#!/bin/bash
### Sweep QM9 targets (0-4) with fixed 1% labeled / 79% unlabeled splits for all trainers, plot test MSE.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 05:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -J gcn_target_sweep
#BSUB -o results/logs/gcn_target_sweep_%J.out
#BSUB -e results/logs/gcn_target_sweep_%J.err

set -euo pipefail

# Start from the directory where bsub was called (repo root)
cd "${LS_SUBCWD:-$PWD}"

# Repo + output dirs relative to submission dir
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"   # <-- add this line

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

source ~/miniconda3/bin/activate gnn-qm9


# Configurable knobs
TARGETS="${TARGETS:-0 1 2 3 4}"               # space-separated list of QM9 target indices
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_target_sweep_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/gcn_target_sweep_loss.png}"

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_PATH TOTAL_EPOCHS VAL_INTERVAL TARGETS


# --- Dry-run mode: only check paths/env, don't run Python ---



python - <<'PY'
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import initialize, compose
import hydra
from omegaconf import OmegaConf

from utils import seed_everything

targets = [int(x) for x in os.environ["TARGETS"].split()]
label_pct = 1.0
unlabel_pct = 79.0
splits = [unlabel_pct / 100.0, label_pct / 100.0, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

trainers = [
    ("semi-supervised-ensemble", "val_MSE", "Supervised Ensemble"),
    ("mean-teacher", "val_MSE_teacher", "Mean Teacher (EMA)"),
    ("NCPSTrainer", "val_MSE", "n-CPS Ensemble"),
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "100")
val_interval = os.environ.get("VAL_INTERVAL", "10")


def evaluate_test(trainer_id, trainer, device, dm):
    """Compute test MSE (denormalized) for a trained trainer."""
    loader = dm.test_dataloader()
    y_mean = dm.y_mean.to(device)
    y_std = dm.y_std.to(device)

    losses = []
    with torch.no_grad():
        if trainer_id == "mean-teacher":
            model = trainer.teacher
            model.eval()
            for batch, targets in loader:
                batch, targets = batch.to(device), targets.to(device)
                preds = model(batch)
                preds_denorm = preds * y_std + y_mean
                targets_denorm = targets * y_std + y_mean
                loss = torch.nn.functional.mse_loss(preds_denorm, targets_denorm)
                losses.append(loss.item())
        else:
            models = trainer.models
            for m in models:
                m.eval()
            for batch, targets in loader:
                batch, targets = batch.to(device), targets.to(device)
                preds_list = [m(batch) for m in models]
                avg_preds = torch.stack(preds_list).mean(0)
                preds_denorm = avg_preds * y_std + y_mean
                targets_denorm = targets * y_std + y_mean
                loss = torch.nn.functional.mse_loss(preds_denorm, targets_denorm)
                losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


results = {name: [] for _, _, name in trainers}

with initialize(config_path="configs", version_base=None):
    for target in targets:
        for trainer_id, metric_key, display_name in trainers:
            run_name = f"{trainer_id}_target{target}_labels1p"
            overrides = [
                f"trainer={trainer_id}",
                "model=gcn",
                f"dataset.init.splits=[{split_str}]",
                f"dataset.init.target={target}",
                f"dataset.init.data_dir={data_dir}",
                "logger.disable=false",
                f"logger.name={run_name}",
                "logger.group=gcn_target_sweep",
                f"trainer.train.total_epochs={total_epochs}",
                f"trainer.train.validation_interval={val_interval}",
            ]

            cfg = compose(config_name="run", overrides=overrides)

            if cfg.device in ["unset", "auto"]:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(cfg.device)

            seed_everything(cfg.seed, cfg.force_deterministic)

            logger = hydra.utils.instantiate(cfg.logger)
            hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            logger.init_run(hparams)

            dm = hydra.utils.instantiate(cfg.dataset.init)
            model = hydra.utils.instantiate(cfg.model.init).to(device)
            if cfg.compile_model:
                model = torch.compile(model)
            models = [model]
            trainer = hydra.utils.instantiate(
                cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device
            )

            print(f"Training {display_name} on target {target} with 1% labeled / 79% unlabeled...", flush=True)
            trainer.train(**cfg.trainer.train)
            test_mse = evaluate_test(trainer_id, trainer, device, dm)
            print(f"Test MSE ({display_name}, target {target}): {test_mse:.6f}", flush=True)

            results[display_name].append({"target": target, "test_loss": test_mse})

            if hasattr(logger, "end_run"):
                logger.end_run()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"metric": "test_MSE", "data": results}, indent=2))
print(f"Saved raw results to {results_json}")

plt.figure(figsize=(7, 5))
for display_name, entries in results.items():
    entries = sorted(entries, key=lambda x: x["target"])
    xs = [e["target"] for e in entries]
    ys = [e["test_loss"] for e in entries]
    plt.plot(xs, ys, marker="o", label=display_name)
plt.xlabel("QM9 target index")
plt.ylabel("Test MSE")
plt.title("GCN target sweep (1% labeled / 79% unlabeled)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
PY
