#!/bin/bash
### Evaluate best lambda (consistency weight) settings on test MSE across targets/seeds.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 07:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -J gcn_eval_best_lambdas
#BSUB -o results/logs/gcn_eval_best_lambdas_%J.out
#BSUB -e results/logs/gcn_eval_best_lambdas_%J.err

set -euo pipefail

# Start from the directory where bsub was called (repo root)
cd "${LS_SUBCWD:-$PWD}"

# Repo + output dirs relative to submission dir
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

source ~/miniconda3/bin/activate gnn-qm9


# Configurable knobs
TARGETS="${TARGETS:-0 1 2 3 4}"          # QM9 targets to evaluate
SEEDS="${SEEDS:-0 1 2}"                     # seeds to average over
TOTAL_EPOCHS="${TOTAL_EPOCHS:-80}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_eval_best_lambdas_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/gcn_eval_best_lambdas_test.png}"
PLOT_VAL_PATH="${PLOT_VAL_PATH:-${OUTPUT_BASE}/gcn_eval_best_lambdas_val.png}"

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_PATH \
       PLOT_VAL_PATH TOTAL_EPOCHS VAL_INTERVAL TARGETS SEEDS


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
seeds = [int(x) for x in os.environ["SEEDS"].split()]
label_pct = 1.0
unlabel_pct = 79.0
splits = [unlabel_pct / 100.0, label_pct / 100.0, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

# Best from lambda sweep: Mean Teacher lambda=0.001, n-CPS lambda=0.001
trainers = [
    ("semi-supervised-ensemble", "Supervised Ensemble", {}),
    ("mean-teacher", "Mean Teacher (EMA)", {"trainer.init.consistency_weight": 0.001}),
    ("NCPSTrainer", "n-CPS Ensemble", {"trainer.init.cps_weight": 0.001}),
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
plot_val_path = pathlib.Path(os.environ["PLOT_VAL_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "80")
val_interval = os.environ.get("VAL_INTERVAL", "10")


def evaluate_loader(loader, trainer_id, trainer, device, dm):
    """Compute MSE (denormalized) for a given dataloader."""
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


results = {name: [] for _, name, _ in trainers}

with initialize(config_path="configs", version_base=None):
    for target in targets:
        for seed in seeds:
            for trainer_id, display_name, overrides_map in trainers:
                run_name = f"{trainer_id}_target{target}_seed{seed}_bestlambda"
                overrides = [
                    f"trainer={trainer_id}",
                    "model=gcn",
                    f"dataset.init.splits=[{split_str}]",
                    f"dataset.init.target={target}",
                    f"dataset.init.data_dir={data_dir}",
                    f"seed={seed}",
                    "logger.disable=false",
                    f"logger.name='{run_name}'",
                    "logger.group=gcn_eval_best_lambdas",
                    f"trainer.train.total_epochs={total_epochs}",
                    f"trainer.train.validation_interval={val_interval}",
                ]
                for k, v in overrides_map.items():
                    overrides.append(f"{k}={v}")

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

                print(f"Training {display_name} | target={target} | seed={seed} | overrides={overrides_map}", flush=True)
                trainer.train(**cfg.trainer.train)
                val_mse = evaluate_loader(dm.val_dataloader(), trainer_id, trainer, device, dm)
                test_mse = evaluate_loader(dm.test_dataloader(), trainer_id, trainer, device, dm)
                print(f"Val/Test MSE ({display_name}, target {target}, seed {seed}): {val_mse:.6f} / {test_mse:.6f}", flush=True)

                results[display_name].append(
                    {
                        "target": target,
                        "seed": seed,
                        "val_loss": val_mse,
                        "test_loss": test_mse,
                        "overrides": overrides_map,
                    }
                )

                if hasattr(logger, "end_run"):
                    logger.end_run()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"metric": "test_MSE", "data": results}, indent=2))
print(f"Saved raw results to {results_json}")

# Plot mean over seeds per target
plt.figure(figsize=(7, 5))
for display_name, entries in results.items():
    by_target = {}
    for e in entries:
        by_target.setdefault(e["target"], []).append(e["test_loss"])
    xs = sorted(by_target)
    ys = [float(np.mean(by_target[t])) for t in xs]
    plt.plot(xs, ys, marker="o", label=display_name)
plt.xlabel("QM9 target index")
plt.ylabel("Test MSE (mean over seeds)")
plt.title("Best lambda evaluation (1% labeled / 79% unlabeled)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")

# Plot mean val MSE over seeds per target
plt.figure(figsize=(7, 5))
for display_name, entries in results.items():
    by_target = {}
    for e in entries:
        by_target.setdefault(e["target"], []).append(e["val_loss"])
    xs = sorted(by_target)
    ys = [float(np.mean(by_target[t])) for t in xs]
    plt.plot(xs, ys, marker="o", label=display_name)
plt.xlabel("QM9 target index")
plt.ylabel("Val MSE (mean over seeds)")
plt.title("Best lambda evaluation (validation, 1% labeled / 79% unlabeled)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plot_val_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_val_path, dpi=200, bbox_inches="tight")
print(f"Saved val plot to {plot_val_path}")
PY
