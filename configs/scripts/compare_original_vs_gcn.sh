#!/bin/bash
### Compare Original_GCN vs improved GCN on QM9 across all trainers
### (Supervised Ensemble, Mean Teacher, n-CPS) with 72% unlabeled / 8% labeled.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 01:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -J compare_original_vs_gcn
#BSUB -o results/logs/compare_original_vs_gcn_%J.out
#BSUB -e results/logs/compare_original_vs_gcn_%J.err

set -euo pipefail

# Start from the directory where bsub was called (repo root)
cd "${LS_SUBCWD:-$PWD}"

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

# Load Python and activate your venv (adjust path if needed)
module purge
module load python3/3.11.13
source ~/venv/gnn311/bin/activate

TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/compare_original_vs_gcn_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/compare_original_vs_gcn_test.png}"

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_PATH TOTAL_EPOCHS VAL_INTERVAL

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

# Use 72% unlabeled, 8% labeled, 10% val, 10% test
splits = [0.72, 0.08, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

models_to_run = [
    ("original_gcn", "Original_GCN"),
    ("gcn", "Improved GCN"),
]

trainers_to_run = [
    ("semi-supervised-ensemble", "Supervised Ensemble"),
    ("mean-teacher", "Mean Teacher"),
    ("NCPSTrainer", "n-CPS Ensemble"),
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "80")
val_interval = os.environ.get("VAL_INTERVAL", "10")


def evaluate(trainer_id, trainer, loader, device, y_mean, y_std):
    """Evaluate on a split, handling each trainer type correctly."""
    if trainer_id == "mean-teacher":
        models = [trainer.teacher]
    else:
        models = trainer.models

    for m in models:
        m.eval()

    losses = []
    with torch.no_grad():
        for batch, targets in loader:
            batch, targets = batch.to(device), targets.to(device)
            preds_list = [m(batch) for m in models]
            avg_preds = torch.stack(preds_list).mean(0)
            preds_denorm = avg_preds * y_std + y_mean
            targets_denorm = targets * y_std + y_mean
            loss = torch.nn.functional.mse_loss(preds_denorm, targets_denorm)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


results = []

with initialize(config_path="configs", version_base=None):
    for trainer_id, trainer_name in trainers_to_run:
        for model_cfg, model_name in models_to_run:
            run_name = f"compare_{trainer_id}_{model_cfg}"

            overrides = [
                f"trainer={trainer_id}",
                f"model={model_cfg}",
                f"dataset.init.splits=[{split_str}]",
                f"dataset.init.data_dir={data_dir}",
                "dataset.init.data_augmentation=false",
                "logger.disable=false",
                f"logger.name={run_name}",
                "logger.group=compare_original_vs_gcn",
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
            trainer = hydra.utils.instantiate(
                cfg.trainer.init, models=[model], logger=logger, datamodule=dm, device=device
            )

            print(f"Training {model_name} with {trainer_name}...", flush=True)
            trainer.train(**cfg.trainer.train)

            y_mean = dm.y_mean.to(device)
            y_std = dm.y_std.to(device)

            val_mse = evaluate(trainer_id, trainer, dm.val_dataloader(), device, y_mean, y_std)
            test_mse = evaluate(trainer_id, trainer, dm.test_dataloader(), device, y_mean, y_std)

            print(
                f"{model_name} + {trainer_name} - "
                f"Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}",
                flush=True,
            )

            results.append(
                {
                    "trainer": trainer_name,
                    "trainer_id": trainer_id,
                    "model": model_name,
                    "model_cfg": model_cfg,
                    "val_mse": val_mse,
                    "test_mse": test_mse,
                }
            )

            if hasattr(logger, "end_run"):
                logger.end_run()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"results": results}, indent=2))
print(f"Saved raw results to {results_json}")

# Simple grouped bar plot: x-axis = model, grouped by trainer (test MSE)
plt.figure(figsize=(8, 5))
trainers = [t[1] for t in trainers_to_run]
models = [m[1] for m in models_to_run]

x = np.arange(len(models))
width = 0.25

for idx, trainer_name in enumerate(trainers):
    ys = []
    for model_name in models:
        entry = next(
            r for r in results if r["trainer"] == trainer_name and r["model"] == model_name
        )
        ys.append(entry["test_mse"])
    plt.bar(x + idx * width, ys, width=width, label=trainer_name)

plt.xticks(x + width, models)
plt.ylabel("Test MSE")
plt.title("Original_GCN vs Improved GCN across trainers (test MSE)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
PY
