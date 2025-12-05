#!/bin/bash
### Compare Original_GCN vs improved GCN across all trainers
### using both QM9DataModule versions (original vs new), for target 2 and seed 0.
### Additionally, record and plot validation MSE per epoch for each combination.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 01:30
#BSUB -R "rusage[mem=10GB]"
#BSUB -J compare_original_vs_gcn_qm9vars
#BSUB -o results/logs/compare_original_vs_gcn_qm9vars_%J.out
#BSUB -e results/logs/compare_original_vs_gcn_qm9vars_%J.err

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
# We want validation metrics every epoch to plot val MSE per epoch.
VAL_INTERVAL="${VAL_INTERVAL:-1}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/compare_original_vs_gcn_qm9vars_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/compare_original_vs_gcn_qm9vars_test.png}"
PLOT_VAL_PATH="${PLOT_VAL_PATH:-${OUTPUT_BASE}/compare_original_vs_gcn_qm9vars_val.png}"

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_PATH PLOT_VAL_PATH TOTAL_EPOCHS VAL_INTERVAL

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

# Fixed: target 2 and seed 0 for all runs
target = 2
fixed_seed = 0

# Use 72% unlabeled, 8% labeled, 10% val, 10% test
splits = [0.72, 0.08, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

models_to_run = [
    ("original_gcn", "Original_GCN"),
    ("gcn", "Improved GCN"),
]

# Each entry: (trainer_cfg_name, display_name, trainer_target_override)
# We override trainer.init._target_ to point to the classes in trainer2.py.
trainers_to_run = [
    (
        "semi-supervised-ensemble",
        "Supervised Ensemble",
        "trainer.init._target_=trainer2.SemiSupervisedEnsemble2",
    ),
    (
        "mean-teacher",
        "Mean Teacher",
        "trainer.init._target_=trainer2.MeanTeacherTrainer2",
    ),
    (
        "NCPSTrainer",
        "n-CPS Ensemble",
        "trainer.init._target_=trainer2.NCPSTrainer2",
    ),
]

# Two QM9 datamodule variants from qm9.py:
#   - QM9DataModule_original (no normalization, simple batching)
#   - QM9DataModule (normalized targets, augmentation support)
dataset_variants = [
    ("qm9_original", "QM9 original", ["dataset.init._target_=qm9.QM9DataModule_original"]),
    ("qm9_new", "QM9 new", []),  # default _target_ from configs/dataset/qm9.yaml
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
plot_val_path = pathlib.Path(os.environ["PLOT_VAL_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "100")
val_interval = os.environ.get("VAL_INTERVAL", "10")


def evaluate(trainer_id, trainer, loader, device, y_mean, y_std):
    """Evaluate on a split, handling each trainer type correctly."""
    if trainer_id.startswith("mean-teacher"):
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
    for trainer_id, trainer_name, trainer_target_override in trainers_to_run:
        for model_cfg, model_name in models_to_run:
            for ds_key, ds_name, ds_overrides in dataset_variants:
                run_name = f"compare_qm9vars_{trainer_id}_{model_cfg}_{ds_key}"

                overrides = [
                    f"trainer={trainer_id}",
                    "dataset=qm9",
                    f"model={model_cfg}",
                    f"dataset.init.splits=[{split_str}]",
                    f"dataset.init.data_dir={data_dir}",
                    f"dataset.init.target={target}",
                    "dataset.init.data_augmentation=false",
                    "logger.disable=false",
                    f"logger.name={run_name}",
                    "logger.group=compare_original_vs_gcn_qm9vars",
                    f"trainer.train.total_epochs={total_epochs}",
                    f"trainer.train.validation_interval={val_interval}",
                    f"seed={fixed_seed}",
                ]

                # Use trainer2.* implementations for this run
                if trainer_target_override:
                    overrides.append(trainer_target_override)

                overrides.extend(ds_overrides)

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

                print(
                    f"Training {model_name} with {trainer_name} on {ds_name} "
                    f"(target={target}, seed={fixed_seed})...",
                    flush=True,
                )
                trainer.train(**cfg.trainer.train)

                # Collect validation-history from trainer (per epoch)
                if trainer_id.startswith("mean-teacher"):
                    epochs_hist = getattr(trainer, "history", {}).get("epoch", [])
                    vals_hist = getattr(trainer, "history", {}).get("val_MSE_teacher", [])
                else:
                    epochs_hist = getattr(trainer, "history", {}).get("epoch", [])
                    vals_hist = getattr(trainer, "history", {}).get("val_MSE", [])
                val_history = [
                    {"epoch": int(e), "val_mse": float(v)}
                    for e, v in zip(epochs_hist, vals_hist)
                ]

                # Handle normalization if available (new QM9) or identity otherwise (original QM9)
                if hasattr(dm, "y_mean") and hasattr(dm, "y_std"):
                    y_mean = dm.y_mean.to(device)
                    y_std = dm.y_std.to(device)
                else:
                    y_mean = torch.tensor(0.0, device=device)
                    y_std = torch.tensor(1.0, device=device)

                val_mse = evaluate(trainer_id, trainer, dm.val_dataloader(), device, y_mean, y_std)
                test_mse = evaluate(trainer_id, trainer, dm.test_dataloader(), device, y_mean, y_std)

                print(
                    f"{model_name} + {trainer_name} on {ds_name} - "
                    f"Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}",
                    flush=True,
                )

                results.append(
                    {
                        "trainer": trainer_name,
                        "trainer_id": trainer_id,
                        "model": model_name,
                        "model_cfg": model_cfg,
                        "dataset_variant": ds_name,
                        "dataset_key": ds_key,
                        "target": target,
                        "seed": fixed_seed,
                        "val_history": val_history,
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

# Simple grouped bar plot: x-axis = model/dataset, grouped by trainer (test MSE)
plt.figure(figsize=(10, 5))
trainers = [t[1] for t in trainers_to_run]
labels = []
for model_cfg, model_name in models_to_run:
    for ds_key, ds_name, _ in dataset_variants:
        labels.append(f"{model_name}\n{ds_name}")

x = np.arange(len(labels))
width = 0.22

for idx, trainer_name in enumerate(trainers):
    ys = []
    for model_cfg, model_name in models_to_run:
        for ds_key, ds_name, _ in dataset_variants:
            entry = next(
                r
                for r in results
                if r["trainer"] == trainer_name
                and r["model"] == model_name
                and r["dataset_key"] == ds_key
            )
            ys.append(entry["test_mse"])
    plt.bar(x + idx * width, ys, width=width, label=trainer_name)

plt.xticks(x + width, labels, rotation=45, ha="right")
plt.ylabel("Test MSE")
plt.title("Original_GCN vs Improved GCN across trainers and QM9 variants (target 2, seed 0)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")

# Validation MSE per epoch plot (all 12 combinations)
plt.figure(figsize=(10, 6))
for entry in results:
    history = entry.get("val_history", [])
    if not history:
        continue
    epochs = [h["epoch"] for h in history]
    vals = [h["val_mse"] for h in history]
    label = f"{entry['trainer']} | {entry['model']} | {entry['dataset_variant']}"
    plt.plot(epochs, vals, label=label)

plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Validation MSE per epoch (target 2, seed 0)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=8)
plt.tight_layout()
plot_val_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_val_path, dpi=200, bbox_inches="tight")
print(f"Saved val-per-epoch plot to {plot_val_path}")
PY
