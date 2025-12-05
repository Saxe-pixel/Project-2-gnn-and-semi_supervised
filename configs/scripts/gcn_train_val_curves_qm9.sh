#!/bin/bash
### Plot train vs validation loss curves for all three trainers
### (Supervised Ensemble, Mean Teacher, n-CPS) using the improved GCN
### and the new QM9 datamodule (target 2).
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 02:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -J gcn_train_val_curves_qm9
#BSUB -o results/logs/gcn_train_val_curves_qm9_%J.out
#BSUB -e results/logs/gcn_train_val_curves_qm9_%J.err

set -euo pipefail

# Start from the directory where bsub was called (repo root)
cd "${LS_SUBCWD:-$PWD}"

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

module purge
module load python3/3.11.13
source ~/venv/gnn311/bin/activate

TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
# Validate every epoch so we can plot full curves
VAL_INTERVAL="${VAL_INTERVAL:-1}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_train_val_curves_qm9_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/gcn_train_val_curves_qm9.png}"

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

target = 2
fixed_seed = 0

# Use the default semi-supervised split from configs/dataset/qm9.yaml:
# 79% unlabeled, 1% labeled, 10% val, 10% test.
splits = [0.79, 0.01, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

# Trainer config name, display name, and trainer2 target class
trainers_to_run = [
    ("semi-supervised-ensemble", "Supervised Ensemble", "trainer2.SemiSupervisedEnsemble"),
    ("mean-teacher", "Mean Teacher", "trainer2.MeanTeacherTrainer"),
    ("NCPSTrainer", "n-CPS Ensemble", "trainer2.NCPSTrainer"),
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
total_epochs = int(os.environ.get("TOTAL_EPOCHS", "100"))
val_interval = int(os.environ.get("VAL_INTERVAL", "1"))


def evaluate(trainer_id, trainer, loader, device, y_mean, y_std):
    """Evaluate denormalized MSE on a given loader."""
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
    for trainer_id, trainer_name, trainer_target in trainers_to_run:
        run_name = f"train_val_curves_{trainer_id}_gcn_qm9"

        overrides = [
            f"trainer={trainer_id}",
            "dataset=qm9",
            "model=gcn",
            f"dataset.init.splits=[{split_str}]",
            f"dataset.init.data_dir={data_dir}",
            f"dataset.init.target={target}",
            "dataset.init.data_augmentation=false",
            "logger.disable=false",
            f"logger.name={run_name}",
            "logger.group=gcn_train_val_curves_qm9",
            f"trainer.train.total_epochs={total_epochs}",
            f"trainer.train.validation_interval={val_interval}",
            f"trainer.init._target_={trainer_target}",
            f"seed={fixed_seed}",
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

        print(
            f"Training {trainer_name} with GCN on QM9 "
            f"(target={target}, seed={fixed_seed})...",
            flush=True,
        )
        trainer.train(**cfg.trainer.train)

        # Collect history: epochs, train supervised loss, and val MSE
        epochs_hist = getattr(trainer, "history", {}).get("epoch", [])
        train_hist = getattr(trainer, "history", {}).get("train_supervised_loss", [])

        if trainer_id == "mean-teacher":
            val_hist = getattr(trainer, "history", {}).get("val_MSE_teacher", [])
        else:
            val_hist = getattr(trainer, "history", {}).get("val_MSE", [])

        # Convert train supervised loss from normalized units to original units
        if hasattr(dm, "y_std"):
            y_std_scalar = float(dm.y_std.detach().cpu().item())
        else:
            y_std_scalar = 1.0
        train_hist_denorm = [float(v) * (y_std_scalar ** 2) for v in train_hist]

        # Final validation and test MSE (denormalized) for reference
        y_mean = dm.y_mean.to(device)
        y_std = dm.y_std.to(device)
        val_mse = evaluate(trainer_id, trainer, dm.val_dataloader(), device, y_mean, y_std)
        test_mse = evaluate(trainer_id, trainer, dm.test_dataloader(), device, y_mean, y_std)

        print(
            f"{trainer_name} - Final Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}",
            flush=True,
        )

        results.append(
            {
                "trainer": trainer_name,
                "trainer_id": trainer_id,
                "target": target,
                "seed": fixed_seed,
                "epochs": [int(e) for e in epochs_hist],
                "train_supervised_loss": train_hist_denorm,
                "val_MSE": [float(v) for v in val_hist],
                "final_val_MSE": val_mse,
                "final_test_MSE": test_mse,
            }
        )

        if hasattr(logger, "end_run"):
            logger.end_run()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"results": results}, indent=2))
print(f"Saved raw results to {results_json}")

# Plot train vs validation curves for all three trainers
plt.figure(figsize=(8, 5))
for entry in results:
    epochs = entry["epochs"]
    train = entry["train_supervised_loss"]
    val = entry["val_MSE"]
    name = entry["trainer"]

    if not epochs:
        continue

    plt.plot(epochs, train, label=f"{name} train", linestyle="-")
    plt.plot(epochs, val, label=f"{name} val", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("MSE loss (denormalized)")
plt.title("Train vs validation loss (GCN + QM9, target 2)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved train/val curves plot to {plot_path}")
PY

