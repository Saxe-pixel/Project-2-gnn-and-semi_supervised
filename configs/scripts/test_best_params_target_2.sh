#!/bin/bash
### FINAL EVALUATION: Run the best Validation configs on the TEST SET for Target 2.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -J gcn_final_test_target2
#BSUB -o results/logs/gcn_final_test_target2_%J.out
#BSUB -e results/logs/gcn_final_test_target2_%J.err

set -euo pipefail

# Start from the directory where bsub was called (repo root)
cd "${LS_SUBCWD:-$PWD}"

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

source ~/miniconda3/bin/activate gnn-qm9

# --- CONFIGURATION ---
TARGET="2"  # HOMO Energy (Primary Target)
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_final_test_target2_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/gcn_final_test_target2_bar.png}"

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_PATH TOTAL_EPOCHS VAL_INTERVAL TARGET

python - <<'PY'
import json
import os
import pathlib
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import initialize, compose
import hydra
from omegaconf import OmegaConf

from utils import seed_everything

target = int(os.environ["TARGET"])
# 1% Labeled / 79% Unlabeled split
label_pct = 1.0
unlabel_pct = 79.0
splits = [unlabel_pct / 100.0, label_pct / 100.0, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

# ==================================================================================
# BEST HYPERPARAMETERS (Derived from Validation Random Search)
# ==================================================================================
best_params = {
    "semi-supervised-ensemble": {
        # Best Supervised (Trial 6): Deep (5 layers), High Dropout (0.39)
        "model.init.hidden_channels": 256,
        "model.init.num_layers": 5,
        "model.init.dropout": 0.39,
        "trainer.init.optimizer.lr": 0.00385,
        "trainer.init.optimizer.weight_decay": 0.00048,
    },
    "mean-teacher": {
        # Best Mean Teacher (Trial 8): Shallow (3 layers), Low Dropout (0.08), High Lambda (0.54)
        "model.init.hidden_channels": 256,
        "model.init.num_layers": 3,
        "model.init.dropout": 0.08,
        "trainer.init.optimizer.lr": 0.00620,
        "trainer.init.optimizer.weight_decay": 5.78e-5,
        "trainer.init.consistency_weight": 0.544,
        "trainer.init.ema_decay": 0.9935,
        "trainer.init.consistency_rampup_epochs": 5,
    },
    "NCPSTrainer": {
        # Best n-CPS (Trial 7): Deep (5 layers), Moderate Dropout (0.13), Lambda (0.20), 4 Models
        "model.init.hidden_channels": 256,
        "model.init.num_layers": 5,
        "model.init.dropout": 0.13,
        "trainer.init.optimizer.lr": 0.00247,
        "trainer.init.optimizer.weight_decay": 0.00019,
        "trainer.init.cps_weight": 0.1993,
        "trainer.init.cps_rampup_epochs": 10,
        "trainer.init.num_models": 4,
    }
}

trainers = [
    ("semi-supervised-ensemble", "Supervised Baseline"),
    ("mean-teacher", "Mean Teacher"),
    ("NCPSTrainer", "n-CPS Ensemble"),
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "100")
val_interval = os.environ.get("VAL_INTERVAL", "10")

def evaluate_test(trainer_id, trainer, device, dm):
    """Compute TEST MSE (denormalized) for a trained trainer."""
    loader = dm.test_dataloader() # <--- CRITICAL: Using Test Set
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

results = []

with initialize(config_path="configs", version_base=None):
    for trainer_id, display_name in trainers:
        run_name = f"{trainer_id}_final_test_target{target}"
        
        # Base overrides
        overrides = [
            f"trainer={trainer_id}",
            "model=gcn",
            f"dataset.init.splits=[{split_str}]",
            f"dataset.init.target={target}",
            f"dataset.init.data_dir={data_dir}",
            "logger.disable=false",
            f"logger.name={run_name}",
            "logger.group=gcn_final_test",
            f"trainer.train.total_epochs={total_epochs}",
            f"trainer.train.validation_interval={val_interval}",
        ]
        
        # Apply Optimized Params for this specific trainer
        specific_params = best_params.get(trainer_id, {})
        for k, v in specific_params.items():
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

        print(f"Training {display_name} (Optimized) | Target {target} | 1% Labeled...", flush=True)
        trainer.train(**cfg.trainer.train)
        test_mse = evaluate_test(trainer_id, trainer, device, dm)
        print(f"FINAL TEST MSE: {test_mse:.6f}", flush=True)

        results.append({
            "method": display_name,
            "test_mse": test_mse,
            "params": specific_params
        })

        if hasattr(logger, "end_run"):
            logger.end_run()
        
        # Cleanup
        del trainer, model, models, dm, cfg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"target": target, "results": results}, indent=2))
print(f"Saved raw results to {results_json}")

# Plotting Bar Chart
plt.figure(figsize=(7, 5))
methods = [r["method"] for r in results]
mses = [r["test_mse"] for r in results]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

bars = plt.bar(methods, mses, color=colors, alpha=0.8, width=0.5)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.ylabel("Test MSE")
plt.title(f"Final Test Performance on Target {target} (Optimized Architectures)")
plt.grid(axis='y', linestyle="--", alpha=0.4)
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
PY