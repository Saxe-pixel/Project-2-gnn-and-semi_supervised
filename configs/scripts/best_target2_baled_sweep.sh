#!/bin/bash
### FINAL EVALUATION: Label Efficiency Sweep on TEST SET (Target 2) with Optimized Configs.
#BSUB -q gpua100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 06:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -J gcn_label_sweep_test
#BSUB -o results/logs/gcn_label_sweep_test_target2_2_%J.out
#BSUB -e results/logs/gcn_label_sweep_test_target2_2_%J.err

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

# --- CONFIGURATION ---
TARGET="2"  # HOMO Energy
# Fractions to sweep: 0.5, 1, 2, 3, 5, 8
LABEL_PCTS="${LABEL_PCTS:-0.5 1 2 3 5 8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_label_sweep_test_results_target2_2.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/gcn_label_sweep_test_curve_target2_2.png}"
SEED="${SEED:-0}"  # Random seed override (matches configs/run.yaml default if unset)

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_PATH TOTAL_EPOCHS VAL_INTERVAL TARGET LABEL_PCTS SEED

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
label_pcts = [float(x) for x in os.environ["LABEL_PCTS"].split()]
seed = int(os.environ.get("SEED", "0"))

# ==================================================================================
# OPTIMIZED HYPERPARAMETERS (Valid for Low-Data Regime)
# ==================================================================================
best_params = {
    "semi-supervised-ensemble": {
        "model.init.hidden_channels": 256,
        "model.init.num_layers": 5,
        "model.init.dropout": 0.39,
        "trainer.init.optimizer.lr": 0.00385,
        "trainer.init.optimizer.weight_decay": 0.00048,
    },
    "mean-teacher": {
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

results = {name: [] for _, name in trainers}

with initialize(config_path="configs", version_base=None):
    for label_pct in label_pcts:
        # Calculate Unlabeled portion to ensure they sum to 80%
        # Train total is 80%, Val is 10%, Test is 10%
        unlabel_pct = 80.0 - label_pct
        
        splits = [unlabel_pct / 100.0, label_pct / 100.0, 0.1, 0.1]
        split_str = ",".join(f"{s:.4f}" for s in splits)
        
        print(f"\n=== Starting Sweep for {label_pct}% Labeled / {unlabel_pct}% Unlabeled ===", flush=True)

        for trainer_id, display_name in trainers:
            run_name = f"{trainer_id}_label{label_pct}_test"
            
            overrides = [
                f"trainer={trainer_id}",
                "model=gcn",
                f"dataset.init.splits=[{split_str}]",
                f"dataset.init.target={target}",
                f"dataset.init.data_dir={data_dir}",
                f"seed={seed}",
                "logger.disable=false",
                f"logger.name={run_name}",
                "logger.group=gcn_label_sweep_test",
                f"trainer.train.total_epochs={total_epochs}",
                f"trainer.train.validation_interval={val_interval}",
            ]
            
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
            
            # Ensure splits are created
            if hasattr(dm, "setup"):
                dm.setup()

            model = hydra.utils.instantiate(cfg.model.init).to(device)
            if cfg.compile_model:
                model = torch.compile(model)
            models = [model]
            trainer = hydra.utils.instantiate(
                cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device
            )

            print(f"Training {display_name} | {label_pct}% Labeled...", flush=True)
            trainer.train(**cfg.trainer.train)
            test_mse = evaluate_test(trainer_id, trainer, device, dm)
            print(f"Test MSE: {test_mse:.6f}", flush=True)

            results[display_name].append({
                "label_pct": label_pct, 
                "unlabel_pct": unlabel_pct,
                "test_loss": test_mse
            })

            if hasattr(logger, "end_run"):
                logger.end_run()
            
            del trainer, model, models, dm, cfg
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"metric": "test_MSE", "data": results}, indent=2))
print(f"Saved raw results to {results_json}")

# Plotting Line Chart
plt.figure(figsize=(8, 6))
for display_name, entries in results.items():
    entries = sorted(entries, key=lambda x: x["label_pct"])
    xs = [e["label_pct"] for e in entries]
    ys = [e["test_loss"] for e in entries]
    plt.plot(xs, ys, marker="o", label=display_name, linewidth=2)

plt.xlabel("Labeled Data Percentage (%)")
plt.ylabel("Test MSE")
plt.title("Label Efficiency: Test Performance vs. Data Scarcity")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
# Optional: Set x-ticks to match the swept values
plt.xticks(label_pcts) 
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
PY
