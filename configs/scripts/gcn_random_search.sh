#!/bin/bash
### Random hyperparameter search for GCN on QM9 (1% labeled / 79% unlabeled), per trainer, using validation MSE.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 05:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -J gcn_random_search
#BSUB -o results/logs/gcn_random_search_%J.out
#BSUB -e results/logs/gcn_random_search_%J.err

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
TRIALS="${TRIALS:-10}"                           # number of samples per trainer
SEED_BASE="${SEED_BASE:-0}"                      # offset for seeds per trial
TOTAL_EPOCHS="${TOTAL_EPOCHS:-80}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_random_search_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/gcn_random_search_val.png}"

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_PATH \
       TOTAL_EPOCHS VAL_INTERVAL TRIALS SEED_BASE

python - <<'PY'
import json
import math
import os
import pathlib
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import initialize, compose
import hydra
from omegaconf import OmegaConf

from utils import seed_everything

# Fixed split: 79% unlabeled, 1% labeled, 10% val, 10% test
label_pct = 1.0
unlabel_pct = 79.0
splits = [unlabel_pct / 100.0, label_pct / 100.0, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

trials = int(os.environ["TRIALS"])
seed_base = int(os.environ["SEED_BASE"])

trainers = [
    ("semi-supervised-ensemble", "val_MSE", "Supervised Ensemble"),
    ("mean-teacher", "val_MSE_teacher", "Mean Teacher (EMA)"),
    ("NCPSTrainer", "val_MSE", "n-CPS Ensemble"),
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "80")
val_interval = os.environ.get("VAL_INTERVAL", "10")

rng = random.Random(0xC0DE)


def sample_common():
    return {
        # NOTE: these go under model.init.*, not model.*
        "model.init.hidden_channels": rng.choice([64, 128, 256]),
        "model.init.num_layers": rng.choice([2, 3, 4, 5]),
        "model.init.dropout": round(rng.uniform(0.0, 0.5), 2),
        "trainer.init.optimizer.lr": 10 ** rng.uniform(-3.5, -2.0),           # ~0.0003-0.01
        "trainer.init.optimizer.weight_decay": 10 ** rng.uniform(-4.5, -2.5), # ~3e-5-3e-3
    }


def sample_mt():
    return {
        "trainer.init.consistency_weight": round(rng.uniform(0.05, 0.7), 3),
        "trainer.init.ema_decay": round(rng.uniform(0.97, 0.999), 4),
        "trainer.init.consistency_rampup_epochs": rng.choice([5, 10, 20, 30]),
    }


def sample_ncps():
    return {
        "trainer.init.cps_weight": round(rng.uniform(0.001, 0.2), 4),
        "trainer.init.cps_rampup_epochs": rng.choice([5, 10, 20]),
        "trainer.init.num_models": rng.choice([3, 4]),
    }


def evaluate_val(trainer_id, trainer, device, dm):
    """Compute validation MSE (denormalized) for a trained trainer."""
    loader = dm.val_dataloader()
    y_mean = dm.y_mean.to(device)
    y_std = dm.y_std.to(device)

    if trainer_id == "mean-teacher":
        model = trainer.teacher
        model.eval()
        losses = []
        with torch.no_grad():
            for batch, targets in loader:
                batch, targets = batch.to(device), targets.to(device)
                preds = model(batch)
                preds_denorm = preds * y_std + y_mean
                targets_denorm = targets * y_std + y_mean
                loss = torch.nn.functional.mse_loss(preds_denorm, targets_denorm)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")
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


results = {name: [] for _, _, name in trainers}

with initialize(config_path="configs", version_base=None):
    for trainer_id, metric_key, display_name in trainers:
        print(f"\n=== Trainer: {display_name} ===", flush=True)
        for trial in range(trials):
            common = sample_common()
            specific = {}
            if trainer_id == "mean-teacher":
                specific = sample_mt()
            elif trainer_id == "NCPSTrainer":
                specific = sample_ncps()

            trial_seed = seed_base + trial
            overrides = [
                f"trainer={trainer_id}",
                "model=gcn",
                f"dataset.init.splits=[{split_str}]",
                f"dataset.init.data_dir={data_dir}",
                f"seed={trial_seed}",
                "logger.disable=false",
                f"logger.name={trainer_id}_trial{trial}",
                "logger.group=gcn_random_search",
                f"trainer.train.total_epochs={total_epochs}",
                f"trainer.train.validation_interval={val_interval}",
            ]

            # add hyperparam overrides
            for k, v in {**common, **specific}.items():
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

            print(f"Trial {trial+1}/{trials} | {display_name} | seed={trial_seed}", flush=True)
            trainer.train(**cfg.trainer.train)
            val_mse = evaluate_val(trainer_id, trainer, device, dm)
            print(f"Validation MSE: {val_mse:.6f}", flush=True)

            results[display_name].append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "val_loss": val_mse,
                    "common": common,
                    "specific": specific,
                }
            )

            if hasattr(logger, "end_run"):
                logger.end_run()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"metric": "val_MSE", "data": results}, indent=2))
print(f"Saved raw results to {results_json}")

# Plot per-trainer: validation MSE vs. trial id
plt.figure(figsize=(8, 5))
for display_name, entries in results.items():
    entries = sorted(entries, key=lambda x: x["trial"])
    xs = [e["trial"] for e in entries]
    ys = [e["val_loss"] for e in entries]
    plt.plot(xs, ys, marker="o", label=display_name)
plt.xlabel("Trial")
plt.ylabel("Validation MSE")
plt.title(f"GCN random search ({trials} trials per trainer)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
PY
