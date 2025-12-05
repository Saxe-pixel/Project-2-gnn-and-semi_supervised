#!/bin/bash
### Sweep consistency weights (lambda) for Mean Teacher and n-CPS on validation MSE, fixed 1% labeled / 79% unlabeled.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 06:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -J gcn_lambda_sweep
#BSUB -o results/logs/gcn_lambda_sweep_%J.out
#BSUB -e results/logs/gcn_lambda_sweep_%J.err

set -euo pipefail

# Start from the directory where bsub was called (repo root)
cd "${LS_SUBCWD:-$PWD}"

# Repo + output dirs relative to submission dir
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"   # <-- add this line

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

module purge
module load python3/3.11.13
source ~/venv/gnn311/bin/activate


# Configurable knobs
LAMBDA_START="${LAMBDA_START:-0.0005}"
LAMBDA_END="${LAMBDA_END:-1.0}"
NUM_LAMBDAS="${NUM_LAMBDAS:-50}"

# Generate 50 equally spaced lambdas in [0.0005, 1.0] by default.
LAMBDA_LIST=$(python - <<PY
start = float("${LAMBDA_START}")
end = float("${LAMBDA_END}")
n = int("${NUM_LAMBDAS}")
if n < 2:
    vals = [start]
else:
    step = (end - start) / (n - 1)
    vals = [start + i * step for i in range(n)]
print(" ".join(f"{v:.6f}" for v in vals))
PY
)

# Allow overriding via environment, but default both trainers to the same grid.
: "${MT_LAMBDAS:=${LAMBDA_LIST}}"
: "${NCPS_LAMBDAS:=${LAMBDA_LIST}}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-80}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_lambda_sweep_results.json}"
PLOT_MT_PATH="${PLOT_MT_PATH:-${OUTPUT_BASE}/gcn_lambda_sweep_mean_teacher.png}"
PLOT_NCPS_PATH="${PLOT_NCPS_PATH:-${OUTPUT_BASE}/gcn_lambda_sweep_ncps.png}"

export REPO_ROOT DATA_DIR RESULTS_JSON PLOT_MT_PATH PLOT_NCPS_PATH \
       TOTAL_EPOCHS VAL_INTERVAL MT_LAMBDAS NCPS_LAMBDAS

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

label_pct = 1.0
unlabel_pct = 79.0
splits = [unlabel_pct / 100.0, label_pct / 100.0, 0.1, 0.1]
split_str = ",".join(f"{s:.4f}" for s in splits)

mt_lambdas = [float(x) for x in os.environ["MT_LAMBDAS"].split()]
ncps_lambdas = [float(x) for x in os.environ["NCPS_LAMBDAS"].split()]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_mt_path = pathlib.Path(os.environ["PLOT_MT_PATH"])
plot_ncps_path = pathlib.Path(os.environ["PLOT_NCPS_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "100")
val_interval = os.environ.get("VAL_INTERVAL", "10")

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

results = {"Mean Teacher": [], "n-CPS Ensemble": []}

with initialize(config_path="configs", version_base=None):
    # Mean Teacher sweep
    for lam in mt_lambdas:
        run_name = f"mean_teacher_lambda_{lam}"
        overrides = [
            "trainer=mean-teacher",
            "model=gcn",
            f"dataset.init.splits=[{split_str}]",
            f"dataset.init.data_dir={data_dir}",
            "logger.disable=false",
            f"logger.name={run_name}",
            "logger.group=gcn_lambda_sweep_mt",
            f"trainer.train.total_epochs={total_epochs}",
            f"trainer.train.validation_interval={val_interval}",
            f"trainer.init.consistency_weight={lam}",
        ]

        cfg = compose(config_name="run", overrides=overrides)
        device = torch.device("cuda" if cfg.device in ["unset", "auto"] and torch.cuda.is_available() else cfg.device if cfg.device not in ["unset", "auto"] else "cpu")

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

        print(f"Mean Teacher: lambda={lam}", flush=True)
        trainer.train(**cfg.trainer.train)
        val_mse = evaluate_val("mean-teacher", trainer, device, dm)
        print(f"Val MSE: {val_mse:.6f}", flush=True)

        results["Mean Teacher"].append({"lambda": lam, "val_loss": val_mse})

        if hasattr(logger, "end_run"):
            logger.end_run()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # n-CPS sweep
    for lam in ncps_lambdas:
        run_name = f"ncps_lambda_{lam}"
        overrides = [
            "trainer=NCPSTrainer",
            "model=gcn",
            f"dataset.init.splits=[{split_str}]",
            f"dataset.init.data_dir={data_dir}",
            "logger.disable=false",
            f"logger.name={run_name}",
            "logger.group=gcn_lambda_sweep_ncps",
            f"trainer.train.total_epochs={total_epochs}",
            f"trainer.train.validation_interval={val_interval}",
            f"trainer.init.cps_weight={lam}",
        ]

        cfg = compose(config_name="run", overrides=overrides)
        device = torch.device("cuda" if cfg.device in ["unset", "auto"] and torch.cuda.is_available() else cfg.device if cfg.device not in ["unset", "auto"] else "cpu")

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

        print(f"n-CPS: lambda={lam}", flush=True)
        trainer.train(**cfg.trainer.train)
        val_mse = evaluate_val("NCPSTrainer", trainer, device, dm)
        print(f"Val MSE: {val_mse:.6f}", flush=True)

        results["n-CPS Ensemble"].append({"lambda": lam, "val_loss": val_mse})

        if hasattr(logger, "end_run"):
            logger.end_run()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"metric": "val_MSE", "data": results}, indent=2))
print(f"Saved raw results to {results_json}")

def plot_curve(entries, title, path):
    entries = sorted(entries, key=lambda x: x["lambda"])
    xs = [e["lambda"] for e in entries]
    ys = [e["val_loss"] for e in entries]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Lambda (consistency weight)")
    plt.ylabel("Validation MSE")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {path}")

plot_curve(results["Mean Teacher"], "Mean Teacher lambda sweep (val MSE)", plot_mt_path)
plot_curve(results["n-CPS Ensemble"], "n-CPS lambda sweep (val MSE)", plot_ncps_path)
PY
