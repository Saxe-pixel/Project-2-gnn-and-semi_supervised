#!/bin/bash
### Sweep labeled percentages for GCN across all three trainers using
### the best val-MSE hyperparameters from gcn_random_search_30trials_results.json.
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 05:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -J gcn_label_sweep_best_30trials
#BSUB -o results/logs/gcn_label_sweep_best_30trials_%J.out
#BSUB -e results/logs/gcn_label_sweep_best_30trials_%J.err

set -euo pipefail

# Start from the directory where bsub was called (repo root)
cd "${LS_SUBCWD:-$PWD}"

# Repo + output dirs relative to submission dir
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

source ~/venv/gnn311/bin/activate

# Optional: override via environment variables before submission, e.g.:
#   TOTAL_EPOCHS=50 VAL_INTERVAL=5 bsub < configs/scripts/gcn_label_sweep_best_30trials.sh
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RANDOM_SEARCH_JSON="${RANDOM_SEARCH_JSON:-${OUTPUT_BASE}/gcn_random_search_30trials_results.json}"
RESULTS_JSON="${RESULTS_JSON:-${OUTPUT_BASE}/gcn_label_sweep_best_30trials_results.json}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_BASE}/gcn_label_sweep_best_30trials_loss.png}"

export REPO_ROOT DATA_DIR RANDOM_SEARCH_JSON RESULTS_JSON PLOT_PATH TOTAL_EPOCHS VAL_INTERVAL

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

label_pcts = [0.5, 1, 2, 5, 8]
unlabel_pcts = [79.5, 79, 78, 75, 72]

assert len(label_pcts) == len(unlabel_pcts), "Label and unlabeled percentage lists must align."

trainers = [
    ("semi-supervised-ensemble", "val_MSE", "Supervised Ensemble"),
    ("mean-teacher", "val_MSE_teacher", "Mean Teacher (EMA)"),
    ("NCPSTrainer", "val_MSE", "n-CPS Ensemble"),
]

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data_dir = os.environ["DATA_DIR"]
random_search_json = pathlib.Path(os.environ["RANDOM_SEARCH_JSON"])
results_json = pathlib.Path(os.environ["RESULTS_JSON"])
plot_path = pathlib.Path(os.environ["PLOT_PATH"])
total_epochs = os.environ.get("TOTAL_EPOCHS", "100")
val_interval = os.environ.get("VAL_INTERVAL", "10")

if not random_search_json.is_file():
    raise FileNotFoundError(f"Random search results not found at {random_search_json}")

with random_search_json.open("r") as f:
    random_results = json.load(f)


def get_best_overrides(display_name: str):
    """Return merged common+specific overrides for the best (lowest val_loss) trial."""
    entries = random_results["data"].get(display_name)
    if not entries:
        raise ValueError(f"No entries found in random-search results for '{display_name}'")
    best = min(entries, key=lambda e: e["val_loss"])
    overrides = {}
    overrides.update(best.get("common", {}))
    overrides.update(best.get("specific", {}))
    return best, overrides


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


# Precompute best hyperparameters per trainer from random-search results
best_trials = {}
for trainer_id, _metric_key, display_name in trainers:
    best_entry, overrides = get_best_overrides(display_name)
    best_trials[trainer_id] = {
        "display": display_name,
        "val_loss": best_entry["val_loss"],
        "trial": best_entry["trial"],
        "seed": best_entry["seed"],
        "overrides": overrides,
    }
    print(
        f"Best trial for {display_name}: "
        f"trial={best_entry['trial']} seed={best_entry['seed']} "
        f"val_loss={best_entry['val_loss']:.6f} overrides={overrides}",
        flush=True,
    )


results = {name: [] for _, _, name in trainers}

with initialize(config_path="configs", version_base=None):
    for labeled_pct, unlabeled_pct in zip(label_pcts, unlabel_pcts):
        splits = [unlabeled_pct / 100.0, labeled_pct / 100.0, 0.1, 0.1]
        split_str = ",".join(f"{s:.4f}" for s in splits)

        for trainer_id, _metric_key, display_name in trainers:
            cfg_best = best_trials[trainer_id]
            overrides_map = cfg_best["overrides"]
            run_name = f"{trainer_id}_labels{str(labeled_pct).replace('.', 'p')}_best30"

            overrides = [
                f"trainer={trainer_id}",
                "model=gcn",
                f"dataset.init.splits=[{split_str}]",
                f"dataset.init.data_dir={data_dir}",
                "logger.disable=false",
                f"logger.name={run_name}",
                "logger.group=gcn_label_sweep_best_30trials",
                f"trainer.train.total_epochs={total_epochs}",
                f"trainer.train.validation_interval={val_interval}",
            ]

            # Add hyperparameter overrides from the best random-search trial
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

            print(
                f"Training {display_name} with {labeled_pct}% labeled / {unlabeled_pct}% unlabeled "
                f"using best-30trials hyperparameters...",
                flush=True,
            )
            trainer.train(**cfg.trainer.train)
            test_mse = evaluate_test(trainer_id, trainer, device, dm)
            print(f"Test MSE ({display_name}, {labeled_pct}% labels): {test_mse:.6f}", flush=True)

            results[display_name].append(
                {
                    "labels_pct": labeled_pct,
                    "unlabeled_pct": unlabeled_pct,
                    "test_loss": test_mse,
                    "best_trial": {
                        "trial": cfg_best["trial"],
                        "seed": cfg_best["seed"],
                        "val_loss": cfg_best["val_loss"],
                        "overrides": overrides_map,
                    },
                }
            )

            if hasattr(logger, "end_run"):
                logger.end_run()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps({"metric": "test_MSE", "data": results}, indent=2))
print(f"Saved raw results to {results_json}")

plt.figure(figsize=(7, 5))
for display_name, entries in results.items():
    entries = sorted(entries, key=lambda x: x["labels_pct"])
    xs = [e["labels_pct"] for e in entries]
    ys = [e["test_loss"] for e in entries]
    plt.plot(xs, ys, marker="o", label=display_name)
plt.xlabel("Labeled data (%)")
plt.ylabel("Test MSE")
plt.title("GCN label sweep with best 30-trial hyperparameters (test loss vs. labeled percentage)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
PY

