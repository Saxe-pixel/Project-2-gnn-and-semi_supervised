# GNN_semi_supervised_drug_discovery

This repository contains code for semi‑supervised molecular property prediction using Graph Neural Networks on the QM9 dataset (DTU course 02456 Deep Learning).

The **main results and figures** are collected in the notebook:
- `notebooks/semisupervised_pipeline.ipynb`


## Running Models Locally

Train a model using Hydra:

```bash
python src/run.py
```

You can override the model and trainer from the command line, e.g.:

```bash
python src/run.py model=gcn         # standard GCN
python src/run.py model=gcn_deep   # deeper GCN
python src/run.py trainer=mean-teacher
python src/run.py trainer=NCPSTrainer
```


## Submitting Jobs on DTU HPC

Submit a job via LSF using one of the scripts in `configs/scripts/`:

```bash
bsub < configs/scripts/gcn_label_sweep.sh
```


## Repository Structure

```text
configs/        - Hydra configs and experiment scripts
src/            - Models, trainers, data modules, utils, run.py
notebooks/      - Main analysis notebook with final figures
results/        - Saved JSON summaries and plots (some useful, some not)
outputs/        - Hydra / W&B run directories
logs/           - HPC log files
requirements.txt
```


## Dataset

The project uses the QM9 dataset from PyTorch Geometric.  
By default (`configs/dataset/qm9.yaml`) we use a low‑label split:

* 79% unlabeled
* 1% labeled
* 10% validation
* 10% test

Some comparison scripts also use 72% / 8% / 10% / 10% splits; see `configs/scripts/compare_original_vs_gcn_qm9_variants.sh`.
In the report, a label sweep was done from 0.5-8%.

## Summary (empirical)

High‑level observations from the experiments (see the notebook for details):

* Semi‑supervised methods (Mean Teacher, n‑CPS) give modest gains when labels are very scarce.
* Architecture depth, hidden size and optimization hyperparameters have a large impact on performance.


## Installation

Install PyTorch and PyTorch Geometric for your hardware (see their official installation guides), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```
