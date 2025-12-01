#!/bin/bash
### General options
### -- specify queue --
#BSUB -q gpuv100

### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm
#BSUB -W 06:00
### request system-memory
#BSUB -R "rusage[mem=8GB]"

### -- set the job Name --
#BSUB -J gnn_semi_sup
### -- Specify the output and error file. %J is the job-id --
#BSUB -o batch/logs/gnn_%J.out
#BSUB -e batch/logs/gnn_%J.err

# -- end of LSF options --

set -euo pipefail

source ~/miniconda3/bin/activate gnn-qm9

SEEDS=(0 1)

for SEED in "${SEEDS[@]}"; do
  # Supervised baseline (no unlabeled data)
  python src/run.py \
    trainer=semi-supervised-ensemble \
    dataset.splits=[0,0.8,0.1,0.1] \
    dataset.data_augmentation=false \
    seed="${SEED}" \
    logger.name="sup_gcn_seed${SEED}"

  # Mean Teacher with augmentation
  python src/run.py \
    trainer=mean-teacher \
    model=gcn \
    dataset.splits=[0.79,0.01,0.1,0.1] \
    dataset.data_augmentation=true \
    seed="${SEED}" \
    logger.name="mt_gcn_seed${SEED}"

  # n-CPS ensemble with GINE backbone and augmentation
  python src/run.py \
    trainer=NCPSTrainer \
    model=ginegcn \
    dataset.splits=[0.79,0.01,0.1,0.1] \
    dataset.data_augmentation=true \
    seed="${SEED}" \
    logger.name="ncps_gine_seed${SEED}"
done
