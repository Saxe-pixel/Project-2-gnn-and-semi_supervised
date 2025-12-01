#!/bin/bash
### Supervised baseline (no unlabeled data)
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 06:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -J gnn_sup
#BSUB -o batch/logs/gnn_sup_%J.out
#BSUB -e batch/logs/gnn_sup_%J.err

set -euo pipefail

mkdir -p batch/logs
source ~/miniconda3/bin/activate gnn-qm9

SEEDS=(0 1)

for SEED in "${SEEDS[@]}"; do
  python src/run.py \
    trainer=semi-supervised-ensemble \
    model=gcn \
    dataset.init.splits=[0.0,0.8,0.1,0.1] \
    dataset.init.data_augmentation=false \
    seed="${SEED}" \
    logger.name="sup_gcn_seed${SEED}"
done
