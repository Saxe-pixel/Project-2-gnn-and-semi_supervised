#!/bin/bash
### n-CPS ensemble with GINE backbone (~1% labeled)
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 06:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -J gnn_ncps1
#BSUB -o batch/logs/gnn_ncps1_%J.out
#BSUB -e batch/logs/gnn_ncps1_%J.err

set -euo pipefail

mkdir -p batch/logs
source ~/miniconda3/bin/activate gnn-qm9

SEEDS=(0 1)

for SEED in "${SEEDS[@]}"; do
  python src/run.py \
    trainer=NCPSTrainer \
    model=ginegcn \
    dataset.init.splits=[0.79,0.01,0.1,0.1] \
    dataset.init.data_augmentation=true \
    seed="${SEED}" \
    logger.name="ncps1pct_gine_seed${SEED}"
done
