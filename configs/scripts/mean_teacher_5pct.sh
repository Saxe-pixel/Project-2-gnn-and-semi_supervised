#!/bin/bash
### Mean Teacher with ~5% labeled to show sensitivity
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 06:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -J gnn_mt5
#BSUB -o batch/logs/gnn_mt5_%J.out
#BSUB -e batch/logs/gnn_mt5_%J.err

set -euo pipefail

mkdir -p batch/logs
source ~/miniconda3/bin/activate gnn-qm9

SEEDS=(0 1)

for SEED in "${SEEDS[@]}"; do
  python src/run.py \
    trainer=mean-teacher \
    model=gcn \
    dataset.init.splits=[0.74,0.05,0.1,0.1] \
    dataset.init.data_augmentation=true \
    seed="${SEED}" \
    logger.name="mt5pct_gcn_seed${SEED}"
done
