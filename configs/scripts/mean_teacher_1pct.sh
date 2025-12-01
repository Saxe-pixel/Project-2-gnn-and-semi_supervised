#!/bin/bash
### Mean Teacher with ~1% labeled
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 06:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -J gnn_mt1
#BSUB -o batch/logs/gnn_mt1_%J.out
#BSUB -e batch/logs/gnn_mt1_%J.err

set -euo pipefail

mkdir -p batch/logs
source ~/miniconda3/bin/activate gnn-qm9

SEEDS=(0 1)

for SEED in "${SEEDS[@]}"; do
  python src/run.py \
    trainer=mean-teacher \
    model=gcn \
    dataset.init.splits=[0.79,0.01,0.1,0.1] \
    dataset.init.data_augmentation=true \
    seed="${SEED}" \
    logger.name="mt1pct_gcn_seed${SEED}"
done
