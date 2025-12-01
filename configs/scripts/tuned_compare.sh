#!/bin/bash
### Tuned comparison: supervised ensemble vs mean-teacher vs n-CPS (≈1% labels) with deeper GCN
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 08:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -J gnn_tuned
#BSUB -o batch/logs/gnn_tuned_%J.out
#BSUB -e batch/logs/gnn_tuned_%J.err

set -euo pipefail

mkdir -p batch/logs
source ~/miniconda3/bin/activate gnn-qm9

SEEDS=(0 1)

for SEED in "${SEEDS[@]}"; do
  # Supervised baseline (still only sees labeled set)
  python src/run.py \
    trainer=semi-supervised-ensemble-tuned \
    model=gcn_deep \
    dataset.init.splits=[0.79,0.01,0.1,0.1] \
    dataset.init.data_augmentation=true \
    seed="${SEED}" \
    logger.name="tuned_sup_deep_seed${SEED}"

  # Mean-Teacher with deeper GCN backbone
  python src/run.py \
    trainer=mean-teacher-tuned \
    model=gcn_deep \
    dataset.init.splits=[0.79,0.01,0.1,0.1] \
    dataset.init.data_augmentation=true \
    seed="${SEED}" \
    logger.name="tuned_mt_deep_seed${SEED}"

  # n-CPS ensemble with GINE backbone
  python src/run.py \
    trainer=NCPSTrainer-tuned \
    model=ginegcn \
    dataset.init.splits=[0.79,0.01,0.1,0.1] \
    dataset.init.data_augmentation=true \
    seed="${SEED}" \
    logger.name="tuned_ncps_gine_seed${SEED}"
done
