#!/usr/bin/env bash

# Quick 0.5% labeled comparison: Mean Teacher vs supervised ensemble
# Uses deep GCN.
# QUEUE (gpuv100/gpua100), walltime/mem as appropriate.

QUEUE="${QUEUE:-gpuv100}"
WALLTIME="${WALLTIME:-01:00}"
MEM="${MEM:-8G}"

cd /zhome/c2/2/186918/Project-2-gnn-and-semi_supervised || exit 1

# Common overrides: 79.5% unlabeled, 0.5% labeled, 10% val, 10% test.
SPLITS='[0.795,0.005,0.1,0.1]'
EPOCHS=50

# Supervised ensemble baseline (acts supervised when only labeled loader is used)
bsub -q "$QUEUE" -n 4 -W "$WALLTIME" -R "rusage[mem=$MEM] span[hosts=1]" -gpu "num=1" \
  -J sup0p5pct_s0 -o batch/logs/sup0p5pct_s0.out -e batch/logs/sup0p5pct_s0.err \
  "cd /zhome/c2/2/186918/Project-2-gnn-and-semi_supervised && \
   python src/run.py trainer=semi-supervised-ensemble-tuned model=gcn_deep \
   dataset.qm9.splits='$SPLITS' dataset.qm9.seed=0 seed=0 \
   trainer.train.total_epochs=$EPOCHS trainer.train.validation_interval=10"

# Mean Teacher
bsub -q "$QUEUE" -n 4 -W "$WALLTIME" -R "rusage[mem=$MEM] span[hosts=1]" -gpu "num=1" \
  -J mt0p5pct_s0 -o batch/logs/mt0p5pct_s0.out -e batch/logs/mt0p5pct_s0.err \
  "cd /zhome/c2/2/186918/Project-2-gnn-and-semi_supervised && \
   python src/run.py trainer=mean-teacher-tuned model=gcn_deep \
   dataset.qm9.splits='$SPLITS' dataset.qm9.seed=0 seed=0 \
   trainer.train.total_epochs=$EPOCHS trainer.train.validation_interval=10"

