### Hyperparameter sweep for NCPSTrainer on QM9 with GCN_CPS

#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100

### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm
#BSUB -W 08:00
### request 4GB of system-memory
#BSUB -R "rusage[mem=4GB]"

### -- set the job Name --
#BSUB -J ncps_sweep
### -- Specify the output and error file. %J is the job-id --
#BSUB -o batch/logs/ncps_sweep%J.out
#BSUB -e batch/logs/ncps_sweep%J.err

# -- end of LSF options --

# Load modules
module purge
module load python3/3.11.13

source ~/venv/gnn311/bin/activate

# Hydra multirun sweep over key NCPSTrainer hyperparameters.
# Each combination becomes a separate W&B run.
python src/run.py -m \
  trainer=NCPSTrainer model=gcn_cps dataset=qm9 \
  logger.group=ncps_sweep \
  trainer.init.cps_weight=0.0002,0.0005,0.001 \
  trainer.init.mask_percentile=0.02,0.05,0.1 \
  trainer.init.strong_node_noise=0.002,0.003 \
  trainer.init.strong_edge_drop=0.002,0.004 \
  seed=0,1

