### Example of batch script

#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc

### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm
#BSUB -W 01:00
### request 2GB of system-memory
#BSUB -R "rusage[mem=2GB]"

### -- set the job Name --
#BSUB -J gnn_train
### -- Specify the output and error file. %J is the job-id --
#BSUB -o batch/logs/test%J.out
#BSUB -e batch/logs/test%J.err

# -- end of LSF options --

### nvidia-smi

# Load modules
module purge
module load python3/3.11.13

source ~/venv/gnn311/bin/activate
python src/run.py trainer=mean-teacher