#!/bin/bash


########################### General options ###########################

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J debug

### -- ask for number of cores (default: 1) --
#BSUB -n 2

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:02

# request system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/debug_%J.out
#BSUB -e /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/debug_%J.err
# -- end of LSF options --

source /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/.venv/bin/activate

echo "Running Flower job on $(hostname)"

ray start --head
sleep 10
ray stop