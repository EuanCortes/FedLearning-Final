#!/bin/bash


########################### General options ###########################

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J test_flower

### -- ask for number of cores (default: 1) --
#BSUB -n 16

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=shared"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:30

# request system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/%J.out
#BSUB -e /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/%J.err
# -- end of LSF options --

source /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/.venv/bin/activate

federation_config="
options.num-supernodes=10 \
options.backend.init-args.address='local' \
options.backend.init-args.num-cpus=${LSB_DJOB_NUMPROC} \
options.backend.init-args.num-gpus=1 \
options.backend.client-resources.num-cpus=2 \
options.backend.client-resources.num-gpus=0.1
"

run_config="
num-partitions=10 \
num-server-rounds=50 \
method='scaffold' \
partition-method='dirichlet' \
fraction-fit=1 \
momentum=0.9 
"

flwr run    \
    --federation-config "$federation_config" \
    --run-config "$run_config" 

python3 clean_cvs.py /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/flower/flower-experiments/client_cvs

