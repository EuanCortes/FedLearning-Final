#!/bin/bash


########################### General options ###########################

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J test_flower

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:05

# request system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/%J.out
#BSUB -e /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/%J.err
# -- end of LSF options --

source /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/.venv/bin/activate

echo "Running Flower job on $(hostname)"

#export CUDA_VISIBLE_DEVICES=0,1

export CUDA_LAUNCH_BLOCKING=1

export TORCH_USE_CUDA_DSA=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#DEVICE_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)

nvidia-smi

flwr run    \
    --federation-config "options.num-supernodes=10" \
    --federation-config "options.backend.init-args.address='local'" \
    --federation-config "options.backend.init-args.num-cpus=${LSB_DJOB_NUMPROC}" \
    --federation-config "options.backend.init-args.num-gpus=1" \
    --federation-config "options.backend.client-resources.num-cpus=2" \
    --federation-config "options.backend.client-resources.num-gpus=0.2" \
    --run-config "num-server-rounds=2"

