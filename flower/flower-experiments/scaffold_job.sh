#!/bin/bash


########################### General options ###########################

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J scaffold_job_array[1-3]

### -- ask for number of cores (default: 1) --
#BSUB -n 16

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=shared"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 01:00

# request system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/scaffold_%J_%I.out
#BSUB -e /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/scaffold_%J_%I.err
# -- end of LSF options --

source /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/.venv/bin/activate

# job idxs 1-6 are for 10 clients, fraction-fit=1, num-server-rounds=50
# job idxs 7-12 are for 100 clients, fraction-fit=0.2, num-server-rounds=100

# each configuration is run 3 times with different partition methods
if ((LSB_JOBINDEX == 1)); then
    partition_method='iid'
elif ((LSB_JOBINDEX == 2)); then
    partition_method='dirichlet'
else
    partition_method='shard'
fi

echo "Job #$LSB_JOBINDEX → method=$method, supernodes=$num_supernodes, fraction=$fraction_fit, rounds=$num_server_rounds, partition=$partition_method"

federation_config="
options.num-supernodes=${num_supernodes} \
options.backend.init-args.address='local' \
options.backend.init-args.num-cpus=${LSB_DJOB_NUMPROC} \
options.backend.init-args.num-gpus=1 \
options.backend.client-resources.num-cpus=2 \
options.backend.client-resources.num-gpus=0.1
"

run_config="
num-server-rounds=${num_server_rounds} \
learning-rate=0.05 \
momentum=0.9 \
batch-size=32 \
local-epochs=10 \
fraction-fit=${fraction_fit} \
method='${method}' \
num-partitions=${num_supernodes} \
partition-method='${partition_method}' \
save-dir='/zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/flower/flower-experiments/client_cvs/${LSB_JOBINDEX}' \
"

echo "Running: flwr run --federation-config '$federation_config' --run-config '$run_config'"

flwr run    \
    --federation-config "$federation_config" \
    --run-config "$run_config"

python3 clean_cvs.py /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/flower/flower-experiments/client_cvs/${LSB_JOBINDEX}

