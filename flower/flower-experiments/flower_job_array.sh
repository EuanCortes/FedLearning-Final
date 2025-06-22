#!/bin/bash


########################### General options ###########################

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J flower_job_array[1-12]

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
#BSUB -o /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/%J_%I.out
#BSUB -e /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/logs/lsf/%J_%I.err
# -- end of LSF options --

source /zhome/94/5/156250/Documents/FederatedLearning/FedLearning-Final/.venv/bin/activate

# job idxs 1-6 are for 10 clients, fraction-fit=1, num-server-rounds=50
# job idxs 7-12 are for 100 clients, fraction-fit=0.2, num-server-rounds=100

if (( LSB_JOBINDEX < 7 )); then
    method='fedavg'             # use fedavg method for job indices 1-6
else
    method='scaffold'           # use scaffold method for job indices 7-12
fi

# first 3 jobs per method are run with 10 clients, fraction-fit=1, num-server-rounds=50
# last 3 jobs per method are run with 100 clients, fraction-fit=0.2, num-server-rounds=100

r=$(( (LSB_JOBINDEX - 1) % 6 ))
if ((r < 3)); then
    num_supernodes=10
    fraction_fit=1
    num_server_rounds=50
else
    num_supernodes=100
    fraction_fit=0.2
    num_server_rounds=100
fi

# each configuration is run 3 times with different partition methods
rr=$(( r % 3 ))
if ((rr == 0)); then
    partition_method='iid'
elif ((rr == 1)); then
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

