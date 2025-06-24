"""flower-experiments: A Flower / PyTorch app."""
from typing import Optional, Tuple, Dict

import torch
import numpy as np

from flwr.common import Context, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fedlearn.model import SmallCNN, test
from fedlearn.server import ScaffoldServer
from fedlearn.data_loader import load_datasets
from fedlearn.strategies import ScaffoldStrategy
from fedlearn.utils import set_parameters, get_parameters

def gen_evaluate_fn(load_dataset_kwargs) -> callable:
    
    def evaluate(
        server_round: int,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
        ) -> Optional[Tuple[float, dict[str, Scalar]]]:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = SmallCNN().to(device)
        _, _, testloader = load_datasets(**load_dataset_kwargs)
        set_parameters(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(
            net=net, 
            device=device, 
            testloader=testloader, 
            criterion=torch.nn.CrossEntropyLoss(),
            )
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        return loss, {"accuracy": accuracy}
    
    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_partitions = context.run_config["num-partitions"]
    method = context.run_config["method"]

    min_fit_clients = min(num_partitions, int(num_partitions * fraction_fit))
    min_evaluate_clients = min(num_partitions, int(num_partitions * fraction_evaluate))

    config = ServerConfig(num_rounds=num_rounds)

    # args for load_datasets
    batch_size = context.run_config["batch-size"]
    partition_method = context.run_config["partition-method"]
    partitioner_kwargs = {
        "num_partitions": num_partitions,
    }
    if partition_method == "dirichlet":
        partitioner_kwargs["alpha"] = context.run_config["dirichlet-alpha"]
        partitioner_kwargs["partition_by"] = "label"
    elif partition_method == "shard":
        partitioner_kwargs["num_shards_per_partition"] = context.run_config["num-shards-per-partition"]
        partitioner_kwargs["partition_by"] = "label"

    # Get cache_dir from run_config, default to "data" in the parent directory
    cache_dir = context.run_config["cache-dir"]
    load_dataset_kwargs = {
        "partition_id": 0,  # Not used in server context, just a placeholder
        "partition_method": partition_method,
        "partitioner_kwargs": partitioner_kwargs,
        "batch_size": batch_size,
        "cache_dir": cache_dir,
    }

    # specify the fedavg strategy
    if method.lower() == "fedavg":
        params = get_parameters(SmallCNN())
        strategy = FedAvg(
            fraction_fit=fraction_fit,                          # Use all clients for training
            fraction_evaluate=fraction_evaluate,                # Use 50% of clients for evaluation
            min_fit_clients=min_fit_clients,                    # Minimum number of clients to train
            min_evaluate_clients=min_evaluate_clients,          # Minimum number of clients to evaluate
            min_available_clients=num_partitions,               # Minimum number of clients available (enforce all clients to be available)
            initial_parameters=ndarrays_to_parameters(params),   # Initial parameters for the model
            evaluate_fn=gen_evaluate_fn(load_dataset_kwargs)    # Pass the evaluation function
        )
        return ServerAppComponents(strategy=strategy, config=config)
    
    # specify the scaffold strategy
    elif method.lower() == "scaffold":
        strategy = ScaffoldStrategy(
            total_num_clients=num_partitions,               # Total number of clients
            fraction_fit=fraction_fit,                      # Use all clients for training, C
            fraction_evaluate=fraction_evaluate,            # Use 50% of clients for evaluation
            min_fit_clients=min_fit_clients,                # Minimum number of clients to train
            min_evaluate_clients=min_evaluate_clients,      # Minimum number of clients to evaluate
            min_available_clients=num_partitions,           # Minimum number of clients available (enforce all clients to be available)
            evaluate_fn=gen_evaluate_fn(load_dataset_kwargs)# Pass the evaluation function
        )

        server = ScaffoldServer(strategy=strategy)

        # Configure the server for 3 rounds of training
        return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

