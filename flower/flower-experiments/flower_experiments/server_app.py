"""flower-experiments: A Flower / PyTorch app."""
from typing import Optional, Tuple, List

import torch
import numpy as np

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fedlearn.model import SmallCNN, test
from fedlearn.server import ScaffoldServer
from fedlearn.data_loader import load_datasets
from fedlearn.utils import set_parameters, get_parameters


def gen_evaluate_fn(load_dataset_kwargs) -> callable:
    """
    Generate an evaluation function for the server.
    """
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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Custom aggregation function for evaluation metrics.
    Simply computes a weighted average of the accuracy across clients,
    based on the number of examples each client used for evaluation.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    # read all simulation parameters from context (defined in pyproject.toml)
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_partitions = context.run_config["num-partitions"]
    method = context.run_config["method"]

    # compute minimum number of clients for training and evaluation
    min_fit_clients = min(num_partitions, int(num_partitions * fraction_fit))
    min_evaluate_clients = min(num_partitions, int(num_partitions * fraction_evaluate))

    # args for load_datasets
    batch_size = context.run_config["batch-size"]
    partition_method = context.run_config["partition-method"]
    cache_dir = context.run_config["cache-dir"]
    data_share_fraction = context.run_config["data-sharing-fraction"]

    # define partitioner_kwargs based on the partition_method
    partitioner_kwargs = {
        "num_partitions": num_partitions,
    }
    if partition_method == "dirichlet":
        partitioner_kwargs["alpha"] = context.run_config["dirichlet-alpha"]
        partitioner_kwargs["partition_by"] = "label"
    elif partition_method == "shard":
        partitioner_kwargs["num_shards_per_partition"] = context.run_config["num-shards-per-partition"]
        partitioner_kwargs["partition_by"] = "label"

    # define load_dataset_kwargs for gen_evaluate_fn
    load_dataset_kwargs = {
        "partition_id": 0,  # Not used in server context, just a placeholder
        "partition_method": partition_method,
        "partitioner_kwargs": partitioner_kwargs,
        "batch_size": batch_size,
        "cache_dir": cache_dir,
        "data_share_fraction": data_share_fraction,
    }

    # define initial params and evaluate_fn (if applicable)
    params = None
    evaluate_fn = None

    if method.lower() != "fedper":
        params = ndarrays_to_parameters(get_parameters(SmallCNN()))
        evaluate_fn = gen_evaluate_fn(load_dataset_kwargs)

    # As we are only performing weighted average aggregation, we can use
    # FedAvg for all methods.
    strategy = FedAvg(
            fraction_fit=fraction_fit,                          # Use all clients for training
            fraction_evaluate=fraction_evaluate,                # Use 50% of clients for evaluation
            min_fit_clients=min_fit_clients,                    # Minimum number of clients to train
            min_evaluate_clients=min_evaluate_clients,          # Minimum number of clients to evaluate
            min_available_clients=num_partitions,               # Minimum number of clients available (enforce all clients to be available)
            initial_parameters=params,                          # Initial parameters for the model
            evaluate_fn=evaluate_fn,                            # Pass the evaluation function
            evaluate_metrics_aggregation_fn=weighted_average,   # Custom aggregation function for evaluation metrics
        )
    
    # define server configuration
    config = ServerConfig(num_rounds=num_rounds)

    if method.lower() == "fedavg" or method.lower() == "fedper":
        return ServerAppComponents(strategy=strategy, config=config)
    
    elif method.lower() == "scaffold":
        server = ScaffoldServer(strategy=strategy)
        return ServerAppComponents(server=server, config=config)
    
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'fedavg', 'scaffold', and 'fedper'.")

# Create ServerApp
app = ServerApp(server_fn=server_fn)

