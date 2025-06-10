from typing import Tuple, Optional, Dict, Union

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ShardPartitioner
from flwr_datasets import FederatedDataset

fds = None  # Global variable to hold the FederatedDataset instance
partitioner = None  # Global variable to hold the partitioner instance

def load_datasets(partition_id: int, 
                  partition_method: str,
                  partitioner_kwargs: Dict[str, Union[str, int, float]],
                  batch_size: int = 64,
                  cache_dir: str = "data",
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    function for loading CIFAR-10 dataset and partitioning it for federated learning.
    Parameters:
        partition_id:   int, the ID of the partition to load.
        num_partitions: int, the total number of partitions to create.
        batch_size:     int, the size of each batch for training and validation.
        cache_dir:      str, the directory to cache the dataset.
        partitioner:    Optional[Partitioner], the partitioner to use for creating partitions.
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: trainloader, valloader, and testloader.
    """
    # Only initialize `FederatedDataset` and partitioner once
    global fds, partitioner
    if partitioner is None:
        partitioner_constructor = None
        if partition_method == "iid":
            partitioner_constructor = IidPartitioner
        elif partition_method == "dirichlet":
            partitioner_constructor = DirichletPartitioner
        elif partition_method == "shard":
            partitioner_constructor = ShardPartitioner
        
        if partitioner_constructor is None:
            raise ValueError(f"Unknown partitioner method: {partition_method}")
        
        partitioner = partitioner_constructor(**partitioner_kwargs)

    if fds is None:
        fds = FederatedDataset(dataset="cifar10", 
                               partitioners={"train": partitioner}, 
                               cache_dir=cache_dir,
                               )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, valloader, testloader
