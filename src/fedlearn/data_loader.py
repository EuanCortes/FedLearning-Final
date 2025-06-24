from typing import Tuple, Dict, Union

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ShardPartitioner
from flwr_datasets import FederatedDataset

fds = None  # Global variable to hold the FederatedDataset instance
partitioner = None  # Global variable to hold the partitioner instance
p_method = None

def load_datasets(partition_id: int, 
                  partition_method: str,
                  partitioner_kwargs: Dict[str, Union[str, int, float]],
                  batch_size: int = 64,
                  cache_dir: str = "data",
                  data_share_fraction: float = 0.0,
                  data_share_seed: int = 42,
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    function for loading CIFAR-10 dataset and partitioning it for federated learning.

    Parameters:
        partition_id:           int, the ID of the partition to load.
        partition_method:       str, the method to use for partitioning the dataset.
        partitioner_kwargs:     Dict[str, Union[str, int, float]], the parameters for the partitioner.
        batch_size:             int, the size of each batch for training and validation.
        cache_dir:              str, the directory to cache the dataset.
        data_share_fraction:    Optional[float], fraction of the global (test) dataset to share with each client.
        data_share_seed:        Optional[int], seed for random splitting of the dataset.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: trainloader, valloader, and testloader.
    """
    # Only initialize `FederatedDataset` and partitioner once
    global fds, partitioner, p_method
    if partitioner is None or p_method != partition_method:
        partitioner_constructor = None

        # choose correct partitioner
        if partition_method == "iid":
            partitioner_constructor = IidPartitioner
        elif partition_method == "dirichlet":
            partitioner_constructor = DirichletPartitioner
        elif partition_method == "shard":
            partitioner_constructor = ShardPartitioner
        
        if partitioner_constructor is None:
            raise ValueError(f"Unknown partitioner method: {partition_method}")
        
        # Initialize partitioner with the provided kwargs
        partitioner = partitioner_constructor(**partitioner_kwargs)
        p_method = partition_method

        # Initialize FederatedDataset with the partitioner
        fds = FederatedDataset(
            dataset="cifar10", 
            partitioners={"train": partitioner}, 
            cache_dir=cache_dir,
        )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # load the partition using the partition_id
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Define the transforms to apply to the images
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Apply the transforms to the partition
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    # split into local train and test sets
    local_trainset = partition_train_test["train"]
    local_testset = partition_train_test["test"]

    # load the global test partition
    testset = fds.load_split("test").with_transform(apply_transforms)

    # Datasharing: share a fraction of the global dataset with each client if specified
    if data_share_fraction > 0.0:
        generator = torch.Generator().manual_seed(data_share_seed)
        testset, shareset = random_split(testset, [1-data_share_fraction, data_share_fraction], generator=generator)
        local_trainset = ConcatDataset([local_trainset, shareset])

    # return pytorch dataloaders
    trainloader = DataLoader(local_trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(local_testset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloader, valloader, testloader
