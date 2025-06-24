from typing import List
from collections import OrderedDict
import numpy as np
import torch    
from flwr.common import (
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)


# define helper functions to set and get model parameters
def get_parameters(net) -> List[np.ndarray]:
    """
    Retrieve model parameters from a PyTorch model and convert them to numpy arrays.

    Args:
        net (torch.nn.Module): The PyTorch model from which to retrieve parameters.

    Returns:
        List[np.ndarray]: A list of numpy arrays representing the model parameters.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set model parameters for a PyTorch model from numpy arrays.

    Args:
        net (torch.nn.Module): The PyTorch model to set parameters for.
        parameters (List[np.ndarray]): A list of numpy arrays representing the model parameters.

    Returns:
        None
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k : torch.tensor(v).to(torch.float32) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def concat_params(parameters: Parameters, global_cv: List[np.ndarray]) -> Parameters:
    """
    Concatenate model parameters and global control variates.

    Args:
        parameters (Parameters): The model parameters to concatenate.
        global_cv (List[np.ndarray]): The global control variates to concatenate.

    Returns:
        Parameters: A new Parameters object containing the concatenated parameters and global control variates.
    """
    parameters_ndarrays = parameters_to_ndarrays(parameters)
    parameters_ndarrays.extend(global_cv)
    return ndarrays_to_parameters(parameters_ndarrays)
