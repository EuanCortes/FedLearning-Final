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
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k : torch.tensor(v).to(torch.float32) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def concat_params(parameters: Parameters, global_cv: List[np.ndarray]) -> Parameters:
    """
    Concatenate model parameters and global control variates.
    """
    parameters_ndarrays = parameters_to_ndarrays(parameters)
    parameters_ndarrays.extend(global_cv)
    return ndarrays_to_parameters(parameters_ndarrays)
