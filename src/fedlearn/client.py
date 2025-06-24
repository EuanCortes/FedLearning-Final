from typing import List, Tuple, Optional

import torch

import numpy as np

from flwr.client import NumPyClient
from flwr.common import Context, ArrayRecord
from fedlearn.utils import get_parameters, set_parameters
from fedlearn.model import train, train_scaffold, test


class FlowerClient(NumPyClient):
    """
    A simple Flower client for federated learning. This is essentially a 
    copy of the FlowerClient from the flower tutorials.
    """
    def __init__(self, 
                 partition_id: int, 
                 net: torch.nn.Module, 
                 trainloader: torch.utils.data.DataLoader, 
                 valloader: torch.utils.data.DataLoader,
                 criterion: torch.nn.Module,
                 num_epochs: int,
                 lr: float,
                 momentum: float,
                 weight_decay: float,
                 device: Optional[int] = None
                 ):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        if device is not None:
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)


    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, 
              self.device,
              self.trainloader, 
              self.criterion,
              self.num_epochs,
              self.lr,
              self.momentum,
              self.weight_decay,
              )
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(
            net=self.net,
            device=self.device,
            testloader=self.valloader,
            criterion=self.criterion
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


class ScaffoldClient(NumPyClient):
    """
    A Flower client that implements the Scaffold algorithm for federated learning.
    """
    def __init__(self, 
                 partition_id: int, 
                 net: torch.nn.Module, 
                 trainloader: torch.utils.data.DataLoader, 
                 valloader: torch.utils.data.DataLoader,
                 criterion: torch.nn.Module,
                 num_epochs: int,
                 lr: float,
                 momentum: float,
                 weight_decay: float,
                 context: Context,
                 device: Optional[int] = None
                 ):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        if device is not None:
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.client_state = context.state.array_records
        self.client_cv_header = "client_cv"


    def get_parameters(self, config: dict) -> List[np.ndarray]:
        return get_parameters(self.net)
    

    # Here is where all the training logic and control variate updates happen
    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:

        # the global parameters are packed together with the global control variates
        # in the form [params, global_cv]. we start by separating them
        params = parameters[:len(parameters) // 2]          # list of np.ndarray
        global_cv = parameters[len(parameters) // 2:]       # list of np.ndarray

        # load the current global model:
        set_parameters(self.net, params)

        # load client control variates, if they exist:
        if self.client_cv_header in self.client_state:
            client_cv = self.client_state[self.client_cv_header].to_numpy_ndarrays()    # list of np.ndarray
        else:
            # if no client control variates exist, initialize them to zero arrays
            client_cv = [np.zeros_like(p) for p in params]  # list of np.ndarray

        client_cv_torch = [torch.tensor(cv).to(torch.float32) for cv in client_cv]  # list of torch.tensor

        # convert global control variates to tensors
        global_cv_torch = [torch.tensor(cv).to(torch.float32) for cv in global_cv]  # list of torch.tensor

        # call the training function
        train_scaffold(
            net=self.net,
            device=self.device,
            trainloader=self.trainloader,
            criterion=self.criterion,
            num_epochs=self.num_epochs,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            global_cv=global_cv_torch,           # passing list of torch.tensor
            client_cv=client_cv_torch            # passing list of torch.tensor
        )

        # update the client control variates
        yi = get_parameters(self.net)           # list of np.ndarray

        # compute coefficient for the control variates
        # 1 / (K * eta) where K is the number of backward passes (num_epochs * len(trainloader))
        coeff = 1. / (self.num_epochs * len(self.trainloader) * self.lr) 

        # define new list for udated client control variates
        client_cv_new = []

        # compute client control variate update, list of np.ndarray
        for xj, yj, cj, cij in zip(params, yi, global_cv, client_cv):
            client_cv_new.append(
                cij - cj + coeff * (xj - yj)
            ) 

        # compute server updates
        server_update_x = [yj - xj for xj, yj in zip(params, yi)]
        server_update_c = [cij_n - cij for cij_n, cij in zip(client_cv_new, client_cv)]

        self.client_state[self.client_cv_header] = ArrayRecord(client_cv_new)

        #concatenate server updates
        server_update = server_update_x + server_update_c

        return server_update, len(self.trainloader.dataset), {}



    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        set_parameters(self.net, parameters)
        avg_loss, accuracy = test(
            net=self.net,
            device=self.device,
            testloader=self.valloader,
            criterion=self.criterion
        )
        return float(avg_loss), len(self.valloader), {"accuracy": accuracy}
    


class FedPerClient(NumPyClient):
    """
    A Flower client that implements the FedPer algorithm for federated learning.
    """
    def __init__(self, 
                 partition_id: int, 
                 net: torch.nn.Module, 
                 trainloader: torch.utils.data.DataLoader, 
                 valloader: torch.utils.data.DataLoader,
                 criterion: torch.nn.Module,
                 num_epochs: int,
                 lr: float,
                 momentum: float,
                 weight_decay: float,
                 context: Context,
                 device: Optional[int] = None,
                 ):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        if device is not None:
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Define the shared parameters (in this case all parameters except the fully connected layers)
        self.shared_parameter_names = [
            name for name in self.net.state_dict().keys() 
            if "fc" not in name
        ]

        # get the client state from the context
        self.client_state = context.state.array_records
        self.client_parameters_header = "client_parameters"


    def _get_shared_parameters(self) -> List[np.ndarray]:
        """Get the shared parameters of the model."""
        return [
            val.cpu().numpy() 
            for name, val in self.net.state_dict().items()
            if name in self.shared_parameter_names
        ]
    

    def _set_shared_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set the shared parameters of the model."""
        if self.client_parameters_header in self.client_state:
            current_parameters = self.client_state[self.client_parameters_header].to_torch_state_dict()
        else:
            current_parameters = self.net.state_dict()
        # Update only the shared parameters
        for name, param in zip(self.shared_parameter_names, parameters):
            current_parameters[name] = torch.tensor(param)

        self.net.load_state_dict(current_parameters, strict=True)


    def get_parameters(self, config: dict) -> List[np.ndarray]:
        return self._get_shared_parameters()

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self._set_shared_parameters(parameters)
        train(self.net, 
              self.device,
              self.trainloader, 
              self.criterion,
              self.num_epochs,
              self.lr,
              self.momentum,
              self.weight_decay,
              )
        
        self.client_state[self.client_parameters_header] = ArrayRecord(
            self.net.state_dict()
        )

        return self._get_shared_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self._set_shared_parameters(parameters)
        loss, accuracy = test(
            net=self.net,
            device=self.device,
            testloader=self.valloader,
            criterion=self.criterion
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
