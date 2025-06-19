"""scaffold: A Flower / PyTorch app."""

import os
from typing import List, Tuple, Optional

import torch

import numpy as np

from flwr.client import NumPyClient
from fedlearn.utils import get_parameters, set_parameters
from fedlearn.model import train, train_scaffold, test


class FlowerClient(NumPyClient):
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
                 device: Optional[int]
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
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)


    #def get_parameters(self, config):
    #    print(f"[Client {self.partition_id}] get_parameters")
    #    return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
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
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(
            net=self.net,
            device=self.device,
            testloader=self.valloader,
            criterion=self.criterion
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


class ScaffoldClient(NumPyClient):
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
                 save_dir: Optional[str],
                 device: Optional[int]
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

        # define directory to save client control variates
        if save_dir is None:
            save_dir = "client_cvs"

        # create directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # define the path to save the client control variates
        self.save_name = os.path.join(save_dir, f"client_{self.partition_id}_cv.pt")

        # initialize client control variates
        self.client_cv = [torch.zeros(param.shape).to(torch.float32) for param in self.net.state_dict().values()]


    # Here is where all the training logic and control variate updates happen
    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:

        # the global parameters are packed together with the global control variates
        # in the form [params, global_cv]. we start by separating them
        params = parameters[:len(parameters) // 2]          # list of np.ndarray
        global_cv = parameters[len(parameters) // 2:]       # list of np.ndarray

        # load the current global model:
        set_parameters(self.net, params)

        # load client control variates, if they exist:
        if os.path.exists(self.save_name):
            self.client_cv = torch.load(self.save_name)     # list of torch.tensor

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
            global_cv=global_cv_torch,          # passing list of torch.tensor
            client_cv=self.client_cv            # passing list of torch.tensor
        )

        # update the client control variates
        yi = get_parameters(self.net)           # list of np.ndarray

        # compute coefficient for the control variates
        # 1 / (K * eta) where K is the number of backward passes (num_epochs * len(trainloader))
        coeff = 1. / (self.num_epochs * len(self.trainloader) * self.lr) 

        client_cv = [cv.numpy() for cv in self.client_cv]  # list of np.ndarray

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

        # convert client cvs back to torch tensors
        self.client_cv = [torch.tensor(cv).to(torch.float32) for cv in client_cv_new]  

        # save the updated client control variates
        torch.save(self.client_cv, self.save_name)

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
    

