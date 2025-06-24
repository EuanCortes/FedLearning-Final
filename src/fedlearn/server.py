from typing import List, Tuple, Optional, Dict
from logging import WARNING, INFO, DEBUG

import numpy as np

from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    )

from flwr.server import Server
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log
from flwr.server.server import FitResultsAndFailures, fit_clients

from fedlearn.utils import concat_params

class ScaffoldServer(Server):
    """
    Server implementation for the SCAFFOLD algorithm
    """

    def __init__(
            self, 
            strategy: Strategy, 
            client_manager: Optional[ClientManager] = None,
        ) -> None:
        
        if client_manager is None:
            client_manager = SimpleClientManager()

        super().__init__(strategy=strategy, client_manager=client_manager)
        
        self.global_cv: List[np.ndarray] = []  # Global control variates for Scaffold
    
    def _get_initial_parameters(
            self, 
            server_round: int, 
            timeout: Optional[float]
        ) -> Parameters: 
        
        parameters = self.strategy.initialize_parameters(self.client_manager)

        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")

            self.global_cv = [
                np.zeros_like(param, dtype=np.float32) for param in parameters_to_ndarrays(parameters)
            ]

            return parameters
        
        log(WARNING, "No initial parameters provided by strategy, shutting down")
        self.disconnect_all_clients()


    def fit_round(
            self,
            server_round: int,
            timeout: Optional[float],
        ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        
        # define client instructions to be passed to "fit_clients" function
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=concat_params(self.parameters, self.global_cv),
            client_manager=self._client_manager,
        )

        # if no clients are selected, return None
        if not client_instructions:
            log(INFO, f"fit_round {server_round}: no clients selected.")
            return None
        
        log(
            DEBUG,
            f"fit_round {server_round}: selected {len(client_instructions)} clients.",
        )

        # Call the "fit_clients" function from flwr.server.server
        # to perform the training on selected clients
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )

        log(DEBUG,
            f"fit_round {server_round}: received {len(results)} results and {len(failures)} failures.",
        )

        # Aggregate the results from the clients
        aggregated_results = self.strategy.aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )

        # Extract the aggregated parameters and control variates
        aggregated_results_combined = []
        if aggregated_results[0] is not None:
            aggregated_results_combined = parameters_to_ndarrays(aggregated_results[0])

        # Split the aggregated results into model parameters and control variates
        aggregated_parameters = aggregated_results_combined[:len(aggregated_results_combined) // 2] # model parameters
        aggregated_cv = aggregated_results_combined[len(aggregated_results_combined) // 2:]         # control variates

        # define the update coefficient for the control variates
        cv_coeff = len(results) / len(self._client_manager.all())

        # Update the global control variates according to
        # global_cv <- global_cv + cv_coeff * aggregated_cv
        # where cv_coeff = |S| / N, |S| is the number of clients that participated in the round
        # and aggregated_cv = (1 / |S|) * sum_{i in S} (c_i^+ - c_i)
        self.global_cv = [
            cv + cv_coeff * new_cv for cv, new_cv in zip(self.global_cv, aggregated_cv)
        ]


        # Update the global model parameters
        # new_parameters = current_parameters + aggregated_parameters
        # where current_parameters are the parameters of the global model before the round
        # and aggregated_parameters = (1 / |S|) * sum_{i in S} (w_i^+ - w)
        current_parameters = parameters_to_ndarrays(self.parameters)
        new_parameters = [
            param + update for param, update in zip(current_parameters, aggregated_parameters)
        ]

        new_parameters = ndarrays_to_parameters(new_parameters)

        return new_parameters, aggregated_results[1], (results, failures)
