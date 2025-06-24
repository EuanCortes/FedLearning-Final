from typing import List, Tuple, Optional, Union, Dict
from logging import WARNING

from flwr.common import ( 
    Parameters,
    Scalar,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    )

from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from fedlearn.model import SmallCNN
from fedlearn.utils import get_parameters

class ScaffoldStrategy(Strategy):
    def __init__(
        self,
        total_num_clients: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        accept_failures: bool = False,
        fit_metrics_aggregation_fn: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.total_num_clients = total_num_clients
        total_num_clients = total_num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn

        self.evaluate_fn = evaluate_fn


    def __repr__(self) -> str:
        return "ScaffoldStrategy"


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = SmallCNN()
        parameters = get_parameters(net)        
        return ndarrays_to_parameters(parameters)


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {}
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = [(client, fit_ins) for client in clients]
        
        return fit_configurations


    def aggregate_fit(self, 
                      server_round: int, 
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        aggregation method for Scaffold strategy.
        """
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        combined_parameters = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        len_combined_parameters = len(combined_parameters[0]) # combined number of model parameters and control variates

        num_samples_all = [fit_res.num_examples for _, fit_res in results]  # number of training samples from each client

        # The "aggregate()" function expects a list of tuples, where each tuple contains
        # the local parameters and the number of samples for that client.
        #aggregation_inputs_parameters = [
        #    (local_params[:len_combined_parameters // 2], num_samples) 
        #    for local_params, num_samples in zip(combined_parameters, num_samples_all)
        #]
        
        #parameters_aggregated = aggregate(aggregation_inputs_parameters)

        #aggregation_inputs_cv = [
        #    (local_params[len_combined_parameters // 2:], num_samples) 
        #    for local_params, num_samples in zip(combined_parameters, num_samples_all)
        #]

        #cv_aggregated = aggregate(aggregation_inputs_cv)

        aggregation_inputs = [
            (local_params, num_samples) 
            for local_params, num_samples in zip(combined_parameters, num_samples_all)
        ]

        parameters_aggregated = aggregate(aggregation_inputs)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics)
                for _, fit_res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        

        return (
            #ndarrays_to_parameters(parameters_aggregated + cv_aggregated),
            ndarrays_to_parameters(parameters_aggregated),
            metrics_aggregated,
        )


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated


    # method for evaluating the global model
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            return None
            # If an evaluation function is provided, use it
        parameters_ndarray = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarray, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


    # boilerplate code
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    # boilerplate code
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


