"""Federated Averaging strategy with Differential Privacy.

Implements FedAvg with server-side noise injection for differential
privacy guarantees in federated learning.
"""

from typing import Optional, List, Tuple, Dict, Callable, Union
from collections import OrderedDict

import numpy as np

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from privacy.noise_injection import GaussianNoiseInjector
from privacy.gradient_clipping import GradientClipper


class DPFedAvg(FedAvg):
    """Federated Averaging with Differential Privacy.
    
    Extends Flower's FedAvg strategy with:
    - Server-side gradient clipping
    - Gaussian noise injection for DP
    - Privacy budget tracking
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
        evaluate_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
        enable_dp: bool = True,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
    ):
        """Initialize DPFedAvg strategy.
        
        Args:
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
            evaluate_fn: Optional server-side evaluation function
            on_fit_config_fn: Function to configure fit rounds
            on_evaluate_config_fn: Function to configure evaluate rounds
            accept_failures: Whether to accept client failures
            initial_parameters: Initial model parameters
            fit_metrics_aggregation_fn: Function to aggregate fit metrics
            evaluate_metrics_aggregation_fn: Function to aggregate eval metrics
            enable_dp: Enable differential privacy
            noise_multiplier: Noise multiplier for DP
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.enable_dp = enable_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        # Privacy components
        if enable_dp:
            self.noise_injector = GaussianNoiseInjector(
                noise_multiplier=noise_multiplier,
                device="cpu",
            )
            self.gradient_clipper = GradientClipper(
                max_grad_norm=max_grad_norm,
            )
        
        # Track current global parameters
        self.current_parameters: Optional[NDArrays] = None
        if initial_parameters is not None:
            self.current_parameters = parameters_to_ndarrays(initial_parameters)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with DP.
        
        Args:
            server_round: Current server round
            results: List of successful client results
            failures: List of failed client results
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            return None, {}
        
        # Extract weights and sample counts
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        if self.enable_dp and self.current_parameters is not None:
            # Compute model deltas (updates)
            deltas = []
            for weights, num_examples in weights_results:
                delta = [
                    new - old
                    for new, old in zip(weights, self.current_parameters)
                ]
                deltas.append((delta, num_examples))
            
            # Clip deltas
            clipped_deltas = []
            for delta, num_examples in deltas:
                # Compute L2 norm of delta
                total_norm = np.sqrt(sum(np.sum(d ** 2) for d in delta))
                
                # Clip if necessary
                clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-6))
                clipped = [d * clip_factor for d in delta]
                clipped_deltas.append((clipped, num_examples))
            
            # Aggregate clipped deltas
            total_examples = sum(num for _, num in clipped_deltas)
            aggregated_delta = [
                np.zeros_like(d) for d in clipped_deltas[0][0]
            ]
            
            for delta, num_examples in clipped_deltas:
                weight = num_examples / total_examples
                for i, d in enumerate(delta):
                    aggregated_delta[i] += weight * d
            
            # Add noise to aggregated delta
            noise_std = self.max_grad_norm * self.noise_multiplier / len(results)
            noised_delta = [
                d + np.random.normal(0, noise_std, d.shape)
                for d in aggregated_delta
            ]
            
            # Apply noised delta to get new parameters
            aggregated_ndarrays = [
                old + delta
                for old, delta in zip(self.current_parameters, noised_delta)
            ]
        else:
            # Standard FedAvg without DP
            total_examples = sum(num for _, num in weights_results)
            aggregated_ndarrays = [
                np.zeros_like(weights_results[0][0][i])
                for i in range(len(weights_results[0][0]))
            ]
            
            for weights, num_examples in weights_results:
                weight = num_examples / total_examples
                for i, w in enumerate(weights):
                    aggregated_ndarrays[i] += weight * w
        
        # Update current parameters
        self.current_parameters = aggregated_ndarrays
        
        # Convert to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        
        # Add DP info to metrics
        if self.enable_dp:
            metrics_aggregated["dp_noise_multiplier"] = self.noise_multiplier
            metrics_aggregated["dp_max_grad_norm"] = self.max_grad_norm
            metrics_aggregated["num_clients"] = len(results)
        
        return parameters_aggregated, metrics_aggregated

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure fit round.
        
        Args:
            server_round: Current server round
            parameters: Current global parameters
            client_manager: Client manager
            
        Returns:
            List of (client, fit_instructions) tuples
        """
        # Update current parameters
        self.current_parameters = parameters_to_ndarrays(parameters)
        
        # Use parent implementation
        return super().configure_fit(server_round, parameters, client_manager)


class SecureAggregationStrategy(DPFedAvg):
    """Strategy with secure aggregation support.
    
    Extends DPFedAvg with placeholder for secure aggregation protocol.
    In production, this would use cryptographic protocols like SMPC.
    """

    def __init__(
        self,
        *args,
        enable_secure_agg: bool = False,
        **kwargs,
    ):
        """Initialize with secure aggregation.
        
        Args:
            enable_secure_agg: Enable secure aggregation
            *args: Arguments for DPFedAvg
            **kwargs: Keyword arguments for DPFedAvg
        """
        super().__init__(*args, **kwargs)
        self.enable_secure_agg = enable_secure_agg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with optional secure aggregation.
        
        Note: This is a placeholder. Real secure aggregation would use
        cryptographic protocols like secret sharing or homomorphic encryption.
        """
        if self.enable_secure_agg:
            # In production, this would:
            # 1. Collect encrypted updates from clients
            # 2. Perform secure sum using SMPC or HE
            # 3. Return aggregated result without learning individual updates
            pass
        
        # Fall back to DP aggregation
        return super().aggregate_fit(server_round, results, failures)
