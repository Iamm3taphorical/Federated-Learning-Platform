"""Federated Learning Server implementation.

Provides the central federation server using Flower framework
with differential privacy integration and database logging.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
from collections import OrderedDict
from datetime import datetime
import uuid
import os

import numpy as np

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from config.settings import get_settings
from models.medical_cnn import create_model
from models.model_utils import get_model_parameters, set_model_parameters
from privacy.privacy_accountant import FederatedPrivacyAccountant


class FederatedServer:
    """Federated Learning Server for medical diagnosis.
    
    Orchestrates federated training across hospital clients with
    differential privacy and database logging.
    """

    def __init__(
        self,
        model_architecture: str = "MedicalCNN",
        num_classes: int = 14,
        in_channels: int = 1,
        server_address: str = "0.0.0.0:8080",
        num_rounds: int = 10,
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        enable_dp: bool = True,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        model_id: Optional[str] = None,
        weights_dir: str = "./weights",
    ):
        """Initialize federated server.
        
        Args:
            model_architecture: Model architecture to use
            num_classes: Number of output classes
            in_channels: Number of input channels
            server_address: Server address for Flower
            num_rounds: Number of training rounds
            min_clients: Minimum clients required
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            enable_dp: Enable differential privacy
            target_epsilon: Target epsilon for DP
            target_delta: Target delta for DP
            noise_multiplier: Noise multiplier for DP
            max_grad_norm: Maximum gradient norm
            model_id: Optional model ID for tracking
            weights_dir: Directory for saving weights
        """
        self.server_address = server_address
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        
        # Privacy settings
        self.enable_dp = enable_dp
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        # Create model
        self.model = create_model(
            architecture=model_architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            use_dp_compatible=enable_dp,
        )
        
        # Model tracking
        self.model_id = model_id or str(uuid.uuid4())
        self.current_round = 0
        self.weights_dir = weights_dir
        os.makedirs(weights_dir, exist_ok=True)
        
        # Privacy accountant
        self.privacy_accountant = FederatedPrivacyAccountant(
            client_sampling_rate=fraction_fit,
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "epsilon": [],
        }

    def get_initial_parameters(self) -> Parameters:
        """Get initial model parameters.
        
        Returns:
            Flower Parameters object
        """
        return ndarrays_to_parameters(get_model_parameters(self.model))

    def create_strategy(self) -> fl.server.strategy.Strategy:
        """Create the aggregation strategy.
        
        Returns:
            Flower Strategy instance
        """
        from server.strategies.fed_avg import DPFedAvg
        
        return DPFedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            initial_parameters=self.get_initial_parameters(),
            enable_dp=self.enable_dp,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            on_fit_config_fn=self._get_fit_config,
            on_evaluate_config_fn=self._get_evaluate_config,
            fit_metrics_aggregation_fn=self._aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=self._aggregate_evaluate_metrics,
        )

    def _get_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """Get configuration for fit round.
        
        Args:
            server_round: Current server round
            
        Returns:
            Configuration dictionary
        """
        settings = get_settings()
        return {
            "round": server_round,
            "local_epochs": settings.local_epochs,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "enable_dp": self.enable_dp,
        }

    def _get_evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        """Get configuration for evaluate round.
        
        Args:
            server_round: Current server round
            
        Returns:
            Configuration dictionary
        """
        return {
            "round": server_round,
        }

    def _aggregate_fit_metrics(
        self,
        metrics: List[Tuple[int, Dict[str, Scalar]]],
    ) -> Dict[str, Scalar]:
        """Aggregate fit metrics from clients.
        
        Args:
            metrics: List of (num_examples, metrics_dict) tuples
            
        Returns:
            Aggregated metrics
        """
        if not metrics:
            return {}
        
        total_examples = sum(num for num, _ in metrics)
        
        aggregated = {}
        for key in metrics[0][1].keys():
            weighted_sum = sum(
                num * m[key] for num, m in metrics if key in m
            )
            aggregated[key] = weighted_sum / total_examples
        
        return aggregated

    def _aggregate_evaluate_metrics(
        self,
        metrics: List[Tuple[int, Dict[str, Scalar]]],
    ) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics from clients.
        
        Args:
            metrics: List of (num_examples, metrics_dict) tuples
            
        Returns:
            Aggregated metrics
        """
        if not metrics:
            return {}
        
        total_examples = sum(num for num, _ in metrics)
        
        aggregated = {}
        for key in metrics[0][1].keys():
            weighted_sum = sum(
                num * m[key] for num, m in metrics if key in m
            )
            aggregated[key] = weighted_sum / total_examples
        
        # Track in history
        if "accuracy" in aggregated:
            self.history["accuracy"].append(aggregated["accuracy"])
        if "loss" in aggregated:
            self.history["loss"].append(aggregated["loss"])
        
        # Track privacy
        epsilon = self.privacy_accountant.get_cumulative_epsilon(self.target_delta)
        aggregated["cumulative_epsilon"] = epsilon
        self.history["epsilon"].append(epsilon)
        
        return aggregated

    def save_model(self, round_num: int) -> str:
        """Save model weights.
        
        Args:
            round_num: Current round number
            
        Returns:
            Path to saved weights
        """
        import torch
        
        path = os.path.join(
            self.weights_dir,
            f"model_{self.model_id}_round{round_num}.pt"
        )
        torch.save({
            "state_dict": self.model.state_dict(),
            "round": round_num,
            "model_id": self.model_id,
            "epsilon": self.privacy_accountant.get_cumulative_epsilon(self.target_delta),
        }, path)
        
        return path

    def start(self) -> Dict[str, Any]:
        """Start the federated learning server.
        
        Returns:
            Training history and final metrics
        """
        print(f"Starting Federated Server on {self.server_address}")
        print(f"Model ID: {self.model_id}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Minimum clients: {self.min_clients}")
        print(f"Differential Privacy: {'Enabled' if self.enable_dp else 'Disabled'}")
        
        if self.enable_dp:
            print(f"  Target ε: {self.target_epsilon}")
            print(f"  Target δ: {self.target_delta}")
            print(f"  Noise multiplier: {self.noise_multiplier}")
            print(f"  Max grad norm: {self.max_grad_norm}")
        
        # Create strategy
        strategy = self.create_strategy()
        
        # Start server
        history = fl.server.start_server(
            server_address=self.server_address,
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )
        
        # Save final model
        final_path = self.save_model(self.num_rounds)
        print(f"Final model saved to: {final_path}")
        
        # Get final privacy budget
        final_epsilon = self.privacy_accountant.get_cumulative_epsilon(self.target_delta)
        
        return {
            "history": self.history,
            "final_epsilon": final_epsilon,
            "final_delta": self.target_delta,
            "model_path": final_path,
            "model_id": self.model_id,
        }


def start_server(
    model_architecture: str = None,
    num_classes: int = None,
    num_rounds: int = None,
    min_clients: int = None,
    enable_dp: bool = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to start the federated server.
    
    Uses settings from environment/config with optional overrides.
    
    Returns:
        Training results
    """
    settings = get_settings()
    
    server = FederatedServer(
        model_architecture=model_architecture or settings.default_model_architecture,
        num_classes=num_classes or 14,
        server_address=settings.fl_server_address,
        num_rounds=num_rounds or settings.num_rounds,
        min_clients=min_clients or settings.min_clients,
        fraction_fit=settings.fraction_fit,
        fraction_evaluate=settings.fraction_evaluate,
        enable_dp=enable_dp if enable_dp is not None else settings.enable_dp,
        target_epsilon=settings.target_epsilon,
        target_delta=settings.target_delta,
        noise_multiplier=settings.noise_multiplier,
        max_grad_norm=settings.max_grad_norm,
        weights_dir=settings.weights_dir,
        **kwargs,
    )
    
    return server.start()
