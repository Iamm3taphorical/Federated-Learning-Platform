"""Base Flower client for federated learning.

Provides the foundation for hospital clients with local training,
evaluation, and differential privacy support.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    Status,
    Code,
)

from models.model_utils import get_model_parameters, set_model_parameters
from privacy.dp_sgd import DPSGDHandler, DPConfig, ManualDPSGD
from privacy.gradient_clipping import GradientClipper


class BaseClient(fl.client.NumPyClient):
    """Base federated learning client.
    
    Implements the Flower NumPyClient interface with support for:
    - Local training with configurable epochs
    - Differential privacy via Opacus or manual DP-SGD
    - Gradient clipping
    - Training metrics collection
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        enable_dp: bool = True,
        max_grad_norm: float = 1.0,
    ):
        """Initialize base client.
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            device: Device for training
            enable_dp: Enable differential privacy
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.enable_dp = enable_dp
        self.max_grad_norm = max_grad_norm
        
        # Training state
        self.current_round = 0
        self.training_history: List[Dict[str, float]] = []
        
        # DP components
        self.gradient_clipper = GradientClipper(max_grad_norm=max_grad_norm)
        self.manual_dp = ManualDPSGD(
            model=model,
            max_grad_norm=max_grad_norm,
            noise_multiplier=1.1,
            device=device,
        ) if enable_dp else None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get current model parameters.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of numpy arrays (model parameters)
        """
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters.
        
        Args:
            parameters: List of numpy arrays
        """
        set_model_parameters(self.model, parameters)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model on local data.
        
        Args:
            parameters: Current global model parameters
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Get training config
        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 32))
        learning_rate = float(config.get("learning_rate", 0.001))
        self.current_round = int(config.get("round", 0))
        
        # Train locally
        start_time = time.time()
        metrics = self._train(
            epochs=local_epochs,
            learning_rate=learning_rate,
        )
        training_time = time.time() - start_time
        
        # Add training metadata
        metrics["training_time"] = training_time
        metrics["client_id"] = self.client_id
        metrics["round"] = self.current_round
        
        # Store in history
        self.training_history.append(metrics)
        
        # Return updated parameters
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            metrics,
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        
        # Use validation loader if available, else train loader
        eval_loader = self.val_loader or self.train_loader
        
        loss, accuracy, metrics = self._evaluate(eval_loader)
        
        return (
            loss,
            len(eval_loader.dataset),
            {"accuracy": accuracy, **metrics},
        )

    def _train(
        self,
        epochs: int = 1,
        learning_rate: float = 0.001,
    ) -> Dict[str, float]:
        """Perform local training.
        
        Args:
            epochs: Number of local epochs
            learning_rate: Learning rate
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        grad_norms = []
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient clipping
                if self.enable_dp:
                    original_norm = self.gradient_clipper.clip_gradients(self.model)
                    grad_norms.append(original_norm)
                
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total,
            "num_batches": num_batches,
        }
        
        if grad_norms:
            metrics["avg_grad_norm"] = sum(grad_norms) / len(grad_norms)
            metrics["max_grad_norm"] = max(grad_norms)
        
        return metrics

    def _evaluate(
        self,
        data_loader: DataLoader,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Evaluate model.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Tuple of (loss, accuracy, additional_metrics)
        """
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = criterion(output, target)
                total_loss += loss.item() * target.size(0)
                
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy, {"num_samples": total}
