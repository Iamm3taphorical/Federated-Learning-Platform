"""Noise injection mechanisms for differential privacy.

Provides Gaussian and Laplacian noise injection for model gradients
and parameters to achieve differential privacy guarantees.
"""

from typing import Dict, OrderedDict, Optional, Union
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn


class NoiseInjector(ABC):
    """Abstract base class for noise injection."""
    
    @abstractmethod
    def add_noise(
        self,
        gradients: Union[torch.Tensor, Dict[str, torch.Tensor]],
        sensitivity: float,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Add noise to gradients or parameters.
        
        Args:
            gradients: Tensor or dict of tensors to add noise to
            sensitivity: Sensitivity (e.g., clipping bound)
            
        Returns:
            Noised gradients/parameters
        """
        pass
    
    @abstractmethod
    def compute_epsilon(
        self,
        sensitivity: float,
        delta: float,
    ) -> float:
        """Compute epsilon for the noise mechanism.
        
        Args:
            sensitivity: Sensitivity of the query
            delta: Delta parameter
            
        Returns:
            Epsilon value
        """
        pass


class GaussianNoiseInjector(NoiseInjector):
    """Gaussian noise injection for (ε,δ)-differential privacy.
    
    Adds calibrated Gaussian noise to achieve differential privacy
    with given epsilon and delta parameters.
    """

    def __init__(
        self,
        noise_multiplier: float,
        device: str = "cpu",
    ):
        """Initialize Gaussian noise injector.
        
        Args:
            noise_multiplier: Ratio of noise std to sensitivity
            device: Device for tensor operations
        """
        self.noise_multiplier = noise_multiplier
        self.device = device

    def add_noise(
        self,
        gradients: Union[torch.Tensor, Dict[str, torch.Tensor]],
        sensitivity: float,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Add Gaussian noise to gradients.
        
        Args:
            gradients: Tensor or dict of tensors
            sensitivity: L2 sensitivity (clipping bound)
            
        Returns:
            Noised gradients
        """
        noise_std = sensitivity * self.noise_multiplier
        
        if isinstance(gradients, torch.Tensor):
            noise = torch.normal(
                mean=0,
                std=noise_std,
                size=gradients.shape,
                device=self.device,
            )
            return gradients + noise
        
        # Handle dict of tensors
        noised_grads = {}
        for key, grad in gradients.items():
            if grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=noise_std,
                    size=grad.shape,
                    device=grad.device,
                )
                noised_grads[key] = grad + noise
            else:
                noised_grads[key] = None
        
        return noised_grads

    def add_noise_to_model(
        self,
        model: nn.Module,
        sensitivity: float,
    ) -> None:
        """Add noise directly to model gradients in-place.
        
        Args:
            model: PyTorch model
            sensitivity: L2 sensitivity
        """
        noise_std = sensitivity * self.noise_multiplier
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=noise_std,
                    size=param.grad.shape,
                    device=param.grad.device,
                )
                param.grad.add_(noise)

    def add_noise_to_weights(
        self,
        weights: OrderedDict[str, torch.Tensor],
        sensitivity: float,
    ) -> OrderedDict[str, torch.Tensor]:
        """Add noise to model weights (for aggregation).
        
        Args:
            weights: Model state dict
            sensitivity: Sensitivity for this round
            
        Returns:
            Noised weights
        """
        noise_std = sensitivity * self.noise_multiplier
        noised_weights = OrderedDict()
        
        for key, weight in weights.items():
            noise = torch.normal(
                mean=0,
                std=noise_std,
                size=weight.shape,
                device=weight.device,
            )
            noised_weights[key] = weight + noise
        
        return noised_weights

    def compute_epsilon(
        self,
        sensitivity: float,
        delta: float,
    ) -> float:
        """Compute epsilon for Gaussian mechanism.
        
        Uses the analytic Gaussian mechanism formula.
        
        Args:
            sensitivity: L2 sensitivity
            delta: Delta parameter
            
        Returns:
            Epsilon value
        """
        if delta <= 0 or delta >= 1:
            return float('inf')
        
        sigma = sensitivity * self.noise_multiplier
        
        # Epsilon for Gaussian mechanism
        epsilon = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / sigma
        
        return epsilon


class LaplacianNoiseInjector(NoiseInjector):
    """Laplacian noise injection for ε-differential privacy.
    
    Adds calibrated Laplacian noise to achieve pure differential privacy
    (without delta).
    """

    def __init__(
        self,
        epsilon: float,
        device: str = "cpu",
    ):
        """Initialize Laplacian noise injector.
        
        Args:
            epsilon: Privacy parameter
            device: Device for tensor operations
        """
        self.epsilon = epsilon
        self.device = device

    def add_noise(
        self,
        gradients: Union[torch.Tensor, Dict[str, torch.Tensor]],
        sensitivity: float,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Add Laplacian noise to gradients.
        
        Args:
            gradients: Tensor or dict of tensors
            sensitivity: L1 sensitivity
            
        Returns:
            Noised gradients
        """
        scale = sensitivity / self.epsilon
        
        if isinstance(gradients, torch.Tensor):
            noise = torch.distributions.Laplace(0, scale).sample(gradients.shape)
            noise = noise.to(self.device)
            return gradients + noise
        
        # Handle dict of tensors
        noised_grads = {}
        for key, grad in gradients.items():
            if grad is not None:
                noise = torch.distributions.Laplace(0, scale).sample(grad.shape)
                noise = noise.to(grad.device)
                noised_grads[key] = grad + noise
            else:
                noised_grads[key] = None
        
        return noised_grads

    def compute_epsilon(
        self,
        sensitivity: float,
        delta: float = 0.0,  # Ignored for Laplace
    ) -> float:
        """Compute epsilon (returns the configured epsilon)."""
        return self.epsilon


def compute_gaussian_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sensitivity: float = 1.0,
) -> float:
    """Compute noise multiplier for target epsilon/delta.
    
    Args:
        target_epsilon: Target epsilon
        target_delta: Target delta
        sensitivity: Sensitivity (default 1.0 for normalized gradients)
        
    Returns:
        Required noise multiplier
    """
    # From the Gaussian mechanism analysis
    return sensitivity * math.sqrt(2 * math.log(1.25 / target_delta)) / target_epsilon
