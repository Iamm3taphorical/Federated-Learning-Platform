"""Differential Privacy SGD (DP-SGD) implementation.

Provides mechanisms for training models with differential privacy guarantees
using gradient clipping and noise addition via Opacus.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants import RDPAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None


@dataclass
class DPConfig:
    """Configuration for differential privacy training."""
    
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    epochs: int = 1
    batch_size: int = 32
    sample_rate: Optional[float] = None


class DPSGDHandler:
    """Handler for DP-SGD training with Opacus.
    
    This class wraps Opacus PrivacyEngine to provide a simple interface
    for training models with differential privacy guarantees.
    
    Example:
        >>> dp_handler = DPSGDHandler(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     data_loader=train_loader,
        ...     config=DPConfig(target_epsilon=8.0)
        ... )
        >>> private_model, private_optimizer, private_loader = dp_handler.make_private()
        >>> # Train using private_model, private_optimizer, private_loader
        >>> epsilon = dp_handler.get_epsilon()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        config: Optional[DPConfig] = None,
    ):
        """Initialize DP-SGD handler.
        
        Args:
            model: PyTorch model to make private
            optimizer: Optimizer for training
            data_loader: Training data loader
            config: DP configuration (epsilon, delta, noise, etc.)
        """
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required for DP-SGD. Install with: pip install opacus"
            )
        
        self.config = config or DPConfig()
        self.original_model = model
        self.original_optimizer = optimizer
        self.original_data_loader = data_loader
        
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.private_model: Optional[nn.Module] = None
        self.private_optimizer: Optional[torch.optim.Optimizer] = None
        self.private_data_loader: Optional[DataLoader] = None
        
        # Calculate sample rate if not provided
        if self.config.sample_rate is None:
            dataset_size = len(data_loader.dataset)
            self.config.sample_rate = self.config.batch_size / dataset_size

    def validate_model(self) -> Tuple[nn.Module, bool]:
        """Validate and fix model for DP training.
        
        Returns:
            Tuple of (fixed_model, was_modified)
        """
        model = self.original_model
        errors = ModuleValidator.validate(model, strict=False)
        
        if errors:
            model = ModuleValidator.fix(model)
            return model, True
        
        return model, False

    def make_private(self) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """Make model, optimizer, and data loader private.
        
        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        # Validate and fix model
        model, was_modified = self.validate_model()
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine()
        
        # Make model and optimizer private
        self.private_model, self.private_optimizer, self.private_data_loader = (
            self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=self.original_optimizer,
                data_loader=self.original_data_loader,
                target_epsilon=self.config.target_epsilon,
                target_delta=self.config.target_delta,
                epochs=self.config.epochs,
                max_grad_norm=self.config.max_grad_norm,
            )
        )
        
        return self.private_model, self.private_optimizer, self.private_data_loader

    def make_private_with_noise(
        self,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """Make private with explicit noise multiplier (instead of epsilon target).
        
        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        model, was_modified = self.validate_model()
        
        self.privacy_engine = PrivacyEngine()
        
        self.private_model, self.private_optimizer, self.private_data_loader = (
            self.privacy_engine.make_private(
                module=model,
                optimizer=self.original_optimizer,
                data_loader=self.original_data_loader,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
            )
        )
        
        return self.private_model, self.private_optimizer, self.private_data_loader

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Get current privacy budget spent (epsilon).
        
        Args:
            delta: Delta value for epsilon computation (uses config default if None)
            
        Returns:
            Current epsilon value
        """
        if self.privacy_engine is None:
            raise RuntimeError("Privacy engine not initialized. Call make_private first.")
        
        delta = delta or self.config.target_delta
        return self.privacy_engine.get_epsilon(delta)

    def get_privacy_spent(self) -> Dict[str, float]:
        """Get full privacy budget information.
        
        Returns:
            Dictionary with epsilon, delta, and other privacy metrics
        """
        if self.privacy_engine is None:
            return {"epsilon": 0.0, "delta": 0.0}
        
        epsilon = self.get_epsilon()
        return {
            "epsilon": epsilon,
            "delta": self.config.target_delta,
            "noise_multiplier": self.config.noise_multiplier,
            "max_grad_norm": self.config.max_grad_norm,
        }

    def step(self) -> None:
        """Step the privacy accountant (called after optimizer.step())."""
        if self.privacy_engine is not None:
            self.privacy_engine.accountant.step(
                noise_multiplier=self.config.noise_multiplier,
                sample_rate=self.config.sample_rate,
            )


def make_private_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, DPSGDHandler]:
    """Convenience function to make a model private.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        data_loader: Training data loader
        target_epsilon: Target epsilon for DP
        target_delta: Target delta for DP
        max_grad_norm: Maximum gradient norm for clipping
        epochs: Number of training epochs
        
    Returns:
        Tuple of (private_model, private_optimizer, private_data_loader, dp_handler)
    """
    config = DPConfig(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        batch_size=data_loader.batch_size or 32,
    )
    
    handler = DPSGDHandler(model, optimizer, data_loader, config)
    private_model, private_optimizer, private_loader = handler.make_private()
    
    return private_model, private_optimizer, private_loader, handler


class ManualDPSGD:
    """Manual DP-SGD implementation for environments without Opacus.
    
    This provides a fallback implementation using manual gradient clipping
    and noise injection. Use DPSGDHandler with Opacus when possible.
    """

    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        device: str = "cpu",
    ):
        """Initialize manual DP-SGD.
        
        Args:
            model: PyTorch model
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier for Gaussian noise
            device: Device for computation
        """
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.device = device

    def clip_gradients(self) -> float:
        """Clip gradients by global norm.
        
        Returns:
            Original gradient norm before clipping
        """
        total_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = math.sqrt(total_norm)
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm

    def add_noise(self, batch_size: int) -> None:
        """Add Gaussian noise to gradients.
        
        Args:
            batch_size: Batch size for noise scaling
        """
        noise_std = self.max_grad_norm * self.noise_multiplier / batch_size
        
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=noise_std,
                    size=param.grad.shape,
                    device=self.device,
                )
                param.grad.data.add_(noise)

    def step(self, batch_size: int) -> float:
        """Perform DP-SGD step: clip gradients and add noise.
        
        Args:
            batch_size: Current batch size
            
        Returns:
            Original gradient norm before clipping
        """
        original_norm = self.clip_gradients()
        self.add_noise(batch_size)
        return original_norm
