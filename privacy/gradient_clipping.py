"""Gradient clipping utilities for differential privacy.

Provides per-sample and global gradient clipping to bound
the sensitivity of gradient-based computations.
"""

from typing import Dict, List, Optional, Tuple, OrderedDict
import math

import torch
import torch.nn as nn


class GradientClipper:
    """Gradient clipping for differential privacy.
    
    Implements both per-sample clipping (for DP-SGD) and global clipping
    (for simpler privacy mechanisms).
    """

    def __init__(
        self,
        max_grad_norm: float,
        norm_type: float = 2.0,
    ):
        """Initialize gradient clipper.
        
        Args:
            max_grad_norm: Maximum gradient norm (clipping bound)
            norm_type: Type of norm (1, 2, or inf)
        """
        self.max_grad_norm = max_grad_norm
        self.norm_type = norm_type

    def clip_gradients(
        self,
        model: nn.Module,
    ) -> float:
        """Clip gradients by global norm.
        
        Args:
            model: PyTorch model with computed gradients
            
        Returns:
            Original gradient norm before clipping
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # Compute total gradient norm
        if self.norm_type == float('inf'):
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
            total_norm = total_norm ** (1.0 / self.norm_type)
        
        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        if clip_coef < 1.0:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        
        return total_norm

    def clip_gradient_dict(
        self,
        gradients: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Clip a dictionary of gradients.
        
        Args:
            gradients: Dictionary of gradient tensors
            
        Returns:
            Tuple of (clipped gradients, original norm)
        """
        # Compute total norm
        if self.norm_type == float('inf'):
            total_norm = max(
                g.abs().max().item() for g in gradients.values() if g is not None
            )
        else:
            total_norm = 0.0
            for g in gradients.values():
                if g is not None:
                    param_norm = g.norm(self.norm_type)
                    total_norm += param_norm.item() ** self.norm_type
            total_norm = total_norm ** (1.0 / self.norm_type)
        
        # Compute clip coefficient
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        # Clip gradients
        clipped = {}
        for key, grad in gradients.items():
            if grad is not None:
                clipped[key] = grad * clip_coef
            else:
                clipped[key] = None
        
        return clipped, total_norm

    def clip_model_updates(
        self,
        old_weights: OrderedDict[str, torch.Tensor],
        new_weights: OrderedDict[str, torch.Tensor],
    ) -> Tuple[OrderedDict[str, torch.Tensor], float]:
        """Clip model weight updates (new - old).
        
        Useful for federated learning where we clip the difference
        between local and global models.
        
        Args:
            old_weights: Original model weights
            new_weights: Updated model weights
            
        Returns:
            Tuple of (clipped updates as new weights, original update norm)
        """
        # Compute updates
        updates = {}
        for key in new_weights:
            if key in old_weights:
                updates[key] = new_weights[key] - old_weights[key]
            else:
                updates[key] = new_weights[key]
        
        # Compute update norm
        total_norm = 0.0
        for update in updates.values():
            if update is not None:
                param_norm = update.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        total_norm = total_norm ** (1.0 / self.norm_type)
        
        # Compute clip coefficient
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        # Apply clipped updates to get new weights
        clipped_weights = OrderedDict()
        for key in new_weights:
            if key in old_weights:
                clipped_update = updates[key] * clip_coef
                clipped_weights[key] = old_weights[key] + clipped_update
            else:
                clipped_weights[key] = new_weights[key]
        
        return clipped_weights, total_norm


class AdaptiveGradientClipper:
    """Adaptive gradient clipping with quantile-based bound estimation.
    
    Adjusts clipping bound based on observed gradient norms to balance
    privacy and utility.
    """

    def __init__(
        self,
        initial_bound: float = 1.0,
        target_quantile: float = 0.5,
        learning_rate: float = 0.01,
        min_bound: float = 0.001,
        max_bound: float = 100.0,
    ):
        """Initialize adaptive clipper.
        
        Args:
            initial_bound: Initial clipping bound
            target_quantile: Target quantile of gradients to clip
            learning_rate: Learning rate for bound adjustment
            min_bound: Minimum allowed bound
            max_bound: Maximum allowed bound
        """
        self.bound = initial_bound
        self.target_quantile = target_quantile
        self.learning_rate = learning_rate
        self.min_bound = min_bound
        self.max_bound = max_bound
        
        self.norm_history: List[float] = []

    def clip_and_update(
        self,
        model: nn.Module,
    ) -> Tuple[float, float]:
        """Clip gradients and update clipping bound.
        
        Args:
            model: PyTorch model with computed gradients
            
        Returns:
            Tuple of (original norm, current bound)
        """
        # Compute gradient norm
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0, self.bound
        
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm)
        
        # Record norm
        self.norm_history.append(total_norm)
        
        # Clip gradients with current bound
        clip_coef = self.bound / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        if clip_coef < 1.0:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        
        # Update bound based on whether we clipped
        # If norm > bound, we want to increase bound (negative gradient)
        # If norm < bound, we want to decrease bound (positive gradient)
        was_clipped = 1.0 if total_norm > self.bound else 0.0
        gradient = was_clipped - self.target_quantile
        
        # Update bound
        self.bound = self.bound * math.exp(-self.learning_rate * gradient)
        self.bound = max(self.min_bound, min(self.max_bound, self.bound))
        
        return total_norm, self.bound

    def get_statistics(self) -> Dict[str, float]:
        """Get clipping statistics.
        
        Returns:
            Dictionary with clipping statistics
        """
        if not self.norm_history:
            return {"current_bound": self.bound}
        
        import statistics
        
        return {
            "current_bound": self.bound,
            "mean_norm": statistics.mean(self.norm_history),
            "median_norm": statistics.median(self.norm_history),
            "max_norm": max(self.norm_history),
            "min_norm": min(self.norm_history),
            "clip_rate": sum(1 for n in self.norm_history if n > self.bound) / len(self.norm_history),
        }

    def reset(self) -> None:
        """Reset norm history."""
        self.norm_history = []
