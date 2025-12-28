"""Model utility functions for federated learning.

Provides functions for extracting, setting, and aggregating model
parameters in federated learning scenarios.
"""

from typing import List, Tuple, Dict, OrderedDict, Optional
from collections import OrderedDict as ODict
import copy

import torch
import torch.nn as nn
import numpy as np


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of numpy arrays.
    
    This format is compatible with Flower's parameter handling.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of numpy arrays representing model parameters
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of numpy arrays.
    
    Args:
        model: PyTorch model to update
        parameters: List of numpy arrays with new parameter values
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = ODict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_model_weights(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Get model weights as an ordered dict (state dict).
    
    Args:
        model: PyTorch model
        
    Returns:
        Model state dict
    """
    return copy.deepcopy(model.state_dict())


def set_model_weights(
    model: nn.Module,
    weights: OrderedDict[str, torch.Tensor],
) -> None:
    """Set model weights from state dict.
    
    Args:
        model: PyTorch model
        weights: State dict with weights
    """
    model.load_state_dict(weights, strict=True)


def compute_model_delta(
    old_weights: OrderedDict[str, torch.Tensor],
    new_weights: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Compute the difference between two model weight sets.
    
    Args:
        old_weights: Original weights
        new_weights: Updated weights
        
    Returns:
        Weight delta (new - old)
    """
    delta = ODict()
    for key in new_weights:
        if key in old_weights:
            delta[key] = new_weights[key] - old_weights[key]
        else:
            delta[key] = new_weights[key]
    return delta


def apply_model_delta(
    base_weights: OrderedDict[str, torch.Tensor],
    delta: OrderedDict[str, torch.Tensor],
    scale: float = 1.0,
) -> OrderedDict[str, torch.Tensor]:
    """Apply a scaled delta to base weights.
    
    Args:
        base_weights: Base model weights
        delta: Weight delta to apply
        scale: Scale factor for delta
        
    Returns:
        Updated weights
    """
    new_weights = ODict()
    for key in base_weights:
        if key in delta:
            new_weights[key] = base_weights[key] + scale * delta[key]
        else:
            new_weights[key] = base_weights[key]
    return new_weights


def average_model_weights(
    weights_list: List[OrderedDict[str, torch.Tensor]],
    sample_counts: Optional[List[int]] = None,
) -> OrderedDict[str, torch.Tensor]:
    """Average multiple model weight sets (FedAvg).
    
    Args:
        weights_list: List of model state dicts
        sample_counts: Optional list of sample counts for weighted averaging
        
    Returns:
        Averaged weights
    """
    if not weights_list:
        raise ValueError("weights_list cannot be empty")
    
    if len(weights_list) == 1:
        return copy.deepcopy(weights_list[0])
    
    # Compute weights for averaging
    if sample_counts is not None:
        if len(sample_counts) != len(weights_list):
            raise ValueError("sample_counts must have same length as weights_list")
        total_samples = sum(sample_counts)
        averaging_weights = [count / total_samples for count in sample_counts]
    else:
        # Equal weights
        averaging_weights = [1.0 / len(weights_list)] * len(weights_list)
    
    # Perform weighted average
    averaged = ODict()
    keys = weights_list[0].keys()
    
    for key in keys:
        # Stack tensors and compute weighted sum
        stacked = torch.stack([w[key].float() for w in weights_list])
        weight_tensor = torch.tensor(averaging_weights).view(-1, *([1] * (stacked.dim() - 1)))
        averaged[key] = (stacked * weight_tensor).sum(dim=0)
    
    return averaged


def average_model_deltas(
    deltas: List[OrderedDict[str, torch.Tensor]],
    sample_counts: Optional[List[int]] = None,
) -> OrderedDict[str, torch.Tensor]:
    """Average multiple model deltas.
    
    Similar to average_model_weights but specifically for deltas.
    
    Args:
        deltas: List of model deltas
        sample_counts: Optional sample counts for weighting
        
    Returns:
        Averaged delta
    """
    return average_model_weights(deltas, sample_counts)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def compute_model_norm(
    weights: OrderedDict[str, torch.Tensor],
    norm_type: float = 2.0,
) -> float:
    """Compute the norm of model weights.
    
    Args:
        weights: Model state dict
        norm_type: Type of norm (1, 2, or inf)
        
    Returns:
        Computed norm
    """
    if norm_type == float('inf'):
        return max(w.abs().max().item() for w in weights.values())
    
    total = 0.0
    for weight in weights.values():
        total += weight.norm(norm_type).item() ** norm_type
    
    return total ** (1.0 / norm_type)


def scale_model_weights(
    weights: OrderedDict[str, torch.Tensor],
    scale: float,
) -> OrderedDict[str, torch.Tensor]:
    """Scale model weights by a factor.
    
    Args:
        weights: Model state dict
        scale: Scale factor
        
    Returns:
        Scaled weights
    """
    return ODict({k: v * scale for k, v in weights.items()})


def add_model_weights(
    weights1: OrderedDict[str, torch.Tensor],
    weights2: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Add two model weight dictionaries.
    
    Args:
        weights1: First weights
        weights2: Second weights
        
    Returns:
        Sum of weights
    """
    return ODict({k: weights1[k] + weights2[k] for k in weights1.keys()})


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    tolerance: float = 1e-6,
) -> Tuple[bool, Dict[str, float]]:
    """Compare two models for equivalence.
    
    Args:
        model1: First model
        model2: Second model
        tolerance: Maximum allowed difference
        
    Returns:
        Tuple of (are_equal, layer_diffs)
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    if state1.keys() != state2.keys():
        return False, {"structure_mismatch": float('inf')}
    
    diffs = {}
    all_equal = True
    
    for key in state1:
        diff = (state1[key] - state2[key]).abs().max().item()
        diffs[key] = diff
        if diff > tolerance:
            all_equal = False
    
    return all_equal, diffs


class ModelCheckpointer:
    """Utility for saving and loading model checkpoints."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        import os
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        name: str,
        round_num: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            name: Checkpoint name
            round_num: Optional round number
            metrics: Optional metrics to save
            
        Returns:
            Path to saved checkpoint
        """
        import os
        
        filename = f"{name}"
        if round_num is not None:
            filename += f"_round{round_num}"
        filename += ".pt"
        
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "round_num": round_num,
            "metrics": metrics or {},
        }
        
        torch.save(checkpoint, path)
        return path

    def load(
        self,
        model: nn.Module,
        path: str,
    ) -> Dict:
        """Load model checkpoint.
        
        Args:
            model: Model to load weights into
            path: Path to checkpoint
            
        Returns:
            Checkpoint metadata (round_num, metrics)
        """
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        return {
            "round_num": checkpoint.get("round_num"),
            "metrics": checkpoint.get("metrics", {}),
        }
