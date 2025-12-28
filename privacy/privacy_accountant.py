"""Privacy Accountant for tracking differential privacy budget.

Implements RDP (Renyi Differential Privacy) accounting for accurate
privacy budget tracking across federated learning rounds.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import math
from abc import ABC, abstractmethod


@dataclass
class PrivacySpent:
    """Record of privacy budget spent."""
    
    epsilon: float
    delta: float
    num_steps: int = 0
    noise_multiplier: float = 0.0
    sample_rate: float = 0.0


@dataclass
class RoundPrivacy:
    """Privacy budget for a single FL round."""
    
    round_number: int
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    num_local_steps: int


class PrivacyAccountant(ABC):
    """Abstract base class for privacy accounting."""
    
    @abstractmethod
    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        """Record a single DP-SGD step."""
        pass
    
    @abstractmethod
    def get_epsilon(self, delta: float) -> float:
        """Compute epsilon for given delta."""
        pass
    
    @abstractmethod
    def get_privacy_spent(self) -> PrivacySpent:
        """Get total privacy budget spent."""
        pass


class MomentsAccountant(PrivacyAccountant):
    """Moments accountant for RDP-based privacy tracking.
    
    Uses the moments accountant technique from "Deep Learning with Differential Privacy"
    (Abadi et al., 2016) to provide tighter privacy bounds than naive composition.
    """

    # Default moment orders for RDP accounting
    DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(
        self,
        moment_orders: Optional[List[float]] = None,
    ):
        """Initialize moments accountant.
        
        Args:
            moment_orders: List of moment orders for RDP computation
        """
        self.orders = moment_orders or self.DEFAULT_ORDERS
        self.steps: List[Tuple[float, float]] = []  # (noise_multiplier, sample_rate)

    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        """Record a single DP-SGD step.
        
        Args:
            noise_multiplier: Noise multiplier used
            sample_rate: Batch sampling rate
        """
        self.steps.append((noise_multiplier, sample_rate))

    def _compute_log_moment(
        self,
        order: float,
        noise_multiplier: float,
        sample_rate: float,
    ) -> float:
        """Compute log moment for a single step.
        
        Uses the formula from Mironov's RDP paper.
        """
        if noise_multiplier == 0:
            return float('inf')
        
        # Simplified RDP computation for Gaussian mechanism
        return order * sample_rate ** 2 / (2 * noise_multiplier ** 2)

    def _compute_rdp(self, order: float) -> float:
        """Compute RDP for all steps at given order."""
        total_rdp = 0.0
        for noise_multiplier, sample_rate in self.steps:
            total_rdp += self._compute_log_moment(order, noise_multiplier, sample_rate)
        return total_rdp

    def get_epsilon(self, delta: float) -> float:
        """Compute epsilon for given delta using RDP.
        
        Args:
            delta: Target delta value
            
        Returns:
            Epsilon value providing (epsilon, delta)-DP
        """
        if not self.steps:
            return 0.0
        
        if delta <= 0:
            return float('inf')
        
        # Compute epsilon for each order and take minimum
        min_epsilon = float('inf')
        
        for order in self.orders:
            rdp = self._compute_rdp(order)
            # Convert RDP to (epsilon, delta)-DP
            epsilon = rdp - math.log(delta) / (order - 1)
            min_epsilon = min(min_epsilon, epsilon)
        
        return max(0.0, min_epsilon)

    def get_delta(self, epsilon: float) -> float:
        """Compute delta for given epsilon.
        
        Args:
            epsilon: Target epsilon value
            
        Returns:
            Delta value providing (epsilon, delta)-DP
        """
        if not self.steps:
            return 0.0
        
        min_delta = 1.0
        
        for order in self.orders:
            rdp = self._compute_rdp(order)
            delta = math.exp((order - 1) * (rdp - epsilon))
            min_delta = min(min_delta, delta)
        
        return min_delta

    def get_privacy_spent(self) -> PrivacySpent:
        """Get total privacy budget spent."""
        if not self.steps:
            return PrivacySpent(epsilon=0.0, delta=0.0)
        
        # Use a common delta value
        delta = 1e-5
        epsilon = self.get_epsilon(delta)
        
        # Get average noise multiplier and sample rate
        avg_noise = sum(s[0] for s in self.steps) / len(self.steps)
        avg_rate = sum(s[1] for s in self.steps) / len(self.steps)
        
        return PrivacySpent(
            epsilon=epsilon,
            delta=delta,
            num_steps=len(self.steps),
            noise_multiplier=avg_noise,
            sample_rate=avg_rate,
        )

    def reset(self) -> None:
        """Reset the accountant."""
        self.steps = []


class FederatedPrivacyAccountant:
    """Privacy accountant for federated learning scenarios.
    
    Tracks privacy budget across multiple FL rounds and clients,
    handling the composition of privacy across rounds.
    """

    def __init__(
        self,
        client_sampling_rate: float = 1.0,
        moment_orders: Optional[List[float]] = None,
    ):
        """Initialize federated privacy accountant.
        
        Args:
            client_sampling_rate: Probability of client being sampled per round
            moment_orders: Moment orders for RDP computation
        """
        self.client_sampling_rate = client_sampling_rate
        self.round_accountants: Dict[int, MomentsAccountant] = {}
        self.global_accountant = MomentsAccountant(moment_orders)
        self.rounds: List[RoundPrivacy] = []

    def start_round(self, round_number: int) -> None:
        """Start tracking a new round.
        
        Args:
            round_number: The round number
        """
        self.round_accountants[round_number] = MomentsAccountant()

    def record_step(
        self,
        round_number: int,
        noise_multiplier: float,
        sample_rate: float,
    ) -> None:
        """Record a training step within a round.
        
        Args:
            round_number: Current round number
            noise_multiplier: Noise multiplier used
            sample_rate: Batch sampling rate
        """
        if round_number not in self.round_accountants:
            self.start_round(round_number)
        
        self.round_accountants[round_number].step(noise_multiplier, sample_rate)
        
        # Also record in global accountant with client sampling
        # This accounts for the amplification from client subsampling
        effective_sample_rate = sample_rate * self.client_sampling_rate
        self.global_accountant.step(noise_multiplier, effective_sample_rate)

    def complete_round(
        self,
        round_number: int,
        noise_multiplier: float,
        max_grad_norm: float,
        delta: float = 1e-5,
    ) -> RoundPrivacy:
        """Complete a round and compute its privacy budget.
        
        Args:
            round_number: Round number
            noise_multiplier: Noise multiplier used
            max_grad_norm: Maximum gradient norm
            delta: Target delta
            
        Returns:
            RoundPrivacy record for this round
        """
        if round_number not in self.round_accountants:
            raise ValueError(f"Round {round_number} not started")
        
        accountant = self.round_accountants[round_number]
        epsilon = accountant.get_epsilon(delta)
        
        round_privacy = RoundPrivacy(
            round_number=round_number,
            epsilon=epsilon,
            delta=delta,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            num_local_steps=len(accountant.steps),
        )
        
        self.rounds.append(round_privacy)
        return round_privacy

    def get_cumulative_epsilon(self, delta: float = 1e-5) -> float:
        """Get cumulative epsilon across all rounds.
        
        Args:
            delta: Target delta
            
        Returns:
            Cumulative epsilon value
        """
        return self.global_accountant.get_epsilon(delta)

    def get_summary(self, delta: float = 1e-5) -> Dict[str, Any]:
        """Get summary of privacy budget spent.
        
        Returns:
            Dictionary with privacy summary
        """
        return {
            "cumulative_epsilon": self.get_cumulative_epsilon(delta),
            "delta": delta,
            "num_rounds": len(self.rounds),
            "total_steps": len(self.global_accountant.steps),
            "client_sampling_rate": self.client_sampling_rate,
            "rounds": [
                {
                    "round": r.round_number,
                    "epsilon": r.epsilon,
                    "steps": r.num_local_steps,
                }
                for r in self.rounds
            ],
        }


def compute_dp_sgd_privacy(
    sample_size: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-5,
) -> float:
    """Compute the privacy budget for DP-SGD training.
    
    Args:
        sample_size: Total number of training samples
        batch_size: Batch size for training
        noise_multiplier: Noise multiplier used
        epochs: Number of training epochs
        delta: Target delta value
        
    Returns:
        Epsilon value for the training run
    """
    if sample_size <= 0 or batch_size <= 0:
        raise ValueError("sample_size and batch_size must be positive")
    
    sample_rate = batch_size / sample_size
    steps = int(epochs * sample_size / batch_size)
    
    accountant = MomentsAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier, sample_rate)
    
    return accountant.get_epsilon(delta)


def compute_noise_multiplier(
    sample_size: int,
    batch_size: int,
    target_epsilon: float,
    epochs: int,
    delta: float = 1e-5,
    tolerance: float = 0.01,
) -> float:
    """Compute noise multiplier to achieve target epsilon.
    
    Uses binary search to find the noise multiplier that achieves
    the target epsilon for given training parameters.
    
    Args:
        sample_size: Total training samples
        batch_size: Batch size
        target_epsilon: Desired epsilon
        epochs: Training epochs
        delta: Target delta
        tolerance: Tolerance for binary search
        
    Returns:
        Required noise multiplier
    """
    low, high = 0.1, 100.0
    
    while high - low > tolerance:
        mid = (low + high) / 2
        epsilon = compute_dp_sgd_privacy(
            sample_size, batch_size, mid, epochs, delta
        )
        
        if epsilon < target_epsilon:
            high = mid
        else:
            low = mid
    
    return high
