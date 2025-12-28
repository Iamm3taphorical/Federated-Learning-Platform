"""Metrics collection for federated learning platform.

Provides utilities for collecting, aggregating, and reporting
training metrics across federated learning rounds.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class RoundMetrics:
    """Metrics for a single training round."""
    
    round_number: int
    loss: float
    accuracy: float
    num_clients: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    epsilon: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collector for federated learning metrics.
    
    Collects and aggregates metrics across training rounds
    for monitoring and analysis.
    """

    def __init__(self, model_id: str, output_dir: str = "./logs"):
        """Initialize metrics collector.
        
        Args:
            model_id: Model being trained
            output_dir: Directory for metrics output
        """
        self.model_id = model_id
        self.output_dir = output_dir
        self.rounds: List[RoundMetrics] = []
        
        os.makedirs(output_dir, exist_ok=True)

    def record_round(
        self,
        round_number: int,
        loss: float,
        accuracy: float,
        num_clients: int,
        epsilon: Optional[float] = None,
        **additional_metrics,
    ) -> RoundMetrics:
        """Record metrics for a training round.
        
        Args:
            round_number: Round number
            loss: Training/validation loss
            accuracy: Accuracy metric
            num_clients: Number of participating clients
            epsilon: Privacy budget spent (if using DP)
            **additional_metrics: Any additional metrics
            
        Returns:
            RoundMetrics object
        """
        metrics = RoundMetrics(
            round_number=round_number,
            loss=loss,
            accuracy=accuracy,
            num_clients=num_clients,
            epsilon=epsilon,
            additional_metrics=additional_metrics,
        )
        
        self.rounds.append(metrics)
        self._save_metrics()
        
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics.
        
        Returns:
            Summary dictionary
        """
        if not self.rounds:
            return {"model_id": self.model_id, "rounds": 0}
        
        losses = [r.loss for r in self.rounds]
        accuracies = [r.accuracy for r in self.rounds]
        
        return {
            "model_id": self.model_id,
            "num_rounds": len(self.rounds),
            "final_loss": losses[-1],
            "final_accuracy": accuracies[-1],
            "best_accuracy": max(accuracies),
            "min_loss": min(losses),
            "avg_clients_per_round": sum(r.num_clients for r in self.rounds) / len(self.rounds),
            "final_epsilon": self.rounds[-1].epsilon if self.rounds[-1].epsilon else None,
        }

    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get history of all rounds.
        
        Returns:
            List of round metrics dictionaries
        """
        return [
            {
                "round": r.round_number,
                "loss": r.loss,
                "accuracy": r.accuracy,
                "num_clients": r.num_clients,
                "epsilon": r.epsilon,
                "timestamp": r.timestamp.isoformat(),
                **r.additional_metrics,
            }
            for r in self.rounds
        ]

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        filepath = os.path.join(self.output_dir, f"metrics_{self.model_id}.json")
        
        data = {
            "model_id": self.model_id,
            "summary": self.get_summary(),
            "rounds": self.get_round_history(),
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: Optional[str] = None) -> str:
        """Export metrics to CSV format.
        
        Args:
            filepath: Optional path for CSV file
            
        Returns:
            Path to created CSV file
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, f"metrics_{self.model_id}.csv")
        
        headers = ["round", "loss", "accuracy", "num_clients", "epsilon", "timestamp"]
        
        with open(filepath, "w") as f:
            f.write(",".join(headers) + "\n")
            for r in self.rounds:
                row = [
                    str(r.round_number),
                    f"{r.loss:.6f}",
                    f"{r.accuracy:.6f}",
                    str(r.num_clients),
                    f"{r.epsilon:.6f}" if r.epsilon else "",
                    r.timestamp.isoformat(),
                ]
                f.write(",".join(row) + "\n")
        
        return filepath


class PrivacyMetricsTracker:
    """Tracker for privacy-related metrics."""

    def __init__(self, target_epsilon: float, target_delta: float):
        """Initialize privacy tracker.
        
        Args:
            target_epsilon: Target epsilon budget
            target_delta: Target delta value
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.epsilon_history: List[float] = []
        self.warnings: List[str] = []

    def record_epsilon(self, epsilon: float, round_number: int) -> None:
        """Record epsilon for a round.
        
        Args:
            epsilon: Current cumulative epsilon
            round_number: Round number
        """
        self.epsilon_history.append(epsilon)
        
        # Check if approaching budget
        if epsilon > 0.8 * self.target_epsilon:
            self.warnings.append(
                f"Round {round_number}: Epsilon {epsilon:.4f} approaching "
                f"target {self.target_epsilon}"
            )
        
        if epsilon > self.target_epsilon:
            self.warnings.append(
                f"Round {round_number}: PRIVACY BUDGET EXCEEDED! "
                f"Epsilon {epsilon:.4f} > target {self.target_epsilon}"
            )

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget.
        
        Returns:
            Remaining epsilon budget
        """
        current = self.epsilon_history[-1] if self.epsilon_history else 0.0
        return max(0.0, self.target_epsilon - current)

    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted.
        
        Returns:
            True if budget is exhausted
        """
        return self.get_remaining_budget() <= 0

    def get_report(self) -> Dict[str, Any]:
        """Get privacy metrics report.
        
        Returns:
            Privacy metrics dictionary
        """
        current = self.epsilon_history[-1] if self.epsilon_history else 0.0
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "current_epsilon": current,
            "remaining_budget": self.get_remaining_budget(),
            "budget_exhausted": self.is_budget_exhausted(),
            "num_rounds": len(self.epsilon_history),
            "warnings": self.warnings,
        }
