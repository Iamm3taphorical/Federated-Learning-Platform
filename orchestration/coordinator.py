"""Federated learning coordinator for orchestrating training workflows.

Provides high-level orchestration of federated training sessions,
coordinating between server, clients, database, and monitoring.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
import asyncio
import threading

from database.connection import get_session
from database.crud import (
    ModelCRUD,
    TrainingRoundCRUD,
    PrivacyBudgetCRUD,
    ModelVersionCRUD,
    AuditLogCRUD,
)
from config.settings import get_settings
from config.constants import RoundStatus, AuditAction
from monitoring.metrics import MetricsCollector, PrivacyMetricsTracker
from monitoring.audit import log_training_event


class FederatedCoordinator:
    """Coordinator for federated learning workflows.
    
    Orchestrates the full lifecycle of federated training:
    - Model creation and versioning
    - Training round management
    - Privacy budget tracking
    - Metrics collection
    - Compliance logging
    """

    def __init__(
        self,
        model_id: Optional[UUID] = None,
        architecture: str = "MedicalCNN",
        modality: str = "xray",
        num_classes: int = 14,
        enable_dp: bool = True,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
    ):
        """Initialize coordinator.
        
        Args:
            model_id: Existing model ID to continue training
            architecture: Model architecture for new model
            modality: Data modality for new model
            num_classes: Number of output classes
            enable_dp: Enable differential privacy
            target_epsilon: Target privacy budget
            target_delta: Target delta value
        """
        self.settings = get_settings()
        self.model_id = model_id
        self.architecture = architecture
        self.modality = modality
        self.num_classes = num_classes
        self.enable_dp = enable_dp
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        # Tracking
        self.current_round: Optional[UUID] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.privacy_tracker: Optional[PrivacyMetricsTracker] = None
        
        # FL server thread
        self._server_thread: Optional[threading.Thread] = None
        self._is_training = False

    def initialize_model(self) -> UUID:
        """Initialize or retrieve model.
        
        Returns:
            Model UUID
        """
        with get_session() as session:
            if self.model_id:
                model = ModelCRUD.get_by_id(session, self.model_id)
                if not model:
                    raise ValueError(f"Model {self.model_id} not found")
            else:
                model = ModelCRUD.create(
                    session=session,
                    architecture=self.architecture,
                    modality=self.modality,
                    hyperparameters={
                        "num_classes": self.num_classes,
                        "enable_dp": self.enable_dp,
                    },
                )
                self.model_id = model.model_id
                
                AuditLogCRUD.create(
                    session=session,
                    actor="coordinator",
                    action=AuditAction.MODEL_CREATE,
                    resource_type="model",
                    resource_id=model.model_id,
                )
        
        # Initialize trackers
        self.metrics_collector = MetricsCollector(
            model_id=str(self.model_id),
            output_dir=self.settings.logs_dir,
        )
        
        if self.enable_dp:
            self.privacy_tracker = PrivacyMetricsTracker(
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
            )
        
        return self.model_id

    def start_training_round(
        self,
        num_rounds: int = 10,
        min_clients: int = 2,
        config: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Start a new training round.
        
        Args:
            num_rounds: Number of FL rounds
            min_clients: Minimum required clients
            config: Additional configuration
            
        Returns:
            Training round UUID
        """
        if self.model_id is None:
            self.initialize_model()
        
        with get_session() as session:
            # Get current round number
            existing_rounds = TrainingRoundCRUD.get_by_model(session, self.model_id)
            round_number = len(existing_rounds) + 1
            
            # Create training round
            training_round = TrainingRoundCRUD.create(
                session=session,
                model_id=self.model_id,
                round_number=round_number,
                config={
                    "num_rounds": num_rounds,
                    "min_clients": min_clients,
                    "enable_dp": self.enable_dp,
                    "target_epsilon": self.target_epsilon,
                    **(config or {}),
                },
            )
            
            # Create privacy budget record
            if self.enable_dp:
                PrivacyBudgetCRUD.create(
                    session=session,
                    round_id=training_round.round_id,
                    epsilon=self.target_epsilon,
                    delta=self.target_delta,
                    noise_multiplier=self.settings.noise_multiplier,
                )
            
            # Update status to in_progress
            TrainingRoundCRUD.update_status(
                session=session,
                round_id=training_round.round_id,
                status=RoundStatus.IN_PROGRESS,
            )
            
            self.current_round = training_round.round_id
            
            log_training_event(
                event_type="start",
                model_id=self.model_id,
                round_number=round_number,
                details={"num_rounds": num_rounds, "min_clients": min_clients},
            )
        
        return self.current_round

    def record_round_metrics(
        self,
        fl_round: int,
        loss: float,
        accuracy: float,
        num_clients: int,
        epsilon: Optional[float] = None,
    ) -> None:
        """Record metrics for a FL round.
        
        Args:
            fl_round: FL round number within session
            loss: Training/validation loss
            accuracy: Accuracy metric
            num_clients: Participating clients
            epsilon: Current cumulative epsilon
        """
        if self.metrics_collector:
            self.metrics_collector.record_round(
                round_number=fl_round,
                loss=loss,
                accuracy=accuracy,
                num_clients=num_clients,
                epsilon=epsilon,
            )
        
        if self.privacy_tracker and epsilon:
            self.privacy_tracker.record_epsilon(epsilon, fl_round)

    def complete_training(
        self,
        final_accuracy: Optional[float] = None,
        final_loss: Optional[float] = None,
        weights_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete the training session.
        
        Args:
            final_accuracy: Final model accuracy
            final_loss: Final loss value
            weights_path: Path to saved weights
            
        Returns:
            Training summary
        """
        if self.current_round is None:
            raise ValueError("No active training round")
        
        with get_session() as session:
            # Update round status
            TrainingRoundCRUD.update_status(
                session=session,
                round_id=self.current_round,
                status=RoundStatus.COMPLETED,
            )
            
            # Create model version
            if final_accuracy is not None or final_loss is not None:
                ModelVersionCRUD.create(
                    session=session,
                    model_id=self.model_id,
                    round_id=self.current_round,
                    accuracy=final_accuracy,
                    loss=final_loss,
                    weights_path=weights_path,
                )
            
            # Update privacy budget if tracking
            if self.privacy_tracker:
                budget = PrivacyBudgetCRUD.get_by_round(session, self.current_round)
                if budget:
                    report = self.privacy_tracker.get_report()
                    PrivacyBudgetCRUD.update_cumulative(
                        session=session,
                        round_id=self.current_round,
                        cumulative_epsilon=report["current_epsilon"],
                        cumulative_delta=self.target_delta,
                    )
            
            log_training_event(
                event_type="complete",
                model_id=self.model_id,
                round_number=0,  # Will be retrieved from DB
                details={
                    "accuracy": final_accuracy,
                    "loss": final_loss,
                },
            )
        
        # Get summary
        summary = {
            "model_id": str(self.model_id),
            "round_id": str(self.current_round),
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "weights_path": weights_path,
        }
        
        if self.metrics_collector:
            summary["metrics_summary"] = self.metrics_collector.get_summary()
        
        if self.privacy_tracker:
            summary["privacy_report"] = self.privacy_tracker.get_report()
        
        self.current_round = None
        self._is_training = False
        
        return summary

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status.
        
        Returns:
            Status dictionary
        """
        status = {
            "model_id": str(self.model_id) if self.model_id else None,
            "current_round": str(self.current_round) if self.current_round else None,
            "is_training": self._is_training,
        }
        
        if self.current_round:
            with get_session() as session:
                round_info = TrainingRoundCRUD.get_by_id(session, self.current_round)
                if round_info:
                    status["round_status"] = round_info.status
                    status["round_number"] = round_info.round_number
        
        if self.metrics_collector:
            status["metrics"] = self.metrics_collector.get_summary()
        
        if self.privacy_tracker:
            status["privacy"] = self.privacy_tracker.get_report()
        
        return status
