"""CRUD operations for the federated learning platform database.

Provides data access layer with type-safe operations for all models.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, and_
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    Hospital,
    Model,
    TrainingRound,
    RoundParticipation,
    ModelVersion,
    PrivacyBudget,
    AuditLog,
)


class HospitalCRUD:
    """CRUD operations for Hospital model."""

    @staticmethod
    def create(
        session: Session,
        name: str,
        public_key: str,
        region: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Hospital:
        """Create a new hospital registration."""
        hospital = Hospital(
            name=name,
            public_key=public_key,
            region=region,
            metadata_=metadata or {},
        )
        session.add(hospital)
        session.flush()
        return hospital

    @staticmethod
    def get_by_id(session: Session, hospital_id: UUID) -> Optional[Hospital]:
        """Get hospital by ID."""
        return session.get(Hospital, hospital_id)

    @staticmethod
    def get_all(
        session: Session,
        active_only: bool = True,
        region: Optional[str] = None,
    ) -> List[Hospital]:
        """Get all hospitals with optional filters."""
        query = select(Hospital)
        if active_only:
            query = query.where(Hospital.is_active == True)
        if region:
            query = query.where(Hospital.region == region)
        return list(session.execute(query).scalars().all())

    @staticmethod
    def update(
        session: Session,
        hospital_id: UUID,
        **kwargs,
    ) -> Optional[Hospital]:
        """Update hospital fields."""
        hospital = session.get(Hospital, hospital_id)
        if hospital:
            for key, value in kwargs.items():
                if hasattr(hospital, key):
                    setattr(hospital, key, value)
            session.flush()
        return hospital

    @staticmethod
    def deactivate(session: Session, hospital_id: UUID) -> bool:
        """Deactivate a hospital (soft delete)."""
        result = session.execute(
            update(Hospital)
            .where(Hospital.hospital_id == hospital_id)
            .values(is_active=False)
        )
        return result.rowcount > 0


class ModelCRUD:
    """CRUD operations for Model."""

    @staticmethod
    def create(
        session: Session,
        architecture: str,
        modality: str,
        description: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Model:
        """Create a new model entry."""
        model = Model(
            architecture=architecture,
            modality=modality,
            description=description,
            hyperparameters=hyperparameters or {},
        )
        session.add(model)
        session.flush()
        return model

    @staticmethod
    def get_by_id(session: Session, model_id: UUID) -> Optional[Model]:
        """Get model by ID."""
        return session.get(Model, model_id)

    @staticmethod
    def get_all(
        session: Session,
        modality: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> List[Model]:
        """Get all models with optional filters."""
        query = select(Model)
        if modality:
            query = query.where(Model.modality == modality)
        if architecture:
            query = query.where(Model.architecture == architecture)
        return list(session.execute(query).scalars().all())


class TrainingRoundCRUD:
    """CRUD operations for TrainingRound."""

    @staticmethod
    def create(
        session: Session,
        model_id: UUID,
        round_number: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingRound:
        """Create a new training round."""
        training_round = TrainingRound(
            model_id=model_id,
            round_number=round_number,
            config=config or {},
            status="pending",
        )
        session.add(training_round)
        session.flush()
        return training_round

    @staticmethod
    def get_by_id(session: Session, round_id: UUID) -> Optional[TrainingRound]:
        """Get training round by ID."""
        return session.get(TrainingRound, round_id)

    @staticmethod
    def get_by_model(
        session: Session,
        model_id: UUID,
        status: Optional[str] = None,
    ) -> List[TrainingRound]:
        """Get all training rounds for a model."""
        query = select(TrainingRound).where(TrainingRound.model_id == model_id)
        if status:
            query = query.where(TrainingRound.status == status)
        query = query.order_by(TrainingRound.round_number)
        return list(session.execute(query).scalars().all())

    @staticmethod
    def start_round(session: Session, round_id: UUID) -> Optional[TrainingRound]:
        """Mark a round as started."""
        training_round = session.get(TrainingRound, round_id)
        if training_round:
            training_round.status = "in_progress"
            training_round.started_at = datetime.utcnow()
            session.flush()
        return training_round

    @staticmethod
    def complete_round(session: Session, round_id: UUID) -> Optional[TrainingRound]:
        """Mark a round as completed."""
        training_round = session.get(TrainingRound, round_id)
        if training_round:
            training_round.status = "completed"
            training_round.completed_at = datetime.utcnow()
            session.flush()
        return training_round

    @staticmethod
    def fail_round(session: Session, round_id: UUID) -> Optional[TrainingRound]:
        """Mark a round as failed."""
        training_round = session.get(TrainingRound, round_id)
        if training_round:
            training_round.status = "failed"
            training_round.completed_at = datetime.utcnow()
            session.flush()
        return training_round

    @staticmethod
    def add_participation(
        session: Session,
        round_id: UUID,
        hospital_id: UUID,
    ) -> RoundParticipation:
        """Add hospital participation to a round."""
        participation = RoundParticipation(
            round_id=round_id,
            hospital_id=hospital_id,
        )
        session.add(participation)
        session.flush()
        return participation

    @staticmethod
    def update_participation(
        session: Session,
        round_id: UUID,
        hospital_id: UUID,
        update_received: bool = True,
        update_size_bytes: Optional[int] = None,
        training_time_seconds: Optional[float] = None,
        local_samples_used: Optional[int] = None,
    ) -> Optional[RoundParticipation]:
        """Update participation record."""
        participation = session.get(
            RoundParticipation, (round_id, hospital_id)
        )
        if participation:
            participation.update_received = update_received
            participation.completed_at = datetime.utcnow()
            if update_size_bytes is not None:
                participation.update_size_bytes = update_size_bytes
            if training_time_seconds is not None:
                participation.training_time_seconds = training_time_seconds
            if local_samples_used is not None:
                participation.local_samples_used = local_samples_used
            session.flush()
        return participation


class ModelVersionCRUD:
    """CRUD operations for ModelVersion."""

    @staticmethod
    def create(
        session: Session,
        model_id: UUID,
        round_id: UUID,
        accuracy: Optional[float] = None,
        auc: Optional[float] = None,
        loss: Optional[float] = None,
        weights_path: Optional[str] = None,
        additional_metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelVersion:
        """Create a new model version."""
        version = ModelVersion(
            model_id=model_id,
            round_id=round_id,
            accuracy=accuracy,
            auc=auc,
            loss=loss,
            weights_path=weights_path,
            additional_metrics=additional_metrics or {},
            **kwargs,
        )
        session.add(version)
        session.flush()
        return version

    @staticmethod
    def get_by_id(session: Session, version_id: UUID) -> Optional[ModelVersion]:
        """Get model version by ID."""
        return session.get(ModelVersion, version_id)

    @staticmethod
    def get_by_model(
        session: Session,
        model_id: UUID,
    ) -> List[ModelVersion]:
        """Get all versions for a model."""
        query = (
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
        )
        return list(session.execute(query).scalars().all())

    @staticmethod
    def get_best_version(
        session: Session,
        model_id: UUID,
        metric: str = "accuracy",
    ) -> Optional[ModelVersion]:
        """Get the best model version by a specific metric."""
        query = (
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(getattr(ModelVersion, metric).desc())
            .limit(1)
        )
        return session.execute(query).scalar_one_or_none()


class PrivacyBudgetCRUD:
    """CRUD operations for PrivacyBudget."""

    @staticmethod
    def create(
        session: Session,
        round_id: UUID,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        max_grad_norm: float = 1.0,
        cumulative_epsilon: Optional[float] = None,
        cumulative_delta: Optional[float] = None,
    ) -> PrivacyBudget:
        """Create privacy budget record for a round."""
        budget = PrivacyBudget(
            round_id=round_id,
            epsilon=epsilon,
            delta=delta,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            cumulative_epsilon=cumulative_epsilon,
            cumulative_delta=cumulative_delta,
        )
        session.add(budget)
        session.flush()
        return budget

    @staticmethod
    def get_by_round(session: Session, round_id: UUID) -> Optional[PrivacyBudget]:
        """Get privacy budget for a round."""
        return session.get(PrivacyBudget, round_id)

    @staticmethod
    def get_cumulative_budget(
        session: Session,
        model_id: UUID,
    ) -> Dict[str, float]:
        """Calculate cumulative privacy budget for a model."""
        query = (
            select(PrivacyBudget)
            .join(TrainingRound)
            .where(TrainingRound.model_id == model_id)
            .order_by(TrainingRound.round_number)
        )
        budgets = list(session.execute(query).scalars().all())
        
        if not budgets:
            return {"epsilon": 0.0, "delta": 0.0}
        
        # Simple composition (advanced composition would use RDP)
        total_epsilon = sum(b.epsilon for b in budgets)
        max_delta = max(b.delta for b in budgets)
        
        return {"epsilon": total_epsilon, "delta": max_delta}


class AuditLogCRUD:
    """CRUD operations for AuditLog."""

    @staticmethod
    def create(
        session: Session,
        actor: str,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> AuditLog:
        """Create an audit log entry."""
        log = AuditLog(
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            details=details or {},
            success=success,
        )
        session.add(log)
        session.flush()
        return log

    @staticmethod
    def get_by_actor(
        session: Session,
        actor: str,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Get audit logs for an actor."""
        query = (
            select(AuditLog)
            .where(AuditLog.actor == actor)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        return list(session.execute(query).scalars().all())

    @staticmethod
    def get_by_resource(
        session: Session,
        resource_type: str,
        resource_id: UUID,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Get audit logs for a resource."""
        query = (
            select(AuditLog)
            .where(
                and_(
                    AuditLog.resource_type == resource_type,
                    AuditLog.resource_id == resource_id,
                )
            )
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        return list(session.execute(query).scalars().all())

    @staticmethod
    def get_recent(
        session: Session,
        limit: int = 100,
        action: Optional[str] = None,
    ) -> List[AuditLog]:
        """Get recent audit logs."""
        query = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
        if action:
            query = query.where(AuditLog.action == action)
        return list(session.execute(query).scalars().all())
