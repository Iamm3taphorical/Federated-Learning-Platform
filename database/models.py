"""SQLAlchemy ORM models for the federated learning platform.

This module defines all database models for tracking federated learning
metadata, privacy budgets, model versions, and audit logs.

Note: This database stores ONLY system metadata - never raw medical data.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    ForeignKey,
    Text,
    BigInteger,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column

Base = declarative_base()


class Hospital(Base):
    """Represents a hospital/institution participating in federated learning.
    
    Stores registration info and public key for secure communication.
    """
    __tablename__ = "hospitals"

    hospital_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    region: Mapped[Optional[str]] = mapped_column(String(100))
    public_key: Mapped[str] = mapped_column(Text, nullable=False)
    registered_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, default=datetime.utcnow
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )

    # Relationships
    participations: Mapped[List["RoundParticipation"]] = relationship(
        "RoundParticipation", back_populates="hospital", lazy="dynamic"
    )
    auth_tokens: Mapped[List["AuthToken"]] = relationship(
        "AuthToken", back_populates="hospital", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<Hospital(id={self.hospital_id}, name='{self.name}')>"


class Model(Base):
    """Represents an ML model architecture used in federated training."""
    __tablename__ = "models"

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    architecture: Mapped[str] = mapped_column(String(100), nullable=False)
    modality: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, default=datetime.utcnow
    )
    description: Mapped[Optional[str]] = mapped_column(Text)
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Relationships
    training_rounds: Mapped[List["TrainingRound"]] = relationship(
        "TrainingRound", back_populates="model", lazy="dynamic"
    )
    versions: Mapped[List["ModelVersion"]] = relationship(
        "ModelVersion", back_populates="model", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.model_id}, arch='{self.architecture}', modality='{self.modality}')>"


class TrainingRound(Base):
    """Represents a single federated learning training round."""
    __tablename__ = "training_rounds"

    round_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("models.model_id", ondelete="CASCADE")
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    config: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="training_rounds")
    participations: Mapped[List["RoundParticipation"]] = relationship(
        "RoundParticipation", back_populates="training_round", lazy="dynamic"
    )
    privacy_budget: Mapped[Optional["PrivacyBudget"]] = relationship(
        "PrivacyBudget", back_populates="training_round", uselist=False
    )
    model_version: Mapped[Optional["ModelVersion"]] = relationship(
        "ModelVersion", back_populates="training_round", uselist=False
    )

    def __repr__(self) -> str:
        return f"<TrainingRound(id={self.round_id}, round={self.round_number}, status='{self.status}')>"


class RoundParticipation(Base):
    """Tracks hospital participation in each training round."""
    __tablename__ = "round_participation"

    round_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("training_rounds.round_id", ondelete="CASCADE"),
        primary_key=True,
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hospitals.hospital_id", ondelete="CASCADE"),
        primary_key=True,
    )
    update_received: Mapped[bool] = mapped_column(Boolean, default=False)
    update_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    training_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    local_samples_used: Mapped[Optional[int]] = mapped_column(Integer)
    joined_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)

    # Relationships
    training_round: Mapped["TrainingRound"] = relationship(
        "TrainingRound", back_populates="participations"
    )
    hospital: Mapped["Hospital"] = relationship(
        "Hospital", back_populates="participations"
    )

    def __repr__(self) -> str:
        return f"<RoundParticipation(round={self.round_id}, hospital={self.hospital_id})>"


class ModelVersion(Base):
    """Stores model version metadata and evaluation metrics."""
    __tablename__ = "model_versions"

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("models.model_id", ondelete="CASCADE")
    )
    round_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_rounds.round_id", ondelete="CASCADE")
    )
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    auc: Mapped[Optional[float]] = mapped_column(Float)
    precision_score: Mapped[Optional[float]] = mapped_column(Float)
    recall_score: Mapped[Optional[float]] = mapped_column(Float)
    f1_score: Mapped[Optional[float]] = mapped_column(Float)
    fairness_score: Mapped[Optional[float]] = mapped_column(Float)
    loss: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)
    weights_path: Mapped[Optional[str]] = mapped_column(Text)
    additional_metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="versions")
    training_round: Mapped["TrainingRound"] = relationship(
        "TrainingRound", back_populates="model_version"
    )

    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.version_id}, accuracy={self.accuracy})>"


class PrivacyBudget(Base):
    """Tracks differential privacy budget per training round."""
    __tablename__ = "privacy_budget"

    round_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("training_rounds.round_id", ondelete="CASCADE"),
        primary_key=True,
    )
    epsilon: Mapped[float] = mapped_column(Float, nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)
    noise_multiplier: Mapped[float] = mapped_column(Float, nullable=False)
    max_grad_norm: Mapped[float] = mapped_column(Float, default=1.0)
    cumulative_epsilon: Mapped[Optional[float]] = mapped_column(Float)
    cumulative_delta: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    training_round: Mapped["TrainingRound"] = relationship(
        "TrainingRound", back_populates="privacy_budget"
    )

    def __repr__(self) -> str:
        return f"<PrivacyBudget(round={self.round_id}, ε={self.epsilon}, δ={self.delta})>"


class AuditLog(Base):
    """Append-only audit log for compliance tracking (HIPAA/GDPR)."""
    __tablename__ = "audit_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(100))
    resource_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    details: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    success: Mapped[bool] = mapped_column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.log_id}, actor='{self.actor}', action='{self.action}')>"


class AuthToken(Base):
    """OAuth tokens for hospital authentication."""
    __tablename__ = "auth_tokens"

    token_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hospitals.hospital_id", ondelete="CASCADE")
    )
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    hospital: Mapped["Hospital"] = relationship("Hospital", back_populates="auth_tokens")

    def __repr__(self) -> str:
        return f"<AuthToken(id={self.token_id}, hospital={self.hospital_id})>"
