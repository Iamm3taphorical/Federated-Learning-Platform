"""Database package for federated learning platform."""

from database.models import (
    Base,
    Hospital,
    Model,
    TrainingRound,
    RoundParticipation,
    ModelVersion,
    PrivacyBudget,
    AuditLog,
    AuthToken,
)
from database.connection import (
    get_engine,
    get_session,
    get_async_session,
    init_db,
)
from database.crud import (
    HospitalCRUD,
    ModelCRUD,
    TrainingRoundCRUD,
    ModelVersionCRUD,
    PrivacyBudgetCRUD,
    AuditLogCRUD,
)

__all__ = [
    # Models
    "Base",
    "Hospital",
    "Model",
    "TrainingRound",
    "RoundParticipation", 
    "ModelVersion",
    "PrivacyBudget",
    "AuditLog",
    "AuthToken",
    # Connection
    "get_engine",
    "get_session",
    "get_async_session",
    "init_db",
    # CRUD
    "HospitalCRUD",
    "ModelCRUD",
    "TrainingRoundCRUD",
    "ModelVersionCRUD",
    "PrivacyBudgetCRUD",
    "AuditLogCRUD",
]
