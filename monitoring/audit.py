"""Audit logging utilities for HIPAA/GDPR compliance.

Provides structured audit logging for all federated learning operations.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
import json
import os

from database.connection import get_session
from database.crud import AuditLogCRUD
from config.constants import AuditAction


def create_audit_log(
    actor: str,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[UUID] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    success: bool = True,
) -> None:
    """Create an audit log entry.
    
    Logs to database and optionally to file for compliance.
    
    Args:
        actor: Who performed the action
        action: What action was performed
        resource_type: Type of resource affected
        resource_id: ID of resource affected
        ip_address: IP address of actor
        details: Additional details (avoid PII)
        success: Whether the action succeeded
    """
    try:
        with get_session() as session:
            AuditLogCRUD.create(
                session=session,
                actor=actor,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                details=details,
                success=success,
            )
    except Exception as e:
        # Fallback to file logging if database fails
        _log_to_file(actor, action, resource_type, resource_id, details, success, str(e))


def _log_to_file(
    actor: str,
    action: str,
    resource_type: Optional[str],
    resource_id: Optional[UUID],
    details: Optional[Dict[str, Any]],
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Fallback file-based audit logging."""
    log_dir = os.getenv("LOGS_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "audit.log")
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "actor": actor,
        "action": action,
        "resource_type": resource_type,
        "resource_id": str(resource_id) if resource_id else None,
        "details": details,
        "success": success,
        "db_error": error,
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_training_event(
    event_type: str,
    model_id: UUID,
    round_number: int,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a training-related event.
    
    Args:
        event_type: Type of training event
        model_id: Model being trained
        round_number: Current round number
        details: Additional event details
    """
    create_audit_log(
        actor="training_system",
        action=f"training.{event_type}",
        resource_type="model",
        resource_id=model_id,
        details={
            "round_number": round_number,
            **(details or {}),
        },
    )


def log_privacy_event(
    event_type: str,
    epsilon: float,
    delta: float,
    round_id: UUID,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a privacy-related event.
    
    Args:
        event_type: Type of privacy event
        epsilon: Current epsilon value
        delta: Current delta value
        round_id: Training round ID
        details: Additional details
    """
    create_audit_log(
        actor="privacy_system",
        action=f"privacy.{event_type}",
        resource_type="training_round",
        resource_id=round_id,
        details={
            "epsilon": epsilon,
            "delta": delta,
            **(details or {}),
        },
    )


def log_client_event(
    event_type: str,
    hospital_id: UUID,
    round_id: Optional[UUID] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a client-related event.
    
    Args:
        event_type: Type of client event (join, leave, update)
        hospital_id: Hospital/client ID
        round_id: Optional training round ID
        details: Additional details
    """
    create_audit_log(
        actor=str(hospital_id),
        action=f"client.{event_type}",
        resource_type="hospital",
        resource_id=hospital_id,
        details={
            "round_id": str(round_id) if round_id else None,
            **(details or {}),
        },
    )
