"""FastAPI application for the federated learning platform.

Provides REST API endpoints for managing hospitals, models,
training rounds, and monitoring federated learning.
"""

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db, init_db
from database.crud import (
    HospitalCRUD,
    ModelCRUD,
    TrainingRoundCRUD,
    ModelVersionCRUD,
    PrivacyBudgetCRUD,
    AuditLogCRUD,
)
from config.settings import get_settings
from config.constants import AuditAction


# Pydantic models for API
class HospitalCreate(BaseModel):
    """Request model for hospital registration."""
    name: str = Field(..., min_length=1, max_length=255)
    region: Optional[str] = None
    public_key: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class HospitalResponse(BaseModel):
    """Response model for hospital data."""
    hospital_id: str
    name: str
    region: Optional[str]
    registered_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class ModelCreate(BaseModel):
    """Request model for creating a model."""
    architecture: str = Field(..., min_length=1, max_length=100)
    modality: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    """Response model for model data."""
    model_id: str
    architecture: str
    modality: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingStartRequest(BaseModel):
    """Request model for starting training."""
    model_id: str
    num_rounds: int = Field(default=10, ge=1, le=1000)
    min_clients: int = Field(default=2, ge=1)
    enable_dp: bool = True
    target_epsilon: float = Field(default=8.0, gt=0)
    target_delta: float = Field(default=1e-5, gt=0)
    noise_multiplier: float = Field(default=1.1, gt=0)


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    round_id: str
    model_id: str
    round_number: int
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class PrivacyBudgetResponse(BaseModel):
    """Response model for privacy budget."""
    epsilon: float
    delta: float
    noise_multiplier: float
    cumulative_epsilon: Optional[float]


class MetricsResponse(BaseModel):
    """Response model for model metrics."""
    version_id: str
    round_id: str
    accuracy: Optional[float]
    auc: Optional[float]
    loss: Optional[float]
    created_at: datetime


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    print(f"Starting Federated Learning Platform API on port {settings.server_port}")
    
    # Initialize database
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization warning: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down API server")


# Create FastAPI app
app = FastAPI(
    title="Federated Learning Platform for Medical Diagnosis",
    description="Privacy-preserving federated learning across healthcare institutions",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Hospital endpoints
@app.post("/hospitals", response_model=HospitalResponse, status_code=status.HTTP_201_CREATED)
async def register_hospital(
    hospital: HospitalCreate,
    db: Session = Depends(get_db),
):
    """Register a new hospital for federated learning."""
    created = HospitalCRUD.create(
        session=db,
        name=hospital.name,
        public_key=hospital.public_key,
        region=hospital.region,
        metadata=hospital.metadata,
    )
    db.commit()
    
    # Audit log
    AuditLogCRUD.create(
        session=db,
        actor="api",
        action=AuditAction.HOSPITAL_REGISTER,
        resource_type="hospital",
        resource_id=created.hospital_id,
        details={"name": hospital.name},
    )
    db.commit()
    
    return HospitalResponse(
        hospital_id=str(created.hospital_id),
        name=created.name,
        region=created.region,
        registered_at=created.registered_at,
        is_active=created.is_active,
    )


@app.get("/hospitals", response_model=List[HospitalResponse])
async def list_hospitals(
    active_only: bool = True,
    region: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all registered hospitals."""
    hospitals = HospitalCRUD.get_all(
        session=db,
        active_only=active_only,
        region=region,
    )
    return [
        HospitalResponse(
            hospital_id=str(h.hospital_id),
            name=h.name,
            region=h.region,
            registered_at=h.registered_at,
            is_active=h.is_active,
        )
        for h in hospitals
    ]


@app.get("/hospitals/{hospital_id}", response_model=HospitalResponse)
async def get_hospital(
    hospital_id: str,
    db: Session = Depends(get_db),
):
    """Get a specific hospital by ID."""
    hospital = HospitalCRUD.get_by_id(db, uuid.UUID(hospital_id))
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    
    return HospitalResponse(
        hospital_id=str(hospital.hospital_id),
        name=hospital.name,
        region=hospital.region,
        registered_at=hospital.registered_at,
        is_active=hospital.is_active,
    )


# Model endpoints
@app.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
):
    """Create a new model definition."""
    created = ModelCRUD.create(
        session=db,
        architecture=model.architecture,
        modality=model.modality,
        description=model.description,
        hyperparameters=model.hyperparameters,
    )
    db.commit()
    
    AuditLogCRUD.create(
        session=db,
        actor="api",
        action=AuditAction.MODEL_CREATE,
        resource_type="model",
        resource_id=created.model_id,
        details={"architecture": model.architecture, "modality": model.modality},
    )
    db.commit()
    
    return ModelResponse(
        model_id=str(created.model_id),
        architecture=created.architecture,
        modality=created.modality,
        description=created.description,
        created_at=created.created_at,
    )


@app.get("/models", response_model=List[ModelResponse])
async def list_models(
    modality: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all models."""
    models = ModelCRUD.get_all(session=db, modality=modality)
    return [
        ModelResponse(
            model_id=str(m.model_id),
            architecture=m.architecture,
            modality=m.modality,
            description=m.description,
            created_at=m.created_at,
        )
        for m in models
    ]


# Training endpoints
@app.post("/training/start", response_model=TrainingStatusResponse)
async def start_training(
    request: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start a new federated training session."""
    # Verify model exists
    model = ModelCRUD.get_by_id(db, uuid.UUID(request.model_id))
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get existing rounds for this model
    existing_rounds = TrainingRoundCRUD.get_by_model(db, model.model_id)
    next_round_num = len(existing_rounds) + 1
    
    # Create training round record
    training_round = TrainingRoundCRUD.create(
        session=db,
        model_id=model.model_id,
        round_number=next_round_num,
        config={
            "num_rounds": request.num_rounds,
            "min_clients": request.min_clients,
            "enable_dp": request.enable_dp,
            "target_epsilon": request.target_epsilon,
            "target_delta": request.target_delta,
            "noise_multiplier": request.noise_multiplier,
        },
    )
    db.commit()
    
    # Create privacy budget record
    PrivacyBudgetCRUD.create(
        session=db,
        round_id=training_round.round_id,
        epsilon=request.target_epsilon,
        delta=request.target_delta,
        noise_multiplier=request.noise_multiplier,
    )
    db.commit()
    
    # Audit log
    AuditLogCRUD.create(
        session=db,
        actor="api",
        action=AuditAction.TRAINING_START,
        resource_type="training_round",
        resource_id=training_round.round_id,
        details={"model_id": request.model_id, "num_rounds": request.num_rounds},
    )
    db.commit()
    
    # Start training in background (in production, this would use a task queue)
    # background_tasks.add_task(run_federated_training, training_round.round_id)
    
    return TrainingStatusResponse(
        round_id=str(training_round.round_id),
        model_id=str(training_round.model_id),
        round_number=training_round.round_number,
        status=training_round.status,
        started_at=training_round.started_at,
        completed_at=training_round.completed_at,
    )


@app.get("/training/status/{round_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    round_id: str,
    db: Session = Depends(get_db),
):
    """Get status of a training round."""
    training_round = TrainingRoundCRUD.get_by_id(db, uuid.UUID(round_id))
    if not training_round:
        raise HTTPException(status_code=404, detail="Training round not found")
    
    return TrainingStatusResponse(
        round_id=str(training_round.round_id),
        model_id=str(training_round.model_id),
        round_number=training_round.round_number,
        status=training_round.status,
        started_at=training_round.started_at,
        completed_at=training_round.completed_at,
    )


@app.get("/training/privacy/{round_id}", response_model=PrivacyBudgetResponse)
async def get_privacy_budget(
    round_id: str,
    db: Session = Depends(get_db),
):
    """Get privacy budget for a training round."""
    budget = PrivacyBudgetCRUD.get_by_round(db, uuid.UUID(round_id))
    if not budget:
        raise HTTPException(status_code=404, detail="Privacy budget not found")
    
    return PrivacyBudgetResponse(
        epsilon=budget.epsilon,
        delta=budget.delta,
        noise_multiplier=budget.noise_multiplier,
        cumulative_epsilon=budget.cumulative_epsilon,
    )


@app.get("/metrics/{model_id}", response_model=List[MetricsResponse])
async def get_model_metrics(
    model_id: str,
    db: Session = Depends(get_db),
):
    """Get evaluation metrics for a model."""
    versions = ModelVersionCRUD.get_by_model(db, uuid.UUID(model_id))
    return [
        MetricsResponse(
            version_id=str(v.version_id),
            round_id=str(v.round_id),
            accuracy=v.accuracy,
            auc=v.auc,
            loss=v.loss,
            created_at=v.created_at,
        )
        for v in versions
    ]


# Audit log endpoint (for compliance)
@app.get("/audit/recent")
async def get_recent_audit_logs(
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get recent audit logs for compliance review."""
    logs = AuditLogCRUD.get_recent(db, limit=limit)
    return [
        {
            "log_id": str(log.log_id),
            "actor": log.actor,
            "action": log.action,
            "resource_type": log.resource_type,
            "resource_id": str(log.resource_id) if log.resource_id else None,
            "timestamp": log.timestamp.isoformat(),
            "success": log.success,
        }
        for log in logs
    ]


def run_api():
    """Run the FastAPI application."""
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "server.app:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )


if __name__ == "__main__":
    run_api()
