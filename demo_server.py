"""Standalone demo server for testing the API.

This runs without requiring flwr or database connections.
"""

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware


# In-memory storage for demo
HOSPITALS = {}
MODELS = {}
TRAINING_ROUNDS = {}


# Pydantic models
class HospitalCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    region: Optional[str] = None
    public_key: str = Field(default="demo_public_key")


class HospitalResponse(BaseModel):
    hospital_id: str
    name: str
    region: Optional[str]
    registered_at: datetime
    is_active: bool


class ModelCreate(BaseModel):
    architecture: str = Field(..., min_length=1)
    modality: str = Field(..., min_length=1)
    description: Optional[str] = None


class ModelResponse(BaseModel):
    model_id: str
    architecture: str
    modality: str
    description: Optional[str]
    created_at: datetime


class TrainingStartRequest(BaseModel):
    model_id: str
    num_rounds: int = Field(default=10, ge=1, le=1000)
    min_clients: int = Field(default=2, ge=1)
    enable_dp: bool = True
    target_epsilon: float = Field(default=8.0, gt=0)


class TrainingStatusResponse(BaseModel):
    round_id: str
    model_id: str
    status: str
    config: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown handlers."""
    print("=" * 60)
    print("Federated Learning Platform API - DEMO MODE")
    print("=" * 60)
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("Health Check: http://127.0.0.1:8000/health")
    print("=" * 60)
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Federated Learning Platform for Medical Diagnosis",
    description="Privacy-preserving federated learning across healthcare institutions (DEMO)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "demo",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Federated Learning Platform API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "status": "running",
        "mode": "demo",
        "endpoints": {
            "health": "GET /health",
            "hospitals": "GET/POST /hospitals",
            "models": "GET/POST /models",
            "training": "POST /training/start, GET /training/status/{id}",
        }
    }


@app.post("/hospitals", response_model=HospitalResponse, status_code=201)
async def register_hospital(hospital: HospitalCreate):
    """Register a new hospital."""
    hospital_id = str(uuid4())
    data = {
        "hospital_id": hospital_id,
        "name": hospital.name,
        "region": hospital.region,
        "registered_at": datetime.utcnow(),
        "is_active": True,
    }
    HOSPITALS[hospital_id] = data
    return HospitalResponse(**data)


@app.get("/hospitals", response_model=List[HospitalResponse])
async def list_hospitals():
    """List all registered hospitals."""
    return [HospitalResponse(**h) for h in HOSPITALS.values()]


@app.post("/models", response_model=ModelResponse, status_code=201)
async def create_model(model: ModelCreate):
    """Create a new model definition."""
    model_id = str(uuid4())
    data = {
        "model_id": model_id,
        "architecture": model.architecture,
        "modality": model.modality,
        "description": model.description,
        "created_at": datetime.utcnow(),
    }
    MODELS[model_id] = data
    return ModelResponse(**data)


@app.get("/models", response_model=List[ModelResponse])
async def list_models():
    """List all models."""
    return [ModelResponse(**m) for m in MODELS.values()]


@app.post("/training/start", response_model=TrainingStatusResponse)
async def start_training(request: TrainingStartRequest):
    """Start a training session (simulated in demo mode)."""
    if request.model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    round_id = str(uuid4())
    data = {
        "round_id": round_id,
        "model_id": request.model_id,
        "status": "pending",
        "config": {
            "num_rounds": request.num_rounds,
            "min_clients": request.min_clients,
            "enable_dp": request.enable_dp,
            "target_epsilon": request.target_epsilon,
        },
    }
    TRAINING_ROUNDS[round_id] = data
    return TrainingStatusResponse(**data)


@app.get("/training/status/{round_id}", response_model=TrainingStatusResponse)
async def get_training_status(round_id: str):
    """Get training status."""
    if round_id not in TRAINING_ROUNDS:
        raise HTTPException(status_code=404, detail="Training round not found")
    return TrainingStatusResponse(**TRAINING_ROUNDS[round_id])


@app.get("/privacy/settings")
async def get_privacy_settings():
    """Get current privacy settings."""
    return {
        "differential_privacy": {
            "enabled": True,
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
            "noise_multiplier": 1.1,
            "max_grad_norm": 1.0,
        },
        "secure_aggregation": {
            "enabled": False,
            "protocol": "none",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
