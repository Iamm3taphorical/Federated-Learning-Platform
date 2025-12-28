"""Application settings using Pydantic BaseSettings.

Loads configuration from environment variables and .env files.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database Configuration
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/federated_medical",
        description="PostgreSQL connection URL",
    )
    database_async_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/federated_medical",
        description="Async PostgreSQL connection URL",
    )

    # Server Configuration
    server_host: str = Field(default="0.0.0.0", description="API server host")
    server_port: int = Field(default=8000, description="API server port")
    fl_server_address: str = Field(
        default="0.0.0.0:8080", description="Flower FL server address"
    )
    num_rounds: int = Field(default=10, description="Number of FL training rounds")
    min_clients: int = Field(
        default=2, description="Minimum clients required to start training"
    )
    fraction_fit: float = Field(
        default=1.0, description="Fraction of clients for training"
    )
    fraction_evaluate: float = Field(
        default=1.0, description="Fraction of clients for evaluation"
    )

    # Privacy Configuration
    enable_dp: bool = Field(
        default=True, description="Enable differential privacy"
    )
    target_epsilon: float = Field(
        default=8.0, description="Target epsilon for DP"
    )
    target_delta: float = Field(
        default=1e-5, description="Target delta for DP"
    )
    max_grad_norm: float = Field(
        default=1.0, description="Maximum gradient norm for clipping"
    )
    noise_multiplier: float = Field(
        default=1.1, description="Noise multiplier for DP-SGD"
    )

    # Secure Aggregation
    enable_secure_agg: bool = Field(
        default=False, description="Enable secure aggregation"
    )

    # Authentication
    secret_key: str = Field(
        default="your-super-secret-key-change-in-production",
        description="JWT secret key",
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Token expiration in minutes"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")

    # Model Configuration
    default_model_architecture: str = Field(
        default="MedicalCNN", description="Default model architecture"
    )
    default_modality: str = Field(
        default="xray", description="Default imaging modality"
    )
    local_epochs: int = Field(
        default=1, description="Local epochs per FL round"
    )
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")

    # Paths
    weights_dir: str = Field(
        default="./weights", description="Directory for model weights"
    )
    logs_dir: str = Field(default="./logs", description="Directory for logs")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings instance (cached)
    """
    return Settings()
