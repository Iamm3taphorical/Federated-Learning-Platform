"""Pytest configuration and fixtures.

Provides test fixtures for database, models, and clients.
"""

import pytest
import os
import tempfile

import torch

# Set test database
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["DATABASE_ASYNC_URL"] = "sqlite+aiosqlite:///test.db"


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    from models.medical_cnn import MedicalCNN
    
    return MedicalCNN(
        num_classes=3,
        in_channels=1,
        input_size=(64, 64),
        use_dp_compatible=True,
    )


@pytest.fixture
def sample_data():
    """Create sample training data."""
    # Random images and labels
    images = torch.randn(32, 1, 64, 64)
    labels = torch.randint(0, 3, (32,))
    return images, labels


@pytest.fixture
def train_loader(sample_data):
    """Create a sample training data loader."""
    from torch.utils.data import TensorDataset, DataLoader
    
    images, labels = sample_data
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def db_session():
    """Create a test database session."""
    from database.connection import get_engine, init_db
    from database.models import Base
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory database
    engine = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()
