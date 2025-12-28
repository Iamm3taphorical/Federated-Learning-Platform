"""Tests for ML models."""

import pytest
import torch


class TestMedicalCNN:
    """Tests for MedicalCNN model."""

    def test_model_creation(self, sample_model):
        """Test model creates correctly."""
        assert sample_model is not None
        
        # Check model has expected number of classes
        output = sample_model(torch.randn(1, 1, 64, 64))
        assert output.shape == (1, 3)

    def test_model_forward_pass(self, sample_model, sample_data):
        """Test forward pass works."""
        images, _ = sample_data
        # Resize to match model input
        images = torch.nn.functional.interpolate(images, size=(64, 64))
        
        output = sample_model(images)
        
        assert output.shape == (32, 3)
        assert not torch.isnan(output).any()

    def test_model_training_step(self, sample_model, train_loader):
        """Test a single training step."""
        sample_model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sample_model.parameters())
        
        for images, labels in train_loader:
            images = torch.nn.functional.interpolate(images, size=(64, 64))
            
            optimizer.zero_grad()
            output = sample_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0
            break


class TestModelUtils:
    """Tests for model utilities."""

    def test_get_set_parameters(self, sample_model):
        """Test getting and setting model parameters."""
        from models.model_utils import get_model_parameters, set_model_parameters
        
        # Get parameters
        params = get_model_parameters(sample_model)
        
        assert isinstance(params, list)
        assert len(params) > 0
        
        # Modify and set back
        modified_params = [p * 2 for p in params]
        set_model_parameters(sample_model, modified_params)
        
        # Verify parameters changed
        new_params = get_model_parameters(sample_model)
        for orig, new in zip(params, new_params):
            assert not (orig == new).all()

    def test_average_model_weights(self, sample_model):
        """Test FedAvg weight averaging."""
        from models.model_utils import get_model_weights, average_model_weights
        from collections import OrderedDict
        
        # Get base weights
        weights1 = get_model_weights(sample_model)
        
        # Create modified weights
        weights2 = OrderedDict()
        for key, value in weights1.items():
            weights2[key] = value + torch.ones_like(value)
        
        # Average
        averaged = average_model_weights([weights1, weights2])
        
        # Should be between original and modified
        for key in weights1:
            expected = (weights1[key] + weights2[key]) / 2
            assert torch.allclose(averaged[key], expected, atol=1e-6)

    def test_weighted_average(self, sample_model):
        """Test weighted averaging."""
        from models.model_utils import get_model_weights, average_model_weights
        from collections import OrderedDict
        
        weights1 = get_model_weights(sample_model)
        weights2 = OrderedDict()
        for key, value in weights1.items():
            weights2[key] = value + torch.ones_like(value)
        
        # Weight towards weights1 (more samples)
        averaged = average_model_weights(
            [weights1, weights2],
            sample_counts=[100, 10],
        )
        
        # Should be closer to weights1
        for key in weights1:
            diff_from_w1 = (averaged[key] - weights1[key]).abs().max()
            diff_from_w2 = (averaged[key] - weights2[key]).abs().max()
            assert diff_from_w1 < diff_from_w2


class TestModelFactory:
    """Tests for model creation factory."""

    def test_create_medical_cnn(self):
        """Test creating MedicalCNN via factory."""
        from models.medical_cnn import create_model
        
        model = create_model(
            architecture="MedicalCNN",
            num_classes=10,
            in_channels=1,
        )
        
        output = model(torch.randn(1, 1, 224, 224))
        assert output.shape == (1, 10)

    def test_create_resnet(self):
        """Test creating MedicalResNet via factory."""
        from models.medical_cnn import create_model
        
        model = create_model(
            architecture="MedicalResNet",
            num_classes=5,
            in_channels=1,
        )
        
        output = model(torch.randn(1, 1, 224, 224))
        assert output.shape == (1, 5)
