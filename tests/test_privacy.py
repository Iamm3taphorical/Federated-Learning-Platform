"""Tests for privacy module."""

import pytest
import torch
import numpy as np


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_clip_gradients(self, sample_model):
        """Test gradient clipping works correctly."""
        from privacy.gradient_clipping import GradientClipper
        
        # Create fake gradients
        for param in sample_model.parameters():
            param.grad = torch.randn_like(param) * 10  # Large gradients
        
        clipper = GradientClipper(max_grad_norm=1.0)
        original_norm = clipper.clip_gradients(sample_model)
        
        # Compute clipped norm
        clipped_norm = 0.0
        for param in sample_model.parameters():
            if param.grad is not None:
                clipped_norm += param.grad.norm(2).item() ** 2
        clipped_norm = np.sqrt(clipped_norm)
        
        # Original should be larger than max
        assert original_norm > 1.0
        # Clipped should be <= max
        assert clipped_norm <= 1.0 + 1e-6


class TestNoiseInjection:
    """Tests for noise injection."""

    def test_gaussian_noise_injection(self):
        """Test Gaussian noise is added correctly."""
        from privacy.noise_injection import GaussianNoiseInjector
        
        injector = GaussianNoiseInjector(noise_multiplier=1.0)
        
        # Test tensor
        original = torch.zeros(100)
        noised = injector.add_noise(original, sensitivity=1.0)
        
        # Should be different from original
        assert not torch.allclose(original, noised)
        
        # Noise should have expected standard deviation (approximately)
        std = noised.std().item()
        assert 0.5 < std < 1.5  # Allow some variance

    def test_gaussian_noise_dict(self):
        """Test noise injection on dictionary of tensors."""
        from privacy.noise_injection import GaussianNoiseInjector
        
        injector = GaussianNoiseInjector(noise_multiplier=1.0)
        
        gradients = {
            "layer1": torch.zeros(50),
            "layer2": torch.zeros(100),
        }
        
        noised = injector.add_noise(gradients, sensitivity=1.0)
        
        assert "layer1" in noised
        assert "layer2" in noised
        assert not torch.allclose(gradients["layer1"], noised["layer1"])


class TestPrivacyAccountant:
    """Tests for privacy accounting."""

    def test_moments_accountant(self):
        """Test moments accountant epsilon computation."""
        from privacy.privacy_accountant import MomentsAccountant
        
        accountant = MomentsAccountant()
        
        # Simulate training steps
        for _ in range(100):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        
        epsilon = accountant.get_epsilon(delta=1e-5)
        
        # Epsilon should be positive and finite
        assert epsilon > 0
        assert np.isfinite(epsilon)

    def test_compute_dp_sgd_privacy(self):
        """Test convenience function for computing privacy."""
        from privacy.privacy_accountant import compute_dp_sgd_privacy
        
        epsilon = compute_dp_sgd_privacy(
            sample_size=1000,
            batch_size=32,
            noise_multiplier=1.0,
            epochs=1,
            delta=1e-5,
        )
        
        assert epsilon > 0
        assert np.isfinite(epsilon)


class TestDPSGD:
    """Tests for DP-SGD implementation."""

    def test_manual_dp_sgd(self, sample_model):
        """Test manual DP-SGD clip and noise."""
        from privacy.dp_sgd import ManualDPSGD
        
        dp_sgd = ManualDPSGD(
            model=sample_model,
            max_grad_norm=1.0,
            noise_multiplier=1.0,
        )
        
        # Create gradients
        for param in sample_model.parameters():
            param.grad = torch.randn_like(param) * 5
        
        # Apply DP-SGD step
        original_norm = dp_sgd.step(batch_size=32)
        
        assert original_norm > 0
