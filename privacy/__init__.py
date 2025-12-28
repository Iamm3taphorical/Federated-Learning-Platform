"""Privacy package for federated learning platform.

Provides differential privacy mechanisms, privacy accounting, and secure aggregation.
"""

from privacy.dp_sgd import DPSGDHandler, make_private_model
from privacy.privacy_accountant import PrivacyAccountant, compute_dp_sgd_privacy
from privacy.noise_injection import GaussianNoiseInjector, LaplacianNoiseInjector
from privacy.gradient_clipping import GradientClipper

__all__ = [
    "DPSGDHandler",
    "make_private_model",
    "PrivacyAccountant",
    "compute_dp_sgd_privacy",
    "GaussianNoiseInjector",
    "LaplacianNoiseInjector",
    "GradientClipper",
]
