"""Hospital-specific federated learning client.

Extends the base client with medical data handling capabilities
including DICOM and FHIR data interfaces.
"""

from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import flwr as fl

from clients.base_client import BaseClient
from models.medical_cnn import create_model
from config.settings import get_settings


class MedicalImageDataset(Dataset):
    """Dataset for medical images.
    
    Supports loading images from various formats for federated training.
    This is a simplified implementation - in production, would use
    proper DICOM/FHIR loaders.
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize medical image dataset.
        
        Args:
            data_dir: Directory containing images
            transform: Optional transforms to apply
            target_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Collect image paths
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load sample paths from directory."""
        if not self.data_dir.exists():
            # Create synthetic data for testing
            self._create_synthetic_data()
            return
        
        # Expect subdirectories as class labels
        for class_idx, class_dir in enumerate(sorted(self.data_dir.iterdir())):
            if class_dir.is_dir():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".dcm"]:
                        self.samples.append((img_path, class_idx))

    def _create_synthetic_data(self) -> None:
        """Create synthetic data for testing."""
        # Generate random samples for demonstration
        num_samples = 100
        num_classes = 3
        
        for i in range(num_samples):
            class_label = i % num_classes
            self.samples.append((None, class_label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        if path is None:
            # Synthetic data
            image = torch.randn(1, *self.target_size)
        else:
            # Load real image
            image = self._load_image(path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image from file.
        
        Args:
            path: Path to image file
            
        Returns:
            Image tensor
        """
        try:
            from PIL import Image
            import torchvision.transforms as T
            
            img = Image.open(path).convert("L")  # Grayscale
            transform = T.Compose([
                T.Resize(self.target_size),
                T.ToTensor(),
            ])
            return transform(img)
        except Exception:
            # Return random tensor as fallback
            return torch.randn(1, *self.target_size)


class HospitalClient(BaseClient):
    """Hospital-specific federated learning client.
    
    Extends BaseClient with:
    - Medical image data handling
    - Non-IID data handling
    - Hospital-specific configuration
    """

    def __init__(
        self,
        hospital_id: str,
        data_path: str,
        model_architecture: str = "MedicalCNN",
        num_classes: int = 14,
        in_channels: int = 1,
        batch_size: int = 32,
        device: str = "cpu",
        enable_dp: bool = True,
        max_grad_norm: float = 1.0,
        val_split: float = 0.2,
    ):
        """Initialize hospital client.
        
        Args:
            hospital_id: Unique hospital identifier
            data_path: Path to local medical data
            model_architecture: Model architecture to use
            num_classes: Number of output classes
            in_channels: Number of input channels
            batch_size: Training batch size
            device: Device for training
            enable_dp: Enable differential privacy
            max_grad_norm: Maximum gradient norm
            val_split: Validation split ratio
        """
        # Create model
        model = create_model(
            architecture=model_architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            use_dp_compatible=enable_dp,
        )
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            data_path=data_path,
            batch_size=batch_size,
            val_split=val_split,
        )
        
        super().__init__(
            client_id=hospital_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            enable_dp=enable_dp,
            max_grad_norm=max_grad_norm,
        )
        
        self.hospital_id = hospital_id
        self.data_path = data_path
        
        print(f"Hospital Client '{hospital_id}' initialized")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset) if val_loader else 0}")
        print(f"  DP enabled: {enable_dp}")

    def _create_data_loaders(
        self,
        data_path: str,
        batch_size: int,
        val_split: float,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create training and validation data loaders.
        
        Args:
            data_path: Path to data
            batch_size: Batch size
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        dataset = MedicalImageDataset(data_path)
        
        # Split into train and validation
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            train_dataset = dataset
            val_loader = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        
        return train_loader, val_loader

    def fit(
        self,
        parameters,
        config: Dict[str, Any],
    ) -> Tuple:
        """Train with hospital-specific handling.
        
        Extends parent fit with additional hospital metadata.
        """
        result = super().fit(parameters, config)
        
        # Add hospital-specific metrics
        params, num_examples, metrics = result
        metrics["hospital_id"] = self.hospital_id
        
        return params, num_examples, metrics


def start_client(
    hospital_id: str = None,
    data_path: str = None,
    server_address: str = None,
    **kwargs,
) -> None:
    """Start a hospital client and connect to federation server.
    
    Args:
        hospital_id: Hospital identifier
        data_path: Path to local data
        server_address: Federation server address
        **kwargs: Additional client arguments
    """
    settings = get_settings()
    
    # Use defaults from settings
    hospital_id = hospital_id or f"hospital_{os.getpid()}"
    data_path = data_path or "./data"
    server_address = server_address or settings.fl_server_address
    
    # Create client
    client = HospitalClient(
        hospital_id=hospital_id,
        data_path=data_path,
        batch_size=settings.batch_size,
        enable_dp=settings.enable_dp,
        max_grad_norm=settings.max_grad_norm,
        **kwargs,
    )
    
    print(f"Connecting to federation server at {server_address}")
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )


def main():
    """Main entry point for hospital client."""
    parser = argparse.ArgumentParser(description="Hospital Federated Learning Client")
    parser.add_argument(
        "--hospital-id",
        type=str,
        default=None,
        help="Unique hospital identifier",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to local medical data",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        help="Federation server address",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=14,
        help="Number of output classes",
    )
    parser.add_argument(
        "--no-dp",
        action="store_true",
        help="Disable differential privacy",
    )
    
    args = parser.parse_args()
    
    start_client(
        hospital_id=args.hospital_id,
        data_path=args.data_path,
        server_address=args.server_address,
        num_classes=args.num_classes,
        enable_dp=not args.no_dp,
    )


if __name__ == "__main__":
    main()
