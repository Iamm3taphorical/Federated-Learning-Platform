"""Medical CNN architectures for federated learning.

Provides CNN models optimized for medical imaging tasks including
X-ray, MRI, CT scan classification and diagnosis.
"""

from typing import Optional, Tuple, Dict, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalCNN(nn.Module):
    """CNN architecture for medical image classification.
    
    A flexible CNN designed for medical imaging that can be adapted
    for different modalities (X-ray, MRI, CT) and classification tasks.
    
    Uses batch normalization (converted to group norm for DP compatibility)
    and dropout for regularization.
    """

    def __init__(
        self,
        num_classes: int = 14,
        in_channels: int = 1,
        input_size: Tuple[int, int] = (224, 224),
        dropout_rate: float = 0.3,
        use_dp_compatible: bool = True,
    ):
        """Initialize MedicalCNN.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            input_size: Expected input image size (height, width)
            dropout_rate: Dropout rate for regularization
            use_dp_compatible: If True, use GroupNorm instead of BatchNorm for DP
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Normalization layer factory
        def norm_layer(channels: int) -> nn.Module:
            if use_dp_compatible:
                # GroupNorm is DP-compatible (no per-sample stats)
                num_groups = min(32, channels)
                return nn.GroupNorm(num_groups, channels)
            else:
                return nn.BatchNorm2d(channels)
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate feature map size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)
            feat_size = self.features(dummy).view(1, -1).size(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        return self.features(x).view(x.size(0), -1)


class ResNetBlock(nn.Module):
    """ResNet-style residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_dp_compatible: bool = True,
    ):
        super().__init__()
        
        def norm_layer(channels: int) -> nn.Module:
            if use_dp_compatible:
                num_groups = min(32, channels)
                return nn.GroupNorm(num_groups, channels)
            else:
                return nn.BatchNorm2d(channels)
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MedicalResNet(nn.Module):
    """ResNet-style architecture for medical imaging.
    
    A lightweight ResNet variant optimized for federated learning
    on medical images with DP compatibility.
    """

    def __init__(
        self,
        num_classes: int = 14,
        in_channels: int = 1,
        num_blocks: Tuple[int, ...] = (2, 2, 2, 2),
        use_dp_compatible: bool = True,
    ):
        """Initialize MedicalResNet.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            num_blocks: Number of residual blocks per stage
            use_dp_compatible: Use DP-compatible normalization
        """
        super().__init__()
        
        self.use_dp_compatible = use_dp_compatible
        
        def norm_layer(channels: int) -> nn.Module:
            if use_dp_compatible:
                num_groups = min(32, channels)
                return nn.GroupNorm(num_groups, channels)
            else:
                return nn.BatchNorm2d(channels)
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, s, self.use_dp_compatible))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def create_model(
    architecture: str = "MedicalCNN",
    num_classes: int = 14,
    in_channels: int = 1,
    input_size: Tuple[int, int] = (224, 224),
    use_dp_compatible: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to create a model.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        in_channels: Number of input channels
        input_size: Input image size
        use_dp_compatible: Use DP-compatible layers
        **kwargs: Additional model-specific arguments
        
    Returns:
        PyTorch model
    """
    architecture = architecture.lower()
    
    if architecture in ("medicalcnn", "cnn"):
        return MedicalCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size,
            use_dp_compatible=use_dp_compatible,
            **kwargs,
        )
    elif architecture in ("medicalresnet", "resnet"):
        return MedicalResNet(
            num_classes=num_classes,
            in_channels=in_channels,
            use_dp_compatible=use_dp_compatible,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
