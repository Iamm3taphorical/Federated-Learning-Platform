"""DICOM data handler for medical imaging.

Provides utilities for loading and preprocessing DICOM medical images
for federated learning training.
"""

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset


class DICOMDataHandler:
    """Handler for DICOM medical imaging data.
    
    Provides loading, preprocessing, and dataset creation for DICOM files.
    Designed for use in federated learning with privacy considerations.
    """

    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        cache_images: bool = False,
    ):
        """Initialize DICOM handler.
        
        Args:
            data_dir: Directory containing DICOM files
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            cache_images: Whether to cache loaded images in memory
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.cache_images = cache_images
        
        self._cache: Dict[str, np.ndarray] = {}
        self._pydicom_available = self._check_pydicom()

    def _check_pydicom(self) -> bool:
        """Check if pydicom is available."""
        try:
            import pydicom
            return True
        except ImportError:
            print("Warning: pydicom not installed. DICOM loading disabled.")
            return False

    def load_dicom(self, file_path: str) -> Optional[np.ndarray]:
        """Load a single DICOM file.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Numpy array of pixel data, or None if loading fails
        """
        if not self._pydicom_available:
            return None
        
        # Check cache
        if self.cache_images and file_path in self._cache:
            return self._cache[file_path]
        
        try:
            import pydicom
            
            # Read DICOM
            dcm = pydicom.dcmread(file_path)
            
            # Get pixel array
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            # Apply windowing if available
            if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                center = float(dcm.WindowCenter)
                width = float(dcm.WindowWidth)
                pixel_array = self._apply_windowing(pixel_array, center, width)
            
            # Normalize
            if self.normalize:
                pixel_array = self._normalize(pixel_array)
            
            # Resize
            pixel_array = self._resize(pixel_array)
            
            # Cache if enabled
            if self.cache_images:
                self._cache[file_path] = pixel_array
            
            return pixel_array
            
        except Exception as e:
            print(f"Error loading DICOM {file_path}: {e}")
            return None

    def _apply_windowing(
        self,
        image: np.ndarray,
        center: float,
        width: float,
    ) -> np.ndarray:
        """Apply windowing to image.
        
        Args:
            image: Input image
            center: Window center
            width: Window width
            
        Returns:
            Windowed image
        """
        lower = center - width / 2
        upper = center + width / 2
        image = np.clip(image, lower, upper)
        return image

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        min_val = image.min()
        max_val = image.max()
        
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
        
        return image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        try:
            from PIL import Image
            
            # Convert to PIL Image
            img = Image.fromarray((image * 255).astype(np.uint8))
            img = img.resize(self.target_size[::-1], Image.Resampling.BILINEAR)
            
            return np.array(img).astype(np.float32) / 255.0
            
        except ImportError:
            # Simple nearest-neighbor resize fallback
            import torch.nn.functional as F
            
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(
                tensor, size=self.target_size, mode='bilinear', align_corners=False
            )
            return resized.squeeze().numpy()

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from DICOM file.
        
        Note: Returns limited metadata to avoid privacy leakage.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Dictionary of safe metadata
        """
        if not self._pydicom_available:
            return {}
        
        try:
            import pydicom
            
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            # Only extract non-identifying metadata
            safe_metadata = {
                "modality": getattr(dcm, 'Modality', 'Unknown'),
                "rows": getattr(dcm, 'Rows', 0),
                "columns": getattr(dcm, 'Columns', 0),
                "bits_allocated": getattr(dcm, 'BitsAllocated', 0),
            }
            
            return safe_metadata
            
        except Exception:
            return {}

    def create_dataset(
        self,
        labels: Optional[Dict[str, int]] = None,
    ) -> Dataset:
        """Create a PyTorch dataset from DICOM directory.
        
        Args:
            labels: Optional mapping of file names to labels
            
        Returns:
            PyTorch Dataset
        """
        return DICOMDataset(self, labels)

    def clear_cache(self) -> None:
        """Clear the image cache."""
        self._cache.clear()


class DICOMDataset(Dataset):
    """PyTorch Dataset for DICOM images."""

    def __init__(
        self,
        handler: DICOMDataHandler,
        labels: Optional[Dict[str, int]] = None,
    ):
        """Initialize DICOM dataset.
        
        Args:
            handler: DICOM data handler
            labels: Mapping of file names to labels
        """
        self.handler = handler
        self.labels = labels or {}
        
        # Find all DICOM files
        self.files: List[Path] = list(handler.data_dir.glob("**/*.dcm"))
        
        # Filter to files with labels if provided
        if labels:
            self.files = [f for f in self.files if f.name in labels]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.files[idx]
        
        # Load image
        image = self.handler.load_dicom(str(file_path))
        
        if image is None:
            # Return random data as fallback
            image = np.random.randn(*self.handler.target_size).astype(np.float32)
        
        # Get label
        label = self.labels.get(file_path.name, 0)
        
        # Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        return tensor, label
