"""FHIR data handler for electronic health records.

Provides utilities for loading and preprocessing FHIR-formatted
health records for federated learning with transformers/tabular models.
"""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class FHIRDataHandler:
    """Handler for FHIR electronic health record data.
    
    Provides loading and preprocessing of FHIR resources for
    federated learning on EHR data. Focuses on privacy-preserving
    feature extraction.
    """

    def __init__(
        self,
        data_dir: str,
        resource_types: Optional[List[str]] = None,
        max_sequence_length: int = 512,
    ):
        """Initialize FHIR handler.
        
        Args:
            data_dir: Directory containing FHIR bundles
            resource_types: List of resource types to extract
            max_sequence_length: Maximum sequence length for tokenization
        """
        self.data_dir = Path(data_dir)
        self.resource_types = resource_types or [
            "Patient",
            "Observation",
            "Condition",
            "MedicationRequest",
            "DiagnosticReport",
        ]
        self.max_sequence_length = max_sequence_length
        
        # Feature vocabulary (built during loading)
        self.vocab: Dict[str, int] = {}
        self.vocab_size = 0

    def load_bundle(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a FHIR bundle from JSON file.
        
        Args:
            file_path: Path to FHIR bundle JSON
            
        Returns:
            Parsed bundle dictionary or None
        """
        try:
            with open(file_path, 'r') as f:
                bundle = json.load(f)
            return bundle
        except Exception as e:
            print(f"Error loading FHIR bundle {file_path}: {e}")
            return None

    def extract_features(
        self,
        bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract features from FHIR bundle.
        
        Note: Only extracts non-identifying clinical features.
        
        Args:
            bundle: FHIR bundle dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            "observation_codes": [],
            "condition_codes": [],
            "medication_codes": [],
            "vital_signs": {},
            "lab_values": {},
        }
        
        # Extract from entries
        entries = bundle.get("entry", [])
        
        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            
            if resource_type == "Observation":
                self._extract_observation(resource, features)
            elif resource_type == "Condition":
                self._extract_condition(resource, features)
            elif resource_type == "MedicationRequest":
                self._extract_medication(resource, features)
        
        return features

    def _extract_observation(
        self,
        resource: Dict[str, Any],
        features: Dict[str, Any],
    ) -> None:
        """Extract features from Observation resource."""
        # Get observation code
        code_obj = resource.get("code", {})
        codings = code_obj.get("coding", [])
        
        for coding in codings:
            code = coding.get("code")
            if code:
                features["observation_codes"].append(code)
        
        # Get value if numeric
        value_quantity = resource.get("valueQuantity", {})
        value = value_quantity.get("value")
        
        if value is not None and codings:
            code = codings[0].get("code", "unknown")
            features["lab_values"][code] = float(value)

    def _extract_condition(
        self,
        resource: Dict[str, Any],
        features: Dict[str, Any],
    ) -> None:
        """Extract features from Condition resource."""
        code_obj = resource.get("code", {})
        codings = code_obj.get("coding", [])
        
        for coding in codings:
            code = coding.get("code")
            if code:
                features["condition_codes"].append(code)

    def _extract_medication(
        self,
        resource: Dict[str, Any],
        features: Dict[str, Any],
    ) -> None:
        """Extract features from MedicationRequest resource."""
        medication = resource.get("medicationCodeableConcept", {})
        codings = medication.get("coding", [])
        
        for coding in codings:
            code = coding.get("code")
            if code:
                features["medication_codes"].append(code)

    def features_to_tensor(
        self,
        features: Dict[str, Any],
    ) -> torch.Tensor:
        """Convert extracted features to tensor.
        
        Args:
            features: Extracted feature dictionary
            
        Returns:
            Feature tensor
        """
        # Build vocabulary if needed
        all_codes = (
            features["observation_codes"] +
            features["condition_codes"] +
            features["medication_codes"]
        )
        
        for code in all_codes:
            if code not in self.vocab:
                self.vocab[code] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        
        # Create feature vector
        feature_vector = np.zeros(max(self.vocab_size, 1000), dtype=np.float32)
        
        # One-hot encode codes
        for code in all_codes:
            idx = self.vocab.get(code, 0)
            if idx < len(feature_vector):
                feature_vector[idx] = 1.0
        
        # Add lab values
        for i, (code, value) in enumerate(features.get("lab_values", {}).items()):
            if 500 + i < len(feature_vector):
                feature_vector[500 + i] = value
        
        return torch.from_numpy(feature_vector)

    def create_dataset(
        self,
        labels: Optional[Dict[str, int]] = None,
    ) -> Dataset:
        """Create PyTorch dataset from FHIR directory.
        
        Args:
            labels: Mapping of file names to labels
            
        Returns:
            PyTorch Dataset
        """
        return FHIRDataset(self, labels)


class FHIRDataset(Dataset):
    """PyTorch Dataset for FHIR EHR data."""

    def __init__(
        self,
        handler: FHIRDataHandler,
        labels: Optional[Dict[str, int]] = None,
    ):
        """Initialize FHIR dataset.
        
        Args:
            handler: FHIR data handler
            labels: Mapping of file names to labels
        """
        self.handler = handler
        self.labels = labels or {}
        
        # Find all JSON files
        self.files: List[Path] = list(handler.data_dir.glob("**/*.json"))
        
        # Filter to files with labels if provided
        if labels:
            self.files = [f for f in self.files if f.name in labels]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.files[idx]
        
        # Load and extract features
        bundle = self.handler.load_bundle(str(file_path))
        
        if bundle is None:
            # Return random data as fallback
            features = torch.randn(1000)
        else:
            extracted = self.handler.extract_features(bundle)
            features = self.handler.features_to_tensor(extracted)
        
        # Get label
        label = self.labels.get(file_path.name, 0)
        
        return features, label
