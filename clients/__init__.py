"""Clients package for federated learning."""

from clients.base_client import BaseClient
from clients.hospital_client import HospitalClient, start_client

__all__ = [
    "BaseClient",
    "HospitalClient",
    "start_client",
]
