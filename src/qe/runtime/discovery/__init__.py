"""Model discovery and auto-routing for free LLM providers."""

from qe.runtime.discovery.schemas import DiscoveredModel, ModelHealthMetrics, TierAssignment
from qe.runtime.discovery.service import ModelDiscoveryService

__all__ = [
    "DiscoveredModel",
    "ModelDiscoveryService",
    "ModelHealthMetrics",
    "TierAssignment",
]
