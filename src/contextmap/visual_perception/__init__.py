"""Public contracts for backend-agnostic visual perception evidence."""

from .region_models import (
    ArtifactReference,
    BackendScore,
    BoundingBox,
    CoordinateConvention,
    InlineMask,
    Region2D,
    RegionCandidate,
    RegionIdentity,
    RegionProvenance,
    RejectedRegionCandidate,
    RejectionReason,
)

__all__ = [
    "ArtifactReference",
    "BackendScore",
    "BoundingBox",
    "CoordinateConvention",
    "InlineMask",
    "Region2D",
    "RegionCandidate",
    "RegionIdentity",
    "RegionProvenance",
    "RejectedRegionCandidate",
    "RejectionReason",
]
