"""Public contract for the Visual Perception capability.

Visual Perception turns one physical
:class:`~contextmap.ingestion.SourceObservation` into backend-agnostic
visual evidence (regions, features, semantic claims) for one configured
:class:`PerceptionRun`, without deciding persistent 3D entity identity,
final semantic meaning, or 2D→3D projection. See
``src/contextmap/visual_perception/docs/README.md`` for the full
capability documentation.
"""

from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    ClaimId,
    FeatureId,
    FeatureScope,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRun,
    PerceptionRunId,
    Region2D,
    RegionId,
    SceneContext,
    SemanticClaim,
    VisualFeature,
)

__all__ = [
    "BackendProvenance",
    "BoundingBox2D",
    "ClaimId",
    "FeatureId",
    "FeatureScope",
    "HypothesisRole",
    "PerceptionResult",
    "PerceptionResultId",
    "PerceptionRun",
    "PerceptionRunId",
    "Region2D",
    "RegionId",
    "SceneContext",
    "SemanticClaim",
    "VisualFeature",
]
