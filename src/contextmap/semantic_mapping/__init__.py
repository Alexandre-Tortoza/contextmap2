"""Public contract for the semantic mapping capability.

Semantic Mapping materializes fused evidence as persistent semantic entities inside one
semantic-map artifact. An :class:`Entity` keeps the exact 3D support, every semantic hypothesis,
the evidence chain and the temporal state; it never decides whether two entities are the same
physical object, which belongs to Entity Resolution. See
``src/contextmap/semantic_mapping/docs/README.md`` for the full capability documentation.
"""

from contextmap.semantic_mapping.evidence import EntityEvidenceLinks, FusedEvidenceRef
from contextmap.semantic_mapping.geometry import EntityGeometry
from contextmap.semantic_mapping.models import (
    Entity,
    EntityId,
    EntityProvenance,
    EntityReference,
    EntitySet,
    ForeignEntityReferenceError,
    SemanticMapId,
    UnknownEntityError,
)
from contextmap.semantic_mapping.semantic_state import EntityHypothesis, EntitySemanticState
from contextmap.semantic_mapping.temporal import EntityTemporalState

__all__ = [
    "Entity",
    "EntityEvidenceLinks",
    "EntityGeometry",
    "EntityHypothesis",
    "EntityId",
    "EntityProvenance",
    "EntityReference",
    "EntitySemanticState",
    "EntitySet",
    "EntityTemporalState",
    "ForeignEntityReferenceError",
    "FusedEvidenceRef",
    "SemanticMapId",
    "UnknownEntityError",
]
