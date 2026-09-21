"""Public contract for the spatial relations capability.

Spatial Relations describes how persistent resolved entities relate in space. A relation is a
versioned predicate between two resolved entities together with the evidence that supports or
contradicts it and an explicit state (supported, rejected or unresolved); it never replaces that
evidence, never corrects an entity and never answers a natural-language query. See
``src/contextmap/spatial_relations/docs/README.md`` for the full capability documentation.
"""

from contextmap.spatial_relations.evidence import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    MeasuredGeometry,
    Quantity,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceId,
    RelationEvidenceProvenance,
    RelationEvidenceStatus,
    evidence_id_for,
)
from contextmap.spatial_relations.frame_conventions import (
    FRAME_CONVENTIONS_POLICY_ID,
    AxisDirection,
    FrameConventionError,
    FrameConventions,
    IncompatibleFrameError,
    UndeclaredAxisError,
)
from contextmap.spatial_relations.models import (
    Relation,
    RelationId,
    RelationProvenance,
    RelationState,
    RelationUncertainty,
    RelationUncertaintyKind,
    relation_id_for,
)
from contextmap.spatial_relations.serialization import (
    decode_relation,
    decode_relation_evidence,
    encode_relation,
    encode_relation_evidence,
)
from contextmap.spatial_relations.taxonomy import (
    PREDICATE_SPECS,
    TAXONOMY_VERSION,
    FrameRequirement,
    PredicateFamily,
    PredicateSpec,
    RelationPredicate,
    predicate_spec,
)

__all__ = [
    "FRAME_CONVENTIONS_POLICY_ID",
    "PREDICATE_SPECS",
    "TAXONOMY_VERSION",
    "AxisDirection",
    "EvidenceCaveat",
    "EvidenceCaveatKind",
    "FrameConventionError",
    "FrameConventions",
    "FrameRequirement",
    "IncompatibleFrameError",
    "MeasuredGeometry",
    "PredicateFamily",
    "PredicateSpec",
    "Quantity",
    "Relation",
    "RelationEvidence",
    "RelationEvidenceChannel",
    "RelationEvidenceId",
    "RelationEvidenceProvenance",
    "RelationEvidenceStatus",
    "RelationId",
    "RelationPredicate",
    "RelationProvenance",
    "RelationState",
    "RelationUncertainty",
    "RelationUncertaintyKind",
    "UndeclaredAxisError",
    "decode_relation",
    "decode_relation_evidence",
    "encode_relation",
    "encode_relation_evidence",
    "evidence_id_for",
    "predicate_spec",
    "relation_id_for",
]
