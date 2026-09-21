"""Public contract for the spatial relations capability.

Spatial Relations describes how persistent resolved entities relate in space. A relation is a
versioned predicate between two resolved entities together with the evidence that supports or
contradicts it and an explicit state (supported, rejected or unresolved); it never replaces that
evidence, never corrects an entity and never answers a natural-language query. See
``src/contextmap/spatial_relations/docs/README.md`` for the full capability documentation.
"""

from contextmap.spatial_relations.candidates import (
    CANDIDATE_POLICY_ID,
    CandidateExclusion,
    CandidateExclusionReason,
    CandidatePolicy,
    CandidateProvenance,
    CandidateReason,
    RelationCandidate,
    RelationCandidateSet,
    SkippedPredicate,
    generate_relation_candidates,
)
from contextmap.spatial_relations.contact_predicates import (
    CONTACT_POLICY_ID,
    CONTACT_PREDICATES,
    ContactPredicatePolicy,
    evaluate_contact_candidates,
    evaluate_contact_predicate,
)
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
from contextmap.spatial_relations.geometric_predicates import (
    GEOMETRIC_POLICY_ID,
    GEOMETRIC_PREDICATES,
    GeometricPredicatePolicy,
    evaluate_geometric_candidates,
    evaluate_geometric_predicate,
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
    "CANDIDATE_POLICY_ID",
    "CONTACT_POLICY_ID",
    "CONTACT_PREDICATES",
    "FRAME_CONVENTIONS_POLICY_ID",
    "GEOMETRIC_POLICY_ID",
    "GEOMETRIC_PREDICATES",
    "PREDICATE_SPECS",
    "TAXONOMY_VERSION",
    "AxisDirection",
    "CandidateExclusion",
    "CandidateExclusionReason",
    "CandidatePolicy",
    "CandidateProvenance",
    "CandidateReason",
    "ContactPredicatePolicy",
    "EvidenceCaveat",
    "EvidenceCaveatKind",
    "FrameConventionError",
    "FrameConventions",
    "FrameRequirement",
    "GeometricPredicatePolicy",
    "IncompatibleFrameError",
    "MeasuredGeometry",
    "PredicateFamily",
    "PredicateSpec",
    "Quantity",
    "Relation",
    "RelationCandidate",
    "RelationCandidateSet",
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
    "SkippedPredicate",
    "UndeclaredAxisError",
    "decode_relation",
    "decode_relation_evidence",
    "encode_relation",
    "encode_relation_evidence",
    "evaluate_contact_candidates",
    "evaluate_contact_predicate",
    "evaluate_geometric_candidates",
    "evaluate_geometric_predicate",
    "evidence_id_for",
    "generate_relation_candidates",
    "predicate_spec",
    "relation_id_for",
]
