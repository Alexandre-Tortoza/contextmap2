"""Public contract for the semantic fusion capability.

Semantic Fusion accumulates evidence from several views over a shared spatial
support and keeps it as :class:`FusedEvidence`: the competing hypotheses, the
exact claims and typed signals behind each, the conflicts between physical
observations and what is simply unknown. It creates no entity identity, elects
no winner and never merges heterogeneous scores into one number. See
``src/contextmap/semantic_fusion/docs/README.md`` for the full capability
documentation.
"""

from contextmap.semantic_fusion.grouping import (
    PHYSICAL_OBSERVATION_GROUPING_POLICY_ID,
    PhysicalObservationGrouping,
    group_by_physical_observation,
)
from contextmap.semantic_fusion.models import (
    EvidenceContribution,
    EvidenceContributionId,
    EvidenceReference,
    EvidenceStance,
    FusedEvidence,
    FusedEvidenceId,
    FusedEvidenceProvenance,
    FusedHypothesis,
    FusedHypothesisId,
    FusionSupport,
    FusionSupportId,
    FusionSupportProvenance,
    HypothesisEvidence,
    ObservationQualityRef,
    PhysicalObservationGroup,
    PointRepresentationRef,
    ScoreReference,
    SupportSignal,
    SupportSignalKind,
    UncertaintyKind,
    UncertaintyRecord,
)
from contextmap.semantic_fusion.support import (
    GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
    ExcludedObservation,
    FusionSupportBuild,
    GeometryOverlapSupportPolicy,
    build_fusion_supports,
)

__all__ = [
    "GEOMETRY_OVERLAP_SUPPORT_POLICY_ID",
    "PHYSICAL_OBSERVATION_GROUPING_POLICY_ID",
    "EvidenceContribution",
    "EvidenceContributionId",
    "EvidenceReference",
    "EvidenceStance",
    "ExcludedObservation",
    "FusedEvidence",
    "FusedEvidenceId",
    "FusedEvidenceProvenance",
    "FusedHypothesis",
    "FusedHypothesisId",
    "FusionSupport",
    "FusionSupportBuild",
    "FusionSupportId",
    "FusionSupportProvenance",
    "GeometryOverlapSupportPolicy",
    "HypothesisEvidence",
    "ObservationQualityRef",
    "PhysicalObservationGroup",
    "PhysicalObservationGrouping",
    "PointRepresentationRef",
    "ScoreReference",
    "SupportSignal",
    "SupportSignalKind",
    "UncertaintyKind",
    "UncertaintyRecord",
    "build_fusion_supports",
    "group_by_physical_observation",
]
