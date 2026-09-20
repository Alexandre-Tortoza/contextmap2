"""Public contract for the semantic fusion capability.

Semantic Fusion accumulates evidence from several views over a shared spatial
support and keeps it as :class:`FusedEvidence`: the competing hypotheses, the
exact claims and typed signals behind each, the conflicts between physical
observations and what is simply unknown. It creates no entity identity, elects
no winner and never merges heterogeneous scores into one number. See
``src/contextmap/semantic_fusion/docs/README.md`` for the full capability
documentation.
"""

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

__all__ = [
    "EvidenceContribution",
    "EvidenceContributionId",
    "EvidenceReference",
    "EvidenceStance",
    "FusedEvidence",
    "FusedEvidenceId",
    "FusedEvidenceProvenance",
    "FusedHypothesis",
    "FusedHypothesisId",
    "FusionSupport",
    "FusionSupportId",
    "FusionSupportProvenance",
    "HypothesisEvidence",
    "ObservationQualityRef",
    "PhysicalObservationGroup",
    "PointRepresentationRef",
    "ScoreReference",
    "SupportSignal",
    "SupportSignalKind",
    "UncertaintyKind",
    "UncertaintyRecord",
]
