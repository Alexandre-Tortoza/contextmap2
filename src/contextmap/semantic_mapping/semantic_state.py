"""The semantic state of an entity.

An entity keeps the semantic evidence accumulated upstream instead of collapsing it into one
label and one confidence. Each hypothesis keeps every claim behind it, with the typed signals
each producer emitted, so a missing score stays distinguishable from a low one and an
abstention from a negative.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.semantic_fusion import (
    EvidenceStance,
    FusedEvidenceId,
    FusedHypothesisId,
    HypothesisEvidence,
)
from contextmap.semantic_mapping._checks import require_canonical, require_present


@dataclass(frozen=True, kw_only=True)
class EntityHypothesis:
    """One semantic candidate of an entity, with every piece of evidence behind it.

    The label is the open-vocabulary text exactly as it was proposed. There is no probability
    and no combined confidence: the typed signals of each evidence item stay separate.

    Attributes:
        fused_evidence_id: The fused evidence the hypothesis comes from.
        hypothesis_id: The hypothesis, local to that fused evidence.
        label: The hypothesis text, verbatim.
        evidence: The claims that support, conflict with or are ambiguous about the hypothesis,
            sorted by contribution and claim and unique.
    """

    fused_evidence_id: FusedEvidenceId
    hypothesis_id: FusedHypothesisId
    label: str
    evidence: tuple[HypothesisEvidence, ...]

    def __post_init__(self) -> None:
        """Validate the identities, the ordering and that something supports the hypothesis.

        Raises:
            ValueError: If an identity or the label is empty, the evidence is not sorted and
                unique, or no evidence item supports the hypothesis.
        """
        require_present(self, "fused_evidence_id", "hypothesis_id", "label")
        require_canonical(
            "evidence", self.evidence, lambda item: (item.contribution_id, item.claim_id)
        )
        if not any(item.stance is EvidenceStance.SUPPORTING for item in self.evidence):
            raise ValueError(
                f"hypothesis {self.label!r} needs at least one supporting evidence item"
            )


@dataclass(frozen=True, kw_only=True)
class EntitySemanticState:
    """What an entity may be, with nothing collapsed.

    Attributes:
        hypotheses: Every candidate interpretation, sorted by fused evidence and hypothesis
            and unique; empty when no view produced a claim.
    """

    hypotheses: tuple[EntityHypothesis, ...] = ()

    def __post_init__(self) -> None:
        """Validate the ordering and that a label is not repeated within one fused evidence.

        Raises:
            ValueError: If the hypotheses are not sorted and unique, or one fused evidence
                lists the same label twice.
        """
        require_canonical(
            "hypotheses",
            self.hypotheses,
            lambda item: (item.fused_evidence_id, item.hypothesis_id),
        )
        labels: set[tuple[str, str]] = set()
        for hypothesis in self.hypotheses:
            key = (hypothesis.fused_evidence_id, hypothesis.label)
            if key in labels:
                raise ValueError(
                    f"label {hypothesis.label!r} appears twice in fused evidence "
                    f"{hypothesis.fused_evidence_id!r}"
                )
            labels.add(key)
