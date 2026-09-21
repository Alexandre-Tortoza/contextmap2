"""The evidence chain behind an entity.

An entity is a compact record that *references* the evidence that created it. It never copies
images, masks, embeddings or fusion payloads: it names the fused evidence, and everything
upstream stays where it was produced.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.semantic_fusion import FusedEvidenceId, FusionSupportId, SemanticFusionRunId
from contextmap.semantic_mapping._checks import require_canonical, require_present


@dataclass(frozen=True, kw_only=True)
class FusedEvidenceRef:
    """Reference to the evidence fused over one support, inside one persisted fusion run.

    Attributes:
        fusion_run_id: The Semantic Fusion run artifact that owns the evidence.
        fused_evidence_id: The fused evidence, local to that run.
        fusion_support_id: The support the evidence was accumulated over.
    """

    fusion_run_id: SemanticFusionRunId
    fused_evidence_id: FusedEvidenceId
    fusion_support_id: FusionSupportId

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "fusion_run_id", "fused_evidence_id", "fusion_support_id")


@dataclass(frozen=True, kw_only=True)
class EntityEvidenceLinks:
    """Where the evidence that supports an entity can be found.

    Attributes:
        fused_evidence: The fused evidence the entity was materialized from, sorted by run and
            evidence and unique; never empty, because an entity with no evidence behind it
            cannot be audited.
    """

    fused_evidence: tuple[FusedEvidenceRef, ...]

    def __post_init__(self) -> None:
        """Validate that the entity has evidence and that it is listed canonically.

        Raises:
            ValueError: If there is no fused evidence, or the references are not sorted and
                unique.
        """
        if not self.fused_evidence:
            raise ValueError("fused_evidence must not be empty: an entity needs evidence")
        require_canonical(
            "fused_evidence",
            self.fused_evidence,
            lambda ref: (ref.fusion_run_id, ref.fused_evidence_id),
        )
