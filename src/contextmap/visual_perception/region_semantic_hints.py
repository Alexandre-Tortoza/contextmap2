"""Link native proposal text to the canonical region its geometry reached, as evidence only.

A region-producing task such as Florence-2 ``<OD>`` or ``<DENSE_REGION_CAPTION>`` returns
text and geometry from the same inference. The candidate keeps that text as
:class:`~contextmap.visual_perception.region_models.NativeRegionText`; this module derives,
from the already materialized candidates and normalization, one :class:`RegionSemanticHint`
per proposal that carries text. The derivation reruns no model, never writes text into
``Region2D`` and never produces a ``SemanticClaim``: the hint is frame evidence that a
consumer may score or ignore.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .models import RegionId
from .normalization import NormalizationResult
from .region_models import NativeRegionText, RegionCandidate, RegionProvenance


class HintContribution(StrEnum):
    """How the hinted proposal's own geometry took part in canonical regions.

    The text describes the proposal's geometry, not the region's: a ``MERGED`` hint names
    the region that absorbed its proposal, but the region's frozen geometry comes from
    another proposal, so its text must not be read as the region's own.
    """

    REPRESENTATIVE = "representative"
    MERGED = "merged"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RegionSemanticHint:
    """Task-native text of one proposal, linked to the canonical region it reached.

    Attributes:
        candidate_id: Globally remapped candidate identity (pass-prefixed).
        source_observation_id: Physical observation the proposal was produced for.
        perception_run_id: Discovery execution that produced the proposal.
        perception_result_id: Result scope in which ``region_id`` is meaningful.
        provenance: Backend, checkpoint, configuration digest, discovery pass and native
            proposal identity of the proposal that carried the text.
        native_text: Task, optional prompt and verbatim text of that proposal.
        contribution: Whether the proposal's geometry is the region's frozen geometry,
            was merged into it, or reached no region.
        region_id: Region the proposal contributed to; ``None`` exactly when
            ``contribution`` is ``REJECTED``.
    """

    candidate_id: str
    source_observation_id: str
    perception_run_id: str
    perception_result_id: str
    provenance: RegionProvenance
    native_text: NativeRegionText
    contribution: HintContribution
    region_id: RegionId | None = None

    def __post_init__(self) -> None:
        """Require identities and a region link consistent with the contribution."""
        identities = (
            self.candidate_id,
            self.source_observation_id,
            self.perception_run_id,
            self.perception_result_id,
        )
        if any(not identity for identity in identities):
            raise ValueError("region semantic hint identity fields must not be empty")
        if (self.region_id is None) != (self.contribution is HintContribution.REJECTED):
            raise ValueError(
                "region semantic hint region_id must be set exactly when its proposal "
                "reached a canonical region"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible evidence record."""
        return {
            "candidate_id": self.candidate_id,
            "source_observation_id": self.source_observation_id,
            "perception_run_id": self.perception_run_id,
            "perception_result_id": self.perception_result_id,
            "provenance": self.provenance.to_dict(),
            "native_text": self.native_text.to_dict(),
            "contribution": self.contribution.value,
            "region_id": None if self.region_id is None else str(self.region_id),
        }


def derive_region_semantic_hints(
    candidates: Sequence[RegionCandidate],
    normalization: NormalizationResult,
) -> tuple[RegionSemanticHint, ...]:
    """Derive one hint per text-carrying candidate from one materialized discovery.

    The link follows the explicit lineage of normalization: a candidate belongs to the
    region whose ``contributor_candidate_ids`` list it, and it is ``MERGED`` exactly when a
    ``MergeDecision`` names it as the merged proposal. A candidate listed by no frozen
    region (rejected, or merged into a group later cut by the region budget) keeps its hint
    with ``REJECTED`` and no region.

    Args:
        candidates: Globally remapped candidates that were normalized, for example
            ``DiscoveryRunResult.candidates``.
        normalization: The normalization of exactly those candidates.

    Returns:
        Hints ordered by ``candidate_id``, independent of the input order.

    Raises:
        ValueError: If candidate ids repeat, or the normalization lineage does not belong
            to these candidates or does not name one representative per region.
    """
    by_id: dict[str, RegionCandidate] = {}
    for candidate in candidates:
        if candidate.candidate_id in by_id:
            raise ValueError("candidate ids must be unique to derive region semantic hints")
        by_id[candidate.candidate_id] = candidate

    merged_into = {
        decision.merged_candidate_id: decision.representative_candidate_id
        for decision in normalization.merge_decisions
    }
    region_of: dict[str, tuple[RegionId, HintContribution]] = {}
    for region in normalization.regions:
        unknown = [item for item in region.contributor_candidate_ids if item not in by_id]
        if unknown:
            raise ValueError(
                f"region {region.region_id} contributor {unknown[0]!r} is not among the "
                "discovery candidates"
            )
        representatives = [
            item for item in region.contributor_candidate_ids if item not in merged_into
        ]
        # Sem exatamente um representante, a região não diz de quem é a geometria congelada,
        # e um texto fundido poderia ser lido como o texto da própria região.
        if len(representatives) != 1 or any(
            merged_into[item] not in region.contributor_candidate_ids
            for item in region.contributor_candidate_ids
            if item in merged_into
        ):
            raise ValueError(
                f"region {region.region_id} merge lineage does not name exactly one "
                "representative among its contributors"
            )
        for item in region.contributor_candidate_ids:
            contribution = (
                HintContribution.MERGED if item in merged_into else HintContribution.REPRESENTATIVE
            )
            region_of[item] = (region.region_id, contribution)

    hints: list[RegionSemanticHint] = []
    for candidate_id in sorted(by_id):
        candidate = by_id[candidate_id]
        if candidate.native_text is None:
            continue
        region_id, contribution = region_of.get(candidate_id, (None, HintContribution.REJECTED))
        hints.append(
            RegionSemanticHint(
                candidate_id=candidate_id,
                source_observation_id=candidate.source_observation_id,
                perception_run_id=candidate.perception_run_id,
                perception_result_id=candidate.perception_result_id,
                provenance=candidate.provenance,
                native_text=candidate.native_text,
                contribution=contribution,
                region_id=region_id,
            )
        )
    return tuple(hints)
