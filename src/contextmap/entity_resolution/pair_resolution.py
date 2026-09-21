"""Collecting the evidence of a pair and resolving the candidate pairs of a run.

:class:`MatchEvidenceBuilder` runs the hard validity gates and then every configured channel,
producing the evidence of one comparison. A channel that is not configured is not evaluated
(``None``); a configured one that cannot compare the pair is present and unavailable. When a gate
fails, no channel is computed at all: every configured channel is recorded as unavailable because
the gate blocked the comparison, so the record names the gate that prevented it and nothing
measured across incompatible coordinates leaks into the evidence.

:func:`resolve_candidate_pairs` is the policy execution service. It takes the pairs that candidate
retrieval produced (never all pairs), builds each pair's evidence and decides it. It has no side
effect on the entities and does not merge anything: turning ``MATCH`` decisions into resolved
entities is a separate, explicit step.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from contextmap.entity_resolution.appearance_comparison import AppearanceComparator
from contextmap.entity_resolution.channels import (
    AppearanceEvidence,
    GeometryEvidence,
    PointRepresentationEvidence,
    SemanticEvidence,
    TemporalEvidence,
    Unavailability,
    UnavailableReason,
)
from contextmap.entity_resolution.decision import ResolutionDecision
from contextmap.entity_resolution.evidence import (
    COMPARISON_GATES_POLICY_ID,
    EntityMatchEvidence,
    MatchEvidenceProvenance,
    comparison_id_for,
    evaluate_comparison_gates,
)
from contextmap.entity_resolution.geometry_comparison import (
    GeometryComparisonPolicy,
    compare_geometry,
)
from contextmap.entity_resolution.models import reference_order
from contextmap.entity_resolution.representation_comparison import RepresentationComparator
from contextmap.entity_resolution.resolution_policy import ConservativeResolutionPolicy, decide
from contextmap.entity_resolution.retrieval import EntityCandidateSet, candidate_pairs
from contextmap.entity_resolution.semantic_comparison import (
    SemanticCompatibilityPolicy,
    compare_semantics,
)
from contextmap.entity_resolution.temporal_comparison import (
    TemporalCompatibilityPolicy,
    compare_temporal,
)
from contextmap.geometric_mapping import GeometrySource
from contextmap.semantic_mapping import Entity, UnknownEntityError


@dataclass(frozen=True, kw_only=True)
class ComparisonChannels:
    """Which evidence channels a run evaluates, and how.

    Geometry is required: the baseline is anchored in it. Every other channel is optional and is
    evaluated only when configured, so the geometry-only path is a configuration with nothing else.

    Attributes:
        geometry: The thresholds of the geometric rules.
        geometry_source: The read boundary of the geometric map, needed only when the geometry
            policy asks for nearest-point statistics.
        semantic: The semantic compatibility rules, or ``None`` to not evaluate the channel.
        temporal: The temporal rule, or ``None`` to not evaluate the channel.
        appearance: The appearance comparator, or ``None`` to not evaluate the channel.
        representation: The 3D structure comparator, or ``None`` to not evaluate the channel.
    """

    geometry: GeometryComparisonPolicy
    geometry_source: GeometrySource | None = None
    semantic: SemanticCompatibilityPolicy | None = None
    temporal: TemporalCompatibilityPolicy | None = None
    appearance: AppearanceComparator | None = None
    representation: RepresentationComparator | None = None


class MatchEvidenceBuilder:
    """Builds the evidence of entity pairs from a fixed set of configured channels."""

    def __init__(self, channels: ComparisonChannels, *, code_version: str | None = None) -> None:
        """Create a builder.

        Args:
            channels: The channels to evaluate.
            code_version: Code revision that produces the evidence, when known.
        """
        self._channels = channels
        self._code_version = code_version

    def build(self, entity_a: Entity, entity_b: Entity) -> EntityMatchEvidence:
        """Collect the evidence of one comparison.

        The result does not depend on the order of the arguments: the pair is put in canonical
        order and the evidence refers to it.

        Args:
            entity_a: One entity.
            entity_b: The other entity.

        Returns:
            The evidence: the gate results and the evidence of every configured channel, or, when a
            gate failed, every configured channel unavailable because the gate blocked it.

        Raises:
            ValueError: If both arguments are the same entity.
        """
        first, second = sorted(
            (entity_a, entity_b), key=lambda item: reference_order(item.reference)
        )
        if first.reference == second.reference:
            raise ValueError("an entity cannot be compared with itself")
        gates = evaluate_comparison_gates(first, second)
        common = {
            "comparison_id": comparison_id_for(first.reference, second.reference),
            "entity_a_ref": first.reference,
            "entity_b_ref": second.reference,
            "gates": gates,
            "provenance": MatchEvidenceProvenance(
                gate_policy_id=COMPARISON_GATES_POLICY_ID, code_version=self._code_version
            ),
        }
        failed = [gate for gate in gates if not gate.passed]
        if failed:
            blocked = Unavailability(
                reason=UnavailableReason.BLOCKED_BY_GATE,
                detail="; ".join(f"{gate.gate_id}: {gate.detail}" for gate in failed),
            )
            return EntityMatchEvidence(**common, **self._blocked_channels(blocked))  # type: ignore[arg-type]
        channels = self._channels
        return EntityMatchEvidence(
            **common,  # type: ignore[arg-type]
            geometry=compare_geometry(
                first, second, channels.geometry, source=channels.geometry_source
            ),
            semantic=None
            if channels.semantic is None
            else compare_semantics(first, second, channels.semantic),
            appearance=None
            if channels.appearance is None
            else channels.appearance.compare(first, second),
            temporal=None
            if channels.temporal is None
            else compare_temporal(first, second, channels.temporal),
            point_representation=None
            if channels.representation is None
            else channels.representation.compare(first, second),
        )

    def _blocked_channels(self, blocked: Unavailability) -> dict[str, object]:
        channels = self._channels
        evidence: dict[str, object] = {
            "geometry": GeometryEvidence(policy=channels.geometry.ref(), unavailable=blocked)
        }
        if channels.semantic is not None:
            evidence["semantic"] = SemanticEvidence(
                policy=channels.semantic.ref(), unavailable=blocked
            )
        if channels.appearance is not None:
            evidence["appearance"] = AppearanceEvidence(
                policy=channels.appearance.policy.ref(), unavailable=blocked
            )
        if channels.temporal is not None:
            evidence["temporal"] = TemporalEvidence(
                policy=channels.temporal.ref(), unavailable=blocked
            )
        if channels.representation is not None:
            evidence["point_representation"] = PointRepresentationEvidence(
                policy=channels.representation.policy.ref(), unavailable=blocked
            )
        return evidence


@dataclass(frozen=True, kw_only=True)
class PairResolution:
    """The evidence of one pair and the decision reached from it.

    Attributes:
        evidence: The typed evidence of the comparison.
        decision: The verdict of the policy on it; ``decision.evidence_ref`` is the evidence's id.
    """

    evidence: EntityMatchEvidence
    decision: ResolutionDecision

    def __post_init__(self) -> None:
        """Validate that the decision was reached from this evidence.

        Raises:
            ValueError: If the decision refers to another comparison.
        """
        if self.decision.evidence_ref != self.evidence.comparison_id:
            raise ValueError("the decision was not reached from this evidence")


def resolve_candidate_pairs(
    entities: Iterable[Entity],
    candidate_sets: Iterable[EntityCandidateSet],
    builder: MatchEvidenceBuilder,
    policy: ConservativeResolutionPolicy,
    *,
    code_version: str | None = None,
) -> tuple[PairResolution, ...]:
    """Build the evidence and decide every candidate pair of a run.

    Only the pairs candidate retrieval produced are compared, each once, in canonical order.

    Args:
        entities: The entities, from which the candidate references are resolved.
        candidate_sets: The candidate sets of the run.
        builder: Collects the evidence of a pair.
        policy: Decides each pair.
        code_version: Code revision that produced the decisions, when known.

    Returns:
        One resolution per distinct candidate pair, in canonical order. The entities are not
        modified and nothing is merged.

    Raises:
        UnknownEntityError: If a candidate set names an entity that was not given.
    """
    by_reference = {entity.reference: entity for entity in entities}
    resolutions: list[PairResolution] = []
    for first_ref, second_ref in candidate_pairs(candidate_sets):
        try:
            first, second = by_reference[first_ref], by_reference[second_ref]
        except KeyError as error:
            raise UnknownEntityError(error.args[0]) from None
        evidence = builder.build(first, second)
        resolutions.append(
            PairResolution(
                evidence=evidence, decision=decide(evidence, policy, code_version=code_version)
            )
        )
    return tuple(resolutions)
