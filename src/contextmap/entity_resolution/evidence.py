"""Match evidence: everything known about whether two entities are the same object.

An :class:`EntityMatchEvidence` is the record of one comparison. It keeps the typed evidence of
every channel that was evaluated, separately and with its own units, and the results of the hard
validity gates that decide whether the comparison could be attempted at all. It carries no score
that summarizes the channels: a downstream consumer can answer *why entities were merged*, *which
evidence supported the match*, *which evidence argued against it* and *which gate prevented the
comparison* from the record itself.

The pair is always in canonical order, ``entity_a_ref`` sorting strictly before ``entity_b_ref``
by ``(semantic_map_id, entity_id)``, so a comparison has one identity whichever way it was asked
and two equivalent records encode to the same bytes. The evidence is *evidence*, not a decision:
deciding is the resolution policy's job, and a comparison never changes an entity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import NewType

from contextmap.entity_resolution._checks import require_canonical, require_present
from contextmap.entity_resolution.channels import (
    AppearanceEvidence,
    ChannelEvidence,
    EvidenceStatus,
    Finding,
    GeometryEvidence,
    MatchChannel,
    PointRepresentationEvidence,
    SemanticEvidence,
    TemporalEvidence,
    UnavailableReason,
)
from contextmap.entity_resolution.models import reference_order
from contextmap.semantic_mapping import Entity, EntityReference

COMPARISON_GATES_POLICY_ID = "entity-comparison-gates-v1"
"""Versioned identity of the hard validity gates evaluated by :func:`evaluate_comparison_gates`."""

ComparisonId = NewType("ComparisonId", str)
"""Identity of one comparison of a pair of entities, derived from the pair."""


def comparison_id_for(entity_a_ref: EntityReference, entity_b_ref: EntityReference) -> ComparisonId:
    """Compute the deterministic identity of the comparison of a pair.

    Args:
        entity_a_ref: The entity whose reference sorts first.
        entity_b_ref: The other entity.

    Returns:
        A pure function of the pair, so the identity does not depend on the order in which pairs
        were retrieved or on any other comparison.
    """
    material = "\n".join((*reference_order(entity_a_ref), *reference_order(entity_b_ref)))
    return ComparisonId(f"comparison--{hashlib.sha256(material.encode()).hexdigest()[:16]}")


@dataclass(frozen=True, kw_only=True)
class GateResult:
    """The outcome of one hard validity gate.

    Attributes:
        gate_id: Identity of the gate.
        passed: Whether the pair may be compared as far as this gate is concerned.
        detail: A deterministic, human-readable explanation, naming the values compared.
    """

    gate_id: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        """Require the gate and an explanation.

        Raises:
            ValueError: If the gate or the detail is empty.
        """
        require_present(self, "gate_id", "detail")


@dataclass(frozen=True, kw_only=True)
class MatchEvidenceProvenance:
    """How the gates of a comparison were evaluated.

    Attributes:
        gate_policy_id: Versioned identity of the gate rules.
        code_version: Code revision that produced the evidence, when known.
    """

    gate_policy_id: str
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Validate the identities.

        Raises:
            ValueError: If the gate policy is empty, or the code version is present but empty.
        """
        require_present(self, "gate_policy_id")
        if self.code_version is not None:
            require_present(self, "code_version")


def evaluate_comparison_gates(entity_a: Entity, entity_b: Entity) -> tuple[GateResult, ...]:
    """Evaluate the hard validity gates of a comparison.

    A failed gate means the pair cannot be compared as given: coordinates of different geometric
    maps or frames are never compared without an explicit alignment, and an entity is never
    compared with itself.

    Args:
        entity_a: One entity.
        entity_b: The other entity.

    Returns:
        One result per gate, sorted by gate identity.
    """
    geometry_a, geometry_b = entity_a.geometry, entity_b.geometry
    results = (
        GateResult(
            gate_id="distinct-entities",
            passed=entity_a.reference != entity_b.reference,
            detail=f"{entity_a.reference.entity_id!r} and {entity_b.reference.entity_id!r}",
        ),
        GateResult(
            gate_id="same-geometric-map",
            passed=geometry_a.geometric_map_id == geometry_b.geometric_map_id,
            detail=f"{geometry_a.geometric_map_id!r} and {geometry_b.geometric_map_id!r}",
        ),
        GateResult(
            gate_id="same-map-frame",
            passed=geometry_a.map_frame == geometry_b.map_frame,
            detail=f"{geometry_a.map_frame!r} and {geometry_b.map_frame!r}",
        ),
    )
    return tuple(sorted(results, key=lambda result: result.gate_id))


@dataclass(frozen=True, kw_only=True)
class EntityMatchEvidence:
    """All the evidence of one comparison between two entities.

    A channel that is ``None`` was **not evaluated** (for example because the policy disabled it);
    a channel that was evaluated but could not compare the pair is present and ``UNAVAILABLE``.

    Attributes:
        comparison_id: Identity of the comparison, derived from the pair.
        entity_a_ref: The entity whose reference sorts first.
        entity_b_ref: The other entity; sorts strictly after ``entity_a_ref``.
        gates: The hard validity gates, sorted by identity and unique.
        provenance: How the gates were evaluated.
        geometry: Geometric evidence.
        semantic: Semantic compatibility evidence.
        appearance: Visual appearance evidence.
        temporal: Temporal compatibility evidence.
        point_representation: Optional 3D structural evidence.
    """

    comparison_id: ComparisonId
    entity_a_ref: EntityReference
    entity_b_ref: EntityReference
    gates: tuple[GateResult, ...]
    provenance: MatchEvidenceProvenance
    geometry: GeometryEvidence | None = None
    semantic: SemanticEvidence | None = None
    appearance: AppearanceEvidence | None = None
    temporal: TemporalEvidence | None = None
    point_representation: PointRepresentationEvidence | None = None

    def __post_init__(self) -> None:
        """Validate the pair, its identity, the gates and that a blocked comparison is empty.

        Raises:
            TypeError: If a channel slot holds the evidence of another channel.
            ValueError: If the pair is not two different entities in canonical order, the
                comparison id is not the one derived from the pair, the gates are not sorted and
                unique, or a comparison blocked by a failed gate carries available evidence.
        """
        require_present(self, "comparison_id")
        order_a, order_b = reference_order(self.entity_a_ref), reference_order(self.entity_b_ref)
        if order_a == order_b:
            raise ValueError("an entity cannot be compared with itself")
        if order_a > order_b:
            raise ValueError("the pair must be in canonical order: entity_a_ref sorts first")
        if self.comparison_id != comparison_id_for(self.entity_a_ref, self.entity_b_ref):
            raise ValueError("comparison_id is not the identity derived from the pair")
        require_canonical("gates", self.gates, lambda gate: (gate.gate_id,))
        for channel, evidence in self._slots():
            if evidence is not None and evidence.channel is not channel:
                raise TypeError(
                    f"{channel.value} must hold {channel.value} evidence, got "
                    f"{evidence.channel.value} evidence"
                )
        if self.blocked:
            for channel, evidence in self.channels():
                unavailable = evidence.unavailable
                if (
                    unavailable is None
                    or unavailable.reason is not UnavailableReason.BLOCKED_BY_GATE
                ):
                    raise ValueError(
                        f"the comparison is blocked by a failed gate, so {channel.value} cannot "
                        f"carry available evidence"
                    )

    @property
    def blocked(self) -> bool:
        """Whether a hard gate failed, so the pair could not be compared."""
        return any(not gate.passed for gate in self.gates)

    def failed_gates(self) -> tuple[GateResult, ...]:
        """The gates that prevented the comparison."""
        return tuple(gate for gate in self.gates if not gate.passed)

    def channels(self) -> tuple[tuple[MatchChannel, ChannelEvidence], ...]:
        """The evaluated channels, in canonical order, each with its evidence."""
        return tuple(
            (channel, evidence) for channel, evidence in self._slots() if evidence is not None
        )

    def channel_status(self, channel: MatchChannel) -> EvidenceStatus | None:
        """What one channel says, or ``None`` when it was not evaluated."""
        for evaluated, evidence in self.channels():
            if evaluated is channel:
                return evidence.status
        return None

    def supporting_channels(self) -> tuple[MatchChannel, ...]:
        """The channels whose findings support the same-object hypothesis."""
        return self._channels_with(EvidenceStatus.SUPPORTING)

    def conflicting_channels(self) -> tuple[MatchChannel, ...]:
        """The channels with a finding that argues against the same-object hypothesis."""
        return self._channels_with(EvidenceStatus.CONFLICTING)

    def unavailable_channels(self) -> tuple[MatchChannel, ...]:
        """The evaluated channels that could not compare the pair."""
        return self._channels_with(EvidenceStatus.UNAVAILABLE)

    def signals(self) -> tuple[tuple[MatchChannel, Finding], ...]:
        """Every supporting or conflicting finding, each with the channel that produced it."""
        return tuple(
            (channel, finding)
            for channel, evidence in self.channels()
            for finding in evidence.findings
            if finding.status in (EvidenceStatus.SUPPORTING, EvidenceStatus.CONFLICTING)
        )

    def _channels_with(self, status: EvidenceStatus) -> tuple[MatchChannel, ...]:
        return tuple(channel for channel, evidence in self.channels() if evidence.status is status)

    def _slots(self) -> tuple[tuple[MatchChannel, ChannelEvidence | None], ...]:
        return (
            (MatchChannel.GEOMETRY, self.geometry),
            (MatchChannel.SEMANTIC, self.semantic),
            (MatchChannel.APPEARANCE, self.appearance),
            (MatchChannel.TEMPORAL, self.temporal),
            (MatchChannel.POINT_REPRESENTATION, self.point_representation),
        )
