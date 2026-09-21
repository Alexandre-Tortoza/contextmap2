"""Entities and relations as the ContextMap composes them.

The map does not own what an entity or a relation *is*: Entity Resolution and Spatial Relations
do. It owns how they are composed into one readable whole. Each record therefore states three
things: its identity inside the map, the upstream record it maps to (never a silent rewrite of
the upstream identity), and the part of the upstream result a consumer needs without opening
the upstream artifact.

Geometry stays authoritative in the geometric-map artifact, so an entity holds
:class:`GeometryReference` values and never coordinates. Uncertainty is never resolved here: an
ambiguous or conflicting entity keeps every hypothesis, and an unresolved relation stays
unresolved. Support values are deliberately not copied; the evidence behind a result stays
reachable through the upstream artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contextmap.artifact._checks import require_canonical, require_present
from contextmap.artifact.provenance import EvidenceOrigin
from contextmap.artifact.references import (
    ContextEntityId,
    ContextEntityReference,
    ContextRelationId,
    UpstreamRecordRef,
)
from contextmap.geometric_mapping import GeometryReference


class AmbiguityStatus(Enum):
    """How settled the semantic state of an entity is.

    The value must agree with the number of hypotheses, so it can be read without inspecting
    them, and a competing state can never be collapsed into one label by the map.

    Attributes:
        UNAMBIGUOUS: Exactly one hypothesis exists and nothing competes with it.
        AMBIGUOUS: Several hypotheses compete, without contradicting observations.
        CONFLICTING: Distinct observations back incompatible hypotheses.
        INSUFFICIENT_EVIDENCE: No hypothesis is supported; an abstention, not negative evidence.
    """

    UNAMBIGUOUS = "unambiguous"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, kw_only=True)
class LabelHypothesis:
    """One thing an entity may be, as an open-vocabulary label.

    A label is a semantic hypothesis, not the identity of an object in the world. There is no
    score: the numbers that support it stay in the upstream artifact, which the origin cites.

    Attributes:
        label: The label text, verbatim.
        origin: How the hypothesis was produced and the evidence it rests on.
    """

    label: str
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        """Require the label.

        Raises:
            ValueError: If the label is blank.
        """
        require_present(self, "label")


@dataclass(frozen=True, kw_only=True)
class ContextSemanticState:
    """What an entity may be, with every hypothesis kept.

    Attributes:
        status: How settled the state is.
        hypotheses: Every label hypothesis, sorted by label and unique. The order carries no
            ranking.
    """

    status: AmbiguityStatus
    hypotheses: tuple[LabelHypothesis, ...]

    def __post_init__(self) -> None:
        """Validate that the status agrees with the hypotheses.

        Raises:
            ValueError: If the hypotheses are not sorted and unique, an unambiguous state does
                not have exactly one, a competing state has fewer than two, or a state with
                insufficient evidence has any.
        """
        require_canonical(
            "hypotheses", self.hypotheses, lambda item: (item.label,), detail="by label "
        )
        count = len(self.hypotheses)
        if self.status is AmbiguityStatus.UNAMBIGUOUS and count != 1:
            raise ValueError(f"an unambiguous state has exactly one hypothesis, got {count}")
        if self.status in (AmbiguityStatus.AMBIGUOUS, AmbiguityStatus.CONFLICTING) and count < 2:
            raise ValueError(
                f"a {self.status.value} state keeps at least two hypotheses, got {count}"
            )
        if self.status is AmbiguityStatus.INSUFFICIENT_EVIDENCE and count:
            raise ValueError(f"a state with insufficient evidence has no hypothesis, got {count}")


@dataclass(frozen=True, kw_only=True)
class ContextEntity:
    """One resolved entity, as the map composes it.

    Attributes:
        entity_id: Identity of the entity, unique inside the map.
        source: The upstream record this entity maps to; the identity mapping is explicit and
            two entities never map to the same record.
        member_entities: The source entities this entity was resolved from, sorted and unique;
            at least one. An entity that merged several keeps all of them.
        resolution_decisions: The resolution decisions behind the entity, sorted and unique;
            empty when no decision was needed.
        geometry_refs: The authoritative geometry support, by reference: at least one element,
            sorted and unique. Never coordinates.
        semantic_state: What the entity may be, with every hypothesis kept.
        origin: How the entity was produced and the evidence it rests on.
    """

    entity_id: ContextEntityId
    source: UpstreamRecordRef
    member_entities: tuple[UpstreamRecordRef, ...]
    resolution_decisions: tuple[UpstreamRecordRef, ...]
    geometry_refs: tuple[GeometryReference, ...]
    semantic_state: ContextSemanticState
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        """Validate the identity, the resolution lineage and the geometry support.

        Raises:
            ValueError: If the identity is blank, the entity has no member entity or no
                geometry support, or a list is not sorted and unique.
        """
        require_present(self, "entity_id")
        if not self.member_entities:
            raise ValueError("member_entities must list the source entities of the entity")
        for name, records in (
            ("member_entities", self.member_entities),
            ("resolution_decisions", self.resolution_decisions),
        ):
            require_canonical(name, records, lambda item: (item.artifact_id, item.record_id))
        if not self.geometry_refs:
            raise ValueError("geometry_refs must list the geometry that supports the entity")
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda item: (item.map_id, item.geometry_id),
        )


class RelationState(Enum):
    """Whether the evidence supports a relation.

    A candidate relation is never a confirmed one, so unresolved and conflicting relations are
    kept in the map with their state instead of being dropped or promoted.

    Attributes:
        SUPPORTED: The evidence supports the relation.
        UNRESOLVED: The evidence neither supports nor refutes it.
        CONFLICTING: The evidence both supports and contradicts it.
    """

    SUPPORTED = "supported"
    UNRESOLVED = "unresolved"
    CONFLICTING = "conflicting"


@dataclass(frozen=True, kw_only=True)
class ContextRelation:
    """One directed relation between two entities of the map.

    Attributes:
        relation_id: Identity of the relation, unique inside the map.
        source: The upstream record this relation maps to; two relations never map to the same
            record.
        subject: The entity the predicate is about.
        predicate: The relation type, from the taxonomy of the producing policy.
        object: The entity the predicate points to. ``subject predicate object`` is directed:
            the relation does not hold in the other direction unless it is stated.
        state: Whether the evidence supports the relation.
        origin: How the relation was derived and the evidence it rests on.
    """

    relation_id: ContextRelationId
    source: UpstreamRecordRef
    subject: ContextEntityReference
    predicate: str
    object: ContextEntityReference
    state: RelationState
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        """Validate the identity, the predicate and that the endpoints differ.

        Raises:
            ValueError: If the identity or the predicate is blank, or the relation relates an
                entity to itself.
        """
        require_present(self, "relation_id", "predicate")
        if self.subject == self.object:
            raise ValueError(
                f"relation {self.relation_id!r} relates entity {self.subject.entity_id!r} to itself"
            )
