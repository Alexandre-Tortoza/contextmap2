"""Entities and relations as the ContextMap composes them.

The map does not own what an entity or a relation *is*: Entity Resolution and Spatial Relations
do, and their identities, states and predicates are used here as they define them. The map owns
how they are composed into one readable whole. Each record therefore states three things: its
identity inside the map, the upstream record it maps to (a :class:`ResolvedEntityReference` for
an entity, the run and relation identities for a relation, never a silent rewrite of the
upstream identity), and the part of the upstream result a consumer needs without opening the
upstream artifact.

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
)
from contextmap.entity_resolution import ResolutionDecisionId, ResolvedEntityReference
from contextmap.geometric_mapping import GeometryReference
from contextmap.semantic_mapping import EntityReference
from contextmap.spatial_relations import (
    RelationId,
    RelationPredicate,
    RelationState,
    RelationUncertaintyKind,
    SpatialRelationsRunId,
)


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
        source: The resolved entity this entity maps to; the identity mapping is explicit and
            two entities never map to the same resolved entity.
        member_entities: The source entities this entity was resolved from, sorted and unique;
            at least one. An entity that merged several keeps all of them.
        resolution_decisions: The resolution decisions that grouped the members, local to the
            resolution run of ``source``, sorted and unique; empty when no decision was needed.
        unresolved_neighbors: Source entities outside this one that a decision left
            ``UNRESOLVED`` against a member: neither merged nor declared distinct. Sorted and
            unique, and never a member.
        geometry_refs: The authoritative geometry support, by reference: at least one element,
            sorted and unique. Never coordinates.
        semantic_state: What the entity may be, with every hypothesis kept.
        origin: How the entity was produced and the evidence it rests on.
    """

    entity_id: ContextEntityId
    source: ResolvedEntityReference
    member_entities: tuple[EntityReference, ...]
    resolution_decisions: tuple[ResolutionDecisionId, ...]
    unresolved_neighbors: tuple[EntityReference, ...]
    geometry_refs: tuple[GeometryReference, ...]
    semantic_state: ContextSemanticState
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        """Validate the identity, the resolution lineage and the geometry support.

        Raises:
            ValueError: If the identity is blank, the entity has no member entity or no
                geometry support, a list is not sorted and unique, or an unresolved neighbor is
                also a member.
        """
        require_present(self, "entity_id")
        if not self.member_entities:
            raise ValueError("member_entities must list the source entities of the entity")
        for name, references in (
            ("member_entities", self.member_entities),
            ("unresolved_neighbors", self.unresolved_neighbors),
        ):
            require_canonical(
                name,
                references,
                lambda item: (str(item.semantic_map_id), str(item.entity_id)),
            )
        require_canonical("resolution_decisions", self.resolution_decisions, lambda item: (item,))
        shared = set(self.member_entities) & set(self.unresolved_neighbors)
        if shared:
            raise ValueError(
                f"an unresolved neighbor cannot also be a member: {sorted(map(repr, shared))}"
            )
        if not self.geometry_refs:
            raise ValueError("geometry_refs must list the geometry that supports the entity")
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda item: (item.map_id, item.geometry_id),
        )


@dataclass(frozen=True, kw_only=True)
class ContextRelation:
    """One directed relation between two entities of the map.

    A candidate relation is never a confirmed one: an unresolved or rejected relation stays in the
    map with its state instead of being dropped or promoted, and a consumer that wants only the
    confirmed ones filters by ``RelationState.SUPPORTED``.

    Attributes:
        relation_id: Identity of the relation, unique inside the map.
        source_run_id: The Spatial Relations run that decided the relation.
        source_relation_id: The relation, local to that run. Together with ``source_run_id`` it
            names the upstream relation; two relations never map to the same one.
        subject: The entity the predicate is about.
        predicate: The canonical predicate of the Spatial Relations taxonomy.
        object: The entity the predicate points to. ``subject predicate object`` is directed:
            the relation does not hold in the other direction unless it is stated.
        state: The state decided by Spatial Relations: supported, rejected or unresolved.
        uncertainty_kinds: Why an unresolved relation could not be decided, sorted and unique;
            empty for a supported or rejected one.
        origin: How the relation was derived and the evidence it rests on.
    """

    relation_id: ContextRelationId
    source_run_id: SpatialRelationsRunId
    source_relation_id: RelationId
    subject: ContextEntityReference
    predicate: RelationPredicate
    object: ContextEntityReference
    state: RelationState
    uncertainty_kinds: tuple[RelationUncertaintyKind, ...]
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        """Validate the identities, the endpoints and that the state agrees with the uncertainty.

        Raises:
            ValueError: If an identity is blank, the relation relates an entity to itself, the
                uncertainty kinds are not sorted and unique, an unresolved relation does not say
                why, or a supported or rejected one carries uncertainty.
        """
        require_present(self, "relation_id", "source_run_id", "source_relation_id")
        if self.subject == self.object:
            raise ValueError(
                f"relation {self.relation_id!r} relates entity {self.subject.entity_id!r} to itself"
            )
        require_canonical(
            "uncertainty_kinds",
            self.uncertainty_kinds,
            lambda item: (item.value,),
            detail="by value ",
        )
        if self.state is RelationState.UNRESOLVED:
            if not self.uncertainty_kinds:
                raise ValueError("an unresolved relation says why: it needs uncertainty kinds")
        elif self.uncertainty_kinds:
            raise ValueError(f"a {self.state.value} relation carries no uncertainty")
