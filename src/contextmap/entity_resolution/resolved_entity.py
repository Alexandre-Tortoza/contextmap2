"""The resolved entity: what explicit ``MATCH`` decisions justify, derived from exact members.

A :class:`ResolvedEntity` is the *belief* that several source entities are one physical object, plus
the exact lineage of that belief. It is derived only from its member
:class:`~contextmap.semantic_mapping.Entity` records and the ``MATCH``
:class:`~contextmap.entity_resolution.ResolutionDecision` records that grouped them; it never
replaces
them. The source entities stay immutable and individually addressable by their own
:class:`~contextmap.semantic_mapping.EntityReference`, and every one of them belongs to exactly one
resolved entity (an entity nothing matched is a resolved entity of one member), so a consumer such
as Spatial Relations needs nothing but resolved entities and never depends on a source mutation.

Identity is scoped to one resolution artifact: a :class:`ResolvedEntityId` is derived from the
sorted
references of the members, so identical immutable inputs always give the same id, and the stable
handle is the :class:`~contextmap.entity_resolution.ResolvedEntityReference`
``(resolution_run_id, resolved_entity_id)``. Nothing implies permanence across artifacts.

Aggregation is exact and never invents confidence:

* **geometry** is the *union of the members' geometry references* (the authoritative support, kept
  as
  references, never copied), with the union of their bounds, which is exact; references held by more
  than one member are counted, not duplicated;
* **semantics** keeps every member hypothesis, attribute and uncertainty record verbatim, with no
  primary hypothesis; the ambiguity of the whole is at least that of any member and becomes
  ambiguous when the members propose different labels, so a merge never looks more certain than its
  parts;
* **time** is recomputed from the exact union of the members' physical observations, each counted
  once; an observation the members share keeps the largest of their inference counts (a lower
  bound, since the members may count the same inference results);
* **evidence links** are the union of the members' links.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import NewType

from contextmap.entity_resolution._checks import require_canonical, require_present
from contextmap.entity_resolution.decision import ResolutionDecisionId
from contextmap.entity_resolution.models import (
    EntityResolutionRunId,
    PolicyRef,
    ResolvedEntityId,
    ResolvedEntityReference,
    reference_order,
)
from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId
from contextmap.semantic_mapping import (
    AmbiguityState,
    EntityAttribute,
    EntityEvidenceLinks,
    EntityHypothesis,
    EntityReference,
    EntityTemporalState,
    EntityUncertainty,
    geometry_set_digest,
)

MATERIALIZATION_POLICY_ID = "connected-components-materialization-v1"
"""Versioned identity of the grouping and aggregation rules described in this module."""

GEOMETRY_UNION_RULE_ID = "union-of-member-geometry-v1"
"""Versioned identity of the rule that derives the geometry of a resolved entity."""

TEMPORAL_UNION_RULE_ID = "resolved-union-temporal-summary-v1"
"""Versioned identity of the rule that derives the temporal state of a resolved entity."""

ContradictionId = NewType("ContradictionId", str)
"""Identity of one transitivity contradiction, derived from the ``DISTINCT`` decision."""

_AMBIGUITY_RANK = {
    AmbiguityState.UNAMBIGUOUS: 0,
    AmbiguityState.INSUFFICIENT_EVIDENCE: 1,
    AmbiguityState.AMBIGUOUS: 2,
    AmbiguityState.CONFLICTING: 3,
}


class UnknownResolvedEntityError(KeyError):
    """Raised when a resolution artifact has no resolved entity with the referenced id."""


class ForeignResolvedEntityReferenceError(ValueError):
    """Raised when a reference names a resolution artifact other than the one resolving it."""


def resolved_entity_id_for(members: tuple[EntityReference, ...]) -> ResolvedEntityId:
    """Compute the deterministic identity of the resolved entity of a set of members.

    Args:
        members: The references of the members, sorted canonically.

    Returns:
        A pure function of the member references, so identical inputs give the same id whatever the
        order of the decisions or of the entities. It is local to a resolution artifact.
    """
    material = "\n".join("/".join(reference_order(reference)) for reference in members)
    return ResolvedEntityId(f"resolved--{hashlib.sha256(material.encode()).hexdigest()[:16]}")


def contradiction_id_for(distinct_decision_id: ResolutionDecisionId) -> ContradictionId:
    """Compute the deterministic identity of the contradiction a ``DISTINCT`` decision causes."""
    digest = hashlib.sha256(distinct_decision_id.encode()).hexdigest()[:16]
    return ContradictionId(f"contradiction--{digest}")


@dataclass(frozen=True, kw_only=True)
class ResolvedMember:
    """One source entity of a resolved entity, with the decisions that grouped it.

    Attributes:
        entity_ref: The source entity, exactly as its semantic map names it.
        matched_by: The ``MATCH`` decisions that involve this member, sorted and unique; empty for
            the only member of a resolved entity nothing matched.
    """

    entity_ref: EntityReference
    matched_by: tuple[ResolutionDecisionId, ...]

    def __post_init__(self) -> None:
        """Require canonical decision ordering.

        Raises:
            ValueError: If the decisions are not sorted and unique.
        """
        require_canonical("matched_by", self.matched_by, lambda item: (item,))


@dataclass(frozen=True, kw_only=True)
class ResolvedGeometry:
    """The union of the geometry supports of the members, as references and exact bounds.

    Attributes:
        geometry_refs: The authoritative support: every geometry reference of every member, sorted
            by ``geometry_id``, unique, never empty, from one geometric map.
        map_frame: The frame of that map; ``bounds`` is expressed in it.
        bounds: The tight axis-aligned box of the union, which is the union of the member bounds.
        duplicate_reference_count: References held by more than one member; each appears once in
            ``geometry_refs``.
        input_geometry_digest: Digest of the exact set of references, so the summary is tied to it.
        summary_rule_id: The versioned rule that derived the summary.
    """

    geometry_refs: tuple[GeometryReference, ...]
    map_frame: str
    bounds: Bounds3D
    duplicate_reference_count: int
    input_geometry_digest: str
    summary_rule_id: str

    def __post_init__(self) -> None:
        """Validate that the support is one canonical set of one map and the summary is its own.

        Raises:
            ValueError: If the support is empty, unsorted, repeated or spans several maps, the
                bounds are in another frame, the duplicate count is negative, or the digest is not
                the one of the support.
        """
        require_present(self, "map_frame", "input_geometry_digest", "summary_rule_id")
        if not self.geometry_refs:
            raise ValueError("geometry_refs must not be empty")
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda reference: (reference.geometry_id,),
            detail="by geometry_id ",
        )
        maps = {reference.map_id for reference in self.geometry_refs}
        if len(maps) > 1:
            raise ValueError(f"geometry_refs must belong to one map, got {sorted(maps)!r}")
        if self.bounds.frame_id != self.map_frame:
            raise ValueError(
                f"bounds are expressed in {self.bounds.frame_id!r}, not the map frame "
                f"{self.map_frame!r}"
            )
        if self.duplicate_reference_count < 0:
            raise ValueError("duplicate_reference_count must not be negative")
        if self.input_geometry_digest != geometry_set_digest(self.geometry_refs):
            raise ValueError("the summary was not derived from exactly these geometry references")

    @property
    def geometric_map_id(self) -> MapId:
        """The immutable geometric map the support belongs to."""
        return self.geometry_refs[0].map_id

    @property
    def extent_m(self) -> tuple[float, float, float]:
        """Side lengths of ``bounds``, in meters."""
        low, high = self.bounds.minimum_m, self.bounds.maximum_m
        return (high[0] - low[0], high[1] - low[1], high[2] - low[2])

    @property
    def bounds_center_m(self) -> tuple[float, float, float]:
        """The center of ``bounds``, in meters: a representative position, not a centroid."""
        low, high = self.bounds.minimum_m, self.bounds.maximum_m
        return ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, (low[2] + high[2]) / 2)


@dataclass(frozen=True, kw_only=True)
class MemberAmbiguity:
    """How settled the semantic state of one member was.

    Attributes:
        entity_ref: The member.
        ambiguity_state: Its state, as its semantic map recorded it.
    """

    entity_ref: EntityReference
    ambiguity_state: AmbiguityState


def derive_resolved_ambiguity(
    members: tuple[MemberAmbiguity, ...], hypotheses: tuple[EntityHypothesis, ...]
) -> AmbiguityState:
    """Derive how settled the merged semantic state is, never more than its parts.

    Args:
        members: The state of every member.
        hypotheses: The union of the member hypotheses.

    Returns:
        ``CONFLICTING`` or ``AMBIGUOUS`` when any member is, ``AMBIGUOUS`` also when the members
        propose different labels, ``INSUFFICIENT_EVIDENCE`` when no member has a hypothesis, and
        ``UNAMBIGUOUS`` otherwise. A member without hypotheses does not make the others uncertain.
    """
    worst = max((item.ambiguity_state for item in members), key=_AMBIGUITY_RANK.__getitem__)
    if worst in (AmbiguityState.CONFLICTING, AmbiguityState.AMBIGUOUS):
        return worst
    if not hypotheses:
        return AmbiguityState.INSUFFICIENT_EVIDENCE
    if len({item.label for item in hypotheses}) > 1:
        return AmbiguityState.AMBIGUOUS
    return AmbiguityState.UNAMBIGUOUS


def uncertainty_key(item: EntityUncertainty) -> tuple[str, ...]:
    """The canonical sort key of an uncertainty record."""
    record = item.record
    return (
        item.fused_evidence_id,
        record.kind.value,
        record.rule_id,
        "|".join(record.hypothesis_ids),
        "|".join(f"{ref.contribution_id}/{ref.claim_id or ''}" for ref in record.evidence),
    )


@dataclass(frozen=True, kw_only=True)
class ResolvedSemanticState:
    """Every member hypothesis, attribute and uncertainty, kept verbatim, with no primary.

    Merging entities never makes the semantics more certain: there is no primary hypothesis, no
    combined confidence, and a disagreement between members is visible in ``ambiguity_state``.

    Attributes:
        hypotheses: The union of the member hypotheses, sorted by fused evidence and hypothesis and
            unique.
        attributes: The union of the member attributes, sorted and unique.
        uncertainty: The union of the member uncertainty records, sorted and unique.
        member_ambiguity: How settled each member was, sorted by member.
        ambiguity_state: How settled the whole is; always the one :func:`derive_resolved_ambiguity`
            gives.
    """

    hypotheses: tuple[EntityHypothesis, ...]
    attributes: tuple[EntityAttribute, ...]
    uncertainty: tuple[EntityUncertainty, ...]
    member_ambiguity: tuple[MemberAmbiguity, ...]
    ambiguity_state: AmbiguityState

    def __post_init__(self) -> None:
        """Validate ordering and that the ambiguity is the one the members imply.

        Raises:
            ValueError: If a collection is not sorted and unique, there is no member, or the
                ambiguity state is not derived from the members and hypotheses.
        """
        if not self.member_ambiguity:
            raise ValueError("member_ambiguity must not be empty")
        require_canonical(
            "hypotheses",
            self.hypotheses,
            lambda item: (item.fused_evidence_id, item.hypothesis_id),
        )
        require_canonical(
            "attributes",
            self.attributes,
            lambda item: (item.name, item.value, item.origin.value, item.derivation_id),
        )
        require_canonical("uncertainty", self.uncertainty, uncertainty_key)
        require_canonical(
            "member_ambiguity", self.member_ambiguity, lambda item: reference_order(item.entity_ref)
        )
        expected = derive_resolved_ambiguity(self.member_ambiguity, self.hypotheses)
        if self.ambiguity_state is not expected:
            raise ValueError(
                f"ambiguity_state {self.ambiguity_state.value!r} does not match the members, "
                f"which imply {expected.value!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ResolvedEntityProvenance:
    """How a resolved entity was materialized.

    Attributes:
        policy: The grouping and aggregation policy and its configuration.
        code_version: Code revision that produced it, when known.
    """

    policy: PolicyRef
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Validate the code version.

        Raises:
            ValueError: If the code version is present but empty.
        """
        if self.code_version is not None:
            require_present(self, "code_version")


@dataclass(frozen=True, kw_only=True)
class ResolvedEntity:
    """One physical object as the resolution believes it, with the exact lineage of the belief.

    Attributes:
        resolved_entity_id: Identity, local to ``resolution_run_id``, derived from the members.
        resolution_run_id: The resolution artifact that owns the entity; makes the identity scope
            explicit on the record itself.
        members: The source entities, sorted canonically and unique, never empty.
        geometry: The union of the members' geometry supports.
        semantic_state: Every member hypothesis, attribute and uncertainty, with no primary.
        evidence: The union of the members' evidence links.
        temporal_state: When and how often the members were seen, recomputed from the exact union.
        resolution_decision_refs: The ``MATCH`` decisions that grouped the members, sorted and
            unique; the union of the members' ``matched_by``.
        unresolved_neighbor_refs: Entities outside this one that a decision left ``UNRESOLVED``
            against a member, sorted and unique.
        contradiction_ids: The transitivity contradictions this entity is withheld by, sorted and
            unique; the members were matched transitively but contradicted by a ``DISTINCT``.
        provenance: How it was materialized.
    """

    resolved_entity_id: ResolvedEntityId
    resolution_run_id: EntityResolutionRunId
    members: tuple[ResolvedMember, ...]
    geometry: ResolvedGeometry
    semantic_state: ResolvedSemanticState
    evidence: EntityEvidenceLinks
    temporal_state: EntityTemporalState
    resolution_decision_refs: tuple[ResolutionDecisionId, ...]
    unresolved_neighbor_refs: tuple[EntityReference, ...]
    contradiction_ids: tuple[ContradictionId, ...]
    provenance: ResolvedEntityProvenance

    def __post_init__(self) -> None:
        """Validate the identity, the lineage and that the parts describe the same members.

        Raises:
            ValueError: If an identity is empty, there is no member, the members or a reference
                list is not sorted and unique, the id is not the one derived from the members, the
                decision refs are not the members' decisions, a neighbor is also a member, or the
                semantic state does not describe exactly the members.
        """
        require_present(self, "resolved_entity_id", "resolution_run_id")
        if not self.members:
            raise ValueError("a resolved entity needs at least one member")
        require_canonical("members", self.members, lambda item: reference_order(item.entity_ref))
        refs = self.member_entity_refs
        if self.resolved_entity_id != resolved_entity_id_for(refs):
            raise ValueError("resolved_entity_id is not the identity derived from the members")
        require_canonical(
            "resolution_decision_refs", self.resolution_decision_refs, lambda item: (item,)
        )
        if set(self.resolution_decision_refs) != {
            decision for member in self.members for decision in member.matched_by
        }:
            raise ValueError("resolution_decision_refs must be the decisions of the members")
        require_canonical(
            "unresolved_neighbor_refs", self.unresolved_neighbor_refs, reference_order
        )
        if set(self.unresolved_neighbor_refs) & set(refs):
            raise ValueError("an unresolved neighbor cannot be a member")
        require_canonical("contradiction_ids", self.contradiction_ids, lambda item: (item,))
        if [item.entity_ref for item in self.semantic_state.member_ambiguity] != list(refs):
            raise ValueError("the semantic state must describe exactly the members")

    @property
    def member_entity_refs(self) -> tuple[EntityReference, ...]:
        """The references of the source entities, in canonical order."""
        return tuple(member.entity_ref for member in self.members)

    @property
    def reference(self) -> ResolvedEntityReference:
        """The stable handle of this resolved entity."""
        return ResolvedEntityReference(
            resolution_run_id=self.resolution_run_id, resolved_entity_id=self.resolved_entity_id
        )


@dataclass(frozen=True, kw_only=True)
class TransitivityContradiction:
    """Matches that chain two entities which another decision says are distinct.

    ``A MATCH B`` and ``B MATCH C`` group ``A`` and ``C``, but ``A DISTINCT C`` says they are not
    the same object. Connected components would silently hide that; it is recorded instead, and the
    entities of the component are withheld from merging.

    Attributes:
        contradiction_id: Identity, derived from the ``DISTINCT`` decision.
        distinct_decision_id: The ``DISTINCT`` decision between two members of one component.
        distinct_pair: The two entities that decision separates.
        match_path: The ``MATCH`` decisions that chain them, in order along the shortest path.
        component: Every entity of the component that was withheld, sorted and unique.
    """

    contradiction_id: ContradictionId
    distinct_decision_id: ResolutionDecisionId
    distinct_pair: tuple[EntityReference, EntityReference]
    match_path: tuple[ResolutionDecisionId, ...]
    component: tuple[EntityReference, ...]

    def __post_init__(self) -> None:
        """Validate the identity and that the pair is inside the withheld component.

        Raises:
            ValueError: If the id is not derived from the decision, the chain is shorter than two
                matches, the component is not sorted and unique or the pair is not inside it.
        """
        if self.contradiction_id != contradiction_id_for(self.distinct_decision_id):
            raise ValueError("contradiction_id is not the identity derived from the decision")
        if len(self.match_path) < 2:
            raise ValueError("a transitivity contradiction needs a chain of at least two matches")
        require_canonical("component", self.component, reference_order)
        if not set(self.distinct_pair) <= set(self.component):
            raise ValueError("the contradicted pair must be inside the withheld component")


@dataclass(frozen=True, kw_only=True)
class ResolvedEntitySet:
    """The resolved entities of one resolution artifact, addressable by reference.

    Attributes:
        resolution_run_id: The resolution artifact the entities belong to.
        entities: Every resolved entity, sorted by id and unique, owned by ``resolution_run_id``,
            whose members are disjoint: a source entity belongs to exactly one.
    """

    resolution_run_id: EntityResolutionRunId
    entities: tuple[ResolvedEntity, ...]

    def __post_init__(self) -> None:
        """Validate ids, ownership and that no source entity is in two resolved entities.

        Raises:
            ValueError: If the identity is empty, the entities are not sorted by id and unique, one
                belongs to another artifact, or a source entity is a member of two of them.
        """
        require_present(self, "resolution_run_id")
        require_canonical("entities", self.entities, lambda item: (item.resolved_entity_id,))
        counts: Counter[EntityReference] = Counter()
        for entity in self.entities:
            if entity.resolution_run_id != self.resolution_run_id:
                raise ValueError(
                    f"resolved entity {entity.resolved_entity_id!r} belongs to resolution run "
                    f"{entity.resolution_run_id!r}, not {self.resolution_run_id!r}"
                )
            counts.update(entity.member_entity_refs)
        repeated = sorted(reference_order(ref) for ref, count in counts.items() if count > 1)
        if repeated:
            raise ValueError(
                f"source entities belong to more than one resolved entity: {repeated!r}"
            )

    def resolve(self, reference: ResolvedEntityReference) -> ResolvedEntity:
        """Resolve a reference to its resolved entity.

        Args:
            reference: A reference to a resolved entity of this artifact.

        Returns:
            The resolved entity the reference names.

        Raises:
            ForeignResolvedEntityReferenceError: If the reference names another artifact: an id is
                only meaningful inside the artifact that owns it.
            UnknownResolvedEntityError: If this artifact has no such resolved entity.
        """
        if reference.resolution_run_id != self.resolution_run_id:
            raise ForeignResolvedEntityReferenceError(
                f"reference names resolution run {reference.resolution_run_id!r}, but this is "
                f"{self.resolution_run_id!r}"
            )
        for entity in self.entities:
            if entity.resolved_entity_id == reference.resolved_entity_id:
                return entity
        raise UnknownResolvedEntityError(reference.resolved_entity_id)

    def resolved_of(self, entity_ref: EntityReference) -> ResolvedEntity:
        """The resolved entity a source entity belongs to.

        Args:
            entity_ref: A source entity.

        Returns:
            The resolved entity that has it as a member.

        Raises:
            KeyError: If no resolved entity has this member.
        """
        for entity in self.entities:
            if entity_ref in entity.member_entity_refs:
                return entity
        raise KeyError(entity_ref)


@dataclass(frozen=True, kw_only=True)
class ResolvedEntityMaterialization:
    """The outcome of materializing resolved entities from decisions.

    Attributes:
        resolved: The resolved entities.
        contradictions: The transitivity contradictions, sorted by id and unique, each surfaced
            explicitly instead of being hidden by the grouping.
        policy: The grouping and aggregation policy and its configuration.
    """

    resolved: ResolvedEntitySet
    contradictions: tuple[TransitivityContradiction, ...]
    policy: PolicyRef

    def __post_init__(self) -> None:
        """Validate ordering and that each contradiction is recorded on the entities it withholds.

        Raises:
            ValueError: If the contradictions are not sorted and unique, or one is not recorded on
                every entity of its component.
        """
        require_canonical(
            "contradictions", self.contradictions, lambda item: (item.contradiction_id,)
        )
        for contradiction in self.contradictions:
            for entity_ref in contradiction.component:
                entity = self.resolved.resolved_of(entity_ref)
                if contradiction.contradiction_id not in entity.contradiction_ids:
                    raise ValueError(
                        f"contradiction {contradiction.contradiction_id!r} is not recorded on "
                        f"the entity {entity_ref.entity_id!r} it withholds"
                    )
