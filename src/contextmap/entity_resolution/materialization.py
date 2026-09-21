"""Materializing resolved entities from source entities and explicit decisions.

The baseline, ``connected-components-materialization-v1``, is deterministic and entirely driven by
the ``MATCH`` decisions it is given: entities linked by ``MATCH`` form one resolved entity (the
connected components of the match graph), and every other entity is a resolved entity of one member.
Nothing is decided here and nothing is hidden. The order of the entities or the decisions never
changes the result, and the source entities are never modified.

Transitivity is the delicate part. ``A MATCH B`` and ``B MATCH C`` group ``A`` and ``C`` even when
some decision says ``A DISTINCT C``. Merging the component anyway would hide that contradiction, so
the rule is conservative and explicit: a component that contains a ``DISTINCT`` pair is **not
merged**, every member stays its own resolved entity, and a
:class:`~contextmap.entity_resolution.resolved_entity.TransitivityContradiction` names the
``DISTINCT`` decision, the chain of ``MATCH`` decisions that links the pair, and the withheld
entities. A component may hold several contradictions; each withheld entity carries the id of
every one of them, so a consumer sees why a probable match was not honored. An ``UNRESOLVED``
decision never merges, and is recorded as an unresolved
neighbor of both entities.

An entity in a group must share the geometric map and frame of the others (their supports are
unioned as references and their bounds are unioned), and a group must share one clock domain;
anything else is an explicit error, never a silent merge.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from contextmap.entity_resolution.decision import (
    ResolutionDecision,
    ResolutionDecisionId,
    ResolutionOutcome,
)
from contextmap.entity_resolution.models import (
    EntityResolutionRunId,
    PolicyRef,
    reference_order,
)
from contextmap.entity_resolution.resolved_entity import (
    GEOMETRY_UNION_RULE_ID,
    MATERIALIZATION_POLICY_ID,
    TEMPORAL_UNION_RULE_ID,
    ContradictionId,
    MemberAmbiguity,
    ResolvedEntity,
    ResolvedEntityMaterialization,
    ResolvedEntityProvenance,
    ResolvedEntitySet,
    ResolvedGeometry,
    ResolvedMember,
    ResolvedSemanticState,
    TransitivityContradiction,
    contradiction_id_for,
    derive_resolved_ambiguity,
    resolved_entity_id_for,
    uncertainty_key,
)
from contextmap.geometric_mapping import Bounds3D, GeometryReference
from contextmap.semantic_mapping import (
    Entity,
    EntityAttribute,
    EntityEvidenceLinks,
    EntityFeatureRef,
    EntityHypothesis,
    EntityReference,
    EntityTemporalState,
    EntityUncertainty,
    FusedEvidenceRef,
    ObservationRef,
    TemporalProvenance,
    geometry_set_digest,
)

_UNMERGE_CONTRADICTORY_COMPONENTS = "unmerge-contradictory-components"

_Record = TypeVar("_Record")


class MaterializationError(ValueError):
    """Raised when the entities or decisions given to materialization cannot be aggregated."""


def materialization_policy() -> PolicyRef:
    """The identity and configuration of the grouping and aggregation rules.

    The rules have no parameters, so the fingerprint covers the versioned rule identities only.
    """
    canonical = json.dumps(
        {
            "policy_id": MATERIALIZATION_POLICY_ID,
            "contradiction_rule": _UNMERGE_CONTRADICTORY_COMPONENTS,
            "geometry_rule": GEOMETRY_UNION_RULE_ID,
            "temporal_rule": TEMPORAL_UNION_RULE_ID,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return PolicyRef(
        policy_id=MATERIALIZATION_POLICY_ID,
        configuration_fingerprint=f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}",
    )


@dataclass(frozen=True)
class _Graph:
    """The ``MATCH`` graph: adjacency by entity, and the decision that links each pair."""

    neighbors: dict[EntityReference, list[EntityReference]]
    link: dict[frozenset[EntityReference], ResolutionDecisionId]


def materialize_resolved_entities(
    entities: Iterable[Entity],
    decisions: Iterable[ResolutionDecision],
    *,
    resolution_run_id: EntityResolutionRunId,
    code_version: str | None = None,
) -> ResolvedEntityMaterialization:
    """Materialize the resolved entities that explicit decisions justify.

    Args:
        entities: Every source entity of the run, in any order; none is modified.
        decisions: The decisions of the run, in any order; only ``MATCH`` decisions group entities.
        resolution_run_id: The identity of the resolution artifact that will own the entities.
        code_version: Code revision that produced them, when known.

    Returns:
        The resolved entities, one per group (a source entity nothing matched is a group of one),
        and the transitivity contradictions, with the contradictory components left unmerged.

    Raises:
        MaterializationError: If an entity or a pair is repeated, a decision names an unknown
            entity, a group spans different geometric maps, frames or clock domains, or two
            members disagree about the same physical observation.
    """
    by_reference = _index_entities(entities)
    decision_list = _validated_decisions(decisions, by_reference)
    graph = _match_graph(decision_list)
    components = _components(by_reference, graph)
    contradictions = _contradictions(decision_list, components, graph)
    withheld: dict[EntityReference, set[ContradictionId]] = {}
    for item in contradictions:
        for reference in item.component:
            withheld.setdefault(reference, set()).add(item.contradiction_id)
    neighbors = _unresolved_neighbors(decision_list)
    policy = materialization_policy()
    resolved: list[ResolvedEntity] = []
    for component in components:
        groups = (
            [(reference,) for reference in component] if component[0] in withheld else [component]
        )
        for group in groups:
            resolved.append(
                _aggregate(
                    tuple(by_reference[reference] for reference in group),
                    graph,
                    neighbors,
                    withheld,
                    resolution_run_id,
                    ResolvedEntityProvenance(policy=policy, code_version=code_version),
                )
            )
    return ResolvedEntityMaterialization(
        resolved=ResolvedEntitySet(
            resolution_run_id=resolution_run_id,
            entities=tuple(sorted(resolved, key=lambda item: item.resolved_entity_id)),
        ),
        contradictions=tuple(sorted(contradictions, key=lambda item: item.contradiction_id)),
        policy=policy,
    )


def _index_entities(entities: Iterable[Entity]) -> dict[EntityReference, Entity]:
    by_reference: dict[EntityReference, Entity] = {}
    for entity in sorted(entities, key=lambda item: reference_order(item.reference)):
        if entity.reference in by_reference:
            raise MaterializationError(f"entity {entity.reference!r} is repeated")
        by_reference[entity.reference] = entity
    return by_reference


def _validated_decisions(
    decisions: Iterable[ResolutionDecision], by_reference: dict[EntityReference, Entity]
) -> list[ResolutionDecision]:
    seen: set[frozenset[EntityReference]] = set()
    checked: list[ResolutionDecision] = []
    for decision in sorted(decisions, key=lambda item: item.decision_id):
        pair = frozenset((decision.entity_a_ref, decision.entity_b_ref))
        unknown = [ref for ref in pair if ref not in by_reference]
        if unknown:
            raise MaterializationError(
                f"decision {decision.decision_id!r} names entities that were not given: "
                f"{sorted(reference_order(ref) for ref in unknown)!r}"
            )
        if pair in seen:
            raise MaterializationError(
                f"the pair of decision {decision.decision_id!r} was decided more than once: "
                f"a pair has one decision per run"
            )
        seen.add(pair)
        checked.append(decision)
    return checked


def _match_graph(decisions: Sequence[ResolutionDecision]) -> _Graph:
    neighbors: dict[EntityReference, list[EntityReference]] = defaultdict(list)
    link: dict[frozenset[EntityReference], ResolutionDecisionId] = {}
    for decision in decisions:
        if decision.decision is not ResolutionOutcome.MATCH:
            continue
        first, second = decision.entity_a_ref, decision.entity_b_ref
        neighbors[first].append(second)
        neighbors[second].append(first)
        link[frozenset((first, second))] = decision.decision_id
    for adjacent in neighbors.values():
        adjacent.sort(key=reference_order)
    return _Graph(neighbors=dict(neighbors), link=link)


def _components(
    by_reference: dict[EntityReference, Entity], graph: _Graph
) -> list[tuple[EntityReference, ...]]:
    """The connected components of the match graph, each sorted, ordered by first member."""
    seen: set[EntityReference] = set()
    components: list[tuple[EntityReference, ...]] = []
    for start in by_reference:
        if start in seen:
            continue
        component = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for other in graph.neighbors.get(current, ()):
                if other not in component:
                    component.add(other)
                    queue.append(other)
        seen |= component
        components.append(tuple(sorted(component, key=reference_order)))
    return components


def _shortest_match_path(
    graph: _Graph, start: EntityReference, goal: EntityReference
) -> tuple[ResolutionDecisionId, ...]:
    """The ``MATCH`` decisions along the shortest chain between two entities of one component."""
    previous: dict[EntityReference, EntityReference | None] = {start: None}
    queue = deque([start])
    while True:
        # Os dois extremos estão no mesmo componente, então o alvo sempre é alcançado.
        current = queue.popleft()
        if current == goal:
            break
        for other in graph.neighbors.get(current, ()):
            if other not in previous:
                previous[other] = current
                queue.append(other)
    path: list[ResolutionDecisionId] = []
    node: EntityReference = goal
    while (parent := previous[node]) is not None:
        path.append(graph.link[frozenset((parent, node))])
        node = parent
    return tuple(reversed(path))


def _contradictions(
    decisions: Sequence[ResolutionDecision],
    components: list[tuple[EntityReference, ...]],
    graph: _Graph,
) -> list[TransitivityContradiction]:
    component_of = {reference: component for component in components for reference in component}
    found: list[TransitivityContradiction] = []
    for decision in decisions:
        if decision.decision is not ResolutionOutcome.DISTINCT:
            continue
        first, second = decision.entity_a_ref, decision.entity_b_ref
        if component_of[first] != component_of[second]:
            continue
        found.append(
            TransitivityContradiction(
                contradiction_id=contradiction_id_for(decision.decision_id),
                distinct_decision_id=decision.decision_id,
                distinct_pair=(first, second),
                match_path=_shortest_match_path(graph, first, second),
                component=component_of[first],
            )
        )
    return found


def _unresolved_neighbors(
    decisions: Sequence[ResolutionDecision],
) -> dict[EntityReference, set[EntityReference]]:
    neighbors: dict[EntityReference, set[EntityReference]] = defaultdict(set)
    for decision in decisions:
        if decision.decision is ResolutionOutcome.UNRESOLVED:
            neighbors[decision.entity_a_ref].add(decision.entity_b_ref)
            neighbors[decision.entity_b_ref].add(decision.entity_a_ref)
    return neighbors


def _aggregate(
    members: tuple[Entity, ...],
    graph: _Graph,
    unresolved: dict[EntityReference, set[EntityReference]],
    withheld: dict[EntityReference, set[ContradictionId]],
    resolution_run_id: EntityResolutionRunId,
    provenance: ResolvedEntityProvenance,
) -> ResolvedEntity:
    references = tuple(member.reference for member in members)
    in_group = set(references)
    matched = {
        reference: tuple(
            sorted(
                decision
                for pair, decision in graph.link.items()
                if reference in pair and pair <= in_group
            )
        )
        for reference in references
    }
    neighbors = sorted(
        {other for reference in references for other in unresolved.get(reference, ())} - in_group,
        key=reference_order,
    )
    return ResolvedEntity(
        resolved_entity_id=resolved_entity_id_for(references),
        resolution_run_id=resolution_run_id,
        members=tuple(
            ResolvedMember(entity_ref=reference, matched_by=matched[reference])
            for reference in references
        ),
        geometry=_geometry(members),
        semantic_state=_semantic_state(members),
        evidence=_evidence(members),
        temporal_state=_temporal_state(members),
        resolution_decision_refs=tuple(sorted({d for ids in matched.values() for d in ids})),
        unresolved_neighbor_refs=tuple(neighbors),
        contradiction_ids=tuple(
            sorted({item for ref in references for item in withheld.get(ref, ())})
        ),
        provenance=provenance,
    )


def _geometry(members: tuple[Entity, ...]) -> ResolvedGeometry:
    keys = {(item.geometry.geometric_map_id, item.geometry.map_frame) for item in members}
    if len(keys) != 1:
        raise MaterializationError(
            f"members span different geometric maps or frames and cannot be grouped: "
            f"{sorted(keys)!r}"
        )
    counts: dict[str, int] = defaultdict(int)
    references: dict[str, GeometryReference] = {}
    for member in members:
        for reference in member.geometry.geometry_refs:
            counts[reference.geometry_id] += 1
            references[reference.geometry_id] = reference
    ordered = tuple(references[key] for key in sorted(references))
    frame = members[0].geometry.map_frame
    low = tuple(min(item.geometry.bounds.minimum_m[axis] for item in members) for axis in range(3))
    high = tuple(max(item.geometry.bounds.maximum_m[axis] for item in members) for axis in range(3))
    return ResolvedGeometry(
        geometry_refs=ordered,
        map_frame=str(frame),
        bounds=Bounds3D(
            frame_id=frame,
            minimum_m=(low[0], low[1], low[2]),
            maximum_m=(high[0], high[1], high[2]),
        ),
        duplicate_reference_count=sum(1 for count in counts.values() if count > 1),
        input_geometry_digest=geometry_set_digest(ordered),
        summary_rule_id=GEOMETRY_UNION_RULE_ID,
    )


def _semantic_state(members: tuple[Entity, ...]) -> ResolvedSemanticState:
    hypotheses: dict[tuple[str, str], EntityHypothesis] = {}
    attributes: dict[tuple[str, str, str, str], EntityAttribute] = {}
    uncertainty: dict[tuple[str, ...], EntityUncertainty] = {}
    for member in members:
        for hypothesis in member.semantic_state.hypotheses:
            _put(hypotheses, (hypothesis.fused_evidence_id, hypothesis.hypothesis_id), hypothesis)
        for attribute in member.semantic_state.attributes:
            key = (attribute.name, attribute.value, attribute.origin.value, attribute.derivation_id)
            _put(attributes, key, attribute)
        for item in member.semantic_state.uncertainty:
            _put(uncertainty, uncertainty_key(item), item)
    ambiguity = tuple(
        MemberAmbiguity(
            entity_ref=member.reference, ambiguity_state=member.semantic_state.ambiguity_state
        )
        for member in members
    )
    union = tuple(hypotheses[key] for key in sorted(hypotheses))
    return ResolvedSemanticState(
        hypotheses=union,
        attributes=tuple(attributes[key] for key in sorted(attributes)),
        uncertainty=tuple(uncertainty[key] for key in sorted(uncertainty)),
        member_ambiguity=ambiguity,
        ambiguity_state=derive_resolved_ambiguity(ambiguity, union),
    )


def _put(table: dict[Any, _Record], key: Any, value: _Record) -> None:
    """Add a record once; the same identity with different content is an aggregation error."""
    if key in table and table[key] != value:
        raise MaterializationError(f"two members disagree about the record {key!r}")
    table[key] = value


def _evidence(members: tuple[Entity, ...]) -> EntityEvidenceLinks:
    fused: dict[tuple[str, str], FusedEvidenceRef] = {}
    features: dict[tuple[str, str], EntityFeatureRef] = {}
    representations: dict[tuple[str, str], Any] = {}
    for member in members:
        for item in member.evidence.fused_evidence:
            _put(fused, (item.fusion_run_id, item.fused_evidence_id), item)
        for feature in member.evidence.visual_feature_refs:
            _put(features, (feature.perception_result_id, feature.feature_id), feature)
        for representation in member.evidence.point_representation_refs:
            _put(
                representations,
                (representation.run_id, representation.representation_id),
                representation,
            )
    return EntityEvidenceLinks(
        fused_evidence=tuple(fused[key] for key in sorted(fused)),
        spatial_observation_ids=tuple(
            sorted({item for member in members for item in member.evidence.spatial_observation_ids})
        ),
        physical_observation_ids=tuple(
            sorted(
                {item for member in members for item in member.evidence.physical_observation_ids}
            )
        ),
        visual_feature_refs=tuple(features[key] for key in sorted(features)),
        point_representation_refs=tuple(representations[key] for key in sorted(representations)),
    )


def _temporal_state(members: tuple[Entity, ...]) -> EntityTemporalState:
    clocks = {item.temporal_state.first_seen.clock_id for item in members}
    if len(clocks) != 1:
        raise MaterializationError(
            f"members span more than one clock domain and cannot be grouped: {sorted(clocks)!r}"
        )
    observations: dict[str, ObservationRef] = {}
    for member in members:
        for ref in member.temporal_state.observation_refs:
            known = observations.get(ref.physical_observation_id)
            if known is not None and known.acquisition_timestamp != ref.acquisition_timestamp:
                raise MaterializationError(
                    f"members disagree about when physical observation "
                    f"{ref.physical_observation_id!r} was acquired"
                )
            # O mesmo frame pode ter sido interpretado pelas mesmas inferências em ambos os membros:
            # o máximo é um limite inferior que nunca conta a mesma inferência duas vezes.
            count = (
                ref.inference_result_count
                if known is None
                else max(known.inference_result_count, ref.inference_result_count)
            )
            observations[ref.physical_observation_id] = ObservationRef(
                physical_observation_id=ref.physical_observation_id,
                acquisition_timestamp=ref.acquisition_timestamp,
                inference_result_count=count,
            )
    refs = tuple(
        sorted(
            observations.values(),
            key=lambda item: (
                item.acquisition_timestamp.total_nanoseconds(),
                item.physical_observation_id,
            ),
        )
    )
    return EntityTemporalState(
        first_seen=refs[0].acquisition_timestamp,
        last_seen=refs[-1].acquisition_timestamp,
        physical_observation_count=len(refs),
        inference_result_count=sum(item.inference_result_count for item in refs),
        observation_refs=refs,
        provenance=TemporalProvenance(
            rule_id=TEMPORAL_UNION_RULE_ID, input_order_chronological=True
        ),
        lifecycle=None,
    )
