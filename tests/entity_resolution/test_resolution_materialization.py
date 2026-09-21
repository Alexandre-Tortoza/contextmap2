"""Resolved entities: exact aggregation, merge lineage, contradictions, deterministic ids."""

from __future__ import annotations

import dataclasses
import json
import random
from itertools import pairwise
from typing import Any

import pytest
from mapping_builders import make_hypothesis, make_semantic_state, timestamp
from resolution_builders import decision_between
from resolution_entity_builders import (
    attribute,
    entity_at,
    entity_over,
    fused_id,
    lattice,
    scene_source,
)
from resolution_feature_fakes import feature_ref

from contextmap.entity_resolution import (
    MATERIALIZATION_POLICY_ID,
    EntityResolutionRunId,
    ForeignResolvedEntityReferenceError,
    MaterializationError,
    MemberAmbiguity,
    ResolutionDecision,
    ResolutionOutcome,
    ResolvedEntity,
    ResolvedEntityMaterialization,
    ResolvedEntityProvenance,
    ResolvedEntityReference,
    ResolvedEntitySet,
    ResolvedMember,
    TransitivityContradiction,
    UnknownResolvedEntityError,
    contradiction_id_for,
    derive_resolved_ambiguity,
    materialize_resolved_entities,
    resolved_entity_id_for,
)
from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import PointRepresentationId, PointRepresentationRunId
from contextmap.semantic_fusion import PointRepresentationRef, UncertaintyKind, UncertaintyRecord
from contextmap.semantic_mapping import (
    AmbiguityState,
    AttributeOrigin,
    Entity,
    EntityReference,
    EntityTemporalState,
    EntityUncertainty,
    SemanticMapId,
)

RUN = EntityResolutionRunId("resolution-run-0001")
MATCH, DISTINCT, UNRESOLVED = (
    ResolutionOutcome.MATCH,
    ResolutionOutcome.DISTINCT,
    ResolutionOutcome.UNRESOLVED,
)


def boxes(*names: str, seconds: dict[str, Any] | None = None) -> dict[str, Entity]:
    """Entities on a line, each with its own fused evidence, so nothing is shared by accident."""
    return {
        name: entity_at(
            name,
            (index * 5.0, 0.0, 0.0),
            support_number=index + 1,
            seconds=(seconds or {}).get(name, (10 + index, 12 + index)),
        )
        for index, name in enumerate(names)
    }


def decide(a: Entity, b: Entity, outcome: ResolutionOutcome) -> ResolutionDecision:
    return decision_between(a.reference, b.reference, outcome)


def materialize(
    entities: dict[str, Entity], *decisions: ResolutionDecision, run: EntityResolutionRunId = RUN
) -> ResolvedEntityMaterialization:
    return materialize_resolved_entities(entities.values(), decisions, resolution_run_id=run)


def members_of(result: ResolvedEntityMaterialization) -> list[list[str]]:
    return sorted(
        [str(ref.entity_id) for ref in item.member_entity_refs] for item in result.resolved.entities
    )


# --- grouping and lineage -----------------------------------------------------------------------


def test_a_match_materializes_one_resolved_entity_and_every_source_stays_addressable() -> None:
    entities = boxes("a", "b", "c")
    match = decide(entities["a"], entities["b"], MATCH)

    result = materialize(entities, match)

    assert members_of(result) == [["a", "b"], ["c"]]
    merged = result.resolved.resolved_of(entities["a"].reference)
    assert merged.resolution_decision_refs == (match.decision_id,)
    assert [item.matched_by for item in merged.members] == [(match.decision_id,)] * 2
    assert result.resolved.resolved_of(entities["c"].reference).resolution_decision_refs == ()
    # Cada entidade de origem pertence a exatamente uma entidade resolvida.
    assert sorted(
        ref.entity_id for item in result.resolved.entities for ref in item.member_entity_refs
    ) == [
        "a",
        "b",
        "c",
    ]


def test_a_chain_of_matches_is_one_entity_with_the_lineage_of_each_member() -> None:
    entities = boxes("a", "b", "c")
    ab, bc = (
        decide(entities["a"], entities["b"], MATCH),
        decide(entities["b"], entities["c"], MATCH),
    )

    result = materialize(entities, ab, bc)

    (merged,) = result.resolved.entities
    lineage = {str(item.entity_ref.entity_id): item.matched_by for item in merged.members}
    assert lineage == {
        "a": (ab.decision_id,),
        "b": tuple(sorted((ab.decision_id, bc.decision_id))),
        "c": (bc.decision_id,),
    }
    assert merged.resolution_decision_refs == tuple(sorted((ab.decision_id, bc.decision_id)))
    assert result.contradictions == ()


def test_only_match_decisions_group_entities() -> None:
    entities = boxes("a", "b", "c", "d")
    decisions = [
        decide(entities["a"], entities["b"], DISTINCT),
        decide(entities["a"], entities["c"], UNRESOLVED),
    ]

    result = materialize(entities, *decisions)

    assert members_of(result) == [["a"], ["b"], ["c"], ["d"]]


def test_unresolved_neighbors_are_recorded_on_both_sides_and_never_merged() -> None:
    entities = boxes("a", "b", "c")
    result = materialize(
        entities,
        decide(entities["a"], entities["b"], MATCH),
        decide(entities["b"], entities["c"], UNRESOLVED),
    )

    merged = result.resolved.resolved_of(entities["a"].reference)
    other = result.resolved.resolved_of(entities["c"].reference)
    assert merged.unresolved_neighbor_refs == (entities["c"].reference,)
    assert other.unresolved_neighbor_refs == (entities["b"].reference,)
    assert len(result.resolved.entities) == 2


def test_the_source_entities_are_never_modified() -> None:
    entities = boxes("a", "b")
    before = {name: entity for name, entity in entities.items()}

    materialize(entities, decide(entities["a"], entities["b"], MATCH))

    assert entities == before


# --- transitivity contradictions ----------------------------------------------------------------


def test_a_contradiction_is_surfaced_and_the_component_is_not_merged() -> None:
    entities = boxes("a", "b", "c")
    ab, bc = (
        decide(entities["a"], entities["b"], MATCH),
        decide(entities["b"], entities["c"], MATCH),
    )
    ac = decide(entities["a"], entities["c"], DISTINCT)

    result = materialize(entities, ab, bc, ac)

    assert members_of(result) == [["a"], ["b"], ["c"]]
    (contradiction,) = result.contradictions
    assert contradiction.distinct_decision_id == ac.decision_id
    assert contradiction.distinct_pair == (entities["a"].reference, entities["c"].reference)
    assert contradiction.match_path == (ab.decision_id, bc.decision_id)
    assert contradiction.component == tuple(entities[name].reference for name in "abc")
    assert contradiction.contradiction_id == contradiction_id_for(ac.decision_id)
    for entity in result.resolved.entities:
        assert entity.contradiction_ids == (contradiction.contradiction_id,)
        assert entity.resolution_decision_refs == ()


def test_the_contradiction_path_is_the_shortest_chain_of_matches() -> None:
    entities = boxes("a", "b", "c", "d")
    chain = [decide(entities[x], entities[y], MATCH) for x, y in ("ab", "bc", "cd")]
    shortcut = decide(entities["a"], entities["d"], MATCH)  # a~d ligado direto
    distinct = decide(entities["b"], entities["d"], DISTINCT)

    result = materialize(entities, *chain, shortcut, distinct)

    (contradiction,) = result.contradictions
    assert len(contradiction.match_path) == 2  # b~a, a~d, mais curto que b~c, c~d? empate: 2
    assert set(contradiction.match_path) <= {item.decision_id for item in (*chain, shortcut)}


def test_a_distinct_pair_in_different_components_is_no_contradiction() -> None:
    entities = boxes("a", "b", "c", "d")
    result = materialize(
        entities,
        decide(entities["a"], entities["b"], MATCH),
        decide(entities["c"], entities["d"], MATCH),
        decide(entities["a"], entities["c"], DISTINCT),
    )

    assert members_of(result) == [["a", "b"], ["c", "d"]]
    assert result.contradictions == ()


def test_only_the_contradictory_component_is_withheld() -> None:
    entities = boxes("a", "b", "c", "d", "e")
    result = materialize(
        entities,
        decide(entities["a"], entities["b"], MATCH),
        decide(entities["b"], entities["c"], MATCH),
        decide(entities["a"], entities["c"], DISTINCT),
        decide(entities["d"], entities["e"], MATCH),
    )

    assert members_of(result) == [["a"], ["b"], ["c"], ["d", "e"]]
    assert result.resolved.resolved_of(entities["d"].reference).contradiction_ids == ()


# --- several contradictions in one component ----------------------------------------------------
# Regressão de uma execução real: um componente de MATCH com DOIS pares DISTINCT levantava
# ValueError, porque cada entidade guardava um único id de contradição.


def chain_with_distinct(
    names: str, distinct: tuple[str, ...]
) -> tuple[dict[str, Entity], list[ResolutionDecision], list[ResolutionDecision]]:
    """A chain of MATCH decisions over ``names`` and the DISTINCT decisions named by pairs."""
    entities = boxes(*names)
    matches = [
        decide(entities[first], entities[second], MATCH) for first, second in pairwise(names)
    ]
    distincts = [decide(entities[pair[0]], entities[pair[1]], DISTINCT) for pair in distinct]
    return entities, matches, distincts


def test_two_contradictions_in_one_component_are_both_recorded_and_nothing_is_merged() -> None:
    entities, matches, distincts = chain_with_distinct("abcd", ("ac", "bd"))
    ab, bc, cd = matches
    ac, bd = distincts

    result = materialize(entities, *matches, *distincts)

    assert members_of(result) == [["a"], ["b"], ["c"], ["d"]]
    by_distinct = {item.distinct_decision_id: item for item in result.contradictions}
    assert set(by_distinct) == {ac.decision_id, bd.decision_id}
    assert by_distinct[ac.decision_id].match_path == (ab.decision_id, bc.decision_id)
    assert by_distinct[bd.decision_id].match_path == (bc.decision_id, cd.decision_id)
    everyone = tuple(entities[name].reference for name in "abcd")
    assert all(item.component == everyone for item in result.contradictions)
    expected = tuple(sorted(item.contradiction_id for item in result.contradictions))
    assert len(expected) == 2
    for entity in result.resolved.entities:
        assert entity.contradiction_ids == expected
        assert entity.resolution_decision_refs == ()


def test_three_contradictions_in_one_component_are_all_recorded() -> None:
    entities, matches, distincts = chain_with_distinct("abcde", ("ac", "bd", "ce"))

    result = materialize(entities, *matches, *distincts)

    assert members_of(result) == [["a"], ["b"], ["c"], ["d"], ["e"]]
    expected = tuple(sorted(contradiction_id_for(item.decision_id) for item in distincts))
    assert tuple(item.contradiction_id for item in result.contradictions) == expected
    for entity in result.resolved.entities:
        assert entity.contradiction_ids == expected


def test_two_contradicted_pairs_that_share_an_entity_are_both_recorded() -> None:
    entities, matches, distincts = chain_with_distinct("abcd", ("ac", "ad"))

    result = materialize(entities, *matches, *distincts)

    assert members_of(result) == [["a"], ["b"], ["c"], ["d"]]
    assert {item.distinct_pair for item in result.contradictions} == {
        (entities["a"].reference, entities["c"].reference),
        (entities["a"].reference, entities["d"].reference),
    }
    expected = tuple(sorted(contradiction_id_for(item.decision_id) for item in distincts))
    for entity in result.resolved.entities:
        assert entity.contradiction_ids == expected


def test_the_contradictions_of_one_component_never_reach_another() -> None:
    entities = boxes("a", "b", "c", "d", "e", "f", "g", "h")
    decisions = [
        decide(entities[x], entities[y], outcome)
        for x, y, outcome in (
            ("a", "b", MATCH),
            ("b", "c", MATCH),
            ("c", "d", MATCH),
            ("a", "c", DISTINCT),
            ("b", "d", DISTINCT),
            ("e", "f", MATCH),
            ("f", "g", MATCH),
            ("e", "g", DISTINCT),
            ("g", "h", MATCH),
        )
    ]

    result = materialize(entities, *decisions)

    counts = {
        name: len(result.resolved.resolved_of(entities[name].reference).contradiction_ids)
        for name in "abcdefgh"
    }
    assert counts == {"a": 2, "b": 2, "c": 2, "d": 2, "e": 1, "f": 1, "g": 1, "h": 1}
    assert len(result.contradictions) == 3


def test_several_contradictions_do_not_depend_on_the_order_of_the_decisions() -> None:
    entities, matches, distincts = chain_with_distinct("abcde", ("ac", "bd", "ce"))
    decisions = [*matches, *distincts]
    shuffled = list(decisions)
    random.Random(7).shuffle(shuffled)

    assert materialize(entities, *shuffled) == materialize(entities, *decisions)


# --- geometry -----------------------------------------------------------------------------------


def test_the_geometry_is_the_union_of_the_member_references_without_duplicates() -> None:
    points = lattice((0, 0, 0), (2.0, 1.0, 1.0), steps=6)
    source = scene_source(dict(enumerate(points)))
    first = entity_over("a", source, range(0, 120), support_number=1)
    second = entity_over("b", source, range(80, 216), support_number=2)

    result = materialize({"a": first, "b": second}, decide(first, second, MATCH))

    (merged,) = result.resolved.entities
    geometry = merged.geometry
    assert len(geometry.geometry_refs) == 216
    assert geometry.duplicate_reference_count == 40
    assert [ref.geometry_id for ref in geometry.geometry_refs] == sorted(
        ref.geometry_id for ref in geometry.geometry_refs
    )
    low_a, high_a = first.geometry.bounds.minimum_m, first.geometry.bounds.maximum_m
    low_b, high_b = second.geometry.bounds.minimum_m, second.geometry.bounds.maximum_m
    low, high = geometry.bounds.minimum_m, geometry.bounds.maximum_m
    assert low == tuple(min(a, b) for a, b in zip(low_a, low_b, strict=True))
    assert high == tuple(max(a, b) for a, b in zip(high_a, high_b, strict=True))
    assert geometry.bounds_center_m == pytest.approx(
        tuple((lo + hi) / 2 for lo, hi in zip(low, high, strict=True))
    )
    assert geometry.extent_m == pytest.approx(
        tuple(hi - lo for lo, hi in zip(low, high, strict=True))
    )
    assert geometry.geometric_map_id == first.geometry.geometric_map_id


def test_members_over_different_geometric_maps_cannot_be_grouped() -> None:
    first = entity_at("a", (0, 0, 0), support_number=1)
    second = entity_at(
        "b",
        (0, 0, 0),
        support_number=2,
        map_id=MapId("map-0002"),
        semantic_map_id=SemanticMapId("semantic-map-0002"),
    )

    with pytest.raises(MaterializationError, match="different geometric maps"):
        materialize({"a": first, "b": second}, decide(first, second, MATCH))


# --- semantics: nothing is invented -------------------------------------------------------------


def with_labels(name: str, number: int, *labels: str, **kwargs: Any) -> Entity:
    hypotheses = tuple(
        make_hypothesis(f"hypothesis-{index:04d}", label, fused_evidence_id=fused_id(number))
        for index, label in enumerate(labels)
    )
    return entity_at(
        name, (number * 5.0, 0.0, 0.0), support_number=number, hypotheses=hypotheses, **kwargs
    )


def test_semantic_alternatives_and_disagreements_are_preserved_with_no_primary() -> None:
    first = with_labels("a", 1, "pallet")
    second = with_labels("b", 2, "pallet", "crate")

    (merged,) = materialize(
        {"a": first, "b": second}, decide(first, second, MATCH)
    ).resolved.entities

    state = merged.semantic_state
    assert sorted(item.label for item in state.hypotheses) == ["crate", "pallet", "pallet"]
    assert state.ambiguity_state is AmbiguityState.AMBIGUOUS
    assert [item.ambiguity_state for item in state.member_ambiguity] == [
        AmbiguityState.UNAMBIGUOUS,
        AmbiguityState.AMBIGUOUS,
    ]
    assert not hasattr(state, "primary_hypothesis")


def test_members_that_propose_different_labels_are_ambiguous_never_more_certain() -> None:
    first, second = with_labels("a", 1, "pallet"), with_labels("b", 2, "person")

    (merged,) = materialize(
        {"a": first, "b": second}, decide(first, second, MATCH)
    ).resolved.entities

    assert merged.semantic_state.ambiguity_state is AmbiguityState.AMBIGUOUS
    assert {item.label for item in merged.semantic_state.hypotheses} == {"pallet", "person"}


def test_members_that_agree_stay_unambiguous_and_a_silent_member_adds_no_doubt() -> None:
    agreeing_a, agreeing_b = with_labels("a", 1, "pallet"), with_labels("b", 2, "pallet")
    silent = with_labels("c", 3)

    (both,) = materialize(
        {"a": agreeing_a, "b": agreeing_b}, decide(agreeing_a, agreeing_b, MATCH)
    ).resolved.entities
    (mixed,) = materialize(
        {"a": agreeing_a, "c": silent}, decide(agreeing_a, silent, MATCH)
    ).resolved.entities

    assert both.semantic_state.ambiguity_state is AmbiguityState.UNAMBIGUOUS
    assert mixed.semantic_state.ambiguity_state is AmbiguityState.UNAMBIGUOUS
    assert [item.ambiguity_state for item in mixed.semantic_state.member_ambiguity] == [
        AmbiguityState.UNAMBIGUOUS,
        AmbiguityState.INSUFFICIENT_EVIDENCE,
    ]


def test_members_without_any_hypothesis_stay_insufficient_evidence() -> None:
    first, second = with_labels("a", 1), with_labels("b", 2)

    (merged,) = materialize(
        {"a": first, "b": second}, decide(first, second, MATCH)
    ).resolved.entities

    assert merged.semantic_state.hypotheses == ()
    assert merged.semantic_state.ambiguity_state is AmbiguityState.INSUFFICIENT_EVIDENCE


def test_attributes_keep_their_origin_and_are_deduplicated() -> None:
    wood = attribute("material", "wood")
    first = with_labels("a", 1, "pallet", attributes=(wood,))
    second = with_labels(
        "b",
        2,
        "pallet",
        attributes=(
            wood,
            attribute("weight", "heavy", origin=AttributeOrigin.EXTERNAL_KNOWLEDGE),
        ),
    )

    (merged,) = materialize(
        {"a": first, "b": second}, decide(first, second, MATCH)
    ).resolved.entities

    assert [(item.name, item.value, item.origin) for item in merged.semantic_state.attributes] == [
        ("material", "wood", AttributeOrigin.OBSERVED),
        ("weight", "heavy", AttributeOrigin.EXTERNAL_KNOWLEDGE),
    ]


def test_two_members_that_disagree_about_one_record_are_refused() -> None:
    first = with_labels("a", 1, "pallet")
    # Mesma evidência fundida e mesmo id de hipótese, conteúdo diferente: um registro contraditório.
    clash = entity_at(
        "b",
        (5.0, 0.0, 0.0),
        support_number=1,
        hypotheses=(make_hypothesis("hypothesis-0000", "crate", fused_evidence_id=fused_id(1)),),
    )

    with pytest.raises(MaterializationError, match="disagree"):
        materialize({"a": first, "b": clash}, decide(first, clash, MATCH))


# --- time and evidence --------------------------------------------------------------------------


def test_the_temporal_state_is_recomputed_from_the_union_of_observations() -> None:
    entities = boxes("a", "b", seconds={"a": (10, 12), "b": (12, 20)})

    (merged,) = materialize(entities, decide(entities["a"], entities["b"], MATCH)).resolved.entities

    state = merged.temporal_state
    assert (state.first_seen.seconds, state.last_seen.seconds) == (10, 20)
    assert state.physical_observation_count == 3  # o frame 12 é compartilhado e conta uma vez
    assert [item.physical_observation_id for item in state.observation_refs] == [
        "frame-0010",
        "frame-0012",
        "frame-0020",
    ]
    assert merged.evidence.physical_observation_ids == tuple(
        item.physical_observation_id for item in state.observation_refs
    )


def test_a_shared_observation_keeps_the_largest_inference_count() -> None:
    first = entity_at("a", (0, 0, 0), support_number=1, seconds=(10, 12), inference_results=2)
    second = entity_at("b", (5, 0, 0), support_number=2, seconds=(12, 20), inference_results=3)

    (merged,) = materialize(
        {"a": first, "b": second}, decide(first, second, MATCH)
    ).resolved.entities

    counts = {
        item.physical_observation_id: item.inference_result_count
        for item in merged.temporal_state.observation_refs
    }
    assert counts == {"frame-0010": 2, "frame-0012": 3, "frame-0020": 3}
    assert merged.temporal_state.inference_result_count == 8


def test_members_of_different_clock_domains_cannot_be_grouped() -> None:
    first = entity_at("a", (0, 0, 0), support_number=1)
    second = entity_at("b", (5, 0, 0), support_number=2, clock_id="another:clock")

    with pytest.raises(MaterializationError, match="clock domain"):
        materialize({"a": first, "b": second}, decide(first, second, MATCH))


def test_the_evidence_links_are_the_union_of_the_members() -> None:
    entities = boxes("a", "b")

    (merged,) = materialize(entities, decide(entities["a"], entities["b"], MATCH)).resolved.entities

    assert [ref.fused_evidence_id for ref in merged.evidence.fused_evidence] == [
        fused_id(1),
        fused_id(2),
    ]
    assert len(merged.evidence.spatial_observation_ids) == 1  # o mesmo id espacial não se duplica


# --- identity and determinism -------------------------------------------------------------------


def test_ids_derive_from_the_members_and_are_scoped_to_the_artifact() -> None:
    entities = boxes("a", "b")
    match = decide(entities["a"], entities["b"], MATCH)

    first = materialize(entities, match, run=EntityResolutionRunId("resolution-run-0001"))
    second = materialize(entities, match, run=EntityResolutionRunId("resolution-run-0002"))

    (one,), (two,) = first.resolved.entities, second.resolved.entities
    assert one.resolved_entity_id == resolved_entity_id_for(one.member_entity_refs)
    assert one.resolved_entity_id == two.resolved_entity_id  # conteúdo idêntico, mesmo id local
    assert one.reference != two.reference  # mas a identidade completa é do artifact
    assert one.reference == ResolvedEntityReference(
        resolution_run_id=EntityResolutionRunId("resolution-run-0001"),
        resolved_entity_id=one.resolved_entity_id,
    )
    source_ids = {ref.entity_id for ref in one.member_entity_refs}
    assert one.resolved_entity_id not in source_ids


def test_the_result_does_not_depend_on_the_order_of_entities_or_decisions() -> None:
    entities = boxes("a", "b", "c", "d", "e")
    decisions = [
        decide(entities["a"], entities["b"], MATCH),
        decide(entities["b"], entities["c"], MATCH),
        decide(entities["a"], entities["c"], DISTINCT),
        decide(entities["d"], entities["e"], MATCH),
        decide(entities["a"], entities["d"], UNRESOLVED),
    ]
    baseline = materialize(entities, *decisions)

    for seed in range(5):
        rng = random.Random(seed)
        shuffled_entities = list(entities.values())
        shuffled_decisions = decisions[:]
        rng.shuffle(shuffled_entities)
        rng.shuffle(shuffled_decisions)

        again = materialize_resolved_entities(
            shuffled_entities, shuffled_decisions, resolution_run_id=RUN
        )

        assert again == baseline


def test_the_materialization_names_its_policy() -> None:
    result = materialize(boxes("a"))

    assert result.policy.policy_id == MATERIALIZATION_POLICY_ID
    assert result.resolved.entities[0].provenance.policy == result.policy


# --- refusals -----------------------------------------------------------------------------------


def test_repeated_unknown_and_twice_decided_inputs_are_refused() -> None:
    entities = boxes("a", "b", "c")
    stranger = entity_at("z", (99, 0, 0), support_number=9)

    with pytest.raises(MaterializationError, match="repeated"):
        materialize_resolved_entities([entities["a"], entities["a"]], [], resolution_run_id=RUN)
    with pytest.raises(MaterializationError, match="not given"):
        materialize(entities, decide(entities["a"], stranger, MATCH))
    match = decide(entities["a"], entities["b"], MATCH)
    other_policy = decide(entities["a"], entities["b"], DISTINCT)
    with pytest.raises(MaterializationError, match="more than once"):
        materialize(entities, match, other_policy)


# --- contracts ----------------------------------------------------------------------------------


def one_merged() -> tuple[dict[str, Entity], ResolvedEntity]:
    entities = boxes("a", "b")
    (merged,) = materialize(entities, decide(entities["a"], entities["b"], MATCH)).resolved.entities
    return entities, merged


def replaced(entity: ResolvedEntity, **changes: Any) -> ResolvedEntity:
    return dataclasses.replace(entity, **changes)


def test_a_resolved_entity_refuses_a_tampered_identity_or_lineage() -> None:
    _, merged = one_merged()

    with pytest.raises(ValueError, match="derived from the members"):
        replaced(merged, resolved_entity_id="resolved--tampered")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decisions of the members"):
        replaced(merged, resolution_decision_refs=())
    with pytest.raises(ValueError, match="neighbor cannot be a member"):
        replaced(merged, unresolved_neighbor_refs=(merged.member_entity_refs[0],))
    with pytest.raises(ValueError, match="at least one member"):
        replaced(merged, members=())
    with pytest.raises(ValueError, match="sorted"):
        replaced(merged, members=tuple(reversed(merged.members)))
    with pytest.raises(ValueError, match="exactly the members"):
        replaced(
            merged,
            members=(
                ResolvedMember(
                    entity_ref=merged.member_entity_refs[0], matched_by=merged.members[0].matched_by
                ),
            ),
            resolved_entity_id=resolved_entity_id_for((merged.member_entity_refs[0],)),
            resolution_decision_refs=merged.members[0].matched_by,
        )


def test_the_semantic_ambiguity_must_be_the_one_the_members_imply() -> None:
    _, merged = one_merged()

    with pytest.raises(ValueError, match="ambiguity_state"):
        replaced(
            merged,
            semantic_state=type(merged.semantic_state)(
                hypotheses=merged.semantic_state.hypotheses,
                attributes=(),
                uncertainty=(),
                member_ambiguity=merged.semantic_state.member_ambiguity,
                ambiguity_state=AmbiguityState.CONFLICTING,
            ),
        )
    assert (
        derive_resolved_ambiguity(
            (
                MemberAmbiguity(
                    entity_ref=merged.member_entity_refs[0],
                    ambiguity_state=AmbiguityState.CONFLICTING,
                ),
            ),
            (),
        )
        is AmbiguityState.CONFLICTING
    )


def test_the_set_resolves_references_and_refuses_foreign_and_unknown_ones() -> None:
    _, merged = one_merged()
    resolved = ResolvedEntitySet(resolution_run_id=RUN, entities=(merged,))

    assert resolved.resolve(merged.reference) is merged
    with pytest.raises(ForeignResolvedEntityReferenceError):
        resolved.resolve(
            ResolvedEntityReference(
                resolution_run_id=EntityResolutionRunId("other-run"),
                resolved_entity_id=merged.resolved_entity_id,
            )
        )
    with pytest.raises(UnknownResolvedEntityError):
        resolved.resolve(
            ResolvedEntityReference(
                resolution_run_id=RUN, resolved_entity_id=resolved_entity_id_for(())
            )
        )
    with pytest.raises(KeyError):
        resolved.resolved_of(EntityReference(semantic_map_id=SemanticMapId("x"), entity_id="y"))  # type: ignore[arg-type]


def test_a_set_refuses_entities_of_another_artifact_and_a_source_entity_in_two() -> None:
    entities, merged = one_merged()
    solo = materialize(entities).resolved.entities[0]

    with pytest.raises(ValueError, match="belongs to resolution run"):
        ResolvedEntitySet(resolution_run_id=EntityResolutionRunId("other-run"), entities=(merged,))
    with pytest.raises(ValueError, match="more than one resolved entity"):
        ResolvedEntitySet(
            resolution_run_id=RUN,
            entities=tuple(sorted((merged, solo), key=lambda item: item.resolved_entity_id)),
        )


def test_a_contradiction_must_hold_a_chain_inside_its_component() -> None:
    entities = boxes("a", "b", "c")
    refs = tuple(entities[name].reference for name in "abc")
    distinct = decide(entities["a"], entities["c"], DISTINCT)

    with pytest.raises(ValueError, match="derived from the decision"):
        TransitivityContradiction(
            contradiction_id="contradiction--x",  # type: ignore[arg-type]
            distinct_decision_id=distinct.decision_id,
            distinct_pair=(refs[0], refs[2]),
            match_path=("d1", "d2"),  # type: ignore[arg-type]
            component=refs,
        )
    with pytest.raises(ValueError, match="at least two matches"):
        TransitivityContradiction(
            contradiction_id=contradiction_id_for(distinct.decision_id),
            distinct_decision_id=distinct.decision_id,
            distinct_pair=(refs[0], refs[2]),
            match_path=("d1",),  # type: ignore[arg-type]
            component=refs,
        )
    with pytest.raises(ValueError, match="inside the withheld component"):
        TransitivityContradiction(
            contradiction_id=contradiction_id_for(distinct.decision_id),
            distinct_decision_id=distinct.decision_id,
            distinct_pair=(refs[0], refs[2]),
            match_path=("d1", "d2"),  # type: ignore[arg-type]
            component=refs[:2],
        )


def test_the_materialization_survives_a_json_round_trip() -> None:
    entities = boxes("a", "b", "c")
    result = materialize(
        entities,
        decide(entities["a"], entities["b"], MATCH),
        decide(entities["b"], entities["c"], MATCH),
        decide(entities["a"], entities["c"], DISTINCT),
    )

    record = json.loads(json.dumps(to_record(result)))

    assert from_record(ResolvedEntityMaterialization, record) == result


# --- the rest of the union, and the record validators -------------------------------------------


def test_evidence_features_representations_and_uncertainty_are_unioned() -> None:
    source = scene_source(dict(enumerate(lattice((0, 0, 0), 1.0))))
    space = "sha256:representation-space"

    def representation(number: int, index: int) -> PointRepresentationRef:
        return PointRepresentationRef(
            representation_id=PointRepresentationId(f"representation-{number:04d}"),
            run_id=PointRepresentationRunId("representation-run-0001"),
            representation_space_id=space,
            geometry_reference=GeometryReference(
                map_id=MapId("map-0001"),
                geometry_id=geometry_id_for(map_id=MapId("map-0001"), index=index),
            ),
        )

    first = entity_over(
        "a", source, range(0, 13), support_number=1, representations=(representation(1, 0),)
    )
    second = entity_over(
        "b", source, range(13, 27), support_number=2, representations=(representation(2, 20),)
    )
    with_features = entity_at(
        "c", (9.0, 0.0, 0.0), support_number=3, features=(feature_ref(10, 0),)
    )
    other_features = entity_at(
        "d", (12.0, 0.0, 0.0), support_number=4, features=(feature_ref(10, 0), feature_ref(20, 1))
    )
    note = EntityUncertainty(
        fused_evidence_id=fused_id(3),
        record=UncertaintyRecord(
            kind=UncertaintyKind.INSUFFICIENT_EVIDENCE,
            hypothesis_ids=(),
            evidence=(),
            rule_id="rule-insufficient",
        ),
    )
    with_note = dataclasses.replace(
        with_features,
        semantic_state=make_semantic_state(
            with_features.semantic_state.hypotheses, uncertainty=(note,)
        ),
    )

    (rep_merged,) = materialize(
        {"a": first, "b": second}, decide(first, second, MATCH)
    ).resolved.entities
    (feature_merged,) = materialize(
        {"c": with_note, "d": other_features}, decide(with_note, other_features, MATCH)
    ).resolved.entities

    assert [
        str(ref.representation_id) for ref in rep_merged.evidence.point_representation_refs
    ] == [
        "representation-0001",
        "representation-0002",
    ]
    assert [str(ref.feature_id) for ref in feature_merged.evidence.visual_feature_refs] == [
        "feature-0010-00",
        "feature-0020-01",
    ]
    assert feature_merged.semantic_state.uncertainty == (note,)


def test_members_that_disagree_about_when_a_frame_was_acquired_are_refused() -> None:
    first = entity_at("a", (0.0, 0.0, 0.0), support_number=1, seconds=(10, 12))
    second = entity_at("b", (5.0, 0.0, 0.0), support_number=2, seconds=(10, 20))
    shifted = tuple(
        dataclasses.replace(ref, acquisition_timestamp=timestamp(11))
        if ref.physical_observation_id == "frame-0010"
        else ref
        for ref in second.temporal_state.observation_refs
    )
    disagreeing = dataclasses.replace(
        second,
        temporal_state=EntityTemporalState(
            first_seen=shifted[0].acquisition_timestamp,
            last_seen=shifted[-1].acquisition_timestamp,
            physical_observation_count=2,
            inference_result_count=2,
            observation_refs=shifted,
            provenance=second.temporal_state.provenance,
        ),
    )

    with pytest.raises(MaterializationError, match="disagree about when"):
        materialize({"a": first, "b": disagreeing}, decide(first, disagreeing, MATCH))


def test_the_geometry_record_refuses_an_inconsistent_support_or_summary() -> None:
    _, merged = one_merged()
    geometry = merged.geometry
    other_map = dataclasses.replace(
        geometry.geometry_refs[0],
        map_id=MapId("map-0002"),
        geometry_id=GeometryId("map-0002--geom-0"),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        dataclasses.replace(geometry, geometry_refs=())
    with pytest.raises(ValueError, match="one map"):
        dataclasses.replace(
            geometry,
            geometry_refs=(*geometry.geometry_refs, other_map),
        )
    with pytest.raises(ValueError, match="map frame"):
        dataclasses.replace(geometry, map_frame="odom")
    with pytest.raises(ValueError, match="negative"):
        dataclasses.replace(geometry, duplicate_reference_count=-1)
    with pytest.raises(ValueError, match="exactly these"):
        dataclasses.replace(geometry, input_geometry_digest="sha256:other")


def test_the_remaining_record_validators_refuse_incoherent_values() -> None:
    _, merged = one_merged()

    with pytest.raises(ValueError, match="member_ambiguity"):
        dataclasses.replace(merged.semantic_state, member_ambiguity=())
    with pytest.raises(ValueError, match="code_version"):
        ResolvedEntityProvenance(policy=merged.provenance.policy, code_version=" ")
    with pytest.raises(ValueError, match="not recorded on"):
        ResolvedEntityMaterialization(
            resolved=ResolvedEntitySet(resolution_run_id=RUN, entities=(merged,)),
            contradictions=(
                TransitivityContradiction(
                    contradiction_id=contradiction_id_for("decision--x"),  # type: ignore[arg-type]
                    distinct_decision_id="decision--x",  # type: ignore[arg-type]
                    distinct_pair=merged.member_entity_refs,  # type: ignore[arg-type]
                    match_path=("d1", "d2"),  # type: ignore[arg-type]
                    component=merged.member_entity_refs,
                ),
            ),
            policy=merged.provenance.policy,
        )
