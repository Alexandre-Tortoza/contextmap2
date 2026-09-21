"""Candidate retrieval: a permissive, deterministic, indexed first step that never decides."""

from __future__ import annotations

import json
import math
import random
from typing import Any

import pytest
from mapping_builders import SEMANTIC_MAP_ID, make_hypothesis
from resolution_entity_builders import entity_at

import contextmap.entity_resolution.retrieval as retrieval_module
from contextmap.entity_resolution import (
    CANDIDATE_RETRIEVAL_POLICY_ID,
    CandidateRetrievalPolicy,
    EntityCandidate,
    EntityCandidateSet,
    EntitySpatialIndex,
    ExclusionReason,
    RetrievalDiagnostics,
    RetrievalReason,
    candidate_pairs,
    decode_candidate_set,
    encode_candidate_set,
    explain_candidacy,
    reference_order,
    retrieve_candidate_sets,
)
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import Entity, EntityId, SemanticMapId, UnknownEntityError

POLICY = CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.2)


def candidates_of(
    entities: list[Entity], source: str, policy: CandidateRetrievalPolicy = POLICY
) -> EntityCandidateSet:
    sets = {
        item.source_entity_ref.entity_id: item for item in retrieve_candidate_sets(entities, policy)
    }
    return sets[EntityId(source)]


def ids(candidate_set: EntityCandidateSet) -> list[str]:
    return [str(ref.entity_id) for ref in candidate_set.candidate_entity_refs]


# --- what is retrieved --------------------------------------------------------------------------


def test_a_nearby_object_is_a_candidate_and_the_reason_is_explicit() -> None:
    entities = [entity_at("a", (0, 0, 0)), entity_at("b", (0.8, 0, 0))]

    found = candidates_of(entities, "a")

    assert ids(found) == ["b"]
    (candidate,) = found.candidates
    assert candidate.reasons == (RetrievalReason.CENTROID_WITHIN_RADIUS,)
    assert candidate.centroid_distance_m == pytest.approx(0.8)
    assert candidate.bounds_gap_m == pytest.approx(0.3)


def test_overlapping_bounds_are_retrieved_even_when_the_centroids_are_far() -> None:
    # Uma parede grande e um objeto pequeno encostado nela: os centroides estão longe.
    entities = [
        entity_at("wall", (0, 0, 0), size=(8.0, 0.2, 3.0)),
        entity_at("box", (3.5, 0.0, 0.0), size=0.4),
    ]

    found = candidates_of(entities, "box")

    assert ids(found) == ["wall"]
    assert found.candidates[0].reasons == (RetrievalReason.BOUNDS_OVERLAP,)
    assert found.candidates[0].centroid_distance_m > POLICY.centroid_radius_m


def test_bounds_within_the_margin_are_retrieved_with_their_own_reason() -> None:
    entities = [entity_at("a", (0, 0, 0)), entity_at("b", (0.65, 0, 0))]
    policy = CandidateRetrievalPolicy(centroid_radius_m=0.5, bounds_margin_m=0.2)

    found = candidates_of(entities, "a", policy)

    assert ids(found) == ["b"]
    assert found.candidates[0].reasons == (RetrievalReason.BOUNDS_WITHIN_MARGIN,)


def test_far_objects_with_the_same_labels_are_excluded_with_a_diagnosable_reason() -> None:
    near, far = entity_at("a", (0, 0, 0)), entity_at("b", (25, 0, 0))

    assert ids(candidates_of([near, far], "a")) == []
    assessment = explain_candidacy(near, far, POLICY)
    assert assessment.exclusion is ExclusionReason.OUTSIDE_SPATIAL_RANGE
    assert assessment.reasons == ()
    assert assessment.centroid_distance_m == pytest.approx(25.0)
    assert "25" in assessment.detail


def test_entities_without_semantic_hypotheses_are_still_candidates() -> None:
    silent = entity_at("a", (0, 0, 0), hypotheses=())
    labeled = entity_at("b", (0.5, 0, 0), hypotheses=(make_hypothesis(label="crate"),))

    assert ids(candidates_of([silent, labeled], "a")) == ["b"]
    assert ids(candidates_of([silent, labeled], "b")) == ["a"]


def test_same_labels_or_different_labels_do_not_change_retrieval() -> None:
    first = entity_at("a", (0, 0, 0), hypotheses=(make_hypothesis(label="pallet"),))
    second = entity_at("b", (0.5, 0, 0), hypotheses=(make_hypothesis(label="person"),))

    assert ids(candidates_of([first, second], "a")) == ["b"]


def test_an_entity_is_never_its_own_candidate() -> None:
    found = candidates_of([entity_at("a", (0, 0, 0)), entity_at("b", (0.5, 0, 0))], "a")

    assert "a" not in ids(found)
    assert (
        explain_candidacy(entity_at("a"), entity_at("a"), POLICY).exclusion
        is ExclusionReason.SAME_ENTITY
    )


# --- map and frame compatibility ----------------------------------------------------------------


def test_entities_over_another_geometric_map_are_never_candidates() -> None:
    here = entity_at("a", (0, 0, 0))
    there = entity_at(
        "b",
        (0, 0, 0),
        map_id=MapId("map-0002"),
        semantic_map_id=SemanticMapId("semantic-map-0002"),
    )

    assert ids(candidates_of([here, there], "a")) == []
    assessment = explain_candidacy(here, there, POLICY)
    assert assessment.exclusion is ExclusionReason.INCOMPATIBLE_MAP
    assert assessment.centroid_distance_m is None
    assert "map-0002" in assessment.detail


def test_entities_in_another_frame_of_the_same_map_id_are_never_candidates() -> None:
    here = entity_at("a", (0, 0, 0))
    there = entity_at("b", (0, 0, 0), frame="odom")

    assert ids(candidates_of([here, there], "a")) == []
    assert explain_candidacy(here, there, POLICY).exclusion is ExclusionReason.INCOMPATIBLE_MAP


def test_entities_of_two_semantic_maps_over_one_geometric_map_can_be_candidates() -> None:
    first = entity_at("a", (0, 0, 0))
    second = entity_at(
        "a", (0.4, 0, 0), semantic_map_id=SemanticMapId("semantic-map-0002"), first_index=5000
    )

    found = retrieve_candidate_sets([first, second], POLICY)

    assert [len(item.candidates) for item in found] == [1, 1]
    assert found[0].candidates[0].entity_ref.semantic_map_id == "semantic-map-0002"


# --- time ---------------------------------------------------------------------------------------


def test_time_never_excludes_when_the_policy_does_not_ask_for_it() -> None:
    early, late = (
        entity_at("a", (0, 0, 0), seconds=(1, 2)),
        entity_at("b", (0.4, 0, 0), seconds=(900, 901)),
    )

    found = candidates_of([early, late], "a")

    assert ids(found) == ["b"]
    assert found.candidates[0].time_gap_ns == 898 * 1_000_000_000


def test_a_declared_time_gap_excludes_entities_seen_far_apart_in_time() -> None:
    policy = CandidateRetrievalPolicy(
        centroid_radius_m=1.0, bounds_margin_m=0.2, max_time_gap_ns=60 * 1_000_000_000
    )
    early, late = (
        entity_at("a", (0, 0, 0), seconds=(1, 2)),
        entity_at("b", (0.4, 0, 0), seconds=(900, 901)),
    )
    close = entity_at("c", (0.4, 0.3, 0), seconds=(30, 31))

    assert ids(candidates_of([early, late, close], "a", policy)) == ["c"]
    assessment = explain_candidacy(early, late, policy)
    assert assessment.exclusion is ExclusionReason.TEMPORAL_GAP_EXCEEDED
    assert RetrievalReason.CENTROID_WITHIN_RADIUS in assessment.reasons


def test_overlapping_observation_intervals_have_no_time_gap() -> None:
    found = candidates_of(
        [entity_at("a", (0, 0, 0), seconds=(1, 10)), entity_at("b", (0.4, 0, 0), seconds=(5, 20))],
        "a",
    )

    assert found.candidates[0].time_gap_ns == 0


def test_timestamps_of_different_clocks_are_not_compared_and_never_drop_a_candidate() -> None:
    policy = CandidateRetrievalPolicy(
        centroid_radius_m=1.0, bounds_margin_m=0.2, max_time_gap_ns=1_000_000_000
    )
    first = entity_at("a", (0, 0, 0), seconds=(1, 2))
    second = entity_at("b", (0.4, 0, 0), seconds=(900, 901), clock_id="another:clock")

    found = candidates_of([first, second], "a", policy)

    assert ids(found) == ["b"]
    assert found.candidates[0].time_gap_ns is None


# --- determinism and structure ------------------------------------------------------------------


def test_retrieval_is_deterministic_whatever_the_input_order() -> None:
    entities = [entity_at(f"e{index:02d}", (index * 0.6, 0, 0)) for index in range(8)]
    shuffled = entities[:]
    random.Random(3).shuffle(shuffled)

    assert retrieve_candidate_sets(entities, POLICY) == retrieve_candidate_sets(shuffled, POLICY)


def test_candidates_are_sorted_unique_and_reasoned() -> None:
    entities = [entity_at(f"e{index:02d}", (index * 0.4, 0, 0)) for index in range(6)]

    for candidate_set in retrieve_candidate_sets(entities, POLICY):
        keys = [reference_order(item.entity_ref) for item in candidate_set.candidates]
        assert keys == sorted(set(keys))
        assert all(item.reasons for item in candidate_set.candidates)


def test_one_candidate_set_per_entity_in_reference_order() -> None:
    entities = [entity_at("b", (0, 0, 0)), entity_at("a", (5, 0, 0))]

    sets = retrieve_candidate_sets(entities, POLICY)

    assert [str(item.source_entity_ref.entity_id) for item in sets] == ["a", "b"]
    assert all(item.candidates == () for item in sets)


def test_a_repeated_entity_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="repeated"):
        retrieve_candidate_sets([entity_at("a", (0, 0, 0)), entity_at("a", (5, 0, 0))], POLICY)


def test_the_candidate_set_is_retrieval_not_a_decision() -> None:
    found = candidates_of([entity_at("a", (0, 0, 0)), entity_at("b", (0.5, 0, 0))], "a")

    assert not any(hasattr(found, name) for name in ("decision", "match", "score", "similarity"))
    assert found.policy.policy_id == CANDIDATE_RETRIEVAL_POLICY_ID
    assert found.policy.configuration_fingerprint == POLICY.fingerprint()


def test_candidate_pairs_are_deduplicated_and_in_canonical_order() -> None:
    entities = [entity_at(name, (index * 0.5, 0, 0)) for index, name in enumerate("abc")]

    pairs = candidate_pairs(retrieve_candidate_sets(entities, POLICY))

    assert [(str(a.entity_id), str(b.entity_id)) for a, b in pairs] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


# --- policy -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"centroid_radius_m": 0.0, "bounds_margin_m": 0.1},
        {"centroid_radius_m": -1.0, "bounds_margin_m": 0.1},
        {"centroid_radius_m": math.inf, "bounds_margin_m": 0.1},
        {"centroid_radius_m": 1.0, "bounds_margin_m": -0.1},
        {"centroid_radius_m": 1.0, "bounds_margin_m": math.nan},
        {"centroid_radius_m": 1.0, "bounds_margin_m": 0.1, "max_time_gap_ns": -1},
    ],
)
def test_the_policy_refuses_impossible_thresholds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CandidateRetrievalPolicy(**kwargs)


def test_the_configuration_fingerprint_changes_with_every_threshold() -> None:
    base = CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.2)
    variants = [
        CandidateRetrievalPolicy(centroid_radius_m=1.5, bounds_margin_m=0.2),
        CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.3),
        CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.2, max_time_gap_ns=5),
    ]

    assert base.fingerprint() == base.fingerprint()
    assert len({base.fingerprint(), *(item.fingerprint() for item in variants)}) == 4


# --- the index agrees with the definition and does not compare everything -----------------------


def _random_scene(count: int, seed: int) -> list[Entity]:
    rng = random.Random(seed)
    entities = []
    for index in range(count):
        center = (rng.uniform(0, 24), rng.uniform(0, 24), rng.uniform(0, 3))
        if index % 40 == 0:
            # Alguns suportes grandes exercitam o caminho das entidades que cobrem muitas células.
            size: Any = (rng.uniform(10, 20), rng.uniform(10, 20), 0.2)
        else:
            size = rng.uniform(0.2, 1.4)
        entities.append(entity_at(f"e{index:04d}", center, size=size))
    return entities


def _brute_force(entities: list[Entity], policy: CandidateRetrievalPolicy) -> dict[Any, set[Any]]:
    return {
        source.reference: {
            other.reference
            for other in entities
            if explain_candidacy(source, other, policy).exclusion is None
        }
        for source in entities
    }


@pytest.mark.parametrize("max_cells_per_entity", [512, 8])
def test_the_index_returns_exactly_what_the_definition_says(
    monkeypatch: pytest.MonkeyPatch, max_cells_per_entity: int
) -> None:
    # Com o teto baixo quase toda entidade cai na lista de "grandes": o resultado não pode mudar.
    monkeypatch.setattr(retrieval_module, "_MAX_CELLS_PER_ENTITY", max_cells_per_entity)
    entities = _random_scene(160, seed=20260921)

    by_source = {
        item.source_entity_ref: set(item.candidate_entity_refs)
        for item in retrieve_candidate_sets(entities, POLICY)
    }

    assert by_source == _brute_force(entities, POLICY)


def test_a_wall_sized_entity_is_retrieved_from_and_for_every_object_on_it() -> None:
    wall = entity_at("wall", (0, 0, 0), size=(30.0, 30.0, 0.2))
    on_wall = [
        entity_at(f"on-{index}", (index * 5.0 - 10, index * 3.0 - 6, 0.0), size=0.4)
        for index in range(5)
    ]
    off_wall = [
        entity_at(f"off-{index}", (index * 5.0 - 10, 0.0, 6.0), size=0.4) for index in range(3)
    ]
    entities = [wall, *on_wall, *off_wall]

    by_source = {item.source_entity_ref: item for item in retrieve_candidate_sets(entities, POLICY)}

    assert set(by_source[wall.reference].candidate_entity_refs) == {
        item.reference for item in on_wall
    }
    assert all(
        wall.reference in by_source[item.reference].candidate_entity_refs for item in on_wall
    )
    assert all(by_source[item.reference].candidate_entity_refs == () for item in off_wall)
    assert {
        ref: set(item.candidate_entity_refs) for ref, item in by_source.items()
    } == _brute_force(entities, POLICY)


def test_retrieval_is_symmetric() -> None:
    entities = _random_scene(120, seed=7)
    by_source = {
        item.source_entity_ref: set(item.candidate_entity_refs)
        for item in retrieve_candidate_sets(entities, POLICY)
    }

    for source, found in by_source.items():
        assert all(source in by_source[other] for other in found)


def test_the_index_examines_a_small_neighborhood_instead_of_every_entity() -> None:
    columns, rows = 40, 30
    entities = [
        entity_at(f"e{row:02d}-{column:02d}", (column * 2.0, row * 2.0, 0.0))
        for row in range(rows)
        for column in range(columns)
    ]
    policy = CandidateRetrievalPolicy(centroid_radius_m=1.2, bounds_margin_m=0.1)

    sets = retrieve_candidate_sets(entities, policy)

    count = len(entities)
    examined = [item.diagnostics.examined_entity_count for item in sets]
    assert all(item.diagnostics.compatible_entity_count == count - 1 for item in sets)
    assert max(examined) <= 30
    assert sum(examined) < 0.02 * count * count
    # Na malha de 2 m com raio 1,2 m ninguém tem vizinho: nada foi perdido nem inventado.
    assert all(item.candidates == () for item in sets)


def test_the_index_can_be_queried_for_one_entity_without_retrieving_the_others() -> None:
    entities = [entity_at(f"e{index:02d}", (index * 0.6, 0, 0)) for index in range(6)]
    index = EntitySpatialIndex(entities, POLICY)

    only = index.retrieve(entities[2].reference)

    assert only == next(
        item
        for item in retrieve_candidate_sets(entities, POLICY)
        if item.source_entity_ref == entities[2].reference
    )
    with pytest.raises(UnknownEntityError):
        index.retrieve(entity_at("missing").reference)


# --- contract and serialization -----------------------------------------------------------------


def test_a_candidate_set_refuses_its_own_source_and_unsorted_or_unreasoned_candidates() -> None:
    found = candidates_of(
        [entity_at("a", (0, 0, 0)), entity_at("b", (0.5, 0, 0)), entity_at("c", (0.7, 0, 0))], "a"
    )
    first, second = found.candidates

    with pytest.raises(ValueError, match="candidates"):
        EntityCandidateSet(
            source_entity_ref=found.source_entity_ref,
            candidates=(second, first),
            policy=found.policy,
            diagnostics=found.diagnostics,
        )
    with pytest.raises(ValueError, match="source"):
        EntityCandidateSet(
            source_entity_ref=first.entity_ref,
            candidates=(first,),
            policy=found.policy,
            diagnostics=found.diagnostics,
        )
    with pytest.raises(ValueError, match="reasons"):
        EntityCandidate(
            entity_ref=first.entity_ref,
            reasons=(),
            centroid_distance_m=1.0,
            bounds_gap_m=0.0,
            time_gap_ns=None,
        )


def test_a_candidate_refuses_unordered_reasons_and_a_negative_time_gap() -> None:
    reference = entity_at("b").reference

    with pytest.raises(ValueError, match="reasons"):
        EntityCandidate(
            entity_ref=reference,
            reasons=(RetrievalReason.BOUNDS_OVERLAP, RetrievalReason.CENTROID_WITHIN_RADIUS),
            centroid_distance_m=0.1,
            bounds_gap_m=0.0,
            time_gap_ns=None,
        )
    with pytest.raises(ValueError, match="time_gap_ns"):
        EntityCandidate(
            entity_ref=reference,
            reasons=(RetrievalReason.BOUNDS_OVERLAP,),
            centroid_distance_m=0.1,
            bounds_gap_m=0.0,
            time_gap_ns=-1,
        )


def test_a_candidate_set_cannot_hold_more_candidates_than_examined_entities() -> None:
    found = candidates_of([entity_at("a", (0, 0, 0)), entity_at("b", (0.5, 0, 0))], "a")

    with pytest.raises(ValueError, match="examined"):
        EntityCandidateSet(
            source_entity_ref=found.source_entity_ref,
            candidates=found.candidates,
            policy=found.policy,
            diagnostics=RetrievalDiagnostics(compatible_entity_count=0, examined_entity_count=0),
        )


def test_diagnostics_are_coherent() -> None:
    with pytest.raises(ValueError, match="examined"):
        RetrievalDiagnostics(compatible_entity_count=3, examined_entity_count=4)
    with pytest.raises(ValueError, match="negative"):
        RetrievalDiagnostics(compatible_entity_count=-1, examined_entity_count=0)


def test_a_candidate_set_survives_a_json_round_trip_and_revalidates() -> None:
    found = candidates_of([entity_at("a", (0, 0, 0)), entity_at("b", (0.5, 0, 0))], "a")

    record = json.loads(json.dumps(encode_candidate_set(found)))

    assert decode_candidate_set(record) == found
    record["candidates"][0]["reasons"] = []
    with pytest.raises(ValueError, match="reasons"):
        decode_candidate_set(record)


def test_the_semantic_map_of_the_fixture_is_the_default() -> None:
    assert entity_at("a").semantic_map_id == SEMANTIC_MAP_ID
