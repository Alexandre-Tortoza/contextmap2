"""Candidate generation: deterministic, geometry-only narrowing that never decides a relation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import weakref
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import ClassVar

import pytest
from relation_builders import entity_ref
from relation_run_fixture import CANDIDATES as STOREROOM
from relation_scene import Scene
from relation_storeroom_fixture import exclusion_ceiling, multiset_digest, storeroom_scene

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.semantic_mapping import EntityGeometry
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    CANDIDATE_POLICY_ID,
    TAXONOMY_VERSION,
    AxisDirection,
    CandidateExclusion,
    CandidateExclusionReason,
    CandidatePolicy,
    CandidateReason,
    FrameConventions,
    FrameRequirement,
    IncompatibleFrameError,
    RelationCandidateSet,
    RelationPredicate,
    candidates,
    encode_candidate_set,
    generate_relation_candidates,
)

Box = tuple[Vector3, Vector3]

CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
NO_AXES = FrameConventions(map_frame="map")


def _policy(
    *predicates: RelationPredicate, proximity: float = 0.5, directional: float = 2.0
) -> CandidatePolicy:
    return CandidatePolicy(
        predicates=predicates or (RelationPredicate.NEXT_TO,),
        proximity_radius_m=proximity,
        directional_radius_m=directional,
    )


def _entities(*boxes: Box) -> dict[ResolvedEntityReference, EntityGeometry]:
    """Entities ``resolved-0001``, ``resolved-0002``... filling the given boxes."""
    scene = Scene()
    for number, (low, high) in enumerate(boxes, start=1):
        scene.add_box(f"e{number}", low, high)
    return {entity_ref(number): scene.geometry(f"e{number}") for number in range(1, len(boxes) + 1)}


def _keys(result: object) -> list[tuple[str, str, str]]:
    return [
        (
            str(item.subject_entity_ref.resolved_entity_id),  # type: ignore[attr-defined]
            item.predicate.value,  # type: ignore[attr-defined]
            str(item.object_entity_ref.resolved_entity_id),  # type: ignore[attr-defined]
        )
        for item in result  # type: ignore[attr-defined]
    ]


NEARBY: tuple[Box, Box] = (
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ((1.3, 0.0, 0.0), (2.3, 1.0, 1.0)),
)
FLOOR: Box = ((0.0, 0.0, 0.0), (4.0, 4.0, 0.1))
CRATE_ON_FLOOR: Box = ((1.0, 1.0, 0.1), (2.0, 2.0, 1.1))


def test_nearby_entities_become_one_proximity_candidate_with_the_measured_gap() -> None:
    result = generate_relation_candidates(
        _entities(*NEARBY), policy=_policy(), conventions=CONVENTIONS
    )
    assert _keys(result.candidates) == [("resolved-0001", "next_to", "resolved-0002")]
    candidate = result.candidates[0]
    assert candidate.reasons == (CandidateReason.WITHIN_PROXIMITY_RADIUS,)
    assert candidate.bounds_gap_m == pytest.approx(0.3)
    assert result.exclusions == ()


def test_a_symmetric_predicate_is_a_candidate_once_per_unordered_pair() -> None:
    symmetric = (
        RelationPredicate.NEXT_TO,
        RelationPredicate.INTERSECTS,
        RelationPredicate.TOUCHING,
    )
    result = generate_relation_candidates(
        _entities(*NEARBY), policy=_policy(*symmetric), conventions=CONVENTIONS
    )
    assert _keys(result.candidates) == [
        ("resolved-0001", "intersects", "resolved-0002"),
        ("resolved-0001", "next_to", "resolved-0002"),
        ("resolved-0001", "touching", "resolved-0002"),
    ]


def test_a_directed_predicate_is_evaluated_in_each_direction_that_can_hold() -> None:
    both_sides = (
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.5), (1.0, 1.0, 1.5)),
    )
    result = generate_relation_candidates(
        _entities(*both_sides),
        policy=_policy(RelationPredicate.LEANING_AGAINST),
        conventions=CONVENTIONS,
    )
    assert _keys(result.candidates) == [
        ("resolved-0001", "leaning_against", "resolved-0002"),
        ("resolved-0002", "leaning_against", "resolved-0001"),
    ]


def test_candidates_are_unique_and_never_include_the_derived_direction() -> None:
    result = generate_relation_candidates(
        _entities(FLOOR, CRATE_ON_FLOOR),
        policy=_policy(RelationPredicate.ABOVE, RelationPredicate.INSIDE),
        conventions=CONVENTIONS,
    )
    keys = _keys(result.candidates) + _keys(result.exclusions)
    assert len(keys) == len(set(keys))
    assert {predicate for _, predicate, _ in keys} <= {"above", "inside"}


def test_pairs_too_far_apart_are_not_enumerated_and_only_counted() -> None:
    far = (NEARBY[0], ((10.0, 0.0, 0.0), (11.0, 1.0, 1.0)))
    result = generate_relation_candidates(
        _entities(*far), policy=_policy(), conventions=CONVENTIONS
    )
    assert result.candidates == ()
    assert result.exclusions == ()
    assert result.entity_count == 2
    assert result.pairs_not_enumerated == 1


def test_a_neighbor_beyond_the_radius_is_excluded_with_its_gap_recorded() -> None:
    diagonal = (NEARBY[0], ((1.4, 1.4, 0.0), (2.4, 2.4, 1.0)))
    result = generate_relation_candidates(
        _entities(*diagonal), policy=_policy(directional=0.5), conventions=CONVENTIONS
    )
    assert result.candidates == ()
    assert len(result.exclusions) == 1
    exclusion = result.exclusions[0]
    assert exclusion.reason is CandidateExclusionReason.BEYOND_PROXIMITY_RADIUS
    assert exclusion.bounds_gap_m == pytest.approx(0.4 * 2**0.5)
    assert result.pairs_not_enumerated == 0


def test_containment_is_a_candidate_only_for_the_entity_that_can_fit_inside() -> None:
    big: Box = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    small: Box = ((4.0, 4.0, 4.0), (5.0, 5.0, 5.0))
    result = generate_relation_candidates(
        _entities(big, small), policy=_policy(RelationPredicate.INSIDE), conventions=CONVENTIONS
    )
    assert _keys(result.candidates) == [("resolved-0002", "inside", "resolved-0001")]
    assert result.candidates[0].reasons == (
        CandidateReason.CONTAINMENT_POSSIBLE,
        CandidateReason.WITHIN_PROXIMITY_RADIUS,
    )
    assert _keys(result.exclusions) == [("resolved-0001", "inside", "resolved-0002")]
    assert result.exclusions[0].reason is CandidateExclusionReason.CONTAINMENT_IMPOSSIBLE


def test_a_directional_candidate_needs_overlapping_footprints_and_the_right_side() -> None:
    result = generate_relation_candidates(
        _entities(FLOOR, CRATE_ON_FLOOR),
        policy=_policy(RelationPredicate.ABOVE),
        conventions=CONVENTIONS,
    )
    assert _keys(result.candidates) == [("resolved-0002", "above", "resolved-0001")]
    assert result.candidates[0].reasons == (
        CandidateReason.FOOTPRINT_OVERLAP,
        CandidateReason.SUBJECT_ON_DIRECTED_SIDE,
        CandidateReason.WITHIN_DIRECTIONAL_RADIUS,
    )
    assert _keys(result.exclusions) == [("resolved-0001", "above", "resolved-0002")]
    assert result.exclusions[0].reason is CandidateExclusionReason.SUBJECT_NOT_ON_DIRECTED_SIDE


def test_stacked_entities_with_disjoint_footprints_are_not_above_candidates() -> None:
    offset_crate: Box = ((6.0, 1.0, 0.1), (7.0, 2.0, 1.1))
    result = generate_relation_candidates(
        _entities(FLOOR, offset_crate),
        policy=_policy(RelationPredicate.ABOVE, directional=3.0),
        conventions=CONVENTIONS,
    )
    assert result.candidates == ()
    assert {item.reason for item in result.exclusions} == {
        CandidateExclusionReason.NO_FOOTPRINT_OVERLAP
    }


def test_directional_reach_is_configured_apart_from_proximity() -> None:
    lamp_above_table = (
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.8)),
        ((0.0, 0.0, 2.5), (1.0, 1.0, 3.0)),
    )
    near_only = generate_relation_candidates(
        _entities(*lamp_above_table),
        policy=_policy(RelationPredicate.ABOVE, directional=1.0),
        conventions=CONVENTIONS,
    )
    long_reach = generate_relation_candidates(
        _entities(*lamp_above_table),
        policy=_policy(RelationPredicate.ABOVE, directional=2.0),
        conventions=CONVENTIONS,
    )
    assert near_only.candidates == ()
    assert _keys(long_reach.candidates) == [("resolved-0002", "above", "resolved-0001")]


def test_depth_predicates_use_the_declared_forward_axis() -> None:
    behind_and_ahead = (
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((1.5, 0.0, 0.0), (2.5, 1.0, 1.0)),
    )
    result = generate_relation_candidates(
        _entities(*behind_and_ahead),
        policy=_policy(RelationPredicate.IN_FRONT_OF),
        conventions=CONVENTIONS,
    )
    assert _keys(result.candidates) == [("resolved-0002", "in_front_of", "resolved-0001")]
    reversed_forward = dataclasses.replace(CONVENTIONS, forward_axis=AxisDirection.NEGATIVE_X)
    flipped = generate_relation_candidates(
        _entities(*behind_and_ahead),
        policy=_policy(RelationPredicate.IN_FRONT_OF),
        conventions=reversed_forward,
    )
    assert _keys(flipped.candidates) == [("resolved-0001", "in_front_of", "resolved-0002")]


def test_a_predicate_whose_axis_is_not_declared_is_skipped_and_reported() -> None:
    result = generate_relation_candidates(
        _entities(FLOOR, CRATE_ON_FLOOR),
        policy=_policy(
            RelationPredicate.NEXT_TO, RelationPredicate.ABOVE, RelationPredicate.IN_FRONT_OF
        ),
        conventions=NO_AXES,
    )
    assert {item.predicate for item in result.candidates} == {RelationPredicate.NEXT_TO}
    skipped = {item.predicate: item.requirement for item in result.skipped_predicates}
    assert skipped == {
        RelationPredicate.ABOVE: FrameRequirement.UP_AXIS,
        RelationPredicate.IN_FRONT_OF: FrameRequirement.UP_AND_FORWARD_AXES,
    }
    assert all(item.detail for item in result.skipped_predicates)


def test_candidate_sets_are_deterministic_and_independent_of_input_order() -> None:
    scene = Scene()
    boxes = {
        1: FLOOR,
        2: CRATE_ON_FLOOR,
        3: ((2.5, 1.0, 0.1), (3.5, 2.0, 1.1)),
        4: ((0.0, 0.0, 1.1), (1.0, 1.0, 2.1)),
    }
    for number, (low, high) in boxes.items():
        scene.add_box(f"e{number}", low, high)
    geometries = {entity_ref(number): scene.geometry(f"e{number}") for number in boxes}
    policy = _policy(
        RelationPredicate.NEXT_TO,
        RelationPredicate.ABOVE,
        RelationPredicate.INSIDE,
        RelationPredicate.TOUCHING,
        directional=3.0,
    )
    forward = generate_relation_candidates(geometries, policy=policy, conventions=CONVENTIONS)
    shuffled: Mapping[ResolvedEntityReference, EntityGeometry] = dict(
        reversed(list(geometries.items()))
    )
    backward = generate_relation_candidates(shuffled, policy=policy, conventions=CONVENTIONS)
    assert forward == backward
    assert forward == generate_relation_candidates(
        geometries, policy=policy, conventions=CONVENTIONS
    )
    assert forward.candidates == tuple(sorted(forward.candidates, key=_sort_key))


def _sort_key(candidate: object) -> tuple[str, str, str, str, str]:
    subject = candidate.subject_entity_ref  # type: ignore[attr-defined]
    obj = candidate.object_entity_ref  # type: ignore[attr-defined]
    return (
        subject.resolution_run_id,
        subject.resolved_entity_id,
        candidate.predicate.value,  # type: ignore[attr-defined]
        obj.resolution_run_id,
        obj.resolved_entity_id,
    )


def test_labels_and_semantics_cannot_influence_the_candidates() -> None:
    # A geração recebe só referências e geometria: não há por onde um label entrar.
    import inspect

    parameters = inspect.signature(generate_relation_candidates).parameters
    assert set(parameters) == {"entities", "policy", "conventions"}


def test_a_large_map_is_narrowed_without_visiting_every_pair() -> None:
    boxes = [((10.0 * i, 0.0, 0.0), (10.0 * i + 1.0, 1.0, 1.0)) for i in range(300)]
    result = generate_relation_candidates(
        _entities(*boxes), policy=_policy(), conventions=CONVENTIONS
    )
    total_pairs = 300 * 299 // 2
    assert result.candidates == ()
    assert result.pairs_not_enumerated == total_pairs
    assert len(result.exclusions) == 0


def test_provenance_records_policy_frame_map_and_conventions() -> None:
    result = generate_relation_candidates(
        _entities(*NEARBY), policy=_policy(), conventions=CONVENTIONS
    )
    provenance = result.provenance
    assert provenance.policy_id == CANDIDATE_POLICY_ID
    assert provenance.configuration_fingerprint == _policy().fingerprint()
    assert provenance.frame_conventions_fingerprint == CONVENTIONS.fingerprint()
    assert provenance.taxonomy_version == TAXONOMY_VERSION
    assert provenance.map_frame == "map"
    assert str(provenance.geometric_map_id) == "map-0001"


def test_an_empty_or_single_entity_set_has_no_candidates() -> None:
    empty = generate_relation_candidates({}, policy=_policy(), conventions=CONVENTIONS)
    single = generate_relation_candidates(
        _entities(NEARBY[0]), policy=_policy(), conventions=CONVENTIONS
    )
    assert empty.candidates == () and empty.entity_count == 0
    assert single.candidates == () and single.entity_count == 1
    assert empty.provenance.geometric_map_id is None


def test_geometry_in_another_frame_is_refused_loudly() -> None:
    geometries = _entities(*NEARBY)
    with pytest.raises(IncompatibleFrameError):
        generate_relation_candidates(
            geometries,
            policy=_policy(),
            conventions=FrameConventions(map_frame="odom"),
        )


def test_entities_of_different_resolution_artifacts_are_refused() -> None:
    scene = Scene()
    scene.add_box("a", *NEARBY[0])
    scene.add_box("b", *NEARBY[1])
    mixed = {
        entity_ref(1, run="resolution-run-0001"): scene.geometry("a"),
        entity_ref(2, run="resolution-run-0002"): scene.geometry("b"),
    }
    with pytest.raises(ValueError, match="resolution"):
        generate_relation_candidates(mixed, policy=_policy(), conventions=CONVENTIONS)


def test_a_policy_lists_only_predicates_that_are_evaluated_directly() -> None:
    with pytest.raises(ValueError, match="BELOW"):
        _policy(RelationPredicate.BELOW)
    with pytest.raises(ValueError, match="ABOVE"):
        _policy(RelationPredicate.BELOW)


def test_a_policy_is_explicit_and_validated() -> None:
    with pytest.raises(ValueError, match="predicates"):
        CandidatePolicy(predicates=(), proximity_radius_m=0.5, directional_radius_m=2.0)
    with pytest.raises(ValueError, match="unique"):
        _policy(RelationPredicate.NEXT_TO, RelationPredicate.NEXT_TO)
    for radius in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="proximity_radius_m"):
            CandidatePolicy(
                predicates=(RelationPredicate.NEXT_TO,),
                proximity_radius_m=radius,
                directional_radius_m=2.0,
            )
        with pytest.raises(ValueError, match="directional_radius_m"):
            CandidatePolicy(
                predicates=(RelationPredicate.NEXT_TO,),
                proximity_radius_m=0.5,
                directional_radius_m=radius,
            )


def test_the_policy_fingerprint_follows_every_choice_and_ignores_listing_order() -> None:
    base = _policy(RelationPredicate.NEXT_TO, RelationPredicate.ABOVE)
    assert (
        base.fingerprint()
        == _policy(RelationPredicate.ABOVE, RelationPredicate.NEXT_TO).fingerprint()
    )
    assert base.fingerprint() != _policy(RelationPredicate.NEXT_TO).fingerprint()
    assert (
        base.fingerprint()
        != _policy(RelationPredicate.NEXT_TO, RelationPredicate.ABOVE, proximity=0.6).fingerprint()
    )
    assert base.fingerprint().startswith("sha256:")


ALL_EVALUATED = (
    RelationPredicate.NEXT_TO,
    RelationPredicate.INTERSECTS,
    RelationPredicate.TOUCHING,
    RelationPredicate.INSIDE,
    RelationPredicate.ABOVE,
    RelationPredicate.IN_FRONT_OF,
    RelationPredicate.ON_TOP_OF,
    RelationPredicate.LEANING_AGAINST,
)


def random_boxes(seed: int, count: int = 4) -> dict[ResolvedEntityReference, EntityGeometry]:
    """``count`` boxes of random size scattered in a 4 m x 4 m x 2 m room."""
    rng = random.Random(seed)
    boxes: list[Box] = []
    for _ in range(count):
        size = (rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.0))
        low = (rng.uniform(0.0, 4.0), rng.uniform(0.0, 4.0), rng.uniform(0.0, 2.0))
        boxes.append((low, (low[0] + size[0], low[1] + size[1], low[2] + size[2])))
    return _entities(*boxes)


def _largest_exclusion_group(result: RelationCandidateSet) -> int:
    groups = Counter((item.predicate, item.reason) for item in result.exclusions)
    return max(groups.values(), default=0)


# #601 (SR-01): gravados antes do teto de exclusões, em cenas em que nenhum grupo
# (predicado, razão) passa de oito exclusões, o menor teto considerado.
RECORDED_CANDIDATE_SETS = {
    1: "27cb21ee5ba422b3ec8296172ed184793ca7eeed90cdaacb4cc7897656206f31",
    4: "6344a2bca0bb021dd62e9070bff09d05c90671930b288304b6062e8658c54b39",
    5: "b348dc0452e9a6dbfd07435448c16b4c7bbfe9184afb3c4ab51eb712ce440333",
    6: "fb914a114a2691050ae9fb7bb65d5db1831bf9f04091ceb61e69167309e79302",
    7: "bb3033f9875fb1a721e1e79c0280ef0307d980b24ca4185799e707d09e14611b",
}


@pytest.mark.parametrize("seed", sorted(RECORDED_CANDIDATE_SETS))
def test_candidate_sets_below_the_exclusion_ceiling_match_the_recorded_sets(seed: int) -> None:
    result = generate_relation_candidates(
        random_boxes(seed),
        policy=_policy(*ALL_EVALUATED, proximity=0.5, directional=2.0),
        conventions=CONVENTIONS,
    )
    records = json.dumps(encode_candidate_set(result), sort_keys=True).encode()
    assert _largest_exclusion_group(result) <= 8
    assert hashlib.sha256(records).hexdigest() == RECORDED_CANDIDATE_SETS[seed]


# --- the exclusion ceiling (#601) ---


def nearest_first(item: CandidateExclusion) -> tuple[float, str, str, str, str, str]:
    """The documented listing order: gap, then subject, predicate and object."""
    return (item.bounds_gap_m, *_sort_key(item))


def by_group(
    exclusions: Iterable[CandidateExclusion],
) -> dict[tuple[RelationPredicate, CandidateExclusionReason], list[CandidateExclusion]]:
    groups: dict[tuple[RelationPredicate, CandidateExclusionReason], list[CandidateExclusion]] = {}
    for item in exclusions:
        groups.setdefault((item.predicate, item.reason), []).append(item)
    return groups


def test_a_group_past_the_ceiling_lists_its_nearest_and_summarizes_all_of_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SR-01: toda exclusão de todo par enumerado ficava listada, sem teto.
    _, _, entities = storeroom_scene()
    exclusion_ceiling(monkeypatch, 10**9)
    whole = generate_relation_candidates(entities, policy=STOREROOM, conventions=CONVENTIONS)
    exclusion_ceiling(monkeypatch, 40)
    capped = generate_relation_candidates(entities, policy=STOREROOM, conventions=CONVENTIONS)
    groups = by_group(whole.exclusions)
    large = {key: items for key, items in groups.items() if len(items) > 40}

    assert large and len(large) < len(groups)
    assert capped.exclusions == tuple(
        item for item in whole.exclusions if (item.predicate, item.reason) not in large
    )
    assert capped.candidates == whole.candidates
    summaries = {(item.predicate, item.reason): item for item in capped.exclusion_summaries}
    assert list(summaries) == sorted(large, key=lambda key: (key[0].value, key[1].value))
    for key, items in large.items():
        summary = summaries[key]
        gaps = [item.bounds_gap_m for item in items]
        assert summary.count == len(items)
        assert summary.nearest == tuple(sorted(items, key=nearest_first)[:40])
        assert (summary.min_gap_m, summary.max_gap_m) == (min(gaps), max(gaps))
        assert summary.digest == multiset_digest(items)
        assert summary.digest == multiset_digest(reversed(items))


class TrackedExclusion(CandidateExclusion):
    """An exclusion whose lifetime a test can observe; records the peak alive at creation."""

    alive: ClassVar[list[weakref.ref[CandidateExclusion]]] = []
    peak: ClassVar[int] = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        TrackedExclusion.alive.append(weakref.ref(self))
        living = sum(ref() is not None for ref in TrackedExclusion.alive)
        TrackedExclusion.peak = max(TrackedExclusion.peak, living)


def test_the_exclusions_held_while_generating_stay_bounded_by_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SR-01: a geração retinha em memória toda exclusão até o fim.
    _, _, entities = storeroom_scene()
    exclusion_ceiling(monkeypatch, 10**9)
    whole = generate_relation_candidates(entities, policy=STOREROOM, conventions=CONVENTIONS)
    exclusion_ceiling(monkeypatch, 4)
    monkeypatch.setattr(candidates, "CandidateExclusion", TrackedExclusion)
    TrackedExclusion.alive, TrackedExclusion.peak = [], 0

    generate_relation_candidates(entities, policy=STOREROOM, conventions=CONVENTIONS)

    assert len(whole.exclusions) > 1000
    # Cada grupo guarda no máximo o teto, mais a exclusão que acabou de ser criada.
    assert TrackedExclusion.peak <= len(by_group(whole.exclusions)) * (4 + 1)


def test_a_summary_must_describe_one_group_and_list_its_nearest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, entities = storeroom_scene()
    exclusion_ceiling(monkeypatch, 4)
    capped = generate_relation_candidates(entities, policy=STOREROOM, conventions=CONVENTIONS)
    summary = capped.exclusion_summaries[0]
    other = capped.exclusion_summaries[1].nearest[0]
    invalid = {
        "listed": {"count": len(summary.nearest)},
        "some, not all": {"nearest": ()},
        "predicate and reason": {"nearest": (*summary.nearest[:-1], other)},
        "sorted": {"nearest": tuple(reversed(summary.nearest))},
        "min_gap_m": {"min_gap_m": summary.min_gap_m + 1.0},
        "max_gap_m": {"max_gap_m": summary.nearest[-1].bounds_gap_m - 0.001},
        "digest": {"digest": summary.digest.replace("sha256-multiset:", "sha256:")},
    }
    for message, changes in invalid.items():
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(summary, **changes)  # type: ignore[arg-type]
    listed_again = (*capped.exclusions, summary.nearest[0])
    with pytest.raises(ValueError, match="listed whole or summarized"):
        dataclasses.replace(capped, exclusions=tuple(sorted(listed_again, key=_sort_key)))
    twice = (capped.exclusion_summaries[0], capped.exclusion_summaries[0])
    with pytest.raises(ValueError, match="exclusion_summaries"):
        dataclasses.replace(capped, exclusion_summaries=twice)


def test_candidate_presence_is_not_relation_evidence() -> None:
    result = generate_relation_candidates(
        _entities(*NEARBY), policy=_policy(), conventions=CONVENTIONS
    )
    fields = {field.name for field in dataclasses.fields(result.candidates[0])}
    assert not {"status", "state", "score", "confidence", "measurements"} & fields
