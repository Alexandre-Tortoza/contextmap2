"""Split candidates: diagnostics for over-merged entities, never a split."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from resolution_entity_builders import entity_over, lattice, scene_source

from contextmap.entity_resolution import (
    SPLIT_DETECTION_POLICY_ID,
    EntityResolutionRunId,
    EvidenceStatus,
    SplitCandidate,
    SplitDetectionPolicy,
    SplitPartition,
    SplitStatus,
    detect_split_candidates,
    materialize_resolved_entities,
)
from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.geometric_mapping import GeometryReference
from contextmap.semantic_mapping import Entity, GeometryResolutionError

POLICY = SplitDetectionPolicy(
    connectivity_radius_m=0.6,
    min_partition_points=5,
    min_partition_fraction=0.1,
    min_gap_m=1.0,
    max_dominant_fraction=0.9,
)


def clumps(
    *centers: tuple[float, float, float], sizes: tuple[int, ...] | None = None
) -> tuple[Entity, Any]:
    """One entity whose support is a lattice clump at each center."""
    points: list[tuple[float, float, float]] = []
    spans: list[range] = []
    for center in centers:
        start = len(points)
        points += lattice(center, 0.6)
        spans.append(range(start, len(points)))
    source = scene_source(dict(enumerate(points)))
    indexes = [index for span in spans for index in span]
    return entity_over("a", source, indexes), source


def by_rule(candidate: SplitCandidate) -> dict[str, EvidenceStatus]:
    return {item.rule_id: item.status for item in candidate.signals}


def test_two_far_pieces_are_suggested_with_their_exact_geometry() -> None:
    entity, source = clumps((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))

    (candidate,) = detect_split_candidates([entity], source=source, policy=POLICY)

    assert candidate.status is SplitStatus.SUGGESTED
    assert candidate.entity_ref == entity.reference
    assert [len(item.geometry_refs) for item in candidate.partitions] == [27, 27]
    rebuilt = sorted(
        (ref for item in candidate.partitions for ref in item.geometry_refs),
        key=lambda ref: ref.geometry_id,
    )
    assert rebuilt == list(entity.geometry.geometry_refs)  # a partição é o suporte exato
    assert by_rule(candidate) == {
        "disconnected-components": EvidenceStatus.SUPPORTING,
        "partition-gap": EvidenceStatus.SUPPORTING,
        "dominant-partition": EvidenceStatus.NEUTRAL,
    }
    assert candidate.policy.policy_id == SPLIT_DETECTION_POLICY_ID
    assert candidate.policy.configuration_fingerprint == POLICY.fingerprint()


def test_a_single_connected_piece_is_not_a_candidate() -> None:
    entity, source = clumps((0.0, 0.0, 0.0))

    assert detect_split_candidates([entity], source=source, policy=POLICY) == ()


def test_a_sparse_scattered_entity_is_rejected_not_split() -> None:
    scattered = [(float(index) * 2.0, 0.0, 0.0) for index in range(4)]
    source = scene_source(dict(enumerate(scattered)))
    entity = entity_over("a", source, range(4))

    (candidate,) = detect_split_candidates([entity], source=source, policy=POLICY)

    assert candidate.status is SplitStatus.REJECTED
    assert by_rule(candidate) == {"disconnected-components": EvidenceStatus.NEUTRAL}


def test_pieces_that_are_close_may_be_one_sparse_object_and_stay_unresolved() -> None:
    entity, source = clumps((0.0, 0.0, 0.0), (1.4, 0.0, 0.0))

    (candidate,) = detect_split_candidates([entity], source=source, policy=POLICY)

    assert candidate.status is SplitStatus.UNRESOLVED
    assert by_rule(candidate)["partition-gap"] is EvidenceStatus.CONFLICTING


def test_a_dominant_piece_with_a_small_far_one_stays_unresolved() -> None:
    points = lattice((0.0, 0.0, 0.0), 0.6, steps=8) + lattice((8.0, 0.0, 0.0), 0.2, steps=2)
    source = scene_source(dict(enumerate(points)))
    entity = entity_over("a", source, range(len(points)))
    policy = SplitDetectionPolicy(
        connectivity_radius_m=0.6,
        min_partition_points=5,
        min_partition_fraction=0.01,
        min_gap_m=1.0,
        max_dominant_fraction=0.9,
    )

    (candidate,) = detect_split_candidates([entity], source=source, policy=policy)

    assert candidate.status is SplitStatus.UNRESOLVED
    assert by_rule(candidate)["dominant-partition"] is EvidenceStatus.CONFLICTING


def test_detection_is_deterministic_and_never_modifies_the_entities() -> None:
    entity, source = clumps((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    before = entity

    first = detect_split_candidates([entity], source=source, policy=POLICY)
    second = detect_split_candidates([entity], source=source, policy=POLICY)

    assert first == second and len(first[0].partitions) == 3
    assert entity == before
    keys = [item.geometry_refs[0].geometry_id for item in first[0].partitions]
    assert keys == sorted(keys)


def test_the_baseline_merge_is_unchanged_whether_or_not_detection_runs() -> None:
    entity, source = clumps((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    run = EntityResolutionRunId("resolution-run-0001")

    without = materialize_resolved_entities([entity], [], resolution_run_id=run)
    detect_split_candidates([entity], source=source, policy=POLICY)
    with_detection = materialize_resolved_entities([entity], [], resolution_run_id=run)

    assert without == with_detection
    assert len(with_detection.resolved.entities) == 1  # nenhuma divisão foi materializada


def test_an_unresolvable_support_fails_loudly() -> None:
    entity, _ = clumps((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    other = scene_source({0: (0.0, 0.0, 0.0)})

    with pytest.raises(GeometryResolutionError):
        detect_split_candidates([entity], source=other, policy=POLICY)


def test_the_candidate_contract_refuses_incoherent_records() -> None:
    entity, source = clumps((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    (candidate,) = detect_split_candidates([entity], source=source, policy=POLICY)
    first, second = candidate.partitions

    with pytest.raises(ValueError, match="at least two"):
        SplitCandidate(
            entity_ref=candidate.entity_ref,
            partitions=(first,),
            signals=candidate.signals,
            status=candidate.status,
            policy=candidate.policy,
        )
    with pytest.raises(ValueError, match="sorted"):
        SplitCandidate(
            entity_ref=candidate.entity_ref,
            partitions=(second, first),
            signals=candidate.signals,
            status=candidate.status,
            policy=candidate.policy,
        )
    overlapping = SplitPartition(
        geometry_refs=tuple(
            sorted({first.geometry_refs[0], *second.geometry_refs}, key=lambda r: r.geometry_id)
        ),
        bounds=second.bounds,
    )
    with pytest.raises(ValueError, match="share a geometry reference"):
        SplitCandidate(
            entity_ref=candidate.entity_ref,
            partitions=(overlapping, second),
            signals=candidate.signals,
            status=candidate.status,
            policy=candidate.policy,
        )
    with pytest.raises(ValueError, match="does not follow"):
        SplitCandidate(
            entity_ref=candidate.entity_ref,
            partitions=candidate.partitions,
            signals=candidate.signals,
            status=SplitStatus.REJECTED,
            policy=candidate.policy,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        SplitPartition(geometry_refs=(), bounds=first.bounds)
    with pytest.raises(ValueError, match="sorted"):
        SplitPartition(geometry_refs=tuple(reversed(first.geometry_refs)), bounds=first.bounds)
    assert isinstance(first.geometry_refs[0], GeometryReference)


@pytest.mark.parametrize(
    "overrides",
    [
        {"connectivity_radius_m": 0.0},
        {"min_gap_m": math.inf},
        {"min_partition_points": 0},
        {"min_partition_fraction": 0.0},
        {"min_partition_fraction": 1.0},
        {"max_dominant_fraction": 0.5},
        {"max_dominant_fraction": 1.1},
    ],
)
def test_the_policy_refuses_impossible_thresholds(overrides: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "connectivity_radius_m": 0.6,
        "min_partition_points": 5,
        "min_partition_fraction": 0.1,
        "min_gap_m": 1.0,
        "max_dominant_fraction": 0.9,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        SplitDetectionPolicy(**values)


def test_the_fingerprint_changes_with_every_threshold_and_the_record_round_trips() -> None:
    def make(**changes: Any) -> SplitDetectionPolicy:
        values: dict[str, Any] = {
            "connectivity_radius_m": 0.6,
            "min_partition_points": 5,
            "min_partition_fraction": 0.1,
            "min_gap_m": 1.0,
            "max_dominant_fraction": 0.9,
        }
        values.update(changes)
        return SplitDetectionPolicy(**values)

    variants = [
        make(connectivity_radius_m=0.7),
        make(min_partition_points=6),
        make(min_partition_fraction=0.2),
        make(min_gap_m=2.0),
        make(max_dominant_fraction=0.8),
    ]
    assert len({POLICY.fingerprint(), *(item.fingerprint() for item in variants)}) == 6

    entity, source = clumps((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    (candidate,) = detect_split_candidates([entity], source=source, policy=POLICY)

    record = json.loads(json.dumps(to_record(candidate)))

    assert from_record(SplitCandidate, record) == candidate
