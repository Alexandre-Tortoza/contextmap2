"""Temporal compatibility: intervals, shared frames and inference counts, no inferred motion."""

from __future__ import annotations

import json
from typing import Any

import pytest
from resolution_entity_builders import entity_at

from contextmap.entity_resolution import (
    TEMPORAL_COMPATIBILITY_POLICY_ID,
    EvidenceStatus,
    TemporalCompatibilityPolicy,
    TemporalEvidence,
    UnavailableReason,
    compare_temporal,
)
from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.semantic_mapping import Entity

SECOND = 1_000_000_000
STATIC = TemporalCompatibilityPolicy(static_scene=True)
RELAXED = TemporalCompatibilityPolicy(static_scene=False)


def seen(entity_id: str, *seconds: int, **kwargs: Any) -> Entity:
    return entity_at(entity_id, seconds=seconds, **kwargs)


def measure(first: Entity, second: Entity, policy: TemporalCompatibilityPolicy = RELAXED) -> Any:
    return compare_temporal(first, second, policy)


def test_overlapping_observation_intervals_report_their_overlap() -> None:
    evidence = measure(seen("a", 1, 10), seen("b", 5, 20))

    assert evidence.measurement.interval_overlap_ns == 5 * SECOND
    assert evidence.measurement.interval_gap_ns == 0
    assert evidence.measurement.clock_id == "fixture:header"


def test_separate_observation_intervals_report_their_gap() -> None:
    evidence = measure(seen("a", 1, 2), seen("b", 10, 12))

    assert evidence.measurement.interval_overlap_ns == 0
    assert evidence.measurement.interval_gap_ns == 8 * SECOND


def test_touching_intervals_have_neither_overlap_nor_gap() -> None:
    evidence = measure(seen("a", 1, 5), seen("b", 5, 9))

    assert (evidence.measurement.interval_overlap_ns, evidence.measurement.interval_gap_ns) == (
        0,
        0,
    )


def test_repeated_views_from_different_frames_share_no_physical_observation() -> None:
    evidence = measure(seen("a", 10, 12), seen("b", 40, 42))

    measurement = evidence.measurement
    assert measurement.shared_physical_observation_count == 0
    assert measurement.union_physical_observation_count == 4
    assert evidence.status is EvidenceStatus.NEUTRAL


def test_a_shared_frame_is_counted_once_in_the_union() -> None:
    evidence = measure(seen("a", 10, 12), seen("b", 12, 14))

    measurement = evidence.measurement
    assert measurement.shared_physical_observation_count == 1
    assert measurement.union_physical_observation_count == 3
    assert (measurement.physical_observation_count_a, measurement.physical_observation_count_b) == (
        2,
        2,
    )


def test_physical_observations_and_inference_results_are_counted_apart() -> None:
    evidence = measure(seen("a", 10, 12, inference_results=3), seen("b", 40))

    measurement = evidence.measurement
    assert measurement.physical_observation_count_a == 2
    assert measurement.inference_result_count_a == 6
    assert measurement.physical_observation_count_b == 1
    assert measurement.inference_result_count_b == 1
    # Repetir a inferência sobre o mesmo frame não cria observações independentes.
    assert measurement.union_physical_observation_count == 3


def test_a_different_clock_makes_the_channel_unavailable_not_negative() -> None:
    evidence = measure(seen("a", 10), seen("b", 10, clock_id="another:clock"))

    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.measurement is None
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.INCOMPATIBLE_DOMAIN
    assert "another:clock" in evidence.unavailable.detail


def test_two_entities_seen_in_one_frame_conflict_only_under_a_declared_static_scene() -> None:
    first, second = seen("a", 10, 12), seen("b", 12, 14)

    relaxed = measure(first, second, RELAXED)
    static = measure(first, second, STATIC)

    finding = next(
        item for item in static.findings if item.rule_id == "static-scene-co-observation"
    )
    assert finding.status is EvidenceStatus.CONFLICTING
    assert finding.metric == "shared_physical_observation_count"
    assert finding.observed == 1
    assert static.status is EvidenceStatus.CONFLICTING
    assert relaxed.status is EvidenceStatus.NEUTRAL
    assert any(item.rule_id == "static-scene-co-observation" for item in relaxed.findings)


def test_the_static_scene_rule_stays_neutral_without_a_shared_frame() -> None:
    assert measure(seen("a", 10), seen("b", 20), STATIC).status is EvidenceStatus.NEUTRAL


def test_time_never_infers_motion_or_reidentification() -> None:
    evidence = measure(seen("a", 1), seen("b", 5000))

    assert evidence.status is EvidenceStatus.NEUTRAL
    assert evidence.measurement.interval_gap_ns == 4999 * SECOND


def test_the_comparison_is_symmetric_and_names_its_policy() -> None:
    first, second = seen("a", 1, 10), seen("b", 5, 20)

    assert measure(first, second) == measure(second, first)
    evidence = measure(first, second, STATIC)
    assert evidence.policy.policy_id == TEMPORAL_COMPATIBILITY_POLICY_ID
    assert evidence.policy.configuration_fingerprint == STATIC.fingerprint()


def test_the_fingerprint_depends_on_the_static_scene_declaration() -> None:
    assert STATIC.fingerprint() != RELAXED.fingerprint()
    assert STATIC.fingerprint() == TemporalCompatibilityPolicy(static_scene=True).fingerprint()


def test_the_policy_requires_an_explicit_declaration() -> None:
    with pytest.raises(TypeError):
        TemporalCompatibilityPolicy()  # type: ignore[call-arg]


def test_the_evidence_survives_a_json_round_trip() -> None:
    evidence = measure(seen("a", 10, 12, inference_results=2), seen("b", 12, 14), STATIC)

    record = json.loads(json.dumps(to_record(evidence)))

    assert from_record(TemporalEvidence, record) == evidence
