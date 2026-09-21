"""Typed per-channel evidence: availability, findings and the measurements each channel keeps."""

from __future__ import annotations

import math
from typing import Any

import pytest
from resolution_builders import (
    EMBEDDING_SPACE,
    appearance_evidence,
    appearance_measurement,
    channel_policy,
    feature_ref,
    finding,
    geometry_evidence,
    geometry_measurement,
    representation_evidence,
    semantic_evidence,
    temporal_evidence,
    temporal_measurement,
    unavailable,
)

from contextmap.entity_resolution import (
    AppearanceEvidence,
    ChannelPolicyRef,
    EvidenceStatus,
    FeatureContribution,
    Finding,
    GeometryEvidence,
    MatchChannel,
    SupportDistance,
    Unavailability,
    UnavailableReason,
)


def test_each_evidence_class_names_its_channel() -> None:
    assert geometry_evidence().channel is MatchChannel.GEOMETRY
    assert semantic_evidence().channel is MatchChannel.SEMANTIC
    assert appearance_evidence().channel is MatchChannel.APPEARANCE
    assert temporal_evidence().channel is MatchChannel.TEMPORAL
    assert representation_evidence().channel is MatchChannel.POINT_REPRESENTATION


def test_a_channel_is_measured_or_unavailable_never_both_and_never_neither() -> None:
    with pytest.raises(ValueError, match="measurement or an unavailability"):
        GeometryEvidence(
            policy=channel_policy(), measurement=geometry_measurement(), unavailable=unavailable()
        )
    with pytest.raises(ValueError, match="measurement or an unavailability"):
        GeometryEvidence(policy=channel_policy())


def test_an_unavailable_channel_reports_no_findings() -> None:
    with pytest.raises(ValueError, match="findings"):
        GeometryEvidence(policy=channel_policy(), unavailable=unavailable(), findings=(finding(),))


def test_an_unavailable_channel_is_not_a_zero_score() -> None:
    evidence = AppearanceEvidence(
        policy=channel_policy(),
        unavailable=unavailable(UnavailableReason.INCOMPATIBLE_DOMAIN, "different spaces"),
    )

    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.measurement is None
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.INCOMPATIBLE_DOMAIN


def test_unavailability_explains_itself() -> None:
    with pytest.raises(ValueError, match="detail"):
        Unavailability(reason=UnavailableReason.MISSING_EVIDENCE, detail=" ")


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((), EvidenceStatus.NEUTRAL),
        ((EvidenceStatus.NEUTRAL,), EvidenceStatus.NEUTRAL),
        ((EvidenceStatus.SUPPORTING,), EvidenceStatus.SUPPORTING),
        ((EvidenceStatus.SUPPORTING, EvidenceStatus.NEUTRAL), EvidenceStatus.SUPPORTING),
        ((EvidenceStatus.CONFLICTING,), EvidenceStatus.CONFLICTING),
        # Um conflito dentro do canal não é diluído pelos achados que apoiam.
        ((EvidenceStatus.SUPPORTING, EvidenceStatus.CONFLICTING), EvidenceStatus.CONFLICTING),
    ],
)
def test_the_channel_status_follows_its_findings(
    statuses: tuple[EvidenceStatus, ...], expected: EvidenceStatus
) -> None:
    evidence = GeometryEvidence(
        policy=channel_policy(),
        measurement=geometry_measurement(),
        findings=tuple(finding(status, f"rule-{index}") for index, status in enumerate(statuses)),
    )

    assert evidence.status is expected


def test_a_finding_cannot_report_unavailability() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        Finding(rule_id="rule-1", status=EvidenceStatus.UNAVAILABLE, detail="nothing")


def test_a_finding_that_names_a_metric_records_the_observed_value() -> None:
    with pytest.raises(ValueError, match="observed"):
        finding(metric="bounds_iou")
    with pytest.raises(ValueError, match="metric"):
        finding(observed=0.5)
    with pytest.raises(ValueError, match="finite"):
        finding(metric="bounds_iou", observed=math.nan)
    recorded = finding(metric="bounds_iou", observed=0.5, threshold=0.3)
    assert (recorded.observed, recorded.threshold) == (0.5, 0.3)


def test_a_finding_needs_a_rule_and_an_explanation() -> None:
    with pytest.raises(ValueError, match="rule_id"):
        Finding(rule_id="", status=EvidenceStatus.NEUTRAL, detail="x")
    with pytest.raises(ValueError, match="detail"):
        Finding(rule_id="rule-1", status=EvidenceStatus.NEUTRAL, detail="")


def test_a_channel_names_the_policy_and_configuration_it_was_measured_under() -> None:
    with pytest.raises(ValueError, match="configuration_fingerprint"):
        ChannelPolicyRef(policy_id="entity-geometry-comparison-v1", configuration_fingerprint="")


# --- geometry -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"centroid_distance_m": -0.1},
        {"centroid_distance_m": math.inf},
        {"bounds_gap_m": -1.0},
        {"bounds_iou": 1.5},
        {"bounds_overlap_fraction_a": -0.1},
        {"extent_ratio": 0.0},
        {"support_count_a": 0},
        {"shared_support_count": 5},
        {"support_jaccard": 0.9},
        {"orientation_angle_rad": 2.0},
        {"caveats": ("b:sparse_support", "a:sparse_support")},
    ],
)
def test_a_geometry_measurement_refuses_impossible_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        geometry_measurement(**overrides)


def test_a_degenerate_bounds_have_no_iou_instead_of_a_zero() -> None:
    measurement = geometry_measurement(
        bounds_iou=None, bounds_overlap_fraction_a=None, bounds_overlap_fraction_b=None
    )

    assert measurement.bounds_iou is None


def test_support_distance_statistics_are_consistent() -> None:
    SupportDistance(a_to_b_mean_m=0.1, b_to_a_mean_m=0.2, hausdorff_m=0.5, points_a=10, points_b=8)
    with pytest.raises(ValueError, match="hausdorff"):
        SupportDistance(
            a_to_b_mean_m=0.3, b_to_a_mean_m=0.2, hausdorff_m=0.1, points_a=10, points_b=8
        )


# --- appearance ---------------------------------------------------------------------------------


def test_an_appearance_measurement_names_the_space_and_the_aggregation() -> None:
    measurement = appearance_measurement()

    assert measurement.embedding_space_id == EMBEDDING_SPACE
    assert measurement.aggregation_id == "physical-observation-prototype-v1"
    assert len(measurement.contributions_a) == len(measurement.contributions_b) == 1


def test_appearance_features_must_live_in_the_declared_space() -> None:
    foreign = FeatureContribution(
        physical_observation_id="frame-0120",
        feature_refs=(feature_ref(1, space="sha256:another-space"),),
    )

    with pytest.raises(ValueError, match="embedding space"):
        appearance_measurement(contributions_a=(foreign,))


def test_appearance_needs_contributions_on_both_sides() -> None:
    with pytest.raises(ValueError, match="contributions_a"):
        appearance_measurement(contributions_a=())


def test_a_physical_observation_appears_once_per_side() -> None:
    one = FeatureContribution(physical_observation_id="frame-0120", feature_refs=(feature_ref(1),))
    again = FeatureContribution(
        physical_observation_id="frame-0120", feature_refs=(feature_ref(3),)
    )

    with pytest.raises(ValueError, match="contributions_a"):
        appearance_measurement(contributions_a=(one, again))


def test_repeated_inference_stays_inside_one_physical_observation() -> None:
    # Duas features do mesmo frame físico ficam juntas: contam como uma observação.
    contribution = FeatureContribution(
        physical_observation_id="frame-0120", feature_refs=(feature_ref(1), feature_ref(3))
    )

    measurement = appearance_measurement(contributions_a=(contribution,))

    assert len(measurement.contributions_a) == 1
    assert len(measurement.contributions_a[0].feature_refs) == 2


def test_a_contribution_needs_sorted_unique_features() -> None:
    with pytest.raises(ValueError, match="feature_refs"):
        FeatureContribution(
            physical_observation_id="frame-0120", feature_refs=(feature_ref(3), feature_ref(1))
        )
    with pytest.raises(ValueError, match="feature_refs"):
        FeatureContribution(physical_observation_id="frame-0120", feature_refs=())


# --- temporal -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"interval_overlap_ns": -1},
        {"interval_overlap_ns": 5, "interval_gap_ns": 5},
        {"shared_physical_observation_count": 3},
        {"union_physical_observation_count": 9},
        {"physical_observation_count_a": 0},
        {"inference_result_count_a": 1},
        {"clock_id": ""},
    ],
)
def test_a_temporal_measurement_refuses_incoherent_counts(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        temporal_measurement(**overrides)


def test_physical_observations_and_inference_results_are_counted_apart() -> None:
    measurement = temporal_measurement()

    assert measurement.physical_observation_count_a == 2
    assert measurement.inference_result_count_a == 3
