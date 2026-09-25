"""Geometric comparison: typed, scale-aware evidence from the 3D support of two entities."""

from __future__ import annotations

import math
from typing import Any

import pytest
from mapping_builders import make_hypothesis
from resolution_entity_builders import entity_at, entity_over, lattice, scene_source

from contextmap.entity_resolution import (
    GEOMETRY_COMPARISON_POLICY_ID,
    EvidenceStatus,
    GeometryComparisonPolicy,
    GeometryEvidence,
    SupportDistancePolicy,
    UnavailableReason,
    compare_geometry,
)
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import (
    GeometryResolutionError,
    GeometrySummaryPolicy,
    OrientationPolicy,
    SemanticMapId,
)

POLICY = GeometryComparisonPolicy(
    min_shared_support_jaccard=0.5,
    min_bounds_iou=0.5,
    min_bounds_containment=0.9,
    min_conflict_gap_m=0.5,
    min_extent_ratio=0.3,
)


def statuses(evidence: GeometryEvidence) -> dict[str, EvidenceStatus]:
    return {item.rule_id: item.status for item in evidence.findings}


def two_objects(
    first_center: tuple[float, float, float],
    second_center: tuple[float, float, float],
    *,
    first_size: Any = 0.6,
    second_size: Any = 0.6,
    policy: GeometryComparisonPolicy = POLICY,
) -> GeometryEvidence:
    points = lattice(first_center, first_size)
    points += lattice(second_center, second_size)
    source = scene_source(dict(enumerate(points)))
    half = len(points) // 2
    first = entity_over("a", source, range(half))
    second = entity_over("b", source, range(half, len(points)))
    return compare_geometry(first, second, policy)


# --- same support and strong overlap ------------------------------------------------------------


def test_the_same_support_gives_the_strongest_geometric_evidence() -> None:
    source = scene_source(dict(enumerate(lattice((0, 0, 0), 0.6))))
    first = entity_over("a", source, range(27))
    second = entity_over("b", source, range(27))

    evidence = compare_geometry(first, second, POLICY)

    measurement = evidence.measurement
    assert measurement is not None
    assert measurement.support_jaccard == 1.0
    assert measurement.shared_support_count == 27
    assert measurement.bounds_iou == pytest.approx(1.0)
    assert measurement.centroid_distance_m == pytest.approx(0.0)
    assert measurement.extent_ratio == pytest.approx(1.0)
    assert evidence.status is EvidenceStatus.SUPPORTING
    assert statuses(evidence)["shared-support"] is EvidenceStatus.SUPPORTING
    assert statuses(evidence)["bounds-iou"] is EvidenceStatus.SUPPORTING


def test_strongly_overlapping_supports_are_supporting_without_sharing_a_point() -> None:
    evidence = two_objects((0, 0, 0), (0.1, 0, 0))

    assert evidence.measurement is not None and evidence.measurement.shared_support_count == 0
    assert evidence.measurement.bounds_iou == pytest.approx(0.5 / 0.7, rel=1e-6)
    assert statuses(evidence)["bounds-iou"] is EvidenceStatus.SUPPORTING
    assert statuses(evidence)["shared-support"] is EvidenceStatus.NEUTRAL
    assert evidence.status is EvidenceStatus.SUPPORTING


def test_every_metric_is_recorded_with_the_threshold_it_was_judged_against() -> None:
    evidence = two_objects((0, 0, 0), (0.1, 0, 0))

    by_rule = {item.rule_id: item for item in evidence.findings}

    assert by_rule["bounds-iou"].metric == "bounds_iou"
    assert by_rule["bounds-iou"].threshold == POLICY.min_bounds_iou
    assert by_rule["bounds-iou"].observed == pytest.approx(0.5 / 0.7, rel=1e-6)
    assert evidence.policy.policy_id == GEOMETRY_COMPARISON_POLICY_ID
    assert evidence.policy.configuration_fingerprint == POLICY.fingerprint()


# --- nearby but distinct ------------------------------------------------------------------------


def test_nearby_but_disjoint_objects_stay_distinguishable() -> None:
    evidence = two_objects((0, 0, 0), (1.5, 0, 0))

    assert evidence.measurement is not None
    assert evidence.measurement.bounds_iou == 0.0
    assert evidence.measurement.bounds_gap_m == pytest.approx(0.9)
    assert statuses(evidence)["bounds-separation"] is EvidenceStatus.CONFLICTING
    assert evidence.status is EvidenceStatus.CONFLICTING


def test_objects_closer_than_the_conflict_gap_are_not_called_conflicting() -> None:
    evidence = two_objects((0, 0, 0), (0.9, 0, 0))

    assert evidence.measurement is not None
    assert evidence.measurement.bounds_gap_m == pytest.approx(0.3)
    assert statuses(evidence)["bounds-separation"] is EvidenceStatus.NEUTRAL
    assert evidence.status is EvidenceStatus.NEUTRAL


def test_a_high_geometric_score_is_still_only_geometry() -> None:
    evidence = two_objects((0, 0, 0), (0.0, 0.0, 0.0))

    assert evidence.status is EvidenceStatus.SUPPORTING
    assert not any(hasattr(evidence, name) for name in ("semantic", "decision", "match"))


# --- partial support, background, degenerate and sparse -----------------------------------------


def test_a_partial_view_contained_in_the_whole_is_supported_by_containment() -> None:
    whole = lattice((0, 0, 0), (0.9, 0.6, 0.6))
    source = scene_source(dict(enumerate(whole)))
    full = entity_over("a", source, range(len(whole)))
    left_half = entity_over(
        "b", source, [index for index, point in enumerate(whole) if point[0] <= 0.0]
    )

    evidence = compare_geometry(full, left_half, POLICY)

    assert evidence.measurement is not None
    assert evidence.measurement.bounds_overlap_fraction_b == pytest.approx(1.0)
    assert evidence.measurement.bounds_iou == pytest.approx(0.5)
    assert evidence.measurement.shared_support_count == len(left_half.geometry.geometry_refs)
    assert statuses(evidence)["bounds-containment"] is EvidenceStatus.SUPPORTING
    assert statuses(evidence)["extent-mismatch"] is EvidenceStatus.NEUTRAL


def test_a_small_object_on_a_large_planar_background_is_not_evidence_of_identity() -> None:
    floor = lattice((0, 0, 0), (12.0, 12.0, 1.0), steps=6)
    box = lattice((1.0, 1.0, 0.0), 0.4)
    points = floor + box
    source = scene_source(dict(enumerate(points)))
    background = entity_over("a-floor", source, range(len(floor)))
    small = entity_over("b-box", source, range(len(floor), len(points)))

    evidence = compare_geometry(background, small, POLICY)

    assert evidence.measurement is not None
    assert evidence.measurement.bounds_overlap_fraction_b == pytest.approx(1.0)
    assert evidence.measurement.extent_ratio is not None and evidence.measurement.extent_ratio < 0.3
    assert statuses(evidence)["bounds-containment"] is EvidenceStatus.NEUTRAL
    assert statuses(evidence)["bounds-iou"] is EvidenceStatus.NEUTRAL
    assert statuses(evidence)["extent-mismatch"] is EvidenceStatus.NEUTRAL
    assert evidence.status is EvidenceStatus.NEUTRAL


def test_very_different_sizes_that_are_not_contained_conflict() -> None:
    evidence = two_objects((0, 0, 0), (1.0, 0, 0), first_size=0.4, second_size=(2.0, 0.6, 0.6))

    assert statuses(evidence)["extent-mismatch"] is EvidenceStatus.CONFLICTING


def test_flat_bounds_have_no_volumetric_overlap_instead_of_a_zero() -> None:
    flat = [(x * 0.2, y * 0.2, 0.0) for x in range(3) for y in range(3)]
    source = scene_source(dict(enumerate(flat)))
    first = entity_over("a", source, range(9))
    second = entity_over("b", source, range(9))

    evidence = compare_geometry(first, second, POLICY)

    assert evidence.measurement is not None
    assert evidence.measurement.bounds_iou is None
    assert evidence.measurement.bounds_overlap_fraction_a is None
    assert "a:degenerate_extent" in evidence.measurement.caveats
    assert statuses(evidence)["bounds-iou"] is EvidenceStatus.NEUTRAL
    assert statuses(evidence)["shared-support"] is EvidenceStatus.SUPPORTING
    assert evidence.status is EvidenceStatus.SUPPORTING


def test_a_flat_box_against_a_volumetric_one_has_no_extent_ratio() -> None:
    plane = [(x * 0.2, y * 0.2, 0.0) for x in range(3) for y in range(3)]
    source = scene_source(dict(enumerate(plane + lattice((0.2, 0.2, 0.0), 0.4))))
    flat = entity_over("a", source, range(9))
    volumetric = entity_over("b", source, range(9, 36))

    evidence = compare_geometry(flat, volumetric, POLICY)

    assert evidence.measurement is not None and evidence.measurement.extent_ratio is None
    assert statuses(evidence)["extent-mismatch"] is EvidenceStatus.NEUTRAL
    assert statuses(evidence)["bounds-iou"] is EvidenceStatus.NEUTRAL


def test_sparse_and_disconnected_supports_are_flagged_as_caveats() -> None:
    far_apart = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (5.0, 0.0, 0.0), (5.1, 0.1, 0.1)]
    two_close = [(0.0, 3.0, 0.0), (0.2, 3.2, 0.2)]
    source = scene_source(dict(enumerate(far_apart + two_close)))
    disconnected = entity_over("a", source, range(4))
    sparse = entity_over("b", source, [4, 5])

    evidence = compare_geometry(disconnected, sparse, POLICY)

    assert evidence.measurement is not None
    assert evidence.measurement.caveats == ("a:disconnected_support", "b:sparse_support")


def test_unequal_support_density_is_kept_visible() -> None:
    points = lattice((0, 0, 0), 0.6, steps=5) + lattice((0.05, 0, 0), 0.5, steps=2)
    source = scene_source(dict(enumerate(points)))
    dense = entity_over("a", source, range(125))
    sparse = entity_over("b", source, range(125, len(points)))

    evidence = compare_geometry(dense, sparse, POLICY)

    assert evidence.measurement is not None
    assert (evidence.measurement.support_count_a, evidence.measurement.support_count_b) == (125, 8)


# --- frames and maps ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "other",
    [
        {"map_id": MapId("map-0002")},
        {"frame": "odom"},
    ],
)
def test_entities_that_cannot_share_a_frame_are_blocked_not_compared(other: dict[str, Any]) -> None:
    first = entity_at("a", (0, 0, 0))
    second = entity_at("b", (0, 0, 0), semantic_map_id=SemanticMapId("semantic-map-0002"), **other)

    evidence = compare_geometry(first, second, POLICY)

    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.measurement is None
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.BLOCKED_BY_GATE


def test_an_entity_is_not_compared_with_itself() -> None:
    entity = entity_at("a", (0, 0, 0))

    evidence = compare_geometry(entity, entity, POLICY)

    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.BLOCKED_BY_GATE


def test_the_comparison_is_symmetric_and_canonically_ordered() -> None:
    source = scene_source(dict(enumerate(lattice((0, 0, 0), 0.6) + lattice((0.4, 0, 0), 0.4))))
    first = entity_over("a", source, range(27))
    second = entity_over("b", source, range(27, 54))

    assert compare_geometry(first, second, POLICY) == compare_geometry(second, first, POLICY)
    measurement = compare_geometry(second, first, POLICY).measurement
    assert measurement is not None and measurement.support_count_a == 27


# --- support distance and orientation -----------------------------------------------------------


def test_support_distances_need_a_source_and_are_optional() -> None:
    policy = GeometryComparisonPolicy(
        min_shared_support_jaccard=0.5,
        min_bounds_iou=0.5,
        min_bounds_containment=0.9,
        min_conflict_gap_m=0.5,
        min_extent_ratio=0.3,
        support_distance=SupportDistancePolicy(max_points_per_side=100, max_mean_distance_m=0.1),
    )
    source = scene_source(dict(enumerate(lattice((0, 0, 0), 0.6) + lattice((0.05, 0, 0), 0.6))))
    first = entity_over("a", source, range(27))
    second = entity_over("b", source, range(27, 54))

    without = compare_geometry(first, second, POLICY)
    with_distance = compare_geometry(first, second, policy, source=source)

    assert without.measurement is not None and without.measurement.support_distance is None
    assert with_distance.measurement is not None
    distance = with_distance.measurement.support_distance
    assert distance is not None
    assert distance.hausdorff_m == pytest.approx(0.05)
    assert distance.a_to_b_mean_m == pytest.approx(0.05)
    assert distance.b_to_a_mean_m == pytest.approx(0.05)
    assert statuses(with_distance)["support-proximity"] is EvidenceStatus.SUPPORTING
    with pytest.raises(ValueError, match="GeometrySource"):
        compare_geometry(first, second, policy)


def test_distant_supports_have_large_nearest_point_distances() -> None:
    policy = GeometryComparisonPolicy(
        min_shared_support_jaccard=0.5,
        min_bounds_iou=0.5,
        min_bounds_containment=0.9,
        min_conflict_gap_m=0.5,
        min_extent_ratio=0.3,
        support_distance=SupportDistancePolicy(max_points_per_side=100, max_mean_distance_m=0.1),
    )
    source = scene_source(dict(enumerate(lattice((0, 0, 0), 0.6) + lattice((2, 0, 0), 0.6))))
    first = entity_over("a", source, range(27))
    second = entity_over("b", source, range(27, 54))

    evidence = compare_geometry(first, second, policy, source=source)

    distance = evidence.measurement.support_distance if evidence.measurement else None
    assert distance is not None
    assert distance.a_to_b_mean_m > 1.4
    assert statuses(evidence)["support-proximity"] is EvidenceStatus.NEUTRAL


def test_support_distances_are_sampled_deterministically_and_the_sample_is_recorded() -> None:
    policy = GeometryComparisonPolicy(
        min_shared_support_jaccard=0.5,
        min_bounds_iou=0.5,
        min_bounds_containment=0.9,
        min_conflict_gap_m=0.5,
        min_extent_ratio=0.3,
        support_distance=SupportDistancePolicy(max_points_per_side=10, max_mean_distance_m=0.1),
    )
    source = scene_source(dict(enumerate(lattice((0, 0, 0), 0.6, steps=5))))
    first = entity_over("a", source, range(0, 100))
    second = entity_over("b", source, range(25, 125))

    one = compare_geometry(first, second, policy, source=source)
    two = compare_geometry(first, second, policy, source=source)

    assert one == two
    assert one.measurement is not None and one.measurement.support_distance is not None
    assert one.measurement.support_distance.points_a == 10
    assert one.measurement.support_distance.points_b == 10
    assert one.measurement.support_count_a == 100


def test_an_unresolvable_support_fails_loudly_instead_of_being_skipped() -> None:
    policy = GeometryComparisonPolicy(
        min_shared_support_jaccard=0.5,
        min_bounds_iou=0.5,
        min_bounds_containment=0.9,
        min_conflict_gap_m=0.5,
        min_extent_ratio=0.3,
        support_distance=SupportDistancePolicy(max_points_per_side=100, max_mean_distance_m=0.1),
    )
    full = scene_source(dict(enumerate(lattice((0, 0, 0), 0.6) + lattice((2, 0, 0), 0.6))))
    first = entity_over("a", full, range(27))
    second = entity_over("b", full, range(27, 54))
    truncated = scene_source(dict(enumerate(lattice((0, 0, 0), 0.6))))

    with pytest.raises(GeometryResolutionError):
        compare_geometry(first, second, policy, source=truncated)


def test_orientation_is_measured_when_both_exist_and_judged_only_when_the_policy_asks() -> None:
    oriented = GeometrySummaryPolicy(
        sparse_point_threshold=3,
        connectivity_radius_m=0.5,
        orientation=OrientationPolicy(min_points=5, min_variance_ratio=1.2),
    )
    slab = lattice((0, 0, 0), (1.2, 0.5, 0.2), steps=4)
    rotated = [(y, x, z) for x, y, z in lattice((3, 0, 0), (1.2, 0.5, 0.2), steps=4)]
    source = scene_source(dict(enumerate(slab + rotated)))
    first = entity_over("a", source, range(len(slab)), policy=oriented)
    second = entity_over("b", source, range(len(slab), len(slab) + len(rotated)), policy=oriented)
    plain = entity_over("c", source, range(len(slab)))
    with_orientation = GeometryComparisonPolicy(
        min_shared_support_jaccard=0.5,
        min_bounds_iou=0.5,
        min_bounds_containment=0.9,
        min_conflict_gap_m=0.5,
        min_extent_ratio=0.3,
        max_orientation_angle_rad=math.radians(30),
    )

    measured = compare_geometry(first, second, with_orientation)
    missing = compare_geometry(first, plain, with_orientation)

    assert measured.measurement is not None
    assert measured.measurement.orientation_angle_rad == pytest.approx(math.pi / 2, abs=1e-3)
    assert statuses(measured)["orientation-mismatch"] is EvidenceStatus.CONFLICTING
    assert missing.measurement is not None and missing.measurement.orientation_angle_rad is None
    assert statuses(missing)["orientation-mismatch"] is EvidenceStatus.NEUTRAL
    assert "orientation-mismatch" not in statuses(compare_geometry(first, second, POLICY))


# --- policy -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_shared_support_jaccard": 0.0},
        {"min_shared_support_jaccard": 1.5},
        {"min_bounds_iou": math.nan},
        {"min_bounds_containment": 0.0},
        {"min_conflict_gap_m": 0.0},
        {"min_conflict_gap_m": math.inf},
        {"min_extent_ratio": 1.2},
        {"max_orientation_angle_rad": -0.1},
        {"max_orientation_angle_rad": 2.0},
    ],
)
def test_the_policy_refuses_impossible_thresholds(overrides: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "min_shared_support_jaccard": 0.5,
        "min_bounds_iou": 0.5,
        "min_bounds_containment": 0.9,
        "min_conflict_gap_m": 0.5,
        "min_extent_ratio": 0.3,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        GeometryComparisonPolicy(**values)


def test_the_support_distance_policy_refuses_impossible_values() -> None:
    with pytest.raises(ValueError, match="max_points_per_side"):
        SupportDistancePolicy(max_points_per_side=0, max_mean_distance_m=0.1)
    with pytest.raises(ValueError, match="max_mean_distance_m"):
        SupportDistancePolicy(max_points_per_side=10, max_mean_distance_m=0.0)


def test_the_fingerprint_changes_with_every_threshold() -> None:
    variants = [
        GeometryComparisonPolicy(
            min_shared_support_jaccard=0.6,
            min_bounds_iou=0.5,
            min_bounds_containment=0.9,
            min_conflict_gap_m=0.5,
            min_extent_ratio=0.3,
        ),
        GeometryComparisonPolicy(
            min_shared_support_jaccard=0.5,
            min_bounds_iou=0.5,
            min_bounds_containment=0.9,
            min_conflict_gap_m=0.5,
            min_extent_ratio=0.3,
            support_distance=SupportDistancePolicy(max_points_per_side=10, max_mean_distance_m=0.1),
        ),
    ]

    assert len({POLICY.fingerprint(), *(item.fingerprint() for item in variants)}) == 3


def test_the_labels_of_the_entities_play_no_part() -> None:
    first = entity_at("a", (0, 0, 0), hypotheses=(make_hypothesis(label="pallet"),))
    person = entity_at("b", (0, 0, 0), hypotheses=(make_hypothesis(label="person"),))
    unlabeled = entity_at("b", (0, 0, 0), hypotheses=())

    assert compare_geometry(first, person, POLICY) == compare_geometry(first, unlabeled, POLICY)
