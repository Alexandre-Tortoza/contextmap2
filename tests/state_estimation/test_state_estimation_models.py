import dataclasses
import math

import pytest
from pose_builders import (
    TRAJECTORY_ID,
    make_pose,
    make_trajectory,
    make_trajectory_provenance,
    timestamp_ns,
)

import contextmap.state_estimation as state_estimation
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.state_estimation import (
    PoseEstimate,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryGap,
    pose_estimate_id_for,
)


def test_public_api_exports_resolve() -> None:
    for name in state_estimation.__all__:
        assert hasattr(state_estimation, name), name


# --- PoseEstimate -----------------------------------------------------------


def test_pose_declares_which_frame_is_transformed_into_which() -> None:
    pose = make_pose(0, parent_frame="map", child_frame="body")

    assert pose.parent_frame == FrameId("map")
    assert pose.child_frame == FrameId("body")


def test_pose_keeps_missing_uncertainty_explicit() -> None:
    assert make_pose(0).covariance is None


def test_pose_accepts_a_6x6_covariance() -> None:
    covariance = tuple(float(i) for i in range(36))

    assert make_pose(0, covariance=covariance).covariance == covariance


@pytest.mark.parametrize("size", [0, 9, 35, 37])
def test_pose_rejects_covariance_with_wrong_size(size: int) -> None:
    with pytest.raises(ValueError, match="covariance"):
        make_pose(0, covariance=(0.0,) * size)


def test_pose_rejects_non_finite_covariance() -> None:
    with pytest.raises(ValueError, match="covariance"):
        make_pose(0, covariance=(math.nan,) + (0.0,) * 35)


def test_pose_can_be_flagged_as_degraded() -> None:
    assert make_pose(0, validity=PoseValidity.DEGRADED).validity is PoseValidity.DEGRADED


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_pose_rejects_non_finite_translation(bad_value: float) -> None:
    with pytest.raises(ValueError, match="translation_m"):
        make_pose(0, translation_m=(0.0, bad_value, 0.0))


def test_pose_rejects_orientation_that_is_not_a_unit_quaternion() -> None:
    with pytest.raises(ValueError, match="unit quaternion"):
        make_pose(0, orientation=(0.0, 0.0, 0.0, 1.01))


def test_pose_rejects_non_finite_orientation() -> None:
    with pytest.raises(ValueError, match="unit quaternion"):
        make_pose(0, orientation=(math.nan, 0.0, 0.0, 1.0))


def test_pose_rejects_identical_parent_and_child_frames() -> None:
    with pytest.raises(ValueError, match="parent_frame and child_frame"):
        make_pose(0, parent_frame="map", child_frame="map")


@pytest.mark.parametrize(
    ("parent_frame", "child_frame"),
    [("", "body"), ("map", "")],
)
def test_pose_rejects_empty_frame_ids(parent_frame: str, child_frame: str) -> None:
    with pytest.raises(ValueError, match="frame"):
        make_pose(0, parent_frame=parent_frame, child_frame=child_frame)


def test_pose_provenance_requires_at_least_one_source_observation() -> None:
    with pytest.raises(ValueError, match="source_observation_ids"):
        PoseProvenance(source_observation_ids=())


def test_pose_provenance_records_declared_conversions() -> None:
    provenance = PoseProvenance(
        source_observation_ids=(SourceObservationId("pose-0000"),),
        conversions_applied=("renormalized orientation (norm=1.0004)",),
    )

    assert provenance.conversions_applied == ("renormalized orientation (norm=1.0004)",)


def test_pose_is_immutable() -> None:
    pose = make_pose(0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        pose.translation_m = (1.0, 1.0, 1.0)  # type: ignore[misc]


def test_pose_estimate_id_is_deterministic_and_scoped_to_its_trajectory() -> None:
    first = pose_estimate_id_for(trajectory_id=TRAJECTORY_ID, index=7)

    assert first == pose_estimate_id_for(trajectory_id=TRAJECTORY_ID, index=7)
    assert first != pose_estimate_id_for(trajectory_id=TRAJECTORY_ID, index=8)
    assert TRAJECTORY_ID in first


# --- Trajectory -------------------------------------------------------------


def test_trajectory_exposes_time_bounds_derived_from_its_poses() -> None:
    trajectory = make_trajectory()

    assert trajectory.time_bounds.start == trajectory.poses[0].timestamp
    assert trajectory.time_bounds.end == trajectory.poses[-1].timestamp
    assert trajectory.time_bounds.duration_ns == 400_000_000


def test_trajectory_keeps_poses_as_an_immutable_tuple() -> None:
    trajectory = make_trajectory()

    assert isinstance(trajectory.poses, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        trajectory.poses = ()  # type: ignore[misc]


def test_trajectory_rejects_an_empty_pose_sequence() -> None:
    with pytest.raises(ValueError, match="at least one pose"):
        Trajectory(
            trajectory_id=TRAJECTORY_ID,
            reference_frame=FrameId("map"),
            body_frame=FrameId("body"),
            poses=(),
            gaps=(),
            provenance=make_trajectory_provenance(),
        )


@pytest.mark.parametrize(
    ("parent_frame", "child_frame"),
    [("odom", "body"), ("map", "imu")],
)
def test_trajectory_rejects_poses_with_other_frame_semantics(
    parent_frame: str, child_frame: str
) -> None:
    poses = [make_pose(0), make_pose(1, parent_frame=parent_frame, child_frame=child_frame)]

    with pytest.raises(ValueError, match="frame"):
        make_trajectory(poses)


def test_trajectory_rejects_mixed_clock_domains() -> None:
    poses = [make_pose(0), make_pose(1, clock_id="another:clock")]

    with pytest.raises(ValueError, match="clock"):
        make_trajectory(poses)


@pytest.mark.parametrize("second_time_ns", [0, -1])
def test_trajectory_requires_strictly_increasing_timestamps(second_time_ns: int) -> None:
    poses = [make_pose(0, time_ns=0), make_pose(1, time_ns=second_time_ns)]

    with pytest.raises(ValueError, match="strictly increasing"):
        make_trajectory(poses)


def test_trajectory_rejects_duplicate_estimate_ids() -> None:
    first = make_pose(0)
    duplicate = dataclasses.replace(first, timestamp=timestamp_ns(100_000_000))

    with pytest.raises(ValueError, match="duplicate estimate_id"):
        make_trajectory([first, duplicate])


def test_trajectory_records_gaps_between_consecutive_poses() -> None:
    poses = [make_pose(0), make_pose(1), make_pose(2, time_ns=1_000_000_000), make_pose(3)]
    poses[3] = make_pose(3, time_ns=1_100_000_000)
    gap = TrajectoryGap(
        previous_estimate_id=poses[1].estimate_id,
        next_estimate_id=poses[2].estimate_id,
        duration_ns=900_000_000,
    )

    trajectory = make_trajectory(poses, gaps=[gap])

    assert trajectory.gaps == (gap,)
    assert trajectory.quality_summary().gap_count == 1


def test_trajectory_rejects_a_gap_that_disagrees_with_the_pose_timestamps() -> None:
    poses = [make_pose(0), make_pose(1)]
    gap = TrajectoryGap(
        previous_estimate_id=poses[0].estimate_id,
        next_estimate_id=poses[1].estimate_id,
        duration_ns=1,
    )

    with pytest.raises(ValueError, match="gap duration"):
        make_trajectory(poses, gaps=[gap])


def test_trajectory_rejects_a_gap_between_non_consecutive_poses() -> None:
    poses = [make_pose(0), make_pose(1), make_pose(2)]
    gap = TrajectoryGap(
        previous_estimate_id=poses[0].estimate_id,
        next_estimate_id=poses[2].estimate_id,
        duration_ns=200_000_000,
    )

    with pytest.raises(ValueError, match="consecutive"):
        make_trajectory(poses, gaps=[gap])


def test_trajectory_rejects_a_gap_that_references_an_unknown_pose() -> None:
    poses = [make_pose(0), make_pose(1)]
    gap = TrajectoryGap(
        previous_estimate_id=poses[0].estimate_id,
        next_estimate_id=pose_estimate_id_for(trajectory_id=TRAJECTORY_ID, index=99),
        duration_ns=100_000_000,
    )

    with pytest.raises(ValueError, match="unknown"):
        make_trajectory(poses, gaps=[gap])


def test_quality_summary_reports_sampling_and_flagged_poses() -> None:
    poses = [
        make_pose(0),
        make_pose(1, validity=PoseValidity.DEGRADED, covariance=(0.0,) * 36),
        make_pose(2, time_ns=250_000_000),
    ]

    quality = make_trajectory(poses).quality_summary()

    assert quality.pose_count == 3
    assert quality.duration_ns == 250_000_000
    assert quality.min_interval_ns == 100_000_000
    assert quality.max_interval_ns == 150_000_000
    assert quality.median_interval_ns == 100_000_000
    assert quality.degraded_pose_count == 1
    assert quality.pose_with_covariance_count == 1
    assert quality.gap_count == 0


def test_quality_summary_of_a_single_pose_has_no_intervals() -> None:
    quality = make_trajectory([make_pose(0)]).quality_summary()

    assert quality.pose_count == 1
    assert quality.duration_ns == 0
    assert quality.min_interval_ns is None
    assert quality.median_interval_ns is None
    assert quality.max_interval_ns is None


def test_pose_estimate_is_the_canonical_pose_type() -> None:
    assert isinstance(make_pose(0), PoseEstimate)
