import math

import pytest
from pose_builders import make_pose, make_trajectory

from contextmap.state_estimation import (
    motion_deltas,
    summarize_distribution,
    summarize_motion,
)


def _yaw(angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def test_distribution_reports_order_statistics_with_nearest_rank_p95() -> None:
    summary = summarize_distribution([100.0, 1.0, 3.0, 2.0, 4.0])

    assert summary is not None
    assert (summary.count, summary.minimum, summary.median, summary.maximum) == (5, 1.0, 3.0, 100.0)
    assert summary.p95 == 100.0


def test_an_empty_distribution_is_absent() -> None:
    assert summarize_distribution([]) is None


def test_constant_translation_gives_a_degenerate_distribution() -> None:
    trajectory = make_trajectory([make_pose(index) for index in range(5)])

    motion = summarize_motion(trajectory)

    assert motion.translation_delta_m is not None
    assert (motion.translation_delta_m.minimum, motion.translation_delta_m.maximum) == (1.0, 1.0)
    assert motion.linear_speed_mps is not None
    assert motion.linear_speed_mps.median == pytest.approx(10.0)
    assert motion.orientation_delta_rad is not None
    assert motion.orientation_delta_rad.maximum == pytest.approx(0.0, abs=1e-12)


def test_orientation_change_is_measured_as_an_angle_and_an_angular_speed() -> None:
    poses = [make_pose(index, orientation=_yaw(0.1 * index)) for index in range(4)]

    motion = summarize_motion(make_trajectory(poses))

    assert motion.orientation_delta_rad is not None
    assert motion.orientation_delta_rad.median == pytest.approx(0.1)
    assert motion.angular_speed_radps is not None
    assert motion.angular_speed_radps.median == pytest.approx(1.0)


def test_deltas_name_the_consecutive_poses_they_come_from() -> None:
    trajectory = make_trajectory([make_pose(index) for index in range(3)])

    deltas = motion_deltas(trajectory)

    assert [(d.previous_estimate_id, d.next_estimate_id) for d in deltas] == [
        (trajectory.poses[0].estimate_id, trajectory.poses[1].estimate_id),
        (trajectory.poses[1].estimate_id, trajectory.poses[2].estimate_id),
    ]
    assert all(d.interval_ns == 100_000_000 for d in deltas)


def test_a_single_pose_has_no_motion() -> None:
    motion = summarize_motion(make_trajectory([make_pose(0)]))

    assert motion.translation_delta_m is None
    assert motion.linear_speed_mps is None
    assert motion_deltas(make_trajectory([make_pose(0)])) == ()
