"""Motion statistics of a trajectory.

Per-interval translation and orientation changes and the speeds derived from
them are what expose a jump, a stall or a timestamp problem in a trajectory.
They are measured, never judged: deciding what counts as a discontinuity needs
a threshold, and thresholds belong to the evaluation that owns the reference
profile, not to the contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from contextmap.shared import quaternion_angle_between
from contextmap.state_estimation.models import PoseEstimateId, Trajectory

_NANOSECONDS_PER_SECOND = 1_000_000_000
_P95 = 0.95


@dataclass(frozen=True, kw_only=True)
class DistributionSummary:
    """Order statistics of a set of measurements.

    Attributes:
        count: Number of measurements.
        minimum: Smallest value.
        median: Median value.
        p95: 95th percentile by nearest rank.
        maximum: Largest value.
    """

    count: int
    minimum: float
    median: float
    p95: float
    maximum: float


def summarize_distribution(values: Sequence[float]) -> DistributionSummary | None:
    """Summarize measurements with order statistics.

    Args:
        values: The measurements, in any order.

    Returns:
        The summary, or ``None`` when there are no measurements.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(_P95 * len(ordered))
    return DistributionSummary(
        count=len(ordered),
        minimum=ordered[0],
        median=median(ordered),
        p95=ordered[rank - 1],
        maximum=ordered[-1],
    )


@dataclass(frozen=True, kw_only=True)
class MotionDelta:
    """Change between two consecutive poses.

    Attributes:
        previous_estimate_id: Earlier pose.
        next_estimate_id: Later pose.
        interval_ns: Time between them, in nanoseconds; always positive.
        translation_delta_m: Distance between the two positions, in meters.
        orientation_delta_rad: Smallest rotation between the two orientations.
        linear_speed_mps: ``translation_delta_m`` over the interval.
        angular_speed_radps: ``orientation_delta_rad`` over the interval.
    """

    previous_estimate_id: PoseEstimateId
    next_estimate_id: PoseEstimateId
    interval_ns: int
    translation_delta_m: float
    orientation_delta_rad: float
    linear_speed_mps: float
    angular_speed_radps: float


def motion_deltas(trajectory: Trajectory) -> tuple[MotionDelta, ...]:
    """Measure the change between every pair of consecutive poses.

    Args:
        trajectory: The trajectory to measure.

    Returns:
        One delta per interval; empty for a single pose.
    """
    deltas: list[MotionDelta] = []
    for earlier, later in pairwise(trajectory.poses):
        interval_ns = later.timestamp.total_nanoseconds() - earlier.timestamp.total_nanoseconds()
        seconds = interval_ns / _NANOSECONDS_PER_SECOND
        translation = math.dist(earlier.translation_m, later.translation_m)
        rotation = quaternion_angle_between(earlier.orientation, later.orientation)
        deltas.append(
            MotionDelta(
                previous_estimate_id=earlier.estimate_id,
                next_estimate_id=later.estimate_id,
                interval_ns=interval_ns,
                translation_delta_m=translation,
                orientation_delta_rad=rotation,
                linear_speed_mps=translation / seconds,
                angular_speed_radps=rotation / seconds,
            )
        )
    return tuple(deltas)


@dataclass(frozen=True, kw_only=True)
class MotionSummary:
    """Distributions of the per-interval motion of a trajectory.

    Each field is ``None`` when the trajectory has fewer than two poses.

    Attributes:
        translation_delta_m: Distance between consecutive positions.
        orientation_delta_rad: Rotation between consecutive orientations.
        linear_speed_mps: Speed derived from translation and interval.
        angular_speed_radps: Angular speed derived from rotation and interval.
    """

    translation_delta_m: DistributionSummary | None
    orientation_delta_rad: DistributionSummary | None
    linear_speed_mps: DistributionSummary | None
    angular_speed_radps: DistributionSummary | None


def summarize_motion(trajectory: Trajectory) -> MotionSummary:
    """Summarize the per-interval motion of a trajectory.

    Args:
        trajectory: The trajectory to summarize.

    Returns:
        The distributions of translation, rotation and speeds.
    """
    deltas = motion_deltas(trajectory)
    return MotionSummary(
        translation_delta_m=summarize_distribution([d.translation_delta_m for d in deltas]),
        orientation_delta_rad=summarize_distribution([d.orientation_delta_rad for d in deltas]),
        linear_speed_mps=summarize_distribution([d.linear_speed_mps for d in deltas]),
        angular_speed_radps=summarize_distribution([d.angular_speed_radps for d in deltas]),
    )
