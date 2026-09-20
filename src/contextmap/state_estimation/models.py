"""Canonical pose and trajectory contracts for State Estimation.

State Estimation owns the *dynamic* part of the rig geometry: where the body
is in the map at time ``t``. Static sensor extrinsics remain calibration owned
by Ingestion (:class:`~contextmap.ingestion.CalibrationSet`), and a pose read
directly from a dataset is only an input measurement
(:class:`~contextmap.ingestion.ExternalPoseMeasurement`), never a
:class:`PoseEstimate` until a backend has validated and published it.

Transform convention
    ``T_parent_child`` maps coordinates expressed in ``child_frame`` into
    ``parent_frame``: ``p_parent = R(orientation) * p_child + translation_m``.
    ``orientation`` is a unit quaternion ordered ``(x, y, z, w)``. The same
    convention is used by ``RigidTransform`` and ``ExternalPoseMeasurement``
    in Ingestion, so no reordering or inversion happens at the boundary. A
    pose is never a bare matrix: every :class:`PoseEstimate` names the frame
    it transforms from and into.

No ROS message, FAST-LIO object or NumPy array appears in these contracts; a
persisted trajectory is readable with the standard library alone. See
``src/contextmap/state_estimation/docs/contracts.md`` for the field reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from statistics import median_low
from typing import NewType

from contextmap.ingestion import Covariance6x6, FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import Quaternion, SourceTimestamp, Vector3, is_unit_quaternion

PoseEstimateId = NewType("PoseEstimateId", str)
"""Identity of a :class:`PoseEstimate`, local to its state-estimation artifact."""

TrajectoryId = NewType("TrajectoryId", str)
"""Identity of a :class:`Trajectory`, local to its state-estimation artifact."""

_COVARIANCE_SIZE = 36


class PoseValidity(Enum):
    """Quality state a backend attaches to one published pose.

    Attributes:
        VALID: The pose passed structural validation and the backend has no
            reason to distrust it.
        DEGRADED: The pose passed structural validation, but the backend
            flagged it as unreliable. It stays in the trajectory so consumers
            can decide, instead of the backend silently dropping it.
    """

    VALID = "valid"
    DEGRADED = "degraded"


def pose_estimate_id_for(*, trajectory_id: TrajectoryId, index: int) -> PoseEstimateId:
    """Compute the deterministic identity of the ``index``-th pose of a trajectory.

    Args:
        trajectory_id: The trajectory that owns the pose.
        index: Zero-based position of the pose in the trajectory.

    Returns:
        A pure function of the inputs, so repeated construction and
        serialization round-trips never need an identity registry.
    """
    return PoseEstimateId(f"{trajectory_id}--pose-{index:06d}")


@dataclass(frozen=True, kw_only=True)
class PoseProvenance:
    """Traceability of one pose back to the observations that produced it.

    Attributes:
        source_observation_ids: Canonical observations the backend consumed
            to publish this pose (external pose samples, or the LiDAR/IMU
            observations of an estimator step). Never empty.
        conversions_applied: Human-readable notes of every normalization the
            backend applied to the source values, e.g. ``"renormalized
            orientation (norm=1.0004)"``. Empty asserts none was necessary.
    """

    source_observation_ids: tuple[SourceObservationId, ...]
    conversions_applied: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require every pose to trace back to at least one observation.

        Raises:
            ValueError: If ``source_observation_ids`` is empty.
        """
        if not self.source_observation_ids:
            raise ValueError("source_observation_ids must contain at least one observation")


@dataclass(frozen=True, kw_only=True)
class PoseEstimate:
    """Dynamic pose ``T_parent_child`` of a body frame at one timestamp.

    Attributes:
        estimate_id: Identity of this pose within its artifact.
        timestamp: Time the pose refers to, with its clock domain.
        parent_frame: Reference frame the pose is expressed in, e.g. ``"map"``.
        child_frame: Frame whose pose is reported, e.g. ``"body"``.
        translation_m: ``(x, y, z)`` of ``child_frame``'s origin in
            ``parent_frame``, in meters.
        orientation: Unit quaternion ``(x, y, z, w)`` rotating ``child_frame``
            axes into ``parent_frame`` axes.
        validity: Quality state attached by the backend.
        provenance: Source observations and conversions behind this pose.
        covariance: Optional row-major 6x6 covariance over ``(x, y, z,
            rotation about x, y, z)`` expressed in ``parent_frame`` (m^2 and
            rad^2, the ROS ``Odometry`` convention). ``None`` means the
            backend reported no uncertainty; it is never fabricated.
    """

    estimate_id: PoseEstimateId
    timestamp: SourceTimestamp
    parent_frame: FrameId
    child_frame: FrameId
    translation_m: Vector3
    orientation: Quaternion
    validity: PoseValidity
    provenance: PoseProvenance
    covariance: Covariance6x6 | None = None

    def __post_init__(self) -> None:
        """Validate frame semantics, units, orientation and covariance.

        Raises:
            ValueError: If an identity or frame is empty, the frames are
                identical, ``translation_m`` is not finite, ``orientation``
                is not a finite unit quaternion, or ``covariance`` is not a
                finite 6x6 matrix.
        """
        if not self.estimate_id:
            raise ValueError("estimate_id must not be empty")
        if not self.parent_frame or not self.child_frame:
            raise ValueError("parent_frame and child_frame must not be empty")
        if self.parent_frame == self.child_frame:
            raise ValueError(
                f"parent_frame and child_frame must differ, got {self.parent_frame!r} for both"
            )
        if not all(math.isfinite(value) for value in self.translation_m):
            raise ValueError(f"translation_m must be finite, got {self.translation_m!r}")
        if not is_unit_quaternion(self.orientation):
            raise ValueError(
                f"orientation must be a finite unit quaternion (x, y, z, w), "
                f"got {self.orientation!r}"
            )
        if self.covariance is not None:
            if len(self.covariance) != _COVARIANCE_SIZE:
                raise ValueError(
                    f"covariance must contain exactly {_COVARIANCE_SIZE} values (row-major 6x6), "
                    f"got {len(self.covariance)}"
                )
            if not all(math.isfinite(value) for value in self.covariance):
                raise ValueError("covariance values must be finite")


@dataclass(frozen=True, kw_only=True)
class TimeBounds:
    """Closed time interval within one clock domain.

    Attributes:
        start: First instant covered.
        end: Last instant covered; not before ``start``.
    """

    start: SourceTimestamp
    end: SourceTimestamp

    def __post_init__(self) -> None:
        """Validate that both ends share a clock and are ordered.

        Raises:
            ValueError: If the clock domains differ or ``end`` precedes ``start``.
        """
        if self.start.clock_id != self.end.clock_id:
            raise ValueError(
                f"time bounds must share one clock domain, got {self.start.clock_id!r} "
                f"and {self.end.clock_id!r}"
            )
        if self.duration_ns < 0:
            raise ValueError("time bounds end must not precede start")

    @property
    def duration_ns(self) -> int:
        """Return the exact interval length in nanoseconds."""
        return self.end.total_nanoseconds() - self.start.total_nanoseconds()


@dataclass(frozen=True, kw_only=True)
class TrajectoryGap:
    """Interval between two consecutive poses across which interpolation is invalid.

    A backend records a gap when its configured tolerance says the trajectory
    cannot be trusted between the two poses. The record is persisted with the
    trajectory so lookups stay reproducible without the backend configuration.

    Attributes:
        previous_estimate_id: Pose at the start of the gap.
        next_estimate_id: The next pose in the trajectory.
        duration_ns: Exact time between the two poses, in nanoseconds.
    """

    previous_estimate_id: PoseEstimateId
    next_estimate_id: PoseEstimateId
    duration_ns: int

    def __post_init__(self) -> None:
        """Validate the gap has a positive duration.

        Raises:
            ValueError: If ``duration_ns`` is not positive.
        """
        if self.duration_ns <= 0:
            raise ValueError(f"gap duration must be positive, got {self.duration_ns} ns")


@dataclass(frozen=True, kw_only=True)
class EstimatorProvenance:
    """Identity of the backend and configuration that produced a trajectory.

    Attributes:
        backend_id: Stable adapter identity, e.g. ``"external_pose"``.
        backend_version: Backend version or source ref.
        configuration_fingerprint: Deterministic hash of the effective backend
            configuration; ``None`` when there is nothing configurable.
    """

    backend_id: str
    backend_version: str
    configuration_fingerprint: str | None = None


@dataclass(frozen=True, kw_only=True)
class TrajectoryProvenance:
    """Run-level traceability shared by every pose of a trajectory.

    Attributes:
        estimator: Backend and configuration identity.
        sequence_artifact_id: Canonical sequence the estimator consumed.
        selection_id: Deterministic identity of the sequence selection.
        calibration_identity: Hash of the static calibration the estimator
            used; ``None`` when the backend needed none.
        code_version: Code revision that produced the trajectory, when known.
    """

    estimator: EstimatorProvenance
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    calibration_identity: str | None = None
    code_version: str | None = None


@dataclass(frozen=True, kw_only=True)
class TrajectoryQualitySummary:
    """Measurable sampling and flagging summary of a trajectory.

    Attributes:
        pose_count: Number of poses.
        duration_ns: Time from first to last pose.
        min_interval_ns: Smallest interval between consecutive poses, or
            ``None`` for a single pose.
        median_interval_ns: Lower median interval, or ``None``.
        max_interval_ns: Largest interval, or ``None``.
        gap_count: Number of recorded :class:`TrajectoryGap` entries.
        degraded_pose_count: Poses flagged :attr:`PoseValidity.DEGRADED`.
        pose_with_covariance_count: Poses carrying a covariance.
    """

    pose_count: int
    duration_ns: int
    min_interval_ns: int | None
    median_interval_ns: int | None
    max_interval_ns: int | None
    gap_count: int
    degraded_pose_count: int
    pose_with_covariance_count: int


@dataclass(frozen=True, kw_only=True)
class Trajectory:
    """Time-ordered sequence of poses of one body frame in one reference frame.

    Attributes:
        trajectory_id: Identity of this trajectory within its artifact.
        reference_frame: Frame every pose is expressed in, e.g. ``"map"``.
        body_frame: Frame every pose describes, e.g. ``"body"``.
        poses: Poses with strictly increasing timestamps in one clock domain.
        gaps: Intervals across which interpolation must not be trusted.
        provenance: Run-level traceability.
    """

    trajectory_id: TrajectoryId
    reference_frame: FrameId
    body_frame: FrameId
    poses: tuple[PoseEstimate, ...]
    gaps: tuple[TrajectoryGap, ...]
    provenance: TrajectoryProvenance

    def __post_init__(self) -> None:
        """Validate frame semantics, clock domain, ordering and gap records.

        Raises:
            ValueError: If there are no poses, a pose declares other frames,
                the clock domains are mixed, timestamps are not strictly
                increasing, an ``estimate_id`` repeats, or a gap disagrees
                with the poses it references.
        """
        if not self.poses:
            raise ValueError("trajectory must contain at least one pose")

        clock_id = self.poses[0].timestamp.clock_id
        index_by_id: dict[PoseEstimateId, int] = {}
        previous_ns: int | None = None
        for index, pose in enumerate(self.poses):
            if pose.parent_frame != self.reference_frame or pose.child_frame != self.body_frame:
                raise ValueError(
                    f"pose {pose.estimate_id!r} declares frames {pose.parent_frame!r} -> "
                    f"{pose.child_frame!r}, but the trajectory is {self.reference_frame!r} -> "
                    f"{self.body_frame!r}"
                )
            if pose.timestamp.clock_id != clock_id:
                raise ValueError(
                    f"all poses must share one clock domain, got {clock_id!r} and "
                    f"{pose.timestamp.clock_id!r}"
                )
            if pose.estimate_id in index_by_id:
                raise ValueError(f"duplicate estimate_id in trajectory: {pose.estimate_id!r}")
            when_ns = pose.timestamp.total_nanoseconds()
            if previous_ns is not None and when_ns <= previous_ns:
                raise ValueError(
                    f"pose timestamps must be strictly increasing, but {pose.estimate_id!r} "
                    f"is not later than its predecessor"
                )
            index_by_id[pose.estimate_id] = index
            previous_ns = when_ns

        for gap in self.gaps:
            previous_index = index_by_id.get(gap.previous_estimate_id)
            next_index = index_by_id.get(gap.next_estimate_id)
            if previous_index is None or next_index is None:
                raise ValueError(
                    f"gap references an unknown pose: {gap.previous_estimate_id!r} -> "
                    f"{gap.next_estimate_id!r}"
                )
            if next_index != previous_index + 1:
                raise ValueError(
                    f"gap must span consecutive poses, got {gap.previous_estimate_id!r} -> "
                    f"{gap.next_estimate_id!r}"
                )
            actual_ns = (
                self.poses[next_index].timestamp.total_nanoseconds()
                - self.poses[previous_index].timestamp.total_nanoseconds()
            )
            if gap.duration_ns != actual_ns:
                raise ValueError(
                    f"gap duration {gap.duration_ns} ns disagrees with the pose timestamps "
                    f"({actual_ns} ns)"
                )

    @property
    def time_bounds(self) -> TimeBounds:
        """Return the interval from the first to the last pose."""
        return TimeBounds(start=self.poses[0].timestamp, end=self.poses[-1].timestamp)

    def quality_summary(self) -> TrajectoryQualitySummary:
        """Summarize sampling and flagged poses.

        Returns:
            Counts and interval statistics computed from the poses alone.
        """
        times_ns = [pose.timestamp.total_nanoseconds() for pose in self.poses]
        intervals = [later - earlier for earlier, later in pairwise(times_ns)]
        return TrajectoryQualitySummary(
            pose_count=len(self.poses),
            duration_ns=times_ns[-1] - times_ns[0],
            min_interval_ns=min(intervals) if intervals else None,
            median_interval_ns=median_low(intervals) if intervals else None,
            max_interval_ns=max(intervals) if intervals else None,
            gap_count=len(self.gaps),
            degraded_pose_count=sum(
                1 for pose in self.poses if pose.validity is PoseValidity.DEGRADED
            ),
            pose_with_covariance_count=sum(1 for pose in self.poses if pose.covariance is not None),
        )
