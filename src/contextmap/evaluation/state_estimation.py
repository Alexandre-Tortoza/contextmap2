"""Deterministic evaluation reports for State Estimation.

A trajectory is not accepted merely because it exists: it must be numerically
valid, temporally coherent, frame-consistent and, when a trusted reference
exists, measurably close to it. This module builds one report schema for every
backend, so an external-pose baseline and an estimator such as FAST-LIO are
compared under the same protocol while each keeps its own identity and
configuration.

Nothing here alters a pipeline output. Quality and execution cost stay in
separate report sections, motion anomalies are only reported against thresholds
a reference profile supplies (none are hardcoded), and a reference trajectory
is used only when it was explicitly declared an evaluation reference: an
arbitrary pose stream is never called ground truth. The alignment applied
before measuring absolute error is always stated in the report.

NumPy is imported only inside the rigid alignment, so importing
:mod:`contextmap.evaluation` adds no base runtime dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from contextmap.ingestion import FrameId, SequenceArtifactId
from contextmap.shared import (
    Quaternion,
    Vector3,
    compose_rigid,
    invert_rigid,
    quaternion_angle_between,
    quaternion_multiply,
    quaternion_norm,
    rotate_vector,
)
from contextmap.state_estimation import (
    ClockDomainMismatchError,
    DistributionSummary,
    EstimatorProvenance,
    LookupPolicy,
    MotionSummary,
    PoseEstimate,
    PoseEstimateId,
    PoseLookupResult,
    ResolvedPose,
    ResolvedTransform,
    StateEstimationResult,
    StateEstimationRunId,
    TemporalAlignmentSummary,
    Trajectory,
    TrajectoryId,
    TrajectoryLookup,
    motion_deltas,
    summarize_distribution,
    summarize_lookups,
    summarize_motion,
)

EVALUATOR_VERSION = "1"
"""Bumped whenever a metric's definition changes, so reports stay comparable."""

_IDENTITY_ROTATION: Quaternion = (0.0, 0.0, 0.0, 1.0)
_MIN_ALIGNMENT_PAIRS = 3


class StateEstimationEvaluationError(ValueError):
    """Raised when an evaluation invariant cannot be satisfied."""


class ReferenceRole(Enum):
    """What a trajectory was declared to be by the reference profile.

    Attributes:
        EVALUATION_REFERENCE: Explicitly declared suitable to evaluate against.
        INPUT_MEASUREMENT: A pose stream used as an estimator input; it is not
            reference data, whatever its file is called.
    """

    EVALUATION_REFERENCE = "evaluation_reference"
    INPUT_MEASUREMENT = "input_measurement"


@dataclass(frozen=True, kw_only=True)
class ReferenceTrajectory:
    """A trajectory together with the role and identity its profile gave it.

    Attributes:
        trajectory: The candidate reference.
        reference_id: Identity of the reference profile that declares its role,
            e.g. ``"reference-profile:corridor@1"``.
        role: The declared role; only ``EVALUATION_REFERENCE`` can be compared against.
    """

    trajectory: Trajectory
    reference_id: str
    role: ReferenceRole

    def __post_init__(self) -> None:
        """Require the profile identity.

        Raises:
            ValueError: If ``reference_id`` is empty.
        """
        if not self.reference_id:
            raise ValueError("reference_id must not be empty")


class AlignmentMethod(Enum):
    """How the estimated trajectory is aligned to the reference before measuring.

    Attributes:
        NONE: No alignment; the error includes any frame offset.
        SE3: Rigid alignment (rotation and translation, no scale) that minimizes
            the position error. It hides a constant frame offset by design, and
            with collinear positions the rotation about the line is undetermined.
    """

    NONE = "none"
    SE3 = "se3"


@dataclass(frozen=True, kw_only=True)
class ReferenceComparisonConfig:
    """Explicit protocol of a comparison against a reference.

    Attributes:
        alignment: How to align before measuring absolute error.
        max_time_difference_ns: Largest distance between an estimated pose and the
            reference pose associated with it; farther poses stay unmatched.
        relative_pair_offset: When set, also measure relative pose error between
            matched pairs ``i`` and ``i + relative_pair_offset``.
    """

    alignment: AlignmentMethod
    max_time_difference_ns: int
    relative_pair_offset: int | None = None

    def __post_init__(self) -> None:
        """Validate the protocol.

        Raises:
            ValueError: If a tolerance or offset is not positive.
        """
        if self.max_time_difference_ns < 0:
            raise ValueError("max_time_difference_ns must not be negative")
        if self.relative_pair_offset is not None and self.relative_pair_offset < 1:
            raise ValueError("relative_pair_offset must be at least 1 when set")


@dataclass(frozen=True, kw_only=True)
class MotionThresholds:
    """Limits a reference profile supplies for motion anomaly detection.

    Every limit is optional; an unset one is never applied. Nothing here has a
    default, so no threshold is inherited from one corridor or dataset.

    Attributes:
        max_translation_delta_m: Largest position change between consecutive poses.
        max_orientation_delta_rad: Largest rotation between consecutive poses.
        max_linear_speed_mps: Largest derived linear speed.
        max_angular_speed_radps: Largest derived angular speed.
    """

    max_translation_delta_m: float | None = None
    max_orientation_delta_rad: float | None = None
    max_linear_speed_mps: float | None = None
    max_angular_speed_radps: float | None = None

    def __post_init__(self) -> None:
        """Validate that every set limit is positive.

        Raises:
            ValueError: If a limit is not positive.
        """
        for name, value in (
            ("max_translation_delta_m", self.max_translation_delta_m),
            ("max_orientation_delta_rad", self.max_orientation_delta_rad),
            ("max_linear_speed_mps", self.max_linear_speed_mps),
            ("max_angular_speed_radps", self.max_angular_speed_radps),
        ):
            if value is not None and not value > 0:
                raise ValueError(f"{name} must be positive when set")


@dataclass(frozen=True, kw_only=True)
class StructuralReport:
    """Structure of the trajectory, re-verified from its numbers.

    Attributes:
        pose_count: Number of poses.
        reference_frame: Frame every pose is expressed in.
        body_frame: Frame every pose describes.
        clock_id: Clock domain of every timestamp.
        all_values_finite: Whether every translation and orientation is finite.
        max_quaternion_norm_error: Largest ``|norm - 1|`` over the orientations.
        min_interval_ns: Smallest interval between consecutive poses, or ``None``.
        gap_count: Recorded trajectory gaps.
        degraded_pose_count: Poses the backend flagged as degraded.
        consumed_observation_count: Observations the backend considered.
        rejected_observation_count: Observations the backend rejected.
    """

    pose_count: int
    reference_frame: FrameId
    body_frame: FrameId
    clock_id: str
    all_values_finite: bool
    max_quaternion_norm_error: float
    min_interval_ns: int | None
    gap_count: int
    degraded_pose_count: int
    consumed_observation_count: int
    rejected_observation_count: int


@dataclass(frozen=True, kw_only=True)
class MotionAnomaly:
    """One interval that exceeds a threshold the profile supplied.

    Attributes:
        kind: ``"translation_delta_m"``, ``"orientation_delta_rad"``,
            ``"linear_speed_mps"`` or ``"angular_speed_radps"``.
        previous_estimate_id: Earlier pose of the interval.
        next_estimate_id: Later pose of the interval.
        value: Measured value.
        threshold: Limit it exceeded.
    """

    kind: str
    previous_estimate_id: PoseEstimateId
    next_estimate_id: PoseEstimateId
    value: float
    threshold: float


@dataclass(frozen=True, kw_only=True)
class MotionReport:
    """Per-interval motion and the anomalies found against supplied thresholds.

    Attributes:
        summary: Distributions of translation, rotation and speeds.
        interval_ns: Distribution of the sample interval in nanoseconds, or ``None``.
        thresholds: The thresholds applied, or ``None`` when none were supplied.
        anomalies: Intervals exceeding a supplied threshold.
    """

    summary: MotionSummary
    interval_ns: DistributionSummary | None
    thresholds: MotionThresholds | None
    anomalies: tuple[MotionAnomaly, ...]


@dataclass(frozen=True, kw_only=True)
class TransformTraceRecord:
    """One reconstructable chain ``T_reference_sensor(t) = T_reference_body(t) * T_body_sensor``.

    Attributes:
        pose_estimate_id: Pose whose ``T_reference_body(t)`` was used.
        timestamp_ns: Time of that pose.
        reference_frame: Frame the composed transform is expressed in.
        sensor_frame: Frame whose coordinates the composed transform maps.
        sensor_translation_m: Translation of ``T_reference_sensor(t)``.
        sensor_rotation: Rotation of ``T_reference_sensor(t)``.
        composition_error_m: Position left by ``T_reference_sensor * T_sensor_body``
            compared with ``T_reference_body``.
        composition_error_rad: Rotation left by the same comparison.
        round_trip_error_m: Position left by ``T * T^-1``.
        round_trip_error_rad: Rotation left by ``T * T^-1``.
    """

    pose_estimate_id: PoseEstimateId
    timestamp_ns: int
    reference_frame: FrameId
    sensor_frame: FrameId
    sensor_translation_m: Vector3
    sensor_rotation: Quaternion
    composition_error_m: float
    composition_error_rad: float
    round_trip_error_m: float
    round_trip_error_rad: float


@dataclass(frozen=True, kw_only=True)
class TransformTraceReport:
    """Sampled transform chains and the worst numerical inconsistency among them.

    Attributes:
        records: The sampled chains, in time order.
        max_composition_error_m: Worst composition position error.
        max_composition_error_rad: Worst composition rotation error.
        max_round_trip_error_m: Worst inverse round-trip position error.
        max_round_trip_error_rad: Worst inverse round-trip rotation error.
    """

    records: tuple[TransformTraceRecord, ...]
    max_composition_error_m: float
    max_composition_error_rad: float
    max_round_trip_error_m: float
    max_round_trip_error_rad: float


@dataclass(frozen=True, kw_only=True)
class ErrorStatistics:
    """Statistics of a set of non-negative errors.

    Attributes:
        count: Number of errors.
        rmse: Root mean square.
        mean: Arithmetic mean.
        median: Median.
        p95: 95th percentile by nearest rank.
        maximum: Largest error.
    """

    count: int
    rmse: float
    mean: float
    median: float
    p95: float
    maximum: float


@dataclass(frozen=True, kw_only=True)
class AssociationReport:
    """How estimated poses were paired with reference poses.

    Attributes:
        matched_count: Estimated poses with a reference pose within the tolerance.
        unmatched_count: Estimated poses left out because none was close enough.
        max_time_difference_ns: Largest time difference among the matches.
    """

    matched_count: int
    unmatched_count: int
    max_time_difference_ns: int


@dataclass(frozen=True, kw_only=True)
class AlignmentReport:
    """The alignment applied before measuring absolute error.

    Attributes:
        method: The method used.
        translation_m: Translation of the transform that maps the estimated frame
            into the reference frame, or ``None`` when no alignment was applied.
        rotation: Rotation of that transform, or ``None``.
    """

    method: AlignmentMethod
    translation_m: Vector3 | None
    rotation: Quaternion | None


@dataclass(frozen=True, kw_only=True)
class RelativeErrorReport:
    """Relative pose error between matched pairs, which alignment does not affect.

    Attributes:
        pair_offset: Index distance between the two poses of each pair.
        pair_count: Number of pairs measured.
        translation_m: Position error of the relative motion, in meters.
        rotation_rad: Rotation error of the relative motion, in radians.
    """

    pair_offset: int
    pair_count: int
    translation_m: ErrorStatistics
    rotation_rad: ErrorStatistics


@dataclass(frozen=True, kw_only=True)
class AccuracyReport:
    """Measured closeness to a trusted reference.

    Attributes:
        reference_id: Identity of the reference profile.
        reference_role: The role that profile declared.
        reference_frame: Reference frame of the reference trajectory.
        reference_body_frame: Body frame of the reference trajectory.
        config: The comparison protocol.
        association: How poses were paired.
        alignment: The alignment applied.
        ate_translation_m: Absolute trajectory error of the position, in meters.
        rotation_error_rad: Absolute orientation error, in radians.
        rpe: Relative pose error, when requested.
    """

    reference_id: str
    reference_role: ReferenceRole
    reference_frame: FrameId
    reference_body_frame: FrameId
    config: ReferenceComparisonConfig
    association: AssociationReport
    alignment: AlignmentReport
    ate_translation_m: ErrorStatistics
    rotation_error_rad: ErrorStatistics
    rpe: RelativeErrorReport | None


@dataclass(frozen=True, kw_only=True)
class CostReport:
    """Execution cost, kept apart from every quality measure.

    Attributes:
        pose_count: Poses produced.
        runtime_s: Wall-clock time the backend took, when measured.
    """

    pose_count: int
    runtime_s: float | None


@dataclass(frozen=True, kw_only=True)
class StateEstimationEvaluationReport:
    """Common evaluation report of one trajectory.

    Attributes:
        evaluator_version: Version of the metric definitions.
        sequence_artifact_id: Canonical sequence the trajectory was estimated from.
        selection_id: Deterministic identity of the selection.
        run_id: Run that persisted the trajectory, when there is one.
        trajectory_id: Identity of the evaluated trajectory.
        estimator: Backend and configuration identity.
        calibration_identity: Hash of the calibration the estimator used.
        structural: Re-verified structure.
        motion: Motion distributions and anomalies.
        temporal: Temporal-alignment metrics of the supplied lookups, or ``None``.
        transform_trace: Sampled transform chains, or ``None``.
        accuracy: Comparison against a trusted reference, or ``None``.
        cost: Execution cost.
    """

    evaluator_version: str
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    run_id: StateEstimationRunId | None
    trajectory_id: TrajectoryId
    estimator: EstimatorProvenance
    calibration_identity: str | None
    structural: StructuralReport
    motion: MotionReport
    temporal: TemporalAlignmentSummary | None
    transform_trace: TransformTraceReport | None
    accuracy: AccuracyReport | None
    cost: CostReport


def evaluate_state_estimation(
    *,
    trajectory: Trajectory,
    run_id: StateEstimationRunId | None = None,
    result: StateEstimationResult | None = None,
    thresholds: MotionThresholds | None = None,
    body_to_sensor: ResolvedTransform | None = None,
    trace_max_records: int = 20,
    lookups: Sequence[PoseLookupResult] = (),
    reference: ReferenceTrajectory | None = None,
    reference_config: ReferenceComparisonConfig | None = None,
    runtime_s: float | None = None,
) -> StateEstimationEvaluationReport:
    """Evaluate one trajectory under the common protocol.

    Args:
        trajectory: The trajectory to evaluate.
        run_id: Run that persisted it, recorded for reproducibility.
        result: The backend result, for the consumed/rejected observation counts.
        thresholds: Motion limits from the reference profile; none are applied
            when omitted.
        body_to_sensor: ``T_body_sensor`` to trace ``T_reference_sensor(t)`` with.
        trace_max_records: Most poses to include in the transform trace.
        lookups: Pose lookups made for the run, for temporal-alignment metrics.
        reference: A trusted reference to compare against.
        reference_config: The comparison protocol; required with ``reference``.
        runtime_s: Wall-clock time the backend took, reported apart from quality.

    Returns:
        The report.

    Raises:
        StateEstimationEvaluationError: If a reference is not declared for
            evaluation, its protocol is missing, its clock domain differs, no pose
            can be associated, or alignment is not defined.
    """
    accuracy = None
    if reference is not None:
        if reference_config is None:
            raise StateEstimationEvaluationError(
                "a reference requires an explicit reference_config"
            )
        accuracy = _compare_with_reference(trajectory, reference, reference_config)
    elif reference_config is not None:
        raise StateEstimationEvaluationError("reference_config was given without a reference")

    provenance = trajectory.provenance
    return StateEstimationEvaluationReport(
        evaluator_version=EVALUATOR_VERSION,
        sequence_artifact_id=provenance.sequence_artifact_id,
        selection_id=provenance.selection_id,
        run_id=run_id,
        trajectory_id=trajectory.trajectory_id,
        estimator=provenance.estimator,
        calibration_identity=provenance.calibration_identity,
        structural=_structural_report(trajectory, result),
        motion=_motion_report(trajectory, thresholds),
        temporal=summarize_lookups(lookups) if lookups else None,
        transform_trace=(
            trace_transform_chain(
                trajectory, body_to_sensor=body_to_sensor, max_records=trace_max_records
            )
            if body_to_sensor is not None
            else None
        ),
        accuracy=accuracy,
        cost=CostReport(pose_count=len(trajectory.poses), runtime_s=runtime_s),
    )


def trace_transform_chain(
    trajectory: Trajectory, *, body_to_sensor: ResolvedTransform, max_records: int = 20
) -> TransformTraceReport:
    """Reconstruct the reference-to-sensor transform chain for sampled poses.

    The chain is ``T_reference_sensor(t) = T_reference_body(t) * T_body_sensor``. For
    each sampled pose it is composed, then checked numerically: composing
    it with ``T_sensor_body`` must give back ``T_reference_body(t)``, and the composed
    transform times its inverse must be the identity.

    Args:
        trajectory: Provides ``T_reference_body(t)``.
        body_to_sensor: The static transform from the body to the sensor.
        max_records: Most poses to sample, spread evenly from first to last.

    Returns:
        The sampled chains and the worst inconsistency among them.

    Raises:
        StateEstimationEvaluationError: If ``body_to_sensor`` is not expressed in the
            trajectory's body frame or ``max_records`` is not positive.
    """
    if body_to_sensor.parent_frame != trajectory.body_frame:
        raise StateEstimationEvaluationError(
            f"T_body_sensor has parent frame {body_to_sensor.parent_frame!r} but the trajectory "
            f"body frame is {trajectory.body_frame!r}"
        )
    if max_records < 1:
        raise StateEstimationEvaluationError("max_records must be at least 1")

    inverse_translation, inverse_rotation = invert_rigid(
        translation=body_to_sensor.translation, rotation=body_to_sensor.rotation
    )
    records: list[TransformTraceRecord] = []
    for index in _sample_indexes(len(trajectory.poses), max_records):
        pose = trajectory.poses[index]
        sensor_translation, sensor_rotation = compose_rigid(
            outer_translation=pose.translation_m,
            outer_rotation=pose.orientation,
            inner_translation=body_to_sensor.translation,
            inner_rotation=body_to_sensor.rotation,
        )
        back_translation, back_rotation = compose_rigid(
            outer_translation=sensor_translation,
            outer_rotation=sensor_rotation,
            inner_translation=inverse_translation,
            inner_rotation=inverse_rotation,
        )
        sensor_inverse_translation, sensor_inverse_rotation = invert_rigid(
            translation=sensor_translation, rotation=sensor_rotation
        )
        round_trip_translation, round_trip_rotation = compose_rigid(
            outer_translation=sensor_translation,
            outer_rotation=sensor_rotation,
            inner_translation=sensor_inverse_translation,
            inner_rotation=sensor_inverse_rotation,
        )
        records.append(
            TransformTraceRecord(
                pose_estimate_id=pose.estimate_id,
                timestamp_ns=pose.timestamp.total_nanoseconds(),
                reference_frame=trajectory.reference_frame,
                sensor_frame=body_to_sensor.child_frame,
                sensor_translation_m=sensor_translation,
                sensor_rotation=sensor_rotation,
                composition_error_m=math.dist(back_translation, pose.translation_m),
                composition_error_rad=quaternion_angle_between(back_rotation, pose.orientation),
                round_trip_error_m=math.hypot(*round_trip_translation),
                round_trip_error_rad=quaternion_angle_between(
                    round_trip_rotation, _IDENTITY_ROTATION
                ),
            )
        )
    return TransformTraceReport(
        records=tuple(records),
        max_composition_error_m=max(r.composition_error_m for r in records),
        max_composition_error_rad=max(r.composition_error_rad for r in records),
        max_round_trip_error_m=max(r.round_trip_error_m for r in records),
        max_round_trip_error_rad=max(r.round_trip_error_rad for r in records),
    )


def _sample_indexes(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


def _structural_report(
    trajectory: Trajectory, result: StateEstimationResult | None
) -> StructuralReport:
    poses = trajectory.poses
    finite = all(
        math.isfinite(value) for pose in poses for value in (*pose.translation_m, *pose.orientation)
    )
    norm_error = max(abs(quaternion_norm(pose.orientation) - 1.0) for pose in poses)
    quality = trajectory.quality_summary()
    return StructuralReport(
        pose_count=len(poses),
        reference_frame=trajectory.reference_frame,
        body_frame=trajectory.body_frame,
        clock_id=poses[0].timestamp.clock_id,
        all_values_finite=finite,
        max_quaternion_norm_error=norm_error,
        min_interval_ns=quality.min_interval_ns,
        gap_count=quality.gap_count,
        degraded_pose_count=quality.degraded_pose_count,
        consumed_observation_count=result.consumed_observation_count if result else 0,
        rejected_observation_count=result.rejected_observation_count if result else 0,
    )


def _motion_report(trajectory: Trajectory, thresholds: MotionThresholds | None) -> MotionReport:
    deltas = motion_deltas(trajectory)
    anomalies: list[MotionAnomaly] = []
    if thresholds is not None:
        for delta in deltas:
            for kind, value, limit in (
                (
                    "translation_delta_m",
                    delta.translation_delta_m,
                    thresholds.max_translation_delta_m,
                ),
                (
                    "orientation_delta_rad",
                    delta.orientation_delta_rad,
                    thresholds.max_orientation_delta_rad,
                ),
                ("linear_speed_mps", delta.linear_speed_mps, thresholds.max_linear_speed_mps),
                (
                    "angular_speed_radps",
                    delta.angular_speed_radps,
                    thresholds.max_angular_speed_radps,
                ),
            ):
                if limit is not None and value > limit:
                    anomalies.append(
                        MotionAnomaly(
                            kind=kind,
                            previous_estimate_id=delta.previous_estimate_id,
                            next_estimate_id=delta.next_estimate_id,
                            value=value,
                            threshold=limit,
                        )
                    )
    return MotionReport(
        summary=summarize_motion(trajectory),
        interval_ns=summarize_distribution([float(delta.interval_ns) for delta in deltas]),
        thresholds=thresholds,
        anomalies=tuple(anomalies),
    )


def _compare_with_reference(
    trajectory: Trajectory, reference: ReferenceTrajectory, config: ReferenceComparisonConfig
) -> AccuracyReport:
    if reference.role is not ReferenceRole.EVALUATION_REFERENCE:
        raise StateEstimationEvaluationError(
            f"{reference.reference_id!r} is declared {reference.role.value!r}, not an "
            "evaluation reference; a pose stream is never treated as ground truth by its name"
        )
    lookup = TrajectoryLookup(reference.trajectory)
    policy = LookupPolicy.nearest(max_time_delta_ns=config.max_time_difference_ns)
    pairs: list[tuple[PoseEstimate, PoseEstimate]] = []
    worst_difference_ns = 0
    for pose in trajectory.poses:
        try:
            outcome = lookup.pose_at(pose.timestamp, policy=policy)
        except ClockDomainMismatchError as error:
            raise StateEstimationEvaluationError(
                f"the estimated trajectory and the reference use different clock domains; "
                f"they are never compared implicitly ({error})"
            ) from error
        if isinstance(outcome, ResolvedPose):
            pairs.append((pose, outcome.pose))
            worst_difference_ns = max(worst_difference_ns, outcome.time_delta_ns)
    if not pairs:
        raise StateEstimationEvaluationError(
            "no pose could be associated with the reference within "
            f"max_time_difference_ns={config.max_time_difference_ns}"
        )

    alignment_rotation: Quaternion | None = None
    alignment_translation: Vector3 | None = None
    if config.alignment is AlignmentMethod.SE3:
        if len(pairs) < _MIN_ALIGNMENT_PAIRS:
            raise StateEstimationEvaluationError(
                f"rigid alignment needs at least three associated poses, got {len(pairs)}"
            )
        alignment_rotation, alignment_translation = _rigid_alignment(
            [estimated.translation_m for estimated, _ in pairs],
            [ref.translation_m for _, ref in pairs],
        )

    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for estimated, ref in pairs:
        position = estimated.translation_m
        orientation = estimated.orientation
        if alignment_rotation is not None and alignment_translation is not None:
            rotated = rotate_vector(alignment_rotation, position)
            position = (
                rotated[0] + alignment_translation[0],
                rotated[1] + alignment_translation[1],
                rotated[2] + alignment_translation[2],
            )
            orientation = quaternion_multiply(alignment_rotation, orientation)
        translation_errors.append(math.dist(position, ref.translation_m))
        rotation_errors.append(quaternion_angle_between(orientation, ref.orientation))

    return AccuracyReport(
        reference_id=reference.reference_id,
        reference_role=reference.role,
        reference_frame=reference.trajectory.reference_frame,
        reference_body_frame=reference.trajectory.body_frame,
        config=config,
        association=AssociationReport(
            matched_count=len(pairs),
            unmatched_count=len(trajectory.poses) - len(pairs),
            max_time_difference_ns=worst_difference_ns,
        ),
        alignment=AlignmentReport(
            method=config.alignment,
            translation_m=alignment_translation,
            rotation=alignment_rotation,
        ),
        ate_translation_m=_error_statistics(translation_errors),
        rotation_error_rad=_error_statistics(rotation_errors),
        rpe=(
            _relative_error(pairs, config.relative_pair_offset)
            if config.relative_pair_offset is not None
            else None
        ),
    )


def _relative_error(
    pairs: list[tuple[PoseEstimate, PoseEstimate]], offset: int
) -> RelativeErrorReport | None:
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for (first_e, first_r), (second_e, second_r) in zip(pairs, pairs[offset:], strict=False):
        estimated_step = _relative_motion(first_e, second_e)
        reference_step = _relative_motion(first_r, second_r)
        inverse_translation, inverse_rotation = invert_rigid(
            translation=reference_step[0], rotation=reference_step[1]
        )
        error_translation, error_rotation = compose_rigid(
            outer_translation=inverse_translation,
            outer_rotation=inverse_rotation,
            inner_translation=estimated_step[0],
            inner_rotation=estimated_step[1],
        )
        translation_errors.append(math.hypot(*error_translation))
        rotation_errors.append(quaternion_angle_between(error_rotation, _IDENTITY_ROTATION))
    if not translation_errors:
        return None
    return RelativeErrorReport(
        pair_offset=offset,
        pair_count=len(translation_errors),
        translation_m=_error_statistics(translation_errors),
        rotation_rad=_error_statistics(rotation_errors),
    )


def _relative_motion(first: PoseEstimate, second: PoseEstimate) -> tuple[Vector3, Quaternion]:
    """``T_first_second = inverse(T_ref_first) * T_ref_second``."""
    inverse_translation, inverse_rotation = invert_rigid(
        translation=first.translation_m, rotation=first.orientation
    )
    return compose_rigid(
        outer_translation=inverse_translation,
        outer_rotation=inverse_rotation,
        inner_translation=second.translation_m,
        inner_rotation=second.orientation,
    )


def _error_statistics(errors: Sequence[float]) -> ErrorStatistics:
    summary = summarize_distribution(errors)
    assert summary is not None  # os chamadores garantem ao menos um erro
    return ErrorStatistics(
        count=summary.count,
        rmse=math.sqrt(sum(value * value for value in errors) / len(errors)),
        mean=sum(errors) / len(errors),
        median=summary.median,
        p95=summary.p95,
        maximum=summary.maximum,
    )


def _rigid_alignment(
    estimated: list[Vector3], reference: list[Vector3]
) -> tuple[Quaternion, Vector3]:
    """Rotation and translation minimizing the position error (Horn/Umeyama, no scale)."""
    import numpy as np

    est = np.asarray(estimated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    est_mean = est.mean(axis=0)
    ref_mean = ref.mean(axis=0)
    covariance = (est - est_mean).T @ (ref - ref_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0])
    rotation = vt.T @ correction @ u.T
    translation = ref_mean - rotation @ est_mean
    matrix = [[float(value) for value in row] for row in rotation]
    return _quaternion_from_matrix(matrix), (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )


def _quaternion_from_matrix(m: list[list[float]]) -> Quaternion:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)


@dataclass(frozen=True, kw_only=True)
class StateEstimationComparisonEntry:
    """One backend's headline measures inside a controlled comparison.

    Attributes:
        estimator: The backend's identity.
        configuration_fingerprint: The backend's configuration fingerprint.
        run_id: The run that persisted its trajectory, when there is one.
        pose_count: Poses produced.
        ate_rmse_m: Absolute trajectory error RMSE, in meters.
        rotation_error_max_rad: Largest absolute orientation error.
        rpe_translation_rmse_m: Relative translation error RMSE, when requested.
        gap_count: Recorded trajectory gaps.
        runtime_s: Wall-clock time, kept apart from the quality measures.
    """

    estimator: EstimatorProvenance
    configuration_fingerprint: str | None
    run_id: StateEstimationRunId | None
    pose_count: int
    ate_rmse_m: float | None
    rotation_error_max_rad: float | None
    rpe_translation_rmse_m: float | None
    gap_count: int
    runtime_s: float | None


@dataclass(frozen=True, kw_only=True)
class StateEstimationComparison:
    """Reports of different backends over the same sequence, selection and reference.

    Attributes:
        sequence_artifact_id: The shared sequence.
        selection_id: The shared selection.
        calibration_identity: The shared calibration.
        reference_id: The shared reference profile.
        comparison_config: The shared comparison protocol.
        entries: One entry per backend, in the order given.
    """

    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    calibration_identity: str | None
    reference_id: str
    comparison_config: ReferenceComparisonConfig
    entries: tuple[StateEstimationComparisonEntry, ...]


def compare_state_estimation_reports(
    reports: Sequence[StateEstimationEvaluationReport],
) -> StateEstimationComparison:
    """Compare backends under one protocol, rejecting anything but the backend changing.

    Args:
        reports: One report per backend, each with a reference comparison.

    Returns:
        The comparison, preserving each backend's identity and configuration.

    Raises:
        StateEstimationEvaluationError: If there are fewer than two reports, a report has
            no reference comparison, or the sequence, selection, calibration, reference or
            comparison protocol differs.
    """
    if len(reports) < 2:
        raise StateEstimationEvaluationError("a comparison needs at least two reports")
    first = reports[0]
    if first.accuracy is None:
        raise StateEstimationEvaluationError("every compared report needs a reference comparison")
    for report in reports[1:]:
        if report.sequence_artifact_id != first.sequence_artifact_id:
            raise StateEstimationEvaluationError("reports use different sequence artifacts")
        if report.selection_id != first.selection_id:
            raise StateEstimationEvaluationError("reports use different selections")
        if report.calibration_identity != first.calibration_identity:
            raise StateEstimationEvaluationError("reports use different calibration identities")
        if report.accuracy is None:
            raise StateEstimationEvaluationError(
                "every compared report needs a reference comparison"
            )
        if (
            report.accuracy.reference_id != first.accuracy.reference_id
            or report.accuracy.config != first.accuracy.config
            or report.evaluator_version != first.evaluator_version
        ):
            raise StateEstimationEvaluationError(
                "reports use a different reference, comparison protocol or evaluator version"
            )
    entries = []
    for report in reports:
        assert report.accuracy is not None
        entries.append(
            StateEstimationComparisonEntry(
                estimator=report.estimator,
                configuration_fingerprint=report.estimator.configuration_fingerprint,
                run_id=report.run_id,
                pose_count=report.structural.pose_count,
                ate_rmse_m=report.accuracy.ate_translation_m.rmse,
                rotation_error_max_rad=report.accuracy.rotation_error_rad.maximum,
                rpe_translation_rmse_m=(
                    report.accuracy.rpe.translation_m.rmse if report.accuracy.rpe else None
                ),
                gap_count=report.structural.gap_count,
                runtime_s=report.cost.runtime_s,
            )
        )
    return StateEstimationComparison(
        sequence_artifact_id=first.sequence_artifact_id,
        selection_id=first.selection_id,
        calibration_identity=first.calibration_identity,
        reference_id=first.accuracy.reference_id,
        comparison_config=first.accuracy.config,
        entries=tuple(entries),
    )


def encode_state_estimation_report(report: StateEstimationEvaluationReport) -> dict[str, Any]:
    """Encode a report as plain JSON-compatible data.

    Args:
        report: The report to encode.

    Returns:
        A record with every identity needed to reproduce the report; ``cost`` is
        the only section that carries runtime.
    """
    return {
        "evaluator_version": report.evaluator_version,
        "sequence_artifact_id": str(report.sequence_artifact_id),
        "selection_id": report.selection_id,
        "run_id": None if report.run_id is None else str(report.run_id),
        "trajectory_id": str(report.trajectory_id),
        "estimator": {
            "backend_id": report.estimator.backend_id,
            "backend_version": report.estimator.backend_version,
            "configuration_fingerprint": report.estimator.configuration_fingerprint,
        },
        "calibration_identity": report.calibration_identity,
        "structural": {
            "pose_count": report.structural.pose_count,
            "reference_frame": str(report.structural.reference_frame),
            "body_frame": str(report.structural.body_frame),
            "clock_id": report.structural.clock_id,
            "all_values_finite": report.structural.all_values_finite,
            "max_quaternion_norm_error": report.structural.max_quaternion_norm_error,
            "min_interval_ns": report.structural.min_interval_ns,
            "gap_count": report.structural.gap_count,
            "degraded_pose_count": report.structural.degraded_pose_count,
            "consumed_observation_count": report.structural.consumed_observation_count,
            "rejected_observation_count": report.structural.rejected_observation_count,
        },
        "motion": {
            "translation_delta_m": _encode_distribution(report.motion.summary.translation_delta_m),
            "orientation_delta_rad": _encode_distribution(
                report.motion.summary.orientation_delta_rad
            ),
            "linear_speed_mps": _encode_distribution(report.motion.summary.linear_speed_mps),
            "angular_speed_radps": _encode_distribution(report.motion.summary.angular_speed_radps),
            "interval_ns": _encode_distribution(report.motion.interval_ns),
            "thresholds": None
            if report.motion.thresholds is None
            else {
                "max_translation_delta_m": report.motion.thresholds.max_translation_delta_m,
                "max_orientation_delta_rad": report.motion.thresholds.max_orientation_delta_rad,
                "max_linear_speed_mps": report.motion.thresholds.max_linear_speed_mps,
                "max_angular_speed_radps": report.motion.thresholds.max_angular_speed_radps,
            },
            "anomalies": [
                {
                    "kind": a.kind,
                    "previous_estimate_id": str(a.previous_estimate_id),
                    "next_estimate_id": str(a.next_estimate_id),
                    "value": a.value,
                    "threshold": a.threshold,
                }
                for a in report.motion.anomalies
            ],
        },
        "temporal": None
        if report.temporal is None
        else {
            "lookup_count": report.temporal.lookup_count,
            "exact_count": report.temporal.exact_count,
            "nearest_count": report.temporal.nearest_count,
            "interpolated_count": report.temporal.interpolated_count,
            "rejected_count": report.temporal.rejected_count,
            "rejections_by_reason": {
                reason.value: count
                for reason, count in report.temporal.rejections_by_reason.items()
            },
            "min_time_delta_ns": report.temporal.min_time_delta_ns,
            "median_time_delta_ns": report.temporal.median_time_delta_ns,
            "max_time_delta_ns": report.temporal.max_time_delta_ns,
        },
        "transform_trace": None
        if report.transform_trace is None
        else {
            "max_composition_error_m": report.transform_trace.max_composition_error_m,
            "max_composition_error_rad": report.transform_trace.max_composition_error_rad,
            "max_round_trip_error_m": report.transform_trace.max_round_trip_error_m,
            "max_round_trip_error_rad": report.transform_trace.max_round_trip_error_rad,
            "records": [
                {
                    "pose_estimate_id": str(r.pose_estimate_id),
                    "timestamp_ns": r.timestamp_ns,
                    "reference_frame": str(r.reference_frame),
                    "sensor_frame": str(r.sensor_frame),
                    "sensor_translation_m": list(r.sensor_translation_m),
                    "sensor_rotation_xyzw": list(r.sensor_rotation),
                    "composition_error_m": r.composition_error_m,
                    "composition_error_rad": r.composition_error_rad,
                    "round_trip_error_m": r.round_trip_error_m,
                    "round_trip_error_rad": r.round_trip_error_rad,
                }
                for r in report.transform_trace.records
            ],
        },
        "reference": None if report.accuracy is None else _encode_reference(report.accuracy),
        "accuracy": None if report.accuracy is None else _encode_accuracy(report.accuracy),
        "cost": {"pose_count": report.cost.pose_count, "runtime_s": report.cost.runtime_s},
    }


def _encode_distribution(summary: DistributionSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "median": summary.median,
        "p95": summary.p95,
        "maximum": summary.maximum,
    }


def _encode_errors(stats: ErrorStatistics) -> dict[str, Any]:
    return {
        "count": stats.count,
        "rmse": stats.rmse,
        "mean": stats.mean,
        "median": stats.median,
        "p95": stats.p95,
        "maximum": stats.maximum,
    }


def _encode_reference(accuracy: AccuracyReport) -> dict[str, Any]:
    alignment = accuracy.alignment
    return {
        "reference_id": accuracy.reference_id,
        "role": accuracy.reference_role.value,
        "reference_frame": str(accuracy.reference_frame),
        "body_frame": str(accuracy.reference_body_frame),
        "config": {
            "alignment": accuracy.config.alignment.value,
            "max_time_difference_ns": accuracy.config.max_time_difference_ns,
            "relative_pair_offset": accuracy.config.relative_pair_offset,
        },
        "alignment": {
            "method": alignment.method.value,
            "translation_m": None
            if alignment.translation_m is None
            else list(alignment.translation_m),
            "rotation_xyzw": None if alignment.rotation is None else list(alignment.rotation),
        },
    }


def _encode_accuracy(accuracy: AccuracyReport) -> dict[str, Any]:
    return {
        "association": {
            "matched_count": accuracy.association.matched_count,
            "unmatched_count": accuracy.association.unmatched_count,
            "max_time_difference_ns": accuracy.association.max_time_difference_ns,
        },
        "ate_translation_m": _encode_errors(accuracy.ate_translation_m),
        "rotation_error_rad": _encode_errors(accuracy.rotation_error_rad),
        "rpe": None
        if accuracy.rpe is None
        else {
            "pair_offset": accuracy.rpe.pair_offset,
            "pair_count": accuracy.rpe.pair_count,
            "translation_m": _encode_errors(accuracy.rpe.translation_m),
            "rotation_rad": _encode_errors(accuracy.rpe.rotation_rad),
        },
    }
