"""FAST-LIO backend: LiDAR-inertial state estimation behind the canonical port.

FAST-LIO is the planned LiDAR-inertial estimator for Solution 1. Downstream
geometry consumes :class:`~contextmap.state_estimation.PoseEstimate` and
:class:`~contextmap.state_estimation.Trajectory`, never ROS or FAST-LIO types,
so everything specific to running FAST-LIO (ROS messages, its process, its
files) lives behind :class:`FastLioRunner`. This module owns the canonical
side: it validates the request, resolves the LiDAR-to-IMU extrinsic from the
canonical calibration (the backend keeps no second copy), hands the runner a
:class:`FastLioJob`, and turns the runner's output into a validated
trajectory.

What the trajectory means: FAST-LIO estimates the pose of the IMU (the body
frame) in the frame anchored at its first pose. That is a local frame, not a
global or reference-data frame, so comparing it with a reference trajectory
needs an explicit, reported alignment. The published poses say nothing about
motion correction of the input scans: the scans stay raw and this backend
records that explicitly, so no consumer assumes deskewing merely because
FAST-LIO was used. There is no fallback: any failure surfaces as a
:class:`FastLioFailure` naming the cause. See
``src/contextmap/state_estimation/docs/backends.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from typing import Protocol, runtime_checkable

from contextmap.ingestion import (
    FrameId,
    ImuObservation,
    LidarObservation,
    current_code_version,
)
from contextmap.shared import Quaternion, SourceTimestamp, Vector3
from contextmap.state_estimation.frame_graph import FrameGraphError, StaticFrameGraph
from contextmap.state_estimation.models import (
    EstimatorProvenance,
    PoseEstimate,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryGap,
    TrajectoryProvenance,
    pose_estimate_id_for,
)
from contextmap.state_estimation.ports import (
    DiagnosticSeverity,
    EstimationDiagnostic,
    MissingEstimatorInputError,
    StateEstimationError,
    StateEstimationRequest,
    StateEstimationResult,
)
from contextmap.state_estimation.pose_validation import (
    canonical_orientation,
    check_pose_values,
    gap_between,
)
from contextmap.state_estimation.preflight import (
    GeometryRequirements,
    StaticRelationRequirement,
    calibration_identity,
)

BACKEND_ID = "fast_lio"
_CODE_PREFIX = f"{BACKEND_ID}."

ParameterValue = str | int | float | bool
"""A primitive value forwarded verbatim to the runner as an estimator parameter."""


class FastLioFailureKind(Enum):
    """Why a FAST-LIO run failed.

    Attributes:
        PROCESS_FAILED: The process could not start or exited with an error.
        TIMEOUT: The process did not finish in time.
        MISSING_OUTPUT: The process finished without writing a trajectory.
        INVALID_OUTPUT: The trajectory could not be parsed or broke a rule.
        INITIALIZATION_FAILED: FAST-LIO reported it could not initialize.
        DIVERGED: FAST-LIO reported that its estimate diverged.
        VERSION_MISMATCH: The runner used another FAST-LIO version than configured.
        EMPTY_OUTPUT: The run finished but published no pose.
    """

    PROCESS_FAILED = "process_failed"
    TIMEOUT = "timeout"
    MISSING_OUTPUT = "missing_output"
    INVALID_OUTPUT = "invalid_output"
    INITIALIZATION_FAILED = "initialization_failed"
    DIVERGED = "diverged"
    VERSION_MISMATCH = "version_mismatch"
    EMPTY_OUTPUT = "empty_output"


class FastLioFailure(StateEstimationError):
    """Raised when FAST-LIO cannot produce a trajectory.

    Attributes:
        kind: The cause.
        detail: What happened, with the numbers or messages involved.
        stderr_tail: The end of the process log, when there is one.
    """

    def __init__(self, *, kind: FastLioFailureKind, detail: str, stderr_tail: str = "") -> None:
        """Build the failure.

        Args:
            kind: The cause.
            detail: What happened.
            stderr_tail: The end of the process log, when there is one.
        """
        super().__init__(f"FAST-LIO {kind.value}: {detail}")
        self.kind = kind
        self.detail = detail
        self.stderr_tail = stderr_tail


@dataclass(frozen=True, kw_only=True)
class FastLioJob:
    """Canonical description of one FAST-LIO run, free of any ROS type.

    Attributes:
        lidar: LiDAR scans in time order, all in ``lidar_frame``.
        imu: IMU samples in time order, all in ``imu_frame``.
        lidar_frame: Frame of every scan.
        imu_frame: Frame of every IMU sample; the body frame whose pose is estimated.
        lidar_in_imu_translation: Translation of ``T_imu_lidar`` in meters, i.e.
            the LiDAR origin expressed in the IMU frame.
        lidar_in_imu_rotation: Rotation of ``T_imu_lidar``, quaternion ``(x, y, z, w)``.
        clock_id: Clock domain of every timestamp in the job.
        scan_period_ns: Duration of one scan in nanoseconds.
        parameters: Estimator parameters forwarded verbatim to the runner.
    """

    lidar: tuple[LidarObservation, ...]
    imu: tuple[ImuObservation, ...]
    lidar_frame: FrameId
    imu_frame: FrameId
    lidar_in_imu_translation: Vector3
    lidar_in_imu_rotation: Quaternion
    clock_id: str
    scan_period_ns: int
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class FastLioRawPose:
    """One pose as the runner read it from FAST-LIO's output, before validation.

    Attributes:
        timestamp_ns: Time of the pose in nanoseconds of the job's clock domain.
            It must fall within the interval of the scan it was estimated from:
            from the scan's timestamp up to one ``scan_period_ns`` later.
        translation: ``(x, y, z)`` of the IMU frame in the FAST-LIO frame, in meters.
        orientation: Quaternion ``(x, y, z, w)`` as reported.
        covariance: Row-major 6x6 covariance over ``(x, y, z, rotation about
            x, y, z)``, or ``None`` when the estimator exposes none.
    """

    timestamp_ns: int
    translation: Vector3
    orientation: Quaternion
    covariance: tuple[float, ...] | None


@dataclass(frozen=True, kw_only=True)
class FastLioRunOutput:
    """A successful run's output.

    Attributes:
        poses: Poses in output order.
        reported_ref: FAST-LIO version or ref the runner says it used, or
            ``None`` when the runner does not report one.
        warnings: Messages the process reported while still succeeding.
    """

    poses: tuple[FastLioRawPose, ...]
    reported_ref: str | None
    warnings: tuple[str, ...]


@runtime_checkable
class FastLioRunner(Protocol):
    """Runs FAST-LIO for a job; the only place ROS and the FAST-LIO process exist."""

    def describe(self) -> Mapping[str, object]:
        """Report the runner's effective configuration as primitive values.

        Returns:
            Values that influence the run (e.g. the command); they enter the
            estimator's configuration fingerprint.
        """
        ...

    def run(self, job: FastLioJob) -> FastLioRunOutput:
        """Execute FAST-LIO for a job.

        Args:
            job: The canonical run description.

        Returns:
            The poses FAST-LIO produced.

        Raises:
            FastLioFailure: If FAST-LIO could not produce a trajectory.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class FastLioConfig:
    """Effective configuration of the FAST-LIO backend.

    Attributes:
        reference_frame: Name given to the frame FAST-LIO anchors at its first
            pose; it is the ``reference_frame`` of the published trajectory.
        body_frame: The IMU frame whose pose FAST-LIO estimates; every IMU
            sample must be expressed in it.
        fast_lio_ref: FAST-LIO version or git ref, required so the run is
            reproducible; a runner reporting another one fails the run.
        scan_period_ns: Duration of one LiDAR scan; a pose is attributed to the
            scan whose interval contains its timestamp.
        max_gap_ns: Interval between consecutive poses beyond which a
            :class:`~contextmap.state_estimation.TrajectoryGap` is recorded.
        orientation_norm_tolerance: Largest ``|norm - 1|`` of a reported
            quaternion that is renormalized (and recorded) instead of rejected.
        parameters: Estimator parameters (LiDAR type, noise, ranges, ...)
            forwarded verbatim to the runner and part of the fingerprint. The
            backend does not interpret them.
    """

    reference_frame: FrameId
    body_frame: FrameId
    fast_lio_ref: str
    scan_period_ns: int
    max_gap_ns: int | None = None
    orientation_norm_tolerance: float = 1e-3
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the configuration is possible.

        Raises:
            ValueError: If a frame or the ref is empty, the frames are equal,
                or a period or tolerance is not a positive/finite value.
        """
        if not self.reference_frame or not self.body_frame:
            raise ValueError("reference_frame and body_frame must not be empty")
        if self.reference_frame == self.body_frame:
            raise ValueError("reference_frame and body_frame must differ")
        if not self.fast_lio_ref:
            raise ValueError("fast_lio_ref must name the FAST-LIO version or ref")
        if self.scan_period_ns <= 0:
            raise ValueError("scan_period_ns must be positive")
        if self.max_gap_ns is not None and self.max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive when set")
        if not (
            math.isfinite(self.orientation_norm_tolerance) and self.orientation_norm_tolerance >= 0
        ):
            raise ValueError("orientation_norm_tolerance must be a non-negative finite number")


class FastLioEstimator:
    """State Estimation backend running FAST-LIO through a :class:`FastLioRunner`."""

    def __init__(self, config: FastLioConfig, runner: FastLioRunner) -> None:
        """Create the backend.

        Args:
            config: Effective configuration.
            runner: Executes FAST-LIO; the composition root provides it.
        """
        self._config = config
        self._runner = runner

    def estimator_provenance(self) -> EstimatorProvenance:
        """Report the FAST-LIO ref and a fingerprint of configuration and runner."""
        config = self._config
        payload = json.dumps(
            {
                "reference_frame": str(config.reference_frame),
                "body_frame": str(config.body_frame),
                "fast_lio_ref": config.fast_lio_ref,
                "scan_period_ns": config.scan_period_ns,
                "max_gap_ns": config.max_gap_ns,
                "orientation_norm_tolerance": config.orientation_norm_tolerance,
                "parameters": dict(config.parameters),
                "runner": dict(self._runner.describe()),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return EstimatorProvenance(
            backend_id=BACKEND_ID,
            backend_version=config.fast_lio_ref,
            configuration_fingerprint=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        )

    def geometry_requirements(self) -> GeometryRequirements:
        """Declare LiDAR and IMU data plus the LiDAR-to-IMU extrinsic; no camera data."""
        return GeometryRequirements(
            capability=f"state_estimation:{BACKEND_ID}",
            modalities=frozenset({"lidar", "imu"}),
            static_relations=(StaticRelationRequirement(from_endpoint="lidar", to_endpoint="imu"),),
            reference_frame=self._config.reference_frame,
            body_frame=self._config.body_frame,
        )

    def estimate(self, request: StateEstimationRequest) -> StateEstimationResult:
        """Run FAST-LIO over the request's LiDAR and IMU observations.

        Args:
            request: Canonical inputs; the calibration must connect the IMU and
                LiDAR frames with a static transform.

        Returns:
            The validated trajectory, an explicit note that input scans are not
            motion corrected, and any warning the runner or gap detection raised.

        Raises:
            MissingEstimatorInputError: If LiDAR, IMU or calibration is absent.
            StateEstimationError: If the streams are inconsistent (frames, clocks,
                ordering, IMU coverage) or the extrinsic is missing.
            FastLioFailure: If FAST-LIO fails or its output is invalid.
        """
        job = self._build_job(request)
        output = self._runner.run(job)
        if output.reported_ref is not None and output.reported_ref != self._config.fast_lio_ref:
            raise FastLioFailure(
                kind=FastLioFailureKind.VERSION_MISMATCH,
                detail=(
                    f"the runner used {output.reported_ref!r} but the configuration "
                    f"declares {self._config.fast_lio_ref!r}"
                ),
            )
        if not output.poses:
            raise FastLioFailure(
                kind=FastLioFailureKind.EMPTY_OUTPUT, detail="FAST-LIO published no pose"
            )

        poses, gaps, gap_diagnostics = self._publish(request, job, output)
        diagnostics = [
            EstimationDiagnostic(
                severity=DiagnosticSeverity.INFO,
                code=f"{_CODE_PREFIX}raw_scans_not_deskewed",
                message=(
                    "input scans are raw: this trajectory does not imply that any scan was "
                    "motion corrected"
                ),
            ),
            *(
                EstimationDiagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code=f"{_CODE_PREFIX}runner_warning",
                    message=warning,
                )
                for warning in output.warnings
            ),
            *gap_diagnostics,
        ]
        trajectory = Trajectory(
            trajectory_id=request.trajectory_id,
            reference_frame=self._config.reference_frame,
            body_frame=self._config.body_frame,
            poses=tuple(poses),
            gaps=tuple(gaps),
            provenance=TrajectoryProvenance(
                estimator=self.estimator_provenance(),
                sequence_artifact_id=request.sequence_artifact_id,
                selection_id=request.selection_id,
                calibration_identity=calibration_identity(request.calibration),
                code_version=current_code_version(),
            ),
        )
        return StateEstimationResult(
            trajectory=trajectory,
            diagnostics=tuple(diagnostics),
            consumed_observation_count=len(job.lidar) + len(job.imu),
            rejected_observation_count=0,
        )

    def _build_job(self, request: StateEstimationRequest) -> FastLioJob:
        config = self._config
        lidar = tuple(o for o in request.observations if isinstance(o, LidarObservation))
        imu = tuple(o for o in request.observations if isinstance(o, ImuObservation))
        if not lidar:
            raise MissingEstimatorInputError(
                "the FAST-LIO backend needs LidarObservation observations, but the request has none"
            )
        if not imu:
            raise MissingEstimatorInputError(
                "the FAST-LIO backend needs ImuObservation observations, but the request has none"
            )
        if request.calibration is None:
            raise MissingEstimatorInputError(
                "the FAST-LIO backend needs the canonical calibration for the LiDAR-to-IMU "
                "extrinsic, but the request has none"
            )

        lidar_frames = {o.frame_id for o in lidar}
        if len(lidar_frames) != 1:
            raise StateEstimationError(
                f"LiDAR observations span several frames {sorted(map(str, lidar_frames))}; "
                "FAST-LIO needs a single LiDAR frame"
            )
        wrong_imu_frames = {o.frame_id for o in imu} - {config.body_frame}
        if wrong_imu_frames:
            raise StateEstimationError(
                f"IMU observations must be in the body frame {config.body_frame!r}, "
                f"found {sorted(map(str, wrong_imu_frames))}"
            )
        clocks = {o.timestamp.clock_id for o in (*lidar, *imu)}
        if len(clocks) != 1:
            raise StateEstimationError(
                f"LiDAR and IMU observations use different clock domains {sorted(clocks)}"
            )
        for name, stream in (("LiDAR", lidar), ("IMU", imu)):
            times = [o.timestamp.total_nanoseconds() for o in stream]
            if any(later <= earlier for earlier, later in pairwise(times)):
                raise StateEstimationError(f"{name} timestamps must be strictly increasing")
        last_scan_end = lidar[-1].timestamp.total_nanoseconds() + config.scan_period_ns
        if (
            imu[0].timestamp.total_nanoseconds() > lidar[0].timestamp.total_nanoseconds()
            or imu[-1].timestamp.total_nanoseconds() < last_scan_end
        ):
            raise StateEstimationError(
                "IMU observations do not cover the LiDAR scans: they must start no later than "
                "the first scan and reach the end of the last one"
            )

        lidar_frame = next(iter(lidar_frames))
        try:
            extrinsic = StaticFrameGraph.from_calibration(request.calibration).resolve(
                config.body_frame, lidar_frame
            )
        except FrameGraphError as error:
            raise StateEstimationError(
                f"no static transform connects the IMU frame {config.body_frame!r} and the "
                f"LiDAR frame {lidar_frame!r} in the calibration: {error}"
            ) from error
        return FastLioJob(
            lidar=lidar,
            imu=imu,
            lidar_frame=lidar_frame,
            imu_frame=config.body_frame,
            lidar_in_imu_translation=extrinsic.translation,
            lidar_in_imu_rotation=extrinsic.rotation,
            clock_id=next(iter(clocks)),
            scan_period_ns=config.scan_period_ns,
            parameters=dict(config.parameters),
        )

    def _publish(
        self, request: StateEstimationRequest, job: FastLioJob, output: FastLioRunOutput
    ) -> tuple[list[PoseEstimate], list[TrajectoryGap], list[EstimationDiagnostic]]:
        config = self._config
        scan_starts = [scan.timestamp.total_nanoseconds() for scan in job.lidar]
        poses: list[PoseEstimate] = []
        gaps: list[TrajectoryGap] = []
        diagnostics: list[EstimationDiagnostic] = []

        for raw in output.poses:
            violation = check_pose_values(
                translation=raw.translation,
                orientation=raw.orientation,
                covariance=raw.covariance,
                orientation_norm_tolerance=config.orientation_norm_tolerance,
            )
            if violation is not None:
                raise FastLioFailure(
                    kind=FastLioFailureKind.INVALID_OUTPUT,
                    detail=f"pose at {raw.timestamp_ns} ns: {violation.detail}",
                )
            if poses and raw.timestamp_ns <= poses[-1].timestamp.total_nanoseconds():
                raise FastLioFailure(
                    kind=FastLioFailureKind.INVALID_OUTPUT,
                    detail=f"pose at {raw.timestamp_ns} ns is not later than the previous pose",
                )
            scan_index = bisect_right(scan_starts, raw.timestamp_ns) - 1
            if scan_index < 0 or raw.timestamp_ns > scan_starts[scan_index] + job.scan_period_ns:
                raise FastLioFailure(
                    kind=FastLioFailureKind.INVALID_OUTPUT,
                    detail=(
                        f"pose at {raw.timestamp_ns} ns falls within no LiDAR scan interval, so "
                        "it cannot be traced to a source observation"
                    ),
                )
            orientation, conversions = canonical_orientation(raw.orientation)
            seconds, nanoseconds = divmod(raw.timestamp_ns, 1_000_000_000)
            pose = PoseEstimate(
                estimate_id=pose_estimate_id_for(
                    trajectory_id=request.trajectory_id, index=len(poses)
                ),
                timestamp=SourceTimestamp(
                    seconds=seconds, nanoseconds=nanoseconds, clock_id=job.clock_id
                ),
                parent_frame=config.reference_frame,
                child_frame=config.body_frame,
                translation_m=raw.translation,
                orientation=orientation,
                validity=PoseValidity.VALID,
                provenance=PoseProvenance(
                    source_observation_ids=(job.lidar[scan_index].observation_id,),
                    conversions_applied=conversions,
                ),
                covariance=raw.covariance,
            )
            gap = gap_between(poses[-1], pose, max_gap_ns=config.max_gap_ns) if poses else None
            if gap is not None:
                gaps.append(gap)
                diagnostics.append(
                    EstimationDiagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        code=f"{_CODE_PREFIX}timestamp_gap",
                        message=(
                            f"{gap.duration_ns} ns between {gap.previous_estimate_id!r} and "
                            f"{gap.next_estimate_id!r} exceeds max_gap_ns={config.max_gap_ns}"
                        ),
                        observation_id=job.lidar[scan_index].observation_id,
                    )
                )
            poses.append(pose)
        return poses, gaps, diagnostics
