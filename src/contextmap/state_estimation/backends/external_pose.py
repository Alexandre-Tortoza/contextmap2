"""ExternalPose backend: publish externally measured poses as a canonical trajectory.

Datasets, simulators and odometry sources may supply poses. Ingestion turns
them into :class:`~contextmap.ingestion.ExternalPoseMeasurement` (frames,
meters and ``(x, y, z, w)`` quaternions already canonical); this backend
validates each measurement, publishes it as a
:class:`~contextmap.state_estimation.PoseEstimate` and assembles a
:class:`~contextmap.state_estimation.Trajectory`. It serves as the first
geometry baseline without depending on an estimator's quality or runtime.

An external pose is an input measurement, not ground truth: nothing here
labels it as reference data. The backend does not know how a dataset names its
files or frames, never rewrites calibration, and never silently repairs a
frame, timestamp or orientation problem: an invalid sample either fails the
run or is dropped and recorded, depending on an explicit policy. See
``src/contextmap/state_estimation/docs/backends.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum

from contextmap.ingestion import (
    ExternalPoseMeasurement,
    FrameId,
    SourceObservationId,
    current_code_version,
)
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
from contextmap.state_estimation.preflight import GeometryRequirements

BACKEND_ID = "external_pose"
BACKEND_VERSION = "1"
"""Bumped whenever the validation or mapping semantics of this backend change."""

_CODE_PREFIX = f"{BACKEND_ID}."


class InvalidSamplePolicy(Enum):
    """What the backend does with a measurement that fails validation.

    Attributes:
        FAIL: Stop the run with :class:`InvalidPoseSampleError`.
        SKIP: Drop the measurement and record a warning diagnostic; the run
            fails only if no valid measurement remains.
    """

    FAIL = "fail"
    SKIP = "skip"


class InvalidPoseSampleError(StateEstimationError):
    """Raised when a measurement is invalid and the policy is ``FAIL``.

    Attributes:
        code: Stable identifier of the violated condition.
        observation_id: The offending measurement.
        detail: What is wrong, with the numbers involved.
    """

    def __init__(self, *, code: str, observation_id: SourceObservationId, detail: str) -> None:
        """Build the error.

        Args:
            code: Stable identifier of the violated condition.
            observation_id: The offending measurement.
            detail: What is wrong, with the numbers involved.
        """
        super().__init__(f"invalid pose sample {observation_id!r} ({code}): {detail}")
        self.code = code
        self.observation_id = observation_id
        self.detail = detail


@dataclass(frozen=True, kw_only=True)
class ExternalPoseConfig:
    """Effective configuration of the ExternalPose backend.

    Attributes:
        reference_frame: Frame every measurement must be expressed in, and
            the ``reference_frame`` of the published trajectory.
        body_frame: Frame whose pose every measurement reports, and the
            ``body_frame`` of the published trajectory. A measurement that
            declares other frames is invalid: frame semantics are never
            inferred or renamed here.
        invalid_sample_policy: What to do with an invalid measurement.
        max_gap_ns: Interval between consecutive poses beyond which a
            :class:`~contextmap.state_estimation.TrajectoryGap` is recorded;
            ``None`` records none. Gaps are reported, never filled.
        orientation_norm_tolerance: Largest ``|norm - 1|`` of a source
            quaternion that is renormalized (and recorded) instead of rejected.
    """

    reference_frame: FrameId
    body_frame: FrameId
    invalid_sample_policy: InvalidSamplePolicy = InvalidSamplePolicy.FAIL
    max_gap_ns: int | None = None
    orientation_norm_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        """Validate that the configuration is possible.

        Raises:
            ValueError: If a frame is empty, the frames are equal, or a
                tolerance is not a positive/finite value.
        """
        if not self.reference_frame or not self.body_frame:
            raise ValueError("reference_frame and body_frame must not be empty")
        if self.reference_frame == self.body_frame:
            raise ValueError("reference_frame and body_frame must differ")
        if self.max_gap_ns is not None and self.max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive when set")
        if not (
            math.isfinite(self.orientation_norm_tolerance) and self.orientation_norm_tolerance >= 0
        ):
            raise ValueError("orientation_norm_tolerance must be a non-negative finite number")

    def fingerprint(self) -> str:
        """Return a deterministic hash of this configuration.

        Returns:
            ``"sha256:<hex digest>"``, different for any changed field.
        """
        payload = json.dumps(
            {
                "reference_frame": str(self.reference_frame),
                "body_frame": str(self.body_frame),
                "invalid_sample_policy": self.invalid_sample_policy.value,
                "max_gap_ns": self.max_gap_ns,
                "orientation_norm_tolerance": self.orientation_norm_tolerance,
            },
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class _Violation:
    code: str
    detail: str


class ExternalPoseEstimator:
    """State Estimation backend over canonical external pose measurements."""

    def __init__(self, config: ExternalPoseConfig) -> None:
        """Create the backend.

        Args:
            config: Effective configuration.
        """
        self._config = config

    def estimator_provenance(self) -> EstimatorProvenance:
        """Report this backend's identity and configuration fingerprint."""
        return EstimatorProvenance(
            backend_id=BACKEND_ID,
            backend_version=BACKEND_VERSION,
            configuration_fingerprint=self._config.fingerprint(),
        )

    def geometry_requirements(self) -> GeometryRequirements:
        """Declare that this backend needs external poses and no calibration."""
        return GeometryRequirements(
            capability=f"state_estimation:{BACKEND_ID}",
            modalities=frozenset({"external_pose"}),
            reference_frame=self._config.reference_frame,
            body_frame=self._config.body_frame,
        )

    def estimate(self, request: StateEstimationRequest) -> StateEstimationResult:
        """Publish the request's external pose measurements as a trajectory.

        Args:
            request: Canonical inputs; only ``ExternalPoseMeasurement``
                observations are used, in the order given.

        Returns:
            The trajectory, one warning diagnostic per rejected measurement
            and per recorded gap, and the consumed/rejected counts.

        Raises:
            MissingEstimatorInputError: If the request has no external pose.
            InvalidPoseSampleError: If a measurement is invalid and the policy
                is ``FAIL``.
            StateEstimationError: If no valid measurement remains.
        """
        measurements = [
            observation
            for observation in request.observations
            if isinstance(observation, ExternalPoseMeasurement)
        ]
        if not measurements:
            raise MissingEstimatorInputError(
                "the external pose backend needs ExternalPoseMeasurement observations, "
                "but the request has none"
            )

        poses: list[PoseEstimate] = []
        gaps: list[TrajectoryGap] = []
        diagnostics: list[EstimationDiagnostic] = []
        rejected = 0

        for measurement in measurements:
            violation = self._violation(measurement, poses)
            if violation is not None:
                if self._config.invalid_sample_policy is InvalidSamplePolicy.FAIL:
                    raise InvalidPoseSampleError(
                        code=violation.code,
                        observation_id=measurement.observation_id,
                        detail=violation.detail,
                    )
                rejected += 1
                diagnostics.append(
                    EstimationDiagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        code=violation.code,
                        message=f"rejected {measurement.observation_id!r}: {violation.detail}",
                        observation_id=measurement.observation_id,
                    )
                )
                continue

            pose = self._publish(measurement, request, index=len(poses))
            gap = (
                gap_between(poses[-1], pose, max_gap_ns=self._config.max_gap_ns) if poses else None
            )
            if gap is not None:
                gaps.append(gap)
                diagnostics.append(
                    EstimationDiagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        code=f"{_CODE_PREFIX}timestamp_gap",
                        message=(
                            f"{gap.duration_ns} ns between {gap.previous_estimate_id!r} and "
                            f"{gap.next_estimate_id!r} exceeds max_gap_ns={self._config.max_gap_ns}"
                        ),
                        observation_id=measurement.observation_id,
                    )
                )
            poses.append(pose)

        if not poses:
            raise StateEstimationError(
                f"the external pose backend found no valid pose sample among {len(measurements)} "
                "measurement(s)"
            )

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
                # Este backend só consome medições de pose; não usa calibração.
                calibration_identity=None,
                code_version=current_code_version(),
            ),
        )
        return StateEstimationResult(
            trajectory=trajectory,
            diagnostics=tuple(diagnostics),
            consumed_observation_count=len(measurements),
            rejected_observation_count=rejected,
        )

    def _violation(
        self, measurement: ExternalPoseMeasurement, accepted: list[PoseEstimate]
    ) -> _Violation | None:
        """Return the first rule the measurement breaks, or ``None`` if it is valid."""
        config = self._config
        if measurement.parent_frame != config.reference_frame or (
            measurement.frame_id != config.body_frame
        ):
            return _Violation(
                code=f"{_CODE_PREFIX}frame_mismatch",
                detail=(
                    f"declares {measurement.parent_frame!r} -> {measurement.frame_id!r}, "
                    f"expected {config.reference_frame!r} -> {config.body_frame!r}"
                ),
            )
        values = check_pose_values(
            translation=measurement.translation,
            orientation=measurement.orientation,
            covariance=measurement.pose_covariance,
            orientation_norm_tolerance=config.orientation_norm_tolerance,
        )
        if values is not None:
            return _Violation(code=f"{_CODE_PREFIX}{values.rule}", detail=values.detail)
        if accepted:
            previous = accepted[-1].timestamp
            if measurement.timestamp.clock_id != previous.clock_id:
                return _Violation(
                    code=f"{_CODE_PREFIX}clock_domain_mismatch",
                    detail=(
                        f"clock {measurement.timestamp.clock_id!r} differs from the trajectory "
                        f"clock {previous.clock_id!r}"
                    ),
                )
            if measurement.timestamp.total_nanoseconds() <= previous.total_nanoseconds():
                return _Violation(
                    code=f"{_CODE_PREFIX}non_increasing_timestamp",
                    detail=(
                        "timestamp is not later than the previous accepted sample "
                        "(duplicate or out of order)"
                    ),
                )
        return None

    def _publish(
        self, measurement: ExternalPoseMeasurement, request: StateEstimationRequest, *, index: int
    ) -> PoseEstimate:
        """Publish a validated measurement as a canonical pose."""
        orientation, conversions = canonical_orientation(measurement.orientation)
        return PoseEstimate(
            estimate_id=pose_estimate_id_for(trajectory_id=request.trajectory_id, index=index),
            timestamp=measurement.timestamp,
            parent_frame=self._config.reference_frame,
            child_frame=self._config.body_frame,
            translation_m=measurement.translation,
            orientation=orientation,
            validity=PoseValidity.VALID,
            provenance=PoseProvenance(
                source_observation_ids=(measurement.observation_id,),
                conversions_applied=conversions,
            ),
            covariance=measurement.pose_covariance,
        )
