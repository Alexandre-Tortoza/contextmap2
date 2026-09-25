"""Calibration, reprojection and temporal-alignment diagnostics of an association frame.

Association is only as trustworthy as its calibration, its pose timing and the map it
projects. These diagnostics make those conditions observable, per frame, without
becoming a semantic confidence:

* everything the frame depended on is recorded: calibration, camera model, pose lookup, RGB
  timestamp, the acquisition window of the geometry, the prepared-image transform and the
  visibility policy;
* the reprojection residual is reported **only** against trusted reference correspondences
  and is never estimated from the association itself;
* a frame that is misaligned in time, or whose reference does not reproject, gets an explicit
  finding instead of a silent pass;
* an optional time-offset sweep helps diagnose a timing error without ever changing the
  calibration or a timestamp.

There is no automatic calibration optimization here, and no diagnostic changes a semantic
claim. Raw components are preserved so a downstream policy decides whether and how to use them.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import ImageObservation, SourceObservationId
from contextmap.sensor_association.camera_models import CameraIdentity
from contextmap.sensor_association.dense_sampling import DenseFeatureSamples, InterpolationPolicy
from contextmap.sensor_association.frame_projection import (
    ExtrinsicRef,
    FrameProjection,
    FrameProjector,
    RejectedProjection,
)
from contextmap.sensor_association.membership import FrameMembership, MembershipStatistics
from contextmap.sensor_association.models import (
    CalibrationRef,
    DepthMetric,
    PoseRef,
    VisibilityState,
)
from contextmap.sensor_association.quality import ReprojectionStatistics, ValueSummary
from contextmap.sensor_association.serialization import encode_calibration_ref, encode_pose_ref
from contextmap.sensor_association.visibility import VisibilityResolution
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import LookupRejection, TimeBounds
from contextmap.visual_perception import PreparedImage

if TYPE_CHECKING:
    from numpy.typing import NDArray

DIAGNOSTICS_DEFINITIONS_VERSION = "association-diagnostics-v1"
"""Versioned identity of the diagnostic definitions in this module."""

_NO_REFERENCE = "no trusted reference correspondences were provided"
_NONE_PROJECTS = "no trusted reference correspondence could be projected"


class FindingSeverity(Enum):
    """How serious a diagnostic finding is.

    Attributes:
        WARNING: A configured tolerance was exceeded; the association ran but should be read
            with care.
        FAILURE: The frame cannot support a meaningful association, or a reference that was
            provided could not be evaluated at all.
    """

    WARNING = "warning"
    FAILURE = "failure"


class FindingCode(Enum):
    """The conditions a frame diagnostic can flag.

    Attributes:
        POSE_TIME_DELTA_EXCEEDS_TOLERANCE: The pose used is further in time from the frame
            than the tolerance allows.
        FRAME_OUTSIDE_MAP_TIME_WINDOW: The frame lies outside the geometry's acquisition
            window by more than the tolerance allows.
        NOTHING_VISIBLE_IN_IMAGE: No map point is visible in the supported prepared image.
        REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE: The 95th percentile residual against the
            trusted reference exceeds the tolerance.
        REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE: Too many reference correspondences could
            not be projected.
        NO_REFERENCE_CORRESPONDENCE_PROJECTS: A trusted reference was provided but none of its
            correspondences could be projected.
    """

    POSE_TIME_DELTA_EXCEEDS_TOLERANCE = "pose_time_delta_exceeds_tolerance"
    FRAME_OUTSIDE_MAP_TIME_WINDOW = "frame_outside_map_time_window"
    NOTHING_VISIBLE_IN_IMAGE = "nothing_visible_in_image"
    REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE = "reprojection_residual_exceeds_tolerance"
    REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE = "reprojection_invalid_rate_exceeds_tolerance"
    NO_REFERENCE_CORRESPONDENCE_PROJECTS = "no_reference_correspondence_projects"


@dataclass(frozen=True, kw_only=True)
class DiagnosticFinding:
    """One explicit warning or failure.

    Attributes:
        code: What was flagged.
        severity: How serious it is.
        message: A human-readable description with the numbers behind it.
        observed: The measured value, when there is one.
        tolerance: The tolerance it was compared with, when there is one.
    """

    code: FindingCode
    severity: FindingSeverity
    message: str
    observed: float | None
    tolerance: float | None


@dataclass(frozen=True, kw_only=True)
class DiagnosticTolerances:
    """The tolerances a frame is checked against. There are no defaults.

    A check is only as meaningful as the profile it comes from, so each tolerance is chosen
    per run; ``None`` disables that check.

    Attributes:
        max_pose_time_delta_ns: Largest accepted distance from the frame timestamp to the
            nearest contributing pose, in nanoseconds.
        max_map_window_offset_ns: Largest accepted distance from the frame timestamp to the
            geometry's acquisition window, in nanoseconds.
        max_reprojection_p95_px: Largest accepted 95th percentile residual, in pixels.
        max_reprojection_invalid_rate: Largest accepted share of reference correspondences that
            could not be projected, within ``[0, 1]``.
    """

    max_pose_time_delta_ns: int | None
    max_map_window_offset_ns: int | None
    max_reprojection_p95_px: float | None
    max_reprojection_invalid_rate: float | None

    def __post_init__(self) -> None:
        """Validate the tolerances.

        Raises:
            ValueError: If a tolerance is negative or not finite, or the invalid rate is above 1.
        """
        for name in (
            "max_pose_time_delta_ns",
            "max_map_window_offset_ns",
            "max_reprojection_p95_px",
            "max_reprojection_invalid_rate",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and not negative, got {value!r}")
        rate = self.max_reprojection_invalid_rate
        if rate is not None and rate > 1.0:
            raise ValueError(f"max_reprojection_invalid_rate must be at most 1, got {rate!r}")


@dataclass(frozen=True, kw_only=True, eq=False)
class TrustedCorrespondences:
    """Reference pairs of map geometry and the pixel where it was really observed.

    They must come from a trusted source (surveyed landmarks, a calibration target); they are
    what makes a reprojection residual meaningful, and nothing here creates them.

    Attributes:
        reference_id: Identity of the trusted correspondence set.
        geometry_indices: ``(K,)`` global indices of the map geometry elements, the same
            identity :func:`~contextmap.geometric_mapping.geometry_id_for` uses.
        observed_pixels: ``(K, 2)`` raw-image pixels where each element was observed.
    """

    reference_id: str
    geometry_indices: NDArray[Any]
    observed_pixels: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate the reference.

        Raises:
            ValueError: If the reference is unnamed or empty, the arrays disagree in shape, or
                a pixel is not finite.
        """
        import numpy as np

        if not self.reference_id:
            raise ValueError("reference_id must not be empty")
        count = self.geometry_indices.shape[0] if self.geometry_indices.ndim == 1 else -1
        if count != self.observed_pixels.shape[0] or self.observed_pixels.shape[1:] != (2,):
            raise ValueError(
                f"geometry_indices must have shape (K,) and observed_pixels shape (K, 2), got "
                f"{self.geometry_indices.shape} and {self.observed_pixels.shape}"
            )
        if count < 1:
            raise ValueError("a trusted reference needs at least one correspondence")
        if not np.isfinite(self.observed_pixels).all():
            raise ValueError("observed_pixels must be finite")


def reprojection_statistics(
    frame: FrameProjection, correspondences: TrustedCorrespondences
) -> ReprojectionStatistics | None:
    """Measure the reprojection residual of a projected frame against a trusted reference.

    The residual of a correspondence is the distance, in raw-image pixels, between where the
    frame projects the geometry and where it was observed. A correspondence the camera model
    cannot project is counted as invalid and contributes no residual. The 95th percentile uses
    linear interpolation between order statistics.

    A correspondence names geometry by its **global** map index, so it is looked up in the
    frame's candidate rows rather than used as one: geometry the frame's candidate policy
    did not select is reported as unevaluated and contributes no residual.

    Args:
        frame: The projection of one camera frame's candidate geometry.
        correspondences: The trusted reference.

    Returns:
        The statistics, or ``None`` when no correspondence could be projected.

    Raises:
        ValueError: If a correspondence names geometry outside the map.
    """
    import numpy as np

    indices = correspondences.geometry_indices
    if indices.min() < 0 or indices.max() >= frame.map_point_count:
        raise ValueError(
            f"geometry_indices must lie within the map's {frame.map_point_count} elements"
        )
    count = int(indices.shape[0])
    rows, evaluated = frame.rows_for(indices)
    valid = np.zeros(count, dtype=bool)
    valid[evaluated] = frame.projectable[rows[evaluated]]
    valid_count = int(valid.sum())
    if valid_count == 0:
        return None
    projected = frame.raw_pixels[rows[valid]]
    residuals = np.hypot(*(projected - correspondences.observed_pixels[valid]).T)
    return ReprojectionStatistics(
        reference_id=correspondences.reference_id,
        correspondence_count=count,
        invalid_count=int((evaluated & ~valid).sum()),
        unevaluated_count=count - int(evaluated.sum()),
        mean_px=float(residuals.mean()),
        median_px=float(np.median(residuals)),
        p95_px=float(np.percentile(residuals, 95)),
        max_px=float(residuals.max()),
    )


@dataclass(frozen=True, kw_only=True)
class DenseSamplingSummary:
    """How one dense feature map was sampled in a frame, for comparing feature paths.

    Attributes:
        feature_id: The dense feature.
        source_artifact_id: The artifact that owns it.
        embedding_space_id: Its feature space.
        interpolation: The interpolation policy used.
        coordinate_transform_id: The transform that produced its sampling geometry.
        sampling_fingerprint: Hash of its complete sampling geometry.
        enhanced: ``True`` when the map is the output of a resolution enhancement.
        enhancement_source_feature_id: The native feature an enhanced map derives from.
        eligible_count: Visible points that were eligible.
        sampled_count: Of those, the ones the grid served.
        out_of_support_count: Of those, the ones it could not serve.
    """

    feature_id: str
    source_artifact_id: str
    embedding_space_id: str
    interpolation: InterpolationPolicy
    coordinate_transform_id: str
    sampling_fingerprint: str
    enhanced: bool
    enhancement_source_feature_id: str | None
    eligible_count: int
    sampled_count: int
    out_of_support_count: int


@dataclass(frozen=True, kw_only=True)
class FrameDiagnostics:
    """The diagnostic report of one association frame.

    The geometry and calibration part is independent of the dense feature path, so a native
    and an enhanced run over the same inputs report identical values there.

    Attributes:
        definitions_version: Version of the diagnostic definitions.
        source_observation_id: The camera frame.
        image_timestamp: When the frame was acquired.
        map_id: The geometric map projected.
        map_time_bounds: The acquisition window of the geometry.
        map_window_offset_ns: Signed distance from the frame timestamp to that window, in
            nanoseconds; ``0`` inside it and ``None`` when the clock domains differ.
        calibration_ref: The exact calibration and camera used.
        camera: The calibrated camera.
        pose_ref: The exact pose used; its ``time_delta_ns`` is the temporal alignment.
        extrinsic: The static body-to-camera extrinsic used.
        image_transform_id: The raw-to-prepared image chain.
        prepared_image_size: ``(width, height)`` of the prepared image.
        visibility_policy_id: The occlusion rule applied.
        visibility_policy_fingerprint: Hash of that rule's parameters.
        depth_metric: How depth is measured.
        point_count: Map points evaluated.
        state_counts: Points per decided visibility state.
        visible_count: Points that are supported and not occluded.
        visible_depth_m: Depth of the visible points, when there are any.
        membership: The region membership summary, when membership was evaluated.
        reprojection: Residual against a trusted reference, when one was provided.
        reprojection_unavailable_reason: Why ``reprojection`` is missing, otherwise ``None``.
        dense_sampling: One summary per dense feature map sampled.
        findings: Explicit warnings and failures.
    """

    definitions_version: str
    source_observation_id: SourceObservationId
    image_timestamp: SourceTimestamp
    map_id: MapId
    map_time_bounds: TimeBounds
    map_window_offset_ns: int | None
    calibration_ref: CalibrationRef
    camera: CameraIdentity
    pose_ref: PoseRef
    extrinsic: ExtrinsicRef
    image_transform_id: str
    prepared_image_size: tuple[int, int]
    visibility_policy_id: str
    visibility_policy_fingerprint: str
    depth_metric: DepthMetric
    point_count: int
    state_counts: dict[VisibilityState, int]
    visible_count: int
    visible_depth_m: ValueSummary | None
    membership: MembershipStatistics | None
    reprojection: ReprojectionStatistics | None
    reprojection_unavailable_reason: str | None
    dense_sampling: tuple[DenseSamplingSummary, ...]
    findings: tuple[DiagnosticFinding, ...]

    @property
    def failed(self) -> bool:
        """Whether any finding is a failure."""
        return any(finding.severity is FindingSeverity.FAILURE for finding in self.findings)

    def to_record(self) -> dict[str, Any]:
        """Return the report as JSON primitives, keeping every raw component."""
        return {
            "definitions_version": self.definitions_version,
            "source_observation_id": str(self.source_observation_id),
            "image_timestamp": self.image_timestamp.to_record(),
            "map": {
                "map_id": str(self.map_id),
                "time_bounds": {
                    "start": self.map_time_bounds.start.to_record(),
                    "end": self.map_time_bounds.end.to_record(),
                },
                "window_offset_ns": self.map_window_offset_ns,
            },
            "calibration_ref": encode_calibration_ref(self.calibration_ref),
            "camera": {
                "calibration_id": str(self.camera.calibration_id),
                "content_hash": self.camera.content_hash,
                "camera_frame": str(self.camera.camera_frame),
                "camera_model_kind": self.camera.camera_model_kind,
                "image_size": list(self.camera.image_size),
            },
            "pose_ref": encode_pose_ref(self.pose_ref),
            "extrinsic": {
                "calibration_identity": self.extrinsic.calibration_identity,
                "parent_frame": str(self.extrinsic.parent_frame),
                "child_frame": str(self.extrinsic.child_frame),
            },
            "image_transform_id": self.image_transform_id,
            "prepared_image_size": list(self.prepared_image_size),
            "visibility_policy": {
                "policy_id": self.visibility_policy_id,
                "fingerprint": self.visibility_policy_fingerprint,
            },
            "depth_metric": self.depth_metric.value,
            "visibility": {
                "point_count": self.point_count,
                "behind_camera": self.state_counts[VisibilityState.BEHIND_CAMERA],
                "outside_image": self.state_counts[VisibilityState.OUTSIDE_IMAGE],
                "outside_valid_support": self.state_counts[VisibilityState.OUTSIDE_VALID_SUPPORT],
                "occluded": self.state_counts[VisibilityState.OCCLUDED],
                "visible": self.visible_count,
                "visible_depth_m": None
                if self.visible_depth_m is None
                else dataclasses.asdict(self.visible_depth_m),
            },
            "membership": None if self.membership is None else dataclasses.asdict(self.membership),
            "reprojection": None
            if self.reprojection is None
            else dataclasses.asdict(self.reprojection),
            "reprojection_unavailable_reason": self.reprojection_unavailable_reason,
            "dense_sampling": [
                {**dataclasses.asdict(summary), "interpolation": summary.interpolation.value}
                for summary in self.dense_sampling
            ],
            "findings": [
                {
                    "code": finding.code.value,
                    "severity": finding.severity.value,
                    "message": finding.message,
                    "observed": finding.observed,
                    "tolerance": finding.tolerance,
                }
                for finding in self.findings
            ],
        }


def diagnose_frame(
    resolution: VisibilityResolution,
    *,
    tolerances: DiagnosticTolerances,
    membership: FrameMembership | None = None,
    correspondences: TrustedCorrespondences | None = None,
    dense_samples: Sequence[DenseFeatureSamples] = (),
) -> FrameDiagnostics:
    """Build the diagnostic report of one association frame.

    Args:
        resolution: The visibility of the frame's points.
        tolerances: The tolerances the frame is checked against.
        membership: The frame's region membership, when it was evaluated.
        correspondences: A trusted reference to measure the reprojection residual against.
        dense_samples: The dense feature samplings of the frame, one per feature map.

    Returns:
        The report, with explicit findings for anything outside the tolerances.
    """
    import numpy as np

    frame = resolution.frame
    findings: list[DiagnosticFinding] = []
    window_offset = _window_offset_ns(frame.image_timestamp, frame.map_time_bounds)

    delta = frame.pose_ref.time_delta_ns
    limit = tolerances.max_pose_time_delta_ns
    if limit is not None and delta > limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.POSE_TIME_DELTA_EXCEEDS_TOLERANCE,
                severity=FindingSeverity.WARNING,
                message=(
                    f"the pose used is {delta} ns from the frame, above the {limit} ns tolerance"
                ),
                observed=delta,
                tolerance=limit,
            )
        )
    window_limit = tolerances.max_map_window_offset_ns
    if window_limit is not None and window_offset is not None and abs(window_offset) > window_limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.FRAME_OUTSIDE_MAP_TIME_WINDOW,
                severity=FindingSeverity.WARNING,
                message=(
                    f"the frame is {abs(window_offset)} ns outside the geometry's acquisition "
                    f"window, above the {window_limit} ns tolerance"
                ),
                observed=abs(window_offset),
                tolerance=window_limit,
            )
        )
    if resolution.visible_count == 0:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.NOTHING_VISIBLE_IN_IMAGE,
                severity=FindingSeverity.FAILURE,
                message="no map point is visible in the supported prepared image",
                observed=0,
                tolerance=None,
            )
        )

    reprojection = None
    reason: str | None = _NO_REFERENCE
    if correspondences is not None:
        reprojection = reprojection_statistics(frame, correspondences)
        reason = None if reprojection is not None else _NONE_PROJECTS
        findings.extend(_reprojection_findings(reprojection, tolerances))

    visible_depth = None
    if resolution.visible_count:
        depth = resolution.depth_m[resolution.visible]
        visible_depth = ValueSummary(
            count=int(depth.shape[0]),
            minimum=float(depth.min()),
            median=float(np.median(depth)),
            maximum=float(depth.max()),
        )

    return FrameDiagnostics(
        definitions_version=DIAGNOSTICS_DEFINITIONS_VERSION,
        source_observation_id=frame.source_observation_id,
        image_timestamp=frame.image_timestamp,
        map_id=frame.map_id,
        map_time_bounds=frame.map_time_bounds,
        map_window_offset_ns=window_offset,
        calibration_ref=frame.calibration_ref,
        camera=frame.camera,
        pose_ref=frame.pose_ref,
        extrinsic=frame.extrinsic,
        image_transform_id=frame.image_transform.transform_id,
        prepared_image_size=frame.image_transform.prepared_size,
        visibility_policy_id=resolution.policy.policy_id,
        visibility_policy_fingerprint=resolution.policy.fingerprint(),
        depth_metric=resolution.depth_metric,
        point_count=len(frame.projectable),
        state_counts=resolution.state_counts(),
        visible_count=resolution.visible_count,
        visible_depth_m=visible_depth,
        membership=None if membership is None else membership.statistics(),
        reprojection=reprojection,
        reprojection_unavailable_reason=reason,
        dense_sampling=tuple(_dense_summary(samples) for samples in dense_samples),
        findings=tuple(findings),
    )


def _reprojection_findings(
    statistics: ReprojectionStatistics | None, tolerances: DiagnosticTolerances
) -> list[DiagnosticFinding]:
    if statistics is None:
        return [
            DiagnosticFinding(
                code=FindingCode.NO_REFERENCE_CORRESPONDENCE_PROJECTS,
                severity=FindingSeverity.FAILURE,
                message=_NONE_PROJECTS,
                observed=None,
                tolerance=None,
            )
        ]
    findings: list[DiagnosticFinding] = []
    p95_limit = tolerances.max_reprojection_p95_px
    if p95_limit is not None and statistics.p95_px > p95_limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE,
                severity=FindingSeverity.WARNING,
                message=(
                    f"the 95th percentile reprojection residual is {statistics.p95_px:.3f} px, "
                    f"above the {p95_limit} px tolerance"
                ),
                observed=statistics.p95_px,
                tolerance=p95_limit,
            )
        )
    rate = statistics.invalid_count / statistics.correspondence_count
    rate_limit = tolerances.max_reprojection_invalid_rate
    if rate_limit is not None and rate > rate_limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE,
                severity=FindingSeverity.WARNING,
                message=(
                    f"{statistics.invalid_count} of {statistics.correspondence_count} reference "
                    f"correspondences could not be projected, above the {rate_limit} tolerance"
                ),
                observed=rate,
                tolerance=rate_limit,
            )
        )
    return findings


def _window_offset_ns(timestamp: SourceTimestamp, bounds: TimeBounds) -> int | None:
    if timestamp.clock_id != bounds.start.clock_id:
        return None
    instant = timestamp.total_nanoseconds()
    start = bounds.start.total_nanoseconds()
    end = bounds.end.total_nanoseconds()
    if instant < start:
        return instant - start
    if instant > end:
        return instant - end
    return 0


def _dense_summary(samples: DenseFeatureSamples) -> DenseSamplingSummary:
    provenance = samples.provenance
    enhancement = provenance.enhancement
    return DenseSamplingSummary(
        feature_id=str(provenance.feature_id),
        source_artifact_id=provenance.source_artifact_id,
        embedding_space_id=provenance.embedding_space_id,
        interpolation=provenance.interpolation,
        coordinate_transform_id=provenance.coordinate_transform_id,
        sampling_fingerprint=provenance.sampling_fingerprint,
        enhanced=enhancement is not None,
        enhancement_source_feature_id=None
        if enhancement is None
        else str(enhancement.source_feature_id),
        eligible_count=int(samples.eligible_indices.shape[0]),
        sampled_count=samples.sampled_count,
        out_of_support_count=samples.out_of_support_count,
    )


@dataclass(frozen=True, kw_only=True)
class TimeOffsetOutcome:
    """The reprojection residual of a frame whose timestamp was shifted by one offset.

    Attributes:
        offset_ns: The offset applied to the frame timestamp, in nanoseconds.
        statistics: The residual against the trusted reference, or ``None`` when the shifted
            frame could not be evaluated.
        rejection: Why the pose lookup rejected the shifted timestamp, when it did.
    """

    offset_ns: int
    statistics: ReprojectionStatistics | None
    rejection: LookupRejection | None


def time_offset_sweep(
    projector: FrameProjector,
    observation: ImageObservation,
    prepared_image: PreparedImage,
    correspondences: TrustedCorrespondences,
    offsets_ns: Sequence[int],
) -> tuple[TimeOffsetOutcome, ...]:
    """Measure the reprojection residual as the frame timestamp is shifted, for diagnosis only.

    Each offset re-projects the frame with the pose looked up at the shifted timestamp and
    measures the residual of the reference against it. The observation, the calibration and
    every timestamp are left untouched; the sweep only reports, and choosing to act on it is
    a separate, explicit decision. Each offset selects its own candidates, because a shifted
    pose moves the camera, so a reference whose geometry falls outside the candidate policy
    at some offset is reported as unevaluated there.

    Args:
        projector: The projector of the map, trajectory and calibration.
        observation: The camera frame.
        prepared_image: The image prepared from that frame.
        correspondences: The trusted reference.
        offsets_ns: The offsets to try, in nanoseconds, added to the frame timestamp.

    Returns:
        One outcome per offset, in the given order.
    """
    outcomes: list[TimeOffsetOutcome] = []
    for offset in offsets_ns:
        shifted = dataclasses.replace(
            observation, timestamp=_shifted(observation.timestamp, offset)
        )
        result = projector.project(shifted, prepared_image)
        if isinstance(result, RejectedProjection):
            outcomes.append(
                TimeOffsetOutcome(offset_ns=offset, statistics=None, rejection=result.rejection)
            )
        else:
            outcomes.append(
                TimeOffsetOutcome(
                    offset_ns=offset,
                    statistics=reprojection_statistics(result, correspondences),
                    rejection=None,
                )
            )
    return tuple(outcomes)


def _shifted(timestamp: SourceTimestamp, offset_ns: int) -> SourceTimestamp:
    seconds, nanoseconds = divmod(timestamp.total_nanoseconds() + offset_ns, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=timestamp.clock_id)
