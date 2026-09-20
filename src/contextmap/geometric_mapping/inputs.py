"""Deterministic assembly of the geometry inputs of one mapping run.

Geometric Mapping consumes canonical observations, never dataset files, ROS
topics or vendor names. This module pairs the LiDAR scans of a selected segment
of a canonical sequence with what their transformation into the map frame will
need, and refuses to guess when something is missing:

* the source frame, the timestamp (with its clock) and the payload identity;
* the pose of the body at the scan's timestamp, resolved through an explicit
  :class:`~contextmap.state_estimation.LookupPolicy` so trajectory coverage is
  checked before any point is touched;
* the static extrinsic ``T_body_source`` from the canonical calibration;
* the point layout, so the coordinates can be decoded without guessing;
* the motion-correction state and the run's policy for it.

A scan that cannot be used becomes a :class:`GeometryInputRejection` with a
reason; a problem that makes the whole assembly meaningless (another sequence,
another calibration, duplicate identities) raises :class:`GeometryInputError`.
Nothing is transformed here. Only ``LidarObservation`` is a geometry source
today: depth-derived point clouds are not a canonical ingestion modality yet,
and no untyped payload abstraction stands in for them.
See ``src/contextmap/geometric_mapping/docs/inputs.md``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from contextmap.geometric_mapping.motion_correction import (
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
    MotionCorrectionVerdict,
    ScanDisposition,
    apply_motion_correction_policy,
    unknown_motion_correction,
    verify_motion_correction,
)
from contextmap.ingestion import (
    CalibrationSet,
    ExplicitIdsSelection,
    FrameId,
    LidarObservation,
    PointFieldDataType,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceSelection,
    SequenceSelectionResult,
    SourceObservationId,
    observation_modality,
    resolve_selection,
    validate_lidar_observation,
)
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import (
    ClockDomainMismatchError,
    FrameGraphError,
    LookupPolicy,
    RejectedLookup,
    ResolvedPose,
    ResolvedTransform,
    StateEstimationRunId,
    StateEstimationRunReader,
    StaticFrameGraph,
    Trajectory,
    TrajectoryId,
    TrajectoryLookup,
    calibration_identity,
)

# O ingestion normaliza os campos numéricos do payload para little-endian.
_COORDINATE_FIELDS = ("x", "y", "z")
_SCALAR_SIZE_BYTES = {PointFieldDataType.FLOAT32: 4, PointFieldDataType.FLOAT64: 8}


class GeometryInputError(ValueError):
    """Raised when the inputs of a run do not belong together."""


class UnsupportedPointCloudLayoutError(ValueError):
    """Raised when a scan's coordinates cannot be decoded without guessing."""


@dataclass(frozen=True, kw_only=True)
class PointCloudLayout:
    """Where the coordinates are in a LiDAR point record.

    The coordinates are meters in the scan's frame, little-endian, as the
    canonical ``LidarObservation`` contract states.

    Attributes:
        x_offset_bytes: Byte offset of ``x`` within a point record.
        y_offset_bytes: Byte offset of ``y`` within a point record.
        z_offset_bytes: Byte offset of ``z`` within a point record.
        point_step_bytes: Bytes per point record.
        scalar: Floating-point type shared by ``x``, ``y`` and ``z``.
    """

    x_offset_bytes: int
    y_offset_bytes: int
    z_offset_bytes: int
    point_step_bytes: int
    scalar: PointFieldDataType


def resolve_point_cloud_layout(observation: LidarObservation) -> PointCloudLayout:
    """Locate the coordinates of a scan, refusing layouts that would need a guess.

    Args:
        observation: The scan.

    Returns:
        The offsets and scalar type of ``x``, ``y`` and ``z``. Other fields, such
        as intensity, are permitted and ignored.

    Raises:
        UnsupportedPointCloudLayoutError: If the payload disagrees with its
            declared size, a coordinate is missing, the coordinates are not one
            floating-point type with a single element each, or they overlap or
            fall outside the point record.
    """
    problems = validate_lidar_observation(observation)
    if problems:
        raise UnsupportedPointCloudLayoutError("; ".join(problems))

    by_name = {point_field.name: point_field for point_field in observation.fields}
    missing = [name for name in _COORDINATE_FIELDS if name not in by_name]
    if missing:
        raise UnsupportedPointCloudLayoutError(
            f"point cloud has no {', '.join(repr(name) for name in missing)} field"
        )
    coordinates = [by_name[name] for name in _COORDINATE_FIELDS]

    scalars = {point_field.data_type for point_field in coordinates}
    if len(scalars) != 1 or next(iter(scalars)) not in _SCALAR_SIZE_BYTES:
        raise UnsupportedPointCloudLayoutError(
            "x, y and z must share one floating-point type (float32 or float64), got "
            f"{sorted(scalar.value for scalar in scalars)}"
        )
    scalar = next(iter(scalars))
    if any(point_field.count != 1 for point_field in coordinates):
        raise UnsupportedPointCloudLayoutError("x, y and z must each have count 1")

    size = _SCALAR_SIZE_BYTES[scalar]
    offsets = sorted(point_field.offset_bytes for point_field in coordinates)
    if any(later - earlier < size for earlier, later in pairwise(offsets)):
        raise UnsupportedPointCloudLayoutError("the coordinate fields overlap")
    if offsets[-1] + size > observation.point_step_bytes:
        raise UnsupportedPointCloudLayoutError(
            f"a coordinate ends beyond point_step_bytes={observation.point_step_bytes}"
        )

    return PointCloudLayout(
        x_offset_bytes=by_name["x"].offset_bytes,
        y_offset_bytes=by_name["y"].offset_bytes,
        z_offset_bytes=by_name["z"].offset_bytes,
        point_step_bytes=observation.point_step_bytes,
        scalar=scalar,
    )


class InputRejectionReason(Enum):
    """Why a scan was left out of the geometry inputs.

    Attributes:
        MISSING_SOURCE_FRAME: The scan declares no frame.
        MISSING_CLOCK_IDENTITY: The scan's timestamp names no clock domain.
        MISSING_PAYLOAD: The scan has no points.
        UNSUPPORTED_LAYOUT: The coordinates cannot be decoded without guessing.
        INCONSISTENT_MOTION_CORRECTION: The declared correction state disagrees
            with the scan.
        MOTION_CORRECTION_REJECTED: The run's policy rejects the scan's
            correction state.
        NO_STATIC_TRANSFORM: The calibration has no static path from the body
            frame to the scan's frame.
        CLOCK_DOMAIN_MISMATCH: The scan is in another clock domain than the
            trajectory.
        POSE_LOOKUP_REJECTED: The lookup policy accepts no pose for the scan's
            timestamp (outside the trajectory, gap, tolerance).
    """

    MISSING_SOURCE_FRAME = "missing_source_frame"
    MISSING_CLOCK_IDENTITY = "missing_clock_identity"
    MISSING_PAYLOAD = "missing_payload"
    UNSUPPORTED_LAYOUT = "unsupported_layout"
    INCONSISTENT_MOTION_CORRECTION = "inconsistent_motion_correction"
    MOTION_CORRECTION_REJECTED = "motion_correction_rejected"
    NO_STATIC_TRANSFORM = "no_static_transform"
    CLOCK_DOMAIN_MISMATCH = "clock_domain_mismatch"
    POSE_LOOKUP_REJECTED = "pose_lookup_rejected"


@dataclass(frozen=True, kw_only=True)
class GeometryInput:
    """One scan with everything its transformation will need.

    ``P_map = T_map_body(t) · T_body_source · P_source``: ``pose`` is the first
    factor, ``static_transform`` the second. No point has been transformed.

    Attributes:
        observation: The source scan, unchanged.
        payload_hash: ``"sha256:<hex>"`` of the scan's payload bytes.
        layout: Where the coordinates are in each point record.
        pose: ``T_map_body(t)`` at the scan's timestamp, with how it was resolved.
        static_transform: ``T_body_source`` from the calibration; ``None`` only
            when the scan is already expressed in the body frame.
        motion_correction: The scan's declared correction state and what the
            run's policy decided about it.
    """

    observation: LidarObservation
    payload_hash: str
    layout: PointCloudLayout
    pose: ResolvedPose
    static_transform: ResolvedTransform | None
    motion_correction: MotionCorrectionVerdict

    @property
    def observation_id(self) -> SourceObservationId:
        """Identity of the physical observation."""
        return self.observation.observation_id

    @property
    def source_frame(self) -> FrameId:
        """Frame the scan's coordinates are expressed in."""
        return self.observation.frame_id

    @property
    def timestamp(self) -> SourceTimestamp:
        """Acquisition time of the scan, with its clock domain."""
        return self.observation.timestamp


@dataclass(frozen=True, kw_only=True)
class GeometryInputRejection:
    """A scan left out of the inputs, kept as data so it can be audited.

    Attributes:
        observation_id: The scan that was left out.
        reason: Why.
        detail: Human-readable specifics.
    """

    observation_id: SourceObservationId
    reason: InputRejectionReason
    detail: str


@dataclass(frozen=True, kw_only=True)
class GeometryInputPlan:
    """The geometry inputs of one run and the lineage they were assembled under.

    Attributes:
        sequence_artifact_id: Canonical sequence the scans come from.
        selection_id: Deterministic identity of the selection over that sequence.
        trajectory_id: Trajectory the poses come from.
        state_estimation_run_id: State Estimation run behind the trajectory, when known.
        map_frame: Frame of the global map (the trajectory's reference frame).
        body_frame: Frame of the rig's body.
        calibration_identity: Identity of the calibration the extrinsics come
            from; ``None`` when the run had none.
        pose_lookup: Policy every pose was resolved under.
        motion_correction_policy: Policy applied to every scan's correction state.
        inputs: Usable scans, in the selection's canonical order.
        rejections: Scans left out, in the selection's canonical order.
        ignored_observation_counts: Selected observations that are not geometry
            sources, by modality; never silently dropped.
    """

    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    map_frame: FrameId
    body_frame: FrameId
    calibration_identity: str | None
    pose_lookup: LookupPolicy
    motion_correction_policy: MotionCorrectionPolicy
    inputs: tuple[GeometryInput, ...]
    rejections: tuple[GeometryInputRejection, ...]
    ignored_observation_counts: Mapping[str, int]


def assemble_geometry_inputs_from_artifacts(
    *,
    sequence: SequenceArtifactReader,
    selection: SequenceSelection,
    run: StateEstimationRunReader,
    pose_lookup: LookupPolicy,
    motion_correction_policy: MotionCorrectionPolicy,
    motion_correction: Mapping[SourceObservationId, MotionCorrectionRecord] | None = None,
) -> GeometryInputPlan:
    """Assemble the inputs of a run from a canonical sequence and a State Estimation run.

    Args:
        sequence: The canonical sequence artifact.
        selection: Which of its observations to map.
        run: The State Estimation run whose trajectory places them.
        pose_lookup: How a pose is resolved at a scan's timestamp.
        motion_correction_policy: What to do with scans not known to be corrected.
        motion_correction: Declared correction state per scan; scans without one
            are ``UNKNOWN``.

    Returns:
        The plan; see :func:`assemble_geometry_inputs`.

    Raises:
        SequenceSelectionError: If the selection does not resolve.
        GeometryInputError: If the inputs do not belong together.
    """
    return assemble_geometry_inputs(
        sequence=resolve_selection(sequence, selection),
        calibration=sequence.read_calibration(),
        trajectory=run.trajectory(),
        state_estimation_run_id=run.manifest.run_id,
        pose_lookup=pose_lookup,
        motion_correction_policy=motion_correction_policy,
        motion_correction=motion_correction,
    )


def assemble_geometry_inputs(
    *,
    sequence: SequenceSelectionResult,
    calibration: CalibrationSet | None,
    trajectory: Trajectory,
    pose_lookup: LookupPolicy,
    motion_correction_policy: MotionCorrectionPolicy,
    state_estimation_run_id: StateEstimationRunId | None = None,
    motion_correction: Mapping[SourceObservationId, MotionCorrectionRecord] | None = None,
) -> GeometryInputPlan:
    """Pair the selected LiDAR scans with their pose, extrinsic and correction state.

    Per scan, the first failing check in this order rejects it: source frame,
    clock identity, payload, layout, correction record, correction policy, static transform,
    pose lookup.

    Args:
        sequence: The resolved selection over a canonical sequence.
        calibration: The sequence's canonical calibration, if any.
        trajectory: The trajectory the poses are resolved from.
        pose_lookup: How a pose is resolved at a scan's timestamp.
        motion_correction_policy: What to do with scans not known to be corrected.
        state_estimation_run_id: The run behind ``trajectory``, recorded in the plan.
        motion_correction: Declared correction state per scan; scans without one
            are ``UNKNOWN``, never inferred.

    Returns:
        The plan: usable scans, rejected scans with reasons, and the lineage.

    Raises:
        GeometryInputError: If the trajectory comes from another sequence or was
            estimated with another calibration, if an observation identity
            repeats, or if an explicit selection names an observation that is
            not a LiDAR scan.
    """
    identity = calibration_identity(calibration)
    _require_compatible_lineage(sequence, identity, trajectory)
    _require_unique_identities(sequence)
    _require_explicit_ids_are_scans(sequence)

    declared = motion_correction or {}
    graph = (
        StaticFrameGraph.from_calibration(calibration)
        if calibration is not None
        else StaticFrameGraph(())
    )
    lookup = TrajectoryLookup(trajectory)

    inputs: list[GeometryInput] = []
    rejections: list[GeometryInputRejection] = []
    ignored: Counter[str] = Counter()
    for observation in sequence.observations:
        if not isinstance(observation, LidarObservation):
            ignored[observation_modality(observation)] += 1
            continue
        record = declared.get(observation.observation_id)
        outcome = _assess_scan(
            observation,
            record=record if record is not None else unknown_motion_correction(observation),
            trajectory=trajectory,
            graph=graph,
            lookup=lookup,
            pose_lookup=pose_lookup,
            motion_correction_policy=motion_correction_policy,
        )
        if isinstance(outcome, GeometryInput):
            inputs.append(outcome)
        else:
            rejections.append(outcome)

    return GeometryInputPlan(
        sequence_artifact_id=sequence.sequence_artifact_id,
        selection_id=sequence.selection_id,
        trajectory_id=trajectory.trajectory_id,
        state_estimation_run_id=state_estimation_run_id,
        map_frame=trajectory.reference_frame,
        body_frame=trajectory.body_frame,
        calibration_identity=identity,
        pose_lookup=pose_lookup,
        motion_correction_policy=motion_correction_policy,
        inputs=tuple(inputs),
        rejections=tuple(rejections),
        ignored_observation_counts=dict(ignored),
    )


def _require_compatible_lineage(
    sequence: SequenceSelectionResult, used_calibration: str | None, trajectory: Trajectory
) -> None:
    provenance = trajectory.provenance
    if provenance.sequence_artifact_id != sequence.sequence_artifact_id:
        raise GeometryInputError(
            f"the trajectory was estimated over sequence {provenance.sequence_artifact_id!r} but "
            f"the scans come from {sequence.sequence_artifact_id!r}"
        )
    # Um backend que não precisou de calibração não declara identidade; então não há o que comparar.
    estimated_with = provenance.calibration_identity
    if estimated_with is not None and estimated_with != used_calibration:
        raise GeometryInputError(
            f"the trajectory was estimated with calibration {estimated_with} but the extrinsics "
            f"would come from calibration {used_calibration}"
        )


def _require_unique_identities(sequence: SequenceSelectionResult) -> None:
    counts = Counter(observation.observation_id for observation in sequence.observations)
    repeated = sorted(str(identity) for identity, count in counts.items() if count > 1)
    if repeated:
        raise GeometryInputError(f"duplicate observation identities in one selection: {repeated}")


def _require_explicit_ids_are_scans(sequence: SequenceSelectionResult) -> None:
    if not isinstance(sequence.selection, ExplicitIdsSelection):
        return
    not_scans = sorted(
        str(observation.observation_id)
        for observation in sequence.observations
        if not isinstance(observation, LidarObservation)
    )
    if not_scans:
        raise GeometryInputError(
            f"an explicit selection names observations that are not LiDAR scans: {not_scans}"
        )


def _assess_scan(
    scan: LidarObservation,
    *,
    record: MotionCorrectionRecord,
    trajectory: Trajectory,
    graph: StaticFrameGraph,
    lookup: TrajectoryLookup,
    pose_lookup: LookupPolicy,
    motion_correction_policy: MotionCorrectionPolicy,
) -> GeometryInput | GeometryInputRejection:
    def reject(reason: InputRejectionReason, detail: str) -> GeometryInputRejection:
        return GeometryInputRejection(
            observation_id=scan.observation_id, reason=reason, detail=detail
        )

    if not scan.frame_id:
        return reject(InputRejectionReason.MISSING_SOURCE_FRAME, "the scan declares no frame")
    if not scan.timestamp.clock_id:
        return reject(
            InputRejectionReason.MISSING_CLOCK_IDENTITY, "the scan's timestamp names no clock"
        )
    if scan.point_count == 0 or not scan.data:
        return reject(InputRejectionReason.MISSING_PAYLOAD, "the scan has no points")
    try:
        layout = resolve_point_cloud_layout(scan)
    except UnsupportedPointCloudLayoutError as error:
        return reject(InputRejectionReason.UNSUPPORTED_LAYOUT, str(error))

    problems = verify_motion_correction(record, scan)
    if problems:
        return reject(InputRejectionReason.INCONSISTENT_MOTION_CORRECTION, "; ".join(problems))
    verdict = apply_motion_correction_policy(record, motion_correction_policy)
    if verdict.disposition is ScanDisposition.REJECT:
        return reject(InputRejectionReason.MOTION_CORRECTION_REJECTED, str(verdict.message))

    static_transform: ResolvedTransform | None = None
    if scan.frame_id != trajectory.body_frame:
        try:
            static_transform = graph.resolve(trajectory.body_frame, scan.frame_id)
        except FrameGraphError as error:
            return reject(InputRejectionReason.NO_STATIC_TRANSFORM, str(error))

    try:
        pose = lookup.pose_for_observation(scan, policy=pose_lookup)
    except ClockDomainMismatchError as error:
        return reject(InputRejectionReason.CLOCK_DOMAIN_MISMATCH, str(error))
    if isinstance(pose, RejectedLookup):
        return reject(
            InputRejectionReason.POSE_LOOKUP_REJECTED, f"{pose.rejection.value}: {pose.detail}"
        )

    return GeometryInput(
        observation=scan,
        payload_hash=f"sha256:{hashlib.sha256(scan.data).hexdigest()}",
        layout=layout,
        pose=pose,
        static_transform=static_transform,
        motion_correction=verdict,
    )
