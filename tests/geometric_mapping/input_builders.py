"""Deterministic sequence, calibration and trajectory builders for geometry input tests."""

from __future__ import annotations

import math

from lidar_builders import CLOCK_ID, make_scan, timestamp_ns

from contextmap.geometric_mapping import (
    GeometryInputPlan,
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
    ScanDisposition,
    assemble_geometry_inputs,
)
from contextmap.ingestion import (
    CalibrationSet,
    ExternalPoseMeasurement,
    FrameId,
    FullSequenceSelection,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    RigidTransform,
    SensorId,
    SequenceArtifactId,
    SequenceSelection,
    SequenceSelectionResult,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
    selection_identity,
)
from contextmap.shared import Quaternion, Vector3
from contextmap.state_estimation import (
    EstimatorProvenance,
    LookupPolicy,
    PoseEstimate,
    PoseProvenance,
    PoseValidity,
    StateEstimationRunId,
    Trajectory,
    TrajectoryId,
    TrajectoryProvenance,
    calibration_identity,
    pose_estimate_id_for,
)

MS = 1_000_000
SEQUENCE_ID = SequenceArtifactId("sequence-0001")
TRAJECTORY_ID = TrajectoryId("run-0001--trajectory")
IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)
QUARTER_TURN_Z: Quaternion = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))


def rigid(
    parent: str,
    child: str,
    translation: Vector3 = (0.0, 0.0, 0.0),
    rotation: Quaternion = IDENTITY,
) -> RigidTransform:
    """Build a static transform ``T_parent_child``."""
    return RigidTransform(
        parent_frame=FrameId(parent),
        child_frame=FrameId(child),
        translation=translation,
        rotation=rotation,
    )


T_BODY_LIDAR = rigid("body", "lidar", (0.5, 0.0, 0.25))


def make_calibration(transforms: tuple[RigidTransform, ...] = (T_BODY_LIDAR,)) -> CalibrationSet:
    """Build a calibration whose only content is static transforms."""
    return CalibrationSet(entries={}, static_transforms=transforms)


def make_pose(
    index: int,
    *,
    time_ns: int | None = None,
    clock_id: str = CLOCK_ID,
    orientation: Quaternion = IDENTITY,
) -> PoseEstimate:
    """Build a valid pose of ``body`` in ``map``; by default 100 ms apart along +x."""
    when = index * 100 * MS if time_ns is None else time_ns
    return PoseEstimate(
        estimate_id=pose_estimate_id_for(trajectory_id=TRAJECTORY_ID, index=index),
        timestamp=timestamp_ns(when, clock_id=clock_id),
        parent_frame=FrameId("map"),
        child_frame=FrameId("body"),
        translation_m=(float(index), 0.0, 0.0),
        orientation=orientation,
        validity=PoseValidity.VALID,
        provenance=PoseProvenance(source_observation_ids=(SourceObservationId(f"pose-{index}"),)),
    )


def make_trajectory(
    count: int = 5,
    *,
    calibration: CalibrationSet | None = None,
    declare_calibration: bool = True,
    sequence_artifact_id: SequenceArtifactId = SEQUENCE_ID,
    clock_id: str = CLOCK_ID,
    orientation: Quaternion = IDENTITY,
) -> Trajectory:
    """Build a trajectory of ``count`` poses 100 ms apart, every one with ``orientation``.

    The trajectory declares the identity of ``calibration`` (the default one)
    unless ``declare_calibration`` is false, as a backend that needed none would.
    """
    used = calibration if calibration is not None else make_calibration()
    return Trajectory(
        trajectory_id=TRAJECTORY_ID,
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
        poses=tuple(
            make_pose(index, clock_id=clock_id, orientation=orientation) for index in range(count)
        ),
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(backend_id="fake_estimator", backend_version="0"),
            sequence_artifact_id=sequence_artifact_id,
            selection_id="full-sequence",
            calibration_identity=calibration_identity(used) if declare_calibration else None,
            code_version="test",
        ),
    )


def make_image(observation_id: str = "frame-0000", *, time_ns: int = 50 * MS) -> ImageObservation:
    """Build a minimal image, which is never a geometry input."""
    return ImageObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("camera0"),
        frame_id=FrameId("camera"),
        timestamp=timestamp_ns(time_ns),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/camera"),
        width=1,
        height=1,
        encoding=ImageEncoding.MONO8,
        data=b"\x00",
    )


def make_imu(observation_id: str = "imu-0000", *, time_ns: int = 50 * MS) -> ImuObservation:
    """Build a minimal IMU sample, which is never a geometry input."""
    return ImuObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("imu0"),
        frame_id=FrameId("imu"),
        timestamp=timestamp_ns(time_ns),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/imu"),
    )


def make_external_pose(index: int) -> ExternalPoseMeasurement:
    """Build the measurement behind ``make_pose(index)``."""
    return ExternalPoseMeasurement(
        observation_id=SourceObservationId(f"pose-{index:04d}"),
        sensor_id=SensorId("external_pose_source"),
        frame_id=FrameId("body"),
        timestamp=timestamp_ns(index * 100 * MS),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/poses"),
        parent_frame=FrameId("map"),
        translation=(float(index), 0.0, 0.0),
        orientation=IDENTITY,
    )


def make_sequence(scan_count: int = 5) -> list[SourceObservation]:
    """LiDAR scans ``scan-0000..`` every 100 ms, with one image and one IMU sample at 50 ms."""
    observations: list[SourceObservation] = []
    for index in range(scan_count):
        observations.append(make_scan(f"scan-{index:04d}", time_ns=index * 100 * MS))
        if index == 0:
            observations.extend([make_image(), make_imu()])
    return observations


def make_selection_result(
    observations: list[SourceObservation],
    selection: SequenceSelection | None = None,
    *,
    sequence_artifact_id: SequenceArtifactId = SEQUENCE_ID,
) -> SequenceSelectionResult:
    """Wrap already selected observations as the result of resolving ``selection``."""
    chosen = selection if selection is not None else FullSequenceSelection()
    return SequenceSelectionResult(
        sequence_artifact_id=sequence_artifact_id,
        selection=chosen,
        selection_id=selection_identity(sequence_artifact_id, chosen),
        observations=tuple(observations),
    )


ACCEPT_ALL = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.ACCEPT)
INTERPOLATED = LookupPolicy.interpolated()
DEFAULT_CALIBRATION = make_calibration()


def assemble_plan(
    observations: list[SourceObservation] | None = None,
    *,
    calibration: CalibrationSet | None = DEFAULT_CALIBRATION,
    trajectory: Trajectory | None = None,
    pose_lookup: LookupPolicy = INTERPOLATED,
    policy: MotionCorrectionPolicy = ACCEPT_ALL,
    selection: SequenceSelection | None = None,
    motion_correction: dict[SourceObservationId, MotionCorrectionRecord] | None = None,
    run_id: StateEstimationRunId | None = None,
) -> GeometryInputPlan:
    """Assemble the geometry inputs of ``observations`` (the default sequence when omitted)."""
    return assemble_geometry_inputs(
        sequence=make_selection_result(
            observations if observations is not None else make_sequence(), selection
        ),
        calibration=calibration,
        trajectory=trajectory if trajectory is not None else make_trajectory(),
        pose_lookup=pose_lookup,
        motion_correction_policy=policy,
        motion_correction=motion_correction,
        state_estimation_run_id=run_id,
    )
