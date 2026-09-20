"""Deterministic world, platform motion and scans for the Geometric Mapping evaluation tests.

A static corridor (two walls and a floor) is observed from ten poses of a moving
platform. Every scan is *simulated* from the known truth: each scan holds the whole
world expressed in the sensor frame, so a correct mapping puts the same world point
at the same map coordinate in every scan, and any mistake in the pose, the extrinsic
or the timing shows up as a measurable disagreement.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from pathlib import Path

from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    GeometryInputPlan,
    MapDebugLevel,
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
    ScanDisposition,
    ScanVoxelPolicy,
    assemble_geometry_inputs,
)
from contextmap.ingestion import (
    CalibrationSet,
    FrameId,
    FullSequenceSelection,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    RigidTransform,
    SensorId,
    SequenceArtifactId,
    SequenceSelectionResult,
    SourceObservationId,
    SourceProvenance,
    selection_identity,
)
from contextmap.shared import (
    Quaternion,
    SourceTimestamp,
    Vector3,
    invert_rigid,
    normalize_quaternion,
    rotate_vector,
)
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
CLOCK_ID = "fixture:header"
SEQUENCE_ID = SequenceArtifactId("sequence-0001")
SEQUENCE = "corridor-synthetic"
SCAN_COUNT = 10
STEP_M = 0.37  # não múltiplo do espaçamento da grade (0,5 m): evita aliasing
IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)


def axis_angle(axis: Vector3, angle_rad: float) -> Quaternion:
    """Unit quaternion for a rotation of ``angle_rad`` about ``axis``."""
    norm = math.sqrt(sum(component * component for component in axis))
    sin = math.sin(angle_rad / 2.0)
    unit = tuple(component / norm for component in axis)
    return normalize_quaternion(
        (unit[0] * sin, unit[1] * sin, unit[2] * sin, math.cos(angle_rad / 2.0))
    )


def _world() -> tuple[Vector3, ...]:
    points: list[Vector3] = []
    for i in range(41):
        x = 0.5 * i
        for j in range(7):
            z = 0.5 * j
            points.append((x, 2.0, z))
            points.append((x, -2.0, z))
        for j in range(9):
            points.append((x, -2.0 + 0.5 * j, 0.0))
    return tuple(points)


WORLD: tuple[Vector3, ...] = _world()

TRUE_STATIC: tuple[Vector3, Quaternion] = ((0.3, 0.0, 1.0), axis_angle((0.1, 0.05, 1.0), 0.2))


def true_pose(index: int) -> tuple[Vector3, Quaternion]:
    """``T_map_body`` of scan ``index``: moves along +x, drifts in y and yaws slowly."""
    return (STEP_M * index, 0.021 * index, 0.0), axis_angle((0.05, 0.0, 1.0), 0.03 * index)


def _through(transform: tuple[Vector3, Quaternion], point: Vector3) -> Vector3:
    translation, rotation = transform
    rotated = rotate_vector(rotation, point)
    return (
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    )


def sensor_view(index: int) -> tuple[Vector3, ...]:
    """The whole world expressed in the sensor frame at scan ``index``."""
    to_body = invert_rigid(translation=true_pose(index)[0], rotation=true_pose(index)[1])
    to_lidar = invert_rigid(translation=TRUE_STATIC[0], rotation=TRUE_STATIC[1])
    return tuple(_through(to_lidar, _through(to_body, point)) for point in WORLD)


def make_scan(index: int, points: Sequence[Vector3]) -> LidarObservation:
    """A float64 scan (no payload quantization) at ``index * 100 ms``."""
    fields = tuple(
        PointFieldDescriptor(name=name, offset_bytes=8 * i, data_type=PointFieldDataType.FLOAT64)
        for i, name in enumerate("xyz")
    )
    seconds, nanoseconds = divmod(index * 100 * MS, 1_000_000_000)
    return LidarObservation(
        observation_id=SourceObservationId(f"scan-{index:04d}"),
        sensor_id=SensorId("lidar0"),
        frame_id=FrameId("lidar"),
        timestamp=SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=CLOCK_ID),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/world"),
        point_count=len(points),
        point_step_bytes=24,
        fields=fields,
        data=b"".join(struct.pack("<3d", *point) for point in points),
    )


def make_calibration(static: tuple[Vector3, Quaternion] = TRUE_STATIC) -> CalibrationSet:
    """``T_body_lidar`` as the calibration."""
    return CalibrationSet(
        entries={},
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("body"),
                child_frame=FrameId("lidar"),
                translation=static[0],
                rotation=static[1],
            ),
        ),
    )


def make_trajectory(
    *,
    calibration: CalibrationSet,
    invert_poses: bool = False,
    time_offset_ns: int = 0,
    pose_count: int = SCAN_COUNT + 2,
) -> Trajectory:
    """The true trajectory, optionally with a deliberate mistake.

    Args:
        calibration: The calibration the trajectory is declared to have used.
        invert_poses: Store ``T_body_map`` values under the ``T_map_body`` labels.
        time_offset_ns: Shift every pose timestamp, so each scan is matched to the
            pose of another instant.
        pose_count: Poses to publish (two more than the scans keeps them covered).
    """
    poses = []
    for index in range(pose_count):
        translation, rotation = true_pose(index)
        if invert_poses:
            translation, rotation = invert_rigid(translation=translation, rotation=rotation)
        total_ns = index * 100 * MS + time_offset_ns
        seconds, nanoseconds = divmod(total_ns, 1_000_000_000)
        poses.append(
            PoseEstimate(
                estimate_id=pose_estimate_id_for(trajectory_id=TrajectoryId("traj"), index=index),
                timestamp=SourceTimestamp(
                    seconds=seconds, nanoseconds=nanoseconds, clock_id=CLOCK_ID
                ),
                parent_frame=FrameId("map"),
                child_frame=FrameId("body"),
                translation_m=translation,
                orientation=rotation,
                validity=PoseValidity.VALID,
                provenance=PoseProvenance(
                    source_observation_ids=(SourceObservationId(f"pose-{index}"),)
                ),
            )
        )
    return Trajectory(
        trajectory_id=TrajectoryId("traj"),
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
        poses=tuple(poses),
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(backend_id="fake", backend_version="0"),
            sequence_artifact_id=SEQUENCE_ID,
            selection_id="full-sequence",
            calibration_identity=calibration_identity(calibration),
            code_version="test",
        ),
    )


ACCEPT_ALL = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.ACCEPT)


def make_plan(
    *,
    static: tuple[Vector3, Quaternion] = TRUE_STATIC,
    invert_poses: bool = False,
    time_offset_ns: int = 0,
    motion_correction: dict[SourceObservationId, MotionCorrectionRecord] | None = None,
    scan_count: int = SCAN_COUNT,
) -> GeometryInputPlan:
    """Assemble the inputs of the simulated scans under the given (possibly wrong) chain.

    The scans are always simulated with the true chain; ``static``, ``invert_poses``
    and ``time_offset_ns`` change what the *mapping* is told.
    """
    calibration = make_calibration(static)
    selection = FullSequenceSelection()
    observations = [make_scan(index, sensor_view(index)) for index in range(scan_count)]
    return assemble_geometry_inputs(
        sequence=SequenceSelectionResult(
            sequence_artifact_id=SEQUENCE_ID,
            selection=selection,
            selection_id=selection_identity(SEQUENCE_ID, selection),
            observations=tuple(observations),
        ),
        calibration=calibration,
        trajectory=make_trajectory(
            calibration=calibration, invert_poses=invert_poses, time_offset_ns=time_offset_ns
        ),
        pose_lookup=LookupPolicy.interpolated(),
        motion_correction_policy=ACCEPT_ALL,
        state_estimation_run_id=StateEstimationRunId("run-0001"),
        motion_correction=motion_correction,
    )


def write_run(
    workspace: Path,
    plan: GeometryInputPlan,
    *,
    index: int = 1,
    aggregation: ScanVoxelPolicy | None = None,
    runtime_s: float | None = None,
    profile: str = "baseline",
) -> GeometricMapArtifactReader:
    """Persist ``plan`` as a mapping run and open it."""
    writer = GeometricMapArtifactWriter(
        workspace_root=workspace,
        sequence_name=SEQUENCE,
        run_id=GeometricMapRunId(f"run-{index:04d}"),
        run_index=index,
        selection_label="full-sequence",
        profile_label=profile,
        debug_level=MapDebugLevel.NONE,
    )
    writer.finalize(plan=plan, aggregation=aggregation, code_version="test", runtime_s=runtime_s)
    run_dir = (
        workspace
        / "runs"
        / "geometric-mapping"
        / SEQUENCE
        / f"run-{index:04d}__full-sequence__{profile}"
    )
    return GeometricMapArtifactReader(run_dir)
