"""A real, tiny GeometricMapArtifact for ContextMapArtifact serialization tests.

The artifact is written by the public ``GeometricMapArtifactWriter`` from synthetic LiDAR scans,
so the reader and validator are exercised against the real payload format. The builders are a
compact copy of what the geometric-mapping tests use: test helper modules are imported by base
name, which is not reliable across test directories.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from contextmap.geometric_mapping import (
    GeometricMapArtifactManifest,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    MotionCorrectionPolicy,
    ScanDisposition,
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
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import (
    EstimatorProvenance,
    LookupPolicy,
    PoseEstimate,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryId,
    TrajectoryProvenance,
    calibration_identity,
    pose_estimate_id_for,
)

CLOCK_ID = "fixture:header"
SEQUENCE_NAME = "corridor-02"
RUN_ID = GeometricMapRunId("map-run-0001")
MAP_ID = f"{SEQUENCE_NAME}--{RUN_ID}"
SCAN_COUNT = 10
POINTS_PER_SCAN = 100
POINT_COUNT = SCAN_COUNT * POINTS_PER_SCAN
_SEQUENCE_ID = SequenceArtifactId("sequence-0001")
_TRAJECTORY_ID = TrajectoryId("run-0001--trajectory")
_MS = 1_000_000


def _timestamp(total_nanoseconds: int) -> SourceTimestamp:
    seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=CLOCK_ID)


def _cloud(points_per_scan: int) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (math.sin(i) * 3.0, math.cos(i) * 2.0, (i % 5) * 0.1) for i in range(points_per_scan)
    )


def _scan(index: int, points_per_scan: int) -> LidarObservation:
    fields = tuple(
        PointFieldDescriptor(name=name, offset_bytes=i * 4, data_type=PointFieldDataType.FLOAT32)
        for i, name in enumerate(("x", "y", "z"))
    )
    return LidarObservation(
        observation_id=SourceObservationId(f"scan-{index:04d}"),
        sensor_id=SensorId("velodyne"),
        frame_id=FrameId("lidar"),
        timestamp=_timestamp(index * 100 * _MS),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/lidar"),
        point_count=points_per_scan,
        point_step_bytes=12,
        fields=fields,
        data=b"".join(struct.pack("<3f", *point) for point in _cloud(points_per_scan)),
        is_dense=True,
    )


def build_geometry_artifact(
    workspace_root: Path, *, scans: int = SCAN_COUNT, points_per_scan: int = POINTS_PER_SCAN
) -> tuple[Path, GeometricMapArtifactManifest]:
    """Write a real GeometricMapArtifact under ``workspace_root`` and return it.

    Args:
        workspace_root: Where the artifact is written; this helper, not the writer, chooses the
            final run directory under it.

    Returns:
        The run directory and its manifest. The map is ``corridor-02--map-run-0001`` with
        ``scans * points_per_scan`` raw points (``POINT_COUNT`` by default) in the frame ``map``.
    """
    calibration = CalibrationSet(
        entries={},
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("body"),
                child_frame=FrameId("lidar"),
                translation=(0.5, 0.0, 0.25),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )
    poses = tuple(
        PoseEstimate(
            estimate_id=pose_estimate_id_for(trajectory_id=_TRAJECTORY_ID, index=index),
            timestamp=_timestamp(index * 100 * _MS),
            parent_frame=FrameId("map"),
            child_frame=FrameId("body"),
            translation_m=(float(index), 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            validity=PoseValidity.VALID,
            provenance=PoseProvenance(
                source_observation_ids=(SourceObservationId(f"pose-{index}"),)
            ),
        )
        for index in range(scans)
    )
    trajectory = Trajectory(
        trajectory_id=_TRAJECTORY_ID,
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
        poses=poses,
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(backend_id="fake_estimator", backend_version="0"),
            sequence_artifact_id=_SEQUENCE_ID,
            selection_id="full-sequence",
            calibration_identity=calibration_identity(calibration),
            code_version="test",
        ),
    )
    selection = FullSequenceSelection()
    plan = assemble_geometry_inputs(
        sequence=SequenceSelectionResult(
            sequence_artifact_id=_SEQUENCE_ID,
            selection=selection,
            selection_id=selection_identity(_SEQUENCE_ID, selection),
            observations=tuple(_scan(index, points_per_scan) for index in range(scans)),
        ),
        calibration=calibration,
        trajectory=trajectory,
        pose_lookup=LookupPolicy.interpolated(),
        motion_correction_policy=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.ACCEPT
        ),
        motion_correction=None,
        state_estimation_run_id=None,
    )
    run_dir = workspace_root / "geometric_mapping"
    manifest = GeometricMapArtifactWriter(
        output_dir=run_dir,
        sequence_name=SEQUENCE_NAME,
        run_id=RUN_ID,
        run_index=1,
    ).finalize(plan=plan, aggregation=None, code_version="test")
    return run_dir, manifest
