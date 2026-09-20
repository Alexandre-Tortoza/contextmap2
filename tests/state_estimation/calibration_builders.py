"""Deterministic calibration and sensor-observation builders for geometry preflight tests."""

from __future__ import annotations

import math

from pose_builders import CLOCK_ID, timestamp_ns

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    PinholeCameraModel,
    RigidTransform,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import (
    Quaternion,
    Vector3,
    compose_rigid,
    invert_rigid,
)

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


def calibration(
    *transforms: RigidTransform,
    entries: dict[CalibrationReferenceId, CalibrationEntry] | None = None,
) -> CalibrationSet:
    """Build a calibration set with the given static transforms."""
    return CalibrationSet(entries=entries or {}, static_transforms=tuple(transforms))


def camera_entry(calibration_id: str, *, frame: str = "camera") -> CalibrationEntry:
    """Build a calibration entry that carries a pinhole camera model."""
    return CalibrationEntry(
        calibration_id=CalibrationReferenceId(calibration_id),
        sensor_id=SensorId("camera0"),
        frame_id=FrameId(frame),
        camera_model=PinholeCameraModel(width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0),
        provenance=CalibrationProvenance(source_type="fixture", source_path="fixtures/camera"),
        content_hash=f"sha256:{calibration_id}",
    )


def _provenance(topic: str) -> SourceProvenance:
    return SourceProvenance(
        source_type="fixture", source_path="fixtures/sensors", source_topic=topic
    )


def lidar(
    observation_id: str = "scan-0001", *, frame: str = "lidar", clock_id: str = CLOCK_ID
) -> LidarObservation:
    """Build an empty LiDAR scan in ``frame``."""
    return LidarObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId(f"sensor-{frame}"),
        frame_id=FrameId(frame),
        timestamp=timestamp_ns(0, clock_id=clock_id),
        provenance=_provenance("/lidar"),
        point_count=0,
        point_step_bytes=16,
        fields=(),
        data=b"",
    )


def imu(
    observation_id: str = "imu-0001", *, frame: str = "imu", clock_id: str = CLOCK_ID
) -> ImuObservation:
    """Build an IMU measurement in ``frame``."""
    return ImuObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId(f"sensor-{frame}"),
        frame_id=FrameId(frame),
        timestamp=timestamp_ns(0, clock_id=clock_id),
        provenance=_provenance("/imu"),
    )


def image(
    observation_id: str = "frame-0001",
    *,
    frame: str = "camera",
    calibration_id: str | None = None,
    clock_id: str = CLOCK_ID,
) -> ImageObservation:
    """Build a 1x1 image in ``frame`` optionally tied to a calibration entry."""
    return ImageObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId(f"sensor-{frame}"),
        frame_id=FrameId(frame),
        timestamp=timestamp_ns(0, clock_id=clock_id),
        provenance=_provenance("/camera"),
        calibration_id=None if calibration_id is None else CalibrationReferenceId(calibration_id),
        width=1,
        height=1,
        encoding=ImageEncoding.MONO8,
        data=b"\x00",
    )


def closing_transform(
    first: RigidTransform, second: RigidTransform, *, parent: str, child: str
) -> RigidTransform:
    """Build the transform that closes a loop consistently.

    ``first`` and ``second`` share a parent frame; the result is
    ``T_first.child_second.child = inverse(first) * second``.
    """
    inverse_translation, inverse_rotation = invert_rigid(
        translation=first.translation, rotation=first.rotation
    )
    translation, rotation = compose_rigid(
        outer_translation=inverse_translation,
        outer_rotation=inverse_rotation,
        inner_translation=second.translation,
        inner_rotation=second.rotation,
    )
    return rigid(parent, child, translation, rotation)
