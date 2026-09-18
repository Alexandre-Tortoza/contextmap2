"""Deterministic ingestion fixtures shared across ingestion tests.

A small, fully in-memory multimodal sequence (RGB + point cloud + IMU +
external pose) plus representative calibration, and intentionally invalid
variants for the major validation paths this milestone defines. Committed
as Python, not a binary fixture file, so it stays inspectable as code and
needs no external download — the same approach already used by the ROS 1
and ROS 2 adapter tests (#44, #45), which build a synthetic bag at test
time instead of checking one in.
"""

from __future__ import annotations

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    PinholeCameraModel,
    PointFieldDataType,
    PointFieldDescriptor,
    RigidTransform,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp


def _clock_id(topic: str) -> str:
    # One clock_id per topic, matching how Ros1BagSourceAdapter/Ros2BagSourceAdapter
    # actually assign clock_id (f"{source_type}:{topic}") — different topics are
    # independent timelines and are never expected to interleave monotonically.
    return f"fixture:{topic}"


def _timestamp(seconds: float, topic: str) -> SourceTimestamp:
    whole = int(seconds)
    nanoseconds = round((seconds - whole) * 1_000_000_000)
    return SourceTimestamp(seconds=whole, nanoseconds=nanoseconds, clock_id=_clock_id(topic))


def _provenance(topic: str) -> SourceProvenance:
    return SourceProvenance(
        source_type="fixture", source_path="fixtures/corridor", source_topic=topic
    )


def build_valid_sequence() -> list[SourceObservation]:
    """A deterministic, valid multimodal sequence: 2 images, 1 LiDAR scan, 1 IMU, 1 pose."""
    return [
        ImageObservation(
            observation_id=SourceObservationId("frame-0001"),
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            timestamp=_timestamp(1.0, "/camera/image_raw"),
            provenance=_provenance("/camera/image_raw"),
            width=2,
            height=1,
            encoding=ImageEncoding.RGB8,
            data=b"\x01\x02\x03\x04\x05\x06",
        ),
        ImageObservation(
            observation_id=SourceObservationId("frame-0002"),
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            timestamp=_timestamp(2.0, "/camera/image_raw"),
            provenance=_provenance("/camera/image_raw"),
            width=2,
            height=1,
            encoding=ImageEncoding.RGB8,
            data=b"\x07\x08\x09\x0a\x0b\x0c",
        ),
        LidarObservation(
            observation_id=SourceObservationId("scan-0001"),
            sensor_id=SensorId("velodyne_top"),
            frame_id=FrameId("velodyne"),
            timestamp=_timestamp(1.0, "/velodyne_points"),
            provenance=_provenance("/velodyne_points"),
            point_count=1,
            point_step_bytes=12,
            fields=(
                PointFieldDescriptor(
                    name="x", offset_bytes=0, data_type=PointFieldDataType.FLOAT32
                ),
                PointFieldDescriptor(
                    name="y", offset_bytes=4, data_type=PointFieldDataType.FLOAT32
                ),
                PointFieldDescriptor(
                    name="z", offset_bytes=8, data_type=PointFieldDataType.FLOAT32
                ),
            ),
            data=b"\x00" * 12,
        ),
        ImuObservation(
            observation_id=SourceObservationId("imu-0001"),
            sensor_id=SensorId("imu0"),
            frame_id=FrameId("imu_link"),
            timestamp=_timestamp(1.0, "/imu/data"),
            provenance=_provenance("/imu/data"),
            linear_acceleration=(0.0, 0.0, 9.81),
            angular_velocity=(0.0, 0.0, 0.0),
            orientation=None,
        ),
        ExternalPoseMeasurement(
            observation_id=SourceObservationId("odom-0001"),
            sensor_id=SensorId("wheel_odometry"),
            frame_id=FrameId("base_link"),
            timestamp=_timestamp(1.0, "/odom"),
            provenance=_provenance("/odom"),
            parent_frame=FrameId("odom"),
            translation=(1.0, 2.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        ),
    ]


def build_calibration_set() -> CalibrationSet:
    """Representative calibration/frame metadata matching :func:`build_valid_sequence`'s frames.

    Covers every ``frame_id`` used by :func:`build_valid_sequence`
    (``front_camera_optical`` via a camera entry, ``velodyne`` via a
    camera-model-less entry, ``imu_link``/``odom`` via static transforms
    to ``base_link``), so
    ``validate_frame_references(build_valid_sequence(), build_calibration_set())``
    reports no problem.
    """
    camera_model = PinholeCameraModel(width=2, height=1, fx=1.0, fy=1.0, cx=1.0, cy=0.5)
    provenance = CalibrationProvenance(source_type="fixture", source_path="fixtures/calib.yaml")
    camera_entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId("front_camera-calib"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        camera_model=camera_model,
        provenance=provenance,
        content_hash="sha256:fixture-camera",
    )
    lidar_entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId("velodyne_top-calib"),
        sensor_id=SensorId("velodyne_top"),
        frame_id=FrameId("velodyne"),
        camera_model=None,
        provenance=provenance,
        content_hash="sha256:fixture-lidar",
    )
    return CalibrationSet(
        entries={
            camera_entry.calibration_id: camera_entry,
            lidar_entry.calibration_id: lidar_entry,
        },
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("base_link"),
                child_frame=FrameId("imu_link"),
                translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            RigidTransform(
                parent_frame=FrameId("odom"),
                child_frame=FrameId("base_link"),
                translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )


def build_image_with_data_size_mismatch() -> ImageObservation:
    """Intentionally invalid: ``data`` length does not match width*height*bytes-per-pixel."""
    return ImageObservation(
        observation_id=SourceObservationId("frame-bad"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=_timestamp(1.0, "/camera/image_raw"),
        provenance=_provenance("/camera/image_raw"),
        width=4,
        height=4,
        encoding=ImageEncoding.RGB8,
        data=b"\x00\x01\x02",
    )


def build_lidar_with_inconsistent_fields() -> LidarObservation:
    """Intentionally invalid: a field's ``offset_bytes`` falls outside ``point_step_bytes``."""
    return LidarObservation(
        observation_id=SourceObservationId("scan-bad"),
        sensor_id=SensorId("velodyne_top"),
        frame_id=FrameId("velodyne"),
        timestamp=_timestamp(1.0, "/velodyne_points"),
        provenance=_provenance("/velodyne_points"),
        point_count=1,
        point_step_bytes=8,
        fields=(
            PointFieldDescriptor(name="x", offset_bytes=0, data_type=PointFieldDataType.FLOAT32),
            PointFieldDescriptor(name="y", offset_bytes=20, data_type=PointFieldDataType.FLOAT32),
        ),
        data=b"\x00" * 8,
    )


def build_non_monotonic_observations() -> list[SourceObservation]:
    """Intentionally invalid: second image's timestamp precedes the first's, same clock."""
    return [
        ImageObservation(
            observation_id=SourceObservationId("frame-a"),
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            timestamp=_timestamp(2.0, "/camera/image_raw"),
            provenance=_provenance("/camera/image_raw"),
            width=1,
            height=1,
            encoding=ImageEncoding.RGB8,
            data=b"\x00\x00\x00",
        ),
        ImageObservation(
            observation_id=SourceObservationId("frame-b"),
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            timestamp=_timestamp(1.0, "/camera/image_raw"),
            provenance=_provenance("/camera/image_raw"),
            width=1,
            height=1,
            encoding=ImageEncoding.RGB8,
            data=b"\x00\x00\x00",
        ),
    ]


def build_observation_with_unknown_frame() -> ImageObservation:
    """Intentionally invalid: ``frame_id`` absent from :func:`build_calibration_set`."""
    return ImageObservation(
        observation_id=SourceObservationId("frame-unknown-frame"),
        sensor_id=SensorId("side_camera"),
        frame_id=FrameId("side_camera_optical"),
        timestamp=_timestamp(1.0, "/side_camera/image_raw"),
        provenance=_provenance("/side_camera/image_raw"),
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00\x00\x00",
    )
