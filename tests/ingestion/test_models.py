import dataclasses

import pytest

from contextmap.ingestion import (
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp


def _timestamp(clock_id: str = "system") -> SourceTimestamp:
    return SourceTimestamp(seconds=1, nanoseconds=0, clock_id=clock_id)


def _provenance(**overrides: object) -> SourceProvenance:
    defaults: dict[str, object] = {
        "source_type": "ros1_bag",
        "source_path": "data/example.bag",
    }
    defaults.update(overrides)
    return SourceProvenance(**defaults)  # type: ignore[arg-type]


def test_image_observation_holds_required_fields() -> None:
    observation = ImageObservation(
        observation_id=SourceObservationId("frame-0001"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=_timestamp(),
        provenance=_provenance(source_topic="/camera/image_raw"),
        width=640,
        height=480,
        encoding=ImageEncoding.RGB8,
        data=b"\x00" * (640 * 480 * 3),
    )

    assert observation.width == 640
    assert observation.height == 480
    assert len(observation.data) == 640 * 480 * 3
    assert observation.calibration_id is None


def test_lidar_observation_holds_field_layout() -> None:
    fields = (
        PointFieldDescriptor(name="x", offset_bytes=0, data_type=PointFieldDataType.FLOAT32),
        PointFieldDescriptor(name="y", offset_bytes=4, data_type=PointFieldDataType.FLOAT32),
        PointFieldDescriptor(name="z", offset_bytes=8, data_type=PointFieldDataType.FLOAT32),
    )

    observation = LidarObservation(
        observation_id=SourceObservationId("scan-0001"),
        sensor_id=SensorId("velodyne_top"),
        frame_id=FrameId("velodyne"),
        timestamp=_timestamp(),
        provenance=_provenance(source_topic="/velodyne_points"),
        point_count=2,
        point_step_bytes=12,
        fields=fields,
        data=b"\x00" * 24,
    )

    assert observation.point_count == 2
    assert observation.fields == fields
    assert observation.is_dense is True


def test_imu_observation_represents_missing_orientation_as_none() -> None:
    observation = ImuObservation(
        observation_id=SourceObservationId("imu-0001"),
        sensor_id=SensorId("imu0"),
        frame_id=FrameId("imu_link"),
        timestamp=_timestamp(),
        provenance=_provenance(source_topic="/imu/data"),
        linear_acceleration=(0.0, 0.0, 9.81),
        angular_velocity=(0.0, 0.0, 0.0),
        orientation=None,
    )

    assert observation.orientation is None
    assert observation.linear_acceleration == (0.0, 0.0, 9.81)


def test_external_pose_measurement_is_not_a_state_estimate() -> None:
    measurement = ExternalPoseMeasurement(
        observation_id=SourceObservationId("odom-0001"),
        sensor_id=SensorId("wheel_odometry"),
        frame_id=FrameId("base_link"),
        timestamp=_timestamp(),
        provenance=_provenance(source_topic="/odom"),
        parent_frame=FrameId("odom"),
        translation=(1.0, 2.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )

    # Ownership rule: an ExternalPoseMeasurement is an input measurement,
    # never a substitute for a canonical PoseEstimate.
    assert type(measurement).__name__ != "PoseEstimate"
    assert measurement.parent_frame != measurement.frame_id


@pytest.mark.parametrize(
    "observation_factory",
    [
        lambda: ImageObservation(
            observation_id=SourceObservationId("frame-0001"),
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            timestamp=_timestamp(),
            provenance=_provenance(),
            width=1,
            height=1,
            encoding=ImageEncoding.MONO8,
            data=b"\x00",
        ),
        lambda: ImuObservation(
            observation_id=SourceObservationId("imu-0001"),
            sensor_id=SensorId("imu0"),
            frame_id=FrameId("imu_link"),
            timestamp=_timestamp(),
            provenance=_provenance(),
        ),
    ],
)
def test_source_observations_are_immutable(observation_factory: object) -> None:
    observation: SourceObservation = observation_factory()  # type: ignore[operator]

    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.sensor_id = SensorId("other")  # type: ignore[misc]


def test_downstream_code_consumes_mixed_modalities_without_branching() -> None:
    """Same code path consumes observations regardless of originating source."""

    def source_identity(observation: SourceObservation) -> tuple[str, str]:
        return observation.observation_id, observation.sensor_id

    ros1_image = ImageObservation(
        observation_id=SourceObservationId("frame-0001"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=_timestamp(),
        provenance=_provenance(source_type="ros1_bag"),
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00\x00\x00",
    )
    ros2_image = ImageObservation(
        observation_id=SourceObservationId("frame-0002"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=_timestamp(),
        provenance=_provenance(source_type="ros2_bag"),
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00\x00\x00",
    )

    identities = [source_identity(obs) for obs in (ros1_image, ros2_image)]

    assert identities == [
        ("frame-0001", "front_camera"),
        ("frame-0002", "front_camera"),
    ]


def test_provenance_defaults_have_no_topic_or_metadata() -> None:
    provenance = _provenance()

    assert provenance.source_topic is None
    assert provenance.source_message_index is None
    assert provenance.raw_metadata == {}
