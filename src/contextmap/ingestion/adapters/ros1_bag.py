"""ROS 1 bag source adapter.

Reads a ROS 1 bag and decodes the topics named in the adapter's
:class:`~contextmap.ingestion.source_adapter.SourceAdapterConfig` into
canonical :data:`~contextmap.ingestion.models.SourceObservation` instances,
never exposing a ROS message object across the
:class:`~contextmap.ingestion.source_adapter.SourceAdapter` boundary. Uses
the pure-Python ``rosbags`` library (optional dependency, extra
``contextmap[ros1]``), so no ROS 1 installation is required — see
``src/contextmap/ingestion/docs/adapters.md``.

v0 scope: RGB (``sensor_msgs/Image``), LiDAR (``sensor_msgs/PointCloud2``),
IMU (``sensor_msgs/Imu``), external pose (``nav_msgs/Odometry``), and
camera calibration (``sensor_msgs/CameraInfo``, via
:meth:`Ros1BagSourceAdapter.read_calibration`). Static/dynamic TF
(``tf2_msgs/TFMessage``) is not decoded in v0.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from contextmap.ingestion.calibration import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationSet,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    PinholeCameraModel,
    compute_content_hash,
)
from contextmap.ingestion.models import (
    CalibrationReferenceId,
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
from contextmap.ingestion.source_adapter import (
    MissingRequiredTopicError,
    SourceAdapterCapabilities,
    SourceAdapterConfig,
    SourceAdapterWarning,
)
from contextmap.shared import SourceTimestamp

_IMAGE_ENCODING_MAP: dict[str, ImageEncoding] = {
    "rgb8": ImageEncoding.RGB8,
    "bgr8": ImageEncoding.BGR8,
    "mono8": ImageEncoding.MONO8,
    "mono16": ImageEncoding.MONO16,
}

_POINTFIELD_TYPE_MAP: dict[int, PointFieldDataType] = {
    1: PointFieldDataType.INT8,
    2: PointFieldDataType.UINT8,
    3: PointFieldDataType.INT16,
    4: PointFieldDataType.UINT16,
    5: PointFieldDataType.INT32,
    6: PointFieldDataType.UINT32,
    7: PointFieldDataType.FLOAT32,
    8: PointFieldDataType.FLOAT64,
}

_PINHOLE_DISTORTION_MODEL_MAP: dict[str, DistortionModel] = {
    "plumb_bob": DistortionModel.PLUMB_BOB,
    "rational_polynomial": DistortionModel.RATIONAL_POLYNOMIAL,
}


class Ros1BagSourceAdapter:
    """Reads a ROS 1 bag and yields canonical source observations."""

    def __init__(self, config: SourceAdapterConfig) -> None:
        """Create an adapter for a configured ROS 1 bag.

        Args:
            config: Adapter configuration (path + topic mapping). By
                convention ``config.source_type`` is ``"ros1_bag"``, but
                this is not validated here.
        """
        self._config = config
        self._typestore = get_typestore(Stores.ROS1_NOETIC)
        self._warnings: list[SourceAdapterWarning] = []

    def capabilities(self) -> SourceAdapterCapabilities:
        """Report which configured modalities are actually present in the bag.

        Returns:
            Capabilities reflecting which configured topics exist in the bag.
        """
        available = self._available_topics()
        mapping = self._config.topics
        return SourceAdapterCapabilities(
            rgb=mapping.rgb is not None and mapping.rgb in available,
            lidar=mapping.lidar is not None and mapping.lidar in available,
            imu=mapping.imu is not None and mapping.imu in available,
            external_pose=mapping.pose is not None and mapping.pose in available,
            calibration=mapping.camera_info is not None and mapping.camera_info in available,
        )

    def read_observations(self) -> Iterator[SourceObservation]:
        """Decode the bag and yield canonical observations.

        Yields:
            One :data:`~contextmap.ingestion.models.SourceObservation` per
            successfully decoded message, in bag order.

        Raises:
            MissingRequiredTopicError: If a topic named in
                ``config.required_topics`` is absent from the bag.
        """
        available = self._available_topics()
        self._check_required_topics(available)

        topic_kinds = self._configured_topic_kinds()
        self._warnings = []
        counters: dict[str, int] = {}

        with Reader(self._config.path) as reader:
            connections = [
                connection for connection in reader.connections if connection.topic in topic_kinds
            ]
            for connection, _bag_timestamp, rawdata in reader.messages(connections=connections):
                topic = connection.topic
                index = counters.get(topic, 0)
                counters[topic] = index + 1
                try:
                    message = self._typestore.deserialize_ros1(rawdata, connection.msgtype)
                    observation = self._decode(
                        kind=topic_kinds[topic],
                        topic=topic,
                        msgtype=connection.msgtype,
                        index=index,
                        message=message,
                    )
                except (KeyError, ValueError, AttributeError) as error:
                    self._warnings.append(
                        SourceAdapterWarning(topic=topic, message_index=index, reason=str(error))
                    )
                    continue
                yield observation

    def warnings(self) -> Sequence[SourceAdapterWarning]:
        """Return warnings from the most recent :meth:`read_observations` call.

        Returns:
            Accumulated warnings, in the order they occurred.
        """
        return tuple(self._warnings)

    def read_calibration(self) -> CalibrationSet | None:
        """Decode camera calibration from the configured ``camera_info`` topic.

        Uses the last message on the topic: intrinsics are treated as
        static for the whole bag in v0. Static/dynamic TF extrinsics are
        not decoded in v0, so the result never has ``static_transforms``.

        Returns:
            A calibration set with one entry, or ``None`` when no
            ``camera_info`` topic is configured or it has no messages.
        """
        topic = self._config.topics.camera_info
        if topic is None:
            return None

        last_message: Any = None
        with Reader(self._config.path) as reader:
            connections = [
                connection for connection in reader.connections if connection.topic == topic
            ]
            for connection, _timestamp, rawdata in reader.messages(connections=connections):
                last_message = self._typestore.deserialize_ros1(rawdata, connection.msgtype)
        if last_message is None:
            return None

        sensor_id = SensorId(_sanitize_topic(self._config.topics.rgb or topic))
        frame_id = FrameId(last_message.header.frame_id)
        camera_model = _decode_camera_model(last_message)
        calibration_id = CalibrationReferenceId(f"{sensor_id}-calib")
        entry = CalibrationEntry(
            calibration_id=calibration_id,
            sensor_id=sensor_id,
            frame_id=frame_id,
            camera_model=camera_model,
            provenance=CalibrationProvenance(
                source_type="ros1_bag",
                source_path=self._config.path,
                original_values={
                    "distortion_model": last_message.distortion_model,
                    "D": [float(value) for value in last_message.D],
                },
            ),
            content_hash=compute_content_hash(
                sensor_id=sensor_id, frame_id=frame_id, camera_model=camera_model
            ),
        )
        return CalibrationSet(entries={calibration_id: entry}, static_transforms=())

    def _available_topics(self) -> frozenset[str]:
        with Reader(self._config.path) as reader:
            return frozenset(connection.topic for connection in reader.connections)

    def _configured_topic_kinds(self) -> dict[str, str]:
        mapping = self._config.topics
        kinds: dict[str, str] = {}
        for kind, topic in (
            ("rgb", mapping.rgb),
            ("lidar", mapping.lidar),
            ("imu", mapping.imu),
            ("pose", mapping.pose),
        ):
            if topic is not None:
                kinds[topic] = kind
        return kinds

    def _check_required_topics(self, available: frozenset[str]) -> None:
        mapping = self._config.topics
        topic_by_field = {
            "rgb": mapping.rgb,
            "camera_info": mapping.camera_info,
            "lidar": mapping.lidar,
            "imu": mapping.imu,
            "pose": mapping.pose,
        }
        missing = [
            field_name
            for field_name in self._config.required_topics
            if topic_by_field[field_name] is None or topic_by_field[field_name] not in available
        ]
        if missing:
            raise MissingRequiredTopicError(
                f"required topics missing from ros1 bag {self._config.path!r}: {sorted(missing)}"
            )

    def _decode(
        self, *, kind: str, topic: str, msgtype: str, index: int, message: Any
    ) -> SourceObservation:
        observation_id = SourceObservationId(f"{_sanitize_topic(topic)}-{index:06d}")
        sensor_id = SensorId(_sanitize_topic(topic))
        provenance = SourceProvenance(
            source_type="ros1_bag",
            source_path=self._config.path,
            source_topic=topic,
            source_message_index=index,
            raw_metadata={"msgtype": msgtype},
        )
        timestamp = _stamp_to_timestamp(message.header.stamp, topic)

        if kind == "rgb":
            return _decode_image(message, observation_id, sensor_id, timestamp, provenance)
        if kind == "lidar":
            return _decode_lidar(message, observation_id, sensor_id, timestamp, provenance)
        if kind == "imu":
            return _decode_imu(message, observation_id, sensor_id, timestamp, provenance)
        if kind == "pose":
            return _decode_pose(message, observation_id, sensor_id, timestamp, provenance)
        raise ValueError(f"unhandled configured topic kind: {kind!r}")


def _sanitize_topic(topic: str) -> str:
    return topic.strip("/").replace("/", "_")


def _stamp_to_timestamp(stamp: Any, topic: str) -> SourceTimestamp:
    return SourceTimestamp(
        seconds=stamp.sec, nanoseconds=stamp.nanosec, clock_id=f"ros1_bag:{topic}"
    )


def _decode_image(
    message: Any,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> ImageObservation:
    encoding = _IMAGE_ENCODING_MAP.get(message.encoding)
    if encoding is None:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}")
    return ImageObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=provenance,
        width=message.width,
        height=message.height,
        encoding=encoding,
        data=message.data.tobytes(),
    )


def _decode_lidar(
    message: Any,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> LidarObservation:
    fields = tuple(
        PointFieldDescriptor(
            name=field.name,
            offset_bytes=field.offset,
            data_type=_POINTFIELD_TYPE_MAP[field.datatype],
            count=field.count,
        )
        for field in message.fields
    )
    return LidarObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=provenance,
        point_count=message.width * message.height,
        point_step_bytes=message.point_step,
        fields=fields,
        is_dense=bool(message.is_dense),
        data=message.data.tobytes(),
    )


def _decode_imu(
    message: Any,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> ImuObservation:
    linear_acceleration = (
        None
        if message.linear_acceleration_covariance[0] == -1.0
        else (
            float(message.linear_acceleration.x),
            float(message.linear_acceleration.y),
            float(message.linear_acceleration.z),
        )
    )
    angular_velocity = (
        None
        if message.angular_velocity_covariance[0] == -1.0
        else (
            float(message.angular_velocity.x),
            float(message.angular_velocity.y),
            float(message.angular_velocity.z),
        )
    )
    orientation = (
        None
        if message.orientation_covariance[0] == -1.0
        else (
            float(message.orientation.x),
            float(message.orientation.y),
            float(message.orientation.z),
            float(message.orientation.w),
        )
    )
    return ImuObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=provenance,
        linear_acceleration=linear_acceleration,
        angular_velocity=angular_velocity,
        orientation=orientation,
    )


def _decode_pose(
    message: Any,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> ExternalPoseMeasurement:
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    return ExternalPoseMeasurement(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.child_frame_id),
        timestamp=timestamp,
        provenance=provenance,
        parent_frame=FrameId(message.header.frame_id),
        translation=(float(position.x), float(position.y), float(position.z)),
        orientation=(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ),
    )


def _decode_camera_model(message: Any) -> CameraModel:
    width = message.width
    height = message.height
    fx = float(message.K[0])
    fy = float(message.K[4])
    cx = float(message.K[2])
    cy = float(message.K[5])

    if message.distortion_model == "equidistant":
        coefficients = [float(value) for value in message.D[:4]]
        coefficients += [0.0] * (4 - len(coefficients))
        k1, k2, k3, k4 = coefficients
        return FisheyeCameraModel(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_coefficients=(k1, k2, k3, k4),
        )

    distortion_model = _PINHOLE_DISTORTION_MODEL_MAP.get(
        message.distortion_model, DistortionModel.NONE
    )
    return PinholeCameraModel(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_model=distortion_model,
        distortion_coefficients=tuple(float(value) for value in message.D),
    )
