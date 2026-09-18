"""Shared decode logic for the rosbags-based ROS 1 and ROS 2 source adapters.

``Ros1BagSourceAdapter`` and ``Ros2BagSourceAdapter`` decode the same
underlying ``sensor_msgs``/``nav_msgs`` message shapes — ``rosbags``
normalizes both ROS distributions to the same field names for every
message used here, except ``sensor_msgs/CameraInfo``'s ``D``/``K`` (ROS 1)
vs ``d``/``k`` (ROS 2) casing. That one real difference stays in each
adapter; everything else that is genuinely identical between the two ROS
versions is defined once, here, instead of duplicated per adapter.

Private to ``contextmap.ingestion.adapters``: both call sites are within
the same capability, so importing this module across them is not a
cross-capability access and is not part of the public adapter contract.
"""

from __future__ import annotations

from typing import Any

from contextmap.ingestion.calibration import (
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    PinholeCameraModel,
)
from contextmap.ingestion.models import (
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp

IMAGE_ENCODING_MAP: dict[str, ImageEncoding] = {
    "rgb8": ImageEncoding.RGB8,
    "bgr8": ImageEncoding.BGR8,
    "mono8": ImageEncoding.MONO8,
    "mono16": ImageEncoding.MONO16,
}

POINTFIELD_TYPE_MAP: dict[int, PointFieldDataType] = {
    1: PointFieldDataType.INT8,
    2: PointFieldDataType.UINT8,
    3: PointFieldDataType.INT16,
    4: PointFieldDataType.UINT16,
    5: PointFieldDataType.INT32,
    6: PointFieldDataType.UINT32,
    7: PointFieldDataType.FLOAT32,
    8: PointFieldDataType.FLOAT64,
}

PINHOLE_DISTORTION_MODEL_MAP: dict[str, DistortionModel] = {
    "plumb_bob": DistortionModel.PLUMB_BOB,
    "rational_polynomial": DistortionModel.RATIONAL_POLYNOMIAL,
}


def sanitize_topic(topic: str) -> str:
    """Turn a ROS topic name into an identifier usable as a sensor/observation ID.

    Args:
        topic: A ROS topic name, e.g. ``"/camera/image_raw"``.

    Returns:
        The topic with leading/trailing slashes stripped and internal
        slashes replaced by underscores, e.g. ``"camera_image_raw"``.
    """
    return topic.strip("/").replace("/", "_")


def stamp_to_timestamp(stamp: Any, *, clock_id: str) -> SourceTimestamp:
    """Convert a ROS ``builtin_interfaces/Time``-shaped stamp into a canonical timestamp.

    Args:
        stamp: A decoded message's ``header.stamp`` (has ``.sec``/``.nanosec``).
        clock_id: Clock domain identity to attach, e.g. ``"ros1_bag:/imu/data"``.

    Returns:
        The canonical timestamp.
    """
    return SourceTimestamp(seconds=stamp.sec, nanoseconds=stamp.nanosec, clock_id=clock_id)


def decode_image(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> ImageObservation:
    """Decode a ``sensor_msgs/Image`` into a canonical :class:`ImageObservation`.

    Args:
        message: A decoded ``sensor_msgs/Image`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.

    Returns:
        The canonical image observation.

    Raises:
        ValueError: If ``message.encoding`` is not a supported encoding.
    """
    encoding = IMAGE_ENCODING_MAP.get(message.encoding)
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


def decode_lidar(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> LidarObservation:
    """Decode a ``sensor_msgs/PointCloud2`` into a canonical :class:`LidarObservation`.

    Args:
        message: A decoded ``sensor_msgs/PointCloud2`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.

    Returns:
        The canonical LiDAR observation.
    """
    fields = tuple(
        PointFieldDescriptor(
            name=field.name,
            offset_bytes=field.offset,
            data_type=POINTFIELD_TYPE_MAP[field.datatype],
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


def decode_imu(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> ImuObservation:
    """Decode a ``sensor_msgs/Imu`` into a canonical :class:`ImuObservation`.

    Each of ``linear_acceleration``/``angular_velocity``/``orientation`` is
    ``None`` when its own covariance's first element is ``-1.0`` — the
    ``sensor_msgs/Imu`` convention for "not provided" — checked
    independently per field.

    Args:
        message: A decoded ``sensor_msgs/Imu`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.

    Returns:
        The canonical IMU observation.
    """
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


def decode_pose(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
) -> ExternalPoseMeasurement:
    """Decode a ``nav_msgs/Odometry`` into a canonical :class:`ExternalPoseMeasurement`.

    ``header.frame_id``/``child_frame_id`` map to
    ``parent_frame``/``frame_id``, matching the ownership rule that an
    externally supplied pose is an input measurement, never a State
    Estimation ``PoseEstimate``.

    Args:
        message: A decoded ``nav_msgs/Odometry`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.

    Returns:
        The canonical external pose measurement.
    """
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


def build_camera_model(
    *,
    width: int,
    height: int,
    k_matrix: Any,
    distortion_model_name: str,
    distortion_coefficients: Any,
) -> CameraModel:
    """Build a canonical camera model from primitive ``CameraInfo`` values.

    Callers extract ``K``/``D`` (ROS 1) or ``k``/``d`` (ROS 2) themselves,
    since that field-name casing is the one real difference between the
    two ROS versions' ``sensor_msgs/CameraInfo``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        k_matrix: The 9-element row-major intrinsic matrix.
        distortion_model_name: ``CameraInfo.distortion_model``.
        distortion_coefficients: The distortion coefficient array.

    Returns:
        A :class:`~contextmap.ingestion.calibration.FisheyeCameraModel` when
        ``distortion_model_name == "equidistant"``, otherwise a
        :class:`~contextmap.ingestion.calibration.PinholeCameraModel`
        (falling back to :attr:`DistortionModel.NONE` for an unrecognized
        model name, leaving validation to catch the resulting
        inconsistency rather than raising here).
    """
    fx = float(k_matrix[0])
    fy = float(k_matrix[4])
    cx = float(k_matrix[2])
    cy = float(k_matrix[5])

    if distortion_model_name == "equidistant":
        coefficients = [float(value) for value in distortion_coefficients[:4]]
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

    distortion_model = PINHOLE_DISTORTION_MODEL_MAP.get(distortion_model_name, DistortionModel.NONE)
    return PinholeCameraModel(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_model=distortion_model,
        distortion_coefficients=tuple(float(value) for value in distortion_coefficients),
    )
