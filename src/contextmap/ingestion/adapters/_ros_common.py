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

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from contextmap.ingestion.calibration import (
    CalibrationEntry,
    CalibrationError,
    CalibrationSet,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    PinholeCameraModel,
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
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.source_adapter import InvalidSourceWindowError, SourceAdapterConfig
from contextmap.shared import SourceTimestamp

IMAGE_ENCODING_MAP: dict[str, ImageEncoding] = {
    "rgb8": ImageEncoding.RGB8,
    "bgr8": ImageEncoding.BGR8,
    "mono8": ImageEncoding.MONO8,
    "mono16": ImageEncoding.MONO16,
}

IMAGE_BYTES_PER_PIXEL: dict[ImageEncoding, int] = {
    ImageEncoding.RGB8: 3,
    ImageEncoding.BGR8: 3,
    ImageEncoding.MONO8: 1,
    ImageEncoding.MONO16: 2,
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

POINTFIELD_SIZE_BYTES: dict[PointFieldDataType, int] = {
    PointFieldDataType.INT8: 1,
    PointFieldDataType.UINT8: 1,
    PointFieldDataType.INT16: 2,
    PointFieldDataType.UINT16: 2,
    PointFieldDataType.INT32: 4,
    PointFieldDataType.UINT32: 4,
    PointFieldDataType.FLOAT32: 4,
    PointFieldDataType.FLOAT64: 8,
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
    calibration_id: CalibrationReferenceId | None = None,
) -> ImageObservation:
    """Decode a ``sensor_msgs/Image`` into a canonical :class:`ImageObservation`.

    Args:
        message: A decoded ``sensor_msgs/Image`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.
        calibration_id: Applicable canonical calibration reference, when available.

    Returns:
        The canonical image observation.

    Raises:
        ValueError: If ``message.encoding`` is not a supported encoding.
    """
    encoding = IMAGE_ENCODING_MAP.get(message.encoding)
    if encoding is None:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}")
    row_payload_size = int(message.width) * IMAGE_BYTES_PER_PIXEL[encoding]
    data = _remove_row_padding(
        message.data.tobytes(),
        height=int(message.height),
        row_step=int(message.step),
        row_payload_size=row_payload_size,
        payload_name="image",
    )
    source_is_bigendian = bool(message.is_bigendian)
    byte_order_normalized = source_is_bigendian and encoding is ImageEncoding.MONO16
    if byte_order_normalized:
        data = _swap_fixed_width_values(data, value_size=2)
    normalized_provenance = _with_raw_metadata(
        provenance,
        source_step_bytes=int(message.step),
        source_is_bigendian=source_is_bigendian,
        row_padding_removed=int(message.step) != row_payload_size,
        byte_order_normalized=byte_order_normalized,
        canonical_byte_order="little",
    )
    return ImageObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=normalized_provenance,
        calibration_id=calibration_id,
        width=message.width,
        height=message.height,
        encoding=encoding,
        data=data,
    )


def decode_lidar(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
    calibration_id: CalibrationReferenceId | None = None,
) -> LidarObservation:
    """Decode a ``sensor_msgs/PointCloud2`` into a canonical :class:`LidarObservation`.

    Args:
        message: A decoded ``sensor_msgs/PointCloud2`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.
        calibration_id: Applicable canonical calibration reference, when available.

    Returns:
        The canonical LiDAR observation.

    Raises:
        ValueError: If a ``PointField.datatype`` is outside the supported
            ``sensor_msgs/PointField`` datatypes.
    """
    fields = tuple(
        PointFieldDescriptor(
            name=field.name,
            offset_bytes=field.offset,
            data_type=_point_field_data_type(field),
            count=field.count,
        )
        for field in message.fields
    )
    row_payload_size = int(message.width) * int(message.point_step)
    data = _remove_row_padding(
        message.data.tobytes(),
        height=int(message.height),
        row_step=int(message.row_step),
        row_payload_size=row_payload_size,
        payload_name="point cloud",
    )
    source_is_bigendian = bool(message.is_bigendian)
    if source_is_bigendian:
        data = _normalize_point_field_byte_order(
            data,
            point_step=int(message.point_step),
            fields=fields,
        )
    normalized_provenance = _with_raw_metadata(
        provenance,
        source_row_step_bytes=int(message.row_step),
        source_is_bigendian=source_is_bigendian,
        row_padding_removed=int(message.row_step) != row_payload_size,
        byte_order_normalized=source_is_bigendian,
        canonical_byte_order="little",
    )
    return LidarObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=normalized_provenance,
        calibration_id=calibration_id,
        point_count=message.width * message.height,
        point_step_bytes=message.point_step,
        fields=fields,
        is_dense=bool(message.is_dense),
        data=data,
    )


def decode_imu(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
    calibration_id: CalibrationReferenceId | None = None,
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
        calibration_id: Applicable canonical calibration reference, when available.

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
        calibration_id=calibration_id,
        linear_acceleration=linear_acceleration,
        angular_velocity=angular_velocity,
        orientation=orientation,
        linear_acceleration_covariance=_available_covariance(
            message.linear_acceleration_covariance
        ),
        angular_velocity_covariance=_available_covariance(message.angular_velocity_covariance),
        orientation_covariance=_available_covariance(message.orientation_covariance),
    )


def decode_pose(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
    calibration_id: CalibrationReferenceId | None = None,
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
        calibration_id: Applicable canonical calibration reference, when available.

    Returns:
        The canonical external pose measurement.
    """
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    linear_velocity = message.twist.twist.linear
    angular_velocity = message.twist.twist.angular
    return ExternalPoseMeasurement(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.child_frame_id),
        timestamp=timestamp,
        provenance=provenance,
        calibration_id=calibration_id,
        parent_frame=FrameId(message.header.frame_id),
        translation=(float(position.x), float(position.y), float(position.z)),
        orientation=(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ),
        pose_covariance=tuple(float(value) for value in message.pose.covariance),
        linear_velocity=(
            float(linear_velocity.x),
            float(linear_velocity.y),
            float(linear_velocity.z),
        ),
        angular_velocity=(
            float(angular_velocity.x),
            float(angular_velocity.y),
            float(angular_velocity.z),
        ),
        twist_covariance=tuple(float(value) for value in message.twist.covariance),
    )


def _point_field_data_type(field: Any) -> PointFieldDataType:
    """Map a ``sensor_msgs/PointField`` datatype code to its canonical data type.

    Args:
        field: A decoded ``sensor_msgs/PointField`` (has ``.name``/``.datatype``).

    Returns:
        The canonical point field data type.

    Raises:
        ValueError: If ``field.datatype`` is not one of the supported codes;
            the message names the field, the received code and the supported
            range, so the resulting adapter warning is actionable.
    """
    data_type = POINTFIELD_TYPE_MAP.get(field.datatype)
    if data_type is None:
        raise ValueError(
            f"unsupported PointField datatype {field.datatype!r} for field {field.name!r}; "
            f"supported datatypes are {min(POINTFIELD_TYPE_MAP)}-{max(POINTFIELD_TYPE_MAP)}"
        )
    return data_type


def _available_covariance(values: Any) -> tuple[float, ...] | None:
    """Return a ROS covariance unless its first value marks data unavailable."""
    if values[0] == -1.0:
        return None
    return tuple(float(value) for value in values)


def _remove_row_padding(
    data: bytes,
    *,
    height: int,
    row_step: int,
    row_payload_size: int,
    payload_name: str,
) -> bytes:
    """Validate a ROS row layout and return a tightly packed payload.

    A payload without row padding is already tightly packed and is returned as
    the same object, never copied.
    """
    if height <= 0 or row_payload_size <= 0:
        raise ValueError(f"{payload_name} dimensions must be positive")
    if row_step < row_payload_size:
        raise ValueError(
            f"{payload_name} row step {row_step} is smaller than payload {row_payload_size}"
        )
    expected_size = height * row_step
    if len(data) != expected_size:
        raise ValueError(
            f"{payload_name} data size {len(data)} does not match height*row_step={expected_size}"
        )
    if row_step == row_payload_size:
        return data
    return b"".join(
        data[row_index * row_step : row_index * row_step + row_payload_size]
        for row_index in range(height)
    )


def _swap_fixed_width_values(data: bytes, *, value_size: int) -> bytes:
    """Reverse every fixed-width scalar in a tightly packed byte buffer."""
    return b"".join(
        data[offset : offset + value_size][::-1] for offset in range(0, len(data), value_size)
    )


def _normalize_point_field_byte_order(
    data: bytes,
    *,
    point_step: int,
    fields: tuple[PointFieldDescriptor, ...],
) -> bytes:
    """Convert declared multibyte point fields from big- to little-endian.

    ``data`` is tightly packed (a whole number of ``point_step`` records, as
    :func:`_remove_row_padding` guarantees). Each field is swapped for every
    point at once, as a ``(points, count, value_size)`` byte view reversed on
    its last axis, with the same bytes as a per-point, per-element loop.
    """
    points = np.frombuffer(data, dtype=np.uint8).reshape(-1, point_step).copy()
    for field in fields:
        value_size = POINTFIELD_SIZE_BYTES[field.data_type]
        field_end = field.offset_bytes + value_size * field.count
        if field.offset_bytes < 0 or field_end > point_step:
            raise ValueError(f"point field {field.name!r} exceeds point_step={point_step}")
        if value_size == 1:
            continue
        values = points[:, field.offset_bytes : field_end].reshape(-1, field.count, value_size)
        points[:, field.offset_bytes : field_end] = values[:, :, ::-1].reshape(
            -1, field.count * value_size
        )
    return points.tobytes()


def _with_raw_metadata(provenance: SourceProvenance, **metadata: object) -> SourceProvenance:
    """Return provenance enriched with primitive normalization metadata."""
    return replace(provenance, raw_metadata={**provenance.raw_metadata, **metadata})


def merge_calibration(
    provided: CalibrationSet | None,
    discovered_entries: tuple[CalibrationEntry, ...],
) -> CalibrationSet | None:
    """Merge configured calibration with source-discovered entries by sensor identity."""
    entries = dict(provided.entries) if provided is not None else {}
    by_sensor: dict[SensorId, CalibrationEntry] = {}
    for entry in entries.values():
        previous = by_sensor.get(entry.sensor_id)
        if previous is not None:
            raise CalibrationError(
                f"multiple calibration entries for sensor_id={entry.sensor_id!r}"
            )
        by_sensor[entry.sensor_id] = entry

    for entry in discovered_entries:
        previous = by_sensor.get(entry.sensor_id)
        if previous is not None:
            if previous.content_hash != entry.content_hash:
                raise CalibrationError(f"conflicting calibration for sensor_id={entry.sensor_id!r}")
            continue
        entries[entry.calibration_id] = entry
        by_sensor[entry.sensor_id] = entry

    if not entries and provided is None:
        return None
    return CalibrationSet(
        entries=entries,
        static_transforms=provided.static_transforms if provided is not None else (),
    )


def calibration_ids_by_sensor(
    calibration: CalibrationSet | None,
) -> dict[SensorId, CalibrationReferenceId]:
    """Build an unambiguous sensor-to-calibration lookup."""
    if calibration is None:
        return {}
    result: dict[SensorId, CalibrationReferenceId] = {}
    for entry in calibration.entries.values():
        if entry.sensor_id in result:
            raise CalibrationError(
                f"multiple calibration entries for sensor_id={entry.sensor_id!r}"
            )
        result[entry.sensor_id] = entry.calibration_id
    return result


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


class StreamingContentHash:
    """Accumulates a content hash over exactly the raw messages an adapter reads (issue #506).

    ``sequence_provenance.compute_source_content_hash`` re-reads a whole
    file/directory in its own separate pass — appropriate once, for a full
    ingestion, but it would cost O(source size) even when a
    :class:`~contextmap.ingestion.source_adapter.SourceWindow` limits
    reading to a small fraction of the source. This accumulates a hash
    incrementally, one message at a time, as an adapter's own
    ``read_observations()`` already reads each message — at no extra I/O —
    so the resulting hash covers exactly what was read: the whole source
    when no window is configured, or only the window otherwise.
    """

    def __init__(self) -> None:
        """Create an empty accumulator (:meth:`hexdigest` returns ``None`` until updated)."""
        self._digest = hashlib.sha256()
        self._touched = False

    def update(self, *, topic: str, timestamp_nanoseconds: int, rawdata: bytes) -> None:
        """Fold one message's identity and raw bytes into the running hash.

        Args:
            topic: The message's source topic.
            timestamp_nanoseconds: The message's recording timestamp.
            rawdata: The message's raw, still-serialized bytes.
        """
        self._touched = True
        self._digest.update(topic.encode("utf-8"))
        self._digest.update(timestamp_nanoseconds.to_bytes(8, "big", signed=True))
        self._digest.update(rawdata)

    def hexdigest(self) -> str | None:
        """Return the accumulated hash, or ``None`` if :meth:`update` was never called.

        Returns:
            ``"sha256:<hex digest>"``, or ``None``.
        """
        return f"sha256:{self._digest.hexdigest()}" if self._touched else None


def resolve_window_bounds(
    config: SourceAdapterConfig,
    *,
    reader: Any,
    topic_kinds: Mapping[str, str],
) -> tuple[int | None, int | None]:
    """Resolve a configured window into nanosecond bounds for the bag reader's own time filter.

    Both ``rosbags.rosbag1.Reader.messages`` and ``rosbags.rosbag2.Reader.messages``
    accept ``start``/``stop`` (nanoseconds, the bag's own per-message
    recording time) and skip the chunk data of any message outside that
    range without decompressing or decoding it — the mechanism that makes
    windowed ingestion cost proportional to the window, not the source.

    Args:
        config: The adapter configuration; ``config.window`` may be ``None``.
        reader: An open ``rosbags`` reader of the configured source. Only
            its connections and index are read, never a message chunk, so
            the caller can reuse the same reader for its other checks.
        topic_kinds: Configured topic name -> modality kind, as built by
            each adapter's own ``_configured_topic_kinds()``.

    Returns:
        ``(start_nanoseconds, stop_nanoseconds)``, both ``None`` when no
        window is configured.

    Raises:
        InvalidSourceWindowError: If ``config.window.clock_id`` does not
            match ``config.resolved_window_clock_id()``, or if the window
            does not overlap the source's actual recording-time range at
            all (never silently truncated to an empty read).
    """
    window = config.window
    if window is None:
        return None, None

    expected_clock_id = config.resolved_window_clock_id()
    if window.clock_id != expected_clock_id:
        raise InvalidSourceWindowError(
            f"window.clock_id {window.clock_id!r} does not match this source's own "
            f"recording-time clock {expected_clock_id!r}; a window is always expressed "
            "in the source's own recording time, never the header clock"
        )

    start_ns = round(window.start_seconds * 1_000_000_000)
    stop_ns = round(window.end_seconds * 1_000_000_000)

    bounds = _recording_time_bounds(reader, topic_kinds)
    if bounds is None:
        return start_ns, stop_ns
    source_min_ns, source_max_ns = bounds
    if stop_ns <= source_min_ns or start_ns > source_max_ns:
        raise InvalidSourceWindowError(
            f"window [{window.start_seconds}, {window.end_seconds}) does not overlap "
            f"the source's recording-time range [{source_min_ns / 1e9}, {source_max_ns / 1e9}] "
            "seconds for the configured topics"
        )
    return start_ns, stop_ns


def _recording_time_bounds(reader: Any, topic_kinds: Mapping[str, str]) -> tuple[int, int] | None:
    """Return ``(min, max)`` recording-time nanoseconds for the configured topics, cheaply.

    Reading either bound never decompresses a message chunk: a ROS 1
    reader already carries a per-connection index (``entry.time``) built
    from the bag's own index records; a ROS 2 reader instead exposes only
    a bag-wide ``start_time``/``end_time`` (from the storage's metadata,
    not per-topic), which is used as a safe, if less precise, superset —
    a window that does not overlap the *whole* bag cannot overlap one of
    its topics either.

    Args:
        reader: An open reader of the configured source.
        topic_kinds: Configured topic name -> modality kind.

    Returns:
        The bounds, or ``None`` when the source has no messages on any
        configured topic (ROS 1) or no messages at all (ROS 2).
    """
    indexes = getattr(reader, "indexes", None)
    if indexes is not None:
        connections = [
            connection for connection in reader.connections if connection.topic in topic_kinds
        ]
        recording_times = [
            entry.time for connection in connections for entry in indexes[connection.id]
        ]
        return (min(recording_times), max(recording_times)) if recording_times else None

    if reader.message_count == 0:
        return None
    return reader.start_time, reader.end_time
