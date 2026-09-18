"""Canonical, backend-agnostic sensor observation contracts.

These types normalize source-specific robotic data (ROS 1 bags, ROS 2 bags,
recorded datasets) into a single representation that downstream capabilities
can consume without importing ROS messages, bag APIs, or dataset-specific
structures. See ``src/contextmap/ingestion/docs/contracts.md`` for the field
reference, unit conventions, and ROS 1/ROS 2 mapping examples.

Each physical modality (image, LiDAR, IMU, external pose) is represented by
its own immutable type instead of a single record with a field per possible
modality. A missing modality is therefore represented by the absence of an
instance, not by a null/zeroed field on a shared record; an optional
sub-field within a modality (e.g. IMU orientation) is represented explicitly
with ``None`` when the source did not provide it.

An externally supplied pose (``ExternalPoseMeasurement``) is an input
measurement read from a source, not the canonical map pose. Ownership of the
canonical ``PoseEstimate`` and trajectory belongs to the State Estimation
capability; this module never produces one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

from contextmap.shared import SourceTimestamp

SourceObservationId = NewType("SourceObservationId", str)
"""Identity of a physical observation, stable across repeated inference runs."""

SensorId = NewType("SensorId", str)
"""Identity of the physical sensor/device that produced an observation."""

FrameId = NewType("FrameId", str)
"""Identity of a coordinate frame an observation or measurement is expressed in.

Full coordinate-frame conventions (handedness, axis order, transform
composition notation) are formalized by the calibration and coordinate-frame
metadata contract; this type only carries the frame's name.
"""

CalibrationReferenceId = NewType("CalibrationReferenceId", str)
"""Reference to a calibration entry owned by the calibration contract.

Ingestion only carries this reference; the calibration data itself is
defined by the calibration and coordinate-frame metadata contract.
"""


class ImageEncoding(Enum):
    """Pixel encoding of an :class:`ImageObservation`, using ROS-style names."""

    RGB8 = "rgb8"
    BGR8 = "bgr8"
    MONO8 = "mono8"
    MONO16 = "mono16"


class PointFieldDataType(Enum):
    """Scalar type of a single named field within a LiDAR point record."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT16 = "int16"
    INT32 = "int32"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"


@dataclass(frozen=True)
class PointFieldDescriptor:
    """Layout of one named field within a packed LiDAR point record.

    Attributes:
        name: Field name, e.g. "x", "y", "z", "intensity", "ring".
        offset_bytes: Byte offset of this field within a point record.
        data_type: Scalar type stored at ``offset_bytes``.
        count: Number of ``data_type`` elements this field holds (1 unless
            the source packs a small array into a single field).
    """

    name: str
    offset_bytes: int
    data_type: PointFieldDataType
    count: int = 1


@dataclass(frozen=True)
class SourceProvenance:
    """Traceability metadata linking a canonical observation to its raw source.

    Attributes:
        source_type: Identifier of the originating adapter family, e.g.
            "ros1_bag", "ros2_bag", "dataset".
        source_path: Path or identity of the source the observation was
            decoded from.
        source_topic: Original topic/channel name, when the source exposes
            one.
        source_message_index: Positional index of the source message within
            its topic/channel, when the source format makes this available.
        raw_metadata: Source-native metadata preserved as primitive values
            (strings, numbers, booleans, and nested mappings/sequences of
            those). No source-native SDK object may be stored here; adapters
            must convert message fields to primitives before constructing
            this provenance.
    """

    source_type: str
    source_path: str
    source_topic: str | None = None
    source_message_index: int | None = None
    raw_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class _SourceObservationBase:
    """Fields shared by every canonical source observation modality.

    Attributes:
        observation_id: Stable identity of the physical observation. Reusing
            the same identity across repeated inference runs is what allows
            those runs to be recognized as re-processing the same physical
            event rather than as a new physical observation.
        sensor_id: Identity of the physical sensor/device that produced this
            observation.
        frame_id: Coordinate frame this observation is expressed in.
        timestamp: Capture time in the source's own clock domain.
        provenance: Traceability metadata back to the raw source.
        calibration_id: Reference to the calibration entry that applies to
            this observation, when the source or its configuration supplies
            one. ``None`` means no calibration is associated yet.
    """

    observation_id: SourceObservationId
    sensor_id: SensorId
    frame_id: FrameId
    timestamp: SourceTimestamp
    provenance: SourceProvenance
    calibration_id: CalibrationReferenceId | None = None


@dataclass(frozen=True, kw_only=True)
class ImageObservation(_SourceObservationBase):
    """A single normalized RGB/mono image frame.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        encoding: Pixel encoding of ``data``.
        data: Row-major, tightly packed pixel buffer (no per-row padding)
            matching ``width``, ``height``, and ``encoding``.
    """

    width: int
    height: int
    encoding: ImageEncoding
    data: bytes


@dataclass(frozen=True, kw_only=True)
class LidarObservation(_SourceObservationBase):
    """A single normalized LiDAR/point-cloud scan.

    Attributes:
        point_count: Number of point records in ``data``.
        point_step_bytes: Number of bytes occupied by a single point record.
        fields: Layout of the named fields packed into each point record.
        is_dense: Whether every point record contains valid, finite values.
            ``False`` indicates the source may include invalid/NaN points.
        data: Row-major, tightly packed buffer of ``point_count`` records,
            each ``point_step_bytes`` long and laid out per ``fields``.
    """

    point_count: int
    point_step_bytes: int
    fields: Sequence[PointFieldDescriptor]
    data: bytes
    is_dense: bool = True


@dataclass(frozen=True, kw_only=True)
class ImuObservation(_SourceObservationBase):
    """A single normalized IMU measurement.

    Attributes:
        linear_acceleration: (x, y, z) in meters per second squared, in
            ``frame_id``. ``None`` when the source does not provide it.
        angular_velocity: (x, y, z) in radians per second, in ``frame_id``.
            ``None`` when the source does not provide it.
        orientation: Unit quaternion (x, y, z, w) expressing this sensor's
            orientation, when the source reports absolute orientation.
            ``None`` when unavailable; a missing orientation must not be
            represented as an identity quaternion.
    """

    linear_acceleration: tuple[float, float, float] | None = None
    angular_velocity: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, kw_only=True)
class ExternalPoseMeasurement(_SourceObservationBase):
    """A pose/odometry measurement read directly from a source.

    This is an input measurement, not the canonical map pose: it is never a
    substitute for, and must never be confused with, a State Estimation
    ``PoseEstimate``. Downstream code that needs the canonical trajectory
    must consume State Estimation's contract, not this type.

    Attributes:
        parent_frame: Reference frame the pose is expressed in, e.g. "odom".
        translation: (x, y, z) position of ``child_frame`` within
            ``parent_frame``, in meters.
        orientation: Unit quaternion (x, y, z, w) of ``child_frame`` within
            ``parent_frame``.

    Note:
        ``frame_id`` (inherited) holds the child frame whose pose is being
        reported, e.g. "base_link".
    """

    parent_frame: FrameId
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


SourceObservation = ImageObservation | LidarObservation | ImuObservation | ExternalPoseMeasurement
"""Any canonical, backend-agnostic sensor observation produced by ingestion."""

MODALITY_NAMES: frozenset[str] = frozenset({"image", "lidar", "imu", "external_pose"})
"""Canonical modality names, one per :data:`SourceObservation` variant."""


def observation_modality(observation: SourceObservation) -> str:
    """Return the canonical modality name of a source observation.

    Args:
        observation: Any canonical source observation.

    Returns:
        One of the names in :data:`MODALITY_NAMES`.

    Raises:
        TypeError: If ``observation`` is not a recognized modality type.
    """
    if isinstance(observation, ImageObservation):
        return "image"
    if isinstance(observation, LidarObservation):
        return "lidar"
    if isinstance(observation, ImuObservation):
        return "imu"
    if isinstance(observation, ExternalPoseMeasurement):
        return "external_pose"
    raise TypeError(f"unsupported observation type: {type(observation)!r}")
