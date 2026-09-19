"""Source adapter boundary: how a raw recorded source becomes canonical observations.

A source adapter decodes one recorded source (a ROS 1 bag, a ROS 2 bag, a
recorded dataset) and yields canonical
:data:`~contextmap.ingestion.models.SourceObservation` instances — one per
physical source message, ungrouped. Grouping/synchronization
(:mod:`contextmap.ingestion.synchronization`) is a separate stage applied
to an adapter's output, not this boundary's concern.

No ROS message object, bag SDK type, or other source-native structure may
ever cross this boundary; every :class:`SourceAdapter` method returns only
canonical ingestion contracts or primitives. Concrete adapters (ROS 1, ROS
2, ...) therefore hold their ROS-specific dependencies and live under
``contextmap.ingestion.adapters.*`` — the only place a capability may
import a heavy source SDK, per ``docs/architecture.md`` and
``tests/architecture/test_boundaries.py``'s ``HEAVY_SDK_ROOTS`` check. A
consumer that only reads an already-persisted
:class:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader`
never needs those dependencies installed. See
``src/contextmap/ingestion/docs/adapters.md`` for the configuration shape
and per-adapter guidance.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from contextmap.ingestion.calibration import CalibrationSet, ensure_valid_calibration_set
from contextmap.ingestion.models import SourceObservation

_KNOWN_TOPIC_FIELDS = frozenset({"rgb", "camera_info", "lidar", "imu", "pose"})


class SourceAdapterError(Exception):
    """Base class for source adapter failures."""


class MissingRequiredTopicError(SourceAdapterError):
    """Raised when a configured required topic/channel is absent from the source."""


class UnsupportedSourceMessageError(SourceAdapterError):
    """Raised when a source message cannot be decoded by this adapter."""


@dataclass(frozen=True, kw_only=True)
class SourceTopicMapping:
    """Named source topic/channel assignments an adapter reads from.

    Attributes:
        rgb: Topic/channel for RGB/image data, or ``None`` when not used.
        camera_info: Topic/channel for camera calibration metadata.
        lidar: Topic/channel for LiDAR/point-cloud data.
        imu: Topic/channel for IMU data.
        pose: Topic/channel for external pose/odometry data.
    """

    rgb: str | None = None
    camera_info: str | None = None
    lidar: str | None = None
    imu: str | None = None
    pose: str | None = None


@dataclass(frozen=True, kw_only=True)
class SourceAdapterConfig:
    """Configuration for constructing a source adapter.

    A concrete adapter that needs configuration beyond this shared shape
    (e.g. a ROS 2 storage/serialization format identity) reads it from
    ``extra`` rather than the boundary growing a field per adapter family.

    Attributes:
        source_type: Adapter family identity, e.g. ``"ros1_bag"``,
            ``"ros2_bag"``, ``"dataset"``.
        path: Path to the recorded source.
        topics: Topic/channel mapping this adapter reads from.
        timestamp_clock_id: Shared identity of the header clock used by the
            configured topics. When omitted, a deterministic identity derived
            from ``source_type`` and ``path`` is used.
        calibration: Externally supplied canonical calibration to merge with
            calibration discovered in the source. This is also the supported
            path for static extrinsics when a source adapter does not decode TF.
        required_topics: Subset of :class:`SourceTopicMapping` field names
            that must be present in the source. A required topic missing
            from the source raises :class:`MissingRequiredTopicError`; it
            is never silently skipped.
        extra: Adapter-specific configuration not covered by the shared
            shape above, as primitive values.
    """

    source_type: str
    path: str
    topics: SourceTopicMapping
    timestamp_clock_id: str | None = None
    calibration: CalibrationSet | None = None
    required_topics: frozenset[str] = frozenset()
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ``required_topics`` names a known topic field.

        Raises:
            ValueError: If any entry in ``required_topics`` is not one of
                :class:`SourceTopicMapping`'s field names.
        """
        unknown = self.required_topics - _KNOWN_TOPIC_FIELDS
        if unknown:
            raise ValueError(f"unknown required_topics entries: {sorted(unknown)}")
        if self.timestamp_clock_id == "":
            raise ValueError("timestamp_clock_id must not be empty")
        if self.calibration is not None:
            ensure_valid_calibration_set(self.calibration)

    def resolved_timestamp_clock_id(self) -> str:
        """Return the explicit or deterministic source-wide header clock identity."""
        return self.timestamp_clock_id or f"{self.source_type}:{self.path}:header"


@dataclass(frozen=True, kw_only=True)
class SourceAdapterCapabilities:
    """Which modalities/metadata a source adapter can produce for its source.

    Attributes:
        rgb: Whether RGB/image observations are available.
        lidar: Whether LiDAR observations are available.
        imu: Whether IMU observations are available.
        external_pose: Whether external pose/odometry observations are
            available.
        calibration: Whether calibration metadata is available.
    """

    rgb: bool = False
    lidar: bool = False
    imu: bool = False
    external_pose: bool = False
    calibration: bool = False


@dataclass(frozen=True, kw_only=True)
class SourceAdapterWarning:
    """A non-fatal problem encountered while decoding a source.

    Attributes:
        topic: Topic/channel the problem occurred on, when known.
        message_index: Positional index of the affected source message,
            when known.
        reason: Human-readable explanation.
    """

    topic: str | None
    message_index: int | None
    reason: str


@runtime_checkable
class SourceAdapter(Protocol):
    """The minimal boundary every source adapter implements.

    Implementations decode one configured source and expose its observations
    and calibration entirely through canonical ingestion contracts; see module
    docs for the heavy-SDK isolation this boundary depends on.
    """

    def capabilities(self) -> SourceAdapterCapabilities:
        """Report which modalities/metadata this source actually provides.

        Returns:
            The capabilities this adapter's configured source has.
        """
        ...

    def read_observations(self) -> Iterator[SourceObservation]:
        """Decode the source and yield canonical observations.

        Malformed or unsupported source messages are skipped and reported
        through :meth:`warnings`, never silently dropped without a trace.

        Yields:
            One :data:`~contextmap.ingestion.models.SourceObservation` per
            physical source message successfully decoded, in source order.

        Raises:
            MissingRequiredTopicError: If a topic named in this adapter's
                ``SourceAdapterConfig.required_topics`` is absent from the
                source.
        """
        ...

    def read_calibration(self) -> CalibrationSet | None:
        """Return normalized calibration available for the configured source.

        Returns:
            Canonical calibration, or ``None`` when the source and its
            configuration provide no calibration.
        """
        ...

    def warnings(self) -> Sequence[SourceAdapterWarning]:
        """Return warnings accumulated by the most recent :meth:`read_observations` call.

        Returns:
            Accumulated warnings, in the order they occurred.
        """
        ...
