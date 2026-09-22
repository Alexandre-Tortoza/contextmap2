"""Pose-file source adapter.

Reads a standalone text pose file — shipped separately from a bag, e.g. a
dataset's ground-truth/odometry trajectory — into canonical
:class:`~contextmap.ingestion.models.ExternalPoseMeasurement` observations,
through the same :class:`~contextmap.ingestion.source_adapter.SourceAdapter`
boundary as :mod:`~contextmap.ingestion.adapters.ros1_bag`/
:mod:`~contextmap.ingestion.adapters.ros2_bag`.

v0 supports exactly one format, TUM (``t x y z qx qy qz qw``: seconds as a
decimal, meters, unit quaternion ``(x, y, z, w)``; blank lines and lines
starting with ``#`` are skipped). The file itself never declares which
frames the pose relates, nor a clock domain compatible with any other
source, so :class:`SourceAdapterConfig.extra` must declare ``format``,
``parent_frame`` and ``body_frame`` explicitly — nothing is inferred from
the file name or content. See
``src/contextmap/ingestion/docs/adapters.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from contextmap.ingestion.calibration import CalibrationSet
from contextmap.ingestion.models import (
    ExternalPoseMeasurement,
    FrameId,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.sequence_provenance import compute_source_content_hash
from contextmap.ingestion.source_adapter import (
    SourceAdapterCapabilities,
    SourceAdapterConfig,
    SourceAdapterError,
    SourceAdapterWarning,
)
from contextmap.shared import SourceTimestamp

_SOURCE_TYPE = "pose_file"
_SUPPORTED_FORMATS = frozenset({"tum"})
_NANOSECONDS_PER_SECOND = Decimal(1_000_000_000)
_DEFAULT_SENSOR_ID = "external_pose_file"


class PoseFileConfigError(SourceAdapterError):
    """Raised when required ``SourceAdapterConfig.extra`` settings are missing or invalid."""


class PoseFileFormatError(SourceAdapterError):
    """Raised internally when one line does not match its declared format.

    Never escapes :meth:`PoseFileSourceAdapter.read_observations`: the
    offending line is skipped and reported through
    :meth:`PoseFileSourceAdapter.warnings` instead, matching every other
    adapter's malformed-message policy (see ``docs/adapters.md``).
    """


@dataclass(frozen=True, kw_only=True)
class _PoseFileSettings:
    """Validated, adapter-specific settings read from ``SourceAdapterConfig.extra``."""

    format: str
    parent_frame: FrameId
    body_frame: FrameId
    sensor_id: SensorId


def _settings_from_config(config: SourceAdapterConfig) -> _PoseFileSettings:
    """Validate and extract this adapter's required ``extra`` configuration.

    Args:
        config: The adapter configuration.

    Returns:
        The validated settings.

    Raises:
        PoseFileConfigError: If ``format`` is not a supported value, or
            ``parent_frame``/``body_frame`` is missing or not a non-empty
            string. Nothing is inferred from the file name or content.
    """
    extra = config.extra
    format_name = extra.get("format")
    if format_name not in _SUPPORTED_FORMATS:
        raise PoseFileConfigError(
            "pose_file adapter requires extra['format'] to be one of "
            f"{sorted(_SUPPORTED_FORMATS)}, got {format_name!r}"
        )
    parent_frame = extra.get("parent_frame")
    if not isinstance(parent_frame, str) or not parent_frame:
        raise PoseFileConfigError(
            "pose_file adapter requires a non-empty extra['parent_frame']; "
            "a pose file never declares its own reference frame"
        )
    body_frame = extra.get("body_frame")
    if not isinstance(body_frame, str) or not body_frame:
        raise PoseFileConfigError(
            "pose_file adapter requires a non-empty extra['body_frame']; "
            "a pose file never declares which frame its pose reports"
        )
    sensor_id = extra.get("sensor_id", _DEFAULT_SENSOR_ID)
    if not isinstance(sensor_id, str) or not sensor_id:
        raise PoseFileConfigError("extra['sensor_id'], when given, must be a non-empty string")
    return _PoseFileSettings(
        format=format_name,
        parent_frame=FrameId(parent_frame),
        body_frame=FrameId(body_frame),
        sensor_id=SensorId(sensor_id),
    )


class PoseFileSourceAdapter:
    """Reads a standalone pose file and yields canonical external pose measurements."""

    def __init__(self, config: SourceAdapterConfig) -> None:
        """Create an adapter for a configured pose file.

        Args:
            config: Adapter configuration; ``config.extra`` must declare
                ``format``, ``parent_frame`` and ``body_frame`` (see module
                docs). By convention ``config.source_type`` is
                ``"pose_file"``, but this is not validated here.

        Raises:
            PoseFileConfigError: If required ``extra`` settings are missing
                or invalid.
        """
        self._config = config
        self._settings = _settings_from_config(config)
        self._warnings: list[SourceAdapterWarning] = []

    def capabilities(self) -> SourceAdapterCapabilities:
        """Report that this adapter provides only external pose (plus configured calibration).

        Returns:
            Capabilities reflecting the one modality this adapter produces.
        """
        return SourceAdapterCapabilities(
            external_pose=True,
            calibration=self._config.calibration is not None,
        )

    def read_observations(self) -> Iterator[SourceObservation]:
        """Decode the pose file and yield canonical external pose measurements.

        The whole file is hashed once per call (its provenance-carried
        content identity), then read line by line; a malformed line is
        skipped and reported through :meth:`warnings`, never silently
        dropped without a trace.

        Yields:
            One :class:`~contextmap.ingestion.models.ExternalPoseMeasurement`
            per successfully parsed, non-comment, non-blank line, in file
            order.

        Raises:
            FileNotFoundError: If the configured path does not exist.
        """
        self._warnings = []
        path = Path(self._config.path)
        content_hash = compute_source_content_hash(path)
        clock_id = self._config.resolved_timestamp_clock_id()

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    timestamp, translation, orientation = _parse_tum_line(line, clock_id=clock_id)
                except PoseFileFormatError as error:
                    self._warnings.append(
                        SourceAdapterWarning(
                            topic=None, message_index=line_number, reason=str(error)
                        )
                    )
                    continue

                observation_id = SourceObservationId(
                    f"{self._settings.sensor_id}-{line_number:06d}"
                )
                provenance = SourceProvenance(
                    source_type=_SOURCE_TYPE,
                    source_path=self._config.path,
                    source_message_index=line_number,
                    raw_metadata={
                        "format": self._settings.format,
                        "source_content_hash": content_hash,
                    },
                )
                yield ExternalPoseMeasurement(
                    observation_id=observation_id,
                    sensor_id=self._settings.sensor_id,
                    frame_id=self._settings.body_frame,
                    timestamp=timestamp,
                    provenance=provenance,
                    parent_frame=self._settings.parent_frame,
                    translation=translation,
                    orientation=orientation,
                )

    def read_calibration(self) -> CalibrationSet | None:
        """Return the configured calibration, when any (a pose file has none of its own).

        Returns:
            ``config.calibration`` unchanged, or ``None``.
        """
        return self._config.calibration

    def warnings(self) -> Sequence[SourceAdapterWarning]:
        """Return warnings from the most recent :meth:`read_observations` call.

        Returns:
            Accumulated warnings, in the order they occurred.
        """
        return tuple(self._warnings)


def _parse_tum_line(
    line: str, *, clock_id: str
) -> tuple[SourceTimestamp, tuple[float, float, float], tuple[float, float, float, float]]:
    """Parse one non-blank, non-comment TUM line.

    Args:
        line: The stripped line text.
        clock_id: Clock domain identity to attach to the resulting timestamp.

    Returns:
        The parsed timestamp, translation and orientation (unnormalized,
        exactly as the file states — numerical validity is State
        Estimation's policy, not this adapter's).

    Raises:
        PoseFileFormatError: If the line does not have exactly 8
            whitespace-separated fields, the timestamp is not a finite
            whole number of nanoseconds, or a field is not a finite number.
    """
    fields = line.split()
    if len(fields) != 8:
        raise PoseFileFormatError(f"expected 't x y z qx qy qz qw' (8 fields), got {len(fields)}")

    try:
        nanoseconds_total = Decimal(fields[0]) * _NANOSECONDS_PER_SECOND
    except (ArithmeticError, InvalidOperation) as error:
        raise PoseFileFormatError(f"timestamp {fields[0]!r} is not a finite number") from error
    is_whole_number = nanoseconds_total == nanoseconds_total.to_integral_value()
    if not nanoseconds_total.is_finite() or not is_whole_number:
        raise PoseFileFormatError(f"timestamp {fields[0]!r} is not a whole number of nanoseconds")
    total_ns = int(nanoseconds_total)
    seconds, nanoseconds = divmod(total_ns, 1_000_000_000)

    try:
        x, y, z, qx, qy, qz, qw = (float(value) for value in fields[1:])
    except ValueError as error:
        raise PoseFileFormatError(f"non-numeric field in {fields[1:]!r}") from error
    if not all(math.isfinite(value) for value in (x, y, z, qx, qy, qz, qw)):
        raise PoseFileFormatError("translation/orientation must be finite")

    return (
        SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id),
        (x, y, z),
        (qx, qy, qz, qw),
    )
