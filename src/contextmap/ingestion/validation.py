"""Structural validation for canonical observations and sequences.

Ingestion errors should be caught before they propagate into perception,
geometry, or association. This module complements the file-level checks
already in :mod:`contextmap.ingestion.sequence_artifact`
(missing/corrupted files, index-to-payload cross-references) and
:func:`contextmap.ingestion.calibration.validate_calibration_set` with
checks over the *content* of decoded observations: image/point-cloud
metadata consistency, timestamp ordering, and frame-reference consistency
against a sequence's calibration. None of these checks require a
persisted artifact, GPU, or model dependency — they run directly over
in-memory :data:`~contextmap.ingestion.models.SourceObservation` sequences,
so they are cheap enough for every pull request.

Every function here follows the same convention as
:meth:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader.verify_integrity`:
it returns a list of human-readable problems, never raises, and an empty
list means no problem was found.
"""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.ingestion.calibration import CalibrationSet
from contextmap.ingestion.models import (
    ImageEncoding,
    ImageObservation,
    LidarObservation,
    PointFieldDataType,
    SourceObservation,
)

_IMAGE_BYTES_PER_PIXEL: dict[ImageEncoding, int] = {
    ImageEncoding.RGB8: 3,
    ImageEncoding.BGR8: 3,
    ImageEncoding.MONO8: 1,
    ImageEncoding.MONO16: 2,
}

_POINT_FIELD_SIZE_BYTES: dict[PointFieldDataType, int] = {
    PointFieldDataType.INT8: 1,
    PointFieldDataType.UINT8: 1,
    PointFieldDataType.INT16: 2,
    PointFieldDataType.UINT16: 2,
    PointFieldDataType.INT32: 4,
    PointFieldDataType.UINT32: 4,
    PointFieldDataType.FLOAT32: 4,
    PointFieldDataType.FLOAT64: 8,
}


def validate_image_observation(observation: ImageObservation) -> list[str]:
    """Check one image observation's metadata for internal consistency.

    Args:
        observation: The image observation to check.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    problems: list[str] = []
    if observation.width <= 0 or observation.height <= 0:
        problems.append(f"{observation.observation_id}: width and height must be positive")
        return problems

    bytes_per_pixel = _IMAGE_BYTES_PER_PIXEL[observation.encoding]
    expected_size = observation.width * observation.height * bytes_per_pixel
    if len(observation.data) != expected_size:
        problems.append(
            f"{observation.observation_id}: image data size {len(observation.data)} does not "
            f"match width*height*bytes_per_pixel={expected_size} for encoding "
            f"{observation.encoding.value!r}"
        )
    return problems


def validate_lidar_observation(observation: LidarObservation) -> list[str]:
    """Check one LiDAR observation's metadata and point-field layout for consistency.

    Every field must lie entirely inside one point record: its first byte at an offset within
    ``point_step_bytes`` and its last element ending at or before it
    (``offset_bytes + size(data_type) * count <= point_step_bytes``), whatever the source's byte
    order.

    Args:
        observation: The LiDAR observation to check.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    problems: list[str] = []
    if observation.point_step_bytes <= 0:
        problems.append(f"{observation.observation_id}: point_step_bytes must be positive")
        return problems

    expected_size = observation.point_count * observation.point_step_bytes
    if len(observation.data) != expected_size:
        problems.append(
            f"{observation.observation_id}: point cloud data size {len(observation.data)} does "
            f"not match point_count*point_step_bytes={expected_size}"
        )

    seen_names: set[str] = set()
    for point_field in observation.fields:
        if point_field.name in seen_names:
            problems.append(
                f"{observation.observation_id}: duplicate point field name {point_field.name!r}"
            )
        seen_names.add(point_field.name)
        field_end = (
            point_field.offset_bytes
            + _POINT_FIELD_SIZE_BYTES[point_field.data_type] * point_field.count
        )
        if point_field.offset_bytes < 0 or (
            point_field.offset_bytes >= observation.point_step_bytes
        ):
            problems.append(
                f"{observation.observation_id}: field {point_field.name!r} offset_bytes "
                f"{point_field.offset_bytes} is out of range for "
                f"point_step_bytes={observation.point_step_bytes}"
            )
        elif field_end > observation.point_step_bytes:
            problems.append(
                f"{observation.observation_id}: field {point_field.name!r} ends at byte "
                f"{field_end}, beyond point_step_bytes={observation.point_step_bytes}"
            )
    return problems


def validate_timestamp_ordering(
    observations: Sequence[SourceObservation], *, allow_duplicates: bool = True
) -> list[str]:
    """Check that observations sharing a clock domain are monotonically ordered.

    Observations are checked in the order given; pass them pre-sorted by
    intended replay order (e.g. a sequence artifact's own index order).
    Observations on different ``clock_id`` values are never compared to
    each other, matching the rule already established for synchronization
    (``docs/synchronization.md``) and selection.

    Args:
        observations: Observations to check, in their intended order.
        allow_duplicates: When ``False``, two observations sharing both a
            ``clock_id`` and a normalized timestamp are also reported.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    problems: list[str] = []
    # Nanossegundos inteiros, como em synchronize(): em epochs reais o float64 não separa
    # diferenças abaixo de ~238 ns e esconderia regressões ou inventaria duplicatas.
    last_by_clock: dict[str, tuple[int, str]] = {}
    for observation in observations:
        clock_id = observation.timestamp.clock_id
        nanoseconds = observation.timestamp.total_nanoseconds()
        previous = last_by_clock.get(clock_id)
        if previous is not None:
            previous_nanoseconds, previous_id = previous
            if nanoseconds < previous_nanoseconds:
                problems.append(
                    f"{observation.observation_id}: timestamp {nanoseconds} ns is before "
                    f"{previous_id}'s {previous_nanoseconds} ns on clock {clock_id!r} "
                    "(non-monotonic)"
                )
            elif nanoseconds == previous_nanoseconds and not allow_duplicates:
                problems.append(
                    f"{observation.observation_id}: duplicate timestamp {nanoseconds} ns on clock "
                    f"{clock_id!r} (shared with {previous_id})"
                )
        last_by_clock[clock_id] = (nanoseconds, str(observation.observation_id))
    return problems


def validate_frame_references(
    observations: Sequence[SourceObservation], calibration: CalibrationSet | None
) -> list[str]:
    """Check that observation ``frame_id`` values are known to a sequence's calibration.

    When ``calibration`` is ``None``, there is nothing to check against and
    this always returns an empty list — the absence of calibration is not
    itself a frame-reference problem.

    Args:
        observations: Observations to check.
        calibration: The sequence's calibration set, or ``None``.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    if calibration is None:
        return []

    known_frames = {str(entry.frame_id) for entry in calibration.entries.values()}
    for transform in calibration.static_transforms:
        known_frames.add(str(transform.parent_frame))
        known_frames.add(str(transform.child_frame))

    problems: list[str] = []
    for observation in observations:
        if str(observation.frame_id) not in known_frames:
            problems.append(
                f"{observation.observation_id}: frame_id {observation.frame_id!r} is not a "
                f"known calibration frame"
            )
    return problems


def validate_observations(
    observations: Sequence[SourceObservation],
    *,
    calibration: CalibrationSet | None = None,
    allow_duplicate_timestamps: bool = True,
) -> list[str]:
    """Run every structural validation check over a set of observations.

    Args:
        observations: Observations to check, in their intended replay
            order.
        calibration: The sequence's calibration set, when available; see
            :func:`validate_frame_references`.
        allow_duplicate_timestamps: Forwarded to
            :func:`validate_timestamp_ordering`.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    problems: list[str] = []
    for observation in observations:
        if isinstance(observation, ImageObservation):
            problems.extend(validate_image_observation(observation))
        elif isinstance(observation, LidarObservation):
            problems.extend(validate_lidar_observation(observation))
    problems.extend(
        validate_timestamp_ordering(observations, allow_duplicates=allow_duplicate_timestamps)
    )
    problems.extend(validate_frame_references(observations, calibration))
    return problems
