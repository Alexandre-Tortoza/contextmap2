"""Human-readable summary and structured debug diagnostics for a sequence.

Diagnostics are debug evidence, never a contractual dependency: a
:class:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader` is
fully usable without ever calling anything in this module (``docs/ARTIFACTS.md``:
"outputs/ é contratual, debug/ não é"). :func:`summarize_observations` lets
a researcher understand what a sequence contains without opening the
original bag; ``SequenceArtifactWriter.set_diagnostics()`` persists that
summary, warnings, synchronization decisions, dropped events, and calibration
frame inventory as inspectable JSON files under ``diagnostics/``, rather than
only console logs. These records remain debug evidence: downstream scientific
stages must depend on the canonical observations and calibration, not on these
diagnostic projections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from contextmap.ingestion.calibration import CalibrationSet, RigidTransform
from contextmap.ingestion.models import (
    MODALITY_NAMES,
    FrameId,
    ImageObservation,
    LidarObservation,
    SourceObservation,
    observation_modality,
)
from contextmap.ingestion.synchronization import (
    DroppedEvent,
    SynchronizationDecision,
    SynchronizationDiagnostics,
)

SCHEMA_VERSION = "0.2.0"
"""Sequence diagnostics schema version."""


@dataclass(frozen=True, kw_only=True)
class ModalitySummary:
    """Aggregate stats for one modality within a sequence.

    Attributes:
        count: Number of observations of this modality.
        first_timestamp_seconds: Earliest normalized timestamp seen, or
            ``None`` when ``count == 0``. Only directly comparable to
            ``last_timestamp_seconds`` when all observations of this
            modality share one ``clock_id``; see ``clock_ids``.
        last_timestamp_seconds: Latest normalized timestamp seen, or
            ``None`` when ``count == 0``.
        clock_ids: Distinct ``clock_id`` values seen for this modality.
            More than one entry means ``first_timestamp_seconds``/
            ``last_timestamp_seconds`` mix clock domains and should not be
            read as a single time range.
    """

    count: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    clock_ids: Sequence[str]


@dataclass(frozen=True, kw_only=True)
class SequenceSummary:
    """Human-readable summary of a sequence's decoded observations.

    Attributes:
        modality_summaries: One :class:`ModalitySummary` per modality in
            :data:`~contextmap.ingestion.models.MODALITY_NAMES`.
        image_resolutions: Distinct ``(width, height)`` pairs seen across
            image observations.
        pointcloud_field_names: Distinct point-field name layouts seen
            across LiDAR observations, each as a tuple in field order.
        warning_count: Number of warnings recorded alongside this summary.
    """

    modality_summaries: Mapping[str, ModalitySummary]
    image_resolutions: Sequence[tuple[int, int]]
    pointcloud_field_names: Sequence[tuple[str, ...]]
    source_types: Sequence[str]
    topic_counts: Mapping[str, int]
    synchronization_status_counts: Mapping[str, int]
    synchronization_offset_nanoseconds: Mapping[str, tuple[int, int, float]]
    calibration_ids: Sequence[str]
    frame_ids: Sequence[str]
    static_transform_count: int
    warning_count: int


@dataclass(frozen=True, kw_only=True)
class FrameGraphDiagnostics:
    """Inspectable calibration/frame inventory for one sequence."""

    calibration_ids: Sequence[str]
    frame_ids: Sequence[str]
    static_transforms: Sequence[RigidTransform]


@dataclass(frozen=True, kw_only=True)
class SequenceDiagnostics:
    """A sequence's persisted diagnostics: summary plus warning text.

    Attributes:
        summary: The sequence summary.
        warnings: Adapter/validation/synchronization warnings collected
            during ingestion, as human-readable text, in the order they
            were recorded.
    """

    summary: SequenceSummary
    warnings: Sequence[str]
    synchronization: SynchronizationDiagnostics | None = None
    frame_graph: FrameGraphDiagnostics | None = None


def summarize_observations(
    observations: Sequence[SourceObservation],
    *,
    warning_count: int = 0,
    synchronization: SynchronizationDiagnostics | None = None,
    calibration: CalibrationSet | None = None,
) -> SequenceSummary:
    """Compute a human-readable summary of a set of observations.

    Args:
        observations: Observations to summarize, in any order.
        warning_count: Number of warnings to record in the summary.
        synchronization: Structured association decisions to summarize, when available.
        calibration: Calibration and frame inventory to summarize, when available.

    Returns:
        The computed summary.
    """
    by_modality: dict[str, list[SourceObservation]] = {name: [] for name in MODALITY_NAMES}
    for observation in observations:
        by_modality[observation_modality(observation)].append(observation)

    modality_summaries: dict[str, ModalitySummary] = {}
    for modality, items in by_modality.items():
        if not items:
            modality_summaries[modality] = ModalitySummary(
                count=0, first_timestamp_seconds=None, last_timestamp_seconds=None, clock_ids=()
            )
            continue
        seconds = [item.timestamp.to_float_seconds() for item in items]
        clock_ids = sorted({item.timestamp.clock_id for item in items})
        modality_summaries[modality] = ModalitySummary(
            count=len(items),
            first_timestamp_seconds=min(seconds),
            last_timestamp_seconds=max(seconds),
            clock_ids=clock_ids,
        )

    image_resolutions = sorted(
        {
            (observation.width, observation.height)
            for observation in observations
            if isinstance(observation, ImageObservation)
        }
    )
    pointcloud_field_names = sorted(
        {
            tuple(field.name for field in observation.fields)
            for observation in observations
            if isinstance(observation, LidarObservation)
        }
    )
    source_types = sorted({observation.provenance.source_type for observation in observations})
    topic_counts: dict[str, int] = {}
    for observation in observations:
        topic = observation.provenance.source_topic
        if topic is not None:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    status_counts: dict[str, int] = {}
    offsets_by_modality: dict[str, list[int]] = {}
    if synchronization is not None:
        for decision in synchronization.decisions:
            status_counts[decision.status] = status_counts.get(decision.status, 0) + 1
            if decision.offset_nanoseconds is not None:
                offsets_by_modality.setdefault(decision.modality, []).append(
                    decision.offset_nanoseconds
                )
    offset_summaries = {
        modality: (
            min(offsets),
            max(offsets),
            sum(abs(offset) for offset in offsets) / len(offsets),
        )
        for modality, offsets in offsets_by_modality.items()
    }

    calibration_ids = (
        sorted(str(calibration_id) for calibration_id in calibration.entries)
        if calibration is not None
        else []
    )
    frame_ids: set[str] = set()
    if calibration is not None:
        frame_ids.update(str(entry.frame_id) for entry in calibration.entries.values())
        for transform in calibration.static_transforms:
            frame_ids.add(str(transform.parent_frame))
            frame_ids.add(str(transform.child_frame))

    return SequenceSummary(
        modality_summaries=modality_summaries,
        image_resolutions=image_resolutions,
        pointcloud_field_names=pointcloud_field_names,
        source_types=source_types,
        topic_counts=dict(sorted(topic_counts.items())),
        synchronization_status_counts=dict(sorted(status_counts.items())),
        synchronization_offset_nanoseconds=offset_summaries,
        calibration_ids=calibration_ids,
        frame_ids=sorted(frame_ids),
        static_transform_count=(
            len(calibration.static_transforms) if calibration is not None else 0
        ),
        warning_count=warning_count,
    )


def encode_diagnostics(diagnostics: SequenceDiagnostics) -> dict[str, Any]:
    """Encode a sequence's diagnostics summary into a JSON-serializable dict.

    This covers ``summary.json`` only; ``warnings.jsonl`` is written
    separately (one JSON object per line) since it is a log-shaped file,
    not a single document.

    Args:
        diagnostics: The diagnostics to encode.

    Returns:
        A dict suitable for ``json.dumps``.
    """
    summary = diagnostics.summary
    return {
        "schema_version": SCHEMA_VERSION,
        "modality_summaries": {
            modality: {
                "count": item.count,
                "first_timestamp_seconds": item.first_timestamp_seconds,
                "last_timestamp_seconds": item.last_timestamp_seconds,
                "clock_ids": list(item.clock_ids),
            }
            for modality, item in summary.modality_summaries.items()
        },
        "image_resolutions": [list(resolution) for resolution in summary.image_resolutions],
        "pointcloud_field_names": [list(names) for names in summary.pointcloud_field_names],
        "source_types": list(summary.source_types),
        "topic_counts": dict(summary.topic_counts),
        "synchronization_status_counts": dict(summary.synchronization_status_counts),
        "synchronization_offset_nanoseconds": {
            modality: {
                "minimum": values[0],
                "maximum": values[1],
                "mean_absolute": values[2],
            }
            for modality, values in summary.synchronization_offset_nanoseconds.items()
        },
        "calibration_ids": list(summary.calibration_ids),
        "frame_ids": list(summary.frame_ids),
        "static_transform_count": summary.static_transform_count,
        "warning_count": summary.warning_count,
    }


def decode_diagnostics_summary(record: dict[str, Any]) -> SequenceSummary:
    """Decode a sequence summary from a dict produced by :func:`encode_diagnostics`.

    Args:
        record: A dict as produced by :func:`encode_diagnostics`.

    Returns:
        The decoded summary.

    Raises:
        ValueError: If ``record["schema_version"]`` is not understood by
            this module.
    """
    schema_version = record.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported sequence diagnostics schema_version: {schema_version!r}")
    return SequenceSummary(
        modality_summaries={
            modality: ModalitySummary(
                count=item["count"],
                first_timestamp_seconds=item["first_timestamp_seconds"],
                last_timestamp_seconds=item["last_timestamp_seconds"],
                clock_ids=list(item["clock_ids"]),
            )
            for modality, item in record["modality_summaries"].items()
        },
        image_resolutions=[tuple(pair) for pair in record["image_resolutions"]],
        pointcloud_field_names=[tuple(names) for names in record["pointcloud_field_names"]],
        source_types=list(record["source_types"]),
        topic_counts=dict(record["topic_counts"]),
        synchronization_status_counts=dict(record["synchronization_status_counts"]),
        synchronization_offset_nanoseconds={
            modality: (
                values["minimum"],
                values["maximum"],
                values["mean_absolute"],
            )
            for modality, values in record["synchronization_offset_nanoseconds"].items()
        },
        calibration_ids=list(record["calibration_ids"]),
        frame_ids=list(record["frame_ids"]),
        static_transform_count=record["static_transform_count"],
        warning_count=record["warning_count"],
    )


def encode_synchronization_decision(decision: SynchronizationDecision) -> dict[str, Any]:
    """Encode one synchronization decision for JSON Lines persistence."""
    return {
        "frame_index": decision.frame_index,
        "anchor_observation_id": decision.anchor_observation_id,
        "modality": decision.modality,
        "selected_observation_id": decision.selected_observation_id,
        "offset_nanoseconds": decision.offset_nanoseconds,
        "status": decision.status,
    }


def decode_synchronization_decision(record: dict[str, Any]) -> SynchronizationDecision:
    """Decode one persisted synchronization decision."""
    return SynchronizationDecision(**record)


def encode_dropped_event(event: DroppedEvent) -> dict[str, Any]:
    """Encode one dropped event without duplicating its observation payload."""
    return {
        "observation_id": str(event.observation.observation_id),
        "modality": observation_modality(event.observation),
        "reason": event.reason,
    }


def decode_dropped_event(
    record: dict[str, Any],
    observations_by_id: Mapping[str, SourceObservation],
) -> DroppedEvent:
    """Decode one dropped event and resolve its canonical observation."""
    observation_id = record["observation_id"]
    try:
        observation = observations_by_id[observation_id]
    except KeyError as error:
        raise ValueError(
            f"dropped event references unknown observation_id={observation_id!r}"
        ) from error
    if observation_modality(observation) != record["modality"]:
        raise ValueError(f"dropped event modality mismatch for observation_id={observation_id!r}")
    return DroppedEvent(observation=observation, reason=record["reason"])


def build_frame_graph_diagnostics(calibration: CalibrationSet) -> FrameGraphDiagnostics:
    """Build a directly inspectable frame inventory from canonical calibration."""
    frames = {str(entry.frame_id) for entry in calibration.entries.values()}
    for transform in calibration.static_transforms:
        frames.add(str(transform.parent_frame))
        frames.add(str(transform.child_frame))
    return FrameGraphDiagnostics(
        calibration_ids=sorted(str(item) for item in calibration.entries),
        frame_ids=sorted(frames),
        static_transforms=tuple(calibration.static_transforms),
    )


def encode_frame_graph(frame_graph: FrameGraphDiagnostics) -> dict[str, Any]:
    """Encode frame-graph diagnostics as a human-inspectable JSON document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "calibration_ids": list(frame_graph.calibration_ids),
        "frame_ids": list(frame_graph.frame_ids),
        "static_transforms": [
            {
                "parent_frame": str(transform.parent_frame),
                "child_frame": str(transform.child_frame),
                "translation": list(transform.translation),
                "rotation_xyzw": list(transform.rotation),
            }
            for transform in frame_graph.static_transforms
        ],
    }


def decode_frame_graph(record: dict[str, Any]) -> FrameGraphDiagnostics:
    """Decode frame-graph diagnostics and reject unsupported schemas."""
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported frame graph schema_version: {record.get('schema_version')!r}"
        )
    return FrameGraphDiagnostics(
        calibration_ids=tuple(record["calibration_ids"]),
        frame_ids=tuple(record["frame_ids"]),
        static_transforms=tuple(
            RigidTransform(
                parent_frame=FrameId(item["parent_frame"]),
                child_frame=FrameId(item["child_frame"]),
                translation=tuple(item["translation"]),
                rotation=tuple(item["rotation_xyzw"]),
            )
            for item in record["static_transforms"]
        ),
    )
