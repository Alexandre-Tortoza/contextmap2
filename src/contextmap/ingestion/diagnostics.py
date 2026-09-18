"""Human-readable summary and structured debug diagnostics for a sequence.

Diagnostics are debug evidence, never a contractual dependency: a
:class:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader` is
fully usable without ever calling anything in this module (``docs/ARTIFACTS.md``:
"outputs/ é contratual, debug/ não é"). :func:`summarize_observations` lets
a researcher understand what a sequence contains without opening the
original bag; ``SequenceArtifactWriter.set_diagnostics()`` persists that
summary plus any warnings collected during ingestion as inspectable JSON
files under ``diagnostics/``, rather than only console logs.

v0 writes two files, ``summary.json`` and ``warnings.jsonl`` — smaller
than the candidate structure sketched in the issue (which also lists
``synchronization.jsonl``, ``dropped-events.jsonl``, ``frame-graph.json``).
A caller that ran :func:`~contextmap.ingestion.synchronization.synchronize`
can format its
:class:`~contextmap.ingestion.synchronization.SynchronizationDiagnostics`
entries as text and pass them into the same ``warnings`` list; a dedicated
file per diagnostic kind is deferred until a real second consumer needs to
parse them independently of the human-readable warning text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from contextmap.ingestion.models import (
    MODALITY_NAMES,
    ImageObservation,
    LidarObservation,
    SourceObservation,
    observation_modality,
)

SCHEMA_VERSION = "0.1.0"
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
    warning_count: int


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


def summarize_observations(
    observations: Sequence[SourceObservation], *, warning_count: int = 0
) -> SequenceSummary:
    """Compute a human-readable summary of a set of observations.

    Args:
        observations: Observations to summarize, in any order.
        warning_count: Number of warnings to record in the summary.

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

    return SequenceSummary(
        modality_summaries=modality_summaries,
        image_resolutions=image_resolutions,
        pointcloud_field_names=pointcloud_field_names,
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
        warning_count=record["warning_count"],
    )
