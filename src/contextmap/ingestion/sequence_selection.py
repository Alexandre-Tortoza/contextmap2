"""Source-independent selection and replay over a canonical sequence artifact.

A research run should be able to process an entire canonical sequence, a
contiguous range, or an explicit subset of observations without rewriting
or copying the underlying :class:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader`
artifact. A :data:`SequenceSelection` is a small, serializable description
of *which* observations to read; :func:`resolve_selection` is the single
replay path every selection kind goes through, so a full-sequence read and
a partial read share the same code and therefore the same observations for
the frames they have in common.

Four selection kinds are supported in v0: full sequence, a frame-index
range, a timestamp range (within one clock domain — see
``docs/synchronization.md`` for why clock domains are never mixed), and an
explicit set of observation identities. Selections do not nest or compose
in v0, and never filter by modality; modality filtering is left to the
downstream stage that actually needs it. See
``src/contextmap/ingestion/docs/selection.md`` for worked examples.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from contextmap.ingestion.models import SourceObservation, SourceObservationId
from contextmap.ingestion.sequence_artifact import SequenceArtifactId, SequenceArtifactReader


class SequenceSelectionError(Exception):
    """Raised when a selection is invalid or cannot be resolved."""


@dataclass(frozen=True)
class FullSequenceSelection:
    """Select every observation in the sequence, in canonical index order."""


@dataclass(frozen=True, kw_only=True)
class FrameRangeSelection:
    """Select observations by position in the canonical index order.

    Attributes:
        start_frame_index: Inclusive lower bound (0-based position in the
            sequence's canonical index order, not a synchronized
            :class:`~contextmap.ingestion.synchronization.ProcessingObservation`
            index, which is a separate, per-run concept).
        end_frame_index: Exclusive upper bound. Values beyond the
            sequence's length are clipped, not an error, so an
            open-ended "from N to the end" range can be written without
            knowing the exact count.
    """

    start_frame_index: int
    end_frame_index: int

    def __post_init__(self) -> None:
        """Validate the range.

        Raises:
            ValueError: If ``start_frame_index`` is negative or
                ``end_frame_index`` is before it.
        """
        if self.start_frame_index < 0:
            raise ValueError(f"start_frame_index must be >= 0, got {self.start_frame_index}")
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("end_frame_index must be >= start_frame_index")


@dataclass(frozen=True, kw_only=True)
class TimestampRangeSelection:
    """Select observations by normalized timestamp, within one clock domain.

    Attributes:
        clock_id: Clock domain ``start_seconds``/``end_seconds`` are
            expressed in. An observation whose ``timestamp.clock_id``
            differs is excluded, never compared as if it were the same
            clock (see ``docs/synchronization.md``).
        start_seconds: Inclusive lower bound of normalized timestamp
            (``timestamp.to_float_seconds()``).
        end_seconds: Exclusive upper bound.
    """

    clock_id: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        """Validate the range.

        Raises:
            ValueError: If ``end_seconds`` is before ``start_seconds``.
        """
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be >= start_seconds")


@dataclass(frozen=True, kw_only=True)
class ExplicitIdsSelection:
    """Select an explicit set of observations by identity.

    Attributes:
        observation_ids: Identities to select. This set's iteration order
            does not determine replay order; replay always follows the
            sequence's own canonical index order.
    """

    observation_ids: frozenset[SourceObservationId]

    def __post_init__(self) -> None:
        """Validate the set.

        Raises:
            ValueError: If ``observation_ids`` is empty.
        """
        if not self.observation_ids:
            raise ValueError("observation_ids must not be empty")


SequenceSelection = (
    FullSequenceSelection | FrameRangeSelection | TimestampRangeSelection | ExplicitIdsSelection
)
"""Any supported selection kind."""


@dataclass(frozen=True, kw_only=True)
class SequenceSelectionResult:
    """The resolved, deterministic result of applying a selection.

    Attributes:
        sequence_artifact_id: Identity of the sequence artifact the
            selection was resolved against.
        selection: The selection that was resolved.
        selection_id: Deterministic identity of
            ``(sequence_artifact_id, selection)``, for reproducibility —
            see :func:`selection_identity`.
        observations: Selected observations, in the sequence's own
            canonical index order. No payload is copied or mutated to
            produce this result; it references the same observations
            :meth:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader.list_observations`
            would return.
    """

    sequence_artifact_id: SequenceArtifactId
    selection: SequenceSelection
    selection_id: str
    observations: Sequence[SourceObservation]


def resolve_selection(
    reader: SequenceArtifactReader, selection: SequenceSelection
) -> SequenceSelectionResult:
    """Resolve a selection against a sequence artifact reader.

    Every selection kind goes through this one replay path: reading a
    :class:`FullSequenceSelection` and reading the corresponding portion of
    a :class:`FrameRangeSelection` produce identical observations for the
    frames they have in common.

    Args:
        reader: Reader for the sequence artifact to select from.
        selection: The selection to resolve.

    Returns:
        The resolved, deterministically ordered result.

    Raises:
        SequenceSelectionError: If ``selection`` references frame indices
            or observation identities that do not exist in the sequence.
    """
    all_observations = reader.list_observations()
    sequence_artifact_id = reader.manifest.artifact_id

    if isinstance(selection, FullSequenceSelection):
        resolved: list[SourceObservation] = list(all_observations)
    elif isinstance(selection, FrameRangeSelection):
        resolved = _resolve_frame_range(all_observations, selection)
    elif isinstance(selection, TimestampRangeSelection):
        resolved = _resolve_timestamp_range(all_observations, selection)
    elif isinstance(selection, ExplicitIdsSelection):
        resolved = _resolve_explicit_ids(all_observations, selection)
    else:
        raise TypeError(f"unsupported selection type: {type(selection)!r}")

    return SequenceSelectionResult(
        sequence_artifact_id=sequence_artifact_id,
        selection=selection,
        selection_id=selection_identity(sequence_artifact_id, selection),
        observations=tuple(resolved),
    )


def _resolve_frame_range(
    observations: Sequence[SourceObservation], selection: FrameRangeSelection
) -> list[SourceObservation]:
    total = len(observations)
    if total > 0 and selection.start_frame_index >= total:
        raise SequenceSelectionError(
            f"start_frame_index {selection.start_frame_index} is out of range "
            f"for {total} observations"
        )
    end = min(selection.end_frame_index, total)
    return list(observations[selection.start_frame_index : end])


def _resolve_timestamp_range(
    observations: Sequence[SourceObservation], selection: TimestampRangeSelection
) -> list[SourceObservation]:
    resolved: list[SourceObservation] = []
    for observation in observations:
        if observation.timestamp.clock_id != selection.clock_id:
            continue
        seconds = observation.timestamp.to_float_seconds()
        if selection.start_seconds <= seconds < selection.end_seconds:
            resolved.append(observation)
    return resolved


def _resolve_explicit_ids(
    observations: Sequence[SourceObservation], selection: ExplicitIdsSelection
) -> list[SourceObservation]:
    known_ids = {observation.observation_id for observation in observations}
    missing = sorted(str(oid) for oid in selection.observation_ids if oid not in known_ids)
    if missing:
        raise SequenceSelectionError(f"observation_ids not found in sequence: {missing}")
    return [
        observation
        for observation in observations
        if observation.observation_id in selection.observation_ids
    ]


def encode_selection(selection: SequenceSelection) -> dict[str, Any]:
    """Encode a selection into a JSON-serializable dict, e.g. for a run manifest.

    Args:
        selection: The selection to encode.

    Returns:
        A dict suitable for ``json.dumps``.
    """
    if isinstance(selection, FullSequenceSelection):
        return {"kind": "full"}
    if isinstance(selection, FrameRangeSelection):
        return {
            "kind": "frame_range",
            "start_frame_index": selection.start_frame_index,
            "end_frame_index": selection.end_frame_index,
        }
    if isinstance(selection, TimestampRangeSelection):
        return {
            "kind": "timestamp_range",
            "clock_id": selection.clock_id,
            "start_seconds": selection.start_seconds,
            "end_seconds": selection.end_seconds,
        }
    if isinstance(selection, ExplicitIdsSelection):
        return {
            "kind": "explicit_ids",
            "observation_ids": sorted(str(oid) for oid in selection.observation_ids),
        }
    raise TypeError(f"unsupported selection type: {type(selection)!r}")


def decode_selection(record: dict[str, Any]) -> SequenceSelection:
    """Decode a selection from a dict produced by :func:`encode_selection`.

    Args:
        record: A dict as produced by :func:`encode_selection`.

    Returns:
        The decoded selection.

    Raises:
        ValueError: If ``record["kind"]`` is not a known selection kind.
    """
    kind = record["kind"]
    if kind == "full":
        return FullSequenceSelection()
    if kind == "frame_range":
        return FrameRangeSelection(
            start_frame_index=record["start_frame_index"],
            end_frame_index=record["end_frame_index"],
        )
    if kind == "timestamp_range":
        return TimestampRangeSelection(
            clock_id=record["clock_id"],
            start_seconds=record["start_seconds"],
            end_seconds=record["end_seconds"],
        )
    if kind == "explicit_ids":
        return ExplicitIdsSelection(
            observation_ids=frozenset(SourceObservationId(oid) for oid in record["observation_ids"])
        )
    raise ValueError(f"unknown selection kind: {kind!r}")


def selection_identity(
    sequence_artifact_id: SequenceArtifactId, selection: SequenceSelection
) -> str:
    """Compute the deterministic identity of a selection over a sequence.

    Args:
        sequence_artifact_id: Identity of the sequence artifact selected
            from.
        selection: The selection to identify.

    Returns:
        ``"sha256:<hex digest>"`` over the canonical encoding of
        ``(sequence_artifact_id, selection)``. Two calls with the same
        arguments always return the same identity.
    """
    payload = json.dumps(
        {
            "sequence_artifact_id": str(sequence_artifact_id),
            "selection": encode_selection(selection),
        },
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
