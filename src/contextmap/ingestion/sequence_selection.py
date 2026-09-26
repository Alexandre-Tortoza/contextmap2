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
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, overload

from contextmap.ingestion.models import SourceObservation, SourceObservationId
from contextmap.ingestion.sequence_artifact import SequenceArtifactId, SequenceArtifactReader

_NANOSECONDS_PER_SECOND = 1_000_000_000


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
            index, which is a separate, per-run concept). A value at or
            beyond the sequence's length is a :class:`SequenceSelectionError`
            when resolved, empty sequences included; only ``0`` over an
            empty sequence resolves, to an empty result.
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
        start_seconds: Inclusive lower bound of normalized timestamp, in seconds. It is
            compared with the exact ``timestamp.total_nanoseconds()``, never with a float
            conversion of the timestamp.
        end_seconds: Exclusive upper bound, compared the same way. Must be
            strictly greater than ``start_seconds``: ``[t, t)`` contains no
            instant, so a zero-width range is rejected as invalid
            configuration, the same rule as
            :class:`~contextmap.ingestion.source_adapter.SourceWindow`.
    """

    clock_id: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        """Validate the range.

        Raises:
            ValueError: If ``end_seconds`` does not come strictly after
                ``start_seconds``.
        """
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be > start_seconds (a range must not be empty)")


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

    Resolving never reads a payload file: which observations match is
    decided from :meth:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader.iter_index`
    alone. The returned ``observations`` decode each matching observation —
    payload included — only when the caller actually accesses it, and do not
    cache it, so peak memory stays proportional to what the caller holds at
    once, never to the size of the selection or of the sequence.

    Args:
        reader: Reader for the sequence artifact to select from.
        selection: The selection to resolve.

    Returns:
        The resolved, deterministically ordered result.

    Raises:
        SequenceSelectionError: If ``selection`` references frame indices
            or observation identities that do not exist in the sequence.
    """
    sequence_artifact_id = reader.manifest.artifact_id

    if isinstance(selection, FullSequenceSelection):
        offsets: list[int] = [entry.offset for entry in reader.iter_index()]
    elif isinstance(selection, FrameRangeSelection):
        offsets = _resolve_frame_range_offsets(reader, selection)
    elif isinstance(selection, TimestampRangeSelection):
        offsets = _resolve_timestamp_range_offsets(reader, selection)
    elif isinstance(selection, ExplicitIdsSelection):
        offsets = _resolve_explicit_ids_offsets(reader, selection)
    else:
        raise TypeError(f"unsupported selection type: {type(selection)!r}")

    observations: Sequence[SourceObservation] = (
        _LazyObservationView(reader, offsets) if offsets else ()
    )
    return SequenceSelectionResult(
        sequence_artifact_id=sequence_artifact_id,
        selection=selection,
        selection_id=selection_identity(sequence_artifact_id, selection),
        observations=observations,
    )


def _resolve_frame_range_offsets(
    reader: SequenceArtifactReader, selection: FrameRangeSelection
) -> list[int]:
    offsets: list[int] = []
    total = 0
    for position, entry in enumerate(reader.iter_index()):
        total = position + 1
        if selection.start_frame_index <= position < selection.end_frame_index:
            offsets.append(entry.offset)
    # start=0 sobre uma sequência vazia é o único início "no fim" aceito: ele descreve a
    # sequência inteira, que por acaso não tem nada. Qualquer outro start fora do intervalo é erro.
    if selection.start_frame_index > 0 and selection.start_frame_index >= total:
        raise SequenceSelectionError(
            f"start_frame_index {selection.start_frame_index} is out of range "
            f"for {total} observations"
        )
    return offsets


def _resolve_timestamp_range_offsets(
    reader: SequenceArtifactReader, selection: TimestampRangeSelection
) -> list[int]:
    start = _ceil_nanoseconds(selection.start_seconds)
    end = _ceil_nanoseconds(selection.end_seconds)
    offsets: list[int] = []
    for entry in reader.iter_index():
        if entry.timestamp.clock_id != selection.clock_id:
            continue
        if start <= entry.timestamp.total_nanoseconds() < end:
            offsets.append(entry.offset)
    return offsets


def _ceil_nanoseconds(seconds: float) -> int | float:
    """Return the smallest whole nanosecond count not below ``seconds``, computed exactly.

    For an integer ``t``, ``t >= x`` iff ``t >= ceil(x)`` and ``t < x`` iff ``t < ceil(x)``, so
    both bounds of a range compare exactly with integer nanoseconds. A non-finite bound is
    returned as is: it already compares correctly with any integer.
    """
    if not math.isfinite(seconds):
        return seconds
    return math.ceil(Fraction(seconds) * _NANOSECONDS_PER_SECOND)


def _resolve_explicit_ids_offsets(
    reader: SequenceArtifactReader, selection: ExplicitIdsSelection
) -> list[int]:
    wanted = set(selection.observation_ids)
    found: dict[SourceObservationId, int] = {}
    for entry in reader.iter_index():
        if entry.observation_id in wanted:
            found[entry.observation_id] = entry.offset
    missing = sorted(str(oid) for oid in wanted if oid not in found)
    if missing:
        raise SequenceSelectionError(f"observation_ids not found in sequence: {missing}")
    return list(found.values())


class _LazyObservationView(Sequence[SourceObservation]):
    """A resolved selection's observations, decoded from disk one at a time.

    Never reads ahead and never caches: each element — payload included — is
    decoded fresh from
    :meth:`~contextmap.ingestion.sequence_artifact.SequenceArtifactReader.observation_at`
    on every access. A caller that iterates the same view twice pays the
    decode cost twice; this trades that cost for the memory guarantee
    :func:`resolve_selection` makes.
    """

    def __init__(self, reader: SequenceArtifactReader, offsets: Sequence[int]) -> None:
        """Build a view over ``offsets`` (already in canonical sequence order).

        Args:
            reader: Reader used to decode each observation on access.
            offsets: Byte offsets (see
                :attr:`~contextmap.ingestion.sequence_artifact.IndexEntry.offset`),
                in canonical sequence order.
        """
        self._reader = reader
        self._offsets = tuple(offsets)

    def __len__(self) -> int:
        """Return the number of observations in this view."""
        return len(self._offsets)

    @overload
    def __getitem__(self, index: int) -> SourceObservation: ...

    @overload
    def __getitem__(self, index: slice) -> list[SourceObservation]: ...

    def __getitem__(self, index: int | slice) -> SourceObservation | list[SourceObservation]:
        """Decode the observation(s) at ``index``, reading their payload now."""
        if isinstance(index, slice):
            return [self._reader.observation_at(offset) for offset in self._offsets[index]]
        return self._reader.observation_at(self._offsets[index])

    def __eq__(self, other: object) -> bool:
        """Compare elementwise against any other sequence of observations."""
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(
            mine == theirs for mine, theirs in zip(self, other, strict=True)
        )

    def __repr__(self) -> str:
        """Return a compact representation that never decodes any element."""
        return f"_LazyObservationView(len={len(self)})"


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
