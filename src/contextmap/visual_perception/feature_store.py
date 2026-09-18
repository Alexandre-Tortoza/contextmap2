"""Durable feature-payload storage: writer, index, and lazy/random-access reader.

A :class:`~contextmap.visual_perception.models.VisualFeature` carries
only lightweight metadata and an opaque ``payload_reference`` — never
the numerical array inline (``models.py``, #48). This module is what
actually persists and later loads that array, separately from
``outputs/results.jsonl`` (``run_artifact.py``, #52), so that opening a
run or reading a feature's metadata never requires loading its
numerical payload into memory.

Storage format is NumPy's native ``.npy`` container: it preserves exact
shape/dtype/values (including non-native byte order, encoded in the
dtype string itself), needs no extra runtime dependency beyond NumPy
(already used internally by capabilities per
``docs/shared-primitives.md`` — never as part of a public contract's
*identity*), and gives trivial random access since every feature is its
own file. NumPy is only ever imported inside the functions that
actually serialize/deserialize array bytes (:meth:`FeatureStoreWriter.write`,
:meth:`FeatureStoreReader.load`) — reading metadata/the index never
requires NumPy to be installed. See
``src/contextmap/visual_perception/docs/feature_store.md`` for the
on-disk layout and integrity guarantees.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import (
    BackendProvenance,
    FeatureId,
    FeatureScope,
    VisualFeature,
)
from contextmap.visual_perception.serialization import decode_provenance, encode_provenance

if TYPE_CHECKING:
    from numpy.typing import NDArray

FEATURE_INDEX_FILENAME = "feature-index.jsonl"
"""Name of the feature store's index file, relative to its root."""


class FeatureStoreError(Exception):
    """Base class for feature payload storage failures."""


class FeaturePayloadIntegrityError(FeatureStoreError):
    """Raised when a persisted feature payload fails an integrity check on load."""


@dataclass(frozen=True, kw_only=True)
class FeaturePayloadEntry:
    """Indexed metadata for one persisted feature payload.

    Every field here is readable without loading the numerical array —
    this is exactly what the index (``feature-index.jsonl``) persists.

    Attributes:
        feature_id: The feature this entry describes.
        source_observation_id: The physical observation the feature was
            produced for.
        scope: Whether the payload is dense, global, or region-scoped
            (see :class:`~contextmap.visual_perception.models.FeatureScope`).
        embedding_space_id: Opaque embedding space reference (see
            :mod:`contextmap.visual_perception.embedding_space`).
        shape: Exact array shape, validated on both write and load.
        dtype: Exact NumPy dtype string (e.g. ``"float32"``,
            ``">f4"`` for non-native byte order); validated on both
            write and load.
        normalization: Normalization metadata carried over from the
            producing :class:`~contextmap.visual_perception.models.VisualFeature`.
        payload_reference: Path to the payload file, relative to the
            feature store root — the same value as the owning
            ``VisualFeature.payload_reference``.
        content_hash: ``"sha256:<hex digest>"`` of the raw file bytes.
        size_bytes: Size of the payload file in bytes.
        provenance: Backend that produced the feature.
    """

    feature_id: FeatureId
    source_observation_id: SourceObservationId
    scope: FeatureScope
    embedding_space_id: str
    shape: tuple[int, ...]
    dtype: str
    normalization: str | None
    payload_reference: str
    content_hash: str
    size_bytes: int
    provenance: BackendProvenance


class FeatureStoreWriter:
    """Writes feature payloads under a feature store root, accumulating an index."""

    def __init__(self, root: Path) -> None:
        """Create a writer rooted at ``root``.

        Args:
            root: Directory payload paths are resolved relative to
                (typically a run artifact's ``outputs/features/``).
        """
        self._root = root
        self._entries: list[FeaturePayloadEntry] = []

    def write(
        self,
        feature: VisualFeature,
        source_observation_id: SourceObservationId,
        array: NDArray[Any],
    ) -> FeaturePayloadEntry:
        """Persist one feature's numerical array to ``feature.payload_reference``.

        Args:
            feature: The feature this array belongs to. Its ``shape``
                and ``dtype`` must match ``array`` exactly.
            source_observation_id: The physical observation the feature
                was produced for.
            array: The array to persist.

        Returns:
            The indexed entry describing the persisted payload.

        Raises:
            FeatureStoreError: If ``array``'s shape/dtype does not match
                ``feature``'s declared metadata, or
                ``feature.payload_reference`` resolves outside ``root``.
        """
        import numpy as np

        if tuple(array.shape) != feature.shape:
            raise FeatureStoreError(
                f"array shape {tuple(array.shape)} does not match "
                f"feature.shape {feature.shape} for feature_id={feature.feature_id!r}"
            )
        if str(array.dtype) != feature.dtype:
            raise FeatureStoreError(
                f"array dtype {array.dtype!s} does not match "
                f"feature.dtype {feature.dtype!r} for feature_id={feature.feature_id!r}"
            )

        full_path = _resolve_within_root(self._root, feature.payload_reference)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        data = buffer.getvalue()
        full_path.write_bytes(data)

        entry = FeaturePayloadEntry(
            feature_id=feature.feature_id,
            source_observation_id=source_observation_id,
            scope=feature.scope,
            embedding_space_id=feature.embedding_space_id,
            shape=feature.shape,
            dtype=feature.dtype,
            normalization=feature.normalization,
            payload_reference=feature.payload_reference,
            content_hash=f"sha256:{hashlib.sha256(data).hexdigest()}",
            size_bytes=len(data),
            provenance=feature.provenance,
        )
        self._entries.append(entry)
        return entry

    def entries(self) -> Sequence[FeaturePayloadEntry]:
        """Every entry written so far, in write order."""
        return tuple(self._entries)


class FeatureStoreReader:
    """Reads a feature store's index, then loads individual payloads lazily."""

    def __init__(self, root: Path, entries: Sequence[FeaturePayloadEntry]) -> None:
        """Wrap an already-loaded index.

        Prefer :meth:`open` to read a persisted index from disk.
        """
        self._root = root
        self._by_id = {entry.feature_id: entry for entry in entries}

    @classmethod
    def open(cls, root: Path) -> FeatureStoreReader:
        """Open a feature store's index without loading any payload.

        Args:
            root: The feature store root (typically a run artifact's
                ``outputs/features/``).

        Returns:
            A reader over the index found at ``root``, or an empty
            reader when ``root`` has no index yet (persisting feature
            payloads is opt-in per run).
        """
        index_path = root / FEATURE_INDEX_FILENAME
        if not index_path.is_file():
            return cls(root, ())

        entries: list[FeaturePayloadEntry] = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    entries.append(decode_feature_payload_entry(json.loads(stripped)))
        return cls(root, entries)

    def feature_ids(self) -> Sequence[FeatureId]:
        """Every feature id with a persisted payload, in index order."""
        return tuple(self._by_id)

    def entry(self, feature_id: FeatureId) -> FeaturePayloadEntry:
        """Return one feature's indexed metadata without loading its payload.

        Raises:
            FeatureStoreError: If no entry exists for ``feature_id``.
        """
        try:
            return self._by_id[feature_id]
        except KeyError:
            raise FeatureStoreError(f"no persisted payload for feature_id={feature_id!r}") from None

    def load(self, feature_id: FeatureId) -> NDArray[Any]:
        """Load and verify one feature's numerical array.

        Args:
            feature_id: The feature to load.

        Returns:
            The array, exactly as written.

        Raises:
            FeatureStoreError: If no entry exists for ``feature_id``, or
                the payload file is missing.
            FeaturePayloadIntegrityError: If the payload's content hash,
                shape, or dtype does not match its indexed entry, or the
                file is not a valid/supported ``.npy`` payload.
        """
        import numpy as np

        entry = self.entry(feature_id)
        full_path = _resolve_within_root(self._root, entry.payload_reference)
        if not full_path.is_file():
            raise FeatureStoreError(f"missing payload file: {entry.payload_reference}")

        data = full_path.read_bytes()
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        if digest != entry.content_hash:
            raise FeaturePayloadIntegrityError(
                f"content hash mismatch for {entry.payload_reference}: "
                f"expected {entry.content_hash}, found {digest}"
            )

        try:
            array: NDArray[Any] = np.load(io.BytesIO(data), allow_pickle=False)
        except (ValueError, OSError) as error:
            raise FeaturePayloadIntegrityError(
                f"unsupported or corrupt payload for {entry.payload_reference}: {error}"
            ) from error

        if tuple(array.shape) != entry.shape:
            raise FeaturePayloadIntegrityError(
                f"shape mismatch for {entry.payload_reference}: "
                f"expected {entry.shape}, found {tuple(array.shape)}"
            )
        if str(array.dtype) != entry.dtype:
            raise FeaturePayloadIntegrityError(
                f"dtype mismatch for {entry.payload_reference}: "
                f"expected {entry.dtype!r}, found {array.dtype!s}"
            )
        return array


def write_feature_index(root: Path, entries: Sequence[FeaturePayloadEntry]) -> None:
    """Write a feature store's index file from a writer's accumulated entries.

    Args:
        root: The feature store root.
        entries: Entries to write, typically from
            :meth:`FeatureStoreWriter.entries`.
    """
    index_path = root / FEATURE_INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        f"{json.dumps(encode_feature_payload_entry(entry), sort_keys=True)}\n" for entry in entries
    )
    index_path.write_text(content, encoding="utf-8")


def encode_feature_payload_entry(entry: FeaturePayloadEntry) -> dict[str, Any]:
    """Encode a :class:`FeaturePayloadEntry` into a JSON-serializable dict."""
    return {
        "feature_id": str(entry.feature_id),
        "source_observation_id": str(entry.source_observation_id),
        "scope": entry.scope.value,
        "embedding_space_id": entry.embedding_space_id,
        "shape": list(entry.shape),
        "dtype": entry.dtype,
        "normalization": entry.normalization,
        "payload_reference": entry.payload_reference,
        "content_hash": entry.content_hash,
        "size_bytes": entry.size_bytes,
        "provenance": encode_provenance(entry.provenance),
    }


def decode_feature_payload_entry(record: dict[str, Any]) -> FeaturePayloadEntry:
    """Decode a :class:`FeaturePayloadEntry` from :func:`encode_feature_payload_entry`'s output."""
    return FeaturePayloadEntry(
        feature_id=FeatureId(record["feature_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        scope=FeatureScope(record["scope"]),
        embedding_space_id=record["embedding_space_id"],
        shape=tuple(record["shape"]),
        dtype=record["dtype"],
        normalization=record["normalization"],
        payload_reference=record["payload_reference"],
        content_hash=record["content_hash"],
        size_bytes=record["size_bytes"],
        provenance=decode_provenance(record["provenance"]),
    )


def _resolve_within_root(root: Path, relative_reference: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / relative_reference).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise FeatureStoreError(
            f"payload_reference resolves outside the feature store boundary: {relative_reference!r}"
        )
    return candidate
