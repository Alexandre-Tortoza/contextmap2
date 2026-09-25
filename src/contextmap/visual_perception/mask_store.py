"""Durable region-mask storage: writer, index, and lazy/random-access reader.

A :class:`~contextmap.visual_perception.models.Region2D`'s mask, when a
mask-based discovery backend (SAM2, SAM3) produced one, used to be
inlined into ``outputs/results.jsonl`` as a flat JSON array of one
integer per pixel across the *whole image* — about 0.92 MB for a single
640x480 mask, regardless of how small the region's own bounding box was
(#378). This module persists that pixel payload separately, the same
way :mod:`contextmap.visual_perception.feature_store` already persists
feature payloads: a compact binary file per mask, referenced by
:class:`~contextmap.visual_perception.models.Region2D.mask_reference`,
plus a small versioned index (``mask-index.jsonl``) carrying the
content hash and shape needed to load and verify it on demand.

Compactness comes from bit-packing (:func:`numpy.packbits`), not from
cropping to the region's bounding box: a full 640x480 mask packs to
38,400 bytes (one bit per pixel) regardless of content, an 8x reduction
over one byte per pixel and roughly a 24x reduction over the JSON array
it replaces, and it keeps the payload's shape identical to
``(image_height, image_width)`` — no crop-offset bookkeeping, no
coordinate transform to invert on load, and no discrepancy with the
``InlineMask`` contract's existing full-image invariant
(``_validate_geometry_bounds`` in ``region_models.py``). This is enough
to bring ``results.jsonl`` back to metadata-only size (moving the pixel
payload out of it dominates the reduction, see ``docs/run_artifact.md``);
a bounding-box crop was considered and rejected as unneeded complexity
for the size this issue actually asks for. Unpacking with
:func:`numpy.unpackbits` (bounded by the exact original bit count) makes
the round trip bit-exact.

As with :mod:`contextmap.visual_perception.feature_store`, NumPy is only
imported inside the functions that serialize/deserialize array bytes —
reading metadata or the index never requires it.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import RegionId
from contextmap.visual_perception.region_models import InlineMask

MASK_INDEX_FILENAME = "mask-index.jsonl"
"""Name of the mask store's index file, relative to its root."""

MASK_INDEX_SCHEMA_VERSION = "0.1.0"
"""Mask index schema version written and understood by this module."""

_MASK_INDEX_RECORD_TYPE = "mask_index"


class MaskStoreError(Exception):
    """Base class for region-mask payload storage failures."""


class MaskPayloadIntegrityError(MaskStoreError):
    """Raised when a persisted mask payload fails an integrity check on load."""


@dataclass(frozen=True, kw_only=True)
class MaskPayloadEntry:
    """Indexed metadata for one persisted region-mask payload.

    Every field here is readable without loading the packed pixel
    payload — this is exactly what the index (``mask-index.jsonl``)
    persists.

    Attributes:
        region_id: The region this entry describes, local to
            ``source_observation_id`` (see
            :class:`~contextmap.visual_perception.models.Region2D`).
        source_observation_id: The physical observation the region was
            proposed for.
        width: Mask width in pixels, equal to the region's image width.
        height: Mask height in pixels, equal to the region's image
            height.
        payload_reference: Path to the payload file, relative to the
            mask store root.
        content_hash: ``"sha256:<hex digest>"`` of the raw (packed)
            file bytes.
        size_bytes: Size of the payload file in bytes.
    """

    region_id: RegionId
    source_observation_id: SourceObservationId
    width: int
    height: int
    payload_reference: str
    content_hash: str
    size_bytes: int


class MaskStoreWriter:
    """Writes region-mask payloads under a mask store root, accumulating an index."""

    def __init__(self, root: Path) -> None:
        """Create a writer rooted at ``root``.

        Args:
            root: Directory payload paths are resolved relative to
                (typically a run artifact's ``outputs/masks/``).
        """
        self._root = root
        self._entries: list[MaskPayloadEntry] = []
        self._keys: set[tuple[SourceObservationId, RegionId]] = set()
        self._payload_paths: set[Path] = set()

    def write(
        self,
        *,
        region_id: RegionId,
        source_observation_id: SourceObservationId,
        mask: InlineMask,
    ) -> MaskPayloadEntry:
        """Bit-pack and persist one region's mask.

        Args:
            region_id: The region this mask belongs to, local to
                ``source_observation_id``.
            source_observation_id: The physical observation the region
                was proposed for.
            mask: The full-image mask to persist.

        Returns:
            The indexed entry describing the persisted payload.

        Raises:
            MaskStoreError: If this ``(source_observation_id, region_id)``
                pair was already written, or the computed payload path
                resolves outside ``root``.
        """
        import numpy as np

        key = (source_observation_id, region_id)
        if key in self._keys:
            raise MaskStoreError(
                "duplicate mask payload key: "
                f"source_observation_id={source_observation_id!r}, region_id={region_id!r}"
            )

        payload_reference = f"{source_observation_id}/{region_id}.npy"
        full_path = _resolve_within_root(self._root, payload_reference)
        if full_path in self._payload_paths:
            raise MaskStoreError(f"duplicate payload_reference: {payload_reference!r}")
        full_path.parent.mkdir(parents=True, exist_ok=True)

        packed = np.packbits(np.array(mask.data, dtype=np.bool_))
        buffer = io.BytesIO()
        np.save(buffer, packed, allow_pickle=False)
        data = buffer.getvalue()
        full_path.write_bytes(data)

        entry = MaskPayloadEntry(
            region_id=region_id,
            source_observation_id=source_observation_id,
            width=mask.width,
            height=mask.height,
            payload_reference=payload_reference,
            content_hash=f"sha256:{hashlib.sha256(data).hexdigest()}",
            size_bytes=len(data),
        )
        self._entries.append(entry)
        self._keys.add(key)
        self._payload_paths.add(full_path)
        return entry

    def entries(self) -> Sequence[MaskPayloadEntry]:
        """Every entry written so far, in write order."""
        return tuple(self._entries)


class MaskStoreReader:
    """Reads a mask store's index, then loads individual payloads lazily."""

    def __init__(self, root: Path, entries: Sequence[MaskPayloadEntry]) -> None:
        """Wrap an already-loaded index.

        Prefer :meth:`open` to read a persisted index from disk.
        """
        self._root = root
        self._by_key: dict[tuple[SourceObservationId, RegionId], MaskPayloadEntry] = {}
        payload_paths: set[Path] = set()
        for entry in entries:
            key = (entry.source_observation_id, entry.region_id)
            if key in self._by_key:
                raise MaskStoreError(
                    "duplicate mask payload key in index: "
                    f"source_observation_id={entry.source_observation_id!r}, "
                    f"region_id={entry.region_id!r}"
                )
            payload_path = _resolve_within_root(root, entry.payload_reference)
            if payload_path in payload_paths:
                raise MaskStoreError(
                    f"duplicate payload_reference in index: {entry.payload_reference!r}"
                )
            self._by_key[key] = entry
            payload_paths.add(payload_path)

    @classmethod
    def open(cls, root: Path) -> MaskStoreReader:
        """Open a mask store's index without loading any payload.

        Args:
            root: The mask store root (typically a run artifact's
                ``outputs/masks/``).

        Returns:
            A reader over the index found at ``root``, or an empty
            reader when ``root`` has no index yet (no region of this
            run carried a mask).
        """
        index_path = root / MASK_INDEX_FILENAME
        if not index_path.is_file():
            return cls(root, ())

        entries: list[MaskPayloadEntry] = []
        header_found = False
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if stripped:
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as error:
                        raise MaskStoreError(
                            f"invalid mask index JSON at line {line_number}: {error}"
                        ) from error
                    if not isinstance(record, dict):
                        raise MaskStoreError(
                            f"mask index record at line {line_number} must be an object"
                        )
                    if not header_found:
                        if record.get("record_type") != _MASK_INDEX_RECORD_TYPE:
                            raise MaskStoreError("mask index is missing its schema header")
                        schema_version = record.get("schema_version")
                        if schema_version != MASK_INDEX_SCHEMA_VERSION:
                            raise MaskStoreError(
                                f"unsupported mask index schema_version: {schema_version!r}"
                            )
                        header_found = True
                    else:
                        try:
                            entries.append(decode_mask_payload_entry(record))
                        except (KeyError, TypeError, ValueError) as error:
                            raise MaskStoreError(
                                f"invalid mask index entry at line {line_number}: {error}"
                            ) from error

        if not header_found:
            raise MaskStoreError("mask index is missing its schema header")
        return cls(root, entries)

    def region_keys(self) -> Sequence[tuple[SourceObservationId, RegionId]]:
        """Every observation-local region key in index order."""
        return tuple(self._by_key)

    def entry(
        self, source_observation_id: SourceObservationId, region_id: RegionId
    ) -> MaskPayloadEntry:
        """Return one mask's indexed metadata without loading its payload.

        Raises:
            MaskStoreError: If no entry exists for the observation-local
                ``region_id``.
        """
        try:
            return self._by_key[(source_observation_id, region_id)]
        except KeyError:
            raise MaskStoreError(
                "no persisted mask for "
                f"source_observation_id={source_observation_id!r}, region_id={region_id!r}"
            ) from None

    def load(self, source_observation_id: SourceObservationId, region_id: RegionId) -> InlineMask:
        """Load, hash-verify, and unpack one region's full-image mask.

        Args:
            source_observation_id: The physical observation owning the region.
            region_id: The observation-local region to load.

        Returns:
            The mask, bit-exact with what was originally persisted.

        Raises:
            MaskStoreError: If no entry exists for ``region_id``, or the
                payload file is missing.
            MaskPayloadIntegrityError: If the payload's content hash does
                not match its indexed entry, it has fewer bits than
                ``width * height``, or the file is not a valid/supported
                ``.npy`` payload.
        """
        import numpy as np

        entry = self.entry(source_observation_id, region_id)
        full_path = _resolve_within_root(self._root, entry.payload_reference)
        if not full_path.is_file():
            raise MaskStoreError(f"missing payload file: {entry.payload_reference}")

        data = full_path.read_bytes()
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        if digest != entry.content_hash:
            raise MaskPayloadIntegrityError(
                f"content hash mismatch for {entry.payload_reference}: "
                f"expected {entry.content_hash}, found {digest}"
            )

        try:
            packed = np.load(io.BytesIO(data), allow_pickle=False)
        except (ValueError, OSError) as error:
            raise MaskPayloadIntegrityError(
                f"unsupported or corrupt payload for {entry.payload_reference}: {error}"
            ) from error

        bit_count = entry.width * entry.height
        unpacked = np.unpackbits(packed, count=bit_count)
        if unpacked.shape[0] != bit_count:
            raise MaskPayloadIntegrityError(
                f"mask payload for {entry.payload_reference} has fewer than {bit_count} bits"
            )
        return InlineMask(
            width=entry.width,
            height=entry.height,
            data=tuple(bool(value) for value in unpacked.tolist()),
        )


def write_mask_index(root: Path, entries: Sequence[MaskPayloadEntry]) -> None:
    """Write a mask store's index file from a writer's accumulated entries.

    Args:
        root: The mask store root.
        entries: Entries to write, typically from
            :meth:`MaskStoreWriter.entries`.
    """
    index_path = root / MASK_INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": _MASK_INDEX_RECORD_TYPE,
        "schema_version": MASK_INDEX_SCHEMA_VERSION,
    }
    content = f"{json.dumps(header, sort_keys=True)}\n" + "".join(
        f"{json.dumps(encode_mask_payload_entry(entry), sort_keys=True)}\n" for entry in entries
    )
    index_path.write_text(content, encoding="utf-8")


def encode_mask_payload_entry(entry: MaskPayloadEntry) -> dict[str, Any]:
    """Encode a :class:`MaskPayloadEntry` into a JSON-serializable dict."""
    return {
        "region_id": str(entry.region_id),
        "source_observation_id": str(entry.source_observation_id),
        "width": entry.width,
        "height": entry.height,
        "payload_reference": entry.payload_reference,
        "content_hash": entry.content_hash,
        "size_bytes": entry.size_bytes,
    }


def decode_mask_payload_entry(record: dict[str, Any]) -> MaskPayloadEntry:
    """Decode a :class:`MaskPayloadEntry` from :func:`encode_mask_payload_entry`'s output."""
    return MaskPayloadEntry(
        region_id=RegionId(record["region_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        width=record["width"],
        height=record["height"],
        payload_reference=record["payload_reference"],
        content_hash=record["content_hash"],
        size_bytes=record["size_bytes"],
    )


def _resolve_within_root(root: Path, relative_reference: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / relative_reference).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise MaskStoreError(
            f"payload_reference resolves outside the mask store boundary: {relative_reference!r}"
        )
    return candidate
