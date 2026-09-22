"""JSON Lines record tables with a byte-offset index.

Entities and relations are stored one canonical record per line, ordered by key, with a separate
index that says where each line starts and how long it is. That gives the two properties a map
consumer needs: the file streams and hashes in one pass, and **one record is read by key without
parsing any other**. The index is derived data: a damaged one is an explicit error, never a
silently wrong record, and the authoritative lines are what a validator rebuilds it from.

Every line is a JSON object with a string ``key``; the rest of the object belongs to whoever
writes the table. Lines are ASCII, with sorted keys and compact separators, so the same records
always produce the same bytes and no line contains a raw newline. The module reads only the
files it is given and never writes to them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from contextmap.artifact.serialization.errors import (
    BrokenIndexError,
    MissingPayloadError,
    RecordNotFoundError,
    RecordTableError,
)

_INDEX_FIELDS = frozenset({"key", "offset", "length"})


def canonical_json_line(record: Mapping[str, Any]) -> bytes:
    """Encode one record as a single canonical JSON line, without the trailing newline.

    Args:
        record: A JSON-safe mapping.

    Returns:
        ASCII bytes with sorted keys and compact separators.

    Raises:
        RecordTableError: If the record holds a non-finite number (which is not valid JSON) or
            a value JSON cannot represent.
    """
    try:
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except ValueError as error:
        raise RecordTableError(
            f"record contains a non-finite number, which is not valid JSON ({error})"
        ) from error
    except TypeError as error:
        raise RecordTableError(f"record is not JSON-serializable ({error})") from error
    return text.encode("ascii")


def document_json(record: Mapping[str, Any]) -> bytes:
    """Encode a JSON document for people and tools: sorted, indented, newline-terminated.

    Args:
        record: A JSON-safe mapping.

    Returns:
        UTF-8 bytes; the same mapping always gives the same bytes.

    Raises:
        RecordTableError: If the document holds a non-finite number or a value JSON cannot
            represent.
    """
    try:
        text = json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
    except ValueError as error:
        raise RecordTableError(
            f"document contains a non-finite number, which is not valid JSON ({error})"
        ) from error
    except TypeError as error:
        raise RecordTableError(f"document is not JSON-serializable ({error})") from error
    return (text + "\n").encode("ascii")


@dataclass(frozen=True)
class EncodedTable:
    """A table ready to be written: its lines and the index that locates them.

    Attributes:
        payload: The lines, one canonical JSON object each, ordered by key.
        index: One ``{"key", "length", "offset"}`` line per record, in the same order.
        record_count: Number of records.
    """

    payload: bytes
    index: bytes
    record_count: int


def encode_record_table(lines: Iterable[Mapping[str, Any]]) -> EncodedTable:
    """Encode records as a table ordered by key, with its offset index.

    The result depends only on the records, not on the order they arrive in.

    Args:
        lines: JSON-safe mappings, each with a non-empty string ``key`` unique in the table.

    Returns:
        The payload and the index.

    Raises:
        RecordTableError: If a key is missing, empty or not a string, a key is duplicated
            (nothing is dropped silently), or a record cannot be encoded as JSON.
    """
    keyed: list[tuple[str, Mapping[str, Any]]] = []
    for line in lines:
        key = line.get("key")
        if not isinstance(key, str) or not key:
            raise RecordTableError(f"every record needs a non-empty string 'key', got {key!r}")
        keyed.append((key, line))
    keyed.sort(key=lambda item: item[0])
    for (previous, _), (current, _) in pairwise(keyed):
        if previous == current:
            raise RecordTableError(f"duplicate key {current!r}")

    payload = bytearray()
    index = bytearray()
    for key, line in keyed:
        try:
            encoded = canonical_json_line(line)
        except RecordTableError as error:
            raise RecordTableError(f"record {key!r}: {error}") from error
        index += canonical_json_line({"key": key, "length": len(encoded), "offset": len(payload)})
        index += b"\n"
        payload += encoded + b"\n"
    return EncodedTable(payload=bytes(payload), index=bytes(index), record_count=len(keyed))


def rebuild_index(payload: bytes) -> bytes:
    """Recompute the offset index of a table payload from the payload alone.

    The payload is authoritative and the index is derived, so a validator rebuilds the index and
    compares it byte for byte with the stored one: any difference means the stored index is not
    the one these lines would produce.

    Args:
        payload: The bytes of a JSON Lines table.

    Returns:
        The index that :func:`encode_record_table` would have written for these lines.

    Raises:
        RecordTableError: If the payload is not a canonical table: a line is not terminated by a
            newline or is not a JSON object with a string ``key``, or the keys are not strictly
            ascending (which is also how a duplicate key shows).
    """
    if payload and not payload.endswith(b"\n"):
        raise RecordTableError("the table is not terminated by a newline")
    index = bytearray()
    offset = 0
    previous: str | None = None
    for text in payload.split(b"\n")[:-1]:
        try:
            line = json.loads(text)
        except ValueError as error:
            raise RecordTableError(
                f"the line at offset {offset} is not valid JSON ({error})"
            ) from error
        key = line.get("key") if isinstance(line, dict) else None
        if not isinstance(key, str) or not key:
            raise RecordTableError(f"the line at offset {offset} has no non-empty string 'key'")
        if previous is not None and key <= previous:
            raise RecordTableError(
                f"the keys are not strictly ascending: {key!r} follows {previous!r} "
                "(a repeated key is a duplicate)"
            )
        index += canonical_json_line({"key": key, "length": len(text), "offset": offset}) + b"\n"
        offset += len(text) + 1
        previous = key
    return bytes(index)


class RecordTable:
    """Read-only, random-access view of a table on disk.

    Opening a table reads and checks only the index, which is proportional to the number of
    records. A record is read, and only then parsed, when it is asked for.
    """

    def __init__(self, payload_path: Path, index_path: Path, *, record_count: int) -> None:
        """Open a table and verify its index against the payload's size.

        Args:
            payload_path: The JSON Lines file.
            index_path: The index of that file.
            record_count: The number of records the manifest declares for the table.

        Raises:
            MissingPayloadError: If either file is missing.
            BrokenIndexError: If the index is malformed, unsorted, not contiguous, disagrees
                with ``record_count``, or does not account for exactly the bytes of the payload.
        """
        for path in (payload_path, index_path):
            if not path.is_file():
                raise MissingPayloadError(f"missing file {path.name} of a record table")
        self._path = payload_path
        entries = _load_index(index_path)
        if len(entries) != record_count:
            raise BrokenIndexError(
                f"{index_path.name} has {len(entries)} entries but the manifest declares "
                f"record_count={record_count}"
            )
        expected_size = _check_contiguous(entries, index_path.name)
        size = payload_path.stat().st_size
        if size != expected_size:
            raise BrokenIndexError(
                f"{payload_path.name} has {size} bytes but {index_path.name} "
                f"accounts for {expected_size} bytes"
            )
        self._entries = {key: (offset, length) for key, offset, length in entries}
        self._keys = tuple(key for key, _, _ in entries)

    @property
    def keys(self) -> tuple[str, ...]:
        """Every key, in the order of the file."""
        return self._keys

    def __len__(self) -> int:
        """Number of records."""
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        """Whether a record with this key exists."""
        return key in self._entries

    def read(self, key: str) -> dict[str, Any]:
        """Read and parse the one record with this key.

        Args:
            key: The record key.

        Returns:
            The parsed line.

        Raises:
            RecordNotFoundError: If there is no such record.
            BrokenIndexError: If the bytes at the indexed position are not one well-formed
                line with that key.
        """
        try:
            offset, length = self._entries[key]
        except KeyError:
            raise RecordNotFoundError(f"no record with key {key!r} in {self._path.name}") from None
        with self._path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(length + 1)
        return self._decode(chunk, key)

    def iter_lines(self) -> Iterator[dict[str, Any]]:
        """Read every record once, in file order.

        Returns:
            The parsed lines.

        Raises:
            BrokenIndexError: If a line is not well formed or does not match the index.
        """
        with self._path.open("rb") as handle:
            for key in self._keys:
                _, length = self._entries[key]
                yield self._decode(handle.read(length + 1), key)

    def _decode(self, chunk: bytes, key: str) -> dict[str, Any]:
        if not chunk.endswith(b"\n"):
            raise BrokenIndexError(
                f"the record {key!r} of {self._path.name} is not terminated by a newline "
                "at its indexed position"
            )
        try:
            line = json.loads(chunk[:-1])
        except ValueError as error:
            raise BrokenIndexError(
                f"the record {key!r} of {self._path.name} is not valid JSON ({error})"
            ) from error
        if not isinstance(line, dict) or line.get("key") != key:
            found = line.get("key") if isinstance(line, dict) else None
            raise BrokenIndexError(
                f"the index of {self._path.name} points at the record {found!r} instead of {key!r}"
            )
        return line


def _load_index(index_path: Path) -> list[tuple[str, int, int]]:
    raw = index_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise BrokenIndexError(f"{index_path.name} is not terminated by a newline")
    entries: list[tuple[str, int, int]] = []
    for number, text in enumerate(raw.split(b"\n")[:-1]):
        try:
            entry = json.loads(text)
        except ValueError as error:
            raise BrokenIndexError(
                f"entry {number} of {index_path.name} is not valid JSON ({error})"
            ) from error
        if not isinstance(entry, dict) or entry.keys() != _INDEX_FIELDS:
            raise BrokenIndexError(
                f"entry {number} of {index_path.name} must have exactly the fields "
                f"{sorted(_INDEX_FIELDS)}"
            )
        key, offset, length = entry["key"], entry["offset"], entry["length"]
        if not isinstance(key, str) or not key:
            raise BrokenIndexError(f"entry {number} of {index_path.name} has an invalid key")
        for name, value in (("offset", offset), ("length", length)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BrokenIndexError(
                    f"entry {number} of {index_path.name} has an invalid {name}: {value!r}"
                )
        entries.append((key, offset, length))
    for (previous, _, _), (current, _, _) in pairwise(entries):
        if previous >= current:
            raise BrokenIndexError(
                f"the keys of {index_path.name} are not strictly sorted: "
                f"{current!r} follows {previous!r}"
            )
    return entries


def _check_contiguous(entries: list[tuple[str, int, int]], index_name: str) -> int:
    """Check that the lines tile the payload with no gap or overlap; return its total size."""
    expected_offset = 0
    for key, offset, length in entries:
        if offset != expected_offset:
            raise BrokenIndexError(
                f"the offset of {key!r} in {index_name} is {offset}, expected {expected_offset}"
            )
        expected_offset = offset + length + 1
    return expected_offset


def encode_entity_relation_index(
    entity_keys: Iterable[str], endpoints: Iterable[tuple[str, str, str]]
) -> bytes:
    """Encode the traversal index from entities to the relations they take part in.

    One line per entity, ordered by key, listing the keys of the relations in which it is the
    subject and those in which it is the object. It answers ``relations_for(entity)`` and
    ``relation -> entity`` traversal without scanning the relation table, and it is derived:
    it is rebuilt from the two tables and never redefines them.

    Args:
        entity_keys: The key of every entity of the map.
        endpoints: ``(relation key, subject entity key, object entity key)`` of every relation.

    Returns:
        The JSON Lines payload.

    Raises:
        RecordTableError: If a relation names an entity that is not in ``entity_keys``.
    """
    known = set(entity_keys)
    as_subject: dict[str, list[str]] = {key: [] for key in known}
    as_object: dict[str, list[str]] = {key: [] for key in known}
    for relation_key, subject_key, object_key in endpoints:
        for role, entity_key in (("subject", subject_key), ("object", object_key)):
            if entity_key not in known:
                raise RecordTableError(
                    f"relation {relation_key!r} has the {role} {entity_key!r}, "
                    "which is not an entity of the map"
                )
        as_subject[subject_key].append(relation_key)
        as_object[object_key].append(relation_key)
    return b"".join(
        canonical_json_line(
            {"key": key, "as_subject": sorted(as_subject[key]), "as_object": sorted(as_object[key])}
        )
        + b"\n"
        for key in sorted(known)
    )
