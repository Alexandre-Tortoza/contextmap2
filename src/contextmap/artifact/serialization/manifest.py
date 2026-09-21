"""The manifest of a ContextMapArtifact: identity, inventory, payload descriptors, dependencies.

``manifest.json`` is the first file anyone opens and the root of the artifact's integrity. It says
which format and schema versions the artifact uses, inventories every contractual file with size
and SHA-256, describes each binary or tabular payload well enough to read it with ordinary tools
(encoding, dtype, shape, unit, frame, semantics) and lists the upstream artifacts the map
depends on, separating what is required to resolve the map from what is only optional evidence.

The manifest is plain JSON with no runtime, model or ROS object in it. See
``src/contextmap/artifact/docs/storage-layout.md`` for the layout and the reasons for each choice.

Identity
    ``content_identity`` is the SHA-256 of every field of the manifest except the write time and
    the dependency locators. It is therefore a pure function of the contractual content: writing
    the same map twice gives the same identity, moving the artifact or its dependencies does not
    change it, and any change to a file, a descriptor, a count or a dependency does.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from contextmap.artifact.serialization.errors import ManifestError, UnsupportedFormatVersionError
from contextmap.artifact.serialization.layout import (
    ARTIFACT_TYPE,
    FORMAT_VERSION,
    SUPPORTED_FORMAT_VERSIONS,
    require_contractual_path,
)
from contextmap.shared import FileEntry

COLUMN_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "uint8": "u1",
        "int32": "i4",
        "uint32": "u4",
        "int64": "i8",
        "uint64": "u8",
        "float32": "f4",
        "float64": "f8",
    }
)
"""Closed vocabulary of column dtypes, each mapped to its NumPy kind and byte width.

Every column is little-endian and C-ordered; there is no object, string or structured dtype, so
a column can never carry a pickle or a model-native tensor.
"""

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_BYTE_ORDER = "little"

_MANIFEST_KEYS = frozenset(
    {
        "artifact_type",
        "format_version",
        "schema_version",
        "context_map_id",
        "content_identity",
        "written_at",
        "code_version",
        "configuration_fingerprint",
        "entity_count",
        "relation_count",
        "payloads",
        "dependencies",
        "file_inventory",
    }
)
_RECORD_PAYLOAD_KEYS = frozenset(
    {"encoding", "path", "role", "semantics", "record_count", "derived_from"}
)
_COLUMN_PAYLOAD_KEYS = frozenset(
    {
        "encoding",
        "path",
        "role",
        "semantics",
        "dtype",
        "byte_order",
        "shape",
        "unit",
        "frame_id",
        "derived_from",
    }
)
_DEPENDENCY_KEYS = frozenset(
    {"artifact_type", "artifact_id", "content_identity", "requirement", "locator"}
)
_INVENTORY_KEYS = frozenset({"path", "size_bytes", "content_hash"})


class PayloadRole(Enum):
    """Whether a payload is authoritative or can be rebuilt from another one.

    Attributes:
        AUTHORITATIVE: The data itself; nothing else in the artifact can recreate it.
        DERIVED_INDEX: An index rebuilt deterministically from other payloads. A corrupt index
            never redefines the data it points into; it is detected by rebuilding it.
    """

    AUTHORITATIVE = "authoritative"
    DERIVED_INDEX = "derived_index"


class Requirement(Enum):
    """How much the map depends on an upstream artifact.

    Attributes:
        REQUIRED: Needed to resolve the map itself, such as the geometry that entities point
            into. A missing or mismatching required dependency makes the artifact unusable.
        OPTIONAL: Needed only to inspect evidence in depth. Its absence is a warning, because
            the core map stays fully readable without it.
    """

    REQUIRED = "required"
    OPTIONAL = "optional"


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a non-empty string, got {value!r}")


def _require_optional_text(value: object, field: str) -> None:
    if value is not None:
        _require_text(value, field)


def _require_count(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{field} must be a non-negative integer, got {value!r}")


def _require_digest(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.match(value) is None:
        raise ManifestError(f"{field} must be 'sha256:<64 hex digits>', got {value!r}")


def _require_supported_format_version(value: object) -> None:
    if value not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_FORMAT_VERSIONS))
        raise UnsupportedFormatVersionError(
            f"unsupported format_version {value!r}; this code reads: {supported}"
        )


def _require_payload_paths(path: str, role: PayloadRole, derived_from: tuple[str, ...]) -> None:
    _require_text(path, "path")
    require_contractual_path(path)
    if role is PayloadRole.AUTHORITATIVE and derived_from:
        raise ManifestError(f"authoritative payload {path!r} must not declare derived_from")
    if role is PayloadRole.DERIVED_INDEX and not derived_from:
        raise ManifestError(f"derived payload {path!r} must declare derived_from")
    for source in derived_from:
        require_contractual_path(source)


@dataclass(frozen=True, kw_only=True)
class RecordPayload:
    """A JSON Lines payload: one canonical record per line.

    Attributes:
        path: Path relative to the artifact directory.
        role: Whether the payload is authoritative or a derived index.
        semantics: What each line means, in a sentence a person can read.
        record_count: Number of lines (records).
        derived_from: Paths of the payloads a derived index is rebuilt from; empty otherwise.
    """

    path: str
    role: PayloadRole
    semantics: str
    record_count: int
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the descriptor.

        Raises:
            ManifestError: If the path is not contractual, the semantics are empty, the record
                count is negative, or ``derived_from`` disagrees with the role.
        """
        _require_payload_paths(self.path, self.role, self.derived_from)
        _require_text(self.semantics, "semantics")
        _require_count(self.record_count, "record_count")


@dataclass(frozen=True, kw_only=True)
class ColumnPayload:
    """A raw little-endian, C-ordered array with a fixed dtype and shape.

    The file has no header: everything needed to read it is here, so a person or a tool reads it
    with the standard library, ``numpy.memmap`` or any other tool without parsing anything.

    Attributes:
        path: Path relative to the artifact directory.
        role: Whether the payload is authoritative or a derived index.
        semantics: What each element and each axis mean.
        dtype: One of :data:`COLUMN_DTYPES`.
        shape: Extent of every axis; a zero axis is an explicit empty table.
        derived_from: Paths of the payloads a derived index is rebuilt from; empty otherwise.
        unit: Unit of the values (for example ``"m"``), or ``None`` when they have none, as
            indexes and counts do.
        frame_id: Coordinate frame of the values, or ``None`` when they are not coordinates.
    """

    path: str
    role: PayloadRole
    semantics: str
    dtype: str
    shape: tuple[int, ...]
    derived_from: tuple[str, ...] = ()
    unit: str | None = None
    frame_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the descriptor.

        Raises:
            ManifestError: If the path is not contractual, the dtype is not in
                :data:`COLUMN_DTYPES`, the shape is not a non-empty tuple of non-negative
                integers, a unit or frame is empty, or ``derived_from`` disagrees with the role.
        """
        _require_payload_paths(self.path, self.role, self.derived_from)
        _require_text(self.semantics, "semantics")
        if self.dtype not in COLUMN_DTYPES:
            raise ManifestError(
                f"dtype of {self.path!r} must be one of {sorted(COLUMN_DTYPES)}, got {self.dtype!r}"
            )
        if not self.shape or any(
            isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
            for extent in self.shape
        ):
            raise ManifestError(
                f"shape must be a non-empty tuple of non-negative integers, got {self.shape!r}"
            )
        _require_optional_text(self.unit, "unit")
        _require_optional_text(self.frame_id, "frame_id")

    @property
    def item_size_bytes(self) -> int:
        """Width of one element in bytes."""
        return int(COLUMN_DTYPES[self.dtype][1:])

    @property
    def expected_size_bytes(self) -> int:
        """Size the file must have: elements times the width of one element."""
        return math.prod(self.shape) * self.item_size_bytes

    @property
    def numpy_typestr(self) -> str:
        """The NumPy type string of the elements, always explicitly little-endian."""
        return "<" + COLUMN_DTYPES[self.dtype]


Payload = RecordPayload | ColumnPayload
"""A payload descriptor: a JSON Lines table or a raw column."""


@dataclass(frozen=True, kw_only=True)
class DependencyRecord:
    """An upstream artifact the map refers to instead of copying.

    Attributes:
        artifact_type: Kind of the upstream artifact, for example ``"geometric_map"``.
        artifact_id: Identity of the artifact inside its own capability; it is what a person or
            a resolver looks for, and is not verified on its own.
        content_identity: Digest of the upstream artifact's contractual inventory
            (:func:`inventory_digest`). It is what makes the reference exact: an artifact found
            by any path is only the dependency if its digest matches.
        requirement: Whether the map needs it to be resolved or only to inspect evidence.
        locator: A hint where to look, relative to the artifact directory. It is transport
            information: never trusted, never part of the identity, and rewritten when the
            artifact travels in a bundle.
    """

    artifact_type: str
    artifact_id: str
    content_identity: str
    requirement: Requirement
    locator: str | None = None

    def __post_init__(self) -> None:
        """Validate the record.

        Raises:
            ManifestError: If a text is empty, the digest is malformed, or the locator is not a
                relative POSIX path.
        """
        _require_text(self.artifact_type, "artifact_type")
        _require_text(self.artifact_id, "artifact_id")
        _require_digest(self.content_identity, "content_identity")
        if self.locator is not None:
            _require_text(self.locator, "locator")
            first = self.locator.split("/")[0]
            if (
                self.locator.startswith("/")
                or "\\" in self.locator
                or "\x00" in self.locator
                or first.endswith(":")
            ):
                raise ManifestError(f"locator must be a relative POSIX path, got {self.locator!r}")

    @property
    def key(self) -> tuple[str, str]:
        """What identifies the dependency within one manifest."""
        return (self.artifact_type, self.artifact_id)


@dataclass(frozen=True, kw_only=True)
class ContextMapArtifactManifest:
    """The authoritative, inspectable description of one ContextMapArtifact.

    Attributes:
        context_map_id: Identity of the map, as the schema defines it.
        schema_version: Version of the map's data semantics (the schema), independent of how
            they are stored.
        format_version: Version of the on-disk layout and encodings.
        content_identity: SHA-256 of the contractual content; see the module documentation.
        written_at: ISO 8601 write time with a UTC offset. Provenance only: it is not part of
            the identity.
        code_version: Revision of the code that wrote the artifact, when known.
        configuration_fingerprint: Identity of the effective configuration, when known.
        entity_count: Records in the entity table.
        relation_count: Records in the relation table.
        payloads: Every binary or tabular payload, sorted by path.
        dependencies: Every upstream artifact the map refers to, sorted by type and id.
        file_inventory: Every contractual file with size and hash, sorted by path. It excludes the
            manifest, the README and, by construction, any ``debug/`` file.
    """

    context_map_id: str
    schema_version: str
    format_version: str
    content_identity: str
    written_at: str
    code_version: str | None
    configuration_fingerprint: str | None
    entity_count: int
    relation_count: int
    payloads: tuple[Payload, ...]
    dependencies: tuple[DependencyRecord, ...]
    file_inventory: tuple[FileEntry, ...]

    def __post_init__(self) -> None:
        """Validate every field and the consistency between them.

        Raises:
            UnsupportedFormatVersionError: If the format version is not supported.
            ManifestError: If a field is malformed, a path is duplicated, an inventoried path is
                not contractual, a payload is not inventoried, or a derived payload names a
                source that is not a payload.
        """
        _require_supported_format_version(self.format_version)
        _require_text(self.context_map_id, "context_map_id")
        _require_text(self.schema_version, "schema_version")
        _require_digest(self.content_identity, "content_identity")
        _require_optional_text(self.code_version, "code_version")
        if self.configuration_fingerprint is not None:
            _require_text(self.configuration_fingerprint, "configuration_fingerprint")
        _require_count(self.entity_count, "entity_count")
        _require_count(self.relation_count, "relation_count")
        _require_write_time(self.written_at)

        inventory_paths = _unique([entry.path for entry in self.file_inventory], "inventory path")
        for entry in self.file_inventory:
            require_contractual_path(entry.path)
        payload_paths = _unique([payload.path for payload in self.payloads], "payload path")
        _unique([dependency.key for dependency in self.dependencies], "dependency")
        for payload in self.payloads:
            if payload.path not in inventory_paths:
                raise ManifestError(f"payload {payload.path!r} is not inventoried")
            for source in payload.derived_from:
                if source not in payload_paths:
                    raise ManifestError(
                        f"derived_from {source!r} of payload {payload.path!r} "
                        "is not a payload of this manifest"
                    )


def _unique(items: list[Any], what: str) -> set[Any]:
    seen: set[Any] = set()
    for item in items:
        if item in seen:
            raise ManifestError(f"duplicate {what}: {item!r}")
        seen.add(item)
    return seen


def _require_write_time(value: object) -> None:
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise ManifestError(f"written_at must be an ISO 8601 time with a UTC offset, got {value!r}")


def create_manifest(
    *,
    context_map_id: str,
    schema_version: str,
    written_at: str,
    code_version: str | None,
    configuration_fingerprint: str | None,
    entity_count: int,
    relation_count: int,
    payloads: Iterable[Payload],
    dependencies: Iterable[DependencyRecord],
    file_inventory: Iterable[FileEntry],
) -> ContextMapArtifactManifest:
    """Build the manifest of a new artifact, sorting its parts and computing its identity.

    Args:
        context_map_id: Identity of the map.
        schema_version: Version of the map's data semantics.
        written_at: ISO 8601 write time with a UTC offset.
        code_version: Revision of the writing code, when known.
        configuration_fingerprint: Identity of the effective configuration, when known.
        entity_count: Records in the entity table.
        relation_count: Records in the relation table.
        payloads: The binary and tabular payload descriptors, in any order.
        dependencies: The upstream artifacts referred to, in any order.
        file_inventory: Every contractual file with size and hash, in any order.

    Returns:
        A manifest of the current format version whose ``content_identity`` is computed.

    Raises:
        ManifestError: If the parts are malformed or inconsistent.
    """
    provisional = ContextMapArtifactManifest(
        context_map_id=context_map_id,
        schema_version=schema_version,
        format_version=FORMAT_VERSION,
        content_identity=_PLACEHOLDER_IDENTITY,
        written_at=written_at,
        code_version=code_version,
        configuration_fingerprint=configuration_fingerprint,
        entity_count=entity_count,
        relation_count=relation_count,
        payloads=tuple(sorted(payloads, key=lambda payload: payload.path)),
        dependencies=tuple(sorted(dependencies, key=lambda dependency: dependency.key)),
        file_inventory=tuple(sorted(file_inventory, key=lambda entry: entry.path)),
    )
    return replace(provisional, content_identity=manifest_content_identity(provisional))


def inventory_digest(entries: Iterable[FileEntry]) -> str:
    """Digest an inventory of files, independent of the order of its entries.

    It identifies the contractual content of an artifact that has an inventory, which is how a
    dependency is pinned to exact bytes.

    Args:
        entries: The ``(path, size, hash)`` entries.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    rows = sorted([entry.path, entry.size_bytes, entry.content_hash] for entry in entries)
    return _digest({"kind": "file-inventory", "files": rows})


def manifest_content_identity(manifest: ContextMapArtifactManifest) -> str:
    """Compute the content identity a manifest should carry.

    The write time and the dependency locators are left out: they describe when and where, not
    what, so the same map written twice, or moved with its dependencies, keeps its identity.

    Args:
        manifest: The manifest to identify; its own ``content_identity`` is ignored.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    record = encode_manifest(manifest)
    del record["written_at"], record["content_identity"]
    for dependency in record["dependencies"]:
        del dependency["locator"]
    return _digest({"kind": "context-map-manifest", "manifest": record})


def _digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def encode_manifest(manifest: ContextMapArtifactManifest) -> dict[str, Any]:
    """Encode a manifest as a JSON-safe record, every list in canonical order.

    Args:
        manifest: The manifest to encode.

    Returns:
        A record that :func:`decode_manifest` turns back into an equal manifest.
    """
    return {
        "artifact_type": ARTIFACT_TYPE,
        "format_version": manifest.format_version,
        "schema_version": manifest.schema_version,
        "context_map_id": manifest.context_map_id,
        "content_identity": manifest.content_identity,
        "written_at": manifest.written_at,
        "code_version": manifest.code_version,
        "configuration_fingerprint": manifest.configuration_fingerprint,
        "entity_count": manifest.entity_count,
        "relation_count": manifest.relation_count,
        "payloads": [
            _encode_payload(payload) for payload in sorted(manifest.payloads, key=_path_of)
        ],
        "dependencies": [
            {
                "artifact_type": dependency.artifact_type,
                "artifact_id": dependency.artifact_id,
                "content_identity": dependency.content_identity,
                "requirement": dependency.requirement.value,
                "locator": dependency.locator,
            }
            for dependency in sorted(manifest.dependencies, key=lambda item: item.key)
        ],
        "file_inventory": [
            {
                "path": entry.path,
                "size_bytes": entry.size_bytes,
                "content_hash": entry.content_hash,
            }
            for entry in sorted(manifest.file_inventory, key=lambda item: item.path)
        ],
    }


def _path_of(payload: Payload) -> str:
    return payload.path


def _encode_payload(payload: Payload) -> dict[str, Any]:
    if isinstance(payload, RecordPayload):
        return {
            "encoding": "jsonl",
            "path": payload.path,
            "role": payload.role.value,
            "semantics": payload.semantics,
            "record_count": payload.record_count,
            "derived_from": list(payload.derived_from),
        }
    return {
        "encoding": "raw-le",
        "path": payload.path,
        "role": payload.role.value,
        "semantics": payload.semantics,
        "dtype": payload.dtype,
        "byte_order": _BYTE_ORDER,
        "shape": list(payload.shape),
        "unit": payload.unit,
        "frame_id": payload.frame_id,
        "derived_from": list(payload.derived_from),
    }


def decode_manifest(record: Mapping[str, Any]) -> ContextMapArtifactManifest:
    """Decode and validate a manifest record read from ``manifest.json``.

    The decoding is strict: a missing field, an unknown field or a malformed value is an error,
    never a default, so damage or a newer format cannot be read as a valid old one. The recorded
    ``content_identity`` is kept as written; comparing it with
    :func:`manifest_content_identity` is how tampering is detected.

    Args:
        record: The parsed JSON.

    Returns:
        The manifest.

    Raises:
        UnsupportedFormatVersionError: If ``format_version`` is not a supported version.
        ManifestError: If the record is not a ContextMapArtifact manifest or is malformed or
            inconsistent.
    """
    if not isinstance(record, Mapping):
        raise ManifestError(f"a manifest must be a JSON object, got {type(record).__name__}")
    if record.get("artifact_type") != ARTIFACT_TYPE:
        raise ManifestError(
            f"artifact_type must be {ARTIFACT_TYPE!r}, got {record.get('artifact_type')!r}"
        )
    _require_supported_format_version(record.get("format_version"))
    _require_keys(record, _MANIFEST_KEYS, "manifest")
    payloads = tuple(_decode_payload(item) for item in _require_list(record, "payloads"))
    dependencies = tuple(_decode_dependency(item) for item in _require_list(record, "dependencies"))
    inventory = tuple(
        _decode_inventory_entry(item) for item in _require_list(record, "file_inventory")
    )
    return ContextMapArtifactManifest(
        context_map_id=record["context_map_id"],
        schema_version=record["schema_version"],
        format_version=record["format_version"],
        content_identity=record["content_identity"],
        written_at=record["written_at"],
        code_version=record["code_version"],
        configuration_fingerprint=record["configuration_fingerprint"],
        entity_count=record["entity_count"],
        relation_count=record["relation_count"],
        payloads=tuple(sorted(payloads, key=_path_of)),
        dependencies=tuple(sorted(dependencies, key=lambda item: item.key)),
        file_inventory=tuple(sorted(inventory, key=lambda item: item.path)),
    )


def _require_keys(record: Mapping[str, Any], expected: frozenset[str], what: str) -> None:
    missing = sorted(expected - record.keys())
    unknown = sorted(record.keys() - expected)
    if missing:
        raise ManifestError(f"{what} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ManifestError(f"{what} has unknown fields: {', '.join(unknown)}")


def _require_list(record: Mapping[str, Any], field: str) -> list[Any]:
    value = record[field]
    if not isinstance(value, list):
        raise ManifestError(f"{field} must be a list, got {type(value).__name__}")
    return value


def _require_object(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _decode_role(value: object) -> PayloadRole:
    try:
        return PayloadRole(value)
    except ValueError:
        allowed = [role.value for role in PayloadRole]
        raise ManifestError(f"role must be one of {allowed}, got {value!r}") from None


def _decode_string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"{field} must be a list of strings, got {value!r}")
    return tuple(value)


def _decode_payload(item: object) -> Payload:
    record = _require_object(item, "a payload descriptor")
    encoding = record.get("encoding")
    if encoding == "jsonl":
        _require_keys(record, _RECORD_PAYLOAD_KEYS, "jsonl payload descriptor")
        return RecordPayload(
            path=record["path"],
            role=_decode_role(record["role"]),
            semantics=record["semantics"],
            record_count=record["record_count"],
            derived_from=_decode_string_list(record["derived_from"], "derived_from"),
        )
    if encoding == "raw-le":
        _require_keys(record, _COLUMN_PAYLOAD_KEYS, "raw-le payload descriptor")
        if record["byte_order"] != _BYTE_ORDER:
            raise ManifestError(f"byte_order must be {_BYTE_ORDER!r}, got {record['byte_order']!r}")
        shape = record["shape"]
        if not isinstance(shape, list):
            raise ManifestError(f"shape must be a list, got {shape!r}")
        return ColumnPayload(
            path=record["path"],
            role=_decode_role(record["role"]),
            semantics=record["semantics"],
            dtype=record["dtype"],
            shape=tuple(shape),
            derived_from=_decode_string_list(record["derived_from"], "derived_from"),
            unit=record["unit"],
            frame_id=record["frame_id"],
        )
    raise ManifestError(f"payload encoding must be 'jsonl' or 'raw-le', got {encoding!r}")


def _decode_dependency(item: object) -> DependencyRecord:
    record = _require_object(item, "a dependency record")
    _require_keys(record, _DEPENDENCY_KEYS, "dependency record")
    try:
        requirement = Requirement(record["requirement"])
    except ValueError:
        raise ManifestError(
            f"requirement must be 'required' or 'optional', got {record['requirement']!r}"
        ) from None
    return DependencyRecord(
        artifact_type=record["artifact_type"],
        artifact_id=record["artifact_id"],
        content_identity=record["content_identity"],
        requirement=requirement,
        locator=record["locator"],
    )


def _decode_inventory_entry(item: object) -> FileEntry:
    record = _require_object(item, "an inventory entry")
    _require_keys(record, _INVENTORY_KEYS, "inventory entry")
    _require_text(record["path"], "path")
    _require_count(record["size_bytes"], "size_bytes")
    _require_digest(record["content_hash"], "content_hash")
    return FileEntry(
        path=record["path"],
        size_bytes=record["size_bytes"],
        content_hash=record["content_hash"],
    )
