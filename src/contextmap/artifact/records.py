"""Canonical record view of the ContextMap schema.

A *record* is the schema written as plain JSON-compatible values: mappings, lists, strings,
numbers, booleans and ``None``. It has one entry per field of the contract, named exactly like
the field, and enums are written as their values. It says nothing about files, formats or
layout: a serializer chooses how records reach storage, and the schema tests use records to
prove that a map survives a round trip without depending on any serializer.

Decoding is strict on purpose. A missing field, an unknown field, a value of the wrong type or
a value that violates an invariant is an error that names where it happened; nothing is
defaulted, ignored or repaired. The schema version is checked before any other field, so a
record written under an unreadable version is refused whole instead of read partially.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import Any, NewType, Union, get_args, get_origin, get_type_hints

from contextmap.artifact.models import ContextMap
from contextmap.artifact.versioning import (
    UnsupportedSchemaVersionError,
    require_supported_schema_version,
)


class ContextMapRecordError(ValueError):
    """Raised when a record does not have the shape of the schema."""


def context_map_to_record(context_map: ContextMap) -> dict[str, Any]:
    """Write a map as a JSON-compatible record.

    Args:
        context_map: The map to write.

    Returns:
        A record with one entry per field, in field order, that ``json.dumps`` accepts as is.
    """
    encoded = _encode(context_map)
    assert isinstance(encoded, dict)
    return encoded


def context_map_from_record(record: Mapping[str, Any]) -> ContextMap:
    """Rebuild a map from a record produced by :func:`context_map_to_record`.

    Args:
        record: The record; every field must be present and no other key may appear.

    Returns:
        The validated map.

    Raises:
        UnsupportedSchemaVersionError: If the record names a schema version this code cannot
            read; raised before any other field is looked at.
        ContextMapRecordError: If a field is missing, unknown, or has the wrong type.
        ValueError: If a value violates an invariant of the contract.
    """
    if not isinstance(record, Mapping):
        raise ContextMapRecordError(f"a record must be a mapping, got {type(record).__name__}")
    version = record.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedSchemaVersionError(
            f"a record needs a text schema version, got {version!r}"
        )
    require_supported_schema_version(version)
    decoded = _decode(ContextMap, record, "")
    assert isinstance(decoded, ContextMap)
    return decoded


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def _decode(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return _decode_optional(annotation, value, path)
    if origin is tuple:
        return _decode_tuple(annotation, value, path)
    if isinstance(annotation, NewType):
        return _decode(annotation.__supertype__, value, path)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _decode_enum(annotation, value, path)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _decode_dataclass(annotation, value, path)
    return _decode_scalar(annotation, value, path)


def _decode_optional(annotation: Any, value: Any, path: str) -> Any:
    members = [member for member in get_args(annotation) if member is not type(None)]
    if len(members) != 1:
        raise TypeError(f"{path or 'record'}: only 'X | None' unions are part of the schema")
    return None if value is None else _decode(members[0], value, path)


def _decode_tuple(annotation: Any, value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ContextMapRecordError(f"{path}: expected a list, got {type(value).__name__}")
    members = get_args(annotation)
    if len(members) == 2 and members[1] is Ellipsis:
        return tuple(
            _decode(members[0], item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    if len(members) != len(value):
        raise ContextMapRecordError(f"{path}: expected {len(members)} values, got {len(value)}")
    return tuple(
        _decode(member, item, f"{path}[{index}]")
        for index, (member, item) in enumerate(zip(members, value, strict=True))
    )


def _decode_enum(annotation: type[Enum], value: Any, path: str) -> Enum:
    try:
        return annotation(value)
    except ValueError:
        allowed = ", ".join(repr(member.value) for member in annotation)
        raise ContextMapRecordError(
            f"{path}: {value!r} is not a {annotation.__name__}; expected one of {allowed}"
        ) from None


def _decode_dataclass(annotation: type[Any], value: Any, path: str) -> Any:
    if not isinstance(value, Mapping):
        raise ContextMapRecordError(f"{path or 'record'}: expected a mapping")
    hints = get_type_hints(annotation)
    known = {field.name for field in fields(annotation)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ContextMapRecordError(f"{path or 'record'}: unknown field {unknown[0]!r}")
    arguments: dict[str, Any] = {}
    for field in fields(annotation):
        location = f"{path}.{field.name}" if path else field.name
        if field.name in value:
            arguments[field.name] = _decode(hints[field.name], value[field.name], location)
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ContextMapRecordError(f"{location}: field is missing")
    return annotation(**arguments)


def _decode_scalar(annotation: Any, value: Any, path: str) -> Any:
    if annotation is bool:
        accepted = isinstance(value, bool)
    elif annotation is int:
        accepted = isinstance(value, int) and not isinstance(value, bool)
    elif annotation is float:
        accepted = isinstance(value, int | float) and not isinstance(value, bool)
    elif annotation is str:
        accepted = isinstance(value, str)
    else:
        raise TypeError(f"{path or 'record'}: unsupported schema type {annotation!r}")
    if not accepted:
        raise ContextMapRecordError(
            f"{path}: expected {annotation.__name__}, got {type(value).__name__}"
        )
    return float(value) if annotation is float else value
