"""Reflective JSON codec for the frozen dataclasses of Entity Resolution.

Every contract of this capability is a frozen dataclass whose fields are primitives, enums,
``NewType`` identities, tuples or other dataclasses. Instead of one hand-written encoder per
contract, this module encodes and decodes them from their type annotations, so a new contract
cannot drift from its codec and every record is decoded *strictly*: a missing field, an unknown
field, a value of the wrong type or an unknown enum member is refused with the path of the
offending value, and the object is rebuilt through its constructor, so every invariant of the
contract is revalidated instead of trusted.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

_T = TypeVar("_T")


def to_record(value: object) -> Any:
    """Encode a contract into JSON primitives.

    Args:
        value: A frozen dataclass of this capability, or one of its field values.

    Returns:
        ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` or ``None``; tuples become lists
        and enums their value.

    Raises:
        TypeError: If a value has no JSON encoding.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [to_record(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_record(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    raise TypeError(f"cannot encode a {type(value).__name__}")


def from_record(cls: type[_T], record: object, *, path: str | None = None) -> _T:
    """Decode a contract from the output of :func:`to_record`, revalidating it.

    Args:
        cls: The frozen dataclass to rebuild.
        record: The decoded JSON object.
        path: Where the record sits in the document, for error messages; defaults to the class.

    Returns:
        The contract, built through its constructor.

    Raises:
        ValueError: If the record is not an object, a field is missing or unknown, a value has the
            wrong type, or the constructor refuses the values.
    """
    where = path or cls.__name__
    if not isinstance(record, Mapping):
        raise ValueError(f"{where}: expected an object, got {type(record).__name__}")
    fields = {field.name: field for field in dataclasses.fields(cls) if field.init}  # type: ignore[arg-type]
    unknown = sorted(set(record) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown!r}")
    hints = _hints(cls)
    values: dict[str, Any] = {}
    for name, field in fields.items():
        if name in record:
            values[name] = _decode(hints[name], record[name], f"{where}.{name}")
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ValueError(f"{where}: missing the field {name!r}")
    try:
        return cls(**values)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    if cls not in _HINTS:
        _HINTS[cls] = get_type_hints(cls)
    return _HINTS[cls]


def _decode(annotation: Any, value: Any, path: str) -> Any:
    supertype = getattr(annotation, "__supertype__", None)
    if supertype is not None:
        return annotation(_decode(supertype, value, path))
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return _decode_optional(annotation, value, path)
    if origin is tuple:
        return _decode_tuple(annotation, value, path)
    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return _decode_enum(annotation, value, path)
        if dataclasses.is_dataclass(annotation):
            return from_record(annotation, value, path=path)
        return _decode_primitive(annotation, value, path)
    raise TypeError(f"{path}: unsupported annotation {annotation!r}")


def _decode_optional(annotation: Any, value: Any, path: str) -> Any:
    members = get_args(annotation)
    present = [member for member in members if member is not type(None)]
    if value is None:
        if len(present) == len(members):
            raise ValueError(f"{path}: must not be null")
        return None
    if len(present) != 1:
        raise TypeError(f"{path}: unsupported union {annotation!r}")
    return _decode(present[0], value, path)


def _decode_tuple(annotation: Any, value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a list, got {type(value).__name__}")
    members = get_args(annotation)
    if len(members) == 2 and members[1] is Ellipsis:
        return tuple(
            _decode(members[0], item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    if len(members) != len(value):
        raise ValueError(f"{path}: expected {len(members)} items, got {len(value)}")
    return tuple(
        _decode(member, item, f"{path}[{index}]")
        for index, (member, item) in enumerate(zip(members, value, strict=True))
    )


def _decode_enum(annotation: type[Enum], value: Any, path: str) -> Enum:
    try:
        return annotation(value)
    except ValueError:
        raise ValueError(f"{path}: {value!r} is not a valid {annotation.__name__}") from None


def _decode_primitive(annotation: type, value: Any, path: str) -> Any:
    if annotation is bool:
        ok = isinstance(value, bool)
    elif annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif annotation is float:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    elif annotation is str:
        ok = isinstance(value, str)
    else:
        raise TypeError(f"{path}: unsupported type {annotation!r}")
    if not ok:
        raise ValueError(f"{path}: expected {annotation.__name__}, got {type(value).__name__}")
    return float(value) if annotation is float else value
