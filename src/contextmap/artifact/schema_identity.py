"""Structural identity of the ContextMap schema.

The schema version says what a change *means*; the fingerprint says what the structure *is*, so a
structural change can never go unnoticed. It covers the names, types and enum values of the
contract, following every type reachable from the root, including the types reused from other
capabilities (geometry references, bounds, timestamps): a change to any of them changes the
artifact schema too.

What the fingerprint cannot see is meaning. Two schemas with the same structure and different
invariants have the same fingerprint, so classifying a change is still a decision recorded in
``src/contextmap/artifact/docs/versioning.md``. The fingerprint guards against the silent kind
of drift: a structural change without a conscious version decision.

The description ignores declaration order and where a type is defined, so reordering fields or
moving a class to another module is not a schema change, while renaming a type is.
"""

from __future__ import annotations

import hashlib
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, NewType, Union, get_args, get_origin, get_type_hints

from contextmap.artifact.models import ContextMap


def describe_schema(root: type[Any] = ContextMap) -> str:
    """Describe the structure of a schema canonically.

    Args:
        root: The root type of the schema; the ContextMap by default.

    Returns:
        A text with one block per dataclass or enum reachable from ``root``: dataclasses list
        their fields with types, enums list their members. Blocks and lines are sorted, so the
        text is stable and diffable.
    """
    reachable: dict[tuple[str, str], type[Any]] = {}
    pending = [root]
    while pending:
        current = pending.pop()
        key = (current.__name__, current.__module__)
        if key in reachable:
            continue
        reachable[key] = current
        if is_dataclass(current):
            hints = get_type_hints(current)
            for field in fields(current):
                pending.extend(_referenced_types(hints[field.name]))

    blocks: list[str] = []
    for _, current in sorted(reachable.items()):
        if isinstance(current, type) and issubclass(current, Enum):
            lines = sorted(f"  {member.name} = {member.value!r}" for member in current)
            blocks.append("\n".join([f"enum {current.__name__}", *lines]))
        else:
            hints = get_type_hints(current)
            lines = sorted(
                f"  {field.name}: {_render(hints[field.name])}" for field in fields(current)
            )
            blocks.append("\n".join([f"dataclass {current.__name__}", *lines]))
    return "\n".join(blocks) + "\n"


def schema_fingerprint(root: type[Any] = ContextMap) -> str:
    """Hash the structure of a schema.

    Args:
        root: The root type of the schema; the ContextMap by default.

    Returns:
        ``"sha256:<hex>"`` of :func:`describe_schema`: equal for equal structure, different when
        a field, a type or an enum member is added, removed, renamed or retyped.
    """
    digest = hashlib.sha256(describe_schema(root).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _referenced_types(annotation: Any) -> list[type[Any]]:
    """List the dataclasses and enums an annotation refers to, through any container."""
    if isinstance(annotation, NewType):
        return _referenced_types(annotation.__supertype__)
    arguments = get_args(annotation)
    if arguments:
        return [
            referenced
            for argument in arguments
            if argument is not Ellipsis
            for referenced in _referenced_types(argument)
        ]
    if isinstance(annotation, type) and (is_dataclass(annotation) or issubclass(annotation, Enum)):
        return [annotation]
    return []


def _render(annotation: Any) -> str:
    """Write an annotation canonically, without the module of the types it names."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return " | ".join(sorted(_render(argument) for argument in get_args(annotation)))
    if origin is tuple:
        rendered = (
            "..." if argument is Ellipsis else _render(argument)
            for argument in get_args(annotation)
        )
        return f"tuple[{', '.join(rendered)}]"
    if isinstance(annotation, NewType):
        return f"{annotation.__name__}({_render(annotation.__supertype__)})"
    if annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    raise TypeError(f"unsupported schema annotation {annotation!r}")
