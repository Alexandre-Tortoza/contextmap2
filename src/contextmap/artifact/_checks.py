"""Validation helpers private to the ContextMap schema.

The schema requires canonical ordering everywhere, so that two equivalent maps always encode
to the same record and a persisted artifact is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import TypeVar

_Item = TypeVar("_Item")


def require_present(owner: object, *names: str) -> None:
    """Require that the named string attributes of ``owner`` are not blank.

    Args:
        owner: The object being validated.
        *names: Attribute names to check.

    Raises:
        ValueError: If an attribute is empty or only whitespace.
    """
    for name in names:
        if not str(getattr(owner, name)).strip():
            raise ValueError(f"{name} must not be empty")


def require_artifact_identity(owner: object, *names: str) -> None:
    """Require artifact identities that are present and are not filesystem paths.

    An artifact is identified by content and lineage, never by where it happens to be stored, so
    a value that looks like a path is refused instead of being interpreted.

    Args:
        owner: The object being validated.
        *names: Attribute names holding an artifact identity.

    Raises:
        ValueError: If an identity is blank or contains a path separator.
    """
    require_present(owner, *names)
    for name in names:
        value = str(getattr(owner, name))
        if "/" in value or "\\" in value:
            raise ValueError(f"{name} must be an identity, not a path: {value!r}")


def require_optional_present(owner: object, *names: str) -> None:
    """Require that the named attributes are either ``None`` or not blank.

    Args:
        owner: The object being validated.
        *names: Attribute names to check.

    Raises:
        ValueError: If an attribute is set but blank.
    """
    for name in names:
        if getattr(owner, name) is not None:
            require_present(owner, name)


def require_canonical(
    name: str, items: Sequence[_Item], key: Callable[[_Item], tuple[str, ...]], *, detail: str = ""
) -> None:
    """Require strictly increasing keys, so equivalent content always encodes identically.

    Args:
        name: Name of the collection, for the error message.
        items: The collection to check.
        key: Sort key of one item.
        detail: What the collection is sorted by, when not obvious.

    Raises:
        ValueError: If the keys are not strictly increasing, which also refuses duplicates.
    """
    keys = [key(item) for item in items]
    if any(left >= right for left, right in pairwise(keys)):
        raise ValueError(f"{name} must be sorted {detail}and unique")
