"""Validation helpers private to Spatial Relations.

The contracts of this capability require canonical ordering everywhere, so that two equivalent
records always encode to the same bytes and a persisted artifact is reproducible.
"""

from __future__ import annotations

import math
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


def require_finite(name: str, value: float) -> None:
    """Require a finite number.

    Raises:
        ValueError: If the value is NaN or infinite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


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
