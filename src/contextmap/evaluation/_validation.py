"""Small field validators shared by evaluation manifests and annotation schemas."""

from __future__ import annotations

import re
from collections.abc import Iterable

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def require_text(name: str, value: str) -> None:
    """Reject empty or blank text."""
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def require_sha256(name: str, value: str) -> None:
    """Reject anything that is not ``sha256:<64 hex digits>``."""
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be 'sha256:<64 hex digits>', got {value!r}")


def require_unique(name: str, values: Iterable[str]) -> None:
    """Reject a repeated value, naming it."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{name} must be unique; {value!r} is repeated")
        seen.add(value)
