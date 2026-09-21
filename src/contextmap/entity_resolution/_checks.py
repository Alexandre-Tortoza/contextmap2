"""Validation helpers private to Entity Resolution.

The contracts of this capability require canonical ordering everywhere, so that two equivalent
records always encode to the same bytes and a persisted artifact is reproducible.
"""

from __future__ import annotations


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
