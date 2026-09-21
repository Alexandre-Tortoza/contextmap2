"""Shared resolution of canonical semantic view payloads for semantic backends."""

from __future__ import annotations

from pathlib import Path


def resolve_view_payload(root: Path, reference: str) -> Path:
    """Resolve a semantic view payload reference below ``root`` without allowing escape.

    Args:
        root: Directory that contains the run/artifact-relative ``outputs/semantic-views/``
            tree named by ``SemanticVisualView.payload_reference``.
        reference: Run-relative payload reference of one canonical visual view.

    Returns:
        The resolved path of an existing regular file inside ``root``.

    Raises:
        ValueError: If the reference resolves outside ``root``.
        FileNotFoundError: If the resolved payload is not an existing file.
    """
    resolved_root = root.resolve()
    candidate = (resolved_root / reference).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"semantic view reference escapes its root: {reference!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"semantic view payload does not exist: {reference!r}")
    return candidate
