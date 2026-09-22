"""Shared reading and integrity check of canonical semantic view payloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

from contextmap.visual_perception.semantic_requests import SemanticVisualView


def read_view_payload(root: Path, view: SemanticVisualView) -> bytes:
    """Read the exact bytes of ``view`` and verify them against ``view.sha256``.

    The request identifies the evidence it supplies by ``SemanticVisualView.sha256``. Every
    backend must therefore decode or transmit only the bytes returned here: the payload is
    read once, hashed, and handed back, so the verified bytes are the consumed bytes and the
    file cannot change between the check and the inference.

    Args:
        root: Directory that contains the run/artifact-relative ``outputs/semantic-views/``
            tree named by ``SemanticVisualView.payload_reference``.
        view: Canonical visual view whose payload reference and SHA-256 identify the bytes.

    Returns:
        The payload bytes, whose SHA-256 equals ``view.sha256``.

    Raises:
        ValueError: If the reference resolves outside ``root`` or the payload bytes do not
            hash to ``view.sha256``.
        FileNotFoundError: If the resolved payload is not an existing file.
    """
    resolved_root = root.resolve()
    candidate = (resolved_root / view.payload_reference).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"semantic view reference escapes its root: {view.payload_reference!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"semantic view payload does not exist: {view.payload_reference!r}")
    payload = candidate.read_bytes()
    found = hashlib.sha256(payload).hexdigest()
    if found != view.sha256:
        raise ValueError(
            f"semantic view payload {view.payload_reference!r} does not match "
            f"SemanticVisualView.sha256 of view {view.view_id!r}: "
            f"expected {view.sha256}, found {found}"
        )
    return payload
