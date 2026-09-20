"""Shared validation for immutable Hugging Face model revisions."""

from __future__ import annotations

import re

_COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_huggingface_commit_revision(revision: str) -> None:
    """Require an immutable, full lowercase Git commit SHA."""
    if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a full 40-character lowercase hexadecimal commit SHA")
