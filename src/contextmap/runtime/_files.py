"""Atomic, never-overwriting publication of the small JSON documents a run persists."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_text(directory: Path, filename: str, text: str) -> Path:
    """Publish ``text`` as ``directory/filename`` atomically, without replacing anything.

    A reader sees the whole file or none of it, and an existing file is never touched:
    the content is written to a temporary file that is then hard-linked into place, which
    fails if the name is taken.

    Args:
        directory: Target directory; created when missing.
        filename: Name of the published file.
        text: UTF-8 content.

    Returns:
        Path of the published file.

    Raises:
        FileExistsError: If ``directory/filename`` already exists.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / filename
    descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        # link falha se o destino existir: publicação sem sobrescrever, atômica.
        os.link(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return final
