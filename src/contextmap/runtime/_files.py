"""Atomic, durable, never-overwriting publication of the small JSON documents a run persists.

A document is forced to stable storage (``fsync``) before it gets its name, and its
directory right after, so after a power loss the name either does not exist or holds
the whole document. Syncing a directory opens it read-only, which is POSIX behavior
(Linux, the platform the project runs on).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_text(directory: Path, filename: str, text: str) -> Path:
    """Publish ``text`` as ``directory/filename`` atomically, without replacing anything.

    A reader sees the whole file or none of it, and an existing file is never touched:
    the content is written to a temporary file that is then hard-linked into place, which
    fails if the name is taken. Costs two ``fsync`` calls: the content before the link and
    the directory after it.

    Args:
        directory: Target directory; created when missing.
        filename: Name of the published file.
        text: UTF-8 content.

    Returns:
        Path of the published file.

    Raises:
        FileExistsError: If ``directory/filename`` already exists.
        OSError: If the content or the directory cannot be forced to stable storage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / filename
    descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    temporary = Path(temporary_name)
    try:
        _write_synced(descriptor, text)
        # link falha se o destino existir: publicação sem sobrescrever, atômica.
        os.link(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    _sync_directory(directory)
    return final


def replace_text(directory: Path, filename: str, text: str) -> Path:
    """Replace ``directory/filename`` atomically with ``text``.

    Only for repairing a document that is known to be invalid; a valid published
    document is never replaced (see :func:`publish_text`). Costs two ``fsync`` calls,
    like :func:`publish_text`.

    Args:
        directory: Target directory; created when missing.
        filename: Name of the file.
        text: UTF-8 content.

    Returns:
        Path of the file.

    Raises:
        OSError: If the content or the directory cannot be forced to stable storage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    temporary = Path(temporary_name)
    try:
        _write_synced(descriptor, text)
        final = directory / filename
        os.replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    _sync_directory(directory)
    return final


def _write_synced(descriptor: int, text: str) -> None:
    """Write ``text`` as UTF-8 through ``descriptor``, force it to stable storage and close it."""
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(directory: Path) -> None:
    """Force a directory's entries (the names linked, replaced or removed in it) to stable storage.

    Raises:
        OSError: If the directory cannot be opened or synced; opening a directory
            read-only to sync it is POSIX and fails on platforms that forbid it.
    """
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
