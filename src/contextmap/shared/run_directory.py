"""Atomic, self-describing run directories shared by every run artifact.

``docs/ARTIFACTS.md`` fixes the rules every run artifact follows regardless of
its payload: an interrupted write must never look like a finished run, a
finished run is immutable and never overwritten, its manifest inventories
every contractual file with size and hash so corruption is detected, and run
indexes are monotonic per capability and sequence and computed from valid runs
on disk (a registry file is only a convenience). This module implements those
rules once. It knows nothing about what a run contains; the owning capability
decides the files, the manifest fields and what makes a run valid.

Publication is durable as well as atomic: every file and directory of a run is
forced to stable storage (``fsync``) before the rename that makes it visible,
and the parent directory right after it. The rename alone only protects against
a process that dies; without the syncs a power loss could persist the rename
before the data and leave a published run with empty or truncated files.
Syncing a directory opens it read-only, which is POSIX behavior (Linux, the
platform the project runs on).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, BinaryIO
from uuid import uuid4

_MANIFEST_FILENAME = "manifest.json"
_README_FILENAME = "README.md"
_TEMPORARY_PREFIX = ".tmp-"

# Arquivos grandes (a geometria de um mapa pode ter gigabytes) nunca são lidos por inteiro.
_HASH_CHUNK_BYTES = 1 << 20


class RunDirectoryError(Exception):
    """Raised when a run directory cannot be written or published safely."""


@dataclass(frozen=True, kw_only=True)
class FileEntry:
    """One file in a run's inventory.

    Attributes:
        path: Path relative to the run directory, with ``/`` separators.
        size_bytes: Size of the file in bytes.
        content_hash: ``"sha256:<hex digest>"`` of the content.
    """

    path: str
    size_bytes: int
    content_hash: str


def file_entry(relative_path: str, data: bytes) -> FileEntry:
    """Build the inventory entry of a file's content.

    Args:
        relative_path: Path relative to the run directory.
        data: The file content.

    Returns:
        The entry with the content's size and SHA-256.
    """
    return FileEntry(
        path=relative_path,
        size_bytes=len(data),
        content_hash=f"sha256:{hashlib.sha256(data).hexdigest()}",
    )


def check_file_inventory(root: Path, inventory: Iterable[FileEntry]) -> list[str]:
    """Compare an inventory with what is actually on disk.

    Files are hashed in chunks, so a file larger than memory can be checked.

    Args:
        root: The run directory.
        inventory: The entries the manifest promises.

    Returns:
        Human-readable problems (a path that is not inside the run, missing
        file, size mismatch, hash mismatch); empty means the inventory matches.
        A path outside the run is reported and never opened.
    """
    problems: list[str] = []
    for entry in inventory:
        # Um manifest adulterado nunca faz o leitor abrir um arquivo fora do run.
        if not is_run_relative_path(entry.path):
            problems.append(f"invalid path in manifest: {entry.path!r}")
            continue
        file_path = root / entry.path
        if not file_path.is_file():
            problems.append(f"missing file referenced by manifest: {entry.path}")
            continue
        size_bytes, content_hash = _hash_file(file_path)
        if size_bytes != entry.size_bytes:
            problems.append(
                f"size mismatch for {entry.path}: expected {entry.size_bytes}, found {size_bytes}"
            )
            continue
        if content_hash != entry.content_hash:
            problems.append(f"content hash mismatch for {entry.path}")
    return problems


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, f"sha256:{digest.hexdigest()}"


class _HashingRawWriter(io.RawIOBase):
    """Raw writer that hashes and counts exactly the bytes it hands to the file."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        view = memoryview(data)
        written = self._handle.write(view) or 0
        self.digest.update(view[:written])
        self.size_bytes += written
        return written


class AtomicRunDirectory:
    """Builds a run directory in a temporary sibling and publishes it atomically.

    Use it as a context manager: files are written under a hidden temporary
    directory, and only :meth:`publish` renames it to its final path, after
    verifying the inventory. Leaving the block without publishing, or with an
    exception, removes the temporary directory, so the final path never exists
    half written. Every file and directory reaches stable storage before that
    rename, so this also holds after a power loss (see :meth:`publish`).
    """

    def __init__(self, final_dir: Path) -> None:
        """Prepare a run directory.

        Args:
            final_dir: Where the finished run will live.

        Raises:
            RunDirectoryError: If ``final_dir`` already exists; a finished run
                is immutable, and rerunning creates a new run instead.
        """
        if final_dir.exists():
            raise RunDirectoryError(f"run directory already exists: {final_dir}")
        self._final_dir = final_dir
        self._tmp_dir = final_dir.parent / f"{_TEMPORARY_PREFIX}{final_dir.name}-{uuid4().hex[:8]}"
        self._entries: dict[str, FileEntry] = {}
        self._written: set[str] = set()
        self._open_streams = 0
        self._failed_stream = False
        self._published = False

    def __enter__(self) -> AtomicRunDirectory:
        """Create the temporary directory."""
        self._tmp_dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Discard the temporary directory unless the run was published."""
        if not self._published:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def write_text(self, relative_path: str, text: str, *, contractual: bool = True) -> None:
        """Write a UTF-8 text file; see :meth:`write_bytes`."""
        self.write_bytes(relative_path, text.encode("utf-8"), contractual=contractual)

    def write_bytes(self, relative_path: str, data: bytes, *, contractual: bool = True) -> None:
        """Write a file, recording it in the inventory when it is contractual.

        The file is forced to stable storage (``fsync``) before the call returns.

        Args:
            relative_path: Path relative to the run directory.
            data: The file content.
            contractual: ``True`` for data a downstream consumer may depend on:
                it enters the inventory, so its loss is detected. ``False`` for
                human-only evidence such as ``debug/`` files: it is written but
                never inventoried, so removing it cannot invalidate the run.

        Raises:
            RunDirectoryError: If the path is not a plain relative path inside
                the run, or the same path was already written.
            OSError: If the file cannot be written or forced to stable storage.
        """
        _require_relative_path(relative_path)
        if relative_path in self._written:
            raise RunDirectoryError(f"path already written in this run: {relative_path}")
        target = self._tmp_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_synced(target, data)
        self._written.add(relative_path)
        if contractual:
            self._entries[relative_path] = file_entry(relative_path, data)

    @contextmanager
    def open_binary(self, relative_path: str, *, contractual: bool = True) -> Iterator[BinaryIO]:
        """Stream a file that may be larger than memory.

        The bytes are hashed and counted as they are written, so the inventory
        entry costs no second read. Use it like :meth:`write_bytes` for payloads
        that are produced incrementally. The file is forced to stable storage
        (``fsync``) when the stream is closed.

        Args:
            relative_path: Path relative to the run directory.
            contractual: See :meth:`write_bytes`.

        Yields:
            A writable binary stream. It must be closed (leave the ``with``
            block) before :meth:`publish`.

        Raises:
            RunDirectoryError: If the path is not a plain relative path inside
                the run, or the same path was already written.
            OSError: If the file cannot be written or forced to stable storage;
                the run can then no longer be published.
        """
        _require_relative_path(relative_path)
        if relative_path in self._written:
            raise RunDirectoryError(f"path already written in this run: {relative_path}")
        target = self._tmp_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._written.add(relative_path)
        self._open_streams += 1
        try:
            with target.open("wb") as handle:
                raw = _HashingRawWriter(handle)
                stream = io.BufferedWriter(raw, buffer_size=_HASH_CHUNK_BYTES)
                try:
                    yield stream
                finally:
                    stream.close()
                _sync_file(handle)
        except BaseException:
            # Um arquivo parcial nunca pode ser publicado nem parecer inventariado.
            self._failed_stream = True
            raise
        finally:
            self._open_streams -= 1
        if contractual:
            self._entries[relative_path] = FileEntry(
                path=relative_path,
                size_bytes=raw.size_bytes,
                content_hash=f"sha256:{raw.digest.hexdigest()}",
            )

    def written_path(self, relative_path: str) -> Path:
        """Return where a file written in this run currently lives.

        For reading back what was written before the run is published, for
        example to open a payload and derive human-only evidence from it. Do not
        write through it: files are only added with :meth:`write_bytes` and
        :meth:`open_binary`, which keep the inventory honest.

        Args:
            relative_path: Path relative to the run directory.

        Raises:
            RunDirectoryError: If no such file was written in this run.
        """
        if relative_path not in self._written:
            raise RunDirectoryError(f"path has not been written in this run: {relative_path}")
        return self._tmp_dir / relative_path

    def inventory(self) -> tuple[FileEntry, ...]:
        """Return the inventory recorded so far, sorted by path.

        It is exactly what :meth:`publish` adds to the manifest: every contractual file written
        with :meth:`write_bytes`, and every contractual stream of :meth:`open_binary` once it is
        closed, with the size and hash counted while it was written. An owner whose manifest
        depends on the inventory (for example a content identity computed over the file hashes)
        builds it from here without reading any file again.

        Returns:
            The entries recorded when this call is made; a stream still open is not among them.
        """
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.path))

    def publish(self, *, manifest: Mapping[str, Any], readme: str) -> None:
        """Write the manifest, verify the inventory and make the run visible.

        The manifest and README are not part of their own inventory.

        Publication is durable: the payloads were forced to stable storage when
        written, the manifest and README are forced here, then every directory of
        the run, and only then the run is renamed to its final path, after which
        the parent directory is forced too. After a power loss the final path
        either does not exist or holds the complete run.

        Cost: one ``fsync`` per file of the run (paid by :meth:`write_bytes` and
        :meth:`open_binary` as each file is written), one per directory of the
        run and one for the parent. Each waits for the storage device, so the
        cost grows with the number of files and with the data not yet flushed,
        not with anything the owning capability computes.

        Args:
            manifest: Manifest fields from the owning capability; the
                ``file_inventory`` is added here.
            readme: Human-readable summary of the run.

        Raises:
            RunDirectoryError: If a stream is open or failed, the inventory does
                not match the files on disk, or the final path appeared in the
                meantime.
            OSError: If a file or directory cannot be forced to stable storage.
                Before the rename the run is not published; when the parent
                directory fails after it, the run is visible but its durability
                is not confirmed.
        """
        if self._open_streams:
            raise RunDirectoryError("a stream is still open; close it before publishing the run")
        if self._failed_stream:
            raise RunDirectoryError("a stream failed while writing; the run cannot be published")
        inventory = self.inventory()
        record = {
            **manifest,
            "file_inventory": [
                {
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "content_hash": entry.content_hash,
                }
                for entry in inventory
            ],
        }
        _write_synced(
            self._tmp_dir / _MANIFEST_FILENAME,
            json.dumps(record, indent=2, sort_keys=True).encode("utf-8"),
        )
        _write_synced(self._tmp_dir / _README_FILENAME, readme.encode("utf-8"))

        problems = check_file_inventory(self._tmp_dir, inventory)
        if problems:
            raise RunDirectoryError(f"internal consistency check failed before publish: {problems}")
        if self._final_dir.exists():
            raise RunDirectoryError(f"run directory already exists: {self._final_dir}")
        # Os dados já estão no disco; os nomes de cada diretório do run também precisam
        # estar antes do rename, senão ele pode sobreviver a uma queda sem os arquivos.
        for directory in self._run_directories():
            _sync_directory(directory)
        self._tmp_dir.rename(self._final_dir)
        self._published = True
        _sync_directory(self._final_dir.parent)

    def _run_directories(self) -> list[Path]:
        """Return every directory of the run: its root and each parent of a written file."""
        parents = {parent for path in self._written for parent in PurePosixPath(path).parents}
        return sorted({self._tmp_dir, *(self._tmp_dir / parent for parent in parents)})


def is_run_relative_path(relative_path: str) -> bool:
    """Tell whether a path is a plain relative path that stays inside a run directory.

    Args:
        relative_path: Path with ``/`` separators, as a manifest or an index records it.

    Returns:
        ``False`` for an empty path, an absolute path, a path with a ``..`` part or ``.``
        alone: joining any of them to the run directory could leave it or name it.
    """
    path = PurePosixPath(relative_path)
    return bool(relative_path) and not (
        path.is_absolute() or ".." in path.parts or path.parts == (".",)
    )


def _write_synced(path: Path, data: bytes) -> None:
    """Write ``data`` to a new file and force it to stable storage before returning."""
    with path.open("wb") as handle:
        handle.write(data)
        _sync_file(handle)


def _sync_file(handle: BinaryIO) -> None:
    """Flush a file's buffers and force its content to stable storage."""
    handle.flush()
    os.fsync(handle.fileno())


def _sync_directory(directory: Path) -> None:
    """Force a directory's entries (the names created or renamed in it) to stable storage.

    Raises:
        OSError: If the directory cannot be opened or synced; opening a directory
            read-only to sync it is POSIX and fails on platforms that forbid it.
    """
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_relative_path(relative_path: str) -> None:
    if not is_run_relative_path(relative_path):
        raise RunDirectoryError(
            f"expected a relative path inside the run directory, got {relative_path!r}"
        )
