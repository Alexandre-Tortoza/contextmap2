"""Atomic, self-describing run directories shared by every run artifact.

``docs/ARTIFACTS.md`` fixes the rules every run artifact follows regardless of
its payload: an interrupted write must never look like a finished run, a
finished run is immutable and never overwritten, its manifest inventories
every contractual file with size and hash so corruption is detected, and run
indexes are monotonic per capability and sequence and computed from valid runs
on disk (a registry file is only a convenience). This module implements those
rules once. It knows nothing about what a run contains; the owning capability
decides the files, the manifest fields and what makes a run valid.
"""

from __future__ import annotations

import hashlib
import io
import json
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
        Human-readable problems (missing file, size mismatch, hash
        mismatch); empty means the inventory matches.
    """
    problems: list[str] = []
    for entry in inventory:
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
    half written.
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
        """
        _require_relative_path(relative_path)
        if relative_path in self._written:
            raise RunDirectoryError(f"path already written in this run: {relative_path}")
        target = self._tmp_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self._written.add(relative_path)
        if contractual:
            self._entries[relative_path] = file_entry(relative_path, data)

    @contextmanager
    def open_binary(self, relative_path: str, *, contractual: bool = True) -> Iterator[BinaryIO]:
        """Stream a file that may be larger than memory.

        The bytes are hashed and counted as they are written, so the inventory
        entry costs no second read. Use it like :meth:`write_bytes` for payloads
        that are produced incrementally.

        Args:
            relative_path: Path relative to the run directory.
            contractual: See :meth:`write_bytes`.

        Yields:
            A writable binary stream. It must be closed (leave the ``with``
            block) before :meth:`publish`.

        Raises:
            RunDirectoryError: If the path is not a plain relative path inside
                the run, or the same path was already written.
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

    def publish(self, *, manifest: Mapping[str, Any], readme: str) -> None:
        """Write the manifest, verify the inventory and make the run visible.

        The manifest and README are not part of their own inventory.

        Args:
            manifest: Manifest fields from the owning capability; the
                ``file_inventory`` is added here.
            readme: Human-readable summary of the run.

        Raises:
            RunDirectoryError: If a stream is open or failed, the inventory does
                not match the files on disk, or the final path appeared in the
                meantime.
        """
        if self._open_streams:
            raise RunDirectoryError("a stream is still open; close it before publishing the run")
        if self._failed_stream:
            raise RunDirectoryError("a stream failed while writing; the run cannot be published")
        inventory = sorted(self._entries.values(), key=lambda entry: entry.path)
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
        (self._tmp_dir / _MANIFEST_FILENAME).write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self._tmp_dir / _README_FILENAME).write_text(readme, encoding="utf-8")

        problems = check_file_inventory(self._tmp_dir, inventory)
        if problems:
            raise RunDirectoryError(f"internal consistency check failed before publish: {problems}")
        if self._final_dir.exists():
            raise RunDirectoryError(f"run directory already exists: {self._final_dir}")
        self._tmp_dir.rename(self._final_dir)
        self._published = True


def _require_relative_path(relative_path: str) -> None:
    path = PurePosixPath(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts or path.parts == (".",):
        raise RunDirectoryError(
            f"expected a relative path inside the run directory, got {relative_path!r}"
        )
