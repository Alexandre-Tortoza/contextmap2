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
import json
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any
from uuid import uuid4

_MANIFEST_FILENAME = "manifest.json"
_README_FILENAME = "README.md"
_REGISTRY_FILENAME = "runs.json"
_TEMPORARY_PREFIX = ".tmp-"


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
        data = file_path.read_bytes()
        if len(data) != entry.size_bytes:
            problems.append(
                f"size mismatch for {entry.path}: expected {entry.size_bytes}, found {len(data)}"
            )
            continue
        if file_entry(entry.path, data).content_hash != entry.content_hash:
            problems.append(f"content hash mismatch for {entry.path}")
    return problems


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

    def publish(self, *, manifest: Mapping[str, Any], readme: str) -> None:
        """Write the manifest, verify the inventory and make the run visible.

        The manifest and README are not part of their own inventory.

        Args:
            manifest: Manifest fields from the owning capability; the
                ``file_inventory`` is added here.
            readme: Human-readable summary of the run.

        Raises:
            RunDirectoryError: If the inventory does not match the files on
                disk, or the final path appeared in the meantime.
        """
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


def _run_directories(sequence_dir: Path) -> list[Path]:
    if not sequence_dir.is_dir():
        return []
    return sorted(
        entry
        for entry in sequence_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(_TEMPORARY_PREFIX)
    )


def next_run_index(sequence_dir: Path, *, index_of: Callable[[Path], int | None]) -> int:
    """Compute the next monotonic run index of a capability's sequence.

    Scans the run directories themselves, never the registry, so an
    interrupted, incomplete or corrupted run is not counted and allocation
    works when the registry is absent or stale.

    Args:
        sequence_dir: Directory holding the runs of one capability and sequence.
        index_of: Returns a directory's run index when it is a complete, valid
            run of the capability, and ``None`` otherwise.

    Returns:
        The next index, starting at ``1`` when there is no valid run.
    """
    indexes = [index for run_dir in _run_directories(sequence_dir) if (index := index_of(run_dir))]
    return max(indexes, default=0) + 1


def write_run_registry(
    sequence_dir: Path, *, describe: Callable[[Path], Mapping[str, Any] | None]
) -> None:
    """Rebuild the ``runs.json`` convenience registry from the valid runs on disk.

    The registry is never the source of truth: it can be deleted and
    regenerated at any time, and a directory that is not a valid run is
    silently left out.

    Args:
        sequence_dir: Directory holding the runs of one capability and sequence.
        describe: Returns the registry record of a valid run, or ``None``.
    """
    if not sequence_dir.is_dir():
        return
    runs = [record for run_dir in _run_directories(sequence_dir) if (record := describe(run_dir))]
    (sequence_dir / _REGISTRY_FILENAME).write_text(
        json.dumps({"runs": runs}, indent=2, sort_keys=True), encoding="utf-8"
    )
