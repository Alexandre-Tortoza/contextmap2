"""Upstream artifacts a ContextMapArtifact refers to instead of copying.

An upstream artifact is pinned by the digest of its own contractual inventory, so any directory
that holds exactly those files *is* the dependency, wherever it lives. The manifest also keeps a
relative ``locator`` as a hint where to look; a hint is never trusted, and it is the only thing
that changes when an artifact travels.

This module reads the ``manifest.json`` and the ``file_inventory`` every run artifact of the
project publishes (see :mod:`contextmap.shared.run_directory`), so it works for any upstream
capability without importing it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from contextmap.artifact.errors import UpstreamArtifactError
from contextmap.artifact.layout import MANIFEST
from contextmap.artifact.manifest import Requirement
from contextmap.shared import FileEntry, check_file_inventory

GEOMETRIC_MAP_ARTIFACT_TYPE = "geometric_map"
"""``artifact_type`` of the dependency that holds the authoritative geometry."""


@dataclass(frozen=True, kw_only=True)
class UpstreamArtifact:
    """An upstream artifact, on disk, that a map is written against.

    Attributes:
        artifact_type: Kind of the artifact, for example ``"semantic_fusion_run"``.
        artifact_id: Identity of the artifact inside its own capability.
        location: Where it is now. The location is used to read and verify it and to compute a
            relative hint; it is never recorded as an identity.
        requirement: Whether the map needs it to be resolved or only to inspect evidence.
    """

    artifact_type: str
    artifact_id: str
    location: Path
    requirement: Requirement


def read_inventory(directory: Path) -> tuple[FileEntry, ...]:
    """Read the file inventory that an artifact directory publishes in its manifest.

    Args:
        directory: A run artifact directory.

    Returns:
        The inventory entries, sorted by path.

    Raises:
        UpstreamArtifactError: If there is no manifest, it is not JSON, or its inventory is
            missing or malformed.
    """
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise UpstreamArtifactError(f"no {MANIFEST} in the upstream artifact {directory.name!r}")
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = record["file_inventory"]
        entries = tuple(
            FileEntry(
                path=item["path"],
                size_bytes=item["size_bytes"],
                content_hash=item["content_hash"],
            )
            for item in listed
        )
    except (ValueError, KeyError, TypeError) as error:
        raise UpstreamArtifactError(
            f"the {MANIFEST} of the upstream artifact {directory.name!r} has no readable "
            f"file_inventory ({error})"
        ) from error
    return tuple(sorted(entries, key=lambda entry: entry.path))


def verify_inventory(directory: Path, inventory: tuple[FileEntry, ...]) -> None:
    """Check every inventoried file of an upstream artifact against the disk.

    Args:
        directory: The artifact directory.
        inventory: What its manifest promises.

    Raises:
        UpstreamArtifactError: Naming every missing, resized or altered file.
    """
    problems = check_file_inventory(directory, inventory)
    if problems:
        raise UpstreamArtifactError(
            f"the upstream artifact {directory.name!r} does not match its inventory: "
            + "; ".join(problems)
        )


def relative_locator(from_directory: Path, to_directory: Path) -> str | None:
    """Compute the relative POSIX hint that leads from one directory to another.

    Args:
        from_directory: The artifact that refers to the other one (it may not exist yet).
        to_directory: The upstream artifact.

    Returns:
        The path such as ``../geometric_mapping/run-0001``, or ``None`` when there is no relative
        path between them (for example different drives), in which case no hint is recorded.
    """
    try:
        relative = os.path.relpath(to_directory.resolve(), from_directory.resolve())
    except ValueError:
        return None
    return PurePosixPath(*Path(relative).parts).as_posix()
