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
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from contextmap.artifact.serialization.errors import UpstreamArtifactError
from contextmap.artifact.serialization.layout import MANIFEST
from contextmap.artifact.serialization.manifest import DependencyRecord, run_artifact_digest
from contextmap.shared import FileEntry, check_file_inventory

GEOMETRIC_MAP_ARTIFACT_TYPE = "geometric_map"
"""``artifact_type`` of the dependency that holds the authoritative geometry."""

ENTITY_RESOLUTION_RUN_ARTIFACT_TYPE = "entity_resolution_run"
"""``artifact_type`` of the dependency that resolves the map's entities."""

SPATIAL_RELATIONS_RUN_ARTIFACT_TYPE = "spatial_relations_run"
"""``artifact_type`` of the dependency that decides the map's relations."""


def _read_manifest_record(directory: Path) -> dict[str, Any]:
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise UpstreamArtifactError(f"no {MANIFEST} in the upstream artifact {directory.name!r}")
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise UpstreamArtifactError(
            f"the {MANIFEST} of the upstream artifact {directory.name!r} is not valid JSON "
            f"({error})"
        ) from error
    if not isinstance(record, dict):
        raise UpstreamArtifactError(
            f"the {MANIFEST} of the upstream artifact {directory.name!r} is not a JSON object"
        )
    return record


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
    record = _read_manifest_record(directory)
    try:
        entries = tuple(
            FileEntry(
                path=item["path"],
                size_bytes=item["size_bytes"],
                content_hash=item["content_hash"],
            )
            for item in record["file_inventory"]
        )
    except (KeyError, TypeError) as error:
        raise UpstreamArtifactError(
            f"the {MANIFEST} of the upstream artifact {directory.name!r} has no readable "
            f"file_inventory ({error})"
        ) from error
    return tuple(sorted(entries, key=lambda entry: entry.path))


def artifact_digest(directory: Path) -> str:
    """Compute the digest that pins an upstream artifact, from its own manifest.

    It is :func:`~contextmap.artifact.serialization.manifest.run_artifact_digest` applied to the
    identity, the schema version and the inventory that the artifact's manifest records: the same
    value the artifact's owner and its other consumers compute for it. A lineage entry of a map
    carries this value as the ``content_identity`` of the artifact it cites.

    Args:
        directory: A run artifact directory.

    Returns:
        ``"sha256:<hex digest>"``.

    Raises:
        UpstreamArtifactError: If the manifest is missing, is not JSON, or lacks an identity
            (``run_id`` or ``artifact_id``), a schema version or a readable inventory.
    """
    record = _read_manifest_record(directory)
    identity = record.get("run_id", record.get("artifact_id"))
    schema_version = record.get("schema_version")
    if not isinstance(identity, str) or not isinstance(schema_version, str):
        raise UpstreamArtifactError(
            f"the {MANIFEST} of the upstream artifact {directory.name!r} records no run_id (or "
            "artifact_id) and schema_version to pin it by"
        )
    return run_artifact_digest(identity, schema_version, read_inventory(directory))


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


class DependencyStatus(Enum):
    """Whether a recorded dependency was found, and whether it is the recorded one.

    Attributes:
        FOUND: A directory was found whose inventory digest is the recorded one.
        MISSING: No directory was found at the given path or at the relative hint.
        MISMATCH: A directory was found but it is not the recorded artifact: its inventory digest
            differs (stale or foreign) or its inventory is unreadable.
    """

    FOUND = "found"
    MISSING = "missing"
    MISMATCH = "mismatch"


@dataclass(frozen=True, kw_only=True)
class DependencyResolution:
    """The outcome of looking for one recorded dependency.

    Attributes:
        record: The dependency as the manifest recorded it.
        status: Whether it was found and matches.
        location: The directory that was examined; ``None`` when there was nowhere to look.
        detail: A sentence explaining the status, for people and reports.
    """

    record: DependencyRecord
    status: DependencyStatus
    location: Path | None
    detail: str


def resolve_dependency(
    record: DependencyRecord, *, artifact_root: Path, dependency_paths: Mapping[str, Path]
) -> DependencyResolution:
    """Look for a recorded dependency and check that it is exactly the recorded artifact.

    An explicit path for the artifact id is used alone: it never falls back to the relative hint,
    because a caller who says where the dependency is must not be answered with another place.
    Without one, the recorded hint is resolved against the artifact directory. Whatever is found
    counts only if the digest of its inventory equals the recorded ``content_identity``; nothing
    is searched for, and no directory is trusted for its name.

    Args:
        record: The dependency the manifest recorded.
        artifact_root: The directory of the artifact that refers to it.
        dependency_paths: Explicit locations, keyed by ``artifact_id``.

    Returns:
        The resolution. Nothing is raised for a missing or stale dependency; the caller decides
        what that means for a required or an optional one.
    """
    if record.artifact_id in dependency_paths:
        candidate: Path | None = dependency_paths[record.artifact_id]
    elif record.locator is not None:
        candidate = (artifact_root / record.locator).resolve()
    else:
        candidate = None
    what = f"{record.artifact_type} {record.artifact_id!r}"
    if candidate is None:
        return DependencyResolution(
            record=record,
            status=DependencyStatus.MISSING,
            location=None,
            detail=f"the {what} has no recorded hint; pass its location explicitly",
        )
    if not candidate.is_dir():
        return DependencyResolution(
            record=record,
            status=DependencyStatus.MISSING,
            location=candidate,
            detail=f"the {what} is not at {candidate.name!r}: no such directory",
        )
    if not (candidate / MANIFEST).is_file():
        return DependencyResolution(
            record=record,
            status=DependencyStatus.MISSING,
            location=candidate,
            detail=f"the {what} is not at {candidate.name!r}: it holds no {MANIFEST}",
        )
    try:
        found = artifact_digest(candidate)
    except UpstreamArtifactError as error:
        return DependencyResolution(
            record=record,
            status=DependencyStatus.MISMATCH,
            location=candidate,
            detail=str(error),
        )
    if found != record.content_identity:
        return DependencyResolution(
            record=record,
            status=DependencyStatus.MISMATCH,
            location=candidate,
            detail=(
                f"the directory {candidate.name!r} is not the {what}: its inventory digest "
                f"{found} differs from the recorded {record.content_identity}"
            ),
        )
    return DependencyResolution(
        record=record,
        status=DependencyStatus.FOUND,
        location=candidate,
        detail=f"the {what} was found and matches",
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
