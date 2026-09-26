"""The handle the runtime holds to one immutable stage artifact."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True, kw_only=True)
class ArtifactRef:
    """A handle to one immutable stage artifact.

    Attributes:
        stage_id: Stage that produced it.
        contract: Artifact kind, for example ``"SequenceArtifact"``.
        artifact_id: Identity of the exact artifact or run, never a directory name.
        content_hash: Hash of the artifact's contractual content, for example the digest of
            its manifest inventory. It is what makes reuse safe: two artifacts with the same
            content hash are interchangeable, whatever their names or run identities. An
            artifact without one can be consumed but never reused or indexed.
        location: Directory of the artifact, relative to the workspace and written with ``/``
            (``<dataset>/<run>/<stage>``). It is what makes a reused artifact *referenced* and
            still openable: the runtime never copies an artifact, so a later run reads the
            directory of the run that wrote it. ``None`` for a handle that names no directory.
    """

    stage_id: str
    contract: str
    artifact_id: str
    content_hash: str | None = None
    location: str | None = None

    def to_document(self) -> dict[str, str | None]:
        """Return the JSON-compatible form persisted in execution records and the index."""
        document: dict[str, str | None] = {
            "stage_id": self.stage_id,
            "contract": self.contract,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
        }
        if self.location is not None:
            document["location"] = self.location
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ArtifactRef:
        """Rebuild a handle from its persisted form.

        Args:
            document: A mapping produced by :meth:`to_document`.

        Returns:
            The handle.

        Raises:
            ValueError: If a field is missing or is not text.
        """
        try:
            values = {name: document[name] for name in ("stage_id", "contract", "artifact_id")}
            content_hash = document.get("content_hash")
            location = document.get("location")
        except KeyError as error:
            raise ValueError(f"an artifact reference lacks {error}") from error
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("an artifact reference needs text stage_id, contract and artifact_id")
        if content_hash is not None and not isinstance(content_hash, str):
            raise ValueError("an artifact reference's content_hash must be text or null")
        if location is not None and not isinstance(location, str):
            raise ValueError("an artifact reference's location must be text when present")
        return cls(content_hash=content_hash, location=location, **values)


def artifact_directory(workspace: Path, ref: ArtifactRef) -> Path:
    """Return the directory a handle points to, inside the workspace.

    Args:
        workspace: The workspace root.
        ref: The handle of an artifact.

    Returns:
        ``<workspace>/<ref.location>``.

    Raises:
        ValueError: If the handle names no location, or the location is empty, absolute or
            climbs out of the workspace (it is refused, never sanitized).
    """
    if ref.location is None:
        raise ValueError(f"artifact {ref.artifact_id!r} has no location: it cannot be opened")
    parts = PurePosixPath(ref.location).parts
    if not ref.location or PurePosixPath(ref.location).is_absolute() or ".." in parts:
        raise ValueError(
            f"artifact {ref.artifact_id!r} has a location {ref.location!r} that is not a path "
            "inside the workspace"
        )
    return workspace.joinpath(*parts)


def inventory_digest(inventory: Sequence[object]) -> str:
    """Return the content hash of an artifact: the digest of its contractual file inventory.

    Args:
        inventory: The ``file_inventory`` of a manifest; each entry has ``path`` and
            ``content_hash``.

    Returns:
        ``sha256:`` of the canonical JSON of ``{path: content_hash}``. Two artifacts with the same
        contractual files have the same digest, whatever their names, ids or timestamps.
    """
    files = {entry.path: entry.content_hash for entry in inventory}  # type: ignore[attr-defined]
    text = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
