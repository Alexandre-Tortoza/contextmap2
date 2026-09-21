"""The handle the runtime holds to one immutable stage artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    """

    stage_id: str
    contract: str
    artifact_id: str
    content_hash: str | None = None

    def to_document(self) -> dict[str, str | None]:
        """Return the JSON-compatible form persisted in execution records and the index."""
        return {
            "stage_id": self.stage_id,
            "contract": self.contract,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
        }

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
        except KeyError as error:
            raise ValueError(f"an artifact reference lacks {error}") from error
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("an artifact reference needs text stage_id, contract and artifact_id")
        if content_hash is not None and not isinstance(content_hash, str):
            raise ValueError("an artifact reference's content_hash must be text or null")
        return cls(content_hash=content_hash, **values)
