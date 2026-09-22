"""Stage reuse: identity keys, the index of completed artifacts and reuse decisions.

A stage artifact is immutable, so its result can be reused whenever everything the result
depends on is identical. This module defines that identity and where completed artifacts
are looked up; the runner (:mod:`contextmap.runtime.pipeline`) applies it stage by stage,
in dependency order, so changing one stage invalidates exactly that stage and the stages
that truly depend on it.

Reuse is never keyed on a directory or a human-readable run name. A key is made of the
identity of the stage's own configuration (contract, backend and parameters), the content
hash of every input artifact, the code identity and any extra identities the caller
declares (calibration, map, selection). There is no global mutable cache service: the
default index is a directory of small immutable JSON entries, one per identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from contextmap.runtime._files import publish_text, replace_text
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.errors import ReuseError

REUSE_SCHEMA_VERSION = "0.1.0"
"""Version of the index entry and reuse key documents."""


@dataclass(frozen=True, kw_only=True)
class ReuseKey:
    """Everything the result of one stage depends on.

    Attributes:
        stage_id: The stage.
        contract: Artifact kind the stage produces.
        stage_config_digest: Identity of the stage's own configuration (its declared
            contract and the backend and parameters of each variation point).
        inputs: For each input, the artifact kind and the content hash it was read with.
        code_identity: Identity of the code and policies that produce the result.
        identities: Extra named identities the caller declares for the stage, such as the
            calibration, the map or the observation selection.
    """

    stage_id: str
    contract: str
    stage_config_digest: str
    inputs: Mapping[str, tuple[str, str]]
    code_identity: str
    identities: Mapping[str, str] = field(default_factory=dict)

    def to_document(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible form of the key."""
        return {
            "schema_version": REUSE_SCHEMA_VERSION,
            "stage_id": self.stage_id,
            "contract": self.contract,
            "stage_config_digest": self.stage_config_digest,
            "inputs": {
                name: {"contract": contract, "content_hash": content_hash}
                for name, (contract, content_hash) in sorted(self.inputs.items())
            },
            "code_identity": self.code_identity,
            "identities": dict(sorted(self.identities.items())),
        }

    @property
    def digest(self) -> str:
        """Return ``"sha256:<hex>"`` of the canonical key."""
        canonical = json.dumps(
            self.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class ReuseDecision:
    """Whether one stage was reused or recomputed, and why.

    The exact prior artifact is named when reused: a decision is never a bare cache-hit
    flag.

    Attributes:
        kind: ``"reused"`` or ``"recomputed"``.
        reason: Why, in words.
        key_digest: Digest of the reuse key, or ``None`` when no key could be built.
        reused_from: The exact prior artifact that was reused.
    """

    kind: Literal["reused", "recomputed"]
    reason: str
    key_digest: str | None = None
    reused_from: ArtifactRef | None = None

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form persisted in execution records."""
        return {
            "kind": self.kind,
            "reason": self.reason,
            "key": self.key_digest,
            "reused_from": None if self.reused_from is None else self.reused_from.to_document(),
        }


@dataclass(frozen=True, kw_only=True)
class StoreLookup:
    """The answer of an artifact store for one reuse key.

    Attributes:
        artifact: The completed, still valid artifact recorded for exactly that key.
        reason: Why there is one, or why not.
    """

    artifact: ArtifactRef | None
    reason: str


class ArtifactStore(Protocol):
    """Where completed stage artifacts are indexed by the identity of what produced them."""

    def find(self, key: ReuseKey) -> StoreLookup:
        """Return the completed, still valid artifact recorded for exactly this key."""
        ...

    def record(self, key: ReuseKey, output: ArtifactRef) -> bool:
        """Index a completed artifact under its key; return whether it became the indexed one."""
        ...


@dataclass(frozen=True, kw_only=True)
class ReusePolicy:
    """How one execution decides between reusing and recomputing.

    Attributes:
        store: Where completed artifacts are looked up and recorded.
        code_identity: Identity of the code and policies producing the results, such as a
            commit or a release. It has no default: reuse across code versions must be a
            decision, not an accident.
        force_recompute: Stages that always run, for an experiment that needs a fresh
            artifact.
        identities: Extra named identities per stage, folded into that stage's key.
    """

    store: ArtifactStore
    code_identity: str
    force_recompute: frozenset[str] = frozenset()
    identities: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an empty code identity: nothing could distinguish two code versions."""
        if not self.code_identity.strip():
            raise ValueError("code_identity must not be empty")


class FileArtifactStore:
    """An index of completed artifacts kept as one small immutable JSON file per identity.

    Entries are written only after a stage completes, atomically and without replacing a
    valid entry, so a crash, a failure or a partial output leaves nothing that could
    satisfy a lookup. A lookup also re-validates the entry against its key and asks the
    ``verify`` callback whether the artifact still exists and is intact; anything that
    fails is treated as absent.

    The first completed artifact of an identity stays indexed: a forced recomputation
    yields a new artifact for its own run and does not displace it.
    """

    def __init__(
        self, root: str | os.PathLike[str], *, verify: Callable[[ArtifactRef], bool]
    ) -> None:
        """Create the store.

        Args:
            root: Directory that holds the entries.
            verify: Tells whether an indexed artifact still exists and is intact, for
                example by opening it with its capability's reader. It has no default: an
                index cannot know whether the artifact behind an entry is still there.
        """
        self._root = Path(root)
        self._verify = verify

    def find(self, key: ReuseKey) -> StoreLookup:
        """Return the artifact recorded for exactly ``key`` if it is still valid.

        Args:
            key: The reuse key of the stage about to run.

        Returns:
            The artifact, or the reason there is none.
        """
        path = self._root / _entry_name(key)
        if not path.is_file():
            return StoreLookup(artifact=None, reason="no prior artifact has this identity")
        entry = _read_entry(path)
        if entry is None:
            return StoreLookup(artifact=None, reason="the index entry is unreadable")
        if entry["key_digest"] != key.digest or entry["key"] != key.to_document():
            return StoreLookup(artifact=None, reason="the index entry does not match its identity")
        try:
            output = ArtifactRef.from_document(entry["output"])
        except ValueError:
            return StoreLookup(artifact=None, reason="the index entry has no valid artifact")
        if (
            output.stage_id != key.stage_id
            or output.contract != key.contract
            or output.content_hash is None
        ):
            return StoreLookup(artifact=None, reason="the indexed artifact does not fit the key")
        if not self._verify(output):
            return StoreLookup(
                artifact=None,
                reason=f"the prior artifact {output.artifact_id!r} failed verification",
            )
        return StoreLookup(artifact=output, reason="an artifact with identical identity exists")

    def record(self, key: ReuseKey, output: ArtifactRef) -> bool:
        """Index a completed artifact under ``key``.

        A valid entry is never replaced. An unreadable, mismatched or no longer verifiable
        entry is repaired with the new artifact.

        Args:
            key: The reuse key the artifact was produced under.
            output: The completed artifact; it needs a content hash.

        Returns:
            Whether ``output`` is now the indexed artifact of this identity.

        Raises:
            ReuseError: If the artifact does not belong to the key or has no content hash.
        """
        if output.content_hash is None:
            raise ReuseError(f"artifact {output.artifact_id!r} has no content hash; not indexable")
        if output.stage_id != key.stage_id or output.contract != key.contract:
            raise ReuseError(
                f"artifact {output.artifact_id!r} ({output.contract!r} from {output.stage_id!r}) "
                f"does not belong to the key of stage {key.stage_id!r} producing {key.contract!r}"
            )
        text = (
            json.dumps(
                {
                    "schema_version": REUSE_SCHEMA_VERSION,
                    "key_digest": key.digest,
                    "key": key.to_document(),
                    "output": output.to_document(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        name = _entry_name(key)
        try:
            publish_text(self._root, name, text)
            return True
        except FileExistsError:
            current = self.find(key)
            if current.artifact is not None:
                return current.artifact == output
            replace_text(self._root, name, text)
            return True


def _entry_name(key: ReuseKey) -> str:
    return key.digest.replace(":", "-") + ".json"


def _read_entry(path: Path) -> dict[str, Any] | None:
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or not {"key_digest", "key", "output"} <= entry.keys():
        return None
    return entry
