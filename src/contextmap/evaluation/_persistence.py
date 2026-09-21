"""Deterministic digests and immutable JSON persistence shared by evaluation manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

_CHUNK_BYTES = 1 << 20


def canonical_json(value: object) -> str:
    """Return the canonical JSON text digests are computed over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: object) -> str:
    """Return ``sha256:<hex>`` of the canonical JSON form of a JSON-compatible value."""
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def file_digest(path: Path) -> str:
    """Return ``sha256:<hex>`` of a file, streamed so large payloads stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_immutable_json(path: Path, value: object, artifact_name: str) -> None:
    """Atomically publish a JSON document, refusing to replace an existing one."""
    if path.exists():
        raise FileExistsError(f"{artifact_name} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"{artifact_name} already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
