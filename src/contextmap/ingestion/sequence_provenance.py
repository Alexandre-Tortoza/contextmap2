"""Provenance, integrity, and content-identity metadata for sequence artifacts.

A canonical sequence artifact (:mod:`contextmap.ingestion.sequence_artifact`,
#39) is reused across many later runs. This module makes it self-describing
enough to answer, from the manifest/provenance files alone, where the data
came from, how it was normalized, and whether it still matches what it
originally recorded — the ingestion-specific slice of the global
auditability convention in ``docs/ARTIFACTS.md``. See
``src/contextmap/ingestion/docs/provenance.md`` for the content-identity
strategy and the local-filesystem-only integrity approach.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"
"""Sequence provenance schema version."""


@dataclass(frozen=True, kw_only=True)
class SequenceProvenance:
    """Provenance and content-identity metadata for one sequence artifact.

    Attributes:
        source_type: Adapter family identity, e.g. ``"ros1_bag"``.
        source_path: Path or identity of the ingested source.
        source_content_hash: Deterministic hash summarizing the source's
            content (see :func:`compute_source_content_hash`); lets a
            reader tell unchanged source data apart from source data
            modified under the same path. ``None`` when not computed.
        ingestion_config: Effective ingestion configuration (topic
            mapping, synchronization tolerance, etc.) as primitive
            values. Never contains secrets/credentials.
        configuration_hash: Deterministic hash of ``ingestion_config``
            (see :func:`compute_configuration_hash`). ``None`` when
            ``ingestion_config`` is empty.
        adapter_type: Adapter family/version identity, e.g.
            ``"ros1_bag"``. ``None`` when not applicable (e.g. a manually
            constructed sequence).
        code_version: Ingestion code identity (package version or commit
            SHA), when available.
        calibration_source_hash: Hash identifying the calibration
            attached to this sequence, when any was attached. ``None``
            when no calibration is attached.
        synchronization_policy: Name of the synchronization policy
            applied before persisting (e.g.
            ``"nearest_within_tolerance"``), or ``None`` when this
            artifact stores unsynchronized per-event observations.
        warnings: Adapter/synchronization warnings or known source
            limitations carried into the artifact, as human-readable text.
    """

    source_type: str
    source_path: str
    source_content_hash: str | None = None
    ingestion_config: Mapping[str, object] = field(default_factory=dict)
    configuration_hash: str | None = None
    adapter_type: str | None = None
    code_version: str | None = None
    calibration_source_hash: str | None = None
    synchronization_policy: str | None = None
    warnings: Sequence[str] = ()


def current_code_version() -> str | None:
    """Return this package's installed version, when available.

    Returns:
        ``contextmap.__version__``, or ``None`` if the package metadata
        is unavailable (e.g. running from a source tree with no install).
    """
    from contextmap import __version__

    return __version__ if __version__ != "0.0.0" else None


def compute_configuration_hash(config: Mapping[str, object]) -> str | None:
    """Compute a deterministic hash of an effective ingestion configuration.

    Args:
        config: Effective ingestion configuration, as primitive values.

    Returns:
        ``"sha256:<hex digest>"``, or ``None`` when ``config`` is empty
        (nothing to distinguish).
    """
    if not config:
        return None
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def compute_source_content_hash(source_path: Path) -> str:
    """Compute a deterministic content hash for a raw source file or directory.

    A single file (e.g. a ROS 1 bag) is hashed directly. A directory (e.g.
    a ROS 2 bag, which is a directory of a storage file plus
    ``metadata.yaml``) is hashed over its sorted relative file list, each
    file's size, and each file's own content hash — never over path or
    mtime alone — so modified content under an unchanged path is detected
    and identical content under a different path is recognized as the
    same source.

    This hashes every byte of the source and is therefore O(source size);
    callers decide when to pay that cost (typically once, at ingestion
    time) rather than on every read — see
    ``src/contextmap/ingestion/docs/provenance.md`` for this trade-off.

    Args:
        source_path: Path to the raw source file or directory.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    if source_path.is_file():
        return f"sha256:{_hash_file(source_path)}"

    entries = [
        (
            file_path.relative_to(source_path).as_posix(),
            file_path.stat().st_size,
            _hash_file(file_path),
        )
        for file_path in sorted(path for path in source_path.rglob("*") if path.is_file())
    ]
    payload = json.dumps(entries, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def compute_content_identity(provenance: SequenceProvenance) -> str:
    """Compute the deterministic content identity of a sequence provenance.

    Two calls return the same identity if and only if ``source_type``,
    ``source_content_hash``, and ``configuration_hash`` are all equal —
    i.e. the same source content normalized by the same effective
    configuration. ``source_path`` is deliberately excluded: identical
    content ingested from a different path is still the same content
    identity, and — since ``source_content_hash`` changes with content —
    the same path with modified content is a different identity. Changing
    synchronization/calibration settings changes ``configuration_hash``
    and therefore this identity, as required by this milestone's content
    identity rule.

    Args:
        provenance: The provenance to compute an identity for.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    payload = json.dumps(
        {
            "source_type": provenance.source_type,
            "source_content_hash": provenance.source_content_hash,
            "configuration_hash": provenance.configuration_hash,
        },
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def encode_provenance(provenance: SequenceProvenance) -> dict[str, Any]:
    """Encode a sequence provenance into a JSON-serializable dict.

    Args:
        provenance: The provenance to encode.

    Returns:
        A dict suitable for ``json.dumps``, including the computed
        :func:`compute_content_identity` under ``"content_identity"`` so
        the value is inspectable without recomputing it.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "source_type": provenance.source_type,
        "source_path": provenance.source_path,
        "source_content_hash": provenance.source_content_hash,
        "ingestion_config": dict(provenance.ingestion_config),
        "configuration_hash": provenance.configuration_hash,
        "adapter_type": provenance.adapter_type,
        "code_version": provenance.code_version,
        "calibration_source_hash": provenance.calibration_source_hash,
        "synchronization_policy": provenance.synchronization_policy,
        "warnings": list(provenance.warnings),
        "content_identity": compute_content_identity(provenance),
    }


def decode_provenance(record: dict[str, Any]) -> SequenceProvenance:
    """Decode a sequence provenance from a dict produced by :func:`encode_provenance`.

    Args:
        record: A dict as produced by :func:`encode_provenance`.

    Returns:
        The decoded provenance.

    Raises:
        ValueError: If ``record["schema_version"]`` is not understood by
            this module.
    """
    schema_version = record.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported sequence provenance schema_version: {schema_version!r}")
    return SequenceProvenance(
        source_type=record["source_type"],
        source_path=record["source_path"],
        source_content_hash=record["source_content_hash"],
        ingestion_config=dict(record["ingestion_config"]),
        configuration_hash=record["configuration_hash"],
        adapter_type=record["adapter_type"],
        code_version=record["code_version"],
        calibration_source_hash=record["calibration_source_hash"],
        synchronization_policy=record["synchronization_policy"],
        warnings=tuple(record["warnings"]),
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
