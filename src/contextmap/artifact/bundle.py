"""Portable bundle export of a ContextMapArtifact, with an explicit dependency closure.

A ContextMapArtifact *references* the artifacts it depends on, so moving it alone keeps the map
readable but leaves the geometry behind. A bundle is a directory that carries the artifact and,
by a declared :class:`ClosurePolicy`, the upstream artifacts it needs, so it can be moved to
another filesystem and opened there with no path from the original workspace.

Exporting is transport, not inference. The bundle keeps every identity and content digest of the
source; what changes is only where dependencies live, which is the ``locator`` hint in the
carried manifest and never part of the content identity. It records what was embedded and what
was left out and why, verifies every byte it copies against the inventory it came from, and
carries only contractual files: no debug data, no stray files, no bag, no checkpoint. A bundle
is marked as a bundle (``artifact_type = "context_map_bundle"``) and cannot be opened as the
artifact it carries, which stays where it is, immutable.

Layout::

    <bundle>/
    ├── manifest.json          # bundle manifest: policy, source identity, what was carried
    ├── README.md
    ├── artifact/              # the ContextMapArtifact; only its dependency hints are rewritten
    └── dependencies/<type>/<id>/   # each embedded upstream artifact, contractual files only
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.artifact.dependencies import (
    DependencyResolution,
    DependencyStatus,
    read_inventory,
    resolve_dependency,
)
from contextmap.artifact.directory import load_manifest
from contextmap.artifact.errors import (
    ArtifactExistsError,
    BundleError,
    ContextMapArtifactError,
)
from contextmap.artifact.layout import BUNDLE_ARTIFACT_TYPE, MANIFEST, README
from contextmap.artifact.manifest import (
    ContextMapArtifactManifest,
    DependencyRecord,
    Requirement,
    encode_manifest,
    manifest_content_identity,
)
from contextmap.artifact.validation import (
    VALIDATOR_VERSION,
    Severity,
    ValidationLevel,
    validate_context_map_artifact,
)
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
)

BUNDLE_FORMAT_VERSION = "0.1.0"
"""Version of the bundle layout and manifest written by this code."""

ARTIFACT_DIRECTORY = "artifact"
DEPENDENCIES_DIRECTORY = "dependencies"

_COPY_CHUNK_BYTES = 1 << 20
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "artifact_type",
        "bundle_format_version",
        "closure_policy",
        "source_context_map_id",
        "source_content_identity",
        "source_format_version",
        "source_schema_version",
        "embedded",
        "omitted",
        "exported_at",
        "verified_with_validator_version",
        "bundle_identity",
        "file_inventory",
    }
)


class ClosurePolicy(Enum):
    """Which upstream artifacts a bundle carries besides the ContextMapArtifact itself.

    Attributes:
        CORE_ONLY: Only the artifact. Every dependency, required or not, is recorded as omitted
            by policy; the bundle cannot resolve geometry on its own.
        REQUIRED: The artifact and every required dependency (the geometry the map resolves
            through). Optional evidence is left out unless selected.
        SELECTED_EVIDENCE: The required closure plus the optional evidence the caller selects.
    """

    CORE_ONLY = "core-only"
    REQUIRED = "core+required"
    SELECTED_EVIDENCE = "core+selected-evidence"


@dataclass(frozen=True, kw_only=True)
class EmbeddedDependency:
    """An upstream artifact the bundle carries.

    Attributes:
        artifact_type: Kind of the upstream artifact.
        artifact_id: Its identity.
        content_identity: Digest of its inventory, exactly as the source artifact recorded it.
        requirement: ``"required"`` or ``"optional"`` in the source artifact.
        location: Where it is inside the bundle, relative to the bundle directory.
        file_count: Files carried (its manifest and every inventoried file).
        size_bytes: Total size of those files.
    """

    artifact_type: str
    artifact_id: str
    content_identity: str
    requirement: str
    location: str
    file_count: int
    size_bytes: int


@dataclass(frozen=True, kw_only=True)
class OmittedDependency:
    """An upstream artifact the bundle does not carry, and why.

    Attributes:
        artifact_type: Kind of the upstream artifact.
        artifact_id: Its identity.
        content_identity: Digest of its inventory, as the source artifact recorded it.
        requirement: ``"required"`` or ``"optional"`` in the source artifact.
        reason: Why it was left out.
    """

    artifact_type: str
    artifact_id: str
    content_identity: str
    requirement: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class BundleManifest:
    """What a bundle is, what it carries and where it came from.

    Attributes:
        bundle_format_version: Version of the bundle layout.
        closure_policy: The policy the bundle was exported under.
        source_context_map_id: Identity of the map in the carried artifact.
        source_content_identity: Content identity of the source artifact, unchanged by export.
        source_format_version: Format version of the source artifact.
        source_schema_version: Schema version of the source artifact.
        embedded: The upstream artifacts carried.
        omitted: The upstream artifacts left out, each with the reason.
        exported_at: ISO 8601 export time. Provenance only: it is not part of the identity.
        verified_with_validator_version: Version of the validator that verified the source
            artifact in full before it was copied.
        bundle_identity: SHA-256 of everything above except ``exported_at``, plus the inventory.
        file_inventory: Every file of the bundle with size and hash, except its own manifest and
            README.
    """

    bundle_format_version: str
    closure_policy: ClosurePolicy
    source_context_map_id: str
    source_content_identity: str
    source_format_version: str
    source_schema_version: str
    embedded: tuple[EmbeddedDependency, ...]
    omitted: tuple[OmittedDependency, ...]
    exported_at: str
    verified_with_validator_version: str
    bundle_identity: str
    file_inventory: tuple[FileEntry, ...]


def export_bundle(
    artifact_dir: Path,
    output_dir: Path,
    *,
    policy: ClosurePolicy,
    evidence: Iterable[tuple[str, str]] = (),
    dependency_paths: Mapping[str, Path] | None = None,
    exported_at: datetime | None = None,
) -> BundleManifest:
    """Export a ContextMapArtifact as a portable bundle.

    The source artifact is verified in full first, then copied byte for byte with each byte
    checked against the inventory it came from. The bundle is published atomically; the source
    and its dependencies are only read.

    Args:
        artifact_dir: The ContextMapArtifact to export.
        output_dir: The final directory of the bundle. It must not exist yet.
        policy: Which upstream artifacts to carry.
        evidence: With :attr:`ClosurePolicy.SELECTED_EVIDENCE`, the ``(artifact_type,
            artifact_id)`` of each optional dependency to carry. Any other policy must not
            select evidence.
        dependency_paths: Where upstream artifacts are now, keyed by artifact id; without an
            entry the relative hint of the source manifest is tried.
        exported_at: Export time recorded as provenance; the current UTC time when omitted.

    Returns:
        The manifest of the published bundle.

    Raises:
        ArtifactExistsError: If ``output_dir`` already exists.
        BundleError: If the source does not verify in full (a missing required dependency or a
            damaged file included), selected evidence is not one the artifact records or cannot
            be found, or a copied byte does not match the inventory it came from.
    """
    if output_dir.exists():
        raise ArtifactExistsError(f"the bundle already exists: {output_dir}")
    selection = set(evidence)
    if selection and policy is not ClosurePolicy.SELECTED_EVIDENCE:
        raise BundleError(
            f"selected evidence needs the {ClosurePolicy.SELECTED_EVIDENCE.value} policy, "
            f"not {policy.value}"
        )
    paths = dependency_paths if dependency_paths is not None else {}

    report = validate_context_map_artifact(
        artifact_dir, level=ValidationLevel.FULL, dependency_paths=paths
    )
    codes = sorted({item.code for item in report.findings if item.severity is Severity.ERROR})
    if codes:
        raise BundleError(
            "the artifact does not verify in full, so it is not exported: " + ", ".join(codes)
        )
    manifest = load_manifest(artifact_dir)

    recorded = {record.key for record in manifest.dependencies}
    for key in sorted(selection - recorded):
        raise BundleError(f"the artifact records no dependency {key[0]} {key[1]!r} to select")
    embedded_records, omitted = _plan(manifest, policy, selection)
    resolutions = {
        record.key: _located(record, artifact_dir, paths, selected=record.key in selection)
        for record in embedded_records
    }

    when = exported_at if exported_at is not None else datetime.now(UTC)
    try:
        with AtomicRunDirectory(output_dir) as run:
            entries: list[FileEntry] = []
            entries += _carry_artifact(run, artifact_dir, manifest, embedded_records)
            embedded: list[EmbeddedDependency] = []
            for record in embedded_records:
                located = resolutions[record.key].location
                assert located is not None
                carried = _carry_dependency(run, located, record)
                entries += carried
                embedded.append(
                    EmbeddedDependency(
                        artifact_type=record.artifact_type,
                        artifact_id=record.artifact_id,
                        content_identity=record.content_identity,
                        requirement=record.requirement.value,
                        location=_dependency_location(record),
                        file_count=len(carried),
                        size_bytes=sum(entry.size_bytes for entry in carried),
                    )
                )
            bundle = _sealed(manifest, policy, embedded, omitted, when, entries)
            run.publish(manifest=encode_bundle_manifest(bundle), readme=_render_readme(bundle))
    except RunDirectoryError as error:
        raise BundleError(str(error)) from error
    return bundle


def _plan(
    manifest: ContextMapArtifactManifest, policy: ClosurePolicy, selection: set[tuple[str, str]]
) -> tuple[list[DependencyRecord], list[OmittedDependency]]:
    """Split the recorded dependencies into the ones to carry and the ones to leave out."""
    carry: list[DependencyRecord] = []
    omitted: list[OmittedDependency] = []
    for record in manifest.dependencies:
        if policy is ClosurePolicy.CORE_ONLY:
            reason: str | None = "left out by the core-only policy"
        elif record.requirement is Requirement.REQUIRED or record.key in selection:
            reason = None
        else:
            reason = "optional evidence was not selected"
        if reason is None:
            carry.append(record)
        else:
            omitted.append(
                OmittedDependency(
                    artifact_type=record.artifact_type,
                    artifact_id=record.artifact_id,
                    content_identity=record.content_identity,
                    requirement=record.requirement.value,
                    reason=reason,
                )
            )
    return carry, omitted


def _located(
    record: DependencyRecord, artifact_dir: Path, paths: Mapping[str, Path], *, selected: bool
) -> DependencyResolution:
    """Find a dependency that has to be carried; not finding it blocks the export."""
    resolution = resolve_dependency(record, artifact_root=artifact_dir, dependency_paths=paths)
    if resolution.status is not DependencyStatus.FOUND:
        kind = "selected evidence" if selected else "required dependency"
        raise BundleError(f"the {kind} cannot be carried: {resolution.detail}")
    return resolution


def _dependency_location(record: DependencyRecord) -> str:
    for part in (record.artifact_type, record.artifact_id):
        if _SAFE_NAME.match(part) is None:
            raise BundleError(
                f"{part!r} cannot name a directory of the bundle: use letters, digits, '.', "
                "'_' and '-'"
            )
    return f"{DEPENDENCIES_DIRECTORY}/{record.artifact_type}/{record.artifact_id}"


def _carry_artifact(
    run: AtomicRunDirectory,
    artifact_dir: Path,
    manifest: ContextMapArtifactManifest,
    carried: list[DependencyRecord],
) -> list[FileEntry]:
    """Copy the artifact; only the transport hints of its dependencies are rewritten."""
    embedded = {record.key for record in carried}
    rewritten = replace(
        manifest,
        dependencies=tuple(
            replace(
                record,
                locator=f"../{_dependency_location(record)}" if record.key in embedded else None,
            )
            for record in manifest.dependencies
        ),
    )
    if manifest_content_identity(rewritten) != manifest.content_identity:
        raise BundleError("rewriting the dependency hints changed the content identity")
    entries = [
        _write_bytes(
            run,
            f"{ARTIFACT_DIRECTORY}/{MANIFEST}",
            json.dumps(encode_manifest(rewritten), indent=2, sort_keys=True).encode("utf-8"),
        )
    ]
    readme = artifact_dir / README
    if readme.is_file():
        entries.append(_copy(run, readme, f"{ARTIFACT_DIRECTORY}/{README}", None))
    for entry in manifest.file_inventory:
        entries.append(
            _copy(run, artifact_dir / entry.path, f"{ARTIFACT_DIRECTORY}/{entry.path}", entry)
        )
    return entries


def _carry_dependency(
    run: AtomicRunDirectory, located: Path, record: DependencyRecord
) -> list[FileEntry]:
    """Copy the contractual files of one upstream artifact: its manifest and its inventory."""
    base = _dependency_location(record)
    entries = [_copy(run, located / MANIFEST, f"{base}/{MANIFEST}", None)]
    readme = located / README
    if readme.is_file():
        entries.append(_copy(run, readme, f"{base}/{README}", None))
    for entry in read_inventory(located):
        entries.append(_copy(run, located / entry.path, f"{base}/{entry.path}", entry))
    return entries


def _write_bytes(run: AtomicRunDirectory, destination: str, data: bytes) -> FileEntry:
    run.write_bytes(destination, data)
    return FileEntry(
        path=destination,
        size_bytes=len(data),
        content_hash=f"sha256:{hashlib.sha256(data).hexdigest()}",
    )


def _copy(
    run: AtomicRunDirectory, source: Path, destination: str, expected: FileEntry | None
) -> FileEntry:
    """Copy one file in chunks, hashing what was read; refuse it if it is not what was promised."""
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, run.open_binary(destination) as sink:
        while chunk := reader.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
            sink.write(chunk)
    content_hash = f"sha256:{digest.hexdigest()}"
    if expected is not None and (size, content_hash) != (
        expected.size_bytes,
        expected.content_hash,
    ):
        raise BundleError(
            f"{expected.path}: the bytes read from the source do not match its inventory "
            "(the file changed while exporting)"
        )
    return FileEntry(path=destination, size_bytes=size, content_hash=content_hash)


def _identity(bundle: BundleManifest) -> str:
    record = encode_bundle_manifest(bundle)
    del record["exported_at"], record["bundle_identity"]
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _sealed(
    manifest: ContextMapArtifactManifest,
    policy: ClosurePolicy,
    embedded: list[EmbeddedDependency],
    omitted: list[OmittedDependency],
    when: datetime,
    entries: list[FileEntry],
) -> BundleManifest:
    provisional = BundleManifest(
        bundle_format_version=BUNDLE_FORMAT_VERSION,
        closure_policy=policy,
        source_context_map_id=manifest.context_map_id,
        source_content_identity=manifest.content_identity,
        source_format_version=manifest.format_version,
        source_schema_version=manifest.schema_version,
        embedded=tuple(sorted(embedded, key=lambda item: (item.artifact_type, item.artifact_id))),
        omitted=tuple(sorted(omitted, key=lambda item: (item.artifact_type, item.artifact_id))),
        exported_at=when.isoformat(),
        verified_with_validator_version=VALIDATOR_VERSION,
        bundle_identity="sha256:" + "0" * 64,
        file_inventory=tuple(sorted(entries, key=lambda entry: entry.path)),
    )
    return replace(provisional, bundle_identity=_identity(provisional))


def _render_readme(bundle: BundleManifest) -> str:
    lines = [
        f"# Bundle of the context map `{bundle.source_context_map_id}`",
        "",
        "This directory is a transport bundle, not an inference run. The context map it carries "
        "is unchanged and is in `artifact/`.",
        "",
        f"- Closure policy: `{bundle.closure_policy.value}`",
        f"- Content identity of the carried artifact: `{bundle.source_content_identity}`",
        f"- Bundle identity: `{bundle.bundle_identity}`",
        "",
        "Carried upstream artifacts:",
        "",
    ]
    lines += [
        f"- `{item.artifact_type}` `{item.artifact_id}` ({item.requirement}) in `{item.location}`"
        for item in bundle.embedded
    ] or ["- none"]
    lines += ["", "Left out:", ""]
    lines += [
        f"- `{item.artifact_type}` `{item.artifact_id}` ({item.requirement}): {item.reason}"
        for item in bundle.omitted
    ] or ["- nothing"]
    lines += [""]
    return "\n".join(lines)


def encode_bundle_manifest(bundle: BundleManifest) -> dict[str, Any]:
    """Encode a bundle manifest as a JSON-compatible record."""
    return {
        "artifact_type": BUNDLE_ARTIFACT_TYPE,
        "bundle_format_version": bundle.bundle_format_version,
        "closure_policy": bundle.closure_policy.value,
        "source_context_map_id": bundle.source_context_map_id,
        "source_content_identity": bundle.source_content_identity,
        "source_format_version": bundle.source_format_version,
        "source_schema_version": bundle.source_schema_version,
        "embedded": [
            {
                "artifact_type": item.artifact_type,
                "artifact_id": item.artifact_id,
                "content_identity": item.content_identity,
                "requirement": item.requirement,
                "location": item.location,
                "file_count": item.file_count,
                "size_bytes": item.size_bytes,
            }
            for item in bundle.embedded
        ],
        "omitted": [
            {
                "artifact_type": item.artifact_type,
                "artifact_id": item.artifact_id,
                "content_identity": item.content_identity,
                "requirement": item.requirement,
                "reason": item.reason,
            }
            for item in bundle.omitted
        ],
        "exported_at": bundle.exported_at,
        "verified_with_validator_version": bundle.verified_with_validator_version,
        "bundle_identity": bundle.bundle_identity,
        "file_inventory": [
            {"path": item.path, "size_bytes": item.size_bytes, "content_hash": item.content_hash}
            for item in bundle.file_inventory
        ],
    }


def read_bundle_manifest(bundle_dir: Path) -> BundleManifest:
    """Read and decode the manifest of a bundle directory.

    Args:
        bundle_dir: The bundle directory.

    Returns:
        The manifest.

    Raises:
        BundleError: If the directory has no manifest, it is not a bundle manifest, or it is
            malformed.
    """
    path = bundle_dir / MANIFEST
    if not path.is_file():
        raise BundleError(f"{bundle_dir.name!r} has no {MANIFEST}: it is not a bundle")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("artifact_type") != BUNDLE_ARTIFACT_TYPE:
            raise BundleError(
                f"{bundle_dir.name!r} is not a bundle: its artifact_type is "
                f"{record.get('artifact_type')!r}, not {BUNDLE_ARTIFACT_TYPE!r}"
            )
        if record.keys() != _MANIFEST_KEYS:
            wrong = sorted(record.keys() ^ _MANIFEST_KEYS)
            raise BundleError(f"the bundle manifest has the wrong fields: {wrong}")
        if record["bundle_format_version"] != BUNDLE_FORMAT_VERSION:
            raise BundleError(
                f"unsupported bundle_format_version {record['bundle_format_version']!r}; "
                f"this code reads {BUNDLE_FORMAT_VERSION}"
            )
        bundle = BundleManifest(
            bundle_format_version=record["bundle_format_version"],
            closure_policy=ClosurePolicy(record["closure_policy"]),
            source_context_map_id=record["source_context_map_id"],
            source_content_identity=record["source_content_identity"],
            source_format_version=record["source_format_version"],
            source_schema_version=record["source_schema_version"],
            embedded=tuple(EmbeddedDependency(**item) for item in record["embedded"]),
            omitted=tuple(OmittedDependency(**item) for item in record["omitted"]),
            exported_at=record["exported_at"],
            verified_with_validator_version=record["verified_with_validator_version"],
            bundle_identity=record["bundle_identity"],
            file_inventory=tuple(FileEntry(**item) for item in record["file_inventory"]),
        )
    except BundleError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise BundleError(
            f"the bundle manifest of {bundle_dir.name!r} is malformed: {error}"
        ) from error
    if _SHA256.match(bundle.bundle_identity) is None:
        raise BundleError("the bundle manifest has a malformed bundle_identity")
    return bundle


def verify_bundle(bundle_dir: Path) -> tuple[str, ...]:
    """Check that a bundle is intact and that its artifact verifies with what it carries.

    Every file of the bundle is hashed against the bundle inventory, the bundle identity is
    recomputed, the carried artifact must still have the content identity the bundle records,
    and the artifact is validated in full. A dependency the bundle declares as omitted is not
    a problem: it is what the policy said. Nothing is repaired.

    Args:
        bundle_dir: The bundle directory.

    Returns:
        The problems found, each a sentence; empty means the bundle is intact.
    """
    try:
        bundle = read_bundle_manifest(bundle_dir)
    except BundleError as error:
        return (str(error),)
    problems: list[str] = []
    if _identity(bundle) != bundle.bundle_identity:
        problems.append(
            f"the bundle identity does not match its manifest: recorded {bundle.bundle_identity}, "
            f"recomputed {_identity(bundle)}"
        )
    problems += check_file_inventory(bundle_dir, bundle.file_inventory)
    artifact_dir = bundle_dir / ARTIFACT_DIRECTORY
    try:
        carried = load_manifest(artifact_dir)
    except ContextMapArtifactError as error:
        return (*problems, f"the carried artifact cannot be read: {error}")
    if carried.content_identity != bundle.source_content_identity:
        problems.append(
            "the carried artifact is not the source the bundle records: content identity "
            f"{carried.content_identity} instead of {bundle.source_content_identity}"
        )
    omitted = {item.artifact_id for item in bundle.omitted}
    report = validate_context_map_artifact(artifact_dir, level=ValidationLevel.FULL)
    for finding in report.findings:
        if finding.severity is not Severity.ERROR:
            continue
        if finding.code == "dependency.required_missing" and finding.subject in omitted:
            continue
        problems.append(f"{finding.code}: {finding.message}")
    return tuple(problems)
