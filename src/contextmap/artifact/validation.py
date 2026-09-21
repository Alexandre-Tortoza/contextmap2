"""Integrity validation of a ContextMapArtifact, independent of the mapping runtime.

The validator answers one question for a consumer that is about to trust an artifact: is it
intact, consistent and complete enough to use? It checks the manifest and its versions, the
inventory of files, the hashes, the descriptors of the payloads, the indexes, the references
between records, the lineage, the declared capabilities against the content, and the upstream
artifacts the map depends on, distinguishing the ones it cannot work without from optional
evidence.

It never raises for a damaged artifact and never repairs one: everything it finds is a finding in
a deterministic, machine-readable :class:`ValidationReport`, with the checks that ran and the
ones that did not. There are two levels. ``STRUCTURAL`` is fast (a ``stat`` per file, the indexes
and the small documents) and its best answer is ``STRUCTURALLY_VALID``; only ``FULL``
(hashes, every record, references, rebuilt indexes, upstream files) may answer ``VERIFIED``. An
unchecked artifact is never reported as valid.

The validator needs no model, ROS or robotics runtime, and reads only the files it is given.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.artifact.dependencies import (
    GEOMETRIC_MAP_ARTIFACT_TYPE,
    DependencyResolution,
    DependencyStatus,
    read_inventory,
    resolve_dependency,
)
from contextmap.artifact.directory import load_manifest
from contextmap.artifact.errors import (
    ArtifactIntegrityError,
    ContextMapArtifactError,
    IncompleteContextMapArtifactError,
    ManifestError,
    RecordTableError,
    UnsupportedArtifactSchemaError,
    UnsupportedFormatVersionError,
    UpstreamArtifactError,
)
from contextmap.artifact.layout import (
    CONTRACTUAL_FILES,
    DEBUG_DIRECTORY,
    ENTITIES,
    ENTITY_INDEX,
    ENTITY_RELATION_INDEX,
    GEOMETRY_REFERENCE,
    LINEAGE,
    MANIFEST,
    MAP_METADATA,
    README,
    RELATION_INDEX,
    RELATIONS,
)
from contextmap.artifact.manifest import (
    ContextMapArtifactManifest,
    DependencyRecord,
    PayloadRole,
    RecordPayload,
    Requirement,
)
from contextmap.artifact.metadata import MapCapability
from contextmap.artifact.models import ContextMap
from contextmap.artifact.records import ContextMapRecordError, context_map_from_record
from contextmap.artifact.tables import (
    RecordTable,
    encode_entity_relation_index,
    rebuild_index,
)
from contextmap.geometric_mapping import GeometricMapArtifactReader, MapArtifactError
from contextmap.shared import FileEntry, check_file_inventory

VALIDATOR_VERSION = "0.1.0"
"""Version of the validator's checks and of the report format it writes."""

_SKIPPED_AT_STRUCTURAL = "not run at the structural level"


class _Skip(Exception):
    """Raised by a check that cannot run because an earlier error already explains why."""


class ValidationLevel(Enum):
    """How much a validation reads.

    Attributes:
        STRUCTURAL: Fast: manifest, versions, one ``stat`` per file, payload descriptors, the
            structure of the indexes, the small documents and where the upstream artifacts are.
            It cannot verify contents.
        FULL: Everything above plus every hash, every record, the references between records,
            the indexes rebuilt from the records, and the upstream artifacts' own files.
    """

    STRUCTURAL = "structural"
    FULL = "full"


class Severity(Enum):
    """How serious a finding is.

    Attributes:
        ERROR: The artifact cannot be trusted for what the finding concerns.
        WARNING: Something a person should know that does not stop the core map from being
            trusted, such as optional evidence that is not there.
    """

    ERROR = "error"
    WARNING = "warning"


class ValidationStatus(Enum):
    """What the validation is entitled to say about the artifact.

    Attributes:
        INVALID: At least one error.
        STRUCTURALLY_VALID: No error, but only the structural level ran: contents, references and
            upstream files were not verified, so the artifact is not called valid.
        VERIFIED: No error at the full level. Checks that this validator cannot perform yet are
            listed as skipped, with the reason, in ``checks``.
    """

    INVALID = "invalid"
    STRUCTURALLY_VALID = "structurally_valid"
    VERIFIED = "verified"


class CheckOutcome(Enum):
    """What happened to one check.

    Attributes:
        PASSED: It ran and found no error.
        FAILED: It ran and found at least one error.
        SKIPPED: It did not run; ``detail`` says why.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileStatus(Enum):
    """What is known about one inventoried file.

    Attributes:
        VERIFIED: Present, with the recorded size and the recorded SHA-256.
        SIZE_OK: Present with the recorded size; the hash was not checked.
        MISSING: Not on disk.
        SIZE_MISMATCH: On disk with another size (truncated or changed).
        HASH_MISMATCH: On disk with the recorded size but another SHA-256.
    """

    VERIFIED = "verified"
    SIZE_OK = "size_ok"
    MISSING = "missing"
    SIZE_MISMATCH = "size_mismatch"
    HASH_MISMATCH = "hash_mismatch"


@dataclass(frozen=True, kw_only=True)
class Finding:
    """One thing the validation found.

    Attributes:
        code: A stable machine-readable identifier such as ``file.hash_mismatch``.
        severity: Error or warning.
        message: A sentence for people.
        subject: What the finding is about (a relative file path, a record key or an artifact
            id); never an absolute path.
    """

    code: str
    severity: Severity
    message: str
    subject: str | None


@dataclass(frozen=True, kw_only=True)
class CheckResult:
    """The outcome of one named check.

    Attributes:
        name: The check, such as ``file_hashes``.
        outcome: Passed, failed or skipped.
        detail: For a skipped check, why; otherwise ``None``.
    """

    name: str
    outcome: CheckOutcome
    detail: str | None


@dataclass(frozen=True, kw_only=True)
class CheckedFile:
    """One inventoried file and what the validation established about it.

    Attributes:
        path: Path relative to the artifact directory.
        size_bytes: Size recorded in the manifest.
        content_hash: SHA-256 recorded in the manifest.
        status: What was verified.
    """

    path: str
    size_bytes: int
    content_hash: str
    status: FileStatus


@dataclass(frozen=True, kw_only=True)
class DependencyReport:
    """What was found for one recorded dependency.

    Attributes:
        artifact_type: Kind of the upstream artifact.
        artifact_id: Its identity.
        requirement: ``"required"`` or ``"optional"``.
        status: ``"found"``, ``"missing"`` or ``"mismatch"``.
        detail: A sentence explaining the status.
    """

    artifact_type: str
    artifact_id: str
    requirement: str
    status: str
    detail: str


@dataclass(frozen=True, kw_only=True)
class ValidationReport:
    """The machine-readable outcome of validating one artifact.

    The report is a function of the artifact and of the level: it carries no time, no absolute
    path and no host information, so validating the same artifact twice, or a copy of it, gives
    the same bytes.

    Attributes:
        validator_version: Version of the validator that produced it.
        level: The level that ran.
        status: What the validation is entitled to say.
        artifact_type: ``artifact_type`` of the manifest, or ``None`` if it could not be read.
        format_version: Format version of the manifest, or ``None``.
        schema_version: Schema version of the manifest, or ``None``.
        context_map_id: Identity of the map, or ``None``.
        content_identity: Content identity recorded in the manifest, or ``None``.
        checks: Every check in a fixed order, with what happened to it.
        files: The inventory that was checked, sorted by path.
        dependencies: The recorded dependencies and what was found for each.
        findings: Errors first, then warnings, each ordered by code and subject.
    """

    validator_version: str
    level: ValidationLevel
    status: ValidationStatus
    artifact_type: str | None
    format_version: str | None
    schema_version: str | None
    context_map_id: str | None
    content_identity: str | None
    checks: tuple[CheckResult, ...]
    files: tuple[CheckedFile, ...]
    dependencies: tuple[DependencyReport, ...]
    findings: tuple[Finding, ...]

    def to_record(self) -> dict[str, Any]:
        """Write the report as JSON-compatible values.

        Returns:
            A mapping with one entry per field; enums are written as their values.
        """
        return {
            "validator_version": self.validator_version,
            "level": self.level.value,
            "status": self.status.value,
            "artifact_type": self.artifact_type,
            "format_version": self.format_version,
            "schema_version": self.schema_version,
            "context_map_id": self.context_map_id,
            "content_identity": self.content_identity,
            "checks": [
                {"name": item.name, "outcome": item.outcome.value, "detail": item.detail}
                for item in self.checks
            ],
            "files": [
                {
                    "path": item.path,
                    "size_bytes": item.size_bytes,
                    "content_hash": item.content_hash,
                    "status": item.status.value,
                }
                for item in self.files
            ],
            "dependencies": [
                {
                    "artifact_type": item.artifact_type,
                    "artifact_id": item.artifact_id,
                    "requirement": item.requirement,
                    "status": item.status,
                    "detail": item.detail,
                }
                for item in self.dependencies
            ],
            "findings": [
                {
                    "code": item.code,
                    "severity": item.severity.value,
                    "message": item.message,
                    "subject": item.subject,
                }
                for item in self.findings
            ],
        }

    def to_json(self) -> str:
        """Write the report as deterministic JSON: sorted keys, indented, newline-terminated."""
        return json.dumps(self.to_record(), sort_keys=True, indent=2) + "\n"


_CHECK_ORDER = (
    "manifest",
    "inventory",
    "file_hashes",
    "payload_descriptors",
    "index_structure",
    "map_record",
    "capabilities",
    "lineage",
    "dependencies",
    "geometry_consistency",
    "reference_integrity",
    "index_rebuild",
    "dependency_integrity",
    "entity_geometry_support",
    "unlisted_files",
)

_MANIFEST_ERROR_CODES: tuple[tuple[type[ContextMapArtifactError], str], ...] = (
    (IncompleteContextMapArtifactError, "artifact.incomplete"),
    (UnsupportedFormatVersionError, "manifest.unsupported_format_version"),
    (UnsupportedArtifactSchemaError, "manifest.unsupported_schema_version"),
    (ArtifactIntegrityError, "manifest.identity_mismatch"),
    (ManifestError, "manifest.malformed"),
)

# Os payloads que todo artifact v0 tem, com o papel e a origem esperados de cada um.
_EXPECTED_PAYLOADS: dict[str, tuple[PayloadRole, tuple[str, ...]]] = {
    ENTITIES: (PayloadRole.AUTHORITATIVE, ()),
    RELATIONS: (PayloadRole.AUTHORITATIVE, ()),
    ENTITY_INDEX: (PayloadRole.DERIVED_INDEX, (ENTITIES,)),
    RELATION_INDEX: (PayloadRole.DERIVED_INDEX, (RELATIONS,)),
    ENTITY_RELATION_INDEX: (PayloadRole.DERIVED_INDEX, (ENTITIES, RELATIONS)),
}


def validate_context_map_artifact(
    path: Path,
    *,
    level: ValidationLevel = ValidationLevel.FULL,
    dependency_paths: Mapping[str, Path] | None = None,
) -> ValidationReport:
    """Validate a ContextMapArtifact directory.

    The artifact is only read. Damage of any kind is a finding in the report, never an exception
    and never repaired.

    Args:
        path: The artifact directory.
        level: How much to read; see :class:`ValidationLevel`.
        dependency_paths: Where upstream artifacts are now, keyed by artifact id. Without an
            entry the relative hint of the manifest is tried.

    Returns:
        The report.
    """
    return _Validation(
        Path(path), level, dependency_paths if dependency_paths is not None else {}
    ).run()


class _Validation:
    """One validation run: the state the checks share and the findings they add."""

    def __init__(
        self, root: Path, level: ValidationLevel, dependency_paths: Mapping[str, Path]
    ) -> None:
        self._root = root
        self._level = level
        self._dependency_paths = dependency_paths
        self._findings: list[Finding] = []
        self._results: dict[str, CheckResult] = {}
        self._manifest: ContextMapArtifactManifest | None = None
        self._files: dict[str, CheckedFile] = {}
        self._usable: set[str] = set()
        self._context_map: ContextMap | None = None
        self._resolutions: list[DependencyResolution] = []
        self._entity_keys: set[str] | None = None
        self._endpoints: list[tuple[str, str, str]] | None = None
        self._endpoint_errors = False

    def run(self) -> ValidationReport:
        """Run every check in a fixed order and assemble the report."""
        self._check("manifest", self._check_manifest)
        if self._manifest is None:
            for name in _CHECK_ORDER[1:]:
                self._skip(name, "the manifest could not be read")
        else:
            self._check("inventory", self._check_inventory)
            self._at_full("file_hashes", self._check_hashes)
            self._check("payload_descriptors", self._check_descriptors)
            self._check("index_structure", self._check_index_structure)
            self._check("map_record", self._check_map_record)
            self._check("capabilities", self._check_capabilities)
            self._check("lineage", self._check_lineage)
            self._check("dependencies", self._check_dependencies)
            self._check("geometry_consistency", self._check_geometry)
            self._at_full("reference_integrity", self._check_references)
            self._at_full("index_rebuild", self._check_index_rebuild)
            self._at_full("dependency_integrity", self._check_upstream_files)
            self._skip(
                "entity_geometry_support",
                "entity records are opaque to the artifact until the schema types them",
            )
            self._check("unlisted_files", self._check_unlisted)
        return self._report()

    def _report(self) -> ValidationReport:
        manifest = self._manifest
        errors = any(item.severity is Severity.ERROR for item in self._findings)
        if errors:
            status = ValidationStatus.INVALID
        elif self._level is ValidationLevel.FULL:
            status = ValidationStatus.VERIFIED
        else:
            status = ValidationStatus.STRUCTURALLY_VALID
        unique = {(item.code, item.subject, item.message): item for item in self._findings}
        findings = sorted(
            unique.values(),
            key=lambda item: (
                item.severity is Severity.WARNING,
                item.code,
                item.subject or "",
                item.message,
            ),
        )
        return ValidationReport(
            validator_version=VALIDATOR_VERSION,
            level=self._level,
            status=status,
            artifact_type=None if manifest is None else "context_map",
            format_version=None if manifest is None else manifest.format_version,
            schema_version=None if manifest is None else manifest.schema_version,
            context_map_id=None if manifest is None else manifest.context_map_id,
            content_identity=None if manifest is None else manifest.content_identity,
            checks=tuple(self._results[name] for name in _CHECK_ORDER),
            files=tuple(self._files[key] for key in sorted(self._files)),
            dependencies=tuple(
                DependencyReport(
                    artifact_type=item.record.artifact_type,
                    artifact_id=item.record.artifact_id,
                    requirement=item.record.requirement.value,
                    status=item.status.value,
                    detail=item.detail,
                )
                for item in sorted(self._resolutions, key=lambda item: item.record.key)
            ),
            findings=tuple(findings),
        )

    # -- plumbing --------------------------------------------------------------------------

    def _error(self, code: str, message: str, subject: str | None = None) -> None:
        self._findings.append(
            Finding(code=code, severity=Severity.ERROR, message=message, subject=subject)
        )

    def _warning(self, code: str, message: str, subject: str | None = None) -> None:
        self._findings.append(
            Finding(code=code, severity=Severity.WARNING, message=message, subject=subject)
        )

    def _skip(self, name: str, reason: str) -> None:
        self._results[name] = CheckResult(name=name, outcome=CheckOutcome.SKIPPED, detail=reason)

    def _at_full(self, name: str, body: Callable[[], None]) -> None:
        if self._level is ValidationLevel.FULL:
            self._check(name, body)
        else:
            self._skip(name, _SKIPPED_AT_STRUCTURAL)

    def _check(self, name: str, body: Callable[[], None]) -> None:
        """Run a check, turning the artifact-level failures it hits into findings."""
        before = sum(1 for item in self._findings if item.severity is Severity.ERROR)
        try:
            body()
        except _Skip as skip:
            self._skip(name, str(skip))
            return
        except (ContextMapArtifactError, OSError, ValueError) as error:
            self._error(f"{name}.failed", f"the {name} check could not complete: {error}")
        after = sum(1 for item in self._findings if item.severity is Severity.ERROR)
        outcome = CheckOutcome.FAILED if after > before else CheckOutcome.PASSED
        self._results[name] = CheckResult(name=name, outcome=outcome, detail=None)

    def _read(self, relative_path: str) -> bytes:
        return (self._root / relative_path).read_bytes()

    def _document(self, relative_path: str) -> dict[str, Any]:
        record = json.loads(self._read(relative_path).decode("utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"{relative_path} must hold a JSON object")
        return record

    def _manifest_required(self) -> ContextMapArtifactManifest:
        assert self._manifest is not None
        return self._manifest

    # -- checks ----------------------------------------------------------------------------

    def _check_manifest(self) -> None:
        try:
            self._manifest = load_manifest(self._root)
        except ContextMapArtifactError as error:
            code = next(
                (code for kind, code in _MANIFEST_ERROR_CODES if isinstance(error, kind)),
                "manifest.failed",
            )
            self._error(code, str(error), MANIFEST)

    def _check_inventory(self) -> None:
        manifest = self._manifest_required()
        listed = {entry.path for entry in manifest.file_inventory}
        for required in CONTRACTUAL_FILES:
            if required not in listed:
                self._error(
                    "inventory.required_file_missing",
                    f"the manifest does not inventory the required file {required!r}",
                    required,
                )
        for entry in manifest.file_inventory:
            file_path = self._root / entry.path
            status = FileStatus.SIZE_OK
            if not file_path.is_file():
                status = FileStatus.MISSING
                self._error("file.missing", f"the file {entry.path!r} is not on disk", entry.path)
            else:
                size = file_path.stat().st_size
                if size != entry.size_bytes:
                    status = FileStatus.SIZE_MISMATCH
                    self._error(
                        "file.size_mismatch",
                        f"{entry.path!r} has {size} bytes but the manifest records "
                        f"{entry.size_bytes}: it was truncated or changed",
                        entry.path,
                    )
                else:
                    self._usable.add(entry.path)
            self._files[entry.path] = _checked(entry, status)

    def _check_hashes(self) -> None:
        for path in sorted(self._usable):
            checked = self._files[path]
            entry = FileEntry(
                path=path, size_bytes=checked.size_bytes, content_hash=checked.content_hash
            )
            if check_file_inventory(self._root, [entry]):
                self._files[path] = _with_status(checked, FileStatus.HASH_MISMATCH)
                self._usable.discard(path)
                self._error(
                    "file.hash_mismatch",
                    f"{path!r} has the recorded size but not the recorded SHA-256: it was changed",
                    path,
                )
            else:
                self._files[path] = _with_status(checked, FileStatus.VERIFIED)

    def _check_descriptors(self) -> None:
        manifest = self._manifest_required()
        described = {payload.path: payload for payload in manifest.payloads}
        counts = {ENTITIES: manifest.entity_count, RELATIONS: manifest.relation_count}
        for path, (role, sources) in _EXPECTED_PAYLOADS.items():
            payload = described.get(path)
            if payload is None:
                self._error(
                    "payload.descriptor_missing", f"the manifest does not describe {path!r}", path
                )
                continue
            if payload.role is not role or payload.derived_from != sources:
                self._error(
                    "payload.descriptor_mismatch",
                    f"the descriptor of {path!r} declares the role {payload.role.value!r} "
                    f"derived from {list(payload.derived_from)}, expected {role.value!r} "
                    f"derived from {list(sources)}",
                    path,
                )
            expected = {
                ENTITY_INDEX: manifest.entity_count,
                RELATION_INDEX: manifest.relation_count,
                ENTITY_RELATION_INDEX: manifest.entity_count,
                **counts,
            }[path]
            if not isinstance(payload, RecordPayload) or payload.record_count != expected:
                self._error(
                    "payload.count_mismatch",
                    f"the descriptor of {path!r} does not declare {expected} records",
                    path,
                )

    def _check_index_structure(self) -> None:
        manifest = self._manifest_required()
        for payload_path, index_path, count in (
            (ENTITIES, ENTITY_INDEX, manifest.entity_count),
            (RELATIONS, RELATION_INDEX, manifest.relation_count),
        ):
            if not {payload_path, index_path} <= self._usable:
                continue
            try:
                RecordTable(self._root / payload_path, self._root / index_path, record_count=count)
            except ContextMapArtifactError as error:
                self._error("index.broken", str(error), index_path)
        if ENTITY_RELATION_INDEX in self._usable:
            try:
                for number, text in enumerate(self._read(ENTITY_RELATION_INDEX).split(b"\n")[:-1]):
                    line = json.loads(text)
                    if not isinstance(line, dict) or line.keys() != {
                        "key",
                        "as_subject",
                        "as_object",
                    }:
                        raise ValueError(f"line {number} has the wrong fields")
            except ValueError as error:
                self._error(
                    "index.broken",
                    f"{ENTITY_RELATION_INDEX} is not a valid traversal index ({error})",
                    ENTITY_RELATION_INDEX,
                )

    def _check_map_record(self) -> None:
        manifest = self._manifest_required()
        if not {MAP_METADATA, GEOMETRY_REFERENCE} <= self._usable:
            raise _Skip(f"{MAP_METADATA} or {GEOMETRY_REFERENCE} is not intact")
        try:
            self._context_map = context_map_from_record(
                {
                    "context_map_id": manifest.context_map_id,
                    "schema_version": manifest.schema_version,
                    "metadata": self._document(MAP_METADATA),
                    "geometry_ref": self._document(GEOMETRY_REFERENCE),
                }
            )
        except (ContextMapRecordError, ValueError) as error:
            self._error(
                "map.invalid",
                f"the map record is not valid under the schema: {error}",
                MAP_METADATA,
            )

    def _check_capabilities(self) -> None:
        manifest = self._manifest_required()
        if self._context_map is None:
            raise _Skip("the map record could not be read")
        declared = self._context_map.metadata.capabilities.content
        for capability, count, code in (
            (MapCapability.ENTITIES, manifest.entity_count, "capabilities.undeclared_entities"),
            (MapCapability.RELATIONS, manifest.relation_count, "capabilities.undeclared_relations"),
        ):
            if count and capability not in declared:
                self._error(
                    code,
                    f"the artifact holds {count} {capability.value} but the map does not declare "
                    f"the {capability.value} capability",
                    MAP_METADATA,
                )

    def _check_lineage(self) -> None:
        manifest = self._manifest_required()
        if LINEAGE not in self._usable:
            raise _Skip(f"{LINEAGE} is not intact")
        try:
            recorded = self._document(LINEAGE)["upstream_artifacts"]
            found = sorted(
                (
                    item["artifact_type"],
                    item["artifact_id"],
                    item["content_identity"],
                    item["requirement"],
                )
                for item in recorded
            )
        except (KeyError, TypeError, ValueError) as error:
            self._error("lineage.malformed", f"{LINEAGE} is not a valid lineage ({error})", LINEAGE)
            return
        expected = sorted(
            (item.artifact_type, item.artifact_id, item.content_identity, item.requirement.value)
            for item in manifest.dependencies
        )
        if found != expected:
            self._error(
                "lineage.mismatch",
                f"the upstream artifacts of {LINEAGE} are not the dependencies of the manifest",
                LINEAGE,
            )

    def _check_dependencies(self) -> None:
        manifest = self._manifest_required()
        for record in manifest.dependencies:
            resolution = resolve_dependency(
                record, artifact_root=self._root, dependency_paths=self._dependency_paths
            )
            self._resolutions.append(resolution)
            subject = record.artifact_id
            if resolution.status is DependencyStatus.MISMATCH:
                self._error("dependency.mismatch", resolution.detail, subject)
            elif resolution.status is DependencyStatus.MISSING:
                if record.requirement is Requirement.REQUIRED:
                    self._error("dependency.required_missing", resolution.detail, subject)
                else:
                    self._warning("dependency.optional_missing", resolution.detail, subject)

    def _geometry_record(self) -> DependencyRecord | None:
        manifest = self._manifest_required()
        return next(
            (
                item
                for item in manifest.dependencies
                if item.artifact_type == GEOMETRIC_MAP_ARTIFACT_TYPE
            ),
            None,
        )

    def _check_geometry(self) -> None:
        if self._context_map is None:
            raise _Skip("the map record could not be read")
        link = self._context_map.geometry_ref
        record = self._geometry_record()
        if record is None:
            self._error("geometry.dependency_missing", "the manifest records no geometric map")
            return
        if record.artifact_id != str(link.map_id):
            self._error(
                "geometry.reference_mismatch",
                f"the map refers to the geometric map {link.map_id!r} but the manifest records "
                f"{record.artifact_id!r}",
                record.artifact_id,
            )
            return
        resolution = next((item for item in self._resolutions if item.record == record), None)
        if resolution is None or resolution.status is not DependencyStatus.FOUND:
            return
        assert resolution.location is not None
        try:
            with GeometricMapArtifactReader(resolution.location) as reader:
                upstream = reader.manifest
        except MapArtifactError as error:
            self._error(
                "geometry.unreadable", f"the geometric map cannot be opened: {error}", link.map_id
            )
            return
        if upstream.map_id != link.map_id:
            self._error(
                "geometry.map_id_mismatch",
                f"the geometric map is {upstream.map_id!r}, the map refers to {link.map_id!r}",
                link.map_id,
            )
        if upstream.point_count != link.point_count:
            self._error(
                "geometry.point_count_mismatch",
                f"the map declares {link.point_count} geometry elements but the geometric map "
                f"has {upstream.point_count}",
                link.map_id,
            )
        frame = self._context_map.metadata.frame.frame_id
        if upstream.map_frame != frame:
            self._error(
                "geometry.frame_mismatch",
                f"the map is expressed in frame {frame!r} but the geometric map is in "
                f"{upstream.map_frame!r}",
                link.map_id,
            )

    def _check_references(self) -> None:
        manifest = self._manifest_required()
        needed = {ENTITIES, ENTITY_INDEX, RELATIONS, RELATION_INDEX}
        if not needed <= self._usable:
            raise _Skip("the record tables are not intact")
        try:
            entities = RecordTable(
                self._root / ENTITIES,
                self._root / ENTITY_INDEX,
                record_count=manifest.entity_count,
            )
            relations = RecordTable(
                self._root / RELATIONS,
                self._root / RELATION_INDEX,
                record_count=manifest.relation_count,
            )
            keys: set[str] = set()
            for line in entities.iter_lines():
                if line.keys() != {"key", "record"}:
                    raise RecordTableError(f"the entity {line['key']!r} has the wrong fields")
                keys.add(line["key"])
            endpoints: list[tuple[str, str, str]] = []
            for line in relations.iter_lines():
                if line.keys() != {"key", "subject", "object", "record"}:
                    raise RecordTableError(f"the relation {line['key']!r} has the wrong fields")
                endpoints.append((line["key"], line["subject"], line["object"]))
        except ContextMapArtifactError as error:
            self._error("index.broken", str(error), ENTITY_INDEX)
            return
        self._entity_keys, self._endpoints = keys, endpoints
        for relation_key, subject, obj in endpoints:
            for role, key in (("subject", subject), ("object", obj)):
                if key not in keys:
                    self._endpoint_errors = True
                    self._error(
                        "reference.relation_endpoint_missing",
                        f"the relation {relation_key!r} has the {role} {key!r}, which is not an "
                        "entity of the map",
                        relation_key,
                    )

    def _check_index_rebuild(self) -> None:
        for payload_path, index_path in ((ENTITIES, ENTITY_INDEX), (RELATIONS, RELATION_INDEX)):
            if not {payload_path, index_path} <= self._usable:
                continue
            try:
                rebuilt = rebuild_index(self._read(payload_path))
            except RecordTableError as error:
                self._error("records.invalid", f"{payload_path}: {error}", payload_path)
                continue
            if rebuilt != self._read(index_path):
                self._error(
                    "index.mismatch",
                    f"{index_path} is not the index that {payload_path} produces",
                    index_path,
                )
        if (
            ENTITY_RELATION_INDEX in self._usable
            and self._entity_keys is not None
            and self._endpoints is not None
            and not self._endpoint_errors
        ):
            rebuilt = encode_entity_relation_index(self._entity_keys, self._endpoints)
            if rebuilt != self._read(ENTITY_RELATION_INDEX):
                self._error(
                    "index.mismatch",
                    f"{ENTITY_RELATION_INDEX} is not the index that the tables produce",
                    ENTITY_RELATION_INDEX,
                )

    def _check_upstream_files(self) -> None:
        for resolution in self._resolutions:
            if resolution.status is not DependencyStatus.FOUND or resolution.location is None:
                continue
            try:
                problems = check_file_inventory(
                    resolution.location, read_inventory(resolution.location)
                )
            except UpstreamArtifactError as error:
                problems = [str(error)]
            if problems:
                self._error(
                    "dependency.upstream_damaged",
                    f"the {resolution.record.artifact_type} {resolution.record.artifact_id!r} "
                    f"does not match its own inventory: {'; '.join(problems)}",
                    resolution.record.artifact_id,
                )

    def _check_unlisted(self) -> None:
        manifest = self._manifest_required()
        listed = {entry.path for entry in manifest.file_inventory} | {MANIFEST, README}
        for path in sorted(self._root.rglob("*")):
            relative = path.relative_to(self._root).as_posix()
            if relative == DEBUG_DIRECTORY or relative.startswith(f"{DEBUG_DIRECTORY}/"):
                if relative == DEBUG_DIRECTORY:
                    self._warning(
                        "debug.present",
                        "a debug directory is present; it is not part of the artifact and is "
                        "never read",
                        relative,
                    )
                continue
            if path.is_file() and relative not in listed:
                self._warning(
                    "file.unlisted",
                    f"{relative!r} is not in the inventory: it is not part of the artifact "
                    "and is never read",
                    relative,
                )


def _checked(entry: FileEntry, status: FileStatus) -> CheckedFile:
    return CheckedFile(
        path=entry.path,
        size_bytes=entry.size_bytes,
        content_hash=entry.content_hash,
        status=status,
    )


def _with_status(checked: CheckedFile, status: FileStatus) -> CheckedFile:
    return CheckedFile(
        path=checked.path,
        size_bytes=checked.size_bytes,
        content_hash=checked.content_hash,
        status=status,
    )
