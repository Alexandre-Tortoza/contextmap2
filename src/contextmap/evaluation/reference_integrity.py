"""Reference-set integrity validation and leakage prevention.

A research evaluation is invalid if tuning and evaluation samples overlap, if
related frames of one physical sequence land on both sides of a split, or if an
annotation quietly carries model output. :func:`validate_reference_set` checks a
manifest (and, when given the reference-set directory, its annotation files)
and reports every finding as a *blocker* or a *warning*, together with the
reference-set version and digest. :func:`require_valid_reference_set` and
:func:`open_validated_reference_set` are the entry points evaluation tooling
uses: they refuse a reference set that has blockers. See
``src/contextmap/evaluation/docs/reference-integrity.md``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import file_digest
from contextmap.evaluation.annotations import (
    AnnotationError,
    AnnotationFamily,
    GeometryAnnotationSet,
    read_annotation_set,
)
from contextmap.evaluation.reference_set import (
    AnnotationFileEntry,
    ProvenanceOrigin,
    ReferenceSample,
    ReferenceSampleId,
    ReferenceSetError,
    ReferenceSetIdentity,
    ReferenceSetManifest,
    ReferenceTrust,
    SplitRole,
    SplitScheme,
    SplitUnit,
    read_reference_set,
)


class IntegritySeverity(Enum):
    """Whether a finding makes a reference set unusable or only deserves attention."""

    BLOCKER = "blocker"
    WARNING = "warning"


@dataclass(frozen=True, kw_only=True)
class IntegrityFinding:
    """One machine-readable integrity finding.

    Attributes:
        severity: A blocker refuses the reference set; a warning does not.
        code: Stable kebab-case identifier of the check that raised it.
        message: What is wrong, in a sentence.
        subjects: The identities involved (samples, annotations, split keys).
    """

    severity: IntegritySeverity
    code: str
    message: str
    subjects: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "subjects": list(self.subjects),
        }


@dataclass(frozen=True, kw_only=True)
class ProvenanceAuditEntry:
    """How one annotation file came to be, read straight from the manifest.

    ``independent_of_model_output`` is true only when the annotation neither
    comes from model inference nor was seeded by a model-run artifact.
    """

    annotation_id: str
    trust: ReferenceTrust
    origin: ProvenanceOrigin
    annotator: str
    method: str
    seeded_from_artifacts: tuple[str, ...]
    reviewed: bool
    independent_of_model_output: bool

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "annotation_id": self.annotation_id,
            "trust": self.trust.value,
            "origin": self.origin.value,
            "annotator": self.annotator,
            "method": self.method,
            "seeded_from_artifacts": list(self.seeded_from_artifacts),
            "reviewed": self.reviewed,
            "independent_of_model_output": self.independent_of_model_output,
        }


@dataclass(frozen=True, kw_only=True)
class ReferenceSetIntegrityReport:
    """The outcome of validating a reference set.

    Attributes:
        reference_set: Identity (id, version, digest) of the validated manifest.
        files_checked: Whether annotation files were read and hashed as well.
        findings: Every blocker and warning, in a deterministic order.
        provenance_audit: One entry per annotation file.
    """

    reference_set: ReferenceSetIdentity
    files_checked: bool
    findings: tuple[IntegrityFinding, ...]
    provenance_audit: tuple[ProvenanceAuditEntry, ...]

    @property
    def blockers(self) -> tuple[IntegrityFinding, ...]:
        """Return the findings that refuse the reference set."""
        return tuple(item for item in self.findings if item.severity is IntegritySeverity.BLOCKER)

    @property
    def warnings(self) -> tuple[IntegrityFinding, ...]:
        """Return the findings that only deserve attention."""
        return tuple(item for item in self.findings if item.severity is IntegritySeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        """Return whether there is no blocker."""
        return not self.blockers

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "reference_set": self.reference_set.to_record(),
            "files_checked": self.files_checked,
            "valid": self.is_valid,
            "blockers": [item.to_record() for item in self.blockers],
            "warnings": [item.to_record() for item in self.warnings],
            "provenance_audit": [item.to_record() for item in self.provenance_audit],
        }


class ReferenceSetIntegrityError(ReferenceSetError):
    """Raised when evaluation tooling is given a reference set with blockers."""

    def __init__(self, report: ReferenceSetIntegrityReport) -> None:
        """Keep the report and name every blocker in the message."""
        self.report = report
        identity = report.reference_set
        listing = "; ".join(f"{item.code}: {item.message}" for item in report.blockers)
        super().__init__(
            f"reference set {identity.reference_set_id!r} version {identity.version!r} "
            f"is invalid: {listing}"
        )


@dataclass(frozen=True, kw_only=True)
class ValidatedReferenceSet:
    """A reference set that passed validation, files included.

    It cannot hold a failing report, a report of another reference set, or a
    report that did not check the annotation files, so evaluation tooling that
    accepts it never sees an invalid reference set.
    """

    manifest: ReferenceSetManifest
    report: ReferenceSetIntegrityReport
    root: Path

    def __post_init__(self) -> None:
        """Refuse a report that does not prove this manifest valid."""
        if not self.report.is_valid:
            raise ReferenceSetIntegrityError(self.report)
        if self.report.reference_set != self.manifest.identity():
            raise ReferenceSetError("the integrity report belongs to another reference set")
        if not self.report.files_checked:
            raise ReferenceSetError("the integrity report did not check the annotation files")


def validate_reference_set(
    manifest: ReferenceSetManifest, root: Path | None = None
) -> ReferenceSetIntegrityReport:
    """Validate a manifest and, given its directory, its annotation files.

    Never raises for an invalid reference set: every problem becomes a finding.
    """
    samples = {item.sample_id: item for item in manifest.samples}
    findings: list[IntegrityFinding] = []
    findings += _check_samples(manifest)
    findings += _check_split_schemes(manifest, samples)
    findings += _check_annotations(manifest)
    if root is not None:
        findings += _check_annotation_files(manifest, samples, root)
    return ReferenceSetIntegrityReport(
        reference_set=manifest.identity(),
        files_checked=root is not None,
        findings=tuple(findings),
        provenance_audit=_provenance_audit(manifest),
    )


def require_valid_reference_set(
    manifest: ReferenceSetManifest, root: Path
) -> ValidatedReferenceSet:
    """Return the validated reference set, refusing one with blockers.

    Raises:
        ReferenceSetIntegrityError: If validation found any blocker.
    """
    report = validate_reference_set(manifest, root)
    if not report.is_valid:
        raise ReferenceSetIntegrityError(report)
    return ValidatedReferenceSet(manifest=manifest, report=report, root=root)


def open_validated_reference_set(root: Path) -> ValidatedReferenceSet:
    """Read the manifest under ``root``, verify its digest and validate it fully.

    Raises:
        ReferenceSetError: If the manifest cannot be trusted.
        ReferenceSetIntegrityError: If validation found any blocker.
    """
    return require_valid_reference_set(read_reference_set(root), root)


def _blocker(code: str, message: str, *subjects: str) -> IntegrityFinding:
    return IntegrityFinding(
        severity=IntegritySeverity.BLOCKER, code=code, message=message, subjects=subjects
    )


def _warning(code: str, message: str, *subjects: str) -> IntegrityFinding:
    return IntegrityFinding(
        severity=IntegritySeverity.WARNING, code=code, message=message, subjects=subjects
    )


def _group_keys(sample: ReferenceSample, unit: SplitUnit) -> tuple[str, ...]:
    if unit is SplitUnit.SEQUENCE:
        return (sample.source_id,)
    return tuple(item.key for item in sample.groups if item.unit is unit)


def _check_samples(manifest: ReferenceSetManifest) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []
    calibration_source = {item.calibration_id: item.source_id for item in manifest.calibrations}

    by_hash: dict[str, list[str]] = defaultdict(list)
    by_observation: dict[str, list[str]] = defaultdict(list)
    for sample in manifest.samples:
        by_hash[sample.content_hash].append(sample.sample_id)
        for observation_id in sample.observation_ids:
            by_observation[observation_id].append(sample.sample_id)
        for calibration_id in sample.calibration_ids:
            if calibration_source[calibration_id] != sample.source_id:
                findings.append(
                    _blocker(
                        "calibration-source-mismatch",
                        f"sample {sample.sample_id!r} uses calibration {calibration_id!r}, which "
                        f"belongs to source {calibration_source[calibration_id]!r} and not to "
                        f"{sample.source_id!r}",
                        sample.sample_id,
                        calibration_id,
                    )
                )
    for content_hash, ids in by_hash.items():
        if len(ids) > 1:
            findings.append(
                _blocker(
                    "duplicate-sample-content",
                    f"samples share the content hash {content_hash}: they are the same content",
                    *ids,
                )
            )
    for shared, owners in by_observation.items():
        if len(owners) > 1:
            findings.append(
                _warning(
                    "shared-observation",
                    f"physical observation {shared!r} belongs to more than one sample",
                    shared,
                    *owners,
                )
            )

    split_members = {
        sample_id
        for scheme in manifest.split_schemes
        for split in scheme.splits
        for sample_id in split.sample_ids
    }
    annotated = {sample_id for entry in manifest.annotations for sample_id in entry.sample_ids}
    for sample in manifest.samples:
        if manifest.split_schemes and sample.sample_id not in split_members:
            findings.append(
                _warning(
                    "sample-not-in-any-split",
                    f"sample {sample.sample_id!r} is in no split of any scheme",
                    sample.sample_id,
                )
            )
        if sample.sample_id not in annotated:
            findings.append(
                _warning(
                    "sample-without-annotation",
                    f"sample {sample.sample_id!r} is not covered by any annotation file",
                    sample.sample_id,
                )
            )
    return findings


def _check_split_schemes(
    manifest: ReferenceSetManifest, samples: dict[ReferenceSampleId, ReferenceSample]
) -> list[IntegrityFinding]:
    if not manifest.split_schemes:
        return [
            _blocker(
                "no-split-scheme",
                "the reference set declares no split scheme, so tuning and evaluation "
                "samples are not separated",
            )
        ]
    findings: list[IntegrityFinding] = []
    tasks: dict[str, list[str]] = defaultdict(list)
    for scheme in manifest.split_schemes:
        tasks[scheme.task].append(scheme.scheme_id)
        findings += _check_scheme(scheme, samples)
    for task, scheme_ids in tasks.items():
        if len(scheme_ids) > 1:
            findings.append(
                _warning(
                    "multiple-schemes-for-task",
                    f"task {task!r} has more than one split scheme; results are not comparable "
                    "across them",
                    *scheme_ids,
                )
            )
    return findings


def _check_scheme(
    scheme: SplitScheme, samples: dict[ReferenceSampleId, ReferenceSample]
) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []
    if not scheme.splits:
        return [
            _blocker("empty-scheme", f"scheme {scheme.scheme_id!r} has no split", scheme.scheme_id)
        ]
    if not any(split.role is SplitRole.TEST for split in scheme.splits):
        findings.append(
            _warning(
                "no-test-split",
                f"scheme {scheme.scheme_id!r} has no held-out test split",
                scheme.scheme_id,
            )
        )

    split_of: dict[ReferenceSampleId, list[str]] = defaultdict(list)
    splits_by_key: dict[str, set[str]] = defaultdict(set)
    splits_by_source: dict[str, set[str]] = defaultdict(set)
    splits_by_observation: dict[str, set[str]] = defaultdict(set)
    for split in scheme.splits:
        if not split.sample_ids:
            findings.append(
                _warning(
                    "empty-split",
                    f"split {split.name!r} of scheme {scheme.scheme_id!r} selects no sample",
                    scheme.scheme_id,
                    split.name,
                )
            )
        for sample_id in split.sample_ids:
            sample = samples[sample_id]
            split_of[sample_id].append(split.name)
            splits_by_source[sample.source_id].add(split.name)
            for observation_id in sample.observation_ids:
                splits_by_observation[observation_id].add(split.name)
            keys = _group_keys(sample, scheme.unit)
            if not keys:
                findings.append(
                    _blocker(
                        "missing-group-key",
                        f"sample {sample_id!r} has no {scheme.unit.value} group key, so scheme "
                        f"{scheme.scheme_id!r} cannot keep its {scheme.unit.value} units together",
                        sample_id,
                        scheme.scheme_id,
                    )
                )
            for key in keys:
                splits_by_key[key].add(split.name)

    for sample_id, split_names in split_of.items():
        if len(split_names) > 1:
            findings.append(
                _blocker(
                    "split-overlap",
                    f"sample {sample_id!r} is in more than one split of scheme "
                    f"{scheme.scheme_id!r}: {sorted(split_names)}",
                    sample_id,
                    scheme.scheme_id,
                )
            )
    for key, key_splits in splits_by_key.items():
        if len(key_splits) > 1:
            findings.append(
                _blocker(
                    "split-leakage",
                    f"{scheme.unit.value} {key!r} straddles splits {sorted(key_splits)} of scheme "
                    f"{scheme.scheme_id!r}",
                    f"{scheme.unit.value}:{key}",
                    scheme.scheme_id,
                )
            )
    for shared, shared_splits in splits_by_observation.items():
        if len(shared_splits) > 1:
            findings.append(
                _blocker(
                    "split-observation-overlap",
                    f"physical observation {shared!r} is in splits {sorted(shared_splits)} of "
                    f"scheme {scheme.scheme_id!r}",
                    shared,
                    scheme.scheme_id,
                )
            )
    if scheme.unit is not SplitUnit.SEQUENCE:
        for source_id, source_splits in splits_by_source.items():
            if len(source_splits) > 1:
                findings.append(
                    _warning(
                        "sequence-shared-across-splits",
                        f"sequence {source_id!r} feeds splits {sorted(source_splits)} of scheme "
                        f"{scheme.scheme_id!r}, whose unit is {scheme.unit.value}",
                        source_id,
                        scheme.scheme_id,
                    )
                )
    findings += _check_adjacency(scheme, samples)
    return findings


def _check_adjacency(
    scheme: SplitScheme, samples: dict[ReferenceSampleId, ReferenceSample]
) -> list[IntegrityFinding]:
    window = scheme.adjacency_window_ns
    if window is None:
        return []
    findings: list[IntegrityFinding] = []
    timeline: dict[tuple[str, str], list[tuple[int, int, str, str]]] = defaultdict(list)
    for split in scheme.splits:
        for sample_id in split.sample_ids:
            sample = samples[sample_id]
            span = sample.time_span
            if span is None:
                findings.append(
                    _blocker(
                        "missing-time-span",
                        f"sample {sample_id!r} has no time span, so its adjacency to other "
                        f"samples cannot be checked for scheme {scheme.scheme_id!r}",
                        sample_id,
                        scheme.scheme_id,
                    )
                )
                continue
            timeline[(sample.source_id, span.start.clock_id)].append(
                (
                    span.start.total_nanoseconds(),
                    span.end.total_nanoseconds(),
                    sample_id,
                    split.name,
                )
            )
    for entries in timeline.values():
        entries.sort()
        for index, (_, end, first, first_split) in enumerate(entries):
            for start, _, second, second_split in entries[index + 1 :]:
                if start > end + window:
                    break
                if first_split != second_split:
                    findings.append(
                        _blocker(
                            "adjacent-split-leakage",
                            f"samples {first!r} ({first_split}) and {second!r} ({second_split}) "
                            f"are within {window} ns of each other but in different splits of "
                            f"scheme {scheme.scheme_id!r}",
                            first,
                            second,
                            scheme.scheme_id,
                        )
                    )
    return findings


def _check_annotations(manifest: ReferenceSetManifest) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []
    provenance = {item.provenance_id: item for item in manifest.provenance}
    used = {entry.provenance_id for entry in manifest.annotations}
    by_path: dict[str, list[str]] = defaultdict(list)
    by_hash: dict[str, list[str]] = defaultdict(list)

    for entry in manifest.annotations:
        by_path[entry.path].append(entry.annotation_id)
        by_hash[entry.content_hash].append(entry.annotation_id)
        try:
            AnnotationFamily.from_schema(entry.schema)
        except AnnotationError:
            findings.append(
                _blocker(
                    "unknown-annotation-schema",
                    f"annotation {entry.annotation_id!r} declares the unsupported schema "
                    f"{entry.schema!r}",
                    entry.annotation_id,
                )
            )
        if not entry.sample_ids:
            findings.append(
                _warning(
                    "annotation-without-samples",
                    f"annotation {entry.annotation_id!r} covers no sample",
                    entry.annotation_id,
                )
            )
        record = provenance[entry.provenance_id]
        if record.seeded_from_artifacts and entry.trust is not ReferenceTrust.DIAGNOSTIC_ONLY:
            if record.review is None:
                findings.append(
                    _blocker(
                        "unreviewed-model-seeding",
                        f"annotation {entry.annotation_id!r} was seeded by model-run artifacts "
                        f"{list(record.seeded_from_artifacts)} and has no explicit review, so it "
                        f"cannot be {entry.trust.value}",
                        entry.annotation_id,
                        *record.seeded_from_artifacts,
                    )
                )
            else:
                findings.append(
                    _warning(
                        "model-seeded-reviewed",
                        f"annotation {entry.annotation_id!r} was seeded by model-run artifacts "
                        f"and reviewed by {record.review.reviewer!r}",
                        entry.annotation_id,
                    )
                )
        if entry.trust is ReferenceTrust.DIAGNOSTIC_ONLY:
            findings.append(
                _warning(
                    "diagnostic-annotation",
                    f"annotation {entry.annotation_id!r} is diagnostic only and cannot back "
                    "acceptance metrics",
                    entry.annotation_id,
                )
            )

    for path, ids in by_path.items():
        if len(ids) > 1:
            findings.append(
                _blocker(
                    "duplicate-annotation-path",
                    f"annotations {ids} point at the same file {path!r}",
                    *ids,
                )
            )
    for content_hash, ids in by_hash.items():
        if len(ids) > 1:
            findings.append(
                _warning(
                    "duplicate-annotation-content",
                    f"annotations {ids} declare the same content hash {content_hash}",
                    *ids,
                )
            )
    for provenance_id in provenance:
        if provenance_id not in used:
            findings.append(
                _warning(
                    "unused-provenance",
                    f"provenance {provenance_id!r} is used by no annotation",
                    provenance_id,
                )
            )
    return findings


def _check_annotation_files(
    manifest: ReferenceSetManifest,
    samples: dict[ReferenceSampleId, ReferenceSample],
    root: Path,
) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []
    resolved_root = root.resolve()
    for entry in manifest.annotations:
        path = (root / entry.path).resolve()
        if not path.is_relative_to(resolved_root):
            findings.append(
                _blocker(
                    "annotation-file-outside-root",
                    f"annotation file {entry.path!r} resolves outside the reference-set root",
                    entry.annotation_id,
                )
            )
        elif not path.is_file():
            findings.append(
                _blocker(
                    "annotation-file-missing",
                    f"annotation file {entry.path!r} does not exist",
                    entry.annotation_id,
                )
            )
        elif file_digest(path) != entry.content_hash:
            findings.append(
                _blocker(
                    "annotation-file-hash-mismatch",
                    f"annotation file {entry.path!r} does not match its declared content hash",
                    entry.annotation_id,
                )
            )
        else:
            findings += _check_annotation_content(entry, samples, path)
    return findings


def _check_annotation_content(
    entry: AnnotationFileEntry,
    samples: dict[ReferenceSampleId, ReferenceSample],
    path: Path,
) -> list[IntegrityFinding]:
    try:
        annotation_set = read_annotation_set(path)
    except AnnotationError as error:
        return [
            _blocker(
                "annotation-unreadable",
                f"annotation file {entry.path!r} cannot be read: {error}",
                entry.annotation_id,
            )
        ]
    findings: list[IntegrityFinding] = []
    if annotation_set.family.schema != entry.schema:
        findings.append(
            _blocker(
                "annotation-schema-mismatch",
                f"annotation file {entry.path!r} follows {annotation_set.family.schema!r} but "
                f"the manifest declares {entry.schema!r}",
                entry.annotation_id,
            )
        )
    declared = set(entry.sample_ids)
    for reference in annotation_set.observation_references():
        sample = samples.get(reference.sample_id)
        if sample is None:
            findings.append(
                _blocker(
                    "annotation-unknown-sample",
                    f"annotation {entry.annotation_id!r} refers to sample "
                    f"{reference.sample_id!r}, which is not in the manifest",
                    entry.annotation_id,
                    reference.sample_id,
                )
            )
            continue
        if reference.sample_id not in declared:
            findings.append(
                _blocker(
                    "annotation-sample-not-declared",
                    f"annotation {entry.annotation_id!r} annotates sample "
                    f"{reference.sample_id!r}, which the manifest entry does not declare",
                    entry.annotation_id,
                    reference.sample_id,
                )
            )
        if (
            reference.observation_id is not None
            and reference.observation_id not in sample.observation_ids
        ):
            findings.append(
                _blocker(
                    "annotation-unknown-observation",
                    f"annotation {entry.annotation_id!r} refers to observation "
                    f"{reference.observation_id!r}, which is not part of sample "
                    f"{reference.sample_id!r}",
                    entry.annotation_id,
                    reference.sample_id,
                    reference.observation_id,
                )
            )
    if isinstance(annotation_set, GeometryAnnotationSet):
        findings += _check_geometry_calibration(entry, annotation_set, samples)
    return findings


def _check_geometry_calibration(
    entry: AnnotationFileEntry,
    annotation_set: GeometryAnnotationSet,
    samples: dict[ReferenceSampleId, ReferenceSample],
) -> Iterable[IntegrityFinding]:
    for correspondence in annotation_set.correspondences:
        sample = samples.get(correspondence.sample_id)
        if sample is not None and correspondence.calibration_id not in sample.calibration_ids:
            yield _blocker(
                "annotation-calibration-mismatch",
                f"correspondence {correspondence.correspondence_id!r} uses calibration "
                f"{correspondence.calibration_id!r}, which is not a calibration of sample "
                f"{correspondence.sample_id!r}",
                entry.annotation_id,
                correspondence.correspondence_id,
            )


def _provenance_audit(manifest: ReferenceSetManifest) -> tuple[ProvenanceAuditEntry, ...]:
    provenance = {item.provenance_id: item for item in manifest.provenance}
    entries = []
    for entry in manifest.annotations:
        record = provenance[entry.provenance_id]
        entries.append(
            ProvenanceAuditEntry(
                annotation_id=entry.annotation_id,
                trust=entry.trust,
                origin=record.origin,
                annotator=record.annotator,
                method=record.method,
                seeded_from_artifacts=record.seeded_from_artifacts,
                reviewed=record.review is not None,
                independent_of_model_output=(
                    record.origin is not ProvenanceOrigin.MODEL_INFERENCE
                    and not record.seeded_from_artifacts
                ),
            )
        )
    return tuple(entries)
