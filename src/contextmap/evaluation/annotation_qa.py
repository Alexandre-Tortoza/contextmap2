"""Quality assurance of reference annotations.

Reference integrity (:mod:`contextmap.evaluation.reference_integrity`) asks
whether the *manifest* can be trusted. This module asks whether the *content*
of the annotation files can: a mask that does not fit its image, a relation
between identities nobody declared, an inverse predicate that contradicts its
inverse, a correspondence outside the frame. Bad annotations invalidate every
result computed against them, so QA runs before an evaluation is official.

The report keeps three things apart:

* **blockers**: the reference set cannot be used until they are fixed;
* **warnings**: suspicious, worth a look, not disqualifying;
* **permissible observations and disagreements**: ambiguity, unknown, partial
  coverage and differences *between annotators*. These are information about the
  data, not defects, and nothing here resolves them: no annotator is silently
  chosen, and each disagreement names the annotation files that hold each side.

See ``src/contextmap/evaluation/docs/annotation-qa.md``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import file_digest
from contextmap.evaluation.annotations import (
    AnnotationError,
    AnnotationFamily,
    AnnotationSet,
    Coverage,
    FrameRegionAnnotation,
    GeometryAnnotationSet,
    IdentityAnnotationSet,
    LabelNormalization,
    RegionAnnotation,
    RegionAnnotationSet,
    RelationAnnotationSet,
    RelationStatus,
    SemanticAnnotationRecord,
    SemanticAnnotationSet,
    SemanticStatus,
    VisibilityAnnotationSet,
    VisibilityLevel,
    read_annotation_set,
)
from contextmap.evaluation.reference_integrity import (
    IntegrityFinding,
    IntegritySeverity,
    ReferenceSetIntegrityReport,
    ValidatedReferenceSet,
    open_validated_reference_set,
    validate_reference_set,
)
from contextmap.evaluation.reference_set import (
    AnnotationFileEntry,
    ReferenceSetError,
    ReferenceSetIdentity,
    ReferenceSetManifest,
)
from contextmap.visual_perception import InlineMask

QA_REPORT_SCHEMA = "contextmap.annotation-qa/v1"
"""Schema identifier of the annotation QA report."""

Cell = tuple[int, int, int, int]
Target = tuple[str, ...]


class AnnotationQaError(ReferenceSetError):
    """Raised when a reference set is refused because its annotations have blockers."""

    def __init__(self, report: AnnotationQaReport) -> None:
        """Keep the report and name every QA blocker in the message."""
        self.report = report
        listing = "; ".join(f"{item.code}: {item.message}" for item in report.blockers)
        super().__init__(f"the reference annotations are inconsistent: {listing}")


@dataclass(frozen=True, kw_only=True)
class AnnotationQaPolicy:
    """The explicit, versioned tolerances QA applies; there are no hidden defaults.

    Attributes:
        policy_id: Version of the policy, cited by the report.
        duplicate_pixel_tolerance_px: Two correspondences of one image closer than
            this many pixels annotate the same pixel.
        duplicate_point_tolerance_m: Two 3D points closer than this many metres are the
            same point; the same pixel mapped further apart is a conflict.
        region_match_iou: Regions of two annotators match at or above this IoU.
    """

    policy_id: str
    duplicate_pixel_tolerance_px: float
    duplicate_point_tolerance_m: float
    region_match_iou: float

    def __post_init__(self) -> None:
        """Require an identity and meaningful tolerances."""
        if not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        for name, value in (
            ("duplicate_pixel_tolerance_px", self.duplicate_pixel_tolerance_px),
            ("duplicate_point_tolerance_m", self.duplicate_point_tolerance_m),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} tolerance must be a positive finite number")
        if not 0.0 < self.region_match_iou <= 1.0:
            raise ValueError("region_match_iou must be an IoU in (0, 1]")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "policy_id": self.policy_id,
            "duplicate_pixel_tolerance_px": self.duplicate_pixel_tolerance_px,
            "duplicate_point_tolerance_m": self.duplicate_point_tolerance_m,
            "region_match_iou": self.region_match_iou,
        }


@dataclass(frozen=True, kw_only=True)
class PermissibleObservation:
    """Ambiguity, unknown or partial coverage: information about the data, not a defect."""

    kind: str
    family: AnnotationFamily
    annotation_id: str
    subject: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "kind": self.kind,
            "family": self.family.value,
            "annotation_id": self.annotation_id,
            "subject": self.subject,
        }


@dataclass(frozen=True, kw_only=True)
class DisagreementRecord:
    """One target on which two annotators differ, with what each side says.

    ``positions`` lists ``(annotation_id, description)`` for every side; nothing
    marks one of them as right.
    """

    subject: str
    kind: str
    positions: tuple[tuple[str, str], ...]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "subject": self.subject,
            "kind": self.kind,
            "positions": [list(item) for item in self.positions],
        }


@dataclass(frozen=True, kw_only=True)
class DisagreementSummary:
    """The comparison of two annotators' files of one family.

    Attributes:
        family: The annotation family compared.
        annotation_ids: The two annotation files, in manifest order.
        annotators: Their annotators, in the same order.
        compared: Targets annotated by both.
        agreements: Compared targets on which they agree.
        disagreements: The targets on which they differ.
    """

    family: AnnotationFamily
    annotation_ids: tuple[str, str]
    annotators: tuple[str, str]
    compared: int
    agreements: int
    disagreements: tuple[DisagreementRecord, ...]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "family": self.family.value,
            "annotation_ids": list(self.annotation_ids),
            "annotators": list(self.annotators),
            "compared": self.compared,
            "agreements": self.agreements,
            "disagreements": [item.to_record() for item in self.disagreements],
        }


@dataclass(frozen=True, kw_only=True)
class AnnotationQaReport:
    """The outcome of checking a reference set's annotation content.

    Attributes:
        reference_set: Identity (id, version, digest) of the checked manifest.
        policy: The tolerances applied.
        integrity: The reference-set integrity report, including the split checks.
        findings: QA blockers and warnings, in a deterministic order.
        permissible: Ambiguity, unknown and partial-coverage observations.
        disagreements: One summary per pair of annotators and family.
        annotation_files_checked: Annotation files that could be read and checked.
    """

    reference_set: ReferenceSetIdentity
    policy: AnnotationQaPolicy
    integrity: ReferenceSetIntegrityReport
    findings: tuple[IntegrityFinding, ...]
    permissible: tuple[PermissibleObservation, ...]
    disagreements: tuple[DisagreementSummary, ...]
    annotation_files_checked: int

    @property
    def blockers(self) -> tuple[IntegrityFinding, ...]:
        """Return the QA findings that refuse the reference set."""
        return tuple(item for item in self.findings if item.severity is IntegritySeverity.BLOCKER)

    @property
    def warnings(self) -> tuple[IntegrityFinding, ...]:
        """Return the QA findings that only deserve attention."""
        return tuple(item for item in self.findings if item.severity is IntegritySeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        """Return whether integrity and QA are both free of blockers."""
        return self.integrity.is_valid and not self.blockers

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "schema": QA_REPORT_SCHEMA,
            "reference_set": self.reference_set.to_record(),
            "policy": self.policy.to_record(),
            "valid": self.is_valid,
            "annotation_files_checked": self.annotation_files_checked,
            "integrity": self.integrity.to_record(),
            "blockers": [item.to_record() for item in self.blockers],
            "warnings": [item.to_record() for item in self.warnings],
            "permissible": [item.to_record() for item in self.permissible],
            "disagreements": [item.to_record() for item in self.disagreements],
        }


@dataclass(frozen=True, kw_only=True)
class CertifiedReferenceSet:
    """A reference set that passed integrity validation and annotation QA.

    Official Solution 1 evaluation runs use ``validated`` (see
    :func:`~contextmap.evaluation.experiment_runner.run_experiment`).
    """

    validated: ValidatedReferenceSet
    qa: AnnotationQaReport
    policy: AnnotationQaPolicy

    def __post_init__(self) -> None:
        """Refuse a QA report that does not prove this reference set clean."""
        if not self.qa.is_valid:
            raise AnnotationQaError(self.qa)
        if self.qa.reference_set != self.validated.manifest.identity():
            raise ReferenceSetError("the QA report belongs to another reference set")


# ------------------------------------------------------------------------------ loading


@dataclass(frozen=True)
class _Loaded:
    entry: AnnotationFileEntry
    annotator: str
    annotation_set: AnnotationSet

    @property
    def annotation_id(self) -> str:
        return self.entry.annotation_id


def _load(manifest: ReferenceSetManifest, root: Path) -> list[_Loaded]:
    annotators = {item.provenance_id: item.annotator for item in manifest.provenance}
    resolved_root = root.resolve()
    loaded: list[_Loaded] = []
    for entry in manifest.annotations:
        path = (root / entry.path).resolve()
        if not path.is_relative_to(resolved_root) or not path.is_file():
            continue
        if file_digest(path) != entry.content_hash:
            continue
        try:
            annotation_set = read_annotation_set(path)
        except AnnotationError:
            continue
        if annotation_set.family.schema != entry.schema:
            continue
        loaded.append(_Loaded(entry, annotators[entry.provenance_id], annotation_set))
    return loaded


class _Findings:
    def __init__(self) -> None:
        self.items: list[IntegrityFinding] = []
        self.permissible: list[PermissibleObservation] = []

    def blocker(self, code: str, message: str, *subjects: str) -> None:
        self.items.append(
            IntegrityFinding(
                severity=IntegritySeverity.BLOCKER, code=code, message=message, subjects=subjects
            )
        )

    def warning(self, code: str, message: str, *subjects: str) -> None:
        self.items.append(
            IntegrityFinding(
                severity=IntegritySeverity.WARNING, code=code, message=message, subjects=subjects
            )
        )

    def note(self, kind: str, family: AnnotationFamily, annotation_id: str, subject: str) -> None:
        self.permissible.append(
            PermissibleObservation(
                kind=kind, family=family, annotation_id=annotation_id, subject=subject
            )
        )


# ------------------------------------------------------------------------------ context


class _Context:
    """Lookups shared by the per-family checks."""

    def __init__(self, manifest: ReferenceSetManifest, loaded: list[_Loaded]) -> None:
        self.manifest = manifest
        self.loaded = loaded
        self.calibrations = {item.calibration_id for item in manifest.calibrations}
        self.frames: dict[tuple[str, str], list[FrameRegionAnnotation]] = defaultdict(list)
        self.identities: dict[str, list[tuple[_Loaded, Any]]] = defaultdict(list)
        self.semantics: list[tuple[_Loaded, SemanticAnnotationSet]] = []
        for item in loaded:
            annotation_set = item.annotation_set
            if isinstance(annotation_set, RegionAnnotationSet):
                for frame in annotation_set.frames:
                    self.frames[(frame.sample_id, frame.observation_id)].append(frame)
            elif isinstance(annotation_set, IdentityAnnotationSet):
                for identity in annotation_set.identities:
                    self.identities[identity.identity_id].append((item, identity))
            elif isinstance(annotation_set, SemanticAnnotationSet):
                self.semantics.append((item, annotation_set))

    def region_ids(self, sample_id: str, observation_id: str) -> set[str] | None:
        """Return the annotated region ids of an observation, or ``None`` if it has no regions."""
        frames = self.frames.get((sample_id, observation_id))
        if not frames:
            return None
        return {region.region_id for frame in frames for region in frame.regions}

    def image_size(self, sample_id: str, observation_id: str) -> tuple[int, int] | None:
        """Return the annotated image size of an observation, if a regions file declares it."""
        frames = self.frames.get((sample_id, observation_id))
        return None if not frames else (frames[0].image_width, frames[0].image_height)

    def occurrences(
        self, identity_id: str, sample_id: str, observation_id: str | None
    ) -> list[Any]:
        """Return the occurrences of an identity in an observation (or anywhere in a sample)."""
        found: list[Any] = []
        for _, identity in self.identities.get(identity_id, []):
            for occurrence in identity.occurrences:
                if occurrence.sample_id != sample_id:
                    continue
                if observation_id is None or occurrence.observation_id == observation_id:
                    found.append(occurrence)
        return found


# ------------------------------------------------------------------------------ regions


def _cells(mask: InlineMask) -> frozenset[int]:
    import numpy as np

    return frozenset(np.flatnonzero(mask.as_array()).tolist())


def _foreground_box(mask: InlineMask) -> Cell | None:
    cells = _cells(mask)
    if not cells:
        return None
    xs = [index % mask.width for index in cells]
    ys = [index // mask.width for index in cells]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _check_frame(item: _Loaded, frame: FrameRegionAnnotation, out: _Findings) -> None:
    subject = f"{item.annotation_id}:{frame.sample_id}/{frame.observation_id}"

    def sized(mask: InlineMask) -> bool:
        return mask.width == frame.image_width and mask.height == frame.image_height

    valid: set[int] = set()
    for area in frame.valid_areas:
        if sized(area):
            valid |= _cells(area)
        else:
            out.blocker(
                "area-size-mismatch",
                f"a valid area of {subject} is {area.width}x{area.height} but the image is "
                f"{frame.image_width}x{frame.image_height}",
                subject,
            )
    exclusion: set[int] = set()
    for area in frame.exclusion_areas:
        if sized(area):
            exclusion |= _cells(area)
        else:
            out.blocker(
                "area-size-mismatch",
                f"an exclusion area of {subject} is {area.width}x{area.height} but the image is "
                f"{frame.image_width}x{frame.image_height}",
                subject,
            )

    geometries: dict[object, str] = {}
    for region in frame.regions:
        region_subject = f"{subject}/{region.region_id}"
        box = region.box
        if box is not None and (box.x_max > frame.image_width or box.y_max > frame.image_height):
            out.blocker(
                "box-out-of-image",
                f"the box of region {region.region_id!r} exceeds the {frame.image_width}x"
                f"{frame.image_height} image",
                region_subject,
            )
        key: object = ("box", box.x_min, box.y_min, box.x_max, box.y_max) if box else None
        foreground: frozenset[int] = frozenset()
        if region.mask is not None:
            key = ("mask", region.mask)
            if not sized(region.mask):
                out.blocker(
                    "mask-size-mismatch",
                    f"the mask of region {region.region_id!r} is {region.mask.width}x"
                    f"{region.mask.height} but the image is {frame.image_width}x"
                    f"{frame.image_height}",
                    region_subject,
                )
            else:
                foreground = _cells(region.mask)
                if not foreground:
                    out.blocker(
                        "empty-mask",
                        f"the mask of region {region.region_id!r} has no foreground pixel",
                        region_subject,
                    )
                elif box is not None:
                    extent = _foreground_box(region.mask)
                    assert extent is not None
                    if (
                        extent[0] < box.x_min
                        or extent[1] < box.y_min
                        or extent[2] > box.x_max
                        or extent[3] > box.y_max
                    ):
                        out.blocker(
                            "mask-outside-box",
                            f"the mask of region {region.region_id!r} extends outside its box",
                            region_subject,
                        )
        if key in geometries:
            out.blocker(
                "duplicate-region-geometry",
                f"regions {geometries[key]!r} and {region.region_id!r} have identical geometry",
                region_subject,
            )
        elif key is not None:
            geometries[key] = region.region_id
        if foreground and exclusion:
            inside = foreground & exclusion
            if inside == foreground:
                out.blocker(
                    "region-in-exclusion-area",
                    f"region {region.region_id!r} lies entirely in an exclusion area",
                    region_subject,
                )
            elif inside:
                out.warning(
                    "region-overlaps-exclusion-area",
                    f"region {region.region_id!r} overlaps an exclusion area",
                    region_subject,
                )
        if foreground and valid:
            outside = foreground - valid
            if outside == foreground:
                out.blocker(
                    "region-outside-valid-area",
                    f"region {region.region_id!r} lies entirely outside the valid area",
                    region_subject,
                )
            elif outside:
                out.warning(
                    "region-extends-outside-valid-area",
                    f"region {region.region_id!r} extends outside the valid area",
                    region_subject,
                )
    if frame.coverage is Coverage.PARTIAL:
        out.note("partial-region-coverage", AnnotationFamily.REGIONS, item.annotation_id, subject)


# ---------------------------------------------------------------------------- geometry


def _check_geometry(
    ctx: _Context,
    item: _Loaded,
    geometry: GeometryAnnotationSet,
    policy: AnnotationQaPolicy,
    out: _Findings,
) -> None:
    frames_of_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for corr in geometry.correspondences:
        subject = f"{item.annotation_id}:{corr.correspondence_id}"
        if corr.calibration_id not in ctx.calibrations:
            out.blocker(
                "geometry-unknown-calibration",
                f"correspondence {corr.correspondence_id!r} uses calibration "
                f"{corr.calibration_id!r}, which the reference set does not declare",
                subject,
            )
        size = ctx.image_size(corr.sample_id, corr.image_observation_id)
        u, v = corr.pixel_uv
        if size is not None and not (0.0 <= u < size[0] and 0.0 <= v < size[1]):
            out.blocker(
                "pixel-outside-image",
                f"correspondence {corr.correspondence_id!r} has pixel ({u}, {v}) outside the "
                f"{size[0]}x{size[1]} image",
                subject,
            )
        if corr.point_source_observation_id is not None:
            frames_of_source[(corr.sample_id, corr.point_source_observation_id)].add(
                corr.point_frame_id
            )
        groups[(corr.sample_id, corr.image_observation_id, corr.calibration_id)].append(corr)
    for (sample, source), frame_ids in frames_of_source.items():
        if len(frame_ids) > 1:
            out.blocker(
                "point-frame-inconsistent",
                f"points of scan {source!r} in sample {sample!r} are expressed in more than one "
                f"frame: {sorted(frame_ids)}",
                f"{item.annotation_id}:{sample}/{source}",
            )
    for members in groups.values():
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                if math.dist(first.pixel_uv, second.pixel_uv) > policy.duplicate_pixel_tolerance_px:
                    continue
                if first.point_frame_id != second.point_frame_id:
                    continue
                subjects = (
                    f"{item.annotation_id}:{first.correspondence_id}",
                    f"{item.annotation_id}:{second.correspondence_id}",
                )
                if math.dist(first.point_m, second.point_m) > policy.duplicate_point_tolerance_m:
                    out.blocker(
                        "conflicting-correspondence",
                        f"correspondences {first.correspondence_id!r} and "
                        f"{second.correspondence_id!r} annotate the same pixel with different "
                        "3D points",
                        *subjects,
                    )
                else:
                    out.warning(
                        "duplicate-correspondence",
                        f"correspondences {first.correspondence_id!r} and "
                        f"{second.correspondence_id!r} annotate the same pixel and point",
                        *subjects,
                    )


# ---------------------------------------------------------------------------- identity


def _record_keys(normalization: LabelNormalization, record: SemanticAnnotationRecord) -> set[str]:
    return {normalization.key(concept) for concept in record.concepts}


def _check_identity(
    ctx: _Context, item: _Loaded, identities: IdentityAnnotationSet, out: _Findings
) -> None:
    scope = {(entry.sample_id, entry.observation_id) for entry in identities.scope}
    claims: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for identity in identities.identities:
        per_observation: dict[tuple[str, str], int] = defaultdict(int)
        concept_sets: list[set[str]] = []
        for occurrence in identity.occurrences:
            key = (occurrence.sample_id, occurrence.observation_id)
            subject = f"{item.annotation_id}:{identity.identity_id}"
            per_observation[key] += 1
            if key not in scope:
                out.warning(
                    "occurrence-outside-scope",
                    f"identity {identity.identity_id!r} appears in {key[1]!r}, which the identity "
                    "scope does not list",
                    subject,
                )
            if occurrence.region_id is None:
                continue
            known = ctx.region_ids(*key)
            if known is not None and occurrence.region_id not in known:
                out.blocker(
                    "identity-unknown-region",
                    f"identity {identity.identity_id!r} points at region "
                    f"{occurrence.region_id!r}, which {key[1]!r} does not have",
                    subject,
                )
            claims[(key[0], key[1], occurrence.region_id)].append(identity.identity_id)
            for _, semantics in ctx.semantics:
                record = semantics.find(key[0], key[1], occurrence.region_id)
                if record is not None and record.status is SemanticStatus.LABELED:
                    concept_sets.append(_record_keys(semantics.normalization, record))
        for observed, count in per_observation.items():
            if count > 1:
                out.warning(
                    "identity-repeated-in-observation",
                    f"identity {identity.identity_id!r} has {count} occurrences in {observed[1]!r}",
                    f"{item.annotation_id}:{identity.identity_id}",
                )
        if any(
            not (first & second)
            for index, first in enumerate(concept_sets)
            for second in concept_sets[index + 1 :]
        ):
            out.warning(
                "identity-semantic-conflict",
                f"identity {identity.identity_id!r} is annotated with disjoint concepts in "
                "different observations",
                f"{item.annotation_id}:{identity.identity_id}",
            )
    for (sample, observation, region), owners in claims.items():
        if len(set(owners)) > 1:
            out.blocker(
                "region-claimed-by-two-identities",
                f"region {region!r} of {observation!r} is claimed by identities "
                f"{sorted(set(owners))}",
                f"{item.annotation_id}:{sample}/{observation}/{region}",
            )
    for entry in identities.scope:
        if entry.coverage is Coverage.PARTIAL:
            out.note(
                "partial-identity-coverage",
                AnnotationFamily.IDENTITY,
                item.annotation_id,
                f"{entry.sample_id}/{entry.observation_id}",
            )


# --------------------------------------------------------------------------- relations


def _check_relations(
    ctx: _Context, item: _Loaded, relations: RelationAnnotationSet, out: _Findings
) -> None:
    key = relations.normalization.key
    identity_ids = set(ctx.identities)
    triples: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for entry in relations.relations:
        subject = f"{item.annotation_id}:{entry.relation_id}"
        unknown = [
            name
            for name in (entry.subject_identity_id, entry.object_identity_id)
            if name not in identity_ids
        ]
        for name in unknown:
            out.blocker(
                "relation-unknown-identity",
                f"relation {entry.relation_id!r} names identity {name!r}, which no identity "
                "annotation declares",
                subject,
            )
        if not unknown:
            for anchor in entry.anchors:
                for name in (entry.subject_identity_id, entry.object_identity_id):
                    if not ctx.occurrences(name, anchor.sample_id, anchor.observation_id):
                        out.blocker(
                            "relation-anchor-without-identity",
                            f"relation {entry.relation_id!r} is anchored in "
                            f"{anchor.observation_id!r}, where identity {name!r} is not seen",
                            subject,
                        )
        triples[(entry.subject_identity_id, key(entry.predicate), entry.object_identity_id)].append(
            entry
        )
        if entry.status is RelationStatus.AMBIGUOUS:
            out.note(
                "ambiguous-relation",
                AnnotationFamily.RELATIONS,
                item.annotation_id,
                entry.relation_id,
            )
        elif entry.status is RelationStatus.UNKNOWN:
            out.note(
                "unknown-relation",
                AnnotationFamily.RELATIONS,
                item.annotation_id,
                entry.relation_id,
            )

    for members in triples.values():
        if len(members) > 1:
            statuses = {member.status for member in members}
            names = [member.relation_id for member in members]
            if len(statuses) > 1:
                out.blocker(
                    "conflicting-relation",
                    f"relations {names} state the same relation with different statuses",
                    *(f"{item.annotation_id}:{name}" for name in names),
                )
            else:
                out.warning(
                    "duplicate-relation",
                    f"relations {names} state the same relation more than once",
                    *(f"{item.annotation_id}:{name}" for name in names),
                )

    decided = {RelationStatus.HOLDS, RelationStatus.DOES_NOT_HOLD}
    reported: set[frozenset[tuple[str, str, str]]] = set()
    for rule in relations.predicate_rules:
        partner = (
            key(rule.predicate)
            if rule.symmetric
            else (None if rule.inverse is None else key(rule.inverse))
        )
        if partner is None:
            continue
        code = "symmetry-violation" if rule.symmetric else "inverse-violation"
        for (subject_id, predicate, object_id), members in triples.items():
            if predicate != key(rule.predicate):
                continue
            mirror = triples.get((object_id, partner, subject_id))
            if not mirror:
                continue
            pair = frozenset({(subject_id, predicate, object_id), (object_id, partner, subject_id)})
            if pair in reported:
                continue
            statuses = {member.status for member in members} | {member.status for member in mirror}
            if decided <= statuses:
                reported.add(pair)
                names = [member.relation_id for member in (*members, *mirror)]
                out.blocker(
                    code,
                    f"relations {names} contradict each other under the "
                    f"{'symmetry' if rule.symmetric else 'inverse'} of {rule.predicate!r}",
                    *(f"{item.annotation_id}:{name}" for name in names),
                )


# -------------------------------------------------------------------- semantics/visibility


def _check_semantics(
    ctx: _Context, item: _Loaded, semantics: SemanticAnnotationSet, out: _Findings
) -> None:
    for record in semantics.records:
        subject = (
            f"{item.annotation_id}:{record.sample_id}/{record.observation_id}/{record.region_id}"
        )
        if record.region_id is not None:
            known = ctx.region_ids(record.sample_id, record.observation_id)
            if known is not None and record.region_id not in known:
                out.blocker(
                    "semantic-unknown-region",
                    f"a semantic record targets region {record.region_id!r}, which "
                    f"{record.observation_id!r} does not have",
                    subject,
                )
        if record.status is SemanticStatus.AMBIGUOUS:
            out.note("ambiguous-semantics", AnnotationFamily.SEMANTICS, item.annotation_id, subject)
        elif record.status is SemanticStatus.UNKNOWN:
            out.note("unknown-semantics", AnnotationFamily.SEMANTICS, item.annotation_id, subject)


_FRACTION_RULES: dict[VisibilityLevel, str] = {
    VisibilityLevel.FULLY_VISIBLE: "none-or-zero",
    VisibilityLevel.NOT_VISIBLE: "none-or-one",
    VisibilityLevel.UNKNOWN: "none",
    VisibilityLevel.PARTIALLY_OCCLUDED: "open",
    VisibilityLevel.HEAVILY_OCCLUDED: "open",
}


def _fraction_consistent(level: VisibilityLevel, fraction: float | None) -> bool:
    rule = _FRACTION_RULES[level]
    if fraction is None:
        return True
    if rule == "none":
        return False
    if rule == "none-or-zero":
        return fraction == 0.0
    if rule == "none-or-one":
        return fraction == 1.0
    return 0.0 < fraction < 1.0


def _check_visibility(
    ctx: _Context, item: _Loaded, visibility: VisibilityAnnotationSet, out: _Findings
) -> None:
    for annotation in visibility.annotations:
        subject = (
            f"{item.annotation_id}:{annotation.sample_id}/{annotation.observation_id}/"
            f"{annotation.region_id or annotation.identity_id}"
        )
        if annotation.region_id is not None:
            known = ctx.region_ids(annotation.sample_id, annotation.observation_id)
            if known is not None and annotation.region_id not in known:
                out.blocker(
                    "visibility-unknown-target",
                    f"visibility targets region {annotation.region_id!r}, which "
                    f"{annotation.observation_id!r} does not have",
                    subject,
                )
        if annotation.identity_id is not None and annotation.identity_id not in ctx.identities:
            out.blocker(
                "visibility-unknown-target",
                f"visibility targets identity {annotation.identity_id!r}, which no identity "
                "annotation declares",
                subject,
            )
        if not _fraction_consistent(annotation.level, annotation.occluded_fraction):
            out.blocker(
                "visibility-level-fraction-mismatch",
                f"level {annotation.level.value!r} contradicts the occluded fraction "
                f"{annotation.occluded_fraction}",
                subject,
            )
        if (
            annotation.level is VisibilityLevel.NOT_VISIBLE
            and annotation.identity_id is not None
            and any(
                occurrence.region_id is not None
                for occurrence in ctx.occurrences(
                    annotation.identity_id, annotation.sample_id, annotation.observation_id
                )
            )
        ):
            out.blocker(
                "visibility-contradicts-occurrence",
                f"identity {annotation.identity_id!r} is not visible in "
                f"{annotation.observation_id!r} yet has an annotated region there",
                subject,
            )
        if annotation.level is VisibilityLevel.UNKNOWN:
            out.note("unknown-visibility", AnnotationFamily.VISIBILITY, item.annotation_id, subject)


# --------------------------------------------------------- multiple annotation files


def _semantic_signature(
    normalization: LabelNormalization, record: SemanticAnnotationRecord
) -> tuple[str, frozenset[str]]:
    return (record.status.value, frozenset(_record_keys(normalization, record)))


def _describe_semantics(signature: tuple[str, frozenset[str]]) -> str:
    return f"{signature[0]}: {sorted(signature[1])}"


def _semantic_targets(
    normalization: LabelNormalization, semantics: SemanticAnnotationSet
) -> dict[Target, tuple[str, frozenset[str]]]:
    return {
        (record.sample_id, record.observation_id, str(record.region_id)): _semantic_signature(
            normalization, record
        )
        for record in semantics.records
    }


def _relation_targets(relations: RelationAnnotationSet) -> dict[Target, str]:
    key = relations.normalization.key
    targets: dict[Target, str] = {}
    for entry in relations.relations:
        targets[(entry.subject_identity_id, key(entry.predicate), entry.object_identity_id)] = (
            entry.status.value
        )
    return targets


def _visibility_targets(visibility: VisibilityAnnotationSet) -> dict[Target, str]:
    return {
        (
            item.sample_id,
            item.observation_id,
            str(item.region_id),
            str(item.identity_id),
        ): item.level.value
        for item in visibility.annotations
    }


def _region_bounds(region: RegionAnnotation) -> tuple[float, float, float, float] | None:
    if region.box is not None:
        box = region.box
        return (box.x_min, box.y_min, box.x_max, box.y_max)
    if region.mask is not None:
        extent = _foreground_box(region.mask)
        if extent is None:
            return None
        return (float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3]))
    return None


def _region_iou(first: RegionAnnotation, second: RegionAnnotation) -> float:
    if (
        first.mask is not None
        and second.mask is not None
        and (first.mask.width, first.mask.height) == (second.mask.width, second.mask.height)
    ):
        left, right = _cells(first.mask), _cells(second.mask)
        union = left | right
        return len(left & right) / len(union) if union else 0.0
    a, b = _region_bounds(first), _region_bounds(second)
    if a is None or b is None:
        return 0.0
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def _compare_frames(
    policy: AnnotationQaPolicy,
    first: FrameRegionAnnotation,
    second: FrameRegionAnnotation,
) -> tuple[list[str], list[str], float | None]:
    """Match two annotators' regions greedily by IoU; return the unmatched ids and mean IoU."""
    pairs = sorted(
        (
            (_region_iou(a, b), a.region_id, b.region_id)
            for a in first.regions
            for b in second.regions
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_a: set[str] = set()
    used_b: set[str] = set()
    matched: list[float] = []
    for iou, name_a, name_b in pairs:
        if iou < policy.region_match_iou:
            break
        if name_a in used_a or name_b in used_b:
            continue
        used_a.add(name_a)
        used_b.add(name_b)
        matched.append(iou)
    only_first = [region.region_id for region in first.regions if region.region_id not in used_a]
    only_second = [region.region_id for region in second.regions if region.region_id not in used_b]
    return only_first, only_second, (sum(matched) / len(matched) if matched else None)


def _compare_pair(
    policy: AnnotationQaPolicy, first: _Loaded, second: _Loaded, out: _Findings
) -> DisagreementSummary | None:
    """Compare two annotation files of one family.

    Two annotators yield a disagreement summary. One annotator twice is not a
    disagreement: identical records are a duplicate (warning) and differing ones a
    conflict (blocker). ``None`` means there is nothing to summarize.
    """
    family = first.annotation_set.family
    ids = (first.annotation_id, second.annotation_id)
    annotators = (first.annotator, second.annotator)
    records: list[DisagreementRecord] = []
    compared = 0
    agreements = 0

    def add(subject: str, kind: str, left: str, right: str) -> None:
        records.append(
            DisagreementRecord(
                subject=subject, kind=kind, positions=((ids[0], left), (ids[1], right))
            )
        )

    a, b = first.annotation_set, second.annotation_set
    if isinstance(a, SemanticAnnotationSet) and isinstance(b, SemanticAnnotationSet):
        if a.normalization != b.normalization:
            out.warning(
                "normalization-differs",
                f"{ids[0]} and {ids[1]} normalize labels differently, so their concepts are not "
                "compared",
                *ids,
            )
            return None
        left, right = _semantic_targets(a.normalization, a), _semantic_targets(b.normalization, b)
        for target in left.keys() & right.keys():
            compared += 1
            if left[target] == right[target]:
                agreements += 1
                continue
            kind = (
                "status-differs"
                if left[target][0] != right[target][0]
                else "partial-concept-overlap"
                if left[target][1] & right[target][1]
                else "disjoint-concepts"
            )
            add(
                "/".join(target),
                kind,
                _describe_semantics(left[target]),
                _describe_semantics(right[target]),
            )
    elif isinstance(a, RelationAnnotationSet) and isinstance(b, RelationAnnotationSet):
        if a.normalization != b.normalization:
            out.warning(
                "normalization-differs",
                f"{ids[0]} and {ids[1]} normalize predicates differently, so they are not compared",
                *ids,
            )
            return None
        left_r, right_r = _relation_targets(a), _relation_targets(b)
        for target in left_r.keys() & right_r.keys():
            compared += 1
            if left_r[target] == right_r[target]:
                agreements += 1
            else:
                add(" ".join(target), "status-differs", left_r[target], right_r[target])
    elif isinstance(a, VisibilityAnnotationSet) and isinstance(b, VisibilityAnnotationSet):
        left_v, right_v = _visibility_targets(a), _visibility_targets(b)
        for target in left_v.keys() & right_v.keys():
            compared += 1
            if left_v[target] == right_v[target]:
                agreements += 1
            else:
                add("/".join(target), "level-differs", left_v[target], right_v[target])
    elif isinstance(a, RegionAnnotationSet) and isinstance(b, RegionAnnotationSet):
        frames_a = {(item.sample_id, item.observation_id): item for item in a.frames}
        frames_b = {(item.sample_id, item.observation_id): item for item in b.frames}
        for target in frames_a.keys() & frames_b.keys():
            compared += 1
            only_first, only_second, mean_iou = _compare_frames(
                policy, frames_a[target], frames_b[target]
            )
            if (
                not only_first
                and not only_second
                and frames_a[target].coverage is frames_b[target].coverage
            ):
                agreements += 1
                continue
            iou = "n/a" if mean_iou is None else f"{mean_iou:.3f}"
            add(
                "/".join(target),
                "region-sets-differ",
                f"{frames_a[target].coverage.value}; unmatched {only_first}; mean IoU {iou}",
                f"{frames_b[target].coverage.value}; unmatched {only_second}; mean IoU {iou}",
            )
    else:
        return None

    if first.annotator == second.annotator:
        if compared == 0:
            return None
        if records:
            out.blocker(
                "conflicting-annotation-records",
                f"{ids[0]} and {ids[1]} are by the same annotator {first.annotator!r} and "
                f"disagree on {sorted(record.subject for record in records)}",
                *ids,
            )
        else:
            out.warning(
                "duplicate-annotation-record",
                f"{ids[0]} and {ids[1]} are by the same annotator {first.annotator!r} and "
                f"annotate {compared} target(s) identically",
                *ids,
            )
        return None
    records.sort(key=lambda item: item.subject)
    return DisagreementSummary(
        family=family,
        annotation_ids=ids,
        annotators=annotators,
        compared=compared,
        agreements=agreements,
        disagreements=tuple(records),
    )


# -------------------------------------------------------------------------- entry points


def check_annotation_quality(
    manifest: ReferenceSetManifest, root: Path, *, policy: AnnotationQaPolicy
) -> AnnotationQaReport:
    """Check the content of every annotation file of a reference set.

    Never raises for a flawed reference set: every problem becomes a finding.
    Files that are missing, modified or unreadable are skipped here and reported
    by the embedded integrity report.
    """
    integrity = validate_reference_set(manifest, root)
    loaded = _load(manifest, root)
    ctx = _Context(manifest, loaded)
    out = _Findings()

    for item in loaded:
        annotation_set = item.annotation_set
        if isinstance(annotation_set, RegionAnnotationSet):
            for frame in annotation_set.frames:
                _check_frame(item, frame, out)
        elif isinstance(annotation_set, GeometryAnnotationSet):
            _check_geometry(ctx, item, annotation_set, policy, out)
        elif isinstance(annotation_set, IdentityAnnotationSet):
            _check_identity(ctx, item, annotation_set, out)
        elif isinstance(annotation_set, RelationAnnotationSet):
            _check_relations(ctx, item, annotation_set, out)
        elif isinstance(annotation_set, SemanticAnnotationSet):
            _check_semantics(ctx, item, annotation_set, out)
        elif isinstance(annotation_set, VisibilityAnnotationSet):
            _check_visibility(ctx, item, annotation_set, out)

    summaries: list[DisagreementSummary] = []
    by_family: dict[AnnotationFamily, list[_Loaded]] = defaultdict(list)
    for item in loaded:
        by_family[item.annotation_set.family].append(item)
    for members in by_family.values():
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                summary = _compare_pair(policy, first, second, out)
                if summary is not None:
                    summaries.append(summary)

    return AnnotationQaReport(
        reference_set=manifest.identity(),
        policy=policy,
        integrity=integrity,
        findings=tuple(out.items),
        permissible=tuple(out.permissible),
        disagreements=tuple(summaries),
        annotation_files_checked=len(loaded),
    )


def certify_reference_set(root: Path, *, policy: AnnotationQaPolicy) -> CertifiedReferenceSet:
    """Open a reference set and require it to pass both integrity validation and annotation QA.

    Raises:
        ReferenceSetError: If the manifest cannot be trusted.
        ReferenceSetIntegrityError: If integrity validation found a blocker.
        AnnotationQaError: If annotation QA found a blocker.
    """
    validated = open_validated_reference_set(root)
    report = check_annotation_quality(validated.manifest, root, policy=policy)
    if not report.is_valid:
        raise AnnotationQaError(report)
    return CertifiedReferenceSet(validated=validated, qa=report, policy=policy)
