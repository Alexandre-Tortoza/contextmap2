"""Versioned reference-set manifest for Solution 1 evaluation.

A reference set fixes *what is evaluated against what*: the source sequences,
the physical observations selected from them, the calibration identities, the
annotation files with their schema versions and provenance, the strata used to
stratify results, and the tuning/development/test splits. It is versioned and
hashed independently of any experiment output. See
``src/contextmap/evaluation/docs/reference-set.md``.

Identity rules enforced here:

* samples are bound to physical ``SourceObservationId`` values, never to the
  outputs of a perception run;
* every annotation file declares its trust level and its provenance; trust is
  never inferred from a file name or extension;
* annotations produced by model inference can only be diagnostic data.

Policy-level integrity (split leakage, duplicate content, provenance audits)
belongs to the integrity validation that consumes this manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, NewType

from contextmap.evaluation._persistence import (
    canonical_digest,
    file_digest,
    write_immutable_json,
)
from contextmap.evaluation._validation import require_sha256, require_text, require_unique
from contextmap.ingestion import CalibrationReferenceId, SourceObservationId
from contextmap.shared import SourceTimestamp

REFERENCE_SET_SCHEMA = "contextmap.reference-set/v1"
"""Schema identifier of the reference-set manifest document."""

MANIFEST_FILENAME = "manifest.json"
"""File name of the manifest inside a reference-set version directory."""

ReferenceSampleId = NewType("ReferenceSampleId", str)
"""Stable identity of one evaluation sample inside a reference set."""


class ReferenceSetError(ValueError):
    """Raised when a reference-set manifest or its files cannot be trusted."""


class SourceKind(Enum):
    """Where the physical data of a source comes from."""

    SEQUENCE_ARTIFACT = "sequence_artifact"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    EXTERNAL_DATASET = "external_dataset"


class ReferenceTrust(Enum):
    """What a reference may be used as.

    The level is declared per annotation file. File names such as ``gt`` or
    extensions such as ``.pcd`` never establish it.
    """

    TRUSTED_GROUND_TRUTH = "trusted_ground_truth"
    APPROXIMATE_ANNOTATION = "approximate_annotation"
    DERIVED_MEASUREMENT = "derived_measurement"
    DIAGNOSTIC_ONLY = "diagnostic_only"


class ProvenanceOrigin(Enum):
    """How an annotation was produced."""

    MANUAL_ANNOTATION = "manual_annotation"
    EXTERNAL_DATASET = "external_dataset"
    SYNTHETIC_GENERATION = "synthetic_generation"
    SENSOR_MEASUREMENT = "sensor_measurement"
    MODEL_INFERENCE = "model_inference"


class SplitRole(Enum):
    """The purpose of a split: what may be tuned on it and what only measured."""

    TUNING = "tuning"
    DEVELOPMENT = "development"
    TEST = "test"


class SplitUnit(Enum):
    """The grouping unit that must not straddle two splits of one scheme."""

    SEQUENCE = "sequence"
    SCENE = "scene"
    TIME_SEGMENT = "time_segment"
    PHYSICAL_OBJECT = "physical_object"
    OTHER = "other"


@dataclass(frozen=True, kw_only=True)
class ReferenceSource:
    """A recorded or generated body of data the samples are drawn from.

    Attributes:
        source_id: Identity of the source inside this manifest.
        kind: Sequence artifact, synthetic fixture or external dataset.
        identity: Identity in its own domain, e.g. a sequence artifact id.
        content_hash: ``sha256:<hex>`` of the source content (or, for a
            synthetic fixture, of its generation parameters).
        license: License of the data, ``"unspecified"`` when unknown.
        redistributable: Whether the data may be copied into the repository.
    """

    source_id: str
    kind: SourceKind
    identity: str
    content_hash: str
    license: str
    redistributable: bool

    def __post_init__(self) -> None:
        """Require identity, a valid hash and an explicit license."""
        require_text("source_id", self.source_id)
        require_text("source identity", self.identity)
        require_text("source license", self.license)
        require_sha256("source content_hash", self.content_hash)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "source_id": self.source_id,
            "kind": self.kind.value,
            "identity": self.identity,
            "content_hash": self.content_hash,
            "license": self.license,
            "redistributable": self.redistributable,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReferenceSource:
        """Rebuild a source from :meth:`to_record` output."""
        return cls(
            source_id=record["source_id"],
            kind=SourceKind(record["kind"]),
            identity=record["identity"],
            content_hash=record["content_hash"],
            license=record["license"],
            redistributable=record["redistributable"],
        )


@dataclass(frozen=True, kw_only=True)
class CalibrationIdentity:
    """A calibration entry the samples and geometric annotations refer to.

    Attributes:
        calibration_id: The ingestion calibration reference identity.
        source_id: The source the calibration belongs to.
        content_hash: ``sha256:<hex>`` of the calibration entry, to detect a
            calibration that changed under the same identity.
    """

    calibration_id: CalibrationReferenceId
    source_id: str
    content_hash: str

    def __post_init__(self) -> None:
        """Require identity and a valid hash."""
        require_text("calibration_id", self.calibration_id)
        require_text("calibration source_id", self.source_id)
        require_sha256("calibration content_hash", self.content_hash)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "calibration_id": self.calibration_id,
            "source_id": self.source_id,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CalibrationIdentity:
        """Rebuild a calibration identity from :meth:`to_record` output."""
        return cls(
            calibration_id=CalibrationReferenceId(record["calibration_id"]),
            source_id=record["source_id"],
            content_hash=record["content_hash"],
        )


@dataclass(frozen=True, kw_only=True)
class StratumDefinition:
    """A declared stratification factor and its allowed values.

    Attributes:
        name: Factor name, e.g. ``"visibility"`` or ``"range_band"``.
        values: The values a sample may take for this factor.
        description: What the factor measures and how it was assigned.
    """

    name: str
    values: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        """Require a name, a description and unique non-empty values."""
        require_text("stratum name", self.name)
        require_text("stratum description", self.description)
        if not self.values:
            raise ValueError(f"stratum {self.name!r} needs at least one value")
        for value in self.values:
            require_text("stratum value", value)
        require_unique(f"value of stratum {self.name!r}", self.values)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"name": self.name, "values": list(self.values), "description": self.description}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> StratumDefinition:
        """Rebuild a definition from :meth:`to_record` output."""
        return cls(
            name=record["name"],
            values=tuple(record["values"]),
            description=record["description"],
        )


@dataclass(frozen=True, kw_only=True)
class SampleStratum:
    """The value one sample takes for a declared stratification factor."""

    name: str
    value: str

    def __post_init__(self) -> None:
        """Require a factor name and a value."""
        require_text("stratum name", self.name)
        require_text("stratum value", self.value)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SampleStratum:
        """Rebuild a stratum assignment from :meth:`to_record` output."""
        return cls(name=record["name"], value=record["value"])


@dataclass(frozen=True, kw_only=True)
class SampleGroup:
    """A grouping key of one sample for one split unit, e.g. its scene."""

    unit: SplitUnit
    key: str

    def __post_init__(self) -> None:
        """Require a key."""
        require_text("group key", self.key)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"unit": self.unit.value, "key": self.key}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SampleGroup:
        """Rebuild a group from :meth:`to_record` output."""
        return cls(unit=SplitUnit(record["unit"]), key=record["key"])


@dataclass(frozen=True, kw_only=True)
class SampleTimeSpan:
    """The time covered by a sample, inside one clock domain."""

    start: SourceTimestamp
    end: SourceTimestamp

    def __post_init__(self) -> None:
        """Require one clock domain and an ordered span."""
        if self.start.clock_id != self.end.clock_id:
            raise ValueError("start and end of a time span must share a clock")
        if self.end.total_nanoseconds() < self.start.total_nanoseconds():
            raise ValueError("the end of a time span must not be before its start")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"start": self.start.to_record(), "end": self.end.to_record()}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SampleTimeSpan:
        """Rebuild a span from :meth:`to_record` output."""
        return cls(
            start=SourceTimestamp.from_record(record["start"]),
            end=SourceTimestamp.from_record(record["end"]),
        )


@dataclass(frozen=True, kw_only=True)
class ReferenceSample:
    """One evaluation sample: a selection of physical observations.

    The identity of a sample is independent of any inference run: it is bound
    to the ``SourceObservationId`` values it selects.

    Attributes:
        sample_id: Identity of the sample inside this manifest.
        source_id: The source the observations are drawn from.
        observation_ids: The selected physical observations, in order.
        calibration_ids: Calibration entries that apply to the sample.
        time_span: The time covered, when known.
        content_hash: ``sha256:<hex>`` of the sample content, so duplicated
            content is detectable and a changed sample changes the version.
        strata: Values for the declared stratification factors.
        groups: Grouping keys used to keep related samples in one split.
    """

    sample_id: ReferenceSampleId
    source_id: str
    observation_ids: tuple[SourceObservationId, ...]
    calibration_ids: tuple[CalibrationReferenceId, ...]
    time_span: SampleTimeSpan | None
    content_hash: str
    strata: tuple[SampleStratum, ...] = ()
    groups: tuple[SampleGroup, ...] = ()

    def __post_init__(self) -> None:
        """Require identity, physical observations and unique assignments."""
        require_text("sample_id", self.sample_id)
        require_text("sample source_id", self.source_id)
        require_sha256("sample content_hash", self.content_hash)
        if not self.observation_ids:
            raise ValueError(f"sample {self.sample_id!r} needs at least one source observation")
        require_unique(f"observation id of sample {self.sample_id!r}", self.observation_ids)
        require_unique(f"calibration id of sample {self.sample_id!r}", self.calibration_ids)
        require_unique(f"stratum of sample {self.sample_id!r}", (item.name for item in self.strata))
        require_unique(
            f"group of sample {self.sample_id!r}",
            (f"{item.unit.value}:{item.key}" for item in self.groups),
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "observation_ids": list(self.observation_ids),
            "calibration_ids": list(self.calibration_ids),
            "time_span": None if self.time_span is None else self.time_span.to_record(),
            "content_hash": self.content_hash,
            "strata": [item.to_record() for item in self.strata],
            "groups": [item.to_record() for item in self.groups],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReferenceSample:
        """Rebuild a sample from :meth:`to_record` output."""
        span = record["time_span"]
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            source_id=record["source_id"],
            observation_ids=tuple(SourceObservationId(item) for item in record["observation_ids"]),
            calibration_ids=tuple(
                CalibrationReferenceId(item) for item in record["calibration_ids"]
            ),
            time_span=None if span is None else SampleTimeSpan.from_record(span),
            content_hash=record["content_hash"],
            strata=tuple(SampleStratum.from_record(item) for item in record["strata"]),
            groups=tuple(SampleGroup.from_record(item) for item in record["groups"]),
        )


@dataclass(frozen=True, kw_only=True)
class ProvenanceReview:
    """An explicit human review of an annotation that a model helped to seed."""

    reviewer: str
    method: str

    def __post_init__(self) -> None:
        """Require who reviewed and how."""
        require_text("reviewer", self.reviewer)
        require_text("review method", self.method)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"reviewer": self.reviewer, "method": self.method}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ProvenanceReview:
        """Rebuild a review from :meth:`to_record` output."""
        return cls(reviewer=record["reviewer"], method=record["method"])


@dataclass(frozen=True, kw_only=True)
class AnnotationProvenance:
    """Who or what produced an annotation, and how, independently of model outputs.

    Attributes:
        provenance_id: Identity referenced by annotation files.
        origin: Manual, external dataset, synthetic, sensor measurement or model.
        annotator: The person, team, dataset or tool that produced the labels.
        method: The protocol or derivation method.
        tool_version: Version of the annotation or generation tool, if any.
        seeded_from_artifacts: Model-run artifacts whose output was used as a
            starting point; empty when no model output entered the annotation.
        review: The explicit review that made a model-seeded annotation
            acceptable as a reference, when there is one.
    """

    provenance_id: str
    origin: ProvenanceOrigin
    annotator: str
    method: str
    tool_version: str | None = None
    seeded_from_artifacts: tuple[str, ...] = ()
    review: ProvenanceReview | None = None

    def __post_init__(self) -> None:
        """Require identity, annotator and method."""
        require_text("provenance_id", self.provenance_id)
        require_text("annotator", self.annotator)
        require_text("annotation method", self.method)
        for artifact in self.seeded_from_artifacts:
            require_text("seeding artifact", artifact)
        require_unique("seeding artifact", self.seeded_from_artifacts)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "provenance_id": self.provenance_id,
            "origin": self.origin.value,
            "annotator": self.annotator,
            "method": self.method,
            "tool_version": self.tool_version,
            "seeded_from_artifacts": list(self.seeded_from_artifacts),
            "review": None if self.review is None else self.review.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> AnnotationProvenance:
        """Rebuild a provenance record from :meth:`to_record` output."""
        review = record["review"]
        return cls(
            provenance_id=record["provenance_id"],
            origin=ProvenanceOrigin(record["origin"]),
            annotator=record["annotator"],
            method=record["method"],
            tool_version=record["tool_version"],
            seeded_from_artifacts=tuple(record["seeded_from_artifacts"]),
            review=None if review is None else ProvenanceReview.from_record(review),
        )


@dataclass(frozen=True, kw_only=True)
class AnnotationFileEntry:
    """One annotation file: where it is, what schema it follows, and how far to trust it.

    Attributes:
        annotation_id: Identity of the file inside this manifest.
        schema: Versioned annotation schema, e.g. ``contextmap.reference.regions/v1``.
        path: Path relative to the reference-set directory, POSIX separators.
        content_hash: ``sha256:<hex>`` of the file.
        trust: Declared trust level; never inferred from the path.
        provenance_id: The provenance record that explains how it was produced.
        sample_ids: The samples the file annotates.
    """

    annotation_id: str
    schema: str
    path: str
    content_hash: str
    trust: ReferenceTrust
    provenance_id: str
    sample_ids: tuple[ReferenceSampleId, ...]

    def __post_init__(self) -> None:
        """Require identity, schema, a contained relative path and a valid hash."""
        require_text("annotation_id", self.annotation_id)
        require_text("annotation schema", self.schema)
        require_text("annotation provenance_id", self.provenance_id)
        require_sha256("annotation content_hash", self.content_hash)
        pure = PurePosixPath(self.path)
        if not pure.parts or pure.is_absolute() or ".." in pure.parts or "\\" in self.path:
            raise ValueError(
                f"annotation path must be a contained relative POSIX path, got {self.path!r}"
            )
        require_unique(f"sample id of annotation {self.annotation_id!r}", self.sample_ids)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "annotation_id": self.annotation_id,
            "schema": self.schema,
            "path": self.path,
            "content_hash": self.content_hash,
            "trust": self.trust.value,
            "provenance_id": self.provenance_id,
            "sample_ids": list(self.sample_ids),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> AnnotationFileEntry:
        """Rebuild an entry from :meth:`to_record` output."""
        return cls(
            annotation_id=record["annotation_id"],
            schema=record["schema"],
            path=record["path"],
            content_hash=record["content_hash"],
            trust=ReferenceTrust(record["trust"]),
            provenance_id=record["provenance_id"],
            sample_ids=tuple(ReferenceSampleId(item) for item in record["sample_ids"]),
        )


@dataclass(frozen=True, kw_only=True)
class ReferenceSplit:
    """An ordered selection of samples that serves one role of a split scheme."""

    name: str
    role: SplitRole
    sample_ids: tuple[ReferenceSampleId, ...]

    def __post_init__(self) -> None:
        """Require a name, and a selection without repeated samples."""
        require_text("split name", self.name)
        require_unique(f"sample id of split {self.name!r}", self.sample_ids)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"name": self.name, "role": self.role.value, "sample_ids": list(self.sample_ids)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReferenceSplit:
        """Rebuild a split from :meth:`to_record` output."""
        return cls(
            name=record["name"],
            role=SplitRole(record["role"]),
            sample_ids=tuple(ReferenceSampleId(item) for item in record["sample_ids"]),
        )


@dataclass(frozen=True, kw_only=True)
class SplitScheme:
    """How the samples are divided for one evaluation task, and why.

    Attributes:
        scheme_id: Identity of the scheme inside this manifest.
        task: The evaluation task or benchmark the scheme serves.
        unit: The grouping unit that must not straddle two splits.
        rationale: Why this unit prevents leakage for this task.
        splits: The splits of the scheme.
        adjacency_window_ns: When set, samples of one source whose time spans
            are closer than this many nanoseconds count as adjacent and must
            stay in one split.
    """

    scheme_id: str
    task: str
    unit: SplitUnit
    rationale: str
    splits: tuple[ReferenceSplit, ...]
    adjacency_window_ns: int | None = None

    def __post_init__(self) -> None:
        """Require an explicit unit rationale and unique split names."""
        require_text("scheme_id", self.scheme_id)
        require_text("scheme task", self.task)
        require_text("scheme rationale", self.rationale)
        require_unique(
            f"split name of scheme {self.scheme_id!r}", (item.name for item in self.splits)
        )
        if self.adjacency_window_ns is not None and self.adjacency_window_ns < 0:
            raise ValueError("adjacency_window_ns must not be negative")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "scheme_id": self.scheme_id,
            "task": self.task,
            "unit": self.unit.value,
            "rationale": self.rationale,
            "splits": [item.to_record() for item in self.splits],
            "adjacency_window_ns": self.adjacency_window_ns,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SplitScheme:
        """Rebuild a scheme from :meth:`to_record` output."""
        return cls(
            scheme_id=record["scheme_id"],
            task=record["task"],
            unit=SplitUnit(record["unit"]),
            rationale=record["rationale"],
            splits=tuple(ReferenceSplit.from_record(item) for item in record["splits"]),
            adjacency_window_ns=record["adjacency_window_ns"],
        )


@dataclass(frozen=True, kw_only=True)
class ReferenceSetIdentity:
    """The identity an evaluation report carries to name the reference set it used."""

    reference_set_id: str
    version: str
    digest: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "reference_set_id": self.reference_set_id,
            "version": self.version,
            "digest": self.digest,
        }


@dataclass(frozen=True, kw_only=True)
class ReferenceSetManifest:
    """The canonical, versioned description of a Solution 1 reference set.

    Attributes:
        reference_set_id: Stable name of the reference set across versions.
        version: Version; it must change whenever annotations or selections do.
        sources: The bodies of data the samples come from.
        calibrations: The calibration identities samples refer to.
        stratum_definitions: The declared stratification factors.
        samples: The ordered evaluation samples.
        provenance: How each annotation was produced.
        annotations: The annotation files, with schema, trust and provenance.
        split_schemes: The tuning/development/test schemes, one per task.
    """

    reference_set_id: str
    version: str
    sources: tuple[ReferenceSource, ...]
    calibrations: tuple[CalibrationIdentity, ...]
    stratum_definitions: tuple[StratumDefinition, ...]
    samples: tuple[ReferenceSample, ...]
    provenance: tuple[AnnotationProvenance, ...]
    annotations: tuple[AnnotationFileEntry, ...]
    split_schemes: tuple[SplitScheme, ...]

    def __post_init__(self) -> None:
        """Require unique identities and references that resolve inside the manifest."""
        require_text("reference_set_id", self.reference_set_id)
        require_text("reference-set version", self.version)
        if not self.sources:
            raise ValueError("a reference set needs at least one source")
        if not self.samples:
            raise ValueError("a reference set needs at least one sample")
        require_unique("source id", (item.source_id for item in self.sources))
        require_unique("calibration id", (item.calibration_id for item in self.calibrations))
        require_unique("stratum name", (item.name for item in self.stratum_definitions))
        require_unique("sample id", (item.sample_id for item in self.samples))
        require_unique("provenance id", (item.provenance_id for item in self.provenance))
        require_unique("annotation id", (item.annotation_id for item in self.annotations))
        require_unique("split scheme id", (item.scheme_id for item in self.split_schemes))
        self._check_references()

    def _check_references(self) -> None:
        source_ids = {item.source_id for item in self.sources}
        calibration_ids = {item.calibration_id for item in self.calibrations}
        sample_ids = {item.sample_id for item in self.samples}
        provenance_by_id = {item.provenance_id: item for item in self.provenance}
        allowed_strata = {item.name: set(item.values) for item in self.stratum_definitions}

        for calibration in self.calibrations:
            if calibration.source_id not in source_ids:
                raise ValueError(
                    f"calibration {calibration.calibration_id!r} references unknown source "
                    f"{calibration.source_id!r}"
                )
        for sample in self.samples:
            if sample.source_id not in source_ids:
                raise ValueError(
                    f"sample {sample.sample_id!r} references unknown source {sample.source_id!r}"
                )
            for calibration_id in sample.calibration_ids:
                if calibration_id not in calibration_ids:
                    raise ValueError(
                        f"sample {sample.sample_id!r} references unknown calibration "
                        f"{calibration_id!r}"
                    )
            for stratum in sample.strata:
                if stratum.value not in allowed_strata.get(stratum.name, set()):
                    raise ValueError(
                        f"sample {sample.sample_id!r} uses undeclared stratum "
                        f"{stratum.name!r}={stratum.value!r}"
                    )
        for entry in self.annotations:
            provenance = provenance_by_id.get(entry.provenance_id)
            if provenance is None:
                raise ValueError(
                    f"annotation {entry.annotation_id!r} references unknown provenance "
                    f"{entry.provenance_id!r}"
                )
            if (
                provenance.origin is ProvenanceOrigin.MODEL_INFERENCE
                and entry.trust is not ReferenceTrust.DIAGNOSTIC_ONLY
            ):
                raise ValueError(
                    f"annotation {entry.annotation_id!r} comes from model inference and can "
                    "only be declared diagnostic_only"
                )
            for sample_id in entry.sample_ids:
                if sample_id not in sample_ids:
                    raise ValueError(
                        f"annotated sample {sample_id!r} of annotation "
                        f"{entry.annotation_id!r} is not in the manifest"
                    )
        for scheme in self.split_schemes:
            for split in scheme.splits:
                for sample_id in split.sample_ids:
                    if sample_id not in sample_ids:
                        raise ValueError(
                            f"split sample {sample_id!r} of split {split.name!r} in scheme "
                            f"{scheme.scheme_id!r} is not in the manifest"
                        )

    def selection(self, scheme_id: str, split_name: str) -> tuple[ReferenceSample, ...]:
        """Return the exact ordered samples of one split.

        Raises:
            ReferenceSetError: If the scheme or the split does not exist.
        """
        scheme = next((item for item in self.split_schemes if item.scheme_id == scheme_id), None)
        if scheme is None:
            raise ReferenceSetError(f"unknown split scheme {scheme_id!r}")
        split = next((item for item in scheme.splits if item.name == split_name), None)
        if split is None:
            raise ReferenceSetError(f"scheme {scheme_id!r} has no split {split_name!r}")
        by_id = {item.sample_id: item for item in self.samples}
        return tuple(by_id[sample_id] for sample_id in split.sample_ids)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": REFERENCE_SET_SCHEMA,
            "reference_set_id": self.reference_set_id,
            "version": self.version,
            "sources": [item.to_record() for item in self.sources],
            "calibrations": [item.to_record() for item in self.calibrations],
            "stratum_definitions": [item.to_record() for item in self.stratum_definitions],
            "samples": [item.to_record() for item in self.samples],
            "provenance": [item.to_record() for item in self.provenance],
            "annotations": [item.to_record() for item in self.annotations],
            "split_schemes": [item.to_record() for item in self.split_schemes],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReferenceSetManifest:
        """Rebuild a manifest from :meth:`to_record` output."""
        return cls(
            reference_set_id=record["reference_set_id"],
            version=record["version"],
            sources=tuple(ReferenceSource.from_record(item) for item in record["sources"]),
            calibrations=tuple(
                CalibrationIdentity.from_record(item) for item in record["calibrations"]
            ),
            stratum_definitions=tuple(
                StratumDefinition.from_record(item) for item in record["stratum_definitions"]
            ),
            samples=tuple(ReferenceSample.from_record(item) for item in record["samples"]),
            provenance=tuple(
                AnnotationProvenance.from_record(item) for item in record["provenance"]
            ),
            annotations=tuple(
                AnnotationFileEntry.from_record(item) for item in record["annotations"]
            ),
            split_schemes=tuple(SplitScheme.from_record(item) for item in record["split_schemes"]),
        )

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the whole manifest content."""
        return canonical_digest(self.to_record())

    def identity(self) -> ReferenceSetIdentity:
        """Return the identity evaluation reports use to cite this reference set."""
        return ReferenceSetIdentity(
            reference_set_id=self.reference_set_id, version=self.version, digest=self.digest()
        )


def encode_reference_set(manifest: ReferenceSetManifest) -> dict[str, Any]:
    """Return the manifest document, including its own digest."""
    return {**manifest.to_record(), "digest": manifest.digest()}


def decode_reference_set(document: Mapping[str, Any]) -> ReferenceSetManifest:
    """Rebuild a manifest and verify its declared digest.

    Raises:
        ReferenceSetError: If the schema is unknown, a field is missing or
            invalid, or the declared digest does not match the content.
    """
    try:
        schema = document["schema"]
        if schema != REFERENCE_SET_SCHEMA:
            raise ReferenceSetError(
                f"unsupported reference-set schema {schema!r}; expected {REFERENCE_SET_SCHEMA!r}"
            )
        declared = document["digest"]
        manifest = ReferenceSetManifest.from_record(document)
    except ReferenceSetError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceSetError(f"invalid reference-set manifest: {error}") from error
    if manifest.digest() != declared:
        raise ReferenceSetError(
            f"reference-set digest mismatch: declared {declared!r}, computed {manifest.digest()!r}"
        )
    return manifest


def write_reference_set(root: Path, manifest: ReferenceSetManifest) -> None:
    """Atomically publish an immutable manifest under ``root``.

    Raises:
        FileExistsError: If a manifest already exists there; a changed
            reference set is published under a new version directory.
    """
    write_immutable_json(
        root / MANIFEST_FILENAME, encode_reference_set(manifest), "reference-set manifest"
    )


def read_reference_set(root: Path) -> ReferenceSetManifest:
    """Read the manifest under ``root`` and verify its digest.

    Raises:
        ReferenceSetError: If the document is not valid JSON or fails
            :func:`decode_reference_set`.
    """
    text = (root / MANIFEST_FILENAME).read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ReferenceSetError(f"reference-set manifest is not valid JSON: {error}") from error
    return decode_reference_set(document)


def verify_annotation_files(manifest: ReferenceSetManifest, root: Path) -> None:
    """Check every annotation file against the hash the manifest declares.

    Raises:
        ReferenceSetError: Listing every missing, escaping or modified file.
    """
    resolved_root = root.resolve()
    problems: list[str] = []
    for entry in manifest.annotations:
        path = (root / entry.path).resolve()
        if not path.is_relative_to(resolved_root):
            problems.append(f"annotation file escapes the reference-set root: {entry.path}")
        elif not path.is_file():
            problems.append(f"annotation file missing: {entry.path}")
        elif file_digest(path) != entry.content_hash:
            problems.append(f"annotation file hash mismatch: {entry.path}")
    if problems:
        raise ReferenceSetError("; ".join(problems))


def require_version_bump_on_change(
    previous: ReferenceSetManifest, current: ReferenceSetManifest
) -> None:
    """Refuse a manifest whose content changed while its version did not.

    Raises:
        ReferenceSetError: If the manifests name different reference sets, or
            share a version but not a digest.
    """
    if previous.reference_set_id != current.reference_set_id:
        raise ReferenceSetError(
            f"reference_set_id differs: {previous.reference_set_id!r} vs "
            f"{current.reference_set_id!r}"
        )
    if previous.version == current.version and previous.digest() != current.digest():
        raise ReferenceSetError(
            f"reference set {current.reference_set_id!r} changed content without a new version "
            f"({current.version!r})"
        )
