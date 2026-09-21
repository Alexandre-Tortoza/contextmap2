"""Common evaluation report envelope shared by every stage.

Each capability keeps its own report with its own semantics. The envelope adds
what all of them must share: which stage was evaluated, the reproducibility
metadata (evaluator, reference set, annotation schemas, input artifacts,
configuration, code and metric registry) and the metric results, split into
**quality** and **performance** so cost is never mistaken for correctness. The
stage-specific report travels inside the envelope untouched. There is no
overall score. See ``src/contextmap/evaluation/docs/metrics.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import write_immutable_json
from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.metrics import (
    EvaluationStage,
    MetricDefinition,
    MetricKind,
    MetricRegistry,
    MetricRegistryIdentity,
    require_annotation_compatibility,
)
from contextmap.evaluation.reference_set import ReferenceSetIdentity

REPORT_SCHEMA = "contextmap.evaluation-report/v1"
"""Schema identifier of the common evaluation report envelope."""


class EvaluationReportError(ValueError):
    """Raised when an evaluation report is not consistent with its registry."""


class MetricStatus(Enum):
    """What an evaluator could say about a metric."""

    VALUE = "value"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, kw_only=True)
class MetricResult:
    """The outcome of one metric, for the whole report or for one stratum.

    ``NOT_APPLICABLE`` (annotations or population missing) and ``UNSUPPORTED``
    (the evaluator or backend cannot produce it) carry no value: a missing
    result is never reported as zero.

    Attributes:
        metric: Metric name in the registry.
        metric_version: Metric version in the registry.
        status: Value, not applicable or unsupported.
        value: The value; present exactly when ``status`` is ``VALUE``.
        sample_count: Size of the population the value is computed over.
        strata: ``(factor, value)`` pairs when the result is for one stratum.
    """

    metric: str
    metric_version: str
    status: MetricStatus
    value: float | None
    sample_count: int | None = None
    strata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Tie the value to the status and require sane counts and strata."""
        require_text("metric", self.metric)
        require_text("metric_version", self.metric_version)
        if self.status is MetricStatus.VALUE:
            if self.value is None:
                raise ValueError(f"a value result of {self.metric!r} needs a value")
            if not math.isfinite(self.value):
                raise ValueError(f"the value of {self.metric!r} must be finite")
        elif self.value is not None:
            raise ValueError(
                f"a {self.status.value} result of {self.metric!r} must not carry a value"
            )
        if self.sample_count is not None and self.sample_count < 0:
            raise ValueError("sample_count must not be negative")
        require_unique(f"stratum factor of {self.metric!r}", (name for name, _ in self.strata))

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "metric": self.metric,
            "metric_version": self.metric_version,
            "status": self.status.value,
            "value": self.value,
            "sample_count": self.sample_count,
            "strata": [list(item) for item in self.strata],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> MetricResult:
        """Rebuild a result from :meth:`to_record` output."""
        return cls(
            metric=record["metric"],
            metric_version=record["metric_version"],
            status=MetricStatus(record["status"]),
            value=record["value"],
            sample_count=record["sample_count"],
            strata=tuple((name, value) for name, value in record["strata"]),
        )


@dataclass(frozen=True, kw_only=True)
class EvaluatorIdentity:
    """The evaluator that produced a report, and its version."""

    evaluator_id: str
    evaluator_version: str

    def __post_init__(self) -> None:
        """Require both parts."""
        require_text("evaluator_id", self.evaluator_id)
        require_text("evaluator_version", self.evaluator_version)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"evaluator_id": self.evaluator_id, "evaluator_version": self.evaluator_version}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> EvaluatorIdentity:
        """Rebuild an identity from :meth:`to_record` output."""
        return cls(
            evaluator_id=record["evaluator_id"], evaluator_version=record["evaluator_version"]
        )


@dataclass(frozen=True, kw_only=True)
class ArtifactIdentity:
    """An artifact the evaluation read: a run, a map, a sequence, a reference."""

    kind: str
    artifact_id: str
    digest: str | None

    def __post_init__(self) -> None:
        """Require what identifies the artifact."""
        require_text("artifact kind", self.kind)
        require_text("artifact_id", self.artifact_id)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"kind": self.kind, "artifact_id": self.artifact_id, "digest": self.digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ArtifactIdentity:
        """Rebuild an identity from :meth:`to_record` output."""
        return cls(kind=record["kind"], artifact_id=record["artifact_id"], digest=record["digest"])


@dataclass(frozen=True, kw_only=True)
class ReproducibilityMetadata:
    """Everything needed to reproduce, or to refuse to compare, an evaluation.

    Attributes:
        evaluator: The evaluator and its version.
        reference_set: Identity of the reference set the metrics used; ``None``
            only for stages whose metrics need no reference.
        annotation_schemas: Exact annotation schemas that were available.
        input_artifacts: The artifacts the evaluation read.
        configuration_digest: Digest of the evaluated configuration, if any.
        code_version: Version of the code that produced the evaluated output.
        metric_registry: Identity of the registry the metrics come from.
    """

    evaluator: EvaluatorIdentity
    reference_set: ReferenceSetIdentity | None
    annotation_schemas: tuple[str, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    configuration_digest: str | None
    code_version: str | None
    metric_registry: MetricRegistryIdentity

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "evaluator": self.evaluator.to_record(),
            "reference_set": None if self.reference_set is None else self.reference_set.to_record(),
            "annotation_schemas": list(self.annotation_schemas),
            "input_artifacts": [item.to_record() for item in self.input_artifacts],
            "configuration_digest": self.configuration_digest,
            "code_version": self.code_version,
            "metric_registry": self.metric_registry.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReproducibilityMetadata:
        """Rebuild the metadata from :meth:`to_record` output."""
        reference = record["reference_set"]
        return cls(
            evaluator=EvaluatorIdentity.from_record(record["evaluator"]),
            reference_set=None
            if reference is None
            else ReferenceSetIdentity(
                reference_set_id=reference["reference_set_id"],
                version=reference["version"],
                digest=reference["digest"],
            ),
            annotation_schemas=tuple(record["annotation_schemas"]),
            input_artifacts=tuple(
                ArtifactIdentity.from_record(item) for item in record["input_artifacts"]
            ),
            configuration_digest=record["configuration_digest"],
            code_version=record["code_version"],
            metric_registry=MetricRegistryIdentity.from_record(record["metric_registry"]),
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationReport:
    """One stage's evaluation: shared metadata, separate quality and performance.

    Build it with :func:`assemble_evaluation_report`, which checks every metric
    against the registry.

    Attributes:
        stage: The stage that was evaluated.
        reproducibility: The shared reproducibility metadata.
        quality_metrics: Correctness results; never performance.
        performance_metrics: Cost results; never quality.
        stage_report: The capability's own report, kept untouched.
    """

    stage: EvaluationStage
    reproducibility: ReproducibilityMetadata
    quality_metrics: tuple[MetricResult, ...]
    performance_metrics: tuple[MetricResult, ...]
    stage_report: Mapping[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "stage": self.stage.value,
            "reproducibility": self.reproducibility.to_record(),
            "quality_metrics": [item.to_record() for item in self.quality_metrics],
            "performance_metrics": [item.to_record() for item in self.performance_metrics],
            "stage_report": self.stage_report,
        }


def validate_evaluation_report(report: EvaluationReport, registry: MetricRegistry) -> None:
    """Check a report against the registry it claims to use.

    Raises:
        EvaluationReportError: If a metric is in the wrong container or stage,
            a value is out of range, a result is repeated, the registry differs
            from the one the report cites, or a stage that needs a reference set
            has none.
        MetricRegistryError: If a metric or version is unknown.
        MetricCompatibilityError: If a value needs annotation schemas that are
            not among the available ones.
    """
    if report.reproducibility.metric_registry != registry.identity():
        raise EvaluationReportError(
            "the report cites a different metric registry than the one it is validated against"
        )
    if report.stage is EvaluationStage.RUNTIME and report.quality_metrics:
        raise EvaluationReportError("a runtime report carries performance metrics only")
    _check_results(report, registry, report.quality_metrics, MetricKind.QUALITY)
    _check_results(report, registry, report.performance_metrics, MetricKind.PERFORMANCE)


def _check_results(
    report: EvaluationReport,
    registry: MetricRegistry,
    results: tuple[MetricResult, ...],
    kind: MetricKind,
) -> None:
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    for result in results:
        definition = registry.get(result.metric, result.metric_version)
        if definition.kind is not kind:
            raise EvaluationReportError(
                f"metric {definition.key} is a {definition.kind.value} metric and cannot be "
                f"reported as {kind.value}"
            )
        if definition.stage is not None and definition.stage is not report.stage:
            raise EvaluationReportError(
                f"metric {definition.key} belongs to stage {definition.stage.value!r}, not "
                f"{report.stage.value!r}"
            )
        key = (result.metric, result.metric_version, tuple(sorted(result.strata)))
        if key in seen:
            raise EvaluationReportError(
                f"metric {definition.key} is repeated for the same strata {result.strata}"
            )
        seen.add(key)
        if result.status is MetricStatus.VALUE:
            _check_value(report, definition, result)


def _check_value(
    report: EvaluationReport, definition: MetricDefinition, result: MetricResult
) -> None:
    assert result.value is not None
    if definition.minimum is not None and result.value < definition.minimum:
        raise EvaluationReportError(
            f"{definition.key} value {result.value} is outside its range "
            f"[{definition.minimum}, {definition.maximum}]"
        )
    if definition.maximum is not None and result.value > definition.maximum:
        raise EvaluationReportError(
            f"{definition.key} value {result.value} is outside its range "
            f"[{definition.minimum}, {definition.maximum}]"
        )
    if definition.required_annotations:
        if report.reproducibility.reference_set is None:
            raise EvaluationReportError(
                f"{definition.key} needs annotations, so the report must cite a reference set"
            )
        require_annotation_compatibility(definition, report.reproducibility.annotation_schemas)


def assemble_evaluation_report(
    registry: MetricRegistry,
    *,
    stage: EvaluationStage,
    reproducibility: ReproducibilityMetadata,
    quality_metrics: tuple[MetricResult, ...],
    performance_metrics: tuple[MetricResult, ...],
    stage_report: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Build a report and validate it against the registry.

    Raises:
        EvaluationReportError: See :func:`validate_evaluation_report`.
        MetricRegistryError: If a metric or version is unknown.
        MetricCompatibilityError: If a value's annotation schemas are unavailable.
    """
    report = EvaluationReport(
        stage=stage,
        reproducibility=reproducibility,
        quality_metrics=quality_metrics,
        performance_metrics=performance_metrics,
        stage_report=stage_report,
    )
    validate_evaluation_report(report, registry)
    return report


def encode_evaluation_report(report: EvaluationReport) -> dict[str, Any]:
    """Return the report document, tagged with its schema."""
    return {"schema": REPORT_SCHEMA, **report.to_record()}


def decode_evaluation_report(
    document: Mapping[str, Any], registry: MetricRegistry
) -> EvaluationReport:
    """Rebuild a report and validate it against the registry again.

    Raises:
        EvaluationReportError: If the schema is unsupported, a field is missing
            or the report is inconsistent with the registry.
    """
    try:
        if document["schema"] != REPORT_SCHEMA:
            raise EvaluationReportError(
                f"unsupported evaluation report schema {document['schema']!r}"
            )
        report = EvaluationReport(
            stage=EvaluationStage(document["stage"]),
            reproducibility=ReproducibilityMetadata.from_record(document["reproducibility"]),
            quality_metrics=tuple(
                MetricResult.from_record(item) for item in document["quality_metrics"]
            ),
            performance_metrics=tuple(
                MetricResult.from_record(item) for item in document["performance_metrics"]
            ),
            stage_report=document["stage_report"],
        )
    except EvaluationReportError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationReportError(f"invalid evaluation report: {error}") from error
    validate_evaluation_report(report, registry)
    return report


def write_evaluation_report(path: Path, report: EvaluationReport) -> None:
    """Atomically publish an immutable report file.

    Raises:
        FileExistsError: If the file already exists.
    """
    write_immutable_json(path, encode_evaluation_report(report), "evaluation report")
