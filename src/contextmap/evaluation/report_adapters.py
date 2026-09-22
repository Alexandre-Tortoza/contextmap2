"""Adapters that lift the existing stage reports into the common report envelope.

The per-capability evaluators already compute their own metrics. These adapters
do not recompute anything: they read what a stage report contains, express the
headline numbers as registry metrics with an explicit population size, and keep
the stage report itself inside the envelope. A metric with no annotated
population becomes ``NOT_APPLICABLE``, never zero. See
``src/contextmap/evaluation/docs/metrics.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import fmean
from typing import Any

from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.metrics import EvaluationStage, MetricRegistry
from contextmap.evaluation.reference_set import ReferenceSetIdentity
from contextmap.evaluation.region_discovery import RegionDiscoveryEvaluationReport
from contextmap.evaluation.report_schema import (
    ArtifactIdentity,
    EvaluationReport,
    EvaluatorIdentity,
    MetricResult,
    MetricStatus,
    ReproducibilityMetadata,
    assemble_evaluation_report,
)
from contextmap.evaluation.semantic_interpretation import (
    SemanticEvaluationReport,
    encode_semantic_evaluation_report,
)

_BYTES_PER_MB = 1_000_000


def _value(name: str, value: float, count: int) -> MetricResult:
    return MetricResult(
        metric=name,
        metric_version="1",
        status=MetricStatus.VALUE,
        value=value,
        sample_count=count,
    )


def _unavailable(name: str, status: MetricStatus, count: int | None = 0) -> MetricResult:
    return MetricResult(
        metric=name, metric_version="1", status=status, value=None, sample_count=count
    )


def wrap_stage_report(
    registry: MetricRegistry,
    *,
    stage: EvaluationStage,
    stage_report: Mapping[str, Any],
    reproducibility: ReproducibilityMetadata,
) -> EvaluationReport:
    """Carry any stage's own encoded report in the envelope, without typed metrics.

    This gives every existing harness the shared reproducibility metadata today;
    an adapter that also lifts its headline numbers into registry metrics can
    replace the call without changing the stage report.
    """
    return assemble_evaluation_report(
        registry,
        stage=stage,
        reproducibility=reproducibility,
        quality_metrics=(),
        performance_metrics=(),
        stage_report=stage_report,
    )


def region_discovery_evaluation_report(
    report: RegionDiscoveryEvaluationReport,
    *,
    registry: MetricRegistry,
    reference_set: ReferenceSetIdentity,
    code_version: str | None = None,
) -> EvaluationReport:
    """Lift a Region Discovery report into the envelope.

    Quality metrics are means over the frames that have a regions annotation;
    frames without one are excluded from the population, not scored as zero.
    """
    annotated = [frame.accuracy for frame in report.frames if frame.accuracy is not None]
    quality: list[MetricResult] = []
    for name, values in (
        ("region.iou.mean", [item.mean_iou for item in annotated]),
        ("region.recall.mean", [item.region_recall for item in annotated]),
        ("region.duplicate_rate.mean", [item.duplicate_region_rate for item in annotated]),
    ):
        quality.append(
            _value(name, fmean(values), len(values))
            if values
            else _unavailable(name, MetricStatus.NOT_APPLICABLE)
        )
    memory_mb = [
        frame.performance.peak_memory_mb
        for frame in report.frames
        if frame.performance.peak_memory_mb is not None
    ]
    count = len(report.frames)
    performance = [
        _value(
            "runtime.wall_time",
            sum(frame.performance.runtime_ms for frame in report.frames) / 1000.0,
            count,
        ),
        _value("runtime.peak_memory", max(memory_mb) * _BYTES_PER_MB, count)
        if memory_mb
        else _unavailable("runtime.peak_memory", MetricStatus.UNSUPPORTED, None),
    ]
    run = report.run
    return assemble_evaluation_report(
        registry,
        stage=EvaluationStage.REGION_DISCOVERY,
        reproducibility=ReproducibilityMetadata(
            evaluator=EvaluatorIdentity(
                evaluator_id="region-discovery-evaluator",
                evaluator_version=report.metric_schema_version,
            ),
            reference_set=reference_set,
            annotation_schemas=(AnnotationFamily.REGIONS.schema,),
            input_artifacts=(
                ArtifactIdentity(
                    kind="perception_run", artifact_id=run.perception_run_id, digest=None
                ),
                ArtifactIdentity(
                    kind="perception_artifact", artifact_id=run.perception_artifact_id, digest=None
                ),
            ),
            configuration_digest=run.config_digest,
            code_version=code_version,
            metric_registry=registry.identity(),
        ),
        quality_metrics=tuple(quality),
        performance_metrics=tuple(performance),
        stage_report=report.to_dict(),
    )


def semantic_interpretation_evaluation_report(
    report: SemanticEvaluationReport,
    *,
    registry: MetricRegistry,
    reference_set: ReferenceSetIdentity,
    code_version: str | None = None,
) -> EvaluationReport:
    """Lift a Semantic Interpretation report into the envelope.

    Rates are over the *assessed* claims (``quality.assessed_claim_count``), never
    the raw claim count: a run can have real claims and no human annotation, so
    those claims are unassessed, not wrong, and the rate is ``NOT_APPLICABLE``
    rather than a fabricated value. The ambiguity-preservation rate's population
    is the primary-run (``repeat_index == 0``) requests with an ambiguous
    annotation; repeats measure stability, not additional physical evidence, and
    must not inflate it. The full semantic report (samples, failures, quality,
    cost, outcomes, consistency and per-stratum quality) travels in the stage
    report through the capability's own encoder, so an ``Enum`` field it
    introduces (for example ``SemanticStratum.source``) stays JSON-safe.
    """
    quality = report.quality
    context = report.context
    if quality.acceptable_claim_rate is not None and quality.unsupported_claim_rate is not None:
        quality_metrics = [
            _value(
                "semantic.acceptable_claim_rate",
                quality.acceptable_claim_rate,
                quality.assessed_claim_count,
            ),
            _value(
                "semantic.unsupported_claim_rate",
                quality.unsupported_claim_rate,
                quality.assessed_claim_count,
            ),
        ]
    else:
        quality_metrics = [
            _unavailable("semantic.acceptable_claim_rate", MetricStatus.NOT_APPLICABLE),
            _unavailable("semantic.unsupported_claim_rate", MetricStatus.NOT_APPLICABLE),
        ]
    ambiguity = quality.ambiguity_preservation_rate
    # A população é só a dos requests primários (repeat_index == 0) cujo alvo tem registro
    # ambíguo: repeats medem estabilidade, não evidência física adicional, e não podem
    # inflar a população usada para calcular a taxa (ver o registro de métricas).
    ambiguous_request_count = sum(
        item.ambiguity_preserved is not None for item in report.samples if item.repeat_index == 0
    )
    quality_metrics.append(
        _value("semantic.ambiguity_preservation_rate", ambiguity, ambiguous_request_count)
        if ambiguity is not None
        else _unavailable("semantic.ambiguity_preservation_rate", MetricStatus.NOT_APPLICABLE)
    )
    cost = report.cost
    performance = [
        _value("runtime.wall_time", cost.total_latency_ms / 1000.0, cost.request_count),
        _value("runtime.peak_memory", float(cost.peak_memory_bytes), cost.request_count)
        if cost.peak_memory_bytes is not None
        else _unavailable("runtime.peak_memory", MetricStatus.UNSUPPORTED, None),
    ]
    return assemble_evaluation_report(
        registry,
        stage=EvaluationStage.SEMANTIC_INTERPRETATION,
        reproducibility=ReproducibilityMetadata(
            evaluator=EvaluatorIdentity(
                evaluator_id="semantic-interpretation-evaluator",
                evaluator_version=context.evaluator_version,
            ),
            reference_set=reference_set,
            annotation_schemas=(AnnotationFamily.SEMANTICS.schema,),
            input_artifacts=(
                ArtifactIdentity(
                    kind="perception_run", artifact_id=context.perception_run_id, digest=None
                ),
                ArtifactIdentity(kind="artifact", artifact_id=context.artifact_id, digest=None),
            ),
            configuration_digest=context.pipeline_configuration_digest,
            code_version=code_version,
            metric_registry=registry.identity(),
        ),
        quality_metrics=tuple(quality_metrics),
        performance_metrics=tuple(performance),
        stage_report=encode_semantic_evaluation_report(report),
    )
