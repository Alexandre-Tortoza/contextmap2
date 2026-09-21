"""Adapters from the existing stage reports into the common envelope."""

import json

import pytest
from reference_set_builders import make_valid_manifest

from contextmap.evaluation.metrics import EvaluationStage, default_metric_registry
from contextmap.evaluation.reference_set import ReferenceSetIdentity
from contextmap.evaluation.region_discovery import (
    DiscoveryDiagnosticsMetrics,
    EvaluationRunDescriptor,
    FrameEvaluation,
    PerformanceMetrics,
    RegionDiscoveryEvaluationReport,
    SegmentationAccuracy,
)
from contextmap.evaluation.report_adapters import (
    region_discovery_evaluation_report,
    semantic_interpretation_evaluation_report,
    wrap_stage_report,
)
from contextmap.evaluation.report_schema import (
    ArtifactIdentity,
    EvaluatorIdentity,
    MetricResult,
    MetricStatus,
    ReproducibilityMetadata,
    decode_evaluation_report,
    encode_evaluation_report,
)
from contextmap.evaluation.semantic_interpretation import (
    SemanticCostReport,
    SemanticEvaluationContext,
    SemanticEvaluationReport,
    SemanticQualityReport,
)

REGISTRY = default_metric_registry()


def _reference() -> ReferenceSetIdentity:
    return make_valid_manifest().identity()


def _run() -> EvaluationRunDescriptor:
    return EvaluationRunDescriptor(
        perception_run_id="run-0001",
        perception_artifact_id="artifact-0001",
        backend_id="sam3",
        backend_version="1",
        checkpoint="sam3-base",
        config_digest="sha256:" + "b" * 64,
        pipeline_graph_digest="sha256:" + "c" * 64,
        strategy="single-pass",
        thresholds=(("match_iou", 0.5),),
        variables=(),
        execution_kind="canned",
    )


def _frame(
    name: str, iou: float | None, *, runtime_ms: float, memory_mb: float | None
) -> FrameEvaluation:
    accuracy = (
        None
        if iou is None
        else SegmentationAccuracy(
            mean_iou=iou,
            mean_dice=iou,
            region_recall=iou / 2,
            coverage=iou,
            over_segmentation_rate=0.0,
            under_segmentation_rate=0.0,
            duplicate_region_rate=0.1,
        )
    )
    return FrameEvaluation(
        frame_id=name,
        source_condition="synthetic",
        diagnostics=DiscoveryDiagnosticsMetrics(
            raw_candidate_count=3,
            accepted_region_count=2,
            rejected_candidate_count=1,
            duplicate_merge_ratio=0.0,
            area_pixels=(10.0, 20.0),
            invalid_geometry_count=0,
            constraint_violation_count=0,
        ),
        performance=PerformanceMetrics(
            runtime_ms=runtime_ms, per_pass_runtime_ms=(runtime_ms,), peak_memory_mb=memory_mb
        ),
        accuracy=accuracy,
    )


def _region_report(*frames: FrameEvaluation) -> RegionDiscoveryEvaluationReport:
    return RegionDiscoveryEvaluationReport(
        schema="contextmap.region-discovery-evaluation/v1",
        metric_schema_version="1.0.0",
        reference_set_version="ci-subset/1.0.0",
        frame_selection=tuple(frame.frame_id for frame in frames),
        run=_run(),
        frames=frames,
    )


def _by_metric(results: tuple[MetricResult, ...]) -> dict[str, MetricResult]:
    return {item.metric: item for item in results}


def test_region_discovery_report_keeps_quality_and_performance_apart() -> None:
    report = _region_report(
        _frame("frame-a", 0.8, runtime_ms=1500.0, memory_mb=100.0),
        _frame("frame-b", 0.4, runtime_ms=500.0, memory_mb=250.0),
        _frame("frame-c", None, runtime_ms=1000.0, memory_mb=None),
    )

    lifted = region_discovery_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference(), code_version="0.0.1"
    )

    quality = _by_metric(lifted.quality_metrics)
    performance = _by_metric(lifted.performance_metrics)
    assert lifted.stage is EvaluationStage.REGION_DISCOVERY
    assert quality["region.iou.mean"].value == pytest.approx(0.6)
    assert quality["region.iou.mean"].sample_count == 2
    assert quality["region.recall.mean"].value == pytest.approx(0.3)
    assert quality["region.duplicate_rate.mean"].value == pytest.approx(0.1)
    assert performance["runtime.wall_time"].value == pytest.approx(3.0)
    assert performance["runtime.wall_time"].sample_count == 3
    assert performance["runtime.peak_memory"].value == pytest.approx(250_000_000.0)
    assert set(quality).isdisjoint(performance)


def test_region_discovery_report_carries_reproducibility_and_the_stage_report() -> None:
    report = _region_report(_frame("frame-a", 0.8, runtime_ms=10.0, memory_mb=None))

    lifted = region_discovery_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    metadata = lifted.reproducibility
    assert metadata.reference_set == _reference()
    assert metadata.configuration_digest == "sha256:" + "b" * 64
    assert [item.artifact_id for item in metadata.input_artifacts] == ["run-0001", "artifact-0001"]
    assert metadata.metric_registry == REGISTRY.identity()
    assert lifted.stage_report == report.to_dict()
    document = json.loads(json.dumps(encode_evaluation_report(lifted)))
    assert decode_evaluation_report(document, REGISTRY) == lifted


def test_unannotated_frames_are_not_applicable_never_zero() -> None:
    report = _region_report(_frame("frame-a", None, runtime_ms=10.0, memory_mb=None))

    lifted = region_discovery_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    assert all(item.status is MetricStatus.NOT_APPLICABLE for item in quality.values())
    assert all(item.value is None for item in quality.values())
    peak = _by_metric(lifted.performance_metrics)["runtime.peak_memory"]
    assert peak.status is MetricStatus.UNSUPPORTED


def _semantic_report(
    *, claim_count: int = 4, ambiguity: float | None = 0.5, peak: int | None = 2048
) -> SemanticEvaluationReport:
    return SemanticEvaluationReport(
        context=SemanticEvaluationContext(
            evaluation_id="semantic-eval-0001",
            reference_set_version="ci-subset/1.0.0",
            selection_id="selection-0001",
            perception_run_id="run-0001",
            artifact_id="artifact-0001",
            pipeline_configuration_digest="sha256:" + "d" * 64,
            evaluator_version="semantic-evaluator/1",
        ),
        matching_policy="casefold-exact/1",
        samples=(),
        failures=(),
        quality=SemanticQualityReport(
            claim_count=claim_count,
            acceptable_claim_rate=0.75,
            unsupported_claim_rate=0.25,
            ambiguity_preservation_rate=ambiguity,
            abstention_count=1,
            duplicate_claim_count=0,
        ),
        cost=SemanticCostReport(
            request_count=2,
            total_latency_ms=800.0,
            retry_count=0,
            input_tokens=10,
            output_tokens=5,
            peak_memory_bytes=peak,
        ),
    )


def test_semantic_report_becomes_quality_and_performance_metrics() -> None:
    lifted = semantic_interpretation_evaluation_report(
        _semantic_report(), registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    performance = _by_metric(lifted.performance_metrics)
    assert lifted.stage is EvaluationStage.SEMANTIC_INTERPRETATION
    assert quality["semantic.acceptable_claim_rate"].value == 0.75
    assert quality["semantic.acceptable_claim_rate"].sample_count == 4
    assert quality["semantic.unsupported_claim_rate"].value == 0.25
    assert quality["semantic.ambiguity_preservation_rate"].value == 0.5
    assert performance["runtime.wall_time"].value == pytest.approx(0.8)
    assert performance["runtime.peak_memory"].value == 2048.0
    assert lifted.stage_report is not None
    assert lifted.stage_report["quality"]["claim_count"] == 4
    assert lifted.reproducibility.evaluator.evaluator_version == "semantic-evaluator/1"


def test_semantic_report_without_ambiguous_annotations_or_claims_stays_not_applicable() -> None:
    lifted = semantic_interpretation_evaluation_report(
        _semantic_report(claim_count=0, ambiguity=None, peak=None),
        registry=REGISTRY,
        reference_set=_reference(),
    )

    quality = _by_metric(lifted.quality_metrics)
    assert all(item.status is MetricStatus.NOT_APPLICABLE for item in quality.values())
    assert _by_metric(lifted.performance_metrics)["runtime.peak_memory"].status is (
        MetricStatus.UNSUPPORTED
    )


def test_any_stage_report_can_ride_in_the_envelope_with_shared_metadata() -> None:
    metadata = ReproducibilityMetadata(
        evaluator=EvaluatorIdentity(
            evaluator_id="state-estimation-evaluator", evaluator_version="1"
        ),
        reference_set=None,
        annotation_schemas=(),
        input_artifacts=(
            ArtifactIdentity(kind="trajectory", artifact_id="traj-0001", digest=None),
        ),
        configuration_digest=None,
        code_version="0.0.1",
        metric_registry=REGISTRY.identity(),
    )

    wrapped = wrap_stage_report(
        REGISTRY,
        stage=EvaluationStage.STATE_ESTIMATION,
        stage_report={"schema": "contextmap.state-estimation-evaluation/v1", "accuracy": {}},
        reproducibility=metadata,
    )

    assert wrapped.quality_metrics == () and wrapped.performance_metrics == ()
    assert wrapped.stage_report is not None
    assert wrapped.reproducibility == metadata
