"""Adapters from the existing stage reports into the common envelope."""

import json
from pathlib import Path

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
    write_evaluation_report,
)
from contextmap.evaluation.semantic_interpretation import (
    SemanticAnnotation,
    SemanticEvaluationContext,
    SemanticEvaluationFailure,
    SemanticEvaluationInput,
    SemanticEvaluationReport,
    SemanticStratum,
    StratumSource,
    evaluate_semantic_interpretation,
)
from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    HypothesisRole,
    PerceptionResultId,
    RegionId,
    SemanticBackendDiagnostics,
    SemanticConfidencePolicy,
    SemanticInferenceProvenance,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticPromptTemplate,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
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
    name: str,
    iou: float | None,
    *,
    runtime_ms: float,
    memory_mb: float | None,
    duplicate_rate: float | None = 0.1,
) -> FrameEvaluation:
    accuracy = (
        None
        if iou is None
        else SegmentationAccuracy(
            mean_iou=iou,
            mean_dice=iou,
            region_recall=iou / 2,
            coverage=iou,
            over_segmentation_rate=None if duplicate_rate is None else 0.0,
            under_segmentation_rate=None if duplicate_rate is None else 0.0,
            duplicate_region_rate=duplicate_rate,
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


def test_a_frame_without_discovered_regions_is_outside_the_duplicate_rate_population() -> None:
    report = _region_report(
        _frame("frame-a", 0.8, runtime_ms=10.0, memory_mb=None),
        _frame("frame-b", 0.0, runtime_ms=10.0, memory_mb=None, duplicate_rate=None),
    )

    lifted = region_discovery_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    duplicate = quality["region.duplicate_rate.mean"]
    assert duplicate.value == pytest.approx(0.1) and duplicate.sample_count == 1
    assert quality["region.iou.mean"].sample_count == 2


def test_duplicate_rate_is_not_applicable_when_no_annotated_frame_discovered_a_region() -> None:
    report = _region_report(
        _frame("frame-a", 0.0, runtime_ms=10.0, memory_mb=None, duplicate_rate=None)
    )

    lifted = region_discovery_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    duplicate = quality["region.duplicate_rate.mean"]
    assert duplicate.status is MetricStatus.NOT_APPLICABLE and duplicate.value is None
    assert quality["region.iou.mean"].status is MetricStatus.VALUE
    assert quality["region.iou.mean"].value == 0.0


def _semantic_context() -> SemanticEvaluationContext:
    return SemanticEvaluationContext(
        evaluation_id="semantic-eval-0001",
        reference_set_version="ci-subset/1.0.0",
        selection_id="selection-0001",
        perception_run_id="run-0001",
        artifact_id="artifact-0001",
        pipeline_configuration_digest="sha256:" + "d" * 64,
        evaluator_version="semantic-evaluator/1",
    )


def _execution(
    request_id: str,
    *,
    hypotheses: tuple[tuple[str, HypothesisRole], ...] = (
        ("wooden pallet", HypothesisRole.PRIMARY),
    ),
    source_observation_id: str = "frame-0001",
    region_id: str = "region-0001",
) -> SemanticInterpretationExecution:
    """Build a real, parsed execution, exactly as ``visual_perception`` produces one."""
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId(request_id),
        source_observation_id=SourceObservationId(source_observation_id),
        perception_result_id=PerceptionResultId("result-0001"),
        mode=SemanticInterpretationMode.REGION,
        visual_views=(
            SemanticVisualView(
                view_id="crop",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/semantic-views/crop.jpg",
                source_observation_id=SourceObservationId(source_observation_id),
                region_id=RegionId(region_id),
                sha256="0" * 64,
            ),
        ),
        region_id=RegionId(region_id),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )
    template = SemanticPromptTemplate.default_for(request.mode)
    rendered = render_semantic_prompt(
        request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
    )
    raw = json.dumps(
        {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": hypothesis,
                    "role": role.value,
                    "category": None,
                    "region_kind": "thing",
                    "attributes": {},
                    "confidence": None,
                }
                for hypothesis, role in hypotheses
            ],
            "scene_context": None,
        }
    )
    provenance = SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="qwen_semantic",
            capability="semantic_interpreter",
            provider="qwen",
            model="qwen-3b",
            version="1",
            configuration_fingerprint="sha256:config",
        ),
        task_identity="qwen-region",
        prompt_template_id="region/v1",
        output_schema_version="semantic-response/1",
    )
    return SemanticInterpretationExecution(
        request=request,
        rendered_prompt=rendered,
        raw_response=raw,
        parsed=parse_semantic_response(
            raw,
            request,
            provenance,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        diagnostics=SemanticBackendDiagnostics(latency_ms=400.0, input_tokens=10, output_tokens=5),
        effective_configuration={"model": "qwen-3b"},
    )


def _semantic_report() -> SemanticEvaluationReport:
    """One minimal evaluated semantic report, shared by the attempt-accounting tests."""
    return evaluate_semantic_interpretation(
        context=_semantic_context(),
        inputs=(
            SemanticEvaluationInput(
                execution=_execution("r1"),
                evidence_variant_id="tight-crop",
                annotation=SemanticAnnotation(acceptable_hypotheses=("wooden pallet",)),
            ),
        ),
    )


def test_semantic_report_becomes_quality_and_performance_metrics() -> None:
    report = evaluate_semantic_interpretation(
        context=_semantic_context(),
        inputs=(
            SemanticEvaluationInput(
                execution=_execution("r1"),
                evidence_variant_id="tight-crop",
                annotation=SemanticAnnotation(acceptable_hypotheses=("wooden pallet",)),
            ),
        ),
    )

    lifted = semantic_interpretation_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    performance = _by_metric(lifted.performance_metrics)
    assert lifted.stage is EvaluationStage.SEMANTIC_INTERPRETATION
    assert quality["semantic.acceptable_claim_rate"].value == 1.0
    assert quality["semantic.acceptable_claim_rate"].sample_count == 1
    assert quality["semantic.unsupported_claim_rate"].value == 0.0
    assert performance["runtime.wall_time"].value == pytest.approx(0.4)
    assert lifted.stage_report is not None
    assert lifted.stage_report["quality"]["claim_count"] == 1
    assert lifted.reproducibility.evaluator.evaluator_version == "semantic-evaluator/1"


def test_claims_without_annotation_are_not_applicable_never_raise() -> None:
    """#444: a run can have real claims and no human annotation; that is not an error."""
    report = evaluate_semantic_interpretation(
        context=_semantic_context(),
        inputs=(SemanticEvaluationInput(execution=_execution("r1"), evidence_variant_id="crop"),),
    )
    assert report.quality.claim_count == 1
    assert report.quality.assessed_claim_count == 0
    assert report.quality.acceptable_claim_rate is None

    lifted = semantic_interpretation_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    assert quality["semantic.acceptable_claim_rate"].status is MetricStatus.NOT_APPLICABLE
    assert quality["semantic.unsupported_claim_rate"].status is MetricStatus.NOT_APPLICABLE
    assert quality["semantic.acceptable_claim_rate"].value is None


def test_mixed_annotated_and_unannotated_population_uses_assessed_claim_count() -> None:
    """#444: the population is the assessed claims, never the raw claim count."""
    annotated = _execution("r1", source_observation_id="frame-0001", region_id="region-0001")
    unannotated = _execution(
        "r2",
        hypotheses=(
            ("wooden pallet", HypothesisRole.PRIMARY),
            ("wooden crate", HypothesisRole.ALTERNATIVE),
        ),
        source_observation_id="frame-0002",
        region_id="region-0002",
    )
    report = evaluate_semantic_interpretation(
        context=_semantic_context(),
        inputs=(
            SemanticEvaluationInput(
                execution=annotated,
                evidence_variant_id="tight-crop",
                annotation=SemanticAnnotation(acceptable_hypotheses=("wooden pallet",)),
            ),
            SemanticEvaluationInput(execution=unannotated, evidence_variant_id="tight-crop"),
        ),
    )
    assert report.quality.claim_count == 3
    assert report.quality.assessed_claim_count == 1

    lifted = semantic_interpretation_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    assert quality["semantic.acceptable_claim_rate"].sample_count == 1
    assert quality["semantic.unsupported_claim_rate"].sample_count == 1
    assert quality["semantic.acceptable_claim_rate"].value == 1.0


def test_ambiguity_population_excludes_repeats() -> None:
    """#444: repeats measure stability, not additional physical samples."""
    ambiguous_pair = (
        ("wooden pallet", HypothesisRole.PRIMARY),
        ("wooden crate", HypothesisRole.ALTERNATIVE),
    )
    annotation = SemanticAnnotation(
        acceptable_hypotheses=("wooden pallet", "wooden crate"), ambiguity_expected=True
    )
    primary = SemanticEvaluationInput(
        execution=_execution("r1", hypotheses=ambiguous_pair),
        evidence_variant_id="tight-crop",
        annotation=annotation,
        repeat_index=0,
    )
    repeat = SemanticEvaluationInput(
        execution=_execution("r1", hypotheses=ambiguous_pair),
        evidence_variant_id="tight-crop",
        annotation=annotation,
        repeat_index=1,
    )
    report = evaluate_semantic_interpretation(context=_semantic_context(), inputs=(primary, repeat))
    assert sum(item.ambiguity_preserved is not None for item in report.samples) == 2
    assert report.quality.ambiguity_preservation_rate == 1.0

    lifted = semantic_interpretation_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    ambiguity = _by_metric(lifted.quality_metrics)["semantic.ambiguity_preservation_rate"]
    assert ambiguity.value == 1.0
    assert ambiguity.sample_count == 1


def test_samples_and_failures_with_strata_round_trip_through_a_persisted_report(
    tmp_path: Path,
) -> None:
    """#444: ``SemanticStratum.source`` is an Enum; the persisted report must still be JSON."""
    stratum = SemanticStratum(scheme="lighting", label="low", source=StratumSource.DERIVED)
    failure = SemanticEvaluationFailure(
        request_id="r-failed",
        evidence_variant_id="tight-crop",
        failure_kind="backend",
        message="timeout",
        mode="region",
        source_observation_id="frame-0002",
        region_id="region-0002",
        strata=(
            SemanticStratum(scheme="visibility", label="occluded", source=StratumSource.DERIVED),
        ),
    )
    report = evaluate_semantic_interpretation(
        context=_semantic_context(),
        inputs=(
            SemanticEvaluationInput(
                execution=_execution("r1"), evidence_variant_id="tight-crop", strata=(stratum,)
            ),
        ),
        failures=(failure,),
    )

    lifted = semantic_interpretation_evaluation_report(
        report, registry=REGISTRY, reference_set=_reference()
    )

    path = tmp_path / "semantic-report.json"
    write_evaluation_report(path, lifted)
    document = json.loads(path.read_text(encoding="utf-8"))
    decoded = decode_evaluation_report(document, REGISTRY)
    assert decoded == lifted
    assert decoded.stage_report is not None
    assert decoded.stage_report["samples"][0]["strata"][0]["source"] == "derived"
    assert decoded.stage_report["failures"][0]["strata"][0]["source"] == "derived"
    assert decoded.stage_report["failures"][0]["message"] == "timeout"


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


def test_the_lifted_report_carries_the_parse_failure_rate() -> None:
    """A stage where a third of the responses never parsed must not read as fully healthy."""
    from contextmap.evaluation import SemanticAttemptAccounting, SemanticMeasurementStatus

    report = _semantic_report()
    lifted = semantic_interpretation_evaluation_report(
        report,
        registry=REGISTRY,
        reference_set=_reference(),
        attempts=SemanticAttemptAccounting(attempted=3, parsed=2, parse_failed=1),
    )

    quality = _by_metric(lifted.quality_metrics)
    assert quality["semantic.parse_failure_rate"].value == pytest.approx(1 / 3)
    assert quality["semantic.parse_failure_rate"].sample_count == 3
    assert lifted.stage_report is not None
    attempts = lifted.stage_report["attempts"]
    assert attempts == {
        "attempted": 3,
        "parsed": 2,
        "parse_failed": 1,
        "measurement_status": SemanticMeasurementStatus.COMPLETE.value,
    }


def test_a_legacy_run_reports_no_parse_failure_rate_instead_of_zero() -> None:
    """Never tracked is not the same as zero failures; reporting 0.0 would hide the gap."""
    from contextmap.evaluation import SemanticAttemptAccounting, SemanticMeasurementStatus

    lifted = semantic_interpretation_evaluation_report(
        _semantic_report(),
        registry=REGISTRY,
        reference_set=_reference(),
        attempts=SemanticAttemptAccounting(
            attempted=616,
            parsed=616,
            parse_failed=0,
            measurement_status=SemanticMeasurementStatus.LEGACY_LOWER_BOUND,
        ),
    )

    quality = _by_metric(lifted.quality_metrics)
    assert quality["semantic.parse_failure_rate"].status is MetricStatus.NOT_APPLICABLE
    assert quality["semantic.parse_failure_rate"].value is None
    assert lifted.stage_report is not None
    assert (
        lifted.stage_report["attempts"]["measurement_status"]
        == SemanticMeasurementStatus.LEGACY_LOWER_BOUND.value
    )


def test_a_report_without_attempt_accounting_stays_backward_compatible() -> None:
    """Callers that do not supply the accounting keep working, with the metric unavailable."""
    lifted = semantic_interpretation_evaluation_report(
        _semantic_report(), registry=REGISTRY, reference_set=_reference()
    )

    quality = _by_metric(lifted.quality_metrics)
    assert quality["semantic.parse_failure_rate"].status is MetricStatus.NOT_APPLICABLE
    assert lifted.stage_report is not None
    assert "attempts" not in lifted.stage_report
