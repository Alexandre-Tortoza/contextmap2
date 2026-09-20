"""Deterministic Semantic Interpretation evaluation tests."""

import json
from dataclasses import replace

import pytest

from contextmap.evaluation.semantic_interpretation import (
    SemanticAnnotation,
    SemanticEvaluationContext,
    SemanticEvaluationError,
    SemanticEvaluationFailure,
    SemanticEvaluationInput,
    compare_semantic_backends,
    evaluate_semantic_interpretation,
)
from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    ClaimId,
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


def _semantic_execution() -> SemanticInterpretationExecution:
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("region-request-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("result-0001"),
        mode=SemanticInterpretationMode.REGION,
        visual_views=(
            SemanticVisualView(
                view_id="crop",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/semantic-views/crop.jpg",
                source_observation_id=SourceObservationId("frame-0001"),
                region_id=RegionId("region-0001"),
                sha256="0" * 64,
            ),
        ),
        region_id=RegionId("region-0001"),
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
                    "hypothesis": "wooden pallet",
                    "role": "primary",
                    "category": None,
                    "region_kind": "thing",
                    "attributes": {},
                    "confidence": None,
                }
            ],
            "scene_context": None,
        }
    )
    provenance = SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="qwen_semantic",
            capability="semantic_interpreter",
            provider="qwen",
            model="model",
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
        diagnostics=SemanticBackendDiagnostics(latency_ms=10, input_tokens=5, output_tokens=3),
        effective_configuration={"model": "model"},
    )


def _context() -> SemanticEvaluationContext:
    return SemanticEvaluationContext(
        evaluation_id="semantic-eval-0001",
        reference_set_version="corridor-semantics/1",
        selection_id="selection-0001",
        perception_run_id="run-0001",
        artifact_id="artifact-0001",
        pipeline_configuration_digest="sha256:pipeline",
        evaluator_version="semantic-evaluator/1",
    )


def test_semantic_report_keeps_quality_and_cost_separate() -> None:
    execution = _semantic_execution()
    claim = execution.parsed.claims[0]
    alternative = replace(
        claim,
        claim_id=ClaimId("claim-alt"),
        hypothesis="wooden crate",
        role=HypothesisRole.ALTERNATIVE,
    )
    execution = replace(
        execution,
        parsed=replace(execution.parsed, claims=(claim, alternative)),
    )
    report = evaluate_semantic_interpretation(
        context=_context(),
        inputs=(
            SemanticEvaluationInput(
                execution=execution,
                evidence_variant_id="tight-crop",
                annotation=SemanticAnnotation(
                    acceptable_hypotheses=("wooden pallet", "wooden crate"),
                    ambiguity_expected=True,
                ),
            ),
        ),
    )

    assert report.quality.acceptable_claim_rate == 1.0
    assert report.quality.unsupported_claim_rate == 0.0
    assert report.quality.ambiguity_preservation_rate == 1.0
    assert report.cost.request_count == 1
    assert report.cost.total_latency_ms == execution.diagnostics.latency_ms
    assert report.matching_policy == "casefold-exact/1"


def test_reports_compare_same_requests_and_expose_failure_kind() -> None:
    execution = _semantic_execution()
    first = evaluate_semantic_interpretation(
        context=_context(),
        inputs=(
            SemanticEvaluationInput(
                execution=execution,
                evidence_variant_id="masked-subject",
                annotation=SemanticAnnotation(acceptable_hypotheses=("wooden pallet",)),
            ),
        ),
        failures=(
            SemanticEvaluationFailure(
                request_id="parser-broken",
                evidence_variant_id="masked-subject",
                failure_kind="parser",
                message="malformed JSON",
            ),
        ),
    )
    second_execution = replace(
        execution,
        parsed=replace(
            execution.parsed,
            claims=(replace(execution.parsed.claims[0], hypothesis="unsupported object"),),
        ),
    )
    second = evaluate_semantic_interpretation(
        context=replace(_context(), evaluation_id="semantic-eval-0002"),
        inputs=(
            SemanticEvaluationInput(
                execution=second_execution,
                evidence_variant_id="masked-subject",
                annotation=SemanticAnnotation(acceptable_hypotheses=("wooden pallet",)),
            ),
        ),
    )

    comparison = compare_semantic_backends((first, second))

    assert first.failures[0].failure_kind == "parser"
    assert comparison.request_ids == ("region-request-0001",)
    assert second.quality.unsupported_claim_rate == 1.0

    with pytest.raises(SemanticEvaluationError, match="same request"):
        compare_semantic_backends((first, replace(second, samples=())))
