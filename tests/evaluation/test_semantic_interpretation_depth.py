"""Depth tests of the Semantic Interpretation evaluation: annotations, strata, cost, repeats.

Fake/contract tests over canonical executions built through the real prompt and parser; they
prove the evaluator's arithmetic and conventions, not the quality of any backend.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from semantic_builders import execution, response_json

from contextmap.evaluation.semantic_interpretation import (
    SceneContextAnnotation,
    SemanticAnnotation,
    SemanticEvaluationContext,
    SemanticEvaluationError,
    SemanticEvaluationFailure,
    SemanticEvaluationInput,
    SemanticStratum,
    StratumSource,
    compare_evidence_variants,
    compare_semantic_backends,
    encode_evidence_variant_comparison,
    encode_semantic_backend_comparison,
    encode_semantic_evaluation_report,
    evaluate_semantic_interpretation,
)
from contextmap.visual_perception import (
    BackendProvenance,
    SemanticInterpretationMode,
    SemanticResponseParseError,
    VisualViewKind,
)

SCENE = SemanticInterpretationMode.SCENE
REGION = SemanticInterpretationMode.REGION


def _context() -> SemanticEvaluationContext:
    return SemanticEvaluationContext(
        evaluation_id="semantic-eval-0001",
        reference_set_version="corridor-semantics/1",
        selection_id="selection-0001",
        perception_run_id="run-0001",
        artifact_id="artifact-0001",
        pipeline_configuration_digest="sha256:pipeline",
        evaluator_version="semantic-evaluator/2",
    )


def _input(
    *,
    annotation: SemanticAnnotation | None = None,
    variant: str = "tight_crop",
    strata: tuple[SemanticStratum, ...] = (),
    repeat_index: int = 0,
    **execution_arguments: object,
) -> SemanticEvaluationInput:
    # Uma execução abstida não carrega claims, então a proveniência do backend é explícita.
    backend = BackendProvenance(
        backend_id=str(execution_arguments.get("backend_id", "qwen_semantic")),
        capability="semantic_interpreter",
        provider="test",
        model="model",
        version="1",
        configuration_fingerprint="sha256:config",
    )
    return SemanticEvaluationInput(
        execution=execution(**execution_arguments),  # type: ignore[arg-type]
        evidence_variant_id=variant,
        annotation=annotation,
        backend=backend,
        strata=strata,
        repeat_index=repeat_index,
    )


def _report(*inputs: SemanticEvaluationInput, failures: tuple[SemanticEvaluationFailure, ...] = ()):  # type: ignore[no-untyped-def]
    return evaluate_semantic_interpretation(
        context=_context(), inputs=tuple(inputs), failures=failures
    )


def test_without_an_annotation_claims_are_unassessed_not_unsupported() -> None:
    report = _report(_input(hypotheses=(("a wooden pallet", "primary"),)))

    (sample,) = report.samples
    assert not sample.annotated
    assert sample.acceptable_claim_count is None
    assert sample.unsupported_claim_count is None
    assert sample.rejected_claim_count is None
    assert sample.claim_count == 1
    assert report.quality.claim_count == 1
    assert report.quality.acceptable_claim_rate is None
    assert report.quality.unsupported_claim_rate is None
    assert report.quality.rejected_claim_rate is None
    assert report.quality.annotated_sample_count == 0


def test_explicitly_rejected_labels_are_the_only_hallucinations() -> None:
    annotation = SemanticAnnotation(
        acceptable_hypotheses=("wooden pallet",), rejected_hypotheses=("forklift",)
    )
    report = _report(
        _input(
            annotation=annotation,
            hypotheses=(
                ("wooden pallet", "primary"),
                ("Forklift", "alternative"),
                ("crate", "alternative"),
            ),
        )
    )

    (sample,) = report.samples
    assert sample.acceptable_claim_count == 1
    assert sample.rejected_claim_count == 1
    assert sample.unsupported_claim_count == 2
    assert report.quality.rejected_claim_rate == pytest.approx(1 / 3)


def test_annotation_rejects_a_label_that_is_both_acceptable_and_rejected() -> None:
    with pytest.raises(ValueError, match="both acceptable and rejected"):
        SemanticAnnotation(acceptable_hypotheses=("door",), rejected_hypotheses=("DOOR",))


def test_expected_abstention_is_scored_and_unexpected_abstention_is_visible() -> None:
    unknown = SemanticAnnotation(abstention_expected=True)
    answerable = SemanticAnnotation(acceptable_hypotheses=("door",))
    report = _report(
        _input(request_id="r1", region="r1", annotation=unknown, abstained=True),
        _input(request_id="r2", region="r2", annotation=unknown, hypotheses=(("door", "primary"),)),
        _input(request_id="r3", region="r3", annotation=answerable, abstained=True),
    )

    quality = report.quality
    assert quality.expected_abstention_count == 2
    assert quality.correct_abstention_count == 1
    assert quality.unexpected_abstention_count == 1
    assert quality.abstention_count == 2


def test_an_annotation_must_say_something_and_an_expected_abstention_excludes_answers() -> None:
    with pytest.raises(ValueError, match="must state"):
        SemanticAnnotation()
    with pytest.raises(ValueError, match="abstention"):
        SemanticAnnotation(abstention_expected=True, acceptable_hypotheses=("door",))


def test_scene_context_fields_are_scored_against_the_annotated_fields_only() -> None:
    annotation = SemanticAnnotation(
        scene_context=SceneContextAnnotation(
            acceptable={
                "environment": ("Outdoor",),
                "lighting": ("sunny",),
                "navigability": ("passable",),
            }
        )
    )
    report = _report(
        _input(
            request_id="scene-1",
            mode=SCENE,
            region=None,
            annotation=annotation,
            variant="full_frame",
            scene={"environment": "outdoor", "lighting": "dim", "layout": "anything"},
            hypotheses=(("a car park", "primary"),),
        )
    )

    (sample,) = report.samples
    outcomes = {item.field: item.outcome for item in sample.scene_field_outcomes}
    assert outcomes == {
        "environment": "correct",
        "lighting": "incorrect",
        "navigability": "missing",
    }
    by_field = {item.field: item for item in report.quality.scene_context}
    assert by_field["environment"].correct == 1
    assert by_field["lighting"].incorrect == 1
    assert by_field["navigability"].missing == 1
    assert "layout" not in by_field


def test_scene_context_annotation_rejects_unknown_fields_and_empty_value_lists() -> None:
    with pytest.raises(ValueError, match="unknown scene field"):
        SceneContextAnnotation(acceptable={"weather": ("rain",)})
    with pytest.raises(ValueError, match="at least one"):
        SceneContextAnnotation(acceptable={"lighting": ()})


def test_visibility_strata_report_quality_and_failures_per_stratum() -> None:
    annotation = SemanticAnnotation(acceptable_hypotheses=("door",), visibility_stratum="occluded")
    derived = SemanticStratum(scheme="region_area", label="large", source=StratumSource.DERIVED)
    report = _report(
        _input(request_id="r1", region="r1", annotation=annotation, strata=(derived,)),
        _input(
            request_id="r2",
            region="r2",
            annotation=SemanticAnnotation(
                acceptable_hypotheses=("door",), visibility_stratum="clear"
            ),
            hypotheses=(("door", "primary"),),
        ),
        failures=(
            SemanticEvaluationFailure(
                request_id="r3",
                evidence_variant_id="tight_crop",
                failure_kind="parser",
                message="bad json",
                strata=(
                    SemanticStratum(
                        scheme="visibility", label="occluded", source=StratumSource.ANNOTATION
                    ),
                ),
            ),
        ),
    )

    by_key = {(item.scheme, item.label): item for item in report.strata}
    occluded = by_key[("visibility", "occluded")]
    assert (occluded.sample_count, occluded.failure_count) == (1, 1)
    assert occluded.source is StratumSource.ANNOTATION
    assert occluded.acceptable_claim_rate == 0.0
    assert by_key[("visibility", "clear")].acceptable_claim_rate == 1.0
    assert by_key[("region_area", "large")].source is StratumSource.DERIVED


def test_a_conflicting_visibility_label_is_rejected() -> None:
    conflicting = SemanticStratum(scheme="visibility", label="clear", source=StratumSource.DERIVED)

    with pytest.raises(SemanticEvaluationError, match="visibility"):
        _report(
            _input(
                annotation=SemanticAnnotation(
                    acceptable_hypotheses=("door",), visibility_stratum="occluded"
                ),
                strata=(conflicting,),
            )
        )


def test_latency_is_reported_as_nearest_rank_percentiles_per_mode_and_overall() -> None:
    inputs = [
        _input(
            request_id=f"r{index}",
            region=f"r{index}",
            latency_ms=float(ms),
            peak_memory_bytes=ms * 1000,
        )
        for index, ms in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ]
    inputs.append(
        _input(
            request_id="scene-1",
            mode=SCENE,
            region=None,
            variant="full_frame",
            latency_ms=500.0,
            peak_memory_bytes=9_000_000,
        )
    )

    report = _report(*inputs)

    cost = report.cost
    by_mode = {item.mode: item for item in cost.by_mode}
    assert (by_mode["region"].latency_p50_ms, by_mode["region"].latency_p95_ms) == (50.0, 100.0)
    assert by_mode["scene"].latency_p50_ms == 500.0
    assert by_mode["region"].peak_memory_bytes == 100_000
    assert cost.latency_p50_ms == 60.0
    assert cost.latency_p95_ms == 500.0
    assert cost.peak_memory_bytes == 9_000_000
    assert cost.percentile_method == "nearest-rank"


def test_outcomes_separate_interpreted_abstained_parser_and_backend_failures_per_mode() -> None:
    report = _report(
        _input(request_id="r1", region="r1"),
        _input(request_id="r2", region="r2", abstained=True),
        failures=(
            SemanticEvaluationFailure(
                request_id="r3",
                evidence_variant_id="tight_crop",
                failure_kind="parser",
                message="bad",
                mode="region",
            ),
            SemanticEvaluationFailure(
                request_id="r4",
                evidence_variant_id="tight_crop",
                failure_kind="backend",
                message="oom",
                mode="region",
            ),
        ),
    )

    (region,) = [item for item in report.outcomes if item.mode == "region"]
    assert (region.requests, region.interpreted, region.abstained) == (4, 1, 1)
    assert (region.parser_failures, region.backend_failures) == (1, 1)


def test_repeats_measure_stability_and_are_not_counted_as_independent_evidence() -> None:
    first = _input(request_id="r1", region="r1", latency_ms=10.0)
    identical = _input(request_id="r1", region="r1", latency_ms=12.0, repeat_index=1)
    other = _input(request_id="r2", region="r2", hypotheses=(("door", "primary"),))
    diverging = _input(
        request_id="r2", region="r2", hypotheses=(("window", "primary"),), repeat_index=1
    )

    report = _report(first, identical, other, diverging)

    assert report.quality.claim_count == 2
    assert report.cost.request_count == 4
    consistency = report.consistency
    assert consistency.compared_request_count == 2
    assert consistency.identical_raw_response_count == 1
    assert consistency.identical_answer_count == 1
    assert consistency.divergent_requests == ("r2::tight_crop",)


def test_a_repeat_that_fails_where_another_succeeded_is_a_divergence() -> None:
    report = _report(
        _input(request_id="r1", region="r1"),
        failures=(
            SemanticEvaluationFailure(
                request_id="r1",
                evidence_variant_id="tight_crop",
                failure_kind="parser",
                message="bad",
                repeat_index=1,
                raw_response="{",
            ),
        ),
    )

    assert report.consistency.compared_request_count == 1
    assert report.consistency.divergent_requests == ("r1::tight_crop",)


def test_duplicate_request_variant_and_repeat_are_rejected() -> None:
    with pytest.raises(SemanticEvaluationError, match="duplicate"):
        _report(_input(request_id="r1"), _input(request_id="r1"))


def test_label_concentration_exposes_structural_or_trivial_label_domination() -> None:
    report = _report(
        _input(request_id="r1", region="r1", hypotheses=(("wall", "primary"),)),
        _input(request_id="r2", region="r2", hypotheses=(("Wall", "primary"),)),
        _input(request_id="r3", region="r3", hypotheses=(("wall", "primary"),)),
        _input(request_id="r4", region="r4", hypotheses=(("door", "primary"),)),
    )

    quality = report.quality
    assert quality.distinct_hypothesis_count == 2
    assert quality.dominant_hypothesis == "wall"
    assert quality.dominant_hypothesis_share == 0.75


def test_evidence_variants_are_compared_on_the_same_regions_with_their_real_channels() -> None:
    door = SemanticAnnotation(acceptable_hypotheses=("door",))
    rows = []
    for region, tight, masked in (("r1", "door", "door"), ("r2", "door", "wall")):
        rows.append(
            _input(
                request_id=f"{region}-t",
                region=region,
                variant="tight_crop",
                annotation=door,
                hypotheses=((tight, "primary"),),
            )
        )
        rows.append(
            _input(
                request_id=f"{region}-m",
                region=region,
                variant="masked_subject",
                annotation=door,
                view_kind=VisualViewKind.MASKED_SUBJECT,
                hypotheses=((masked, "primary"),),
            )
        )
    rows.append(_input(request_id="r3-t", region="r3", variant="tight_crop", annotation=door))

    comparison = compare_evidence_variants(_report(*rows), baseline_variant_id="tight_crop")

    assert comparison.paired_region_count == 2
    baseline, masked_entry = comparison.entries
    assert baseline.variant_id == "tight_crop"
    assert baseline.evidence_channels == ("view:tight_crop",)
    assert masked_entry.evidence_channels == ("view:masked_subject",)
    assert baseline.acceptable_claim_rate == 1.0
    assert masked_entry.acceptable_claim_rate == 0.5
    assert masked_entry.acceptable_claim_rate_delta == -0.5
    assert baseline.prompt_template_ids == ("region/v1",)
    # Sensibilidade à evidência sem anotação: quantas regiões mantêm a hipótese primária.
    assert (baseline.primary_comparable_count, baseline.primary_agreement_count) == (None, None)
    assert (masked_entry.primary_comparable_count, masked_entry.primary_agreement_count) == (2, 1)


def test_scene_context_evidence_is_a_channel_and_variants_need_a_shared_region() -> None:
    plain = _input(request_id="r1-a", region="r1", variant="plain")
    with_context = _input(
        request_id="r1-b", region="r1", variant="with_scene_context", scene_context_reference=True
    )
    comparison = compare_evidence_variants(
        _report(plain, with_context), baseline_variant_id="plain"
    )
    assert comparison.entries[1].evidence_channels == ("scene_context", "view:tight_crop")

    with pytest.raises(SemanticEvaluationError, match="baseline"):
        compare_evidence_variants(_report(plain), baseline_variant_id="missing")
    with pytest.raises(SemanticEvaluationError, match="no region"):
        compare_evidence_variants(
            _report(plain, _input(request_id="r9", region="r9", variant="other")),
            baseline_variant_id="plain",
        )


def _region_failure(
    request_id: str,
    *,
    kind: str = "parser",
    frame: str = "frame-1",
    region: str | None = None,
    variant: str = "tight_crop",
) -> SemanticEvaluationFailure:
    """Build a region failure that carries the full physical identity of its request."""
    return SemanticEvaluationFailure(
        request_id=request_id,
        evidence_variant_id=variant,
        failure_kind=kind,
        message="failed",
        mode="region",
        source_observation_id=frame,
        region_id=region or request_id,
    )


def test_backend_comparison_aligns_on_attempted_requests_and_lists_outcomes() -> None:
    first = _report(
        _input(request_id="r1", region="r1", hypotheses=(("door", "primary"),)),
        _input(request_id="r2", region="r2", hypotheses=(("wall", "primary"),)),
        failures=(_region_failure("r3", kind="parser"),),
    )
    second = _report(
        _input(
            request_id="r1",
            region="r1",
            hypotheses=(("Door", "primary"),),
            backend_id="florence2_semantic",
        ),
        _input(
            request_id="r3",
            region="r3",
            hypotheses=(("floor", "primary"),),
            backend_id="florence2_semantic",
        ),
        failures=(_region_failure("r2", kind="backend"),),
    )

    comparison = compare_semantic_backends((first, second))

    outcomes = {
        (item.request_id, item.evidence_variant_id): item.outcomes for item in comparison.outcomes
    }
    assert outcomes[("r1", "tight_crop")] == ("claims", "claims")
    assert outcomes[("r2", "tight_crop")] == ("claims", "backend_failure")
    assert outcomes[("r3", "tight_crop")] == ("parser_failure", "claims")
    assert comparison.primary_agreement_count == 1
    assert comparison.primary_comparable_count == 1

    with pytest.raises(SemanticEvaluationError, match="same request"):
        compare_semantic_backends((first, _report(_input(request_id="r1", region="r1"))))


def test_backend_comparison_outcomes_record_the_physical_identity_they_aligned() -> None:
    first = _report(_input(request_id="r1", region="region-a", frame="frame-7"))
    second = _report(
        _input(request_id="r1", region="region-a", frame="frame-7", backend_id="florence2_semantic")
    )

    (outcome,) = compare_semantic_backends((first, second)).outcomes

    assert (outcome.source_observation_id, outcome.region_id, outcome.mode) == (
        "frame-7",
        "region-a",
        "region",
    )
    encoded = encode_semantic_backend_comparison(compare_semantic_backends((first, second)))
    assert encoded["outcomes"][0]["source_observation_id"] == "frame-7"


@pytest.mark.parametrize(
    ("other", "divergent_identity"),
    [
        (dict(frame="frame-2", region="r1"), "frame-2"),
        (dict(frame="frame-1", region="region-elsewhere"), "region-elsewhere"),
    ],
)
def test_backend_comparison_rejects_the_same_request_ids_over_other_observations_or_regions(
    other: dict[str, str], divergent_identity: str
) -> None:
    first = _report(
        _input(request_id="r1", region="r1", frame="frame-1"),
        _input(request_id="r2", region="r2", frame="frame-1"),
    )
    second = _report(
        _input(request_id="r1", **other, backend_id="florence2_semantic"),  # type: ignore[arg-type]
        _input(request_id="r2", region="r2", frame="frame-1", backend_id="florence2_semantic"),
    )

    with pytest.raises(SemanticEvaluationError, match="same request") as caught:
        compare_semantic_backends((first, second))

    assert divergent_identity in str(caught.value)


def test_backend_comparison_rejects_the_same_request_id_across_scene_and_region_modes() -> None:
    region = _report(_input(request_id="r1", region="r1", frame="frame-1"))
    scene = _report(
        _input(request_id="r1", mode=SCENE, region=None, frame="frame-1", variant="tight_crop")
    )

    with pytest.raises(SemanticEvaluationError, match="same request"):
        compare_semantic_backends((region, scene))


def test_backend_comparison_matches_a_failure_by_its_physical_identity() -> None:
    sample = _report(_input(request_id="r1", region="r1", frame="frame-1"))
    failure_on_the_same_input = _report(
        failures=(_region_failure("r1", kind="backend", frame="frame-1", region="r1"),)
    )
    failure_on_another_frame = _report(
        failures=(_region_failure("r1", kind="backend", frame="frame-2", region="r1"),)
    )
    failure_on_another_region = _report(
        failures=(_region_failure("r1", kind="backend", frame="frame-1", region="r9"),)
    )

    (outcome,) = compare_semantic_backends((sample, failure_on_the_same_input)).outcomes
    assert outcome.outcomes == ("claims", "backend_failure")
    for other in (failure_on_another_frame, failure_on_another_region):
        with pytest.raises(SemanticEvaluationError, match="same request"):
            compare_semantic_backends((sample, other))


@pytest.mark.parametrize(
    "failure",
    [
        SemanticEvaluationFailure(
            request_id="r1", evidence_variant_id="tight_crop", failure_kind="backend", message="x"
        ),
        SemanticEvaluationFailure(
            request_id="r1",
            evidence_variant_id="tight_crop",
            failure_kind="backend",
            message="x",
            mode="region",
            region_id="r1",
        ),
        SemanticEvaluationFailure(
            request_id="r1",
            evidence_variant_id="tight_crop",
            failure_kind="backend",
            message="x",
            source_observation_id="frame-1",
            region_id="r1",
        ),
        SemanticEvaluationFailure(
            request_id="r1",
            evidence_variant_id="tight_crop",
            failure_kind="backend",
            message="x",
            mode="region",
            source_observation_id="frame-1",
        ),
        SemanticEvaluationFailure(
            request_id="r1",
            evidence_variant_id="tight_crop",
            failure_kind="backend",
            message="x",
            mode="scene",
            source_observation_id="frame-1",
            region_id="r1",
        ),
    ],
    ids=["no-identity", "no-observation", "no-mode", "region-without-region", "scene-with-region"],
)
def test_backend_comparison_rejects_failures_without_enough_physical_identity(
    failure: SemanticEvaluationFailure,
) -> None:
    complete = _report(_input(request_id="r1", region="r1", frame="frame-1"))

    with pytest.raises(SemanticEvaluationError, match="physical identity"):
        compare_semantic_backends((complete, _report(failures=(failure,))))


@pytest.mark.parametrize(
    "field", ["reference_set_version", "selection_id", "perception_run_id", "evaluator_version"]
)
def test_backend_comparison_rejects_reports_from_different_experimental_contexts(
    field: str,
) -> None:
    first = _report(_input(request_id="r1", region="r1"))
    second = _report(_input(request_id="r1", region="r1", backend_id="florence2_semantic"))
    other_context = replace(second.context, **{field: "another/1"})

    with pytest.raises(SemanticEvaluationError, match=field):
        compare_semantic_backends((first, replace(second, context=other_context)))


def test_backend_comparison_rejects_reports_with_different_matching_policies() -> None:
    first = _report(_input(request_id="r1", region="r1"))
    second = _report(_input(request_id="r1", region="r1", backend_id="florence2_semantic"))

    with pytest.raises(SemanticEvaluationError, match="matching_policy"):
        compare_semantic_backends((first, replace(second, matching_policy="other-matcher/1")))


def test_backend_comparison_accepts_the_identities_that_belong_to_each_backend() -> None:
    first = _report(_input(request_id="r1", region="r1"))
    second = _report(_input(request_id="r1", region="r1", backend_id="florence2_semantic"))
    own_context = replace(
        second.context,
        evaluation_id="semantic-eval-0002",
        artifact_id="artifact-0002",
        pipeline_configuration_digest="sha256:another-backend-pipeline",
    )

    comparison = compare_semantic_backends((first, replace(second, context=own_context)))

    assert comparison.primary_comparable_count == 1


def test_a_parse_error_carries_the_raw_response_it_rejected() -> None:
    from contextmap.visual_perception import SemanticInterpretationRequest, parse_semantic_response
    from contextmap.visual_perception.semantic_prompt import SemanticConfidencePolicy

    ok = execution()
    request: SemanticInterpretationRequest = ok.request
    provenance = ok.parsed.claims[0].provenance

    with pytest.raises(SemanticResponseParseError) as caught:
        parse_semantic_response(
            '{"abstained": false, "claims": [',
            request,
            provenance,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )

    assert caught.value.raw_response == '{"abstained": false, "claims": ['


def test_failures_are_classified_from_the_exception_with_raw_response_and_identity() -> None:
    ok = execution(request_id="r1", region="r1", frame="frame-9")
    parse_error = SemanticResponseParseError(
        "claim[0] is missing required fields", raw_response="{}"
    )

    parser_failure = SemanticEvaluationFailure.from_error(
        ok.request, evidence_variant_id="tight_crop", error=parse_error, repeat_index=1
    )
    backend_failure = SemanticEvaluationFailure.from_error(
        ok.request, evidence_variant_id="tight_crop", error=RuntimeError("out of memory")
    )

    assert parser_failure.failure_kind == "parser"
    assert parser_failure.raw_response == "{}"
    assert parser_failure.repeat_index == 1
    assert (
        parser_failure.mode,
        parser_failure.source_observation_id,
        parser_failure.region_id,
    ) == (
        "region",
        "frame-9",
        "r1",
    )
    assert backend_failure.failure_kind == "backend"
    assert backend_failure.raw_response is None
    assert "out of memory" in backend_failure.message


def test_the_report_and_comparisons_encode_to_json_with_every_identity() -> None:
    annotation = SemanticAnnotation(
        acceptable_hypotheses=("door",), rejected_hypotheses=("wall",), visibility_stratum="clear"
    )
    report = _report(
        _input(
            annotation=annotation,
            hypotheses=(("door", "primary"), ("wall", "alternative")),
            peak_memory_bytes=1024,
        ),
        _input(request_id="r2", region="r2"),
        failures=(
            SemanticEvaluationFailure(
                request_id="r9",
                evidence_variant_id="tight_crop",
                failure_kind="parser",
                message="bad",
                raw_response="{",
                mode="region",
            ),
        ),
    )

    encoded = encode_semantic_evaluation_report(report)
    text = json.dumps(encoded, sort_keys=True, allow_nan=False)

    assert json.loads(text) == json.loads(json.dumps(encoded))
    assert encoded["context"]["evaluator_version"] == "semantic-evaluator/2"
    assert encoded["matching_policy"] == "casefold-exact/1"
    assert encoded["samples"][0]["backend"]["backend_id"] == "qwen_semantic"
    assert encoded["samples"][0]["prompt_fingerprint"].startswith("sha256:")
    assert encoded["failures"][0]["raw_response"] == "{"
    assert encoded["quality"]["acceptable_claim_rate"] == pytest.approx(0.5)
    assert encoded["quality"]["rejected_claim_rate"] == pytest.approx(0.5)
    assert encoded["cost"]["latency_p50_ms"] == 10.0
    assert encoded["cost"]["percentile_method"] == "nearest-rank"
    assert encoded["outcomes"][0]["mode"] == "region"
    assert encoded["consistency"]["compared_request_count"] == 0

    other = _report(
        _input(backend_id="florence2_semantic"),
        _input(request_id="r2", region="r2", backend_id="florence2_semantic"),
    )
    comparison = compare_semantic_backends((replace(report, failures=()), other))
    variants = compare_evidence_variants(report, baseline_variant_id="tight_crop")
    json.dumps(encode_semantic_backend_comparison(comparison), allow_nan=False)
    json.dumps(encode_evidence_variant_comparison(variants), allow_nan=False)


def test_scene_only_response_json_helper_stays_a_valid_canonical_document() -> None:
    parsed = json.loads(response_json((("hall", "primary"),), mode=SCENE))

    assert parsed["scene_context"] == {}
