"""Evaluator reproducibility tests: identical inputs must give equivalent reports."""

import itertools
from dataclasses import replace

import pytest
from reference_set_builders import make_valid_manifest

from contextmap.evaluation.evaluator_reproducibility import (
    EvaluatorNondeterminismError,
    NondeterministicField,
    check_evaluator_reproducibility,
    compare_evaluation_reports,
    require_reproducible,
)
from contextmap.evaluation.metrics import EvaluationStage, default_metric_registry
from contextmap.evaluation.report_schema import (
    ArtifactIdentity,
    EvaluationReport,
    EvaluatorIdentity,
    MetricResult,
    MetricStatus,
    ReproducibilityMetadata,
    assemble_evaluation_report,
)

REGISTRY = default_metric_registry()
REGIONS = "contextmap.reference.regions/v1"


def _result(name: str, value: float, count: int = 4) -> MetricResult:
    return MetricResult(
        metric=name,
        metric_version="1",
        status=MetricStatus.VALUE,
        value=value,
        sample_count=count,
    )


def _report(
    *,
    iou: float = 0.5,
    recall: float = 0.25,
    wall: float = 1.5,
    stage_report: dict[str, object] | None = None,
    reference_version: str = "1.0.0",
    with_recall: bool = True,
) -> EvaluationReport:
    reference = replace(make_valid_manifest().identity(), version=reference_version)
    quality = (_result("region.iou.mean", iou),) + (
        (_result("region.recall.mean", recall),) if with_recall else ()
    )
    return assemble_evaluation_report(
        REGISTRY,
        stage=EvaluationStage.REGION_DISCOVERY,
        reproducibility=ReproducibilityMetadata(
            evaluator=EvaluatorIdentity(
                evaluator_id="region-discovery-evaluator", evaluator_version="1"
            ),
            reference_set=reference,
            annotation_schemas=(REGIONS,),
            input_artifacts=(
                ArtifactIdentity(kind="perception_run", artifact_id="run-1", digest=None),
            ),
            configuration_digest="sha256:" + "a" * 64,
            code_version="0.0.1",
            metric_registry=REGISTRY.identity(),
        ),
        quality_metrics=quality,
        performance_metrics=(_result("runtime.wall_time", wall),),
        stage_report=stage_report
        if stage_report is not None
        else {"frames": [1, 2], "seed_trace": "a"},
    )


def test_a_deterministic_evaluator_is_reproducible() -> None:
    check = check_evaluator_reproducibility(lambda: _report())

    assert check.reproducible
    assert check.differences == ()
    assert check.repetitions == 2
    assert check.declared == ()
    require_reproducible(check)


def test_resource_measurements_may_vary_and_the_check_says_so() -> None:
    walls = itertools.count(start=1)

    check = check_evaluator_reproducibility(lambda: _report(wall=float(next(walls))), repetitions=3)

    assert check.reproducible
    assert check.performance_values_excluded
    assert "resource" in check.performance_exclusion_reason
    assert check.repetitions == 3


def test_a_change_in_which_resource_metrics_exist_is_still_a_difference() -> None:
    first = _report()
    other = replace(
        first,
        performance_metrics=(
            MetricResult(
                metric="runtime.wall_time",
                metric_version="1",
                status=MetricStatus.UNSUPPORTED,
                value=None,
                sample_count=None,
            ),
        ),
    )

    differences = compare_evaluation_reports(first, other)

    assert "performance_metrics[0].status" in differences
    assert "performance_metrics[0].sample_count" in differences


def test_a_quality_value_that_changes_between_runs_is_a_nondeterministic_evaluator() -> None:
    values = itertools.count(start=1)

    check = check_evaluator_reproducibility(lambda: _report(iou=next(values) / 10))

    assert not check.reproducible
    assert "quality_metrics[0].value" in check.differences
    with pytest.raises(EvaluatorNondeterminismError, match="quality_metrics"):
        require_reproducible(check)


def test_nondeterminism_in_the_stage_report_must_be_declared_with_a_reason() -> None:
    traces = itertools.count()

    def evaluate() -> EvaluationReport:
        return _report(stage_report={"frames": [1, 2], "seed_trace": str(next(traces))})

    undeclared = check_evaluator_reproducibility(evaluate)
    declared = check_evaluator_reproducibility(
        evaluate,
        nondeterministic=(
            NondeterministicField(
                location="stage_report:seed_trace",
                reason="GPU atomics reorder floating-point accumulation between runs",
            ),
        ),
    )

    assert "stage_report.seed_trace" in undeclared.differences
    assert declared.reproducible
    assert declared.declared[0].reason.startswith("GPU atomics")
    assert (
        declared.to_record()["declared_nondeterminism"][0]["location"] == "stage_report:seed_trace"
    )


def test_only_the_declared_stage_report_field_is_excused() -> None:
    traces = itertools.count()

    def evaluate() -> EvaluationReport:
        n = next(traces)
        return _report(stage_report={"frames": [n], "seed_trace": str(n)})

    check = check_evaluator_reproducibility(
        evaluate,
        nondeterministic=(
            NondeterministicField(location="stage_report:seed_trace", reason="unordered reduction"),
        ),
    )

    assert not check.reproducible
    assert check.differences == ("stage_report.frames[0]",)


def test_a_quality_metric_can_be_declared_nondeterministic_and_the_others_stay_strict() -> None:
    counter = itertools.count(start=1)

    def evaluate() -> EvaluationReport:
        n = next(counter)
        return _report(iou=n / 10, recall=n / 10)

    check = check_evaluator_reproducibility(
        evaluate,
        nondeterministic=(
            NondeterministicField(
                location="quality_metric:region.iou.mean",
                reason="matching uses a randomized tie-break on equal IoU",
            ),
        ),
    )

    assert check.differences == ("quality_metrics[1].value",)


def test_declared_nondeterminism_needs_a_location_and_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        NondeterministicField(location="stage_report:x", reason=" ")
    with pytest.raises(ValueError, match="location"):
        NondeterministicField(location="everything", reason="because")
    with pytest.raises(ValueError, match="location"):
        NondeterministicField(location="stage_report:", reason="because")


def test_a_drifting_reference_set_or_metric_set_is_not_reproducible() -> None:
    base = _report()

    reference = compare_evaluation_reports(base, _report(reference_version="2.0.0"))
    metrics = compare_evaluation_reports(base, _report(with_recall=False))

    assert "reproducibility.reference_set.version" in reference
    assert any(item.startswith("quality_metrics") for item in metrics)


def test_identical_reports_have_no_differences() -> None:
    assert compare_evaluation_reports(_report(), _report(wall=99.0)) == ()


def test_at_least_two_runs_are_needed_to_compare() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        check_evaluator_reproducibility(lambda: _report(), repetitions=1)


def test_the_check_is_machine_readable() -> None:
    check = check_evaluator_reproducibility(lambda: _report())

    record = check.to_record()

    assert record["reproducible"] is True
    assert record["repetitions"] == 2
    assert record["performance_values_excluded"] is True
    assert record["differences"] == []
