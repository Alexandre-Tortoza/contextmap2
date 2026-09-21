"""Common evaluation report envelope tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from reference_set_builders import make_valid_manifest

from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.metrics import (
    EvaluationStage,
    MetricCompatibilityError,
    MetricRegistryError,
    default_metric_registry,
)
from contextmap.evaluation.report_schema import (
    REPORT_SCHEMA,
    ArtifactIdentity,
    EvaluationReport,
    EvaluationReportError,
    EvaluatorIdentity,
    MetricResult,
    MetricStatus,
    ReproducibilityMetadata,
    assemble_evaluation_report,
    decode_evaluation_report,
    encode_evaluation_report,
    write_evaluation_report,
)

REGISTRY = default_metric_registry()
REGIONS = AnnotationFamily.REGIONS.schema


def _metadata(**overrides: object) -> ReproducibilityMetadata:
    fields: dict[str, object] = {
        "evaluator": EvaluatorIdentity(
            evaluator_id="region-discovery-evaluator", evaluator_version="1"
        ),
        "reference_set": make_valid_manifest().identity(),
        "annotation_schemas": (REGIONS,),
        "input_artifacts": (
            ArtifactIdentity(kind="perception_run", artifact_id="run-0001", digest=None),
        ),
        "configuration_digest": "sha256:" + "a" * 64,
        "code_version": "0.0.1",
        "metric_registry": REGISTRY.identity(),
    }
    fields.update(overrides)
    return ReproducibilityMetadata(**fields)  # type: ignore[arg-type]


def _value(name: str = "region.iou.mean", value: float = 0.5, **overrides: object) -> MetricResult:
    fields: dict[str, object] = {
        "metric": name,
        "metric_version": "1",
        "status": MetricStatus.VALUE,
        "value": value,
        "sample_count": 4,
    }
    fields.update(overrides)
    return MetricResult(**fields)  # type: ignore[arg-type]


def _report(**overrides: object) -> EvaluationReport:
    fields: dict[str, object] = {
        "stage": EvaluationStage.REGION_DISCOVERY,
        "reproducibility": _metadata(),
        "quality_metrics": (_value(),),
        "performance_metrics": (_value("runtime.wall_time", 1.5, sample_count=4),),
        "stage_report": {"schema": "contextmap.region-discovery-evaluation/v1", "frames": []},
    }
    fields.update(overrides)
    return assemble_evaluation_report(REGISTRY, **fields)  # type: ignore[arg-type]


def test_reports_share_reproducibility_metadata_and_keep_stage_specific_metrics() -> None:
    report = _report()

    document = json.loads(json.dumps(encode_evaluation_report(report)))

    assert document["schema"] == REPORT_SCHEMA
    assert document["stage"] == "region_discovery"
    reference = report.reproducibility.reference_set
    assert reference is not None
    assert document["reproducibility"]["reference_set"]["digest"] == reference.digest
    assert document["reproducibility"]["metric_registry"]["digest"] == REGISTRY.identity().digest
    assert document["stage_report"]["frames"] == []
    assert decode_evaluation_report(document, REGISTRY) == report


def test_quality_and_performance_are_separate_and_cannot_be_mistaken() -> None:
    with pytest.raises(EvaluationReportError, match="performance"):
        _report(quality_metrics=(_value("runtime.wall_time", 1.5),))
    with pytest.raises(EvaluationReportError, match="quality"):
        _report(performance_metrics=(_value("region.iou.mean", 0.5),))
    report = _report()
    assert {item.metric for item in report.quality_metrics}.isdisjoint(
        {item.metric for item in report.performance_metrics}
    )
    assert not hasattr(report, "overall_score")


def test_unknown_metrics_and_versions_are_rejected() -> None:
    with pytest.raises(MetricRegistryError, match="unknown"):
        _report(quality_metrics=(_value("region.overall.score", 0.9),))
    with pytest.raises(MetricRegistryError, match="unknown"):
        _report(quality_metrics=(_value(metric_version="9"),))


def test_a_metric_of_another_stage_is_rejected() -> None:
    with pytest.raises(EvaluationReportError, match="stage"):
        _report(quality_metrics=(_value("semantic.acceptable_claim_rate", 0.5),))


def test_missing_annotations_are_not_applicable_never_a_fabricated_zero() -> None:
    not_applicable = MetricResult(
        metric="region.iou.mean",
        metric_version="1",
        status=MetricStatus.NOT_APPLICABLE,
        value=None,
        sample_count=0,
    )

    report = _report(quality_metrics=(not_applicable,))

    assert report.quality_metrics[0].status is MetricStatus.NOT_APPLICABLE
    with pytest.raises(ValueError, match="value"):
        replace(not_applicable, value=0.0)
    with pytest.raises(ValueError, match="value"):
        replace(_value(), value=None)
    with pytest.raises(ValueError, match="finite"):
        _value(value=float("nan"))
    unsupported = replace(not_applicable, status=MetricStatus.UNSUPPORTED)
    assert _report(quality_metrics=(unsupported,)).quality_metrics[0].status is (
        MetricStatus.UNSUPPORTED
    )


def test_values_must_respect_the_declared_range() -> None:
    with pytest.raises(EvaluationReportError, match="range"):
        _report(quality_metrics=(_value(value=1.5),))
    with pytest.raises(EvaluationReportError, match="range"):
        _report(performance_metrics=(_value("runtime.wall_time", -1.0),))


def test_a_value_needs_its_annotation_schema_to_be_available() -> None:
    incompatible = _metadata(annotation_schemas=("contextmap.reference.regions/v2",))
    without = _metadata(annotation_schemas=())

    with pytest.raises(MetricCompatibilityError, match="regions"):
        _report(reproducibility=incompatible)
    with pytest.raises(MetricCompatibilityError, match="regions"):
        _report(reproducibility=without)
    not_applicable = MetricResult(
        metric="region.iou.mean",
        metric_version="1",
        status=MetricStatus.NOT_APPLICABLE,
        value=None,
        sample_count=0,
    )
    honest = _report(reproducibility=without, quality_metrics=(not_applicable,))
    assert honest.quality_metrics[0].status is MetricStatus.NOT_APPLICABLE


def test_annotation_dependent_quality_needs_a_reference_set() -> None:
    with pytest.raises(EvaluationReportError, match="reference set"):
        _report(reproducibility=_metadata(reference_set=None))
    integrity = _report(
        stage=EvaluationStage.INGESTION_INTEGRITY,
        reproducibility=_metadata(reference_set=None, annotation_schemas=()),
        quality_metrics=(_value("ingestion.integrity.violations", 0.0, sample_count=12),),
        performance_metrics=(),
        stage_report=None,
    )
    assert integrity.reproducibility.reference_set is None


def test_runtime_reports_carry_performance_metrics_only() -> None:
    runtime = _report(
        stage=EvaluationStage.RUNTIME,
        reproducibility=_metadata(reference_set=None, annotation_schemas=()),
        quality_metrics=(),
        performance_metrics=(
            _value("runtime.wall_time", 12.0),
            _value("runtime.peak_memory", 1_000_000.0),
        ),
        stage_report=None,
    )

    assert runtime.quality_metrics == ()
    with pytest.raises(EvaluationReportError, match="runtime"):
        _report(stage=EvaluationStage.RUNTIME, reproducibility=_metadata())


def test_a_metric_result_appears_once_per_stratum() -> None:
    near = _value(strata=(("range_band", "near"),))
    far = _value(value=0.25, strata=(("range_band", "far"),))

    report = _report(quality_metrics=(_value(), near, far))

    assert len(report.quality_metrics) == 3
    with pytest.raises(EvaluationReportError, match="repeated"):
        _report(quality_metrics=(_value(), _value(value=0.6)))
    with pytest.raises(ValueError, match="strat"):
        _value(strata=(("range_band", "near"), ("range_band", "far")))


def test_decoding_revalidates_against_the_registry() -> None:
    document = encode_evaluation_report(_report())
    document["quality_metrics"][0]["value"] = 9.0  # type: ignore[index]

    with pytest.raises(EvaluationReportError, match="range"):
        decode_evaluation_report(document, REGISTRY)
    stale = encode_evaluation_report(_report())
    stale["reproducibility"]["metric_registry"]["digest"] = "sha256:" + "0" * 64  # type: ignore[index]
    with pytest.raises(EvaluationReportError, match="registry"):
        decode_evaluation_report(stale, REGISTRY)
    with pytest.raises(EvaluationReportError, match="schema"):
        decode_evaluation_report(
            {**encode_evaluation_report(_report()), "schema": "x/v0"}, REGISTRY
        )
    broken = encode_evaluation_report(_report())
    del broken["stage"]
    with pytest.raises(EvaluationReportError, match="stage"):
        decode_evaluation_report(broken, REGISTRY)


def test_reports_are_written_immutably(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "region-discovery.json"
    report = _report()

    write_evaluation_report(path, report)

    assert (
        decode_evaluation_report(json.loads(path.read_text(encoding="utf-8")), REGISTRY) == report
    )
    with pytest.raises(FileExistsError):
        write_evaluation_report(path, report)
