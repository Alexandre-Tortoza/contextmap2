"""Experiment execution, per-arm run manifests and comparison tests."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from experiment_builders import (
    REGISTRY,
    backend_experiment,
    factorial_experiment,
    make_executor,
    make_report,
    pinned,
    reference_with_semantics,
    topology_experiment,
    validated_reference,
)
from reference_set_builders import content_hash, make_valid_manifest

from contextmap.evaluation.experiment_runner import (
    ArmExecution,
    ArmStatus,
    ArmUnavailableError,
    FailureKind,
    IncompleteComparisonError,
    read_verified_document,
    require_complete_comparison,
    run_experiment,
)
from contextmap.evaluation.experiments import (
    ExperimentError,
    ExperimentManifest,
    ExperimentPurpose,
    experiment_artifact_identity,
)
from contextmap.evaluation.metrics import EvaluationStage, MetricKind
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet
from contextmap.evaluation.report_schema import MetricResult, MetricStatus

BACKEND_VALUES = {
    "baseline": {"semantic.acceptable_claim_rate": 0.6, "semantic.unsupported_claim_rate": 0.4},
    "arm-1": {"semantic.acceptable_claim_rate": 0.8, "semantic.unsupported_claim_rate": 0.2},
}
FUSION_VALUES = {
    "baseline": {"fusion.reference_recovery.rate": 0.5},
    "arm-1": {"fusion.reference_recovery.rate": 0.6},
    "arm-2": {"fusion.reference_recovery.rate": 0.7},
    "arm-3": {"fusion.reference_recovery.rate": 0.4},
}
SENSOR_VALUES = {
    "baseline": {"association.visible_support.ratio": 0.5},
    "arm-1": {"association.visible_support.ratio": 0.75},
}


@pytest.fixture
def validated(tmp_path: Path) -> ValidatedReferenceSet:
    return validated_reference(tmp_path / "reference")


def _run(
    validated: ValidatedReferenceSet,
    root: Path,
    manifest: ExperimentManifest,
    executor: Any,
) -> Any:
    return run_experiment(
        manifest, executor=executor, registry=REGISTRY, reference_set=validated, root=root
    )


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


# --------------------------------------------------------------------------- happy path


def test_every_arm_gets_a_run_manifest_and_a_report_and_the_comparison_is_emitted(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, calls = make_executor(BACKEND_VALUES)
    root = tmp_path / "run"

    run = _run(validated, root, manifest, executor)

    assert calls == ["baseline", "arm-1"]
    for name in ("experiment.json", "comparison.json"):
        assert (root / name).is_file()
    for arm in ("baseline", "arm-1"):
        assert (root / "arms" / arm / "run.json").is_file()
        assert (root / "arms" / arm / "report.json").is_file()
    assert [item.status for item in run.arm_runs] == [ArmStatus.COMPLETED, ArmStatus.COMPLETED]
    assert run.comparison.complete
    assert set(run.reports) == {"baseline", "arm-1"}
    assert read_verified_document(root / "comparison.json")["complete"] is True


def test_the_run_manifest_preserves_topology_backend_config_and_reference_set_identity(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, _ = make_executor(BACKEND_VALUES)

    run = _run(validated, tmp_path / "run", manifest, executor)

    arm = manifest.arm("arm-1")
    record = read_verified_document(tmp_path / "run" / "arms" / "arm-1" / "run.json")
    assert record["arm_id"] == "arm-1"
    assert record["topology_digest"] == arm.topology.digest()
    interpreter = next(
        item
        for item in record["topology"]["stages"]
        if item["stage_id"] == "semantic_interpretation"
    )
    assert interpreter["implementation"]["backend_id"] == "gemini"
    assert interpreter["implementation"]["model"] == "gemini-pro"
    assert interpreter["implementation"]["configuration_digest"].startswith("sha256:")
    assert record["reference_set"] == validated.manifest.identity().to_record()
    assert record["experiment"]["digest"] == manifest.digest()
    assert record["experiment"] == experiment_artifact_identity(manifest).to_record()
    assert run.arm_runs[1].topology_digest == arm.topology.digest()


def test_the_comparison_lists_metrics_side_by_side_without_a_winner(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, _ = make_executor(BACKEND_VALUES)

    run = _run(validated, tmp_path / "run", manifest, executor)

    by_metric = {item.metric: item for item in run.comparison.metrics}
    acceptable = by_metric["semantic.acceptable_claim_rate"]
    entries = {item.arm_id: item for item in acceptable.entries}
    assert acceptable.kind is MetricKind.QUALITY
    assert entries["baseline"].value == 0.6 and entries["baseline"].delta_from_baseline is None
    assert entries["arm-1"].value == 0.8
    assert entries["arm-1"].delta_from_baseline == pytest.approx(0.2)
    assert acceptable.same_population
    forbidden = {"winner", "score", "rank", "best", "overall", "ranking"}
    assert not (_keys(run.comparison.to_record()) & forbidden)


def test_quality_and_resource_metrics_stay_in_separate_kinds(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, _ = make_executor(BACKEND_VALUES)

    run = _run(validated, tmp_path / "run", manifest, executor)

    kinds = {item.metric: item.kind for item in run.comparison.metrics}
    assert kinds["runtime.wall_time"] is MetricKind.PERFORMANCE
    assert kinds["runtime.peak_memory"] is MetricKind.PERFORMANCE
    assert kinds["semantic.unsupported_claim_rate"] is MetricKind.QUALITY
    assert {item.metric for item in run.comparison.metrics} == {
        "semantic.acceptable_claim_rate",
        "semantic.unsupported_claim_rate",
        "runtime.wall_time",
        "runtime.peak_memory",
    }


def test_a_dag_insertion_comparison_proves_the_shared_upstream_artifact(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = topology_experiment(validated.manifest)
    executor, _ = make_executor(SENSOR_VALUES)

    run = _run(validated, tmp_path / "run", manifest, executor)

    shared = {item.stage_id: item for item in run.comparison.shared_artifacts}
    assert shared["dino"].artifact == pinned("perception_run", "dino-run-X")
    assert shared["dino"].arm_ids == ("baseline", "arm-1")
    assert shared["ingestion"].artifact == pinned("sequence", "sequence-0001")
    assert "sensor_association" not in shared
    assert "feature_resolution_enhancement" not in shared


def test_a_full_factorial_runs_every_arm_with_deltas_against_the_baseline(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = factorial_experiment(validated.manifest)
    executor, calls = make_executor(FUSION_VALUES)

    run = _run(validated, tmp_path / "run", manifest, executor)

    assert calls == ["baseline", "arm-1", "arm-2", "arm-3"]
    recovery = next(item for item in run.comparison.metrics if item.metric.startswith("fusion."))
    deltas = {item.arm_id: item.delta_from_baseline for item in recovery.entries}
    assert deltas["arm-2"] == pytest.approx(0.2)
    assert deltas["arm-3"] == pytest.approx(-0.1)
    assert run.comparison.shared_artifacts[0].stage_id == "sensor_association"


def test_repeated_inference_does_not_multiply_the_physical_samples(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = replace(backend_experiment(validated.manifest), repetitions_per_sample=5)
    executor, _ = make_executor(BACKEND_VALUES)

    run = _run(validated, tmp_path / "run", manifest, executor)

    assert run.comparison.physical_sample_count == len(manifest.selection.sample_ids) == 2
    assert run.comparison.repetitions_per_sample == 5


def test_a_run_is_reproducible_from_the_manifest(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    first_executor, _ = make_executor(BACKEND_VALUES)
    second_executor, _ = make_executor(BACKEND_VALUES)

    first = _run(validated, tmp_path / "first", manifest, first_executor)
    second = _run(validated, tmp_path / "second", manifest, second_executor)

    assert first.comparison.digest() == second.comparison.digest()
    assert (tmp_path / "first" / "comparison.json").read_bytes() == (
        tmp_path / "second" / "comparison.json"
    ).read_bytes()


def test_a_run_directory_is_immutable(validated: ValidatedReferenceSet, tmp_path: Path) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, _ = make_executor(BACKEND_VALUES)
    root = tmp_path / "run"
    _run(validated, root, manifest, executor)

    with pytest.raises(FileExistsError):
        _run(validated, root, manifest, make_executor(BACKEND_VALUES)[0])


# ----------------------------------------------------------- failed or unavailable arms


def test_an_unavailable_arm_stays_explicit_and_no_fallback_runs(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, calls = make_executor(
        BACKEND_VALUES, failures={"arm-1": ArmUnavailableError("GEMINI_API_KEY is not set")}
    )

    run = _run(validated, tmp_path / "run", manifest, executor)

    failed = run.arm_runs[1]
    assert calls == ["baseline", "arm-1"]
    assert failed.status is ArmStatus.UNAVAILABLE
    assert failed.failure is not None
    assert failed.failure.kind is FailureKind.UNAVAILABLE
    assert "GEMINI_API_KEY" in failed.failure.message
    assert "arm-1" not in run.reports
    assert (tmp_path / "run" / "arms" / "arm-1" / "run.json").is_file()
    assert not (tmp_path / "run" / "arms" / "arm-1" / "report.json").exists()


def test_a_failed_arm_cannot_masquerade_as_a_comparable_result(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, _ = make_executor(
        BACKEND_VALUES, failures={"arm-1": RuntimeError("CUDA out of memory")}
    )

    run = _run(validated, tmp_path / "run", manifest, executor)

    comparison = run.comparison
    failed = next(item for item in comparison.arms if item.arm_id == "arm-1")
    assert failed.status is ArmStatus.FAILED
    assert failed.failure is not None
    assert (failed.failure.kind, failed.failure.error_type) == (FailureKind.ERROR, "RuntimeError")
    assert "CUDA out of memory" in failed.failure.message
    assert not failed.comparable
    assert not comparison.complete
    assert all(entry.arm_id != "arm-1" for item in comparison.metrics for entry in item.entries)
    with pytest.raises(IncompleteComparisonError, match="arm-1"):
        require_complete_comparison(comparison)
    document = read_verified_document(tmp_path / "run" / "comparison.json")
    assert document["complete"] is False


def test_a_failed_baseline_leaves_no_deltas_and_an_incomplete_comparison(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, calls = make_executor(BACKEND_VALUES, failures={"baseline": RuntimeError("boom")})

    run = _run(validated, tmp_path / "run", manifest, executor)

    assert calls == ["baseline", "arm-1"]
    assert not run.comparison.complete
    assert all(
        entry.delta_from_baseline is None
        for item in run.comparison.metrics
        for entry in item.entries
    )


def test_a_complete_comparison_passes_the_guard(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)

    run = _run(validated, tmp_path / "run", manifest, make_executor(BACKEND_VALUES)[0])

    require_complete_comparison(run.comparison)


def _other_reference(execution: ArmExecution) -> ArmExecution:
    other = replace(execution.report.reproducibility.reference_set, version="9.9.9")  # type: ignore[type-var]
    return replace(
        execution,
        report=replace(
            execution.report,
            reproducibility=replace(execution.report.reproducibility, reference_set=other),
        ),
    )


def _wrong_configuration(execution: ArmExecution) -> ArmExecution:
    return replace(
        execution,
        report=replace(
            execution.report,
            reproducibility=replace(
                execution.report.reproducibility, configuration_digest=content_hash("other")
            ),
        ),
    )


def _not_bound_to_the_manifest(execution: ArmExecution) -> ArmExecution:
    return replace(
        execution,
        report=replace(
            execution.report,
            reproducibility=replace(execution.report.reproducibility, input_artifacts=()),
        ),
    )


def _missing_quality_metric(execution: ArmExecution) -> ArmExecution:
    return replace(
        execution,
        report=replace(execution.report, quality_metrics=execution.report.quality_metrics[:1]),
    )


def _missing_resource_metric(execution: ArmExecution) -> ArmExecution:
    return replace(execution, report=replace(execution.report, performance_metrics=()))


def _recomputed_pinned_artifact(execution: ArmExecution) -> ArmExecution:
    changed = tuple(
        replace(item, artifact=pinned("perception_run", "recomputed"))
        if item.stage_id == "region_discovery"
        else item
        for item in execution.stage_artifacts
    )
    return replace(execution, stage_artifacts=changed)


def _missing_stage_artifact(execution: ArmExecution) -> ArmExecution:
    return replace(execution, stage_artifacts=execution.stage_artifacts[:-1])


def _another_stage(execution: ArmExecution) -> ArmExecution:
    return replace(
        execution, report=replace(execution.report, stage=EvaluationStage.REGION_DISCOVERY)
    )


@pytest.mark.parametrize(
    ("tweak", "reason"),
    [
        (_other_reference, "reference set"),
        (_wrong_configuration, "configuration"),
        (_not_bound_to_the_manifest, "experiment manifest"),
        (_missing_quality_metric, "quality metric"),
        (_missing_resource_metric, "resource"),
        (_recomputed_pinned_artifact, "pinned"),
        (_missing_stage_artifact, "stage artifact"),
        (_another_stage, "stage"),
    ],
)
def test_an_inconsistent_result_is_a_failed_arm_not_a_comparable_one(
    validated: ValidatedReferenceSet,
    tmp_path: Path,
    tweak: Callable[[ArmExecution], ArmExecution],
    reason: str,
) -> None:
    manifest = backend_experiment(validated.manifest)
    executor, _ = make_executor(
        BACKEND_VALUES,
        tweak=lambda _manifest, arm, result: tweak(result) if arm.arm_id == "arm-1" else result,
    )

    run = _run(validated, tmp_path / "run", manifest, executor)

    failed = run.arm_runs[1]
    assert failed.status is ArmStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.kind is FailureKind.INVALID_RESULT
    assert reason in failed.failure.message
    assert not run.comparison.complete
    assert "arm-1" not in run.reports


def test_an_unsupported_resource_metric_is_reported_as_such(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)

    def unsupported_memory(
        manifest: ExperimentManifest, arm: Any, result: ArmExecution
    ) -> ArmExecution:
        if arm.arm_id != "arm-1":
            return result
        count = len(manifest.selection.sample_ids)
        performance = (
            MetricResult(
                metric="runtime.wall_time",
                metric_version="1",
                status=MetricStatus.VALUE,
                value=2.0,
                sample_count=count,
            ),
            MetricResult(
                metric="runtime.peak_memory",
                metric_version="1",
                status=MetricStatus.UNSUPPORTED,
                value=None,
                sample_count=None,
            ),
        )
        return replace(
            result,
            report=make_report(manifest, arm, BACKEND_VALUES[arm.arm_id], performance=performance),
        )

    run = _run(
        validated,
        tmp_path / "run",
        manifest,
        make_executor(BACKEND_VALUES, tweak=unsupported_memory)[0],
    )

    memory = next(item for item in run.comparison.metrics if item.metric == "runtime.peak_memory")
    entry = next(item for item in memory.entries if item.arm_id == "arm-1")
    assert run.comparison.complete
    assert entry.status is MetricStatus.UNSUPPORTED
    assert entry.value is None and entry.delta_from_baseline is None


# ------------------------------------------------------------------------- prerequisites


def test_an_experiment_of_another_reference_set_version_is_refused_before_running(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    other = make_valid_manifest(version="2.0.0", annotations=reference_with_semantics().annotations)
    manifest = backend_experiment(other)
    executor, calls = make_executor(BACKEND_VALUES)

    with pytest.raises(ExperimentError, match="reference set"):
        _run(validated, tmp_path / "run", manifest, executor)

    assert calls == []
    assert not (tmp_path / "run").exists()


def test_tuning_on_the_held_out_split_is_refused_before_running(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(
        validated.manifest, purpose=ExperimentPurpose.TUNING, split="test"
    )
    executor, calls = make_executor(BACKEND_VALUES)

    with pytest.raises(ExperimentError, match="held-out"):
        _run(validated, tmp_path / "run", manifest, executor)

    assert calls == []


def test_a_verified_document_rejects_tampering(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = backend_experiment(validated.manifest)
    _run(validated, tmp_path / "run", manifest, make_executor(BACKEND_VALUES)[0])
    path = tmp_path / "run" / "comparison.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["complete"] = False
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ExperimentError, match="digest"):
        read_verified_document(path)
