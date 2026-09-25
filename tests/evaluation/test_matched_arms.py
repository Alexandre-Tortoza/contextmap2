"""Matched-arm invariants of prompt, evidence and backend ablations (#545).

An arm advertised as varying one factor must share every other scientific input with the
reference arm. Planned differences are verified field by field when the manifest is built;
executed differences are verified when the comparison is built, and an arm that differs in
anything undeclared never enters the matched metrics.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from experiment_builders import (
    FUSION_CHANNELS,
    FUSION_POLICY,
    INTERPRETER_BACKEND,
    PROMPT_POLICY,
    QWEN_REVISION,
    REGISTRY,
    SEMANTICS,
    VIEW_POLICY,
    factorial_experiment,
    make_executor,
    make_report,
    pinned,
    reference_with_semantics,
    replace_stage_artifact,
    request_policy_experiment,
    request_policy_topology,
    semantic_configuration,
    validated_reference,
)
from reference_set_builders import content_hash

from contextmap.evaluation.experiment_runner import (
    ArmExecution,
    ArmStatus,
    ArmUnavailableError,
    FailureKind,
    IncompleteComparisonError,
    MatchStatus,
    read_verified_document,
    require_complete_comparison,
    run_experiment,
)
from contextmap.evaluation.experiments import (
    AblationMode,
    ArmDifference,
    DifferenceKind,
    ExperimentArm,
    ExperimentError,
    ExperimentManifest,
    StageImplementation,
    VariationKind,
    arm_differences,
    decode_experiment,
    encode_experiment,
)
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet
from contextmap.evaluation.reference_set import ReferenceSetManifest
from contextmap.evaluation.report_schema import ArtifactIdentity, EvaluatorIdentity

SEMANTIC = "semantic_interpretation"
QUALITY = {"semantic.acceptable_claim_rate": 0.6, "semantic.unsupported_claim_rate": 0.4}


@pytest.fixture
def reference() -> ReferenceSetManifest:
    return reference_with_semantics()


@pytest.fixture
def validated(tmp_path: Path) -> ValidatedReferenceSet:
    return validated_reference(tmp_path / "reference")


def _values(manifest: ExperimentManifest) -> dict[str, dict[str, float]]:
    return {arm.arm_id: dict(QUALITY) for arm in manifest.arms}


def _run(
    validated: ValidatedReferenceSet, root: Path, manifest: ExperimentManifest, **kw: Any
) -> Any:
    executor, _ = make_executor(_values(manifest), **kw)
    return run_experiment(
        manifest, executor=executor, registry=REGISTRY, reference_set=validated, root=root
    )


def _with_topology(
    manifest: ExperimentManifest, arm_id: str, **kw: Any
) -> tuple[ExperimentArm, ...]:
    """Return the manifest's arms with ``arm_id`` resolved to another request-policy topology."""
    return tuple(
        replace(arm, topology=request_policy_topology(**kw)) if arm.arm_id == arm_id else arm
        for arm in manifest.arms
    )


def _comparison_arm(run: Any, arm_id: str) -> Any:
    return next(item for item in run.comparison.arms if item.arm_id == arm_id)


# ------------------------------------------------------------- planned: declared factors


def test_a_prompt_only_ablation_records_exactly_its_declared_difference(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference)
    baseline, prompt = manifest.arms

    differences = arm_differences(manifest.variables, baseline, prompt)

    assert differences == (
        ArmDifference(
            stage_id=SEMANTIC,
            kind=DifferenceKind.CONFIGURATION,
            field="request_policy.prompt_policy",
            reference_value='"region/v1"',
            value='"region/v2"',
            declared_by=("prompt_policy",),
        ),
    )


def test_a_view_only_ablation_records_the_ordered_views_it_changes(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference, (VIEW_POLICY,))
    baseline, views = manifest.arms

    (difference,) = arm_differences(manifest.variables, baseline, views)

    assert difference.field == "request_policy.view_policy"
    assert (difference.reference_value, difference.value) == (
        '["tight_crop"]',
        '["tight_crop","masked_subject"]',
    )
    assert difference.declared_by == ("view_policy",)


def test_a_backend_only_ablation_keeps_the_request_policy_matched(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference, (INTERPRETER_BACKEND,))
    baseline, gemini = manifest.arms

    differences = arm_differences(manifest.variables, baseline, gemini)

    assert {item.declared_by for item in differences} == {("semantic_backend",)}
    assert DifferenceKind.BACKEND in {item.kind for item in differences}
    fields = {item.field for item in differences if item.field is not None}
    assert "interpreter.backend" in fields
    assert "interpreter.qwen.revision" in fields
    assert "interpreter.gemini.max_output_tokens" in fields
    assert not any(name.startswith("request_policy") for name in fields)


def test_a_full_factorial_attributes_each_field_to_its_own_variable(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(
        reference, (PROMPT_POLICY, VIEW_POLICY), mode=AblationMode.FULL_FACTORIAL
    )
    both = next(
        arm
        for arm in manifest.arms
        if arm.assignment
        == {"prompt_policy": "region/v2", "view_policy": "tight_crop+masked_subject"}
    )

    differences = arm_differences(manifest.variables, manifest.baseline_arm, both)

    assert {(item.field, item.declared_by) for item in differences} == {
        ("request_policy.prompt_policy", ("prompt_policy",)),
        ("request_policy.view_policy", ("view_policy",)),
    }


# ------------------------------------------------------------ planned: hidden confounders


def test_an_accidental_model_revision_change_in_a_prompt_ablation_is_refused(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference)
    drifted = "2" * 40

    with pytest.raises(ExperimentError, match="undeclared change") as error:
        replace(
            manifest,
            arms=_with_topology(
                manifest,
                "arm-1",
                changes={
                    "request_policy.prompt_policy": "region/v2",
                    "interpreter.qwen.revision": drifted,
                },
            ),
        )

    message = str(error.value)
    assert "interpreter.qwen.revision" in message
    assert QWEN_REVISION in message and drifted in message
    assert "request_policy.prompt_policy" not in message


def test_an_accidental_output_schema_change_is_refused_with_both_values(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference)

    with pytest.raises(ExperimentError, match="undeclared change") as error:
        replace(
            manifest,
            arms=_with_topology(
                manifest,
                "arm-1",
                changes={
                    "request_policy.prompt_policy": "region/v2",
                    "request_policy.output_schema": "semantic-response/2",
                },
            ),
        )

    message = str(error.value)
    assert "request_policy.output_schema" in message
    assert "semantic-response/1" in message and "semantic-response/2" in message


def test_generation_settings_cannot_change_under_a_prompt_ablation(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference)

    with pytest.raises(ExperimentError, match=r"interpreter\.qwen\.max_new_tokens"):
        replace(
            manifest,
            arms=_with_topology(
                manifest,
                "arm-1",
                changes={
                    "request_policy.prompt_policy": "region/v2",
                    "interpreter.qwen.max_new_tokens": 512,
                },
            ),
        )


def test_a_backend_ablation_cannot_also_change_the_request_policy(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference, (INTERPRETER_BACKEND,))

    with pytest.raises(ExperimentError, match=r"request_policy\.view_policy"):
        replace(
            manifest,
            arms=_with_topology(
                manifest,
                "arm-1",
                backend="gemini",
                changes={"request_policy.view_policy": ["full_frame"]},
            ),
        )


def test_a_changed_physical_sequence_is_refused_with_both_identities(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference)

    with pytest.raises(ExperimentError, match="undeclared change") as error:
        replace(
            manifest,
            arms=_with_topology(
                manifest,
                "arm-1",
                changes={"request_policy.prompt_policy": "region/v2"},
                sequence="sequence-0002",
            ),
        )

    message = str(error.value)
    assert "'ingestion'" in message
    assert "sequence-0001" in message and "sequence-0002" in message


def test_a_configuration_variable_without_declared_fields_cannot_change_recorded_fields(
    reference: ReferenceSetManifest,
) -> None:
    undeclared = replace(PROMPT_POLICY, configuration_fields=())

    with pytest.raises(ExperimentError, match=r"request_policy\.prompt_policy"):
        request_policy_experiment(reference, (undeclared,))


def test_declared_fields_cannot_vouch_for_a_stage_that_records_only_its_digest(
    reference: ReferenceSetManifest,
) -> None:
    manifest = factorial_experiment(reference)
    only_the_policy = replace(FUSION_POLICY, configuration_fields=("accumulation.policy",))

    with pytest.raises(ExperimentError, match="the configuration of stage 'semantic_fusion'"):
        replace(manifest, variables=(only_the_policy, FUSION_CHANNELS))


def test_a_stage_records_its_configuration_in_every_arm_or_in_none(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference)
    baseline, prompt = manifest.arms
    semantic = next(item for item in prompt.topology.stages if item.stage_id == SEMANTIC)
    digest_only = replace(semantic.implementation, configuration=None)
    stages = tuple(
        replace(item, implementation=digest_only) if item.stage_id == SEMANTIC else item
        for item in prompt.topology.stages
    )

    with pytest.raises(ExperimentError, match="field by field"):
        replace(
            manifest,
            arms=(baseline, replace(prompt, topology=replace(prompt.topology, stages=stages))),
        )


def test_the_configuration_digest_is_the_digest_of_the_recorded_fields() -> None:
    configuration = semantic_configuration()
    recorded = StageImplementation.from_configuration(
        backend_id="qwen", backend_version="1", model="m", configuration=configuration
    )

    assert recorded.configuration == configuration
    with pytest.raises(ValueError, match="digest of the recorded configuration"):
        replace(recorded, configuration_digest=content_hash("something else"))
    with pytest.raises(ValueError, match="JSON"):
        StageImplementation.from_configuration(
            backend_id="qwen", backend_version="1", model="m", configuration={"x": object()}
        )


def test_a_topology_variable_declares_stages_not_configuration_fields() -> None:
    with pytest.raises(ValueError, match="topology"):
        replace(PROMPT_POLICY, kind=VariationKind.TOPOLOGY)
    with pytest.raises(ValueError, match="unique"):
        replace(PROMPT_POLICY, configuration_fields=("a", "a"))
    with pytest.raises(ValueError, match="empty"):
        replace(PROMPT_POLICY, configuration_fields=(" ",))


def test_recorded_fields_and_declared_fields_round_trip_in_the_manifest(
    reference: ReferenceSetManifest,
) -> None:
    manifest = request_policy_experiment(reference, (INTERPRETER_BACKEND,))

    document = json.loads(json.dumps(encode_experiment(manifest)))
    restored = decode_experiment(document)

    assert restored == manifest
    assert restored.digest() == manifest.digest()
    assert document["variables"][0]["configuration_fields"] == ["interpreter"]
    stage = next(
        item for item in document["arms"][1]["topology"]["stages"] if item["stage_id"] == SEMANTIC
    )
    assert stage["implementation"]["configuration"]["interpreter"]["backend"] == "gemini"


# ------------------------------------------------------------ executed: matched comparison


def test_a_prompt_only_run_is_a_matched_comparison_with_its_identities_reported(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = request_policy_experiment(validated.manifest)

    run = _run(validated, tmp_path / "run", manifest)

    comparison = run.comparison
    arm = _comparison_arm(run, "arm-1")
    assert arm.match is not None
    assert arm.match.status is MatchStatus.MATCHED
    assert arm.match.reference_arm_id == "baseline"
    assert arm.match.factors == (("prompt_policy", "region/v2"),)
    assert [item.field for item in arm.match.declared_differences] == [
        "request_policy.prompt_policy"
    ]
    assert arm.match.undeclared_differences == ()
    assert arm.match.matched_sample_count == comparison.physical_sample_count == 2
    assert _comparison_arm(run, "baseline").match is None
    assert comparison.complete
    require_complete_comparison(comparison)

    document = read_verified_document(tmp_path / "run" / "comparison.json")
    assert document["matching"] == {
        "reference_arm_id": "baseline",
        "matched": 1,
        "invalid": 0,
        "incomplete": 0,
        "excluded_arm_ids": [],
    }
    record = next(item for item in document["arms"] if item["arm_id"] == "arm-1")
    assert record["assignments"] == [["prompt_policy", "region/v2"]]
    configurations = dict(record["stage_configurations"])
    semantic = manifest.arm("arm-1").topology.stage(SEMANTIC)
    assert configurations[SEMANTIC] == semantic.implementation.configuration_digest
    artifacts = {item["stage_id"]: item["artifact"] for item in record["stage_artifacts"]}
    assert artifacts["ingestion"] == pinned("sequence", "sequence-0001").to_record()
    assert record["match"]["status"] == "matched"
    assert record["match"]["declared_differences"][0]["declared_by"] == ["prompt_policy"]


def test_an_unaffected_stage_that_produced_other_content_invalidates_the_pair(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = request_policy_experiment(validated.manifest)
    other_pose = ArtifactIdentity(
        kind="stage_output", artifact_id="state_estimation-other", digest=content_hash("pose-2")
    )

    def drift(_manifest: Any, arm: ExperimentArm, result: ArmExecution) -> ArmExecution:
        if arm.arm_id != "arm-1":
            return result
        return replace_stage_artifact(result, "state_estimation", other_pose)

    run = _run(validated, tmp_path / "run", manifest, tweak=drift)

    arm = _comparison_arm(run, "arm-1")
    assert arm.status is ArmStatus.COMPLETED
    assert arm.match.status is MatchStatus.INVALID
    (difference,) = arm.match.undeclared_differences
    assert (difference.stage_id, difference.kind) == (
        "state_estimation",
        DifferenceKind.STAGE_ARTIFACT,
    )
    assert difference.declared_by == ()
    assert "state_estimation-other" in (difference.value or "")
    assert arm.match.matched_sample_count == 0
    assert not arm.comparable
    assert "arm-1" in run.reports
    assert (tmp_path / "run" / "arms" / "arm-1" / "report.json").is_file()
    assert all(
        entry.arm_id != "arm-1" for metric in run.comparison.metrics for entry in metric.entries
    )
    assert not run.comparison.complete
    with pytest.raises(IncompleteComparisonError, match="state_estimation"):
        require_complete_comparison(run.comparison)
    document = read_verified_document(tmp_path / "run" / "comparison.json")
    assert document["matching"]["invalid"] == 1
    assert document["matching"]["excluded_arm_ids"] == ["arm-1"]


def _other_evaluator(execution: ArmExecution) -> ArmExecution:
    metadata = replace(
        execution.report.reproducibility,
        evaluator=EvaluatorIdentity(evaluator_id="fake-evaluator", evaluator_version="2"),
    )
    return replace(execution, report=replace(execution.report, reproducibility=metadata))


def _other_code(execution: ArmExecution) -> ArmExecution:
    metadata = replace(execution.report.reproducibility, code_version="0.0.2")
    return replace(execution, report=replace(execution.report, reproducibility=metadata))


def _other_annotations(execution: ArmExecution) -> ArmExecution:
    metadata = replace(execution.report.reproducibility, annotation_schemas=(SEMANTICS,))
    return replace(execution, report=replace(execution.report, reproducibility=metadata))


@pytest.mark.parametrize(
    ("tweak", "kind"),
    [
        (_other_evaluator, DifferenceKind.EVALUATOR),
        (_other_code, DifferenceKind.CODE_VERSION),
        (_other_annotations, DifferenceKind.ANNOTATION_SCHEMAS),
    ],
)
def test_an_arm_evaluated_by_other_code_or_references_is_not_matched(
    validated: ValidatedReferenceSet, tmp_path: Path, tweak: Any, kind: DifferenceKind
) -> None:
    manifest = request_policy_experiment(validated.manifest)

    run = _run(
        validated,
        tmp_path / "run",
        manifest,
        tweak=lambda _m, arm, result: tweak(result) if arm.arm_id == "arm-1" else result,
    )

    arm = _comparison_arm(run, "arm-1")
    assert arm.match.status is MatchStatus.INVALID
    assert [item.kind for item in arm.match.undeclared_differences] == [kind]
    assert not run.comparison.complete


def test_a_valid_backend_only_run_is_matched_with_the_backend_as_its_factor(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = request_policy_experiment(validated.manifest, (INTERPRETER_BACKEND,))

    run = _run(validated, tmp_path / "run", manifest)

    arm = _comparison_arm(run, "arm-1")
    assert arm.match.status is MatchStatus.MATCHED
    assert arm.match.factors == (("semantic_backend", "gemini"),)
    assert all(item.declared_by == ("semantic_backend",) for item in arm.match.declared_differences)


# ------------------------------------------ executed: physical inputs, repeats and failures


def test_an_arm_that_evaluated_another_reference_set_or_sequence_stays_an_excluded_result(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    three_prompts = replace(PROMPT_POLICY, values=("region/v1", "region/v2", "region/v3"))
    manifest = request_policy_experiment(validated.manifest, (three_prompts,))

    def drift(_manifest: Any, arm: ExperimentArm, result: ArmExecution) -> ArmExecution:
        if arm.arm_id == "arm-1":
            other = replace(manifest.selection.reference_set, version="9.9.9")
            return replace(
                result,
                report=make_report(manifest, arm, QUALITY, reference_set=other),
            )
        if arm.arm_id == "arm-2":
            return replace_stage_artifact(result, "ingestion", pinned("sequence", "sequence-0002"))
        return result

    run = _run(validated, tmp_path / "run", manifest, tweak=drift)

    for arm_id, reason in (("arm-1", "reference set"), ("arm-2", "sequence-0002")):
        arm = _comparison_arm(run, arm_id)
        assert arm.status is ArmStatus.FAILED
        assert arm.failure.kind is FailureKind.INVALID_RESULT
        assert reason in arm.failure.message
        assert arm.match.status is MatchStatus.INCOMPLETE
        assert arm_id in (arm.match.reason or "")
        assert not arm.comparable
    document = read_verified_document(tmp_path / "run" / "comparison.json")
    assert document["matching"]["incomplete"] == 2
    assert document["matching"]["matched"] == 0
    assert document["matching"]["excluded_arm_ids"] == ["arm-1", "arm-2"]


def test_repeated_prompt_variants_over_one_observation_are_not_new_samples(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    three_prompts = replace(PROMPT_POLICY, values=("region/v1", "region/v2", "region/v3"))
    manifest = replace(
        request_policy_experiment(validated.manifest, (three_prompts,)), repetitions_per_sample=3
    )

    run = _run(validated, tmp_path / "run", manifest)

    comparison = run.comparison
    assert comparison.physical_sample_count == 2
    assert comparison.repetitions_per_sample == 3
    for arm_id in ("arm-1", "arm-2"):
        match = _comparison_arm(run, arm_id).match
        assert match.status is MatchStatus.MATCHED
        # Três prompts x três repetições sobre as mesmas duas amostras continuam duas amostras.
        assert match.matched_sample_count == 2
    document = read_verified_document(tmp_path / "run" / "comparison.json")
    assert document["physical_sample_count"] == 2
    assert document["repetitions_per_sample"] == 3


def test_failed_and_unavailable_arms_are_kept_and_never_trimmed_into_a_matched_set(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = request_policy_experiment(
        validated.manifest, (PROMPT_POLICY, VIEW_POLICY), mode=AblationMode.ONE_AT_A_TIME
    )
    failures = {
        "arm-1": RuntimeError("CUDA out of memory"),
        "arm-2": ArmUnavailableError("the masked-subject view needs SAM masks that were skipped"),
    }

    run = _run(validated, tmp_path / "run", manifest, failures=failures)

    comparison = run.comparison
    assert [item.arm_id for item in comparison.arms] == ["baseline", "arm-1", "arm-2"]
    oom, skipped = _comparison_arm(run, "arm-1"), _comparison_arm(run, "arm-2")
    assert (oom.status, skipped.status) == (ArmStatus.FAILED, ArmStatus.UNAVAILABLE)
    assert oom.match.status is skipped.match.status is MatchStatus.INCOMPLETE
    assert "CUDA out of memory" in (oom.match.reason or "")
    assert oom.match.factors == (("prompt_policy", "region/v2"),)
    assert [item.field for item in skipped.match.declared_differences] == [
        "request_policy.view_policy"
    ]
    assert not comparison.complete
    assert comparison.incomplete_arm_ids == ("arm-1", "arm-2")
    document = read_verified_document(tmp_path / "run" / "comparison.json")
    assert document["matching"]["incomplete"] == 2
    assert {item["arm_id"] for item in document["arms"]} == {"baseline", "arm-1", "arm-2"}


def test_a_failed_reference_arm_leaves_every_pair_incomplete(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = request_policy_experiment(validated.manifest)

    run = _run(validated, tmp_path / "run", manifest, failures={"baseline": RuntimeError("boom")})

    match = _comparison_arm(run, "arm-1").match
    assert match.status is MatchStatus.INCOMPLETE
    assert "baseline" in (match.reason or "")
    assert match.matched_sample_count == 0
