"""Experiment manifest, ablation matrix and controlled-variation tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from experiment_builders import (
    FUSION_CHANNELS,
    FUSION_POLICY,
    REGISTRY,
    SEMANTIC_BACKEND,
    backend_experiment,
    factorial_experiment,
    modify_stage,
    pinned,
    reference_with_semantics,
    stage,
    topology_experiment,
)
from reference_set_builders import content_hash, make_valid_manifest

from contextmap.evaluation.experiments import (
    EXPERIMENT_SCHEMA,
    AblationMode,
    ExperimentError,
    ExperimentPurpose,
    FixedControl,
    MetricRef,
    ResolvedTopology,
    SelectionBinding,
    VariationKind,
    ablation_cells,
    decode_experiment,
    encode_experiment,
    read_experiment,
    validate_experiment_manifest,
    write_experiment,
)
from contextmap.evaluation.metrics import EvaluationStage
from contextmap.evaluation.reference_set import ReferenceSetError
from contextmap.evaluation.report_schema import ArtifactIdentity


@pytest.fixture
def reference():  # type: ignore[no-untyped-def]
    return reference_with_semantics()


# ---------------------------------------------------------------- matrix and identity


def test_the_manifest_round_trips_with_a_stable_digest(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = topology_experiment(reference)

    document = json.loads(json.dumps(encode_experiment(manifest)))
    restored = decode_experiment(document)

    assert document["schema"] == EXPERIMENT_SCHEMA
    assert restored == manifest
    assert restored.digest() == manifest.digest() == document["digest"]
    assert manifest.digest().startswith("sha256:")


def test_every_binding_of_the_manifest_changes_its_digest(reference) -> None:  # type: ignore[no-untyped-def]
    base = backend_experiment(reference)
    variants = {
        "selection": backend_experiment(
            reference, split="tuning", purpose=ExperimentPurpose.TUNING
        ),
        "base-config": replace(base, base_configuration_digest=content_hash("other")),
        "controls": replace(base, fixed_controls=(FixedControl(name="seed", value="1"),)),
        "metrics": backend_experiment(reference, quality=("semantic.acceptable_claim_rate",)),
        "resource": backend_experiment(reference, resource=("runtime.wall_time",)),
        "repetitions": replace(base, repetitions_per_sample=3),
        "version": replace(base, version="1.0.1"),
        "topology": replace(
            base,
            arms=tuple(
                modify_stage(
                    arm, "region_discovery", implementation=stage("x", "y", "sam2").implementation
                )
                for arm in base.arms
            ),
        ),
    }

    digests = {base.digest(), *(item.digest() for item in variants.values())}

    assert len(digests) == len(variants) + 1


def test_ablation_cells_are_deterministic_and_start_at_the_baseline() -> None:
    one_at_a_time = ablation_cells((FUSION_POLICY, FUSION_CHANNELS), AblationMode.ONE_AT_A_TIME)
    factorial = ablation_cells((FUSION_POLICY, FUSION_CHANNELS), AblationMode.FULL_FACTORIAL)

    baseline = (("fusion_policy", "uniform"), ("fusion_channels", "all"))
    assert one_at_a_time[0] == factorial[0] == baseline
    assert one_at_a_time == (
        baseline,
        (("fusion_policy", "quality_aware"), ("fusion_channels", "all")),
        (("fusion_policy", "uniform"), ("fusion_channels", "claims_only")),
    )
    assert len(factorial) == 4
    assert len(set(factorial)) == 4
    assert (
        ablation_cells((FUSION_POLICY, FUSION_CHANNELS), AblationMode.FULL_FACTORIAL) == factorial
    )


def test_the_arms_must_be_exactly_the_cells_of_the_declared_matrix(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = factorial_experiment(reference)
    arms = manifest.arms

    with pytest.raises(ExperimentError, match="ablation matrix"):
        replace(manifest, arms=arms[:-1])
    with pytest.raises(ExperimentError, match="ablation matrix"):
        replace(manifest, mode=AblationMode.ONE_AT_A_TIME)
    with pytest.raises(ValueError, match="unique"):
        replace(manifest, arms=(*arms, arms[1]))


def test_the_baseline_arm_holds_every_baseline_value(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)

    with pytest.raises(ExperimentError, match="baseline arm"):
        replace(manifest, baseline_arm_id="arm-1")
    with pytest.raises(ExperimentError, match="baseline arm"):
        replace(manifest, baseline_arm_id="missing")


def test_variables_need_two_values_and_a_declared_baseline() -> None:
    with pytest.raises(ValueError, match="at least two"):
        replace(SEMANTIC_BACKEND, values=("qwen",))
    with pytest.raises(ValueError, match="baseline"):
        replace(SEMANTIC_BACKEND, baseline_value="other")
    with pytest.raises(ValueError, match="unique"):
        replace(SEMANTIC_BACKEND, values=("qwen", "qwen"))
    assert {kind.value for kind in VariationKind} == {
        "backend",
        "policy",
        "configuration",
        "evidence_channels",
        "topology",
    }


# ----------------------------------------------------------- only declared variables


def test_a_backend_change_in_an_untouched_stage_is_refused(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    baseline, other = manifest.arms
    drifted = modify_stage(
        other, "region_discovery", implementation=stage("x", "y", "sam2").implementation
    )

    with pytest.raises(ExperimentError, match="undeclared change"):
        replace(manifest, arms=(baseline, drifted))


def test_a_configuration_change_in_an_untouched_stage_is_refused(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    baseline, other = manifest.arms
    region = next(item for item in other.topology.stages if item.stage_id == "region_discovery")
    tuned = replace(region.implementation, configuration_digest=content_hash("tuned-threshold"))

    with pytest.raises(ExperimentError, match="undeclared change"):
        replace(
            manifest, arms=(baseline, modify_stage(other, "region_discovery", implementation=tuned))
        )


def test_an_inserted_stage_or_rewired_dependency_needs_a_topology_variable(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    baseline, other = manifest.arms
    inserted = replace(
        other,
        topology=ResolvedTopology(
            stages=(*other.topology.stages, stage("extra", "x", "extra", depends_on=("ingestion",)))
        ),
    )
    rewired = modify_stage(other, "semantic_interpretation", depends_on=("ingestion",))

    with pytest.raises(ExperimentError, match="undeclared change"):
        replace(manifest, arms=(baseline, inserted))
    with pytest.raises(ExperimentError, match="undeclared change"):
        replace(manifest, arms=(baseline, rewired))


def test_a_variable_may_only_change_what_its_kind_allows(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    baseline, other = manifest.arms
    rewired = replace(
        other,
        topology=modify_stage(
            baseline, "semantic_interpretation", depends_on=("ingestion",)
        ).topology,
    )
    as_topology = replace(SEMANTIC_BACKEND, kind=VariationKind.TOPOLOGY)
    as_policy = replace(SEMANTIC_BACKEND, kind=VariationKind.POLICY)

    assert replace(manifest, variables=(as_topology,), arms=(baseline, rewired)).arms[1] == rewired
    with pytest.raises(ExperimentError, match="undeclared change"):
        replace(manifest, variables=(as_policy,))
    with pytest.raises(ExperimentError, match="undeclared change"):
        replace(manifest, variables=(as_topology,))


def test_a_variable_that_changes_nothing_is_refused(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    baseline, other = manifest.arms
    identical = replace(other, topology=baseline.topology)

    with pytest.raises(ExperimentError, match="changes none"):
        replace(manifest, arms=(baseline, identical))


def test_a_touched_stage_must_exist_in_some_arm(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    ghost = replace(SEMANTIC_BACKEND, touches=("no-such-stage",))

    with pytest.raises(ExperimentError, match="unknown stage"):
        replace(manifest, variables=(ghost,))


def test_every_arm_assigns_every_variable_an_allowed_value(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    baseline, other = manifest.arms

    with pytest.raises(ExperimentError, match="allows only"):
        replace(
            manifest, arms=(baseline, replace(other, assignments=(("semantic_backend", "llama"),)))
        )
    with pytest.raises(ExperimentError, match="does not assign"):
        replace(manifest, arms=(baseline, replace(other, assignments=())))


def test_fixed_controls_are_unique_and_never_variables(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)

    with pytest.raises(ValueError, match="control"):
        replace(
            manifest,
            fixed_controls=(
                FixedControl(name="seed", value="0"),
                FixedControl(name="seed", value="1"),
            ),
        )
    with pytest.raises(ExperimentError, match="control"):
        replace(manifest, fixed_controls=(FixedControl(name="semantic_backend", value="qwen"),))


# ------------------------------------------------------------------ DAG-stage ablation


def test_a_dag_insertion_shares_the_exact_upstream_artifact(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = topology_experiment(reference)
    native, enhanced = manifest.arms

    native_dino = next(item for item in native.topology.stages if item.stage_id == "dino")
    enhanced_dino = next(item for item in enhanced.topology.stages if item.stage_id == "dino")

    assert native_dino.artifact == enhanced_dino.artifact == pinned("perception_run", "dino-run-X")
    assert "feature_resolution_enhancement" not in {
        item.stage_id for item in native.topology.stages
    }
    assert next(
        item for item in enhanced.topology.stages if item.stage_id == "sensor_association"
    ).depends_on == ("feature_resolution_enhancement",)
    assert native.topology.digest() != enhanced.topology.digest()


def test_an_arm_cannot_swap_the_pinned_artifact_of_an_untouched_stage(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = topology_experiment(reference)
    native, enhanced = manifest.arms

    other_artifact = modify_stage(enhanced, "dino", artifact=pinned("perception_run", "dino-run-Y"))
    unpinned = modify_stage(enhanced, "dino", artifact=None)

    with pytest.raises(ExperimentError, match="undeclared change"):
        replace(manifest, arms=(native, other_artifact))
    with pytest.raises(ExperimentError, match="pinned immutable artifact"):
        replace(manifest, arms=(modify_stage(native, "dino", artifact=None), unpinned))


def test_the_upstream_of_a_touched_stage_must_be_pinned(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    unpinned = tuple(modify_stage(arm, "region_discovery", artifact=None) for arm in manifest.arms)

    with pytest.raises(ExperimentError, match="pinned immutable artifact"):
        replace(manifest, arms=unpinned)


def test_a_pinned_artifact_needs_a_digest() -> None:
    with pytest.raises(ValueError, match="digest"):
        stage("x", "y", "z", artifact=ArtifactIdentity(kind="run", artifact_id="r", digest=None))


def test_topologies_reject_missing_self_and_cyclic_dependencies() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ResolvedTopology(stages=(stage("a", "c", "b", depends_on=("missing",)),))
    with pytest.raises(ValueError, match="itself"):
        ResolvedTopology(stages=(stage("a", "c", "b", depends_on=("a",)),))
    with pytest.raises(ValueError, match="cycle"):
        ResolvedTopology(
            stages=(
                stage("a", "c", "b", depends_on=("b",)),
                stage("b", "c", "b", depends_on=("a",)),
            )
        )
    with pytest.raises(ValueError, match="unique"):
        ResolvedTopology(stages=(stage("a", "c", "b"), stage("a", "c", "b")))


def test_a_full_factorial_may_change_one_stage_through_two_variables(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = factorial_experiment(reference)

    assert len(manifest.arms) == 4
    digests = {arm.topology.digest() for arm in manifest.arms}
    assert len(digests) == 4


# ------------------------------------------------------------ reference set and registry


def test_the_selection_reproduces_the_exact_samples_of_the_reference_set(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)

    validate_experiment_manifest(manifest, reference_set=reference, registry=REGISTRY)

    expected = [item.sample_id for item in reference.selection("regions-by-sequence", "test")]
    assert list(manifest.selection.sample_ids) == expected
    assert manifest.selection.reference_set == reference.identity()


def test_a_selection_of_another_reference_set_version_is_refused(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    other = make_valid_manifest(version="2.0.0", annotations=reference.annotations)

    with pytest.raises(ExperimentError, match="reference set"):
        validate_experiment_manifest(manifest, reference_set=other, registry=REGISTRY)


def test_a_selection_that_does_not_match_its_split_is_refused(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    reordered = replace(
        manifest,
        selection=replace(
            manifest.selection, sample_ids=tuple(reversed(manifest.selection.sample_ids))
        ),
    )
    other_split = replace(manifest, selection=replace(manifest.selection, split="missing"))

    with pytest.raises(ExperimentError, match="selection"):
        validate_experiment_manifest(reordered, reference_set=reference, registry=REGISTRY)
    with pytest.raises((ExperimentError, ReferenceSetError), match="split"):
        validate_experiment_manifest(other_split, reference_set=reference, registry=REGISTRY)


def test_tuning_on_the_held_out_split_is_refused(reference) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ExperimentError, match="held-out"):
        validate_experiment_manifest(
            backend_experiment(reference, purpose=ExperimentPurpose.TUNING, split="test"),
            reference_set=reference,
            registry=REGISTRY,
        )
    validate_experiment_manifest(
        backend_experiment(reference, purpose=ExperimentPurpose.TUNING, split="tuning"),
        reference_set=reference,
        registry=REGISTRY,
    )


def test_metrics_must_exist_belong_to_the_stage_and_have_their_annotations(reference) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(Exception, match="unknown"):
        validate_experiment_manifest(
            backend_experiment(reference, quality=("semantic.overall.score",)),
            reference_set=reference,
            registry=REGISTRY,
        )
    with pytest.raises(ExperimentError, match="stage"):
        validate_experiment_manifest(
            backend_experiment(reference, quality=("region.iou.mean",)),
            reference_set=reference,
            registry=REGISTRY,
        )
    with pytest.raises(ExperimentError, match="quality"):
        validate_experiment_manifest(
            backend_experiment(reference, quality=("runtime.wall_time",)),
            reference_set=reference,
            registry=REGISTRY,
        )
    with pytest.raises(ExperimentError, match="performance"):
        validate_experiment_manifest(
            backend_experiment(reference, resource=("semantic.acceptable_claim_rate",)),
            reference_set=reference,
            registry=REGISTRY,
        )
    without_semantics = make_valid_manifest()
    manifest = backend_experiment(reference)
    with pytest.raises(Exception, match="semantics"):
        validate_experiment_manifest(
            replace(
                manifest,
                selection=SelectionBinding.for_split(
                    without_semantics, "regions-by-sequence", "test"
                ),
            ),
            reference_set=without_semantics,
            registry=REGISTRY,
        )


def test_the_manifest_must_cite_the_registry_it_is_validated_against(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)
    other_registry = replace(REGISTRY, registry_version="2")

    with pytest.raises(ExperimentError, match="registry"):
        validate_experiment_manifest(manifest, reference_set=reference, registry=other_registry)


def test_repeated_inference_is_not_a_new_physical_sample(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = replace(backend_experiment(reference), repetitions_per_sample=5)

    assert manifest.physical_sample_count == len(manifest.selection.sample_ids)
    with pytest.raises(ValueError, match="repetitions"):
        replace(manifest, repetitions_per_sample=0)


def test_metric_and_resource_policies_are_declared(reference) -> None:  # type: ignore[no-untyped-def]
    manifest = backend_experiment(reference)

    assert MetricRef(name="runtime.wall_time", version="1") in manifest.resource_capture
    assert manifest.evaluated_stage is EvaluationStage.SEMANTIC_INTERPRETATION
    with pytest.raises(ExperimentError, match="quality metric"):
        replace(manifest, quality_metrics=())
    with pytest.raises(ValueError, match="unique"):
        replace(manifest, quality_metrics=(manifest.quality_metrics[0],) * 2)


# --------------------------------------------------------------------------- files


def test_manifests_are_written_immutably_and_verified_on_read(reference, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    manifest = topology_experiment(reference)

    write_experiment(tmp_path, manifest)

    assert read_experiment(tmp_path) == manifest
    with pytest.raises(FileExistsError):
        write_experiment(tmp_path, manifest)
    path = tmp_path / "experiment.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["repetitions_per_sample"] = 9
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ExperimentError, match="digest"):
        read_experiment(tmp_path)


def test_unknown_schemas_and_missing_fields_are_refused(reference) -> None:  # type: ignore[no-untyped-def]
    document = encode_experiment(backend_experiment(reference))

    with pytest.raises(ExperimentError, match="schema"):
        decode_experiment({**document, "schema": "contextmap.experiment/v0"})
    broken = dict(document)
    del broken["arms"]
    with pytest.raises(ExperimentError, match="arms"):
        decode_experiment(broken)
