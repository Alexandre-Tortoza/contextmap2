"""Optional-technique protocol tests: identical upstream artifacts, declared variable only."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from experiment_builders import REGISTRY, pinned, stage
from technique_builders import (
    feature_resolution_protocol,
    feature_resolution_topologies,
    quality_fusion_protocol,
    quality_fusion_topologies,
    technique_reference,
)

from contextmap.evaluation.experiments import (
    ExperimentError,
    ResolvedTopology,
    VariationKind,
    validate_experiment_manifest,
)
from contextmap.evaluation.metrics import EvaluationStage, MetricKind, MetricRegistryError
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet
from contextmap.evaluation.technique_protocols import (
    BASELINE_ARM_ID,
    FEATURE_EXTRACTION_STAGE,
    FEATURE_RESOLUTION_STRATA,
    OBSERVATION_QUALITY_STRATA,
    RESOLUTION_ENHANCEMENT_STAGE,
    RESOURCE_METRICS,
    SEMANTIC_FUSION_STAGE,
    SENSOR_ASSOCIATION_STAGE,
    TECHNIQUE_FEATURE_RESOLUTION,
    TECHNIQUE_QUALITY_AWARE_FUSION,
    VARIANT_ARM_ID,
    TechniqueProtocolError,
    protocol_metric_names,
)
from contextmap.sensor_association import QualityComponent


@pytest.fixture
def validated(tmp_path: Path) -> ValidatedReferenceSet:
    return technique_reference(tmp_path / "reference")


def _rewrite(topology: ResolvedTopology, stage_id: str, **changes: object) -> ResolvedTopology:
    return ResolvedTopology(
        stages=tuple(
            replace(item, **changes) if item.stage_id == stage_id else item  # type: ignore[arg-type]
            for item in topology.stages
        )
    )


# ------------------------------------------------------------ experiment A: resolution


def test_feature_resolution_is_one_controlled_experiment_per_evaluated_stage(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)

    assert protocol.technique_id == TECHNIQUE_FEATURE_RESOLUTION
    assert protocol.stages == (
        EvaluationStage.SENSOR_ASSOCIATION,
        EvaluationStage.SEMANTIC_FUSION,
        EvaluationStage.ENTITY_RESOLUTION,
        EvaluationStage.SPATIAL_RELATIONS,
    )
    for experiment in protocol.experiments:
        assert [arm.arm_id for arm in experiment.arms] == [BASELINE_ARM_ID, VARIANT_ARM_ID]
        assert experiment.baseline_arm_id == BASELINE_ARM_ID
        assert experiment.variables[0].kind is VariationKind.TOPOLOGY
        assert experiment.version == "1.0.0"
        validate_experiment_manifest(
            experiment, reference_set=validated.manifest, registry=REGISTRY
        )


def test_native_and_enhanced_arms_share_the_exact_upstream_dino_artifact(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)

    for experiment in protocol.experiments:
        native = experiment.arm(BASELINE_ARM_ID).topology
        enhanced = experiment.arm(VARIANT_ARM_ID).topology
        expected = pinned("perception_run", "dino-run-X")
        assert native.stage(FEATURE_EXTRACTION_STAGE).artifact == expected
        assert enhanced.stage(FEATURE_EXTRACTION_STAGE).artifact == expected
        assert enhanced.stage(RESOLUTION_ENHANCEMENT_STAGE).depends_on == (
            FEATURE_EXTRACTION_STAGE,
        )
        assert enhanced.stage(SENSOR_ASSOCIATION_STAGE).depends_on[0] == (
            RESOLUTION_ENHANCEMENT_STAGE
        )
        assert RESOLUTION_ENHANCEMENT_STAGE not in {item.stage_id for item in native.stages}


def test_all_experiments_of_a_protocol_compare_the_same_topologies_and_samples(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)

    digests = {tuple(arm.topology.digest() for arm in item.arms) for item in protocol.experiments}
    selections = {item.selection for item in protocol.experiments}
    assert len(digests) == 1
    assert len(selections) == 1


def test_metrics_are_reported_per_stage_and_costs_apart_from_quality(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)

    names = protocol_metric_names(protocol)

    assert names[EvaluationStage.SENSOR_ASSOCIATION] == (
        "association.reprojection_error.median",
        "association.visible_support.ratio",
        "association.feature_anchoring.rate",
    )
    assert "fusion.reference_recovery.rate" in names[EvaluationStage.SEMANTIC_FUSION]
    assert "fusion.ambiguity_retention.rate" in names[EvaluationStage.SEMANTIC_FUSION]
    assert "entity.semantic_accuracy.rate" in names[EvaluationStage.ENTITY_RESOLUTION]
    assert "relations.f1" in names[EvaluationStage.SPATIAL_RELATIONS]
    for experiment in protocol.experiments:
        assert tuple(item.name for item in experiment.resource_capture) == RESOURCE_METRICS
        for reference in experiment.quality_metrics:
            assert REGISTRY.get(reference.name, reference.version).kind is MetricKind.QUALITY
        for reference in experiment.resource_capture:
            assert REGISTRY.get(reference.name, reference.version).kind is MetricKind.PERFORMANCE


def test_the_protocol_names_the_strata_it_reports_by(validated: ValidatedReferenceSet) -> None:
    protocol = feature_resolution_protocol(validated.manifest)

    assert protocol.stratification_factors == FEATURE_RESOLUTION_STRATA
    assert set(FEATURE_RESOLUTION_STRATA) == {
        "range_band",
        "visibility",
        "support_density",
        "valid_region",
        "projected_size",
    }
    assert protocol.undeclared_factors(validated.manifest) == ()


def test_factors_the_reference_set_does_not_declare_are_reported_as_undeclared(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)
    sparse = replace(
        validated.manifest,
        stratum_definitions=tuple(
            item for item in validated.manifest.stratum_definitions if item.name == "visibility"
        ),
    )

    assert protocol.undeclared_factors(sparse) == (
        "range_band",
        "support_density",
        "valid_region",
        "projected_size",
    )


def test_a_change_beyond_the_enhancement_stage_is_refused(validated: ValidatedReferenceSet) -> None:
    _, enhanced = feature_resolution_topologies()
    other_backend = _rewrite(
        enhanced,
        SENSOR_ASSOCIATION_STAGE,
        implementation=stage("x", "y", "other-association").implementation,
    )

    with pytest.raises(ExperimentError, match="undeclared change"):
        feature_resolution_protocol(validated.manifest, enhanced=other_backend)


def test_a_different_dino_artifact_in_the_enhanced_arm_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    _, other = feature_resolution_topologies(dino_artifact="dino-run-Y")

    with pytest.raises(ExperimentError, match="undeclared change"):
        feature_resolution_protocol(validated.manifest, enhanced=other)


def test_an_unpinned_upstream_is_refused(validated: ValidatedReferenceSet) -> None:
    baseline, enhanced = feature_resolution_topologies()

    with pytest.raises(ExperimentError, match="pinned immutable artifact"):
        feature_resolution_protocol(
            validated.manifest,
            baseline=_rewrite(baseline, "geometric_mapping", artifact=None),
            enhanced=_rewrite(enhanced, "geometric_mapping", artifact=None),
        )


def _baseline_with_the_enhancement_stage() -> dict[str, ResolvedTopology]:
    return {"baseline": feature_resolution_topologies()[1]}


def _enhanced_without_the_enhancement_stage() -> dict[str, ResolvedTopology]:
    baseline, _ = feature_resolution_topologies()
    return {"enhanced": baseline}


def _enhancement_not_consuming_the_extraction() -> dict[str, ResolvedTopology]:
    _, enhanced = feature_resolution_topologies()
    return {"enhanced": _rewrite(enhanced, RESOLUTION_ENHANCEMENT_STAGE, depends_on=("ingestion",))}


@pytest.mark.parametrize(
    ("build", "problem"),
    [
        (_baseline_with_the_enhancement_stage, "must not contain"),
        (_enhanced_without_the_enhancement_stage, "no stage"),
        (_enhancement_not_consuming_the_extraction, "must consume"),
    ],
)
def test_topologies_that_are_not_the_protocol_are_refused(
    validated: ValidatedReferenceSet,
    build: Callable[[], dict[str, ResolvedTopology]],
    problem: str,
) -> None:
    with pytest.raises(TechniqueProtocolError, match=problem):
        feature_resolution_protocol(validated.manifest, **build())


def test_a_registry_without_the_protocol_metrics_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    incomplete = replace(
        REGISTRY,
        definitions=tuple(
            item
            for item in REGISTRY.definitions
            if item.name != "association.feature_anchoring.rate"
        ),
    )

    with pytest.raises(MetricRegistryError, match="unknown"):
        feature_resolution_protocol(validated.manifest, registry=incomplete)


def test_the_protocol_digest_binds_every_experiment(validated: ValidatedReferenceSet) -> None:
    first = feature_resolution_protocol(validated.manifest)
    same = feature_resolution_protocol(validated.manifest)
    other = feature_resolution_protocol(validated.manifest, protocol_version="1.0.1")

    assert first.digest() == same.digest()
    assert first.digest() != other.digest()
    assert first.to_record()["experiments"][0]["digest"] == first.experiments[0].digest()


def test_a_protocol_needs_experiments_over_distinct_stages(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)

    with pytest.raises(TechniqueProtocolError, match="one experiment per"):
        replace(protocol, experiments=(protocol.experiments[0], protocol.experiments[0]))
    with pytest.raises(TechniqueProtocolError, match="at least one"):
        replace(protocol, experiments=())
    with pytest.raises(TechniqueProtocolError, match="does not evaluate"):
        replace(protocol, experiments=protocol.experiments[:1]).experiment_for(
            EvaluationStage.SPATIAL_RELATIONS
        )


# ----------------------------------------------------------- experiment B: fusion policy


def test_quality_aware_fusion_shares_the_exact_association_artifact(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = quality_fusion_protocol(validated.manifest)

    assert protocol.technique_id == TECHNIQUE_QUALITY_AWARE_FUSION
    assert protocol.stages == (
        EvaluationStage.SEMANTIC_FUSION,
        EvaluationStage.ENTITY_RESOLUTION,
        EvaluationStage.SPATIAL_RELATIONS,
    )
    for experiment in protocol.experiments:
        uniform = experiment.arm(BASELINE_ARM_ID).topology
        aware = experiment.arm(VARIANT_ARM_ID).topology
        expected = pinned("association_run", "association-run-0001")
        assert uniform.stage(SENSOR_ASSOCIATION_STAGE).artifact == expected
        assert aware.stage(SENSOR_ASSOCIATION_STAGE).artifact == expected
        assert experiment.variables[0].kind is VariationKind.POLICY
        assert (
            uniform.stage(SEMANTIC_FUSION_STAGE).implementation.configuration_digest
            != aware.stage(SEMANTIC_FUSION_STAGE).implementation.configuration_digest
        )
        validate_experiment_manifest(
            experiment, reference_set=validated.manifest, registry=REGISTRY
        )


def test_the_quality_strata_cover_every_observation_quality_component(
    validated: ValidatedReferenceSet,
) -> None:
    protocol = quality_fusion_protocol(validated.manifest)

    assert protocol.stratification_factors == OBSERVATION_QUALITY_STRATA
    for component in QualityComponent:
        assert f"quality.{component.value}" in protocol.stratification_factors
    assert {"ambiguity_level", "conflict_level"} <= set(protocol.stratification_factors)
    assert set(protocol.undeclared_factors(validated.manifest)) == (
        set(OBSERVATION_QUALITY_STRATA) - {"ambiguity_level"}
    )


def test_a_fusion_arm_that_changes_more_than_the_policy_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    _, aware = quality_fusion_topologies()
    changed = _rewrite(
        aware, "entity_resolution", implementation=stage("x", "y", "other-resolver").implementation
    )

    with pytest.raises(ExperimentError, match="undeclared change"):
        quality_fusion_protocol(validated.manifest, quality_aware=changed)


def test_a_fusion_arm_with_another_association_artifact_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    _, aware = quality_fusion_topologies()
    other = _rewrite(
        aware, SENSOR_ASSOCIATION_STAGE, artifact=pinned("association_run", "association-run-0002")
    )

    with pytest.raises(ExperimentError, match="undeclared change"):
        quality_fusion_protocol(validated.manifest, quality_aware=other)


def test_a_fusion_topology_without_the_fusion_stage_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    _, aware = quality_fusion_topologies()
    without = ResolvedTopology(
        stages=tuple(item for item in aware.stages if item.stage_id == SENSOR_ASSOCIATION_STAGE)
    )

    with pytest.raises(TechniqueProtocolError, match="no stage"):
        quality_fusion_protocol(validated.manifest, quality_aware=without)
