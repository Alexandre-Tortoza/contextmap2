"""Builders for optional-technique protocols: topologies, strata and stratified executors."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiment_builders import REGISTRY, pinned, stage, stage_artifacts_for
from qa_builders import clean_sets, write_qa_reference
from reference_set_builders import content_hash

from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.experiment_runner import (
    ArmExecution,
    ArmExecutor,
    ComparisonManifest,
    run_experiment,
)
from contextmap.evaluation.experiments import (
    ExperimentArm,
    ExperimentManifest,
    ResolvedTopology,
    SelectionBinding,
    experiment_artifact_identity,
)
from contextmap.evaluation.metrics import EvaluationStage
from contextmap.evaluation.reference_integrity import (
    ValidatedReferenceSet,
    require_valid_reference_set,
)
from contextmap.evaluation.reference_set import ReferenceSetManifest, StratumDefinition
from contextmap.evaluation.report_schema import (
    EvaluatorIdentity,
    MetricResult,
    MetricStatus,
    ReproducibilityMetadata,
    assemble_evaluation_report,
)
from contextmap.evaluation.technique_protocols import (
    FEATURE_EXTRACTION_STAGE,
    RESOLUTION_ENHANCEMENT_STAGE,
    SEMANTIC_FUSION_STAGE,
    SENSOR_ASSOCIATION_STAGE,
    TechniqueProtocol,
    build_feature_resolution_protocol,
    build_quality_aware_fusion_protocol,
)

StrataKey = tuple[tuple[str, str], ...]
Values = Mapping[str, Mapping[str, Mapping[StrataKey, tuple[float | None, int]]]]
Costs = Mapping[str, Mapping[str, Mapping[StrataKey, float]]]

ALL_SCHEMAS = tuple(family.schema for family in AnnotationFamily)

STRATA = (
    StratumDefinition(
        name="visibility", values=("clear", "occluded"), description="occlusion of the subject"
    ),
    StratumDefinition(
        name="range_band", values=("near", "far"), description="depth of the support"
    ),
    StratumDefinition(
        name="support_density", values=("sparse", "dense"), description="support points per pixel"
    ),
    StratumDefinition(
        name="valid_region", values=("inside", "border"), description="image valid region"
    ),
    StratumDefinition(
        name="projected_size", values=("small", "large"), description="projected support size"
    ),
    StratumDefinition(
        name="ambiguity_level", values=("low", "high"), description="ambiguity of the evidence"
    ),
)


def technique_reference(
    root: Path, strata: tuple[StratumDefinition, ...] = STRATA
) -> ValidatedReferenceSet:
    """Write a reference set with every annotation family and the given strata."""
    manifest = write_qa_reference(root, clean_sets(), stratum_definitions=strata)
    return require_valid_reference_set(manifest, root)


# ------------------------------------------------------------------------- topologies


def _downstream(association_stage: str, fusion_config: str = "uniform") -> list[Any]:
    return [
        stage(
            SEMANTIC_FUSION_STAGE,
            "semantic_fusion",
            "fusion",
            depends_on=(association_stage,),
            config=fusion_config,
        ),
        stage(
            "entity_resolution",
            "entity_resolution",
            "resolver",
            depends_on=(SEMANTIC_FUSION_STAGE,),
        ),
        stage(
            "spatial_relations",
            "spatial_relations",
            "relations",
            depends_on=("entity_resolution",),
        ),
    ]


def feature_resolution_topologies(
    *, dino_artifact: str = "dino-run-X"
) -> tuple[ResolvedTopology, ResolvedTopology]:
    """Return ``(native, enhanced)``: the enhanced path reuses the very same DINO artifact."""
    ingestion = stage(
        "ingestion", "ingestion", "ros-adapter", artifact=pinned("sequence", "sequence-0001")
    )
    geometry = stage(
        "geometric_mapping",
        "geometric_mapping",
        "mapper",
        depends_on=("ingestion",),
        artifact=pinned("geometric_map", "map-0001"),
    )
    dino = stage(
        FEATURE_EXTRACTION_STAGE,
        "visual_perception",
        "dinov3",
        depends_on=("ingestion",),
        artifact=pinned("perception_run", dino_artifact),
    )
    native = ResolvedTopology(
        stages=(
            ingestion,
            geometry,
            dino,
            stage(
                SENSOR_ASSOCIATION_STAGE,
                "sensor_association",
                "association",
                depends_on=(FEATURE_EXTRACTION_STAGE, "geometric_mapping"),
            ),
            *_downstream(SENSOR_ASSOCIATION_STAGE),
        )
    )
    enhanced = ResolvedTopology(
        stages=(
            ingestion,
            geometry,
            dino,
            stage(
                RESOLUTION_ENHANCEMENT_STAGE,
                "visual_perception",
                "resolution-enhancer",
                depends_on=(FEATURE_EXTRACTION_STAGE,),
            ),
            stage(
                SENSOR_ASSOCIATION_STAGE,
                "sensor_association",
                "association",
                depends_on=(RESOLUTION_ENHANCEMENT_STAGE, "geometric_mapping"),
            ),
            *_downstream(SENSOR_ASSOCIATION_STAGE),
        )
    )
    return native, enhanced


def quality_fusion_topologies() -> tuple[ResolvedTopology, ResolvedTopology]:
    """Return ``(uniform, quality_aware)`` over one pinned association artifact."""

    def topology(config: str) -> ResolvedTopology:
        return ResolvedTopology(
            stages=(
                stage(
                    SENSOR_ASSOCIATION_STAGE,
                    "sensor_association",
                    "association",
                    artifact=pinned("association_run", "association-run-0001"),
                ),
                *_downstream(SENSOR_ASSOCIATION_STAGE, fusion_config=config),
            )
        )

    return topology("uniform"), topology("quality-aware")


# --------------------------------------------------------------------------- protocols


def selection_for(reference: ReferenceSetManifest) -> SelectionBinding:
    return SelectionBinding.for_split(reference, "regions-by-sequence", "test")


def feature_resolution_protocol(
    reference: ReferenceSetManifest, **overrides: Any
) -> TechniqueProtocol:
    baseline, enhanced = feature_resolution_topologies()
    fields: dict[str, Any] = {
        "protocol_version": "1.0.0",
        "selection": selection_for(reference),
        "base_configuration_digest": content_hash("base-configuration"),
        "baseline": baseline,
        "enhanced": enhanced,
        "registry": REGISTRY,
    }
    fields.update(overrides)
    return build_feature_resolution_protocol(**fields)


def quality_fusion_protocol(reference: ReferenceSetManifest, **overrides: Any) -> TechniqueProtocol:
    uniform, quality_aware = quality_fusion_topologies()
    fields: dict[str, Any] = {
        "protocol_version": "1.0.0",
        "selection": selection_for(reference),
        "base_configuration_digest": content_hash("base-configuration"),
        "uniform": uniform,
        "quality_aware": quality_aware,
        "registry": REGISTRY,
    }
    fields.update(overrides)
    return build_quality_aware_fusion_protocol(**fields)


# ---------------------------------------------------------------------------- executor


def _quality_results(
    manifest: ExperimentManifest, arm_id: str, values: Values
) -> tuple[MetricResult, ...]:
    count = len(manifest.selection.sample_ids)
    results: list[MetricResult] = []
    for reference in manifest.quality_metrics:
        per_strata = values.get(arm_id, {}).get(reference.name, {(): (0.5, count)})
        for strata, (value, samples) in per_strata.items():
            results.append(
                MetricResult(
                    metric=reference.name,
                    metric_version=reference.version,
                    status=MetricStatus.VALUE if value is not None else MetricStatus.NOT_APPLICABLE,
                    value=value,
                    sample_count=samples,
                    strata=strata,
                )
            )
    return tuple(results)


def _cost_results(
    manifest: ExperimentManifest, arm_id: str, costs: Costs
) -> tuple[MetricResult, ...]:
    count = len(manifest.selection.sample_ids)
    results: list[MetricResult] = []
    for reference in manifest.resource_capture:
        per_strata = costs.get(arm_id, {}).get(reference.name, {(): 1.0})
        for strata, value in per_strata.items():
            results.append(
                MetricResult(
                    metric=reference.name,
                    metric_version=reference.version,
                    status=MetricStatus.VALUE,
                    value=value,
                    sample_count=count,
                    strata=strata,
                )
            )
    return tuple(results)


def technique_executor(
    values: Values,
    *,
    costs: Costs | None = None,
    failures: Mapping[str, BaseException] | None = None,
) -> ArmExecutor:
    """Return an executor that reports canned, stratified values for every experiment."""

    def execute(manifest: ExperimentManifest, arm: ExperimentArm) -> ArmExecution:
        if failures and arm.arm_id in failures:
            raise failures[arm.arm_id]
        report = assemble_evaluation_report(
            REGISTRY,
            stage=manifest.evaluated_stage,
            reproducibility=ReproducibilityMetadata(
                evaluator=EvaluatorIdentity(evaluator_id="fake-evaluator", evaluator_version="1"),
                reference_set=manifest.selection.reference_set,
                annotation_schemas=ALL_SCHEMAS,
                input_artifacts=(experiment_artifact_identity(manifest),),
                configuration_digest=arm.topology.digest(),
                code_version="0.0.1",
                metric_registry=REGISTRY.identity(),
            ),
            quality_metrics=_quality_results(manifest, arm.arm_id, values),
            performance_metrics=_cost_results(manifest, arm.arm_id, costs or {}),
            stage_report=None,
        )
        return ArmExecution(report=report, stage_artifacts=stage_artifacts_for(arm))

    return execute


def run_protocol(
    protocol: TechniqueProtocol,
    validated: ValidatedReferenceSet,
    root: Path,
    executor: ArmExecutor,
    *,
    only: tuple[EvaluationStage, ...] | None = None,
) -> dict[EvaluationStage, ComparisonManifest]:
    """Run every (or the chosen) experiment of a protocol and return the comparisons."""
    comparisons: dict[EvaluationStage, ComparisonManifest] = {}
    for experiment in protocol.experiments:
        if only is not None and experiment.evaluated_stage not in only:
            continue
        comparisons[experiment.evaluated_stage] = run_experiment(
            experiment,
            executor=executor,
            registry=REGISTRY,
            reference_set=validated,
            root=root / experiment.evaluated_stage.value,
        ).comparison
    return comparisons
