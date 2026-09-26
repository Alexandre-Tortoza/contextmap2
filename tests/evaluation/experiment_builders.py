"""Builders for experiment manifests, arms and fake arm executors in tests."""

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from reference_set_builders import content_hash, make_annotation, make_valid_manifest, sample_id

from contextmap.evaluation.annotations import (
    CASEFOLD_EXACT_POLICY,
    AnnotationFamily,
    LabelNormalization,
    SemanticAnnotationRecord,
    SemanticAnnotationSet,
    SemanticStatus,
    write_annotation_set,
)
from contextmap.evaluation.experiment_runner import ArmExecution, ArmExecutor, StageArtifact
from contextmap.evaluation.experiments import (
    AblationMode,
    ExperimentArm,
    ExperimentManifest,
    ExperimentPurpose,
    ExperimentVariable,
    FixedControl,
    MetricRef,
    ResolvedTopology,
    SelectionBinding,
    StageImplementation,
    TopologyStage,
    VariationKind,
    ablation_cells,
    experiment_artifact_identity,
)
from contextmap.evaluation.metrics import EvaluationStage, default_metric_registry
from contextmap.evaluation.reference_integrity import (
    ValidatedReferenceSet,
    require_valid_reference_set,
)
from contextmap.evaluation.reference_set import ReferenceSetManifest
from contextmap.evaluation.report_schema import (
    ArtifactIdentity,
    EvaluationReport,
    EvaluatorIdentity,
    MetricResult,
    MetricStatus,
    ReproducibilityMetadata,
    assemble_evaluation_report,
)
from contextmap.ingestion import SourceObservationId

REGISTRY = default_metric_registry()
SEMANTICS = AnnotationFamily.SEMANTICS.schema
REGIONS = AnnotationFamily.REGIONS.schema
SCHEME = "regions-by-sequence"


# ------------------------------------------------------------------- reference sets


def _semantics_for(*indices: int) -> SemanticAnnotationSet:
    return SemanticAnnotationSet(
        normalization=LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY),
        records=tuple(
            SemanticAnnotationRecord(
                sample_id=sample_id(index),
                observation_id=SourceObservationId(f"frame-{index:04d}"),
                region_id=f"object-{index}",
                status=SemanticStatus.LABELED,
                concepts=("pallet",),
            )
            for index in indices
        ),
    )


def reference_with_semantics() -> ReferenceSetManifest:
    """Return a structurally valid manifest that declares regions and semantics files.

    The files are not written, so it serves manifest-level checks only.
    """
    regions = make_annotation(samples=(0, 1, 2, 3))
    semantics = make_annotation(
        annotation_id="ann-semantics",
        path="annotations/semantics.json",
        schema=SEMANTICS,
        samples=(0, 1, 2, 3),
    )
    return make_valid_manifest(annotations=(regions, semantics))


def write_reference_with_semantics(root: Path) -> ReferenceSetManifest:
    """Write the annotation files under ``root`` and return the matching manifest."""
    from reference_set_builders import regions_for

    regions_hash = write_annotation_set(
        root / "annotations" / "regions.json", regions_for(0, 1, 2, 3)
    )
    semantics_hash = write_annotation_set(
        root / "annotations" / "semantics.json", _semantics_for(0, 1, 2, 3)
    )
    regions = make_annotation(samples=(0, 1, 2, 3), file_hash=regions_hash)
    semantics = make_annotation(
        annotation_id="ann-semantics",
        path="annotations/semantics.json",
        schema=SEMANTICS,
        file_hash=semantics_hash,
        samples=(0, 1, 2, 3),
    )
    return make_valid_manifest(annotations=(regions, semantics))


def validated_reference(root: Path) -> ValidatedReferenceSet:
    """Write and validate a reference set on disk."""
    manifest = write_reference_with_semantics(root)
    return require_valid_reference_set(manifest, root)


# ---------------------------------------------------------------------- topologies


def implementation(
    backend: str, *, model: str | None = None, config: str | None = None
) -> StageImplementation:
    return StageImplementation(
        backend_id=backend,
        backend_version="1",
        model=model,
        configuration_digest=content_hash(config or f"{backend}-config"),
    )


def pinned(kind: str, artifact_id: str) -> ArtifactIdentity:
    return ArtifactIdentity(kind=kind, artifact_id=artifact_id, digest=content_hash(artifact_id))


def stage(
    stage_id: str,
    capability: str,
    backend: str,
    *,
    depends_on: tuple[str, ...] = (),
    artifact: ArtifactIdentity | None = None,
    model: str | None = None,
    config: str | None = None,
) -> TopologyStage:
    return TopologyStage(
        stage_id=stage_id,
        capability=capability,
        implementation=implementation(backend, model=model, config=config),
        depends_on=depends_on,
        artifact=artifact,
    )


def modify_stage(arm: ExperimentArm, stage_id: str, **changes: object) -> ExperimentArm:
    """Return a copy of ``arm`` whose stage ``stage_id`` has the given fields replaced."""
    stages = tuple(
        replace(item, **changes) if item.stage_id == stage_id else item  # type: ignore[arg-type]
        for item in arm.topology.stages
    )
    return replace(arm, topology=ResolvedTopology(stages=stages))


def semantic_topology(backend: str, *, model: str) -> ResolvedTopology:
    return ResolvedTopology(
        stages=(
            stage(
                "ingestion",
                "ingestion",
                "ros-adapter",
                artifact=pinned("sequence", "sequence-0001"),
            ),
            stage(
                "region_discovery",
                "visual_perception",
                "sam3",
                depends_on=("ingestion",),
                artifact=pinned("perception_run", "perception-run-0001"),
            ),
            stage(
                "semantic_interpretation",
                "visual_perception",
                backend,
                depends_on=("region_discovery",),
                model=model,
            ),
        )
    )


def sensor_topology(*, enhanced: bool) -> ResolvedTopology:
    dino = stage(
        "dino",
        "visual_perception",
        "dinov3",
        depends_on=("ingestion",),
        artifact=pinned("perception_run", "dino-run-X"),
    )
    ingestion = stage(
        "ingestion", "ingestion", "ros-adapter", artifact=pinned("sequence", "sequence-0001")
    )
    if enhanced:
        enhancement = stage(
            "feature_resolution_enhancement",
            "visual_perception",
            "resolution-enhancer",
            depends_on=("dino",),
        )
        association = stage(
            "sensor_association",
            "sensor_association",
            "association",
            depends_on=("feature_resolution_enhancement",),
        )
        return ResolvedTopology(stages=(ingestion, dino, enhancement, association))
    association = stage(
        "sensor_association", "sensor_association", "association", depends_on=("dino",)
    )
    return ResolvedTopology(stages=(ingestion, dino, association))


def fusion_topology(*, quality_aware: bool, channels: str) -> ResolvedTopology:
    policy = "quality-aware" if quality_aware else "uniform"
    return ResolvedTopology(
        stages=(
            stage(
                "sensor_association",
                "sensor_association",
                "association",
                artifact=pinned("association_run", "association-run-0001"),
            ),
            stage(
                "semantic_fusion",
                "semantic_fusion",
                "fusion",
                depends_on=("sensor_association",),
                config=f"{policy}|{channels}",
            ),
        )
    )


# ---------------------------------------------------------------------- experiments


def selection_for(reference: ReferenceSetManifest, split: str = "test") -> SelectionBinding:
    return SelectionBinding.for_split(reference, SCHEME, split)


def arms_for(
    variables: Sequence[ExperimentVariable],
    mode: AblationMode,
    topology_for: Callable[[Mapping[str, str]], ResolvedTopology],
) -> tuple[ExperimentArm, ...]:
    """Build one arm per ablation cell, naming the baseline ``baseline``."""
    arms = []
    for index, cell in enumerate(ablation_cells(tuple(variables), mode)):
        assignments = dict(cell)
        is_baseline = all(assignments[item.name] == item.baseline_value for item in variables)
        arms.append(
            ExperimentArm(
                arm_id="baseline" if is_baseline else f"arm-{index}",
                assignments=cell,
                topology=topology_for(assignments),
            )
        )
    return tuple(arms)


def _experiment(
    reference: ReferenceSetManifest,
    *,
    experiment_id: str,
    stage_id: EvaluationStage,
    variables: tuple[ExperimentVariable, ...],
    arms: tuple[ExperimentArm, ...],
    mode: AblationMode,
    quality: tuple[str, ...],
    resource: tuple[str, ...] = ("runtime.wall_time", "runtime.peak_memory"),
    purpose: ExperimentPurpose = ExperimentPurpose.EVALUATION,
    split: str = "test",
) -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id=experiment_id,
        version="1.0.0",
        description="controlled comparison",
        purpose=purpose,
        evaluated_stage=stage_id,
        selection=selection_for(reference, split),
        repetitions_per_sample=1,
        base_configuration_digest=content_hash("base-configuration"),
        variables=variables,
        mode=mode,
        baseline_arm_id="baseline",
        arms=arms,
        fixed_controls=(
            FixedControl(name="prompt_template", value="region/v1"),
            FixedControl(name="seed", value="0"),
        ),
        quality_metrics=tuple(MetricRef(name=name, version="1") for name in quality),
        resource_capture=tuple(MetricRef(name=name, version="1") for name in resource),
        registry=REGISTRY.identity(),
    )


SEMANTIC_BACKEND = ExperimentVariable(
    name="semantic_backend",
    kind=VariationKind.BACKEND,
    touches=("semantic_interpretation",),
    values=("qwen", "gemini"),
    baseline_value="qwen",
    description="local Qwen versus the remote Gemini interpreter",
)
FEATURE_RESOLUTION = ExperimentVariable(
    name="feature_resolution",
    kind=VariationKind.TOPOLOGY,
    touches=("feature_resolution_enhancement", "sensor_association"),
    values=("native", "enhanced"),
    baseline_value="native",
    description="dense features as extracted versus through resolution enhancement",
)
FUSION_POLICY = ExperimentVariable(
    name="fusion_policy",
    kind=VariationKind.POLICY,
    touches=("semantic_fusion",),
    values=("uniform", "quality_aware"),
    baseline_value="uniform",
    description="uniform versus quality-aware evidence accumulation",
)
FUSION_CHANNELS = ExperimentVariable(
    name="fusion_channels",
    kind=VariationKind.EVIDENCE_CHANNELS,
    touches=("semantic_fusion",),
    values=("all", "claims_only"),
    baseline_value="all",
    description="all evidence channels versus the claims channel alone",
)


def backend_experiment(reference: ReferenceSetManifest, **overrides: object) -> ExperimentManifest:
    variables = (SEMANTIC_BACKEND,)
    models = {"qwen": "qwen-3b", "gemini": "gemini-pro"}
    arms = arms_for(
        variables,
        AblationMode.ONE_AT_A_TIME,
        lambda assignment: semantic_topology(
            assignment["semantic_backend"], model=models[assignment["semantic_backend"]]
        ),
    )
    fields: dict[str, object] = {
        "experiment_id": "semantic-backends",
        "stage_id": EvaluationStage.SEMANTIC_INTERPRETATION,
        "variables": variables,
        "arms": arms,
        "mode": AblationMode.ONE_AT_A_TIME,
        "quality": ("semantic.acceptable_claim_rate", "semantic.unsupported_claim_rate"),
    }
    fields.update(overrides)
    return _experiment(reference, **fields)  # type: ignore[arg-type]


def topology_experiment(reference: ReferenceSetManifest, **overrides: object) -> ExperimentManifest:
    variables = (FEATURE_RESOLUTION,)
    arms = arms_for(
        variables,
        AblationMode.ONE_AT_A_TIME,
        lambda assignment: sensor_topology(enhanced=assignment["feature_resolution"] == "enhanced"),
    )
    fields: dict[str, object] = {
        "experiment_id": "feature-resolution",
        "stage_id": EvaluationStage.SENSOR_ASSOCIATION,
        "variables": variables,
        "arms": arms,
        "mode": AblationMode.ONE_AT_A_TIME,
        "quality": ("association.visible_support.ratio",),
    }
    fields.update(overrides)
    return _experiment(reference, **fields)  # type: ignore[arg-type]


def factorial_experiment(
    reference: ReferenceSetManifest, **overrides: object
) -> ExperimentManifest:
    variables = (FUSION_POLICY, FUSION_CHANNELS)
    arms = arms_for(
        variables,
        AblationMode.FULL_FACTORIAL,
        lambda assignment: fusion_topology(
            quality_aware=assignment["fusion_policy"] == "quality_aware",
            channels=assignment["fusion_channels"],
        ),
    )
    fields: dict[str, object] = {
        "experiment_id": "fusion-ablation",
        "stage_id": EvaluationStage.SEMANTIC_FUSION,
        "variables": variables,
        "arms": arms,
        "mode": AblationMode.FULL_FACTORIAL,
        "quality": ("fusion.reference_recovery.rate",),
    }
    fields.update(overrides)
    return _experiment(reference, **fields)  # type: ignore[arg-type]


# --------------------------------------------------------- semantic request-policy ablations

QWEN_REVISION = "1" * 40
"""Immutable Hugging Face commit the Qwen arms are pinned to."""

REQUEST_POLICY: dict[str, Any] = {
    "prompt_policy": "region/v1",
    "view_policy": ["tight_crop"],
    "scene_context": "none",
    "output_schema": "semantic-response/1",
}
"""Backend-neutral request policy of the semantic stage: what is asked, over which views."""

INTERPRETERS: dict[str, tuple[str, dict[str, Any]]] = {
    "qwen": (
        "Qwen/Qwen3-VL-4B-Instruct",
        {
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": QWEN_REVISION,
            "precision": "bfloat16",
            "max_new_tokens": 256,
            "temperature": 0.0,
        },
    ),
    "gemini": (
        "gemini-2.5-flash",
        {"model": "gemini-2.5-flash", "max_output_tokens": 256, "temperature": 0.0},
    ),
}
"""Model and backend parameters of each interpreter, as its effective configuration records them."""

VIEWS = {
    "tight_crop": ["tight_crop"],
    "tight_crop+masked_subject": ["tight_crop", "masked_subject"],
}

PROMPT_POLICY = ExperimentVariable(
    name="prompt_policy",
    kind=VariationKind.POLICY,
    touches=("semantic_interpretation",),
    values=("region/v1", "region/v2"),
    baseline_value="region/v1",
    configuration_fields=("request_policy.prompt_policy",),
    description="the canonical region prompt versus its revised wording",
)
VIEW_POLICY = ExperimentVariable(
    name="view_policy",
    kind=VariationKind.EVIDENCE_CHANNELS,
    touches=("semantic_interpretation",),
    values=tuple(VIEWS),
    baseline_value="tight_crop",
    configuration_fields=("request_policy.view_policy",),
    description="the tight crop alone versus the tight crop followed by the masked subject",
)
INTERPRETER_BACKEND = ExperimentVariable(
    name="semantic_backend",
    kind=VariationKind.BACKEND,
    touches=("semantic_interpretation",),
    values=("qwen", "gemini"),
    baseline_value="qwen",
    configuration_fields=("interpreter",),
    description="local Qwen versus Gemini under the same request policy",
)


def semantic_configuration(
    backend: str = "qwen", changes: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the semantic stage's effective configuration with dotted-path ``changes`` applied."""
    document: dict[str, Any] = {
        "request_policy": copy.deepcopy(REQUEST_POLICY),
        "interpreter": {"backend": backend, backend: copy.deepcopy(INTERPRETERS[backend][1])},
    }
    for path, value in (changes or {}).items():
        *parents, leaf = path.split(".")
        node = document
        for key in parents:
            node = node[key]
        node[leaf] = value
    return document


def request_policy_topology(
    backend: str = "qwen",
    changes: Mapping[str, Any] | None = None,
    *,
    sequence: str = "sequence-0001",
) -> ResolvedTopology:
    """Return a topology whose semantic stage records its configuration field by field.

    ``state_estimation`` (the pose) runs in every arm without being pinned: it is not upstream
    of the semantic stage, so only the executed artifacts can show that every arm shares it.
    """
    return ResolvedTopology(
        stages=(
            stage("ingestion", "ingestion", "ros-adapter", artifact=pinned("sequence", sequence)),
            stage(
                "state_estimation", "state_estimation", "external-pose", depends_on=("ingestion",)
            ),
            stage(
                "region_discovery",
                "visual_perception",
                "sam3",
                depends_on=("ingestion",),
                artifact=pinned("perception_run", "perception-run-0001"),
            ),
            TopologyStage(
                stage_id="semantic_interpretation",
                capability="visual_perception",
                implementation=StageImplementation.from_configuration(
                    backend_id=backend,
                    backend_version="1",
                    model=INTERPRETERS[backend][0],
                    configuration=semantic_configuration(backend, changes),
                ),
                depends_on=("region_discovery",),
            ),
        )
    )


def request_policy_changes(assignment: Mapping[str, str]) -> dict[str, Any]:
    """Return the configuration changes an assignment of request-policy variables makes."""
    changes: dict[str, Any] = {}
    if "prompt_policy" in assignment:
        changes["request_policy.prompt_policy"] = assignment["prompt_policy"]
    if "view_policy" in assignment:
        changes["request_policy.view_policy"] = VIEWS[assignment["view_policy"]]
    return changes


def request_policy_experiment(
    reference: ReferenceSetManifest,
    variables: tuple[ExperimentVariable, ...] = (PROMPT_POLICY,),
    *,
    mode: AblationMode = AblationMode.ONE_AT_A_TIME,
    **overrides: object,
) -> ExperimentManifest:
    """Return a semantic request-policy ablation whose arms differ only as declared."""
    arms = arms_for(
        variables,
        mode,
        lambda assignment: request_policy_topology(
            assignment.get("semantic_backend", "qwen"), request_policy_changes(assignment)
        ),
    )
    fields: dict[str, object] = {
        "experiment_id": "semantic-request-policy",
        "stage_id": EvaluationStage.SEMANTIC_INTERPRETATION,
        "variables": variables,
        "arms": arms,
        "mode": mode,
        "quality": ("semantic.acceptable_claim_rate", "semantic.unsupported_claim_rate"),
    }
    fields.update(overrides)
    return _experiment(reference, **fields)  # type: ignore[arg-type]


# ------------------------------------------------------------------------ executors


def _metric(name: str, value: float, count: int) -> MetricResult:
    return MetricResult(
        metric=name,
        metric_version="1",
        status=MetricStatus.VALUE,
        value=value,
        sample_count=count,
    )


def make_report(
    manifest: ExperimentManifest,
    arm: ExperimentArm,
    values: Mapping[str, float],
    *,
    reference_set: object = None,
    configuration_digest: str | None = None,
    quality: Sequence[MetricResult] | None = None,
    performance: Sequence[MetricResult] | None = None,
) -> EvaluationReport:
    count = len(manifest.selection.sample_ids)
    quality_results = (
        tuple(quality)
        if quality is not None
        else tuple(_metric(ref.name, values[ref.name], count) for ref in manifest.quality_metrics)
    )
    performance_results = (
        tuple(performance)
        if performance is not None
        else tuple(_metric(ref.name, 1.0, count) for ref in manifest.resource_capture)
    )
    return assemble_evaluation_report(
        REGISTRY,
        stage=manifest.evaluated_stage,
        reproducibility=ReproducibilityMetadata(
            evaluator=EvaluatorIdentity(evaluator_id="fake-evaluator", evaluator_version="1"),
            reference_set=reference_set or manifest.selection.reference_set,  # type: ignore[arg-type]
            annotation_schemas=(REGIONS, SEMANTICS),
            input_artifacts=(experiment_artifact_identity(manifest),),
            configuration_digest=configuration_digest or arm.topology.digest(),
            code_version="0.0.1",
            metric_registry=REGISTRY.identity(),
        ),
        quality_metrics=quality_results,
        performance_metrics=performance_results,
        stage_report={"arm": arm.arm_id},
    )


def stage_artifacts_for(arm: ExperimentArm) -> tuple[StageArtifact, ...]:
    """Report the pinned artifact of a reused stage and a deterministic one for an executed stage.

    An executed stage's output is identified by its own record and the artifacts it consumed,
    like a deterministic pipeline: an unaffected stage produces the same content in every arm.
    """
    stages = {item.stage_id: item for item in arm.topology.stages}
    produced: dict[str, ArtifactIdentity] = {}

    def artifact_of(stage_id: str) -> ArtifactIdentity:
        if stage_id not in produced:
            item = stages[stage_id]
            if item.artifact is not None:
                produced[stage_id] = item.artifact
            else:
                inputs = [artifact_of(name).digest for name in item.depends_on]
                digest = content_hash(json.dumps([item.to_record(), inputs], sort_keys=True))
                produced[stage_id] = ArtifactIdentity(
                    kind="stage_output", artifact_id=f"{stage_id}-{digest[7:19]}", digest=digest
                )
        return produced[stage_id]

    return tuple(
        StageArtifact(stage_id=item.stage_id, artifact=artifact_of(item.stage_id))
        for item in arm.topology.stages
    )


def replace_stage_artifact(
    execution: ArmExecution, stage_id: str, artifact: ArtifactIdentity
) -> ArmExecution:
    """Return ``execution`` reporting ``artifact`` for ``stage_id``."""
    return replace(
        execution,
        stage_artifacts=tuple(
            replace(item, artifact=artifact) if item.stage_id == stage_id else item
            for item in execution.stage_artifacts
        ),
    )


def make_executor(
    values: Mapping[str, Mapping[str, float]],
    *,
    failures: Mapping[str, BaseException] | None = None,
    tweak: Callable[[ExperimentManifest, ExperimentArm, ArmExecution], ArmExecution] | None = None,
) -> tuple[ArmExecutor, list[str]]:
    """Return an executor that reports canned values per arm, and the log of executed arms."""
    calls: list[str] = []

    def execute(manifest: ExperimentManifest, arm: ExperimentArm) -> ArmExecution:
        calls.append(arm.arm_id)
        if failures and arm.arm_id in failures:
            raise failures[arm.arm_id]
        result = ArmExecution(
            report=make_report(manifest, arm, values[arm.arm_id]),
            stage_artifacts=stage_artifacts_for(arm),
        )
        return tweak(manifest, arm, result) if tweak else result

    return execute, calls
