"""Controlled evaluation protocols for the optional techniques of Solution 1.

Two techniques are optional and must earn their place with evidence from our own
reference scenarios, not from external papers:

* **feature-resolution enhancement** between dense feature extraction and Sensor
  Association: native dense features against the same immutable extraction
  artifact passed through ``FeatureResolutionEnhancement``;
* **quality-aware Semantic Fusion**: uniform fusion against fusion weighted by
  the canonical ``ObservationQuality``, over the exact same association artifact.

A protocol is a set of :class:`~contextmap.evaluation.experiments.ExperimentManifest`
that differ only in the stage they evaluate. Each experiment is a controlled
comparison (only the declared variable varies, the varied part consumes pinned
immutable artifacts), so the *identical upstream artifact identities* the
acceptance criteria ask for are enforced by construction. One experiment is
built per evaluated stage because a report measures one stage; the metrics of
each stage are reported separately, and the strata a protocol reports on are
named here and must be declared by the reference set. See
``src/contextmap/evaluation/docs/optional-techniques.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from contextmap.evaluation._persistence import canonical_digest
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
    VariationKind,
)
from contextmap.evaluation.metrics import EvaluationStage, MetricKind, MetricRegistry
from contextmap.evaluation.reference_set import ReferenceSetManifest
from contextmap.sensor_association import QualityComponent

TECHNIQUE_FEATURE_RESOLUTION = "feature-resolution-enhancement"
TECHNIQUE_QUALITY_AWARE_FUSION = "quality-aware-fusion"

FEATURE_EXTRACTION_STAGE = "dense_feature_extraction"
RESOLUTION_ENHANCEMENT_STAGE = "feature_resolution_enhancement"
SENSOR_ASSOCIATION_STAGE = "sensor_association"
SEMANTIC_FUSION_STAGE = "semantic_fusion"

BASELINE_ARM_ID = "baseline"
VARIANT_ARM_ID = "variant"

FEATURE_RESOLUTION_STRATA = (
    "range_band",
    "visibility",
    "support_density",
    "valid_region",
    "projected_size",
)
"""Range/distance, visibility/occlusion, LiDAR support density, image (fisheye)
valid region and projected support size."""

OBSERVATION_QUALITY_STRATA = (
    *(f"quality.{component.value}" for component in QualityComponent),
    "ambiguity_level",
    "conflict_level",
)
"""One factor per canonical ``ObservationQuality`` component, plus ambiguity and conflict."""

RESOURCE_METRICS = (
    "runtime.wall_time",
    "runtime.peak_memory",
    "runtime.storage_size",
    "runtime.throughput",
    "runtime.failure_rate",
)
"""The cost metrics every experiment captures, reported apart from quality."""

_STAGE = EvaluationStage

_FEATURE_RESOLUTION_METRICS: tuple[tuple[EvaluationStage, tuple[str, ...]], ...] = (
    (
        _STAGE.SENSOR_ASSOCIATION,
        (
            "association.reprojection_error.median",
            "association.visible_support.ratio",
            "association.feature_anchoring.rate",
        ),
    ),
    (
        _STAGE.SEMANTIC_FUSION,
        (
            "fusion.reference_recovery.rate",
            "fusion.view_consistency.rate",
            "fusion.ambiguity_retention.rate",
        ),
    ),
    (
        _STAGE.ENTITY_RESOLUTION,
        (
            "entity.false_merge.rate",
            "entity.duplicate.rate",
            "entity.semantic_accuracy.rate",
        ),
    ),
    (_STAGE.SPATIAL_RELATIONS, ("relations.f1", "relations.negative_violation.rate")),
)

_QUALITY_AWARE_FUSION_METRICS = _FEATURE_RESOLUTION_METRICS[1:]
"""The association artifact is identical across arms, so its metrics cannot differ."""


class TechniqueProtocolError(ValueError):
    """Raised when topologies do not form the protocol they claim to."""


@dataclass(frozen=True, kw_only=True)
class TechniqueProtocol:
    """A technique's controlled comparison, as one experiment per evaluated stage.

    Attributes:
        technique_id: The optional technique under evaluation.
        protocol_version: Version of the protocol; also the version of its experiments.
        variable_name: The single declared variable.
        stratification_factors: Strata the results are reported by. Only the
            factors the reference set declares are used; the rest are reported
            as not available, never guessed.
        experiments: One experiment per evaluated stage, sharing arms and topologies.
    """

    technique_id: str
    protocol_version: str
    variable_name: str
    stratification_factors: tuple[str, ...]
    experiments: tuple[ExperimentManifest, ...]

    def __post_init__(self) -> None:
        """Require experiments over distinct stages that compare the same arms."""
        if not self.experiments:
            raise TechniqueProtocolError("a protocol needs at least one experiment")
        stages = [item.evaluated_stage for item in self.experiments]
        if len(set(stages)) != len(stages):
            raise TechniqueProtocolError("a protocol has one experiment per evaluated stage")
        first = self.experiments[0]
        for item in self.experiments[1:]:
            if [arm.topology.digest() for arm in item.arms] != [
                arm.topology.digest() for arm in first.arms
            ]:
                raise TechniqueProtocolError(
                    "every experiment of a protocol must compare the same resolved topologies"
                )
            if item.selection != first.selection:
                raise TechniqueProtocolError(
                    "every experiment of a protocol must run on the same selection"
                )

    @property
    def stages(self) -> tuple[EvaluationStage, ...]:
        """Return the evaluated stages, in experiment order."""
        return tuple(item.evaluated_stage for item in self.experiments)

    def experiment_for(self, stage: EvaluationStage) -> ExperimentManifest:
        """Return the experiment that evaluates a stage."""
        for item in self.experiments:
            if item.evaluated_stage is stage:
                return item
        raise TechniqueProtocolError(f"the protocol does not evaluate stage {stage.value!r}")

    def undeclared_factors(self, reference_set: ReferenceSetManifest) -> tuple[str, ...]:
        """Return the protocol's factors the reference set does not declare."""
        declared = {item.name for item in reference_set.stratum_definitions}
        return tuple(name for name in self.stratification_factors if name not in declared)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "technique_id": self.technique_id,
            "protocol_version": self.protocol_version,
            "variable_name": self.variable_name,
            "stratification_factors": list(self.stratification_factors),
            "experiments": [
                {
                    "experiment_id": item.experiment_id,
                    "evaluated_stage": item.evaluated_stage.value,
                    "digest": item.digest(),
                }
                for item in self.experiments
            ],
        }

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the protocol."""
        return canonical_digest(self.to_record())


def _require_stage(topology: ResolvedTopology, stage_id: str, role: str) -> None:
    if stage_id not in {item.stage_id for item in topology.stages}:
        raise TechniqueProtocolError(f"the {role} topology has no stage {stage_id!r}")


def _depends_on(topology: ResolvedTopology, stage_id: str, dependency: str, role: str) -> None:
    if dependency not in topology.stage(stage_id).depends_on:
        raise TechniqueProtocolError(
            f"in the {role} topology, stage {stage_id!r} must consume stage {dependency!r}"
        )


def _experiments(
    *,
    technique_id: str,
    protocol_version: str,
    variable: ExperimentVariable,
    baseline: ResolvedTopology,
    variant: ResolvedTopology,
    stage_metrics: Sequence[tuple[EvaluationStage, tuple[str, ...]]],
    selection: SelectionBinding,
    base_configuration_digest: str,
    registry: MetricRegistry,
    fixed_controls: Sequence[FixedControl],
    repetitions_per_sample: int,
    description: str,
) -> tuple[ExperimentManifest, ...]:
    arms = (
        ExperimentArm(
            arm_id=BASELINE_ARM_ID,
            assignments=((variable.name, variable.baseline_value),),
            topology=baseline,
            description="the technique off",
        ),
        ExperimentArm(
            arm_id=VARIANT_ARM_ID,
            assignments=((variable.name, variable.values[1]),),
            topology=variant,
            description="the technique on",
        ),
    )
    experiments: list[ExperimentManifest] = []
    for stage, names in stage_metrics:
        for name in names:
            definition = registry.get(name, "1")
            if definition.kind is not MetricKind.QUALITY or definition.stage is not stage:
                raise TechniqueProtocolError(
                    f"metric {definition.key} is not a quality metric of stage {stage.value!r}"
                )
        experiments.append(
            ExperimentManifest(
                experiment_id=f"{technique_id}--{stage.value}",
                version=protocol_version,
                description=f"{description}, measured at {stage.value}",
                purpose=ExperimentPurpose.EVALUATION,
                evaluated_stage=stage,
                selection=selection,
                repetitions_per_sample=repetitions_per_sample,
                base_configuration_digest=base_configuration_digest,
                variables=(variable,),
                mode=AblationMode.ONE_AT_A_TIME,
                baseline_arm_id=BASELINE_ARM_ID,
                arms=arms,
                fixed_controls=(
                    FixedControl(
                        name="technique_protocol", value=f"{technique_id}/{protocol_version}"
                    ),
                    *fixed_controls,
                ),
                quality_metrics=tuple(MetricRef(name=name, version="1") for name in names),
                resource_capture=tuple(
                    MetricRef(name=name, version="1") for name in RESOURCE_METRICS
                ),
                registry=registry.identity(),
            )
        )
    return tuple(experiments)


def build_feature_resolution_protocol(
    *,
    protocol_version: str,
    selection: SelectionBinding,
    base_configuration_digest: str,
    baseline: ResolvedTopology,
    enhanced: ResolvedTopology,
    registry: MetricRegistry,
    fixed_controls: Sequence[FixedControl] = (),
    repetitions_per_sample: int = 1,
) -> TechniqueProtocol:
    """Build Experiment A: native dense features against the enhanced feature path.

    ``baseline`` runs ``dense_feature_extraction -> sensor_association``; ``enhanced``
    runs the **same** pinned extraction artifact through ``feature_resolution_enhancement``
    before ``sensor_association``. Everything else (source observations, prepared images,
    calibration, pose, geometric map, regions, downstream policies) must be identical
    and pinned: the experiment manifest refuses any undeclared difference.

    Raises:
        TechniqueProtocolError: If the topologies do not have that shape.
        ExperimentError: If they differ in anything undeclared or consume an unpinned upstream.
    """
    for topology, role in ((baseline, "baseline"), (enhanced, "enhanced")):
        _require_stage(topology, FEATURE_EXTRACTION_STAGE, role)
        _require_stage(topology, SENSOR_ASSOCIATION_STAGE, role)
    if RESOLUTION_ENHANCEMENT_STAGE in {item.stage_id for item in baseline.stages}:
        raise TechniqueProtocolError("the baseline topology must not contain the enhancement stage")
    _require_stage(enhanced, RESOLUTION_ENHANCEMENT_STAGE, "enhanced")
    _depends_on(baseline, SENSOR_ASSOCIATION_STAGE, FEATURE_EXTRACTION_STAGE, "baseline")
    _depends_on(enhanced, RESOLUTION_ENHANCEMENT_STAGE, FEATURE_EXTRACTION_STAGE, "enhanced")
    _depends_on(enhanced, SENSOR_ASSOCIATION_STAGE, RESOLUTION_ENHANCEMENT_STAGE, "enhanced")
    variable = ExperimentVariable(
        name="feature_resolution",
        kind=VariationKind.TOPOLOGY,
        touches=(RESOLUTION_ENHANCEMENT_STAGE, SENSOR_ASSOCIATION_STAGE),
        values=("native", "enhanced"),
        baseline_value="native",
        description="dense features as extracted versus through FeatureResolutionEnhancement",
    )
    experiments = _experiments(
        technique_id=TECHNIQUE_FEATURE_RESOLUTION,
        protocol_version=protocol_version,
        variable=variable,
        baseline=baseline,
        variant=enhanced,
        stage_metrics=_FEATURE_RESOLUTION_METRICS,
        selection=selection,
        base_configuration_digest=base_configuration_digest,
        registry=registry,
        fixed_controls=fixed_controls,
        repetitions_per_sample=repetitions_per_sample,
        description="native dense features against resolution-enhanced features",
    )
    return TechniqueProtocol(
        technique_id=TECHNIQUE_FEATURE_RESOLUTION,
        protocol_version=protocol_version,
        variable_name=variable.name,
        stratification_factors=FEATURE_RESOLUTION_STRATA,
        experiments=experiments,
    )


def build_quality_aware_fusion_protocol(
    *,
    protocol_version: str,
    selection: SelectionBinding,
    base_configuration_digest: str,
    uniform: ResolvedTopology,
    quality_aware: ResolvedTopology,
    registry: MetricRegistry,
    fixed_controls: Sequence[FixedControl] = (),
    repetitions_per_sample: int = 1,
) -> TechniqueProtocol:
    """Build Experiment B: uniform Semantic Fusion against quality-aware fusion.

    Both topologies consume the **same** pinned Sensor Association artifact (the same
    association and fusion-support evidence and physical observations) and differ only in
    the configuration of ``semantic_fusion``.

    Raises:
        TechniqueProtocolError: If the topologies do not have that shape.
        ExperimentError: If they differ in anything undeclared or consume an unpinned upstream.
    """
    for topology, role in ((uniform, "uniform"), (quality_aware, "quality-aware")):
        _require_stage(topology, SENSOR_ASSOCIATION_STAGE, role)
        _require_stage(topology, SEMANTIC_FUSION_STAGE, role)
        _depends_on(topology, SEMANTIC_FUSION_STAGE, SENSOR_ASSOCIATION_STAGE, role)
    variable = ExperimentVariable(
        name="fusion_policy",
        kind=VariationKind.POLICY,
        touches=(SEMANTIC_FUSION_STAGE,),
        values=("uniform", "quality_aware"),
        baseline_value="uniform",
        description="baseline/uniform evidence accumulation versus the quality-aware policy",
    )
    experiments = _experiments(
        technique_id=TECHNIQUE_QUALITY_AWARE_FUSION,
        protocol_version=protocol_version,
        variable=variable,
        baseline=uniform,
        variant=quality_aware,
        stage_metrics=_QUALITY_AWARE_FUSION_METRICS,
        selection=selection,
        base_configuration_digest=base_configuration_digest,
        registry=registry,
        fixed_controls=fixed_controls,
        repetitions_per_sample=repetitions_per_sample,
        description="uniform fusion against quality-aware fusion over the same evidence",
    )
    return TechniqueProtocol(
        technique_id=TECHNIQUE_QUALITY_AWARE_FUSION,
        protocol_version=protocol_version,
        variable_name=variable.name,
        stratification_factors=OBSERVATION_QUALITY_STRATA,
        experiments=experiments,
    )


def protocol_metric_names(protocol: TechniqueProtocol) -> Mapping[EvaluationStage, tuple[str, ...]]:
    """Return the quality metrics each stage of a protocol reports."""
    return {
        item.evaluated_stage: tuple(ref.name for ref in item.quality_metrics)
        for item in protocol.experiments
    }
