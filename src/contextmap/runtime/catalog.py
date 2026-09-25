"""Static catalog of the stages, variation points and backends the runtime can select.

The catalog is pure data. It names what exists, declares what each backend needs to
run (optional modules, environment secrets) and which stages the canonical topology
contains. It imports no backend, loads no model and decides no scientific value: a
backend parameter, a threshold or a checkpoint belongs to the capability that owns it
and to the user's configuration, never to this table.

A *component* is one variation point owned by a capability, identified as
``"<capability>.<slot>"`` (for example ``"visual_perception.region_discovery"``). A
*stage* is a node of the runtime topology; it owns the components it needs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

CANONICAL_PROFILE_ID = "canonical/1"
"""Versioned identity of the canonical Solution 1 topology and profile.

Before v0.1.0 ships, this is the repository's one and only topology, end to end: ingestion
through ``context_map``. There is no released consumer yet to protect from a topology change, so
this identity is free to keep evolving with the pipeline itself until the release. Once v0.1.0
ships, a topology change starts a new versioned identity instead (``canonical/2`` and so on,
never a silent mutation of a published one) -- that discipline begins at the release, not before
it.
"""

_ROSBAGS_HINT = "pip install 'contextmap[ros1]'"

# Contratos que atravessam as arestas do DAG: os artifacts imutáveis de docs/PIPELINE.md.
SEQUENCE = "SequenceArtifact"
PERCEPTION = "PerceptionRunArtifact"
TRAJECTORY = "StateEstimationRunArtifact"
GEOMETRY = "GeometricMapArtifact"
ASSOCIATION = "SensorAssociationRunArtifact"
REPRESENTATION = "PointRepresentationRunArtifact"
FUSION = "SemanticFusionRunArtifact"
ENTITIES = "SemanticEntityArtifact"
RESOLUTION = "EntityResolutionRunArtifact"
RELATIONS = "SpatialRelationsRunArtifact"
CONTEXT_MAP = "ContextMapArtifact"


@dataclass(frozen=True, kw_only=True)
class BackendSpec:
    """What one selectable backend or policy needs in order to run.

    Attributes:
        backend_id: Identity used in configuration. For a policy it is the policy
            identity the owning capability already versions.
        requires: Top-level optional modules the backend's *bundled* code imports when
            it is constructed. A backend whose model runtime the caller supplies lists
            none: that runtime's modules are the caller's concern. They are only looked
            up, never imported, by discovery.
        secrets: Names of the environment variables that carry credentials this
            backend needs. Values are never part of configuration.
        device_parameter: Name of the backend parameter that receives
            ``resources.device`` when the backend does not set it explicitly, or
            ``None`` when the backend has no device notion.
        install_hint: How to obtain ``requires``, shown when a module is missing.
    """

    backend_id: str
    requires: tuple[str, ...] = ()
    secrets: tuple[str, ...] = ()
    device_parameter: str | None = None
    install_hint: str = ""


@dataclass(frozen=True, kw_only=True)
class ComponentSpec:
    """One variation point of a capability and the backends that can fill it.

    Attributes:
        capability: Owner capability package name.
        slot: Name of the variation point inside the capability.
        backends: Selectable backends keyed by ``backend_id``.
        optional: Whether a stage that owns this component still composes with no backend
            selected for it. It marks a genuinely optional variation point (for example one
            evidence channel of a decision that evaluates several independently): its absence
            is a valid, explicit configuration, never a default standing in for the missing
            choice. ``False`` for every variation point a stage always needs to run.
    """

    capability: str
    slot: str
    backends: Mapping[str, BackendSpec]
    optional: bool = False

    @property
    def component_id(self) -> str:
        """Return the ``"<capability>.<slot>"`` identity used across the runtime."""
        return f"{self.capability}.{self.slot}"


@dataclass(frozen=True, kw_only=True)
class StageInput:
    """One typed input of a stage, wired to the stage that produces it.

    Attributes:
        name: Input name, unique inside the stage.
        contract: Identity of the artifact kind the input consumes.
        source: Stage that produces it in the base topology.
        optional: Whether the stage runs without it. An optional input is dropped when
            its source stage does not take part, and wired when it does.
        multiple: Whether the stage accepts several runs of its source stage at once, kept
            as distinct evidence (for example repeated perception runs over the same
            observations). Every other input accepts exactly one run.
    """

    name: str
    contract: str
    source: str
    optional: bool = False
    multiple: bool = False


@dataclass(frozen=True, kw_only=True)
class Interception:
    """Where an optional stage inserts itself between a producer and a consumer.

    The intercepting stage consumes what the consumer's input used to read and produces
    the same contract, so the consumer keeps reading its declared contract and never
    learns which stage produced it.

    Attributes:
        consumer: Stage whose input is redirected.
        input_name: The redirected input of ``consumer``.
    """

    consumer: str
    input_name: str


@dataclass(frozen=True, kw_only=True)
class StageDeclaration:
    """One node of a runtime topology.

    Attributes:
        stage_id: Identity of the stage inside its preset.
        capability: Owner capability package name.
        optional: Whether a configuration may disable the stage.
        default_enabled: Whether the stage takes part when configuration is silent.
        available: Whether the capability that owns the stage is implemented.
        unavailable_reason: Why the stage cannot run, when ``available`` is false.
        components: Component identities this stage needs a backend for.
        inputs: Typed inputs, each wired to its producing stage.
        output: Identity of the artifact kind the stage produces.
        intercepts: For an optional stage, the input it inserts itself into.
    """

    stage_id: str
    capability: str
    optional: bool = False
    default_enabled: bool = True
    available: bool = True
    unavailable_reason: str = ""
    components: tuple[str, ...] = ()
    inputs: tuple[StageInput, ...] = ()
    output: str | None = None
    intercepts: Interception | None = None


@dataclass(frozen=True, kw_only=True)
class RuntimePreset:
    """A versioned, named selection of runtime stages.

    Two presets that select different stages use different ``preset_id``s: a version
    is never reused for a different topology.

    Attributes:
        preset_id: Versioned identity, for example ``"canonical/1"``.
        stages: The stages, in canonical documentation order.
        description: What the preset is for.
    """

    preset_id: str
    stages: tuple[StageDeclaration, ...]
    description: str = ""

    def __post_init__(self) -> None:
        """Reject a preset that declares one stage identity twice."""
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError(f"preset {self.preset_id!r} declares a stage twice")

    def stage(self, stage_id: str) -> StageDeclaration:
        """Return one stage of the preset.

        Args:
            stage_id: Identity of the stage.

        Returns:
            The declaration.

        Raises:
            KeyError: If the preset has no such stage.
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)


def _component(
    capability: str, slot: str, *backends: BackendSpec, optional: bool = False
) -> ComponentSpec:
    return ComponentSpec(
        capability=capability,
        slot=slot,
        backends={backend.backend_id: backend for backend in backends},
        optional=optional,
    )


_COMPONENT_LIST: tuple[ComponentSpec, ...] = (
    _component(
        "ingestion",
        "source_adapter",
        BackendSpec(backend_id="ros1_bag", requires=("rosbags",), install_hint=_ROSBAGS_HINT),
        BackendSpec(backend_id="ros2_bag", requires=("rosbags",), install_hint=_ROSBAGS_HINT),
    ),
    _component(
        "visual_perception",
        "region_discovery",
        BackendSpec(backend_id="sam2", device_parameter="device"),
        BackendSpec(backend_id="sam3", device_parameter="device"),
        BackendSpec(backend_id="florence2", device_parameter="device"),
    ),
    _component(
        "visual_perception",
        "dense_features",
        BackendSpec(
            backend_id="dinov2",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
        BackendSpec(
            backend_id="dinov3",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
    ),
    _component(
        "visual_perception",
        "region_features",
        BackendSpec(
            backend_id="clip",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
        BackendSpec(
            backend_id="alphaclip",
            requires=("torch", "alpha_clip", "PIL"),
            device_parameter="device",
        ),
    ),
    _component(
        "visual_perception",
        "semantic_interpretation",
        BackendSpec(backend_id="qwen", device_parameter="device"),
        BackendSpec(backend_id="gemini", secrets=("GEMINI_API_KEY",)),
        BackendSpec(backend_id="florence2", device_parameter="device"),
    ),
    _component(
        "state_estimation",
        "estimator",
        BackendSpec(backend_id="external_pose"),
        BackendSpec(backend_id="fast_lio", requires=("rosbags",), install_hint=_ROSBAGS_HINT),
    ),
    _component(
        "geometric_mapping",
        "pose_lookup",
        BackendSpec(backend_id="lookup-policy-v1"),
    ),
    _component(
        "geometric_mapping",
        "motion_correction",
        BackendSpec(backend_id="motion-correction-v1"),
    ),
    _component(
        "sensor_association",
        "occlusion",
        BackendSpec(backend_id="conservative-depth-support-v1"),
    ),
    _component(
        "sensor_association",
        "tolerances",
        BackendSpec(backend_id="diagnostic-tolerances-v1"),
    ),
    _component(
        "sensor_association",
        "pose_policy",
        BackendSpec(backend_id="lookup-policy-v1"),
    ),
    _component(
        "point_representation",
        "encoder",
        BackendSpec(backend_id="geometric_descriptor"),
        BackendSpec(backend_id="ptv3", device_parameter="device"),
    ),
    _component(
        "semantic_fusion",
        "support",
        BackendSpec(backend_id="geometry-jaccard-support-v1"),
    ),
    _component(
        "semantic_fusion",
        "accumulation",
        BackendSpec(backend_id="baseline-evidence-accumulation-v1"),
        BackendSpec(backend_id="quality-aware-evidence-accumulation-v1"),
    ),
    _component(
        "entity_resolution",
        "retrieval",
        BackendSpec(backend_id="entity-candidate-retrieval-v1"),
    ),
    _component(
        "entity_resolution",
        "resolution",
        BackendSpec(backend_id="conservative-staged-resolution-v1"),
    ),
    _component(
        "entity_resolution",
        "geometry_comparison",
        BackendSpec(backend_id="entity-geometry-comparison-v1"),
    ),
    _component(
        "entity_resolution",
        "semantic_compatibility",
        BackendSpec(backend_id="entity-semantic-compatibility-v1"),
        optional=True,
    ),
    _component(
        "entity_resolution",
        "temporal_compatibility",
        BackendSpec(backend_id="entity-temporal-compatibility-v1"),
        optional=True,
    ),
    _component(
        "entity_resolution",
        "appearance",
        BackendSpec(backend_id="entity-appearance-comparison-v1"),
        optional=True,
    ),
    _component(
        "entity_resolution",
        "representation",
        BackendSpec(backend_id="entity-representation-comparison-v1"),
        optional=True,
    ),
    _component(
        "semantic_mapping",
        "geometry_summary",
        BackendSpec(backend_id="entity-geometry-summary-v1"),
    ),
    _component(
        "spatial_relations",
        "frame_conventions",
        BackendSpec(backend_id="map-frame-conventions-v1"),
    ),
    _component(
        "spatial_relations",
        "candidate",
        BackendSpec(backend_id="bounds-neighborhood-candidates-v1"),
    ),
    _component(
        "spatial_relations",
        "geometry_summary",
        BackendSpec(backend_id="entity-geometry-summary-v1"),
    ),
    _component(
        "spatial_relations",
        "geometric_predicate",
        BackendSpec(backend_id="bounds-geometric-predicates-v1"),
        optional=True,
    ),
    _component(
        "spatial_relations",
        "contact_predicate",
        BackendSpec(backend_id="point-contact-predicates-v1"),
        optional=True,
    ),
)

COMPONENTS: Mapping[str, ComponentSpec] = {
    component.component_id: component for component in _COMPONENT_LIST
}
"""Every selectable variation point, keyed by ``"<capability>.<slot>"``."""

CANONICAL_PRESET = RuntimePreset(
    preset_id=CANONICAL_PROFILE_ID,
    description=(
        "The Solution 1 topology, end to end: from a recorded source to the ContextMapArtifact. "
        "Pre-v0.1.0, this is the repository's only topology and it is free to grow with the "
        "pipeline; a topology change starts a new versioned identity only once the release ships."
    ),
    stages=(
        StageDeclaration(
            stage_id="ingestion",
            capability="ingestion",
            components=("ingestion.source_adapter",),
            output=SEQUENCE,
        ),
        # Issue #555: an opt-in second ingestion stage, publishing an auxiliary, pose-only
        # SequenceArtifact for state_estimation to merge with the main one. No component: its
        # executor is always injected by the caller (see IngestionStageExecutor), same as
        # "ingestion" itself -- an incremental bridge, not general multi-source ingestion.
        StageDeclaration(
            stage_id="pose_ingestion",
            capability="ingestion",
            optional=True,
            default_enabled=False,
            output=SEQUENCE,
        ),
        StageDeclaration(
            stage_id="visual_perception",
            capability="visual_perception",
            components=(
                "visual_perception.region_discovery",
                "visual_perception.dense_features",
                "visual_perception.region_features",
                "visual_perception.semantic_interpretation",
            ),
            inputs=(StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),),
            output=PERCEPTION,
        ),
        StageDeclaration(
            stage_id="state_estimation",
            capability="state_estimation",
            components=("state_estimation.estimator",),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(
                    name="pose_sequence",
                    contract=SEQUENCE,
                    source="pose_ingestion",
                    optional=True,
                ),
            ),
            output=TRAJECTORY,
        ),
        StageDeclaration(
            stage_id="geometric_mapping",
            capability="geometric_mapping",
            components=("geometric_mapping.pose_lookup", "geometric_mapping.motion_correction"),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(name="trajectory", contract=TRAJECTORY, source="state_estimation"),
            ),
            output=GEOMETRY,
        ),
        StageDeclaration(
            stage_id="sensor_association",
            capability="sensor_association",
            components=(
                "sensor_association.occlusion",
                "sensor_association.tolerances",
                "sensor_association.pose_policy",
            ),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(
                    name="perception",
                    contract=PERCEPTION,
                    source="visual_perception",
                    multiple=True,
                ),
                StageInput(name="trajectory", contract=TRAJECTORY, source="state_estimation"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
            ),
            output=ASSOCIATION,
        ),
        StageDeclaration(
            stage_id="point_representation",
            capability="point_representation",
            optional=True,
            default_enabled=False,
            components=("point_representation.encoder",),
            inputs=(
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
                StageInput(
                    name="association",
                    contract=ASSOCIATION,
                    source="sensor_association",
                    optional=True,
                ),
            ),
            output=REPRESENTATION,
        ),
        StageDeclaration(
            stage_id="semantic_fusion",
            capability="semantic_fusion",
            components=("semantic_fusion.support", "semantic_fusion.accumulation"),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(
                    name="association",
                    contract=ASSOCIATION,
                    source="sensor_association",
                    multiple=True,
                ),
                StageInput(
                    name="perception",
                    contract=PERCEPTION,
                    source="visual_perception",
                    multiple=True,
                ),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
                StageInput(
                    name="representation",
                    contract=REPRESENTATION,
                    source="point_representation",
                    optional=True,
                ),
            ),
            output=FUSION,
        ),
        StageDeclaration(
            stage_id="semantic_mapping",
            capability="semantic_mapping",
            components=("semantic_mapping.geometry_summary",),
            inputs=(
                StageInput(name="fusion", contract=FUSION, source="semantic_fusion"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
            ),
            output=ENTITIES,
        ),
        StageDeclaration(
            stage_id="entity_resolution",
            capability="entity_resolution",
            components=(
                "entity_resolution.retrieval",
                "entity_resolution.resolution",
                "entity_resolution.geometry_comparison",
                "entity_resolution.semantic_compatibility",
                "entity_resolution.temporal_compatibility",
                "entity_resolution.appearance",
                "entity_resolution.representation",
            ),
            inputs=(StageInput(name="entities", contract=ENTITIES, source="semantic_mapping"),),
            output=RESOLUTION,
        ),
        StageDeclaration(
            stage_id="spatial_relations",
            capability="spatial_relations",
            components=(
                "spatial_relations.frame_conventions",
                "spatial_relations.candidate",
                "spatial_relations.geometry_summary",
                "spatial_relations.geometric_predicate",
                "spatial_relations.contact_predicate",
            ),
            inputs=(
                StageInput(name="entities", contract=RESOLUTION, source="entity_resolution"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
            ),
            output=RELATIONS,
        ),
        StageDeclaration(
            stage_id="context_map",
            capability="artifact",
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
                StageInput(name="entities", contract=RESOLUTION, source="entity_resolution"),
                StageInput(name="relations", contract=RELATIONS, source="spatial_relations"),
            ),
            output=CONTEXT_MAP,
        ),
    ),
)

PRESETS: Mapping[str, RuntimePreset] = {
    CANONICAL_PRESET.preset_id: CANONICAL_PRESET,
}
"""Known presets. A profile of the same identity starts from each of them."""
