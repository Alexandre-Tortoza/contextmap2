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
"""Versioned identity of the canonical Solution 1 topology and profile."""

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
    """

    capability: str
    slot: str
    backends: Mapping[str, BackendSpec]

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


def _component(capability: str, slot: str, *backends: BackendSpec) -> ComponentSpec:
    return ComponentSpec(
        capability=capability,
        slot=slot,
        backends={backend.backend_id: backend for backend in backends},
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
)

COMPONENTS: Mapping[str, ComponentSpec] = {
    component.component_id: component for component in _COMPONENT_LIST
}
"""Every selectable variation point, keyed by ``"<capability>.<slot>"``."""


def _unimplemented(
    stage_id: str,
    capability: str,
    milestone: int,
    *,
    inputs: tuple[StageInput, ...],
    output: str,
) -> StageDeclaration:
    return StageDeclaration(
        stage_id=stage_id,
        capability=capability,
        available=False,
        unavailable_reason=(
            f"the {capability} capability is not implemented yet (milestone #{milestone})"
        ),
        inputs=inputs,
        output=output,
    )


CANONICAL_PRESET = RuntimePreset(
    preset_id=CANONICAL_PROFILE_ID,
    description=(
        "Full Solution 1 topology, from a recorded source to the ContextMapArtifact. "
        "Stages whose capability is not implemented yet are declared unavailable and "
        "reported explicitly instead of being skipped."
    ),
    stages=(
        StageDeclaration(
            stage_id="ingestion",
            capability="ingestion",
            components=("ingestion.source_adapter",),
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
            inputs=(StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),),
            output=TRAJECTORY,
        ),
        StageDeclaration(
            stage_id="geometric_mapping",
            capability="geometric_mapping",
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(name="trajectory", contract=TRAJECTORY, source="state_estimation"),
            ),
            output=GEOMETRY,
        ),
        StageDeclaration(
            stage_id="sensor_association",
            capability="sensor_association",
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
        _unimplemented(
            "semantic_mapping",
            "semantic_mapping",
            12,
            inputs=(StageInput(name="fusion", contract=FUSION, source="semantic_fusion"),),
            output=ENTITIES,
        ),
        _unimplemented(
            "entity_resolution",
            "entity_resolution",
            13,
            inputs=(StageInput(name="entities", contract=ENTITIES, source="semantic_mapping"),),
            output=RESOLUTION,
        ),
        _unimplemented(
            "spatial_relations",
            "spatial_relations",
            14,
            inputs=(StageInput(name="entities", contract=RESOLUTION, source="entity_resolution"),),
            output=RELATIONS,
        ),
        _unimplemented(
            "context_map",
            "artifact",
            15,
            inputs=(
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
                StageInput(name="entities", contract=RESOLUTION, source="entity_resolution"),
                StageInput(name="relations", contract=RELATIONS, source="spatial_relations"),
            ),
            output=CONTEXT_MAP,
        ),
    ),
)

PRESETS: Mapping[str, RuntimePreset] = {CANONICAL_PRESET.preset_id: CANONICAL_PRESET}
"""Known presets. A profile of the same identity starts from each of them."""
