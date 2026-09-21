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
    """

    stage_id: str
    capability: str
    optional: bool = False
    default_enabled: bool = True
    available: bool = True
    unavailable_reason: str = ""
    components: tuple[str, ...] = ()


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


def _unimplemented(stage_id: str, capability: str, milestone: int) -> StageDeclaration:
    return StageDeclaration(
        stage_id=stage_id,
        capability=capability,
        available=False,
        unavailable_reason=(
            f"the {capability} capability is not implemented yet (milestone #{milestone})"
        ),
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
        ),
        StageDeclaration(
            stage_id="state_estimation",
            capability="state_estimation",
            components=("state_estimation.estimator",),
        ),
        StageDeclaration(stage_id="geometric_mapping", capability="geometric_mapping"),
        StageDeclaration(stage_id="sensor_association", capability="sensor_association"),
        StageDeclaration(
            stage_id="point_representation",
            capability="point_representation",
            optional=True,
            default_enabled=False,
            components=("point_representation.encoder",),
        ),
        StageDeclaration(
            stage_id="semantic_fusion",
            capability="semantic_fusion",
            components=("semantic_fusion.support", "semantic_fusion.accumulation"),
        ),
        _unimplemented("semantic_mapping", "semantic_mapping", 12),
        _unimplemented("entity_resolution", "entity_resolution", 13),
        _unimplemented("spatial_relations", "spatial_relations", 14),
        _unimplemented("context_map", "artifact", 15),
    ),
)

PRESETS: Mapping[str, RuntimePreset] = {CANONICAL_PRESET.preset_id: CANONICAL_PRESET}
"""Known presets. A profile of the same identity starts from each of them."""
