"""Versioned pipeline presets and declarative stage graph configuration.

This module answers a narrower question than
:mod:`contextmap.visual_perception.service`: not "how is a resolved stage
graph executed" (that is :func:`~contextmap.visual_perception.service.execute_stage_graph`),
but "how is a stage graph produced from a small, versioned, JSON-serializable
configuration instead of being hand-assembled in code every time".

Three concepts only, matching the issue's boundary:

- :class:`StageSpec` — one scientifically meaningful processing step:
  its capability, how its named inputs are wired to upstream
  ``stage_id``s, which backend identity fills it (or ``None`` for a
  stage whose output is supplied externally, e.g. image preparation),
  and its resolved parameters.
- A concrete backend instance (constructed by a caller-supplied
  :data:`StageBackendFactory`) — the real, replaceable substitution
  point. This module never constructs one itself and never imports a
  model SDK.
- :class:`PipelinePreset` — a versioned, named selection of stages,
  dependencies, backends, and parameters (e.g. ``"canonical/1"``).

This is deliberately not a generic plugin/workflow engine: the executable
capabilities a :class:`StageSpec` can declare form the small, fixed table
in ``_CAPABILITY_ADAPTERS``. Public ports that are not wired into the graph,
such as ``SemanticScorer``, are intentionally absent. See
``src/contextmap/visual_perception/docs/pipeline.md`` for the full
design rationale, the canonical preset's topology, and a worked example
of inserting an optional stage without touching downstream capability
code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from contextmap.visual_perception.dense_region_association import DenseFeatureMap
from contextmap.visual_perception.feature_resolution_enhancement import (
    enhance_feature_resolution,
)
from contextmap.visual_perception.models import (
    BackendProvenance,
    FeatureScope,
    PreparedImage,
    Region2D,
    SceneContext,
    SemanticClaim,
    VisualFeature,
)
from contextmap.visual_perception.ports import (
    FeatureExtractor,
    FeatureResolutionEnhancement,
    RegionDiscovery,
    SemanticInterpreter,
)
from contextmap.visual_perception.serialization import (
    encode_provenance as encode_backend_provenance,
)
from contextmap.visual_perception.service import StageDefinition

PIPELINE_SCHEMA_VERSION = "0.2.0"
"""Pipeline preset encoding schema version."""


class PipelineConfigError(Exception):
    """Raised when a pipeline preset is structurally invalid or cannot be resolved."""


@dataclass(frozen=True, kw_only=True)
class StageSpec:
    """One declarative stage node within a :class:`PipelinePreset`.

    Attributes:
        stage_id: Unique identity of this stage within its preset.
        capability: Which capability this stage exercises. For a
            backend stage (``backend_id`` is not ``None``), must be one
            of the capabilities in ``_CAPABILITY_ADAPTERS`` — the fixed,
            small set this module knows how to invoke. For a source
            stage (``backend_id`` is ``None``), this is a free-form
            label (e.g. ``"image_preparation"``) used only for
            diagnostics; nothing in this module calls it.
        inputs: Named input -> upstream ``stage_id`` this stage reads
            from once that stage has succeeded, e.g.
            ``{"image": "image_preparation", "regions": "region_discovery"}``.
            A downstream stage never branches on which concrete backend
            produced an upstream input — only on the named input's
            declared contract.
        backend_id: Identity of the backend implementation to resolve
            for this stage (e.g. ``"dinov2"``, ``"fake_region_discovery"``).
            ``None`` marks a source stage whose output is supplied
            externally at :meth:`ResolvedPipeline.build_stage_graph`
            time (e.g. the prepared image), never by a capability
            backend.
        optional: Documents that an alternative preset may omit this
            stage without that omission being a mistake. Enforcement is
            structural, not a runtime flag: a preset that omits an
            optional stage must also update whichever downstream
            stage's ``inputs`` referenced it, or validation rejects the
            preset for depending on an unknown stage.
        parameters: Resolved, primitive-valued configuration passed to
            this stage's :data:`StageBackendFactory`, e.g.
            ``{"confidence_threshold": 0.5}``. Never a place for
            secrets/credentials.
        feature_scope: Output scope selected for a ``feature_extractor``
            stage. ``REGION`` makes the ``regions`` input mandatory;
            ``DENSE`` and ``GLOBAL`` accept only the prepared image.
            Must be ``None`` for every other capability.
    """

    stage_id: str
    capability: str
    inputs: Mapping[str, str] = field(default_factory=dict)
    backend_id: str | None = None
    optional: bool = False
    parameters: Mapping[str, object] = field(default_factory=dict)
    feature_scope: FeatureScope | None = None

    @property
    def depends_on(self) -> frozenset[str]:
        """Upstream ``stage_id``s this stage's ``inputs`` reference."""
        return frozenset(self.inputs.values())


@dataclass(frozen=True, kw_only=True)
class PipelinePreset:
    """A versioned, named Visual Perception pipeline configuration.

    Attributes:
        preset_id: Versioned identity, e.g. ``"canonical/1"``. Two
            presets that select different stages/backends/parameters
            must use different ``preset_id``s — a version number is
            never reused for a different topology.
        stages: The declarative stage graph. Order does not need to
            already be dependency-sorted.
        description: Human-readable summary of what this preset is for.
    """

    preset_id: str
    stages: tuple[StageSpec, ...]
    description: str = ""


StageBackendFactory = Callable[[Mapping[str, object]], object]
"""Constructs one stage's backend instance from its resolved ``parameters``.

Never called until :func:`validate_pipeline_preset` has already
succeeded for the preset being resolved — this is what keeps a heavy
model from loading before its graph/config passed validation.
"""


@runtime_checkable
class _ProvenanceCapable(Protocol):
    def backend_provenance(self) -> BackendProvenance: ...


def _run_region_discovery(backend: object, inputs: Mapping[str, object]) -> Sequence[Region2D]:
    assert isinstance(backend, RegionDiscovery)
    image = inputs["image"]
    assert isinstance(image, PreparedImage)
    return backend.discover(image)


def _run_feature_extractor(
    backend: object, inputs: Mapping[str, object]
) -> Sequence[VisualFeature]:
    assert isinstance(backend, FeatureExtractor)
    image = inputs["image"]
    assert isinstance(image, PreparedImage)
    regions = inputs.get("regions", ())
    assert isinstance(regions, Sequence)
    return backend.extract(image, regions=regions)  # type: ignore[arg-type]


def _run_feature_resolution_enhancement(
    backend: object, inputs: Mapping[str, object]
) -> DenseFeatureMap:
    assert isinstance(backend, FeatureResolutionEnhancement)
    dense_map = inputs["dense_map"]
    assert isinstance(dense_map, DenseFeatureMap)
    return enhance_feature_resolution(dense_map, enhancer=backend)


def _run_scene_interpretation(backend: object, inputs: Mapping[str, object]) -> SceneContext | None:
    assert isinstance(backend, SemanticInterpreter)
    image = inputs["image"]
    assert isinstance(image, PreparedImage)
    return backend.interpret_scene(image)


def _run_region_interpretation(
    backend: object, inputs: Mapping[str, object]
) -> Sequence[SemanticClaim]:
    assert isinstance(backend, SemanticInterpreter)
    image = inputs["image"]
    regions = inputs["regions"]
    assert isinstance(image, PreparedImage)
    assert isinstance(regions, Sequence)
    return backend.interpret_regions(image, regions)  # type: ignore[arg-type]


_CAPABILITY_ADAPTERS: Mapping[str, Callable[[object, Mapping[str, object]], object]] = {
    "feature_resolution_enhancement": _run_feature_resolution_enhancement,
    "region_discovery": _run_region_discovery,
    "feature_extractor": _run_feature_extractor,
    "scene_interpretation": _run_scene_interpretation,
    "region_interpretation": _run_region_interpretation,
}

KNOWN_CAPABILITIES = frozenset(_CAPABILITY_ADAPTERS)
"""Capability names this module knows how to invoke on a resolved backend.

Adding a new executable capability means adding one adapter function
to ``_CAPABILITY_ADAPTERS`` — never a generic dispatch/plugin mechanism.
"""

_BASE_REQUIRED_INPUTS: Mapping[str, frozenset[str]] = {
    "region_discovery": frozenset({"image"}),
    "feature_extractor": frozenset({"image"}),
    "feature_resolution_enhancement": frozenset({"dense_map"}),
    "scene_interpretation": frozenset({"image"}),
    "region_interpretation": frozenset({"image", "regions"}),
}

_INPUT_PRODUCER_CAPABILITIES: Mapping[str, frozenset[str]] = {
    "dense_map": frozenset({"dense_feature_map_source", "feature_resolution_enhancement"}),
    "image": frozenset({"image_preparation"}),
    "regions": frozenset({"region_discovery"}),
}


def validate_pipeline_preset(preset: PipelinePreset) -> None:
    """Validate a preset's structure without constructing any backend.

    Checks stage dependency order, named input signatures and producer
    compatibility, that every referenced stage exists, that every
    backend stage declares a known capability, and that the graph has no
    cycle — before any heavy model is loaded.

    Args:
        preset: The preset to validate.

    Raises:
        PipelineConfigError: If ``preset`` has a duplicate ``stage_id``,
            a stage depending on an unknown ``stage_id``, a backend
            stage with an unrecognized ``capability``, an invalid input
            signature or producer, or a cycle.
    """
    by_id: dict[str, StageSpec] = {}
    for stage in preset.stages:
        if stage.stage_id in by_id:
            raise PipelineConfigError(
                f"preset {preset.preset_id!r}: duplicate stage_id {stage.stage_id!r}"
            )
        by_id[stage.stage_id] = stage

    for stage in preset.stages:
        unknown_deps = stage.depends_on - by_id.keys()
        if unknown_deps:
            raise PipelineConfigError(
                f"preset {preset.preset_id!r}: stage {stage.stage_id!r} depends on "
                f"unknown stage(s): {sorted(unknown_deps)}"
            )
        if stage.backend_id is not None and stage.capability not in _CAPABILITY_ADAPTERS:
            raise PipelineConfigError(
                f"preset {preset.preset_id!r}: stage {stage.stage_id!r} has no capability "
                f"adapter for {stage.capability!r}; known capabilities: "
                f"{sorted(_CAPABILITY_ADAPTERS)}"
            )
    _ensure_acyclic(preset.preset_id, by_id)

    for stage in preset.stages:
        if stage.backend_id is not None:
            _validate_stage_inputs(preset.preset_id, stage, by_id)


def _validate_stage_inputs(
    preset_id: str, stage: StageSpec, by_id: Mapping[str, StageSpec]
) -> None:
    """Validate one backend stage's named input contract."""
    required_inputs = _BASE_REQUIRED_INPUTS[stage.capability]
    if stage.capability == "feature_extractor":
        if stage.feature_scope is None:
            raise PipelineConfigError(
                f"preset {preset_id!r}: feature_extractor stage {stage.stage_id!r} "
                "must declare feature_scope"
            )
        if stage.feature_scope is FeatureScope.REGION:
            required_inputs |= {"regions"}
    elif stage.feature_scope is not None:
        raise PipelineConfigError(
            f"preset {preset_id!r}: stage {stage.stage_id!r} declares feature_scope "
            "but is not a feature_extractor"
        )

    declared_inputs = frozenset(stage.inputs)
    missing_inputs = required_inputs - declared_inputs
    if missing_inputs:
        raise PipelineConfigError(
            f"preset {preset_id!r}: stage {stage.stage_id!r} is missing required "
            f"input(s): {sorted(missing_inputs)}"
        )

    allowed_inputs = required_inputs
    unexpected_inputs = declared_inputs - allowed_inputs
    if unexpected_inputs:
        raise PipelineConfigError(
            f"preset {preset_id!r}: stage {stage.stage_id!r} declares unexpected "
            f"input(s): {sorted(unexpected_inputs)}; allowed inputs: {sorted(allowed_inputs)}"
        )

    for input_name, upstream_id in stage.inputs.items():
        producer_capability = by_id[upstream_id].capability
        compatible_capabilities = _INPUT_PRODUCER_CAPABILITIES[input_name]
        if producer_capability not in compatible_capabilities:
            raise PipelineConfigError(
                f"preset {preset_id!r}: stage {stage.stage_id!r} input {input_name!r} "
                f"must come from capability {sorted(compatible_capabilities)}, not "
                f"{producer_capability!r}"
            )


def _ensure_acyclic(preset_id: str, by_id: Mapping[str, StageSpec]) -> None:
    resolved_ids: set[str] = set()
    remaining = dict(by_id)
    while remaining:
        ready = [
            stage_id for stage_id, stage in remaining.items() if stage.depends_on <= resolved_ids
        ]
        if not ready:
            raise PipelineConfigError(
                f"preset {preset_id!r}: cycle detected among stages: {sorted(remaining)}"
            )
        for stage_id in ready:
            resolved_ids.add(stage_id)
            del remaining[stage_id]


@dataclass(frozen=True, kw_only=True)
class ResolvedPipeline:
    """A preset with every backend stage's backend instance already constructed.

    Resolving happens once per run (heavy models load exactly once
    here); :meth:`build_stage_graph` is then cheap to call once per
    processed observation, only injecting that observation's source
    stage output(s) (e.g. its :class:`PreparedImage`).
    """

    preset: PipelinePreset
    backends: Mapping[str, object]

    def build_stage_graph(self, seed_outputs: Mapping[str, object]) -> tuple[StageDefinition, ...]:
        """Build one executable stage graph, injecting this observation's source output(s).

        Args:
            seed_outputs: ``stage_id`` -> output, one entry per source
                stage in the preset (``backend_id is None``), e.g.
                ``{"image_preparation": prepared_image}``.

        Returns:
            A :class:`StageDefinition` sequence ready for
            :func:`~contextmap.visual_perception.service.execute_stage_graph`.

        Raises:
            PipelineConfigError: If ``seed_outputs`` is missing an entry
                for one of the preset's source stages.
        """
        source_stage_ids = {
            stage.stage_id for stage in self.preset.stages if stage.backend_id is None
        }
        missing = source_stage_ids - seed_outputs.keys()
        if missing:
            raise PipelineConfigError(
                f"preset {self.preset.preset_id!r}: missing seed output(s) for source "
                f"stage(s): {sorted(missing)}"
            )

        definitions: list[StageDefinition] = []
        for stage in self.preset.stages:
            if stage.backend_id is None:
                definitions.append(_seeded_stage_definition(stage, seed_outputs[stage.stage_id]))
                continue
            definitions.append(_backend_stage_definition(stage, self.backends[stage.stage_id]))
        return tuple(definitions)

    def backend_provenance(self) -> Mapping[str, BackendProvenance]:
        """Report every resolved backend's provenance, keyed by ``stage_id``.

        Returns:
            One :class:`BackendProvenance` per backend stage, for
            persisting alongside a :class:`~contextmap.visual_perception.models.PerceptionRun`.
        """
        provenance: dict[str, BackendProvenance] = {}
        for stage_id, backend in self.backends.items():
            assert isinstance(backend, _ProvenanceCapable)
            provenance[stage_id] = backend.backend_provenance()
        return provenance

    def configuration_digest(self) -> str:
        """Compute a deterministic reproducibility digest for this resolved pipeline.

        Two resolutions produce the same digest if and only if they
        share the same preset content (``preset_id``, stages,
        parameters) and the same resolved backend provenance (model
        identity, version, configuration fingerprint) for every backend
        stage — i.e. the same preset resolved against a different model
        checkpoint yields a different digest.

        Returns:
            ``"sha256:<hex digest>"``.
        """
        payload = {
            "preset": encode_pipeline_preset(self.preset),
            "backend_provenance": {
                stage_id: encode_backend_provenance(provenance)
                for stage_id, provenance in self.backend_provenance().items()
            },
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def resolve_pipeline(
    preset: PipelinePreset, *, backend_factories: Mapping[str, StageBackendFactory]
) -> ResolvedPipeline:
    """Validate a preset, then construct every backend stage's backend exactly once.

    Args:
        preset: The preset to resolve.
        backend_factories: ``stage_id`` -> factory, one entry per
            backend stage in ``preset`` (``backend_id is not None``).
            Never called for a stage that does not survive validation.

    Returns:
        The resolved pipeline, ready for repeated
        :meth:`ResolvedPipeline.build_stage_graph` calls.

    Raises:
        PipelineConfigError: If ``preset`` fails :func:`validate_pipeline_preset`,
            or ``backend_factories`` is missing an entry for one of
            ``preset``'s backend stages.
    """
    validate_pipeline_preset(preset)

    backend_stage_ids = {stage.stage_id for stage in preset.stages if stage.backend_id is not None}
    missing = backend_stage_ids - backend_factories.keys()
    if missing:
        raise PipelineConfigError(f"missing backend factory for stage(s): {sorted(missing)}")

    backends = {
        stage.stage_id: backend_factories[stage.stage_id](stage.parameters)
        for stage in preset.stages
        if stage.backend_id is not None
    }
    return ResolvedPipeline(preset=preset, backends=backends)


def encode_stage_spec(stage: StageSpec) -> dict[str, Any]:
    """Encode a :class:`StageSpec` into a JSON-serializable dict."""
    return {
        "stage_id": stage.stage_id,
        "capability": stage.capability,
        "inputs": dict(stage.inputs),
        "backend_id": stage.backend_id,
        "optional": stage.optional,
        "parameters": dict(stage.parameters),
        "feature_scope": stage.feature_scope.value if stage.feature_scope is not None else None,
    }


def decode_stage_spec(record: Mapping[str, Any]) -> StageSpec:
    """Decode a :class:`StageSpec` from :func:`encode_stage_spec`'s output."""
    return StageSpec(
        stage_id=record["stage_id"],
        capability=record["capability"],
        inputs=dict(record["inputs"]),
        backend_id=record["backend_id"],
        optional=record["optional"],
        parameters=dict(record["parameters"]),
        feature_scope=(
            FeatureScope(record["feature_scope"]) if record["feature_scope"] is not None else None
        ),
    )


def encode_pipeline_preset(preset: PipelinePreset) -> dict[str, Any]:
    """Encode a :class:`PipelinePreset` into a JSON-serializable dict."""
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "preset_id": preset.preset_id,
        "description": preset.description,
        "stages": [encode_stage_spec(stage) for stage in preset.stages],
    }


def decode_pipeline_preset(record: Mapping[str, Any]) -> PipelinePreset:
    """Decode a :class:`PipelinePreset` from :func:`encode_pipeline_preset`'s output.

    Raises:
        PipelineConfigError: If ``record["schema_version"]`` is not
            understood by this module.
    """
    schema_version = record.get("schema_version")
    if schema_version != PIPELINE_SCHEMA_VERSION:
        raise PipelineConfigError(f"unsupported pipeline preset schema_version: {schema_version!r}")
    return PipelinePreset(
        preset_id=record["preset_id"],
        description=record.get("description", ""),
        stages=tuple(decode_stage_spec(stage) for stage in record["stages"]),
    )


def _seeded_stage_definition(stage: StageSpec, value: object) -> StageDefinition:
    def run(_ctx: Mapping[str, object], *, value: object = value) -> object:
        return value

    return StageDefinition(
        stage_id=stage.stage_id,
        capability=stage.capability,
        depends_on=stage.depends_on,
        run=run,
    )


def _backend_stage_definition(stage: StageSpec, backend: object) -> StageDefinition:
    adapter = _CAPABILITY_ADAPTERS[stage.capability]
    input_names = dict(stage.inputs)

    def run(
        ctx: Mapping[str, object],
        *,
        backend: object = backend,
        adapter: Callable[[object, Mapping[str, object]], object] = adapter,
        input_names: Mapping[str, str] = input_names,
    ) -> object:
        resolved_inputs = {name: ctx[upstream_id] for name, upstream_id in input_names.items()}
        return adapter(backend, resolved_inputs)

    return StageDefinition(
        stage_id=stage.stage_id, capability=stage.capability, depends_on=stage.depends_on, run=run
    )


CANONICAL_PRESET_V1 = PipelinePreset(
    preset_id="canonical/1",
    description=(
        "Currently validated default Visual Perception topology: region "
        "discovery over the prepared image, dense and region-scoped "
        "feature extraction, and scene- and region-level semantic "
        "interpretation."
    ),
    stages=(
        StageSpec(stage_id="image_preparation", capability="image_preparation"),
        StageSpec(
            stage_id="region_discovery",
            capability="region_discovery",
            inputs={"image": "image_preparation"},
            backend_id="region_discovery/canonical",
        ),
        StageSpec(
            stage_id="dense_feature_extraction",
            capability="feature_extractor",
            inputs={"image": "image_preparation"},
            backend_id="dense_feature_extractor/canonical",
            feature_scope=FeatureScope.DENSE,
        ),
        StageSpec(
            stage_id="region_feature_extraction",
            capability="feature_extractor",
            inputs={"image": "image_preparation", "regions": "region_discovery"},
            backend_id="region_feature_extractor/canonical",
            feature_scope=FeatureScope.REGION,
        ),
        StageSpec(
            stage_id="scene_interpretation",
            capability="scene_interpretation",
            inputs={"image": "image_preparation"},
            backend_id="semantic_interpreter/canonical",
        ),
        StageSpec(
            stage_id="region_interpretation",
            capability="region_interpretation",
            inputs={"image": "image_preparation", "regions": "region_discovery"},
            backend_id="semantic_interpreter/canonical",
        ),
    ),
)
"""The versioned canonical/default Visual Perception preset.

Implementing a capability/backend never makes it part of this preset
automatically — inclusion here is a separate, explicit decision from
implementation existing. See
``src/contextmap/visual_perception/docs/pipeline.md`` for the reasoning
and a worked example of a non-canonical preset that inserts an optional
feature-resolution-enhancement stage between ``dense_feature_extraction``
and its consumers without changing this module.
"""
