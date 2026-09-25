"""Composition root: turn an effective configuration into concrete implementations.

This is the single place where concrete backends are named and constructed. A
capability never imports the runtime and a downstream stage never learns which backend
produced its input: every implementation leaves here behind the port its capability
publishes, so swapping a backend is a configuration change and no downstream source
changes.

Three rules keep the composition honest:

- Constructing is not loading. A backend's parameters are validated by the capability's
  own configuration class, and its optional modules and secrets are checked, before any
  model is requested. Bundled loaders are lazy; a model the repository does not know how
  to load is supplied by the caller through a :data:`RuntimeProvider`.
- There is no fallback. A selected backend that cannot be built raises a specific error;
  nothing substitutes another backend or drops the stage.
- No scientific policy lives here. Factories only translate configured values into the
  capability's own types; defaults, ranges and meaning belong to the capability.

Backend modules are imported lazily, inside the factory that needs them, so composing a
configuration imports only the backends it selects. There is no registry, plugin system
or service locator: the table of factories below is explicit and closed.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from contextmap.runtime.catalog import COMPONENTS, PRESETS, StageDeclaration
from contextmap.runtime.coercion import ParameterError, build_config
from contextmap.runtime.config import (
    ComponentConfig,
    ConfigurationError,
    EffectiveConfig,
    ResolvedSecrets,
    check_component_availability,
    check_selection,
)
from contextmap.runtime.errors import (
    BackendConfigurationError,
    BackendRuntimeMissingError,
    BackendUnavailableError,
    CompositionError,
    ProviderConfigurationError,
    StageUnavailableError,
)

if TYPE_CHECKING:
    from contextmap.entity_resolution import (
        CandidateRetrievalPolicy,
        ComparisonChannels,
        ConservativeResolutionPolicy,
    )
    from contextmap.geometric_mapping import MotionCorrectionPolicy
    from contextmap.ingestion import SourceAdapter, SourceAdapterConfig
    from contextmap.point_representation import PointEncoder
    from contextmap.runtime.pipeline import StageExecutor
    from contextmap.semantic_fusion import (
        BaselineAccumulationPolicy,
        GeometryOverlapSupportPolicy,
        QualityAwareAccumulationPolicy,
    )
    from contextmap.semantic_mapping import GeometrySummaryPolicy, SemanticMapId
    from contextmap.sensor_association import (
        CandidateGeometryPolicy,
        DiagnosticTolerances,
        OcclusionPolicy,
    )
    from contextmap.spatial_relations import RelationsRunPolicies
    from contextmap.state_estimation import LookupPolicy, StateEstimator
    from contextmap.visual_perception import (
        FeatureExtractor,
        PerceptionRunId,
        RegionDiscovery,
        SemanticInterpreter,
    )

RuntimeProvider = Callable[[Any, ResolvedSecrets], object]
"""Builds the model runtime or client of a backend the repository cannot load itself.

It receives the capability's own configuration object, already validated, and the secrets
that backend declares (and only those), and returns whatever runtime the backend's
adapter expects (for example a SAM3 runtime or a Gemini client). Loading the model is the
provider's business, so heavy SDKs stay out of the runtime package.
"""


def resolve_provider(component_id: str, target: str) -> RuntimeProvider:
    """Resolve a ``"resources.providers"`` target into the :data:`RuntimeProvider` it names.

    This is the declarative counterpart of passing a :data:`RuntimeProvider` in Python: a
    configuration names, instead of embeds, the callable the composition root asks for a
    model runtime. The imported attribute *is* the provider -- there is no intermediate
    factory or wrapper, so it must already accept ``(config, secrets)`` and return the
    runtime, exactly like a provider supplied through ``compose(providers=...)``.

    Security posture: only the ``"module:attribute"`` syntax is accepted, never ``eval``;
    no module is ever installed automatically; and this is called lazily, by
    :meth:`_Context.runtime`/:meth:`_Context.optional_runtime`, only for a component that is
    actually being composed -- never eagerly for every target a document declares. A
    ``resources.providers`` target is Python code that is imported and then called at
    runtime, the same trust boundary as any other configuration that names executable code:
    configuration from an untrusted source must never be resolved this way.

    Args:
        component_id: Variation point the target was declared for, used only to build an
            actionable error.
        target: ``"module_name:attribute"``.

    Returns:
        The imported attribute, already verified callable.

    Raises:
        ProviderConfigurationError: If ``target`` is malformed, its module cannot be
            imported, the attribute does not exist on it, or the attribute is not callable.
    """
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ProviderConfigurationError(
            component_id, target, "must look like 'module:attribute' with both parts non-empty"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ProviderConfigurationError(
            component_id, target, f"module {module_name!r} could not be imported: {error}"
        ) from error
    try:
        attribute = getattr(module, attribute_name)
    except AttributeError as error:
        raise ProviderConfigurationError(
            component_id, target, f"module {module_name!r} has no attribute {attribute_name!r}"
        ) from error
    if not callable(attribute):
        raise ProviderConfigurationError(component_id, target, "is not callable")
    return cast("RuntimeProvider", attribute)


FeatureFactory = Callable[["FeatureBuildScope"], "FeatureExtractor"]
SourceAdapterFactory = Callable[["SourceAdapterConfig"], "SourceAdapter"]


@dataclass(frozen=True, kw_only=True)
class FeatureBuildScope:
    """Values a feature backend needs that exist only while one run is executing.

    Feature extractors write their payloads into the run that owns them, so they can only
    be constructed once that run exists. Everything here is execution state, not
    configuration.

    Attributes:
        run_id: Perception run that owns the produced features.
        feature_stage_id: Stage identity that namespaces the features inside the run.
        source_artifact_id: Run or artifact that owns the persisted payloads.
        payload_sink: Receives payload arrays before the run artifact is finalized.
        prepared_image_root: Directory that resolves prepared-image payload references.
        mask_source: Supplies region masks to a mask-conditioned backend.
        checkpoint_root: Directory that resolves local checkpoint paths.
    """

    run_id: PerceptionRunId
    feature_stage_id: str
    source_artifact_id: str
    payload_sink: Any
    prepared_image_root: Path | None = None
    mask_source: Any = None
    checkpoint_root: Path | None = None


@dataclass(frozen=True, kw_only=True)
class ComposedRuntime:
    """The implementations built for one effective configuration.

    A field is ``None`` when its stage was not composed: not selected, not requested, or
    without an implemented capability (see ``unavailable_stages``).

    Attributes:
        effective: The configuration everything below was built from.
        stages: Identities of the stages that were composed, in topology order.
        unavailable_stages: Enabled stages skipped because their capability does not
            exist yet, with the reason. Nothing stands in for them.
        source_adapter: Builds the configured source adapter for one ingestion request.
        region_discovery: Region discovery backend.
        dense_features: Builds the dense feature extractor once a run scope exists.
        region_features: Builds the region feature extractor once a run scope exists.
        semantic_interpreter: Semantic interpretation backend.
        state_estimator: State estimation backend.
        geometric_mapping_pose_lookup: Pose lookup rule Geometric Mapping uses to place
            each scan.
        motion_correction: Disposition of scans that are not known to be corrected.
        point_encoder: Point encoder, only when the optional stage is selected.
        support_policy: Semantic Fusion support policy.
        accumulation_policy: Semantic Fusion accumulation policy.
        association_candidates: Which map geometry each Sensor Association frame evaluates
            before projection; its range is ``None`` unless ``policies.association_max_range_m``
            sets one, which keeps the whole map evaluated exactly as before (#562).
        occlusion_policy: Sensor Association visibility rule.
        association_tolerances: Sensor Association diagnostic tolerances.
        association_pose_policy: Pose lookup rule Sensor Association uses per frame.
        semantic_mapping_geometry_summary: Semantic Mapping's entity-geometry summary policy.
        entity_retrieval_policy: Entity Resolution candidate retrieval policy.
        entity_comparison_channels: The evidence channels Entity Resolution evaluates; geometry
            is always present, every other channel is ``None`` when not selected.
        entity_resolution_policy: Entity Resolution's conservative decision policy.
        spatial_relations_policies: Spatial Relations' effective policies: frame conventions,
            candidate generation and the geometry-summary policy are always present, the
            predicate evaluators are ``None`` when not selected. ``geometry_summary`` here is
            the only source ``SpatialRelationsExecutor`` reads it from, so the policy it uses
            to summarize geometry and the one persisted in the run's own provenance can never
            diverge (review of PR #540, second round).
    """

    effective: EffectiveConfig
    stages: tuple[str, ...]
    unavailable_stages: Mapping[str, str] = field(default_factory=dict)
    source_adapter: SourceAdapterFactory | None = None
    region_discovery: RegionDiscovery | None = None
    dense_features: FeatureFactory | None = None
    region_features: FeatureFactory | None = None
    semantic_interpreter: SemanticInterpreter | None = None
    state_estimator: StateEstimator | None = None
    geometric_mapping_pose_lookup: LookupPolicy | None = None
    motion_correction: MotionCorrectionPolicy | None = None
    point_encoder: PointEncoder | None = None
    support_policy: GeometryOverlapSupportPolicy | None = None
    accumulation_policy: BaselineAccumulationPolicy | QualityAwareAccumulationPolicy | None = None
    occlusion_policy: OcclusionPolicy | None = None
    association_tolerances: DiagnosticTolerances | None = None
    association_pose_policy: LookupPolicy | None = None
    association_candidates: CandidateGeometryPolicy | None = None
    semantic_mapping_geometry_summary: GeometrySummaryPolicy | None = None
    entity_retrieval_policy: CandidateRetrievalPolicy | None = None
    entity_comparison_channels: ComparisonChannels | None = None
    entity_resolution_policy: ConservativeResolutionPolicy | None = None
    spatial_relations_policies: RelationsRunPolicies | None = None


def compose(
    effective: EffectiveConfig,
    *,
    providers: Mapping[str, RuntimeProvider] | None = None,
    stages: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    on_provider_override: Callable[[str], None] | None = None,
) -> ComposedRuntime:
    """Construct the implementations an effective configuration selects.

    Args:
        effective: The resolved configuration.
        providers: Model runtimes or clients supplied by the caller, keyed by component
            identity (``"<capability>.<slot>"``), for backends without a bundled loader
            and to override a bundled one. For a component this leaves unset, a
            ``resources.providers`` target declared in ``effective`` is resolved instead,
            lazily, only if that component is actually being composed; see
            :func:`resolve_provider`.
        stages: The stages to compose, or ``None`` for every enabled stage whose
            capability is implemented.
        environ: Environment to read secrets from; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed;
            defaults to an :mod:`importlib` lookup.
        on_provider_override: Called with a component identity when ``providers`` supplies
            an entry for it that wins over a ``resources.providers`` target ``effective``
            also declares for the same component -- the only case where a caller-embedded
            provider silently takes precedence over what the configuration itself names.
            Never called when the two never disagree, in particular on every ordinary run
            through the installed CLI, which never passes ``providers``.

    Returns:
        The composed implementations. No model has been loaded.

    Raises:
        ConfigurationError: If a requested stage is unknown or disabled, or a composed
            stage has a variation point with no backend selected.
        StageUnavailableError: If a requested stage has no implemented capability.
        BackendConfigurationError: If a capability rejects its backend's parameters.
        BackendUnavailableError: If a backend needs a module or secret that is missing.
        BackendRuntimeMissingError: If a backend needs a runtime nobody supplied and the
            configuration declares no ``resources.providers`` target for it either.
        ProviderConfigurationError: If a ``resources.providers`` target this component needs
            is malformed or cannot be resolved into a callable.
    """
    preset = PRESETS[effective.config.pipeline.preset]
    enabled = effective.config.pipeline.stages
    unavailable: dict[str, str] = {}
    selected: list[StageDeclaration]
    if stages is None:
        selected = []
        for stage in preset.stages:
            if not enabled.get(stage.stage_id, False):
                continue
            if stage.available:
                selected.append(stage)
            else:
                unavailable[stage.stage_id] = stage.unavailable_reason
    else:
        requested = list(stages)
        known = {stage.stage_id for stage in preset.stages}
        for stage_id in requested:
            if stage_id not in known:
                raise ConfigurationError.single(
                    f"unknown stage {stage_id!r}; known stages: {', '.join(sorted(known))}",
                    path="stages",
                )
            declaration = preset.stage(stage_id)
            if not declaration.available:
                raise StageUnavailableError(stage_id, declaration.unavailable_reason)
            if not enabled.get(stage_id, False):
                raise ConfigurationError.single(
                    f"stage {stage_id!r} is disabled in the configuration", path="stages"
                )
        selected = [stage for stage in preset.stages if stage.stage_id in requested]

    _require_selection(effective, selected)
    context = _Context(
        effective=effective,
        providers=dict(providers or {}),
        environ=os.environ if environ is None else environ,
        module_available=module_available,
        on_provider_override=on_provider_override,
    )
    attributes: dict[str, object] = {}
    for stage in selected:
        attributes.update(_STAGE_COMPOSERS[stage.stage_id](context))
    return ComposedRuntime(
        effective=effective,
        stages=tuple(stage.stage_id for stage in selected),
        unavailable_stages=unavailable,
        **cast("dict[str, Any]", attributes),
    )


def composable_backends() -> frozenset[tuple[str, str]]:
    """List every ``(component_id, backend_id)`` this composition root can construct.

    Returns:
        The pairs with a factory. The catalog and this table must agree; a test enforces it.
    """
    return frozenset(
        (component_id, backend_id)
        for component_id, factories in _FACTORIES.items()
        for backend_id in factories
    )


def _require_selection(effective: EffectiveConfig, selected: list[StageDeclaration]) -> None:
    """Fail when a composed stage has a variation point with no backend."""
    owned = {component_id for stage in selected for component_id in stage.components}
    problems = [
        problem
        for problem in check_selection(effective.config)
        if problem.path.removeprefix("components.") in owned
    ]
    if problems:
        raise ConfigurationError(problems)


@dataclass(frozen=True, kw_only=True)
class _Context:
    """What one composition needs besides the configuration: providers and environment."""

    effective: EffectiveConfig
    providers: Mapping[str, RuntimeProvider]
    environ: Mapping[str, str]
    module_available: Callable[[str], bool] | None
    on_provider_override: Callable[[str], None] | None = None

    def component(self, component_id: str) -> ComponentConfig:
        return self.effective.config.components[component_id]

    def build(
        self,
        component_id: str,
        config_type: type[Any],
        *,
        extras: Mapping[str, type[Any]] | None = None,
        required: Collection[str] = (),
        fixed: Mapping[str, object] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Validate a backend's parameters through the capability's own configuration.

        ``extras`` are reserved parameter groups that build a second object next to the
        configuration (for example a support policy); ``fixed`` are values the slot itself
        decides and a user cannot contradict.
        """
        component = self.component(component_id)
        backend = component.backend or ""
        parameters: dict[str, Any] = dict(component.parameters)
        problems: list[str] = []
        extra_values: dict[str, Any] = {}
        for name, extra_type in (extras or {}).items():
            if name not in parameters:
                if name in required:
                    problems.append(f"missing required parameter: {name}")
                continue
            try:
                extra_values[name] = build_config(extra_type, parameters.pop(name))
            except ParameterError as error:
                problems.extend(f"{name}: {problem}" for problem in error.problems)
        for name, value in (fixed or {}).items():
            if name in parameters and parameters[name] != value:
                problems.append(f"{name}: fixed to {value!r} by the {component_id} slot")
            parameters[name] = value
        config: Any = None
        try:
            config = build_config(config_type, parameters)
        except ParameterError as error:
            problems.extend(error.problems)
        if problems:
            raise BackendConfigurationError(component_id, backend, problems)
        return config, extra_values

    def ensure_available(self, component_id: str) -> None:
        """Fail when a bundled code path needs a module or secret that is missing.

        A component whose runtime a provider supplies -- explicitly, or through a declared
        ``resources.providers`` target -- never needs its own bundled module: that heavy
        import is the provider's business, never ``contextmap.runtime``'s (see the module
        docstring). Only the module check is skipped this way; a missing secret is still
        reported, since a provider is not necessarily what reads it.
        """
        declared = self._declared_provider_target(component_id)
        provider_expected = component_id in self.providers or declared is not None
        problems = check_component_availability(
            component_id,
            self.component(component_id),
            environ=self.environ,
            module_available=self.module_available,
            check_modules=not provider_expected,
        )
        if problems:
            raise BackendUnavailableError(problems)

    def secrets(self, component_id: str) -> ResolvedSecrets:
        """Hold only the secrets the selected backend declares."""
        backend = self.component(component_id).backend or ""
        names = COMPONENTS[component_id].backends[backend].secrets
        return ResolvedSecrets(names, {n: self.environ[n] for n in names if self.environ.get(n)})

    def _declared_provider_target(self, component_id: str) -> str | None:
        """Read the ``resources.providers`` target the configuration declares, if any."""
        return self.effective.config.resources.providers.get(component_id)

    def _resolved_provider(self, component_id: str) -> RuntimeProvider | None:
        """Resolve this component's provider, or ``None`` when nobody supplies one.

        Precedence: an explicit ``providers=`` entry always wins, even over a
        ``resources.providers`` target the configuration declares for the same component --
        and that override is reported through ``on_provider_override`` exactly when it
        actually happens (both exist for the same component). Otherwise, a declared target
        is resolved lazily, right here, only because this component is genuinely being
        composed; the target is never resolved for a component nobody asked to build.
        """
        explicit = self.providers.get(component_id)
        target = self._declared_provider_target(component_id)
        if explicit is not None:
            if target is not None and self.on_provider_override is not None:
                self.on_provider_override(component_id)
            return explicit
        if target is not None:
            return resolve_provider(component_id, target)
        return None

    def runtime(self, component_id: str, config: Any, protocol: str) -> Any:
        """Ask the caller's or the configuration's provider for a backend's runtime.

        Raises:
            BackendRuntimeMissingError: If neither an explicit provider nor a declared
                ``resources.providers`` target supplies one.
            ProviderConfigurationError: If a declared target cannot be resolved.
        """
        provider = self._resolved_provider(component_id)
        if provider is None:
            backend = self.component(component_id).backend or ""
            raise BackendRuntimeMissingError(component_id, backend, protocol)
        return provider(config, self.secrets(component_id))

    def optional_runtime(self, component_id: str, config: Any) -> Any:
        """Return the resolved runtime for a backend that also bundles a loader, or ``None``."""
        provider = self._resolved_provider(component_id)
        return None if provider is None else provider(config, self.secrets(component_id))


Factory = Callable[[_Context, str], Any]


# --- ingestion --------------------------------------------------------------------


def _source_adapter(context: _Context, component_id: str) -> SourceAdapterFactory:
    component = context.component(component_id)
    backend = component.backend or ""
    if component.parameters:
        raise BackendConfigurationError(
            component_id,
            backend,
            [
                f"takes no parameters; the source path and topics come with each request: "
                f"{', '.join(sorted(component.parameters))}"
            ],
        )
    context.ensure_available(component_id)

    def build(request: SourceAdapterConfig) -> SourceAdapter:
        if request.source_type != backend:
            raise BackendConfigurationError(
                component_id,
                backend,
                [
                    f"the request asks for source_type {request.source_type!r}; the configuration "
                    f"selects {backend!r} and never substitutes another adapter"
                ],
            )
        if backend == "ros1_bag":
            from contextmap.ingestion.adapters.ros1_bag import Ros1BagSourceAdapter

            return Ros1BagSourceAdapter(request)
        from contextmap.ingestion.adapters.ros2_bag import Ros2BagSourceAdapter

        return Ros2BagSourceAdapter(request)

    return build


# --- visual perception: region discovery -------------------------------------------


def _region_extras() -> dict[str, type[Any]]:
    from contextmap.visual_perception import DiscoveryPassConfig, NormalizationConfig

    return {"pass_config": DiscoveryPassConfig, "normalization_config": NormalizationConfig}


def _sam2(context: _Context, component_id: str) -> RegionDiscovery:
    from contextmap.visual_perception.backends.sam2 import Sam2Config, Sam2RegionDiscovery

    config, extras = context.build(component_id, Sam2Config, extras=_region_extras())
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "Sam2Runtime")
    return Sam2RegionDiscovery(config=config, runtime=runtime, **extras)


def _sam3(context: _Context, component_id: str) -> RegionDiscovery:
    from contextmap.visual_perception.backends.sam3 import Sam3Config, Sam3RegionDiscovery

    config, extras = context.build(component_id, Sam3Config, extras=_region_extras())
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "Sam3Runtime")
    return Sam3RegionDiscovery(config=config, runtime=runtime, **extras)


def _florence2_regions(context: _Context, component_id: str) -> RegionDiscovery:
    from contextmap.visual_perception.backends.florence2 import (
        Florence2Config,
        Florence2RegionDiscovery,
    )

    config, extras = context.build(component_id, Florence2Config, extras=_region_extras())
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "Florence2Runtime")
    return Florence2RegionDiscovery(config=config, runtime=runtime, **extras)


# --- visual perception: features (run-scoped) --------------------------------------


def _dinov2(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.backends.dinov2 import DinoV2Config, DinoV2DenseFeatureBackend

    config, _ = context.build(component_id, DinoV2Config)
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return DinoV2DenseFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            source_artifact_id=scope.source_artifact_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _dinov3(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.backends.dinov3 import DinoV3Config, DinoV3DenseFeatureBackend

    config, _ = context.build(component_id, DinoV3Config)
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return DinoV3DenseFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            source_artifact_id=scope.source_artifact_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _siglip2(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.backends.siglip2 import (
        Siglip2Config,
        Siglip2FeatureBackend,
    )

    # O slot de features densas fixa o escopo: um SigLIP2 GLOBAL não produz DenseFeatureMap.
    config, _ = context.build(component_id, Siglip2Config, fixed={"scope": "dense"})
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return Siglip2FeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            source_artifact_id=scope.source_artifact_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _clip(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.backends.clip import ClipConfig, ClipVisualFeatureBackend

    # O slot de features por região fixa o escopo: não é uma escolha do usuário.
    config, _ = context.build(component_id, ClipConfig, fixed={"scope": "region"})
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return ClipVisualFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _alphaclip(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.backends.alphaclip import (
        AlphaClipConfig,
        AlphaClipRegionFeatureBackend,
    )

    config, _ = context.build(component_id, AlphaClipConfig)
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        if scope.mask_source is None:
            raise BackendConfigurationError(
                component_id,
                "alphaclip",
                ["needs a mask_source in the FeatureBuildScope: it is conditioned on region masks"],
            )
        return AlphaClipRegionFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            mask_source=scope.mask_source,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
            checkpoint_root=scope.checkpoint_root,
        )

    return build


# --- visual perception: semantic interpretation ------------------------------------


def _qwen(context: _Context, component_id: str) -> SemanticInterpreter:
    from contextmap.visual_perception.backends.qwen import (
        QwenSemanticConfig,
        QwenSemanticInterpreter,
    )

    config, _ = context.build(component_id, QwenSemanticConfig)
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "QwenRuntime")
    return QwenSemanticInterpreter(config=config, runtime=runtime)


def _gemini(context: _Context, component_id: str) -> SemanticInterpreter:
    from contextmap.visual_perception.backends.gemini import (
        GeminiSemanticConfig,
        GeminiSemanticInterpreter,
    )

    config, _ = context.build(component_id, GeminiSemanticConfig)
    context.ensure_available(component_id)
    client = context.runtime(component_id, config, "GeminiClient")
    return GeminiSemanticInterpreter(config=config, client=client)


def _florence2_semantic(context: _Context, component_id: str) -> SemanticInterpreter:
    from contextmap.visual_perception.backends.florence2_semantic import (
        Florence2SemanticConfig,
        Florence2SemanticInterpreter,
    )

    config, _ = context.build(component_id, Florence2SemanticConfig)
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "Florence2SemanticRuntime")
    return Florence2SemanticInterpreter(config=config, runtime=runtime)


# --- state estimation --------------------------------------------------------------


def _external_pose(context: _Context, component_id: str) -> StateEstimator:
    from contextmap.state_estimation.backends.external_pose import (
        ExternalPoseConfig,
        ExternalPoseEstimator,
    )

    config, _ = context.build(component_id, ExternalPoseConfig)
    context.ensure_available(component_id)
    return ExternalPoseEstimator(config)


@dataclass(frozen=True, kw_only=True)
class _NoParameters:
    """A backend whose configuration is made only of reserved parameter groups."""


@dataclass(frozen=True, kw_only=True)
class _FastLioRunnerParameters:
    """Parameters of the bundled subprocess runner, as written in configuration."""

    command: tuple[str, ...]
    timeout_s: float
    work_root: Path | None = None


def _fast_lio(context: _Context, component_id: str) -> StateEstimator:
    from contextmap.state_estimation.backends.fast_lio import FastLioConfig, FastLioEstimator
    from contextmap.state_estimation.backends.fast_lio_process import SubprocessFastLioRunner

    supplied = component_id in context.providers
    config, extras = context.build(
        component_id,
        FastLioConfig,
        extras={"runner": _FastLioRunnerParameters},
        required=() if supplied else ("runner",),
    )
    context.ensure_available(component_id)
    if supplied:
        runner = context.runtime(component_id, config, "FastLioRunner")
    else:
        parameters = extras["runner"]
        try:
            runner = SubprocessFastLioRunner(
                command=list(parameters.command),
                timeout_s=parameters.timeout_s,
                work_root=parameters.work_root,
            )
        except ValueError as error:
            raise BackendConfigurationError(
                component_id, "fast_lio", [f"runner: {error}"]
            ) from error
    return FastLioEstimator(config, runner)


# --- geometric mapping ---------------------------------------------------------------


def _lookup_policy(context: _Context, component_id: str) -> Any:
    from contextmap.state_estimation import LookupPolicy

    policy, _ = context.build(component_id, LookupPolicy)
    return policy


def _motion_correction(context: _Context, component_id: str) -> Any:
    from contextmap.geometric_mapping import MotionCorrectionPolicy

    policy, _ = context.build(component_id, MotionCorrectionPolicy)
    return policy


# --- sensor association --------------------------------------------------------------


def _occlusion_policy(context: _Context, component_id: str) -> Any:
    from contextmap.sensor_association import OcclusionPolicy

    policy, _ = context.build(component_id, OcclusionPolicy)
    return policy


def _diagnostic_tolerances(context: _Context, component_id: str) -> Any:
    from contextmap.sensor_association import DiagnosticTolerances

    policy, _ = context.build(component_id, DiagnosticTolerances)
    return policy


# --- entity resolution ---------------------------------------------------------------


def _entity_retrieval_policy(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import CandidateRetrievalPolicy

    policy, _ = context.build(component_id, CandidateRetrievalPolicy)
    return policy


def _entity_resolution_policy(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import ConservativeResolutionPolicy

    policy, _ = context.build(component_id, ConservativeResolutionPolicy)
    return policy


def _entity_geometry_comparison(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import GeometryComparisonPolicy

    policy, _ = context.build(component_id, GeometryComparisonPolicy)
    return policy


def _entity_semantic_compatibility(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import SemanticCompatibilityPolicy

    policy, _ = context.build(component_id, SemanticCompatibilityPolicy)
    return policy


def _entity_temporal_compatibility(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import TemporalCompatibilityPolicy

    policy, _ = context.build(component_id, TemporalCompatibilityPolicy)
    return policy


def _entity_appearance(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import AppearanceComparator, AppearanceComparisonPolicy

    config, _ = context.build(component_id, AppearanceComparisonPolicy)
    context.ensure_available(component_id)
    source = context.runtime(component_id, config, "FeatureVectorSource")
    return AppearanceComparator(source=source, policy=config)


def _entity_representation(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import (
        RepresentationComparator,
        RepresentationComparisonPolicy,
    )

    config, _ = context.build(component_id, RepresentationComparisonPolicy)
    context.ensure_available(component_id)
    source = context.runtime(component_id, config, "RepresentationVectorSource")
    return RepresentationComparator(source=source, policy=config)


# --- spatial relations -----------------------------------------------------------------


def _frame_conventions(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import FrameConventions

    policy, _ = context.build(component_id, FrameConventions)
    return policy


def _candidate_policy(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import CandidatePolicy

    policy, _ = context.build(component_id, CandidatePolicy)
    return policy


def _geometry_summary_policy(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_mapping import GeometrySummaryPolicy

    policy, _ = context.build(component_id, GeometrySummaryPolicy)
    return policy


def _geometric_predicate_policy(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import GeometricPredicatePolicy

    policy, _ = context.build(component_id, GeometricPredicatePolicy)
    return policy


def _contact_predicate_policy(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import ContactPredicatePolicy

    policy, _ = context.build(component_id, ContactPredicatePolicy)
    return policy


# --- point representation ----------------------------------------------------------


def _geometric_descriptor(context: _Context, component_id: str) -> PointEncoder:
    from contextmap.point_representation import SupportPolicy
    from contextmap.point_representation.backends.geometric_descriptor import (
        GeometricDescriptorEncoder,
    )

    _, extras = context.build(
        component_id,
        _NoParameters,
        extras={"support_policy": SupportPolicy},
        required=("support_policy",),
    )
    context.ensure_available(component_id)
    try:
        return GeometricDescriptorEncoder(extras["support_policy"])
    except ValueError as error:
        raise BackendConfigurationError(
            component_id, "geometric_descriptor", [f"support_policy: {error}"]
        ) from error


def _ptv3(context: _Context, component_id: str) -> PointEncoder:
    from contextmap.point_representation import SupportPolicy
    from contextmap.point_representation.backends.ptv3 import PTv3Config, PTv3PointEncoder

    config, extras = context.build(
        component_id,
        PTv3Config,
        extras={"support_policy": SupportPolicy},
        required=("support_policy",),
    )
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "PTv3Runtime")
    try:
        return PTv3PointEncoder(
            config=config, support_policy=extras["support_policy"], runtime=runtime
        )
    except ValueError as error:
        raise BackendConfigurationError(
            component_id, "ptv3", [f"support_policy: {error}"]
        ) from error


# --- semantic fusion ---------------------------------------------------------------


def _geometry_overlap_support(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_fusion import GeometryOverlapSupportPolicy

    policy, _ = context.build(component_id, GeometryOverlapSupportPolicy)
    return policy


def _baseline_accumulation(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_fusion import BaselineAccumulationPolicy

    policy, _ = context.build(component_id, BaselineAccumulationPolicy)
    return policy


def _quality_aware_accumulation(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_fusion import QualityAwareAccumulationPolicy

    policy, _ = context.build(component_id, QualityAwareAccumulationPolicy)
    return policy


_FACTORIES: Mapping[str, Mapping[str, Factory]] = {
    "ingestion.source_adapter": {"ros1_bag": _source_adapter, "ros2_bag": _source_adapter},
    "visual_perception.region_discovery": {
        "sam2": _sam2,
        "sam3": _sam3,
        "florence2": _florence2_regions,
    },
    "visual_perception.dense_features": {
        "dinov2": _dinov2,
        "dinov3": _dinov3,
        "siglip2": _siglip2,
    },
    "visual_perception.region_features": {"clip": _clip, "alphaclip": _alphaclip},
    "visual_perception.semantic_interpretation": {
        "qwen": _qwen,
        "gemini": _gemini,
        "florence2": _florence2_semantic,
    },
    "state_estimation.estimator": {"external_pose": _external_pose, "fast_lio": _fast_lio},
    "geometric_mapping.pose_lookup": {"lookup-policy-v1": _lookup_policy},
    "geometric_mapping.motion_correction": {"motion-correction-v1": _motion_correction},
    "sensor_association.occlusion": {"conservative-depth-support-v1": _occlusion_policy},
    "sensor_association.tolerances": {"diagnostic-tolerances-v1": _diagnostic_tolerances},
    "sensor_association.pose_policy": {"lookup-policy-v1": _lookup_policy},
    "point_representation.encoder": {"geometric_descriptor": _geometric_descriptor, "ptv3": _ptv3},
    "semantic_fusion.support": {"geometry-jaccard-support-v1": _geometry_overlap_support},
    "semantic_fusion.accumulation": {
        "baseline-evidence-accumulation-v1": _baseline_accumulation,
        "quality-aware-evidence-accumulation-v1": _quality_aware_accumulation,
    },
    "entity_resolution.retrieval": {"entity-candidate-retrieval-v1": _entity_retrieval_policy},
    "entity_resolution.resolution": {
        "conservative-staged-resolution-v1": _entity_resolution_policy
    },
    "entity_resolution.geometry_comparison": {
        "entity-geometry-comparison-v1": _entity_geometry_comparison
    },
    "entity_resolution.semantic_compatibility": {
        "entity-semantic-compatibility-v1": _entity_semantic_compatibility
    },
    "entity_resolution.temporal_compatibility": {
        "entity-temporal-compatibility-v1": _entity_temporal_compatibility
    },
    "entity_resolution.appearance": {"entity-appearance-comparison-v1": _entity_appearance},
    "entity_resolution.representation": {
        "entity-representation-comparison-v1": _entity_representation
    },
    "semantic_mapping.geometry_summary": {"entity-geometry-summary-v1": _geometry_summary_policy},
    "spatial_relations.frame_conventions": {"map-frame-conventions-v1": _frame_conventions},
    "spatial_relations.candidate": {"bounds-neighborhood-candidates-v1": _candidate_policy},
    "spatial_relations.geometry_summary": {"entity-geometry-summary-v1": _geometry_summary_policy},
    "spatial_relations.geometric_predicate": {
        "bounds-geometric-predicates-v1": _geometric_predicate_policy
    },
    "spatial_relations.contact_predicate": {
        "point-contact-predicates-v1": _contact_predicate_policy
    },
}


def _construct(context: _Context, component_id: str) -> Any:
    """Build one variation point with the factory of its selected backend."""
    backend = context.component(component_id).backend
    assert backend is not None  # a seleção completa já foi exigida em compose().
    return _FACTORIES[component_id][backend](context, component_id)


def _construct_optional(context: _Context, component_id: str) -> Any | None:
    """Build a genuinely optional variation point, or ``None`` when it was not selected.

    Unlike :func:`_construct`, a missing backend here is not a configuration error: the
    catalog marks the component ``optional`` (see :class:`~contextmap.runtime.catalog.
    ComponentSpec`), so :func:`~contextmap.runtime.config.check_component_selection` never
    requires it, and its absence must never be replaced by a default policy.
    """
    if context.component(component_id).backend is None:
        return None
    return _construct(context, component_id)


# --- stages ------------------------------------------------------------------------


def _compose_ingestion(context: _Context) -> dict[str, object]:
    return {"source_adapter": _construct(context, "ingestion.source_adapter")}


def _compose_visual_perception(context: _Context) -> dict[str, object]:
    return {
        "region_discovery": _construct(context, "visual_perception.region_discovery"),
        "dense_features": _construct(context, "visual_perception.dense_features"),
        "region_features": _construct(context, "visual_perception.region_features"),
        "semantic_interpreter": _construct(context, "visual_perception.semantic_interpretation"),
    }


def _compose_state_estimation(context: _Context) -> dict[str, object]:
    return {"state_estimator": _construct(context, "state_estimation.estimator")}


def _compose_geometric_mapping(context: _Context) -> dict[str, object]:
    return {
        "geometric_mapping_pose_lookup": _construct(context, "geometric_mapping.pose_lookup"),
        "motion_correction": _construct(context, "geometric_mapping.motion_correction"),
    }


def _compose_sensor_association(context: _Context) -> dict[str, object]:
    from contextmap.sensor_association import CandidateGeometryPolicy

    return {
        "occlusion_policy": _construct(context, "sensor_association.occlusion"),
        "association_tolerances": _construct(context, "sensor_association.tolerances"),
        "association_pose_policy": _construct(context, "sensor_association.pose_policy"),
        # A política de candidatos é um policy cruzado da execução (#562), não um ponto de
        # variação de backend: o padrão None avalia o mapa inteiro, exatamente como antes.
        "association_candidates": CandidateGeometryPolicy(
            max_range_m=context.effective.config.policies.association_max_range_m
        ),
    }


def _compose_point_representation(context: _Context) -> dict[str, object]:
    return {"point_encoder": _construct(context, "point_representation.encoder")}


def _compose_semantic_fusion(context: _Context) -> dict[str, object]:
    return {
        "support_policy": _construct(context, "semantic_fusion.support"),
        "accumulation_policy": _construct(context, "semantic_fusion.accumulation"),
    }


def _compose_nothing(context: _Context) -> dict[str, object]:
    """Compose a stage with no variation point: its service is stateless capability code."""
    return {}


def _compose_entity_resolution(context: _Context) -> dict[str, object]:
    from contextmap.entity_resolution import ComparisonChannels

    channels = ComparisonChannels(
        geometry=_construct(context, "entity_resolution.geometry_comparison"),
        semantic=_construct_optional(context, "entity_resolution.semantic_compatibility"),
        temporal=_construct_optional(context, "entity_resolution.temporal_compatibility"),
        appearance=_construct_optional(context, "entity_resolution.appearance"),
        representation=_construct_optional(context, "entity_resolution.representation"),
    )
    return {
        "entity_retrieval_policy": _construct(context, "entity_resolution.retrieval"),
        "entity_comparison_channels": channels,
        "entity_resolution_policy": _construct(context, "entity_resolution.resolution"),
    }


def _compose_semantic_mapping(context: _Context) -> dict[str, object]:
    return {
        "semantic_mapping_geometry_summary": _construct(
            context, "semantic_mapping.geometry_summary"
        )
    }


def _compose_spatial_relations(context: _Context) -> dict[str, object]:
    from contextmap.spatial_relations import RelationsRunPolicies

    policies = RelationsRunPolicies(
        frame_conventions=_construct(context, "spatial_relations.frame_conventions"),
        candidate=_construct(context, "spatial_relations.candidate"),
        geometry_summary=_construct(context, "spatial_relations.geometry_summary"),
        geometric=_construct_optional(context, "spatial_relations.geometric_predicate"),
        contact=_construct_optional(context, "spatial_relations.contact_predicate"),
    )
    return {"spatial_relations_policies": policies}


_STAGE_COMPOSERS: Mapping[str, Callable[[_Context], dict[str, object]]] = {
    "ingestion": _compose_ingestion,
    "pose_ingestion": _compose_nothing,
    "visual_perception": _compose_visual_perception,
    "state_estimation": _compose_state_estimation,
    "geometric_mapping": _compose_geometric_mapping,
    "sensor_association": _compose_sensor_association,
    "point_representation": _compose_point_representation,
    "semantic_fusion": _compose_semantic_fusion,
    "semantic_mapping": _compose_semantic_mapping,
    "entity_resolution": _compose_entity_resolution,
    "spatial_relations": _compose_spatial_relations,
    "context_map": _compose_nothing,
}


def composed_stages() -> frozenset[str]:
    """List the stages this composition root can compose.

    Returns:
        Stage identities with a composer. Every available stage of the catalog must be here.
    """
    return frozenset(_STAGE_COMPOSERS)


def compose_executors(
    effective: EffectiveConfig,
    *,
    providers: Mapping[str, RuntimeProvider] | None = None,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    on_provider_override: Callable[[str], None] | None = None,
    semantic_map_id: SemanticMapId | None = None,
    code_digest: str | None = None,
    code_version: str | None = None,
) -> dict[str, StageExecutor]:
    """Compose the real :class:`~contextmap.runtime.pipeline.StageExecutor` a DAG run needs.

    This is the automatic counterpart of :func:`compose`: instead of handing back the
    composed backends and policies, it wraps each one into the concrete executor class of
    ``contextmap.runtime.executors`` its stage needs, keyed by ``stage_id``, exactly as a
    caller previously had to build them by hand. A stage whose variation points are not
    (yet) fully selected in ``effective``, whose selected backend rejects its own
    configured parameters, or whose capability the executor does not support, is left out
    -- never filled with a placeholder or allowed to abort composing the other stages. The
    existing ``missing_executors``/"no executor is registered" preflight reporting already
    explains why such a stage will not run; this function never hides that behind a guess.

    ``state_estimation``, ``geometric_mapping``, ``sensor_association``, ``semantic_fusion``,
    ``visual_perception``, ``semantic_mapping``, ``entity_resolution``, ``spatial_relations`` and
    ``context_map`` can be composed this way: each needs only the effective configuration and the
    upstream artifacts the DAG already carries -- except ``semantic_mapping``, which also needs
    ``semantic_map_id`` and ``code_digest`` (see below).

    - ``ingestion`` and ``pose_ingestion`` (issue #555's opt-in auxiliary pose stage, see
      ``catalog.py``) are not composed here: :class:`~contextmap.runtime.ingestion_service.
      IngestionStageExecutor` needs a concrete ``IngestionRequest`` (source path, topics,
      synchronization tolerance) that is per-invocation input, never part of a resolved
      configuration -- it is what the ``ingest`` command's own flags build. A caller that
      wants either stage to run inside :func:`~contextmap.runtime.pipeline.run_plan`
      still injects its own :class:`~contextmap.runtime.ingestion_service.
      IngestionStageExecutor` explicitly (a second instance, under ``stage_id=
      "pose_ingestion"``, for the auxiliary one); the ordinary canonical path is to run
      ``contextmap ingest`` first and feed its published artifact to ``run``/``stage`` as a
      provided or selected input.
    - ``visual_perception`` is composed only when all four of its variation points
      (``region_discovery``, ``dense_features``, ``region_features``,
      ``semantic_interpretation``) are genuinely selected and available -- a partially
      configured capability never gets a partial executor (#507).
    - ``point_representation`` has no real executor yet (its backend is GPU/model
      dependent): it stays absent, exactly as before.
    - ``semantic_fusion`` is composed only when the selected accumulation backend is the
      one :class:`~contextmap.runtime.executors.SemanticFusionExecutor` actually runs
      (``baseline-evidence-accumulation-v1``); the quality-aware accumulation backend has
      no executor yet, so it is left out rather than run through the wrong policy.
    - ``semantic_mapping`` is composed only when the caller supplies both ``semantic_map_id``
      (identity of the persistent semantic map this run's entities belong to) and
      ``code_digest`` (digest of the code that produced the run): neither is a configuration
      value or derivable from ``effective``, and :class:`~contextmap.runtime.executors.
      SemanticMappingExecutor`'s writer refuses an empty ``code_digest``, so this function
      never invents one. Without both, ``semantic_mapping`` stays absent and its artifact must
      be supplied (``provided``/``selections``) for ``entity_resolution`` to consume, exactly
      as before this parameter existed.
    - ``entity_resolution`` always evaluates the required geometry channel; every other
      channel (``semantic``, ``temporal``, ``appearance``, ``representation``) is evaluated
      only when its own component is selected, and is ``None`` -- never a default policy --
      when it is not.
    - ``spatial_relations`` always applies the required frame conventions, candidate policy
      and geometry summary; the geometric and contact predicate evaluators run only when
      their own component is selected.
    - ``context_map`` has no variation point of its own, so composing it never fails once the
      preset declares it. Its ``up_axis`` comes from the same ``spatial_relations``
      ``FrameConventions`` this function already composed above, when available -- never
      re-derived independently -- and stays unknown otherwise.

    Args:
        effective: The resolved configuration.
        providers: Model runtimes or clients for backends without a bundled loader; see
            :func:`compose`.
        environ: Environment to read secrets from; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
        on_provider_override: Called with a component identity whenever ``providers``
            overrides a ``resources.providers`` target ``effective`` also declares for it;
            see :func:`compose`.
        semantic_map_id: Identity of the persistent semantic map ``semantic_mapping``'s
            entities belong to. Required, together with ``code_digest``, for this function to
            compose ``semantic_mapping``; see above.
        code_digest: Digest of the code producing the run, exactly as the caller supplies it --
            this function never computes one from a source tree. Required, together with
            ``semantic_map_id``, for this function to compose ``semantic_mapping``; see above.
        code_version: Code revision that produced the run, threaded into every composed
            executor that records one. ``None`` leaves it unset, exactly as before this
            parameter existed -- never fabricated from a source tree.

    Returns:
        One executor per stage that could genuinely be composed from ``effective``. Never
        raises: a stage this cannot build for any reason (incomplete selection, a rejected
        parameter, a missing module or secret, or an unresolvable declared provider target)
        is simply absent from the result, one stage at a time, so one broken stage never
        costs the others their real executor.
    """
    from contextmap.entity_resolution import MatchEvidenceBuilder
    from contextmap.runtime.executors import (
        ContextMapExecutor,
        EntityResolutionExecutor,
        GeometricMappingExecutor,
        SemanticFusionExecutor,
        SemanticMappingExecutor,
        SensorAssociationExecutor,
        SpatialRelationsExecutor,
        StateEstimationExecutor,
        VisualPerceptionExecutor,
    )
    from contextmap.semantic_fusion import BaselineAccumulationPolicy
    from contextmap.semantic_mapping import EntityMaterializationPolicy

    def _compose_stage(stage_id: str) -> ComposedRuntime | None:
        try:
            return compose(
                effective,
                stages=[stage_id],
                providers=providers,
                environ=environ,
                module_available=module_available,
                on_provider_override=on_provider_override,
            )
        except (ConfigurationError, CompositionError):
            # Seleção incompleta, estágio desabilitado, ou backend selecionado que rejeita seus
            # próprios parâmetros ou módulo/segredo ausente: ausência honesta, nunca um erro que
            # aborte a composição dos outros estágios. O preflight já relata "sem executor".
            return None

    executors: dict[str, StageExecutor] = {}

    visual_perception = _compose_stage("visual_perception")
    if visual_perception is not None:
        assert visual_perception.region_discovery is not None
        assert visual_perception.dense_features is not None
        assert visual_perception.region_features is not None
        assert visual_perception.semantic_interpreter is not None
        executors["visual_perception"] = VisualPerceptionExecutor(
            region_discovery=visual_perception.region_discovery,
            dense_features=visual_perception.dense_features,
            region_features=visual_perception.region_features,
            semantic_interpreter=visual_perception.semantic_interpreter,
        )

    state_estimation = _compose_stage("state_estimation")
    if state_estimation is not None:
        assert state_estimation.state_estimator is not None
        executors["state_estimation"] = StateEstimationExecutor(
            state_estimation.state_estimator,
            allow_ground_truth_trajectory=(
                effective.config.policies.trajectory_mode == "allow_ground_truth"
            ),
        )

    geometric_mapping = _compose_stage("geometric_mapping")
    if geometric_mapping is not None:
        assert geometric_mapping.geometric_mapping_pose_lookup is not None
        assert geometric_mapping.motion_correction is not None
        executors["geometric_mapping"] = GeometricMappingExecutor(
            pose_lookup=geometric_mapping.geometric_mapping_pose_lookup,
            motion_correction=geometric_mapping.motion_correction,
            code_version=code_version,
        )

    sensor_association = _compose_stage("sensor_association")
    if sensor_association is not None:
        assert sensor_association.occlusion_policy is not None
        assert sensor_association.association_tolerances is not None
        assert sensor_association.association_pose_policy is not None
        assert sensor_association.association_candidates is not None
        executors["sensor_association"] = SensorAssociationExecutor(
            candidates=sensor_association.association_candidates,
            occlusion=sensor_association.occlusion_policy,
            tolerances=sensor_association.association_tolerances,
            pose_policy=sensor_association.association_pose_policy,
            code_version=code_version,
        )

    semantic_fusion = _compose_stage("semantic_fusion")
    if semantic_fusion is not None:
        assert semantic_fusion.support_policy is not None
        assert semantic_fusion.accumulation_policy is not None
        if isinstance(semantic_fusion.accumulation_policy, BaselineAccumulationPolicy):
            executors["semantic_fusion"] = SemanticFusionExecutor(
                support_policy=semantic_fusion.support_policy,
                accumulation_policy=semantic_fusion.accumulation_policy,
                code_version=code_version,
            )

    if semantic_map_id is not None and code_digest is not None:
        semantic_mapping = _compose_stage("semantic_mapping")
        if semantic_mapping is not None:
            assert semantic_mapping.semantic_mapping_geometry_summary is not None
            executors["semantic_mapping"] = SemanticMappingExecutor(
                policy=EntityMaterializationPolicy(
                    geometry=semantic_mapping.semantic_mapping_geometry_summary
                ),
                semantic_map_id=semantic_map_id,
                code_digest=code_digest,
                code_version=code_version,
            )

    entity_resolution = _compose_stage("entity_resolution")
    if entity_resolution is not None:
        assert entity_resolution.entity_retrieval_policy is not None
        assert entity_resolution.entity_comparison_channels is not None
        assert entity_resolution.entity_resolution_policy is not None
        executors["entity_resolution"] = EntityResolutionExecutor(
            retrieval=entity_resolution.entity_retrieval_policy,
            builder=MatchEvidenceBuilder(entity_resolution.entity_comparison_channels),
            resolution=entity_resolution.entity_resolution_policy,
            code_version=code_version,
        )

    spatial_relations = _compose_stage("spatial_relations")
    if spatial_relations is not None:
        assert spatial_relations.spatial_relations_policies is not None
        executors["spatial_relations"] = SpatialRelationsExecutor(
            policies=spatial_relations.spatial_relations_policies,
            code_version=code_version,
        )

    context_map = _compose_stage("context_map")
    if context_map is not None:
        # context_map has no variation point of its own, so composing it never fails once the
        # preset declares it. Its up_axis comes from the same FrameConventions spatial_relations
        # already composed above, when available -- never re-derived independently -- and stays
        # None otherwise (MapFrame allows that).
        up_axis = (
            spatial_relations.spatial_relations_policies.frame_conventions.up_axis
            if spatial_relations is not None and spatial_relations.spatial_relations_policies
            else None
        )
        executors["context_map"] = ContextMapExecutor(up_axis=up_axis, code_version=code_version)

    return executors
