"""Region grounding (#569) and its SAM2 refinement (#568), composed and executed from config.

Grounding is an optional variation point of the ``visual_perception`` stage. Its queries
are a reserved parameter group (``query_set``) that becomes one explicit request per image
and query; they never reach the backend configuration or its fingerprint.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.runtime.composition import (
    RegionGroundingPlan,
    RuntimeProvider,
    compose,
    compose_executors,
)
from contextmap.runtime.errors import BackendConfigurationError, BackendUnavailableError
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingQuery,
    GroundingQueryPolicy,
    GroundingTask,
    PreparedImage,
    Region2D,
    RegionGroundingCapabilities,
    RegionGroundingExecution,
    RegionGroundingRequest,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.visual_perception.backends.locateanything import (
    CATEGORY_DETECTION_POLICY,
    POINTING_POLICY,
    LocateAnythingConfig,
    LocateAnythingGeneration,
    LocateAnythingRegionGrounding,
    TransformersLocateAnythingRuntime,
)

GROUNDING = "visual_perception.region_grounding"
REGION = "visual_perception.region_discovery"
INTERPRETER = "visual_perception.semantic_interpretation"
REVISION = "c" * 40
CATEGORIES = {
    "task": "category_detection",
    "policy_id": CATEGORY_DETECTION_POLICY,
    "geometry": "box",
    "categories": ["chair", "table"],
}
POINTING = {
    "task": "phrase_grounding",
    "policy_id": POINTING_POLICY,
    "geometry": "point",
    "text": "the door handle",
}


def _document(*queries: dict[str, Any], **parameters: Any) -> dict[str, Any]:
    document = selected_document()
    block: dict[str, Any] = {
        "model": "nvidia/LocateAnything-3B",
        "revision": REVISION,
        "dtype": "bfloat16",
        "generation_mode": "hybrid",
        "max_new_tokens": 2048,
        "temperature": 0.0,
        "text_attention": "sdpa",
        "vision_attention": "sdpa",
        "query_set": {"queries": list(queries or (CATEGORIES,))},
    }
    block.update(parameters)
    document["components"]["visual_perception"]["region_grounding"] = {
        "backend": "locateanything",
        "locateanything": block,
    }
    return document


class _FakeLocateAnythingRuntime:
    def generate(self, *, image: Any, prompt: str, config: Any) -> LocateAnythingGeneration:
        return LocateAnythingGeneration(text="<|im_end|>")


def _providers(**extra: RuntimeProvider) -> dict[str, RuntimeProvider]:
    providers: dict[str, RuntimeProvider] = {
        REGION: lambda _config, _secrets: object(),
        INTERPRETER: lambda _config, _secrets: object(),
    }
    providers.update(extra)
    return providers


def _compose(tmp_path: Path, document: dict[str, Any], **options: Any) -> Any:
    options.setdefault("module_available", lambda _name: True)
    options.setdefault("environ", {})
    options.setdefault("providers", _providers())
    return compose(effective_from(tmp_path, document), stages=["visual_perception"], **options)


def test_region_grounding_is_optional_and_absent_unless_selected(tmp_path: Path) -> None:
    composed = _compose(tmp_path, selected_document())

    assert composed.region_grounding is None
    assert composed.region_discovery is not None


def test_locateanything_is_selectable_with_explicit_queries(tmp_path: Path) -> None:
    composed = _compose(tmp_path, _document(CATEGORIES, POINTING))

    plan = composed.region_grounding
    assert isinstance(plan, RegionGroundingPlan)
    assert plan.queries == (
        GroundingQuery(
            task=GroundingTask.CATEGORY_DETECTION,
            policy_id=CATEGORY_DETECTION_POLICY,
            geometry=GroundingGeometry.BOX,
            categories=("chair", "table"),
        ),
        GroundingQuery(
            task=GroundingTask.PHRASE_GROUNDING,
            policy_id=POINTING_POLICY,
            geometry=GroundingGeometry.POINT,
            text="the door handle",
        ),
    )


def test_the_bundled_runtime_is_built_per_run_and_loads_nothing_at_composition(
    tmp_path: Path,
) -> None:
    plan = _compose(tmp_path, _document()).region_grounding

    backend = plan.factory(tmp_path)

    assert isinstance(backend, LocateAnythingRegionGrounding)
    assert isinstance(backend._runtime, TransformersLocateAnythingRuntime)  # type: ignore[attr-defined]
    assert backend._runtime._loaded is None  # type: ignore[attr-defined]


def test_the_resolved_device_reaches_the_backend_configuration(tmp_path: Path) -> None:
    backend = _compose(tmp_path, _document()).region_grounding.factory(tmp_path)

    assert backend._config.device == "cpu"  # type: ignore[attr-defined]


def test_a_provider_runtime_replaces_the_bundled_loader(tmp_path: Path) -> None:
    calls: list[Any] = []

    def provide(config: Any, secrets: Any) -> object:
        calls.append(config)
        return _FakeLocateAnythingRuntime()

    composed = _compose(tmp_path, _document(), providers=_providers(**{GROUNDING: provide}))
    backend = composed.region_grounding.factory(tmp_path)

    assert isinstance(backend._runtime, _FakeLocateAnythingRuntime)  # type: ignore[attr-defined]
    assert isinstance(calls[0], LocateAnythingConfig)


def test_the_query_set_is_required(tmp_path: Path) -> None:
    document = _document()
    del document["components"]["visual_perception"]["region_grounding"]["locateanything"][
        "query_set"
    ]

    with pytest.raises(BackendConfigurationError, match="query_set"):
        _compose(tmp_path, document)


@pytest.mark.parametrize(
    "query",
    [
        {**CATEGORIES, "geometry": "point"},
        {**CATEGORIES, "policy_id": "locateanything.category-detection/9"},
        {**CATEGORIES, "categories": ["chair</c>table"]},
        {**CATEGORIES, "categories": []},
    ],
    ids=["unsupported-geometry", "unknown-policy", "separator", "no-category"],
)
def test_an_unservable_query_fails_at_composition_before_any_model(
    tmp_path: Path, query: dict[str, Any]
) -> None:
    with pytest.raises(BackendConfigurationError, match="query_set"):
        _compose(tmp_path, _document(query))


def test_a_repeated_query_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigurationError, match="unique"):
        _compose(tmp_path, _document(CATEGORIES, CATEGORIES))


def test_invalid_inference_settings_fail_at_composition(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigurationError, match="revision"):
        _compose(tmp_path, _document(revision="main"))
    with pytest.raises(BackendConfigurationError, match="text_attention"):
        _compose(tmp_path, _document(text_attention="la_flash"))


def test_a_missing_sdk_is_reported_before_any_model(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailableError, match="torch"):
        _compose(tmp_path, _document(), module_available=lambda name: name != "torch")


def test_queries_change_the_run_identity_but_not_the_backend_fingerprint(
    tmp_path: Path,
) -> None:
    first_document = _document(CATEGORIES)
    second_document = _document({**CATEGORIES, "categories": ["table", "chair"]})

    first = effective_from(tmp_path, first_document)
    first_backend = _compose(tmp_path, first_document).region_grounding.factory(tmp_path)
    second = effective_from(tmp_path, second_document)
    second_backend = _compose(tmp_path, second_document).region_grounding.factory(tmp_path)

    assert first.digest != second.digest
    assert (
        first_backend.backend_provenance().configuration_fingerprint
        == second_backend.backend_provenance().configuration_fingerprint
    )


def test_compose_executors_wires_grounding_into_visual_perception(tmp_path: Path) -> None:
    from contextmap.runtime.executors import VisualPerceptionExecutor

    executors = compose_executors(
        effective_from(tmp_path, _document()),
        providers=_providers(),
        module_available=lambda _name: True,
        environ={},
    )

    executor = executors["visual_perception"]
    assert isinstance(executor, VisualPerceptionExecutor)
    assert isinstance(executor._region_grounding, RegionGroundingPlan)  # type: ignore[attr-defined]


# --- execution -----------------------------------------------------------------------------


class _FakeGrounding:
    """Answers every query with one box over the whole image, recording each request."""

    def __init__(self) -> None:
        self.requests: list[RegionGroundingRequest] = []

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="sha256:fake-grounding",
        )

    def capabilities(self) -> RegionGroundingCapabilities:
        return RegionGroundingCapabilities(
            query_policies=(
                GroundingQueryPolicy(
                    policy_id="fake.phrase/1",
                    task=GroundingTask.PHRASE_GROUNDING,
                    geometry=GroundingGeometry.BOX,
                ),
            )
        )

    def ground(self, request: RegionGroundingRequest) -> RegionGroundingExecution:
        self.requests.append(request)
        return RegionGroundingExecution(
            request=request,
            provenance=self.backend_provenance(),
            rendered_prompt=f"find {request.query.text}",
            raw_response="<box><0><0><1000><1000></box>",
            outputs=(
                GroundingOutput(
                    output_index=0,
                    native_text="<box><0><0><1000><1000></box>",
                    label=request.query.text,
                    box=BoundingBox2D(
                        x=0.0, y=0.0, width=request.image.width, height=request.image.height
                    ),
                ),
            ),
            diagnostics=GroundingDiagnostics(latency_ms=1.0),
            effective_configuration=MappingProxyType({"backend": "fake"}),
        )


def _run_visual_perception(
    tmp_path: Path, *, region_refinement: Any = None
) -> tuple[Any, list[Path]]:
    """Run the real executor over two frames with fake backends; return reader and roots.

    Orchestration only: every backend is a fake, everything else (sequence, prepared-image
    materialization, stage graph, persisted run) is real.
    """
    from contextmap.ingestion import (
        FrameId,
        ImageEncoding,
        ImageObservation,
        SensorId,
        SequenceArtifactId,
        SequenceArtifactWriter,
        SourceObservationId,
        SourceProvenance,
    )
    from contextmap.runtime import ArtifactRef
    from contextmap.runtime.executors import VisualPerceptionExecutor, inventory_digest
    from contextmap.runtime.pipeline import StageRequest
    from contextmap.shared import SourceTimestamp
    from contextmap.visual_perception import (
        SEMANTIC_PROMPT_TEMPLATES,
        FeatureScope,
        PerceptionRunReader,
        SemanticBackendDiagnostics,
        SemanticConfidencePolicy,
        SemanticInferenceProvenance,
        SemanticInterpretationExecution,
        VisualFeature,
        parse_semantic_response,
        render_semantic_prompt,
    )

    class _NoRegions:
        def backend_provenance(self) -> BackendProvenance:
            return BackendProvenance(
                backend_id="fake",
                capability="region_discovery",
                provider="fake",
                model="f",
                version="0",
            )

        def discover(self, image: PreparedImage) -> list[Region2D]:
            return []

    class _NoFeatures:
        def __init__(self, scope: FeatureScope) -> None:
            self._scope = scope

        def backend_provenance(self) -> BackendProvenance:
            return BackendProvenance(
                backend_id="fake_features",
                capability="feature_extractor",
                provider="fake",
                model="f",
                version="0",
            )

        def required_scope(self) -> FeatureScope:
            return self._scope

        def extract(
            self, image: PreparedImage, regions: Sequence[Region2D] = ()
        ) -> Sequence[VisualFeature]:
            return []

    class _AbstainingInterpreter:
        def backend_provenance(self) -> BackendProvenance:
            return BackendProvenance(
                backend_id="fake_semantic",
                capability="semantic_interpreter",
                provider="fake",
                model="f",
                version="0",
            )

        def interpret(self, request: Any) -> SemanticInterpretationExecution:
            import json

            template = SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id]
            rendered = render_semantic_prompt(
                request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
            )
            raw = json.dumps({"abstained": True, "claims": [], "scene_context": None})
            provenance = SemanticInferenceProvenance(
                backend=self.backend_provenance(),
                task_identity="fake",
                prompt_template_id=request.prompt_template_id,
                output_schema_version=request.requested_output_schema,
            )
            return SemanticInterpretationExecution(
                request=request,
                rendered_prompt=rendered,
                raw_response=raw,
                parsed=parse_semantic_response(
                    raw,
                    request,
                    provenance,
                    confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
                ),
                diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
                effective_configuration={"backend": "fake"},
            )

    grounding = _FakeGrounding()
    roots: list[Path] = []

    def factory(prepared_image_root: Path) -> _FakeGrounding:
        roots.append(prepared_image_root)
        return grounding

    queries = tuple(
        GroundingQuery(
            task=GroundingTask.PHRASE_GROUNDING,
            policy_id="fake.phrase/1",
            geometry=GroundingGeometry.BOX,
            text=text,
        )
        for text in ("the red chair", "the table")
    )
    executor = VisualPerceptionExecutor(
        region_discovery=_NoRegions(),  # type: ignore[arg-type]
        dense_features=lambda _scope: _NoFeatures(FeatureScope.DENSE),  # type: ignore[arg-type]
        region_features=lambda _scope: _NoFeatures(FeatureScope.REGION),  # type: ignore[arg-type]
        semantic_interpreter=_AbstainingInterpreter(),  # type: ignore[arg-type]
        semantic_request_policy=SemanticRequestPolicy.from_prompt_policy(
            SemanticPromptPolicy(scene="scene/v1", region="region/v1"),
            views=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
        ),
        region_grounding=RegionGroundingPlan(queries=queries, factory=factory),  # type: ignore[arg-type]
        region_refinement=region_refinement,
    )

    workspace = tmp_path / "ws"
    ingestion_dir = workspace / "corridor-02" / "run-0001" / "ingestion"
    with SequenceArtifactWriter(
        output_dir=ingestion_dir,
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-0001"),
    ) as writer:
        for index in range(2):
            writer.add_observation(
                ImageObservation(
                    observation_id=SourceObservationId(f"frame-{index:04d}"),
                    sensor_id=SensorId("camera_1"),
                    frame_id=FrameId("camera_1_optical"),
                    timestamp=SourceTimestamp(seconds=index, nanoseconds=0, clock_id="fixture"),
                    provenance=SourceProvenance(source_type="fixture", source_path="fixtures"),
                    width=2,
                    height=2,
                    encoding=ImageEncoding.RGB8,
                    data=bytes([index, 20, 30] * 4),
                )
            )
        manifest = writer.finalize()
    output_dir = workspace / "corridor-02" / "run-0001" / "visual_perception"
    request = StageRequest(
        stage_id="visual_perception",
        inputs={
            "sequence": (
                ArtifactRef(
                    stage_id="ingestion",
                    contract="SequenceArtifact",
                    artifact_id=str(manifest.artifact_id),
                    content_hash=inventory_digest(manifest.file_inventory),
                    location="corridor-02/run-0001/ingestion",
                ),
            )
        },
        components={},
        config_digest="sha256:test",
        output_dir=output_dir,
        workspace=workspace,
    )

    executor.execute(request)
    return PerceptionRunReader(output_dir), roots


def test_the_executor_turns_each_query_into_an_explicit_persisted_request(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")  # Pillow materializa a imagem preparada; não é dependência base.

    reader, roots = _run_visual_perception(tmp_path)

    assert reader.verify_integrity() == []
    assert "region_grounding" in reader.manifest.enabled_capabilities
    assert len(roots) == 1  # um backend (e um carregamento de modelo) por run
    executions = reader.list_region_groundings()
    assert len(executions) == 4
    assert {(str(e.request.source_observation_id), e.request.query.text) for e in executions} == {
        (f"frame-{index:04d}", text)
        for index in range(2)
        for text in ("the red chair", "the table")
    }
    assert all(e.request.configuration_fingerprint == "sha256:fake-grounding" for e in executions)
    for result in reader.list_results():
        grounded = [r for r in result.regions if r.provenance.capability == "region_grounding"]
        assert len(grounded) == 2


# --- refinement (#568) ------------------------------------------------------------------------

REFINEMENT = "visual_perception.region_refinement"


def _refined_document(**sam2: Any) -> dict[str, Any]:
    document = _document()
    document["components"]["visual_perception"]["region_refinement"] = {
        "backend": "sam2",
        "sam2": {"checkpoint": "sam2.1_hiera_large", "model_version": "2.1", **sam2},
    }
    return document


class _FakePromptRuntime:
    """Answers every prompt with a full-image mask, like a SAM2 that sees one object."""

    def predict_prompts(self, **kwargs: Any) -> tuple[Any, ...]:
        from contextmap.visual_perception.backends.sam2 import Sam2PromptedMask

        pixels = kwargs["width"] * kwargs["height"]
        return tuple(
            Sam2PromptedMask(mask=(True,) * pixels, predicted_iou=0.9) for _ in kwargs["prompts"]
        )


def _refinement_providers(calls: list[Any] | None = None) -> dict[str, RuntimeProvider]:
    def provide(config: Any, secrets: Any) -> object:
        (calls if calls is not None else []).append(config)
        return _FakePromptRuntime()

    return _providers(**{REFINEMENT: provide})


def test_refinement_is_optional_and_absent_unless_selected(tmp_path: Path) -> None:
    assert _compose(tmp_path, _document()).region_refinement is None


def test_sam2_refinement_is_selectable_with_a_provided_prompt_runtime(tmp_path: Path) -> None:
    from contextmap.visual_perception.backends.sam2 import (
        Sam2PromptRefinement,
        Sam2RefinementConfig,
    )

    calls: list[Any] = []
    composed = _compose(tmp_path, _refined_document(), providers=_refinement_providers(calls))

    refiner = composed.region_refinement(tmp_path)

    assert isinstance(refiner, Sam2PromptRefinement)
    assert isinstance(calls[0], Sam2RefinementConfig)
    assert calls[0].device == "cpu"
    assert len(calls) == 1  # o provider é pedido uma vez na composição, não por run


def test_refinement_needs_a_provided_sam2_runtime(tmp_path: Path) -> None:
    from contextmap.runtime.errors import BackendRuntimeMissingError

    with pytest.raises(BackendRuntimeMissingError):
        _compose(tmp_path, _refined_document())


def test_refinement_without_grounding_is_an_explicit_configuration_error(tmp_path: Path) -> None:
    from contextmap.runtime import ConfigurationError

    document = _refined_document()
    del document["components"]["visual_perception"]["region_grounding"]

    with pytest.raises(ConfigurationError, match="region_grounding"):
        _compose(tmp_path, document, providers=_refinement_providers())


def test_grounding_and_refinement_are_ablated_independently(tmp_path: Path) -> None:
    grounding_only = _compose(tmp_path, _document())
    refined = _compose(tmp_path, _refined_document(), providers=_refinement_providers())
    other_refiner = _compose(
        tmp_path, _refined_document(mask_threshold=0.5), providers=_refinement_providers()
    )

    def grounding_fingerprint(composed: Any) -> Any:
        return composed.region_grounding.factory(tmp_path).backend_provenance()

    assert grounding_fingerprint(grounding_only) == grounding_fingerprint(refined)
    assert grounding_fingerprint(refined) == grounding_fingerprint(other_refiner)
    assert (
        refined.region_refinement(tmp_path).backend_provenance().configuration_fingerprint
        != other_refiner.region_refinement(tmp_path).backend_provenance().configuration_fingerprint
    )


def test_compose_executors_wires_refinement_into_visual_perception(tmp_path: Path) -> None:
    executors = compose_executors(
        effective_from(tmp_path, _refined_document()),
        providers=_refinement_providers(),
        module_available=lambda _name: True,
        environ={},
    )

    assert executors["visual_perception"]._region_refinement is not None  # type: ignore[attr-defined]


def test_the_executor_refines_every_grounding_proposal_and_persists_it(tmp_path: Path) -> None:
    pytest.importorskip("PIL")  # Pillow materializa a imagem preparada; não é dependência base.
    from contextmap.visual_perception.backends.sam2 import (
        Sam2PromptRefinement,
        Sam2RefinementConfig,
    )

    config = Sam2RefinementConfig(checkpoint="sam2.1_hiera_large")
    roots: list[Path] = []

    def refinement(prepared_image_root: Path) -> Sam2PromptRefinement:
        roots.append(prepared_image_root)
        return Sam2PromptRefinement(
            config=config, runtime=_FakePromptRuntime(), prepared_image_root=prepared_image_root
        )

    reader, _ = _run_visual_perception(tmp_path, region_refinement=refinement)

    assert reader.verify_integrity() == []
    assert "region_refinement" in reader.manifest.enabled_capabilities
    assert len(roots) == 1
    refinements = reader.list_region_refinements()
    assert len(refinements) == 2  # uma requisição por imagem, com todas as propostas dela
    groundings = {str(item.request_id): item for item in reader.list_region_groundings()}
    for execution in refinements:
        assert len(execution.request.prompts) == 2
        assert {str(p.grounding_request_id) for p in execution.request.prompts} <= set(groundings)
    for result in reader.list_results():
        capabilities = [region.provenance.capability for region in result.regions]
        assert capabilities.count("region_grounding") == 2
        refined = [r for r in result.regions if r.provenance.capability == "region_refinement"]
        assert len(refined) == 2
        for region in refined:
            assert region.mask_reference is not None
            assert (
                reader.mask_store().load(result.source_observation_id, region.region_id).area == 4
            )
