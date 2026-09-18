from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import pytest
from fakes import (
    FakeDenseFeatureExtractor,
    FakeRegionDiscovery,
    FakeRegionFeatureExtractor,
    FakeSemanticInterpreter,
)

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    KNOWN_CAPABILITIES,
    BackendProvenance,
    BoundingBox2D,
    PerceptionResultId,
    PerceptionRunId,
    PipelineConfigError,
    PipelinePreset,
    PreparedImage,
    StageBackendFactory,
    StageSpec,
    assemble_perception_result,
    decode_pipeline_preset,
    encode_pipeline_preset,
    execute_stage_graph,
    resolve_pipeline,
)


def _prepared_image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0124"),
        payload_reference="debug/frame-0124/prepared.jpg",
        width=640,
        height=480,
    )


def _canonical_backend_factories(
    *, region_discovery: object | None = None, call_log: list[str] | None = None
) -> dict[str, StageBackendFactory]:
    def _record(stage_id: str, factory: Callable[[], object]) -> StageBackendFactory:
        def wrapped(_parameters: Mapping[str, object]) -> object:
            if call_log is not None:
                call_log.append(stage_id)
            return factory()

        return wrapped

    return {
        "region_discovery": _record(
            "region_discovery", lambda: region_discovery or FakeRegionDiscovery()
        ),
        "dense_feature_extraction": _record("dense_feature_extraction", FakeDenseFeatureExtractor),
        "region_feature_extraction": _record(
            "region_feature_extraction", FakeRegionFeatureExtractor
        ),
        "scene_interpretation": _record("scene_interpretation", FakeSemanticInterpreter),
        "region_interpretation": _record("region_interpretation", FakeSemanticInterpreter),
    }


def test_canonical_preset_resolves_and_produces_full_evidence() -> None:
    image = _prepared_image()
    resolved = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories()
    )
    stages = resolved.build_stage_graph({"image_preparation": image})
    outcomes = execute_stage_graph(stages)

    result = assemble_perception_result(
        result_id=PerceptionResultId("run-0001--frame-0124"),
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        outcomes=outcomes,
        region_stage_id="region_discovery",
        feature_stage_ids=("dense_feature_extraction", "region_feature_extraction"),
        claim_stage_ids=("region_interpretation",),
        scene_context_stage_id="scene_interpretation",
    )

    assert len(result.regions) == 1
    assert len(result.features) == 2
    assert len(result.claims) == 1
    assert result.scene_context is not None


def test_heavy_backends_are_not_constructed_until_validation_succeeds() -> None:
    call_log: list[str] = []
    invalid_preset = PipelinePreset(
        preset_id="broken/1",
        stages=(
            StageSpec(
                stage_id="region_discovery",
                capability="region_discovery",
                inputs={"image": "does_not_exist"},
                backend_id="region_discovery/canonical",
            ),
        ),
    )

    with pytest.raises(PipelineConfigError, match="unknown stage"):
        resolve_pipeline(
            invalid_preset, backend_factories=_canonical_backend_factories(call_log=call_log)
        )

    assert call_log == []


def test_backends_are_resolved_once_and_reused_across_many_observations() -> None:
    call_log: list[str] = []
    resolved = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories(call_log=call_log)
    )
    assert call_log.count("region_discovery") == 1

    for observation_id in ("frame-0001", "frame-0002", "frame-0003"):
        image = PreparedImage(
            source_observation_id=SourceObservationId(observation_id),
            payload_reference=f"debug/{observation_id}/prepared.jpg",
            width=640,
            height=480,
        )
        stages = resolved.build_stage_graph({"image_preparation": image})
        execute_stage_graph(stages)

    # The backend factory ran exactly once at resolve time, not once per
    # observation: build_stage_graph only ever reuses the already
    # resolved backend instance.
    assert call_log.count("region_discovery") == 1


def test_validate_rejects_duplicate_stage_id() -> None:
    preset = PipelinePreset(
        preset_id="broken/1",
        stages=(
            StageSpec(stage_id="a", capability="image_preparation"),
            StageSpec(stage_id="a", capability="image_preparation"),
        ),
    )
    with pytest.raises(PipelineConfigError, match="duplicate stage_id"):
        resolve_pipeline(preset, backend_factories={})


def test_validate_rejects_cycle() -> None:
    preset = PipelinePreset(
        preset_id="broken/1",
        stages=(
            StageSpec(
                stage_id="a",
                capability="region_discovery",
                inputs={"image": "b"},
                backend_id="fake",
            ),
            StageSpec(
                stage_id="b",
                capability="region_discovery",
                inputs={"image": "a"},
                backend_id="fake",
            ),
        ),
    )
    factories: dict[str, StageBackendFactory] = {
        "a": lambda _p: object(),
        "b": lambda _p: object(),
    }
    with pytest.raises(PipelineConfigError, match="cycle"):
        resolve_pipeline(preset, backend_factories=factories)


def test_validate_rejects_unknown_capability() -> None:
    preset = PipelinePreset(
        preset_id="broken/1",
        stages=(StageSpec(stage_id="a", capability="not_a_real_capability", backend_id="fake"),),
    )
    factories: dict[str, StageBackendFactory] = {"a": lambda _p: object()}
    with pytest.raises(PipelineConfigError, match="no capability adapter"):
        resolve_pipeline(preset, backend_factories=factories)


def test_optional_stage_can_be_inserted_without_touching_downstream_capability_code() -> None:
    """A non-canonical preset can route region_feature_extraction's 'regions'
    input through an extra optional stage instead of region_discovery
    directly — without editing this module's capability adapters at all.
    """

    def _passthrough_regions(image: PreparedImage) -> Sequence:
        return FakeRegionDiscovery(boxes=(BoundingBox2D(x=1, y=1, width=5, height=5),)).discover(
            image
        )

    class _FakeRegionRefinement:
        def backend_provenance(self) -> BackendProvenance:
            return BackendProvenance(
                backend_id="fake_region_refinement",
                capability="region_discovery",
                provider="fake",
                model="fake",
                version="0.1",
            )

        def discover(self, image: PreparedImage) -> Sequence:
            return _passthrough_regions(image)

    preset = PipelinePreset(
        preset_id="experimental/region-refinement",
        description="Inserts an optional region-refinement stage before feature extraction.",
        stages=(
            StageSpec(stage_id="image_preparation", capability="image_preparation"),
            StageSpec(
                stage_id="region_discovery",
                capability="region_discovery",
                inputs={"image": "image_preparation"},
                backend_id="region_discovery/canonical",
            ),
            StageSpec(
                stage_id="region_refinement",
                capability="region_discovery",
                inputs={"image": "image_preparation"},
                backend_id="region_refinement/experimental",
                optional=True,
            ),
            StageSpec(
                stage_id="region_feature_extraction",
                capability="feature_extractor",
                inputs={"image": "image_preparation", "regions": "region_refinement"},
                backend_id="region_feature_extractor/canonical",
            ),
        ),
    )

    resolved = resolve_pipeline(
        preset,
        backend_factories={
            "region_discovery": lambda _p: FakeRegionDiscovery(),
            "region_refinement": lambda _p: _FakeRegionRefinement(),
            "region_feature_extraction": lambda _p: FakeRegionFeatureExtractor(),
        },
    )
    stages = resolved.build_stage_graph({"image_preparation": _prepared_image()})
    outcomes = execute_stage_graph(stages)
    by_id = {outcome.stage_id: outcome for outcome in outcomes}

    region_features = by_id["region_feature_extraction"].output
    assert by_id["region_refinement"].output is not None
    assert isinstance(region_features, Sequence)
    assert len(region_features) == 1


def test_configuration_digest_is_stable_and_changes_with_backend_identity() -> None:
    resolved_a = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories()
    )
    resolved_b = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories()
    )
    assert resolved_a.configuration_digest() == resolved_b.configuration_digest()

    class _DifferentRegionDiscovery(FakeRegionDiscovery):
        def backend_provenance(self) -> BackendProvenance:
            return replace(super().backend_provenance(), backend_id="a_different_backend")

    resolved_c = resolve_pipeline(
        CANONICAL_PRESET_V1,
        backend_factories=_canonical_backend_factories(
            region_discovery=_DifferentRegionDiscovery()
        ),
    )
    assert resolved_c.configuration_digest() != resolved_a.configuration_digest()


def test_pipeline_preset_encode_decode_round_trip() -> None:
    decoded = decode_pipeline_preset(encode_pipeline_preset(CANONICAL_PRESET_V1))
    assert decoded == CANONICAL_PRESET_V1


def test_known_capabilities_matches_canonical_preset_backend_stages() -> None:
    used = {
        stage.capability for stage in CANONICAL_PRESET_V1.stages if stage.backend_id is not None
    }
    assert used <= KNOWN_CAPABILITIES
