from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    KNOWN_CAPABILITIES,
    BackendProvenance,
    BoundingBox2D,
    DenseFeatureMap,
    DenseFeatureSampling,
    FeatureId,
    FeatureResolutionEnhancement,
    FeatureResolutionEnhancementError,
    FeatureResolutionEnhancementProvenance,
    FeatureScope,
    PipelineConfigError,
    PipelinePreset,
    Region2D,
    RegionId,
    StageBackendFactory,
    StageSpec,
    VisualFeature,
    enhance_feature_resolution,
    execute_stage_graph,
    pool_region_feature,
    resolve_pipeline,
)

_SOURCE_PROVENANCE = BackendProvenance(
    backend_id="fake-dino",
    capability="feature_extractor",
    provider="fake",
    model="synthetic-dense",
    version="1",
    configuration_fingerprint="sha256:dino-config",
)
_ENHANCEMENT_PROVENANCE = BackendProvenance(
    backend_id="fake-resolution-enhancer",
    capability="feature_resolution_enhancement",
    provider="fake",
    model="synthetic-upsampler",
    version="1",
    configuration_fingerprint="sha256:enhancement-config",
)
_REGION_PROVENANCE = BackendProvenance(
    backend_id="fake-region",
    capability="region_discovery",
    provider="fake",
    model="synthetic-region",
    version="1",
)


def _source_map() -> DenseFeatureMap:
    return DenseFeatureMap(
        feature=VisualFeature(
            feature_id=FeatureId("dense-native-0001"),
            scope=FeatureScope.DENSE,
            embedding_space_id="sha256:source-space",
            shape=(2, 2, 2),
            dtype="float32",
            normalization="l2",
            payload_reference="frame-0001/dense-native-0001.npy",
            provenance=_SOURCE_PROVENANCE,
        ),
        sampling=DenseFeatureSampling(
            grid_width=2,
            grid_height=2,
            source_image_width=8,
            source_image_height=4,
            origin_x=0.0,
            origin_y=0.0,
            stride_x=4.0,
            stride_y=2.0,
            support_width=4.0,
            support_height=2.0,
            coordinate_transform_id="prepared-to-native-grid-v1",
        ),
        source_artifact_id="dino-run-0001",
    )


def _enhancement_lineage(
    source: DenseFeatureMap, *, output_embedding_space_id: str | None = None
) -> FeatureResolutionEnhancementProvenance:
    return FeatureResolutionEnhancementProvenance(
        source_feature_id=source.feature.feature_id,
        source_artifact_id=source.source_artifact_id,
        source_payload_reference=source.feature.payload_reference,
        source_embedding_space_id=source.feature.embedding_space_id,
        source_dtype=source.feature.dtype,
        source_normalization=source.feature.normalization,
        source_coordinate_transform_id=source.sampling.coordinate_transform_id,
        backend=_ENHANCEMENT_PROVENANCE,
        input_grid_size=(source.sampling.grid_width, source.sampling.grid_height),
        output_grid_size=(4, 4),
        source_image_size=(
            source.sampling.source_image_width,
            source.sampling.source_image_height,
        ),
        output_embedding_space_id=(output_embedding_space_id or source.feature.embedding_space_id),
        device="cpu",
        precision="float32",
        duration_seconds=0.25,
        peak_memory_bytes=4096,
    )


def _enhanced_map(
    source: DenseFeatureMap,
    *,
    output_embedding_space_id: str | None = None,
    lineage: FeatureResolutionEnhancementProvenance | None = None,
) -> DenseFeatureMap:
    embedding_space_id = output_embedding_space_id or source.feature.embedding_space_id
    return DenseFeatureMap(
        feature=VisualFeature(
            feature_id=FeatureId("dense-enhanced-0001"),
            scope=FeatureScope.DENSE,
            embedding_space_id=embedding_space_id,
            shape=(4, 4, 2),
            dtype="float32",
            normalization="l2",
            payload_reference="frame-0001/dense-enhanced-0001.npy",
            provenance=_ENHANCEMENT_PROVENANCE,
        ),
        sampling=DenseFeatureSampling(
            grid_width=4,
            grid_height=4,
            source_image_width=8,
            source_image_height=4,
            origin_x=0.0,
            origin_y=0.0,
            stride_x=2.0,
            stride_y=1.0,
            support_width=2.0,
            support_height=1.0,
            coordinate_transform_id="prepared-to-enhanced-grid-v1",
        ),
        source_artifact_id=source.source_artifact_id,
        enhancement=lineage
        or _enhancement_lineage(source, output_embedding_space_id=output_embedding_space_id),
    )


class _FakeEnhancer:
    def __init__(self, *, mutate: dict[str, Any] | None = None) -> None:
        self._mutate = mutate or {}
        self.calls: list[DenseFeatureMap] = []

    def backend_provenance(self) -> BackendProvenance:
        return _ENHANCEMENT_PROVENANCE

    def enhance(self, source: DenseFeatureMap) -> DenseFeatureMap:
        self.calls.append(source)
        output = _enhanced_map(source)
        if not self._mutate:
            return output
        if "lineage" in self._mutate:
            return replace(output, enhancement=self._mutate["lineage"])
        return replace(output, **self._mutate)


def test_enhancement_port_produces_a_separate_auditable_dense_map() -> None:
    source = _source_map()
    enhancer = _FakeEnhancer()
    source_before = source

    output = enhance_feature_resolution(source, enhancer=enhancer)

    assert isinstance(enhancer, FeatureResolutionEnhancement)
    assert enhancer.calls == [source]
    assert source == source_before
    assert output is not source
    assert output.feature.feature_id != source.feature.feature_id
    assert output.feature.payload_reference != source.feature.payload_reference
    assert output.source_artifact_id == source.source_artifact_id
    assert output.feature.embedding_space_id == source.feature.embedding_space_id
    assert output.sampling.grid_width == 4
    assert output.sampling.grid_height == 4
    assert output.enhancement is not None
    assert output.enhancement.source_artifact_id == "dino-run-0001"
    assert output.enhancement.backend == _ENHANCEMENT_PROVENANCE
    assert output.enhancement.device == "cpu"
    assert output.enhancement.precision == "float32"
    assert output.enhancement.duration_seconds == 0.25
    assert output.enhancement.peak_memory_bytes == 4096


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("feature_id", "feature identity"),
        ("payload_reference", "payload reference"),
    ],
)
def test_same_artifact_still_requires_distinct_feature_and_payload(
    mutation: str, message: str
) -> None:
    source = _source_map()
    output = _enhanced_map(source)
    if mutation == "feature_id":
        feature = replace(output.feature, feature_id=source.feature.feature_id)
    else:
        feature = replace(output.feature, payload_reference=source.feature.payload_reference)
    invalid = replace(output, feature=feature)

    class _InvalidEnhancer(_FakeEnhancer):
        def enhance(self, source: DenseFeatureMap) -> DenseFeatureMap:
            return invalid

    with pytest.raises(FeatureResolutionEnhancementError, match=message):
        enhance_feature_resolution(source, enhancer=_InvalidEnhancer())


def test_changed_representation_space_is_allowed_only_when_explicit() -> None:
    source = _source_map()
    distinct_space = "sha256:enhanced-space"
    output = _enhanced_map(source, output_embedding_space_id=distinct_space)

    class _DistinctSpaceEnhancer(_FakeEnhancer):
        def enhance(self, source: DenseFeatureMap) -> DenseFeatureMap:
            return output

    validated = enhance_feature_resolution(source, enhancer=_DistinctSpaceEnhancer())
    assert validated.feature.embedding_space_id == distinct_space
    assert validated.enhancement is not None
    assert validated.enhancement.source_embedding_space_id != distinct_space
    assert validated.enhancement.output_embedding_space_id == distinct_space


@pytest.mark.parametrize(
    "mutation",
    [
        "source_feature_id",
        "source_artifact_id",
        "source_payload_reference",
        "source_embedding_space_id",
        "source_dtype",
        "source_normalization",
        "source_coordinate_transform_id",
        "input_grid_size",
        "source_image_size",
        "backend",
    ],
)
def test_enhancement_rejects_incorrect_source_or_backend_lineage(mutation: str) -> None:
    source = _source_map()
    lineage = _enhancement_lineage(source)
    replacements: dict[str, Any] = {
        "source_feature_id": FeatureId("other-feature"),
        "source_artifact_id": "other-artifact",
        "source_payload_reference": "other.npy",
        "source_embedding_space_id": "sha256:other-space",
        "source_dtype": "float16",
        "source_normalization": "none",
        "source_coordinate_transform_id": "other-transform",
        "input_grid_size": (1, 1),
        "source_image_size": (16, 8),
        "backend": replace(_ENHANCEMENT_PROVENANCE, model="other-model"),
    }
    changed = replace(lineage, **{mutation: replacements[mutation]})
    enhancer = _FakeEnhancer(mutate={"lineage": changed})

    with pytest.raises(FeatureResolutionEnhancementError, match="lineage"):
        enhance_feature_resolution(source, enhancer=enhancer)


def test_enhancement_rejects_non_increasing_or_inconsistent_output() -> None:
    source = _source_map()
    valid = _enhanced_map(source)

    same_grid_feature = replace(valid.feature, shape=(2, 2, 2))
    same_grid_sampling = replace(
        valid.sampling,
        grid_width=2,
        grid_height=2,
        stride_x=4.0,
        stride_y=2.0,
        support_width=4.0,
        support_height=2.0,
    )
    assert valid.enhancement is not None
    same_grid_lineage = replace(valid.enhancement, output_grid_size=(2, 2))
    same_grid = replace(
        valid,
        feature=same_grid_feature,
        sampling=same_grid_sampling,
        enhancement=same_grid_lineage,
    )

    class _StaticEnhancer(_FakeEnhancer):
        def __init__(self, output: DenseFeatureMap) -> None:
            super().__init__()
            self.output = output

        def enhance(self, source: DenseFeatureMap) -> DenseFeatureMap:
            return self.output

    with pytest.raises(FeatureResolutionEnhancementError, match="increase"):
        enhance_feature_resolution(source, enhancer=_StaticEnhancer(same_grid))

    wrong_image = replace(
        valid,
        sampling=replace(valid.sampling, source_image_width=16),
    )
    with pytest.raises(FeatureResolutionEnhancementError, match="source image"):
        enhance_feature_resolution(source, enhancer=_StaticEnhancer(wrong_image))


def test_native_and_enhanced_maps_use_the_same_region_pooling_consumer() -> None:
    source = _source_map()
    enhanced = enhance_feature_resolution(source, enhancer=_FakeEnhancer())
    native_payload = np.zeros(source.feature.shape, dtype=np.float32)
    native_payload[..., 0] = 1.0
    enhanced_payload = np.zeros(enhanced.feature.shape, dtype=np.float32)
    enhanced_payload[..., 0] = 1.0
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=1, y=1, width=5, height=2),
        provenance=_REGION_PROVENANCE,
    )

    native_region = pool_region_feature(native_payload, dense_map=source, region=region)
    enhanced_region = pool_region_feature(enhanced_payload, dense_map=enhanced, region=region)

    np.testing.assert_array_equal(native_region.vector, enhanced_region.vector)
    assert native_region.embedding_space_id == enhanced_region.embedding_space_id
    assert (
        native_region.provenance.coordinate_transform_id
        != enhanced_region.provenance.coordinate_transform_id
    )


def test_optional_stage_is_absent_from_canonical_preset_and_runs_when_selected() -> None:
    assert "feature_resolution_enhancement" in KNOWN_CAPABILITIES
    assert all(
        stage.capability != "feature_resolution_enhancement" for stage in CANONICAL_PRESET_V1.stages
    )

    source = _source_map()
    enhancer = _FakeEnhancer()
    preset = PipelinePreset(
        preset_id="experimental/dense-resolution-enhancement-v1",
        stages=(
            StageSpec(
                stage_id="native_dense_feature",
                capability="dense_feature_map_source",
            ),
            StageSpec(
                stage_id="feature_resolution_enhancement",
                capability="feature_resolution_enhancement",
                inputs={"dense_map": "native_dense_feature"},
                backend_id="fake-resolution-enhancer",
                optional=True,
                parameters={"device": "cpu", "precision": "float32"},
            ),
        ),
    )
    factory_calls: list[dict[str, object]] = []

    def factory(parameters: Mapping[str, object]) -> _FakeEnhancer:
        factory_calls.append(dict(parameters))
        return enhancer

    factories: dict[str, StageBackendFactory] = {
        "feature_resolution_enhancement": factory,
    }
    resolved = resolve_pipeline(preset, backend_factories=factories)
    outcomes = execute_stage_graph(resolved.build_stage_graph({"native_dense_feature": source}))

    assert factory_calls == [{"device": "cpu", "precision": "float32"}]
    assert outcomes[-1].output == _enhanced_map(source)
    assert resolved.backend_provenance()["feature_resolution_enhancement"] == (
        _ENHANCEMENT_PROVENANCE
    )
    assert resolved.configuration_digest().startswith("sha256:")


def test_selected_optional_stage_requires_its_backend_factory() -> None:
    preset = PipelinePreset(
        preset_id="experimental/dense-resolution-enhancement-v1",
        stages=(
            StageSpec(stage_id="native", capability="dense_feature_map_source"),
            StageSpec(
                stage_id="enhance",
                capability="feature_resolution_enhancement",
                inputs={"dense_map": "native"},
                backend_id="missing-enhancer",
                optional=True,
            ),
        ),
    )

    with pytest.raises(PipelineConfigError, match="missing backend factory"):
        resolve_pipeline(preset, backend_factories={})
