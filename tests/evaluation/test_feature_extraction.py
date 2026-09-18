from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.evaluation import (
    FeatureEvaluationContext,
    FeatureEvaluationError,
    FeatureEvaluationReport,
    assert_repeatable_feature_outputs,
    compare_feature_map_resolutions,
    encode_feature_evaluation_report,
    evaluate_feature_payload,
)
from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    DenseFeatureMap,
    DenseFeatureSampling,
    EmbeddingSpace,
    FeatureId,
    FeatureScope,
    FeatureStoreReader,
    FeatureStoreWriter,
    Region2D,
    RegionId,
    VisualFeature,
    embedding_space_fingerprint,
    pool_region_feature,
    write_feature_index,
)

_PROVENANCE = BackendProvenance(
    backend_id="fake-dino",
    capability="feature_extractor",
    provider="fake",
    model="synthetic-dense",
    version="1",
    configuration_fingerprint="sha256:fake-config",
)
_REGION_PROVENANCE = BackendProvenance(
    backend_id="fake-region",
    capability="region_discovery",
    provider="fake",
    model="synthetic-mask",
    version="1",
)
_SPACE = EmbeddingSpace(
    family="fake-dino",
    model="synthetic-dense",
    version="1",
    checkpoint="sha256:model",
    layer="patch-embeddings",
    dimension=2,
    normalization="l2",
)
_SPACE_ID = embedding_space_fingerprint(_SPACE)


def _context() -> FeatureEvaluationContext:
    return FeatureEvaluationContext(
        evaluation_id="feature-eval-0001",
        source_observation_id=SourceObservationId("frame-0001"),
        source_image_content_hash="sha256:image",
        source_model_artifact_id="dino-artifact-0001",
        region_set_id="regions-0001",
        downstream_configuration_fingerprint="sha256:downstream",
    )


def _dense_map(
    *,
    grid_width: int = 2,
    grid_height: int = 2,
    transform_id: str = "native-grid-v1",
    feature_id: str = "dense-0001",
    embedding_space_id: str = _SPACE_ID,
) -> DenseFeatureMap:
    feature = VisualFeature(
        feature_id=FeatureId(feature_id),
        scope=FeatureScope.DENSE,
        embedding_space_id=embedding_space_id,
        shape=(grid_height, grid_width, 2),
        dtype="float32",
        normalization="l2",
        payload_reference=f"frame-0001/{feature_id}.npy",
        provenance=_PROVENANCE,
    )
    return DenseFeatureMap(
        feature=feature,
        sampling=DenseFeatureSampling(
            grid_width=grid_width,
            grid_height=grid_height,
            source_image_width=8,
            source_image_height=4,
            origin_x=0.0,
            origin_y=0.0,
            stride_x=8.0 / grid_width,
            stride_y=4.0 / grid_height,
            support_width=8.0 / grid_width,
            support_height=4.0 / grid_height,
            coordinate_transform_id=transform_id,
        ),
        source_artifact_id="perception-run-0001",
    )


def _unit_payload(*, grid_width: int = 2, grid_height: int = 2) -> np.ndarray:
    payload = np.zeros((grid_height, grid_width, 2), dtype=np.float32)
    payload[..., 0] = 1.0
    return payload


def _evaluate(
    dense_map: DenseFeatureMap,
    payload: np.ndarray,
    *,
    context: FeatureEvaluationContext | None = None,
    duration_seconds: float = 0.25,
) -> FeatureEvaluationReport:
    return evaluate_feature_payload(
        payload,
        feature=dense_map.feature,
        embedding_space=_SPACE,
        context=context or _context(),
        dense_map=dense_map,
        duration_seconds=duration_seconds,
        peak_memory_bytes=4096,
        processed_items=2,
    )


def test_evaluation_report_separates_correctness_spatial_and_cost_metrics() -> None:
    dense_map = _dense_map()
    report = evaluate_feature_payload(
        _unit_payload(),
        feature=dense_map.feature,
        embedding_space=_SPACE,
        context=_context(),
        dense_map=dense_map,
        duration_seconds=0.25,
        peak_memory_bytes=4096,
        processed_items=2,
    )

    assert report.numerical.element_count == 8
    assert report.numerical.all_finite is True
    assert report.numerical.vector_norm_min == pytest.approx(1.0)
    assert report.numerical.vector_norm_max == pytest.approx(1.0)
    assert report.spatial is not None
    assert report.spatial.source_image_size == (8, 4)
    assert report.spatial.grid_size == (2, 2)
    assert report.spatial.coordinate_transform_id == "native-grid-v1"
    assert report.cost.duration_seconds == 0.25
    assert report.cost.peak_memory_bytes == 4096
    assert report.cost.payload_size_bytes == _unit_payload().nbytes
    assert report.cost.throughput_items_per_second == 8.0

    encoded = encode_feature_evaluation_report(report)
    assert encoded["embedding_space"]["checkpoint"] == "sha256:model"
    assert encoded["backend"]["configuration_fingerprint"] == "sha256:fake-config"
    assert encoded["context"]["source_model_artifact_id"] == "dino-artifact-0001"
    assert encoded["spatial"]["coordinate_transform_id"] == "native-grid-v1"
    assert set(encoded) >= {"numerical", "spatial", "cost"}
    assert "quality_score" not in encoded


@pytest.mark.parametrize(
    ("scope", "shape", "region_id"),
    [
        (FeatureScope.GLOBAL, (2,), None),
        (FeatureScope.REGION, (2,), RegionId("region-0001")),
    ],
)
def test_global_and_region_features_use_the_same_numerical_validation(
    scope: FeatureScope, shape: tuple[int, ...], region_id: RegionId | None
) -> None:
    feature = VisualFeature(
        feature_id=FeatureId(f"{scope.value}-0001"),
        scope=scope,
        embedding_space_id=_SPACE_ID,
        shape=shape,
        dtype="float32",
        normalization="l2",
        payload_reference=f"{scope.value}.npy",
        provenance=_PROVENANCE,
        region_id=region_id,
    )

    report = evaluate_feature_payload(
        np.array([0.6, 0.8], dtype=np.float32),
        feature=feature,
        embedding_space=_SPACE,
        context=_context(),
        duration_seconds=0.1,
        peak_memory_bytes=128,
    )

    assert report.scope is scope
    assert report.spatial is None
    assert report.numerical.vector_norm_max == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("payload", "feature_change", "space", "message"),
    [
        (_unit_payload()[:, :1], {}, _SPACE, "shape"),
        (_unit_payload().astype(np.float64), {}, _SPACE, "dtype"),
        (
            np.full((2, 2, 2), np.nan, dtype=np.float32),
            {},
            _SPACE,
            "finite",
        ),
        (
            np.full((2, 2, 2), 0.5, dtype=np.float32),
            {},
            _SPACE,
            "l2 normalization",
        ),
        (
            _unit_payload(),
            {"embedding_space_id": "sha256:other"},
            _SPACE,
            "embedding space",
        ),
        (
            _unit_payload(),
            {"embedding_space_id": embedding_space_fingerprint(replace(_SPACE, dimension=3))},
            replace(_SPACE, dimension=3),
            "embedding dimension",
        ),
    ],
)
def test_evaluation_rejects_numerical_and_compatibility_regressions(
    payload: np.ndarray,
    feature_change: dict[str, str],
    space: EmbeddingSpace,
    message: str,
) -> None:
    dense_map = _dense_map()
    feature = dense_map.feature
    if feature_change:
        feature = replace(feature, embedding_space_id=feature_change["embedding_space_id"])
    evaluated_map = replace(dense_map, feature=feature)

    with pytest.raises(FeatureEvaluationError, match=message):
        evaluate_feature_payload(
            payload,
            feature=feature,
            embedding_space=space,
            context=_context(),
            dense_map=evaluated_map,
            duration_seconds=0.1,
            peak_memory_bytes=128,
        )


def test_repeatability_compares_output_metadata_and_payload_but_not_cost() -> None:
    first_map = _dense_map(feature_id="dense-run-a")
    second_map = replace(
        _dense_map(feature_id="dense-run-b"),
        source_artifact_id="perception-run-0002",
    )
    first = _evaluate(first_map, _unit_payload(), duration_seconds=0.25)
    second = _evaluate(second_map, _unit_payload(), duration_seconds=0.5)

    assert_repeatable_feature_outputs(first, second)

    changed = _evaluate(
        second_map,
        np.array(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        duration_seconds=0.5,
    )
    with pytest.raises(FeatureEvaluationError, match="payload"):
        assert_repeatable_feature_outputs(first, changed)


def test_native_and_enhanced_maps_share_pooling_and_comparison_contracts() -> None:
    native_map = _dense_map()
    enhanced_map = _dense_map(
        grid_width=4,
        grid_height=4,
        transform_id="enhanced-grid-v1",
        feature_id="dense-enhanced-0001",
    )
    native_payload = _unit_payload()
    enhanced_payload = _unit_payload(grid_width=4, grid_height=4)
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=1, y=1, width=5, height=2),
        provenance=_REGION_PROVENANCE,
        mask_reference="masks/region-0001.npy",
    )
    mask = np.array(
        [[True, True, False, False, False], [True, True, False, False, False]],
        dtype=np.bool_,
    )

    native_region = pool_region_feature(
        native_payload, dense_map=native_map, region=region, mask=mask
    )
    enhanced_region = pool_region_feature(
        enhanced_payload, dense_map=enhanced_map, region=region, mask=mask
    )
    np.testing.assert_array_equal(native_region.vector, enhanced_region.vector)

    comparison = compare_feature_map_resolutions(
        native=_evaluate(native_map, native_payload),
        enhanced=_evaluate(enhanced_map, enhanced_payload),
    )
    assert comparison.context == _context()
    assert comparison.native.spatial is not None
    assert comparison.enhanced.spatial is not None
    assert comparison.native.spatial.grid_size == (2, 2)
    assert comparison.enhanced.spatial.grid_size == (4, 4)
    assert comparison.native.embedding_space == comparison.enhanced.embedding_space


def test_resolution_comparison_rejects_changed_control_or_embedding_space() -> None:
    native = _evaluate(_dense_map(), _unit_payload())
    changed_context = replace(_context(), region_set_id="regions-0002")
    enhanced = _evaluate(
        _dense_map(
            grid_width=4,
            grid_height=4,
            transform_id="enhanced-grid-v1",
            feature_id="dense-enhanced-0001",
        ),
        _unit_payload(grid_width=4, grid_height=4),
        context=changed_context,
    )
    with pytest.raises(FeatureEvaluationError, match="experimental context"):
        compare_feature_map_resolutions(native=native, enhanced=enhanced)

    other_space = replace(_SPACE, checkpoint="sha256:other-model")
    other_map = _dense_map(
        grid_width=4,
        grid_height=4,
        transform_id="enhanced-grid-v1",
        feature_id="dense-enhanced-0002",
        embedding_space_id=embedding_space_fingerprint(other_space),
    )
    other_report = evaluate_feature_payload(
        _unit_payload(grid_width=4, grid_height=4),
        feature=other_map.feature,
        embedding_space=other_space,
        context=_context(),
        dense_map=other_map,
        duration_seconds=0.1,
        peak_memory_bytes=128,
    )
    with pytest.raises(FeatureEvaluationError, match="embedding space"):
        compare_feature_map_resolutions(native=native, enhanced=other_report)


def test_aspect_ratio_crop_and_boundary_transforms_have_known_grid_support() -> None:
    dense_map = _dense_map(grid_width=4, grid_height=2)
    region = Region2D(
        region_id=RegionId("region-boundary"),
        bounding_box=BoundingBox2D(x=7, y=3, width=2, height=2),
        provenance=_REGION_PROVENANCE,
    )

    result = pool_region_feature(
        _unit_payload(grid_width=4, grid_height=2),
        dense_map=dense_map,
        region=region,
    )

    assert result.diagnostics.cell_row_range == (1, 2)
    assert result.diagnostics.cell_col_range == (3, 4)
    assert result.diagnostics.total_weight == 1
    assert result.diagnostics.coverage_fraction == 0.25


def test_persisted_payload_can_be_loaded_lazily_and_evaluated(tmp_path: Path) -> None:
    dense_map = _dense_map()
    payload = _unit_payload()
    writer = FeatureStoreWriter(tmp_path)
    writer.write(dense_map.feature, SourceObservationId("frame-0001"), payload)
    write_feature_index(tmp_path, writer.entries())

    reader = FeatureStoreReader.open(tmp_path)
    assert reader.entry(dense_map.feature.feature_id).shape == dense_map.feature.shape
    loaded = reader.load(dense_map.feature.feature_id)
    report = _evaluate(dense_map, loaded)

    assert report.payload_hash.startswith("sha256:")
    assert report.cost.payload_size_bytes == payload.nbytes
