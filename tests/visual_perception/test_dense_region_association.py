from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    DenseFeatureMap,
    DenseFeatureSampling,
    EmptyRegionSupportError,
    FeatureId,
    FeatureScope,
    Region2D,
    RegionAssociationError,
    RegionId,
    VisualFeature,
    map_box_to_grid_cells,
    pool_region_feature,
)

_SOURCE_PROVENANCE = BackendProvenance(
    backend_id="fake-dense",
    capability="feature_extractor",
    provider="fake",
    model="synthetic-grid",
    version="1",
)
_REGION_PROVENANCE = BackendProvenance(
    backend_id="fake-region",
    capability="region_discovery",
    provider="fake",
    model="synthetic-mask",
    version="1",
)


def _dense_map(*, transform_id: str = "prepared-to-grid-v1") -> DenseFeatureMap:
    feature = VisualFeature(
        feature_id=FeatureId("dense-0001"),
        scope=FeatureScope.DENSE,
        embedding_space_id="space:synthetic",
        shape=(4, 4, 1),
        dtype="float32",
        normalization="none",
        payload_reference="frame-0001/dense-0001.npy",
        provenance=_SOURCE_PROVENANCE,
    )
    return DenseFeatureMap(
        feature=feature,
        sampling=DenseFeatureSampling(
            grid_width=4,
            grid_height=4,
            source_image_width=8,
            source_image_height=8,
            origin_x=0.0,
            origin_y=0.0,
            stride_x=2.0,
            stride_y=2.0,
            support_width=2.0,
            support_height=2.0,
            coordinate_transform_id=transform_id,
        ),
        source_artifact_id="perception-run-0001",
    )


def _region(*, x: int = 2, y: int = 2, width: int = 4, height: int = 4) -> Region2D:
    return Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=x, y=y, width=width, height=height),
        provenance=_REGION_PROVENANCE,
        mask_reference="masks/region-0001.npy",
    )


def _array() -> np.ndarray:
    return np.arange(16, dtype=np.float32).reshape(4, 4, 1)


def test_dense_feature_map_requires_dense_three_dimensional_feature() -> None:
    feature = VisualFeature(
        feature_id=FeatureId("global-0001"),
        scope=FeatureScope.GLOBAL,
        embedding_space_id="space:synthetic",
        shape=(1,),
        dtype="float32",
        payload_reference="global.npy",
        provenance=_SOURCE_PROVENANCE,
    )

    with pytest.raises(ValueError, match="DENSE"):
        DenseFeatureMap(
            feature=feature,
            sampling=_dense_map().sampling,
            source_artifact_id="perception-run-0001",
        )


def test_dense_feature_map_validates_grid_shape_against_feature_metadata() -> None:
    feature = _dense_map().feature
    sampling = DenseFeatureSampling(
        grid_width=3,
        grid_height=4,
        source_image_width=8,
        source_image_height=8,
        origin_x=0.0,
        origin_y=0.0,
        stride_x=2.0,
        stride_y=2.0,
        support_width=2.0,
        support_height=2.0,
        coordinate_transform_id="prepared-to-grid-v1",
    )

    with pytest.raises(ValueError, match="feature shape"):
        DenseFeatureMap(
            feature=feature, sampling=sampling, source_artifact_id="perception-run-0001"
        )


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (BoundingBox2D(x=2, y=2, width=4, height=4), (1, 3, 1, 3)),
        (BoundingBox2D(x=0, y=0, width=1, height=1), (0, 1, 0, 1)),
        (BoundingBox2D(x=7, y=7, width=2, height=2), (3, 4, 3, 4)),
        (BoundingBox2D(x=8, y=8, width=1, height=1), (4, 4, 4, 4)),
    ],
)
def test_map_box_to_grid_cells_has_known_support(
    box: BoundingBox2D, expected: tuple[int, int, int, int]
) -> None:
    assert map_box_to_grid_cells(box, dense_map=_dense_map()) == expected


def test_map_box_to_grid_cells_respects_explicit_origin_stride_and_support() -> None:
    native = _dense_map()
    shifted_overlapping = DenseFeatureMap(
        feature=native.feature,
        sampling=DenseFeatureSampling(
            grid_width=4,
            grid_height=4,
            source_image_width=8,
            source_image_height=8,
            origin_x=1.0,
            origin_y=1.0,
            stride_x=2.0,
            stride_y=2.0,
            support_width=4.0,
            support_height=4.0,
            coordinate_transform_id="shifted-overlapping-v1",
        ),
        source_artifact_id="enhanced-run-0001",
    )

    assert map_box_to_grid_cells(
        BoundingBox2D(x=4, y=4, width=1, height=1), dense_map=shifted_overlapping
    ) == (0, 2, 0, 2)


def test_mask_aware_pooling_uses_only_supported_cells() -> None:
    mask = np.array(
        [
            [True, True, True, True],
            [True, True, True, True],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=np.bool_,
    )

    result = pool_region_feature(_array(), dense_map=_dense_map(), region=_region(), mask=mask)

    np.testing.assert_array_equal(result.vector, np.array([5.5], dtype=np.float32))
    assert result.embedding_space_id == "space:synthetic"
    assert result.diagnostics.cell_row_range == (1, 3)
    assert result.diagnostics.cell_col_range == (1, 3)
    assert result.diagnostics.contributing_cell_count == 2
    assert result.diagnostics.total_weight == 8
    assert result.diagnostics.coverage_fraction == 1.0


def test_box_only_pooling_uses_the_same_weighted_mean_policy() -> None:
    result = pool_region_feature(_array(), dense_map=_dense_map(), region=_region())

    np.testing.assert_array_equal(result.vector, np.array([7.5], dtype=np.float32))
    assert result.diagnostics.contributing_cell_count == 4
    assert result.diagnostics.total_weight == 16
    assert result.provenance.pooling_policy == "mask_weighted_mean_preserve_l2_v2"


def test_pooling_renormalizes_l2_vectors_to_preserve_embedding_space() -> None:
    dense_map = _dense_map()
    l2_feature = replace(
        dense_map.feature,
        shape=(4, 4, 2),
        normalization="l2",
    )
    l2_map = replace(dense_map, feature=l2_feature)
    array = np.zeros((4, 4, 2), dtype=np.float32)
    array[:, :2, 0] = 1.0
    array[:, 2:, 1] = 1.0

    result = pool_region_feature(array, dense_map=l2_map, region=_region())

    np.testing.assert_allclose(
        result.vector,
        np.array([2**-0.5, 2**-0.5], dtype=np.float32),
        rtol=1e-6,
    )
    assert np.linalg.norm(result.vector) == pytest.approx(1.0)
    assert result.normalization == "l2"
    assert result.provenance.pooling_policy == "mask_weighted_mean_preserve_l2_v2"


def test_pooling_rejects_zero_vector_when_l2_normalization_must_be_preserved() -> None:
    dense_map = _dense_map()
    l2_map = replace(
        dense_map,
        feature=replace(dense_map.feature, shape=(4, 4, 2), normalization="l2"),
    )
    array = np.zeros((4, 4, 2), dtype=np.float32)
    array[:, :2, 0] = 1.0
    array[:, 2:, 0] = -1.0

    with pytest.raises(RegionAssociationError, match="cannot preserve l2 normalization"):
        pool_region_feature(array, dense_map=l2_map, region=_region())


def test_tiny_boundary_mask_is_clipped_to_source_image() -> None:
    mask = np.ones((2, 2), dtype=np.bool_)

    result = pool_region_feature(
        _array(), dense_map=_dense_map(), region=_region(x=7, y=7, width=2, height=2), mask=mask
    )

    np.testing.assert_array_equal(result.vector, np.array([15.0], dtype=np.float32))
    assert result.diagnostics.total_weight == 1
    assert result.diagnostics.coverage_fraction == 0.25


def test_empty_mask_has_explicit_failure() -> None:
    with pytest.raises(EmptyRegionSupportError, match="no supported pixels"):
        pool_region_feature(
            _array(),
            dense_map=_dense_map(),
            region=_region(),
            mask=np.zeros((4, 4), dtype=np.bool_),
        )


def test_mask_shape_and_dtype_are_validated() -> None:
    with pytest.raises(RegionAssociationError, match="mask shape"):
        pool_region_feature(
            _array(),
            dense_map=_dense_map(),
            region=_region(),
            mask=np.ones((8, 8), dtype=np.bool_),
        )

    with pytest.raises(RegionAssociationError, match="boolean"):
        pool_region_feature(
            _array(),
            dense_map=_dense_map(),
            region=_region(),
            mask=np.ones((4, 4), dtype=np.uint8),
        )


def test_array_metadata_is_validated_before_pooling() -> None:
    with pytest.raises(RegionAssociationError, match="array shape"):
        pool_region_feature(
            np.zeros((2, 2, 1), dtype=np.float32), dense_map=_dense_map(), region=_region()
        )

    with pytest.raises(RegionAssociationError, match="array dtype"):
        pool_region_feature(_array().astype(np.float64), dense_map=_dense_map(), region=_region())


def test_provenance_identifies_source_region_policy_and_transform_deterministically() -> None:
    first = pool_region_feature(_array(), dense_map=_dense_map(), region=_region())
    second = pool_region_feature(_array(), dense_map=_dense_map(), region=_region())

    assert first.provenance == second.provenance
    assert first.provenance.source_feature_id == FeatureId("dense-0001")
    assert first.provenance.source_payload_reference == "frame-0001/dense-0001.npy"
    assert first.provenance.source_artifact_id == "perception-run-0001"
    assert first.provenance.region_id == RegionId("region-0001")
    assert first.provenance.coordinate_transform_id == "prepared-to-grid-v1"
    assert first.provenance.source_mask_reference == "masks/region-0001.npy"
    assert first.provenance.mask_content_hash is None
    assert first.provenance.configuration_fingerprint.startswith("sha256:")

    changed_transform = pool_region_feature(
        _array(), dense_map=_dense_map(transform_id="enhanced-to-grid-v2"), region=_region()
    )
    assert (
        first.provenance.configuration_fingerprint
        != changed_transform.provenance.configuration_fingerprint
    )

    masked = pool_region_feature(
        _array(), dense_map=_dense_map(), region=_region(), mask=np.ones((4, 4), dtype=np.bool_)
    )
    assert masked.provenance.mask_content_hash is not None
