"""Deterministic dense feature maps for the association tests."""

from __future__ import annotations

import dataclasses

from perception_builders import make_feature, make_result

from contextmap.visual_perception import (
    BackendProvenance,
    DenseFeatureMap,
    DenseFeatureSampling,
    FeatureResolutionEnhancementProvenance,
    FeatureScope,
    PerceptionResult,
    VisualFeature,
)

IMAGE = (640, 480)


def make_sampling(
    grid: tuple[int, int] = (40, 30),
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    stride: tuple[float, float] = (16.0, 16.0),
    support: tuple[float, float] = (16.0, 16.0),
    image: tuple[int, int] = IMAGE,
    transform_id: str = "native-grid-v1",
) -> DenseFeatureSampling:
    return DenseFeatureSampling(
        grid_width=grid[0],
        grid_height=grid[1],
        source_image_width=image[0],
        source_image_height=image[1],
        origin_x=origin[0],
        origin_y=origin[1],
        stride_x=stride[0],
        stride_y=stride[1],
        support_width=support[0],
        support_height=support[1],
        coordinate_transform_id=transform_id,
    )


def make_dense_feature(
    sampling: DenseFeatureSampling,
    *,
    feature_id: str = "dense-native",
    channels: int = 2,
    space: str = "dinov2:b14",
    normalization: str | None = None,
) -> VisualFeature:
    base = make_feature(feature_id, FeatureScope.DENSE, space=space)
    return dataclasses.replace(
        base,
        shape=(sampling.grid_height, sampling.grid_width, channels),
        normalization=normalization,
    )


def make_dense_map(
    sampling: DenseFeatureSampling | None = None,
    *,
    enhancement: FeatureResolutionEnhancementProvenance | None = None,
    **feature_options: object,
) -> DenseFeatureMap:
    sampling = sampling if sampling is not None else make_sampling()
    return DenseFeatureMap(
        feature=make_dense_feature(sampling, **feature_options),  # type: ignore[arg-type]
        sampling=sampling,
        source_artifact_id="perception-artifact-0001",
        enhancement=enhancement,
    )


def make_dense_result(*dense_maps: DenseFeatureMap) -> PerceptionResult:
    return make_result([], features=[dense_map.feature for dense_map in dense_maps])


def make_enhancement(
    native: DenseFeatureMap, output: DenseFeatureSampling
) -> FeatureResolutionEnhancementProvenance:
    return FeatureResolutionEnhancementProvenance(
        source_feature_id=native.feature.feature_id,
        source_artifact_id=native.source_artifact_id,
        source_payload_reference=native.feature.payload_reference,
        source_embedding_space_id=native.feature.embedding_space_id,
        source_dtype=native.feature.dtype,
        source_normalization=native.feature.normalization,
        source_coordinate_transform_id=native.sampling.coordinate_transform_id,
        backend=BackendProvenance(
            backend_id="fake_enhancer",
            capability="feature_resolution_enhancement",
            provider="fake",
            model="fake",
            version="0",
        ),
        input_grid_size=(native.sampling.grid_width, native.sampling.grid_height),
        output_grid_size=(output.grid_width, output.grid_height),
        source_image_size=(output.source_image_width, output.source_image_height),
        output_embedding_space_id=native.feature.embedding_space_id,
        device="cpu",
        precision="float32",
        duration_seconds=0.1,
        peak_memory_bytes=1024,
    )
