"""A SigLIP2 dense map reaches the geometry through the same backend-neutral path as DINO's."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from perception_builders import RUN_ID, make_result
from projection_builders import make_prepared_image, scene_frame

from contextmap.ingestion import SourceObservationId
from contextmap.sensor_association.dense_sampling import (
    InterpolationPolicy,
    sample_dense_features,
)
from contextmap.sensor_association.visibility import OcclusionPolicy, resolve_visibility
from contextmap.visual_perception import (
    DenseFeatureMap,
    FeatureScope,
    PreparedImage,
    VisualFeature,
)
from contextmap.visual_perception.backends.dinov3 import (
    DinoV3Config,
    DinoV3DenseFeatureBackend,
    DinoV3NativeOutput,
)
from contextmap.visual_perception.backends.siglip2 import (
    Siglip2Config,
    Siglip2FeatureBackend,
    Siglip2NativeOutput,
)

POLICY = OcclusionPolicy(
    cell_size_px=4,
    neighborhood_radius_cells=2,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)
_REVISION = "e" * 40
_INPUT, _PATCH = 512, 16
_GRID = _INPUT // _PATCH


def _coordinate_grid() -> NDArray[np.float32]:
    """The value of cell ``(row, column)`` is ``(row, column)``."""
    rows, columns = np.meshgrid(np.arange(_GRID), np.arange(_GRID), indexing="ij")
    return np.stack((rows, columns), axis=-1).astype(np.float32)


class _Sink:
    def add_feature_payload(
        self, feature: VisualFeature, source_observation_id: SourceObservationId, array: Any
    ) -> None:
        """Payload persistence is not what these tests exercise."""


class _Runtime:
    def __init__(self, output: Any) -> None:
        self._output = output

    def infer(self, image: PreparedImage) -> Any:
        return self._output


def _siglip2_map(image: PreparedImage) -> tuple[DenseFeatureMap, NDArray[Any]]:
    backend = Siglip2FeatureBackend(
        config=Siglip2Config(
            checkpoint="google/siglip2-so400m-patch16-512",
            revision=_REVISION,
            scope=FeatureScope.DENSE,
            input_size=_INPUT,
        ),
        run_id=RUN_ID,
        feature_stage_id="dense_feature_extraction",
        source_artifact_id="perception-artifact-0001",
        payload_sink=_Sink(),
        runtime=_Runtime(
            Siglip2NativeOutput(
                array=_coordinate_grid(),
                model_input_width=_INPUT,
                model_input_height=_INPUT,
                patch_width=_PATCH,
                patch_height=_PATCH,
            )
        ),
    )
    extraction = backend.extract_dense(image)
    return extraction.dense_map, extraction.array


def _dinov3_map(image: PreparedImage) -> tuple[DenseFeatureMap, NDArray[Any]]:
    backend = DinoV3DenseFeatureBackend(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m",
            revision=_REVISION,
            input_width=_INPUT,
            input_height=_INPUT,
        ),
        run_id=RUN_ID,
        feature_stage_id="dense_feature_extraction",
        source_artifact_id="perception-artifact-0001",
        payload_sink=_Sink(),
        runtime=_Runtime(
            DinoV3NativeOutput(
                array=_coordinate_grid(),
                model_input_width=_INPUT,
                model_input_height=_INPUT,
                patch_width=_PATCH,
                patch_height=_PATCH,
                register_token_count=4,
            )
        ),
    )
    extraction = backend.extract_dense(image)
    return extraction.dense_map, extraction.array


@pytest.mark.parametrize(
    ("build", "backend_id"),
    [(_siglip2_map, "siglip2_huggingface"), (_dinov3_map, "dinov3_huggingface")],
    ids=["siglip2", "dinov3"],
)
def test_the_same_native_grid_samples_the_same_cells_whatever_the_backend(
    build: Callable[[PreparedImage], tuple[DenseFeatureMap, NDArray[Any]]], backend_id: str
) -> None:
    prepared = make_prepared_image()
    dense_map, array = build(prepared)
    frame = scene_frame((100, 100, 3.0), (5, 5, 3.0), (634, 474, 3.0), prepared=prepared)

    samples = sample_dense_features(
        resolve_visibility(frame, POLICY),
        make_result([], features=[dense_map.feature]),
        dense_map,
        interpolation=InterpolationPolicy.NEAREST,
    )

    # 640x480 sobre uma grade 32x32: células de 20x15 px, lidas só pela geometria declarada.
    assert samples.sampled.tolist() == [True, True, True]
    assert samples.gather(array).tolist() == [[6.0, 5.0], [0.0, 0.0], [31.0, 31.0]]
    assert samples.provenance.extractor.backend_id == backend_id
    assert samples.provenance.embedding_space_id == dense_map.feature.embedding_space_id
    assert samples.provenance.coordinate_transform_id == dense_map.sampling.coordinate_transform_id
