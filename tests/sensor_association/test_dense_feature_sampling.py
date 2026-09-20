import dataclasses
import json
import math

import numpy as np
import pytest
from dense_builders import (
    make_dense_map,
    make_dense_result,
    make_enhancement,
    make_sampling,
)
from numpy.typing import NDArray
from perception_builders import make_result
from projection_builders import (
    artifact,
    make_prepared_image,
    map_point_for_pixel,
    project_frame,
    scene_frame,
)

from contextmap.ingestion import SourceObservationId
from contextmap.sensor_association.dense_sampling import (
    SAMPLING_POLICY_ID,
    DenseFeatureSamples,
    InterpolationPolicy,
    sample_dense_features,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.visibility import OcclusionPolicy, resolve_visibility
from contextmap.visual_perception import (
    DenseFeatureMap,
    DenseFeatureSampling,
    FeatureId,
    PerceptionResult,
    ResizeOperation,
)

POLICY = OcclusionPolicy(
    cell_size_px=4,
    neighborhood_radius_cells=2,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)


def _coordinate_field(dense_map: DenseFeatureMap) -> NDArray[np.float32]:
    """A payload whose value at cell ``(row, column)`` is ``(row, column)``: linear in the grid."""
    height, width, _ = dense_map.feature.shape
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack((rows, columns), axis=-1).astype(np.float32)


def _sample(
    frame: FrameProjection,
    dense_map: DenseFeatureMap,
    *,
    interpolation: InterpolationPolicy = InterpolationPolicy.NEAREST,
    result: PerceptionResult | None = None,
) -> DenseFeatureSamples:
    return sample_dense_features(
        resolve_visibility(frame, POLICY),
        result if result is not None else make_dense_result(dense_map),
        dense_map,
        interpolation=interpolation,
    )


# --- Nearest ----------------------------------------------------------------


def test_nearest_takes_the_cell_that_contains_the_prepared_pixel() -> None:
    dense_map = make_dense_map()
    frame = scene_frame((100, 100, 3.0), (5, 5, 3.0), (634, 474, 3.0))

    samples = _sample(frame, dense_map)

    assert samples.sampled.tolist() == [True, True, True]
    vectors = samples.gather(_coordinate_field(dense_map))
    # O valor de uma célula é (linha, coluna).
    assert vectors.tolist() == [[6.0, 6.0], [0.0, 0.0], [29.0, 39.0]]


def test_nearest_records_indices_and_never_the_vectors() -> None:
    dense_map = make_dense_map()
    samples = _sample(scene_frame((100, 100, 3.0), (300, 200, 3.0)), dense_map)

    assert samples.cell_rows.shape == (2, 1)
    assert samples.cell_cols.shape == (2, 1)
    assert samples.weights.tolist() == [[1.0], [1.0]]
    names = {field.name for field in dataclasses.fields(samples)}
    assert not names & {"vectors", "features", "array", "payload"}


def test_overlapping_supports_pick_the_cell_with_the_nearest_center() -> None:
    # Campos receptivos de 40 px a cada 16 px: vários suportes contêm o pixel.
    sampling = make_sampling(support=(40.0, 40.0), origin=(-12.0, -12.0), grid=(41, 31))
    dense_map = make_dense_map(sampling)
    pixels = [(100, 100), (333, 217), (15, 8), (620, 470)]
    frame = scene_frame(*[(u, v, 3.0) for u, v in pixels])

    samples = _sample(frame, dense_map)

    for index, (u, v) in enumerate(pixels):
        expected = _brute_force_nearest(sampling, u + 0.5, v + 0.5)
        assert (int(samples.cell_rows[index, 0]), int(samples.cell_cols[index, 0])) == expected


def _brute_force_nearest(
    sampling: DenseFeatureSampling, edge_x: float, edge_y: float
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for row in range(sampling.grid_height):
        for column in range(sampling.grid_width):
            center_x = sampling.origin_x + column * sampling.stride_x + sampling.support_width / 2
            center_y = sampling.origin_y + row * sampling.stride_y + sampling.support_height / 2
            distance = math.hypot(edge_x - center_x, edge_y - center_y)
            if best is None or distance < best[0]:
                best = (distance, row, column)
    assert best is not None
    return best[1], best[2]


def test_a_stride_larger_than_the_support_leaves_gaps_that_are_out_of_support() -> None:
    sampling = make_sampling(support=(8.0, 8.0))
    dense_map = make_dense_map(sampling)
    # Suporte da célula 0: [0, 8); o pixel 12 (borda 12.5) cai na lacuna até a célula 1 (16).
    frame = scene_frame((3, 3, 3.0), (12, 3, 3.0))

    samples = _sample(frame, dense_map)

    assert samples.sampled.tolist() == [True, False]
    assert samples.out_of_support_count == 1


def test_a_pixel_beyond_the_grid_is_out_of_support_not_clamped() -> None:
    # 45 células de 14 px cobrem 630 px de 640.
    sampling = make_sampling(grid=(45, 34), stride=(14.0, 14.0), support=(14.0, 14.0))
    dense_map = make_dense_map(sampling)
    frame = scene_frame((100, 100, 3.0), (635, 100, 3.0), (100, 478, 3.0))

    samples = _sample(frame, dense_map)

    assert samples.sampled.tolist() == [True, False, False]
    assert samples.sampled_count == 1


# --- Bilinear ---------------------------------------------------------------


def test_bilinear_reproduces_a_linear_field_exactly() -> None:
    dense_map = make_dense_map()
    frame = scene_frame((100, 100, 3.0), (333, 217, 3.0), (52, 401, 3.0))

    samples = _sample(frame, dense_map, interpolation=InterpolationPolicy.BILINEAR)
    vectors = samples.gather(_coordinate_field(dense_map))

    # Coordenada contínua da grade: g = (borda - origem - suporte / 2) / passo.
    expected = [
        [(v + 0.5 - 8.0) / 16.0, (u + 0.5 - 8.0) / 16.0]
        for u, v in [(100, 100), (333, 217), (52, 401)]
    ]
    np.testing.assert_allclose(vectors, expected, atol=1e-5)
    assert samples.cell_rows.shape == (3, 4)
    np.testing.assert_allclose(samples.weights.sum(axis=1), 1.0)


def test_bilinear_needs_the_pixel_between_the_outer_cell_centers() -> None:
    dense_map = make_dense_map()
    # Centro da célula 0 em 8; da última (39), em 632. Antes ou depois, faltam vizinhos.
    frame = scene_frame((3, 100, 3.0), (100, 100, 3.0), (637, 100, 3.0))

    samples = _sample(frame, dense_map, interpolation=InterpolationPolicy.BILINEAR)

    assert samples.sampled.tolist() == [False, True, False]
    nearest = _sample(frame, dense_map)
    assert nearest.sampled.tolist() == [True, True, True]


def test_bilinear_next_to_the_last_center_reads_the_last_two_columns() -> None:
    dense_map = make_dense_map()
    # Perto do último centro (borda 632.0) mas antes dele, para não depender de igualdade exata.
    frame = scene_frame((631.4, 100, 3.0))

    samples = _sample(frame, dense_map, interpolation=InterpolationPolicy.BILINEAR)

    assert samples.sampled.tolist() == [True]
    np.testing.assert_allclose(
        samples.gather(_coordinate_field(dense_map)),
        [[(100.5 - 8.0) / 16.0, (631.9 - 8.0) / 16.0]],
        atol=1e-4,
    )


def test_an_l2_normalized_feature_stays_normalized_after_interpolation() -> None:
    sampling = make_sampling()
    dense_map = make_dense_map(sampling, normalization="l2", channels=2)
    height, width = sampling.grid_height, sampling.grid_width
    angles = np.linspace(0.0, math.pi / 2, width)[None, :].repeat(height, axis=0)
    array = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)
    frame = scene_frame((100, 100, 3.0), (333, 217, 3.0))

    bilinear = _sample(frame, dense_map, interpolation=InterpolationPolicy.BILINEAR)
    nearest = _sample(frame, dense_map)

    np.testing.assert_allclose(np.linalg.norm(bilinear.gather(array), axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(nearest.gather(array), axis=1), 1.0, atol=1e-6)


# --- Eligibility ------------------------------------------------------------


def test_only_visible_points_are_sampled() -> None:
    dense_map = make_dense_map()
    frame = project_frame(
        [
            map_point_for_pixel(100, 100, 2.0),
            map_point_for_pixel(100, 100, 6.0),
            (-3.0, 0.0, 0.0),
            (0.5, -10.0, 0.0),
        ]
    )

    samples = _sample(frame, dense_map)

    assert samples.eligible_indices.tolist() == [0]
    assert samples.sampled_point_indices.tolist() == [0]


# --- Native and enhanced maps share one path --------------------------------


def test_a_compatible_enhanced_map_is_sampled_exactly_like_a_native_one() -> None:
    native = make_dense_map()
    fine = make_sampling(
        (80, 60), stride=(8.0, 8.0), support=(8.0, 8.0), transform_id="enhanced-grid-v1"
    )
    enhanced = make_dense_map(
        fine, feature_id="dense-enhanced", enhancement=make_enhancement(native, fine)
    )
    frame = scene_frame((100, 100, 3.0))

    coarse_samples = _sample(frame, native)
    fine_samples = _sample(frame, enhanced)

    assert (int(coarse_samples.cell_rows[0, 0]), int(coarse_samples.cell_cols[0, 0])) == (6, 6)
    assert (int(fine_samples.cell_rows[0, 0]), int(fine_samples.cell_cols[0, 0])) == (12, 12)
    assert coarse_samples.provenance.enhancement is None
    assert fine_samples.provenance.enhancement is not None
    assert fine_samples.provenance.enhancement.source_feature_id == native.feature.feature_id


# --- Preflight --------------------------------------------------------------


def test_a_dense_map_over_another_image_size_fails_preflight() -> None:
    dense_map = make_dense_map(make_sampling(image=(800, 600)))

    with pytest.raises(AssociationInputError, match="prepared image"):
        _sample(scene_frame((100, 100, 3.0)), dense_map)


def test_a_dense_map_over_the_prepared_image_of_a_resized_frame_is_accepted() -> None:
    prepared = make_prepared_image(
        (
            ResizeOperation(
                width=320, height=240, output_image=artifact("half"), provenance_source="c"
            ),
        )
    )
    dense_map = make_dense_map(make_sampling((20, 15), image=(320, 240)))

    samples = _sample(scene_frame((100, 100, 3.0), prepared=prepared), dense_map)

    # Pixel cru (100, 100) -> preparado (49.75, 49.75): borda 50.25, célula 3.
    assert (int(samples.cell_rows[0, 0]), int(samples.cell_cols[0, 0])) == (3, 3)


def test_a_feature_that_is_not_in_the_perception_result_fails_preflight() -> None:
    dense_map = make_dense_map()
    other = make_dense_map(feature_id="dense-other")

    with pytest.raises(AssociationInputError, match="feature"):
        _sample(scene_frame((100, 100, 3.0)), dense_map, result=make_dense_result(other))


def test_a_feature_in_another_embedding_space_than_the_result_fails_preflight() -> None:
    dense_map = make_dense_map()
    same_id_other_space = make_dense_map(space="clip:vitl14")

    with pytest.raises(AssociationInputError, match="embedding"):
        _sample(
            scene_frame((100, 100, 3.0)), dense_map, result=make_dense_result(same_id_other_space)
        )


def test_a_result_of_another_observation_fails_preflight() -> None:
    dense_map = make_dense_map()
    result = make_result(
        [], features=[dense_map.feature], observation_id=SourceObservationId("frame-9999")
    )

    with pytest.raises(AssociationInputError, match="observation"):
        _sample(scene_frame((100, 100, 3.0)), dense_map, result=result)


def test_an_enhancement_that_disagrees_with_its_sampling_fails_preflight() -> None:
    native = make_dense_map()
    fine = make_sampling((80, 60), stride=(8.0, 8.0), support=(8.0, 8.0))
    wrong_grid = dataclasses.replace(make_enhancement(native, fine), output_grid_size=(79, 60))
    wrong_space = dataclasses.replace(
        make_enhancement(native, fine), output_embedding_space_id="another:space"
    )

    for enhancement in (wrong_grid, wrong_space):
        enhanced = make_dense_map(fine, feature_id="dense-enhanced", enhancement=enhancement)
        with pytest.raises(AssociationInputError, match="enhancement"):
            _sample(scene_frame((100, 100, 3.0)), enhanced)


def test_the_payload_must_match_the_declared_feature() -> None:
    dense_map = make_dense_map()
    samples = _sample(scene_frame((100, 100, 3.0)), dense_map)

    with pytest.raises(ValueError, match="shape"):
        samples.gather(np.zeros((30, 40, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="dtype"):
        samples.gather(np.zeros((30, 40, 2), dtype=np.float64))


# --- Reproducibility and provenance -----------------------------------------


def test_the_records_are_reproducible() -> None:
    dense_map = make_dense_map()
    frame = scene_frame((100, 100, 3.0), (300, 200, 3.0))

    first = _sample(frame, dense_map, interpolation=InterpolationPolicy.BILINEAR)
    second = _sample(frame, dense_map, interpolation=InterpolationPolicy.BILINEAR)

    assert np.array_equal(first.cell_rows, second.cell_rows)
    assert np.array_equal(first.weights, second.weights)
    assert first.provenance == second.provenance


def test_the_sampling_names_every_source_it_depended_on() -> None:
    dense_map = make_dense_map()
    frame = scene_frame((100, 100, 3.0))

    provenance = _sample(frame, dense_map, interpolation=InterpolationPolicy.BILINEAR).provenance

    assert provenance.policy_id == SAMPLING_POLICY_ID == "dense-feature-sampling-v1"
    assert provenance.interpolation is InterpolationPolicy.BILINEAR
    assert provenance.feature_id == FeatureId("dense-native")
    assert provenance.source_artifact_id == "perception-artifact-0001"
    assert provenance.payload_reference == "features/dense-native.npy"
    assert provenance.embedding_space_id == "dinov2:b14"
    assert provenance.extractor.backend_id == "fake_extractor"
    assert provenance.coordinate_transform_id == "native-grid-v1"
    assert provenance.sampling_fingerprint.startswith("sha256:")
    assert provenance.map_id == frame.map_id
    assert provenance.source_observation_id == frame.source_observation_id
    assert provenance.image_transform_id == frame.image_transform.transform_id
    assert provenance.calibration_ref == frame.calibration_ref
    assert provenance.pose_ref == frame.pose_ref
    assert provenance.visibility_policy_id == POLICY.policy_id
    assert provenance.visibility_policy_fingerprint == POLICY.fingerprint()


def test_the_sampling_fingerprint_tracks_the_whole_sampling_geometry() -> None:
    frame = scene_frame((100, 100, 3.0))

    def fingerprint(sampling: DenseFeatureSampling) -> str:
        return _sample(frame, make_dense_map(sampling)).provenance.sampling_fingerprint

    base = fingerprint(make_sampling())
    assert base == fingerprint(make_sampling())
    assert base != fingerprint(make_sampling(origin=(1.0, 0.0)))
    assert base != fingerprint(make_sampling(transform_id="other-v1"))


def test_the_provenance_record_is_json() -> None:
    dense_map = make_dense_map()
    fine = make_sampling((80, 60), stride=(8.0, 8.0), support=(8.0, 8.0))
    enhanced = make_dense_map(
        fine, feature_id="dense-enhanced", enhancement=make_enhancement(dense_map, fine)
    )

    for candidate in (dense_map, enhanced):
        provenance = _sample(scene_frame((100, 100, 3.0)), candidate).provenance
        record = json.loads(json.dumps(provenance.to_record()))
        assert record["policy_id"] == SAMPLING_POLICY_ID
        assert record["feature_id"] == str(candidate.feature.feature_id)
        assert (record["enhancement"] is None) == (candidate.enhancement is None)
