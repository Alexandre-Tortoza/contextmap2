import numpy as np
import pytest
from projection_builders import IMAGE_HEIGHT, IMAGE_WIDTH, artifact, make_prepared_image

from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.image_transform import raw_to_prepared_transform
from contextmap.visual_perception import (
    BoundingBox,
    CropOperation,
    NormalizeOperation,
    PreparedImage,
    RectifyOperation,
    ResizeOperation,
    TransformationRecord,
)

RAW = (IMAGE_WIDTH, IMAGE_HEIGHT)


def _resize(width: int, height: int) -> ResizeOperation:
    return ResizeOperation(
        width=width,
        height=height,
        output_image=artifact(f"resize-{width}"),
        provenance_source="cfg",
    )


def _crop(x_min: float, y_min: float, x_max: float, y_max: float) -> CropOperation:
    return CropOperation(
        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        output_image=artifact("crop"),
        provenance_source="cfg",
    )


def _map(operations: tuple[object, ...], *pixels: tuple[float, float]) -> np.ndarray:
    prepared = make_prepared_image(operations)  # type: ignore[arg-type]
    transform = raw_to_prepared_transform(raw_size=RAW, prepared_image=prepared)
    return transform.map_pixels(np.array(pixels))


def test_without_operations_the_prepared_image_is_the_raw_image() -> None:
    transform = raw_to_prepared_transform(raw_size=RAW, prepared_image=make_prepared_image())

    np.testing.assert_allclose(transform.map_pixels(np.array([[12.5, 7.25]])), [[12.5, 7.25]])
    assert transform.steps == ()
    assert transform.prepared_size == RAW


def test_a_resize_scales_about_the_pixel_edges_not_the_pixel_centers() -> None:
    # Halving a 640x480 image: the raw pixel centered at 0 covers [-0.5, 0.5) and
    # lands at -0.25 in the half-size image, where one pixel covers two raw pixels.
    result = _map((_resize(320, 240),), (0.0, 0.0), (1.0, 1.0), (639.0, 479.0))

    np.testing.assert_allclose(result, [[-0.25, -0.25], [0.25, 0.25], [319.25, 239.25]])


def test_an_anisotropic_resize_scales_each_axis_by_its_own_ratio() -> None:
    result = _map((_resize(1280, 240),), (10.0, 10.0))

    np.testing.assert_allclose(result, [[(10.5 * 2) - 0.5, (10.5 * 0.5) - 0.5]])


def test_a_crop_translates_by_its_origin() -> None:
    result = _map((_crop(100, 50, 500, 350),), (445.0, 177.5), (100.0, 50.0))

    np.testing.assert_allclose(result, [[345.0, 127.5], [0.0, 0.0]])


def test_operations_compose_in_the_recorded_order() -> None:
    crop_then_resize = _map((_crop(100, 50, 500, 350), _resize(200, 150)), (445.0, 177.5))
    resize_then_crop = _map((_resize(320, 240), _crop(50, 25, 250, 175)), (445.0, 177.5))

    np.testing.assert_allclose(crop_then_resize, [[172.25, 63.5]])
    np.testing.assert_allclose(resize_then_crop, [[222.25 - 50.0, 88.5 - 25.0]])


def test_a_normalization_does_not_move_pixels() -> None:
    operation = NormalizeOperation(
        method="imagenet", output_image=artifact("norm"), provenance_source="cfg"
    )

    np.testing.assert_allclose(_map((operation,), (10.0, 20.0)), [[10.0, 20.0]])


def test_unprojectable_pixels_stay_unprojectable() -> None:
    result = _map((_resize(320, 240),), (float("nan"), float("nan")))

    assert np.isnan(result).all()


def test_the_prepared_extent_is_half_open_about_the_pixel_edges() -> None:
    transform = raw_to_prepared_transform(
        raw_size=RAW, prepared_image=make_prepared_image((_resize(320, 240),))
    )
    prepared = np.array(
        [[-0.5, -0.5], [319.49, 239.49], [319.5, 10.0], [-0.51, 10.0], [np.nan, 1.0]]
    )

    assert transform.in_prepared_image(prepared).tolist() == [True, True, False, False, False]


def test_the_transform_identity_is_deterministic_and_tracks_every_parameter() -> None:
    def transform_id(*operations: object) -> str:
        prepared = make_prepared_image(operations)  # type: ignore[arg-type]
        return raw_to_prepared_transform(raw_size=RAW, prepared_image=prepared).transform_id

    same = transform_id(_crop(100, 50, 500, 350), _resize(200, 150))
    assert same == transform_id(_crop(100, 50, 500, 350), _resize(200, 150))
    assert same.startswith("sha256:")
    assert same != transform_id(_crop(101, 50, 501, 350), _resize(200, 150))
    assert same != transform_id(_crop(100, 50, 500, 350), _resize(200, 151))
    assert transform_id() != transform_id(_resize(640, 480))


def test_the_chain_must_start_from_the_raw_image_the_calibration_describes() -> None:
    prepared = make_prepared_image((_resize(320, 240),), raw_size=(800, 600))

    with pytest.raises(AssociationInputError, match="raw image"):
        raw_to_prepared_transform(raw_size=RAW, prepared_image=prepared)


def test_the_chain_must_end_at_the_prepared_image_size() -> None:
    prepared = make_prepared_image((_resize(320, 240),))
    broken = PreparedImage(
        source_observation_id=prepared.source_observation_id,
        payload_reference=prepared.payload_reference,
        width=640,
        height=480,
        transformations=prepared.transformations,
    )

    with pytest.raises(AssociationInputError, match="prepared image"):
        raw_to_prepared_transform(raw_size=RAW, prepared_image=broken)


def test_a_chain_with_a_gap_is_rejected() -> None:
    prepared = make_prepared_image((_resize(320, 240), _resize(160, 120)))
    records = list(prepared.transformations)
    gapped = TransformationRecord(
        operation=records[1].operation,
        provenance_source=records[1].provenance_source,
        parameters=records[1].parameters,
        input_dimensions=(400, 300),
        output_dimensions=records[1].output_dimensions,
        output_image=records[1].output_image,
    )
    broken = PreparedImage(
        source_observation_id=prepared.source_observation_id,
        payload_reference=prepared.payload_reference,
        width=prepared.width,
        height=prepared.height,
        transformations=(records[0], gapped),
    )

    with pytest.raises(AssociationInputError, match="contiguous"):
        raw_to_prepared_transform(raw_size=RAW, prepared_image=broken)


def test_a_rectification_is_rejected_because_its_remap_is_not_recorded() -> None:
    operation = RectifyOperation(
        calibration_id="camera-calib", output_image=artifact("rect"), provenance_source="cfg"
    )
    prepared = make_prepared_image((operation,))

    with pytest.raises(AssociationInputError, match="rectif"):
        raw_to_prepared_transform(raw_size=RAW, prepared_image=prepared)


def test_a_crop_record_that_disagrees_with_its_output_size_is_rejected() -> None:
    prepared = make_prepared_image((_crop(100, 50, 500, 350),))
    record = prepared.transformations[0]
    lying = TransformationRecord(
        operation="crop",
        provenance_source=record.provenance_source,
        parameters=record.parameters,
        input_dimensions=record.input_dimensions,
        output_dimensions=(399, 300),
        output_image=record.output_image,
    )
    broken = PreparedImage(
        source_observation_id=prepared.source_observation_id,
        payload_reference=prepared.payload_reference,
        width=399,
        height=300,
        transformations=(lying,),
    )

    with pytest.raises(AssociationInputError, match="crop"):
        raw_to_prepared_transform(raw_size=RAW, prepared_image=broken)


def test_an_unknown_operation_is_rejected_instead_of_guessed() -> None:
    prepared = make_prepared_image()
    record = TransformationRecord(
        operation="warp_perspective",
        provenance_source="cfg",
        parameters=(),
        input_dimensions=RAW,
        output_dimensions=RAW,
        output_image=artifact("warp"),
    )
    broken = PreparedImage(
        source_observation_id=prepared.source_observation_id,
        payload_reference=prepared.payload_reference,
        width=RAW[0],
        height=RAW[1],
        transformations=(record,),
    )

    with pytest.raises(AssociationInputError, match="warp_perspective"):
        raw_to_prepared_transform(raw_size=RAW, prepared_image=broken)
