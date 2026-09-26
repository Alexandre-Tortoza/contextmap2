from dataclasses import FrozenInstanceError
from hashlib import sha256

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import ArtifactReference, BoundingBox, InlineMask
from contextmap.visual_perception.image_preparation import (
    CropOperation,
    ExclusionRegion,
    PreparedImage,
    ResizeOperation,
    SourceImage,
    ValidRegion,
    prepare_image,
)


def _reference(name: str) -> ArtifactReference:
    return ArtifactReference(
        uri=f"outputs/images/{name}.png",
        sha256=sha256(name.encode()).hexdigest(),
        media_type="image/png",
    )


def _mask(width: int, height: int, foreground: set[tuple[int, int]]) -> InlineMask:
    return InlineMask(
        np.array(
            tuple((x, y) in foreground for y in range(height) for x in range(width)), dtype=bool
        ).reshape(height, width)
    )


def test_zero_configuration_is_an_auditable_no_op() -> None:
    source = SourceImage(
        source_observation_id="frame-1", image=_reference("source"), width=8, height=6
    )

    prepared = prepare_image(source)

    assert prepared == PreparedImage(
        source_observation_id=SourceObservationId("frame-1"),
        payload_reference=_reference("source").uri,
        payload_artifact=_reference("source"),
        width=8,
        height=6,
        transformations=(),
    )
    with pytest.raises(FrozenInstanceError):
        source.width = 4  # type: ignore[misc]


def test_resize_and_crop_preserve_ordered_transform_provenance() -> None:
    source = SourceImage(
        source_observation_id="frame-1", image=_reference("source"), width=8, height=6
    )

    prepared = prepare_image(
        source,
        operations=(
            ResizeOperation(
                width=16,
                height=12,
                output_image=_reference("resized"),
                provenance_source="camera-profile-v2",
            ),
            CropOperation(
                box=BoundingBox(x_min=2, y_min=1, x_max=12, y_max=9),
                output_image=_reference("cropped"),
                provenance_source="experiment-config-sha256:abc",
            ),
        ),
    )

    assert (prepared.width, prepared.height) == (10, 8)
    assert prepared.payload_artifact == _reference("cropped")
    assert [record.operation for record in prepared.transformations] == ["resize", "crop"]
    assert prepared.transformations[0].input_dimensions == (8, 6)
    assert prepared.transformations[0].output_dimensions == (16, 12)
    assert prepared.transformations[0].provenance_source == "camera-profile-v2"
    assert prepared.transformations[1].parameters == (
        ("x_min", 2.0),
        ("y_min", 1.0),
        ("x_max", 12.0),
        ("y_max", 9.0),
    )
    assert prepared.transformations[1].to_dict()["provenance_source"] == (
        "experiment-config-sha256:abc"
    )


def test_explicit_valid_and_exclusion_regions_are_optional_and_named() -> None:
    source = SourceImage(
        source_observation_id="frame-1", image=_reference("source"), width=4, height=3
    )
    valid = ValidRegion(
        mask=_mask(4, 3, {(0, 0), (1, 0), (0, 1), (1, 1)}),
        reason="lens_valid_area",
        source="camera-profile-v2",
    )
    exclusion = ExclusionRegion(
        name="rig",
        mask=_mask(4, 3, {(0, 2), (1, 2)}),
        reason="visible camera rig",
        source="sequence-config",
    )

    prepared = prepare_image(source, valid_region=valid, exclusion_regions=(exclusion,))

    assert prepared.valid_region == valid
    assert prepared.exclusion_regions == (exclusion,)
    assert prepared.transformations == ()


def test_preparation_rejects_geometry_that_does_not_match_final_image_space() -> None:
    source = SourceImage(
        source_observation_id="frame-1", image=_reference("source"), width=4, height=3
    )

    with pytest.raises(ValueError, match="final prepared image dimensions"):
        prepare_image(
            source,
            valid_region=ValidRegion(mask=_mask(2, 2, {(0, 0)}), reason="bad", source="test"),
        )

    with pytest.raises(ValueError, match="crop extends outside"):
        prepare_image(
            source,
            operations=(
                CropOperation(
                    box=BoundingBox(x_min=0, y_min=0, x_max=5, y_max=2),
                    output_image=_reference("cropped"),
                    provenance_source="test-config",
                ),
            ),
        )

    with pytest.raises(ValueError, match="provenance source"):
        ResizeOperation(
            width=8,
            height=6,
            output_image=_reference("invalid-source"),
            provenance_source="",
        )

    duplicate = ExclusionRegion(
        name="overlay",
        mask=_mask(4, 3, {(0, 0)}),
        reason="overlay",
        source="test",
    )
    with pytest.raises(ValueError, match="unique names"):
        prepare_image(source, exclusion_regions=(duplicate, duplicate))
