import copy
import pickle
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from mask_cases import inline_mask

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BackendScore,
    BoundingBox,
    BoundingBox2D,
    CoordinateConvention,
    InlineMask,
    Region2D,
    RegionCandidate,
    RegionId,
    RegionProvenance,
    RejectedRegionCandidate,
    RejectionReason,
)


def _provenance(*, proposal_id: str = "proposal-1") -> RegionProvenance:
    return RegionProvenance(
        backend_id="sam-test",
        backend_version="1.0",
        checkpoint="checkpoint-a",
        config_digest="sha256:config",
        discovery_pass_id="full-frame",
        native_proposal_id=proposal_id,
        query="grid",
    )


def test_region_candidate_round_trips_backend_neutral_geometry() -> None:
    candidate = RegionCandidate(
        candidate_id="candidate-1",
        source_observation_id=SourceObservationId("frame-12"),
        perception_run_id="run-4",
        perception_result_id="result-7",
        image_width=4,
        image_height=3,
        bounding_box=BoundingBox(x_min=1.0, y_min=0.0, x_max=3.0, y_max=2.0),
        mask=InlineMask(
            np.array(
                (False, True, True, False, False, True, True, False, False, False, False, False),
                dtype=bool,
            ).reshape(3, 4)
        ),
        score=BackendScore(
            name="predicted_iou",
            value=0.82,
            semantics="SAM-native predicted mask IoU; not calibrated across backends",
        ),
        provenance=_provenance(),
        native_metadata=(("stability_score", 0.91),),
    )

    restored = RegionCandidate.from_dict(candidate.to_dict())

    assert restored == candidate
    assert restored.score is not None
    assert restored.score.name == "predicted_iou"
    assert restored.coordinate_convention is CoordinateConvention.PIXEL_XY_TOP_LEFT


def test_region_candidate_allows_box_only_geometry_and_no_score() -> None:
    candidate = RegionCandidate(
        candidate_id="florence-box-1",
        source_observation_id=SourceObservationId("frame-12"),
        perception_run_id="run-4",
        perception_result_id="result-7",
        image_width=20,
        image_height=10,
        bounding_box=BoundingBox(x_min=2.0, y_min=3.0, x_max=8.0, y_max=9.0),
        provenance=_provenance(),
    )

    assert candidate.mask is None
    assert candidate.score is None
    assert RegionCandidate.from_dict(candidate.to_dict()) == candidate


def test_region_candidate_rejects_missing_or_out_of_bounds_geometry() -> None:
    with pytest.raises(ValueError, match="at least one geometry"):
        RegionCandidate(
            candidate_id="invalid",
            source_observation_id="frame-12",
            perception_run_id="run-4",
            perception_result_id="result-7",
            image_width=10,
            image_height=8,
            provenance=_provenance(),
        )

    with pytest.raises(ValueError, match="image bounds"):
        RegionCandidate(
            candidate_id="invalid",
            source_observation_id="frame-12",
            perception_run_id="run-4",
            perception_result_id="result-7",
            image_width=10,
            image_height=8,
            provenance=_provenance(),
            bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=11.0, y_max=2.0),
        )


def test_region_geometry_is_frozen_and_identity_is_result_local() -> None:
    first = Region2D(
        region_id=RegionId("region-1"),
        bounding_box=BoundingBox2D(x=1.0, y=0.0, width=2.0, height=2.0),
        provenance=BackendProvenance(
            backend_id="sam-test",
            capability="region_discovery",
            provider="test",
            model="checkpoint-a",
            version="1.0",
            configuration_fingerprint="sha256:config",
        ),
        source_observation_id=SourceObservationId("frame-12"),
        image_width=4,
        image_height=3,
        mask_reference="outputs/masks/region-1.pbm",
        area_pixels=4,
        contributor_candidate_ids=("candidate-1",),
        discovery_provenance=(_provenance(),),
    )

    assert ("result-a", first.region_id) != ("result-b", first.region_id)
    assert first.to_dict()["region_id"] == "region-1"
    with pytest.raises(FrozenInstanceError):
        first.area_pixels = 8  # type: ignore[misc]


def test_candidate_requires_materialized_mask_when_a_mask_reference_is_present() -> None:
    with pytest.raises(ValueError, match="requires a materialized mask"):
        RegionCandidate(
            candidate_id="candidate-1",
            source_observation_id="frame-12",
            perception_run_id="run-4",
            perception_result_id="result-7",
            image_width=4,
            image_height=3,
            bounding_box=BoundingBox(x_min=1.0, y_min=0.0, x_max=3.0, y_max=2.0),
            mask_reference=ArtifactReference(
                uri="outputs/masks/candidate-1.pbm",
                sha256="a" * 64,
                media_type="image/x-portable-bitmap",
            ),
            provenance=_provenance(),
        )


def test_rejected_candidate_preserves_machine_readable_reason() -> None:
    rejected = RejectedRegionCandidate(
        candidate_id="candidate-9",
        reason=RejectionReason.INVALID_GEOMETRY,
        detail="mask has no foreground pixels",
        discovery_pass_id="tile-3",
    )

    assert RejectedRegionCandidate.from_dict(rejected.to_dict()) == rejected
    assert rejected.to_dict()["reason"] == "invalid_geometry"


def test_an_inline_mask_holds_one_byte_per_pixel() -> None:
    # #593: um tuple de bools custa ~2,36 MB numa máscara 640x480; um byte por pixel, ~300 KB.
    mask = inline_mask(np.zeros((480, 640), dtype=bool))

    assert mask.as_array().nbytes == 640 * 480


def _pixels() -> np.ndarray[Any, Any]:
    pixels = np.zeros((3, 4), dtype=bool)
    pixels[1, 2] = True
    return pixels


def test_an_inline_mask_owns_an_immutable_copy_of_its_pixels() -> None:
    pixels = _pixels()
    mask = InlineMask(pixels)
    pixels[0, 0] = True  # o chamador continua dono do array que passou

    view = mask.as_array()

    assert view.tolist() == _pixels().tolist()
    with pytest.raises(ValueError, match="read-only"):
        view[0, 0] = True
    base: object = view
    while isinstance(base, np.ndarray):
        # Nem a view nem nada abaixo dela pode voltar a ser gravável.
        with pytest.raises(ValueError):
            base.setflags(write=True)
        base = base.base
    view.shape = (4, 3)  # a view é do chamador: remodelá-la nunca alcança a máscara
    assert mask.as_array().shape == (3, 4)


def test_an_inline_mask_exposes_its_shape_area_and_pixels() -> None:
    mask = InlineMask(_pixels())

    assert (mask.width, mask.height, mask.area) == (4, 3, 1)
    assert mask.value_at(2, 1) is True
    assert mask.value_at(0, 0) is False
    with pytest.raises(IndexError):
        mask.value_at(4, 0)
    with pytest.raises(FrozenInstanceError):
        mask._pixels = np.ones((3, 4), dtype=bool)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("pixels", "error"),
    [
        (np.zeros((3, 4), dtype=np.uint8), TypeError),
        (np.zeros(12, dtype=bool), ValueError),
        (np.zeros((0, 4), dtype=bool), ValueError),
    ],
)
def test_an_inline_mask_refuses_pixels_that_are_not_a_boolean_image(
    pixels: np.ndarray[Any, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        InlineMask(pixels)


def test_inline_masks_are_values() -> None:
    mask = InlineMask(_pixels())
    same = InlineMask(_pixels().copy())

    assert mask == same
    assert hash(mask) == hash(same)
    assert {mask: "region"}[same] == "region"
    assert mask != InlineMask(~_pixels())
    # Mesmos bytes, outra forma: não é a mesma máscara.
    assert InlineMask(np.zeros((2, 6), dtype=bool)) != InlineMask(np.zeros((3, 4), dtype=bool))
    assert mask != "mask"
    assert copy.deepcopy(mask) == mask
    assert pickle.loads(pickle.dumps(mask)) == mask


def test_an_inline_mask_keeps_its_serialized_form() -> None:
    mask = InlineMask(_pixels())

    document = mask.to_dict()

    assert document == {
        "storage": "inline",
        "width": 4,
        "height": 3,
        "data": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    }
    assert all(type(value) is int for value in document["data"])  # type: ignore[attr-defined]
    assert InlineMask.from_dict(document) == mask
