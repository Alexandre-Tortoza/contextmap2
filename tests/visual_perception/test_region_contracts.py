from dataclasses import FrozenInstanceError

import pytest

from contextmap.visual_perception import (
    ArtifactReference,
    BackendScore,
    BoundingBox,
    CoordinateConvention,
    InlineMask,
    Region2D,
    RegionCandidate,
    RegionIdentity,
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
        source_observation_id="frame-12",
        perception_run_id="run-4",
        perception_result_id="result-7",
        image_width=4,
        image_height=3,
        bounding_box=BoundingBox(x_min=1.0, y_min=0.0, x_max=3.0, y_max=2.0),
        mask=InlineMask(
            width=4,
            height=3,
            data=(False, True, True, False, False, True, True, False, False, False, False, False),
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
        source_observation_id="frame-12",
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
        identity=RegionIdentity(
            perception_run_id="run-a", perception_result_id="result-a", region_id="region-1"
        ),
        source_observation_id="frame-12",
        image_width=4,
        image_height=3,
        bounding_box=BoundingBox(x_min=1.0, y_min=0.0, x_max=3.0, y_max=2.0),
        mask=ArtifactReference(
            uri="outputs/masks/region-1.pbm",
            sha256="a" * 64,
            media_type="image/x-portable-bitmap",
        ),
        area_pixels=4,
        contributor_candidate_ids=("candidate-1",),
        provenance=(_provenance(),),
    )
    same_local_id_in_another_result = RegionIdentity(
        perception_run_id="run-b", perception_result_id="result-b", region_id="region-1"
    )

    assert first.identity != same_local_id_in_another_result
    assert Region2D.from_dict(first.to_dict()) == first
    with pytest.raises(FrozenInstanceError):
        first.area_pixels = 8  # type: ignore[misc]


def test_rejected_candidate_preserves_machine_readable_reason() -> None:
    rejected = RejectedRegionCandidate(
        candidate_id="candidate-9",
        reason=RejectionReason.INVALID_GEOMETRY,
        detail="mask has no foreground pixels",
        discovery_pass_id="tile-3",
    )

    assert RejectedRegionCandidate.from_dict(rejected.to_dict()) == rejected
    assert rejected.to_dict()["reason"] == "invalid_geometry"
