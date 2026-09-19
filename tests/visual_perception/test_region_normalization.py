from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox,
    ExclusionRegion,
    InlineMask,
    PreparedImage,
    RegionCandidate,
    RegionProvenance,
    RejectionReason,
    ValidRegion,
)
from contextmap.visual_perception.normalization import (
    MergeKind,
    NormalizationConfig,
    normalize_regions,
)

WIDTH = 8
HEIGHT = 6


def _backend_provenance(backend: str = "fake") -> BackendProvenance:
    return BackendProvenance(
        backend_id=backend,
        capability="region_discovery",
        provider="test-provider",
        model=f"{backend}-checkpoint",
        version="1",
        configuration_fingerprint=f"sha256:{backend}",
    )


def _mask(box: BoundingBox | None) -> InlineMask:
    return InlineMask(
        width=WIDTH,
        height=HEIGHT,
        data=tuple(
            box is not None and box.x_min <= x < box.x_max and box.y_min <= y < box.y_max
            for y in range(HEIGHT)
            for x in range(WIDTH)
        ),
    )


def _prepared(
    *, valid: InlineMask | None = None, exclusion: InlineMask | None = None
) -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-1"),
        payload_reference="outputs/frame.png",
        payload_artifact=ArtifactReference(
            uri="outputs/frame.png",
            sha256=sha256(b"frame").hexdigest(),
            media_type="image/png",
        ),
        width=WIDTH,
        height=HEIGHT,
        transformations=(),
        valid_region=(
            None if valid is None else ValidRegion(valid, "configured valid area", "test")
        ),
        exclusion_regions=(
            () if exclusion is None else (ExclusionRegion("rig", exclusion, "visible rig", "test"),)
        ),
    )


def _candidate(
    candidate_id: str,
    box: BoundingBox | None,
    *,
    backend: str = "fake",
    discovery_pass: str = "full-frame",
    mask: InlineMask | None = None,
) -> RegionCandidate:
    return RegionCandidate(
        candidate_id=candidate_id,
        source_observation_id="frame-1",
        perception_run_id="run-1",
        perception_result_id="result-1",
        image_width=WIDTH,
        image_height=HEIGHT,
        bounding_box=box,
        mask=_mask(box) if mask is None else mask,
        provenance=RegionProvenance(
            backend_id=backend,
            backend_version="1",
            checkpoint=f"{backend}-checkpoint",
            config_digest=f"sha256:{backend}",
            discovery_pass_id=discovery_pass,
            native_proposal_id=candidate_id,
        ),
    )


def test_duplicate_full_frame_and_tile_proposals_merge_with_lineage() -> None:
    first = _candidate(
        "candidate-a",
        BoundingBox(1, 1, 4, 4),
        backend="sam3",
        discovery_pass="full-frame",
    )
    duplicate = _candidate(
        "candidate-b",
        BoundingBox(1, 1, 4, 4),
        backend="sam3",
        discovery_pass="tile-0001",
    )

    provenance = _backend_provenance("sam3")
    result = normalize_regions((duplicate, first), _prepared(), provenance)

    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.region_id == "region-0001"
    assert region.contributor_candidate_ids == ("candidate-a", "candidate-b")
    assert [item.discovery_pass_id for item in region.discovery_provenance] == [
        "full-frame",
        "tile-0001",
    ]
    assert result.merge_decisions[0].kind is MergeKind.IOU_DUPLICATE
    assert result.merge_decisions[0].iou == 1.0
    assert result.rejected[0].reason is RejectionReason.MERGED_DUPLICATE
    assert region.provenance is provenance
    with pytest.raises(FrozenInstanceError):
        region.bounding_box = BoundingBox(0, 0, 1, 1)  # type: ignore[misc, assignment]


def test_contained_fragment_merges_without_comparing_backend_scores() -> None:
    container = _candidate("a-container", BoundingBox(1, 1, 6, 5), backend="sam2")
    fragment = _candidate("b-fragment", BoundingBox(2, 2, 4, 4), backend="sam2")
    config = NormalizationConfig(duplicate_iou_threshold=0.95, containment_threshold=0.9)

    result = normalize_regions(
        (fragment, container), _prepared(), _backend_provenance("sam2"), config
    )

    assert len(result.regions) == 1
    assert result.regions[0].contributor_candidate_ids == ("a-container", "b-fragment")
    assert result.merge_decisions[0].kind is MergeKind.CONTAINMENT


def test_invalid_area_and_configured_spatial_constraints_have_explicit_reasons() -> None:
    empty = _candidate("empty", None, mask=_mask(None))
    too_small = _candidate("small", BoundingBox(0, 0, 1, 1))
    outside_valid = _candidate("outside", BoundingBox(6, 0, 8, 2))
    excluded = _candidate("excluded", BoundingBox(2, 3, 4, 5))
    valid_mask = _mask(BoundingBox(0, 0, 5, 6))
    exclusion_mask = _mask(BoundingBox(0, 3, 5, 6))
    config = NormalizationConfig(
        minimum_area_pixels=2,
        minimum_valid_fraction=0.75,
        maximum_exclusion_fraction=0.1,
    )

    result = normalize_regions(
        (empty, too_small, outside_valid, excluded),
        _prepared(valid=valid_mask, exclusion=exclusion_mask),
        _backend_provenance(),
        config,
    )

    reasons = {item.candidate_id: item.reason for item in result.rejected}
    assert reasons == {
        "empty": RejectionReason.INVALID_GEOMETRY,
        "small": RejectionReason.AREA_BELOW_MINIMUM,
        "outside": RejectionReason.OUTSIDE_VALID_REGION,
        "excluded": RejectionReason.EXCLUSION_OVERLAP,
    }


def test_constraints_are_irrelevant_when_the_prepared_image_declares_none() -> None:
    candidate = _candidate("outside", BoundingBox(6, 0, 8, 2))
    config = NormalizationConfig(
        minimum_valid_fraction=1.0,
        maximum_exclusion_fraction=0.0,
    )

    result = normalize_regions((candidate,), _prepared(), _backend_provenance(), config)

    assert [region.contributor_candidate_ids for region in result.regions] == [("outside",)]
    assert result.rejected == ()


def test_region_budget_is_deterministic_and_configuration_is_provenance_visible() -> None:
    candidates = (
        _candidate("c", BoundingBox(6, 0, 8, 2)),
        _candidate("a", BoundingBox(0, 0, 2, 2)),
        _candidate("b", BoundingBox(3, 0, 5, 2)),
    )
    config = NormalizationConfig(maximum_regions=2)

    result = normalize_regions(candidates, _prepared(), _backend_provenance(), config)

    assert [region.contributor_candidate_ids for region in result.regions] == [("a",), ("b",)]
    assert result.rejected[0].candidate_id == "c"
    assert result.rejected[0].reason is RejectionReason.REGION_BUDGET_EXCEEDED
    assert result.config_digest == config.digest
