import tracemalloc
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from typing import Any

import numpy as np
import pytest
from mask_cases import (
    GOLDEN,
    NORMALIZATION_SEEDS,
    inline_mask,
    normalization_case,
    normalization_digest,
)

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
from contextmap.visual_perception import normalization as normalization_module
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
        np.array(
            tuple(
                box is not None and box.x_min <= x < box.x_max and box.y_min <= y < box.y_max
                for y in range(HEIGHT)
                for x in range(WIDTH)
            ),
            dtype=bool,
        ).reshape(HEIGHT, WIDTH)
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


# --- #593: comportamento registrado e gates estruturais ------------------------------------


@pytest.mark.parametrize("seed", NORMALIZATION_SEEDS)
def test_normalization_matches_the_recorded_behaviour(seed: int) -> None:
    result = normalize_regions(*normalization_case(seed))

    assert normalization_digest(result) == GOLDEN["normalization"][str(seed)]


def _vga_candidates(boxes: list[tuple[int, int, int, int]]) -> tuple[RegionCandidate, ...]:
    candidates = []
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        pixels = np.zeros((480, 640), dtype=bool)
        pixels[y0:y1, x0:x1] = True
        candidates.append(
            RegionCandidate(
                candidate_id=f"candidate-{index:02d}",
                source_observation_id="frame-1",
                perception_run_id="run-1",
                perception_result_id="result-1",
                image_width=640,
                image_height=480,
                mask=inline_mask(pixels),
                provenance=RegionProvenance(
                    backend_id="fake",
                    backend_version="1",
                    checkpoint="fake-checkpoint",
                    config_digest="sha256:fake",
                    discovery_pass_id="full-frame",
                    native_proposal_id=f"native-{index}",
                ),
            )
        )
    return tuple(candidates)


def _vga_image() -> PreparedImage:
    return replace(_prepared(), width=640, height=480)


# Uma grade 4x3 de retângulos disjuntos de 150x150 numa imagem 640x480 (o cenário da auditoria).
_GRID = [(x * 160, y * 160, x * 160 + 150, y * 160 + 150) for y in range(3) for x in range(4)]


def test_the_normalized_geometry_holds_no_pixel_sets() -> None:
    annotations = normalization_module._Geometry.__annotations__.values()

    assert not any("frozenset" in str(annotation) for annotation in annotations)


def test_only_candidates_whose_boxes_meet_are_compared_pixel_by_pixel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compared = 0
    real = normalization_module._overlap

    def counting(*args: Any) -> tuple[float, float]:
        nonlocal compared
        compared += 1
        return real(*args)

    monkeypatch.setattr(normalization_module, "_overlap", counting)
    # A grade disjunta mais um candidato sobre o primeiro retângulo.
    candidates = _vga_candidates([*_GRID, (10, 10, 140, 140)])

    result = normalize_regions(candidates, _vga_image(), _backend_provenance())

    assert len(result.regions) == 12
    assert compared == 1


def test_normalizing_twelve_vga_masks_keeps_a_bounded_transient_peak() -> None:
    # Complemento do gate estrutural: o pico de memória transitória, medido por tracemalloc.
    # Referência medida com pixels em frozenset: 48 MB neste cenário.
    candidates = _vga_candidates(_GRID)
    image = _vga_image()
    tracemalloc.start()
    try:
        normalize_regions(candidates, image, _backend_provenance())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 3_000_000
