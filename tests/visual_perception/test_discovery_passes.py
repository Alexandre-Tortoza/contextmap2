from collections.abc import Callable
from hashlib import sha256

import pytest
from mask_cases import GOLDEN, REMAP_SEEDS, digest, remap_case

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox,
    InlineMask,
    PreparedImage,
    RegionCandidate,
    RegionProvenance,
)
from contextmap.visual_perception.discovery import (
    BackendDiagnostics,
    BorderPolicy,
    DiscoveryInput,
    DiscoveryOutput,
    DiscoveryPassConfig,
    PassKind,
    RegionCandidateDiscovery,
    TilingConfig,
    _expand_mask,
    _resize_mask,
    build_discovery_passes,
    run_discovery_passes,
)


def _image(width: int = 6, height: int = 4) -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-1"),
        payload_reference="outputs/frame.png",
        payload_artifact=ArtifactReference(
            uri="outputs/frame.png",
            sha256=sha256(b"frame").hexdigest(),
            media_type="image/png",
        ),
        width=width,
        height=height,
        transformations=(),
    )


class FakeDiscovery:
    def __init__(
        self, *, touch_right_border: bool = False, relative_geometry: bool = False
    ) -> None:
        self.inputs: list[DiscoveryInput] = []
        self.touch_right_border = touch_right_border
        self.relative_geometry = relative_geometry

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="test",
            model="none",
            version="1",
            configuration_fingerprint="sha256:fake",
        )

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        self.inputs.append(discovery_input)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        if self.relative_geometry:
            box = BoundingBox(
                x_min=width * 0.25,
                y_min=height * 0.25,
                x_max=width * 0.5,
                y_max=height * 0.5,
            )
        else:
            x_max = float(width) if self.touch_right_border else min(2.0, float(width))
            box = BoundingBox(x_min=1.0, y_min=1.0, x_max=x_max, y_max=min(2.0, height))
        mask_data = tuple(
            box.x_min <= x < box.x_max and box.y_min <= y < box.y_max
            for y in range(height)
            for x in range(width)
        )
        candidate = RegionCandidate(
            candidate_id="proposal-1",
            source_observation_id=discovery_input.prepared_image.source_observation_id,
            perception_run_id=discovery_input.perception_run_id,
            perception_result_id=discovery_input.perception_result_id,
            image_width=width,
            image_height=height,
            bounding_box=box,
            mask=InlineMask(width=width, height=height, data=mask_data),
            provenance=RegionProvenance(
                backend_id="fake",
                backend_version="1",
                checkpoint="none",
                config_digest="sha256:fake",
                discovery_pass_id=discovery_input.discovery_pass.pass_id,
                native_proposal_id="proposal-1",
            ),
        )
        return DiscoveryOutput(
            candidates=(candidate,),
            diagnostics=BackendDiagnostics(
                duration_ms=1.5,
                proposal_count=1,
                warnings=(),
            ),
        )


def test_full_frame_is_the_default_backend_neutral_pass() -> None:
    backend = FakeDiscovery()

    result = run_discovery_passes(
        prepared_image=_image(),
        backend=backend,
        perception_run_id="run-1",
        perception_result_id="result-1",
    )

    assert isinstance(backend, RegionCandidateDiscovery)
    assert len(backend.inputs) == 1
    assert backend.inputs[0].discovery_pass.kind is PassKind.FULL_FRAME
    assert result.candidates[0].image_width == 6
    assert result.candidates[0].candidate_id == "full-frame/proposal-1"
    assert result.rejected == ()


def test_overlapping_tiles_are_deterministic_and_remapped_to_global_coordinates() -> None:
    backend = FakeDiscovery()
    config = DiscoveryPassConfig(
        tiling=TilingConfig(tile_width=4, tile_height=3, overlap_x=1, overlap_y=1),
        include_full_frame=False,
    )

    passes = build_discovery_passes(_image(), config)
    result = run_discovery_passes(
        prepared_image=_image(),
        backend=backend,
        perception_run_id="run-1",
        perception_result_id="result-1",
        config=config,
    )

    assert [item.pass_id for item in passes] == ["tile-0000", "tile-0001", "tile-0002", "tile-0003"]
    assert [item.window.to_dict() for item in passes] == [
        {"x_min": 0.0, "y_min": 0.0, "x_max": 4.0, "y_max": 3.0},
        {"x_min": 2.0, "y_min": 0.0, "x_max": 6.0, "y_max": 3.0},
        {"x_min": 0.0, "y_min": 1.0, "x_max": 4.0, "y_max": 4.0},
        {"x_min": 2.0, "y_min": 1.0, "x_max": 6.0, "y_max": 4.0},
    ]
    assert result.candidates[1].bounding_box == BoundingBox(
        x_min=3.0, y_min=1.0, x_max=4.0, y_max=2.0
    )
    assert result.candidates[3].mask is not None
    assert result.candidates[3].image_width == 6
    assert result.candidates[3].provenance.discovery_pass_id == "tile-0003"


def test_internal_tile_border_rejection_is_explicit_and_auditable() -> None:
    config = DiscoveryPassConfig(
        tiling=TilingConfig(
            tile_width=4,
            tile_height=4,
            overlap_x=1,
            overlap_y=0,
            border_policy=BorderPolicy.REJECT_INTERNAL_BORDER,
        ),
        include_full_frame=False,
    )

    result = run_discovery_passes(
        prepared_image=_image(width=6, height=4),
        backend=FakeDiscovery(touch_right_border=True),
        perception_run_id="run-1",
        perception_result_id="result-1",
        config=config,
    )

    assert [item.candidate_id for item in result.rejected] == ["tile-0000/proposal-1"]
    assert result.rejected[0].reason.value == "tile_border_truncation"
    assert [item.candidate_id for item in result.candidates] == ["tile-0001/proposal-1"]


def test_tile_scale_changes_model_input_and_preserves_global_coordinates() -> None:
    baseline_backend = FakeDiscovery(relative_geometry=True)
    scaled_backend = FakeDiscovery(relative_geometry=True)
    baseline_config = DiscoveryPassConfig(
        include_full_frame=False,
        tiling=TilingConfig(tile_width=4, tile_height=4, scale=1.0),
    )
    scaled_config = DiscoveryPassConfig(
        include_full_frame=False,
        tiling=TilingConfig(tile_width=4, tile_height=4, scale=2.0),
    )

    baseline = run_discovery_passes(
        prepared_image=_image(width=4, height=4),
        backend=baseline_backend,
        perception_run_id="run-1",
        perception_result_id="result-1",
        config=baseline_config,
    )
    scaled = run_discovery_passes(
        prepared_image=_image(width=4, height=4),
        backend=scaled_backend,
        perception_run_id="run-1",
        perception_result_id="result-1",
        config=scaled_config,
    )

    assert baseline_backend.inputs[0].discovery_pass.input_width == 4
    assert scaled_backend.inputs[0].discovery_pass.input_width == 8
    assert baseline.candidates[0].bounding_box == scaled.candidates[0].bounding_box
    assert baseline.candidates[0].mask == scaled.candidates[0].mask


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: TilingConfig(tile_width=0, tile_height=2), "positive"),
        (lambda: TilingConfig(tile_width=2, tile_height=2, overlap_x=2), "smaller"),
    ],
)
def test_invalid_tiling_configuration_fails_before_backend_execution(
    factory: Callable[[], TilingConfig], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("seed", REMAP_SEEDS)
def test_tile_mask_remapping_matches_the_recorded_behaviour(seed: int) -> None:
    # #593: resize por vizinho mais próximo e expansão na imagem, registrados antes da troca.
    mask, (window_width, window_height), (image_width, image_height, x0, y0) = remap_case(seed)

    resized = _resize_mask(mask, window_width, window_height)
    expanded = _expand_mask(resized, image_width, image_height, x0, y0)

    remapped = {"resized": resized.to_dict(), "expanded": expanded.to_dict()}
    assert digest(remapped) == GOLDEN["remap"][str(seed)]
