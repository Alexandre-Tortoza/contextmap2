from collections.abc import Callable
from hashlib import sha256

import pytest

from contextmap.visual_perception import (
    ArtifactReference,
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
    RegionDiscovery,
    TilingConfig,
    build_discovery_passes,
    run_discovery_passes,
)


def _image(width: int = 6, height: int = 4) -> PreparedImage:
    return PreparedImage(
        source_observation_id="frame-1",
        image=ArtifactReference(
            uri="outputs/frame.png",
            sha256=sha256(b"frame").hexdigest(),
            media_type="image/png",
        ),
        width=width,
        height=height,
        transformations=(),
    )


class FakeDiscovery:
    def __init__(self, *, touch_right_border: bool = False) -> None:
        self.inputs: list[DiscoveryInput] = []
        self.touch_right_border = touch_right_border

    def discover(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        self.inputs.append(discovery_input)
        width = int(discovery_input.discovery_pass.window.width)
        height = int(discovery_input.discovery_pass.window.height)
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

    assert isinstance(backend, RegionDiscovery)
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
