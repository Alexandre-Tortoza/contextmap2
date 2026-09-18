import json
from hashlib import sha256

import pytest

from contextmap.visual_perception import ArtifactReference, BoundingBox, PreparedImage
from contextmap.visual_perception.backends.florence2 import (
    Florence2Config,
    Florence2NativeOutput,
    Florence2NativeRegion,
    Florence2RegionDiscovery,
)
from contextmap.visual_perception.discovery import DiscoveryInput, DiscoveryPass, PassKind


def _input() -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id="frame-florence",
            image=ArtifactReference(
                uri="outputs/florence.png",
                sha256=sha256(b"florence").hexdigest(),
                media_type="image/png",
            ),
            width=6,
            height=4,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=6, y_max=4),
        ),
        perception_run_id="run-florence",
        perception_result_id="result-florence",
    )


class FakeFlorence2Runtime:
    def predict(
        self, discovery_input: DiscoveryInput, config: Florence2Config
    ) -> Florence2NativeOutput:
        return Florence2NativeOutput(
            regions=(
                Florence2NativeRegion(
                    proposal_id="box-1",
                    box=(1.0, 1.0, 3.0, 3.0),
                    score=None,
                    parsed_text="a requested region",
                ),
                Florence2NativeRegion(
                    proposal_id="mask-2",
                    box=(3.0, 0.0, 5.0, 2.0),
                    score=0.9,
                    mask=(False, False, False, True, True, False)
                    + (False, False, False, True, True, False)
                    + (False,) * 12,
                    parsed_text="second parsed region",
                ),
            ),
            warnings=(),
            parsing_diagnostics=(("unparsed_token_count", 0),),
        )


def test_florence2_normalizes_boxes_and_masks_without_semantic_promotion() -> None:
    config = Florence2Config(
        checkpoint="microsoft/Florence-2-large",
        model_version="1.0",
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        prompt="visible regions",
        device="cuda:0",
        generation_settings=(("num_beams", 3),),
    )
    backend = Florence2RegionDiscovery(config=config, runtime=FakeFlorence2Runtime())

    output = backend.discover(_input())

    assert len(output.candidates) == 2
    box_candidate, mask_candidate = output.candidates
    assert box_candidate.mask is None
    assert box_candidate.score is None
    assert dict(box_candidate.native_metadata)["parsed_text"] == "a requested region"
    assert mask_candidate.mask is not None
    assert mask_candidate.score is not None
    assert mask_candidate.provenance.backend_id == "florence2_region_discovery"
    assert mask_candidate.provenance.query == (
        "<REFERRING_EXPRESSION_SEGMENTATION>:visible regions"
    )
    assert dict(output.diagnostics.metadata)["task"] == config.task
    assert dict(output.diagnostics.metadata)["unparsed_token_count"] == 0
    assert "hypothesis" not in box_candidate.to_dict()
    json.dumps(box_candidate.to_dict())


def test_florence2_configuration_requires_explicit_region_task() -> None:
    with pytest.raises(ValueError, match="task"):
        Florence2Config(checkpoint="florence", task="")
    with pytest.raises(ValueError, match="box_threshold"):
        Florence2Config(checkpoint="florence", task="region", box_threshold=2.0)


def test_florence2_invalid_native_mask_fails_with_parsing_context() -> None:
    class InvalidRuntime:
        def predict(
            self, discovery_input: DiscoveryInput, config: Florence2Config
        ) -> Florence2NativeOutput:
            return Florence2NativeOutput(
                regions=(
                    Florence2NativeRegion(
                        proposal_id="bad-mask",
                        box=(0.0, 0.0, 2.0, 2.0),
                        score=0.8,
                        mask=(True,),
                    ),
                )
            )

    backend = Florence2RegionDiscovery(
        config=Florence2Config(checkpoint="florence", task="region"),
        runtime=InvalidRuntime(),
    )

    with pytest.raises(ValueError, match=r"bad-mask.*mask length"):
        backend.discover(_input())
