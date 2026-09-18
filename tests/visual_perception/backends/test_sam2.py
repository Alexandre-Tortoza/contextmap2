import json
from hashlib import sha256

import pytest

from contextmap.visual_perception import ArtifactReference, BoundingBox, PreparedImage
from contextmap.visual_perception.backends.sam2 import (
    Sam2Config,
    Sam2NativeProposal,
    Sam2RegionDiscovery,
)
from contextmap.visual_perception.discovery import (
    DiscoveryInput,
    DiscoveryPass,
    PassKind,
    RegionDiscovery,
)


def _input() -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id="frame-2",
            image=ArtifactReference(
                uri="outputs/frame-2.png",
                sha256=sha256(b"frame-2").hexdigest(),
                media_type="image/png",
            ),
            width=8,
            height=6,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="tile-0002",
            kind=PassKind.TILE,
            window=BoundingBox(x_min=2, y_min=1, x_max=6, y_max=5),
        ),
        perception_run_id="run-3",
        perception_result_id="result-9",
    )


class FakeSam2Runtime:
    def __init__(self) -> None:
        self.received: list[DiscoveryInput] = []

    def predict(
        self, discovery_input: DiscoveryInput, config: Sam2Config
    ) -> tuple[Sam2NativeProposal, ...]:
        self.received.append(discovery_input)
        return (
            Sam2NativeProposal(
                proposal_id="sam2-1",
                box=(0.0, 0.0, 2.0, 2.0),
                mask=(True, True, False, False) + (False,) * 12,
                predicted_iou=0.91,
                stability_score=0.87,
            ),
            Sam2NativeProposal(
                proposal_id="sam2-low",
                box=(2.0, 2.0, 4.0, 4.0),
                mask=(False,) * 10 + (True, True, False, False, True, True),
                predicted_iou=0.2,
                stability_score=0.4,
            ),
        )


def test_sam2_normalizes_native_proposals_with_complete_provenance() -> None:
    runtime = FakeSam2Runtime()
    config = Sam2Config(
        checkpoint="facebook/sam2-hiera-large",
        model_version="2.1",
        device="cuda:0",
        precision="float16",
        predicted_iou_threshold=0.8,
        stability_threshold=0.8,
        automatic_mask_settings=(("points_per_side", 32),),
    )
    backend = Sam2RegionDiscovery(config=config, runtime=runtime)

    output = backend.discover(_input())

    assert isinstance(backend, RegionDiscovery)
    assert len(runtime.received) == 1
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    assert candidate.candidate_id == "sam2-1"
    assert candidate.image_width == 4
    assert candidate.image_height == 4
    assert candidate.score is not None
    assert candidate.score.name == "predicted_iou"
    assert candidate.provenance.backend_id == "sam2"
    assert candidate.provenance.checkpoint == "facebook/sam2-hiera-large"
    assert candidate.provenance.discovery_pass_id == "tile-0002"
    assert candidate.provenance.config_digest == config.digest
    assert dict(candidate.native_metadata)["stability_score"] == 0.87
    assert dict(output.diagnostics.metadata)["raw_proposal_count"] == 2
    json.dumps(candidate.to_dict())


def test_sam2_configuration_rejects_invalid_thresholds_and_unknown_values() -> None:
    with pytest.raises(ValueError, match="predicted_iou_threshold"):
        Sam2Config(checkpoint="sam2", predicted_iou_threshold=1.1)
    with pytest.raises(ValueError, match="unique"):
        Sam2Config(
            checkpoint="sam2",
            automatic_mask_settings=(("points_per_side", 16), ("points_per_side", 32)),
        )


def test_sam2_native_shape_errors_fail_explicitly() -> None:
    class InvalidRuntime:
        def predict(
            self, discovery_input: DiscoveryInput, config: Sam2Config
        ) -> tuple[Sam2NativeProposal, ...]:
            return (
                Sam2NativeProposal(
                    proposal_id="bad",
                    box=(0.0, 0.0, 2.0, 2.0),
                    mask=(True,),
                    predicted_iou=0.9,
                    stability_score=0.9,
                ),
            )

    backend = Sam2RegionDiscovery(config=Sam2Config(checkpoint="sam2"), runtime=InvalidRuntime())

    with pytest.raises(ValueError, match="mask length"):
        backend.discover(_input())
