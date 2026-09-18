import json
from hashlib import sha256

import pytest

from contextmap.visual_perception import ArtifactReference, BoundingBox, PreparedImage
from contextmap.visual_perception.backends.sam3 import (
    Sam3Config,
    Sam3NativeOutput,
    Sam3NativeProposal,
    Sam3RegionDiscovery,
    Sam3Strategy,
)
from contextmap.visual_perception.discovery import DiscoveryInput, DiscoveryPass, PassKind


def _input() -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id="frame-8",
            image=ArtifactReference(
                uri="outputs/frame-8.png",
                sha256=sha256(b"frame-8").hexdigest(),
                media_type="image/png",
            ),
            width=5,
            height=4,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=5, y_max=4),
        ),
        perception_run_id="run-sam3",
        perception_result_id="result-sam3",
    )


class FakeSam3Runtime:
    def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
        return Sam3NativeOutput(
            proposals=(
                Sam3NativeProposal(
                    proposal_id="proposal-3",
                    box=(1.0, 1.0, 4.0, 3.0),
                    mask=(False,) * 6
                    + (True, True, True, False, False)
                    + (True, True, True, False, False)
                    + (False,) * 4,
                    score_name="mask_score",
                    score=0.93,
                    query_id="query-1",
                    metadata=(("decoder_pass", 2),),
                ),
            ),
            warnings=("runtime used deterministic precision",),
            metadata=(("peak_memory_mb", 512.0),),
        )


def test_sam3_preserves_strategy_query_and_native_score_semantics() -> None:
    config = Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        device="cuda:0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompt="all movable items",
        score_threshold=0.8,
        mask_threshold=0.5,
        strategy_settings=(("max_queries", 8),),
    )
    backend = Sam3RegionDiscovery(config=config, runtime=FakeSam3Runtime())

    output = backend.discover(_input())

    candidate = output.candidates[0]
    assert candidate.score is not None
    assert candidate.score.name == "mask_score"
    assert "not calibrated" in candidate.score.semantics
    assert candidate.provenance.backend_id == "sam3"
    assert candidate.provenance.query == "text_prompt:all movable items:query-1"
    assert candidate.provenance.config_digest == config.digest
    assert dict(candidate.native_metadata)["decoder_pass"] == 2
    assert output.diagnostics.warnings == ("runtime used deterministic precision",)
    assert dict(output.diagnostics.metadata)["strategy"] == "text_prompt"
    assert dict(output.diagnostics.metadata)["peak_memory_mb"] == 512.0
    json.dumps(candidate.to_dict())


def test_sam3_strategy_configuration_is_explicit() -> None:
    with pytest.raises(ValueError, match="requires a prompt"):
        Sam3Config(checkpoint="sam3", strategy=Sam3Strategy.TEXT_PROMPT)
    with pytest.raises(ValueError, match="does not consume a text prompt"):
        Sam3Config(
            checkpoint="sam3",
            strategy=Sam3Strategy.AUTOMATIC,
            prompt="hidden architectural default",
        )
    with pytest.raises(ValueError, match="score_threshold"):
        Sam3Config(checkpoint="sam3", score_threshold=-0.1)


def test_sam3_does_not_fall_back_when_configured_runtime_fails() -> None:
    class FailingRuntime:
        def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
            raise RuntimeError("configured strategy is unavailable")

    backend = Sam3RegionDiscovery(
        config=Sam3Config(checkpoint="sam3", strategy=Sam3Strategy.AUTOMATIC),
        runtime=FailingRuntime(),
    )

    with pytest.raises(RuntimeError, match="configured strategy is unavailable"):
        backend.discover(_input())
