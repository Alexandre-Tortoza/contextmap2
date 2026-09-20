"""Contract tests for the Florence-2 semantic interpreter adapter."""

import json

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    PerceptionResultId,
    RegionId,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.visual_perception.backends.florence2_semantic import (
    Florence2SemanticConfig,
    Florence2SemanticInterpreter,
    Florence2SemanticResponse,
)


class _Runtime:
    def generate(self, **kwargs: object) -> Florence2SemanticResponse:
        return Florence2SemanticResponse(
            text=json.dumps(
                {
                    "abstained": False,
                    "claims": [
                        {
                            "hypothesis": "fire extinguisher",
                            "role": "primary",
                            "category": None,
                            "region_kind": "thing",
                            "attributes": {},
                            "confidence": None,
                        }
                    ],
                    "scene_context": None,
                }
            ),
            peak_memory_bytes=2048,
            warnings=("fake runtime",),
        )


def _adapter() -> Florence2SemanticInterpreter:
    return Florence2SemanticInterpreter(
        config=Florence2SemanticConfig(
            checkpoint="microsoft/Florence-2-large",
            revision="0123456789abcdef0123456789abcdef01234567",
            task="<REGION_TO_DESCRIPTION>",
            supported_modes=frozenset({SemanticInterpretationMode.REGION}),
            device="cuda:0",
            precision="bfloat16",
            max_new_tokens=128,
            temperature=0.0,
        ),
        runtime=_Runtime(),
    )


def _request(adapter: Florence2SemanticInterpreter) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("florence-region-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("result-0001"),
        mode=SemanticInterpretationMode.REGION,
        region_id=RegionId("region-0001"),
        visual_views=(
            SemanticVisualView(
                view_id="crop",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/semantic-views/region-0001.jpg",
                source_observation_id=SourceObservationId("frame-0001"),
                region_id=RegionId("region-0001"),
                sha256="0" * 64,
            ),
        ),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


def test_florence2_semantic_adapter_is_distinct_and_canonical() -> None:
    adapter = _adapter()

    execution = adapter.interpret(_request(adapter))

    assert isinstance(adapter, SemanticInterpreter)
    assert execution.parsed.claims[0].hypothesis == "fire extinguisher"
    assert execution.parsed.claims[0].confidence is None
    assert execution.parsed.claims[0].provenance.task_identity == (
        "florence2-<REGION_TO_DESCRIPTION>-region"
    )
    assert execution.effective_configuration["revision"] == (
        "0123456789abcdef0123456789abcdef01234567"
    )
    assert execution.diagnostics.peak_memory_bytes == 2048
    assert adapter.backend_provenance().backend_id == "florence2_semantic"


def test_florence2_rejects_unsupported_mode_and_invalid_configuration() -> None:
    adapter = _adapter()
    request = _request(adapter)
    scene_request = SemanticInterpretationRequest(
        **{
            **request.__dict__,
            "mode": SemanticInterpretationMode.SCENE,
            "region_id": None,
        }
    )
    with pytest.raises(ValueError, match="does not support scene"):
        adapter.interpret(scene_request)

    with pytest.raises(ValueError, match="supported mode"):
        Florence2SemanticConfig(
            checkpoint="model",
            revision="0123456789abcdef0123456789abcdef01234567",
            task="<TASK>",
            supported_modes=frozenset(),
            device="cpu",
            precision="float32",
            max_new_tokens=1,
            temperature=0,
        )
