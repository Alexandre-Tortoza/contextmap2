"""Contract tests for the Gemini semantic interpreter adapter."""

import json

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    PerceptionResultId,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.visual_perception.backends.gemini import (
    GeminiProviderResponse,
    GeminiSemanticConfig,
    GeminiSemanticError,
    GeminiSemanticInterpreter,
    GeminiTransientError,
)


def _response(confidence: float | None = None) -> GeminiProviderResponse:
    return GeminiProviderResponse(
        text=json.dumps(
            {
                "abstained": False,
                "claims": [
                    {
                        "hypothesis": "warehouse",
                        "role": "primary",
                        "category": None,
                        "region_kind": None,
                        "attributes": {},
                        "confidence": confidence,
                    }
                ],
                "scene_context": {"scene_type": "warehouse"},
            }
        ),
        input_tokens=10,
        output_tokens=8,
    )


class _Client:
    def __init__(self, failures: int = 0, confidence: float | None = None) -> None:
        self.failures = failures
        self.confidence = confidence
        self.calls = 0

    def generate(self, **kwargs: object) -> GeminiProviderResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise GeminiTransientError("rate limited")
        return _response(self.confidence)


def _adapter(client: _Client, *, retries: int = 2) -> GeminiSemanticInterpreter:
    return GeminiSemanticInterpreter(
        config=GeminiSemanticConfig(
            model="gemini-2.5-pro",
            timeout_s=30,
            max_retries=retries,
            temperature=0,
            thinking_budget=128,
        ),
        client=client,
    )


def _request(adapter: GeminiSemanticInterpreter) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("run-0001--frame-0001"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(
            SemanticVisualView(
                view_id="full",
                kind=VisualViewKind.FULL_FRAME,
                payload_reference="outputs/full.jpg",
                source_observation_id=SourceObservationId("frame-0001"),
                sha256="0" * 64,
            ),
        ),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


def test_gemini_retries_transient_failure_and_preserves_usage() -> None:
    client = _Client(failures=1)
    adapter = _adapter(client)

    execution = adapter.interpret(_request(adapter))

    assert isinstance(adapter, SemanticInterpreter)
    assert execution.parsed.scene_context is not None
    assert execution.diagnostics.retries == 1
    assert execution.diagnostics.input_tokens == 10
    assert set(execution.effective_configuration) == {
        "model",
        "timeout_s",
        "max_retries",
        "temperature",
        "thinking_budget",
    }


def test_gemini_exhausted_retries_are_explicit_without_fallback() -> None:
    client = _Client(failures=3)
    adapter = _adapter(client, retries=1)

    with pytest.raises(GeminiSemanticError, match="exhausted"):
        adapter.interpret(_request(adapter))

    assert client.calls == 2


def test_gemini_rejects_model_reported_confidence() -> None:
    adapter = _adapter(_Client(confidence=0.93))

    with pytest.raises(ValueError, match="confidence must be null"):
        adapter.interpret(_request(adapter))
