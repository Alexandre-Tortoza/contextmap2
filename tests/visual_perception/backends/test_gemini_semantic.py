"""Contract tests for the Gemini semantic interpreter adapter (fake client, no provider)."""

import json
from dataclasses import replace

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SEMANTIC_PROMPT_TEMPLATES,
    PerceptionResultId,
    RegionId,
    SemanticConfidencePolicy,
    SemanticInterpretationFailedError,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    render_semantic_prompt,
)
from contextmap.visual_perception.backends import gemini
from contextmap.visual_perception.backends.gemini import (
    GeminiProviderResponse,
    GeminiSemanticConfig,
    GeminiSemanticError,
    GeminiSemanticInterpreter,
    GeminiTransientError,
)
from contextmap.visual_perception.semantic_backend import encode_semantic_execution


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
    def __init__(
        self,
        failures: int = 0,
        confidence: float | None = None,
        terminal: Exception | None = None,
        text: str | None = None,
    ) -> None:
        self.failures = failures
        self.confidence = confidence
        self.terminal = terminal
        self.text = text
        self.calls = 0
        self.received_views: list[object] = []
        self.prompts: list[object] = []

    def generate(self, **kwargs: object) -> GeminiProviderResponse:
        self.calls += 1
        self.received_views.append(kwargs["visual_views"])
        self.prompts.append(kwargs["prompt"])
        if self.terminal is not None:
            raise self.terminal
        if self.calls <= self.failures:
            raise GeminiTransientError("rate limited")
        if self.text is not None:
            return GeminiProviderResponse(text=self.text)
        return _response(self.confidence)


def _config(**overrides: object) -> GeminiSemanticConfig:
    values: dict[str, object] = {
        "model": "gemini-2.5-pro",
        "timeout_s": 30,
        "max_retries": 2,
        "temperature": 0,
        "thinking_budget": 128,
    }
    values.update(overrides)
    return GeminiSemanticConfig(**values)  # type: ignore[arg-type]


def _adapter(
    client: _Client, *, retries: int = 2, waits: list[int] | None = None
) -> GeminiSemanticInterpreter:
    recorded = [] if waits is None else waits
    return GeminiSemanticInterpreter(
        config=_config(max_retries=retries),
        client=client,
        retry_wait=recorded.append,
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
                payload_reference="outputs/semantic-views/full.jpg",
                source_observation_id=SourceObservationId("frame-0001"),
                sha256="0" * 64,
            ),
        ),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


def _region_request(adapter: GeminiSemanticInterpreter) -> SemanticInterpretationRequest:
    region = RegionId("region-0003")
    return replace(
        _request(adapter),
        request_id=SemanticRequestId("region-0003"),
        mode=SemanticInterpretationMode.REGION,
        region_id=region,
        visual_views=(
            SemanticVisualView(
                view_id="tight",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/semantic-views/tight.jpg",
                source_observation_id=SourceObservationId("frame-0001"),
                region_id=region,
                sha256="0" * 64,
            ),
        ),
        prompt_template_id="region-abstention/v1",
    )


def test_gemini_consumes_exactly_the_prompt_policy_the_request_selects() -> None:
    """#542: the same backend-neutral policy Qwen renders, never an internal default."""
    client = _Client(text=json.dumps({"abstained": True, "claims": [], "scene_context": None}))
    adapter = _adapter(client)
    request = _region_request(adapter)

    execution = adapter.interpret(request)

    expected = render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )
    assert client.prompts == [expected.text]
    assert execution.rendered_prompt == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt_template_id": "scene/v9"}, "unknown semantic prompt template 'scene/v9'"),
        ({"prompt_template_id": "region/v1"}, "mode must match"),
        ({"requested_output_schema": "semantic-response/2"}, "schema must match"),
    ],
)
def test_gemini_refuses_a_prompt_policy_it_cannot_render_before_calling_the_provider(
    changes: dict[str, str], message: str
) -> None:
    client = _Client()
    adapter = _adapter(client)

    with pytest.raises(ValueError, match=message):
        adapter.interpret(replace(_request(adapter), **changes))  # type: ignore[arg-type]

    assert client.calls == 0


def test_gemini_retries_transient_failure_and_preserves_usage() -> None:
    waits: list[int] = []
    client = _Client(failures=1)
    adapter = _adapter(client, waits=waits)

    execution = adapter.interpret(_request(adapter))

    assert isinstance(adapter, SemanticInterpreter)
    assert execution.parsed.scene_context is not None
    assert execution.diagnostics.retries == 1
    assert execution.diagnostics.input_tokens == 10
    assert waits == [1]
    assert set(execution.effective_configuration) == {
        "model",
        "timeout_s",
        "max_retries",
        "temperature",
        "thinking_budget",
        "structured_output",
        "retry_backoff_s",
    }


def test_gemini_hands_the_full_view_identity_to_the_client_on_every_attempt() -> None:
    client = _Client(failures=1)
    adapter = _adapter(client)
    request = _request(adapter)

    adapter.interpret(request)

    # O cliente verifica o sha256 antes de enviar bytes, então recebe as views inteiras.
    assert client.received_views == [request.visual_views, request.visual_views]


def test_gemini_exhausted_retries_are_explicit_without_fallback() -> None:
    client = _Client(failures=3)
    adapter = _adapter(client, retries=1)

    with pytest.raises(GeminiSemanticError, match="exhausted"):
        adapter.interpret(_request(adapter))

    assert client.calls == 2


def test_gemini_rejects_model_reported_confidence() -> None:
    adapter = _adapter(_Client(confidence=0.93))

    # Still rejected, but the drifting response is preserved as evidence now.
    with pytest.raises(SemanticInterpretationFailedError, match="confidence must be null"):
        adapter.interpret(_request(adapter))


def test_a_terminal_provider_error_is_not_retried_and_not_replaced() -> None:
    client = _Client(terminal=GeminiSemanticError("Gemini rejected the request: 403"))
    adapter = _adapter(client)

    with pytest.raises(GeminiSemanticError, match="403"):
        adapter.interpret(_request(adapter))

    assert client.calls == 1


def test_malformed_structured_output_is_a_parser_failure_and_is_not_retried() -> None:
    client = _Client(text='{"abstained": false, "claims": [')
    adapter = _adapter(client)

    with pytest.raises(SemanticInterpretationFailedError, match="malformed") as raised:
        adapter.interpret(_request(adapter))

    assert client.calls == 1
    # A truncated response is exactly what you need preserved to diagnose the truncation.
    assert raised.value.failure.raw_response == '{"abstained": false, "claims": ['
    assert raised.value.failure.failure.kind == "SemanticResponseParseError"


def test_an_empty_response_is_an_explicit_provider_failure_not_an_abstention() -> None:
    adapter = _adapter(_Client(text="   "))

    with pytest.raises(GeminiSemanticError, match="empty or blocked"):
        adapter.interpret(_request(adapter))


def test_the_default_wait_between_attempts_backs_off_exponentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(gemini.time, "sleep", slept.append)
    client = _Client(failures=3)
    adapter = GeminiSemanticInterpreter(
        config=_config(max_retries=3, retry_backoff_s=0.5), client=client
    )

    adapter.interpret(_request(adapter))

    assert slept == [0.5, 1.0, 2.0]


def test_the_default_wait_is_capped() -> None:
    config = _config(max_retries=10, retry_backoff_s=10)

    assert gemini.backoff_seconds(config, attempt=1) == 10
    assert gemini.backoff_seconds(config, attempt=10) == gemini.MAX_BACKOFF_SECONDS


def test_structured_output_and_backoff_change_the_fingerprint_and_are_secret_free() -> None:
    plain = GeminiSemanticInterpreter(config=_config(structured_output=False), client=_Client())
    structured = GeminiSemanticInterpreter(config=_config(structured_output=True), client=_Client())

    assert plain.configuration_fingerprint != structured.configuration_fingerprint
    serialized = json.dumps(_config().to_dict()).casefold()
    assert "key" not in serialized
    assert "secret" not in serialized


def test_configuration_validates_the_new_settings() -> None:
    with pytest.raises(ValueError, match="retry_backoff_s"):
        _config(retry_backoff_s=-1)


def test_the_persisted_execution_record_carries_no_credential_field() -> None:
    adapter = _adapter(_Client())
    execution = adapter.interpret(_request(adapter))

    encoded = json.dumps(encode_semantic_execution(execution, raw_response_reference="raw.txt"))

    assert "api_key" not in encoded.casefold()
    assert "[REDACTED]" not in encoded
