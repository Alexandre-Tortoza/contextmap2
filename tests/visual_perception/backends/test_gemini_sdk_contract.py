"""Fake/contract tests running the real google-genai SDK against a mocked HTTP transport.

The SDK is optional, so these tests skip when it is not installed. When it is, nothing
leaves the process: ``httpx.MockTransport`` answers every request, which pins the request
shape and the mapping of real SDK errors and responses without contacting Gemini. They
complement ``test_gemini_client.py``, whose fake modules must keep matching this behavior.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("google.genai")

from contextmap.ingestion import SourceObservationId  # noqa: E402
from contextmap.visual_perception import (  # noqa: E402
    PerceptionResultId,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.visual_perception.backends.gemini import (  # noqa: E402
    GeminiSemanticConfig,
    GeminiSemanticError,
    GeminiSemanticInterpreter,
    GeminiTransientError,
    GoogleGenAIGeminiClient,
)
from contextmap.visual_perception.semantic_backend import encode_semantic_execution  # noqa: E402

KEY = "test-only-fake-credential-0123456789"
REFERENCE = "outputs/semantic-views/full.png"
PNG = b"\x89PNG\r\n\x1a\n-fake-pixels"
CANONICAL = json.dumps(
    {
        "abstained": False,
        "claims": [
            {
                "hypothesis": "corridor",
                "role": "primary",
                "category": None,
                "region_kind": None,
                "attributes": {},
                "confidence": None,
            }
        ],
        "scene_context": {"scene_type": "corridor"},
    }
)

Handler = Callable[[Any], Any]


def _ok(text: str = CANONICAL) -> Any:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "thoughtsTokenCount": 3,
                "totalTokenCount": 21,
            },
        },
    )


def _error(code: int, message: str, status: str) -> Any:
    return httpx.Response(
        code, json={"error": {"code": code, "message": message, "status": status}}
    )


def _client(tmp_path: Path, handler: Handler) -> GoogleGenAIGeminiClient:
    (tmp_path / REFERENCE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / REFERENCE).write_bytes(PNG)
    return GoogleGenAIGeminiClient(
        view_root=tmp_path,
        api_key=KEY,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _config(**overrides: Any) -> GeminiSemanticConfig:
    values: dict[str, Any] = {
        "model": "gemini-2.5-flash",
        "timeout_s": 7.0,
        "max_retries": 2,
        "temperature": 0.0,
        "thinking_budget": 0,
    }
    values.update(overrides)
    return GeminiSemanticConfig(**values)


def _visual_view(payload: bytes = PNG) -> SemanticVisualView:
    """Describe a canonical view whose ``sha256`` identifies exactly ``payload``."""
    return SemanticVisualView(
        view_id="full",
        kind=VisualViewKind.FULL_FRAME,
        payload_reference=REFERENCE,
        source_observation_id=SourceObservationId("frame-1"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _generate(client: GoogleGenAIGeminiClient, config: GeminiSemanticConfig | None = None) -> Any:
    return client.generate(
        visual_views=(_visual_view(),),
        prompt="canonical prompt",
        config=config or _config(),
    )


def test_the_request_reaches_the_transport_with_the_key_only_in_the_header(
    tmp_path: Path,
) -> None:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return _ok()

    _generate(_client(tmp_path, handler))

    (request,) = seen
    assert request.url.path.endswith("/models/gemini-2.5-flash:generateContent")
    assert request.headers["x-goog-api-key"] == KEY
    body = request.content.decode()
    assert KEY not in body
    payload = json.loads(body)
    parts = payload["contents"][0]["parts"]
    inline = parts[0]["inlineData"]
    # O SDK converte ``inline_data`` para camelCase mas deixa ``mime_type`` em snake_case
    # (observado no google-genai 2.25.0; a mesma inconsistência aparece no
    # ``thinking_budget`` abaixo). A grafia da chave é escolha do SDK, não do nosso
    # contrato: o que este teste fixa é o tipo declarado e os bytes exatos.
    assert (inline.get("mimeType") or inline["mime_type"]) == "image/png"
    # Os bytes enviados são exatamente os verificados contra o sha256 da view.
    assert base64.b64decode(inline["data"]) == PNG
    assert parts[1]["text"] == "canonical prompt"
    generation = payload["generationConfig"]
    assert generation["temperature"] == 0.0
    assert generation["responseMimeType"] == "application/json"
    assert generation["thinkingConfig"]["thinking_budget"] == 0
    assert request.extensions["timeout"]["read"] == 7.0


def test_text_and_usage_are_mapped_from_a_real_sdk_response(tmp_path: Path) -> None:
    response = _generate(_client(tmp_path, lambda request: _ok()))

    assert json.loads(response.text)["claims"][0]["hypothesis"] == "corridor"
    assert response.input_tokens == 10
    assert response.output_tokens == 11


@pytest.mark.parametrize("code", [429, 500, 503])
def test_real_rate_limit_and_server_errors_are_transient(code: int, tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: _error(code, "try later", "UNAVAILABLE"))

    with pytest.raises(GeminiTransientError, match=str(code)):
        _generate(client)


def test_a_real_sdk_timeout_is_transient(tmp_path: Path) -> None:
    def handler(request: Any) -> Any:
        raise httpx.ReadTimeout("read timed out", request=request)

    with pytest.raises(GeminiTransientError, match="timed out"):
        _generate(_client(tmp_path, handler))


@pytest.mark.parametrize("code", [400, 403])
def test_real_client_errors_are_terminal_and_the_key_is_redacted(code: int, tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: _error(code, f"API key {KEY} not valid", "DENIED"))

    with pytest.raises(GeminiSemanticError) as caught:
        _generate(client)

    assert not isinstance(caught.value, GeminiTransientError)
    assert KEY not in str(caught.value)
    assert caught.value.__cause__ is None


def test_a_real_blocked_prompt_and_a_safety_stop_are_terminal(tmp_path: Path) -> None:
    blocked = httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(GeminiSemanticError, match="blocked the prompt"):
        _generate(_client(tmp_path, lambda request: blocked))

    stopped = httpx.Response(200, json={"candidates": [{"finishReason": "SAFETY"}]})
    with pytest.raises(GeminiSemanticError, match="SAFETY"):
        _generate(_client(tmp_path, lambda request: stopped))


def test_the_adapter_retries_a_real_429_then_succeeds_without_persisting_the_key(
    tmp_path: Path,
) -> None:
    answers = iter([_error(429, "slow down", "RESOURCE_EXHAUSTED"), _ok()])
    adapter = GeminiSemanticInterpreter(
        config=_config(),
        client=_client(tmp_path, lambda request: next(answers)),
        retry_wait=lambda attempt: None,
    )
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-1"),
        source_observation_id=SourceObservationId("frame-1"),
        perception_result_id=PerceptionResultId("run--frame-1"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(_visual_view(),),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )

    execution = adapter.interpret(request)

    assert execution.diagnostics.retries == 1
    assert execution.parsed.scene_context is not None
    encoded = json.dumps(encode_semantic_execution(execution, raw_response_reference="raw.txt"))
    assert KEY not in encoded


def test_a_view_whose_bytes_diverge_from_the_request_sha256_never_reaches_the_transport(
    tmp_path: Path,
) -> None:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return _ok()

    adapter = GeminiSemanticInterpreter(
        config=_config(), client=_client(tmp_path, handler), retry_wait=lambda attempt: None
    )
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-1"),
        source_observation_id=SourceObservationId("frame-1"),
        perception_result_id=PerceptionResultId("run--frame-1"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(_visual_view(b"the bytes the request identifies"),),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )

    with pytest.raises(GeminiSemanticError, match="sha256"):
        adapter.interpret(request)

    assert seen == []
