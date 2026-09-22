"""Fake/contract tests for the google-genai Gemini client using fake SDK modules.

No provider is contacted: the SDK modules are replaced, so these tests prove how the client
maps configuration, views and provider failures, not the behavior of the Gemini service.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    PerceptionResultId,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.visual_perception.backends import gemini
from contextmap.visual_perception.backends.gemini import (
    GeminiCredentialError,
    GeminiDependencyError,
    GeminiSemanticConfig,
    GeminiSemanticError,
    GeminiSemanticInterpreter,
    GeminiTransientError,
    GoogleGenAIGeminiClient,
)

KEY = "test-only-fake-credential-0123456789"
REFERENCE = "outputs/semantic-views/full.png"
IMAGE = b"image bytes"


class FakeAPIError(Exception):
    def __init__(
        self, code: int, message: str = "provider message", status: str = "STATUS"
    ) -> None:
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.status = status
        self.message = message


class FakeTimeout(Exception):
    """Stands for httpx.TimeoutException."""


class FakeTransportError(Exception):
    """Stands for httpx.TransportError."""


def _response(
    text: str | None = '{"abstained": true, "claims": [], "scene_context": null}',
    *,
    finish: str | None = "STOP",
    block: str | None = None,
    candidates: bool = True,
    usage: tuple[int | None, int | None, int | None] | None = (11, 7, 5),
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        prompt_feedback=None if block is None else SimpleNamespace(block_reason=_enum(block)),
        candidates=[SimpleNamespace(finish_reason=None if finish is None else _enum(finish))]
        if candidates
        else [],
        usage_metadata=None
        if usage is None
        else SimpleNamespace(
            prompt_token_count=usage[0],
            candidates_token_count=usage[1],
            thoughts_token_count=usage[2],
        ),
    )


def _enum(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


class FakeModels:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeSdk:
    """The four SDK modules the client imports, plus what they were asked to do."""

    def __init__(self, outcome: object | None = None) -> None:
        self.models = FakeModels(_response() if outcome is None else outcome)
        self.client_kwargs: list[dict[str, Any]] = []
        sdk = self

        class Client:
            def __init__(self, **kwargs: Any) -> None:
                sdk.client_kwargs.append(kwargs)
                self.models = sdk.models

        def record(kind: str) -> Any:
            def make(**kwargs: Any) -> SimpleNamespace:
                return SimpleNamespace(kind=kind, **kwargs)

            return make

        self.modules = {
            "google.genai": SimpleNamespace(Client=Client),
            "google.genai.types": SimpleNamespace(
                HttpOptions=record("http_options"),
                ThinkingConfig=record("thinking_config"),
                GenerateContentConfig=record("config"),
                Part=SimpleNamespace(from_bytes=record("part")),
            ),
            "google.genai.errors": SimpleNamespace(APIError=FakeAPIError),
            "httpx": SimpleNamespace(
                TimeoutException=FakeTimeout, TransportError=FakeTransportError
            ),
        }

    def install(self, monkeypatch: pytest.MonkeyPatch, *, missing: str | None = None) -> None:
        def import_module(name: str) -> ModuleType | SimpleNamespace:
            if name == missing:
                raise ModuleNotFoundError(name)
            return self.modules[name]

        monkeypatch.setattr(gemini, "importlib", SimpleNamespace(import_module=import_module))


def _config(**overrides: Any) -> GeminiSemanticConfig:
    values: dict[str, Any] = {
        "model": "gemini-2.5-pro",
        "timeout_s": 12.5,
        "max_retries": 2,
        "temperature": 0.0,
        "thinking_budget": 128,
    }
    values.update(overrides)
    return GeminiSemanticConfig(**values)


def _client(tmp_path: Path, *, image: str = REFERENCE, key: str | None = KEY) -> Any:
    path = tmp_path / image
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(IMAGE)
    return GoogleGenAIGeminiClient(view_root=tmp_path, api_key=key)


def _visual_view(reference: str = REFERENCE, payload: bytes = IMAGE) -> SemanticVisualView:
    """Describe a canonical view whose ``sha256`` identifies exactly ``payload``."""
    return SemanticVisualView(
        view_id="full",
        kind=VisualViewKind.FULL_FRAME,
        payload_reference=reference,
        source_observation_id=SourceObservationId("frame-0001"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _generate(client: Any, config: GeminiSemanticConfig, reference: str = REFERENCE) -> Any:
    return client.generate(
        visual_views=(_visual_view(reference),), prompt="canonical prompt", config=config
    )


def test_a_missing_key_fails_at_construction_and_names_only_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiCredentialError, match="GEMINI_API_KEY"):
        GoogleGenAIGeminiClient(view_root=tmp_path)
    with pytest.raises(GeminiCredentialError):
        GoogleGenAIGeminiClient(view_root=tmp_path, api_key="   ")


def test_the_key_is_read_from_the_named_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSdk()
    sdk.install(monkeypatch)
    monkeypatch.setenv("CONTEXTMAP_TEST_KEY", KEY)
    (tmp_path / REFERENCE).parent.mkdir(parents=True)
    (tmp_path / REFERENCE).write_bytes(b"image bytes")
    client = GoogleGenAIGeminiClient(view_root=tmp_path, api_key_env="CONTEXTMAP_TEST_KEY")

    _generate(client, _config())

    assert sdk.client_kwargs[0]["api_key"] == KEY


def test_the_key_never_appears_in_repr_or_str(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert KEY not in repr(client)
    assert KEY not in str(client)


def test_a_missing_sdk_is_a_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk().install(monkeypatch, missing="google.genai")

    with pytest.raises(GeminiDependencyError, match="google-genai"):
        _generate(_client(tmp_path), _config())


def test_the_request_carries_the_views_then_the_prompt_and_the_generation_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSdk()
    sdk.install(monkeypatch)

    _generate(_client(tmp_path), _config())

    (call,) = sdk.models.calls
    assert call["model"] == "gemini-2.5-pro"
    image, text = call["contents"]
    assert (image.kind, image.data, image.mime_type) == ("part", b"image bytes", "image/png")
    assert text == "canonical prompt"
    settings = call["config"]
    assert settings.temperature == 0.0
    assert settings.response_mime_type == "application/json"
    assert settings.thinking_config.thinking_budget == 128
    assert settings.http_options.timeout == 12500
    assert sdk.client_kwargs[0]["api_key"] == KEY


def test_structured_output_and_thinking_are_only_requested_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSdk()
    sdk.install(monkeypatch)

    _generate(_client(tmp_path), _config(structured_output=False, thinking_budget=None))

    settings = sdk.models.calls[0]["config"]
    assert not hasattr(settings, "response_mime_type")
    assert not hasattr(settings, "thinking_config")


@pytest.mark.parametrize(
    ("name", "mime"),
    [
        ("a.png", "image/png"),
        ("a.jpg", "image/jpeg"),
        ("a.jpeg", "image/jpeg"),
        ("a.webp", "image/webp"),
    ],
)
def test_the_image_mime_type_follows_the_payload_suffix(
    name: str, mime: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSdk()
    sdk.install(monkeypatch)
    reference = f"outputs/semantic-views/{name}"

    _generate(_client(tmp_path, image=reference), _config(), reference)

    assert sdk.models.calls[0]["contents"][0].mime_type == mime


def test_an_unsupported_format_a_linked_escape_or_a_missing_payload_is_rejected_before_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSdk()
    sdk.install(monkeypatch)
    reference = "outputs/semantic-views/a.bmp"
    with pytest.raises(GeminiSemanticError, match="format"):
        _generate(_client(tmp_path, image=reference), _config(), reference)

    root = tmp_path / "root"
    (root / "outputs" / "semantic-views").mkdir(parents=True)
    (tmp_path / "secret.png").write_bytes(IMAGE)
    (root / REFERENCE).symlink_to(tmp_path / "secret.png")
    linked = GoogleGenAIGeminiClient(view_root=root, api_key=KEY)
    with pytest.raises(GeminiSemanticError, match="escapes"):
        _generate(linked, _config())

    empty_root = GoogleGenAIGeminiClient(view_root=tmp_path / "empty", api_key=KEY)
    with pytest.raises(GeminiSemanticError, match="does not exist"):
        _generate(empty_root, _config())

    assert sdk.models.calls == []


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_rate_limits_timeouts_and_server_errors_are_transient(
    code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(FakeAPIError(code)).install(monkeypatch)

    with pytest.raises(GeminiTransientError, match=str(code)):
        _generate(_client(tmp_path), _config())


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_terminal(
    code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(FakeAPIError(code)).install(monkeypatch)

    with pytest.raises(GeminiSemanticError, match=str(code)) as caught:
        _generate(_client(tmp_path), _config())

    assert not isinstance(caught.value, GeminiTransientError)


def test_timeouts_and_transport_failures_are_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(FakeTimeout("read timed out")).install(monkeypatch)
    with pytest.raises(GeminiTransientError, match="timed out"):
        _generate(_client(tmp_path), _config(timeout_s=3))

    FakeSdk(FakeTransportError("connection reset")).install(monkeypatch)
    with pytest.raises(GeminiTransientError, match="transport"):
        _generate(_client(tmp_path), _config())


def test_provider_messages_are_redacted_and_the_original_exception_is_not_chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaking = FakeAPIError(403, message=f"API key {KEY} is not valid")
    FakeSdk(leaking).install(monkeypatch)

    with pytest.raises(GeminiSemanticError) as caught:
        _generate(_client(tmp_path), _config())

    assert KEY not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_unexpected_sdk_exceptions_are_terminal_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(ValueError(f"bad header x-goog-api-key: {KEY}")).install(monkeypatch)

    with pytest.raises(GeminiSemanticError, match="ValueError") as caught:
        _generate(_client(tmp_path), _config())

    assert KEY not in str(caught.value)
    assert not isinstance(caught.value, GeminiTransientError)


def test_text_and_usage_are_mapped_with_thinking_counted_as_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(_response(text="payload", usage=(11, 7, 5))).install(monkeypatch)

    response = _generate(_client(tmp_path), _config())

    assert response.text == "payload"
    assert response.input_tokens == 11
    assert response.output_tokens == 12
    assert response.warnings == ()


def test_missing_usage_is_reported_as_unknown_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(_response(usage=None)).install(monkeypatch)

    response = _generate(_client(tmp_path), _config())

    assert (response.input_tokens, response.output_tokens) == (None, None)


def test_a_blocked_prompt_is_a_terminal_error_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(_response(text=None, candidates=False, block="SAFETY")).install(monkeypatch)

    with pytest.raises(GeminiSemanticError, match="blocked the prompt: SAFETY"):
        _generate(_client(tmp_path), _config())


def test_no_candidates_or_a_blocking_finish_reason_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(_response(text=None, candidates=False)).install(monkeypatch)
    with pytest.raises(GeminiSemanticError, match="no candidates"):
        _generate(_client(tmp_path), _config())

    FakeSdk(_response(text="partial", finish="SAFETY")).install(monkeypatch)
    with pytest.raises(GeminiSemanticError, match="finish_reason=SAFETY"):
        _generate(_client(tmp_path), _config())


def test_an_empty_text_is_terminal_and_names_the_finish_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(_response(text=None, finish="STOP")).install(monkeypatch)

    with pytest.raises(GeminiSemanticError, match=r"empty.*STOP"):
        _generate(_client(tmp_path), _config())


def test_hitting_the_output_limit_returns_the_text_with_a_truncation_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSdk(_response(text='{"abstained"', finish="MAX_TOKENS")).install(monkeypatch)

    response = _generate(_client(tmp_path), _config())

    assert response.text == '{"abstained"'
    assert any("MAX_TOKENS" in warning for warning in response.warnings)


def test_a_view_whose_bytes_diverge_from_the_request_sha256_is_never_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSdk()
    sdk.install(monkeypatch)
    adapter = GeminiSemanticInterpreter(
        config=_config(), client=_client(tmp_path), retry_wait=lambda retry: None
    )
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("run-0001--frame-0001"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(_visual_view(payload=b"the bytes the request identifies"),),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )

    with pytest.raises(GeminiSemanticError, match="sha256"):
        adapter.interpret(request)

    assert sdk.models.calls == []
