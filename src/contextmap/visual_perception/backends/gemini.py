"""Gemini adapter for canonical scene and region Semantic Interpretation."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, cast

from contextmap.visual_perception.backends._semantic_views import read_view_payload
from contextmap.visual_perception.models import BackendProvenance, SemanticInferenceProvenance
from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_backend import (
    SemanticBackendDiagnostics,
    SemanticInterpretationExecution,
    semantic_failure_from_parse_error,
)
from contextmap.visual_perception.semantic_prompt import (
    SemanticConfidencePolicy,
    SemanticPromptTemplate,
    SemanticResponseParseError,
    parse_semantic_response,
    render_semantic_prompt,
)
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    SemanticVisualView,
    VisualViewKind,
    validate_semantic_request,
)

MAX_BACKOFF_SECONDS = 60.0
"""Upper bound of the wait between two attempts of the same request."""

_REDACTED = "[REDACTED]"
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_TRANSIENT_HTTP_CODES = frozenset({408, 429})
_COMPLETE_FINISH_REASONS = frozenset({"STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"})


class GeminiTransientError(RuntimeError):
    """A retryable timeout, rate-limit, or provider/server failure."""


class GeminiSemanticError(RuntimeError):
    """A terminal Gemini semantic request failure."""


class GeminiDependencyError(GeminiSemanticError):
    """Raised when the optional ``google-genai`` package is unavailable."""


class GeminiCredentialError(GeminiSemanticError):
    """Raised when no Gemini API key is configured."""


@dataclass(frozen=True, kw_only=True)
class GeminiSemanticConfig:
    """Secret-free effective Gemini request configuration.

    Attributes:
        model: Provider model name, for example ``gemini-2.5-pro``.
        timeout_s: Per-attempt request timeout in seconds.
        max_retries: Retries after the first attempt for transient failures.
        temperature: Sampling temperature.
        thinking_budget: Reasoning-token budget, or ``None`` to leave the model default.
        structured_output: Request JSON output (``response_mime_type=application/json``).
            The schema itself stays in the versioned canonical prompt; provider-enforced
            schemas are not requested because the API accepts only a subset of JSON Schema.
        retry_backoff_s: Base of the exponential wait between attempts: attempt ``n`` waits
            ``retry_backoff_s * 2**(n - 1)`` seconds, capped at :data:`MAX_BACKOFF_SECONDS`.
    """

    model: str
    timeout_s: float
    max_retries: int
    temperature: float
    thinking_budget: int | None = None
    structured_output: bool = True
    retry_backoff_s: float = 1.0

    def __post_init__(self) -> None:
        """Validate provider settings without accepting credential material."""
        if not self.model.strip():
            raise ValueError("Gemini model must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("Gemini timeout_s must be positive")
        if self.max_retries < 0:
            raise ValueError("Gemini max_retries must be non-negative")
        if self.temperature < 0:
            raise ValueError("Gemini temperature must be non-negative")
        if self.thinking_budget is not None and self.thinking_budget < 0:
            raise ValueError("Gemini thinking_budget must be non-negative")
        if not math.isfinite(self.retry_backoff_s) or self.retry_backoff_s < 0:
            raise ValueError("Gemini retry_backoff_s must be finite and non-negative")

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return the complete persistable configuration, which contains no secrets."""
        return {key: cast(JsonScalar, value) for key, value in asdict(self).items()}


def backoff_seconds(config: GeminiSemanticConfig, *, attempt: int) -> float:
    """Return the wait before retry number ``attempt`` (1-based), capped."""
    return float(min(config.retry_backoff_s * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS))


@dataclass(frozen=True, kw_only=True)
class GeminiProviderResponse:
    """SDK-neutral Gemini response and usage metadata."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    warnings: tuple[str, ...] = ()


class GeminiClient(Protocol):
    """Internal seam around the provider SDK and credential handling."""

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: GeminiSemanticConfig,
    ) -> GeminiProviderResponse:
        """Call Gemini or raise a typed retryable/terminal error.

        An implementation must obtain each view's bytes through ``read_view_payload`` (or
        an equivalent check) and must not send a payload whose SHA-256 differs from
        ``SemanticVisualView.sha256``: the request identifies its evidence by that hash, and
        bytes that reach a remote provider cannot be taken back.
        """
        ...


class GeminiSemanticInterpreter:
    """Execute canonical semantic requests through a configured Gemini client."""

    def __init__(
        self,
        *,
        config: GeminiSemanticConfig,
        client: GeminiClient,
        retry_wait: Callable[[int], None] | None = None,
    ) -> None:
        """Bind a secret-owning client without accepting or persisting credentials.

        Args:
            config: Effective, secret-free configuration.
            client: Provider client that owns the credentials.
            retry_wait: Called with the 1-based retry number before each retry. It defaults
                to sleeping the exponential backoff of ``config``.
        """
        self._config = config
        self._client = client
        self._retry_wait = retry_wait or (
            lambda retry: time.sleep(backoff_seconds(config, attempt=retry))
        )
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        self.configuration_fingerprint = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )

    def backend_provenance(self) -> BackendProvenance:
        """Return provider/model/config identity without credentials."""
        return BackendProvenance(
            backend_id="gemini_semantic",
            capability="semantic_interpreter",
            provider="google",
            model=self._config.model,
            version="1",
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare scene/region image request support."""
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset(SemanticInterpretationMode),
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=False,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Call Gemini with bounded retries and canonical parsing, never fallback."""
        validate_semantic_request(request, self.capabilities())
        if request.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("Gemini request configuration fingerprint does not match adapter")
        template = SemanticPromptTemplate.default_for(request.mode)
        rendered = render_semantic_prompt(
            request,
            template,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )
        started = time.monotonic()
        response: GeminiProviderResponse | None = None
        retries = 0
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.generate(
                    visual_views=request.visual_views,
                    prompt=rendered.text,
                    config=self._config,
                )
                break
            except GeminiTransientError as error:
                if attempt >= self._config.max_retries:
                    raise GeminiSemanticError(
                        f"Gemini request exhausted {self._config.max_retries} retries"
                    ) from error
                retries += 1
                self._retry_wait(retries)
        if response is None or not response.text.strip():
            raise GeminiSemanticError("Gemini returned an empty or blocked response")
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"gemini-{request.mode.value}-interpretation",
            prompt_template_id=template.template_id,
            output_schema_version=template.output_schema_version,
            raw_response_reference=(
                f"debug/40-semantic-interpretation/{request.request_id}/raw-response.txt"
            ),
        )
        configuration: Mapping[str, JsonScalar] = MappingProxyType(self._config.to_dict())
        diagnostics = SemanticBackendDiagnostics(
            latency_ms=(time.monotonic() - started) * 1000,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            retries=retries,
            warnings=response.warnings,
        )
        try:
            parsed = parse_semantic_response(
                response.text,
                request,
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            )
        except SemanticResponseParseError as error:
            raise semantic_failure_from_parse_error(
                error,
                request=request,
                rendered_prompt=rendered,
                raw_response=response.text,
                provenance=provenance,
                diagnostics=diagnostics,
                effective_configuration=configuration,
            ) from error
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=response.text,
            parsed=parsed,
            diagnostics=diagnostics,
            effective_configuration=configuration,
        )


class GoogleGenAIGeminiClient:
    """``google-genai`` implementation of :class:`GeminiClient`.

    Constructing the client never contacts Gemini. ``generate`` sends the referenced image
    bytes and the canonical prompt to Google's Gemini API, so a caller must decide that
    those frames may leave the machine before invoking it.

    The API key is held privately, never appears in ``repr``, and is scrubbed from every
    message this client raises. Provider exceptions are not chained (``from None``) because
    their text could echo the key.
    """

    def __init__(
        self,
        *,
        view_root: Path,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        http_client: Any = None,
    ) -> None:
        """Bind credentials and the view root without importing the SDK.

        Args:
            view_root: Directory containing the run-relative ``outputs/semantic-views/``.
            api_key: Explicit key. When omitted, the ``api_key_env`` variable is read.
            api_key_env: Environment variable holding the key.
            http_client: Optional ``httpx.Client`` used by the SDK, for proxies or for tests
                with a mocked transport.

        Raises:
            GeminiCredentialError: If no non-empty key is available.
        """
        key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        if not key.strip():
            raise GeminiCredentialError(
                f"Gemini API key is not configured: set the {api_key_env} environment variable"
            )
        self._api_key = key
        self._view_root = view_root
        self._http_client = http_client
        self._sdk: Any = None
        self._sdk_client: Any = None

    def __repr__(self) -> str:
        """Describe the client without exposing the credential."""
        return f"GoogleGenAIGeminiClient(view_root={str(self._view_root)!r}, api_key={_REDACTED})"

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: GeminiSemanticConfig,
    ) -> GeminiProviderResponse:
        """Send the views and prompt to Gemini and return its text and usage.

        Every view is read and verified against ``SemanticVisualView.sha256`` before the first
        byte is handed to the SDK, so no request is made when any payload diverges from what
        the request recorded.

        Raises:
            GeminiTransientError: For timeouts, transport failures, 408/429 and 5xx.
            GeminiSemanticError: For any terminal failure: a view payload that is missing,
                escapes the view root, has an unsupported format or does not match its
                recorded SHA-256; other HTTP errors, a blocked prompt or response, no
                candidates, or empty text.
            GeminiDependencyError: If ``google-genai`` is not installed.
        """
        sdk = self._load_sdk()
        parts = [self._image_part(sdk, view) for view in visual_views]
        settings: dict[str, Any] = {
            "temperature": config.temperature,
            "http_options": sdk.types.HttpOptions(timeout=int(config.timeout_s * 1000)),
        }
        if config.structured_output:
            settings["response_mime_type"] = "application/json"
        if config.thinking_budget is not None:
            settings["thinking_config"] = sdk.types.ThinkingConfig(
                thinking_budget=config.thinking_budget
            )
        try:
            response = self._client(sdk).models.generate_content(
                model=config.model,
                contents=[*parts, prompt],
                config=sdk.types.GenerateContentConfig(**settings),
            )
        except Exception as error:
            raise self._classify(sdk, error, config) from None
        return _interpret_response(response)

    def _load_sdk(self) -> Any:
        """Import the optional SDK modules once."""
        if self._sdk is None:
            try:
                self._sdk = _Sdk(
                    genai=importlib.import_module("google.genai"),
                    types=importlib.import_module("google.genai.types"),
                    errors=importlib.import_module("google.genai.errors"),
                    httpx=importlib.import_module("httpx"),
                )
            except ModuleNotFoundError as error:
                raise GeminiDependencyError(
                    "Gemini requires the google-genai package: pip install google-genai"
                ) from error
        return self._sdk

    def _client(self, sdk: Any) -> Any:
        """Create the SDK client once, with the injected HTTP client when given."""
        if self._sdk_client is None:
            options: dict[str, Any] = {"api_key": self._api_key}
            if self._http_client is not None:
                options["http_options"] = sdk.types.HttpOptions(httpx_client=self._http_client)
            self._sdk_client = sdk.genai.Client(**options)
        return self._sdk_client

    def _image_part(self, sdk: Any, view: SemanticVisualView) -> Any:
        """Read one verified view payload and wrap it as an inline image part."""
        suffix = PurePosixPath(view.payload_reference).suffix
        mime_type = _MIME_BY_SUFFIX.get(suffix.casefold())
        if mime_type is None:
            raise GeminiSemanticError(
                f"Gemini does not accept the image format {suffix!r}: "
                f"use one of {sorted(_MIME_BY_SUFFIX)}"
            )
        try:
            payload = read_view_payload(self._view_root, view)
        except (ValueError, FileNotFoundError) as error:
            raise GeminiSemanticError(f"Gemini view payload rejected: {error}") from None
        return sdk.types.Part.from_bytes(data=payload, mime_type=mime_type)

    def _redact(self, text: str) -> str:
        """Remove the API key from text that may reach logs or exceptions."""
        return text.replace(self._api_key, _REDACTED)

    def _classify(
        self, sdk: Any, error: Exception, config: GeminiSemanticConfig
    ) -> GeminiSemanticError | GeminiTransientError:
        """Map an SDK/transport exception to a retryable or terminal Gemini error."""
        if isinstance(error, sdk.errors.APIError):
            code = int(error.code)
            detail = self._redact(f"HTTP {code} {error.status or ''}: {error.message or ''}")
            if code in _TRANSIENT_HTTP_CODES or code >= 500:
                return GeminiTransientError(f"Gemini is rate limiting or unavailable ({detail})")
            return GeminiSemanticError(f"Gemini rejected the request ({detail})")
        if isinstance(error, sdk.httpx.TimeoutException):
            return GeminiTransientError(f"Gemini request timed out after {config.timeout_s:g}s")
        if isinstance(error, sdk.httpx.TransportError):
            name = type(error).__name__
            return GeminiTransientError(
                f"Gemini transport failure: {name}: {self._redact(str(error))}"
            )
        return GeminiSemanticError(
            f"Gemini request failed with {type(error).__name__}: {self._redact(str(error))}"
        )


@dataclass(frozen=True, kw_only=True)
class _Sdk:
    """The lazily imported SDK modules used by the client."""

    genai: Any
    types: Any
    errors: Any
    httpx: Any


def _enum_name(value: Any) -> str:
    """Return the member name of an SDK enum, or its text."""
    return str(getattr(value, "name", value))


def _interpret_response(response: Any) -> GeminiProviderResponse:
    """Turn a provider response into text and usage, or a terminal error.

    A blocked prompt, a response stopped for a reason other than completion, a response
    without candidates, and empty text are terminal: none is a semantic abstention.
    Reaching the output limit returns the (probably truncated) text with a warning.
    """
    feedback = getattr(response, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None)
    if block is not None and _enum_name(block) != "BLOCKED_REASON_UNSPECIFIED":
        raise GeminiSemanticError(f"Gemini blocked the prompt: {_enum_name(block)}")
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise GeminiSemanticError("Gemini returned no candidates")
    finish_value = getattr(candidates[0], "finish_reason", None)
    finish = None if finish_value is None else _enum_name(finish_value)
    if finish is not None and finish not in _COMPLETE_FINISH_REASONS:
        raise GeminiSemanticError(f"Gemini blocked the response: finish_reason={finish}")
    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise GeminiSemanticError(f"Gemini returned an empty response (finish_reason={finish})")
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None)
    answer_tokens = getattr(usage, "candidates_token_count", None)
    thinking_tokens = getattr(usage, "thoughts_token_count", None)
    # Os tokens de raciocínio são faturados como saída, então entram na contagem de saída.
    output_tokens = (
        None
        if answer_tokens is None and thinking_tokens is None
        else (answer_tokens or 0) + (thinking_tokens or 0)
    )
    warnings = (
        ("finish_reason=MAX_TOKENS; the structured response may be truncated",)
        if finish == "MAX_TOKENS"
        else ()
    )
    return GeminiProviderResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        warnings=warnings,
    )
