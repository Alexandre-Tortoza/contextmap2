"""Gemini adapter for canonical scene and region Semantic Interpretation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Protocol, cast

from contextmap.visual_perception.models import BackendProvenance, SemanticInferenceProvenance
from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_backend import (
    SemanticBackendDiagnostics,
    SemanticInterpretationExecution,
)
from contextmap.visual_perception.semantic_prompt import (
    SemanticPromptTemplate,
    parse_semantic_response,
    render_semantic_prompt,
)
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    VisualViewKind,
    validate_semantic_request,
)


class GeminiTransientError(RuntimeError):
    """A retryable timeout, rate-limit, or provider/server failure."""


class GeminiSemanticError(RuntimeError):
    """A terminal Gemini semantic request failure."""


@dataclass(frozen=True, kw_only=True)
class GeminiSemanticConfig:
    """Secret-free effective Gemini request configuration."""

    model: str
    timeout_s: float
    max_retries: int
    temperature: float
    thinking_budget: int | None = None

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

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return the complete persistable configuration, which contains no secrets."""
        return {key: cast(JsonScalar, value) for key, value in asdict(self).items()}


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
        visual_payload_references: tuple[str, ...],
        prompt: str,
        config: GeminiSemanticConfig,
    ) -> GeminiProviderResponse:
        """Call Gemini or raise a typed retryable/terminal error."""
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
        """Bind a secret-owning client without accepting or persisting credentials."""
        self._config = config
        self._client = client
        self._retry_wait = retry_wait or (lambda attempt: None)
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
        rendered = render_semantic_prompt(request, template)
        started = time.monotonic()
        response: GeminiProviderResponse | None = None
        retries = 0
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.generate(
                    visual_payload_references=tuple(
                        view.payload_reference for view in request.visual_views
                    ),
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
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=response.text,
            parsed=parse_semantic_response(response.text, request, provenance),
            diagnostics=SemanticBackendDiagnostics(
                latency_ms=(time.monotonic() - started) * 1000,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                retries=retries,
                warnings=response.warnings,
            ),
            effective_configuration=configuration,
        )
