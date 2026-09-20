"""Qwen adapter for canonical scene and region Semantic Interpretation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Protocol, cast

from contextmap.visual_perception.models import (
    BackendProvenance,
    SemanticInferenceProvenance,
)
from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_backend import (
    SemanticBackendDiagnostics,
    SemanticInterpretationExecution,
)
from contextmap.visual_perception.semantic_prompt import (
    SemanticConfidencePolicy,
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


@dataclass(frozen=True, kw_only=True)
class QwenSemanticConfig:
    """Effective local Qwen generation configuration."""

    model: str
    device: str
    precision: str
    max_new_tokens: int
    temperature: float
    quantization: str | None = None

    def __post_init__(self) -> None:
        """Validate model/runtime settings before model construction."""
        for name, value in (
            ("model", self.model),
            ("device", self.device),
            ("precision", self.precision),
        ):
            if not value.strip():
                raise ValueError(f"Qwen {name} must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("Qwen max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("Qwen temperature must be non-negative")
        if self.quantization not in {None, "4bit", "8bit"}:
            raise ValueError("Qwen quantization must be None, '4bit', or '8bit'")

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return a secret-free, JSON-compatible effective configuration."""
        return cast_config(asdict(self))


@dataclass(frozen=True, kw_only=True)
class QwenGenerationResponse:
    """SDK-neutral result returned by the injected local Qwen runtime."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_memory_bytes: int | None = None
    warnings: tuple[str, ...] = ()


class QwenRuntime(Protocol):
    """Internal boundary isolating transformers/torch/Qwen native objects."""

    def generate(
        self,
        *,
        visual_payload_references: tuple[str, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        """Generate one structured response from canonical payload references."""
        ...


class QwenSemanticInterpreter:
    """Interpret canonical semantic requests with an explicitly configured Qwen runtime."""

    def __init__(self, *, config: QwenSemanticConfig, runtime: QwenRuntime) -> None:
        """Construct the adapter without loading a model outside the injected runtime."""
        self._config = config
        self._runtime = runtime
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        self.configuration_fingerprint = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )

    def backend_provenance(self) -> BackendProvenance:
        """Return Qwen model and effective-configuration identity."""
        return BackendProvenance(
            backend_id="qwen_semantic",
            capability="semantic_interpreter",
            provider="qwen",
            model=self._config.model,
            version="1",
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare the canonical evidence this adapter maps to Qwen inputs."""
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset(
                {SemanticInterpretationMode.SCENE, SemanticInterpretationMode.REGION}
            ),
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=False,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Render, execute, and parse one request without fabricating confidence."""
        validate_semantic_request(request, self.capabilities())
        if request.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("Qwen request configuration fingerprint does not match adapter")
        template = SemanticPromptTemplate.default_for(request.mode)
        rendered = render_semantic_prompt(
            request,
            template,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )
        started = time.monotonic()
        response = self._runtime.generate(
            visual_payload_references=tuple(
                view.payload_reference for view in request.visual_views
            ),
            prompt=rendered.text,
            config=self._config,
        )
        latency_ms = (time.monotonic() - started) * 1000
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"qwen-{request.mode.value}-interpretation",
            prompt_template_id=template.template_id,
            output_schema_version=template.output_schema_version,
            raw_response_reference=(
                f"debug/40-semantic-interpretation/{request.request_id}/raw-response.txt"
            ),
        )
        parsed = parse_semantic_response(
            response.text,
            request,
            provenance,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )
        configuration: Mapping[str, JsonScalar] = MappingProxyType(self._config.to_dict())
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=response.text,
            parsed=parsed,
            diagnostics=SemanticBackendDiagnostics(
                latency_ms=latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                peak_memory_bytes=response.peak_memory_bytes,
                warnings=response.warnings,
            ),
            effective_configuration=configuration,
        )


def cast_config(value: dict[str, object]) -> dict[str, JsonScalar]:
    """Narrow dataclass serialization to the scalar Qwen config contract."""
    return {key: cast(JsonScalar, item) for key, item in value.items()}
