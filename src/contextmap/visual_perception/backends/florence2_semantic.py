"""Florence-2 adapter for canonical Semantic Interpretation requests."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from contextmap.visual_perception.models import BackendProvenance, SemanticInferenceProvenance
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
class Florence2SemanticConfig:
    """Secret-free Florence-2 semantic task and generation configuration."""

    checkpoint: str
    revision: str
    task: str
    supported_modes: frozenset[SemanticInterpretationMode]
    device: str
    precision: str
    max_new_tokens: int
    temperature: float

    def __post_init__(self) -> None:
        """Validate task, immutable checkpoint identity, and generation settings."""
        for name, value in (
            ("checkpoint", self.checkpoint),
            ("task", self.task),
            ("device", self.device),
            ("precision", self.precision),
        ):
            if not value.strip():
                raise ValueError(f"Florence-2 {name} must not be empty")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("Florence-2 revision must be a full lowercase Git commit SHA")
        if not self.supported_modes:
            raise ValueError("Florence-2 must declare at least one supported mode")
        if self.max_new_tokens <= 0:
            raise ValueError("Florence-2 max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Florence-2 temperature must be finite and non-negative")

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return a stable, SDK-free effective configuration."""
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "task": self.task,
            "supported_modes": ",".join(sorted(mode.value for mode in self.supported_modes)),
            "device": self.device,
            "precision": self.precision,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


@dataclass(frozen=True, kw_only=True)
class Florence2SemanticResponse:
    """SDK-neutral Florence-2 generation result."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_memory_bytes: int | None = None
    warnings: tuple[str, ...] = ()


class Florence2SemanticRuntime(Protocol):
    """Internal boundary around Florence-2 processor/model native objects."""

    def generate(
        self,
        *,
        visual_payload_references: tuple[str, ...],
        prompt: str,
        config: Florence2SemanticConfig,
    ) -> Florence2SemanticResponse:
        """Execute the selected semantic task and return structured text."""
        ...


class Florence2SemanticInterpreter:
    """Interpret scene/region evidence through a distinct Florence-2 capability adapter."""

    def __init__(
        self, *, config: Florence2SemanticConfig, runtime: Florence2SemanticRuntime
    ) -> None:
        """Bind configuration to a model lifecycle supplied by the composition root."""
        self._config = config
        self._runtime = runtime
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        self.configuration_fingerprint = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )

    def backend_provenance(self) -> BackendProvenance:
        """Return checkpoint/task/config identity for every produced output."""
        return BackendProvenance(
            backend_id="florence2_semantic",
            capability="semantic_interpreter",
            provider="microsoft",
            model=f"{self._config.checkpoint}@{self._config.revision}",
            version="1",
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare configured scene/region support without changing the public port."""
        return SemanticInterpreterCapabilities(
            supported_modes=self._config.supported_modes,
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=False,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Run the configured task and parse it through the shared canonical boundary."""
        validate_semantic_request(request, self.capabilities())
        if request.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("Florence-2 request configuration fingerprint does not match adapter")
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
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"florence2-{self._config.task}-{request.mode.value}",
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
            parsed=parse_semantic_response(
                response.text,
                request,
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            ),
            diagnostics=SemanticBackendDiagnostics(
                latency_ms=(time.monotonic() - started) * 1000,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                peak_memory_bytes=response.peak_memory_bytes,
                warnings=response.warnings,
            ),
            effective_configuration=configuration,
        )
