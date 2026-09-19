"""Shared execution records for replaceable semantic interpreter adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_prompt import (
    ParsedSemanticResponse,
    RenderedSemanticPrompt,
)
from contextmap.visual_perception.semantic_requests import SemanticInterpretationRequest


@dataclass(frozen=True, kw_only=True)
class SemanticBackendDiagnostics:
    """Comparable runtime diagnostics for one semantic backend call."""

    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_memory_bytes: int | None = None
    retries: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject negative durations, counters, or memory measurements."""
        if self.latency_ms < 0:
            raise ValueError("semantic backend latency_ms must be non-negative")
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("peak_memory_bytes", self.peak_memory_bytes),
            ("retries", self.retries),
        ):
            if value is not None and value < 0:
                raise ValueError(f"semantic backend {name} must be non-negative")


@dataclass(frozen=True, kw_only=True)
class SemanticInterpretationExecution:
    """Transient result separating request, raw response, parsing, and metrics."""

    request: SemanticInterpretationRequest
    rendered_prompt: RenderedSemanticPrompt
    raw_response: str
    parsed: ParsedSemanticResponse
    diagnostics: SemanticBackendDiagnostics
    effective_configuration: Mapping[str, JsonScalar]
