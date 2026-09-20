"""Shared execution records for replaceable semantic interpreter adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_prompt import (
    ParsedSemanticResponse,
    RenderedSemanticPrompt,
    SemanticParseDiagnostic,
)
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationRequest,
    decode_semantic_request,
    encode_semantic_request,
)
from contextmap.visual_perception.serialization import (
    decode_claim,
    decode_scene_context,
    encode_claim,
    encode_scene_context,
)


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


def encode_semantic_execution(
    execution: SemanticInterpretationExecution,
    *,
    raw_response_reference: str,
) -> dict[str, Any]:
    """Encode one semantic execution while keeping raw text in a referenced file."""
    return {
        "request": encode_semantic_request(execution.request),
        "rendered_prompt": {
            "template_id": execution.rendered_prompt.template_id,
            "output_schema_version": execution.rendered_prompt.output_schema_version,
            "text": execution.rendered_prompt.text,
            "fingerprint": execution.rendered_prompt.fingerprint,
        },
        "raw_response_reference": raw_response_reference,
        "raw_response": execution.raw_response,
        "raw_response_sha256": execution.parsed.raw_response_sha256,
        "parsed": {
            "claims": [encode_claim(claim) for claim in execution.parsed.claims],
            "scene_context": (
                None
                if execution.parsed.scene_context is None
                else encode_scene_context(execution.parsed.scene_context)
            ),
            "abstained": execution.parsed.abstained,
            "diagnostics": [
                {"code": diagnostic.code, "message": diagnostic.message}
                for diagnostic in execution.parsed.diagnostics
            ],
        },
        "diagnostics": {
            "latency_ms": execution.diagnostics.latency_ms,
            "input_tokens": execution.diagnostics.input_tokens,
            "output_tokens": execution.diagnostics.output_tokens,
            "peak_memory_bytes": execution.diagnostics.peak_memory_bytes,
            "retries": execution.diagnostics.retries,
            "warnings": list(execution.diagnostics.warnings),
        },
        "effective_configuration": dict(execution.effective_configuration),
    }


def decode_semantic_execution(
    record: dict[str, Any],
) -> SemanticInterpretationExecution:
    """Decode one semantic execution and verify its contractual raw response."""
    raw_response = record["raw_response"]
    if not isinstance(raw_response, str):
        raise ValueError("semantic execution raw_response must be a string")
    raw_response_sha256 = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    if raw_response_sha256 != record["raw_response_sha256"]:
        raise ValueError("semantic execution raw response hash does not match its record")
    raw_prompt = record["rendered_prompt"]
    raw_parsed = record["parsed"]
    raw_diagnostics = record["diagnostics"]
    return SemanticInterpretationExecution(
        request=decode_semantic_request(record["request"]),
        rendered_prompt=RenderedSemanticPrompt(
            template_id=raw_prompt["template_id"],
            output_schema_version=raw_prompt["output_schema_version"],
            text=raw_prompt["text"],
            fingerprint=raw_prompt["fingerprint"],
        ),
        raw_response=raw_response,
        parsed=ParsedSemanticResponse(
            raw_response_sha256=raw_response_sha256,
            claims=tuple(decode_claim(item) for item in raw_parsed["claims"]),
            scene_context=(
                None
                if raw_parsed["scene_context"] is None
                else decode_scene_context(raw_parsed["scene_context"])
            ),
            abstained=raw_parsed["abstained"],
            diagnostics=tuple(
                SemanticParseDiagnostic(code=item["code"], message=item["message"])
                for item in raw_parsed["diagnostics"]
            ),
        ),
        diagnostics=SemanticBackendDiagnostics(
            latency_ms=raw_diagnostics["latency_ms"],
            input_tokens=raw_diagnostics["input_tokens"],
            output_tokens=raw_diagnostics["output_tokens"],
            peak_memory_bytes=raw_diagnostics["peak_memory_bytes"],
            retries=raw_diagnostics["retries"],
            warnings=tuple(raw_diagnostics["warnings"]),
        ),
        effective_configuration=MappingProxyType(
            {
                key: cast(JsonScalar, value)
                for key, value in record["effective_configuration"].items()
            }
        ),
    )
