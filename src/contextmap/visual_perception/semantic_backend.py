"""Shared execution records for replaceable semantic interpreter adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from contextmap.visual_perception.models import SemanticInferenceProvenance
from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_audit import redact_semantic_secrets
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
class SemanticVisualInputMeasurement:
    """What one visual view became at the model input, measured from the backend's processor.

    It is an execution diagnostic, not configuration: the same budget gives different sizes
    for views of different shapes, so it never enters a configuration fingerprint.

    Attributes:
        view_id: The ``SemanticVisualView.view_id`` this measurement describes.
        height_px: Image height, in pixels, after the backend's own resize.
        width_px: Image width, in pixels, after the backend's own resize.
        visual_tokens: Language-model tokens this view occupied in the prompt.
    """

    view_id: str
    height_px: int
    width_px: int
    visual_tokens: int

    def __post_init__(self) -> None:
        """Reject a measurement that cannot describe a real model input."""
        if not self.view_id.strip():
            raise ValueError("visual input view_id must not be empty")
        for name, value in (("height_px", self.height_px), ("width_px", self.width_px)):
            if value <= 0:
                raise ValueError(f"visual input {name} must be positive")
        if self.visual_tokens < 0:
            raise ValueError("visual input visual_tokens must be non-negative")


@dataclass(frozen=True, kw_only=True)
class SemanticBackendDiagnostics:
    """Comparable runtime diagnostics for one semantic backend call.

    Attributes:
        latency_ms: Wall-clock duration of the backend call.
        input_tokens: Prompt tokens, visual tokens included, when the backend measures them.
        output_tokens: Generated tokens, when the backend measures them.
        peak_memory_bytes: Peak memory the backend reports for the call.
        retries: Retried attempts before this outcome.
        warnings: Backend warnings about the call.
        visual_inputs: One measurement per request view, in request order, or ``None`` when
            the backend does not measure what its processor made of the views.
    """

    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_memory_bytes: int | None = None
    retries: int = 0
    warnings: tuple[str, ...] = ()
    visual_inputs: tuple[SemanticVisualInputMeasurement, ...] | None = None

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


def encode_semantic_backend_diagnostics(diagnostics: SemanticBackendDiagnostics) -> dict[str, Any]:
    """Encode diagnostics exactly as every semantic record and the human audit persist them."""
    return {
        "latency_ms": diagnostics.latency_ms,
        "input_tokens": diagnostics.input_tokens,
        "output_tokens": diagnostics.output_tokens,
        "peak_memory_bytes": diagnostics.peak_memory_bytes,
        "retries": diagnostics.retries,
        "warnings": list(diagnostics.warnings),
        "visual_inputs": (
            None
            if diagnostics.visual_inputs is None
            else [
                {
                    "view_id": item.view_id,
                    "height_px": item.height_px,
                    "width_px": item.width_px,
                    "visual_tokens": item.visual_tokens,
                }
                for item in diagnostics.visual_inputs
            ]
        ),
    }


def _decode_diagnostics(record: Mapping[str, Any]) -> SemanticBackendDiagnostics:
    """Decode persisted diagnostics; a record without ``visual_inputs`` never measured them."""
    # Registros gravados antes do #526 (schema 0.5.0 do run artifact) não mediam as views:
    # a ausência significa exatamente "não medido", como null, e mantém legível a evidência já
    # congelada sob a mesma versão de schema.
    raw_visual_inputs = record.get("visual_inputs")
    return SemanticBackendDiagnostics(
        latency_ms=record["latency_ms"],
        input_tokens=record["input_tokens"],
        output_tokens=record["output_tokens"],
        peak_memory_bytes=record["peak_memory_bytes"],
        retries=record["retries"],
        warnings=tuple(record["warnings"]),
        visual_inputs=(
            None
            if raw_visual_inputs is None
            else tuple(
                SemanticVisualInputMeasurement(
                    view_id=item["view_id"],
                    height_px=item["height_px"],
                    width_px=item["width_px"],
                    visual_tokens=item["visual_tokens"],
                )
                for item in raw_visual_inputs
            )
        ),
    )


def _require_measured_views(
    request: SemanticInterpretationRequest, diagnostics: SemanticBackendDiagnostics
) -> None:
    """Refuse visual input measurements that are not exactly the request's views, in order.

    Raises:
        ValueError: If a view is missing, extra, or out of order in the measurements.
    """
    if diagnostics.visual_inputs is None:
        return
    measured = [item.view_id for item in diagnostics.visual_inputs]
    expected = [view.view_id for view in request.visual_views]
    if measured != expected:
        raise ValueError(
            f"visual input measurements {measured} do not describe the request views "
            f"{expected} in order"
        )


def _require_selected_prompt(
    request: SemanticInterpretationRequest, rendered_prompt: RenderedSemanticPrompt
) -> None:
    """Refuse attempt evidence whose rendered prompt is not the policy the request selected.

    Raises:
        ValueError: If the template identity or the output schema differs from the request's.
    """
    if rendered_prompt.template_id != request.prompt_template_id:
        raise ValueError(
            f"rendered prompt {rendered_prompt.template_id!r} is not the prompt template the "
            f"request selected ({request.prompt_template_id!r})"
        )
    if rendered_prompt.output_schema_version != request.requested_output_schema:
        raise ValueError(
            f"rendered prompt output schema {rendered_prompt.output_schema_version!r} is not "
            f"the one the request selected ({request.requested_output_schema!r})"
        )


@dataclass(frozen=True, kw_only=True)
class SemanticParseFailure:
    """Why a real, observed backend response could not become canonical semantic evidence.

    Attributes:
        kind: The parser exception's type name, verbatim. No taxonomy is invented here:
            classifying failure modes is an evaluation concern, and inventing categories
            before the evidence exists would be guessing.
        message: The parser's own message, verbatim.
    """

    kind: str
    message: str

    def __post_init__(self) -> None:
        """Reject an empty failure kind or message."""
        if not self.kind.strip():
            raise ValueError("semantic parse failure kind must not be empty")
        if not self.message.strip():
            raise ValueError("semantic parse failure message must not be empty")


@dataclass(frozen=True, kw_only=True)
class FailedSemanticInterpretation:
    """A real backend call whose response was observed but never materialized into claims.

    This is evidence, not a log line. The model was invoked, it answered, and the answer is
    preserved verbatim together with everything needed to audit it; only the parse into
    :class:`ParsedSemanticResponse` failed. It shares its attempt identity
    (``request.request_id``) with the success stream, so ``attempted`` can be reconciled
    against ``parsed`` without guessing.

    ``raw_response_sha256`` is computed exactly as the success path computes it, so the
    identity of the observed evidence never depends on whether the parser worked.

    Attributes:
        request: The canonical request that was issued.
        rendered_prompt: The exact prompt sent, with the template id and the output schema
            version that the parser was holding the response to.
        raw_response: The exact response text the backend produced.
        raw_response_sha256: SHA-256 of ``raw_response``, same rule as the success stream.
        provenance: Backend, model and version that produced the response.
        diagnostics: The backend's own runtime diagnostics for the call.
        failure: Why the parse was rejected.
        effective_configuration: The backend configuration in effect, redacted on encode.
        occurred_at: ISO 8601 UTC time of the attempt. Recorded for auditing; it is not part
            of the evidence's content identity, so run-to-run comparison must exclude it.
    """

    request: SemanticInterpretationRequest
    rendered_prompt: RenderedSemanticPrompt
    raw_response: str
    raw_response_sha256: str
    provenance: SemanticInferenceProvenance
    diagnostics: SemanticBackendDiagnostics
    failure: SemanticParseFailure
    effective_configuration: Mapping[str, JsonScalar]
    occurred_at: str

    def __post_init__(self) -> None:
        """Require the recorded hash, prompt and view measurements to match their request."""
        expected = hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest()
        if self.raw_response_sha256 != expected:
            raise ValueError("raw_response_sha256 does not match raw_response")
        _require_selected_prompt(self.request, self.rendered_prompt)
        _require_measured_views(self.request, self.diagnostics)


def encode_failed_semantic_interpretation(
    failed: FailedSemanticInterpretation,
) -> dict[str, Any]:
    """Encode one failed interpretation for ``outputs/semantic-interpretation-failures.jsonl``."""
    return {
        "request": encode_semantic_request(failed.request),
        "rendered_prompt": {
            "template_id": failed.rendered_prompt.template_id,
            "output_schema_version": failed.rendered_prompt.output_schema_version,
            "text": failed.rendered_prompt.text,
            "fingerprint": failed.rendered_prompt.fingerprint,
        },
        "raw_response": failed.raw_response,
        "raw_response_sha256": failed.raw_response_sha256,
        "provenance": {
            "backend": {
                "backend_id": failed.provenance.backend.backend_id,
                "capability": failed.provenance.backend.capability,
                "provider": failed.provenance.backend.provider,
                "model": failed.provenance.backend.model,
                "version": failed.provenance.backend.version,
                "configuration_fingerprint": failed.provenance.backend.configuration_fingerprint,
            },
            "task_identity": failed.provenance.task_identity,
            "prompt_template_id": failed.provenance.prompt_template_id,
            "output_schema_version": failed.provenance.output_schema_version,
            "raw_response_reference": failed.provenance.raw_response_reference,
        },
        "diagnostics": encode_semantic_backend_diagnostics(failed.diagnostics),
        "parse_failure": {"kind": failed.failure.kind, "message": failed.failure.message},
        "effective_configuration": redact_semantic_secrets(dict(failed.effective_configuration)),
        "occurred_at": failed.occurred_at,
    }


def decode_failed_semantic_interpretation(
    record: dict[str, Any],
) -> FailedSemanticInterpretation:
    """Decode one failed interpretation and verify its recorded response hash."""
    from contextmap.visual_perception.models import BackendProvenance

    raw_response = record["raw_response"]
    if not isinstance(raw_response, str):
        raise ValueError("failed semantic interpretation raw_response must be a string")
    raw_prompt = record["rendered_prompt"]
    raw_provenance = record["provenance"]
    raw_backend = raw_provenance["backend"]
    raw_failure = record["parse_failure"]
    return FailedSemanticInterpretation(
        request=decode_semantic_request(record["request"]),
        rendered_prompt=RenderedSemanticPrompt(
            template_id=raw_prompt["template_id"],
            output_schema_version=raw_prompt["output_schema_version"],
            text=raw_prompt["text"],
            fingerprint=raw_prompt["fingerprint"],
        ),
        raw_response=raw_response,
        raw_response_sha256=record["raw_response_sha256"],
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id=raw_backend["backend_id"],
                capability=raw_backend["capability"],
                provider=raw_backend["provider"],
                model=raw_backend["model"],
                version=raw_backend["version"],
                configuration_fingerprint=raw_backend["configuration_fingerprint"],
            ),
            task_identity=raw_provenance["task_identity"],
            prompt_template_id=raw_provenance["prompt_template_id"],
            output_schema_version=raw_provenance["output_schema_version"],
            raw_response_reference=raw_provenance["raw_response_reference"],
        ),
        diagnostics=_decode_diagnostics(record["diagnostics"]),
        failure=SemanticParseFailure(kind=raw_failure["kind"], message=raw_failure["message"]),
        effective_configuration=MappingProxyType(dict(record["effective_configuration"])),
        occurred_at=record["occurred_at"],
    )


class SemanticInterpretationFailedError(RuntimeError):
    """Raised when a backend produced a real response the parser could not materialize.

    ``interpret()`` keeps returning only materialized executions, so the success contract is
    unchanged; a caller that wants the evidence catches this and persists
    :attr:`failure` through
    :meth:`~contextmap.visual_perception.PerceptionRunWriter.add_failed_semantic_interpretation`.

    Attributes:
        failure: The observed-but-unparsed interpretation, complete enough to audit.
    """

    def __init__(self, failure: FailedSemanticInterpretation) -> None:
        """Create the error around one preserved failed interpretation."""
        super().__init__(f"{failure.failure.kind}: {failure.failure.message}")
        self.failure = failure


def semantic_failure_from_parse_error(
    error: Exception,
    *,
    request: SemanticInterpretationRequest,
    rendered_prompt: RenderedSemanticPrompt,
    raw_response: str,
    provenance: SemanticInferenceProvenance,
    diagnostics: SemanticBackendDiagnostics,
    effective_configuration: Mapping[str, JsonScalar],
    occurred_at: str | None = None,
) -> SemanticInterpretationFailedError:
    """Wrap a parser rejection together with everything needed to audit the attempt.

    Args:
        error: The parser's own exception; its type name and message are kept verbatim.
        request: The request that was issued.
        rendered_prompt: The prompt actually sent.
        raw_response: The exact response the backend produced.
        provenance: Backend, model and version behind the response.
        diagnostics: The backend's runtime diagnostics for the call.
        effective_configuration: Backend configuration in effect.
        occurred_at: ISO 8601 UTC time of the attempt; defaults to now.

    Returns:
        The error to raise, carrying the preserved evidence.
    """
    from datetime import UTC, datetime

    return SemanticInterpretationFailedError(
        FailedSemanticInterpretation(
            request=request,
            rendered_prompt=rendered_prompt,
            raw_response=raw_response,
            raw_response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            provenance=provenance,
            diagnostics=diagnostics,
            failure=SemanticParseFailure(kind=type(error).__name__, message=str(error)),
            effective_configuration=effective_configuration,
            occurred_at=occurred_at or datetime.now(UTC).isoformat(),
        )
    )


@dataclass(frozen=True, kw_only=True)
class SemanticInterpretationExecution:
    """Transient result separating request, raw response, parsing, and metrics.

    ``rendered_prompt`` is always the policy ``request`` selected (its
    ``prompt_template_id`` and ``requested_output_schema``): the request identifies the prompt
    the interpreter actually consumed.
    """

    request: SemanticInterpretationRequest
    rendered_prompt: RenderedSemanticPrompt
    raw_response: str
    parsed: ParsedSemanticResponse
    diagnostics: SemanticBackendDiagnostics
    effective_configuration: Mapping[str, JsonScalar]

    def __post_init__(self) -> None:
        """Refuse a rendered prompt or view measurements other than the request's own."""
        _require_selected_prompt(self.request, self.rendered_prompt)
        _require_measured_views(self.request, self.diagnostics)


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
        "diagnostics": encode_semantic_backend_diagnostics(execution.diagnostics),
        "effective_configuration": redact_semantic_secrets(dict(execution.effective_configuration)),
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
        diagnostics=_decode_diagnostics(record["diagnostics"]),
        effective_configuration=MappingProxyType(
            {
                key: cast(JsonScalar, value)
                for key, value in record["effective_configuration"].items()
            }
        ),
    )
