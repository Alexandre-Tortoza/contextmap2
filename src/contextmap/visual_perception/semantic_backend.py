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
        """Require the recorded hash to match the response it describes."""
        expected = hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest()
        if self.raw_response_sha256 != expected:
            raise ValueError("raw_response_sha256 does not match raw_response")


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
        "diagnostics": {
            "latency_ms": failed.diagnostics.latency_ms,
            "input_tokens": failed.diagnostics.input_tokens,
            "output_tokens": failed.diagnostics.output_tokens,
            "peak_memory_bytes": failed.diagnostics.peak_memory_bytes,
            "retries": failed.diagnostics.retries,
            "warnings": list(failed.diagnostics.warnings),
        },
        "parse_failure": {"kind": failed.failure.kind, "message": failed.failure.message},
        "effective_configuration": redact_semantic_secrets(dict(failed.effective_configuration)),
        "occurred_at": failed.occurred_at,
    }


def decode_failed_semantic_interpretation(
    record: dict[str, Any],
) -> FailedSemanticInterpretation:
    """Decode one persisted failed interpretation and verify its recorded response hash.

    Raises:
        KeyError: If a field of the record is missing.
        ValueError: If the raw response is not a string, or the provenance names no
            ``raw_response_reference``: a backend leaves it ``None``, and the writer that
            persists the record always materializes it.
    """
    from contextmap.visual_perception.models import BackendProvenance

    raw_response = record["raw_response"]
    if not isinstance(raw_response, str):
        raise ValueError("failed semantic interpretation raw_response must be a string")
    raw_prompt = record["rendered_prompt"]
    raw_provenance = record["provenance"]
    if not isinstance(raw_provenance["raw_response_reference"], str):
        raise ValueError(
            "failed semantic interpretation provenance raw_response_reference must be a string"
        )
    raw_backend = raw_provenance["backend"]
    raw_diagnostics = record["diagnostics"]
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
        diagnostics=SemanticBackendDiagnostics(
            latency_ms=raw_diagnostics["latency_ms"],
            input_tokens=raw_diagnostics["input_tokens"],
            output_tokens=raw_diagnostics["output_tokens"],
            peak_memory_bytes=raw_diagnostics["peak_memory_bytes"],
            retries=raw_diagnostics["retries"],
            warnings=tuple(raw_diagnostics["warnings"]),
        ),
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
    """Encode one semantic execution as the record persisted under ``raw_response_reference``.

    The raw response is inline in the record; ``raw_response_reference`` names the contractual
    record that holds it. The run writer materializes that reference, the same one on every claim
    and on the scene context of the execution, so :func:`decode_semantic_execution` can hold them
    to it.
    """
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
        "effective_configuration": redact_semantic_secrets(dict(execution.effective_configuration)),
    }


def decode_semantic_execution(
    record: dict[str, Any],
) -> SemanticInterpretationExecution:
    """Decode one persisted semantic execution and verify its contractual raw response.

    Raises:
        KeyError: If a field of the record is missing, ``raw_response_reference`` included:
            the execution does not carry it, but the record contract does.
        ValueError: If the raw response or its reference is not a string, the raw response
            does not match its recorded hash, or a claim or the scene context names another
            ``raw_response_reference`` than the record.
    """
    raw_response_reference = record["raw_response_reference"]
    if not isinstance(raw_response_reference, str):
        raise ValueError("semantic execution raw_response_reference must be a string")
    raw_response = record["raw_response"]
    if not isinstance(raw_response, str):
        raise ValueError("semantic execution raw_response must be a string")
    raw_response_sha256 = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    if raw_response_sha256 != record["raw_response_sha256"]:
        raise ValueError("semantic execution raw response hash does not match its record")
    raw_prompt = record["rendered_prompt"]
    raw_parsed = record["parsed"]
    raw_diagnostics = record["diagnostics"]
    parsed = ParsedSemanticResponse(
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
    )
    stray = sorted(
        {
            repr(provenance.raw_response_reference)
            for provenance in parsed.provenances()
            if provenance.raw_response_reference != raw_response_reference
        }
    )
    if stray:
        raise ValueError(
            "semantic execution claims and scene context must share the record's "
            f"raw_response_reference {raw_response_reference!r}, found {', '.join(stray)}"
        )
    return SemanticInterpretationExecution(
        request=decode_semantic_request(record["request"]),
        rendered_prompt=RenderedSemanticPrompt(
            template_id=raw_prompt["template_id"],
            output_schema_version=raw_prompt["output_schema_version"],
            text=raw_prompt["text"],
            fingerprint=raw_prompt["fingerprint"],
        ),
        raw_response=raw_response,
        parsed=parsed,
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
