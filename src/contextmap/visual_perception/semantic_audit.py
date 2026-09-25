"""Human-readable Semantic Interpretation audit artifacts and secret redaction."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from contextmap.visual_perception.semantic_requests import encode_semantic_request
from contextmap.visual_perception.serialization import encode_claim, encode_scene_context

if TYPE_CHECKING:
    from contextmap.visual_perception.semantic_backend import (
        FailedSemanticInterpretation,
        SemanticInterpretationExecution,
    )

SEMANTIC_DEBUG_ROOT = "debug/40-semantic-interpretation"
"""Run-relative root for non-contractual semantic diagnostics."""

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "access_token",
    "refresh_token",
}


class SemanticDebugLevel(Enum):
    """Amount of non-contractual semantic diagnostics to persist."""

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


def redact_semantic_secrets(value: Any) -> Any:
    """Recursively redact known credential fields before serialization."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_")
            redacted[key] = (
                "[REDACTED]" if normalized in _SENSITIVE_KEYS else redact_semantic_secrets(item)
            )
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_semantic_secrets(item) for item in value]
    return value


def write_semantic_audit(
    *,
    run_root: Path,
    execution: SemanticInterpretationExecution | FailedSemanticInterpretation,
    debug_level: SemanticDebugLevel,
) -> tuple[str, ...]:
    """Write request/prompt/outcome/diagnostics for one attempt, per the debug level.

    Accepts either outcome of a real backend call. Both land in the same
    ``debug/40-semantic-interpretation/<request-id>/`` directory and share request, prompt,
    diagnostics and (at ``FULL``) the raw response; they differ only in how the outcome is
    recorded — ``parsed-response.json`` for an execution that materialized, or
    ``parse-failure.json`` for one the parser rejected. One writer, so the two can never drift
    apart.
    """
    if debug_level is SemanticDebugLevel.NONE:
        return ()
    from contextmap.visual_perception.semantic_backend import FailedSemanticInterpretation

    request_id = str(execution.request.request_id)
    request_path = PurePosixPath(request_id)
    if request_path.name != request_id or request_id in {".", ".."}:
        raise ValueError(f"semantic request_id is not a safe path segment: {request_id!r}")
    relative_root = f"{SEMANTIC_DEBUG_ROOT}/{request_id}"
    root = run_root / relative_root
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(execution, FailedSemanticInterpretation):
        return _write_shared_audit(
            root=root,
            relative_root=relative_root,
            execution=execution,
            debug_level=debug_level,
            outcome={
                "parse-failure.json": json.dumps(
                    {
                        "kind": execution.failure.kind,
                        "message": execution.failure.message,
                        "raw_response_sha256": execution.raw_response_sha256,
                        "occurred_at": execution.occurred_at,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            },
        )
    parsed = {
        "abstained": execution.parsed.abstained,
        "raw_response_sha256": execution.parsed.raw_response_sha256,
        "diagnostics": [
            {"code": item.code, "message": item.message} for item in execution.parsed.diagnostics
        ],
        "claims": [encode_claim(claim) for claim in execution.parsed.claims],
        "scene_context": (
            None
            if execution.parsed.scene_context is None
            else encode_scene_context(execution.parsed.scene_context)
        ),
    }
    outcome = {"parsed-response.json": json.dumps(parsed, indent=2, sort_keys=True) + "\n"}
    if execution.parsed.scene_context is None:
        outcome["semantic-claims.json"] = (
            json.dumps(parsed["claims"], indent=2, sort_keys=True) + "\n"
        )
    else:
        outcome["scene-context.json"] = (
            json.dumps(parsed["scene_context"], indent=2, sort_keys=True) + "\n"
        )
    return _write_shared_audit(
        root=root,
        relative_root=relative_root,
        execution=execution,
        debug_level=debug_level,
        outcome=outcome,
    )


def _write_shared_audit(
    *,
    root: Path,
    relative_root: str,
    execution: SemanticInterpretationExecution | FailedSemanticInterpretation,
    debug_level: SemanticDebugLevel,
    outcome: dict[str, str],
) -> tuple[str, ...]:
    """Write what both outcomes share, plus the caller's outcome-specific files."""
    # Import tardio: semantic_backend importa este módulo (redação) no carregamento.
    from contextmap.visual_perception.semantic_backend import encode_semantic_backend_diagnostics

    records: dict[str, str] = {
        "request.json": json.dumps(
            redact_semantic_secrets(encode_semantic_request(execution.request)),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "prompt.txt": execution.rendered_prompt.text,
        "diagnostics.json": json.dumps(
            redact_semantic_secrets(
                {
                    **encode_semantic_backend_diagnostics(execution.diagnostics),
                    "effective_configuration": dict(execution.effective_configuration),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        **outcome,
    }
    if debug_level is SemanticDebugLevel.FULL:
        records["raw-response.txt"] = execution.raw_response
    for filename, content in records.items():
        (root / filename).write_text(content, encoding="utf-8")
    return tuple(f"{relative_root}/{filename}" for filename in sorted(records))
