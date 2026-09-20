"""Human-readable Semantic Interpretation audit artifacts and secret redaction."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from contextmap.visual_perception.semantic_requests import encode_semantic_request
from contextmap.visual_perception.serialization import encode_claim, encode_scene_context

if TYPE_CHECKING:
    from contextmap.visual_perception.semantic_backend import SemanticInterpretationExecution

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
    execution: SemanticInterpretationExecution,
    debug_level: SemanticDebugLevel,
) -> tuple[str, ...]:
    """Write request/prompt/parsed/final diagnostics according to the debug level."""
    if debug_level is SemanticDebugLevel.NONE:
        return ()
    request_id = str(execution.request.request_id)
    request_path = PurePosixPath(request_id)
    if request_path.name != request_id or request_id in {".", ".."}:
        raise ValueError(f"semantic request_id is not a safe path segment: {request_id!r}")
    relative_root = f"{SEMANTIC_DEBUG_ROOT}/{request_id}"
    root = run_root / relative_root
    root.mkdir(parents=True, exist_ok=True)
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
    records: dict[str, str] = {
        "request.json": json.dumps(
            redact_semantic_secrets(encode_semantic_request(execution.request)),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "prompt.txt": execution.rendered_prompt.text,
        "parsed-response.json": json.dumps(parsed, indent=2, sort_keys=True) + "\n",
        "diagnostics.json": json.dumps(
            redact_semantic_secrets(
                {
                    "latency_ms": execution.diagnostics.latency_ms,
                    "input_tokens": execution.diagnostics.input_tokens,
                    "output_tokens": execution.diagnostics.output_tokens,
                    "peak_memory_bytes": execution.diagnostics.peak_memory_bytes,
                    "retries": execution.diagnostics.retries,
                    "warnings": list(execution.diagnostics.warnings),
                    "effective_configuration": dict(execution.effective_configuration),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }
    if execution.parsed.scene_context is None:
        records["semantic-claims.json"] = (
            json.dumps(parsed["claims"], indent=2, sort_keys=True) + "\n"
        )
    else:
        records["scene-context.json"] = (
            json.dumps(parsed["scene_context"], indent=2, sort_keys=True) + "\n"
        )
    if debug_level is SemanticDebugLevel.FULL:
        records["raw-response.txt"] = execution.raw_response
    for filename, content in records.items():
        (root / filename).write_text(content, encoding="utf-8")
    return tuple(f"{relative_root}/{filename}" for filename in sorted(records))
