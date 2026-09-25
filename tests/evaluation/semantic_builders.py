"""Builders of real canonical semantic executions for evaluation tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SEMANTIC_PROMPT_TEMPLATES,
    BackendProvenance,
    PerceptionResultId,
    RegionId,
    SemanticBackendDiagnostics,
    SemanticConfidencePolicy,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticPromptTemplate,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
)


def response_json(
    hypotheses: Sequence[tuple[str, str]] = (("wooden pallet", "primary"),),
    *,
    abstained: bool = False,
    scene: dict[str, str | None] | None = None,
    mode: SemanticInterpretationMode = SemanticInterpretationMode.REGION,
) -> str:
    """Return a canonical ``semantic-response/1`` document."""
    claims: list[dict[str, Any]] = [
        {
            "hypothesis": hypothesis,
            "role": role,
            "category": None,
            "region_kind": None,
            "attributes": {},
            "confidence": None,
        }
        for hypothesis, role in hypotheses
    ]
    scene_context: dict[str, Any] | None = None
    if mode is SemanticInterpretationMode.SCENE:
        scene_context = {} if scene is None else dict(scene)
    if abstained:
        return json.dumps({"abstained": True, "claims": [], "scene_context": None})
    return json.dumps({"abstained": False, "claims": claims, "scene_context": scene_context})


def execution(
    *,
    request_id: str = "region-1",
    mode: SemanticInterpretationMode = SemanticInterpretationMode.REGION,
    frame: str = "frame-1",
    region: str | None = "region-1",
    hypotheses: Sequence[tuple[str, str]] = (("wooden pallet", "primary"),),
    abstained: bool = False,
    scene: dict[str, str | None] | None = None,
    latency_ms: float = 10.0,
    peak_memory_bytes: int | None = None,
    input_tokens: int | None = 5,
    output_tokens: int | None = 3,
    retries: int = 0,
    view_kind: VisualViewKind | None = None,
    scene_context_reference: bool = False,
    backend_id: str = "qwen_semantic",
    raw_response: str | None = None,
    template_id: str | None = None,
) -> SemanticInterpretationExecution:
    """Build and parse one execution through the real prompt and parser."""
    is_scene = mode is SemanticInterpretationMode.SCENE
    kind = view_kind or (VisualViewKind.FULL_FRAME if is_scene else VisualViewKind.TIGHT_CROP)
    region_id = None if is_scene else RegionId(region or "region-1")
    default_template = SEMANTIC_PROMPT_TEMPLATES[f"{mode.value}/v1"]
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId(request_id),
        source_observation_id=SourceObservationId(frame),
        perception_result_id=PerceptionResultId(f"result-{frame}"),
        mode=mode,
        visual_views=(
            SemanticVisualView(
                view_id=f"view-{request_id}-{kind.value}",
                kind=kind,
                payload_reference=f"outputs/semantic-views/{request_id}-{kind.value}.png",
                source_observation_id=SourceObservationId(frame),
                sha256="0" * 64,
                region_id=region_id if kind is not VisualViewKind.FULL_FRAME else None,
            ),
        ),
        region_id=region_id,
        prompt_template_id=template_id or default_template.template_id,
        requested_output_schema=default_template.output_schema_version,
        configuration_fingerprint="sha256:config",
        scene_context_reference=(
            SemanticEvidenceReference(evidence_type="scene_context", evidence_id=f"result-{frame}")
            if scene_context_reference
            else None
        ),
    )
    template = SemanticPromptTemplate(
        template_id=request.prompt_template_id,
        mode=mode,
        output_schema_version=default_template.output_schema_version,
        instructions=default_template.instructions,
    )
    rendered = render_semantic_prompt(
        request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
    )
    raw = raw_response or response_json(hypotheses, abstained=abstained, scene=scene, mode=mode)
    provenance = SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id=backend_id,
            capability="semantic_interpreter",
            provider="test",
            model="model",
            version="1",
            configuration_fingerprint="sha256:config",
        ),
        task_identity=f"{backend_id}-{mode.value}",
        prompt_template_id=request.prompt_template_id,
        output_schema_version=request.requested_output_schema,
    )
    return SemanticInterpretationExecution(
        request=request,
        rendered_prompt=rendered,
        raw_response=raw,
        parsed=parse_semantic_response(
            raw, request, provenance, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
        ),
        diagnostics=SemanticBackendDiagnostics(
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            peak_memory_bytes=peak_memory_bytes,
            retries=retries,
        ),
        effective_configuration={"model": "model"},
    )
