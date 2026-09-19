"""Versioned semantic prompts and backend-neutral structured response parsing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

from contextmap.visual_perception.models import (
    ClaimId,
    HypothesisRole,
    SceneContext,
    SemanticAttribute,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticRegionKind,
)
from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
)


class SemanticResponseParseError(ValueError):
    """Raised when a backend response cannot become canonical semantic evidence."""


@dataclass(frozen=True, kw_only=True)
class SemanticPromptTemplate:
    """One immutable semantic instruction policy with a versioned identity."""

    template_id: str
    mode: SemanticInterpretationMode
    output_schema_version: str
    instructions: str

    def __post_init__(self) -> None:
        """Reject unnamed or empty prompt policy fields."""
        for field_name, value in (
            ("template_id", self.template_id),
            ("output_schema_version", self.output_schema_version),
            ("instructions", self.instructions),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")

    @classmethod
    def default_for(cls, mode: SemanticInterpretationMode) -> SemanticPromptTemplate:
        """Return the canonical v1 template for one interpretation mode."""
        if mode is SemanticInterpretationMode.SCENE:
            return cls(
                template_id="scene/v1",
                mode=mode,
                output_schema_version="semantic-response/1",
                instructions=(
                    "Describe only visible scene-level evidence. Preserve ambiguity, "
                    "abstain when unsupported, and never invent confidence."
                ),
            )
        return cls(
            template_id="region/v1",
            mode=mode,
            output_schema_version="semantic-response/1",
            instructions=(
                "Describe only the referenced region. Return one primary hypothesis, "
                "preserve plausible alternatives, and never invent confidence."
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RenderedSemanticPrompt:
    """Auditable deterministic rendering of one template and canonical request."""

    template_id: str
    output_schema_version: str
    text: str
    fingerprint: str


@dataclass(frozen=True, kw_only=True)
class SemanticParseDiagnostic:
    """Record a parser decision without changing semantic content."""

    code: str
    message: str


@dataclass(frozen=True, kw_only=True)
class ParsedSemanticResponse:
    """Canonical result of parsing one raw backend response."""

    raw_response_sha256: str
    claims: tuple[SemanticClaim, ...]
    scene_context: SceneContext | None
    abstained: bool
    diagnostics: tuple[SemanticParseDiagnostic, ...] = ()


def render_semantic_prompt(
    request: SemanticInterpretationRequest,
    template: SemanticPromptTemplate,
) -> RenderedSemanticPrompt:
    """Render a deterministic prompt after checking template/request identity."""
    if template.mode is not request.mode:
        raise ValueError("prompt template mode must match semantic request mode")
    if template.template_id != request.prompt_template_id:
        raise ValueError("prompt template identity must match semantic request")
    if template.output_schema_version != request.requested_output_schema:
        raise ValueError("prompt output schema must match semantic request")

    request_summary = {
        "request_id": str(request.request_id),
        "source_observation_id": str(request.source_observation_id),
        "perception_result_id": str(request.perception_result_id),
        "mode": request.mode.value,
        "region_id": None if request.region_id is None else str(request.region_id),
        "visual_views": [
            {"view_id": view.view_id, "kind": view.kind.value} for view in request.visual_views
        ],
        "visual_features": [
            {
                "feature_id": str(feature.feature_id),
                "embedding_space_id": feature.embedding_space_id,
            }
            for feature in request.visual_features
        ],
        "scene_context_reference": (
            None
            if request.scene_context_reference is None
            else request.scene_context_reference.evidence_id
        ),
    }
    schema = {
        "abstained": "boolean",
        "claims": [
            {
                "hypothesis": "string",
                "role": "primary | alternative",
                "category": "string | null",
                "region_kind": "thing | stuff | null",
                "attributes": "object with scalar values",
                "confidence": "number in [0,1] | null",
            }
        ],
        "scene_context": (
            "object with scene_type/environment/layout/lighting/visibility/navigability | null"
        ),
    }
    text = "\n\n".join(
        (
            f"Template: {template.template_id}",
            template.instructions,
            "Request:\n" + json.dumps(request_summary, sort_keys=True, separators=(",", ":")),
            (
                f"Output schema ({template.output_schema_version}):\n"
                + json.dumps(schema, sort_keys=True, separators=(",", ":"))
            ),
            "Return exactly one JSON object and no explanatory text.",
        )
    )
    fingerprint = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RenderedSemanticPrompt(
        template_id=template.template_id,
        output_schema_version=template.output_schema_version,
        text=text,
        fingerprint=fingerprint,
    )


def parse_semantic_response(
    raw_response: str,
    request: SemanticInterpretationRequest,
    provenance: SemanticInferenceProvenance,
) -> ParsedSemanticResponse:
    """Parse strict JSON, allowing only recorded non-semantic code-fence removal."""
    _validate_parse_identity(request, provenance)
    if not raw_response.strip():
        raise SemanticResponseParseError("raw response must not be empty")

    normalized, diagnostics = _remove_code_fence(raw_response)
    try:
        decoded = json.loads(normalized, parse_constant=_reject_non_finite_json)
    except (json.JSONDecodeError, SemanticResponseParseError) as error:
        raise SemanticResponseParseError(f"malformed structured response: {error}") from error
    data = _mapping(decoded, "semantic response")
    _require_exact_keys(data, {"abstained", "claims", "scene_context"}, "semantic response")
    abstained = data["abstained"]
    if not isinstance(abstained, bool):
        raise SemanticResponseParseError("abstained must be a boolean")
    raw_claims = data["claims"]
    if not isinstance(raw_claims, list):
        raise SemanticResponseParseError("claims must be an array")
    if abstained:
        if raw_claims or data["scene_context"] is not None:
            raise SemanticResponseParseError("abstained response must not contain semantic output")
        return ParsedSemanticResponse(
            raw_response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            claims=(),
            scene_context=None,
            abstained=True,
            diagnostics=diagnostics,
        )
    if not raw_claims:
        raise SemanticResponseParseError("non-abstained response requires at least one claim")

    claims = tuple(
        _parse_claim(item, request=request, provenance=provenance, index=index)
        for index, item in enumerate(raw_claims)
    )
    if request.mode is SemanticInterpretationMode.REGION:
        if data["scene_context"] is not None:
            raise SemanticResponseParseError("region response scene_context must be null")
        if sum(claim.role is HypothesisRole.PRIMARY for claim in claims) != 1:
            raise SemanticResponseParseError("region response requires exactly one primary claim")
        output_claims = claims
        scene_context = None
    else:
        raw_context = _mapping(data["scene_context"], "scene_context")
        allowed_context_keys = {
            "scene_type",
            "environment",
            "layout",
            "lighting",
            "visibility",
            "navigability",
        }
        unexpected = set(raw_context) - allowed_context_keys
        if unexpected:
            raise SemanticResponseParseError(
                f"scene_context contains unexpected fields: {sorted(unexpected)}"
            )
        scene_context = SceneContext(
            source_observation_id=request.source_observation_id,
            perception_result_id=request.perception_result_id,
            provenance=provenance,
            scene_type=_optional_string(raw_context.get("scene_type"), "scene_type"),
            environment=_optional_string(raw_context.get("environment"), "environment"),
            layout=_optional_string(raw_context.get("layout"), "layout"),
            lighting=_optional_string(raw_context.get("lighting"), "lighting"),
            visibility=_optional_string(raw_context.get("visibility"), "visibility"),
            navigability=_optional_string(raw_context.get("navigability"), "navigability"),
            claims=claims,
            evidence_references=request.evidence_references(),
        )
        output_claims = ()

    return ParsedSemanticResponse(
        raw_response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        claims=output_claims,
        scene_context=scene_context,
        abstained=False,
        diagnostics=diagnostics,
    )


def _validate_parse_identity(
    request: SemanticInterpretationRequest,
    provenance: SemanticInferenceProvenance,
) -> None:
    if provenance.prompt_template_id != request.prompt_template_id:
        raise SemanticResponseParseError("provenance prompt template does not match request")
    if provenance.output_schema_version != request.requested_output_schema:
        raise SemanticResponseParseError("provenance output schema does not match request")
    backend_fingerprint = provenance.backend.configuration_fingerprint
    if backend_fingerprint is not None and backend_fingerprint != request.configuration_fingerprint:
        raise SemanticResponseParseError("provenance configuration does not match request")


def _remove_code_fence(
    raw_response: str,
) -> tuple[str, tuple[SemanticParseDiagnostic, ...]]:
    stripped = raw_response.strip()
    if not stripped.startswith("```"):
        return stripped, ()
    lines = stripped.splitlines()
    if (
        len(lines) < 3
        or lines[-1].strip() != "```"
        or lines[0].strip()
        not in {
            "```",
            "```json",
        }
    ):
        raise SemanticResponseParseError("malformed code fence")
    return (
        "\n".join(lines[1:-1]).strip(),
        (
            SemanticParseDiagnostic(
                code="removed_code_fence",
                message="Removed one outer JSON code fence without changing its content.",
            ),
        ),
    )


def _parse_claim(
    value: object,
    *,
    request: SemanticInterpretationRequest,
    provenance: SemanticInferenceProvenance,
    index: int,
) -> SemanticClaim:
    data = _mapping(value, f"claim[{index}]")
    allowed = {"hypothesis", "role", "category", "region_kind", "attributes", "confidence"}
    unexpected = set(data) - allowed
    if unexpected:
        raise SemanticResponseParseError(
            f"claim[{index}] contains unexpected fields: {sorted(unexpected)}"
        )
    hypothesis = data.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise SemanticResponseParseError(f"claim[{index}].hypothesis must be a non-empty string")
    try:
        role = HypothesisRole(data.get("role"))
    except ValueError as error:
        raise SemanticResponseParseError(
            f"claim[{index}].role must be primary or alternative"
        ) from error
    category = _optional_string(data.get("category"), f"claim[{index}].category")
    raw_region_kind = data.get("region_kind")
    try:
        region_kind = None if raw_region_kind is None else SemanticRegionKind(raw_region_kind)
    except ValueError as error:
        raise SemanticResponseParseError(
            f"claim[{index}].region_kind must be thing, stuff, or null"
        ) from error
    raw_attributes = data.get("attributes", {})
    attributes = _mapping(raw_attributes, f"claim[{index}].attributes")
    parsed_attributes: list[SemanticAttribute] = []
    for name, attribute_value in sorted(attributes.items()):
        if not isinstance(name, str) or not name.strip():
            raise SemanticResponseParseError(f"claim[{index}] attribute name must not be empty")
        if not _is_json_scalar(attribute_value):
            raise SemanticResponseParseError(
                f"claim[{index}] attribute {name!r} must have a scalar value"
            )
        parsed_attributes.append(
            SemanticAttribute(name=name, value=cast(JsonScalar, attribute_value))
        )
    confidence = data.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise SemanticResponseParseError(f"claim[{index}].confidence must be in [0, 1] or null")
    return SemanticClaim(
        claim_id=ClaimId(
            f"{request.perception_result_id}--semantic-{request.request_id}-{index:04d}"
        ),
        source_observation_id=request.source_observation_id,
        perception_result_id=request.perception_result_id,
        hypothesis=hypothesis,
        role=role,
        provenance=provenance,
        category=category,
        region_kind=region_kind,
        attributes=tuple(parsed_attributes),
        confidence=None if confidence is None else float(confidence),
        region_id=request.region_id,
        evidence_references=request.evidence_references(),
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SemanticResponseParseError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _require_exact_keys(data: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = keys - data.keys()
    unexpected = data.keys() - keys
    if missing:
        raise SemanticResponseParseError(f"{name} is missing required fields: {sorted(missing)}")
    if unexpected:
        raise SemanticResponseParseError(f"{name} contains unexpected fields: {sorted(unexpected)}")


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SemanticResponseParseError(f"{name} must be a non-empty string or null")
    return value


def _is_json_scalar(value: object) -> bool:
    if isinstance(value, float) and not isfinite(value):
        return False
    return value is None or isinstance(value, (str, int, float, bool))


def _reject_non_finite_json(value: str) -> object:
    raise SemanticResponseParseError(f"non-finite JSON value is not allowed: {value}")
