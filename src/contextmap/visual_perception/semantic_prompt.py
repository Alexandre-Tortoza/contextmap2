"""Versioned semantic prompts and backend-neutral structured response parsing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
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
    """Raised when a backend response cannot become canonical semantic evidence.

    Attributes:
        raw_response: The exact response text that was rejected, so a failed parse never
            loses what the model said (truncation and schema drift are diagnosed from it).
            ``None`` only when the error was raised outside :func:`parse_semantic_response`.
    """

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        """Create the error with an optional rejected raw response."""
        super().__init__(message)
        self.raw_response = raw_response


class SemanticConfidencePolicy(Enum):
    """Declare whether a parser caller owns meaningful numeric confidence."""

    UNSCORED_ONLY = "unscored_only"
    MEASURED = "measured"


SEMANTIC_RESPONSE_SCHEMA = "semantic-response/1"
"""The one output schema :func:`render_semantic_prompt` renders and the parser implements."""

SCENE_CONTEXT_RENDERING = "scene-context/1"
"""Versioned form in which a template that conditions on scene context renders it (#529).

The section names this version, then either states that no scene context was supplied or
gives the template's framing instructions followed by the context as canonical JSON (sorted
keys): the six structured fields and each claim's hypothesis, role, category, region kind
and attributes. Confidence is never rendered, so no uncalibrated number is fed back.
"""


@dataclass(frozen=True, kw_only=True)
class SemanticPromptTemplate:
    """One immutable semantic instruction policy with a versioned identity.

    A template is selected by its ``template_id``, never by a backend: the request names it
    (``SemanticInterpretationRequest.prompt_template_id``) and every instruction-following
    interpreter renders exactly the catalog template of that identity
    (:data:`SEMANTIC_PROMPT_TEMPLATES`) or refuses the request before inference.
    """

    template_id: str
    mode: SemanticInterpretationMode
    output_schema_version: str
    instructions: str
    scene_context_instructions: str | None = None

    def __post_init__(self) -> None:
        """Reject unnamed or empty policy fields and an output schema nothing implements.

        ``scene_context_instructions`` makes a region template render a scene-context
        section (:data:`SCENE_CONTEXT_RENDERING`); a scene template never does, so scene
        interpretation cannot depend on its own output.
        """
        for field_name, value in (
            ("template_id", self.template_id),
            ("output_schema_version", self.output_schema_version),
            ("instructions", self.instructions),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.output_schema_version != SEMANTIC_RESPONSE_SCHEMA:
            # O schema renderizado e o parser só conhecem semantic-response/1: um template que
            # declarasse outra versão prometeria ao modelo um contrato que ninguém implementa.
            raise ValueError(
                f"prompt template {self.template_id!r} declares output schema "
                f"{self.output_schema_version!r}; the renderer and parser implement only "
                f"{SEMANTIC_RESPONSE_SCHEMA!r}"
            )
        if self.scene_context_instructions is not None:
            if self.mode is not SemanticInterpretationMode.REGION:
                raise ValueError(
                    f"prompt template {self.template_id!r}: only a region template may render "
                    "scene context; a scene request conditioned on scene context is circular"
                )
            if not self.scene_context_instructions.strip():
                raise ValueError("scene_context_instructions must not be empty when declared")


_TEMPLATES = (
    SemanticPromptTemplate(
        template_id="scene/v1",
        mode=SemanticInterpretationMode.SCENE,
        output_schema_version=SEMANTIC_RESPONSE_SCHEMA,
        instructions=(
            "Describe only visible scene-level evidence. Preserve ambiguity, "
            "abstain when unsupported, and never invent confidence: omit it or "
            "set it to null."
        ),
    ),
    SemanticPromptTemplate(
        template_id="region/v1",
        mode=SemanticInterpretationMode.REGION,
        output_schema_version=SEMANTIC_RESPONSE_SCHEMA,
        instructions=(
            "Describe only the referenced region. Return one primary hypothesis, "
            "preserve plausible alternatives, and never invent confidence: omit it "
            "or set it to null."
        ),
    ),
    # Alternativa não canônica e não avaliada (#542): existe para que a seleção de prompt seja
    # exercitável ponta a ponta. Varia só a instrução de abstenção em relação a region/v1; as
    # famílias de prompt reais são avaliadas em #525.
    SemanticPromptTemplate(
        template_id="region-abstention/v1",
        mode=SemanticInterpretationMode.REGION,
        output_schema_version=SEMANTIC_RESPONSE_SCHEMA,
        instructions=(
            "Describe only the referenced region. Abstain instead of guessing when the "
            "visible evidence does not support a hypothesis; otherwise return one primary "
            "hypothesis and preserve plausible alternatives. Never invent confidence: omit "
            "it or set it to null."
        ),
    ),
    # Condicionamento por contexto de cena (#529): mesmas instruções de region/v1 mais a seção
    # de contexto; sem contexto, a seção diz explicitamente que nenhum foi fornecido, então
    # um par com/sem contexto varia só esse insumo.
    SemanticPromptTemplate(
        template_id="region-scene-context/v1",
        mode=SemanticInterpretationMode.REGION,
        output_schema_version=SEMANTIC_RESPONSE_SCHEMA,
        instructions=(
            "Describe only the referenced region. Return one primary hypothesis, "
            "preserve plausible alternatives, and never invent confidence: omit it "
            "or set it to null."
        ),
        scene_context_instructions=(
            "The scene context below is a prior hypothesis from another inference over the "
            "same image, not ground truth. Use it only to disambiguate what the region "
            "shows; never describe the scene instead of the region, and disregard it where "
            "the visible evidence contradicts it."
        ),
    ),
)

SEMANTIC_PROMPT_TEMPLATES: Mapping[str, SemanticPromptTemplate] = MappingProxyType(
    {template.template_id: template for template in _TEMPLATES}
)
"""Every versioned instruction template, keyed by its identity; closed and immutable.

``scene/v1`` and ``region/v1`` are the canonical policies. ``region-abstention/v1`` is a
non-canonical, unevaluated alternative. ``region-scene-context/v1`` renders the scene context
a region request is conditioned on (#529); its instructions are those of ``region/v1``.
A new prompt policy is a new entry with a new identity, never an edit of an existing one:
an identity names exactly one instruction text and output schema.
"""


def semantic_prompt_template(template_id: str) -> SemanticPromptTemplate:
    """Return the catalog template a request or configuration names.

    Raises:
        ValueError: If no template has that identity. Nothing substitutes another one.
    """
    template = SEMANTIC_PROMPT_TEMPLATES.get(template_id)
    if template is None:
        raise ValueError(
            f"unknown semantic prompt template {template_id!r}; "
            f"known: {sorted(SEMANTIC_PROMPT_TEMPLATES)}"
        )
    return template


@dataclass(frozen=True, kw_only=True)
class SemanticPromptPolicy:
    """The instruction template each interpretation mode renders, selected before inference.

    This is the declarative, serializable selection a run configures for an
    instruction-following interpreter (Qwen, Gemini). It changes the prompt actually rendered
    and consumed, independently of the backend, its generation settings and the visual views.
    A task-native interpreter (Florence-2) consumes its task prompt instead and does not take
    this policy.

    Attributes:
        scene: Identity of the :data:`SEMANTIC_PROMPT_TEMPLATES` entry scene requests name.
        region: Identity of the entry region requests name.
        region_scene_context: Whether each region request is conditioned on the scene
            context its observation's scene request produced (#529). Requires a region
            template that renders scene context. ``False`` conditions nothing: with such a
            template the prompt then states that no scene context was supplied, which is the
            disabled arm of a matched ablation.
    """

    scene: str
    region: str
    region_scene_context: bool = False

    def __post_init__(self) -> None:
        """Require each identity to name a catalog template of its own mode."""
        for mode, template_id in (
            (SemanticInterpretationMode.SCENE, self.scene),
            (SemanticInterpretationMode.REGION, self.region),
        ):
            template = semantic_prompt_template(template_id)
            if template.mode is not mode:
                raise ValueError(
                    f"{mode.value} prompt policy {template_id!r} is a "
                    f"{template.mode.value} template"
                )
        region = semantic_prompt_template(self.region)
        if self.region_scene_context and region.scene_context_instructions is None:
            raise ValueError(
                f"region prompt template {self.region!r} does not render scene context; "
                "select one that does (region-scene-context/v1) to condition region requests"
            )

    def template_for(self, mode: SemanticInterpretationMode) -> SemanticPromptTemplate:
        """Return the selected template of one mode."""
        template_id = self.scene if mode is SemanticInterpretationMode.SCENE else self.region
        return SEMANTIC_PROMPT_TEMPLATES[template_id]


@dataclass(frozen=True, kw_only=True)
class RenderedSemanticPrompt:
    """Auditable deterministic rendering of one prompt policy for one canonical request.

    Attributes:
        template_id: Identity of the policy rendered; equal to the request's
            ``prompt_template_id``.
        output_schema_version: Schema the response is held to.
        text: The exact text the interpreter consumes.
        fingerprint: ``"sha256:<hex>"`` of ``text`` encoded as UTF-8.
    """

    template_id: str
    output_schema_version: str
    text: str
    fingerprint: str

    def __post_init__(self) -> None:
        """Refuse a fingerprint that does not describe ``text``."""
        expected = "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.fingerprint != expected:
            raise ValueError("rendered prompt fingerprint does not match its text")


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
    *,
    confidence_policy: SemanticConfidencePolicy,
) -> RenderedSemanticPrompt:
    """Render a deterministic prompt after checking template/request identity.

    Raises:
        ValueError: If the template's mode, identity or output schema differs from what the
            request selected; a template the request did not name is never rendered.
    """
    if template.mode is not request.mode:
        raise ValueError("prompt template mode must match semantic request mode")
    if template.template_id != request.prompt_template_id:
        raise ValueError("prompt template identity must match semantic request")
    if template.output_schema_version != request.requested_output_schema:
        raise ValueError("prompt output schema must match semantic request")
    if request.scene_context is not None and template.scene_context_instructions is None:
        # Aceitar o contexto e não renderizá-lo seria descartá-lo em silêncio.
        raise ValueError(
            f"prompt template {template.template_id!r} does not render scene context, but the "
            "request is conditioned on one"
        )

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
        "supporting_metadata": [
            {"name": item.name, "value": item.value} for item in request.supporting_metadata
        ],
    }
    # Sob UNSCORED_ONLY confidence so pode ser null, entao ela sai de "required": o schema
    # passa a dizer o mesmo que a instrucao "never invent confidence" e o que o parser aceita.
    claim_required = ["hypothesis", "role", "category", "region_kind", "attributes"]
    if confidence_policy is not SemanticConfidencePolicy.UNSCORED_ONLY:
        claim_required.append("confidence")
    claim_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": claim_required,
        "properties": {
            "hypothesis": {"type": "string", "minLength": 1},
            "role": {"enum": ["primary", "alternative"]},
            "category": {
                "anyOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "null"},
                ]
            },
            "region_kind": {"enum": ["thing", "stuff", None]},
            "attributes": {
                "type": "object",
                "propertyNames": {"minLength": 1},
                "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
            },
            "confidence": (
                {"type": "null"}
                if confidence_policy is SemanticConfidencePolicy.UNSCORED_ONLY
                else {"type": ["number", "null"], "minimum": 0, "maximum": 1}
            ),
        },
    }
    scene_context_schema: dict[str, object]
    minimum_claims: int
    if request.mode is SemanticInterpretationMode.REGION:
        minimum_claims = 1
        scene_context_schema = {"type": "null"}
    else:
        minimum_claims = 0
        scene_fields = (
            "scene_type",
            "environment",
            "layout",
            "lighting",
            "visibility",
            "navigability",
        )
        scene_context_schema = {
            "additionalProperties": False,
            "properties": {
                name: {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                }
                for name in scene_fields
            },
            "type": "object",
        }
    claims_schema: dict[str, object] = {
        "type": "array",
        "minItems": minimum_claims,
        "items": claim_schema,
    }
    if request.mode is SemanticInterpretationMode.REGION:
        claims_schema["description"] = "Exactly one item must have role=primary."
    non_abstained_schema: dict[str, object] = {
        "properties": {
            "abstained": {"const": False},
            "claims": claims_schema,
            "scene_context": scene_context_schema,
        },
    }
    if request.mode is SemanticInterpretationMode.SCENE:
        non_abstained_schema["anyOf"] = [
            {"properties": {"claims": {"minItems": 1}}},
            {
                "properties": {
                    "scene_context": {
                        "anyOf": [
                            {
                                "required": [name],
                                "properties": {name: {"type": "string", "minLength": 1}},
                            }
                            for name in scene_fields
                        ]
                    }
                }
            },
        ]
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["abstained", "claims", "scene_context"],
        "oneOf": [
            {
                "properties": {
                    "abstained": {"const": True},
                    "claims": {"type": "array", "maxItems": 0},
                    "scene_context": {"type": "null"},
                },
            },
            non_abstained_schema,
        ],
    }
    sections = [
        f"Template: {template.template_id}",
        template.instructions,
        "Request:\n" + json.dumps(request_summary, sort_keys=True, separators=(",", ":")),
    ]
    if template.scene_context_instructions is not None:
        sections.append(_scene_context_section(request, template.scene_context_instructions))
    sections.extend(
        (
            (
                f"Output schema ({template.output_schema_version}):\n"
                + json.dumps(schema, sort_keys=True, separators=(",", ":"))
            ),
            "Return exactly one JSON object and no explanatory text.",
        )
    )
    text = "\n\n".join(sections)
    fingerprint = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RenderedSemanticPrompt(
        template_id=template.template_id,
        output_schema_version=template.output_schema_version,
        text=text,
        fingerprint=fingerprint,
    )


def _scene_context_section(request: SemanticInterpretationRequest, instructions: str) -> str:
    """Render the scene context a region request carries, or state that it carries none."""
    heading = f"Scene context ({SCENE_CONTEXT_RENDERING}):"
    context = request.scene_context
    if context is None:
        return f"{heading} none supplied for this request."
    rendered = {
        "scene_type": context.scene_type,
        "environment": context.environment,
        "layout": context.layout,
        "lighting": context.lighting,
        "visibility": context.visibility,
        "navigability": context.navigability,
        "claims": [
            {
                "hypothesis": claim.hypothesis,
                "role": claim.role.value,
                "category": claim.category,
                "region_kind": None if claim.region_kind is None else claim.region_kind.value,
                "attributes": {item.name: item.value for item in claim.attributes},
            }
            for claim in context.claims
        ],
    }
    return "\n".join(
        (heading, instructions, json.dumps(rendered, sort_keys=True, separators=(",", ":")))
    )


def parse_semantic_response(
    raw_response: str,
    request: SemanticInterpretationRequest,
    provenance: SemanticInferenceProvenance,
    *,
    confidence_policy: SemanticConfidencePolicy,
) -> ParsedSemanticResponse:
    """Parse strict JSON, allowing only recorded non-semantic normalizations.

    The two accepted normalizations are removing one outer code fence and, in region
    mode, treating an omitted ``scene_context`` as ``null``. The key is null-only in that
    mode, so its absence carries the same information; both cases add a
    :class:`SemanticParseDiagnostic` to the result.

    Raises:
        SemanticResponseParseError: If the response is not canonical. The error carries the
            rejected text in ``raw_response``.
    """
    try:
        return _parse_semantic_response(
            raw_response, request, provenance, confidence_policy=confidence_policy
        )
    except SemanticResponseParseError as error:
        if error.raw_response is None:
            error.raw_response = raw_response
        raise


def _parse_semantic_response(
    raw_response: str,
    request: SemanticInterpretationRequest,
    provenance: SemanticInferenceProvenance,
    *,
    confidence_policy: SemanticConfidencePolicy,
) -> ParsedSemanticResponse:
    _validate_parse_identity(request, provenance)
    if not raw_response.strip():
        raise SemanticResponseParseError("raw response must not be empty")

    normalized, diagnostics = _remove_code_fence(raw_response)
    try:
        decoded = json.loads(normalized, parse_constant=_reject_non_finite_json)
    except (json.JSONDecodeError, SemanticResponseParseError) as error:
        raise SemanticResponseParseError(f"malformed structured response: {error}") from error
    data = _mapping(decoded, "semantic response")
    response_keys = {"abstained", "claims", "scene_context"}
    scene_mode = request.mode is SemanticInterpretationMode.SCENE
    _require_keys(
        data,
        required=response_keys if scene_mode else response_keys - {"scene_context"},
        allowed=response_keys,
        name="semantic response",
    )
    if "scene_context" not in data:
        diagnostics = (
            *diagnostics,
            SemanticParseDiagnostic(
                code="defaulted_null_scene_context",
                message="Region response omitted the null-only scene_context key; treated as null.",
            ),
        )
    raw_scene_context = data.get("scene_context")
    abstained = data["abstained"]
    if not isinstance(abstained, bool):
        raise SemanticResponseParseError("abstained must be a boolean")
    raw_claims = data["claims"]
    if not isinstance(raw_claims, list):
        raise SemanticResponseParseError("claims must be an array")
    if abstained:
        if raw_claims or raw_scene_context is not None:
            raise SemanticResponseParseError("abstained response must not contain semantic output")
        return ParsedSemanticResponse(
            raw_response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            claims=(),
            scene_context=None,
            abstained=True,
            diagnostics=diagnostics,
        )
    claims = tuple(
        _parse_claim(
            item,
            request=request,
            provenance=provenance,
            confidence_policy=confidence_policy,
            index=index,
        )
        for index, item in enumerate(raw_claims)
    )
    if request.mode is SemanticInterpretationMode.REGION:
        if not claims:
            raise SemanticResponseParseError("non-abstained region response requires a claim")
        if raw_scene_context is not None:
            raise SemanticResponseParseError("region response scene_context must be null")
        if sum(claim.role is HypothesisRole.PRIMARY for claim in claims) != 1:
            raise SemanticResponseParseError("region response requires exactly one primary claim")
        output_claims = claims
        scene_context = None
    else:
        raw_context = _mapping(raw_scene_context, "scene_context")
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
        context_values = {
            "scene_type": _optional_string(raw_context.get("scene_type"), "scene_type"),
            "environment": _optional_string(raw_context.get("environment"), "environment"),
            "layout": _optional_string(raw_context.get("layout"), "layout"),
            "lighting": _optional_string(raw_context.get("lighting"), "lighting"),
            "visibility": _optional_string(raw_context.get("visibility"), "visibility"),
            "navigability": _optional_string(raw_context.get("navigability"), "navigability"),
        }
        if not claims and all(value is None for value in context_values.values()):
            raise SemanticResponseParseError(
                "non-abstained scene response requires semantic evidence"
            )
        scene_context = SceneContext(
            source_observation_id=request.source_observation_id,
            perception_result_id=request.perception_result_id,
            provenance=provenance,
            **context_values,
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
    confidence_policy: SemanticConfidencePolicy,
    index: int,
) -> SemanticClaim:
    data = _mapping(value, f"claim[{index}]")
    allowed = {"hypothesis", "role", "category", "region_kind", "attributes", "confidence"}
    if confidence_policy is SemanticConfidencePolicy.UNSCORED_ONLY:
        # Sob esta politica o unico valor legal de confidence e null, entao exigir a chave
        # presente nao acrescenta informacao -- so contradiz a instrucao "never invent
        # confidence" do proprio prompt. Um modelo que obedece e omite a chave perdia a claim
        # inteira: 246 das 457 respostas rejeitadas num run real de 360 frames do corridor-02.
        required = allowed - {"confidence"}
        _require_keys(data, required=required, allowed=allowed, name=f"claim[{index}]")
    else:
        _require_exact_keys(data, allowed, f"claim[{index}]")
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
    if confidence is not None and confidence_policy is SemanticConfidencePolicy.UNSCORED_ONLY:
        raise SemanticResponseParseError(
            f"claim[{index}].confidence must be null; model-reported confidence is uncalibrated"
        )
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
    _require_keys(data, required=keys, allowed=keys, name=name)


def _require_keys(
    data: Mapping[str, Any], *, required: set[str], allowed: set[str], name: str
) -> None:
    missing = required - data.keys()
    unexpected = data.keys() - allowed
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
