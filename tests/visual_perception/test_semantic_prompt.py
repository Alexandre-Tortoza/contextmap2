"""Tests for versioned semantic prompts and strict structured parsing."""

import json

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    HypothesisRole,
    PerceptionResultId,
    RegionId,
    SemanticInferenceProvenance,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticPromptTemplate,
    SemanticRequestId,
    SemanticRequestMetadata,
    SemanticResponseParseError,
    SemanticVisualView,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
)


def _request(mode: SemanticInterpretationMode) -> SemanticInterpretationRequest:
    region_id = RegionId("region-0007") if mode is SemanticInterpretationMode.REGION else None
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId(f"{mode.value}-request-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        mode=mode,
        region_id=region_id,
        visual_views=(
            SemanticVisualView(
                view_id="view-0001",
                kind=(
                    VisualViewKind.TIGHT_CROP
                    if region_id is not None
                    else VisualViewKind.FULL_FRAME
                ),
                payload_reference="outputs/views/input.jpg",
                source_observation_id=SourceObservationId("frame-0124"),
                region_id=region_id,
            ),
        ),
        prompt_template_id=f"{mode.value}/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )


def _provenance(mode: SemanticInterpretationMode) -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="fake",
            capability="semantic_interpreter",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="sha256:config",
        ),
        task_identity=f"{mode.value}-interpretation",
        prompt_template_id=f"{mode.value}/v1",
        output_schema_version="semantic-response/1",
        raw_response_reference="debug/raw-response.txt",
    )


def test_prompt_rendering_is_deterministic_and_version_distinguishable() -> None:
    request = _request(SemanticInterpretationMode.REGION)
    v1 = SemanticPromptTemplate.default_for(request.mode)
    v2 = SemanticPromptTemplate(
        template_id="region/v2",
        mode=SemanticInterpretationMode.REGION,
        output_schema_version="semantic-response/1",
        instructions="Describe only the visible region.",
    )

    first = render_semantic_prompt(request, v1)
    second = render_semantic_prompt(request, v1)
    changed = render_semantic_prompt(
        SemanticInterpretationRequest(**{**request.__dict__, "prompt_template_id": "region/v2"}),
        v2,
    )

    assert first == second
    assert first.fingerprint != changed.fingerprint
    assert request.request_id in first.text
    assert "semantic-response/1" in first.text


def test_prompt_includes_supporting_metadata_and_mode_specific_schema() -> None:
    region_request = SemanticInterpretationRequest(
        **{
            **_request(SemanticInterpretationMode.REGION).__dict__,
            "supporting_metadata": (SemanticRequestMetadata(name="camera_height_m", value=1.2),),
        }
    )
    region_prompt = render_semantic_prompt(
        region_request, SemanticPromptTemplate.default_for(region_request.mode)
    )

    assert '"supporting_metadata":[{"name":"camera_height_m","value":1.2}]' in region_prompt.text
    assert '"scene_context":{"type":"null"}' in region_prompt.text
    assert '"minItems":1' in region_prompt.text
    assert '"confidence":{"type":"null"}' in region_prompt.text

    scene_request = _request(SemanticInterpretationMode.SCENE)
    scene_prompt = render_semantic_prompt(
        scene_request, SemanticPromptTemplate.default_for(scene_request.mode)
    )

    assert '"scene_context":{"additionalProperties":false' in scene_prompt.text
    assert '"minItems":0' in scene_prompt.text


def test_region_parser_preserves_primary_alternative_and_unscored_claim() -> None:
    request = _request(SemanticInterpretationMode.REGION)
    raw = json.dumps(
        {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": "wooden pallet",
                    "role": "primary",
                    "category": "load carrier",
                    "region_kind": "thing",
                    "attributes": {"material": "wood"},
                    "confidence": None,
                },
                {
                    "hypothesis": "wooden crate",
                    "role": "alternative",
                    "category": None,
                    "region_kind": "thing",
                    "attributes": {},
                    "confidence": None,
                },
            ],
            "scene_context": None,
        }
    )

    parsed = parse_semantic_response(raw, request, _provenance(request.mode))

    assert [claim.role for claim in parsed.claims] == [
        HypothesisRole.PRIMARY,
        HypothesisRole.ALTERNATIVE,
    ]
    assert parsed.claims[0].hypothesis == "wooden pallet"
    assert parsed.claims[0].confidence is None
    assert parsed.claims[0].attributes[0].value == "wood"
    assert parsed.raw_response_sha256
    assert parsed.scene_context is None


def test_scene_parser_produces_structured_context() -> None:
    request = _request(SemanticInterpretationMode.SCENE)
    raw = json.dumps(
        {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": "warehouse aisle",
                    "role": "primary",
                    "category": None,
                    "region_kind": None,
                    "attributes": {},
                    "confidence": None,
                }
            ],
            "scene_context": {
                "scene_type": "warehouse",
                "environment": "indoor",
                "layout": "aisles",
                "lighting": "artificial",
                "visibility": "clear",
                "navigability": "partially obstructed",
            },
        }
    )

    parsed = parse_semantic_response(raw, request, _provenance(request.mode))

    assert parsed.claims == ()
    assert parsed.scene_context is not None
    assert parsed.scene_context.scene_type == "warehouse"
    assert parsed.scene_context.claims[0].region_id is None


def test_scene_parser_accepts_structured_context_without_redundant_claim() -> None:
    request = _request(SemanticInterpretationMode.SCENE)
    raw = json.dumps(
        {
            "abstained": False,
            "claims": [],
            "scene_context": {"scene_type": "warehouse", "layout": "aisles"},
        }
    )

    parsed = parse_semantic_response(raw, request, _provenance(request.mode))

    assert parsed.scene_context is not None
    assert parsed.scene_context.scene_type == "warehouse"
    assert parsed.scene_context.claims == ()


def test_parser_rejects_model_reported_confidence() -> None:
    request = _request(SemanticInterpretationMode.REGION)
    raw = json.dumps(
        {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": "pallet",
                    "role": "primary",
                    "category": None,
                    "region_kind": "thing",
                    "attributes": {},
                    "confidence": 0.93,
                }
            ],
            "scene_context": None,
        }
    )

    with pytest.raises(SemanticResponseParseError, match="confidence must be null"):
        parse_semantic_response(raw, request, _provenance(request.mode))


def test_parser_records_safe_code_fence_repair_but_rejects_semantic_repairs() -> None:
    request = _request(SemanticInterpretationMode.REGION)
    valid = {
        "abstained": True,
        "claims": [],
        "scene_context": None,
    }
    repaired = parse_semantic_response(
        f"```json\n{json.dumps(valid)}\n```", request, _provenance(request.mode)
    )

    assert repaired.abstained is True
    assert repaired.diagnostics[0].code == "removed_code_fence"

    invalid = {**valid, "abstained": False, "claims": [{"role": "primary"}]}
    with pytest.raises(SemanticResponseParseError, match="hypothesis"):
        parse_semantic_response(json.dumps(invalid), request, _provenance(request.mode))

    invalid_confidence = {
        **valid,
        "abstained": False,
        "claims": [
            {
                "hypothesis": "pallet",
                "role": "primary",
                "category": None,
                "region_kind": "thing",
                "attributes": {},
                "confidence": 9,
            }
        ],
    }
    with pytest.raises(SemanticResponseParseError, match="confidence"):
        parse_semantic_response(json.dumps(invalid_confidence), request, _provenance(request.mode))
