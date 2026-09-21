"""Tests for versioned semantic prompts and strict structured parsing."""

import json

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    HypothesisRole,
    ParsedSemanticResponse,
    PerceptionResultId,
    RegionId,
    RenderedSemanticPrompt,
    SemanticConfidencePolicy,
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
                payload_reference="outputs/semantic-views/input.jpg",
                source_observation_id=SourceObservationId("frame-0124"),
                region_id=region_id,
                sha256="0" * 64,
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


def _render(
    request: SemanticInterpretationRequest,
    template: SemanticPromptTemplate,
    policy: SemanticConfidencePolicy = SemanticConfidencePolicy.UNSCORED_ONLY,
) -> RenderedSemanticPrompt:
    return render_semantic_prompt(request, template, confidence_policy=policy)


def _parse(
    raw: str,
    request: SemanticInterpretationRequest,
    policy: SemanticConfidencePolicy = SemanticConfidencePolicy.UNSCORED_ONLY,
) -> ParsedSemanticResponse:
    return parse_semantic_response(
        raw,
        request,
        _provenance(request.mode),
        confidence_policy=policy,
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

    first = _render(request, v1)
    second = _render(request, v1)
    changed = _render(
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
    region_prompt = _render(region_request, SemanticPromptTemplate.default_for(region_request.mode))

    assert '"supporting_metadata":[{"name":"camera_height_m","value":1.2}]' in region_prompt.text
    assert '"scene_context":{"type":"null"}' in region_prompt.text
    assert '"minItems":1' in region_prompt.text
    assert '"confidence":{"type":"null"}' in region_prompt.text

    scene_request = _request(SemanticInterpretationMode.SCENE)
    scene_prompt = _render(scene_request, SemanticPromptTemplate.default_for(scene_request.mode))

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

    parsed = _parse(raw, request)

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

    parsed = _parse(raw, request)

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

    parsed = _parse(raw, request)

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
        _parse(raw, request)


def test_parser_preserves_explicitly_measured_confidence() -> None:
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

    parsed = _parse(raw, request, SemanticConfidencePolicy.MEASURED)

    assert parsed.claims[0].confidence == 0.93


def test_scene_parser_rejects_non_abstained_empty_evidence() -> None:
    request = _request(SemanticInterpretationMode.SCENE)
    for scene_context in ({}, {"scene_type": None, "layout": None}):
        raw = json.dumps(
            {
                "abstained": False,
                "claims": [],
                "scene_context": scene_context,
            }
        )

        with pytest.raises(SemanticResponseParseError, match="semantic evidence"):
            _parse(raw, request)


def test_parser_records_safe_code_fence_repair_but_rejects_semantic_repairs() -> None:
    request = _request(SemanticInterpretationMode.REGION)
    valid = {
        "abstained": True,
        "claims": [],
        "scene_context": None,
    }
    repaired = _parse(f"```json\n{json.dumps(valid)}\n```", request)

    assert repaired.abstained is True
    assert repaired.diagnostics[0].code == "removed_code_fence"

    invalid = {**valid, "abstained": False, "claims": [{"role": "primary"}]}
    with pytest.raises(SemanticResponseParseError, match="hypothesis"):
        _parse(json.dumps(invalid), request)

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
        _parse(json.dumps(invalid_confidence), request)


_PRIMARY_CLAIM = {
    "hypothesis": "wooden pallet",
    "role": "primary",
    "category": "load carrier",
    "region_kind": "thing",
    "attributes": {"material": "wood"},
    "confidence": None,
}


def test_region_parser_treats_an_omitted_null_scene_context_as_null_and_records_it() -> None:
    request = _request(SemanticInterpretationMode.REGION)
    with_key = _parse(
        json.dumps({"abstained": False, "claims": [_PRIMARY_CLAIM], "scene_context": None}),
        request,
    )

    without_key = _parse(json.dumps({"abstained": False, "claims": [_PRIMARY_CLAIM]}), request)

    assert without_key.claims == with_key.claims
    assert without_key.scene_context is None
    assert with_key.diagnostics == ()
    assert [item.code for item in without_key.diagnostics] == ["defaulted_null_scene_context"]


def test_region_parser_accepts_an_abstention_without_scene_context() -> None:
    parsed = _parse(
        json.dumps({"abstained": True, "claims": []}),
        _request(SemanticInterpretationMode.REGION),
    )

    assert parsed.abstained is True
    assert [item.code for item in parsed.diagnostics] == ["defaulted_null_scene_context"]


def test_region_parser_records_the_code_fence_and_the_defaulted_key_in_order() -> None:
    raw = "```json\n" + json.dumps({"abstained": False, "claims": [_PRIMARY_CLAIM]}) + "\n```"

    parsed = _parse(raw, _request(SemanticInterpretationMode.REGION))

    assert [item.code for item in parsed.diagnostics] == [
        "removed_code_fence",
        "defaulted_null_scene_context",
    ]


def test_region_parser_still_rejects_a_non_null_scene_context() -> None:
    raw = json.dumps(
        {"abstained": False, "claims": [_PRIMARY_CLAIM], "scene_context": {"scene_type": "x"}}
    )

    with pytest.raises(SemanticResponseParseError, match="scene_context must be null"):
        _parse(raw, _request(SemanticInterpretationMode.REGION))


def test_scene_parser_still_requires_scene_context() -> None:
    raw = json.dumps({"abstained": False, "claims": [_PRIMARY_CLAIM]})

    with pytest.raises(SemanticResponseParseError, match=r"missing required fields.*scene_context"):
        _parse(raw, _request(SemanticInterpretationMode.SCENE))


def test_region_parser_still_rejects_other_missing_or_unexpected_keys() -> None:
    request = _request(SemanticInterpretationMode.REGION)

    with pytest.raises(SemanticResponseParseError, match=r"missing required fields.*claims"):
        _parse(json.dumps({"abstained": False}), request)
    with pytest.raises(SemanticResponseParseError, match=r"unexpected fields.*notes"):
        _parse(
            json.dumps({"abstained": False, "claims": [_PRIMARY_CLAIM], "notes": "extra"}),
            request,
        )
