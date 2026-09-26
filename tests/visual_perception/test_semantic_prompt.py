"""Tests for versioned semantic prompts and strict structured parsing."""

import hashlib
import json
from dataclasses import replace

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SCENE_CONTEXT_RENDERING,
    SEMANTIC_PROMPT_TEMPLATES,
    BackendProvenance,
    ClaimId,
    HypothesisRole,
    ParsedSemanticResponse,
    PerceptionResultId,
    RegionId,
    RenderedSemanticPrompt,
    SceneContext,
    SemanticClaim,
    SemanticConfidencePolicy,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticPromptPolicy,
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
    v1 = SEMANTIC_PROMPT_TEMPLATES["region/v1"]
    alternative = SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"]

    first = _render(request, v1)
    second = _render(request, v1)
    changed = _render(replace(request, prompt_template_id="region-abstention/v1"), alternative)

    assert first == second
    assert first.fingerprint != changed.fingerprint
    assert changed.template_id == "region-abstention/v1"
    assert changed.output_schema_version == first.output_schema_version
    assert request.request_id in first.text
    assert "semantic-response/1" in first.text


def test_canonical_policies_render_byte_for_byte_as_before_they_became_explicit() -> None:
    """#542: ``region/v1``/``scene/v1`` left a hidden default for the explicit catalog.

    The fingerprints were rendered by the pre-#542 ``SemanticPromptTemplate.default_for()``;
    moving the canonical policies into the catalog must not change a single byte of them.
    """
    region = _render(
        _request(SemanticInterpretationMode.REGION), SEMANTIC_PROMPT_TEMPLATES["region/v1"]
    )
    scene = _render(
        _request(SemanticInterpretationMode.SCENE), SEMANTIC_PROMPT_TEMPLATES["scene/v1"]
    )

    assert region.fingerprint == (
        "sha256:47703b45c13d8fd3f5a7e4edb85a8fc470e9c75ad96a527dfeda6c1c89a0766d"
    )
    assert scene.fingerprint == (
        "sha256:920a9c590207b8c3e7877e360f5097bbaf3f9f7fb0d5a9971bc6cb3f4df52644"
    )


def test_a_published_policy_identity_keeps_its_content() -> None:
    """An edited instruction is a new policy with a new identity, never the same one."""
    request = replace(
        _request(SemanticInterpretationMode.REGION), prompt_template_id="region-abstention/v1"
    )

    rendered = _render(request, SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"])

    assert rendered.fingerprint == (
        "sha256:55ebbed97b77f2d89f4e8243ce885cc902f2cd8ece736f25f525d636fbca7766"
    )


def test_the_catalog_keys_every_template_by_its_own_versioned_identity() -> None:
    assert {"scene/v1", "region/v1", "region-abstention/v1"} <= set(SEMANTIC_PROMPT_TEMPLATES)
    for template_id, template in SEMANTIC_PROMPT_TEMPLATES.items():
        assert template.template_id == template_id
        assert template.output_schema_version == "semantic-response/1"


def test_a_prompt_policy_selects_one_catalog_template_per_mode() -> None:
    policy = SemanticPromptPolicy(scene="scene/v1", region="region-abstention/v1")

    assert (
        policy.template_for(SemanticInterpretationMode.SCENE)
        is SEMANTIC_PROMPT_TEMPLATES["scene/v1"]
    )
    assert (
        policy.template_for(SemanticInterpretationMode.REGION)
        is SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"]
    )


@pytest.mark.parametrize(
    ("scene", "region", "message"),
    [
        ("scene/v9", "region/v1", "unknown semantic prompt template 'scene/v9'"),
        ("scene/v1", "region/v9", "unknown semantic prompt template 'region/v9'"),
        ("region/v1", "region/v1", "scene prompt policy 'region/v1' is a region template"),
        ("scene/v1", "scene/v1", "region prompt policy 'scene/v1' is a scene template"),
    ],
)
def test_a_prompt_policy_refuses_an_unknown_or_cross_mode_template(
    scene: str, region: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SemanticPromptPolicy(scene=scene, region=region)


def test_a_template_must_declare_the_output_schema_the_renderer_and_parser_implement() -> None:
    with pytest.raises(ValueError, match="semantic-response/1"):
        SemanticPromptTemplate(
            template_id="region/schema-2",
            mode=SemanticInterpretationMode.REGION,
            output_schema_version="semantic-response/2",
            instructions="Describe only the referenced region.",
        )


def test_rendering_refuses_a_template_the_request_did_not_select() -> None:
    region = _request(SemanticInterpretationMode.REGION)
    scene = _request(SemanticInterpretationMode.SCENE)

    with pytest.raises(ValueError, match="identity must match"):
        _render(region, SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"])
    with pytest.raises(ValueError, match="mode must match"):
        _render(scene, SEMANTIC_PROMPT_TEMPLATES["region/v1"])
    with pytest.raises(ValueError, match="schema must match"):
        _render(
            replace(region, requested_output_schema="semantic-response/2"),
            SEMANTIC_PROMPT_TEMPLATES["region/v1"],
        )


def test_a_rendered_prompt_cannot_carry_the_fingerprint_of_other_text() -> None:
    rendered = _render(
        _request(SemanticInterpretationMode.REGION), SEMANTIC_PROMPT_TEMPLATES["region/v1"]
    )

    assert rendered.fingerprint == (
        "sha256:" + hashlib.sha256(rendered.text.encode("utf-8")).hexdigest()
    )
    with pytest.raises(ValueError, match="fingerprint"):
        replace(rendered, text=rendered.text + " ")


def test_prompt_includes_supporting_metadata_and_mode_specific_schema() -> None:
    region_request = replace(
        _request(SemanticInterpretationMode.REGION),
        supporting_metadata=(SemanticRequestMetadata(name="camera_height_m", value=1.2),),
    )
    region_prompt = _render(region_request, SEMANTIC_PROMPT_TEMPLATES["region/v1"])

    assert '"supporting_metadata":[{"name":"camera_height_m","value":1.2}]' in region_prompt.text
    assert '"scene_context":{"type":"null"}' in region_prompt.text
    assert '"minItems":1' in region_prompt.text
    assert '"confidence":{"type":"null"}' in region_prompt.text

    scene_request = _request(SemanticInterpretationMode.SCENE)
    scene_prompt = _render(scene_request, SEMANTIC_PROMPT_TEMPLATES["scene/v1"])

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


def test_an_omitted_confidence_parses_as_null_when_only_null_is_legal() -> None:
    """The dominant real failure: the prompt and the schema contradicted each other.

    `region/v1` instructs the model to "never invent confidence", while the claim schema
    marked `confidence` required and the parser rejected the claim when the key was absent.
    Under UNSCORED_ONLY the only legal value is null, so demanding the key is ceremony that
    cost 246 of 457 rejected responses in a real 360-frame corridor-02 run.
    """
    request = _request(SemanticInterpretationMode.REGION)
    raw = json.dumps(
        {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": "wooden pallet",
                    "role": "primary",
                    "category": None,
                    "region_kind": "thing",
                    "attributes": {},
                }
            ],
            "scene_context": None,
        }
    )

    parsed = parse_semantic_response(
        raw,
        request,
        _provenance(SemanticInterpretationMode.REGION),
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )

    assert parsed.claims[0].hypothesis == "wooden pallet"
    assert parsed.claims[0].confidence is None, "an absent confidence is null, never invented"


def test_an_omitted_confidence_is_still_refused_when_a_number_is_meaningful() -> None:
    """Under MEASURED the value carries information, so its absence stays an error."""
    request = _request(SemanticInterpretationMode.REGION)
    raw = json.dumps(
        {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": "wooden pallet",
                    "role": "primary",
                    "category": None,
                    "region_kind": "thing",
                    "attributes": {},
                }
            ],
            "scene_context": None,
        }
    )

    with pytest.raises(SemanticResponseParseError, match="confidence"):
        parse_semantic_response(
            raw,
            request,
            _provenance(SemanticInterpretationMode.REGION),
            confidence_policy=SemanticConfidencePolicy.MEASURED,
        )


def _scene_context(**fields: object) -> SceneContext:
    values: dict[str, object] = {
        "source_observation_id": SourceObservationId("frame-0124"),
        "perception_result_id": PerceptionResultId("run-0001--frame-0124"),
        "provenance": _provenance(SemanticInterpretationMode.SCENE),
        "scene_type": "warehouse aisle",
        "lighting": "artificial",
    }
    values.update(fields)
    return SceneContext(**values)  # type: ignore[arg-type]


def _conditioned(context: SceneContext | None) -> SemanticInterpretationRequest:
    return replace(
        _request(SemanticInterpretationMode.REGION),
        prompt_template_id="region-scene-context/v1",
        scene_context_reference=(
            None
            if context is None
            else SemanticEvidenceReference(
                evidence_type="scene_context", evidence_id=str(context.perception_result_id)
            )
        ),
        scene_context=context,
    )


class TestSceneContextConditioning:
    """#529: a prior scene hypothesis is rendered explicitly, deterministically and versioned."""

    def test_the_selected_scene_context_is_rendered_as_a_prior_hypothesis(self) -> None:
        template = SEMANTIC_PROMPT_TEMPLATES["region-scene-context/v1"]

        rendered = _render(_conditioned(_scene_context()), template)

        assert template.scene_context_instructions is not None
        assert template.scene_context_instructions in rendered.text
        assert '"scene_type":"warehouse aisle"' in rendered.text
        assert '"lighting":"artificial"' in rendered.text
        assert f"Scene context ({SCENE_CONTEXT_RENDERING}):" in rendered.text
        assert rendered == _render(_conditioned(_scene_context()), template)

    def test_the_scene_claims_are_rendered_without_any_confidence(self) -> None:
        claim = SemanticClaim(
            claim_id=ClaimId("run-0001--frame-0124--claim-0000"),
            source_observation_id=SourceObservationId("frame-0124"),
            perception_result_id=PerceptionResultId("run-0001--frame-0124"),
            hypothesis="loading dock",
            role=HypothesisRole.PRIMARY,
            provenance=_provenance(SemanticInterpretationMode.SCENE),
            confidence=0.9,
        )

        rendered = _render(
            _conditioned(_scene_context(claims=(claim,))),
            SEMANTIC_PROMPT_TEMPLATES["region-scene-context/v1"],
        )

        assert '"hypothesis":"loading dock"' in rendered.text
        assert "0.9" not in rendered.text

    def test_disabling_the_context_keeps_the_template_and_says_so_explicitly(self) -> None:
        template = SEMANTIC_PROMPT_TEMPLATES["region-scene-context/v1"]

        without = _render(_conditioned(None), template)
        with_context = _render(_conditioned(_scene_context()), template)

        assert f"Scene context ({SCENE_CONTEXT_RENDERING}): none" in without.text
        assert template.scene_context_instructions is not None
        assert template.scene_context_instructions not in without.text
        assert without.template_id == with_context.template_id
        assert without.fingerprint != with_context.fingerprint

    def test_a_template_that_cannot_render_scene_context_refuses_one(self) -> None:
        request = replace(_conditioned(_scene_context()), prompt_template_id="region/v1")

        with pytest.raises(ValueError, match="does not render scene context"):
            _render(request, SEMANTIC_PROMPT_TEMPLATES["region/v1"])

    def test_the_canonical_region_policy_never_renders_a_scene_context_section(self) -> None:
        rendered = _render(
            _request(SemanticInterpretationMode.REGION), SEMANTIC_PROMPT_TEMPLATES["region/v1"]
        )

        assert "Scene context" not in rendered.text

    def test_only_a_region_template_may_render_scene_context(self) -> None:
        with pytest.raises(ValueError, match="scene"):
            SemanticPromptTemplate(
                template_id="scene/conditioned",
                mode=SemanticInterpretationMode.SCENE,
                output_schema_version="semantic-response/1",
                instructions="Describe the scene.",
                scene_context_instructions="Use the prior scene.",
            )

    def test_a_policy_conditions_region_requests_only_with_a_template_that_renders_it(
        self,
    ) -> None:
        enabled = SemanticPromptPolicy(
            scene="scene/v1", region="region-scene-context/v1", region_scene_context=True
        )
        disabled = SemanticPromptPolicy(scene="scene/v1", region="region-scene-context/v1")

        assert enabled.region_scene_context
        assert not disabled.region_scene_context
        with pytest.raises(ValueError, match="does not render scene context"):
            SemanticPromptPolicy(scene="scene/v1", region="region/v1", region_scene_context=True)
