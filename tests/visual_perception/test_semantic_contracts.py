"""Contract tests for canonical Semantic Interpretation evidence."""

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SEMANTIC_PROMPT_TEMPLATES,
    BackendProvenance,
    ClaimId,
    HypothesisRole,
    PerceptionResultId,
    RegionId,
    SceneContext,
    SemanticAttribute,
    SemanticClaim,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticRegionKind,
)

SOURCE_ID = SourceObservationId("frame-0124")
RESULT_ID = PerceptionResultId("run-0001--frame-0124")


def _provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="fake_semantic_interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="fake-v1",
            version="1",
            configuration_fingerprint="sha256:config",
        ),
        task_identity="region-labeling",
        prompt_template_id="region/v1",
        output_schema_version="semantic-response/1",
        raw_response_reference="debug/40-semantic-interpretation/raw-response.txt",
    )


def _claim(**overrides: object) -> SemanticClaim:
    values: dict[str, object] = {
        "claim_id": ClaimId("claim-0001"),
        "source_observation_id": SOURCE_ID,
        "perception_result_id": RESULT_ID,
        "hypothesis": "wooden pallet",
        "role": HypothesisRole.PRIMARY,
        "provenance": _provenance(),
        "region_id": RegionId("region-0007"),
        "region_kind": SemanticRegionKind.THING,
        "attributes": (SemanticAttribute(name="material", value="wood"),),
        "evidence_references": (
            SemanticEvidenceReference(evidence_type="region", evidence_id="region-0007"),
        ),
    }
    values.update(overrides)
    return SemanticClaim(**values)  # type: ignore[arg-type]


def test_claim_preserves_scored_and_unscored_alternative_hypotheses() -> None:
    primary = _claim(confidence=None)
    alternative = _claim(
        claim_id=ClaimId("claim-0002"),
        hypothesis="wooden crate",
        role=HypothesisRole.ALTERNATIVE,
        confidence=0.42,
    )

    assert primary.confidence is None
    assert alternative.confidence == 0.42
    assert primary.role is HypothesisRole.PRIMARY
    assert alternative.role is HypothesisRole.ALTERNATIVE
    assert primary.attributes[0].value == "wood"


def test_claim_rejects_empty_hypothesis_and_duplicate_evidence() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        _claim(hypothesis="  ")

    reference = SemanticEvidenceReference(evidence_type="region", evidence_id="region-0007")
    with pytest.raises(ValueError, match="evidence"):
        _claim(evidence_references=(reference, reference))


def test_scene_context_is_distinct_from_region_claims_and_keeps_context_fields() -> None:
    scene_claim = _claim(
        claim_id=ClaimId("claim-scene-0001"),
        hypothesis="industrial storage area",
        region_id=None,
        region_kind=None,
    )
    context = SceneContext(
        source_observation_id=SOURCE_ID,
        perception_result_id=RESULT_ID,
        provenance=_provenance(),
        scene_type="warehouse",
        environment="indoor",
        layout="aisles",
        lighting="artificial",
        visibility="clear",
        navigability="partially obstructed",
        claims=(scene_claim,),
        evidence_references=(
            SemanticEvidenceReference(evidence_type="prepared_image", evidence_id="frame-0124"),
        ),
    )

    assert context.scene_type == "warehouse"
    assert context.claims == (scene_claim,)

    with pytest.raises(ValueError, match="region_id"):
        SceneContext(
            source_observation_id=SOURCE_ID,
            perception_result_id=RESULT_ID,
            provenance=_provenance(),
            claims=(_claim(),),
        )


def test_scene_context_rejects_claim_from_another_inference_result() -> None:
    claim = _claim(region_id=None, perception_result_id=PerceptionResultId("another-result"))

    with pytest.raises(ValueError, match="perception_result_id"):
        SceneContext(
            source_observation_id=SOURCE_ID,
            perception_result_id=RESULT_ID,
            provenance=_provenance(),
            claims=(claim,),
        )


def _scene_provenance() -> SemanticInferenceProvenance:
    """``_provenance()`` describes a region call; a SCENE request needs a matching template."""
    from dataclasses import replace

    return replace(_provenance(), prompt_template_id="scene/v1", task_identity="scene-labeling")


def _failed_interpretation() -> object:
    """A real backend call whose response was observed but could not be materialized."""
    import hashlib
    import json

    from contextmap.visual_perception import (
        FailedSemanticInterpretation,
        SemanticBackendDiagnostics,
        SemanticConfidencePolicy,
        SemanticInterpretationMode,
        SemanticInterpretationRequest,
        SemanticParseFailure,
        SemanticRequestId,
        SemanticVisualView,
        VisualViewKind,
        render_semantic_prompt,
    )

    payload = b"pixels"
    view = SemanticVisualView(
        view_id="v-frame-0124-scene",
        kind=VisualViewKind.FULL_FRAME,
        payload_reference="outputs/semantic-views/frame-0124__scene.png",
        source_observation_id=SOURCE_ID,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-frame-0124"),
        source_observation_id=SOURCE_ID,
        perception_result_id=RESULT_ID,
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(view,),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )
    rendered = render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )
    # A real Qwen failure from the campaign: the model omitted the required key.
    raw = json.dumps({"abstained": False, "claims": [{"hypothesis": "a door"}]})
    return FailedSemanticInterpretation(
        request=request,
        rendered_prompt=rendered,
        raw_response=raw,
        raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        provenance=_scene_provenance(),
        diagnostics=SemanticBackendDiagnostics(latency_ms=12.5),
        failure=SemanticParseFailure(
            kind="SemanticResponseParseError",
            message="claim[0] is missing required fields: ['confidence']",
        ),
        effective_configuration={"backend": "qwen"},
        occurred_at="2026-01-01T00:00:00+00:00",
    )


def test_failed_interpretation_preserves_the_observed_response_as_evidence() -> None:
    """An invalid response is still evidence; only its materialization failed (PR #438 review)."""
    failed = _failed_interpretation()

    assert failed.raw_response  # type: ignore[attr-defined]
    assert failed.failure.kind == "SemanticResponseParseError"  # type: ignore[attr-defined]
    assert "confidence" in failed.failure.message  # type: ignore[attr-defined]
    # Shares the attempt identity with the success stream.
    assert str(failed.request.request_id) == "scene-frame-0124"  # type: ignore[attr-defined]
    assert failed.provenance.backend.model == "fake-v1"  # type: ignore[attr-defined]


def test_failed_interpretation_hashes_the_response_exactly_like_the_success_path() -> None:
    """Evidence identity must not depend on whether the parser succeeded."""
    import hashlib

    from contextmap.visual_perception import (
        SemanticConfidencePolicy,
        parse_semantic_response,
    )

    failed = _failed_interpretation()
    raw = failed.raw_response  # type: ignore[attr-defined]
    parsed_ok = parse_semantic_response(
        '{"abstained": true, "claims": [], "scene_context": null}',
        failed.request,  # type: ignore[attr-defined]
        _scene_provenance(),
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )

    assert (
        parsed_ok.raw_response_sha256
        == hashlib.sha256(b'{"abstained": true, "claims": [], "scene_context": null}').hexdigest()
    )
    assert failed.raw_response_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()  # type: ignore[attr-defined]


def test_failed_interpretation_round_trips_through_its_encoding() -> None:
    """The failures stream is a first-class artifact, not a throwaway log."""
    from contextmap.visual_perception import (
        decode_failed_semantic_interpretation,
        encode_failed_semantic_interpretation,
    )

    failed = _failed_interpretation()
    encoded = encode_failed_semantic_interpretation(failed)  # type: ignore[arg-type]

    assert encoded["raw_response_sha256"] == failed.raw_response_sha256  # type: ignore[attr-defined]
    assert encoded["parse_failure"]["kind"] == "SemanticResponseParseError"
    assert decode_failed_semantic_interpretation(encoded) == failed


def test_a_rejected_response_raises_with_the_evidence_attached() -> None:
    """The backend must not reduce a real response to an error string (PR #438 review)."""
    import hashlib
    import json

    from contextmap.visual_perception import (
        SemanticBackendDiagnostics,
        SemanticConfidencePolicy,
        SemanticInterpretationFailedError,
        parse_semantic_response,
        semantic_failure_from_parse_error,
    )
    from contextmap.visual_perception.semantic_prompt import SemanticResponseParseError

    failed = _failed_interpretation()
    raw = failed.raw_response  # type: ignore[attr-defined]

    with pytest.raises(SemanticResponseParseError) as rejected:
        parse_semantic_response(
            raw,
            failed.request,  # type: ignore[attr-defined]
            _scene_provenance(),
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )

    error = semantic_failure_from_parse_error(
        rejected.value,
        request=failed.request,  # type: ignore[attr-defined]
        rendered_prompt=failed.rendered_prompt,  # type: ignore[attr-defined]
        raw_response=raw,
        provenance=_scene_provenance(),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
        effective_configuration={"backend": "qwen"},
    )

    assert isinstance(error, SemanticInterpretationFailedError)
    assert error.failure.raw_response == raw
    assert error.failure.raw_response_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert error.failure.failure.kind == "SemanticResponseParseError"
    # The parser's own message is kept verbatim; no failure taxonomy is invented here.
    assert error.failure.failure.message == str(rejected.value)
    assert json.loads(error.failure.raw_response)  # the observed response is still readable


@pytest.mark.parametrize(
    ("field", "value"),
    [("template_id", "scene/other"), ("output_schema_version", "semantic-response/2")],
)
def test_attempt_evidence_records_only_the_prompt_policy_its_request_selected(
    field: str, value: str
) -> None:
    """#542: no backend can record a prompt other than the one the request names.

    Both outcomes of a backend call are held to it, so an adapter that rendered a hidden
    default (or anything else) cannot publish evidence claiming the requested policy.
    """
    from dataclasses import replace

    from fakes import FakeSemanticInterpreter

    failed = _failed_interpretation()
    substituted = replace(failed.rendered_prompt, **{field: value})  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="prompt"):
        replace(failed, rendered_prompt=substituted)  # type: ignore[type-var]

    execution = FakeSemanticInterpreter(result_id=RESULT_ID).interpret(failed.request)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="prompt"):
        replace(execution, rendered_prompt=replace(execution.rendered_prompt, **{field: value}))
