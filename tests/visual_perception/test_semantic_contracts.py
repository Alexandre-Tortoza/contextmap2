"""Contract tests for canonical Semantic Interpretation evidence."""

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
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
