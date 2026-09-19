from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    FeatureScope,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    Region2D,
    RegionId,
    SceneContext,
    SemanticClaim,
    SemanticInferenceProvenance,
    VisualFeature,
    decode_perception_result,
    encode_perception_result,
)
from contextmap.visual_perception.models import ClaimId, FeatureId

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0.1"
)
_SEMANTIC_PROVENANCE = SemanticInferenceProvenance(
    backend=BackendProvenance(
        backend_id="fake-semantic",
        capability="semantic_interpreter",
        provider="fake",
        model="fake",
        version="0.1",
    ),
    task_identity="region-labeling",
    prompt_template_id="region/v1",
    output_schema_version="semantic-response/1",
)


def test_perception_result_with_full_evidence_round_trips() -> None:
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=1, y=2, width=10, height=20),
        provenance=_PROVENANCE,
        mask_reference="debug/frame-0001/regions/region-0001/mask.png",
        region_kind="object",
    )
    feature = VisualFeature(
        feature_id=FeatureId("feat-0001"),
        scope=FeatureScope.REGION,
        embedding_space_id="alphaclip-vitl",
        shape=(768,),
        dtype="float32",
        payload_reference="features/feat-0001.bin",
        provenance=_PROVENANCE,
        region_id=region.region_id,
        normalization="l2",
    )
    claim = SemanticClaim(
        claim_id=ClaimId("claim-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        hypothesis="a doorway",
        role=HypothesisRole.ALTERNATIVE,
        provenance=_SEMANTIC_PROVENANCE,
        category="architecture",
        confidence=0.42,
        region_id=region.region_id,
    )
    scene_claim = SemanticClaim(
        claim_id=ClaimId("claim-scene-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        hypothesis="an indoor corridor",
        role=HypothesisRole.PRIMARY,
        provenance=_SEMANTIC_PROVENANCE,
    )
    result = PerceptionResult(
        result_id=PerceptionResultId("run-0001--frame-0124"),
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(region,),
        features=(feature,),
        claims=(claim,),
        scene_context=SceneContext(
            source_observation_id=SourceObservationId("frame-0124"),
            perception_result_id=PerceptionResultId("run-0001--frame-0124"),
            claims=(scene_claim,),
            provenance=_SEMANTIC_PROVENANCE,
        ),
    )

    decoded = decode_perception_result(encode_perception_result(result))

    assert decoded == result


def test_perception_result_with_no_evidence_round_trips() -> None:
    result = PerceptionResult(
        result_id=PerceptionResultId("run-0001--frame-0124"),
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
    )

    decoded = decode_perception_result(encode_perception_result(result))

    assert decoded == result
    assert decoded.scene_context is None
