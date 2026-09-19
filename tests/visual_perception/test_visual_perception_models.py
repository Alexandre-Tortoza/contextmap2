import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    FeatureScope,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRun,
    PerceptionRunId,
    Region2D,
    RegionId,
    SceneContext,
    SemanticClaim,
    SemanticInferenceProvenance,
    VisualFeature,
)
from contextmap.visual_perception.models import ClaimId, FeatureId


def _provenance(capability: str = "region_discovery") -> BackendProvenance:
    return BackendProvenance(
        backend_id="fake", capability=capability, provider="fake", model="fake", version="0.1"
    )


def _region(region_id: str = "region-0001", **overrides: object) -> Region2D:
    defaults: dict[str, object] = {
        "region_id": RegionId(region_id),
        "bounding_box": BoundingBox2D(x=0, y=0, width=10, height=10),
        "provenance": _provenance(),
    }
    defaults.update(overrides)
    return Region2D(**defaults)  # type: ignore[arg-type]


def _semantic_provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=_provenance("semantic_interpreter"),
        task_identity="region-labeling",
        prompt_template_id="region/v1",
        output_schema_version="semantic-response/1",
    )


def _semantic_claim(**overrides: object) -> SemanticClaim:
    defaults: dict[str, object] = {
        "claim_id": ClaimId("claim-0001"),
        "source_observation_id": SourceObservationId("frame-0124"),
        "perception_result_id": PerceptionResultId("result-0001"),
        "hypothesis": "a doorway",
        "role": HypothesisRole.PRIMARY,
        "provenance": _semantic_provenance(),
    }
    defaults.update(overrides)
    return SemanticClaim(**defaults)  # type: ignore[arg-type]


def test_region_bounding_box_rejects_non_positive_extent() -> None:
    with pytest.raises(ValueError, match="positive"):
        BoundingBox2D(x=0, y=0, width=0, height=10)


def test_rejected_region_requires_reason_only_when_rejected() -> None:
    with pytest.raises(ValueError, match="rejection_reason"):
        _region(is_accepted=True, rejection_reason="too small")

    rejected = _region(is_accepted=False, rejection_reason="too small")
    assert rejected.rejection_reason == "too small"


def test_visual_feature_requires_region_id_only_when_region_scoped() -> None:
    with pytest.raises(ValueError, match="region_id is required"):
        VisualFeature(
            feature_id=FeatureId("feat-0001"),
            scope=FeatureScope.REGION,
            embedding_space_id="dinov3-vitl",
            shape=(384,),
            dtype="float32",
            payload_reference="features/feat-0001.bin",
            provenance=_provenance("feature_extractor"),
        )

    with pytest.raises(ValueError, match="must be None"):
        VisualFeature(
            feature_id=FeatureId("feat-0001"),
            scope=FeatureScope.DENSE,
            embedding_space_id="dinov3-vitl",
            shape=(64, 64, 384),
            dtype="float32",
            payload_reference="features/feat-0001.bin",
            provenance=_provenance("feature_extractor"),
            region_id=RegionId("region-0001"),
        )

    dense = VisualFeature(
        feature_id=FeatureId("feat-0002"),
        scope=FeatureScope.DENSE,
        embedding_space_id="dinov3-vitl",
        shape=(64, 64, 384),
        dtype="float32",
        payload_reference="features/feat-0002.bin",
        provenance=_provenance("feature_extractor"),
    )
    assert dense.region_id is None


def test_semantic_claim_confidence_none_is_distinguishable_from_scored() -> None:
    unscored = _semantic_claim()
    scored = _semantic_claim(
        claim_id=ClaimId("claim-0002"),
        confidence=1.0,
    )

    assert unscored.confidence is None
    assert scored.confidence == 1.0
    assert unscored.confidence != scored.confidence


def test_semantic_claim_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _semantic_claim(confidence=1.5)


def test_alternative_and_primary_claims_remain_distinguishable() -> None:
    primary = _semantic_claim()
    alternative = _semantic_claim(
        claim_id=ClaimId("claim-0002"),
        hypothesis="a window",
        role=HypothesisRole.ALTERNATIVE,
    )

    assert primary.role is HypothesisRole.PRIMARY
    assert alternative.role is HypothesisRole.ALTERNATIVE


def test_scene_context_rejects_region_scoped_claims() -> None:
    region_scoped_claim = _semantic_claim(region_id=RegionId("region-0001"))

    with pytest.raises(ValueError, match="region_id"):
        SceneContext(
            source_observation_id=SourceObservationId("frame-0124"),
            perception_result_id=PerceptionResultId("result-0001"),
            claims=(region_scoped_claim,),
            provenance=_semantic_provenance(),
        )


def test_perception_result_rejects_duplicate_region_id() -> None:
    with pytest.raises(ValueError, match="duplicate region_id"):
        PerceptionResult(
            result_id=PerceptionResultId("result-0001"),
            source_observation_id=SourceObservationId("frame-0124"),
            run_id=PerceptionRunId("run-0001"),
            sequence_artifact_id="corridor-02-a1b2c3",
            created_at="2026-01-01T00:00:00+00:00",
            regions=(_region("region-0001"), _region("region-0001")),
        )


def test_perception_result_rejects_duplicate_feature_id() -> None:
    feature = VisualFeature(
        feature_id=FeatureId("feat-0001"),
        scope=FeatureScope.DENSE,
        embedding_space_id="dinov3-vitl",
        shape=(64, 64, 384),
        dtype="float32",
        payload_reference="features/feat-0001.bin",
        provenance=_provenance("feature_extractor"),
    )

    with pytest.raises(ValueError, match="duplicate feature_id"):
        PerceptionResult(
            result_id=PerceptionResultId("result-0001"),
            source_observation_id=SourceObservationId("frame-0124"),
            run_id=PerceptionRunId("run-0001"),
            sequence_artifact_id="corridor-02-a1b2c3",
            created_at="2026-01-01T00:00:00+00:00",
            features=(feature, feature),
        )


def test_perception_result_rejects_duplicate_claim_id() -> None:
    claim = _semantic_claim()

    with pytest.raises(ValueError, match="duplicate claim_id"):
        PerceptionResult(
            result_id=PerceptionResultId("result-0001"),
            source_observation_id=SourceObservationId("frame-0124"),
            run_id=PerceptionRunId("run-0001"),
            sequence_artifact_id="corridor-02-a1b2c3",
            created_at="2026-01-01T00:00:00+00:00",
            claims=(claim, claim),
        )


def test_perception_result_rejects_feature_referencing_unknown_region() -> None:
    dangling_feature = VisualFeature(
        feature_id=FeatureId("feat-0001"),
        scope=FeatureScope.REGION,
        embedding_space_id="alphaclip-vitl",
        shape=(768,),
        dtype="float32",
        payload_reference="features/feat-0001.bin",
        provenance=_provenance("feature_extractor"),
        region_id=RegionId("region-does-not-exist"),
    )

    with pytest.raises(ValueError, match="unknown region_id"):
        PerceptionResult(
            result_id=PerceptionResultId("result-0001"),
            source_observation_id=SourceObservationId("frame-0124"),
            run_id=PerceptionRunId("run-0001"),
            sequence_artifact_id="corridor-02-a1b2c3",
            created_at="2026-01-01T00:00:00+00:00",
            regions=(_region("region-0001"),),
            features=(dangling_feature,),
        )


def test_repeated_processing_of_one_observation_yields_distinct_results() -> None:
    """Same physical frame, two runs, two distinct PerceptionResult identities."""
    source_observation_id = SourceObservationId("frame-0124")

    result_a = PerceptionResult(
        result_id=PerceptionResultId("result-0001"),
        source_observation_id=source_observation_id,
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
    )
    result_b = PerceptionResult(
        result_id=PerceptionResultId("result-0002"),
        source_observation_id=source_observation_id,
        run_id=PerceptionRunId("run-0002"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:05:00+00:00",
    )

    assert result_a.result_id != result_b.result_id
    assert result_a.source_observation_id == result_b.source_observation_id == source_observation_id


def test_dense_feature_can_exist_without_any_region() -> None:
    dense = VisualFeature(
        feature_id=FeatureId("feat-0001"),
        scope=FeatureScope.DENSE,
        embedding_space_id="dinov3-vitl",
        shape=(64, 64, 384),
        dtype="float32",
        payload_reference="features/feat-0001.bin",
        provenance=_provenance("feature_extractor"),
    )
    result = PerceptionResult(
        result_id=PerceptionResultId("result-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(),
        features=(dense,),
    )

    assert result.regions == ()
    assert result.features == (dense,)


def test_perception_run_rejects_negative_run_index() -> None:
    with pytest.raises(ValueError, match="run_index"):
        PerceptionRun(
            run_id=PerceptionRunId("run-0001"),
            run_index=-1,
            sequence_artifact_id="corridor-02-a1b2c3",
            selection_id="sha256:aaaa",
            enabled_capabilities=frozenset({"region_discovery"}),
            backend_provenance={"region_discovery": _provenance()},
        )
