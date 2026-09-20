"""Tests for auditable semantic scoring over canonical visual features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    ClaimId,
    FeatureId,
    FeatureScope,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PipelinePreset,
    ScoreId,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticScore,
    SemanticScoreType,
    StageSpec,
    VisualFeature,
    assemble_perception_result,
    decode_perception_result,
    encode_perception_result,
    execute_stage_graph,
    resolve_pipeline,
)
from contextmap.visual_perception.backends.semantic_scoring import (
    AlphaClipSemanticScorer,
    ClipSemanticScorer,
    SemanticScoringError,
    TextEmbeddingBatch,
)

OBSERVATION_ID = SourceObservationId("frame-0001")
RESULT_ID = PerceptionResultId("result-0001")
SPACE_ID = "sha256:clip-space"


def _claim(*, region_id: str | None = None) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId("claim-0001"),
        source_observation_id=OBSERVATION_ID,
        perception_result_id=RESULT_ID,
        hypothesis="wooden chair",
        role=HypothesisRole.PRIMARY,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="fake-interpreter",
                capability="semantic_interpreter",
                provider="fake",
                model="fake",
                version="1",
            ),
            task_identity="region-labeling",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        ),
        region_id=None if region_id is None else region_id,  # type: ignore[arg-type]
    )


def _feature(*, scope: FeatureScope, region_id: str | None = None) -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId("feature-0001"),
        scope=scope,
        embedding_space_id=SPACE_ID,
        shape=(3,),
        dtype="float32",
        payload_reference="feature-0001.npy",
        provenance=BackendProvenance(
            backend_id="clip-visual",
            capability="feature_extractor",
            provider="openai",
            model="clip-vit-b-32",
            version="1",
        ),
        region_id=None if region_id is None else region_id,  # type: ignore[arg-type]
        normalization="l2",
    )


class _Payloads:
    def load(self, source_observation_id: SourceObservationId, feature_id: FeatureId) -> np.ndarray:
        assert source_observation_id == OBSERVATION_ID
        assert feature_id == FeatureId("feature-0001")
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class _TextEncoder:
    def encode(self, hypotheses: Sequence[str]) -> TextEmbeddingBatch:
        assert tuple(hypotheses) == ("wooden chair",)
        return TextEmbeddingBatch(
            values=np.asarray([[0.6, 0.8, 0.0]], dtype=np.float32),
            embedding_space_id=SPACE_ID,
            normalization="l2",
        )


def _scorer_provenance(backend_id: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id=backend_id,
        capability="semantic_scorer",
        provider="openai",
        model="clip-vit-b-32",
        version="1",
        configuration_fingerprint="sha256:scorer-config",
    )


def test_clip_scores_global_claim_with_raw_cosine_similarity() -> None:
    claim = _claim()
    feature = _feature(scope=FeatureScope.GLOBAL)
    scorer = ClipSemanticScorer(
        provenance=_scorer_provenance("clip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )

    scores = scorer.score((claim,), (feature,))

    assert len(scores) == 1
    assert scores[0].value == pytest.approx(0.6)
    assert scores[0].score_type is SemanticScoreType.COSINE_SIMILARITY
    assert scores[0].calibrated_probability is None
    assert claim.confidence is None


def test_alphaclip_requires_matching_region_feature() -> None:
    claim = _claim(region_id="region-0001")
    feature = _feature(scope=FeatureScope.REGION, region_id="region-0002")
    scorer = AlphaClipSemanticScorer(
        provenance=_scorer_provenance("alphaclip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )

    assert scorer.score((claim,), (feature,)) == ()


def test_scorer_rejects_embedding_space_and_normalization_mismatch() -> None:
    scorer = ClipSemanticScorer(
        provenance=_scorer_provenance("clip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )
    feature = _feature(scope=FeatureScope.GLOBAL)

    with pytest.raises(SemanticScoringError, match="embedding space"):
        scorer.score(
            (_claim(),), (VisualFeature(**{**feature.__dict__, "embedding_space_id": "other"}),)
        )
    with pytest.raises(SemanticScoringError, match="l2-normalized"):
        scorer.score((_claim(),), (VisualFeature(**{**feature.__dict__, "normalization": "none"}),))


def test_semantic_score_is_serialized_separately_from_claim() -> None:
    claim = _claim()
    feature = _feature(scope=FeatureScope.GLOBAL)
    score = SemanticScore(
        score_id=ScoreId("score-0001"),
        claim_id=claim.claim_id,
        feature_id=feature.feature_id,
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=-0.25,
        calibrated_probability=None,
        embedding_space_id=SPACE_ID,
        source_observation_id=OBSERVATION_ID,
        perception_result_id=RESULT_ID,
        provenance=_scorer_provenance("clip-scorer"),
    )
    result = PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=OBSERVATION_ID,
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
        features=(feature,),
        claims=(claim,),
        semantic_scores=(score,),
    )

    restored = decode_perception_result(encode_perception_result(result))

    assert restored.semantic_scores == (score,)
    assert restored.claims[0].confidence is None


def test_semantic_score_rejects_probability_fabrication_and_non_finite_value() -> None:
    values = {
        "score_id": ScoreId("score-0001"),
        "claim_id": ClaimId("claim-0001"),
        "feature_id": FeatureId("feature-0001"),
        "score_type": SemanticScoreType.COSINE_SIMILARITY,
        "value": 0.2,
        "calibrated_probability": None,
        "embedding_space_id": SPACE_ID,
        "source_observation_id": OBSERVATION_ID,
        "perception_result_id": RESULT_ID,
        "provenance": _scorer_provenance("clip-scorer"),
    }
    with pytest.raises(ValueError, match="finite"):
        SemanticScore(**{**values, "value": float("nan")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="calibrated_probability"):
        SemanticScore(**{**values, "calibrated_probability": 1.2})  # type: ignore[arg-type]


def test_semantic_scorer_is_an_executable_stage_graph_capability() -> None:
    scorer = ClipSemanticScorer(
        provenance=_scorer_provenance("clip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )
    preset = PipelinePreset(
        preset_id="semantic-scoring/1",
        stages=(
            StageSpec(stage_id="claims", capability="semantic_claims"),
            StageSpec(stage_id="features", capability="visual_features"),
            StageSpec(
                stage_id="scores",
                capability="semantic_scorer",
                inputs={"claims": "claims", "features": "features"},
                backend_id="clip-scorer",
            ),
        ),
    )
    resolved = resolve_pipeline(preset, backend_factories={"scores": lambda _parameters: scorer})

    outcomes = execute_stage_graph(
        resolved.build_stage_graph(
            {"claims": (_claim(),), "features": (_feature(scope=FeatureScope.GLOBAL),)}
        )
    )

    assert outcomes[-1].output[0].value == pytest.approx(0.6)  # type: ignore[index]
    result = assemble_perception_result(
        result_id=RESULT_ID,
        source_observation_id=OBSERVATION_ID,
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
        outcomes=outcomes,
        claim_stage_ids=("claims",),
        feature_stage_ids=("features",),
        semantic_score_stage_ids=("scores",),
    )
    assert len(result.semantic_scores) == 1
