"""CLIP-family semantic scorers over persisted canonical visual features."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import (
    BackendProvenance,
    FeatureId,
    FeatureScope,
    ScoreId,
    SemanticClaim,
    SemanticScore,
    SemanticScoreType,
    VisualFeature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SemanticScoringError(ValueError):
    """Raised when canonical evidence cannot be scored without changing semantics."""


@dataclass(frozen=True, kw_only=True)
class TextEmbeddingBatch:
    """SDK-free text embeddings returned by a scorer-specific runtime."""

    values: NDArray[Any]
    embedding_space_id: str
    normalization: str

    def __post_init__(self) -> None:
        """Validate batch shape and declared metadata."""
        if self.values.ndim != 2 or any(dimension <= 0 for dimension in self.values.shape):
            raise ValueError("text embedding values must have shape (hypothesis_count, dimension)")
        if not self.embedding_space_id:
            raise ValueError("text embedding_space_id must not be empty")
        if not self.normalization:
            raise ValueError("text normalization must not be empty")


class FeaturePayloadSource(Protocol):
    """Minimal lazy feature-payload boundary required by semantic scoring."""

    def load(
        self, source_observation_id: SourceObservationId, feature_id: FeatureId
    ) -> NDArray[Any]:
        """Load one canonical visual feature payload."""
        ...


class TextEmbeddingEncoder(Protocol):
    """Model-specific text projection into the configured visual space."""

    def encode(self, hypotheses: Sequence[str]) -> TextEmbeddingBatch:
        """Encode hypotheses in input order."""
        ...


class _CosineSemanticScorer:
    """Shared exact cosine rule for model-specific capability adapters."""

    _scope: FeatureScope

    def __init__(
        self,
        *,
        provenance: BackendProvenance,
        embedding_space_id: str,
        payloads: FeaturePayloadSource,
        text_encoder: TextEmbeddingEncoder,
    ) -> None:
        if provenance.capability != "semantic_scorer":
            raise ValueError("scorer provenance capability must be semantic_scorer")
        for field_name, value in (
            ("backend_id", provenance.backend_id),
            ("provider", provenance.provider),
            ("model", provenance.model),
            ("version", provenance.version),
            ("configuration_fingerprint", provenance.configuration_fingerprint),
        ):
            if value is None or not value.strip():
                raise ValueError(f"scorer provenance {field_name} must not be empty")
        if not embedding_space_id:
            raise ValueError("embedding_space_id must not be empty")
        self._provenance = provenance
        self._embedding_space_id = embedding_space_id
        self._payloads = payloads
        self._text_encoder = text_encoder

    def backend_provenance(self) -> BackendProvenance:
        """Return the immutable scorer/model identity."""
        return self._provenance

    def score(
        self, claims: Sequence[SemanticClaim], features: Sequence[VisualFeature]
    ) -> Sequence[SemanticScore]:
        """Score every claim/feature pair compatible with this adapter's scope."""
        if not claims:
            return ()
        batch = self._text_encoder.encode(tuple(claim.hypothesis for claim in claims))
        if batch.values.shape[0] != len(claims):
            raise SemanticScoringError("text encoder returned the wrong hypothesis count")
        self._validate_space(batch.embedding_space_id, "text")
        if batch.normalization != "l2":
            raise SemanticScoringError("text embeddings must be declared l2-normalized")

        scores: list[SemanticScore] = []
        for claim_index, claim in enumerate(claims):
            candidates = tuple(
                feature
                for feature in features
                if feature.scope is self._scope and self._matches_claim_scope(claim, feature)
            )
            for feature in candidates:
                self._validate_feature(feature)
                visual = self._payloads.load(claim.source_observation_id, feature.feature_id)
                text = batch.values[claim_index]
                value = _normalized_dot(text, visual, expected_shape=feature.shape)
                scores.append(
                    SemanticScore(
                        score_id=_score_id(claim, feature, self._provenance),
                        claim_id=claim.claim_id,
                        feature_id=feature.feature_id,
                        score_type=SemanticScoreType.COSINE_SIMILARITY,
                        value=value,
                        calibrated_probability=None,
                        embedding_space_id=self._embedding_space_id,
                        source_observation_id=claim.source_observation_id,
                        perception_result_id=claim.perception_result_id,
                        provenance=self._provenance,
                    )
                )
        return tuple(scores)

    def _matches_claim_scope(self, claim: SemanticClaim, feature: VisualFeature) -> bool:
        if self._scope is FeatureScope.GLOBAL:
            return claim.region_id is None
        return claim.region_id is not None and feature.region_id == claim.region_id

    def _validate_space(self, embedding_space_id: str, source: str) -> None:
        if embedding_space_id != self._embedding_space_id:
            raise SemanticScoringError(
                f"{source} embedding space {embedding_space_id!r} does not match scorer "
                f"embedding space {self._embedding_space_id!r}"
            )

    def _validate_feature(self, feature: VisualFeature) -> None:
        self._validate_space(feature.embedding_space_id, "visual feature")
        if feature.normalization != "l2":
            raise SemanticScoringError("visual features must be declared l2-normalized")
        if len(feature.shape) != 1:
            raise SemanticScoringError("semantic scoring requires one-dimensional feature vectors")


class ClipSemanticScorer(_CosineSemanticScorer):
    """Score scene claims against global CLIP features."""

    _scope = FeatureScope.GLOBAL


class AlphaClipSemanticScorer(_CosineSemanticScorer):
    """Score region claims against AlphaCLIP features for the same frozen region."""

    _scope = FeatureScope.REGION


def _normalized_dot(
    text: NDArray[Any], visual: NDArray[Any], *, expected_shape: tuple[int, ...]
) -> float:
    import numpy as np

    if tuple(visual.shape) != expected_shape or visual.ndim != 1:
        raise SemanticScoringError(
            f"visual payload shape {tuple(visual.shape)} does not match feature shape "
            f"{expected_shape}"
        )
    if text.ndim != 1 or text.shape != visual.shape:
        raise SemanticScoringError("text and visual embedding dimensions do not match")
    if not np.isfinite(text).all() or not np.isfinite(visual).all():
        raise SemanticScoringError("text and visual embeddings must contain only finite values")
    text_norm = float(np.linalg.norm(text))
    visual_norm = float(np.linalg.norm(visual))
    if not math.isclose(text_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise SemanticScoringError("text embedding is not l2-normalized")
    if not math.isclose(visual_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise SemanticScoringError("visual feature is not l2-normalized")
    value = float(np.dot(text, visual))
    return max(-1.0, min(1.0, value))


def _score_id(
    claim: SemanticClaim, feature: VisualFeature, provenance: BackendProvenance
) -> ScoreId:
    payload = json.dumps(
        {
            "claim_id": str(claim.claim_id),
            "feature_id": str(feature.feature_id),
            "perception_result_id": str(claim.perception_result_id),
            "scorer_id": provenance.backend_id,
            "configuration_fingerprint": provenance.configuration_fingerprint,
            "score_type": SemanticScoreType.COSINE_SIMILARITY.value,
        },
        sort_keys=True,
    ).encode("utf-8")
    return ScoreId(f"score-{hashlib.sha256(payload).hexdigest()[:24]}")
