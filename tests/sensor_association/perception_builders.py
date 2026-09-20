"""Deterministic Visual Perception evidence for the association tests."""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    ClaimId,
    FeatureId,
    FeatureScope,
    HypothesisRole,
    InlineMask,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    Region2D,
    RegionId,
    SemanticClaim,
    SemanticInferenceProvenance,
    VisualFeature,
)

RESULT_ID = PerceptionResultId("run-0001--frame-0001")
RUN_ID = PerceptionRunId("run-0001")
OBSERVATION_ID = SourceObservationId("frame-0001")
SEQUENCE_ID = "sequence-0001"

_REGION_BACKEND = BackendProvenance(
    backend_id="fake_region_discovery",
    capability="region_discovery",
    provider="fake",
    model="fake",
    version="0",
)


def rect_mask(width: int, height: int, x0: int, y0: int, x1: int, y1: int) -> InlineMask:
    """A mask that is foreground on ``x0 <= x < x1`` and ``y0 <= y < y1``."""
    return InlineMask(
        width=width,
        height=height,
        data=tuple(x0 <= x < x1 and y0 <= y < y1 for y in range(height) for x in range(width)),
    )


def make_region(
    region_id: str,
    mask: InlineMask | None,
    *,
    accepted: bool = True,
    size: tuple[int, int] | None = None,
) -> Region2D:
    """A region in the space of ``mask`` (or of ``size`` for a box-only region)."""
    width, height = (mask.width, mask.height) if mask is not None else size or (640, 480)
    return Region2D(
        region_id=RegionId(region_id),
        bounding_box=BoundingBox2D(x=0.0, y=0.0, width=10.0, height=10.0),
        provenance=_REGION_BACKEND,
        mask=mask,
        image_width=width,
        image_height=height,
        is_accepted=accepted,
        rejection_reason=None if accepted else "fixture rejection",
        source_observation_id=OBSERVATION_ID,
    )


def make_feature(
    feature_id: str, scope: FeatureScope, *, region_id: str | None = None, space: str = "dinov2:b14"
) -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        scope=scope,
        embedding_space_id=space,
        shape=(4,),
        dtype="float32",
        payload_reference=f"features/{feature_id}.npy",
        provenance=BackendProvenance(
            backend_id="fake_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0",
        ),
        region_id=None if region_id is None else RegionId(region_id),
    )


def make_claim(claim_id: str, *, region_id: str | None = None) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId(claim_id),
        source_observation_id=OBSERVATION_ID,
        perception_result_id=RESULT_ID,
        hypothesis="a hypothesis, not a label",
        role=HypothesisRole.PRIMARY,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="fake_interpreter",
                capability="semantic_interpreter",
                provider="fake",
                model="fake",
                version="0",
            ),
            task_identity="task",
            prompt_template_id="prompt-v1",
            output_schema_version="schema-v1",
        ),
        region_id=None if region_id is None else RegionId(region_id),
    )


def make_result(
    regions: Sequence[Region2D],
    *,
    features: Sequence[VisualFeature] = (),
    claims: Sequence[SemanticClaim] = (),
    observation_id: SourceObservationId = OBSERVATION_ID,
) -> PerceptionResult:
    return PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=observation_id,
        run_id=RUN_ID,
        sequence_artifact_id=SEQUENCE_ID,
        created_at="2026-01-01T00:00:00Z",
        regions=tuple(regions),
        features=tuple(features),
        claims=tuple(claims),
    )
