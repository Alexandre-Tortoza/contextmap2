"""JSON encode/decode for the public visual evidence model.

Every function here is a pure, symmetric pair (``encode_x``/``decode_x``)
over the dataclasses in :mod:`contextmap.visual_perception.models`. This
is the shape :mod:`contextmap.visual_perception.run_artifact` persists to
``outputs/*.jsonl``, but the functions have no dependency on the artifact
layout themselves — any caller that needs a JSON-serializable view of one
of these contracts can use them directly.
"""

from __future__ import annotations

from typing import Any

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    ClaimId,
    FeatureId,
    FeatureScope,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    Region2D,
    RegionId,
    SceneContext,
    SemanticAttribute,
    SemanticClaim,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticRegionKind,
    VisualFeature,
)
from contextmap.visual_perception.region_models import (
    CoordinateConvention,
    InlineMask,
    RegionProvenance,
)


def encode_provenance(provenance: BackendProvenance) -> dict[str, Any]:
    """Encode a :class:`BackendProvenance` into a JSON-serializable dict."""
    return {
        "backend_id": provenance.backend_id,
        "capability": provenance.capability,
        "provider": provenance.provider,
        "model": provenance.model,
        "version": provenance.version,
        "configuration_fingerprint": provenance.configuration_fingerprint,
    }


def decode_provenance(record: dict[str, Any]) -> BackendProvenance:
    """Decode a :class:`BackendProvenance` from :func:`encode_provenance`'s output."""
    return BackendProvenance(**record)


def encode_bounding_box(box: BoundingBox2D) -> dict[str, float]:
    """Encode a :class:`BoundingBox2D` into a JSON-serializable dict."""
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def decode_bounding_box(record: dict[str, Any]) -> BoundingBox2D:
    """Decode a :class:`BoundingBox2D` from :func:`encode_bounding_box`'s output."""
    return BoundingBox2D(**record)


def encode_region(region: Region2D) -> dict[str, Any]:
    """Encode a :class:`Region2D` into a JSON-serializable dict."""
    return {
        "region_id": str(region.region_id),
        "bounding_box": encode_bounding_box(region.bounding_box),
        "provenance": encode_provenance(region.provenance),
        "mask_reference": region.mask_reference,
        "region_kind": region.region_kind,
        "is_accepted": region.is_accepted,
        "rejection_reason": region.rejection_reason,
        "source_observation_id": (
            None if region.source_observation_id is None else str(region.source_observation_id)
        ),
        "image_width": region.image_width,
        "image_height": region.image_height,
        "area_pixels": region.area_pixels,
        "contributor_candidate_ids": list(region.contributor_candidate_ids),
        "discovery_provenance": [item.to_dict() for item in region.discovery_provenance],
        "mask": None if region.mask is None else region.mask.to_dict(),
        "coordinate_convention": region.coordinate_convention.value,
    }


def decode_region(record: dict[str, Any]) -> Region2D:
    """Decode a :class:`Region2D` from :func:`encode_region`'s output."""
    source_observation_id = record.get("source_observation_id")
    raw_mask = record.get("mask")
    return Region2D(
        region_id=RegionId(record["region_id"]),
        bounding_box=decode_bounding_box(record["bounding_box"]),
        provenance=decode_provenance(record["provenance"]),
        mask_reference=record["mask_reference"],
        region_kind=record["region_kind"],
        is_accepted=record["is_accepted"],
        rejection_reason=record["rejection_reason"],
        source_observation_id=(
            None if source_observation_id is None else SourceObservationId(source_observation_id)
        ),
        image_width=record.get("image_width"),
        image_height=record.get("image_height"),
        area_pixels=record.get("area_pixels"),
        contributor_candidate_ids=tuple(record.get("contributor_candidate_ids", ())),
        discovery_provenance=tuple(
            RegionProvenance.from_dict(item) for item in record.get("discovery_provenance", ())
        ),
        mask=None if raw_mask is None else InlineMask.from_dict(raw_mask),
        coordinate_convention=CoordinateConvention(
            record.get("coordinate_convention", CoordinateConvention.PIXEL_XY_TOP_LEFT.value)
        ),
    )


def encode_feature(feature: VisualFeature) -> dict[str, Any]:
    """Encode a :class:`VisualFeature` into a JSON-serializable dict."""
    return {
        "feature_id": str(feature.feature_id),
        "scope": feature.scope.value,
        "embedding_space_id": feature.embedding_space_id,
        "shape": list(feature.shape),
        "dtype": feature.dtype,
        "payload_reference": feature.payload_reference,
        "provenance": encode_provenance(feature.provenance),
        "region_id": str(feature.region_id) if feature.region_id is not None else None,
        "normalization": feature.normalization,
    }


def decode_feature(record: dict[str, Any]) -> VisualFeature:
    """Decode a :class:`VisualFeature` from :func:`encode_feature`'s output."""
    return VisualFeature(
        feature_id=FeatureId(record["feature_id"]),
        scope=FeatureScope(record["scope"]),
        embedding_space_id=record["embedding_space_id"],
        shape=tuple(record["shape"]),
        dtype=record["dtype"],
        payload_reference=record["payload_reference"],
        provenance=decode_provenance(record["provenance"]),
        region_id=RegionId(record["region_id"]) if record["region_id"] is not None else None,
        normalization=record["normalization"],
    )


def encode_claim(claim: SemanticClaim) -> dict[str, Any]:
    """Encode a :class:`SemanticClaim` into a JSON-serializable dict."""
    return {
        "claim_id": str(claim.claim_id),
        "source_observation_id": str(claim.source_observation_id),
        "perception_result_id": str(claim.perception_result_id),
        "hypothesis": claim.hypothesis,
        "role": claim.role.value,
        "provenance": encode_semantic_provenance(claim.provenance),
        "category": claim.category,
        "region_kind": None if claim.region_kind is None else claim.region_kind.value,
        "attributes": [
            {"name": attribute.name, "value": attribute.value} for attribute in claim.attributes
        ],
        "confidence": claim.confidence,
        "region_id": str(claim.region_id) if claim.region_id is not None else None,
        "evidence_references": [
            {
                "evidence_type": reference.evidence_type,
                "evidence_id": reference.evidence_id,
            }
            for reference in claim.evidence_references
        ],
    }


def decode_claim(record: dict[str, Any]) -> SemanticClaim:
    """Decode a :class:`SemanticClaim` from :func:`encode_claim`'s output."""
    return SemanticClaim(
        claim_id=ClaimId(record["claim_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        perception_result_id=PerceptionResultId(record["perception_result_id"]),
        hypothesis=record["hypothesis"],
        role=HypothesisRole(record["role"]),
        provenance=decode_semantic_provenance(record["provenance"]),
        category=record["category"],
        region_kind=(
            None if record["region_kind"] is None else SemanticRegionKind(record["region_kind"])
        ),
        attributes=tuple(
            SemanticAttribute(name=item["name"], value=item["value"])
            for item in record["attributes"]
        ),
        confidence=record["confidence"],
        region_id=RegionId(record["region_id"]) if record["region_id"] is not None else None,
        evidence_references=tuple(
            SemanticEvidenceReference(
                evidence_type=item["evidence_type"], evidence_id=item["evidence_id"]
            )
            for item in record["evidence_references"]
        ),
    )


def encode_semantic_provenance(
    provenance: SemanticInferenceProvenance,
) -> dict[str, Any]:
    """Encode semantic inference provenance into a JSON-serializable dict."""
    return {
        "backend": encode_provenance(provenance.backend),
        "task_identity": provenance.task_identity,
        "prompt_template_id": provenance.prompt_template_id,
        "output_schema_version": provenance.output_schema_version,
        "raw_response_reference": provenance.raw_response_reference,
    }


def decode_semantic_provenance(record: dict[str, Any]) -> SemanticInferenceProvenance:
    """Decode semantic inference provenance from its canonical representation."""
    return SemanticInferenceProvenance(
        backend=decode_provenance(record["backend"]),
        task_identity=record["task_identity"],
        prompt_template_id=record["prompt_template_id"],
        output_schema_version=record["output_schema_version"],
        raw_response_reference=record["raw_response_reference"],
    )


def encode_scene_context(scene_context: SceneContext) -> dict[str, Any]:
    """Encode a :class:`SceneContext` into a JSON-serializable dict."""
    return {
        "source_observation_id": str(scene_context.source_observation_id),
        "perception_result_id": str(scene_context.perception_result_id),
        "claims": [encode_claim(claim) for claim in scene_context.claims],
        "provenance": encode_semantic_provenance(scene_context.provenance),
        "scene_type": scene_context.scene_type,
        "environment": scene_context.environment,
        "layout": scene_context.layout,
        "lighting": scene_context.lighting,
        "visibility": scene_context.visibility,
        "navigability": scene_context.navigability,
        "evidence_references": [
            {
                "evidence_type": reference.evidence_type,
                "evidence_id": reference.evidence_id,
            }
            for reference in scene_context.evidence_references
        ],
    }


def decode_scene_context(record: dict[str, Any]) -> SceneContext:
    """Decode a :class:`SceneContext` from :func:`encode_scene_context`'s output."""
    return SceneContext(
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        perception_result_id=PerceptionResultId(record["perception_result_id"]),
        claims=tuple(decode_claim(item) for item in record["claims"]),
        provenance=decode_semantic_provenance(record["provenance"]),
        scene_type=record["scene_type"],
        environment=record["environment"],
        layout=record["layout"],
        lighting=record["lighting"],
        visibility=record["visibility"],
        navigability=record["navigability"],
        evidence_references=tuple(
            SemanticEvidenceReference(
                evidence_type=item["evidence_type"], evidence_id=item["evidence_id"]
            )
            for item in record["evidence_references"]
        ),
    )


def encode_perception_result(result: PerceptionResult) -> dict[str, Any]:
    """Encode a :class:`PerceptionResult` into a JSON-serializable dict."""
    return {
        "result_id": str(result.result_id),
        "source_observation_id": str(result.source_observation_id),
        "run_id": str(result.run_id),
        "sequence_artifact_id": result.sequence_artifact_id,
        "created_at": result.created_at,
        "regions": [encode_region(region) for region in result.regions],
        "features": [encode_feature(feature) for feature in result.features],
        "claims": [encode_claim(claim) for claim in result.claims],
        "scene_context": (
            encode_scene_context(result.scene_context) if result.scene_context is not None else None
        ),
    }


def decode_perception_result(record: dict[str, Any]) -> PerceptionResult:
    """Decode a :class:`PerceptionResult` from :func:`encode_perception_result`'s output."""
    return PerceptionResult(
        result_id=PerceptionResultId(record["result_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        run_id=PerceptionRunId(record["run_id"]),
        sequence_artifact_id=record["sequence_artifact_id"],
        created_at=record["created_at"],
        regions=tuple(decode_region(item) for item in record["regions"]),
        features=tuple(decode_feature(item) for item in record["features"]),
        claims=tuple(decode_claim(item) for item in record["claims"]),
        scene_context=(
            decode_scene_context(record["scene_context"])
            if record["scene_context"] is not None
            else None
        ),
    )
