"""Canonical request boundary for scene and region semantic interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import (
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    RegionId,
    SemanticEvidenceReference,
)
from contextmap.visual_perception.region_models import JsonScalar

SemanticRequestId = NewType("SemanticRequestId", str)
"""Identity of one auditable semantic backend request."""


class SemanticInterpretationMode(Enum):
    """Scope of the semantic inference requested from an interpreter."""

    SCENE = "scene"
    REGION = "region"


class VisualViewKind(Enum):
    """Canonical visual evidence views that an interpreter may consume."""

    FULL_FRAME = "full_frame"
    MASKED_SUBJECT = "masked_subject"
    TIGHT_CROP = "tight_crop"
    CONTEXTUAL_CROP = "contextual_crop"


@dataclass(frozen=True, kw_only=True)
class SemanticVisualView:
    """Reference one exact image view supplied to semantic inference."""

    view_id: str
    kind: VisualViewKind
    payload_reference: str
    source_observation_id: SourceObservationId
    sha256: str
    region_id: RegionId | None = None

    def __post_init__(self) -> None:
        """Validate identity, payload, scope, and optional content hash."""
        if not self.view_id.strip():
            raise ValueError("visual view id must not be empty")
        if not self.payload_reference.strip():
            raise ValueError("visual view payload_reference must not be empty")
        if self.kind is VisualViewKind.FULL_FRAME and self.region_id is not None:
            raise ValueError("full-frame visual view must not reference a region_id")
        if self.kind is not VisualViewKind.FULL_FRAME and self.region_id is None:
            raise ValueError("region visual view requires region_id")
        invalid = any(character not in "0123456789abcdef" for character in self.sha256)
        if len(self.sha256) != 64 or invalid:
            raise ValueError("visual view sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, kw_only=True)
class SemanticFeatureReference:
    """Reference a selected visual feature and its exact embedding space."""

    feature_id: FeatureId
    embedding_space_id: str
    scope: FeatureScope
    region_id: RegionId | None = None

    def __post_init__(self) -> None:
        """Validate feature scope and embedding-space identity."""
        if not self.embedding_space_id.strip():
            raise ValueError("embedding_space_id must not be empty")
        if self.scope is FeatureScope.REGION and self.region_id is None:
            raise ValueError("region feature reference requires region_id")
        if self.scope is not FeatureScope.REGION and self.region_id is not None:
            raise ValueError("region_id must be None for non-region feature references")


@dataclass(frozen=True, kw_only=True)
class SemanticRequestMetadata:
    """One explicit scalar metadata item supplied to semantic inference."""

    name: str
    value: JsonScalar

    def __post_init__(self) -> None:
        """Reject unnamed metadata."""
        if not self.name.strip():
            raise ValueError("semantic request metadata name must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticInterpretationRequest:
    """Backend-neutral, reproducible selection of semantic input evidence."""

    request_id: SemanticRequestId
    source_observation_id: SourceObservationId
    perception_result_id: PerceptionResultId
    mode: SemanticInterpretationMode
    visual_views: tuple[SemanticVisualView, ...]
    prompt_template_id: str
    requested_output_schema: str
    configuration_fingerprint: str
    region_id: RegionId | None = None
    visual_features: tuple[SemanticFeatureReference, ...] = ()
    scene_context_reference: SemanticEvidenceReference | None = None
    supporting_metadata: tuple[SemanticRequestMetadata, ...] = ()

    def __post_init__(self) -> None:
        """Reject incomplete, ambiguous, or cross-observation requests."""
        if not str(self.request_id).strip():
            raise ValueError("request_id must not be empty")
        if self.mode is SemanticInterpretationMode.SCENE and self.region_id is not None:
            raise ValueError("region_id must be None for scene mode")
        if self.mode is SemanticInterpretationMode.REGION and self.region_id is None:
            raise ValueError("region_id is required for region mode")
        if not self.visual_views:
            raise ValueError("at least one visual view is required")
        for field_name, value in (
            ("prompt_template_id", self.prompt_template_id),
            ("requested_output_schema", self.requested_output_schema),
            ("configuration_fingerprint", self.configuration_fingerprint),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if any(
            view.source_observation_id != self.source_observation_id for view in self.visual_views
        ):
            raise ValueError("visual view source_observation_id must match request")
        if self.region_id is not None and any(
            view.region_id is not None and view.region_id != self.region_id
            for view in self.visual_views
        ):
            raise ValueError("visual view region_id does not match request region")
        if self.region_id is not None and any(
            feature.region_id is not None and feature.region_id != self.region_id
            for feature in self.visual_features
        ):
            raise ValueError("visual feature region_id does not match request region")
        view_ids = [view.view_id for view in self.visual_views]
        if len(set(view_ids)) != len(view_ids):
            raise ValueError("visual view ids must be unique")
        feature_ids = [feature.feature_id for feature in self.visual_features]
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("visual feature ids must be unique")
        metadata_names = [item.name for item in self.supporting_metadata]
        if len(set(metadata_names)) != len(metadata_names):
            raise ValueError("semantic request metadata names must be unique")

    def evidence_references(self) -> tuple[SemanticEvidenceReference, ...]:
        """Return the complete canonical evidence identity set for this request."""
        references = [
            SemanticEvidenceReference(evidence_type="visual_view", evidence_id=view.view_id)
            for view in self.visual_views
        ]
        references.extend(
            SemanticEvidenceReference(
                evidence_type="visual_feature", evidence_id=str(feature.feature_id)
            )
            for feature in self.visual_features
        )
        if self.scene_context_reference is not None:
            references.append(self.scene_context_reference)
        return tuple(references)


@dataclass(frozen=True, kw_only=True)
class SemanticInterpreterCapabilities:
    """Declare which canonical request evidence a semantic backend accepts."""

    supported_modes: frozenset[SemanticInterpretationMode]
    supported_view_kinds: frozenset[VisualViewKind]
    accepts_visual_features: bool
    accepts_scene_context: bool
    required_view_kinds: frozenset[VisualViewKind] = frozenset()

    def __post_init__(self) -> None:
        """Require useful modes/views and consistent required views."""
        if not self.supported_modes:
            raise ValueError("semantic interpreter must support at least one mode")
        if not self.supported_view_kinds:
            raise ValueError("semantic interpreter must support at least one visual view kind")
        if not self.required_view_kinds <= self.supported_view_kinds:
            raise ValueError("required visual view kinds must also be supported")


def validate_semantic_request(
    request: SemanticInterpretationRequest,
    capabilities: SemanticInterpreterCapabilities,
) -> None:
    """Validate a request against a backend declaration before model execution."""
    if request.mode not in capabilities.supported_modes:
        raise ValueError(f"semantic interpreter does not support {request.mode.value} mode")
    requested_view_kinds = frozenset(view.kind for view in request.visual_views)
    unsupported_views = requested_view_kinds - capabilities.supported_view_kinds
    if unsupported_views:
        values = sorted(kind.value for kind in unsupported_views)
        raise ValueError(f"semantic interpreter does not support visual view kinds: {values}")
    missing_views = capabilities.required_view_kinds - requested_view_kinds
    if missing_views:
        values = sorted(kind.value for kind in missing_views)
        raise ValueError(f"semantic request is missing required visual view kinds: {values}")
    if request.visual_features and not capabilities.accepts_visual_features:
        raise ValueError("semantic interpreter does not accept visual features")
    if request.scene_context_reference is not None and not capabilities.accepts_scene_context:
        raise ValueError("semantic interpreter does not accept scene context")


def encode_semantic_request(request: SemanticInterpretationRequest) -> dict[str, Any]:
    """Encode a semantic request into a JSON-compatible audit record."""
    return {
        "request_id": str(request.request_id),
        "source_observation_id": str(request.source_observation_id),
        "perception_result_id": str(request.perception_result_id),
        "mode": request.mode.value,
        "region_id": None if request.region_id is None else str(request.region_id),
        "visual_views": [
            {
                "view_id": view.view_id,
                "kind": view.kind.value,
                "payload_reference": view.payload_reference,
                "source_observation_id": str(view.source_observation_id),
                "region_id": None if view.region_id is None else str(view.region_id),
                "sha256": view.sha256,
            }
            for view in request.visual_views
        ],
        "visual_features": [
            {
                "feature_id": str(feature.feature_id),
                "embedding_space_id": feature.embedding_space_id,
                "scope": feature.scope.value,
                "region_id": None if feature.region_id is None else str(feature.region_id),
            }
            for feature in request.visual_features
        ],
        "scene_context_reference": (
            None
            if request.scene_context_reference is None
            else {
                "evidence_type": request.scene_context_reference.evidence_type,
                "evidence_id": request.scene_context_reference.evidence_id,
            }
        ),
        "supporting_metadata": [
            {"name": item.name, "value": item.value} for item in request.supporting_metadata
        ],
        "prompt_template_id": request.prompt_template_id,
        "requested_output_schema": request.requested_output_schema,
        "configuration_fingerprint": request.configuration_fingerprint,
    }


def decode_semantic_request(record: dict[str, Any]) -> SemanticInterpretationRequest:
    """Restore a semantic request from its canonical audit representation."""
    raw_context = record["scene_context_reference"]
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId(record["request_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        perception_result_id=PerceptionResultId(record["perception_result_id"]),
        mode=SemanticInterpretationMode(record["mode"]),
        region_id=None if record["region_id"] is None else RegionId(record["region_id"]),
        visual_views=tuple(
            SemanticVisualView(
                view_id=item["view_id"],
                kind=VisualViewKind(item["kind"]),
                payload_reference=item["payload_reference"],
                source_observation_id=SourceObservationId(item["source_observation_id"]),
                region_id=(None if item["region_id"] is None else RegionId(item["region_id"])),
                sha256=item["sha256"],
            )
            for item in record["visual_views"]
        ),
        visual_features=tuple(
            SemanticFeatureReference(
                feature_id=FeatureId(item["feature_id"]),
                embedding_space_id=item["embedding_space_id"],
                scope=FeatureScope(item["scope"]),
                region_id=(None if item["region_id"] is None else RegionId(item["region_id"])),
            )
            for item in record["visual_features"]
        ),
        scene_context_reference=(
            None
            if raw_context is None
            else SemanticEvidenceReference(
                evidence_type=raw_context["evidence_type"],
                evidence_id=raw_context["evidence_id"],
            )
        ),
        supporting_metadata=tuple(
            SemanticRequestMetadata(name=item["name"], value=item["value"])
            for item in record["supporting_metadata"]
        ),
        prompt_template_id=record["prompt_template_id"],
        requested_output_schema=record["requested_output_schema"],
        configuration_fingerprint=record["configuration_fingerprint"],
    )
