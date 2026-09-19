"""Deterministic fake Visual Perception backends for core contract tests.

These satisfy the capability ports (#49) with fixed, predictable
evidence — no GPU, model download, or network dependency — so the
orchestration/persistence/evidence-view layers can be validated
end-to-end before any real backend exists. They belong to test/support
infrastructure only and must never be used as a silent production
fallback.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    FeatureScope,
    HypothesisRole,
    PerceptionResultId,
    PreparedImage,
    Region2D,
    RegionId,
    SceneContext,
    SemanticBackendDiagnostics,
    SemanticClaim,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    SemanticPromptTemplate,
    VisualFeature,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
)
from contextmap.visual_perception.models import ClaimId, FeatureId


class FakeRegionDiscovery:
    """Deterministic RegionDiscovery: one region per configured box."""

    def __init__(self, boxes: Sequence[BoundingBox2D] = ()) -> None:
        self._boxes = tuple(boxes) or (BoundingBox2D(x=0, y=0, width=10, height=10),)

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_region_discovery",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def discover(self, image: PreparedImage) -> Sequence[Region2D]:
        provenance = self.backend_provenance()
        return tuple(
            Region2D(
                region_id=RegionId(f"region-{index:04d}"), bounding_box=box, provenance=provenance
            )
            for index, box in enumerate(self._boxes)
        )


class FakeDenseFeatureExtractor:
    """Deterministic FeatureExtractor producing one dense feature, no regions needed."""

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_dense_feature_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def required_scope(self) -> FeatureScope:
        return FeatureScope.DENSE

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        return (
            VisualFeature(
                feature_id=FeatureId("feature-dense-0000"),
                scope=FeatureScope.DENSE,
                embedding_space_id="fake-dense-space",
                shape=(4, 4, 8),
                dtype="float32",
                payload_reference=f"features/{image.source_observation_id}-dense.bin",
                provenance=self.backend_provenance(),
            ),
        )


class FakeRegionFeatureExtractor:
    """Deterministic FeatureExtractor producing one feature per given region."""

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_region_feature_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def required_scope(self) -> FeatureScope:
        return FeatureScope.REGION

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        provenance = self.backend_provenance()
        return tuple(
            VisualFeature(
                feature_id=FeatureId(f"feature-{region.region_id}"),
                scope=FeatureScope.REGION,
                embedding_space_id="fake-region-space",
                shape=(768,),
                dtype="float32",
                payload_reference=f"features/{region.region_id}.bin",
                provenance=provenance,
                region_id=region.region_id,
            )
            for region in regions
        )


class FakeSemanticInterpreter:
    """Deterministic SemanticInterpreter producing scene- and region-level claims."""

    def __init__(self, result_id: PerceptionResultId | None = None) -> None:
        self._result_id = result_id or PerceptionResultId("run-0001--frame-0124")

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_semantic_interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset(SemanticInterpretationMode),
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=False,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        template = SemanticPromptTemplate.default_for(request.mode)
        rendered = render_semantic_prompt(request, template)
        raw_response = json.dumps(
            {
                "abstained": False,
                "claims": [
                    {
                        "hypothesis": (
                            "an indoor corridor"
                            if request.mode is SemanticInterpretationMode.SCENE
                            else "a fake object"
                        ),
                        "role": "primary",
                        "category": None,
                        "region_kind": None,
                        "attributes": {},
                        "confidence": None,
                    }
                ],
                "scene_context": (
                    {"scene_type": "corridor"}
                    if request.mode is SemanticInterpretationMode.SCENE
                    else None
                ),
            }
        )
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"fake-{request.mode.value}",
            prompt_template_id=request.prompt_template_id,
            output_schema_version=request.requested_output_schema,
        )
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=raw_response,
            parsed=parse_semantic_response(raw_response, request, provenance),
            diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
            effective_configuration={"backend": "fake"},
        )

    def interpret_scene(self, image: PreparedImage) -> SceneContext | None:
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity="scene-description",
            prompt_template_id="scene/v1",
            output_schema_version="semantic-response/1",
        )
        return SceneContext(
            source_observation_id=image.source_observation_id,
            perception_result_id=self._result_id,
            claims=(
                SemanticClaim(
                    claim_id=ClaimId("claim-scene-0000"),
                    source_observation_id=image.source_observation_id,
                    perception_result_id=self._result_id,
                    hypothesis="an indoor corridor",
                    role=HypothesisRole.PRIMARY,
                    provenance=provenance,
                    evidence_references=(
                        SemanticEvidenceReference(
                            evidence_type="prepared_image",
                            evidence_id=image.payload_reference,
                        ),
                    ),
                ),
            ),
            provenance=provenance,
        )

    def interpret_regions(
        self, image: PreparedImage, regions: Sequence[Region2D]
    ) -> Sequence[SemanticClaim]:
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity="region-labeling",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        )
        return tuple(
            SemanticClaim(
                claim_id=ClaimId(f"claim-{region.region_id}"),
                source_observation_id=image.source_observation_id,
                perception_result_id=self._result_id,
                hypothesis="a fake object",
                role=HypothesisRole.PRIMARY,
                provenance=provenance,
                region_id=region.region_id,
                evidence_references=(
                    SemanticEvidenceReference(
                        evidence_type="region", evidence_id=str(region.region_id)
                    ),
                ),
            )
            for region in regions
        )


class FakeFailingRegionDiscovery:
    """A RegionDiscovery that always fails, for negative-path tests."""

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_failing_region_discovery",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def discover(self, image: PreparedImage) -> Sequence[Region2D]:
        raise RuntimeError("region discovery backend unavailable")
