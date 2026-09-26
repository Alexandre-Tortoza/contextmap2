"""Deterministic Visual Perception backends for the runtime executor tests.

The executor's orchestration is what these tests prove (#507), not the backends' science:
region discovery, feature extraction and semantic interpretation are fakes, while the
sequence, the prepared images, the resolved stage graph and the written run are real.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from contextmap.runtime.executors import VisualPerceptionExecutor
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    FeatureId,
    FeatureScope,
    PreparedImage,
    Region2D,
    RegionId,
    VisualFeature,
)


class _FakeRegionDiscovery:
    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_region_discovery",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def discover(self, image: PreparedImage) -> list[Region2D]:
        return [
            Region2D(
                region_id=RegionId("region-0000"),
                bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
                provenance=self.backend_provenance(),
            )
        ]


class _FakeFeatureExtractor:
    def __init__(self, scope: FeatureScope) -> None:
        self._scope = scope

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id=f"fake_{self._scope.value}_feature_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def required_scope(self) -> FeatureScope:
        return self._scope

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        if self._scope is FeatureScope.DENSE:
            return [
                VisualFeature(
                    feature_id=FeatureId("feature-dense-0000"),
                    scope=FeatureScope.DENSE,
                    embedding_space_id="fake-dense-space",
                    shape=(2,),
                    dtype="float32",
                    payload_reference=f"{image.source_observation_id}-dense.npy",
                    provenance=self.backend_provenance(),
                )
            ]
        return [
            VisualFeature(
                feature_id=FeatureId(f"feature-{region.region_id}"),
                scope=FeatureScope.REGION,
                embedding_space_id="fake-region-space",
                shape=(2,),
                dtype="float32",
                payload_reference=f"{image.source_observation_id}-{region.region_id}.npy",
                provenance=self.backend_provenance(),
                region_id=region.region_id,
            )
            for region in regions
        ]


class _FakeSemanticInterpreter:
    """Implements the real ``interpret()`` port, like every real backend does today.

    The executor bridges this to its own legacy ``interpret_scene``/``interpret_regions``
    dispatch (see ``_LegacySemanticInterpreterBridge``); a fake standing in for a real
    backend must match what a real backend implements.
    """

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_semantic_interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def interpret(self, request: Any) -> object:
        import json

        from contextmap.visual_perception import (
            SemanticBackendDiagnostics,
            SemanticConfidencePolicy,
            SemanticInferenceProvenance,
            SemanticInterpretationExecution,
            SemanticPromptTemplate,
            parse_semantic_response,
            render_semantic_prompt,
        )

        template = SemanticPromptTemplate.default_for(request.mode)  # type: ignore[attr-defined]
        rendered = render_semantic_prompt(
            request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
        )
        raw_response = json.dumps({"abstained": True, "claims": [], "scene_context": None})
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"fake-{request.mode.value}",  # type: ignore[attr-defined]
            prompt_template_id=request.prompt_template_id,  # type: ignore[attr-defined]
            output_schema_version=request.requested_output_schema,  # type: ignore[attr-defined]
        )
        return SemanticInterpretationExecution(
            request=request,  # type: ignore[arg-type]
            rendered_prompt=rendered,
            raw_response=raw_response,
            parsed=parse_semantic_response(
                raw_response,
                request,  # type: ignore[arg-type]
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            ),
            diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
            effective_configuration={"backend": "fake"},
        )


def stub_perception_executor() -> VisualPerceptionExecutor:
    """A real executor over deterministic fake backends: one region and two features per image."""
    return VisualPerceptionExecutor(
        region_discovery=_FakeRegionDiscovery(),  # type: ignore[arg-type]
        dense_features=lambda _scope: _FakeFeatureExtractor(FeatureScope.DENSE),  # type: ignore[arg-type]
        region_features=lambda _scope: _FakeFeatureExtractor(FeatureScope.REGION),  # type: ignore[arg-type]
        semantic_interpreter=_FakeSemanticInterpreter(),  # type: ignore[arg-type]
    )
