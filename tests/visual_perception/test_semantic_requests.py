"""Tests for auditable Semantic Interpretation requests."""

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    RegionId,
    SemanticEvidenceReference,
    SemanticFeatureReference,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    SemanticRequestId,
    SemanticRequestMetadata,
    SemanticViewConstruction,
    SemanticVisualView,
    VisualViewKind,
    decode_semantic_request,
    encode_semantic_request,
    validate_semantic_request,
)

SOURCE_ID = SourceObservationId("frame-0124")
RESULT_ID = PerceptionResultId("run-0001--frame-0124")
REGION_ID = RegionId("region-0007")


def _view(kind: VisualViewKind = VisualViewKind.TIGHT_CROP) -> SemanticVisualView:
    return SemanticVisualView(
        view_id="view-0001",
        kind=kind,
        payload_reference="outputs/semantic-views/region-0007-tight.jpg",
        source_observation_id=SOURCE_ID,
        region_id=None if kind is VisualViewKind.FULL_FRAME else REGION_ID,
        sha256="0" * 64,
    )


def _request(**overrides: object) -> SemanticInterpretationRequest:
    values: dict[str, object] = {
        "request_id": SemanticRequestId("request-0001"),
        "source_observation_id": SOURCE_ID,
        "perception_result_id": RESULT_ID,
        "mode": SemanticInterpretationMode.REGION,
        "region_id": REGION_ID,
        "visual_views": (_view(),),
        "prompt_template_id": "region/v1",
        "requested_output_schema": "semantic-response/1",
        "configuration_fingerprint": "sha256:config",
    }
    values.update(overrides)
    return SemanticInterpretationRequest(**values)  # type: ignore[arg-type]


def test_region_request_can_use_pixels_without_visual_features() -> None:
    request = _request()

    assert request.visual_features == ()
    assert request.region_id == REGION_ID
    assert request.evidence_references() == (
        SemanticEvidenceReference(evidence_type="visual_view", evidence_id="view-0001"),
    )


def test_request_makes_scene_context_features_and_metadata_explicit() -> None:
    request = _request(
        visual_features=(
            SemanticFeatureReference(
                feature_id=FeatureId("feature-0007"),
                embedding_space_id="sha256:clip-space",
                scope=FeatureScope.REGION,
                region_id=REGION_ID,
            ),
        ),
        scene_context_reference=SemanticEvidenceReference(
            evidence_type="scene_context", evidence_id=str(RESULT_ID)
        ),
        supporting_metadata=(SemanticRequestMetadata(name="camera", value="front"),),
    )

    assert request.visual_features[0].embedding_space_id == "sha256:clip-space"
    assert request.scene_context_reference is not None
    assert {reference.evidence_type for reference in request.evidence_references()} == {
        "visual_view",
        "visual_feature",
        "scene_context",
    }
    assert decode_semantic_request(encode_semantic_request(request)) == request


def test_request_rejects_invalid_scene_region_scope_before_execution() -> None:
    with pytest.raises(ValueError, match="region_id must be None"):
        _request(mode=SemanticInterpretationMode.SCENE)

    with pytest.raises(ValueError, match="region_id is required"):
        _request(region_id=None)

    with pytest.raises(ValueError, match="does not match request region"):
        _request(
            visual_views=(
                SemanticVisualView(
                    view_id="wrong-region",
                    kind=VisualViewKind.MASKED_SUBJECT,
                    payload_reference="outputs/semantic-views/wrong.png",
                    source_observation_id=SOURCE_ID,
                    region_id=RegionId("region-9999"),
                    sha256="1" * 64,
                ),
            )
        )


def test_visual_view_rejects_non_artifact_path_before_backend_execution() -> None:
    with pytest.raises(ValueError, match="outputs/semantic-views"):
        SemanticVisualView(
            view_id="unsafe-view",
            kind=VisualViewKind.TIGHT_CROP,
            payload_reference="../outside.jpg",
            source_observation_id=SOURCE_ID,
            region_id=REGION_ID,
            sha256="0" * 64,
        )


def test_capability_validation_rejects_evidence_a_backend_cannot_consume() -> None:
    capabilities = SemanticInterpreterCapabilities(
        supported_modes=frozenset({SemanticInterpretationMode.REGION}),
        supported_view_kinds=frozenset({VisualViewKind.TIGHT_CROP}),
        accepts_visual_features=False,
        accepts_scene_context=False,
    )
    feature_assisted = _request(
        visual_features=(
            SemanticFeatureReference(
                feature_id=FeatureId("feature-0007"),
                embedding_space_id="sha256:clip-space",
                scope=FeatureScope.REGION,
                region_id=REGION_ID,
            ),
        )
    )

    with pytest.raises(ValueError, match="visual features"):
        validate_semantic_request(feature_assisted, capabilities)

    validate_semantic_request(_request(), capabilities)


def test_a_view_construction_record_round_trips_with_the_request() -> None:
    """#524: how each view was cut from its source image is persisted with the request."""
    view = SemanticVisualView(
        view_id="view-0001",
        kind=VisualViewKind.CONTEXTUAL_CROP,
        payload_reference="outputs/semantic-views/region-0007-context.png",
        source_observation_id=SOURCE_ID,
        region_id=REGION_ID,
        sha256="0" * 64,
        construction=SemanticViewConstruction(
            policy_fingerprint="sha256:policy",
            source_image_sha256="1" * 64,
            pixel_bounds=(0, 2, 40, 30),
        ),
    )
    request = _request(visual_views=(view,))

    encoded = encode_semantic_request(request)

    assert encoded["visual_views"][0]["construction"] == {
        "policy_fingerprint": "sha256:policy",
        "source_image_sha256": "1" * 64,
        "pixel_bounds": [0, 2, 40, 30],
    }
    assert decode_semantic_request(encoded) == request
    assert encode_semantic_request(_request())["visual_views"][0]["construction"] is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"policy_fingerprint": " "}, "policy_fingerprint"),
        ({"source_image_sha256": "xyz"}, "source_image_sha256"),
        ({"pixel_bounds": (5, 0, 5, 3)}, "pixel_bounds"),
        ({"pixel_bounds": (-1, 0, 5, 3)}, "pixel_bounds"),
    ],
)
def test_a_view_construction_record_refuses_an_unusable_lineage(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "policy_fingerprint": "sha256:policy",
        "source_image_sha256": "1" * 64,
        "pixel_bounds": (0, 0, 4, 3),
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        SemanticViewConstruction(**values)  # type: ignore[arg-type]


def test_capabilities_can_bound_the_number_of_views_a_request_carries() -> None:
    capabilities = SemanticInterpreterCapabilities(
        supported_modes=frozenset({SemanticInterpretationMode.REGION}),
        supported_view_kinds=frozenset({VisualViewKind.TIGHT_CROP, VisualViewKind.MASKED_SUBJECT}),
        accepts_visual_features=False,
        accepts_scene_context=False,
        max_visual_views=1,
    )
    masked = SemanticVisualView(
        view_id="view-0002",
        kind=VisualViewKind.MASKED_SUBJECT,
        payload_reference="outputs/semantic-views/region-0007-masked.png",
        source_observation_id=SOURCE_ID,
        region_id=REGION_ID,
        sha256="0" * 64,
    )

    validate_semantic_request(_request(), capabilities)
    with pytest.raises(ValueError, match="at most 1 visual view"):
        validate_semantic_request(_request(visual_views=(_view(), masked)), capabilities)
    with pytest.raises(ValueError, match="max_visual_views"):
        SemanticInterpreterCapabilities(
            supported_modes=frozenset({SemanticInterpretationMode.REGION}),
            supported_view_kinds=frozenset({VisualViewKind.TIGHT_CROP}),
            accepts_visual_features=False,
            accepts_scene_context=False,
            max_visual_views=0,
        )
