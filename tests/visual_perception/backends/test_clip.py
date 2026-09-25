from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    FeatureExtractor,
    FeatureScope,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PreparedImage,
    Region2D,
    RegionId,
    StageDefinition,
    StageStatus,
    VisualFeature,
    embedding_space_fingerprint,
    execute_stage_graph,
    feature_id_for,
)
from contextmap.visual_perception.backends.clip import (
    ClipConfig,
    ClipDependencyError,
    ClipInferenceError,
    ClipNativeOutput,
    ClipView,
    ClipVisualFeatureBackend,
    HuggingFaceClipRuntime,
)

_REVISION = "c" * 40


class FakeClipRuntime:
    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self.array = array
        self.calls: list[tuple[PreparedImage, tuple[object, ...]]] = []

    def encode(self, image: PreparedImage, views: Sequence[ClipView]) -> ClipNativeOutput:
        self.calls.append((image, tuple(views)))
        return ClipNativeOutput(
            array=self.array.copy(), elapsed_seconds=0.25, warnings=("fake warning",)
        )


class RecordingPayloadSink:
    def __init__(self) -> None:
        self.calls: list[tuple[object, SourceObservationId, np.ndarray[Any, Any]]] = []

    def add_feature_payload(
        self,
        feature: object,
        source_observation_id: SourceObservationId,
        array: np.ndarray[Any, Any],
    ) -> None:
        self.calls.append((feature, source_observation_id, array.copy()))


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference="prepared/frame-0001.png",
        width=100,
        height=80,
        transformations=(),
    )


def _region(
    region_id: str, *, x: int, y: int, width: int, height: int, accepted: bool = True
) -> Region2D:
    return Region2D(
        region_id=RegionId(region_id),
        bounding_box=BoundingBox2D(x=x, y=y, width=width, height=height),
        provenance=BackendProvenance(
            backend_id="fake-region",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="1",
        ),
        is_accepted=accepted,
        rejection_reason=None if accepted else "fixture rejection",
    )


def _backend(
    *,
    scope: FeatureScope,
    array: np.ndarray[Any, Any],
    context_padding_fraction: float = 0.0,
    feature_stage_id: str = "global_feature_extraction",
) -> tuple[ClipVisualFeatureBackend, FakeClipRuntime, RecordingPayloadSink]:
    runtime = FakeClipRuntime(array)
    sink = RecordingPayloadSink()
    backend = ClipVisualFeatureBackend(
        config=ClipConfig(
            checkpoint="openai/clip-vit-base-patch32",
            revision=_REVISION,
            scope=scope,
            device="cpu",
            precision="float32",
            input_width=224,
            input_height=224,
            local_files_only=True,
            l2_normalize=True,
            crop_policy="context_box" if context_padding_fraction else "tight_box",
            context_padding_fraction=context_padding_fraction,
        ),
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id=feature_stage_id,
        payload_sink=sink,
        runtime=runtime,
    )
    return backend, runtime, sink


def test_global_mode_satisfies_port_and_emits_one_global_feature() -> None:
    backend, runtime, sink = _backend(
        scope=FeatureScope.GLOBAL,
        array=np.array([[3.0, 4.0]], dtype=np.float32),
    )

    assert isinstance(backend, FeatureExtractor)
    assert backend.required_scope() is FeatureScope.GLOBAL
    extraction = backend.extract_visual(_image())

    assert len(extraction.features) == 1
    feature = extraction.features[0]
    assert feature.scope is FeatureScope.GLOBAL
    assert feature.region_id is None
    assert feature.normalization == "l2"
    np.testing.assert_allclose(extraction.array, np.array([[0.6, 0.8]], dtype=np.float32))
    assert extraction.views[0].crop_box == BoundingBox2D(x=0, y=0, width=100, height=80)
    assert extraction.views[0].source_region_id is None
    assert len(runtime.calls) == 1
    assert len(sink.calls) == 1


def test_region_mode_preserves_region_and_exact_crop_view_provenance() -> None:
    backend, _, sink = _backend(
        scope=FeatureScope.REGION,
        array=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        context_padding_fraction=0.25,
    )
    regions = (
        _region("region-a", x=10, y=20, width=20, height=10),
        _region("region-b", x=90, y=70, width=20, height=20),
    )

    extraction = backend.extract_visual(_image(), regions)

    assert [feature.region_id for feature in extraction.features] == [
        RegionId("region-a"),
        RegionId("region-b"),
    ]
    assert extraction.views[0].crop_box == BoundingBox2D(x=5, y=17, width=30, height=16)
    assert extraction.views[1].crop_box == BoundingBox2D(x=85, y=65, width=15, height=15)
    assert extraction.views[0].crop_policy == "context_box:0.25"
    assert extraction.views[0].coordinate_transform_id.startswith("sha256:")
    assert extraction.features[0].provenance.configuration_fingerprint != (
        extraction.features[1].provenance.configuration_fingerprint
    )
    assert len(sink.calls) == 2
    assert extraction.diagnostics.warnings == ("fake warning", "region-b: crop clipped to image")


def test_embedding_space_is_language_aligned_but_no_scoring_occurs() -> None:
    backend, _, _ = _backend(
        scope=FeatureScope.GLOBAL,
        array=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )

    extraction = backend.extract_visual(_image())
    space = extraction.embedding_space
    assert space is not None

    assert space.family == "clip"
    assert space.checkpoint == f"openai/clip-vit-base-patch32@{_REVISION}"
    assert space.layer == "image_projection"
    assert space.dimension == 3
    assert extraction.features[0].embedding_space_id == embedding_space_fingerprint(space)
    assert not hasattr(backend, "score")


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.array([[np.nan, 1.0]], dtype=np.float32), "finite"),
        (np.zeros((1, 2), dtype=np.float32), "zero-norm"),
    ],
)
def test_invalid_numerical_payload_is_rejected_before_persistence(
    array: np.ndarray[Any, Any], message: str
) -> None:
    backend, _, sink = _backend(scope=FeatureScope.GLOBAL, array=array)

    with pytest.raises(ClipInferenceError, match=message):
        backend.extract_visual(_image())

    assert sink.calls == []


def test_extract_port_returns_persisted_metadata_for_configured_scope() -> None:
    backend, _, sink = _backend(
        scope=FeatureScope.REGION,
        array=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    region = _region("region-a", x=1, y=2, width=3, height=4)

    features = backend.extract(_image(), (region,))

    assert tuple(features) == (sink.calls[0][0],)
    assert features[0].region_id == RegionId("region-a")


def test_global_and_region_modes_share_model_space_but_preserve_distinct_view_provenance() -> None:
    global_backend, _, _ = _backend(
        scope=FeatureScope.GLOBAL, array=np.ones((1, 2), dtype=np.float32)
    )
    region_backend, _, _ = _backend(
        scope=FeatureScope.REGION, array=np.ones((1, 2), dtype=np.float32)
    )

    global_result = global_backend.extract_visual(_image())
    region_result = region_backend.extract_visual(
        _image(), (_region("region-a", x=0, y=0, width=10, height=10),)
    )

    assert global_result.embedding_space == region_result.embedding_space
    assert (
        global_result.features[0].embedding_space_id == region_result.features[0].embedding_space_id
    )
    assert (
        global_result.features[0].provenance.configuration_fingerprint
        != region_result.features[0].provenance.configuration_fingerprint
    )


def test_feature_identity_is_unique_across_composed_feature_stages() -> None:
    region_backend, _, _ = _backend(
        scope=FeatureScope.REGION,
        array=np.ones((1, 2), dtype=np.float32),
        feature_stage_id="region_feature_extraction",
    )

    region = _region("region-a", x=0, y=0, width=10, height=10)
    region_feature = region_backend.extract_visual(_image(), (region,)).features[0]
    result_id = PerceptionResultId("run-0001--frame-0001")
    dense_feature_id = feature_id_for(
        result_id=result_id, producer_id="dense_feature_extraction", index=0
    )
    dense_feature = VisualFeature(
        feature_id=dense_feature_id,
        scope=FeatureScope.DENSE,
        embedding_space_id="dinov2-space",
        shape=(2, 2, 2),
        dtype="float32",
        normalization="none",
        payload_reference=f"features/{dense_feature_id}.npy",
        provenance=BackendProvenance(
            backend_id="fake_dinov2",
            capability="feature_extractor",
            provider="fake",
            model="fake-dinov2",
            version="1",
        ),
    )

    result = PerceptionResult(
        result_id=result_id,
        source_observation_id=_image().source_observation_id,
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="sequence-0001",
        created_at="2026-09-19T00:00:00+00:00",
        regions=(region,),
        features=(dense_feature, region_feature),
    )

    assert len({feature.feature_id for feature in result.features}) == 2
    assert len({feature.payload_reference for feature in result.features}) == 2


def test_region_mode_requires_accepted_regions_and_matching_output_count() -> None:
    backend, _, _ = _backend(scope=FeatureScope.REGION, array=np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ClipInferenceError, match="accepted regions"):
        backend.extract_visual(
            _image(), (_region("rejected", x=0, y=0, width=2, height=2, accepted=False),)
        )

    wrong_count, _, _ = _backend(scope=FeatureScope.REGION, array=np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ClipInferenceError, match="returned 2 embeddings for 1 views"):
        wrong_count.extract_visual(_image(), (_region("region-a", x=0, y=0, width=2, height=2),))


def test_region_mode_with_zero_regions_succeeds_with_zero_features() -> None:
    """A legitimate empty discovery result must not fail region-scoped CLIP (#380)."""
    backend, runtime, sink = _backend(
        scope=FeatureScope.REGION, array=np.ones((1, 2), dtype=np.float32)
    )

    features = backend.extract(_image(), regions=())

    assert features == ()
    assert runtime.calls == []
    assert sink.calls == []

    extraction = backend.extract_visual(_image(), regions=())
    assert extraction.features == ()
    assert extraction.embedding_space is None
    assert extraction.views == ()
    assert extraction.diagnostics.view_count == 0


def test_stage_graph_succeeds_when_region_discovery_finds_nothing() -> None:
    """A zero-region upstream stage must not turn the feature stage FAILED (#380)."""
    backend, _, _ = _backend(scope=FeatureScope.REGION, array=np.ones((1, 2), dtype=np.float32))
    stages = (
        StageDefinition(
            stage_id="region_discovery",
            capability="region_discovery",
            run=lambda _context: (),
        ),
        StageDefinition(
            stage_id="region_feature_extraction",
            capability="feature_extractor",
            run=lambda context: backend.extract(
                _image(),
                regions=context["region_discovery"],  # type: ignore[arg-type]
            ),
            depends_on=frozenset({"region_discovery"}),
        ),
    )

    outcomes = {outcome.stage_id: outcome for outcome in execute_stage_graph(stages)}

    assert outcomes["region_discovery"].status is StageStatus.SUCCEEDED
    assert outcomes["region_feature_extraction"].status is StageStatus.SUCCEEDED
    assert outcomes["region_feature_extraction"].error is None
    assert outcomes["region_feature_extraction"].output == ()


@pytest.mark.parametrize("elapsed_seconds", [float("nan"), float("inf")])
def test_native_diagnostics_require_finite_elapsed_time(elapsed_seconds: float) -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        ClipNativeOutput(
            array=np.ones((1, 2), dtype=np.float32),
            elapsed_seconds=elapsed_seconds,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", ""),
        ("revision", ""),
        ("revision", "main"),
        ("scope", FeatureScope.DENSE),
        ("device", "tpu"),
        ("precision", "int8"),
        ("crop_policy", "mask"),
        ("context_padding_fraction", -0.1),
        ("context_padding_fraction", float("nan")),
        ("context_padding_fraction", float("inf")),
        ("payload_prefix", ""),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "checkpoint": "openai/clip-vit-base-patch32",
        "revision": _REVISION,
        "scope": FeatureScope.GLOBAL,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        ClipConfig(**values)  # type: ignore[arg-type]


def test_missing_sdk_dependencies_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "contextmap.visual_perception.backends.clip.importlib.import_module",
        missing_dependency,
    )
    runtime = HuggingFaceClipRuntime(
        config=ClipConfig(
            checkpoint="openai/clip-vit-base-patch32",
            revision=_REVISION,
            scope=FeatureScope.GLOBAL,
        ),
        prepared_image_root=tmp_path,
    )

    with pytest.raises(ClipDependencyError, match="torch, transformers, and Pillow"):
        runtime.encode(_image(), ())
