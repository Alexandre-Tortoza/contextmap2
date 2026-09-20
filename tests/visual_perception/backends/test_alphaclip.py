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
    EmbeddingSpace,
    FeatureExtractor,
    FeatureScope,
    FeatureStoreReader,
    FeatureStoreWriter,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PreparedImage,
    Region2D,
    RegionId,
    VisualFeature,
    embedding_space_fingerprint,
    feature_id_for,
    write_feature_index,
)
from contextmap.visual_perception.backends.alphaclip import (
    AlphaClipConfig,
    AlphaClipDependencyError,
    AlphaClipInferenceError,
    AlphaClipNativeOutput,
    AlphaClipRegionFeatureBackend,
    AlphaClipRequest,
    OfficialAlphaClipRuntime,
    _normalized_rgb_array,
    _validate_batch_geometry,
)


class DictMaskSource:
    def __init__(self, masks: dict[RegionId, np.ndarray[Any, Any]]) -> None:
        self.masks = masks

    def load_box_local_mask(self, region: Region2D) -> np.ndarray[Any, Any]:
        return self.masks[region.region_id].copy()


class FakeAlphaClipRuntime:
    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self.array = array
        self.calls: list[tuple[PreparedImage, tuple[AlphaClipRequest, ...]]] = []

    def encode(
        self, image: PreparedImage, requests: Sequence[AlphaClipRequest]
    ) -> AlphaClipNativeOutput:
        self.calls.append((image, tuple(requests)))
        return AlphaClipNativeOutput(
            array=self.array.copy(),
            elapsed_seconds=0.5,
            peak_memory_bytes=1024,
            warnings=("fake runtime warning",),
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
        width=6,
        height=4,
        transformations=(),
    )


def _region(
    region_id: str = "region-a", *, x: int = 1, y: int = 1, width: int = 2, height: int = 2
) -> Region2D:
    return Region2D(
        region_id=RegionId(region_id),
        bounding_box=BoundingBox2D(x=x, y=y, width=width, height=height),
        mask_reference=f"masks/{region_id}.npy",
        provenance=BackendProvenance(
            backend_id="fake-region",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="1",
        ),
    )


def _backend(
    *,
    masks: dict[RegionId, np.ndarray[Any, Any]],
    array: np.ndarray[Any, Any],
    view_policy: str = "full_image",
    feature_stage_id: str = "region_feature_extraction",
) -> tuple[AlphaClipRegionFeatureBackend, FakeAlphaClipRuntime, RecordingPayloadSink]:
    runtime = FakeAlphaClipRuntime(array)
    sink = RecordingPayloadSink()
    backend = AlphaClipRegionFeatureBackend(
        config=AlphaClipConfig(
            model_name="ViT-B/16",
            base_checkpoint_path="checkpoints/clip-vit-b16.pt",
            alpha_checkpoint_path="checkpoints/alphaclip-vit-b16.pth",
            checkpoint_fingerprint="sha256:alphaclip123",
            device="cpu",
            precision="float32",
            input_width=4,
            input_height=4,
            view_policy=view_policy,
            context_padding_fraction=0.5 if view_policy == "context_box" else 0.0,
            mask_interpolation="nearest",
        ),
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id=feature_stage_id,
        mask_source=DictMaskSource(masks),
        payload_sink=sink,
        runtime=runtime,
    )
    return backend, runtime, sink


def test_backend_satisfies_region_feature_port_and_places_box_local_mask() -> None:
    region = _region()
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    backend, runtime, sink = _backend(
        masks={region.region_id: mask}, array=np.array([[3.0, 4.0]], dtype=np.float32)
    )

    assert isinstance(backend, FeatureExtractor)
    assert backend.required_scope() is FeatureScope.REGION
    extraction = backend.extract_masked(_image(), (region,))

    assert extraction.features[0].region_id == region.region_id
    assert extraction.features[0].scope is FeatureScope.REGION
    assert extraction.views[0].source_mask_reference == "masks/region-a.npy"
    assert extraction.views[0].mask_content_hash.startswith("sha256:")
    assert extraction.views[0].mask_resize_interpolation == "nearest"
    request = runtime.calls[0][1][0]
    expected_full_mask = np.zeros((4, 6), dtype=np.bool_)
    expected_full_mask[1:3, 1:3] = mask
    np.testing.assert_array_equal(request.mask, expected_full_mask)
    np.testing.assert_allclose(extraction.array, np.array([[0.6, 0.8]], dtype=np.float32))
    assert sink.calls[0][0] == extraction.features[0]


def test_context_view_crops_image_and_mask_without_mutating_region() -> None:
    region = _region(x=0, y=0, width=2, height=2)
    original_box = region.bounding_box
    backend, runtime, _ = _backend(
        masks={region.region_id: np.ones((2, 2), dtype=np.bool_)},
        array=np.array([[1.0, 0.0]], dtype=np.float32),
        view_policy="context_box",
    )

    extraction = backend.extract_masked(_image(), (region,))

    assert region.bounding_box == original_box
    assert extraction.views[0].crop_box == BoundingBox2D(x=0, y=0, width=3, height=3)
    assert extraction.views[0].view_policy == "context_box:0.5"
    assert runtime.calls[0][1][0].mask.shape == (3, 3)
    assert extraction.diagnostics.warnings == (
        "fake runtime warning",
        "region-a: context view clipped to image",
    )


def test_alphaclip_has_distinct_embedding_space_from_ordinary_clip() -> None:
    region = _region()
    backend, _, _ = _backend(
        masks={region.region_id: np.ones((2, 2), dtype=np.bool_)},
        array=np.array([[1.0, 2.0]], dtype=np.float32),
    )

    extraction = backend.extract_masked(_image(), (region,))
    alpha_space = extraction.embedding_space
    clip_space = EmbeddingSpace(
        family="clip",
        model="ViT-B/16",
        version="1",
        checkpoint="sha256:alphaclip123",
        layer="image_projection",
        dimension=2,
        normalization="l2",
    )

    assert alpha_space.family == "alphaclip"
    assert alpha_space.layer == "alpha_conditioned_image_projection"
    assert embedding_space_fingerprint(alpha_space) != embedding_space_fingerprint(clip_space)
    assert not hasattr(backend, "score")


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.array([[1.0, np.inf]], dtype=np.float32), "finite"),
        (np.zeros((1, 2), dtype=np.float32), "zero-norm"),
    ],
)
def test_invalid_numerical_payload_is_rejected_before_persistence(
    array: np.ndarray[Any, Any], message: str
) -> None:
    region = _region()
    backend, _, sink = _backend(
        masks={region.region_id: np.ones((2, 2), dtype=np.bool_)},
        array=array,
    )

    with pytest.raises(AlphaClipInferenceError, match=message):
        backend.extract_masked(_image(), (region,))

    assert sink.calls == []


def test_extract_port_returns_same_persisted_features_and_diagnostics() -> None:
    region = _region()
    backend, _, sink = _backend(
        masks={region.region_id: np.ones((2, 2), dtype=np.bool_)},
        array=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    features = backend.extract(_image(), (region,))

    assert tuple(features) == (sink.calls[0][0],)
    extraction = backend.extract_masked(_image(), (region,))
    assert extraction.diagnostics.elapsed_seconds == 0.5
    assert extraction.diagnostics.peak_memory_bytes == 1024


def test_feature_identity_is_unique_across_composed_feature_stages() -> None:
    region = _region()
    masks = {region.region_id: np.ones((2, 2), dtype=np.bool_)}
    alpha_backend, _, _ = _backend(
        masks=masks,
        array=np.ones((1, 2), dtype=np.float32),
        feature_stage_id="alphaclip_region_features",
    )
    alpha_feature = alpha_backend.extract_masked(_image(), (region,)).features[0]
    result_id = PerceptionResultId("run-0001--frame-0001")
    dense_feature_id = feature_id_for(result_id=result_id, index=0)
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
        features=(dense_feature, alpha_feature),
    )

    assert len({feature.feature_id for feature in result.features}) == 2
    assert len({feature.payload_reference for feature in result.features}) == 2


def test_output_persists_and_reopens_through_canonical_feature_store(tmp_path: Path) -> None:
    region = _region()
    writer = FeatureStoreWriter(tmp_path)

    class StoreSink:
        def add_feature_payload(
            self,
            feature: object,
            source_observation_id: SourceObservationId,
            array: np.ndarray[Any, Any],
        ) -> None:
            from contextmap.visual_perception import VisualFeature

            assert isinstance(feature, VisualFeature)
            writer.write(feature, source_observation_id, array)

    runtime = FakeAlphaClipRuntime(np.array([[3.0, 4.0]], dtype=np.float32))
    backend = AlphaClipRegionFeatureBackend(
        config=AlphaClipConfig(
            model_name="ViT-B/16",
            base_checkpoint_path="base.pt",
            alpha_checkpoint_path="alpha.pth",
            checkpoint_fingerprint="sha256:abc",
        ),
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id="region_feature_extraction",
        mask_source=DictMaskSource({region.region_id: np.ones((2, 2), dtype=np.bool_)}),
        payload_sink=StoreSink(),
        runtime=runtime,
    )

    feature = backend.extract(_image(), (region,))[0]
    write_feature_index(tmp_path, writer.entries())
    reopened = FeatureStoreReader.open(tmp_path)

    np.testing.assert_allclose(
        reopened.load(_image().source_observation_id, feature.feature_id),
        np.array([0.6, 0.8]),
    )
    assert feature.region_id == region.region_id


def test_mask_shape_dtype_empty_support_and_missing_reference_fail_explicitly() -> None:
    region = _region()
    cases = (
        (np.ones((4, 6), dtype=np.bool_), "mask shape"),
        (np.ones((2, 2), dtype=np.uint8), "boolean"),
        (np.zeros((2, 2), dtype=np.bool_), "empty"),
    )
    for mask, message in cases:
        backend, _, _ = _backend(
            masks={region.region_id: mask}, array=np.ones((1, 2), dtype=np.float32)
        )
        with pytest.raises(AlphaClipInferenceError, match=message):
            backend.extract_masked(_image(), (region,))

    no_reference = Region2D(
        region_id=RegionId("no-mask"),
        bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
        provenance=region.provenance,
    )
    backend, _, _ = _backend(
        masks={no_reference.region_id: np.ones((2, 2), dtype=np.bool_)},
        array=np.ones((1, 2), dtype=np.float32),
    )
    with pytest.raises(AlphaClipInferenceError, match="mask_reference"):
        backend.extract_masked(_image(), (no_reference,))


def test_region_count_must_match_runtime_output() -> None:
    region = _region()
    backend, _, _ = _backend(
        masks={region.region_id: np.ones((2, 2), dtype=np.bool_)},
        array=np.ones((2, 2), dtype=np.float32),
    )
    with pytest.raises(AlphaClipInferenceError, match="2 embeddings for 1 requests"):
        backend.extract_masked(_image(), (region,))


@pytest.mark.parametrize("elapsed_seconds", [float("nan"), float("inf")])
def test_native_diagnostics_require_finite_elapsed_time(elapsed_seconds: float) -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        AlphaClipNativeOutput(
            array=np.ones((1, 2), dtype=np.float32),
            elapsed_seconds=elapsed_seconds,
            peak_memory_bytes=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("checkpoint_fingerprint", ""),
        ("device", "tpu"),
        ("precision", "int8"),
        ("view_policy", "mask_only"),
        ("mask_interpolation", "bilinear"),
        ("context_padding_fraction", float("nan")),
        ("context_padding_fraction", float("inf")),
        ("payload_prefix", ""),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "model_name": "ViT-B/16",
        "base_checkpoint_path": "base.pt",
        "alpha_checkpoint_path": "alpha.pth",
        "checkpoint_fingerprint": "sha256:abc",
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        AlphaClipConfig(**values)  # type: ignore[arg-type]


def test_missing_sdk_dependencies_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "contextmap.visual_perception.backends.alphaclip.importlib.import_module",
        missing_dependency,
    )
    runtime = OfficialAlphaClipRuntime(
        config=AlphaClipConfig(
            model_name="ViT-B/16",
            base_checkpoint_path="base.pt",
            alpha_checkpoint_path="alpha.pth",
            checkpoint_fingerprint="sha256:abc",
        ),
        prepared_image_root=tmp_path,
        checkpoint_root=tmp_path,
    )

    with pytest.raises(AlphaClipDependencyError, match="torch, alpha_clip, and Pillow"):
        runtime.encode(_image(), ())


def test_rgb_normalization_preserves_the_declared_model_input_geometry() -> None:
    resized_rgb = np.zeros((4, 6, 3), dtype=np.uint8)

    normalized = _normalized_rgb_array(
        resized_rgb,
        expected_width=6,
        expected_height=4,
    )

    assert normalized.shape == (3, 4, 6)


def test_runtime_rejects_misaligned_rgb_and_alpha_batches_before_inference() -> None:
    image_batch = np.zeros((2, 3, 4, 6), dtype=np.float32)
    alpha_batch = np.zeros((2, 1, 4, 5), dtype=np.float32)

    with pytest.raises(AlphaClipInferenceError, match="RGB and alpha batch geometry"):
        _validate_batch_geometry(image_batch, alpha_batch)
