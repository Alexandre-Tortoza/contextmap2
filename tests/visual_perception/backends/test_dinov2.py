from __future__ import annotations

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
    VisualFeature,
    embedding_space_fingerprint,
    feature_id_for,
    pool_region_feature,
)
from contextmap.visual_perception.backends.dinov2 import (
    DinoV2Config,
    DinoV2DenseFeatureBackend,
    DinoV2DependencyError,
    DinoV2InferenceError,
    DinoV2NativeOutput,
    HuggingFaceDinoV2Runtime,
    _patch_tokens_and_register_count,
)


class FakeDinoV2Runtime:
    """Runtime fake returning fixed native patch features."""

    def __init__(self, output: DinoV2NativeOutput) -> None:
        self.output = output
        self.images: list[PreparedImage] = []

    def infer(self, image: PreparedImage) -> DinoV2NativeOutput:
        self.images.append(image)
        return self.output


class RecordingPayloadSink:
    """Records payloads through the PerceptionRunWriter-compatible method."""

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
        width=12,
        height=8,
        transformations=(),
    )


def _native_output() -> DinoV2NativeOutput:
    return DinoV2NativeOutput(
        array=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        model_input_width=6,
        model_input_height=4,
        patch_width=2,
        patch_height=2,
    )


def _backend(
    *,
    runtime: FakeDinoV2Runtime | None = None,
    sink: RecordingPayloadSink | None = None,
    l2_normalize: bool = False,
    feature_stage_id: str = "dense_feature_extraction",
) -> tuple[DinoV2DenseFeatureBackend, FakeDinoV2Runtime, RecordingPayloadSink]:
    selected_runtime = runtime or FakeDinoV2Runtime(_native_output())
    selected_sink = sink or RecordingPayloadSink()
    backend = DinoV2DenseFeatureBackend(
        config=DinoV2Config(
            checkpoint="facebook/dinov2-small",
            revision="commit-abc123",
            device="cpu",
            precision="float32",
            input_width=6,
            input_height=4,
            local_files_only=True,
            l2_normalize=l2_normalize,
        ),
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id=feature_stage_id,
        source_artifact_id="perception-run-0001",
        payload_sink=selected_sink,
        runtime=selected_runtime,
    )
    return backend, selected_runtime, selected_sink


def test_backend_satisfies_feature_extractor_port() -> None:
    backend, _, _ = _backend()

    assert isinstance(backend, FeatureExtractor)
    assert backend.required_scope() is FeatureScope.DENSE


def test_extract_dense_emits_canonical_metadata_sampling_and_payload() -> None:
    backend, runtime, sink = _backend()

    extraction = backend.extract_dense(_image())

    assert runtime.images == [_image()]
    assert extraction.array.shape == (2, 3, 4)
    assert extraction.dense_map.feature.shape == (2, 3, 4)
    assert extraction.dense_map.feature.dtype == "float32"
    assert extraction.dense_map.feature.scope is FeatureScope.DENSE
    assert extraction.dense_map.feature.normalization == "none"
    assert extraction.dense_map.source_artifact_id == "perception-run-0001"
    assert extraction.dense_map.sampling.grid_width == 3
    assert extraction.dense_map.sampling.grid_height == 2
    assert extraction.dense_map.sampling.source_image_width == 12
    assert extraction.dense_map.sampling.source_image_height == 8
    assert extraction.dense_map.sampling.stride_x == 4.0
    assert extraction.dense_map.sampling.stride_y == 4.0
    assert extraction.dense_map.sampling.support_width == 4.0
    assert extraction.dense_map.sampling.support_height == 4.0
    assert extraction.dense_map.sampling.coordinate_transform_id.startswith("sha256:")
    assert len(sink.calls) == 1
    persisted_feature, persisted_observation_id, persisted_array = sink.calls[0]
    assert persisted_feature == extraction.dense_map.feature
    assert persisted_observation_id == SourceObservationId("frame-0001")
    np.testing.assert_array_equal(persisted_array, extraction.array)


def test_extract_port_returns_the_same_canonical_feature_and_persists_payload() -> None:
    backend, _, sink = _backend()

    features = backend.extract(_image())

    assert len(features) == 1
    feature = features[0]
    assert feature == sink.calls[0][0]
    payload_reference = feature.payload_reference
    assert payload_reference is not None
    assert payload_reference.endswith(".npy")


def test_embedding_space_and_backend_provenance_are_exact_and_auditable() -> None:
    backend, _, _ = _backend()

    extraction = backend.extract_dense(_image())
    space = extraction.embedding_space
    feature = extraction.dense_map.feature
    provenance = backend.backend_provenance()

    assert space.family == "dinov2"
    assert space.model == "facebook/dinov2-small"
    assert space.checkpoint == "facebook/dinov2-small@commit-abc123"
    assert space.layer == "last_hidden_state.patch_tokens_after_cls_and_0_registers"
    assert space.dimension == 4
    assert space.normalization == "none"
    assert feature.embedding_space_id == embedding_space_fingerprint(space)
    assert provenance.backend_id == "dinov2_huggingface"
    assert provenance.model == "facebook/dinov2-small"
    assert provenance.version == "commit-abc123"
    assert provenance.configuration_fingerprint is not None


def test_identical_image_model_and_config_are_deterministic() -> None:
    first_backend, _, _ = _backend()
    second_backend, _, _ = _backend()

    first = first_backend.extract_dense(_image())
    second = second_backend.extract_dense(_image())

    assert first.dense_map == second.dense_map
    assert first.embedding_space == second.embedding_space
    np.testing.assert_array_equal(first.array, second.array)


def test_feature_identity_is_unique_across_composed_feature_stages() -> None:
    dense_backend, _, _ = _backend(feature_stage_id="dense_feature_extraction")
    dense_feature = dense_backend.extract_dense(_image()).dense_map.feature
    result_id = PerceptionResultId("run-0001--frame-0001")
    global_feature_id = feature_id_for(result_id=result_id, index=0)
    global_feature = VisualFeature(
        feature_id=global_feature_id,
        scope=FeatureScope.GLOBAL,
        embedding_space_id="global-space",
        shape=(4,),
        dtype="float32",
        normalization="none",
        payload_reference=f"features/{global_feature_id}.npy",
        provenance=BackendProvenance(
            backend_id="fake_global_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake-global",
            version="1",
        ),
    )

    result = PerceptionResult(
        result_id=result_id,
        source_observation_id=_image().source_observation_id,
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="sequence-0001",
        created_at="2026-09-19T00:00:00+00:00",
        features=(dense_feature, global_feature),
    )

    assert len({feature.feature_id for feature in result.features}) == 2
    assert len({feature.payload_reference for feature in result.features}) == 2


def test_register_tokens_are_removed_from_the_spatial_patch_layout() -> None:
    hidden_state = np.arange(22, dtype=np.float32).reshape(1, 11, 2)

    patch_tokens, register_count = _patch_tokens_and_register_count(
        hidden_state,
        type("Config", (), {"num_register_tokens": 4})(),
    )

    assert register_count == 4
    np.testing.assert_array_equal(patch_tokens, hidden_state[:, 5:, :])


def test_register_token_layout_changes_embedding_space_identity() -> None:
    without_registers, _, _ = _backend()
    with_registers, _, _ = _backend(
        runtime=FakeDinoV2Runtime(
            DinoV2NativeOutput(
                array=_native_output().array,
                model_input_width=6,
                model_input_height=4,
                patch_width=2,
                patch_height=2,
                register_token_count=4,
            )
        )
    )

    base = without_registers.extract_dense(_image())
    registered = with_registers.extract_dense(_image())

    registered_layer = registered.embedding_space.layer
    assert registered_layer is not None
    assert registered_layer.endswith("_4_registers")
    assert base.dense_map.feature.embedding_space_id != (
        registered.dense_map.feature.embedding_space_id
    )


def test_l2_normalization_is_explicit_and_deterministic() -> None:
    backend, _, _ = _backend(l2_normalize=True)

    extraction = backend.extract_dense(_image())

    assert extraction.dense_map.feature.normalization == "l2"
    assert extraction.embedding_space.normalization == "l2"
    norms = np.linalg.norm(extraction.array, axis=-1)
    np.testing.assert_allclose(norms, np.ones((2, 3)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.full((2, 3, 4), np.nan, dtype=np.float32), "finite"),
        (np.zeros((2, 3, 4), dtype=np.float32), "zero-norm"),
    ],
)
def test_invalid_numerical_payload_is_rejected_before_persistence(
    array: np.ndarray[Any, Any], message: str
) -> None:
    native = _native_output()
    backend, _, sink = _backend(
        runtime=FakeDinoV2Runtime(
            DinoV2NativeOutput(
                array=array,
                model_input_width=native.model_input_width,
                model_input_height=native.model_input_height,
                patch_width=native.patch_width,
                patch_height=native.patch_height,
            )
        ),
        l2_normalize=message == "zero-norm",
    )

    with pytest.raises(DinoV2InferenceError, match=message):
        backend.extract_dense(_image())

    assert sink.calls == []


def test_native_output_feeds_common_region_pooling_without_upsampling() -> None:
    backend, _, _ = _backend()
    extraction = backend.extract_dense(_image())
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=0, y=0, width=4, height=4),
        provenance=BackendProvenance(
            backend_id="fake-region",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="1",
        ),
    )

    pooled = pool_region_feature(extraction.array, dense_map=extraction.dense_map, region=region)

    np.testing.assert_array_equal(pooled.vector, extraction.array[0, 0])
    assert extraction.array.shape == (2, 3, 4)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"checkpoint": ""}, "checkpoint"),
        ({"revision": ""}, "revision"),
        ({"device": "tpu"}, "device"),
        ({"precision": "int8"}, "precision"),
        ({"input_width": 0}, "input_width"),
        ({"payload_prefix": ""}, "payload_prefix"),
    ],
)
def test_config_rejects_invalid_execution_identity(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "checkpoint": "facebook/dinov2-small",
        "revision": "commit-abc123",
        "device": "cpu",
        "precision": "float32",
        "input_width": 6,
        "input_height": 4,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        DinoV2Config(**values)  # type: ignore[arg-type]


def test_runtime_output_shape_and_configured_input_are_validated() -> None:
    invalid_output = DinoV2NativeOutput(
        array=np.zeros((2, 2, 4), dtype=np.float32),
        model_input_width=6,
        model_input_height=4,
        patch_width=2,
        patch_height=2,
    )
    backend, _, _ = _backend(runtime=FakeDinoV2Runtime(invalid_output))

    with pytest.raises(DinoV2InferenceError, match="grid shape"):
        backend.extract_dense(_image())


def test_runtime_failure_is_not_replaced_by_a_fallback() -> None:
    class FailingRuntime:
        def infer(self, image: PreparedImage) -> DinoV2NativeOutput:
            raise DinoV2InferenceError("configured DINOv2 device failed")

    backend = DinoV2DenseFeatureBackend(
        config=DinoV2Config(
            checkpoint="facebook/dinov2-small",
            revision="commit-abc123",
            device="cuda",
            precision="float16",
            input_width=6,
            input_height=4,
        ),
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id="dense_feature_extraction",
        source_artifact_id="perception-run-0001",
        payload_sink=RecordingPayloadSink(),
        runtime=FailingRuntime(),
    )

    with pytest.raises(DinoV2InferenceError, match="configured DINOv2 device failed"):
        backend.extract(_image())


def test_missing_huggingface_dependencies_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "contextmap.visual_perception.backends.dinov2.importlib.import_module",
        missing_dependency,
    )
    runtime = HuggingFaceDinoV2Runtime(
        config=DinoV2Config(
            checkpoint="facebook/dinov2-small",
            revision="commit-abc123",
            device="cpu",
            precision="float32",
            input_width=224,
            input_height=224,
        ),
        prepared_image_root=tmp_path,
    )

    with pytest.raises(DinoV2DependencyError, match="torch, transformers, and Pillow"):
        runtime.infer(_image())
