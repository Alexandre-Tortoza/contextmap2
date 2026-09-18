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
    PerceptionRunId,
    PreparedImage,
    Region2D,
    RegionId,
    embedding_space_fingerprint,
    pool_region_feature,
)
from contextmap.visual_perception.backends.dinov3 import (
    DinoV3Config,
    DinoV3DenseFeatureBackend,
    DinoV3DependencyError,
    DinoV3InferenceError,
    DinoV3NativeOutput,
    HuggingFaceDinoV3Runtime,
)


class FakeDinoV3Runtime:
    def __init__(self, output: DinoV3NativeOutput) -> None:
        self.output = output
        self.images: list[PreparedImage] = []

    def infer(self, image: PreparedImage) -> DinoV3NativeOutput:
        self.images.append(image)
        return self.output


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
        width=12,
        height=8,
        transformations=("rectify:v1",),
    )


def _native_output() -> DinoV3NativeOutput:
    return DinoV3NativeOutput(
        array=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        model_input_width=6,
        model_input_height=4,
        patch_width=2,
        patch_height=2,
        register_token_count=4,
    )


def _backend(
    *, output: DinoV3NativeOutput | None = None, l2_normalize: bool = False
) -> tuple[DinoV3DenseFeatureBackend, FakeDinoV3Runtime, RecordingPayloadSink]:
    runtime = FakeDinoV3Runtime(output or _native_output())
    sink = RecordingPayloadSink()
    backend = DinoV3DenseFeatureBackend(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m",
            revision="commit-def456",
            device="cpu",
            precision="float32",
            input_width=6,
            input_height=4,
            local_files_only=True,
            l2_normalize=l2_normalize,
        ),
        run_id=PerceptionRunId("run-0001"),
        source_artifact_id="perception-run-0001",
        payload_sink=sink,
        runtime=runtime,
    )
    return backend, runtime, sink


def test_backend_satisfies_port_and_emits_native_canonical_map() -> None:
    backend, runtime, sink = _backend()

    assert isinstance(backend, FeatureExtractor)
    assert backend.required_scope() is FeatureScope.DENSE
    extraction = backend.extract_dense(_image())

    assert runtime.images == [_image()]
    assert extraction.array.shape == (2, 3, 4)
    assert extraction.dense_map.feature.shape == (2, 3, 4)
    assert extraction.dense_map.feature.dtype == "float32"
    assert extraction.dense_map.feature.normalization == "none"
    assert extraction.dense_map.source_artifact_id == "perception-run-0001"
    assert extraction.dense_map.sampling.grid_width == 3
    assert extraction.dense_map.sampling.grid_height == 2
    assert extraction.dense_map.sampling.stride_x == 4.0
    assert extraction.dense_map.sampling.stride_y == 4.0
    assert extraction.register_token_count == 4
    assert sink.calls[0][0] == extraction.dense_map.feature
    np.testing.assert_array_equal(sink.calls[0][2], extraction.array)


def test_extract_port_persists_payload_and_returns_feature_metadata() -> None:
    backend, _, sink = _backend()

    features = backend.extract(_image())

    assert len(features) == 1
    assert features[0] == sink.calls[0][0]
    assert features[0].scope is FeatureScope.DENSE


def test_embedding_identity_distinguishes_dinov3_and_records_register_policy() -> None:
    backend, _, _ = _backend()
    extraction = backend.extract_dense(_image())
    space = extraction.embedding_space
    provenance = backend.backend_provenance()

    assert space.family == "dinov3"
    assert space.model == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert space.checkpoint == "facebook/dinov3-vits16-pretrain-lvd1689m@commit-def456"
    assert space.layer == "last_hidden_state.patch_tokens_after_registers"
    assert space.dimension == 4
    assert extraction.dense_map.feature.embedding_space_id == embedding_space_fingerprint(space)
    assert provenance.backend_id == "dinov3_huggingface"
    assert provenance.configuration_fingerprint is not None


def test_same_input_and_config_are_deterministic_and_l2_is_explicit() -> None:
    first_backend, _, _ = _backend(l2_normalize=True)
    second_backend, _, _ = _backend(l2_normalize=True)

    first = first_backend.extract_dense(_image())
    second = second_backend.extract_dense(_image())

    assert first.dense_map == second.dense_map
    np.testing.assert_array_equal(first.array, second.array)
    np.testing.assert_allclose(
        np.linalg.norm(first.array, axis=-1), np.ones((2, 3)), rtol=1e-6, atol=1e-6
    )
    assert first.dense_map.feature.normalization == "l2"


def test_native_map_uses_common_region_pooling_without_backend_branch() -> None:
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", ""),
        ("revision", ""),
        ("device", "tpu"),
        ("precision", "int8"),
        ("input_height", 0),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "checkpoint": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "revision": "commit-def456",
        "input_width": 6,
        "input_height": 4,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        DinoV3Config(**values)  # type: ignore[arg-type]


def test_native_grid_mismatch_fails_explicitly() -> None:
    output = DinoV3NativeOutput(
        array=np.zeros((2, 2, 4), dtype=np.float32),
        model_input_width=6,
        model_input_height=4,
        patch_width=2,
        patch_height=2,
        register_token_count=4,
    )
    backend, _, _ = _backend(output=output)

    with pytest.raises(DinoV3InferenceError, match="grid shape"):
        backend.extract_dense(_image())


def test_runtime_failure_has_no_fallback() -> None:
    class FailingRuntime:
        def infer(self, image: PreparedImage) -> DinoV3NativeOutput:
            raise DinoV3InferenceError("DINOv3 checkpoint failed")

    backend = DinoV3DenseFeatureBackend(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m",
            revision="commit-def456",
            input_width=6,
            input_height=4,
        ),
        run_id=PerceptionRunId("run-0001"),
        source_artifact_id="perception-run-0001",
        payload_sink=RecordingPayloadSink(),
        runtime=FailingRuntime(),
    )

    with pytest.raises(DinoV3InferenceError, match="checkpoint failed"):
        backend.extract(_image())


def test_missing_sdk_dependencies_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "contextmap.visual_perception.backends.dinov3.importlib.import_module",
        missing_dependency,
    )
    runtime = HuggingFaceDinoV3Runtime(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m",
            revision="commit-def456",
        ),
        prepared_image_root=tmp_path,
    )

    with pytest.raises(DinoV3DependencyError, match="torch, transformers, and Pillow"):
        runtime.infer(_image())
