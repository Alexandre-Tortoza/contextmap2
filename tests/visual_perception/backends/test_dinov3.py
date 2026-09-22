from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    BackendProvenance,
    BoundingBox2D,
    DenseFeatureDiagnostic,
    DenseFeatureMap,
    DenseFeatureSampling,
    FeatureDebugLevel,
    FeatureEventStatus,
    FeatureExtractionDiagnostic,
    FeatureExtractor,
    FeatureScope,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    Region2D,
    RegionId,
    VisualFeature,
    embedding_space_fingerprint,
    feature_id_for,
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

_REVISION = "b" * 40


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
        transformations=(),
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
    *,
    output: DinoV3NativeOutput | None = None,
    l2_normalize: bool = False,
    feature_stage_id: str = "dense_feature_extraction",
    source_artifact_id: str = "perception-run-0001",
) -> tuple[DinoV3DenseFeatureBackend, FakeDinoV3Runtime, RecordingPayloadSink]:
    runtime = FakeDinoV3Runtime(output or _native_output())
    sink = RecordingPayloadSink()
    backend = DinoV3DenseFeatureBackend(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m",
            revision=_REVISION,
            device="cpu",
            precision="float32",
            input_width=6,
            input_height=4,
            local_files_only=True,
            l2_normalize=l2_normalize,
        ),
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id=feature_stage_id,
        source_artifact_id=source_artifact_id,
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
    assert space.checkpoint == f"facebook/dinov3-vits16-pretrain-lvd1689m@{_REVISION}"
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


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.full((2, 3, 4), np.inf, dtype=np.float32), "finite"),
        (np.zeros((2, 3, 4), dtype=np.float32), "zero-norm"),
    ],
)
def test_invalid_numerical_payload_is_rejected_before_persistence(
    array: np.ndarray[Any, Any], message: str
) -> None:
    native = _native_output()
    backend, _, sink = _backend(
        output=DinoV3NativeOutput(
            array=array,
            model_input_width=native.model_input_width,
            model_input_height=native.model_input_height,
            patch_width=native.patch_width,
            patch_height=native.patch_height,
            register_token_count=native.register_token_count,
        ),
        l2_normalize=message == "zero-norm",
    )

    with pytest.raises(DinoV3InferenceError, match=message):
        backend.extract_dense(_image())

    assert sink.calls == []


def test_feature_identity_is_unique_across_composed_feature_stages() -> None:
    dense_backend, _, _ = _backend(feature_stage_id="dense_feature_extraction")
    dense_feature = dense_backend.extract_dense(_image()).dense_map.feature
    result_id = PerceptionResultId("run-0001--frame-0001")
    global_feature_id = feature_id_for(
        result_id=result_id, producer_id="global_feature_extraction", index=0
    )
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
        ("revision", "main"),
        ("device", "tpu"),
        ("precision", "int8"),
        ("input_height", 0),
        ("payload_prefix", ""),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "checkpoint": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "revision": _REVISION,
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
            revision=_REVISION,
            input_width=6,
            input_height=4,
        ),
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id="dense_feature_extraction",
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
            revision=_REVISION,
        ),
        prepared_image_root=tmp_path,
    )

    with pytest.raises(DinoV3DependencyError, match="torch, transformers, and Pillow"):
        runtime.infer(_image())


def test_persisted_required_metrics_rebuild_the_dense_map_and_reproduce_pooling(
    tmp_path: Path,
) -> None:
    """Regression from the real DINOv3 run: the mapping must not depend on debug or assumptions."""
    # O dono da feature é o run que a persiste: o writer exige source_artifact_id == run_id.
    backend, _, sink = _backend(source_artifact_id="run-0001")
    extraction = backend.extract_dense(_image())
    feature = extraction.dense_map.feature
    sampling = extraction.dense_map.sampling
    writer = PerceptionRunWriter(
        workspace_root=tmp_path,
        sequence_name="corridor",
        run_id=PerceptionRunId("run-0001"),
        run_index=1,
        sequence_artifact_id="corridor-artifact",
        selection_id="selection-0001",
        enabled_capabilities=frozenset({"feature_extractor"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:pipeline",
        selection_label="frame-0001",
        profile_label="dinov3",
        feature_debug_level=FeatureDebugLevel.NONE,
    )
    writer.add_feature_payload(feature, _image().source_observation_id, extraction.array)
    writer.add_result(
        PerceptionResult(
            result_id=PerceptionResultId("run-0001--frame-0001"),
            source_observation_id=_image().source_observation_id,
            run_id=PerceptionRunId("run-0001"),
            sequence_artifact_id="corridor-artifact",
            created_at="2026-01-01T00:00:00+00:00",
            features=(feature,),
        )
    )
    writer.add_feature_diagnostic(
        FeatureExtractionDiagnostic(
            event_id="dense-frame-0001",
            source_observation_id=_image().source_observation_id,
            source_prepared_image_reference=_image().payload_reference,
            source_image_width=sampling.source_image_width,
            source_image_height=sampling.source_image_height,
            stage_id="dense_feature_extraction",
            status=FeatureEventStatus.SUCCEEDED,
            backend=feature.provenance,
            feature_id=feature.feature_id,
            scope=FeatureScope.DENSE,
            embedding_space=extraction.embedding_space,
            output_shape=feature.shape,
            dtype=feature.dtype,
            normalization=feature.normalization,
            payload_reference=feature.payload_reference,
            dense=DenseFeatureDiagnostic(
                source_artifact_id=extraction.dense_map.source_artifact_id,
                grid_width=sampling.grid_width,
                grid_height=sampling.grid_height,
                origin_x=sampling.origin_x,
                origin_y=sampling.origin_y,
                stride_x=sampling.stride_x,
                stride_y=sampling.stride_y,
                support_width=sampling.support_width,
                support_height=sampling.support_height,
                coordinate_transform_id=sampling.coordinate_transform_id,
            ),
        )
    )
    writer.finalize()

    run_dir = tmp_path / "runs" / "visual-perception" / "corridor" / "run-0001__frame-0001__dinov3"
    assert not (run_dir / "debug").exists()
    record = json.loads(
        (run_dir / "metrics" / "feature-extraction.jsonl").read_text(encoding="utf-8")
    )
    persisted = record["dense"]
    rebuilt_sampling = DenseFeatureSampling(
        **{key: persisted[key] for key in DenseFeatureSampling.__dataclass_fields__}
    )
    reader = PerceptionRunReader(run_dir)
    stored = reader.list_results()[0].features[0]
    rebuilt_map = DenseFeatureMap(
        feature=stored,
        sampling=rebuilt_sampling,
        source_artifact_id=persisted["source_artifact_id"],
    )
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=2, y=1, width=8, height=6),
        provenance=BackendProvenance(
            backend_id="fake-region",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="1",
        ),
    )

    loaded = reader.feature_store().load(_image().source_observation_id, stored.feature_id)
    from_artifact = pool_region_feature(loaded, dense_map=rebuilt_map, region=region)
    from_memory = pool_region_feature(
        extraction.array, dense_map=extraction.dense_map, region=region
    )

    assert rebuilt_sampling == sampling
    np.testing.assert_array_equal(from_artifact.vector, from_memory.vector)
    assert sink.calls[0][0] == feature
