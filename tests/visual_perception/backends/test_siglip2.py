"""Contract tests for the direct SigLIP2 feature backend with a deterministic fake encoder."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
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
    EmbeddingSpace,
    EmbeddingSpaceMismatchError,
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
    embedding_space_fingerprint,
    ensure_compatible_features,
    pool_region_feature,
)
from contextmap.visual_perception.backends import siglip2
from contextmap.visual_perception.backends.siglip2 import (
    HuggingFaceSiglip2Runtime,
    Siglip2Config,
    Siglip2DependencyError,
    Siglip2FeatureBackend,
    Siglip2InferenceError,
    Siglip2ModelLoadError,
    Siglip2NativeOutput,
    Siglip2ScopeError,
)

_CHECKPOINT = "google/siglip2-so400m-patch16-512"
_REVISION = "c" * 40
_CHANNELS = 4


class FakeSiglip2Runtime:
    """Deterministic stand-in for the SigLIP2 vision encoder."""

    def __init__(self, output: Siglip2NativeOutput) -> None:
        self.output = output
        self.images: list[PreparedImage] = []

    def infer(self, image: PreparedImage) -> Siglip2NativeOutput:
        self.images.append(image)
        return self.output


class RecordingPayloadSink:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, SourceObservationId, np.ndarray[Any, Any]]] = []

    def add_feature_payload(
        self,
        feature: Any,
        source_observation_id: SourceObservationId,
        array: np.ndarray[Any, Any],
    ) -> None:
        self.calls.append((feature, source_observation_id, array.copy()))


def _image(width: int = 640, height: int = 480) -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference="prepared/frame-0001.png",
        width=width,
        height=height,
        transformations=(),
    )


def _dense_output(
    *, input_size: int = 512, patch: int = 16, channels: int = _CHANNELS
) -> Siglip2NativeOutput:
    grid = input_size // patch
    values = np.arange(grid * grid * channels, dtype=np.float32).reshape(grid, grid, channels)
    return Siglip2NativeOutput(
        array=values + 1.0,
        model_input_width=input_size,
        model_input_height=input_size,
        patch_width=patch,
        patch_height=patch,
    )


def _global_output(*, input_size: int = 512) -> Siglip2NativeOutput:
    return Siglip2NativeOutput(
        array=np.array([3.0, 0.0, 4.0, 0.0], dtype=np.float32),
        model_input_width=input_size,
        model_input_height=input_size,
        patch_width=16,
        patch_height=16,
    )


def _config(**overrides: Any) -> Siglip2Config:
    values: dict[str, Any] = {
        "checkpoint": _CHECKPOINT,
        "revision": _REVISION,
        "scope": FeatureScope.DENSE,
        "input_size": 512,
    }
    values.update(overrides)
    return Siglip2Config(**values)


def _backend(
    *,
    config: Siglip2Config | None = None,
    output: Siglip2NativeOutput | None = None,
    source_artifact_id: str = "perception-run-0001",
) -> tuple[Siglip2FeatureBackend, FakeSiglip2Runtime, RecordingPayloadSink]:
    config = config or _config()
    default_output = _global_output() if config.scope is FeatureScope.GLOBAL else _dense_output()
    runtime = FakeSiglip2Runtime(output or default_output)
    sink = RecordingPayloadSink()
    backend = Siglip2FeatureBackend(
        config=config,
        run_id=PerceptionRunId("run-0001"),
        feature_stage_id="dense_feature_extraction",
        source_artifact_id=source_artifact_id,
        payload_sink=sink,
        runtime=runtime,
    )
    return backend, runtime, sink


def _region(box: BoundingBox2D) -> Region2D:
    return Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=box,
        provenance=BackendProvenance(
            backend_id="fake-region",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="1",
        ),
    )


# --- Port and dense geometry --------------------------------------------------------


def test_dense_backend_satisfies_the_port_and_declares_its_only_scope() -> None:
    backend, runtime, _ = _backend()

    assert isinstance(backend, FeatureExtractor)
    assert backend.required_scope() is FeatureScope.DENSE
    features = backend.extract(_image())

    assert runtime.images == [_image()]
    assert [feature.scope for feature in features] == [FeatureScope.DENSE]
    assert features[0].region_id is None


@pytest.mark.parametrize(
    ("image_size", "input_size", "patch", "grid"),
    [
        # 640x480 → 512x512: células anisotrópicas de 20x15 px na imagem preparada.
        ((640, 480), 512, 16, 32),
        ((1000, 750), 384, 16, 24),
        # patch 14 sobre 384 px: 27 células cobrem 378 px e a faixa final não tem célula.
        ((640, 480), 384, 14, 27),
    ],
)
def test_dense_grid_geometry_is_exact_for_synthetic_image_dimensions(
    image_size: tuple[int, int], input_size: int, patch: int, grid: int
) -> None:
    width, height = image_size
    backend, _, _ = _backend(
        config=_config(input_size=input_size),
        output=_dense_output(input_size=input_size, patch=patch),
    )

    sampling = backend.extract_dense(_image(width, height)).dense_map.sampling

    assert (sampling.grid_width, sampling.grid_height) == (grid, grid)
    assert (sampling.source_image_width, sampling.source_image_height) == (width, height)
    assert (sampling.origin_x, sampling.origin_y) == (0.0, 0.0)
    assert sampling.stride_x == patch * width / input_size
    assert sampling.stride_y == patch * height / input_size
    assert (sampling.support_width, sampling.support_height) == (
        sampling.stride_x,
        sampling.stride_y,
    )
    covered = grid * patch
    assert grid * sampling.stride_x == pytest.approx(covered * width / input_size)
    assert grid * sampling.stride_y == pytest.approx(covered * height / input_size)


def test_the_native_patch_grid_is_persisted_without_any_resampling() -> None:
    native = _dense_output()
    backend, _, sink = _backend(output=native)

    # Uma imagem preparada muito maior que a entrada do modelo continua com a grade nativa 32x32.
    extraction = backend.extract_dense(_image(2048, 1536))

    feature = extraction.dense_map.feature
    assert feature.shape == native.array.shape == (32, 32, _CHANNELS)
    assert feature.dtype == "float32"
    assert feature.normalization == "none"
    assert feature.payload_reference == f"features/{feature.feature_id}.npy"
    assert len(sink.calls) == 1
    persisted_feature, observation_id, persisted = sink.calls[0]
    assert persisted_feature == feature
    assert observation_id == _image().source_observation_id
    np.testing.assert_array_equal(persisted, native.array)
    np.testing.assert_array_equal(extraction.array, native.array)
    assert extraction.dense_map.source_artifact_id == "perception-run-0001"


def test_a_runtime_grid_that_disagrees_with_the_input_geometry_is_rejected_not_resampled() -> None:
    wrong = Siglip2NativeOutput(
        array=np.ones((16, 16, _CHANNELS), dtype=np.float32),
        model_input_width=512,
        model_input_height=512,
        patch_width=16,
        patch_height=16,
    )
    backend, _, sink = _backend(output=wrong)

    with pytest.raises(Siglip2InferenceError, match="grid shape"):
        backend.extract_dense(_image())

    assert sink.calls == []


def test_a_runtime_input_size_other_than_the_configured_one_is_rejected() -> None:
    backend, _, sink = _backend(output=_dense_output(input_size=384))

    with pytest.raises(Siglip2InferenceError, match="input dimensions"):
        backend.extract_dense(_image())

    assert sink.calls == []


def test_a_runtime_dtype_other_than_the_configured_precision_is_rejected() -> None:
    native = _dense_output()
    backend, _, _ = _backend(
        output=dataclasses.replace(native, array=native.array.astype(np.float16))
    )

    with pytest.raises(Siglip2InferenceError, match="precision"):
        backend.extract_dense(_image())


def test_the_native_map_uses_the_common_region_pooling_without_a_backend_branch() -> None:
    backend, _, _ = _backend()
    extraction = backend.extract_dense(_image())

    # Um box de 20x15 px na origem é exatamente a célula (0, 0) da grade 32x32 sobre 640x480.
    pooled = pool_region_feature(
        extraction.array,
        dense_map=extraction.dense_map,
        region=_region(BoundingBox2D(x=0, y=0, width=20, height=15)),
    )

    np.testing.assert_array_equal(pooled.vector, extraction.array[0, 0])


# --- Embedding space identity ------------------------------------------------------


def test_the_dense_embedding_space_names_checkpoint_output_normalization_and_preprocessing() -> (
    None
):
    backend, _, _ = _backend()

    extraction = backend.extract_dense(_image())

    expected = EmbeddingSpace(
        family="siglip2",
        model=_CHECKPOINT,
        version=_REVISION,
        checkpoint=f"{_CHECKPOINT}@{_REVISION}",
        layer=(
            "last_hidden_state.post_layernorm_patch_tokens"
            "|preprocessing=pillow_direct_bicubic_resize_512x512"
            "_processor_rescale_normalize_no_crop_v1"
        ),
        dimension=_CHANNELS,
        normalization="none",
    )
    assert extraction.embedding_space == expected
    assert extraction.dense_map.feature.embedding_space_id == embedding_space_fingerprint(expected)


def test_the_embedding_space_fingerprint_is_deterministic_across_instances() -> None:
    first, _, _ = _backend()
    second, _, _ = _backend()

    first_ids = {first.extract(_image())[0].embedding_space_id for _ in range(2)}
    second_id = second.extract(_image())[0].embedding_space_id

    assert first_ids == {second_id}


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "d" * 40},
        {"checkpoint": "google/siglip2-base-patch16-512"},
        {"resize_filter": "bilinear"},
        {"input_size": 384},
        {"l2_normalize": True},
        {"scope": FeatureScope.GLOBAL},
    ],
    ids=["revision", "checkpoint", "resize_filter", "input_size", "normalization", "scope"],
)
def test_a_revision_or_preprocessing_change_breaks_embedding_compatibility(
    change: dict[str, Any],
) -> None:
    reference_backend, _, _ = _backend()
    reference = reference_backend.extract(_image())[0]
    changed_config = _config(**change)
    output = None
    if "input_size" in change:
        output = _dense_output(input_size=change["input_size"])
    changed_backend, _, _ = _backend(config=changed_config, output=output)

    changed = changed_backend.extract(_image())[0]

    assert changed.embedding_space_id != reference.embedding_space_id
    with pytest.raises(EmbeddingSpaceMismatchError):
        ensure_compatible_features(reference, changed)


@pytest.mark.parametrize(
    "change",
    [{"device": "cuda"}, {"local_files_only": False}, {"payload_prefix": "siglip2"}],
    ids=["device", "local_files_only", "payload_prefix"],
)
def test_execution_settings_keep_the_embedding_space_but_not_the_configuration_identity(
    change: dict[str, Any],
) -> None:
    reference_backend, _, _ = _backend()
    changed_backend, _, _ = _backend(config=_config(**change))

    reference = reference_backend.extract(_image())[0]
    changed = changed_backend.extract(_image())[0]

    ensure_compatible_features(reference, changed)
    assert (
        changed.provenance.configuration_fingerprint
        != reference.provenance.configuration_fingerprint
    )


def test_the_backend_provenance_records_the_pinned_checkpoint() -> None:
    backend, _, _ = _backend()

    provenance = backend.backend_provenance()

    assert provenance.backend_id == "siglip2_huggingface"
    assert provenance.capability == "feature_extractor"
    assert provenance.provider == "huggingface"
    assert provenance.model == _CHECKPOINT
    assert provenance.version == _REVISION
    assert provenance.configuration_fingerprint is not None
    assert provenance.configuration_fingerprint.startswith("sha256:")


# --- Global scope ------------------------------------------------------------------


def test_the_global_backend_emits_one_pooled_image_vector() -> None:
    backend, _, sink = _backend(config=_config(scope=FeatureScope.GLOBAL, l2_normalize=True))

    assert backend.required_scope() is FeatureScope.GLOBAL
    extraction = backend.extract_global(_image())

    feature = extraction.feature
    assert feature.scope is FeatureScope.GLOBAL
    assert feature.region_id is None
    assert feature.shape == (4,)
    assert feature.normalization == "l2"
    np.testing.assert_allclose(extraction.array, [0.6, 0.0, 0.8, 0.0], rtol=1e-6)
    assert extraction.embedding_space.layer == (
        "pooler_output.attention_pooling_head"
        "|preprocessing=pillow_direct_bicubic_resize_512x512"
        "_processor_rescale_normalize_no_crop_v1"
    )
    assert feature.embedding_space_id == embedding_space_fingerprint(extraction.embedding_space)
    assert [call[0] for call in sink.calls] == [feature]
    assert backend.extract(_image()) == (feature,)


# --- Scope validation --------------------------------------------------------------


def test_a_region_scope_is_rejected_by_the_configuration() -> None:
    with pytest.raises(ValueError, match="scope"):
        _config(scope=FeatureScope.REGION)


def test_a_dense_instance_refuses_global_extraction() -> None:
    backend, runtime, _ = _backend()

    with pytest.raises(Siglip2ScopeError, match="DENSE"):
        backend.extract_global(_image())

    assert runtime.images == []


def test_a_global_instance_refuses_dense_extraction() -> None:
    backend, runtime, _ = _backend(config=_config(scope=FeatureScope.GLOBAL))

    with pytest.raises(Siglip2ScopeError, match="GLOBAL"):
        backend.extract_dense(_image())

    assert runtime.images == []


@pytest.mark.parametrize(
    ("scope", "output"),
    [(FeatureScope.DENSE, _global_output()), (FeatureScope.GLOBAL, _dense_output())],
    ids=["dense-gets-pooled", "global-gets-grid"],
)
def test_a_runtime_output_of_the_other_scope_is_rejected(
    scope: FeatureScope, output: Siglip2NativeOutput
) -> None:
    backend, _, sink = _backend(config=_config(scope=scope), output=output)

    with pytest.raises(Siglip2InferenceError, match="shape"):
        backend.extract(_image())

    assert sink.calls == []


# --- Numerical and configuration validation ----------------------------------------


@pytest.mark.parametrize(
    ("value", "l2_normalize", "message"),
    [(np.inf, False, "finite"), (0.0, True, "zero-norm")],
)
def test_an_invalid_numerical_payload_is_rejected_before_persistence(
    value: float, l2_normalize: bool, message: str
) -> None:
    native = _dense_output()
    backend, _, sink = _backend(
        config=_config(l2_normalize=l2_normalize),
        output=dataclasses.replace(native, array=np.full_like(native.array, value)),
    )

    with pytest.raises(Siglip2InferenceError, match=message):
        backend.extract_dense(_image())

    assert sink.calls == []


def test_l2_normalization_is_explicit_per_patch_vector() -> None:
    backend, _, _ = _backend(config=_config(l2_normalize=True))

    extraction = backend.extract_dense(_image())

    assert extraction.dense_map.feature.normalization == "l2"
    np.testing.assert_allclose(
        np.linalg.norm(extraction.array, axis=-1), np.ones((32, 32)), rtol=1e-6
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", ""),
        ("revision", "main"),
        ("device", "tpu"),
        ("precision", "bfloat16"),
        ("input_size", 0),
        ("resize_filter", "lanczos"),
        ("payload_prefix", "../escape"),
        ("code_version", ""),
    ],
)
def test_an_invalid_configuration_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


# --- Persistence through the existing feature artifact ------------------------------


def test_the_payload_and_dense_geometry_round_trip_through_the_perception_run(
    tmp_path: Path,
) -> None:
    # O dono da feature é o run que a persiste: o writer exige source_artifact_id == run_id.
    backend, _, _ = _backend(source_artifact_id="run-0001")
    extraction = backend.extract_dense(_image())
    feature = extraction.dense_map.feature
    sampling = extraction.dense_map.sampling
    writer = PerceptionRunWriter(
        output_dir=tmp_path / "run-0001",
        sequence_name="corridor",
        run_id=PerceptionRunId("run-0001"),
        run_index=1,
        sequence_artifact_id="corridor-artifact",
        selection_id="selection-0001",
        enabled_capabilities=frozenset({"feature_extractor"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:pipeline",
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

    run_dir = tmp_path / "run-0001"
    record = json.loads(
        (run_dir / "metrics" / "feature-extraction.jsonl").read_text(encoding="utf-8")
    )
    rebuilt_sampling = DenseFeatureSampling(
        **{key: record["dense"][key] for key in DenseFeatureSampling.__dataclass_fields__}
    )
    reader = PerceptionRunReader(run_dir)
    stored = reader.list_results()[0].features[0]
    loaded = reader.feature_store().load(_image().source_observation_id, stored.feature_id)
    rebuilt = DenseFeatureMap(
        feature=stored, sampling=rebuilt_sampling, source_artifact_id="run-0001"
    )

    assert stored == feature
    assert rebuilt == extraction.dense_map
    np.testing.assert_array_equal(loaded, extraction.array)


# --- Hugging Face runtime with fake SDK modules -------------------------------------


class _FakeTensor:
    """The slice of the torch tensor API the runtime uses, over a NumPy array."""

    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    def __getitem__(self, key: Any) -> _FakeTensor:
        return _FakeTensor(self.array[key])

    def reshape(self, *shape: int) -> _FakeTensor:
        return _FakeTensor(self.array.reshape(*shape))

    def detach(self) -> _FakeTensor:
        return self

    def to(self, *args: Any, **kwargs: Any) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.array


class _FakePilImage:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.resizes: list[tuple[tuple[int, int], object]] = []

    def convert(self, mode: str) -> _FakePilImage:
        return self

    def resize(self, size: tuple[int, int], resample: object) -> _FakePilImage:
        self.resizes.append((size, resample))
        return _FakePilImage(size)


class _FakeImageModule:
    Resampling = SimpleNamespace(BICUBIC="bicubic-filter", BILINEAR="bilinear-filter")

    def __init__(self, size: tuple[int, int]) -> None:
        self.decoded = _FakePilImage(size)

    @contextmanager
    def open(self, path: Path) -> Iterator[_FakePilImage]:
        yield self.decoded


class _FakeProcessor:
    def __call__(self, *, images: list[_FakePilImage], **kwargs: Any) -> dict[str, _FakeTensor]:
        width, height = images[0].size
        return {"pixel_values": _FakeTensor(np.zeros((1, 3, height, width), dtype=np.float32))}


class _FakeVisionModel:
    """Emits row-major patch tokens whose value encodes their ``(row, column)``."""

    def __init__(self, *, patch: int, image_size: int, with_head: bool = True) -> None:
        self.config = SimpleNamespace(patch_size=patch, image_size=image_size)
        self._with_head = with_head
        self.calls = 0

    def to(self, device: str) -> _FakeVisionModel:
        return self

    def eval(self) -> _FakeVisionModel:
        return self

    def __call__(self, *, pixel_values: _FakeTensor) -> SimpleNamespace:
        self.calls += 1
        _, _, height, width = pixel_values.shape
        patch = self.config.patch_size
        rows, columns = np.meshgrid(
            np.arange(height // patch), np.arange(width // patch), indexing="ij"
        )
        tokens = np.stack((rows, columns), axis=-1).reshape(1, -1, 2).astype(np.float32)
        pooled = np.array([[1.0, 2.0]], dtype=np.float32) if self._with_head else None
        return SimpleNamespace(
            last_hidden_state=_FakeTensor(tokens),
            pooler_output=None if pooled is None else _FakeTensor(pooled),
        )


class _FakeTransformers:
    def __init__(self, *, model_type: str = "siglip", native_size: int = 512, **model: Any):
        self.model = _FakeVisionModel(image_size=native_size, **{"patch": 16, **model})
        self.checkpoint_config = SimpleNamespace(
            model_type=model_type,
            vision_config=SimpleNamespace(image_size=native_size, patch_size=16),
        )
        self.loaded: list[str] = []

    @property
    def AutoConfig(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load("config", self.checkpoint_config))

    @property
    def AutoImageProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load("processor", _FakeProcessor()))

    @property
    def SiglipVisionModel(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load("model", self.model))

    def _load(self, name: str, value: object) -> Any:
        def load(*args: Any, **kwargs: Any) -> object:
            self.loaded.append(name)
            return value

        return load


def _install_sdks(
    monkeypatch: pytest.MonkeyPatch, transformers: _FakeTransformers, images: _FakeImageModule
) -> None:
    torch = SimpleNamespace(float32=np.float32, float16=np.float16, inference_mode=nullcontext)
    sdks = {"torch": torch, "transformers": transformers, "PIL.Image": images}
    monkeypatch.setattr(
        siglip2, "importlib", SimpleNamespace(import_module=lambda name: sdks[name])
    )


def _runtime(tmp_path: Path, config: Siglip2Config) -> HuggingFaceSiglip2Runtime:
    (tmp_path / "prepared").mkdir(exist_ok=True)
    (tmp_path / "prepared" / "frame-0001.png").write_bytes(b"png")
    return HuggingFaceSiglip2Runtime(config=config, prepared_image_root=tmp_path)


def test_the_runtime_reshapes_row_major_patch_tokens_into_the_native_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = _FakeTransformers()
    images = _FakeImageModule((640, 480))
    _install_sdks(monkeypatch, transformers, images)

    native = _runtime(tmp_path, _config()).infer(_image())

    assert native.array.shape == (32, 32, 2)
    np.testing.assert_array_equal(native.array[5, 7], [5.0, 7.0])
    np.testing.assert_array_equal(native.array[31, 0], [31.0, 0.0])
    assert (native.model_input_width, native.model_input_height) == (512, 512)
    assert (native.patch_width, native.patch_height) == (16, 16)
    assert images.decoded.resizes == [((512, 512), "bicubic-filter")]


def test_the_runtime_resizes_with_the_configured_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = _FakeImageModule((640, 480))
    _install_sdks(monkeypatch, _FakeTransformers(), images)

    _runtime(tmp_path, _config(resize_filter="bilinear")).infer(_image())

    assert images.decoded.resizes == [((512, 512), "bilinear-filter")]


def test_the_global_runtime_returns_the_attention_pooled_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_sdks(monkeypatch, _FakeTransformers(), _FakeImageModule((640, 480)))

    native = _runtime(tmp_path, _config(scope=FeatureScope.GLOBAL)).infer(_image())

    np.testing.assert_array_equal(native.array, [1.0, 2.0])


def test_a_checkpoint_without_a_pooling_head_cannot_produce_global_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_sdks(monkeypatch, _FakeTransformers(with_head=False), _FakeImageModule((640, 480)))

    with pytest.raises(Siglip2InferenceError, match="pooling head"):
        _runtime(tmp_path, _config(scope=FeatureScope.GLOBAL)).infer(_image())


def test_a_naflex_checkpoint_is_refused_before_its_weights_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = _FakeTransformers(model_type="siglip2")
    _install_sdks(monkeypatch, transformers, _FakeImageModule((640, 480)))

    with pytest.raises(Siglip2ModelLoadError, match="NaFlex"):
        _runtime(tmp_path, _config()).infer(_image())

    assert transformers.loaded == ["config"]


def test_an_input_size_other_than_the_native_resolution_is_refused_before_weights_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = _FakeTransformers(native_size=384)
    _install_sdks(monkeypatch, transformers, _FakeImageModule((640, 480)))

    with pytest.raises(Siglip2ModelLoadError, match="position embeddings"):
        _runtime(tmp_path, _config()).infer(_image())

    assert transformers.loaded == ["config"]


def test_a_decoded_image_that_disagrees_with_its_metadata_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_sdks(monkeypatch, _FakeTransformers(), _FakeImageModule((320, 240)))

    with pytest.raises(Siglip2InferenceError, match="does not match"):
        _runtime(tmp_path, _config()).infer(_image())


def test_missing_sdk_dependencies_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(siglip2, "importlib", SimpleNamespace(import_module=missing))

    with pytest.raises(Siglip2DependencyError, match="torch, transformers, and Pillow"):
        _runtime(tmp_path, _config()).infer(_image())
