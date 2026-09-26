"""Contract tests for the LocateAnything runtime and its reproducible configuration (#569).

Fake ``torch``/``transformers``/``PIL``/``batch_utils`` modules stand in for the SDKs, so
these tests prove how the configuration maps onto the upstream loading and generation
calls and which failures are explicit. They never load a model, need a GPU or a network,
and say nothing about LocateAnything's quality.
"""

from __future__ import annotations

import hashlib
import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import ArtifactReference, PreparedImage
from contextmap.visual_perception.backends import locateanything
from contextmap.visual_perception.backends.locateanything import (
    LocateAnythingConfig,
    LocateAnythingDependencyError,
    LocateAnythingDeviceError,
    LocateAnythingGenerationMode,
    LocateAnythingInferenceError,
    LocateAnythingModelLoadError,
    LocateAnythingRuntimeMode,
    TransformersLocateAnythingRuntime,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
PIXELS = b"png bytes are never decoded by the fake SDK"
ANSWER = "<ref>chair</ref><box><100><200><300><400></box><|im_end|>"
HISTORY = [("mtp", "<ref>chair</ref>"), ("mtp", "<box><100><200><300><400></box><|im_end|>")]
STATS = "\nStatistic Info, num_tokens=8; generate_time(s)=0.2; switch_to_ar=0\n"


def _config(**overrides: Any) -> LocateAnythingConfig:
    values: dict[str, Any] = {
        "model": "nvidia/LocateAnything-3B",
        "revision": REVISION,
        "device": "cuda",
        "dtype": "bfloat16",
        "generation_mode": LocateAnythingGenerationMode.HYBRID,
        "max_new_tokens": 2048,
        "temperature": 0.0,
        "text_attention": "sdpa",
        "vision_attention": "flash_attention_2",
    }
    values.update(overrides)
    return LocateAnythingConfig(**values)


def _batch_config(**overrides: Any) -> LocateAnythingConfig:
    values: dict[str, Any] = {
        "runtime": LocateAnythingRuntimeMode.BATCH,
        "text_attention": "la_flash",
        "scheduler": "pipeline",
        "group_size": 0,
    }
    values.update(overrides)
    return _config(**values)


# --- configuration --------------------------------------------------------------------------


def test_the_standard_configuration_validates_without_importing_anything() -> None:
    before = set(sys.modules)

    config = _config()

    assert config.runtime is LocateAnythingRuntimeMode.STANDARD
    assert config.local_files_only
    assert not {"torch", "transformers"} & (set(sys.modules) - before)


@pytest.mark.parametrize("revision", ["main", "v1.0", "0123456", REVISION.upper(), ""])
def test_a_reproducible_configuration_requires_a_pinned_commit(revision: str) -> None:
    with pytest.raises(ValueError, match="revision"):
        _config(revision=revision)


def test_a_local_model_directory_must_be_the_snapshot_of_the_pinned_revision() -> None:
    _config(model=f"/models/hub/models--nvidia--LocateAnything-3B/snapshots/{REVISION}")

    with pytest.raises(ValueError, match="snapshot"):
        _config(model="/models/LocateAnything-3B")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"text_attention": "la_flash"}, "text_attention"),
        ({"text_attention": "flash_attention_2"}, "text_attention"),
        ({"text_attention": "auto"}, "text_attention"),
        ({"vision_attention": "auto"}, "vision_attention"),
        ({"vision_attention": "la_flash"}, "vision_attention"),
        ({"scheduler": "pipeline"}, "scheduler"),
        ({"group_size": 2}, "group_size"),
        ({"device": "cpu", "vision_attention": "flash_attention_2"}, "CUDA"),
        (
            {
                "device": "cpu",
                "dtype": "float32",
                "text_attention": "magi",
                "vision_attention": "sdpa",
            },
            "CUDA",
        ),
    ],
)
def test_unknown_attention_and_runtime_combinations_fail_explicitly_for_the_standard_runtime(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scheduler": None}, "scheduler"),
        ({"scheduler": "fastest"}, "scheduler"),
        ({"group_size": None}, "group_size"),
        ({"group_size": -1}, "group_size"),
        ({"generation_mode": LocateAnythingGenerationMode.FAST}, "hybrid"),
        ({"text_attention": "flash_attention_2"}, "text_attention"),
        ({"dtype": "float16"}, "bfloat16"),
        (
            {
                "device": "cpu",
                "dtype": "float32",
                "text_attention": "sdpa",
                "vision_attention": "sdpa",
            },
            "CUDA",
        ),
    ],
)
def test_unknown_attention_and_runtime_combinations_fail_explicitly_for_the_batch_runtime(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _batch_config(**overrides)


def test_a_cpu_configuration_is_valid_with_portable_attention() -> None:
    config = _config(device="cpu", dtype="float32", vision_attention="sdpa")

    assert config.device == "cpu"


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime": LocateAnythingRuntimeMode.BATCH, "scheduler": "pipeline", "group_size": 0},
        {"text_attention": "eager"},
        {"vision_attention": "sdpa"},
    ],
)
def test_runtime_and_attention_settings_change_the_configuration_fingerprint(
    overrides: dict[str, Any],
) -> None:
    assert _config(**overrides).fingerprint != _config().fingerprint


def test_batch_scheduler_and_group_size_change_the_configuration_fingerprint() -> None:
    assert _batch_config(scheduler="eager").fingerprint != _batch_config().fingerprint
    assert _batch_config(group_size=4).fingerprint != _batch_config().fingerprint


def test_the_asset_download_policy_is_recorded_but_is_not_an_inference_setting() -> None:
    offline, online = _config(), _config(local_files_only=False)

    assert offline.fingerprint == online.fingerprint
    assert online.to_dict()["local_files_only"] is False


# --- fake SDK ---------------------------------------------------------------------------------


class FakeCuda:
    def __init__(self, *, available: bool = True, capability: tuple[int, int] = (8, 0)) -> None:
        self.available = available
        self.capability = capability
        self.selected: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def get_device_capability(self, device: str) -> tuple[int, int]:
        return self.capability

    def get_device_name(self, device: str) -> str:
        return "NVIDIA A100-SXM4-80GB"

    def set_device(self, device: str) -> None:
        self.selected.append(device)

    def reset_peak_memory_stats(self, device: str) -> None:
        return None

    def max_memory_allocated(self, device: str) -> int:
        return 7_000_000_000


class FakeTorch(ModuleType):
    float32 = "torch.float32"
    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"

    def __init__(self, cuda: FakeCuda | None = None) -> None:
        super().__init__("torch")
        self.__version__ = "2.8.0+cu128"
        self.version = SimpleNamespace(cuda="12.8")
        self.cuda = cuda or FakeCuda()

    @contextmanager
    def no_grad(self) -> Iterator[None]:
        yield


class FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dtype: str | None = None

    def to(self, dtype: str) -> FakeTensor:
        self.dtype = dtype
        return self


class FakeInputs(dict[str, Any]):
    device: str | None = None

    def to(self, device: str) -> FakeInputs:
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.inputs = FakeInputs(
            pixel_values=FakeTensor("pixel_values"),
            input_ids=FakeTensor("input_ids"),
            attention_mask=FakeTensor("attention_mask"),
            image_grid_hws=FakeTensor("image_grid_hws"),
        )

    def py_apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages = messages
        return "templated"

    def process_vision_info(self, messages: list[dict[str, Any]]) -> tuple[list[Any], None]:
        return [messages[0]["content"][0]["image"]], None

    def __call__(self, **kwargs: Any) -> FakeInputs:
        assert kwargs["text"] == ["templated"]
        assert kwargs["return_tensors"] == "pt"
        return self.inputs


class FakeModel:
    def __init__(self, config: Any, dtype: str, *, answer: str) -> None:
        self.config = config
        self.dtype = dtype
        self.answer = answer
        self.moved_to: str | None = None
        self.evaluating = False
        self.generate_kwargs: dict[str, Any] = {}

    def to(self, device: str) -> FakeModel:
        self.moved_to = device
        return self

    def eval(self) -> FakeModel:
        self.evaluating = True
        return self

    def generate(self, **kwargs: Any) -> tuple[str, list[tuple[str, str]], str]:
        self.generate_kwargs = kwargs
        print(STATS)  # o generate() oficial imprime as estatísticas quando verbose=True
        return self.answer, list(HISTORY), STATS


class FakeTransformers(ModuleType):
    """Records how the runtime loads config, tokenizer, processor and model."""

    def __init__(
        self,
        *,
        answer: str = ANSWER,
        commit: str | None = REVISION,
        vision_attention_after_load: str | None = None,
        dtype_after_load: str | None = None,
    ) -> None:
        super().__init__("transformers")
        self.__version__ = "4.57.1"
        self.answer = answer
        self.commit = commit
        self.vision_attention_after_load = vision_attention_after_load
        self.dtype_after_load = dtype_after_load
        self.processor = FakeProcessor()
        self.model: FakeModel | None = None
        self.calls: dict[str, list[tuple[str, dict[str, Any]]]] = {
            "config": [],
            "tokenizer": [],
            "processor": [],
            "model": [],
        }

    def _config(self, model: str, **kwargs: Any) -> Any:
        self.calls["config"].append((model, kwargs))
        return SimpleNamespace(
            _attn_implementation="magi",
            _commit_hash=self.commit,
            text_config=SimpleNamespace(_attn_implementation=None),
            vision_config=SimpleNamespace(_attn_implementation=None),
        )

    def _tokenizer(self, model: str, **kwargs: Any) -> object:
        self.calls["tokenizer"].append((model, kwargs))
        return SimpleNamespace(name="tokenizer")

    def _processor(self, model: str, **kwargs: Any) -> FakeProcessor:
        self.calls["processor"].append((model, kwargs))
        return self.processor

    def _model(self, model: str, **kwargs: Any) -> FakeModel:
        self.calls["model"].append((model, kwargs))
        config = kwargs["config"]
        if self.vision_attention_after_load is not None:
            # Simula o fallback silencioso do upstream (flash_attention_2 -> sdpa).
            config.vision_config._attn_implementation = self.vision_attention_after_load
        self.model = FakeModel(config, self.dtype_after_load or kwargs["dtype"], answer=self.answer)
        return self.model

    @property
    def AutoConfig(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._config)

    @property
    def AutoTokenizer(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._tokenizer)

    @property
    def AutoProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._processor)

    @property
    def AutoModel(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._model)


class FakeImage:
    def __init__(self, source: io.BytesIO) -> None:
        self.payload = source.read()
        self.mode = "P"

    def convert(self, mode: str) -> FakeImage:
        self.mode = mode
        return self


class FakeBatchUtils(ModuleType):
    def __init__(self, *, stats: object = None, dtype: str = "torch.bfloat16") -> None:
        super().__init__("batch_utils")
        self.stats = {"switch_to_ar": 2, "boxes": 1} if stats is None else stats
        self.dtype = dtype
        self.load_calls = 0
        self.generate_calls: list[tuple[list[tuple[Any, str]], dict[str, Any]]] = []

    def load(self) -> tuple[object, object, object]:
        self.load_calls += 1
        return object(), object(), SimpleNamespace(dtype=self.dtype)

    def generate_batch_hybrid(self, pairs: list[tuple[Any, str]], **kwargs: Any) -> list[str]:
        self.generate_calls.append((pairs, kwargs))
        return [ANSWER for _ in pairs]

    def get_last_hybrid_stats(self) -> object:
        return self.stats


class FakeHub(ModuleType):
    def __init__(self, snapshot: Path) -> None:
        super().__init__("huggingface_hub")
        self.__version__ = "0.35.0"
        self.snapshot = snapshot
        self.calls: list[dict[str, Any]] = []

    def snapshot_download(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return str(self.snapshot)


class Sdk:
    """The fake modules one test installs, and every import the runtime attempted."""

    def __init__(self, modules: dict[str, object]) -> None:
        self.modules = modules
        self.imported: list[str] = []

    def import_module(self, name: str) -> object:
        self.imported.append(name)
        if name not in self.modules:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return self.modules[name]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transformers: FakeTransformers | None = None,
    torch: FakeTorch | None = None,
    extra: dict[str, object] | None = None,
    missing: tuple[str, ...] = (),
) -> Sdk:
    modules: dict[str, object] = {
        "torch": torch or FakeTorch(),
        "transformers": transformers or FakeTransformers(),
        "PIL.Image": SimpleNamespace(open=FakeImage),
        "flash_attn": SimpleNamespace(__version__="2.8.3"),
        "magi_attention": SimpleNamespace(__version__="1.0.5"),
        **(extra or {}),
    }
    for name in missing:
        modules.pop(name, None)
    sdk = Sdk(modules)
    monkeypatch.setattr(
        locateanything, "importlib", SimpleNamespace(import_module=sdk.import_module)
    )
    return sdk


def _image(tmp_path: Path, payload: bytes = PIXELS, *, sha256: str | None = None) -> PreparedImage:
    (tmp_path / "frame-0124.png").write_bytes(payload)
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0124"),
        payload_reference="frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="frame-0124.png",
            sha256=sha256 or hashlib.sha256(payload).hexdigest(),
            media_type="image/png",
        ),
        width=640,
        height=480,
    )


def _runtime(
    tmp_path: Path, config: LocateAnythingConfig, **kwargs: Any
) -> TransformersLocateAnythingRuntime:
    return TransformersLocateAnythingRuntime(config=config, prepared_image_root=tmp_path, **kwargs)


# --- standard runtime ---------------------------------------------------------------------------


def test_constructing_the_runtime_imports_no_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _install(monkeypatch)

    _runtime(tmp_path, _config())

    assert sdk.imported == []


def test_heavy_loading_happens_once_for_every_request_of_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = FakeTransformers()
    _install(monkeypatch, transformers=transformers)
    config = _config()
    runtime = _runtime(tmp_path, config)
    image = _image(tmp_path)

    first = runtime.generate(image=image, prompt="Point to: x.", config=config)
    second = runtime.generate(image=image, prompt="Point to: y.", config=config)

    assert len(transformers.calls["model"]) == 1
    assert len(transformers.calls["processor"]) == 1
    assert dict(first.native_diagnostics).keys() == {"runtime.cold_load_ms"}
    assert second.native_diagnostics == ()


def test_loading_uses_the_pinned_revision_the_remote_code_and_never_downloads_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = FakeTransformers()
    _install(monkeypatch, transformers=transformers)
    runtime = _runtime(tmp_path, _config())

    runtime.load()

    for kind in ("config", "tokenizer", "processor", "model"):
        ((model, kwargs),) = transformers.calls[kind]
        assert model == "nvidia/LocateAnything-3B"
        assert kwargs["revision"] == REVISION
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is True
    assert transformers.calls["model"][0][1]["dtype"] == "torch.bfloat16"
    assert transformers.model is not None
    assert transformers.model.moved_to == "cuda"
    assert transformers.model.evaluating


def test_downloading_is_an_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = FakeTransformers()
    _install(monkeypatch, transformers=transformers)

    _runtime(tmp_path, _config(local_files_only=False)).load()

    assert transformers.calls["model"][0][1]["local_files_only"] is False


def test_the_configured_attention_paths_are_applied_to_the_model_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = FakeTransformers()
    _install(monkeypatch, transformers=transformers)

    _runtime(tmp_path, _config(text_attention="eager", vision_attention="sdpa")).load()

    loaded = transformers.calls["model"][0][1]["config"]
    assert loaded._attn_implementation == "eager"
    assert loaded.text_config._attn_implementation == "eager"
    assert loaded.vision_config._attn_implementation == "sdpa"


def test_a_silent_upstream_attention_fallback_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, transformers=FakeTransformers(vision_attention_after_load="sdpa"))

    with pytest.raises(LocateAnythingModelLoadError, match="flash_attention_2"):
        _runtime(tmp_path, _config()).load()


def test_a_missing_flash_attention_package_fails_before_the_model_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = FakeTransformers()
    _install(monkeypatch, transformers=transformers, missing=("flash_attn",))

    with pytest.raises(LocateAnythingDependencyError, match="flash_attn"):
        _runtime(tmp_path, _config()).load()
    assert transformers.calls["model"] == []


def test_magi_attention_requires_its_package_and_a_hopper_or_newer_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(text_attention="magi")
    _install(monkeypatch, missing=("magi_attention",))
    with pytest.raises(LocateAnythingDependencyError, match="magi_attention"):
        _runtime(tmp_path, config).load()

    _install(monkeypatch, torch=FakeTorch(FakeCuda(capability=(8, 0))))
    with pytest.raises(LocateAnythingDeviceError, match="Hopper"):
        _runtime(tmp_path, config).load()

    _install(monkeypatch, torch=FakeTorch(FakeCuda(capability=(9, 0))))
    _runtime(tmp_path, config).load()


def test_an_unavailable_cuda_device_fails_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, torch=FakeTorch(FakeCuda(available=False)))

    with pytest.raises(LocateAnythingDeviceError, match="CUDA"):
        _runtime(tmp_path, _config()).load()


def test_missing_sdks_fail_with_a_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, missing=("transformers",))

    with pytest.raises(LocateAnythingDependencyError, match="transformers"):
        _runtime(tmp_path, _config()).load()


def test_a_substituted_checkpoint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, transformers=FakeTransformers(commit="f" * 40))

    with pytest.raises(LocateAnythingModelLoadError, match="commit"):
        _runtime(tmp_path, _config()).load()


def test_a_model_loaded_at_another_dtype_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, transformers=FakeTransformers(dtype_after_load="torch.float32"))

    with pytest.raises(LocateAnythingModelLoadError, match="dtype"):
        _runtime(tmp_path, _config()).load()


def test_generation_follows_the_upstream_worker_call_with_the_configured_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transformers = FakeTransformers()
    _install(monkeypatch, transformers=transformers)
    config = _config(
        generation_mode=LocateAnythingGenerationMode.SLOW,
        temperature=0.0,
        top_p=0.8,
        top_k=0,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )

    generation = _runtime(tmp_path, config).generate(
        image=_image(tmp_path), prompt="Point to: the door.", config=config
    )

    assert transformers.model is not None
    kwargs = transformers.model.generate_kwargs
    assert kwargs["generation_mode"] == "slow"
    assert kwargs["max_new_tokens"] == 512
    assert kwargs["temperature"] == 0.0
    assert kwargs["do_sample"] is False
    assert kwargs["top_p"] == 0.8
    assert kwargs["top_k"] is None
    assert kwargs["repetition_penalty"] == 1.05
    assert kwargs["use_cache"] is True
    assert kwargs["verbose"] is True
    assert kwargs["pixel_values"].dtype == "torch.bfloat16"
    assert transformers.processor.inputs.device == "cuda"
    message = transformers.processor.messages[0]["content"]
    assert message[1] == {"type": "text", "text": "Point to: the door."}
    assert message[0]["image"].payload == PIXELS
    assert message[0]["image"].mode == "RGB"
    assert generation.text == ANSWER
    assert generation.decode_steps == tuple(HISTORY)
    assert generation.statistics == STATS
    assert generation.peak_memory_bytes == 7_000_000_000
    assert capsys.readouterr().out == ""


def test_the_runtime_identity_records_library_and_device_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    config = _config()

    generation = _runtime(tmp_path, config).generate(
        image=_image(tmp_path), prompt="p", config=config
    )

    identity = dict(generation.runtime_identity)
    assert identity["torch"] == "2.8.0+cu128"
    assert identity["torch.cuda"] == "12.8"
    assert identity["transformers"] == "4.57.1"
    assert identity["flash_attn"] == "2.8.3"
    assert identity["device_name"] == "NVIDIA A100-SXM4-80GB"
    assert identity["model_commit"] == REVISION
    assert identity["runtime"] == "standard"
    assert "python" in identity


def test_an_image_that_changed_since_the_request_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    config = _config()
    image = _image(tmp_path, sha256="e" * 64)

    with pytest.raises(LocateAnythingInferenceError, match="sha256"):
        _runtime(tmp_path, config).generate(image=image, prompt="p", config=config)


def test_an_image_reference_outside_the_prepared_image_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    config = _config()
    image = _image(tmp_path)
    escaping = PreparedImage(
        source_observation_id=image.source_observation_id,
        payload_reference="../frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="../frame-0124.png",
            sha256=hashlib.sha256(PIXELS).hexdigest(),
            media_type="image/png",
        ),
        width=640,
        height=480,
    )

    with pytest.raises(LocateAnythingInferenceError, match="outside"):
        _runtime(tmp_path, config).generate(image=escaping, prompt="p", config=config)


def test_the_runtime_refuses_a_configuration_it_was_not_built_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)

    with pytest.raises(ValueError, match="configuration"):
        _runtime(tmp_path, _config()).generate(
            image=_image(tmp_path), prompt="p", config=_config(temperature=0.5)
        )


# --- batch runtime -----------------------------------------------------------------------------


def test_the_batch_runtime_loads_the_pinned_snapshot_strictly_and_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    batch_utils, hub = FakeBatchUtils(), FakeHub(snapshot)
    _install(monkeypatch, extra={"batch_utils": batch_utils, "huggingface_hub": hub})
    monkeypatch.setattr(sys, "path", list(sys.path))
    environ: dict[str, str] = {"LA_FLASH_STRICT_ATTN": "0"}
    config = _batch_config()
    runtime = _runtime(tmp_path, config, environ=environ)
    image = _image(tmp_path)

    first = runtime.generate(image=image, prompt="Point to: x.", config=config)
    runtime.generate(image=image, prompt="Point to: y.", config=config)

    assert batch_utils.load_calls == 1
    assert hub.calls == [
        {"repo_id": "nvidia/LocateAnything-3B", "revision": REVISION, "local_files_only": True}
    ]
    assert sys.path[0] == str(snapshot)
    assert environ == {
        "LA_FLASH_MODEL": str(snapshot),
        "LA_FLASH_ATTN": "la_flash",
        "LA_FLASH_VISION_ATTN": "flash_attention_2",
        "LA_FLASH_HYBRID_SCHEDULER": "pipeline",
        "LA_FLASH_HYBRID_GROUP_SIZE": "0",
        "LA_FLASH_STRICT_ATTN": "1",
    }
    pairs, kwargs = batch_utils.generate_calls[0]
    assert [prompt for _, prompt in pairs] == ["Point to: x."]
    assert kwargs == {
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": None,
        "repetition_penalty": 1.1,
        "max_new_tokens": 2048,
        "scheduler": "pipeline",
        "group_size": 0,
    }
    assert first.text == ANSWER
    assert first.statistics == '{"boxes": 1, "switch_to_ar": 2}'
    assert first.decode_steps == ()
    assert dict(first.runtime_identity)["runtime"] == "batch"


def test_the_batch_runtime_refuses_a_snapshot_of_another_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / ("f" * 40)
    snapshot.mkdir(parents=True)
    batch_utils = FakeBatchUtils()
    _install(monkeypatch, extra={"batch_utils": batch_utils, "huggingface_hub": FakeHub(snapshot)})
    monkeypatch.setattr(sys, "path", list(sys.path))

    with pytest.raises(LocateAnythingModelLoadError, match="revision"):
        _runtime(tmp_path, _batch_config(), environ={}).load()
    assert batch_utils.load_calls == 0


def test_the_batch_runtime_requires_the_release_batch_utilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    _install(monkeypatch, extra={"huggingface_hub": FakeHub(snapshot)})
    monkeypatch.setattr(sys, "path", list(sys.path))

    with pytest.raises(LocateAnythingDependencyError, match="batch_utils"):
        _runtime(tmp_path, _batch_config(), environ={}).load()
