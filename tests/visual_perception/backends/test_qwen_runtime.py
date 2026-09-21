"""Contract tests for the transformers-backed Qwen runtime, using fake SDK modules.

These tests are fake/contract tests: they prove how the runtime maps the canonical
configuration onto the transformers/bitsandbytes calls, not the quality of any real
Qwen checkpoint. Real-model execution is recorded separately.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from contextmap.visual_perception.backends import qwen
from contextmap.visual_perception.backends.qwen import (
    HuggingFaceQwenRuntime,
    QwenDependencyError,
    QwenDeviceError,
    QwenInferenceError,
    QwenModelLoadError,
    QwenSemanticConfig,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
VIEW_REFERENCE = "outputs/semantic-views/region-0007.png"


class FakeTokens:
    """Minimal 2-D token tensor supporting ``tokens[:, start:]`` and ``.shape``."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, key: tuple[slice, slice]) -> FakeTokens:
        _, columns = key
        return FakeTokens([row[columns] for row in self.rows])


class FakeInputs(dict[str, Any]):
    """Model inputs that record the device they were moved to."""

    device: str | None = None

    def to(self, device: str) -> FakeInputs:
        self.device = device
        return self


class FakeProcessor:
    def __init__(self, *, prompt_tokens: int = 7) -> None:
        self.prompt_tokens = prompt_tokens
        self.messages: list[dict[str, Any]] = []
        self.template_kwargs: dict[str, Any] = {}
        self.inputs: FakeInputs | None = None
        self.decoded_text = '{"abstained": true, "claims": [], "scene_context": null}'

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> FakeInputs:
        self.messages = messages
        self.template_kwargs = kwargs
        self.inputs = FakeInputs(input_ids=FakeTokens([[1] * self.prompt_tokens]))
        return self.inputs

    def batch_decode(self, tokens: FakeTokens, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens
        return [self.decoded_text]


class FakeModel:
    def __init__(self, *, quantization_config: object | None, new_tokens: int = 5) -> None:
        # Um PretrainedConfig sem quantização não possui o atributo, em vez de guardar None.
        self.config = SimpleNamespace(
            **({} if quantization_config is None else {"quantization_config": quantization_config})
        )
        self.new_tokens = new_tokens
        self.moved_to: str | None = None
        self.evaluating = False
        self.generate_kwargs: dict[str, Any] = {}

    def to(self, device: str) -> FakeModel:
        self.moved_to = device
        return self

    def eval(self) -> FakeModel:
        self.evaluating = True
        return self

    def generate(self, **kwargs: Any) -> FakeTokens:
        self.generate_kwargs = kwargs
        prompt = kwargs["input_ids"].rows[0]
        return FakeTokens([[*prompt, *([9] * self.new_tokens)]])


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeTransformers(ModuleType):
    """Records how the runtime loads the processor and the model."""

    def __init__(
        self,
        *,
        report_quantization: bool = True,
        new_tokens: int = 5,
        model_error: Exception | None = None,
    ) -> None:
        super().__init__("transformers")
        self.report_quantization = report_quantization
        self.model_error = model_error
        self.processor = FakeProcessor()
        self.model: FakeModel | None = None
        self.new_tokens = new_tokens
        self.processor_kwargs: dict[str, Any] = {}
        self.model_kwargs: dict[str, Any] = {}
        self.BitsAndBytesConfig = FakeBitsAndBytesConfig

    def _load_processor(self, model: str, **kwargs: Any) -> FakeProcessor:
        self.processor_kwargs = {"model": model, **kwargs}
        return self.processor

    def _load_model(self, model: str, **kwargs: Any) -> FakeModel:
        self.model_kwargs = {"model": model, **kwargs}
        if self.model_error is not None:
            raise self.model_error
        applied = kwargs.get("quantization_config") if self.report_quantization else None
        self.model = FakeModel(quantization_config=applied, new_tokens=self.new_tokens)
        return self.model

    @property
    def AutoProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_processor)

    @property
    def AutoModelForImageTextToText(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_model)


class FakeCuda:
    def __init__(self, *, available: bool = True, peak: int = 4_000_000_000) -> None:
        self.available = available
        self.peak = peak
        self.reset_devices: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def reset_peak_memory_stats(self, device: str) -> None:
        self.reset_devices.append(device)

    def max_memory_allocated(self, device: str) -> int:
        return self.peak


class FakeTorch(ModuleType):
    float32 = "torch.float32"
    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"

    def __init__(self, *, cuda: FakeCuda | None = None) -> None:
        super().__init__("torch")
        self.cuda = cuda or FakeCuda()

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        yield


class FakeImage:
    def __init__(self, path: Path) -> None:
        self.path = path

    def convert(self, mode: str) -> FakeImage:
        assert mode == "RGB"
        return self

    def __enter__(self) -> FakeImage:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _config(**overrides: Any) -> QwenSemanticConfig:
    values: dict[str, Any] = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": REVISION,
        "device": "cuda",
        "precision": "bfloat16",
        "max_new_tokens": 64,
        "temperature": 0.0,
    }
    values.update(overrides)
    return QwenSemanticConfig(**values)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transformers: FakeTransformers | None = None,
    torch: FakeTorch | None = None,
    missing: str | None = None,
) -> tuple[FakeTransformers, FakeTorch]:
    transformers = transformers or FakeTransformers()
    torch = torch or FakeTorch()
    sdks = {
        "torch": torch,
        "transformers": transformers,
        "PIL.Image": SimpleNamespace(open=lambda path: FakeImage(Path(path))),
    }

    def import_module(name: str) -> object:
        if name == missing:
            raise ModuleNotFoundError(name)
        return sdks[name]

    monkeypatch.setattr(qwen, "importlib", SimpleNamespace(import_module=import_module))
    return transformers, torch


def _view(tmp_path: Path) -> Path:
    path = tmp_path / VIEW_REFERENCE
    path.parent.mkdir(parents=True)
    path.write_bytes(b"png bytes are never decoded by the fake SDK")
    return path


def _generate(runtime: HuggingFaceQwenRuntime, config: QwenSemanticConfig) -> Any:
    return runtime.generate(
        visual_payload_references=(VIEW_REFERENCE,), prompt="canonical prompt", config=config
    )


def test_four_bit_quantization_is_applied_as_nf4_with_the_configured_compute_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(quantization="4bit")

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    quantization = transformers.model_kwargs["quantization_config"]
    assert isinstance(quantization, FakeBitsAndBytesConfig)
    assert quantization.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "torch.bfloat16",
    }
    assert transformers.model_kwargs["device_map"] == {"": "cuda"}
    assert transformers.model is not None
    assert transformers.model.moved_to is None


def test_eight_bit_quantization_is_applied_through_bitsandbytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(quantization="8bit")

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    quantization = transformers.model_kwargs["quantization_config"]
    assert quantization.kwargs == {"load_in_8bit": True}


def test_without_quantization_the_model_is_loaded_at_precision_and_moved_to_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert "quantization_config" not in transformers.model_kwargs
    assert "device_map" not in transformers.model_kwargs
    assert transformers.model_kwargs["dtype"] == "torch.bfloat16"
    assert transformers.model is not None
    assert transformers.model.moved_to == "cuda"
    assert transformers.model.evaluating


def test_loading_uses_the_pinned_revision_and_never_downloads_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    for kwargs in (transformers.processor_kwargs, transformers.model_kwargs):
        assert kwargs["model"] == "Qwen/Qwen3-VL-4B-Instruct"
        assert kwargs["revision"] == REVISION
        assert kwargs["local_files_only"] is True


def test_requested_quantization_that_the_loaded_model_does_not_report_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, transformers=FakeTransformers(report_quantization=False))
    _view(tmp_path)
    config = _config(quantization="4bit")

    with pytest.raises(QwenModelLoadError, match=r"quantization.*not applied"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)


def test_quantization_without_a_cuda_device_is_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(device="cpu", precision="float32", quantization="8bit")

    with pytest.raises(QwenDeviceError, match="bitsandbytes"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert transformers.model is None


def test_unavailable_cuda_is_an_explicit_device_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, torch=FakeTorch(cuda=FakeCuda(available=False)))
    _view(tmp_path)
    config = _config()

    with pytest.raises(QwenDeviceError, match="CUDA"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)


def test_unsupported_precision_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    _view(tmp_path)
    config = _config(precision="int4")

    with pytest.raises(QwenDeviceError, match="precision"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)


def test_missing_sdk_is_a_dependency_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, missing="transformers")
    _view(tmp_path)
    config = _config()

    with pytest.raises(QwenDependencyError, match="transformers"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)


def test_unloadable_checkpoint_is_a_model_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        transformers=FakeTransformers(model_error=OSError("checkpoint not in local cache")),
    )
    _view(tmp_path)
    config = _config()

    with pytest.raises(QwenModelLoadError, match="could not load"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)


def test_a_missing_package_during_loading_is_a_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        transformers=FakeTransformers(model_error=ImportError("requires the latest bitsandbytes")),
    )
    _view(tmp_path)
    config = _config(quantization="4bit")

    with pytest.raises(QwenDependencyError, match="bitsandbytes"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)


def test_the_runtime_requires_a_pinned_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        HuggingFaceQwenRuntime(config=_config(revision=None), view_root=Path("."))


def test_generation_sends_every_view_then_the_canonical_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    path = _view(tmp_path)
    config = _config()

    response = _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    (message,) = transformers.processor.messages
    assert message["role"] == "user"
    assert [part["type"] for part in message["content"]] == ["image", "text"]
    assert message["content"][0]["image"].path == path
    assert message["content"][1]["text"] == "canonical prompt"
    assert transformers.processor.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert transformers.processor.inputs is not None
    assert transformers.processor.inputs.device == "cuda"
    assert response.text == transformers.processor.decoded_text


def test_zero_temperature_is_greedy_and_does_not_pass_sampling_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(max_new_tokens=96)

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert transformers.model is not None
    kwargs = transformers.model.generate_kwargs
    assert kwargs["max_new_tokens"] == 96
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] is None
    assert kwargs["top_p"] is None
    assert kwargs["top_k"] is None


def test_positive_temperature_samples_with_that_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(temperature=0.4)

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert transformers.model is not None
    assert transformers.model.generate_kwargs["do_sample"] is True
    assert transformers.model.generate_kwargs["temperature"] == 0.4


def test_response_reports_token_counts_and_process_peak_gpu_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, torch = _install(monkeypatch)
    _view(tmp_path)
    config = _config()

    response = _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert response.input_tokens == 7
    assert response.output_tokens == 5
    assert response.peak_memory_bytes == 4_000_000_000
    assert torch.cuda.reset_devices == ["cuda"]
    assert response.warnings == ()


def test_cpu_generation_reports_no_gpu_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    _view(tmp_path)
    config = _config(device="cpu", precision="float32")

    response = _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert response.peak_memory_bytes is None


def test_hitting_the_token_limit_is_reported_because_the_json_is_probably_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, transformers=FakeTransformers(new_tokens=64))
    _view(tmp_path)
    config = _config(max_new_tokens=64)

    response = _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert len(response.warnings) == 1
    assert "max_new_tokens=64" in response.warnings[0]


def test_model_is_loaded_once_and_reused_across_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()
    runtime = HuggingFaceQwenRuntime(config=config, view_root=tmp_path)

    _generate(runtime, config)
    first = transformers.model
    _generate(runtime, config)

    assert transformers.model is first


def test_generation_for_another_effective_configuration_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    _view(tmp_path)
    runtime = HuggingFaceQwenRuntime(config=_config(), view_root=tmp_path)

    with pytest.raises(ValueError, match="another configuration"):
        _generate(runtime, _config(quantization="4bit"))


def test_view_references_cannot_escape_the_view_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.png").write_bytes(b"outside")
    config = _config()
    runtime = HuggingFaceQwenRuntime(config=config, view_root=root)

    with pytest.raises(QwenInferenceError, match="escapes"):
        runtime.generate(
            visual_payload_references=("../secret.png",), prompt="prompt", config=config
        )


def test_missing_view_payload_is_an_inference_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    config = _config()
    runtime = HuggingFaceQwenRuntime(config=config, view_root=tmp_path)

    with pytest.raises(QwenInferenceError, match="does not exist"):
        _generate(runtime, config)
