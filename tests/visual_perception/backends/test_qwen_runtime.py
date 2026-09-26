"""Contract tests for the transformers-backed Qwen runtime, using fake SDK modules.

These tests are fake/contract tests: they prove how the runtime maps the canonical
configuration onto the transformers/bitsandbytes calls, not the quality of any real
Qwen checkpoint. Real-model execution is recorded separately.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    PerceptionResultId,
    RegionId,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.visual_perception.backends import qwen
from contextmap.visual_perception.backends.qwen import (
    HuggingFaceQwenRuntime,
    QwenDependencyError,
    QwenDeviceError,
    QwenInferenceError,
    QwenModelLoadError,
    QwenOutOfMemoryError,
    QwenSemanticConfig,
    QwenSemanticInterpreter,
)
from contextmap.visual_perception.semantic_backend import SemanticVisualInputMeasurement

REVISION = "0123456789abcdef0123456789abcdef01234567"
VIEW_REFERENCE = "outputs/semantic-views/region-0007.png"
VIEW_PAYLOAD = b"png bytes are never decoded by the fake SDK"


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


class FakeGrid:
    """Stands in for the ``image_grid_thw`` tensor: one ``(t, h, w)`` patch grid per image."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[int]]:
        return [list(row) for row in self.rows]


# O default do preprocessor_config.json do Qwen3-VL-4B-Instruct: 256 a 16384 tokens visuais.
CHECKPOINT_SIZE = {"shortest_edge": 65_536, "longest_edge": 16_777_216}
DEFAULT_GRID = [1, 16, 20]
"""A 256x320 px image in 16 px patches: 320 patches, merged 2x2 into 80 visual tokens."""


class FakeImageProcessor:
    """The Qwen2-VL image processor surface the runtime reads: budget and patch geometry."""

    def __init__(self) -> None:
        self.size: dict[str, int] = dict(CHECKPOINT_SIZE)
        self.patch_size = 16
        self.merge_size = 2


class FakeProcessor:
    def __init__(self, *, prompt_tokens: int = 7, grid_rows: list[list[int]] | None = None) -> None:
        self.prompt_tokens = prompt_tokens
        self.grid_rows = grid_rows
        self.image_processor = FakeImageProcessor()
        self.messages: list[dict[str, Any]] = []
        self.template_kwargs: dict[str, Any] = {}
        self.inputs: FakeInputs | None = None
        self.decoded_text = '{"abstained": true, "claims": [], "scene_context": null}'

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> FakeInputs:
        self.messages = messages
        self.template_kwargs = kwargs
        images = [part for part in messages[0]["content"] if part["type"] == "image"]
        rows = self.grid_rows if self.grid_rows is not None else [DEFAULT_GRID for _ in images]
        self.inputs = FakeInputs(
            input_ids=FakeTokens([[1] * self.prompt_tokens]), image_grid_thw=FakeGrid(rows)
        )
        return self.inputs

    def batch_decode(self, tokens: FakeTokens, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens
        return [self.decoded_text]


class FakeModel:
    def __init__(
        self,
        *,
        quantization_config: object | None,
        new_tokens: int = 5,
        generate_error: Exception | None = None,
    ) -> None:
        # Um PretrainedConfig sem quantização não possui o atributo, em vez de guardar None.
        self.config = SimpleNamespace(
            **({} if quantization_config is None else {"quantization_config": quantization_config})
        )
        self.new_tokens = new_tokens
        self.generate_error = generate_error
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
        if self.generate_error is not None:
            raise self.generate_error
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
        generate_error: Exception | None = None,
        processor: FakeProcessor | None = None,
        apply_budget: bool = True,
    ) -> None:
        super().__init__("transformers")
        self.report_quantization = report_quantization
        self.model_error = model_error
        self.generate_error = generate_error
        self.apply_budget = apply_budget
        self.processor = processor or FakeProcessor()
        self.model: FakeModel | None = None
        self.new_tokens = new_tokens
        self.processor_kwargs: dict[str, Any] = {}
        self.model_kwargs: dict[str, Any] = {}
        self.opened_payloads: list[bytes] = []
        self.BitsAndBytesConfig = FakeBitsAndBytesConfig

    def _open_image(self, source: io.BytesIO) -> FakeImage:
        image = FakeImage(source)
        self.opened_payloads.append(image.payload)
        return image

    def _load_processor(self, model: str, **kwargs: Any) -> FakeProcessor:
        self.processor_kwargs = {"model": model, **kwargs}
        if self.apply_budget:
            # Como o Qwen2VLImageProcessor do transformers 5.x: min/max_pixels viram ``size``.
            size = self.processor.image_processor.size
            if "min_pixels" in kwargs:
                size["shortest_edge"] = kwargs["min_pixels"]
            if "max_pixels" in kwargs:
                size["longest_edge"] = kwargs["max_pixels"]
        return self.processor

    def _load_model(self, model: str, **kwargs: Any) -> FakeModel:
        self.model_kwargs = {"model": model, **kwargs}
        if self.model_error is not None:
            raise self.model_error
        applied = kwargs.get("quantization_config") if self.report_quantization else None
        self.model = FakeModel(
            quantization_config=applied,
            new_tokens=self.new_tokens,
            generate_error=self.generate_error,
        )
        return self.model

    @property
    def AutoProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_processor)

    @property
    def AutoModelForImageTextToText(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_model)


class FakeOutOfMemoryError(RuntimeError):
    """Stands in for ``torch.cuda.OutOfMemoryError``."""


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

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
    """Records the exact bytes the runtime decoded, since the SDK is never really invoked."""

    def __init__(self, source: io.BytesIO) -> None:
        self.payload = source.read()

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
        "PIL.Image": SimpleNamespace(open=transformers._open_image),
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
    path.write_bytes(VIEW_PAYLOAD)
    return path


def _visual_view(
    payload: bytes = VIEW_PAYLOAD, *, reference: str = VIEW_REFERENCE
) -> SemanticVisualView:
    """Describe a canonical view whose ``sha256`` identifies exactly ``payload``."""
    return SemanticVisualView(
        view_id="tight-crop",
        kind=VisualViewKind.TIGHT_CROP,
        payload_reference=reference,
        source_observation_id=SourceObservationId("frame-0124"),
        region_id=RegionId("region-0007"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _generate(runtime: HuggingFaceQwenRuntime, config: QwenSemanticConfig) -> Any:
    return runtime.generate(
        visual_views=(_visual_view(),), prompt="canonical prompt", config=config
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
    _view(tmp_path)
    config = _config()

    response = _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    (message,) = transformers.processor.messages
    assert message["role"] == "user"
    assert [part["type"] for part in message["content"]] == ["image", "text"]
    # A imagem é decodificada dos bytes verificados, não relida do caminho.
    assert message["content"][0]["image"].payload == VIEW_PAYLOAD
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


def test_several_views_reach_the_model_in_the_declared_order_before_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#524: a multi-view request is one message whose images keep the policy order."""
    transformers, _ = _install(monkeypatch)
    payloads = {"masked": b"masked subject", "tight": b"tight crop", "context": b"context crop"}
    views = []
    for name, payload in payloads.items():
        reference = f"outputs/semantic-views/{name}.png"
        (tmp_path / reference).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / reference).write_bytes(payload)
        views.append(_visual_view(payload, reference=reference))
    config = _config()

    HuggingFaceQwenRuntime(config=config, view_root=tmp_path).generate(
        visual_views=tuple(views), prompt="canonical prompt", config=config
    )

    (message,) = transformers.processor.messages
    assert [part["type"] for part in message["content"]] == ["image", "image", "image", "text"]
    assert [part["image"].payload for part in message["content"][:3]] == list(payloads.values())


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


def test_view_references_cannot_escape_the_view_root_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    root = tmp_path / "root"
    (root / "outputs" / "semantic-views").mkdir(parents=True)
    (tmp_path / "secret.png").write_bytes(b"outside")
    (root / VIEW_REFERENCE).symlink_to(tmp_path / "secret.png")
    config = _config()
    runtime = HuggingFaceQwenRuntime(config=config, view_root=root)

    with pytest.raises(QwenInferenceError, match="escapes"):
        runtime.generate(visual_views=(_visual_view(b"outside"),), prompt="prompt", config=config)


def test_missing_view_payload_is_an_inference_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    config = _config()
    runtime = HuggingFaceQwenRuntime(config=config, view_root=tmp_path)

    with pytest.raises(QwenInferenceError, match="does not exist"):
        _generate(runtime, config)


def test_a_view_whose_bytes_diverge_from_the_request_sha256_is_never_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()
    adapter = QwenSemanticInterpreter(
        config=config, runtime=HuggingFaceQwenRuntime(config=config, view_root=tmp_path)
    )
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("request-0007"),
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        mode=SemanticInterpretationMode.REGION,
        region_id=RegionId("region-0007"),
        visual_views=(
            SemanticVisualView(
                view_id="tight-crop",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference=VIEW_REFERENCE,
                source_observation_id=SourceObservationId("frame-0124"),
                region_id=RegionId("region-0007"),
                sha256=hashlib.sha256(b"the bytes the request identifies").hexdigest(),
            ),
        ),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )

    with pytest.raises(QwenInferenceError, match="sha256"):
        adapter.interpret(request)

    assert transformers.opened_payloads == []
    assert transformers.processor.messages == []
    assert transformers.model is None or transformers.model.generate_kwargs == {}


def test_one_diverging_view_among_several_prevents_the_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    second_reference = "outputs/semantic-views/region-0008.png"
    (tmp_path / second_reference).write_bytes(b"changed after the request was built")
    config = _config()
    runtime = HuggingFaceQwenRuntime(config=config, view_root=tmp_path)
    views = (
        _visual_view(),
        _visual_view(b"what the request recorded", reference=second_reference),
    )

    with pytest.raises(QwenInferenceError, match=r"region-0008.*sha256"):
        runtime.generate(visual_views=views, prompt="prompt", config=config)

    # A primeira view estava íntegra e foi aberta, mas nenhuma inferência aconteceu.
    assert transformers.opened_payloads == [VIEW_PAYLOAD]
    assert transformers.processor.messages == []
    assert transformers.model is not None
    assert transformers.model.generate_kwargs == {}


# --- visual input budget (#526) ------------------------------------------------------

BUDGET = {"min_pixels": 256 * 32 * 32, "max_pixels": 1280 * 32 * 32}


def _second_view(tmp_path: Path) -> SemanticVisualView:
    reference = "outputs/semantic-views/region-0007-context.png"
    payload = b"contextual crop bytes"
    (tmp_path / reference).write_bytes(payload)
    return SemanticVisualView(
        view_id="contextual-crop",
        kind=VisualViewKind.CONTEXTUAL_CROP,
        payload_reference=reference,
        source_observation_id=SourceObservationId("frame-0124"),
        region_id=RegionId("region-0007"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_an_unset_budget_constructs_the_processor_exactly_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backward-identical default: the checkpoint's own processor budget, nothing added."""
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert transformers.processor_kwargs == {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": REVISION,
        "local_files_only": True,
    }
    assert transformers.processor.image_processor.size == CHECKPOINT_SIZE


def test_a_configured_budget_is_applied_when_the_processor_is_constructed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(**BUDGET)

    _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert transformers.processor_kwargs["min_pixels"] == 262_144
    assert transformers.processor_kwargs["max_pixels"] == 1_310_720
    assert transformers.processor.image_processor.size == {
        "shortest_edge": 262_144,
        "longest_edge": 1_310_720,
    }


def test_a_processor_that_ignores_the_budget_fails_before_any_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Like quantization: a configuration that names a budget never runs under another one."""
    transformers, _ = _install(monkeypatch, transformers=FakeTransformers(apply_budget=False))
    _view(tmp_path)
    config = _config(**BUDGET)

    with pytest.raises(QwenModelLoadError, match=r"visual input budget.*not applied"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert transformers.model is None, "the weights are never loaded for an unapplied budget"


def test_a_budget_below_one_merged_patch_cannot_be_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every edge is at least patch_size x merge_size, so 32x32 px is the smallest image."""
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(min_pixels=256, max_pixels=32 * 32 - 1)

    with pytest.raises(QwenModelLoadError, match=r"max_pixels=1023.*1024"):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert transformers.model is None, "the weights are never loaded for an unenforceable budget"


def test_each_view_records_the_size_and_visual_tokens_the_processor_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = FakeProcessor(grid_rows=[[1, 16, 20], [1, 28, 14]])
    _install(monkeypatch, transformers=FakeTransformers(processor=processor))
    _view(tmp_path)
    config = _config(**BUDGET)
    views = (_visual_view(), _second_view(tmp_path))

    response = HuggingFaceQwenRuntime(config=config, view_root=tmp_path).generate(
        visual_views=views, prompt="canonical prompt", config=config
    )

    assert response.visual_inputs == (
        SemanticVisualInputMeasurement(
            view_id="tight-crop", height_px=256, width_px=320, visual_tokens=80
        ),
        SemanticVisualInputMeasurement(
            view_id="contextual-crop", height_px=448, width_px=224, visual_tokens=98
        ),
    )


def test_a_processor_output_that_does_not_account_for_every_view_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No image is silently dropped or added between the request and the model."""
    processor = FakeProcessor(grid_rows=[[1, 16, 20]])
    transformers, _ = _install(monkeypatch, transformers=FakeTransformers(processor=processor))
    _view(tmp_path)
    config = _config()
    views = (_visual_view(), _second_view(tmp_path))

    with pytest.raises(QwenInferenceError, match=r"1 image grid.*2 view"):
        HuggingFaceQwenRuntime(config=config, view_root=tmp_path).generate(
            visual_views=views, prompt="canonical prompt", config=config
        )

    assert transformers.model is not None
    assert transformers.model.generate_kwargs == {}


def test_running_out_of_device_memory_is_attributed_to_the_recorded_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        transformers=FakeTransformers(generate_error=FakeOutOfMemoryError("CUDA out of memory")),
    )
    _view(tmp_path)
    config = _config(**BUDGET)

    with pytest.raises(QwenOutOfMemoryError) as raised:
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    message = str(raised.value)
    assert isinstance(raised.value, QwenInferenceError)
    assert "out of device memory on cuda" in message
    assert "1 view" in message
    assert "configured visual input budget min_pixels=262144, max_pixels=1310720" in message
    assert "tight-crop: 256x320 px (HxW), 80 visual tokens" in message
    assert "input_tokens=7" in message
    assert "peak_memory_bytes=4000000000" in message
    assert "CUDA out of memory" in message


def test_without_a_budget_an_out_of_memory_failure_names_the_checkpoint_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The processor default in effect is recorded, never left as an unknown."""
    _install(
        monkeypatch,
        transformers=FakeTransformers(generate_error=FakeOutOfMemoryError("CUDA out of memory")),
    )
    _view(tmp_path)
    config = _config()

    with pytest.raises(
        QwenOutOfMemoryError,
        match=r"checkpoint processor default visual input "
        r"budget min_pixels=65536, max_pixels=16777216",
    ):
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)


def test_other_generation_failures_are_not_reported_as_out_of_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, transformers=FakeTransformers(generate_error=RuntimeError("boom")))
    _view(tmp_path)
    config = _config()

    with pytest.raises(QwenInferenceError, match="boom") as raised:
        _generate(HuggingFaceQwenRuntime(config=config, view_root=tmp_path), config)

    assert not isinstance(raised.value, QwenOutOfMemoryError)
