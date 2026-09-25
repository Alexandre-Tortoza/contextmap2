"""Contract tests for the transformers-backed Florence-2 semantic runtime (fake SDK modules).

These are fake/contract tests: they prove how the runtime maps the configuration and the
task prompt onto the transformers calls, not the quality of the Florence-2 checkpoint.
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
from contextmap.visual_perception.backends import florence2_semantic
from contextmap.visual_perception.backends.florence2_semantic import (
    Florence2SemanticConfig,
    Florence2SemanticError,
    Florence2SemanticInterpreter,
    HuggingFaceFlorence2SemanticRuntime,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
VIEW_REFERENCE = "outputs/semantic-views/region-0007.png"
VIEW_PAYLOAD = b"png bytes are never decoded by the fake SDK"
TASK_PROMPT = "<REGION_TO_CATEGORY><loc_0><loc_0><loc_999><loc_999>"


class FakeTokens:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]))


class FakeInputs(dict[str, Any]):
    """Model inputs that record where and at which dtype they were moved."""

    moved: tuple[str, object] | None = None

    def to(self, device: str, dtype: object = None) -> FakeInputs:
        self.moved = (device, dtype)
        return self


class FakeProcessor:
    def __init__(self, *, parsed_text: str = "car<loc_0><loc_0><loc_999><loc_999>") -> None:
        self.parsed_text = parsed_text
        self.call: dict[str, Any] = {}
        self.inputs: FakeInputs | None = None
        self.post_process_call: dict[str, Any] = {}
        self.decode_skip_special_tokens: bool | None = None

    def __call__(self, *, text: str, images: object, return_tensors: str) -> FakeInputs:
        self.call = {"text": text, "images": images, "return_tensors": return_tensors}
        self.inputs = FakeInputs(input_ids=FakeTokens([[1] * 11]))
        return self.inputs

    def batch_decode(self, tokens: FakeTokens, *, skip_special_tokens: bool) -> list[str]:
        self.decode_skip_special_tokens = skip_special_tokens
        return ["</s><s>car<loc_0><loc_0><loc_999><loc_999></s>"]

    def post_process_generation(
        self, text: str, *, task: str, image_size: tuple[int, int]
    ) -> dict[str, str]:
        self.post_process_call = {"text": text, "task": task, "image_size": image_size}
        return {task: self.parsed_text}


class FakeModel:
    def __init__(self) -> None:
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
        return FakeTokens([[2, 0, 5, 6, 7, 2]])


class FakeTransformers(ModuleType):
    def __init__(
        self,
        *,
        parsed_text: str = "car<loc_0><loc_0><loc_999><loc_999>",
        error: Exception | None = None,
    ) -> None:
        super().__init__("transformers")
        self.processor = FakeProcessor(parsed_text=parsed_text)
        self.model = FakeModel()
        self.error = error
        self.processor_kwargs: dict[str, Any] = {}
        self.model_kwargs: dict[str, Any] = {}
        self.opened_payloads: list[bytes] = []

    def _open_image(self, source: io.BytesIO) -> FakeImage:
        image = FakeImage(source)
        self.opened_payloads.append(image.payload)
        return image

    def _load_processor(self, model: str, **kwargs: Any) -> FakeProcessor:
        self.processor_kwargs = {"model": model, **kwargs}
        return self.processor

    def _load_model(self, model: str, **kwargs: Any) -> FakeModel:
        self.model_kwargs = {"model": model, **kwargs}
        if self.error is not None:
            raise self.error
        return self.model

    @property
    def AutoProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_processor)

    @property
    def Florence2ForConditionalGeneration(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_model)


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.reset_devices: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def reset_peak_memory_stats(self, device: str) -> None:
        self.reset_devices.append(device)

    def max_memory_allocated(self, device: str) -> int:
        return 1_500_000_000


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

    size = (320, 200)

    def __init__(self, source: io.BytesIO) -> None:
        self.payload = source.read()

    def convert(self, mode: str) -> FakeImage:
        assert mode == "RGB"
        return self

    def __enter__(self) -> FakeImage:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _config(**overrides: Any) -> Florence2SemanticConfig:
    values: dict[str, Any] = {
        "checkpoint": "florence-community/Florence-2-large",
        "revision": REVISION,
        "task": "<REGION_TO_CATEGORY>",
        "supported_modes": frozenset({SemanticInterpretationMode.REGION}),
        "device": "cuda",
        "precision": "bfloat16",
        "max_new_tokens": 64,
        "temperature": 0.0,
    }
    values.update(overrides)
    return Florence2SemanticConfig(**values)


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

    monkeypatch.setattr(
        florence2_semantic, "importlib", SimpleNamespace(import_module=import_module)
    )
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


def _generate(runtime: HuggingFaceFlorence2SemanticRuntime, config: Florence2SemanticConfig) -> Any:
    return runtime.generate(visual_views=(_visual_view(),), task_prompt=TASK_PROMPT, config=config)


def test_the_pinned_revision_is_loaded_locally_without_remote_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()

    _generate(HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config)

    for kwargs in (transformers.processor_kwargs, transformers.model_kwargs):
        assert kwargs["model"] == "florence-community/Florence-2-large"
        assert kwargs["revision"] == REVISION
        assert kwargs["local_files_only"] is True
        assert "trust_remote_code" not in kwargs
    assert transformers.model_kwargs["dtype"] == "torch.bfloat16"
    assert transformers.model.moved_to == "cuda"
    assert transformers.model.evaluating


def test_generation_feeds_the_exact_view_and_the_task_prompt_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, torch = _install(monkeypatch)
    _view(tmp_path)
    config = _config(max_new_tokens=48)

    response = _generate(
        HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config
    )

    processor = transformers.processor
    assert processor.call["text"] == TASK_PROMPT
    # A imagem é decodificada dos bytes verificados, não relida do caminho.
    assert processor.call["images"].payload == VIEW_PAYLOAD
    assert processor.call["return_tensors"] == "pt"
    assert processor.inputs is not None
    assert processor.inputs.moved == ("cuda", "torch.bfloat16")
    kwargs = transformers.model.generate_kwargs
    assert kwargs["max_new_tokens"] == 48
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert response.input_tokens == 11
    assert response.output_tokens == 6
    assert response.peak_memory_bytes == 1_500_000_000
    assert torch.cuda.reset_devices == ["cuda"]


def test_the_official_task_parser_is_applied_and_the_echoed_region_tokens_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()

    response = _generate(
        HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config
    )

    assert transformers.processor.decode_skip_special_tokens is False
    assert transformers.processor.post_process_call == {
        "text": "</s><s>car<loc_0><loc_0><loc_999><loc_999></s>",
        "task": "<REGION_TO_CATEGORY>",
        "image_size": (320, 200),
    }
    assert response.text == "car"


def test_positive_temperature_samples_with_that_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(temperature=0.3)

    _generate(HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config)

    assert transformers.model.generate_kwargs["do_sample"] is True
    assert transformers.model.generate_kwargs["temperature"] == 0.3


def test_model_is_loaded_once_and_cpu_runs_report_no_gpu_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config(device="cpu", precision="float32")
    runtime = HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path)

    first = _generate(runtime, config)
    model = transformers.model
    _generate(runtime, config)

    assert transformers.model is model
    assert first.peak_memory_bytes is None


def test_unavailable_cuda_and_unsupported_precision_are_device_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, torch=FakeTorch(cuda=FakeCuda(available=False)))
    _view(tmp_path)
    config = _config()
    with pytest.raises(Florence2SemanticError, match="CUDA"):
        _generate(HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config)

    _install(monkeypatch)
    config = _config(precision="int4")
    with pytest.raises(Florence2SemanticError, match="precision"):
        _generate(HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config)


def test_missing_sdk_and_unloadable_checkpoint_are_explicit_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _view(tmp_path)

    _install(monkeypatch, missing="transformers")
    with pytest.raises(Florence2SemanticError, match="requires torch"):
        _generate(HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config)

    _install(monkeypatch, transformers=FakeTransformers(error=OSError("not in local cache")))
    with pytest.raises(Florence2SemanticError, match="could not load"):
        _generate(HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path), config)


def test_generation_for_another_configuration_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    _view(tmp_path)
    runtime = HuggingFaceFlorence2SemanticRuntime(config=_config(), view_root=tmp_path)

    with pytest.raises(ValueError, match="another configuration"):
        _generate(runtime, _config(task="<REGION_TO_DESCRIPTION>"))


def test_view_references_cannot_escape_the_view_root_through_a_link_or_be_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    root = tmp_path / "root"
    (root / "outputs" / "semantic-views").mkdir(parents=True)
    (tmp_path / "secret.png").write_bytes(b"outside")
    linked_reference = "outputs/semantic-views/linked.png"
    (root / linked_reference).symlink_to(tmp_path / "secret.png")
    config = _config()
    runtime = HuggingFaceFlorence2SemanticRuntime(config=config, view_root=root)

    with pytest.raises(Florence2SemanticError, match="escapes"):
        runtime.generate(
            visual_views=(_visual_view(b"outside", reference=linked_reference),),
            task_prompt=TASK_PROMPT,
            config=config,
        )
    with pytest.raises(Florence2SemanticError, match="does not exist"):
        _generate(runtime, config)


def test_a_view_whose_bytes_diverge_from_the_request_sha256_is_never_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers, _ = _install(monkeypatch)
    _view(tmp_path)
    config = _config()
    adapter = Florence2SemanticInterpreter(
        config=config,
        runtime=HuggingFaceFlorence2SemanticRuntime(config=config, view_root=tmp_path),
    )
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("florence-region-0007"),
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
        prompt_template_id=adapter.prompt_template_id,
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )

    with pytest.raises(Florence2SemanticError, match="sha256"):
        adapter.interpret(request)

    assert transformers.opened_payloads == []
    assert transformers.processor.call == {}
    assert transformers.model.generate_kwargs == {}
