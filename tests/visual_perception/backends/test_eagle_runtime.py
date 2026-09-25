"""Contract tests for the transformers-backed Eagle 2.5 runtime, using fake SDK modules.

These are fake/contract tests: they prove how the runtime maps the canonical configuration
and views onto the ``eagle_2_5_vl`` remote-code processor and model calls documented by
NVlabs, not the quality of any real Eagle checkpoint. No weights are ever loaded.
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
from contextmap.visual_perception.backends import eagle2_5
from contextmap.visual_perception.backends.eagle2_5 import (
    EagleDependencyError,
    EagleDeviceError,
    EagleInferenceError,
    EagleModelLoadError,
    EagleSemanticConfig,
    EagleSemanticInterpreter,
    HuggingFaceEagleRuntime,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
TIGHT_REFERENCE = "outputs/semantic-views/region-0007-tight.png"
CONTEXT_REFERENCE = "outputs/semantic-views/region-0007-context.png"
TIGHT_PAYLOAD = b"tight crop bytes are never decoded by the fake SDK"
CONTEXT_PAYLOAD = b"contextual crop bytes are never decoded by the fake SDK"
CHAT_TEXT = "<|im_start|>user\n<image-1><image-2>canonical prompt<|im_end|>"


class FakeTokens:
    """Minimal 2-D token tensor exposing ``.shape``."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]))


class FakeTiles:
    """Stands in for ``pixel_values``: one row per tile the processor produced."""

    def __init__(self, count: int) -> None:
        self.shape = (count, 3, 448, 448)


class FakeInputs(dict[str, Any]):
    """Processor output that records the device it was moved to."""

    device: str | None = None

    def to(self, device: str) -> FakeInputs:
        self.device = device
        return self


class FakeProcessor:
    """Records the chat-template and processor calls of the Eagle 2.5 remote processor."""

    def __init__(self, *, tiles_per_image: int | None = None, prompt_tokens: int = 9) -> None:
        self.tiles_per_image = tiles_per_image
        self.prompt_tokens = prompt_tokens
        self.messages: list[dict[str, Any]] = []
        self.template_kwargs: dict[str, Any] = {}
        self.call_kwargs: dict[str, Any] = {}
        self.inputs: FakeInputs | None = None
        self.decoded: FakeTokens | None = None
        self.decode_kwargs: dict[str, Any] = {}
        self.decoded_text = '{"abstained": true, "claims": [], "scene_context": null}'

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.messages = messages
        self.template_kwargs = kwargs
        return CHAT_TEXT

    def __call__(self, **kwargs: Any) -> FakeInputs:
        self.call_kwargs = kwargs
        images = kwargs["images"]
        # Por padrão o processor falso honra o orçamento configurado, como o processor real.
        per_image = (
            kwargs["images_kwargs"]["max_dynamic_tiles"]
            if self.tiles_per_image is None
            else self.tiles_per_image
        )
        self.inputs = FakeInputs(
            input_ids=FakeTokens([[1] * self.prompt_tokens]),
            attention_mask=FakeTokens([[1] * self.prompt_tokens]),
            pixel_values=FakeTiles(per_image * len(images)),
        )
        return self.inputs

    def batch_decode(self, tokens: FakeTokens, **kwargs: Any) -> list[str]:
        self.decoded = tokens
        self.decode_kwargs = kwargs
        return [self.decoded_text]


class FakeModel:
    """Eagle's ``generate`` feeds ``inputs_embeds`` to the LLM: it returns only new tokens."""

    def __init__(self, *, new_tokens: int = 5) -> None:
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
        return FakeTokens([[9] * self.new_tokens])


class FakeTransformers(ModuleType):
    """Records how the runtime loads the remote-code processor and model."""

    def __init__(
        self,
        *,
        processor: FakeProcessor | None = None,
        new_tokens: int = 5,
        model_error: Exception | None = None,
    ) -> None:
        super().__init__("transformers")
        self.processor = processor or FakeProcessor()
        self.new_tokens = new_tokens
        self.model_error = model_error
        self.model: FakeModel | None = None
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
        if self.model_error is not None:
            raise self.model_error
        self.model = FakeModel(new_tokens=self.new_tokens)
        return self.model

    @property
    def AutoProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_processor)

    @property
    def AutoModel(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_model)


class FakeCuda:
    def __init__(self, *, available: bool = True, peak: int = 18_000_000_000) -> None:
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


def _config(**overrides: Any) -> EagleSemanticConfig:
    values: dict[str, Any] = {
        "model": "nvidia/Eagle2.5-8B",
        "revision": REVISION,
        "device": "cuda",
        "precision": "bfloat16",
        "max_new_tokens": 64,
        "temperature": 0.0,
        "max_dynamic_tiles": 4,
    }
    values.update(overrides)
    return EagleSemanticConfig(**values)


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

    monkeypatch.setattr(eagle2_5, "importlib", SimpleNamespace(import_module=import_module))
    return transformers, torch


def _write(root: Path, reference: str, payload: bytes) -> None:
    path = root / reference
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _visual_view(
    view_id: str,
    kind: VisualViewKind,
    reference: str,
    payload: bytes,
) -> SemanticVisualView:
    """Describe a canonical view whose ``sha256`` identifies exactly ``payload``."""
    return SemanticVisualView(
        view_id=view_id,
        kind=kind,
        payload_reference=reference,
        source_observation_id=SourceObservationId("frame-0124"),
        region_id=RegionId("region-0007"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _views() -> tuple[SemanticVisualView, ...]:
    """Two views in a deliberately non-alphabetical order: context first, tight crop second."""
    return (
        _visual_view("context", VisualViewKind.CONTEXTUAL_CROP, CONTEXT_REFERENCE, CONTEXT_PAYLOAD),
        _visual_view("tight", VisualViewKind.TIGHT_CROP, TIGHT_REFERENCE, TIGHT_PAYLOAD),
    )


def _root(tmp_path: Path) -> Path:
    _write(tmp_path, TIGHT_REFERENCE, TIGHT_PAYLOAD)
    _write(tmp_path, CONTEXT_REFERENCE, CONTEXT_PAYLOAD)
    return tmp_path


def _generate(runtime: HuggingFaceEagleRuntime, config: EagleSemanticConfig) -> Any:
    return runtime.generate(visual_views=_views(), prompt="canonical prompt", config=config)


class TestLoading:
    def test_the_pinned_remote_code_revision_is_loaded_from_the_local_cache_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config()

        _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        for kwargs in (transformers.processor_kwargs, transformers.model_kwargs):
            assert kwargs["model"] == "nvidia/Eagle2.5-8B"
            assert kwargs["revision"] == REVISION
            assert kwargs["local_files_only"] is True
            # O código do modelo é remoto: só é executado na revisão imutável fixada.
            assert kwargs["trust_remote_code"] is True
        assert transformers.processor_kwargs["use_fast"] is True
        assert transformers.model_kwargs["torch_dtype"] == "torch.bfloat16"

    def test_the_model_is_moved_to_the_configured_device_in_eval_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config()

        _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        assert transformers.model is not None
        assert transformers.model.moved_to == "cuda"
        assert transformers.model.evaluating

    def test_the_runtime_requires_a_pinned_revision(self) -> None:
        with pytest.raises(ValueError, match="revision"):
            HuggingFaceEagleRuntime(config=_config(revision=None), view_root=Path("."))

    def test_the_model_is_loaded_once_and_reused_across_requests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config()
        runtime = HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path))

        runtime.load()
        first = transformers.model
        _generate(runtime, config)
        _generate(runtime, config)

        assert transformers.model is first

    def test_generation_for_another_effective_configuration_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        runtime = HuggingFaceEagleRuntime(config=_config(), view_root=_root(tmp_path))

        with pytest.raises(ValueError, match="another configuration"):
            _generate(runtime, _config(max_dynamic_tiles=12))

    def test_a_missing_sdk_is_a_dependency_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, missing="transformers")
        config = _config()

        with pytest.raises(EagleDependencyError, match="transformers"):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

    def test_a_package_the_remote_code_imports_is_a_dependency_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            transformers=FakeTransformers(model_error=ImportError("requires flash_attn")),
        )
        config = _config()

        with pytest.raises(EagleDependencyError, match="flash_attn"):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

    def test_an_unloadable_checkpoint_is_a_model_load_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            transformers=FakeTransformers(model_error=OSError("gated repository not in cache")),
        )
        config = _config()

        with pytest.raises(EagleModelLoadError, match=r"could not load nvidia/Eagle2.5-8B@"):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

    def test_an_unavailable_cuda_device_is_an_explicit_device_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch, torch=FakeTorch(cuda=FakeCuda(available=False)))
        config = _config()

        with pytest.raises(EagleDeviceError, match="CUDA"):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        assert transformers.model is None

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"precision": "int4"}, "precision"),
            ({"device": "cpu", "precision": "float16"}, "float16"),
        ],
    )
    def test_an_unusable_precision_is_rejected_before_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        change: dict[str, str],
        message: str,
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config(**change)

        with pytest.raises(EagleDeviceError, match=message):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        assert transformers.model is None


class TestInputs:
    def test_every_view_reaches_the_model_in_request_order_followed_by_the_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config()

        _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        processor = transformers.processor
        (message,) = processor.messages
        assert message["role"] == "user"
        assert [part["type"] for part in message["content"]] == ["image", "image", "text"]
        # Cada imagem é decodificada dos bytes verificados, na ordem exata do request.
        assert [part["image"].payload for part in message["content"][:2]] == [
            CONTEXT_PAYLOAD,
            TIGHT_PAYLOAD,
        ]
        assert message["content"][2]["text"] == "canonical prompt"
        assert processor.template_kwargs == {"tokenize": False, "add_generation_prompt": True}
        # O processor substitui <image-k> pela k-ésima imagem da lista: mesma ordem do chat.
        assert processor.call_kwargs["text"] == [CHAT_TEXT]
        assert [image.payload for image in processor.call_kwargs["images"]] == [
            CONTEXT_PAYLOAD,
            TIGHT_PAYLOAD,
        ]
        assert processor.call_kwargs["return_tensors"] == "pt"
        assert processor.inputs is not None
        assert processor.inputs.device == "cuda"

    def test_the_configured_tiling_budget_is_passed_explicitly_to_the_processor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config(max_dynamic_tiles=6, min_dynamic_tiles=2, use_thumbnail=False)

        _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        assert transformers.processor.call_kwargs["images_kwargs"] == {
            "max_dynamic_tiles": 6,
            "min_dynamic_tiles": 2,
            "use_thumbnail": False,
        }

    def test_tiles_beyond_the_configured_budget_are_never_inferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Um processor que ignorasse o orçamento produziria as 12 tiles padrão do checkpoint.
        transformers, _ = _install(
            monkeypatch,
            transformers=FakeTransformers(processor=FakeProcessor(tiles_per_image=13)),
        )
        config = _config(max_dynamic_tiles=4)

        with pytest.raises(EagleInferenceError, match=r"26 tiles.*at most 10"):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        assert transformers.model is not None
        assert transformers.model.generate_kwargs == {}

    def test_the_thumbnail_tile_is_within_the_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            transformers=FakeTransformers(processor=FakeProcessor(tiles_per_image=5)),
        )
        config = _config(max_dynamic_tiles=4, use_thumbnail=True)

        response = _generate(
            HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config
        )

        assert response.text


class TestViewIntegrity:
    def test_a_view_whose_bytes_diverge_from_the_request_sha256_is_never_inferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        _write(tmp_path, TIGHT_REFERENCE, TIGHT_PAYLOAD)
        config = _config()
        adapter = EagleSemanticInterpreter(
            config=config, runtime=HuggingFaceEagleRuntime(config=config, view_root=tmp_path)
        )
        request = SemanticInterpretationRequest(
            request_id=SemanticRequestId("request-0007"),
            source_observation_id=SourceObservationId("frame-0124"),
            perception_result_id=PerceptionResultId("run-0001--frame-0124"),
            mode=SemanticInterpretationMode.REGION,
            region_id=RegionId("region-0007"),
            visual_views=(
                _visual_view(
                    "tight",
                    VisualViewKind.TIGHT_CROP,
                    TIGHT_REFERENCE,
                    b"the bytes the request identifies",
                ),
            ),
            prompt_template_id="region/v1",
            requested_output_schema="semantic-response/1",
            configuration_fingerprint=adapter.configuration_fingerprint,
        )

        with pytest.raises(EagleInferenceError, match="sha256"):
            adapter.interpret(request)

        assert transformers.opened_payloads == []
        assert transformers.processor.messages == []
        assert transformers.model is None or transformers.model.generate_kwargs == {}

    def test_one_diverging_view_among_several_prevents_the_inference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        _write(tmp_path, CONTEXT_REFERENCE, CONTEXT_PAYLOAD)
        _write(tmp_path, TIGHT_REFERENCE, b"changed after the request was built")
        config = _config()

        with pytest.raises(EagleInferenceError, match=r"region-0007-tight.*sha256"):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=tmp_path), config)

        # A primeira view estava íntegra e foi aberta, mas nenhuma inferência aconteceu.
        assert transformers.opened_payloads == [CONTEXT_PAYLOAD]
        assert transformers.processor.call_kwargs == {}
        assert transformers.model is not None
        assert transformers.model.generate_kwargs == {}

    def test_a_missing_view_payload_is_an_inference_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        config = _config()

        with pytest.raises(EagleInferenceError, match="does not exist"):
            _generate(HuggingFaceEagleRuntime(config=config, view_root=tmp_path), config)

    def test_view_references_cannot_escape_the_view_root_through_a_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        root = tmp_path / "root"
        (root / "outputs" / "semantic-views").mkdir(parents=True)
        (tmp_path / "secret.png").write_bytes(b"outside")
        (root / TIGHT_REFERENCE).symlink_to(tmp_path / "secret.png")
        config = _config()
        runtime = HuggingFaceEagleRuntime(config=config, view_root=root)
        view = _visual_view("tight", VisualViewKind.TIGHT_CROP, TIGHT_REFERENCE, b"outside")

        with pytest.raises(EagleInferenceError, match="escapes"):
            runtime.generate(visual_views=(view,), prompt="prompt", config=config)


class TestGeneration:
    def test_zero_temperature_is_greedy_and_does_not_pass_sampling_parameters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config(max_new_tokens=96)

        _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        assert transformers.model is not None
        kwargs = transformers.model.generate_kwargs
        assert kwargs["max_new_tokens"] == 96
        assert kwargs["do_sample"] is False
        assert kwargs["temperature"] is None
        assert kwargs["top_p"] is None
        assert kwargs["top_k"] is None
        # As entradas do processor (ids, máscara, tiles) chegam inteiras ao generate do Eagle.
        assert transformers.processor.inputs is not None
        for name in ("input_ids", "attention_mask", "pixel_values"):
            assert kwargs[name] is transformers.processor.inputs[name]

    def test_positive_temperature_samples_with_that_temperature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, _ = _install(monkeypatch)
        config = _config(temperature=0.4)

        _generate(HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config)

        assert transformers.model is not None
        assert transformers.model.generate_kwargs["do_sample"] is True
        assert transformers.model.generate_kwargs["temperature"] == 0.4

    def test_the_generated_ids_are_the_answer_and_are_decoded_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformers, torch = _install(monkeypatch)
        config = _config()

        response = _generate(
            HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config
        )

        processor = transformers.processor
        assert response.text == processor.decoded_text
        assert processor.decoded is not None
        assert processor.decoded.shape == (1, 5)  # nada é cortado: o prompt não volta no output
        assert processor.decode_kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        assert response.input_tokens == 9
        assert response.output_tokens == 5
        assert response.peak_memory_bytes == 18_000_000_000
        assert torch.cuda.reset_devices == ["cuda"]
        assert response.warnings == ()

    def test_cpu_generation_reports_no_gpu_memory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        config = _config(device="cpu", precision="float32")

        response = _generate(
            HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config
        )

        assert response.peak_memory_bytes is None

    def test_hitting_the_token_limit_is_reported_because_the_json_is_probably_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, transformers=FakeTransformers(new_tokens=64))
        config = _config(max_new_tokens=64)

        response = _generate(
            HuggingFaceEagleRuntime(config=config, view_root=_root(tmp_path)), config
        )

        assert len(response.warnings) == 1
        assert "max_new_tokens=64" in response.warnings[0]
