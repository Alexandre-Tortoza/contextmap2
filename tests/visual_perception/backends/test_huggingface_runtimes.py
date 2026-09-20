"""Contract tests for the preprocessing and loading shared by the Hugging Face runtimes."""

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from contextmap.visual_perception import FeatureScope
from contextmap.visual_perception.backends import clip, dinov2, dinov3
from contextmap.visual_perception.backends._huggingface import preprocess_pixel_values

REVISION = "0123456789abcdef0123456789abcdef01234567"


@dataclass
class FakeImage:
    name: str
    resizes: list[tuple[tuple[int, int], object]] = field(default_factory=list)

    def resize(self, size: tuple[int, int], resample: object) -> "FakeImage":
        self.resizes.append((size, resample))
        return FakeImage(f"{self.name}@{size[0]}x{size[1]}")


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"pixel_values": "pixel-values"}


def test_preprocessing_resizes_with_pillow_and_leaves_only_normalization_to_the_processor() -> None:
    processor = RecordingProcessor()
    first, second = FakeImage("first"), FakeImage("second")

    pixel_values = preprocess_pixel_values(
        processor=processor,
        images=[first, second],
        width=448,
        height=336,
        resample="bicubic",
    )

    assert pixel_values == "pixel-values"
    assert first.resizes == [((448, 336), "bicubic")]
    assert second.resizes == [((448, 336), "bicubic")]
    assert processor.calls == [
        {
            "images": [FakeImage("first@448x336"), FakeImage("second@448x336")],
            "return_tensors": "pt",
            "do_resize": False,
            "do_center_crop": False,
        }
    ]


class FakeModel:
    def to(self, device: str) -> "FakeModel":
        return self

    def eval(self) -> "FakeModel":
        return self


class FakeTransformers(ModuleType):
    """Records how the adapter loads the processor and the model."""

    def __init__(self, processor_error: Exception | None = None) -> None:
        super().__init__("transformers")
        self.model_kwargs: dict[str, Any] = {}
        self._processor_error = processor_error

    def _load_processor(self, *args: Any, **kwargs: Any) -> object:
        if self._processor_error is not None:
            raise self._processor_error
        return object()

    def _load_model(self, *args: Any, **kwargs: Any) -> FakeModel:
        self.model_kwargs = kwargs
        return FakeModel()

    @property
    def AutoImageProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_processor)

    @property
    def AutoProcessor(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_processor)

    @property
    def AutoModel(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_model)

    @property
    def CLIPModel(self) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=self._load_model)


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(float32="torch.float32", float16="torch.float16")


def _runtime(adapter: str, tmp_path: Path) -> tuple[Any, ModuleType, type[Exception]]:
    if adapter == "dinov2":
        config = dinov2.DinoV2Config(checkpoint="facebook/dinov2-base", revision=REVISION)
        return (
            dinov2.HuggingFaceDinoV2Runtime(config=config, prepared_image_root=tmp_path),
            dinov2,
            dinov2.DinoV2DependencyError,
        )
    if adapter == "dinov3":
        config3 = dinov3.DinoV3Config(checkpoint="facebook/dinov3-vits16", revision=REVISION)
        return (
            dinov3.HuggingFaceDinoV3Runtime(config=config3, prepared_image_root=tmp_path),
            dinov3,
            dinov3.DinoV3DependencyError,
        )
    clip_config = clip.ClipConfig(
        checkpoint="openai/clip-vit-large-patch14", revision=REVISION, scope=FeatureScope.GLOBAL
    )
    return (
        clip.HuggingFaceClipRuntime(config=clip_config, prepared_image_root=tmp_path),
        clip,
        clip.ClipDependencyError,
    )


def _install_sdks(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, transformers: FakeTransformers
) -> None:
    sdks = {"torch": _fake_torch(), "transformers": transformers, "PIL.Image": SimpleNamespace()}

    def import_module(name: str) -> object:
        return sdks[name]

    monkeypatch.setattr(module, "importlib", SimpleNamespace(import_module=import_module))


@pytest.mark.parametrize("adapter", ["dinov2", "dinov3", "clip"])
def test_runtimes_load_the_model_with_dtype_instead_of_the_deprecated_torch_dtype(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, module, _ = _runtime(adapter, tmp_path)
    transformers = FakeTransformers()
    _install_sdks(monkeypatch, module, transformers)

    runtime._ensure_loaded()

    assert transformers.model_kwargs["dtype"] == "torch.float32"
    assert "torch_dtype" not in transformers.model_kwargs


@pytest.mark.parametrize("adapter", ["dinov2", "dinov3", "clip"])
def test_a_missing_optional_import_in_the_processor_is_a_dependency_error(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, module, dependency_error = _runtime(adapter, tmp_path)
    transformers = FakeTransformers(
        processor_error=ImportError("AutoImageProcessor requires the Torchvision library")
    )
    _install_sdks(monkeypatch, module, transformers)

    with pytest.raises(dependency_error, match=r"(?i)torchvision"):
        runtime._ensure_loaded()


@pytest.mark.parametrize("adapter", ["dinov2", "dinov3", "clip"])
def test_other_load_failures_stay_model_load_errors(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, module, dependency_error = _runtime(adapter, tmp_path)
    transformers = FakeTransformers(processor_error=OSError("checkpoint not in local cache"))
    _install_sdks(monkeypatch, module, transformers)

    with pytest.raises(Exception, match="could not load") as caught:
        runtime._ensure_loaded()

    assert not isinstance(caught.value, dependency_error)
