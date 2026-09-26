from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from hashlib import sha256
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from fakes import FakeLoadedModel
from mask_cases import BACKEND_SEEDS, GOLDEN, digest, sam3_candidates

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    AuditedRegionDiscovery,
    BoundingBox,
    InlineMask,
    PreparedImage,
    Region2D,
    RegionDiscovery,
)
from contextmap.visual_perception.backends import sam3 as sam3_module
from contextmap.visual_perception.backends.sam3 import (
    Sam3Config,
    Sam3ImageProcessorRuntime,
    Sam3NativeOutput,
    Sam3NativeProposal,
    Sam3RegionDiscovery,
    Sam3Strategy,
)
from contextmap.visual_perception.discovery import DiscoveryInput, DiscoveryPass, PassKind
from contextmap.visual_perception.normalization import normalize_regions
from contextmap.visual_perception.region_models import RejectionReason


@dataclass(frozen=True)
class MaterializedImage:
    size: tuple[int, int]
    token: tuple[str, str]


class InferenceModeTorch(ModuleType):
    """Fake torch whose inference-mode switch the fake SDK reads, as it would read the real one."""

    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"

    def __init__(self) -> None:
        super().__init__("torch")
        self._inference_mode = False
        self.autocasts: list[dict[str, object]] = []

    def is_inference_mode_enabled(self) -> bool:
        return self._inference_mode

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        previous, self._inference_mode = self._inference_mode, True
        try:
            yield
        finally:
            self._inference_mode = previous

    def autocast(self, **kwargs: object) -> AbstractContextManager[None]:
        self.autocasts.append(kwargs)
        return nullcontext()


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> InferenceModeTorch:
    """The official runtime always enters ``torch.inference_mode()``; torch itself is optional."""
    torch = InferenceModeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _materialized_image(discovery_input: DiscoveryInput) -> MaterializedImage:
    discovery_pass = discovery_input.discovery_pass
    return MaterializedImage(
        size=(discovery_pass.input_width, discovery_pass.input_height),
        token=("image", discovery_pass.pass_id),
    )


def _input() -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id=SourceObservationId("frame-8"),
            payload_reference="outputs/frame-8.png",
            payload_artifact=ArtifactReference(
                uri="outputs/frame-8.png",
                sha256=sha256(b"frame-8").hexdigest(),
                media_type="image/png",
            ),
            width=5,
            height=4,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=5, y_max=4),
        ),
        perception_run_id="run-sam3",
        perception_result_id="result-sam3",
    )


class FakeSam3Runtime:
    def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
        return Sam3NativeOutput(
            proposals=(
                Sam3NativeProposal(
                    proposal_id="proposal-3",
                    box=(1.0, 1.0, 4.0, 3.0),
                    mask=_mask_5x4(x_range=range(1, 4), y_range=range(1, 3)),
                    score_name="mask_score",
                    score=0.93,
                    query_id="query-1",
                    metadata=(("decoder_pass", 2),),
                ),
            ),
            warnings=("runtime used deterministic precision",),
            metadata=(("peak_memory_mb", 512.0),),
        )


def test_sam3_preserves_strategy_query_and_native_score_semantics() -> None:
    config = Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        device="cuda:0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompt="all movable items",
        score_threshold=0.8,
        mask_threshold=0.5,
        strategy_settings=(("max_queries", 8),),
    )
    backend = Sam3RegionDiscovery(config=config, runtime=FakeSam3Runtime())

    output = backend.discover_candidates(_input())

    candidate = output.candidates[0]
    assert candidate.score is not None
    assert candidate.score.name == "mask_score"
    assert "not calibrated" in candidate.score.semantics
    assert candidate.provenance.backend_id == "sam3"
    assert candidate.provenance.query == "text_prompt:all movable items:query-1"
    assert candidate.provenance.config_digest == config.digest
    assert dict(candidate.native_metadata)["decoder_pass"] == 2
    assert output.diagnostics.warnings == ("runtime used deterministic precision",)
    assert dict(output.diagnostics.metadata)["strategy"] == "text_prompt"
    assert dict(output.diagnostics.metadata)["peak_memory_mb"] == 512.0
    json.dumps(candidate.to_dict())
    assert isinstance(backend, RegionDiscovery)
    regions = backend.discover(_input().prepared_image)
    assert all(isinstance(region, Region2D) for region in regions)
    assert regions[0].provenance == backend.backend_provenance()


def test_sam3_reports_the_audit_of_the_regions_it_discovers() -> None:
    """#611: the regions of the public port come with the passes, rejections and merges."""
    config = Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompt="all movable items",
        score_threshold=0.8,
    )
    backend = Sam3RegionDiscovery(config=config, runtime=FakeSam3Runtime())
    image = _input().prepared_image

    audited = backend.discover_audited(image)

    assert isinstance(backend, AuditedRegionDiscovery)
    assert audited.regions == backend.discover(image)
    assert audited.audit.source_observation_id == image.source_observation_id
    assert audited.audit.backend == backend.backend_provenance()
    assert [discovery_pass.pass_id for discovery_pass in audited.audit.passes] == ["full-frame"]


def test_sam3_strategy_configuration_is_explicit() -> None:
    with pytest.raises(ValueError, match="requires a prompt"):
        Sam3Config(checkpoint="sam3", model_version="3.0", strategy=Sam3Strategy.TEXT_PROMPT)
    with pytest.raises(ValueError, match="does not consume a text prompt"):
        Sam3Config(
            checkpoint="sam3",
            model_version="3.0",
            strategy=Sam3Strategy.AUTOMATIC,
            prompt="hidden architectural default",
        )
    with pytest.raises(ValueError, match="score_threshold"):
        Sam3Config(checkpoint="sam3", model_version="3.0", score_threshold=-0.1)


def test_sam3_does_not_fall_back_when_configured_runtime_fails() -> None:
    class FailingRuntime:
        def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
            raise RuntimeError("configured strategy is unavailable")

    backend = Sam3RegionDiscovery(
        config=Sam3Config(checkpoint="sam3", model_version="3.0", strategy=Sam3Strategy.AUTOMATIC),
        runtime=FailingRuntime(),
    )

    with pytest.raises(RuntimeError, match="configured strategy is unavailable"):
        backend.discover_candidates(_input())


def test_official_sam3_text_processor_output_is_detached_and_thresholded() -> None:
    class NativeArray:
        def __init__(self, value: object) -> None:
            self._value = value

        def tolist(self) -> object:
            return self._value

    class ImageProcessor:
        model = FakeLoadedModel()

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def set_image(self, image: object) -> object:
            self.calls.append(("image", image))
            return {"image_state": "encoded"}

        def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
            self.calls.append(("threshold", threshold))
            return state

        def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
            self.calls.append(("prompt", prompt))
            if not isinstance(state, dict):
                raise TypeError("test state must be a dictionary")
            return {
                **state,
                "boxes": NativeArray([[1.0, 1.0, 4.0, 3.0]]),
                "scores": NativeArray([0.92]),
                "masks_logits": NativeArray(
                    [
                        [
                            [0.2, 0.8, 0.8, 0.2, 0.1],
                            [0.2, 0.8, 0.8, 0.2, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1],
                        ]
                    ]
                ),
            }

    processor = ImageProcessor()
    runtime = Sam3ImageProcessorRuntime(
        processor=processor,
        image_loader=_materialized_image,
    )
    config = Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompt="movable item",
        score_threshold=0.7,
        mask_threshold=0.5,
    )

    output = runtime.predict(_input(), config)

    assert processor.calls == [
        ("image", MaterializedImage((5, 4), ("image", "full-frame"))),
        ("threshold", 0.7),
        ("prompt", "movable item"),
    ]
    assert output.proposals[0].box == (1.0, 1.0, 4.0, 3.0)
    assert output.proposals[0].mask.as_array()[0, 1:3].tolist() == [True, True]
    assert output.proposals[0].score == 0.92
    assert output.proposals[0].query_id == "text-prompt-000000"


def test_official_sam3_runtime_rejects_an_unimplemented_strategy_without_fallback() -> None:
    class UnusedProcessor:
        model = FakeLoadedModel()

        def set_image(self, image: object) -> object:
            raise AssertionError("unsupported strategy must fail before inference")

        def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
            raise AssertionError("unsupported strategy must fail before inference")

        def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
            raise AssertionError("unsupported strategy must fail before inference")

    runtime = Sam3ImageProcessorRuntime(
        processor=UnusedProcessor(),
        image_loader=lambda discovery_input: object(),
    )
    config = Sam3Config(checkpoint="sam3", model_version="3.0", strategy=Sam3Strategy.AUTOMATIC)

    with pytest.raises(ValueError, match="supports only text_prompt"):
        runtime.predict(_input(), config)


def _config() -> Sam3Config:
    return Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompt="floor",
        score_threshold=0.5,
    )


def _proposal(box: tuple[float, float, float, float], mask: InlineMask) -> Sam3NativeProposal:
    return Sam3NativeProposal(
        proposal_id="proposal-1",
        box=box,
        mask=mask,
        score_name="concept_score",
        score=0.9,
        query_id="text-prompt-000000",
    )


class ProposalRuntime:
    def __init__(self, *proposals: Sam3NativeProposal) -> None:
        self._proposals = proposals

    def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
        return Sam3NativeOutput(proposals=self._proposals)


def _mask_5x4(*, x_range: range, y_range: range) -> InlineMask:
    return InlineMask(
        np.array([[x in x_range and y in y_range for x in range(5)] for y in range(4)])
    )


def test_sam3_derives_the_candidate_box_from_the_mask_and_keeps_the_native_box() -> None:
    # A caixa nativa vem de um head independente e aqui não contém a máscara.
    proposal = _proposal((1.0, 1.0, 4.0, 3.0), _mask_5x4(x_range=range(0, 5), y_range=range(0, 2)))
    backend = Sam3RegionDiscovery(config=_config(), runtime=ProposalRuntime(proposal))

    candidate = backend.discover_candidates(_input()).candidates[0]

    assert candidate.bounding_box == BoundingBox(x_min=0, y_min=0, x_max=5, y_max=2)
    metadata = dict(candidate.native_metadata)
    assert (
        metadata["native_box_x_min"],
        metadata["native_box_y_min"],
        metadata["native_box_x_max"],
        metadata["native_box_y_max"],
    ) == (1.0, 1.0, 4.0, 3.0)
    assert metadata["native_box_contains_mask"] is False
    assert len(backend.discover(_input().prepared_image)) == 1


def test_sam3_native_box_outside_the_image_does_not_abort_discovery() -> None:
    proposal = _proposal((-0.4, 0.0, 5.6, 4.2), _mask_5x4(x_range=range(0, 5), y_range=range(2, 4)))
    backend = Sam3RegionDiscovery(config=_config(), runtime=ProposalRuntime(proposal))

    candidate = backend.discover_candidates(_input()).candidates[0]

    assert candidate.bounding_box == BoundingBox(x_min=0, y_min=2, x_max=5, y_max=4)
    assert dict(candidate.native_metadata)["native_box_x_min"] == -0.4
    assert dict(candidate.native_metadata)["native_box_contains_mask"] is True
    regions = backend.discover(_input().prepared_image)
    assert len(regions) == 1
    assert regions[0].area_pixels == 10


def test_sam3_empty_masks_are_rejected_explicitly_by_normalization() -> None:
    empty = InlineMask(np.zeros((4, 5), dtype=bool))
    inside = _proposal((1.0, 1.0, 3.0, 3.0), empty)
    outside = Sam3NativeProposal(
        proposal_id="proposal-2",
        box=(-5.0, -5.0, -1.0, -1.0),
        mask=empty,
        score_name="concept_score",
        score=0.8,
        query_id="text-prompt-000001",
    )
    backend = Sam3RegionDiscovery(config=_config(), runtime=ProposalRuntime(inside, outside))

    output = backend.discover_candidates(_input())
    result = normalize_regions(
        output.candidates, _input().prepared_image, backend.backend_provenance()
    )

    assert len(output.candidates) == 2
    assert result.regions == ()
    assert {item.reason for item in result.rejected} == {RejectionReason.INVALID_GEOMETRY}


class RecordingProcessor:
    """Official-processor stand-in that records call order into a shared event list."""

    def __init__(self, events: list[str], *, model: object | None = None) -> None:
        self._events = events
        self.model = model or FakeLoadedModel("cuda:0")

    def set_image(self, image: object) -> object:
        self._events.append("image")
        return {}

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        self._events.append("threshold")
        return state

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        self._events.append("prompt")
        return {"boxes": [], "scores": [], "masks": []}


class RecordingContext:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self) -> None:
        self._events.append("enter")

    def __exit__(self, *exc_info: object) -> None:
        self._events.append("exit")


def _text_prompt_config(precision: str) -> Sam3Config:
    return Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        device="cuda:0",
        precision=precision,
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompt="floor",
    )


def test_sam3_precision_must_be_a_supported_inference_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        Sam3Config(checkpoint="sam3", model_version="3.0", precision="int8")
    for precision in ("float32", "float16", "bfloat16"):
        assert (
            Sam3Config(checkpoint="sam3", model_version="3.0", precision=precision).precision
            == precision
        )


def test_official_sam3_runtime_runs_the_sdk_inside_the_configured_inference_context() -> None:
    events: list[str] = []
    received: list[Sam3Config] = []

    def autocast(config: Sam3Config) -> RecordingContext:
        received.append(config)
        return RecordingContext(events)

    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor(events),
        image_loader=_materialized_image,
        autocast=autocast,
    )
    config = _text_prompt_config("bfloat16")

    runtime.predict(_input(), config)

    assert received == [config]
    assert events == ["enter", "image", "threshold", "prompt", "exit"]


def test_official_sam3_runtime_float32_runs_without_autocast(
    fake_torch: InferenceModeTorch,
) -> None:
    events: list[str] = []
    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor(events), image_loader=_materialized_image
    )

    runtime.predict(_input(), _text_prompt_config("float32"))

    assert events == ["image", "threshold", "prompt"]
    assert fake_torch.autocasts == []


def test_official_sam3_runtime_default_context_is_torch_autocast(
    fake_torch: InferenceModeTorch,
) -> None:
    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor([]), image_loader=_materialized_image
    )

    runtime.predict(_input(), _text_prompt_config("bfloat16"))

    assert fake_torch.autocasts == [{"device_type": "cuda", "dtype": "torch.bfloat16"}]


def test_official_sam3_runtime_without_torch_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(sam3_module, "import_module", unavailable)
    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor([]), image_loader=_materialized_image
    )

    with pytest.raises(RuntimeError, match="requires torch inference_mode"):
        runtime.predict(_input(), _text_prompt_config("float32"))


class InferenceModeRecordingProcessor:
    """Official-processor stand-in that records whether each SDK call ran in inference mode."""

    def __init__(self, torch: InferenceModeTorch) -> None:
        self._torch = torch
        self.model = FakeLoadedModel("cuda:0")
        self.inference_mode: list[bool] = []

    def set_image(self, image: object) -> object:
        self.inference_mode.append(self._torch.is_inference_mode_enabled())
        return {}

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        self.inference_mode.append(self._torch.is_inference_mode_enabled())
        return state

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        self.inference_mode.append(self._torch.is_inference_mode_enabled())
        return {"boxes": [], "scores": [], "masks": []}


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_official_sam3_runtime_runs_the_sdk_in_torch_inference_mode(
    precision: str, fake_torch: InferenceModeTorch
) -> None:
    processor = InferenceModeRecordingProcessor(fake_torch)
    runtime = Sam3ImageProcessorRuntime(processor=processor, image_loader=_materialized_image)

    runtime.predict(_input(), _text_prompt_config(precision))

    assert processor.inference_mode == [True, True, True]
    assert not fake_torch.is_inference_mode_enabled()


def test_sam3_configuration_requires_an_explicit_model_version() -> None:
    with pytest.raises(TypeError, match="model_version"):
        Sam3Config(checkpoint="facebook/sam3")  # type: ignore[call-arg]


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_official_sam3_runtime_rejects_a_model_on_another_device(device: str) -> None:
    events: list[str] = []
    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor(events, model=FakeLoadedModel(device, "float32")),
        image_loader=_materialized_image,
    )

    with pytest.raises(ValueError, match="device 'cuda:0'"):
        runtime.predict(_input(), _text_prompt_config("bfloat16"))

    assert events == []


@pytest.mark.parametrize("seed", BACKEND_SEEDS)
def test_sam3_mask_conversion_matches_the_recorded_behaviour(seed: int) -> None:
    # #593: do resultado nativo do SDK ao RegionCandidate, registrado antes da vetorização.
    assert digest(sam3_candidates(seed)) == GOLDEN["sam3"][str(seed)]


def test_sam3_tensor_masks_are_detached_to_the_cpu_in_double_precision() -> None:
    # #593: o tensor do SDK (GPU, bfloat16) vira um array de uma vez, nunca uma lista por pixel.
    calls: list[object] = []

    class FakeTensor:
        def __init__(self, array: np.ndarray[Any, Any]) -> None:
            self._array = array

        def detach(self) -> FakeTensor:
            calls.append("detach")
            return self

        def to(self, device: str) -> FakeTensor:
            calls.append(("to", device))
            return self

        def double(self) -> FakeTensor:
            calls.append("double")
            return FakeTensor(self._array.astype(np.float64))

        def numpy(self) -> np.ndarray[Any, Any]:
            calls.append("numpy")
            return self._array

        def tolist(self) -> object:
            raise AssertionError("a tensor mask must not become one Python object per pixel")

    logits = np.full((1, 1, 4, 5), 0.2, dtype=np.float32)
    logits[0, 0, 1, 1:4] = 0.9

    class Processor:
        model = FakeLoadedModel()

        def set_image(self, image: object) -> object:
            return {}

        def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
            return state

        def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
            return {
                "boxes": [[1.0, 1.0, 4.0, 2.0]],
                "scores": [0.9],
                "masks_logits": FakeTensor(logits),
            }

    runtime = Sam3ImageProcessorRuntime(processor=Processor(), image_loader=_materialized_image)

    output = runtime.predict(_input(), _config())

    assert calls == ["detach", ("to", "cpu"), "double", "numpy"]
    assert output.proposals[0].mask.as_array()[1].tolist() == [False, True, True, True, False]
