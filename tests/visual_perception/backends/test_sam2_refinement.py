"""Contract tests for SAM2 prompted refinement of grounding proposals (#568).

A fake prompt runtime and a fake official predictor stand in for SAM2, so these tests pin
the mapping from canonical prompts to the SAM2 image-predictor API and back to canonical
refinement outcomes, without a model, a GPU or a network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingTask,
    PerceptionRunId,
    PreparedImage,
    RefinementPrompt,
    RefinementRejectionReason,
    RefinementRequestError,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionRefinement,
    RegionRefinementRequest,
    perception_result_id_for,
    refinement_prompts_from,
)
from contextmap.visual_perception.backends import sam2
from contextmap.visual_perception.backends.sam2 import (
    Sam2Config,
    Sam2ImagePredictorRuntime,
    Sam2PromptedMask,
    Sam2PromptRefinement,
    Sam2RefinementConfig,
)

WIDTH, HEIGHT = 8, 6
PIXELS = b"png bytes the fake decoder never parses"
RESULT_ID = perception_result_id_for(
    run_id=PerceptionRunId("run-0001"), source_observation_id=SourceObservationId("frame-0124")
)


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
        width=WIDTH,
        height=HEIGHT,
    )


def _prompts(image: PreparedImage) -> tuple[RefinementPrompt, ...]:
    grounding = RegionGroundingExecution(
        request=RegionGroundingRequest(
            perception_result_id=RESULT_ID,
            image=image,
            query=GroundingQuery(
                task=GroundingTask.PHRASE_GROUNDING,
                policy_id="fake.phrase/1",
                geometry=GroundingGeometry.BOX,
                text="the chair",
            ),
            configuration_fingerprint="sha256:grounding",
        ),
        provenance=BackendProvenance(
            backend_id="fake_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="sha256:grounding",
        ),
        rendered_prompt="find the chair",
        raw_response="...",
        outputs=(
            GroundingOutput(
                output_index=0,
                native_text="box",
                box=BoundingBox2D(x=1.0, y=1.0, width=4.0, height=3.0),
            ),
            GroundingOutput(
                output_index=1, native_text="point", point=GroundingPoint(x=8.0, y=6.0)
            ),
        ),
        diagnostics=GroundingDiagnostics(latency_ms=1.0),
        effective_configuration=MappingProxyType({}),
    )
    return refinement_prompts_from(grounding)


def _config(**overrides: Any) -> Sam2RefinementConfig:
    values: dict[str, Any] = {"checkpoint": "sam2.1_hiera_large", "model_version": "2.1"}
    values.update(overrides)
    return Sam2RefinementConfig(**values)


def _mask(*cells: tuple[int, int]) -> tuple[bool, ...]:
    return tuple((x, y) in set(cells) for y in range(HEIGHT) for x in range(WIDTH))


class FakePromptRuntime:
    def __init__(self, masks: tuple[Sam2PromptedMask, ...]) -> None:
        self.masks = masks
        self.calls: list[dict[str, Any]] = []

    def predict_prompts(self, **kwargs: Any) -> tuple[Sam2PromptedMask, ...]:
        self.calls.append(kwargs)
        return self.masks


def _refiner(
    tmp_path: Path, masks: tuple[Sam2PromptedMask, ...], **overrides: Any
) -> tuple[Sam2PromptRefinement, FakePromptRuntime]:
    runtime = FakePromptRuntime(masks)
    refiner = Sam2PromptRefinement(
        config=_config(**overrides), runtime=runtime, prepared_image_root=tmp_path
    )
    return refiner, runtime


def _request(refiner: Sam2PromptRefinement, image: PreparedImage) -> RegionRefinementRequest:
    return RegionRefinementRequest(
        perception_result_id=RESULT_ID,
        image=image,
        prompts=_prompts(image),
        configuration_fingerprint=str(refiner.backend_provenance().configuration_fingerprint),
    )


GOOD = (
    Sam2PromptedMask(mask=_mask((2, 2), (3, 2), (2, 3)), predicted_iou=0.93),
    Sam2PromptedMask(mask=_mask(), predicted_iou=0.12),
)


# --- configuration ------------------------------------------------------------------------------


def test_the_refiner_identity_is_its_own_not_the_discovery_one() -> None:
    assert (
        _config().digest != Sam2Config(checkpoint="sam2.1_hiera_large", model_version="2.1").digest
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"checkpoint": "sam2.1_hiera_small"},
        {"model_version": "2.0"},
        {"device": "cuda"},
        {"precision": "bfloat16"},
        {"mask_threshold": 0.5},
        {"max_hole_area": 10.0},
        {"max_sprinkle_area": 10.0},
    ],
)
def test_every_refiner_setting_changes_the_refinement_identity(overrides: dict[str, Any]) -> None:
    assert _config(**overrides).digest != _config().digest


@pytest.mark.parametrize(
    "overrides",
    [
        {"checkpoint": ""},
        {"precision": "int8"},
        {"mask_threshold": float("nan")},
        {"max_hole_area": -1.0},
        {"max_sprinkle_area": float("inf")},
    ],
)
def test_invalid_refiner_settings_fail_before_any_model(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


# --- adapter ------------------------------------------------------------------------------------


def test_the_refiner_satisfies_the_port_with_box_and_point_prompts(tmp_path: Path) -> None:
    refiner, _ = _refiner(tmp_path, GOOD)

    assert isinstance(refiner, RegionRefinement)
    assert refiner.capabilities().prompt_geometries == {
        GroundingGeometry.BOX,
        GroundingGeometry.POINT,
    }
    provenance = refiner.backend_provenance()
    assert (provenance.backend_id, provenance.capability) == ("sam2", "region_refinement")
    assert provenance.configuration_fingerprint == _config().digest


def test_the_runtime_receives_the_verified_image_bytes_and_every_prompt(tmp_path: Path) -> None:
    refiner, runtime = _refiner(tmp_path, GOOD)
    image = _image(tmp_path)

    refiner.refine(_request(refiner, image))

    (call,) = runtime.calls
    assert call["image"] == PIXELS
    assert (call["width"], call["height"]) == (WIDTH, HEIGHT)
    assert call["prompts"] == _prompts(image)
    assert call["config"] == _config()


def test_masks_become_refined_regions_or_explicit_rejections(tmp_path: Path) -> None:
    refiner, _ = _refiner(tmp_path, GOOD)
    image = _image(tmp_path)

    execution = refiner.refine(_request(refiner, image))

    box, point = execution.outcomes
    assert box.region is not None
    assert box.region.provenance == refiner.backend_provenance()
    assert box.region.area_pixels == 3
    assert point.region is None
    assert point.rejection_reason is RefinementRejectionReason.EMPTY_MASK
    assert execution.effective_configuration["checkpoint"] == "sam2.1_hiera_large"


def test_the_sam2_predicted_iou_is_a_native_score_never_a_confidence(tmp_path: Path) -> None:
    refiner, _ = _refiner(tmp_path, GOOD)

    execution = refiner.refine(_request(refiner, _image(tmp_path)))

    (score,) = execution.outcomes[0].native_scores
    assert (score.name, score.value) == ("predicted_iou", 0.93)
    assert "not a calibrated probability" in score.semantics
    assert not hasattr(execution.outcomes[0], "confidence")


def test_an_image_changed_since_the_request_is_refused_before_inference(tmp_path: Path) -> None:
    refiner, runtime = _refiner(tmp_path, GOOD)
    image = _image(tmp_path, sha256="e" * 64)

    with pytest.raises(ValueError, match="sha256"):
        refiner.refine(_request(refiner, image))
    assert runtime.calls == []


def test_a_request_for_another_refiner_configuration_is_refused_before_inference(
    tmp_path: Path,
) -> None:
    refiner, runtime = _refiner(tmp_path, GOOD)
    image = _image(tmp_path)
    request = RegionRefinementRequest(
        perception_result_id=RESULT_ID,
        image=image,
        prompts=_prompts(image),
        configuration_fingerprint="sha256:other",
    )

    with pytest.raises(RefinementRequestError, match="fingerprint"):
        refiner.refine(request)
    assert runtime.calls == []


@pytest.mark.parametrize(
    "masks",
    [
        GOOD[:1],
        (GOOD[0], Sam2PromptedMask(mask=(True,) * 5, predicted_iou=0.5)),
    ],
    ids=["missing-mask", "wrong-shape"],
)
def test_a_runtime_answer_that_breaks_the_seam_is_an_error(
    tmp_path: Path, masks: tuple[Sam2PromptedMask, ...]
) -> None:
    refiner, _ = _refiner(tmp_path, masks)

    with pytest.raises(ValueError, match="SAM2"):
        refiner.refine(_request(refiner, _image(tmp_path)))


def test_a_prompted_mask_score_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        Sam2PromptedMask(mask=_mask(), predicted_iou=float("nan"))


# --- official image-predictor runtime --------------------------------------------------------


class FakePredictor:
    """Records the official ``SAM2ImagePredictor`` calls the runtime makes."""

    def __init__(self, *, mask: np.ndarray | None = None) -> None:
        self.images: list[object] = []
        self.calls: list[dict[str, Any]] = []
        self.mask = mask if mask is not None else np.zeros((HEIGHT, WIDTH), dtype=bool)

    def set_image(self, image: object) -> None:
        self.images.append(image)

    def predict(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls.append(kwargs)
        return self.mask[None, ...], np.array([0.75], dtype=np.float32), np.zeros((1, 4, 4))


def _decoded(payload: bytes) -> np.ndarray:
    assert payload == PIXELS
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def _runtime(
    predictor: FakePredictor, config: Sam2RefinementConfig, contexts: list[str] | None = None
) -> Sam2ImagePredictorRuntime:
    @contextmanager
    def autocast(active: Sam2RefinementConfig) -> Iterator[None]:
        (contexts if contexts is not None else []).append(active.precision)
        yield

    return Sam2ImagePredictorRuntime(
        predictor=predictor,
        config_digest=config.digest,
        image_decoder=_decoded,
        autocast=autocast,
    )


def test_the_predictor_encodes_the_image_once_and_prompts_each_proposal(tmp_path: Path) -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[2:4, 2:5] = True
    predictor = FakePredictor(mask=mask)
    config = _config(precision="bfloat16")
    contexts: list[str] = []
    prompts = _prompts(_image(tmp_path))

    answers = _runtime(predictor, config, contexts).predict_prompts(
        image=PIXELS, width=WIDTH, height=HEIGHT, prompts=prompts, config=config
    )

    assert len(predictor.images) == 1
    assert contexts == ["bfloat16"]
    box_call, point_call = predictor.calls
    assert box_call["box"].tolist() == [1.0, 1.0, 5.0, 4.0]
    assert box_call["multimask_output"] is False
    assert "point_coords" not in box_call
    assert point_call["point_coords"].tolist() == [[8.0, 6.0]]
    assert point_call["point_labels"].tolist() == [1]
    assert point_call["multimask_output"] is False
    assert answers[0].mask == tuple(mask.reshape(-1).tolist())
    assert answers[0].predicted_iou == pytest.approx(0.75)


def test_the_predictor_runtime_refuses_configuration_or_image_drift(tmp_path: Path) -> None:
    config = _config()
    runtime = _runtime(FakePredictor(), config)
    prompts = _prompts(_image(tmp_path))

    with pytest.raises(ValueError, match="configuration digest"):
        runtime.predict_prompts(
            image=PIXELS, width=WIDTH, height=HEIGHT, prompts=prompts, config=_config(device="cuda")
        )
    with pytest.raises(ValueError, match="dimensions"):
        runtime.predict_prompts(
            image=PIXELS, width=WIDTH + 1, height=HEIGHT, prompts=prompts, config=config
        )


def test_the_official_predictor_is_built_from_the_already_loaded_sam2_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[object, dict[str, Any]]] = []

    class OfficialPredictor(FakePredictor):
        def __init__(self, model: object, **kwargs: Any) -> None:
            super().__init__()
            built.append((model, kwargs))

    modules = {"sam2.sam2_image_predictor": SimpleNamespace(SAM2ImagePredictor=OfficialPredictor)}
    monkeypatch.setattr(sam2, "import_module", lambda name: modules[name])
    model = object()
    config = _config(mask_threshold=0.25, max_hole_area=4.0, max_sprinkle_area=2.0)

    Sam2ImagePredictorRuntime.from_model(model=model, config=config)

    assert built == [
        (model, {"mask_threshold": 0.25, "max_hole_area": 4.0, "max_sprinkle_area": 2.0})
    ]
