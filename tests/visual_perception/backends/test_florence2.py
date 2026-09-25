from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BoundingBox,
    NativeRegionText,
    PreparedImage,
    Region2D,
    RegionDiscovery,
)
from contextmap.visual_perception.backends.florence2 import (
    Florence2Config,
    Florence2NativeOutput,
    Florence2NativeRegion,
    Florence2RegionDiscovery,
    TransformersFlorence2Runtime,
)
from contextmap.visual_perception.discovery import DiscoveryInput, DiscoveryPass, PassKind


@dataclass(frozen=True)
class MaterializedImage:
    size: tuple[int, int]
    token: tuple[str, str]


def _materialized_image(discovery_input: DiscoveryInput) -> MaterializedImage:
    discovery_pass = discovery_input.discovery_pass
    return MaterializedImage(
        size=(discovery_pass.input_width, discovery_pass.input_height),
        token=("image", discovery_pass.pass_id),
    )


def _input() -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id=SourceObservationId("frame-florence"),
            payload_reference="outputs/florence.png",
            payload_artifact=ArtifactReference(
                uri="outputs/florence.png",
                sha256=sha256(b"florence").hexdigest(),
                media_type="image/png",
            ),
            width=6,
            height=4,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=6, y_max=4),
        ),
        perception_run_id="run-florence",
        perception_result_id="result-florence",
    )


class FakeFlorence2Runtime:
    def predict(
        self, discovery_input: DiscoveryInput, config: Florence2Config
    ) -> Florence2NativeOutput:
        return Florence2NativeOutput(
            regions=(
                Florence2NativeRegion(
                    proposal_id="box-1",
                    box=(1.0, 1.0, 3.0, 3.0),
                    score=None,
                    parsed_text="a requested region",
                ),
                Florence2NativeRegion(
                    proposal_id="mask-2",
                    box=(3.0, 0.0, 5.0, 2.0),
                    score=0.9,
                    mask=(False, False, False, True, True, False)
                    + (False, False, False, True, True, False)
                    + (False,) * 12,
                    parsed_text="second parsed region",
                ),
            ),
            warnings=(),
            parsing_diagnostics=(("unparsed_token_count", 0),),
        )


def test_florence2_normalizes_boxes_and_masks_without_semantic_promotion() -> None:
    config = Florence2Config(
        checkpoint="microsoft/Florence-2-large",
        model_version="1.0",
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        prompt="visible regions",
        device="cuda:0",
        generation_settings=(("num_beams", 3),),
    )
    backend = Florence2RegionDiscovery(config=config, runtime=FakeFlorence2Runtime())

    output = backend.discover_candidates(_input())

    assert len(output.candidates) == 2
    box_candidate, mask_candidate = output.candidates
    assert box_candidate.mask is None
    assert box_candidate.score is None
    assert box_candidate.native_text == NativeRegionText(
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        prompt="visible regions",
        text="a requested region",
    )
    assert mask_candidate.mask is not None
    assert mask_candidate.score is not None
    assert mask_candidate.provenance.backend_id == "florence2_region_discovery"
    assert mask_candidate.provenance.query == (
        "<REFERRING_EXPRESSION_SEGMENTATION>:visible regions"
    )
    assert dict(output.diagnostics.metadata)["task"] == config.task
    assert dict(output.diagnostics.metadata)["unparsed_token_count"] == 0
    assert "hypothesis" not in box_candidate.to_dict()
    json.dumps(box_candidate.to_dict())
    assert isinstance(backend, RegionDiscovery)
    regions = backend.discover(_input().prepared_image)
    assert all(isinstance(region, Region2D) for region in regions)
    assert regions[0].provenance == backend.backend_provenance()


def test_florence2_configuration_requires_explicit_region_task() -> None:
    with pytest.raises(ValueError, match="task"):
        Florence2Config(checkpoint="florence", task="")
    with pytest.raises(ValueError, match="box_threshold"):
        Florence2Config(checkpoint="florence", task="region", box_threshold=2.0)
    with pytest.raises(ValueError, match="region-producing task"):
        Florence2Config(checkpoint="florence", task="<CAPTION>")


def test_florence2_invalid_native_mask_fails_with_parsing_context() -> None:
    class InvalidRuntime:
        def predict(
            self, discovery_input: DiscoveryInput, config: Florence2Config
        ) -> Florence2NativeOutput:
            return Florence2NativeOutput(
                regions=(
                    Florence2NativeRegion(
                        proposal_id="bad-mask",
                        box=(0.0, 0.0, 2.0, 2.0),
                        score=0.8,
                        mask=(True,),
                    ),
                )
            )

    backend = Florence2RegionDiscovery(
        config=Florence2Config(checkpoint="florence", task="<REGION_PROPOSAL>"),
        runtime=InvalidRuntime(),
    )

    with pytest.raises(ValueError, match=r"bad-mask.*mask length"):
        backend.discover_candidates(_input())


class ModelInputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__({"input_ids": "ids", "pixel_values": "pixels"})
        self.devices: list[str] = []

    def to(self, device: str) -> ModelInputs:
        self.devices.append(device)
        return self


class FlorenceModel:
    def __init__(self) -> None:
        self.received: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        self.received.append(kwargs)
        return "generated-token-ids"


class FlorenceProcessor:
    def __init__(self, parsed: dict[str, object]) -> None:
        self.parsed = parsed
        self.inputs = ModelInputs()
        self.calls: list[tuple[str, object]] = []

    def __call__(self, *, text: str, images: object, return_tensors: str) -> ModelInputs:
        self.calls.append(("encode", (text, images, return_tensors)))
        return self.inputs

    def batch_decode(self, sequences: object, *, skip_special_tokens: bool) -> list[str]:
        self.calls.append(("decode", (sequences, skip_special_tokens)))
        return ["<generated>native Florence output</generated>"]

    def post_process_generation(
        self, text: str, *, task: str, image_size: tuple[int, int]
    ) -> dict[str, object]:
        self.calls.append(("parse", (text, task, image_size)))
        return {task: self.parsed}


def test_official_florence2_runtime_executes_and_parses_region_boxes() -> None:
    processor = FlorenceProcessor(
        {
            "bboxes": [[1.0, 1.0, 4.0, 3.0]],
            "labels": ["movable item"],
            "scores": [0.83],
        }
    )
    model = FlorenceModel()
    runtime = TransformersFlorence2Runtime(
        model=model,
        processor=processor,
        image_loader=_materialized_image,
    )
    config = Florence2Config(
        checkpoint="florence-community/Florence-2-base",
        task="<REGION_PROPOSAL>",
        device="cuda:0",
        generation_settings=(("max_new_tokens", 128), ("num_beams", 3)),
    )

    output = runtime.predict(_input(), config)

    assert processor.inputs.devices == ["cuda:0"]
    assert model.received == [
        {
            "input_ids": "ids",
            "pixel_values": "pixels",
            "max_new_tokens": 128,
            "num_beams": 3,
        }
    ]
    assert output.regions[0].box == (1.0, 1.0, 4.0, 3.0)
    assert output.regions[0].score == 0.83
    assert output.regions[0].parsed_text == "movable item"
    assert dict(output.parsing_diagnostics)["box_count"] == 1
    assert processor.calls[-1] == (
        "parse",
        ("<generated>native Florence output</generated>", "<REGION_PROPOSAL>", (6, 4)),
    )


def test_official_florence2_runtime_rasterizes_parsed_polygons() -> None:
    processor = FlorenceProcessor(
        {
            "polygons": [[[1.0, 1.0, 4.0, 1.0, 4.0, 3.0, 1.0, 3.0]]],
            "labels": ["requested geometry"],
        }
    )
    runtime = TransformersFlorence2Runtime(
        model=FlorenceModel(),
        processor=processor,
        image_loader=_materialized_image,
    )
    config = Florence2Config(
        checkpoint="florence-community/Florence-2-base",
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        prompt="movable item",
    )

    output = runtime.predict(_input(), config)

    region = output.regions[0]
    assert region.box == (1.0, 1.0, 4.0, 3.0)
    assert region.mask is not None
    assert sum(region.mask) == 6
    assert region.parsed_text == "requested geometry"
    assert dict(output.parsing_diagnostics)["polygon_count"] == 1


_CHAIR_BOX = [0.0, 0.0, 2.0, 2.0]
_TABLE_BOX = [3.0, 1.0, 5.0, 3.0]
_SQUARE_POLYGON = [[[3.0, 1.0, 5.0, 1.0, 5.0, 3.0, 3.0, 3.0]]]


@pytest.mark.parametrize(
    ("task", "prompt", "parsed", "expected"),
    [
        pytest.param(
            "<REGION_PROPOSAL>",
            None,
            {"bboxes": [_CHAIR_BOX, _TABLE_BOX], "labels": ["", ""]},
            [("florence2-box-000000", None), ("florence2-box-000001", None)],
            id="region-proposal-geometry-only",
        ),
        pytest.param(
            "<OD>",
            None,
            {"bboxes": [_CHAIR_BOX, _TABLE_BOX], "labels": ["chair", "table"]},
            [("florence2-box-000000", "chair"), ("florence2-box-000001", "table")],
            id="od-boxes-and-labels",
        ),
        pytest.param(
            "<DENSE_REGION_CAPTION>",
            None,
            {"bboxes": [_CHAIR_BOX, _TABLE_BOX], "labels": ["a wooden chair", "a low table"]},
            [
                ("florence2-box-000000", "a wooden chair"),
                ("florence2-box-000001", "a low table"),
            ],
            id="dense-region-caption-descriptions",
        ),
        pytest.param(
            "<OPEN_VOCABULARY_DETECTION>",
            "chair",
            {
                "bboxes": [_CHAIR_BOX],
                "bboxes_labels": ["chair"],
                "polygons": [],
                "polygons_labels": [],
            },
            [("florence2-box-000000", "chair")],
            id="open-vocabulary-boxes",
        ),
        pytest.param(
            "<OPEN_VOCABULARY_DETECTION>",
            "table",
            {
                "bboxes": [],
                "bboxes_labels": [],
                "polygons": _SQUARE_POLYGON,
                "polygons_labels": ["table"],
            },
            [("florence2-polygon-000000", "table")],
            id="open-vocabulary-polygons",
        ),
        pytest.param(
            "<REFERRING_EXPRESSION_SEGMENTATION>",
            "the low table",
            {"polygons": _SQUARE_POLYGON, "labels": [""]},
            [("florence2-polygon-000000", None)],
            id="referring-segmentation-geometry-only",
        ),
    ],
)
def test_task_native_text_is_typed_evidence_of_its_own_proposal_and_task(
    task: str,
    prompt: str | None,
    parsed: dict[str, object],
    expected: list[tuple[str, str | None]],
) -> None:
    config = Florence2Config(
        checkpoint="florence-community/Florence-2-large", task=task, prompt=prompt
    )
    runtime = TransformersFlorence2Runtime(
        model=FlorenceModel(), processor=FlorenceProcessor(parsed), image_loader=_materialized_image
    )
    backend = Florence2RegionDiscovery(config=config, runtime=runtime)

    candidates = backend.discover_candidates(_input()).candidates

    assert [
        (
            candidate.provenance.native_proposal_id,
            None if candidate.native_text is None else candidate.native_text.text,
        )
        for candidate in candidates
    ] == expected
    for candidate in candidates:
        assert "parsed_text" not in dict(candidate.native_metadata)
        if candidate.native_text is not None:
            assert candidate.native_text == NativeRegionText(
                task=task, prompt=prompt, text=candidate.native_text.text
            )
        if candidate.provenance.native_proposal_id.startswith("florence2-polygon"):
            assert candidate.mask is not None


class _TextToggledRuntime:
    def __init__(self, *, with_text: bool) -> None:
        self._with_text = with_text

    def predict(
        self, discovery_input: DiscoveryInput, config: Florence2Config
    ) -> Florence2NativeOutput:
        return Florence2NativeOutput(
            regions=tuple(
                Florence2NativeRegion(
                    proposal_id=f"florence2-box-{index:06d}",
                    box=box,
                    score=None,
                    parsed_text=label if self._with_text else None,
                )
                for index, (box, label) in enumerate(
                    (((0.0, 0.0, 2.0, 2.0), "chair"), ((3.0, 1.0, 5.0, 3.0), "table"))
                )
            )
        )


def test_downstream_consumers_can_ignore_native_text_entirely() -> None:
    config = Florence2Config(checkpoint="florence-community/Florence-2-large", task="<OD>")
    image = _input().prepared_image

    with_text = Florence2RegionDiscovery(
        config=config, runtime=_TextToggledRuntime(with_text=True)
    ).discover(image)
    without_text = Florence2RegionDiscovery(
        config=config, runtime=_TextToggledRuntime(with_text=False)
    ).discover(image)

    assert with_text == without_text
    assert all("chair" not in json.dumps(region.to_dict()) for region in with_text)
