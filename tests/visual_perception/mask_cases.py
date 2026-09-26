"""Seeded mask scenarios and their pinned outputs: the oracle of the mask representation change.

Issue #593 replaces the ``tuple[bool, ...]`` behind ``InlineMask`` with a NumPy array. The
behaviour must not move: the digests in ``mask_golden.json`` were recorded from the tuple-based
implementation, over the canonical JSON of what normalization, tile remapping and the mask store
produce, so any difference in a region, a rejection, a merge decision, a remapped pixel or a
persisted byte (even an ``int`` that became a ``float``) changes a digest.

Issue #608 re-recorded the normalization digests on purpose: the merge representative became
the largest-area member (``largest_area_v1``), and that policy joined the config digest every
normalization digest covers.

``inline_mask`` is the only place these tests build a mask from pixels, so the representation
change touches one helper and never the scenarios or the digests.
"""

from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox,
    ExclusionRegion,
    InlineMask,
    PreparedImage,
    RegionCandidate,
    RegionProvenance,
    ValidRegion,
)
from contextmap.visual_perception.backends.florence2 import (
    Florence2Config,
    Florence2RegionDiscovery,
    TransformersFlorence2Runtime,
)
from contextmap.visual_perception.backends.sam2 import (
    Sam2AutomaticMaskRuntime,
    Sam2Config,
    Sam2RegionDiscovery,
)
from contextmap.visual_perception.backends.sam3 import (
    Sam3Config,
    Sam3ImageProcessorRuntime,
    Sam3RegionDiscovery,
    Sam3Strategy,
)
from contextmap.visual_perception.discovery import DiscoveryInput, DiscoveryPass, PassKind
from contextmap.visual_perception.normalization import NormalizationConfig, NormalizationResult

GOLDEN: dict[str, dict[str, str]] = json.loads(
    Path(__file__).with_name("mask_golden.json").read_text(encoding="utf-8")
)
NORMALIZATION_SEEDS = range(80)
REMAP_SEEDS = range(40)
MASK_STORE_SEEDS = range(12)

_BACKEND = BackendProvenance(
    backend_id="fake",
    capability="region_discovery",
    provider="test-provider",
    model="fake-checkpoint",
    version="1",
    configuration_fingerprint="sha256:fake",
)


def inline_mask(pixels: np.ndarray[Any, Any]) -> InlineMask:
    """Build a mask from a ``(height, width)`` boolean array."""
    return InlineMask(pixels)


def digest(document: object) -> str:
    """SHA-256 of a document's canonical JSON."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def normalization_digest(result: NormalizationResult) -> str:
    """Digest of everything a normalization decides."""
    return digest(
        {
            "regions": [region.to_dict() for region in result.regions],
            "rejected": [item.to_dict() for item in result.rejected],
            "merge_decisions": [item.to_dict() for item in result.merge_decisions],
            "config_digest": result.config_digest,
        }
    )


def normalization_case(
    seed: int,
) -> tuple[tuple[RegionCandidate, ...], PreparedImage, BackendProvenance, NormalizationConfig]:
    """A reproducible mix of masks, boxes, duplicates, nesting, constraints and edge thresholds."""
    rng = random.Random(seed)
    width, height = rng.choice([(20, 14), (17, 11), (24, 9)])
    minimum_area = rng.choice([1.0, 2.0, 6.0])
    config = NormalizationConfig(
        minimum_area_pixels=minimum_area,
        maximum_area_pixels=rng.choice([None, None, 120.0]),
        minimum_valid_fraction=rng.choice([1.0, 0.75, 0.0]),
        maximum_exclusion_fraction=rng.choice([0.0, 0.25, 1.0]),
        # Limiares 0 e 1 incluídos: com 0, todo par casa, mesmo sem nenhum pixel em comum.
        duplicate_iou_threshold=rng.choice([0.8, 0.5, 0.3, 0.0, 1.0]),
        containment_threshold=rng.choice([0.95, 0.6, 1.0, 0.0]),
        maximum_regions=rng.choice([None, None, 3]),
    )
    image = PreparedImage(
        source_observation_id=SourceObservationId("frame-1"),
        payload_reference="outputs/frame.png",
        payload_artifact=ArtifactReference(
            uri="outputs/frame.png",
            sha256=sha256(b"frame").hexdigest(),
            media_type="image/png",
        ),
        width=width,
        height=height,
        transformations=(),
        valid_region=(
            ValidRegion(inline_mask(_rectangle(rng, width, height, large=True)), "valid", "test")
            if rng.random() < 0.4
            else None
        ),
        exclusion_regions=(
            (ExclusionRegion("rig", inline_mask(_rectangle(rng, width, height)), "rig", "test"),)
            if rng.random() < 0.4
            else ()
        ),
    )
    masks: list[np.ndarray[Any, Any]] = []
    candidates: list[RegionCandidate] = []
    for index in range(rng.randrange(0, 14)):
        kind = rng.choice(
            ["rect", "rect", "blob", "copy", "grow", "box", "fraction", "boxed", "misfit", "empty"]
        )
        mask: np.ndarray[Any, Any] | None = None
        box: BoundingBox | None = None
        if kind in ("copy", "grow") and masks:
            mask = rng.choice(masks).copy()
            if kind == "grow":
                for _ in range(rng.randrange(1, 6)):
                    mask[rng.randrange(height), rng.randrange(width)] ^= True
        elif kind == "blob":
            mask = _rectangle(rng, width, height) & (
                np.random.default_rng(seed + index).random((height, width)) < 0.6
            )
        elif kind == "empty":
            mask = np.zeros((height, width), dtype=bool)
        elif kind == "box":
            box = _integer_box(rng, width, height)
        elif kind == "fraction":
            box = _fractional_box(rng, width, height)
        else:
            mask = _rectangle(rng, width, height)
            if kind == "boxed":
                box = _enclosing_box(rng, mask, width, height)
            elif kind == "misfit":
                box = _integer_box(rng, width, height)
        if mask is not None:
            masks.append(mask)
        if mask is None and box is None:
            box = _integer_box(rng, width, height)
        candidates.append(
            RegionCandidate(
                candidate_id=f"c-{rng.randrange(1000):03d}-{index:02d}",
                source_observation_id="frame-1",
                perception_run_id="run-1",
                perception_result_id="result-1",
                image_width=width,
                image_height=height,
                provenance=RegionProvenance(
                    backend_id="fake",
                    backend_version="1",
                    checkpoint="fake-checkpoint",
                    config_digest="sha256:fake",
                    discovery_pass_id=rng.choice(["full-frame", "tile-0001"]),
                    native_proposal_id=f"native-{index}",
                ),
                bounding_box=box,
                mask=None if mask is None else inline_mask(mask),
            )
        )
    return tuple(candidates), image, _BACKEND, config


def remap_case(seed: int) -> tuple[InlineMask, tuple[int, int], tuple[int, int, int, int]]:
    """A tile mask, the window size it is resized to, and the image it is expanded into."""
    rng = random.Random(seed)
    width, height = rng.randrange(1, 13), rng.randrange(1, 10)
    pixels = np.random.default_rng(seed).random((height, width)) < rng.choice([0.1, 0.5, 0.9])
    window_width, window_height = rng.randrange(1, 25), rng.randrange(1, 19)
    image_width = window_width + rng.randrange(0, 7)
    image_height = window_height + rng.randrange(0, 5)
    x_offset = rng.randrange(0, image_width - window_width + 1)
    y_offset = rng.randrange(0, image_height - window_height + 1)
    return (
        inline_mask(pixels),
        (window_width, window_height),
        (image_width, image_height, x_offset, y_offset),
    )


def mask_store_case(seed: int) -> InlineMask:
    """A full-image mask of an awkward size, so packing has a partial last byte."""
    rng = random.Random(seed)
    width, height = rng.randrange(1, 40), rng.randrange(1, 30)
    return inline_mask(np.random.default_rng(seed).random((height, width)) < rng.random())


def _rectangle(
    rng: random.Random, width: int, height: int, *, large: bool = False
) -> np.ndarray[Any, Any]:
    minimum_width, minimum_height = (width // 2, height // 2) if large else (1, 1)
    rect_width = rng.randrange(minimum_width, width + 1)
    rect_height = rng.randrange(minimum_height, height + 1)
    x0 = rng.randrange(0, width - rect_width + 1)
    y0 = rng.randrange(0, height - rect_height + 1)
    pixels = np.zeros((height, width), dtype=bool)
    pixels[y0 : y0 + rect_height, x0 : x0 + rect_width] = True
    return pixels


def _integer_box(rng: random.Random, width: int, height: int) -> BoundingBox:
    x0, y0 = rng.randrange(0, width), rng.randrange(0, height)
    return BoundingBox(x0, y0, rng.randrange(x0 + 1, width + 1), rng.randrange(y0 + 1, height + 1))


def _fractional_box(rng: random.Random, width: int, height: int) -> BoundingBox:
    # Caixas fracionárias, algumas menores que um pixel: nenhum centro de pixel dentro delas.
    x0 = rng.uniform(0, width - 0.2)
    y0 = rng.uniform(0, height - 0.2)
    return BoundingBox(
        x0,
        y0,
        rng.uniform(x0 + 0.1, min(width, x0 + 6)),
        rng.uniform(y0 + 0.1, min(height, y0 + 5)),
    )


def _enclosing_box(
    rng: random.Random, pixels: np.ndarray[Any, Any], width: int, height: int
) -> BoundingBox:
    rows = np.flatnonzero(pixels.any(axis=1))
    columns = np.flatnonzero(pixels.any(axis=0))
    return BoundingBox(
        max(0, int(columns[0]) - rng.randrange(0, 2)),
        max(0, int(rows[0]) - rng.randrange(0, 2)),
        min(width, int(columns[-1]) + 1 + rng.randrange(0, 2)),
        min(height, int(rows[-1]) + 1 + rng.randrange(0, 2)),
    )


# --- Backends: do resultado nativo do SDK ao RegionCandidate -------------------------------

BACKEND_SEEDS = range(24)


class NativeArray:
    """A tensor stand-in that only offers ``tolist()``, like the SDK values the fakes mimic."""

    def __init__(self, value: object) -> None:
        self._value = value

    def tolist(self) -> object:
        return self._value


class _MaterializedImage:
    """A PIL-like image: ``size`` is ``(width, height)``."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size


def _discovery_input(rng: random.Random) -> DiscoveryInput:
    width, height = rng.randrange(6, 26), rng.randrange(5, 19)
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id=SourceObservationId("frame-backend"),
            payload_reference="outputs/backend.png",
            payload_artifact=ArtifactReference(
                uri="outputs/backend.png",
                sha256=sha256(b"backend").hexdigest(),
                media_type="image/png",
            ),
            width=width,
            height=height,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=width, y_max=height),
        ),
        perception_run_id="run-backend",
        perception_result_id="result-backend",
    )


def _image_loader(discovery_input: DiscoveryInput) -> _MaterializedImage:
    discovery_pass = discovery_input.discovery_pass
    return _MaterializedImage((discovery_pass.input_width, discovery_pass.input_height))


def _wrap(rng: random.Random, array: np.ndarray[Any, Any]) -> object:
    """The same values as an array, a nested list, or a ``tolist()``-only object."""
    kind = rng.choice(["array", "list", "native"])
    if kind == "array":
        return array
    return array.tolist() if kind == "list" else NativeArray(array.tolist())


def sam2_candidates(seed: int) -> list[dict[str, object]]:
    """SAM2 automatic-mask records through the official runtime and the adapter."""
    rng = random.Random(seed)
    discovery_input = _discovery_input(rng)
    width = discovery_input.discovery_pass.input_width
    height = discovery_input.discovery_pass.input_height
    records = []
    for index in range(rng.randrange(0, 7)):
        pixels = _rectangle(rng, width, height) & (
            np.random.default_rng(seed * 31 + index).random((height, width)) < 0.8
        )
        # O SDK devolve bool; 0/1 inteiros também são máscaras binárias válidas.
        segmentation = pixels.astype(np.uint8) if rng.random() < 0.3 else pixels
        rows = np.flatnonzero(pixels.any(axis=1))
        columns = np.flatnonzero(pixels.any(axis=0))
        # Convenção do SDK: bbox = [x0, y0, x1 - x0, y1 - y0] com índices inclusivos.
        bbox = (
            [
                float(columns[0]),
                float(rows[0]),
                float(columns[-1] - columns[0]),
                float(rows[-1] - rows[0]),
            ]
            if rows.size
            else [0.0, 0.0, 0.0, 0.0]
        )
        records.append(
            {
                "segmentation": _wrap(rng, segmentation),
                "bbox": bbox,
                "area": int(pixels.sum()),
                "predicted_iou": rng.random(),
                "stability_score": rng.random(),
            }
        )

    class Generator:
        def generate(self, image: object) -> list[dict[str, object]]:
            return records

    config = Sam2Config(
        checkpoint="facebook/sam2-hiera-large",
        predicted_iou_threshold=rng.choice([0.0, 0.5]),
        stability_threshold=rng.choice([0.0, 0.3]),
    )
    runtime = Sam2AutomaticMaskRuntime(
        mask_generator=Generator(), image_loader=_image_loader, config_digest=config.digest
    )
    output = Sam2RegionDiscovery(config=config, runtime=runtime).discover_candidates(
        discovery_input
    )
    return [candidate.to_dict() for candidate in output.candidates]


def sam3_candidates(seed: int) -> list[dict[str, object]]:
    """SAM3 text-prompt processor output through the official runtime and the adapter."""
    rng = random.Random(seed)
    discovery_input = _discovery_input(rng)
    width = discovery_input.discovery_pass.input_width
    height = discovery_input.discovery_pass.input_height
    count = rng.randrange(0, 6)
    generator = np.random.default_rng(seed)
    logits = generator.normal(0.4, 0.3, size=(count, 1, height, width)).astype(np.float32)
    boxes = [
        [
            rng.uniform(-3, width),
            rng.uniform(-3, height),
            rng.uniform(0, width + 3),
            rng.uniform(0, height + 3),
        ]
        for _ in range(count)
    ]
    boxes = [[x0, y0, max(x1, x0 + 0.5), max(y1, y0 + 0.5)] for x0, y0, x1, y1 in boxes]
    mask_threshold = rng.choice([0.5, 0.3, 0.1, 0.7])
    # Pixels exatamente em float32(limiar): em 0.7 ele arredonda para baixo, então o pixel fica
    # abaixo do limiar em float64 e não pode virar primeiro plano por comparação em float32.
    for proposal in range(count):
        for _ in range(3):
            logits[proposal, 0, rng.randrange(height), rng.randrange(width)] = mask_threshold
    masks_key = rng.choice(["masks_logits", "masks_logits", "masks"])
    masks: np.ndarray[Any, Any] = logits
    if masks_key == "masks":
        masks = logits > 0.5
    if rng.random() < 0.5:
        masks = masks[:, 0]  # sem o eixo de canal
    output = {
        "boxes": _wrap(rng, np.array(boxes, dtype=np.float32).reshape(count, 4)),
        "scores": _wrap(rng, generator.random(count).astype(np.float32)),
        masks_key: _wrap(rng, masks),
    }

    class Processor:
        def set_image(self, image: object) -> object:
            return {"image": "encoded"}

        def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
            return state

        def set_text_prompt(self, *, state: object, prompt: str) -> dict[str, object]:
            return output

    config = Sam3Config(
        checkpoint="facebook/sam3",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompt="movable item",
        mask_threshold=mask_threshold,
    )
    runtime = Sam3ImageProcessorRuntime(processor=Processor(), image_loader=_image_loader)
    candidates = Sam3RegionDiscovery(config=config, runtime=runtime).discover_candidates(
        discovery_input
    )
    return [candidate.to_dict() for candidate in candidates.candidates]


def florence2_candidates(seed: int) -> list[dict[str, object]]:
    """Florence-2 polygons and boxes through the official runtime and the adapter."""
    rng = random.Random(seed)
    discovery_input = _discovery_input(rng)
    width = discovery_input.discovery_pass.input_width
    height = discovery_input.discovery_pass.input_height
    polygons: list[object] = []
    for _ in range(rng.randrange(0, 5)):
        contours = [_polygon(rng, width, height) for _ in range(rng.choice([1, 1, 2]))]
        polygons.append(contours[0] if len(contours) == 1 and rng.random() < 0.5 else contours)
    parsed: dict[str, object] = {
        "polygons": polygons,
        "labels": [f"region {index}" for index in range(len(polygons))],
    }

    class Model:
        def generate(self, **kwargs: object) -> object:
            return "generated"

    class Inputs(dict[str, object]):
        def to(self, device: str) -> Inputs:
            return self

    class Processor:
        def __call__(self, *, text: str, images: object, return_tensors: str) -> Inputs:
            return Inputs({"input_ids": "ids", "pixel_values": "pixels"})

        def batch_decode(self, sequences: object, *, skip_special_tokens: bool) -> list[str]:
            return ["<generated>"]

        def post_process_generation(
            self, text: str, *, task: str, image_size: tuple[int, int]
        ) -> dict[str, object]:
            return {task: parsed}

    config = Florence2Config(
        checkpoint="florence-community/Florence-2-base",
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        prompt="movable item",
    )
    runtime = TransformersFlorence2Runtime(
        model=Model(), processor=Processor(), image_loader=_image_loader
    )
    output = Florence2RegionDiscovery(config=config, runtime=runtime).discover_candidates(
        discovery_input
    )
    return [candidate.to_dict() for candidate in output.candidates]


def _polygon(rng: random.Random, width: int, height: int) -> list[float]:
    """A random polygon inside the image; some vertices sit exactly on pixel centers."""
    coordinates: list[float] = []
    for _ in range(rng.randrange(3, 9)):
        if rng.random() < 0.3:
            coordinates += [rng.randrange(width) + 0.5, rng.randrange(height) + 0.5]
        else:
            coordinates += [rng.uniform(0, width), rng.uniform(0, height)]
    # Garante uma caixa com área: um vértice em cada canto oposto de uma faixa mínima.
    coordinates += [min(coordinates[::2]) + 1.0, min(coordinates[1::2]) + 1.0]
    return [
        min(max(value, 0.0), float(bound))
        for value, bound in zip(coordinates, [width, height] * (len(coordinates) // 2), strict=True)
    ]
