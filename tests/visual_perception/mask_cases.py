"""Seeded mask scenarios and their pinned outputs: the oracle of the mask representation change.

Issue #593 replaces the ``tuple[bool, ...]`` behind ``InlineMask`` with a NumPy array. The
behaviour must not move: the digests in ``mask_golden.json`` were recorded from the tuple-based
implementation, over the canonical JSON of what normalization, tile remapping and the mask store
produce, so any difference in a region, a rejection, a merge decision, a remapped pixel or a
persisted byte (even an ``int`` that became a ``float``) changes a digest.

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
    height, width = pixels.shape
    return InlineMask(width=width, height=height, data=tuple(bool(value) for value in pixels.flat))


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
