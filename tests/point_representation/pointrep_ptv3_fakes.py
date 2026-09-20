"""A deterministic PTv3 runtime double: the adapter is tested with no torch or CUDA."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from contextmap.point_representation.backends.ptv3 import PTv3Config, PTv3Inference
from contextmap.shared import Vector3

CHECKPOINT_HASH = "sha256:" + "ab" * 32


def make_config(**overrides: Any) -> PTv3Config:
    values: dict[str, Any] = {
        "variant": "ptv3-base",
        "checkpoint": "ptv3-scannet-v1",
        "checkpoint_hash": CHECKPOINT_HASH,
        "device": "cuda:0",
        "precision": "float32",
        "grid_size_m": 0.05,
        "output_dimension": 8,
        "pooling": "center",
        "normalization": "none",
        "min_support_points": 3,
    }
    values.update(overrides)
    return PTv3Config(**values)


class FakePTv3Runtime:
    """Returns a vector that is a pure function of the coordinates and the pooling."""

    def __init__(self, *, peak_memory_bytes: int | None = 1_000_000) -> None:
        self.calls: list[tuple[tuple[Vector3, ...], int, PTv3Config]] = []
        self._peak_memory_bytes = peak_memory_bytes

    def infer(
        self, *, coordinates_m: Sequence[Vector3], center_index: int, config: PTv3Config
    ) -> PTv3Inference:
        self.calls.append((tuple(coordinates_m), center_index, config))
        if config.pooling == "center":
            anchor = coordinates_m[center_index]
        else:
            count = len(coordinates_m)
            anchor = (
                sum(p[0] for p in coordinates_m) / count,
                sum(p[1] for p in coordinates_m) / count,
                sum(p[2] for p in coordinates_m) / count,
            )
        vector = tuple(
            anchor[index % 3] * (index + 1) + 0.01 * len(coordinates_m)
            for index in range(config.output_dimension)
        )
        return PTv3Inference(vector=vector, peak_memory_bytes=self._peak_memory_bytes)
