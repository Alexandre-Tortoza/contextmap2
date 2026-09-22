"""Axis-aligned box arithmetic shared by candidate retrieval and geometric comparison.

Every function works on :class:`~contextmap.geometric_mapping.Bounds3D` boxes that the caller has
already established to be in one common frame; nothing here infers or converts frames.
"""

from __future__ import annotations

import math

from contextmap.geometric_mapping import Bounds3D


def bounds_gap(first: Bounds3D, second: Bounds3D) -> float:
    """Euclidean distance between two boxes; zero when they overlap or touch."""
    separations = (
        max(0.0, low_b - high_a, low_a - high_b)
        for low_a, high_a, low_b, high_b in zip(
            first.minimum_m, first.maximum_m, second.minimum_m, second.maximum_m, strict=True
        )
    )
    return math.sqrt(sum(value * value for value in separations))


def box_volume(bounds: Bounds3D) -> float:
    """Volume of a box in cubic meters; zero when it is flat on some axis."""
    return math.prod(
        high - low for low, high in zip(bounds.minimum_m, bounds.maximum_m, strict=True)
    )


def intersection_volume(first: Bounds3D, second: Bounds3D) -> float:
    """Volume shared by two boxes in cubic meters; zero when they do not overlap in volume."""
    return math.prod(
        max(0.0, min(high_a, high_b) - max(low_a, low_b))
        for low_a, high_a, low_b, high_b in zip(
            first.minimum_m, first.maximum_m, second.minimum_m, second.maximum_m, strict=True
        )
    )


def extent_ratio(first: Bounds3D, second: Bounds3D) -> float | None:
    """Smallest per-axis ratio between the smaller and the larger side of two boxes.

    Returns:
        A value in ``(0, 1]``; ``1`` for an axis on which both boxes are flat. ``None`` when a
        box is flat on an axis on which the other is not, so the ratio would be zero.
    """
    ratios: list[float] = []
    for low_a, high_a, low_b, high_b in zip(
        first.minimum_m, first.maximum_m, second.minimum_m, second.maximum_m, strict=True
    ):
        side_a, side_b = high_a - low_a, high_b - low_b
        smaller, larger = min(side_a, side_b), max(side_a, side_b)
        if larger == 0.0:
            ratios.append(1.0)
        elif smaller == 0.0:
            return None
        else:
            ratios.append(smaller / larger)
    return min(ratios)
