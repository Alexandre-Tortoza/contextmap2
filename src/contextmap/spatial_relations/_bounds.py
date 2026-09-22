"""Arithmetic on axis-aligned bounds shared by candidate generation and the geometric predicates.

Both stages reason about the same boxes, so the rules that read them (how far apart two boxes are,
how much their intervals share along an axis, how a box looks along a signed direction) live here
once. Every function is pure, works in the frame of the boxes it is given, and never reads a label:
callers are responsible for having checked that both boxes are expressed in one frame.
"""

from __future__ import annotations

import math

from contextmap.geometric_mapping import Bounds3D
from contextmap.spatial_relations.frame_conventions import AxisDirection


def axis_overlap_m(first: Bounds3D, second: Bounds3D, axis: int) -> float:
    """Measure how much two boxes share along one axis, in meters.

    Args:
        first: One box.
        second: The other box, in the same frame.
        axis: ``0`` for x, ``1`` for y and ``2`` for z.

    Returns:
        The length of the shared interval: positive when the intervals overlap, ``0.0`` when they
        only touch, and the negative of the separation when they are apart.
    """
    return min(first.maximum_m[axis], second.maximum_m[axis]) - max(
        first.minimum_m[axis], second.minimum_m[axis]
    )


def bounds_gap_m(first: Bounds3D, second: Bounds3D) -> float:
    """Measure the distance between the closest faces of two boxes, in meters.

    Args:
        first: One box.
        second: The other box, in the same frame.

    Returns:
        ``0.0`` when the boxes overlap or touch, otherwise the Euclidean distance between the
        closest points of the two boxes.
    """
    separations = (max(0.0, -axis_overlap_m(first, second, axis)) for axis in range(3))
    return math.sqrt(sum(separation * separation for separation in separations))


def directed_interval(bounds: Bounds3D, direction: AxisDirection) -> tuple[float, float]:
    """Read a box along a signed axis, so that larger values are further in that direction.

    Args:
        bounds: The box.
        direction: The signed axis to read along.

    Returns:
        ``(low, high)`` of the box measured in the direction: the coordinates themselves for a
        positive axis and their negatives, swapped, for a negative one.
    """
    low = bounds.minimum_m[direction.axis_index]
    high = bounds.maximum_m[direction.axis_index]
    if direction.sign == 1:
        return (low, high)
    return (-high, -low)


def cross_section_axes(direction: AxisDirection) -> tuple[int, int]:
    """Name the two axes perpendicular to a direction.

    Args:
        direction: A signed axis.

    Returns:
        The indexes of the other two axes, in increasing order.
    """
    first, second = (axis for axis in range(3) if axis != direction.axis_index)
    return (first, second)
