"""The bounds arithmetic that candidate generation and the geometric predicates share."""

from __future__ import annotations

import math

import pytest

from contextmap.geometric_mapping import Bounds3D
from contextmap.ingestion import FrameId
from contextmap.spatial_relations import AxisDirection
from contextmap.spatial_relations._bounds import (
    axis_overlap_m,
    bounds_gap_m,
    cross_section_axes,
    directed_interval,
)

FRAME = FrameId("map")


def _box(low: tuple[float, float, float], high: tuple[float, float, float]) -> Bounds3D:
    return Bounds3D(frame_id=FRAME, minimum_m=low, maximum_m=high)


def test_overlap_is_the_length_of_the_shared_interval() -> None:
    first = _box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    second = _box((1.0, 0.5, 5.0), (4.0, 1.0, 6.0))
    assert axis_overlap_m(first, second, 0) == 1.0
    assert axis_overlap_m(first, second, 1) == 0.5
    assert axis_overlap_m(first, second, 2) == -3.0


def test_overlap_is_zero_for_touching_faces_and_symmetric() -> None:
    first = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    second = _box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    assert axis_overlap_m(first, second, 0) == 0.0
    assert axis_overlap_m(second, first, 0) == 0.0


def test_gap_is_zero_when_boxes_overlap_or_touch() -> None:
    first = _box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    assert bounds_gap_m(first, _box((1.0, 1.0, 1.0), (3.0, 3.0, 3.0))) == 0.0
    assert bounds_gap_m(first, _box((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))) == 0.0


def test_gap_is_the_euclidean_distance_between_the_closest_faces() -> None:
    first = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    along_one_axis = _box((4.0, 0.0, 0.0), (5.0, 1.0, 1.0))
    diagonal = _box((4.0, 5.0, 0.0), (5.0, 6.0, 1.0))
    assert bounds_gap_m(first, along_one_axis) == 3.0
    assert bounds_gap_m(first, diagonal) == pytest.approx(math.hypot(3.0, 4.0))
    assert bounds_gap_m(diagonal, first) == bounds_gap_m(first, diagonal)


def test_a_directed_interval_reads_the_box_along_a_signed_axis() -> None:
    box = _box((1.0, 2.0, 3.0), (4.0, 6.0, 9.0))
    assert directed_interval(box, AxisDirection.POSITIVE_Z) == (3.0, 9.0)
    assert directed_interval(box, AxisDirection.NEGATIVE_Z) == (-9.0, -3.0)
    assert directed_interval(box, AxisDirection.POSITIVE_X) == (1.0, 4.0)
    assert directed_interval(box, AxisDirection.NEGATIVE_Y) == (-6.0, -2.0)


def test_the_cross_section_of_an_axis_is_the_other_two() -> None:
    assert cross_section_axes(AxisDirection.POSITIVE_Z) == (0, 1)
    assert cross_section_axes(AxisDirection.NEGATIVE_Y) == (0, 2)
    assert cross_section_axes(AxisDirection.POSITIVE_X) == (1, 2)
