import random

import numpy as np
import pytest
from map_builders import MAP_ID, accumulate, make_scans, open_geometry

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryBlock,
    GeometryBlockSource,
    GeometryPoint,
    GeometryReference,
    PackedGeometry,
    geometry_id_for,
    geometry_index_of,
)
from contextmap.ingestion import FrameId

MAP = FrameId("map")


def _box(minimum: tuple[float, float, float], maximum: tuple[float, float, float]) -> Bounds3D:
    return Bounds3D(frame_id=MAP, minimum_m=minimum, maximum_m=maximum)


def _corridor(scan_count: int = 40, points_per_scan: int = 60) -> PackedGeometry:
    """Scans along +x, one meter apart, each covering a random cloud around its pose."""
    generator = random.Random(7)
    points = tuple(
        (generator.uniform(-2.0, 2.0), generator.uniform(-1.0, 1.0), generator.uniform(0.0, 2.0))
        for _ in range(points_per_scan)
    )
    return open_geometry(*accumulate(make_scans(scan_count, points=points)))


def _gathered(
    geometry: PackedGeometry, *, bounds: Bounds3D | None = None, block_points: int = 4096
) -> tuple[list[int], np.ndarray]:
    blocks = list(geometry.iter_blocks(bounds=bounds, block_points=block_points))
    if not blocks:
        return [], np.empty((0, 3), dtype=np.float64)
    indices = np.concatenate([block.indices for block in blocks])
    coordinates = np.concatenate([block.coordinates_m for block in blocks])
    return [int(index) for index in indices], coordinates


def _index_of(point: GeometryPoint) -> int:
    return geometry_index_of(map_id=MAP_ID, geometry_id=point.geometry_id)


# --- The port ------------------------------------------------------------------------------


def test_the_packed_geometry_is_a_block_source() -> None:
    assert isinstance(_corridor(scan_count=2), GeometryBlockSource)


def test_every_block_declares_the_map_and_frame_its_coordinates_belong_to() -> None:
    geometry = _corridor(scan_count=3, points_per_scan=10)

    blocks = list(geometry.iter_blocks())

    assert blocks
    for block in blocks:
        assert block.map_id == geometry.geometric_map.map_id
        assert block.frame_id == geometry.geometric_map.frame_id


# --- Identity is global, never the row ------------------------------------------------------


def test_a_block_carries_the_global_index_of_every_row_not_its_own_position() -> None:
    geometry = _corridor(scan_count=6, points_per_scan=10)

    # Uma caixa que só alcança os scans do meio: nenhuma linha começa no índice 0.
    indices, coordinates = _gathered(geometry, bounds=_box((3.0, -1.0, 0.0), (3.2, 1.0, 2.0)))

    assert indices and min(indices) > 0
    for row, index in enumerate(indices):
        reference = GeometryReference(
            map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index)
        )
        np.testing.assert_array_equal(coordinates[row], geometry.get(reference).coordinates_m)


# --- Equivalence with the per-point boundary -----------------------------------------------


def test_blocks_hold_exactly_the_geometry_the_per_point_query_returns_in_the_same_order() -> None:
    geometry = _corridor()
    generator = random.Random(11)
    for _ in range(40):
        low = (
            generator.uniform(-3.0, 43.0),
            generator.uniform(-1.5, 1.0),
            generator.uniform(0.0, 1.5),
        )
        size = (
            generator.uniform(0.1, 6.0),
            generator.uniform(0.1, 3.0),
            generator.uniform(0.1, 3.0),
        )
        box = _box(low, (low[0] + size[0], low[1] + size[1], low[2] + size[2]))
        expected = list(geometry.query_bounds(box))

        indices, coordinates = _gathered(geometry, bounds=box)

        assert indices == [_index_of(point) for point in expected]
        np.testing.assert_array_equal(
            coordinates,
            np.array([point.coordinates_m for point in expected], dtype=np.float64).reshape(-1, 3),
        )


def test_without_bounds_the_blocks_cover_the_whole_map_in_index_order() -> None:
    geometry = _corridor(scan_count=5, points_per_scan=8)

    indices, coordinates = _gathered(geometry, block_points=7)

    assert indices == list(range(geometry.geometric_map.point_count))
    np.testing.assert_array_equal(
        coordinates,
        np.array([point.coordinates_m for point in geometry.iter_geometry()], dtype=np.float64),
    )


def test_a_box_that_reaches_no_scan_yields_no_block() -> None:
    geometry = _corridor(scan_count=3, points_per_scan=8)

    far_away = _box((500.0, 500.0, 500.0), (501.0, 501.0, 501.0))

    assert list(geometry.iter_blocks(bounds=far_away)) == []


# --- Bounded blocks ------------------------------------------------------------------------


def test_no_block_is_larger_than_the_requested_size() -> None:
    geometry = _corridor(scan_count=5, points_per_scan=8)

    blocks = list(geometry.iter_blocks(block_points=7))

    assert blocks and all(len(block) <= 7 for block in blocks)
    assert sum(len(block) for block in blocks) == geometry.geometric_map.point_count


def test_a_block_size_that_is_not_positive_is_rejected() -> None:
    geometry = _corridor(scan_count=2, points_per_scan=4)

    with pytest.raises(ValueError, match="block_points"):
        list(geometry.iter_blocks(block_points=0))


def test_bounds_in_another_frame_are_never_reinterpreted() -> None:
    geometry = _corridor(scan_count=2, points_per_scan=4)
    elsewhere = Bounds3D(
        frame_id=FrameId("odom"), minimum_m=(-1.0, -1.0, -1.0), maximum_m=(1.0, 1.0, 1.0)
    )

    with pytest.raises(ValueError, match="frame"):
        list(geometry.iter_blocks(bounds=elsewhere))


# --- The contract of the block itself ------------------------------------------------------


def test_a_block_whose_arrays_disagree_is_rejected() -> None:
    with pytest.raises(ValueError, match="one index per row"):
        GeometryBlock(
            map_id=MAP_ID,
            frame_id=MAP,
            indices=np.array([0, 1], dtype=np.int64),
            coordinates_m=np.zeros((3, 3)),
        )


def test_a_block_whose_coordinates_are_not_three_dimensional_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        GeometryBlock(
            map_id=MAP_ID,
            frame_id=MAP,
            indices=np.array([0], dtype=np.int64),
            coordinates_m=np.zeros((1, 2)),
        )


def test_a_block_whose_indices_are_not_integers_is_rejected() -> None:
    """A float index would be truncated by ``reference()``, naming a different element."""
    with pytest.raises(ValueError, match="integer array"):
        GeometryBlock(
            map_id=MAP_ID,
            frame_id=MAP,
            indices=np.array([1.5, 2.5]),
            coordinates_m=np.zeros((2, 3)),
        )


def test_every_block_a_packed_map_yields_carries_integer_indices() -> None:
    geometry = _corridor(scan_count=3, points_per_scan=8)

    for block in geometry.iter_blocks(block_points=5):
        assert np.issubdtype(block.indices.dtype, np.integer)
