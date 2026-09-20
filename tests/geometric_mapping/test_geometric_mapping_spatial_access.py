import dataclasses
import random
from collections.abc import Iterator

import pytest
from map_builders import MAP_ID, accumulate, make_scans, open_geometry

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
    PackedGeometry,
    ScanVoxelPolicy,
    SpatialIndexMetadata,
    geometry_id_for,
)
from contextmap.ingestion import FrameId

MAP = FrameId("map")
NAN = float("nan")


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


def _reference(index: int) -> GeometryReference:
    return GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))


def _references(points: Iterator[GeometryPoint]) -> list[GeometryReference]:
    return [point.reference for point in points]


def _brute_force(geometry: GeometrySource, box: Bounds3D) -> list[GeometryReference]:
    return [
        point.reference
        for point in geometry.iter_geometry()
        if box.contains(point.coordinates_m, frame_id=MAP)
    ]


# --- What the map declares -----------------------------------------------------------------


def test_the_map_declares_that_its_spatial_index_is_derived_from_the_geometry() -> None:
    accumulated, _ = accumulate(make_scans(3))

    assert accumulated.geometric_map.spatial_index == SpatialIndexMetadata(
        kind="scan_bounds", parameters={}, is_derived=True
    )


def test_the_map_bounds_are_the_tight_envelope_of_the_geometry_and_verify_clean() -> None:
    geometry = _corridor()

    coordinates = [point.coordinates_m for point in geometry.iter_geometry()]

    assert geometry.geometric_map.bounds == Bounds3D.enclosing(coordinates, frame_id=MAP)
    assert geometry.verify_index() == []


# --- Queries -------------------------------------------------------------------------------


def test_a_pruned_query_returns_exactly_what_a_brute_force_filter_returns() -> None:
    geometry = _corridor()
    generator = random.Random(11)
    for _ in range(60):
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

        assert _references(geometry.query_bounds(box)) == _brute_force(geometry, box)


def test_scans_whose_bounds_miss_the_box_are_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = _corridor(scan_count=40, points_per_scan=60)
    read: list[tuple[int, int]] = []
    original = PackedGeometry._iter_rows

    def spy(self: PackedGeometry, first: int, stop: int):  # type: ignore[no-untyped-def]
        read.append((first, stop))
        return original(self, first, stop)

    monkeypatch.setattr(PackedGeometry, "_iter_rows", spy)

    # Só os scans 20 a 22 (x de 20 a 22, com ±2 m de nuvem) alcançam esta caixa.
    found = list(geometry.query_bounds(_box((21.0, -1.0, 0.0), (21.5, 1.0, 2.0))))

    assert found
    assert read and all(first >= 17 * 60 and stop <= 25 * 60 for first, stop in read)
    assert sum(stop - first for first, stop in read) < 8 * 60


def test_a_box_that_holds_a_whole_scan_returns_every_point_of_it() -> None:
    geometry = _corridor(scan_count=5)
    everything = _box((-10.0, -10.0, -10.0), (50.0, 10.0, 10.0))

    assert _references(geometry.query_bounds(everything)) == [
        point.reference for point in geometry.iter_geometry()
    ]


def test_a_scan_that_contributed_no_geometry_is_skipped() -> None:
    empty = make_scans(1, points=((NAN, NAN, NAN),))[0]
    accumulated, packed = accumulate([empty, *make_scans(2, first_scan=1)])
    geometry = open_geometry(accumulated, packed)

    assert len(list(geometry.query_bounds(accumulated.geometric_map.bounds))) == 4
    assert geometry.verify_index() == []


def test_an_aggregated_map_is_queried_and_verified_like_any_other() -> None:
    points = (*((1.0 + 0.01 * i, 0.0, 0.0) for i in range(20)), (9.0, 0.0, 0.0))
    accumulated, packed = accumulate(
        make_scans(3, points=points), aggregation=ScanVoxelPolicy(cell_m=0.5)
    )
    geometry = open_geometry(accumulated, packed)
    box = _box((0.0, -1.0, 0.0), (3.0, 1.0, 1.0))

    assert _references(geometry.query_bounds(box)) == _brute_force(geometry, box)
    assert geometry.verify_index() == []


# --- The index is derived, never authoritative ----------------------------------------------


def test_a_corrupt_scan_bounds_entry_is_reported() -> None:
    accumulated, packed = accumulate(make_scans(3))
    first, second, third = accumulated.scans
    assert second.bounds is not None
    shrunk = dataclasses.replace(
        second,
        bounds=Bounds3D(frame_id=MAP, minimum_m=(50.0, 50.0, 50.0), maximum_m=(51.0, 51.0, 51.0)),
    )
    geometry = PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=(first, shrunk, third), records=packed
    )

    problems = geometry.verify_index()

    assert len(problems) == 1 and "scan 1" in problems[0]


def test_a_corrupt_index_never_changes_the_geometry_it_indexes() -> None:
    accumulated, packed = accumulate(make_scans(3))
    first, second, third = accumulated.scans
    wrong = dataclasses.replace(
        second,
        bounds=Bounds3D(frame_id=MAP, minimum_m=(50.0, 50.0, 50.0), maximum_m=(51.0, 51.0, 51.0)),
    )
    honest = open_geometry(accumulated, packed)
    corrupt = PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=(first, wrong, third), records=packed
    )

    assert list(corrupt.iter_geometry()) == list(honest.iter_geometry())
    reference = GeometryReference(
        map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=3)
    )
    assert corrupt.get(reference) == honest.get(reference)


def test_map_bounds_that_disagree_with_the_geometry_are_reported() -> None:
    accumulated, packed = accumulate(make_scans(3))
    wrong_map = dataclasses.replace(
        accumulated.geometric_map,
        bounds=Bounds3D(frame_id=MAP, minimum_m=(0.0, 0.0, 0.0), maximum_m=(1.0, 1.0, 1.0)),
    )
    geometry = PackedGeometry(geometric_map=wrong_map, scans=accumulated.scans, records=packed)

    problems = geometry.verify_index()

    assert any("map bounds" in problem for problem in problems)


def test_rebuilding_the_index_keeps_ids_and_coordinates_and_restores_the_entries() -> None:
    accumulated, packed = accumulate(make_scans(4))
    honest = open_geometry(accumulated, packed)
    damaged = tuple(
        dataclasses.replace(
            scan, bounds=Bounds3D(frame_id=MAP, minimum_m=(9.0,) * 3, maximum_m=(9.5,) * 3)
        )
        for scan in accumulated.scans
    )
    corrupt = PackedGeometry(geometric_map=accumulated.geometric_map, scans=damaged, records=packed)

    rebuilt = corrupt.rebuild_scan_bounds()
    reopened = PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=rebuilt, records=packed
    )

    assert rebuilt == accumulated.scans
    assert reopened.verify_index() == []
    assert list(reopened.iter_geometry()) == list(honest.iter_geometry())


# --- Callers do not depend on the index -----------------------------------------------------


class _ListGeometry:
    """A geometry source with no index at all, only the port."""

    def __init__(self, source: PackedGeometry) -> None:
        self._points = list(source.iter_geometry())
        self._map = source.geometric_map

    @property
    def geometric_map(self) -> GeometricMap:
        return self._map

    def get(self, reference: GeometryReference) -> GeometryPoint:
        for point in self._points:
            if point.reference == reference:
                return point
        raise KeyError(reference)

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        return iter(self._points)

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        return (
            p for p in self._points if bounds.contains(p.coordinates_m, frame_id=bounds.frame_id)
        )


def test_a_caller_gets_the_same_answers_from_any_geometry_source() -> None:
    packed = _corridor(scan_count=12, points_per_scan=25)
    plain = _ListGeometry(packed)
    box = _box((3.0, -1.0, 0.2), (7.5, 0.5, 1.8))

    for source in (packed, plain):
        assert isinstance(source, GeometrySource)
    assert _references(packed.query_bounds(box)) == _references(plain.query_bounds(box))
    assert _references(packed.iter_geometry()) == _references(plain.iter_geometry())
    reference = GeometryReference(
        map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=5)
    )
    assert packed.get(reference) == plain.get(reference)


def test_the_boundary_returns_geometry_not_index_entries() -> None:
    geometry = _corridor(scan_count=3)

    found = list(geometry.query_bounds(geometry.geometric_map.bounds))

    assert all(isinstance(point, GeometryPoint) for point in found)
    assert len(found) == geometry.geometric_map.point_count
