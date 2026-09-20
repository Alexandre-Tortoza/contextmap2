"""In-memory Geometric Mapping fixtures for Point Representation tests.

Two independent ``GeometrySource`` implementations answer the same queries with
different index behavior, so tests can show that support extraction depends
only on the public boundary and never on how a map is indexed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryPoint,
    GeometryReference,
    MapId,
    TransformLineage,
    geometry_id_for,
)
from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import SourceTimestamp, Vector3
from contextmap.state_estimation import (
    LookupPolicy,
    StateEstimationRunId,
    TimeBounds,
    TrajectoryId,
)

MAP_ID = MapId("map-0001")
MAP_FRAME = "map"
_CLOCK = "fixture:header"


def _timestamp(nanoseconds: int) -> SourceTimestamp:
    return SourceTimestamp(seconds=0, nanoseconds=nanoseconds, clock_id=_CLOCK)


def make_geometry_point(
    index: int,
    coordinates_m: Vector3,
    *,
    map_id: MapId = MAP_ID,
    map_frame: str = MAP_FRAME,
) -> GeometryPoint:
    """A measured point already expressed in the map frame (no transform steps)."""
    return GeometryPoint(
        geometry_id=geometry_id_for(map_id=map_id, index=index),
        map_id=map_id,
        map_frame=FrameId(map_frame),
        coordinates_m=coordinates_m,
        source_frame=FrameId(map_frame),
        source_coordinates_m=coordinates_m,
        source_observation_id=SourceObservationId("lidar-frame-00001"),
        source_point_index=index,
        acquisition_timestamp=_timestamp(1),
        transform_lineage=TransformLineage(steps=()),
    )


def make_geometric_map(
    points: Sequence[GeometryPoint],
    *,
    map_id: MapId = MAP_ID,
    frame: str = MAP_FRAME,
    bounds: Bounds3D | None = None,
) -> GeometricMap:
    return GeometricMap(
        map_id=map_id,
        frame_id=FrameId(frame),
        point_count=len(points),
        bounds=bounds
        if bounds is not None
        else Bounds3D.enclosing((p.coordinates_m for p in points), frame_id=FrameId(frame)),
        source_observation_ids=(SourceObservationId("lidar-frame-00001"),),
        time_bounds=TimeBounds(start=_timestamp(1), end=_timestamp(2)),
        spatial_index=None,
        provenance=GeometricMapProvenance(
            sequence_artifact_id=SequenceArtifactId("sequence-0001"),
            selection_id="full-sequence",
            trajectory_id=TrajectoryId("run-0001--trajectory"),
            state_estimation_run_id=StateEstimationRunId("run-0001"),
            calibration_identity="sha256:calibration",
            pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=250_000_000),
        ),
    )


class LinearScanSource:
    """A ``GeometrySource`` that answers box queries by scanning every point."""

    def __init__(
        self,
        points: Sequence[GeometryPoint],
        *,
        map_id: MapId = MAP_ID,
        frame: str = MAP_FRAME,
        bounds: Bounds3D | None = None,
    ) -> None:
        self._points = tuple(points)
        self._by_id = {point.geometry_id: point for point in self._points}
        self._map = make_geometric_map(self._points, map_id=map_id, frame=frame, bounds=bounds)
        self.queries: list[Bounds3D] = []

    @property
    def geometric_map(self) -> GeometricMap:
        return self._map

    def get(self, reference: GeometryReference) -> GeometryPoint:
        if reference.map_id != self._map.map_id or reference.geometry_id not in self._by_id:
            raise KeyError(reference)
        return self._by_id[reference.geometry_id]

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        return iter(self._points)

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        if bounds.frame_id != self._map.frame_id:
            raise ValueError("query bounds are expressed in another frame")
        self.queries.append(bounds)
        return (
            point
            for point in self._points
            if bounds.contains(point.coordinates_m, frame_id=point.map_frame)
        )


class GridIndexedSource(LinearScanSource):
    """A ``GeometrySource`` backed by a cell hash, yielding results in cell order."""

    def __init__(self, points: Sequence[GeometryPoint], *, cell_size_m: float = 0.5) -> None:
        super().__init__(points)
        self._cell_size_m = cell_size_m
        self._cells: dict[tuple[int, int, int], list[GeometryPoint]] = {}
        for point in reversed(self._points):
            self._cells.setdefault(self._cell_of(point.coordinates_m), []).append(point)

    def _cell_of(self, coordinates_m: Vector3) -> tuple[int, int, int]:
        x, y, z = (math.floor(value / self._cell_size_m) for value in coordinates_m)
        return (x, y, z)

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        if bounds.frame_id != self._map.frame_id:
            raise ValueError("query bounds are expressed in another frame")
        self.queries.append(bounds)
        low = self._cell_of(bounds.minimum_m)
        high = self._cell_of(bounds.maximum_m)
        for cx in range(low[0], high[0] + 1):
            for cy in range(low[1], high[1] + 1):
                for cz in range(low[2], high[2] + 1):
                    for point in self._cells.get((cx, cy, cz), ()):
                        if bounds.contains(point.coordinates_m, frame_id=point.map_frame):
                            yield point


class RepeatingSource(LinearScanSource):
    """A faulty source that yields every match twice."""

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        for point in super().query_bounds(bounds):
            yield point
            yield point


class ForeignFrameSource(LinearScanSource):
    """A faulty source whose query results claim another map frame."""

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        for point in super().query_bounds(bounds):
            yield make_geometry_point(
                int(str(point.geometry_id).rsplit("-", 1)[1]),
                point.coordinates_m,
                map_frame="other",
            )


def points_from(coordinates: Sequence[Vector3]) -> list[GeometryPoint]:
    """Points whose geometry index is their position in ``coordinates``."""
    return [make_geometry_point(index, xyz) for index, xyz in enumerate(coordinates)]


def line_of_points(count: int, *, spacing_m: float = 0.25) -> list[GeometryPoint]:
    """Points on the x axis; a dyadic spacing keeps every distance exactly representable."""
    return points_from([(index * spacing_m, 0.0, 0.0) for index in range(count)])


def random_cloud(count: int, *, extent_m: float = 4.0, seed: int = 7) -> list[GeometryPoint]:
    """A deterministic cloud on a 1/64 m lattice, so equal distances occur as ties."""
    rng = random.Random(seed)
    lattice = int(extent_m * 64)
    return points_from(
        [
            (rng.randrange(lattice) / 64, rng.randrange(lattice) / 64, rng.randrange(lattice) / 64)
            for _ in range(count)
        ]
    )
