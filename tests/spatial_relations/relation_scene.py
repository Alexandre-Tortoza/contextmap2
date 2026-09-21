"""Deterministic 3D scenes for Spatial Relations tests.

A scene holds named point sets in one geometric map. ``geometry(name)`` summarizes a set with the
real Semantic Mapping algorithm, so the bounds, centroid, statistics and diagnostics that the
relation evaluators read are exactly what an entity would carry.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from itertools import product
from pathlib import Path

from contextmap.geometric_mapping import GeometryReference, GeometrySource, MapId, geometry_id_for
from contextmap.semantic_mapping import (
    EntityGeometry,
    GeometrySummaryPolicy,
    OrientationPolicy,
    summarize_geometry,
)
from contextmap.shared import Vector3

MAP_ID = MapId("map-0001")
FRAME = "map"

# O raio de conectividade é grande de propósito: os cenários de teste são caixas cheias e só os
# testes de suporte desconectado ou esparso pedem uma política diferente.
CONNECTED_POLICY = GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=1000.0)
STRICT_POLICY = GeometrySummaryPolicy(sparse_point_threshold=10, connectivity_radius_m=0.5)
# Nuvens densas (malhas de 0,05 a 0,1 m) usam um raio pequeno: com um raio enorme a checagem de
# conectividade de Semantic Mapping é quadrática e os testes ficam lentos sem ganho.
LATTICE_POLICY = GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.15)
ORIENTED_LATTICE_POLICY = GeometrySummaryPolicy(
    sparse_point_threshold=3,
    connectivity_radius_m=0.15,
    orientation=OrientationPolicy(min_points=3, min_variance_ratio=2.0),
)

_MAPPING_TESTS = str(Path(__file__).resolve().parents[1] / "semantic_mapping")


def _lattice(low: float, high: float) -> tuple[float, ...]:
    return (low,) if low == high else (low, (low + high) / 2.0, high)


def in_memory_source(
    map_id: MapId, coordinates_by_index: dict[int, Vector3], *, frame: str
) -> GeometrySource:
    """Serve points from memory, reusing the fake of the Semantic Mapping tests."""
    # O fake vive nos testes de Semantic Mapping: o diretório entra no caminho de importação só
    # quando um cenário precisa dele, sem um segundo conftest que colidiria no mypy.
    if _MAPPING_TESTS not in sys.path:
        sys.path.insert(0, _MAPPING_TESTS)
    from mapping_geometry_fake import InMemoryGeometrySource

    return InMemoryGeometrySource(map_id, coordinates_by_index, frame=frame)


class Scene:
    """Named point sets of one geometric map, and the geometry summarized from each."""

    def __init__(self, *, map_id: MapId = MAP_ID, frame: str = FRAME) -> None:
        self.map_id = map_id
        self.frame = frame
        self._coordinates: dict[int, Vector3] = {}
        self._members: dict[str, list[int]] = {}
        self._source: GeometrySource | None = None

    def add_points(self, name: str, points: Sequence[Vector3]) -> None:
        """Add a named set made of exactly these points."""
        indexes = []
        for point in points:
            index = len(self._coordinates)
            self._coordinates[index] = point
            indexes.append(index)
        self._members[name] = indexes
        self._source = None

    def add_box(self, name: str, minimum: Vector3, maximum: Vector3) -> None:
        """Add a named set filling an axis-aligned box with a lattice of points.

        A flat axis collapses to a single layer, so a plane or a line is a valid support.
        """
        axes = [_lattice(low, high) for low, high in zip(minimum, maximum, strict=True)]
        self.add_points(name, [(x, y, z) for x, y, z in product(*axes)])

    def add_lattice(self, name: str, minimum: Vector3, maximum: Vector3, spacing: float) -> None:
        """Add a named set filling a box with points every ``spacing`` meters, faces included."""
        axes = []
        for low, high in zip(minimum, maximum, strict=True):
            steps = max(1, round((high - low) / spacing))
            axes.append(tuple(low + (high - low) * step / steps for step in range(steps + 1)))
        self.add_points(name, [(x, y, z) for x, y, z in product(*axes)])

    def source(self) -> GeometrySource:
        """The in-memory geometry source of every point of the scene, built once per change."""
        if self._source is None:
            self._source = in_memory_source(self.map_id, self._coordinates, frame=self.frame)
        return self._source

    def references(self, name: str) -> tuple[GeometryReference, ...]:
        """The geometry references of a named set."""
        return tuple(
            GeometryReference(
                map_id=self.map_id, geometry_id=geometry_id_for(map_id=self.map_id, index=index)
            )
            for index in self._members[name]
        )

    def geometry(
        self, name: str, *, policy: GeometrySummaryPolicy = CONNECTED_POLICY
    ) -> EntityGeometry:
        """Summarize a named set as the geometry of an entity."""
        return summarize_geometry(self.references(name), source=self.source(), policy=policy)


def box_geometry(
    minimum: Vector3,
    maximum: Vector3,
    *,
    map_id: MapId = MAP_ID,
    frame: str = FRAME,
    policy: GeometrySummaryPolicy = CONNECTED_POLICY,
) -> EntityGeometry:
    """The geometry of one entity that fills a box, in a scene of its own."""
    scene = Scene(map_id=map_id, frame=frame)
    scene.add_box("only", minimum, maximum)
    return scene.geometry("only", policy=policy)
