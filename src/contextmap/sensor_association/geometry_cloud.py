"""Vectorizable view of a persistent geometric map.

Projection works on whole arrays, but the read boundary of Geometric Mapping,
:class:`~contextmap.geometric_mapping.GeometrySource`, yields one
:class:`~contextmap.geometric_mapping.GeometryPoint` at a time. :class:`GeometryCloud`
reads a map once into a coordinate array so every frame can be projected without a
Python loop over points.

Only the authoritative map-frame coordinate is kept, not a copy of the points: a
geometry reference is recomputed from the position, which is sound because the map
identity is positional (:func:`~contextmap.geometric_mapping.geometry_id_for`) and
the loader verifies it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping import (
    GeometricMap,
    GeometryReference,
    GeometrySource,
    geometry_id_for,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True, eq=False)
class GeometryCloud:
    """The coordinates of one map, in iteration order, ready for vectorized projection.

    Attributes:
        geometric_map: The map's identity, frame and provenance.
        coordinates_m: ``(N, 3)`` authoritative map-frame coordinates, in meters, where
            row ``i`` is the geometry ``geometry_id_for(map_id=..., index=i)``.
    """

    geometric_map: GeometricMap
    coordinates_m: NDArray[Any]

    def __len__(self) -> int:
        """Return the number of geometry elements."""
        return int(self.coordinates_m.shape[0])

    @classmethod
    def from_source(cls, source: GeometrySource) -> GeometryCloud:
        """Read every geometry element of a map once.

        Args:
            source: The read boundary of one immutable map.

        Returns:
            The cloud of that map.

        Raises:
            ValueError: If the source yields a different number of points than the map
                declares, a point belongs to another map or frame, or a geometry id is
                not the positional one ``geometry_id_for(map_id, index)``.
        """
        import numpy as np

        geometric_map = source.geometric_map
        coordinates = np.empty((geometric_map.point_count, 3), dtype=np.float64)
        count = 0
        for index, point in enumerate(source.iter_geometry()):
            if index >= geometric_map.point_count:
                raise ValueError(
                    f"the source yields more geometry than the declared point_count "
                    f"{geometric_map.point_count}"
                )
            if point.map_id != geometric_map.map_id or point.map_frame != geometric_map.frame_id:
                raise ValueError(
                    f"geometry {point.geometry_id!r} does not belong to map "
                    f"{geometric_map.map_id!r} in frame {geometric_map.frame_id!r}"
                )
            if point.geometry_id != geometry_id_for(map_id=geometric_map.map_id, index=index):
                raise ValueError(
                    f"geometry ids must be positional, but element {index} is {point.geometry_id!r}"
                )
            coordinates[index] = point.coordinates_m
            count += 1
        if count != geometric_map.point_count:
            raise ValueError(
                f"the source yields {count} points but the map declares point_count "
                f"{geometric_map.point_count}"
            )
        return cls(geometric_map=geometric_map, coordinates_m=coordinates)

    def reference(self, index: int) -> GeometryReference:
        """Return the stable reference of the ``index``-th geometry element."""
        map_id = self.geometric_map.map_id
        return GeometryReference(
            map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
        )

    def references(self, indices: Iterable[int]) -> tuple[GeometryReference, ...]:
        """Return the references of the given positions, in the given order."""
        return tuple(self.reference(int(index)) for index in indices)
