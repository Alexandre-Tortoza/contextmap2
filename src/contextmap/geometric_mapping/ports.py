"""The read boundary other capabilities use to reach persistent geometry.

Sensor Association, Point Representation and later stages need to resolve
geometry by reference and by spatial extent without depending on how a map is
stored or indexed. :class:`GeometrySource` is that boundary. The authoritative
data is the persisted geometry: an index is an implementation detail behind
these methods, and a corrupt derived index must never redefine coordinates.

The boundary owns no semantics: there is no label, feature, entity or
nearest-semantic query, and no projection into images.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from contextmap.geometric_mapping.models import (
    Bounds3D,
    GeometricMap,
    GeometryPoint,
    GeometryReference,
)


@runtime_checkable
class GeometrySource(Protocol):
    """Capability port: read persistent geometry of one immutable map."""

    @property
    def geometric_map(self) -> GeometricMap:
        """The map's identity, frame, bounds and provenance."""
        ...

    def get(self, reference: GeometryReference) -> GeometryPoint:
        """Resolve a reference to its authoritative geometry.

        Args:
            reference: A reference into this map.

        Returns:
            The same point every time, with its coordinates and lineage.

        Raises:
            KeyError: If the reference does not belong to this map.
        """
        ...

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        """Iterate every geometry element in a deterministic order."""
        ...

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        """Iterate the geometry inside a box, boundaries included.

        Args:
            bounds: The box; it must be expressed in this map's frame.

        Returns:
            The elements whose map coordinates lie inside or on a face of ``bounds``.

        Raises:
            ValueError: If ``bounds`` is expressed in another frame.
        """
        ...
