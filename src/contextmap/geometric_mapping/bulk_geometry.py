"""Vectorized reads of persistent geometry, in blocks that carry global identity.

:class:`~contextmap.geometric_mapping.GeometrySource` resolves geometry one
:class:`~contextmap.geometric_mapping.GeometryPoint` at a time, which is the right
boundary for auditing a single element but the wrong one for a consumer that must
evaluate millions of coordinates per camera frame: building one dataclass per point
dominates the cost and the point's lineage is not even read.

This module adds the array-shaped half of the same read boundary. A
:class:`GeometryBlock` is a slab of authoritative map-frame coordinates plus, for
every row, the **global geometry index** the row came from. The two are always
carried together, so a consumer that subsets, reorders or culls rows never loses the
persistent identity of what it kept: row ``i`` of a block means the geometry
``geometry_id_for(map_id, indices[i])``, never ``geometry_id_for(map_id, i)``.

A block owns its arrays. They are copies, not views into a memory-mapped payload, so
a block stays valid after its source is closed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from contextmap.geometric_mapping.models import (
    Bounds3D,
    GeometricMap,
    GeometryReference,
    MapId,
    geometry_id_for,
)
from contextmap.ingestion import FrameId

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_BLOCK_POINTS = 1_000_000
"""Rows per block when a caller states no size: about 24 MB of coordinates.

It bounds the working set of a full-map pass without making the vectorized work per
block small enough for per-block overhead to matter.
"""


@dataclass(frozen=True, kw_only=True, eq=False)
class GeometryBlock:
    """Map-frame coordinates of some geometry of one map, with their global indices.

    Attributes:
        map_id: The immutable map artifact the geometry belongs to.
        frame_id: The map frame ``coordinates_m`` is expressed in.
        indices: ``(N,)`` zero-based global index in the map of each row.
        coordinates_m: ``(N, 3)`` authoritative map-frame coordinates, in meters.
    """

    map_id: MapId
    frame_id: FrameId
    indices: NDArray[Any]
    coordinates_m: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate that the arrays describe the same points.

        Raises:
            ValueError: If the coordinates are not ``(N, 3)`` or there is not exactly
                one index per coordinate row.
        """
        if self.coordinates_m.ndim != 2 or self.coordinates_m.shape[1] != 3:
            raise ValueError(
                f"coordinates_m must have shape (N, 3), got {self.coordinates_m.shape}"
            )
        if self.indices.shape != (self.coordinates_m.shape[0],):
            raise ValueError(
                f"a block needs one index per row: {self.indices.shape} indices for "
                f"{self.coordinates_m.shape[0]} coordinates"
            )

    def __len__(self) -> int:
        """Return the number of geometry elements in the block."""
        return int(self.coordinates_m.shape[0])

    def reference(self, row: int) -> GeometryReference:
        """Return the persistent reference of one row, from its global index.

        Args:
            row: Position in this block, which is **not** a geometry index.

        Returns:
            The reference of the map element the row was read from.
        """
        index = int(self.indices[row])
        return GeometryReference(
            map_id=self.map_id, geometry_id=geometry_id_for(map_id=self.map_id, index=index)
        )


@runtime_checkable
class GeometryBlockSource(Protocol):
    """Capability port: read persistent geometry of one immutable map in array blocks.

    It is the vectorized counterpart of
    :class:`~contextmap.geometric_mapping.GeometrySource` and answers the same
    question about the same authoritative data; only the shape of the answer differs.
    A source may implement one, the other, or both.
    """

    @property
    def geometric_map(self) -> GeometricMap:
        """The map's identity, frame, bounds and provenance."""
        ...

    def iter_blocks(
        self, *, bounds: Bounds3D | None = None, block_points: int = DEFAULT_BLOCK_POINTS
    ) -> Iterator[GeometryBlock]:
        """Iterate the geometry of the map as blocks, in global index order.

        Args:
            bounds: Restricts the result to the geometry inside the box, boundaries
                included, exactly as
                :meth:`~contextmap.geometric_mapping.GeometrySource.query_bounds`
                does; ``None`` covers the whole map.
            block_points: Maximum rows per block, so a caller can bound its working
                set independently of the map's size.

        Returns:
            Blocks whose concatenation holds every selected element exactly once, in
            increasing global index. A block is never empty.

        Raises:
            ValueError: If ``block_points`` is not positive, or ``bounds`` is
                expressed in another frame than the map.
        """
        ...
