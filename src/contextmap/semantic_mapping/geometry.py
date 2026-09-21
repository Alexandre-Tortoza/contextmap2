"""The 3D support of an entity.

An entity keeps the *exact* persistent geometry that justifies its spatial existence, as
references into an immutable geometric map. Nothing here copies XYZ: the geometry the
references point at stays authoritative, and any summary of it is derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.geometric_mapping import GeometryReference, MapId
from contextmap.ingestion import FrameId
from contextmap.semantic_mapping._checks import require_canonical, require_present


@dataclass(frozen=True, kw_only=True)
class EntityGeometry:
    """The persistent geometry that supports one entity.

    Attributes:
        geometry_refs: References to the supporting geometry, sorted by ``geometry_id``,
            unique and never empty: an entity without real 3D support is not an entity.
        map_frame: The frame of the map the geometry belongs to; every spatial summary of the
            entity is expressed in it.
    """

    geometry_refs: tuple[GeometryReference, ...]
    map_frame: FrameId

    def __post_init__(self) -> None:
        """Validate that the support is present, canonical and from a single map.

        Raises:
            ValueError: If the frame is empty, the support is empty, the references are not
                sorted by geometry and unique, or they span more than one map.
        """
        require_present(self, "map_frame")
        if not self.geometry_refs:
            raise ValueError("geometry_refs must not be empty: an entity needs 3D support")
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda reference: (reference.geometry_id,),
            detail="by geometry_id ",
        )
        maps = {reference.map_id for reference in self.geometry_refs}
        if len(maps) > 1:
            raise ValueError(f"geometry_refs must belong to one map, got {sorted(maps)!r}")

    @property
    def geometric_map_id(self) -> MapId:
        """The immutable geometric map the support belongs to."""
        return self.geometry_refs[0].map_id
