"""Public contract for the geometric mapping capability.

Geometric Mapping owns persistent 3D geometry in one global map frame: the
authoritative XYZ of every point, a stable reference to it, and the lineage that
says how each coordinate was produced. It stops before semantics: labels,
features, claims, entities and relations are attached by other capabilities
through :class:`GeometryReference`. See
``src/contextmap/geometric_mapping/docs/README.md`` for the full capability
documentation.
"""

from contextmap.geometric_mapping.models import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryId,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    MapId,
    PointOrigin,
    SpatialIndexMetadata,
    SpatialIndexParameter,
    TransformKind,
    TransformLineage,
    TransformStep,
    geometry_id_for,
)
from contextmap.geometric_mapping.ports import GeometrySource

__all__ = [
    "Bounds3D",
    "GeometricMap",
    "GeometricMapProvenance",
    "GeometryId",
    "GeometryPoint",
    "GeometryPointProvenance",
    "GeometryReference",
    "GeometrySource",
    "MapId",
    "PointOrigin",
    "SpatialIndexMetadata",
    "SpatialIndexParameter",
    "TransformKind",
    "TransformLineage",
    "TransformStep",
    "geometry_id_for",
]
