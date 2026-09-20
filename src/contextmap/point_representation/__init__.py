"""Public contract for the point representation capability.

Point Representation owns numerical descriptions of the local 3D structure
around persistent geometry: what geometry supported each vector, which
representation space it lives in, and which encoder produced it. It is a 3D
evidence channel kept distinct from visual features and semantic claims; it
owns no label, fusion or entity. See
``src/contextmap/point_representation/docs/README.md`` for the full capability
documentation.
"""

from contextmap.point_representation.compatibility import (
    RepresentationSpaceMismatchError,
    ensure_compatible_representation_spaces,
    ensure_compatible_representations,
    representation_space_fingerprint,
)
from contextmap.point_representation.models import (
    CenteringMode,
    CoordinatePreparation,
    EncoderIdentity,
    NeighborhoodMethod,
    PointRepresentation,
    PointRepresentationId,
    PointRepresentationRunId,
    PointSupport,
    PreparedSupport,
    RepresentationProvenance,
    RepresentationSpace,
    ScaleNormalization,
    SupportPolicy,
    SupportStatistics,
    SupportType,
    representation_id_for,
)
from contextmap.point_representation.support import SupportExtractor

__all__ = [
    "CenteringMode",
    "CoordinatePreparation",
    "EncoderIdentity",
    "NeighborhoodMethod",
    "PointRepresentation",
    "PointRepresentationId",
    "PointRepresentationRunId",
    "PointSupport",
    "PreparedSupport",
    "RepresentationProvenance",
    "RepresentationSpace",
    "RepresentationSpaceMismatchError",
    "ScaleNormalization",
    "SupportExtractor",
    "SupportPolicy",
    "SupportStatistics",
    "SupportType",
    "ensure_compatible_representation_spaces",
    "ensure_compatible_representations",
    "representation_id_for",
    "representation_space_fingerprint",
]
