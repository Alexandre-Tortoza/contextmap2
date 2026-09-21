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
    FailedSupport,
    FailureReason,
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
from contextmap.point_representation.ports import (
    EncodedVector,
    PointEncoder,
    UnencodableSupportError,
)
from contextmap.point_representation.run_artifact import (
    IncompleteRunArtifactError,
    PointRepresentationDebugLevel,
    PointRepresentationRunManifest,
    PointRepresentationRunReader,
    PointRepresentationRunWriter,
    RunArtifactError,
    center_selection_id,
)
from contextmap.point_representation.serialization import encode_support_policy
from contextmap.point_representation.service import (
    EncodedRepresentation,
    RepresentationMetrics,
    RepresentationService,
)
from contextmap.point_representation.support import SupportExtractor

__all__ = [
    "CenteringMode",
    "CoordinatePreparation",
    "EncodedRepresentation",
    "EncodedVector",
    "EncoderIdentity",
    "FailedSupport",
    "FailureReason",
    "IncompleteRunArtifactError",
    "NeighborhoodMethod",
    "PointEncoder",
    "PointRepresentation",
    "PointRepresentationDebugLevel",
    "PointRepresentationId",
    "PointRepresentationRunId",
    "PointRepresentationRunManifest",
    "PointRepresentationRunReader",
    "PointRepresentationRunWriter",
    "PointSupport",
    "PreparedSupport",
    "RepresentationMetrics",
    "RepresentationProvenance",
    "RepresentationService",
    "RepresentationSpace",
    "RepresentationSpaceMismatchError",
    "RunArtifactError",
    "ScaleNormalization",
    "SupportExtractor",
    "SupportPolicy",
    "SupportStatistics",
    "SupportType",
    "UnencodableSupportError",
    "center_selection_id",
    "encode_support_policy",
    "ensure_compatible_representation_spaces",
    "ensure_compatible_representations",
    "representation_id_for",
    "representation_space_fingerprint",
]
