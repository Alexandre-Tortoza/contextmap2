"""Public contract for the sensor association capability.

Sensor Association turns 2D visual evidence into references to persistent 3D
geometry. For each region of one image it produces a
:class:`SpatialObservation`: the map elements that are visible and inside the
region, plus the diagnostics that explain the ones that are not. It is evidence
-- it does not label, identify, fuse or resolve entities, and it copies neither
XYZ, embeddings nor claims. See
``src/contextmap/sensor_association/docs/README.md`` for the full capability
documentation.
"""

from contextmap.sensor_association.camera_models import (
    CameraIdentity,
    CameraProjection,
    PixelProjection,
    camera_projection_for,
)
from contextmap.sensor_association.models import (
    AssociationProvenance,
    CalibrationRef,
    PixelCoordinate,
    PointCorrespondence,
    PoseRef,
    ProjectionSummary,
    SemanticClaimRef,
    SpatialObservation,
    SpatialObservationId,
    VisibilityDiagnostics,
    VisibilityState,
    VisualFeatureRef,
    spatial_observation_id_for,
)

__all__ = [
    "AssociationProvenance",
    "CalibrationRef",
    "CameraIdentity",
    "CameraProjection",
    "PixelCoordinate",
    "PixelProjection",
    "PointCorrespondence",
    "PoseRef",
    "ProjectionSummary",
    "SemanticClaimRef",
    "SpatialObservation",
    "SpatialObservationId",
    "VisibilityDiagnostics",
    "VisibilityState",
    "VisualFeatureRef",
    "camera_projection_for",
    "spatial_observation_id_for",
]
