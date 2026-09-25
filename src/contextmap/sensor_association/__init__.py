"""Public contract for the sensor association capability.

Sensor Association turns 2D visual evidence into references to persistent 3D
geometry. Each frame evaluates the map geometry its
:class:`CandidateGeometryPolicy` selects, and every candidate row carries the global
geometry index it came from, so a reference always names a map element and never
a position in an array. For each region of one image it produces a
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
from contextmap.sensor_association.candidate_geometry import (
    CANDIDATE_GEOMETRY_POLICY_ID,
    CandidateGeometryCloud,
    CandidateGeometryPolicy,
    CandidateGeometryReport,
    CandidateGeometrySelection,
    select_candidate_geometry,
)
from contextmap.sensor_association.dense_sampling import InterpolationPolicy
from contextmap.sensor_association.diagnostics import (
    DiagnosticTolerances,
    ReprojectionAttempt,
    ReprojectionOutcome,
    TrustedCorrespondences,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.models import (
    AssociationProvenance,
    CalibrationRef,
    DepthMetric,
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
from contextmap.sensor_association.quality import (
    ObservationQuality,
    QualityComponent,
    ReprojectionStatistics,
    ValueSummary,
)
from contextmap.sensor_association.run_artifact import (
    IncompleteRunArtifactError,
    RunArtifactError,
    SensorAssociationDebugLevel,
    SensorAssociationRunId,
    SensorAssociationRunManifest,
    SensorAssociationRunReader,
    SensorAssociationRunTransaction,
    SensorAssociationRunWriter,
)
from contextmap.sensor_association.service import (
    AssociationFrameInput,
    DenseChannel,
    FrameAssociation,
    FrameSink,
    SensorAssociationOutcome,
    SensorAssociationRequest,
    SensorAssociationService,
)
from contextmap.sensor_association.visibility import OcclusionPolicy

__all__ = [
    "CANDIDATE_GEOMETRY_POLICY_ID",
    "AssociationFrameInput",
    "AssociationInputError",
    "AssociationProvenance",
    "CalibrationRef",
    "CameraIdentity",
    "CameraProjection",
    "CandidateGeometryCloud",
    "CandidateGeometryPolicy",
    "CandidateGeometryReport",
    "CandidateGeometrySelection",
    "DenseChannel",
    "DepthMetric",
    "DiagnosticTolerances",
    "FrameAssociation",
    "FrameSink",
    "IncompleteRunArtifactError",
    "InterpolationPolicy",
    "ObservationQuality",
    "OcclusionPolicy",
    "PixelCoordinate",
    "PixelProjection",
    "PointCorrespondence",
    "PoseRef",
    "ProjectionSummary",
    "QualityComponent",
    "ReprojectionAttempt",
    "ReprojectionOutcome",
    "ReprojectionStatistics",
    "RunArtifactError",
    "SemanticClaimRef",
    "SensorAssociationDebugLevel",
    "SensorAssociationOutcome",
    "SensorAssociationRequest",
    "SensorAssociationRunId",
    "SensorAssociationRunManifest",
    "SensorAssociationRunReader",
    "SensorAssociationRunTransaction",
    "SensorAssociationRunWriter",
    "SensorAssociationService",
    "SpatialObservation",
    "SpatialObservationId",
    "TrustedCorrespondences",
    "ValueSummary",
    "VisibilityDiagnostics",
    "VisibilityState",
    "VisualFeatureRef",
    "camera_projection_for",
    "select_candidate_geometry",
    "spatial_observation_id_for",
]
