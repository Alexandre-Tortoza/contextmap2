"""Public contract for the state estimation capability.

State Estimation owns the dynamic pose of the rig: where the body frame is in
the map frame over time. It publishes backend-agnostic
:class:`PoseEstimate` and :class:`Trajectory` contracts so Geometric Mapping
and Sensor Association never depend on ROS messages, an estimator's native
types, or a dataset's pose format. See
``src/contextmap/state_estimation/docs/README.md`` for the full capability
documentation.
"""

from contextmap.state_estimation.frame_graph import (
    FrameGraphError,
    LoopInconsistency,
    ResolvedTransform,
    StaticFrameGraph,
)
from contextmap.state_estimation.lookup import (
    ClockDomainMismatchError,
    LookupMode,
    LookupOutcome,
    LookupPolicy,
    LookupRejection,
    PoseLookupResult,
    RejectedLookup,
    ResolvedPose,
    TemporalAlignmentSummary,
    TrajectoryLookup,
    summarize_lookups,
)
from contextmap.state_estimation.models import (
    EstimatorProvenance,
    PoseEstimate,
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
    TimeBounds,
    Trajectory,
    TrajectoryGap,
    TrajectoryId,
    TrajectoryProvenance,
    TrajectoryQualitySummary,
    pose_estimate_id_for,
)
from contextmap.state_estimation.ports import (
    DiagnosticSeverity,
    EstimationDiagnostic,
    MissingEstimatorInputError,
    StateEstimationError,
    StateEstimationRequest,
    StateEstimationResult,
    StateEstimator,
)
from contextmap.state_estimation.preflight import (
    BODY_ENDPOINT,
    ClockCheck,
    DownstreamReadiness,
    FrameGraphSummary,
    GeometryPreflightReport,
    GeometryRequirements,
    PreflightFinding,
    PreflightStatus,
    PreflightTolerances,
    StaticRelationRequirement,
    TransformCheck,
    calibration_identity,
    run_geometry_preflight,
)
from contextmap.state_estimation.service import (
    GeometryPreflightError,
    StateEstimationOutcome,
    execute_state_estimation,
)

__all__ = [
    "BODY_ENDPOINT",
    "ClockCheck",
    "ClockDomainMismatchError",
    "DiagnosticSeverity",
    "DownstreamReadiness",
    "EstimationDiagnostic",
    "EstimatorProvenance",
    "FrameGraphError",
    "FrameGraphSummary",
    "GeometryPreflightError",
    "GeometryPreflightReport",
    "GeometryRequirements",
    "LookupMode",
    "LookupOutcome",
    "LookupPolicy",
    "LookupRejection",
    "LoopInconsistency",
    "MissingEstimatorInputError",
    "PoseEstimate",
    "PoseEstimateId",
    "PoseLookupResult",
    "PoseProvenance",
    "PoseValidity",
    "PreflightFinding",
    "PreflightStatus",
    "PreflightTolerances",
    "RejectedLookup",
    "ResolvedPose",
    "ResolvedTransform",
    "StateEstimationError",
    "StateEstimationOutcome",
    "StateEstimationRequest",
    "StateEstimationResult",
    "StateEstimator",
    "StaticFrameGraph",
    "StaticRelationRequirement",
    "TemporalAlignmentSummary",
    "TimeBounds",
    "Trajectory",
    "TrajectoryGap",
    "TrajectoryId",
    "TrajectoryLookup",
    "TrajectoryProvenance",
    "TrajectoryQualitySummary",
    "TransformCheck",
    "calibration_identity",
    "execute_state_estimation",
    "pose_estimate_id_for",
    "run_geometry_preflight",
    "summarize_lookups",
]
