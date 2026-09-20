"""Public contract for the state estimation capability.

State Estimation owns the dynamic pose of the rig: where the body frame is in
the map frame over time. It publishes backend-agnostic
:class:`PoseEstimate` and :class:`Trajectory` contracts so Geometric Mapping
and Sensor Association never depend on ROS messages, an estimator's native
types, or a dataset's pose format. See
``src/contextmap/state_estimation/docs/README.md`` for the full capability
documentation.
"""

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

__all__ = [
    "EstimatorProvenance",
    "PoseEstimate",
    "PoseEstimateId",
    "PoseProvenance",
    "PoseValidity",
    "TimeBounds",
    "Trajectory",
    "TrajectoryGap",
    "TrajectoryId",
    "TrajectoryProvenance",
    "TrajectoryQualitySummary",
    "pose_estimate_id_for",
]
