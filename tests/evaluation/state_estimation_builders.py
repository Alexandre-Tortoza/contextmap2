"""Deterministic trajectory builders for the State Estimation evaluation tests."""

from __future__ import annotations

import math
from collections.abc import Sequence

from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import Quaternion, SourceTimestamp, Vector3
from contextmap.state_estimation import (
    EstimatorProvenance,
    PoseEstimate,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryId,
    TrajectoryProvenance,
    pose_estimate_id_for,
)

MS = 1_000_000
CLOCK_ID = "fixture:header"
IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)


def yaw(angle: float) -> Quaternion:
    """Quaternion for a rotation of ``angle`` radians about z."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def trajectory_from(
    samples: Sequence[tuple[int, Vector3, Quaternion]],
    *,
    trajectory_id: str = "trajectory",
    clock_id: str = CLOCK_ID,
    backend_id: str = "fake_estimator",
    configuration: str = "cfg-1",
    selection_id: str = "full-sequence",
    calibration_identity: str | None = "sha256:calibration",
) -> Trajectory:
    """Build a trajectory from ``(time_ns, translation_m, orientation)`` samples."""
    identity = TrajectoryId(trajectory_id)
    poses = []
    for index, (time_ns, translation, orientation) in enumerate(samples):
        seconds, nanoseconds = divmod(time_ns, 1_000_000_000)
        poses.append(
            PoseEstimate(
                estimate_id=pose_estimate_id_for(trajectory_id=identity, index=index),
                timestamp=SourceTimestamp(
                    seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id
                ),
                parent_frame=FrameId("map"),
                child_frame=FrameId("body"),
                translation_m=translation,
                orientation=orientation,
                validity=PoseValidity.VALID,
                provenance=PoseProvenance(
                    source_observation_ids=(SourceObservationId(f"obs-{index}"),)
                ),
            )
        )
    return Trajectory(
        trajectory_id=identity,
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
        poses=tuple(poses),
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(
                backend_id=backend_id,
                backend_version="1",
                configuration_fingerprint=f"sha256:{configuration}",
            ),
            sequence_artifact_id=SequenceArtifactId("sequence-0001"),
            selection_id=selection_id,
            calibration_identity=calibration_identity,
        ),
    )


def straight_line(
    count: int = 6,
    *,
    step_m: float = 1.0,
    interval_ns: int = 100 * MS,
    offset: Vector3 = (0.0, 0.0, 0.0),
    time_offset_ns: int = 0,
    **options: object,
) -> Trajectory:
    """Build poses moving ``step_m`` along +x every ``interval_ns``."""
    return trajectory_from(
        [
            (
                time_offset_ns + index * interval_ns,
                (offset[0] + index * step_m, offset[1], offset[2]),
                IDENTITY,
            )
            for index in range(count)
        ],
        **options,  # type: ignore[arg-type]
    )
