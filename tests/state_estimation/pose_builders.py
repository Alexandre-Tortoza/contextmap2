"""Deterministic builders for State Estimation tests.

Kept as Python (no binary fixtures) so the synthetic poses stay inspectable and
every test states the geometry it depends on.
"""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import (
    EstimatorProvenance,
    PoseEstimate,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryGap,
    TrajectoryId,
    TrajectoryProvenance,
    pose_estimate_id_for,
)

CLOCK_ID = "fixture:header"
TRAJECTORY_ID = TrajectoryId("run-0001--trajectory")
IDENTITY_ORIENTATION = (0.0, 0.0, 0.0, 1.0)


def timestamp_ns(total_nanoseconds: int, *, clock_id: str = CLOCK_ID) -> SourceTimestamp:
    """Build a timestamp from an exact integer number of nanoseconds."""
    seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)


def make_pose(
    index: int,
    *,
    time_ns: int | None = None,
    translation_m: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] = IDENTITY_ORIENTATION,
    parent_frame: str = "map",
    child_frame: str = "body",
    validity: PoseValidity = PoseValidity.VALID,
    covariance: tuple[float, ...] | None = None,
    clock_id: str = CLOCK_ID,
    trajectory_id: TrajectoryId = TRAJECTORY_ID,
) -> PoseEstimate:
    """Build one valid pose; by default 100 ms apart along +x."""
    when = index * 100_000_000 if time_ns is None else time_ns
    return PoseEstimate(
        estimate_id=pose_estimate_id_for(trajectory_id=trajectory_id, index=index),
        timestamp=timestamp_ns(when, clock_id=clock_id),
        parent_frame=FrameId(parent_frame),
        child_frame=FrameId(child_frame),
        translation_m=(float(index), 0.0, 0.0) if translation_m is None else translation_m,
        orientation=orientation,
        validity=validity,
        provenance=PoseProvenance(
            source_observation_ids=(SourceObservationId(f"pose-{index:04d}"),),
        ),
        covariance=covariance,
    )


def make_trajectory_provenance() -> TrajectoryProvenance:
    """Build the run-level provenance shared by the test trajectories."""
    return TrajectoryProvenance(
        estimator=EstimatorProvenance(
            backend_id="fake_estimator",
            backend_version="0",
            configuration_fingerprint="sha256:cfg",
        ),
        sequence_artifact_id=SequenceArtifactId("sequence-0001"),
        selection_id="full-sequence",
        calibration_identity="sha256:calibration",
        code_version="test",
    )


def make_trajectory(
    poses: Sequence[PoseEstimate] | None = None,
    *,
    gaps: Sequence[TrajectoryGap] = (),
    reference_frame: str = "map",
    body_frame: str = "body",
) -> Trajectory:
    """Build a valid trajectory; defaults to five poses 100 ms apart."""
    return Trajectory(
        trajectory_id=TRAJECTORY_ID,
        reference_frame=FrameId(reference_frame),
        body_frame=FrameId(body_frame),
        poses=tuple(poses) if poses is not None else tuple(make_pose(i) for i in range(5)),
        gaps=tuple(gaps),
        provenance=make_trajectory_provenance(),
    )
