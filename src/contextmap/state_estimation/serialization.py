"""JSON-friendly encoding of the State Estimation contracts.

Records contain only JSON primitives, so a persisted trajectory can be read
without ROS, an estimator or NumPy. Poses and trajectory metadata are encoded
separately because a run artifact stores poses as one record per line and the
trajectory-level metadata once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation.models import (
    EstimatorProvenance,
    PoseEstimate,
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryGap,
    TrajectoryId,
    TrajectoryProvenance,
)


def encode_pose_estimate(pose: PoseEstimate) -> dict[str, Any]:
    """Encode a pose into a JSON-serializable record.

    Args:
        pose: The pose to encode.

    Returns:
        A record whose ``translation_m`` is in meters and whose
        ``orientation_xyzw`` follows the ``(x, y, z, w)`` order.
    """
    return {
        "estimate_id": str(pose.estimate_id),
        "timestamp": _encode_timestamp(pose.timestamp),
        "parent_frame": str(pose.parent_frame),
        "child_frame": str(pose.child_frame),
        "translation_m": list(pose.translation_m),
        "orientation_xyzw": list(pose.orientation),
        "validity": pose.validity.value,
        "covariance": None if pose.covariance is None else list(pose.covariance),
        "provenance": {
            "source_observation_ids": [
                str(item) for item in pose.provenance.source_observation_ids
            ],
            "conversions_applied": list(pose.provenance.conversions_applied),
            "derived_from": [str(item) for item in pose.provenance.derived_from],
        },
    }


def decode_pose_estimate(record: Mapping[str, Any]) -> PoseEstimate:
    """Decode a pose from a record produced by :func:`encode_pose_estimate`.

    Args:
        record: The persisted record.

    Returns:
        The validated pose.

    Raises:
        ValueError: If the record is malformed or violates the pose contract.
    """
    tx, ty, tz = record["translation_m"]
    qx, qy, qz, qw = record["orientation_xyzw"]
    covariance = record["covariance"]
    provenance = record["provenance"]
    return PoseEstimate(
        estimate_id=PoseEstimateId(record["estimate_id"]),
        timestamp=_decode_timestamp(record["timestamp"]),
        parent_frame=FrameId(record["parent_frame"]),
        child_frame=FrameId(record["child_frame"]),
        translation_m=(tx, ty, tz),
        orientation=(qx, qy, qz, qw),
        validity=PoseValidity(record["validity"]),
        provenance=PoseProvenance(
            source_observation_ids=tuple(
                SourceObservationId(item) for item in provenance["source_observation_ids"]
            ),
            conversions_applied=tuple(provenance["conversions_applied"]),
            derived_from=tuple(PoseEstimateId(item) for item in provenance["derived_from"]),
        ),
        covariance=None if covariance is None else tuple(covariance),
    )


def encode_trajectory_metadata(trajectory: Trajectory) -> dict[str, Any]:
    """Encode everything about a trajectory except its poses.

    Args:
        trajectory: The trajectory to encode.

    Returns:
        A JSON-serializable record; combine it with the poses in
        :func:`decode_trajectory`.
    """
    provenance = trajectory.provenance
    return {
        "trajectory_id": str(trajectory.trajectory_id),
        "reference_frame": str(trajectory.reference_frame),
        "body_frame": str(trajectory.body_frame),
        "gaps": [
            {
                "previous_estimate_id": str(gap.previous_estimate_id),
                "next_estimate_id": str(gap.next_estimate_id),
                "duration_ns": gap.duration_ns,
            }
            for gap in trajectory.gaps
        ],
        "provenance": {
            "estimator": {
                "backend_id": provenance.estimator.backend_id,
                "backend_version": provenance.estimator.backend_version,
                "configuration_fingerprint": provenance.estimator.configuration_fingerprint,
            },
            "sequence_artifact_id": str(provenance.sequence_artifact_id),
            "selection_id": provenance.selection_id,
            "calibration_identity": provenance.calibration_identity,
            "code_version": provenance.code_version,
            "auxiliary_sequence_artifact_id": (
                None
                if provenance.auxiliary_sequence_artifact_id is None
                else str(provenance.auxiliary_sequence_artifact_id)
            ),
            "auxiliary_selection_id": provenance.auxiliary_selection_id,
        },
    }


def decode_trajectory(metadata: Mapping[str, Any], poses: Sequence[PoseEstimate]) -> Trajectory:
    """Rebuild a trajectory from its metadata record and its poses.

    Args:
        metadata: Record produced by :func:`encode_trajectory_metadata`.
        poses: The poses, in trajectory order.

    Returns:
        The validated trajectory.

    Raises:
        ValueError: If the poses and metadata violate the trajectory contract.
    """
    provenance = metadata["provenance"]
    estimator = provenance["estimator"]
    return Trajectory(
        trajectory_id=TrajectoryId(metadata["trajectory_id"]),
        reference_frame=FrameId(metadata["reference_frame"]),
        body_frame=FrameId(metadata["body_frame"]),
        poses=tuple(poses),
        gaps=tuple(
            TrajectoryGap(
                previous_estimate_id=PoseEstimateId(gap["previous_estimate_id"]),
                next_estimate_id=PoseEstimateId(gap["next_estimate_id"]),
                duration_ns=gap["duration_ns"],
            )
            for gap in metadata["gaps"]
        ),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(
                backend_id=estimator["backend_id"],
                backend_version=estimator["backend_version"],
                configuration_fingerprint=estimator["configuration_fingerprint"],
            ),
            sequence_artifact_id=SequenceArtifactId(provenance["sequence_artifact_id"]),
            selection_id=provenance["selection_id"],
            calibration_identity=provenance["calibration_identity"],
            code_version=provenance["code_version"],
            # Records gravados antes da #594 não carregam a linhagem auxiliar.
            auxiliary_sequence_artifact_id=(
                None
                if provenance.get("auxiliary_sequence_artifact_id") is None
                else SequenceArtifactId(provenance["auxiliary_sequence_artifact_id"])
            ),
            auxiliary_selection_id=provenance.get("auxiliary_selection_id"),
        ),
    )


def _encode_timestamp(timestamp: SourceTimestamp) -> dict[str, Any]:
    return {
        "seconds": timestamp.seconds,
        "nanoseconds": timestamp.nanoseconds,
        "clock_id": timestamp.clock_id,
    }


def _decode_timestamp(record: Mapping[str, Any]) -> SourceTimestamp:
    return SourceTimestamp(
        seconds=record["seconds"],
        nanoseconds=record["nanoseconds"],
        clock_id=record["clock_id"],
    )
