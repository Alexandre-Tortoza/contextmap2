import dataclasses
import json
import math

import pytest
from pose_builders import make_pose, make_trajectory

from contextmap.ingestion import SequenceArtifactId, SourceObservationId
from contextmap.state_estimation import (
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
    TrajectoryGap,
)
from contextmap.state_estimation.serialization import (
    decode_pose_estimate,
    decode_trajectory,
    encode_pose_estimate,
    encode_trajectory_metadata,
)


def _through_json(record: dict[str, object]) -> dict[str, object]:
    decoded: dict[str, object] = json.loads(json.dumps(record))
    return decoded


def test_pose_round_trips_through_json() -> None:
    pose = make_pose(
        3,
        translation_m=(1.5, -2.25, 0.125),
        orientation=(0.0, 0.0, math.sin(0.1), math.cos(0.1)),
        covariance=tuple(float(i) / 7 for i in range(36)),
        validity=PoseValidity.DEGRADED,
    )

    assert decode_pose_estimate(_through_json(encode_pose_estimate(pose))) == pose


def test_missing_covariance_stays_explicit_in_the_encoded_record() -> None:
    record = encode_pose_estimate(make_pose(0))

    assert record["covariance"] is None
    assert decode_pose_estimate(_through_json(record)).covariance is None


def test_encoded_pose_states_frames_units_and_clock_domain() -> None:
    record = encode_pose_estimate(make_pose(2, parent_frame="map", child_frame="body"))

    assert record["parent_frame"] == "map"
    assert record["child_frame"] == "body"
    assert record["timestamp"] == {
        "seconds": 0,
        "nanoseconds": 200_000_000,
        "clock_id": "fixture:header",
    }
    assert "translation_m" in record
    assert record["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]


def test_decode_rejects_an_unknown_validity_state() -> None:
    record = _through_json(encode_pose_estimate(make_pose(0)))
    record["validity"] = "bogus"

    with pytest.raises(ValueError, match="bogus"):
        decode_pose_estimate(record)


def test_trajectory_metadata_excludes_poses_and_round_trips_with_them() -> None:
    poses = [make_pose(0), make_pose(1), make_pose(2, time_ns=1_000_000_000)]
    gap = TrajectoryGap(
        previous_estimate_id=poses[1].estimate_id,
        next_estimate_id=poses[2].estimate_id,
        duration_ns=900_000_000,
    )
    trajectory = make_trajectory(poses, gaps=[gap])

    metadata = _through_json(encode_trajectory_metadata(trajectory))

    assert "poses" not in metadata
    assert decode_trajectory(metadata, trajectory.poses) == trajectory


def test_trajectory_metadata_preserves_run_level_provenance() -> None:
    metadata = encode_trajectory_metadata(make_trajectory())

    provenance = metadata["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["sequence_artifact_id"] == "sequence-0001"
    assert provenance["selection_id"] == "full-sequence"
    assert provenance["calibration_identity"] == "sha256:calibration"
    assert provenance["estimator"] == {
        "backend_id": "fake_estimator",
        "backend_version": "0",
        "configuration_fingerprint": "sha256:cfg",
    }


def test_trajectory_metadata_round_trips_the_auxiliary_sequence_lineage() -> None:
    # #594: a sequência auxiliar mesclada (#555) é linhagem da trajetória, não só do manifesto.
    trajectory = make_trajectory()
    trajectory = dataclasses.replace(
        trajectory,
        provenance=dataclasses.replace(
            trajectory.provenance,
            auxiliary_sequence_artifact_id=SequenceArtifactId("sequence-aux-0001"),
            auxiliary_selection_id="aux-selection",
        ),
    )

    metadata = _through_json(encode_trajectory_metadata(trajectory))

    assert decode_trajectory(metadata, trajectory.poses) == trajectory


def test_trajectory_metadata_without_auxiliary_lineage_decodes_as_absent() -> None:
    # Records gravados antes da #594 não têm os campos auxiliares.
    trajectory = make_trajectory()
    metadata = _through_json(encode_trajectory_metadata(trajectory))
    provenance = metadata["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("auxiliary_sequence_artifact_id", None)
    provenance.pop("auxiliary_selection_id", None)

    decoded = decode_trajectory(metadata, trajectory.poses)

    assert decoded.provenance.auxiliary_sequence_artifact_id is None
    assert decoded.provenance.auxiliary_selection_id is None


def test_derived_pose_round_trips_with_the_estimates_it_was_derived_from() -> None:
    pose = dataclasses.replace(
        make_pose(0),
        provenance=PoseProvenance(
            source_observation_ids=(SourceObservationId("pose-0000"),),
            conversions_applied=("interpolated",),
            derived_from=(PoseEstimateId("estimate-a"), PoseEstimateId("estimate-b")),
        ),
    )

    record = _through_json(encode_pose_estimate(pose))

    assert record["provenance"]["derived_from"] == ["estimate-a", "estimate-b"]  # type: ignore[index]
    assert decode_pose_estimate(record) == pose
