"""Issue #555: state_estimation merges an optional auxiliary pose SequenceArtifact.

``StateEstimationExecutor`` reads only one named input ("sequence") before this issue; these
tests exercise the bridge that lets it also read an optional "pose_sequence" input (an
auxiliary, pose-only ``SequenceArtifact`` from the new ``pose_ingestion`` stage), validate the
two sequences' clock relationship, and merge their observations -- without ``StateEstimation
Request``/``TrajectoryProvenance`` gaining any new field (issue #555's own non-goal).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextmap.ingestion import (
    ExternalPoseMeasurement,
    FrameId,
    FullSequenceSelection,
    ImuObservation,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SourceObservationId,
    SourceProvenance,
    selection_identity,
)
from contextmap.ingestion.sequence_provenance import SequenceProvenance
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import ExecutorError, StateEstimationExecutor
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import GeometryPreflightError, StateEstimationRunReader
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)

_CLOCK = "corridor-02-header"


def _pose(
    index: int, *, seconds: int, pose_role: str | None = "ground_truth"
) -> ExternalPoseMeasurement:
    return ExternalPoseMeasurement(
        observation_id=SourceObservationId(f"pose-{index:04d}"),
        sensor_id=SensorId("external_pose_source"),
        frame_id=FrameId("body"),
        timestamp=SourceTimestamp(seconds=seconds, nanoseconds=0, clock_id=_CLOCK),
        provenance=SourceProvenance(
            source_type="pose_file",
            source_path="fixtures/pose.txt",
            raw_metadata={} if pose_role is None else {"pose_role": pose_role},
        ),
        parent_frame=FrameId("map"),
        translation=(float(index), 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )


def _imu(index: int, *, seconds: int) -> ImuObservation:
    """A main-sequence anchor observation: gives the main sequence a real timestamp range to
    validate an auxiliary pose sequence's clock against, without adding any ExternalPose
    Measurement of its own (ExternalPoseEstimator only ever looks at those)."""
    return ImuObservation(
        observation_id=SourceObservationId(f"imu-{index:04d}"),
        sensor_id=SensorId("imu"),
        frame_id=FrameId("imu"),
        timestamp=SourceTimestamp(seconds=seconds, nanoseconds=0, clock_id=_CLOCK),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/imu"),
    )


def _write_sequence(
    directory: Path,
    *,
    artifact_id: str,
    observations: list[ExternalPoseMeasurement] | list[ImuObservation],
) -> None:
    with SequenceArtifactWriter(
        output_dir=directory, sequence_name="test", artifact_id=SequenceArtifactId(artifact_id)
    ) as writer:
        for observation in observations:
            writer.add_observation(observation)
        writer.set_provenance(
            SequenceProvenance(source_type="pose_file", source_path="fixtures/pose.txt")
        )
        writer.finalize()


def _request(
    tmp_path: Path,
    *,
    pose_observations: list[ExternalPoseMeasurement] | None,
) -> StageRequest:
    workspace = tmp_path / "ws"
    main_dir = workspace / "run-0001" / "ingestion"
    _write_sequence(
        main_dir,
        artifact_id="main-sequence",
        observations=[_imu(i, seconds=i) for i in range(3)],
    )
    inputs = {
        "sequence": (
            ArtifactRef(
                stage_id="ingestion",
                contract="SequenceArtifact",
                artifact_id="main-sequence",
                content_hash="sha256:main",
                location="run-0001/ingestion",
            ),
        ),
    }
    if pose_observations is not None:
        pose_dir = workspace / "run-0001" / "pose_ingestion"
        _write_sequence(pose_dir, artifact_id="pose-sequence", observations=pose_observations)
        inputs["pose_sequence"] = (
            ArtifactRef(
                stage_id="pose_ingestion",
                contract="SequenceArtifact",
                artifact_id="pose-sequence",
                content_hash="sha256:pose",
                location="run-0001/pose_ingestion",
            ),
        )
    return StageRequest(
        stage_id="state_estimation",
        inputs=inputs,
        components={},
        config_digest="test",
        output_dir=workspace / "run-0001" / "state_estimation",
        workspace=workspace,
    )


def _estimator() -> ExternalPoseEstimator:
    return ExternalPoseEstimator(
        ExternalPoseConfig(reference_frame=FrameId("map"), body_frame=FrameId("body"))
    )


class TestNoAuxiliaryPoseInput:
    def test_behaves_exactly_as_before_the_bridge_existed(self, tmp_path: Path) -> None:
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=None)

        with pytest.raises(GeometryPreflightError):
            executor.execute(request)


class TestMergingAnOperationalAuxiliaryPose:
    def test_odometry_role_is_merged_in_and_drives_the_trajectory(self, tmp_path: Path) -> None:
        poses = [_pose(i, seconds=i, pose_role="odometry") for i in range(3)]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        ref = executor.execute(request)

        assert ref.contract == "StateEstimationRunArtifact"

    def test_external_localization_role_is_merged_in(self, tmp_path: Path) -> None:
        poses = [_pose(i, seconds=i, pose_role="external_localization") for i in range(3)]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        executor.execute(request)  # does not raise


class TestGroundTruthSafeguard:
    def test_ground_truth_pose_is_ignored_by_default(self, tmp_path: Path) -> None:
        """Issue #555 acceptance criterion: default behavior with ground-truth pose present is
        unchanged from today -- exactly the "no auxiliary" MissingEstimatorInputError."""
        poses = [_pose(i, seconds=i, pose_role="ground_truth") for i in range(3)]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        with pytest.raises(GeometryPreflightError):
            executor.execute(request)

    def test_ground_truth_pose_is_used_with_the_explicit_opt_in(self, tmp_path: Path) -> None:
        poses = [_pose(i, seconds=i, pose_role="ground_truth") for i in range(3)]
        executor = StateEstimationExecutor(_estimator(), allow_ground_truth_trajectory=True)
        request = _request(tmp_path, pose_observations=poses)

        executor.execute(request)  # does not raise: the opt-in let it through

    def test_a_disjoint_clock_ground_truth_pose_is_still_ignored_by_default(
        self, tmp_path: Path
    ) -> None:
        """The ground-truth safeguard must be checked before clock plausibility: a run in the
        default ``operational_only`` mode drops a ground-truth-tagged auxiliary pose unconditio-
        nally, so an incompatible clock on data that was never going to be used must not block
        the run either -- exactly the same outcome as a well-behaved ground-truth pose."""
        far_future = 100_000_000
        poses = [_pose(i, seconds=far_future + i, pose_role="ground_truth") for i in range(3)]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        with pytest.raises(GeometryPreflightError):
            executor.execute(request)

    def test_a_disjoint_clock_ground_truth_pose_is_rejected_with_the_opt_in(
        self, tmp_path: Path
    ) -> None:
        """With the opt-in, the ground-truth pose is no longer dropped, so it must actually pass
        the clock check like any other merged auxiliary sequence."""
        far_future = 100_000_000
        poses = [_pose(i, seconds=far_future + i, pose_role="ground_truth") for i in range(3)]
        executor = StateEstimationExecutor(_estimator(), allow_ground_truth_trajectory=True)
        request = _request(tmp_path, pose_observations=poses)

        with pytest.raises(ExecutorError, match="clock"):
            executor.execute(request)


class TestPoseRoleValidation:
    def test_missing_pose_role_on_the_auxiliary_sequence_fails_loudly(self, tmp_path: Path) -> None:
        poses = [_pose(i, seconds=i, pose_role=None) for i in range(2)]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        with pytest.raises(ExecutorError, match="pose_role"):
            executor.execute(request)

    def test_inconsistent_pose_role_on_the_auxiliary_sequence_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        poses = [
            _pose(0, seconds=0, pose_role="ground_truth"),
            _pose(1, seconds=1, pose_role="odometry"),
        ]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        with pytest.raises(ExecutorError, match="pose_role"):
            executor.execute(request)


class TestClockCompatibilityValidation:
    def test_a_disjoint_clock_range_refuses_to_merge(self, tmp_path: Path) -> None:
        far_future = 100_000_000
        poses = [_pose(i, seconds=far_future + i, pose_role="odometry") for i in range(3)]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        with pytest.raises(ExecutorError, match="clock"):
            executor.execute(request)


class TestAuxiliaryLineageIsPersisted:
    """A ``StateEstimationRunArtifact`` must name every artifact that actually contributed to
    its trajectory, so a consumer holding only this run can close the lineage of every pose it
    carries -- not just the main sequence's."""

    def test_a_merged_auxiliary_pose_is_named_in_the_run_s_own_manifest(
        self, tmp_path: Path
    ) -> None:
        poses = [_pose(i, seconds=i, pose_role="odometry") for i in range(3)]
        executor = StateEstimationExecutor(_estimator())
        request = _request(tmp_path, pose_observations=poses)

        executor.execute(request)

        assert request.output_dir is not None
        run = StateEstimationRunReader(request.output_dir)
        manifest = run.manifest
        assert manifest.auxiliary_sequence_artifact_id == SequenceArtifactId("pose-sequence")
        assert manifest.auxiliary_selection_id == selection_identity(
            SequenceArtifactId("pose-sequence"), FullSequenceSelection()
        )

        # A consumer holding only this run must be able to close the lineage of every pose: open
        # the artifact the manifest itself names, and confirm each pose's source observations
        # actually resolve in it.
        pose_dir = tmp_path / "ws" / "run-0001" / "pose_ingestion"
        pose_sequence = SequenceArtifactReader(pose_dir)
        pose_observation_ids = {
            observation.observation_id for observation in pose_sequence.list_observations()
        }
        trajectory = run.trajectory()
        for pose in trajectory.poses:
            assert set(pose.provenance.source_observation_ids) <= pose_observation_ids

    def test_a_dropped_ground_truth_auxiliary_leaves_no_auxiliary_lineage(
        self, tmp_path: Path
    ) -> None:
        """The default safeguard drops the ground-truth pose from the merge entirely; since it
        never contributed, the manifest must not claim it did. The main sequence carries its own
        operational poses here, so the run succeeds independently of the dropped auxiliary."""
        workspace = tmp_path / "ws"
        main_dir = workspace / "run-0001" / "ingestion"
        _write_sequence(
            main_dir,
            artifact_id="main-sequence",
            observations=[_pose(i, seconds=i, pose_role="odometry") for i in range(3)],
        )
        pose_dir = workspace / "run-0001" / "pose_ingestion"
        _write_sequence(
            pose_dir,
            artifact_id="pose-sequence",
            observations=[_pose(i, seconds=i, pose_role="ground_truth") for i in range(3)],
        )
        request = StageRequest(
            stage_id="state_estimation",
            inputs={
                "sequence": (
                    ArtifactRef(
                        stage_id="ingestion",
                        contract="SequenceArtifact",
                        artifact_id="main-sequence",
                        content_hash="sha256:main",
                        location="run-0001/ingestion",
                    ),
                ),
                "pose_sequence": (
                    ArtifactRef(
                        stage_id="pose_ingestion",
                        contract="SequenceArtifact",
                        artifact_id="pose-sequence",
                        content_hash="sha256:pose",
                        location="run-0001/pose_ingestion",
                    ),
                ),
            },
            components={},
            config_digest="test",
            output_dir=workspace / "run-0001" / "state_estimation",
            workspace=workspace,
        )
        executor = StateEstimationExecutor(_estimator())  # no ground-truth opt-in: default

        executor.execute(request)

        assert request.output_dir is not None
        manifest = StateEstimationRunReader(request.output_dir).manifest
        assert manifest.auxiliary_sequence_artifact_id is None
        assert manifest.auxiliary_selection_id is None
