import math
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from calibration_builders import (
    QUARTER_TURN_Z,
    calibration,
    imu_with_motion,
    lidar_with_points,
    rigid,
    sensor_run,
)
from pose_builders import make_request

import contextmap.state_estimation.backends.fast_lio as fast_lio_module
from contextmap.ingestion import FrameId, ImuObservation, LidarObservation, SourceObservation
from contextmap.state_estimation import (
    DiagnosticSeverity,
    GeometryPreflightError,
    MissingEstimatorInputError,
    PoseValidity,
    StateEstimationError,
    StateEstimationRequest,
    StateEstimationRunId,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    StateEstimator,
    calibration_identity,
    execute_state_estimation,
)
from contextmap.state_estimation.backends.fast_lio import (
    FastLioConfig,
    FastLioEstimator,
    FastLioFailure,
    FastLioFailureKind,
    FastLioJob,
    FastLioRawPose,
    FastLioRunOutput,
)

MS = 1_000_000
EXTRINSIC = rigid("imu", "velodyne", (0.1, 0.0, 0.2), QUARTER_TURN_Z)


def _config(**overrides: object) -> FastLioConfig:
    values: dict[str, object] = {
        "reference_frame": FrameId("fast_lio_init"),
        "body_frame": FrameId("imu"),
        "fast_lio_ref": "v1.0.0",
        "scan_period_ns": 100 * MS,
    }
    values.update(overrides)
    return FastLioConfig(**values)  # type: ignore[arg-type]


def _poses_for(job: FastLioJob) -> tuple[FastLioRawPose, ...]:
    """One pose 50 ms into each scan interval, moving 1 m per scan along +x."""
    return tuple(
        FastLioRawPose(
            timestamp_ns=scan.timestamp.total_nanoseconds() + 50 * MS,
            translation=(float(index), 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            covariance=None,
        )
        for index, scan in enumerate(job.lidar)
    )


class _FakeRunner:
    """Stands in for the FAST-LIO process: records the job and returns canned poses."""

    def __init__(
        self,
        make_output: Callable[[FastLioJob], FastLioRunOutput] | None = None,
        failure: FastLioFailure | None = None,
    ) -> None:
        self.jobs: list[FastLioJob] = []
        self._make_output = make_output or (
            lambda job: FastLioRunOutput(poses=_poses_for(job), reported_ref="v1.0.0", warnings=())
        )
        self._failure = failure

    def describe(self) -> Mapping[str, object]:
        return {"runner": "fake"}

    def run(self, job: FastLioJob) -> FastLioRunOutput:
        self.jobs.append(job)
        if self._failure is not None:
            raise self._failure
        return self._make_output(job)


def _request(
    observations: list[SourceObservation] | None = None, *, with_calibration: bool = True
) -> StateEstimationRequest:
    request = make_request(observations if observations is not None else sensor_run())
    return StateEstimationRequest(
        trajectory_id=request.trajectory_id,
        sequence_artifact_id=request.sequence_artifact_id,
        selection_id=request.selection_id,
        observations=request.observations,
        calibration=calibration(EXTRINSIC) if with_calibration else None,
    )


# --- Canonical output -------------------------------------------------------


def test_publishes_a_canonical_trajectory_with_explicit_frames_and_provenance() -> None:
    runner = _FakeRunner()
    estimator = FastLioEstimator(_config(), runner)
    request = _request()

    result = estimator.estimate(request)

    trajectory = result.trajectory
    assert (trajectory.reference_frame, trajectory.body_frame) == (
        FrameId("fast_lio_init"),
        FrameId("imu"),
    )
    assert [pose.translation_m for pose in trajectory.poses] == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    assert all(pose.validity is PoseValidity.VALID for pose in trajectory.poses)
    provenance = trajectory.provenance
    assert provenance.estimator == estimator.estimator_provenance()
    assert provenance.estimator.backend_id == "fast_lio"
    assert provenance.estimator.backend_version == "v1.0.0"
    assert provenance.calibration_identity == calibration_identity(request.calibration)
    assert provenance.sequence_artifact_id == request.sequence_artifact_id


def test_every_pose_traces_back_to_the_scan_it_was_estimated_from() -> None:
    result = FastLioEstimator(_config(), _FakeRunner()).estimate(_request())

    assert [pose.provenance.source_observation_ids[0] for pose in result.trajectory.poses] == [
        "scan-0000",
        "scan-0001",
        "scan-0002",
    ]


def test_the_extrinsic_comes_from_canonical_calibration_with_explicit_direction() -> None:
    runner = _FakeRunner()

    FastLioEstimator(_config(), runner).estimate(_request())

    (job,) = runner.jobs
    assert (job.imu_frame, job.lidar_frame) == (FrameId("imu"), FrameId("velodyne"))
    assert job.lidar_in_imu_translation == pytest.approx((0.1, 0.0, 0.2))
    assert job.lidar_in_imu_rotation == pytest.approx(QUARTER_TURN_Z)
    assert len(job.lidar) == 3
    assert len(job.imu) == 32
    assert job.clock_id == "fixture:header"
    assert job.scan_period_ns == 100 * MS


def test_the_run_states_that_input_scans_are_not_motion_corrected() -> None:
    result = FastLioEstimator(_config(), _FakeRunner()).estimate(_request())

    (diagnostic,) = result.diagnostics
    assert diagnostic.code == "fast_lio.raw_scans_not_deskewed"
    assert diagnostic.severity is DiagnosticSeverity.INFO
    assert result.consumed_observation_count == 3 + 32
    assert result.rejected_observation_count == 0


def test_covariance_is_kept_only_when_the_estimator_reports_it() -> None:
    covariance = tuple(float(i) for i in range(36))

    def output(job: FastLioJob) -> FastLioRunOutput:
        poses = list(_poses_for(job))
        poses[0] = FastLioRawPose(
            timestamp_ns=poses[0].timestamp_ns,
            translation=poses[0].translation,
            orientation=poses[0].orientation,
            covariance=covariance,
        )
        return FastLioRunOutput(poses=tuple(poses), reported_ref="v1.0.0", warnings=())

    poses = FastLioEstimator(_config(), _FakeRunner(output)).estimate(_request()).trajectory.poses

    assert poses[0].covariance == covariance
    assert poses[1].covariance is None


def test_runner_warnings_and_gaps_are_recorded_not_hidden() -> None:
    def output(job: FastLioJob) -> FastLioRunOutput:
        first, _, third = _poses_for(job)
        return FastLioRunOutput(
            poses=(first, third), reported_ref="v1.0.0", warnings=("degenerate",)
        )

    result = FastLioEstimator(_config(max_gap_ns=150 * MS), _FakeRunner(output)).estimate(
        _request()
    )

    codes = [diagnostic.code for diagnostic in result.diagnostics]
    assert "fast_lio.runner_warning" in codes
    assert "fast_lio.timestamp_gap" in codes
    assert len(result.trajectory.gaps) == 1


def test_fast_lio_is_a_state_estimator_that_declares_its_geometry_requirements() -> None:
    estimator = FastLioEstimator(_config(), _FakeRunner())

    requirements = estimator.geometry_requirements()

    assert isinstance(estimator, StateEstimator)
    assert requirements.modalities == frozenset({"lidar", "imu"})
    assert [(r.from_endpoint, r.to_endpoint) for r in requirements.static_relations] == [
        ("lidar", "imu")
    ]
    assert (requirements.reference_frame, requirements.body_frame) == (
        FrameId("fast_lio_init"),
        FrameId("imu"),
    )
    assert requirements.camera_model_modalities == frozenset()


# --- Inputs are validated before the estimator runs -------------------------


@pytest.mark.parametrize("missing", ["LidarObservation", "ImuObservation"])
def test_missing_lidar_or_imu_is_a_missing_input(missing: str) -> None:
    everything = sensor_run()
    observations = _imus(everything) if missing == "LidarObservation" else _lidars(everything)
    runner = _FakeRunner()

    with pytest.raises(MissingEstimatorInputError, match=missing):
        FastLioEstimator(_config(), runner).estimate(_request(observations))

    assert runner.jobs == []


def test_a_missing_extrinsic_never_starts_the_estimator() -> None:
    runner = _FakeRunner()

    with pytest.raises(MissingEstimatorInputError, match="calibration"):
        FastLioEstimator(_config(), runner).estimate(_request(with_calibration=False))
    unrelated = StateEstimationRequest(
        **{**_request().__dict__, "calibration": calibration(rigid("world", "anchor"))}
    )
    with pytest.raises(StateEstimationError, match="static"):
        FastLioEstimator(_config(), runner).estimate(unrelated)

    assert runner.jobs == []


def _lidars(observations: list[SourceObservation]) -> list[SourceObservation]:
    return [o for o in observations if isinstance(o, LidarObservation)]


def _imus(observations: list[SourceObservation]) -> list[SourceObservation]:
    return [o for o in observations if isinstance(o, ImuObservation)]


def _imu_stream(
    count: int, *, frame: str = "imu", clock_id: str = "fixture:header"
) -> list[SourceObservation]:
    return [
        imu_with_motion(f"imu-{i:05d}", time_ns=i * 10 * MS, frame=frame, clock_id=clock_id)
        for i in range(count)
    ]


def _inconsistent_streams() -> dict[str, tuple[list[SourceObservation], str]]:
    scans = _lidars(sensor_run())
    return {
        "lidar spans two frames": (
            [
                *scans,
                lidar_with_points("scan-x", time_ns=350 * MS, frame="other"),
                *_imu_stream(40),
            ],
            "frame",
        ),
        "imu is not in the body frame": ([*scans, *_imu_stream(40, frame="base")], "body frame"),
        "lidar and imu use different clocks": (
            [*scans, *_imu_stream(40, clock_id="another:clock")],
            "clock",
        ),
        "imu stops before the last scan ends": ([*scans, *_imu_stream(25)], "cover"),
        "lidar timestamps go backwards": (
            [*scans, lidar_with_points("scan-early", time_ns=10 * MS), *_imu_stream(40)],
            "increasing",
        ),
    }


@pytest.mark.parametrize("case", list(_inconsistent_streams()))
def test_inconsistent_sensor_streams_are_rejected_with_an_actionable_error(case: str) -> None:
    observations, needle = _inconsistent_streams()[case]
    runner = _FakeRunner()

    with pytest.raises(StateEstimationError, match=needle):
        FastLioEstimator(_config(), runner).estimate(_request(observations))

    assert runner.jobs == []


# --- Failures are structured and never fall back ----------------------------


@pytest.mark.parametrize("kind", list(FastLioFailureKind))
def test_runner_failures_surface_with_their_kind_and_no_fallback(kind: FastLioFailureKind) -> None:
    failure = FastLioFailure(kind=kind, detail="boom", stderr_tail="last log line")
    estimator = FastLioEstimator(_config(), _FakeRunner(failure=failure))

    with pytest.raises(FastLioFailure) as raised:
        estimator.estimate(_request())

    assert raised.value.kind is kind
    assert "boom" in str(raised.value)
    assert raised.value.stderr_tail == "last log line"


def test_a_run_that_publishes_no_pose_fails() -> None:
    runner = _FakeRunner(lambda job: FastLioRunOutput(poses=(), reported_ref="v1.0.0", warnings=()))

    with pytest.raises(FastLioFailure) as raised:
        FastLioEstimator(_config(), runner).estimate(_request())

    assert raised.value.kind is FastLioFailureKind.EMPTY_OUTPUT


def test_a_different_fast_lio_version_than_the_configured_one_fails() -> None:
    runner = _FakeRunner(
        lambda job: FastLioRunOutput(poses=_poses_for(job), reported_ref="v9.9.9", warnings=())
    )

    with pytest.raises(FastLioFailure) as raised:
        FastLioEstimator(_config(), runner).estimate(_request())

    assert raised.value.kind is FastLioFailureKind.VERSION_MISMATCH


def _with_pose(index: int, **changes: object) -> Callable[[FastLioJob], FastLioRunOutput]:
    def build(job: FastLioJob) -> FastLioRunOutput:
        poses = list(_poses_for(job))
        base = poses[index]
        values = {
            "timestamp_ns": base.timestamp_ns,
            "translation": base.translation,
            "orientation": base.orientation,
            "covariance": base.covariance,
            **changes,
        }
        poses[index] = FastLioRawPose(**values)  # type: ignore[arg-type]
        return FastLioRunOutput(poses=tuple(poses), reported_ref="v1.0.0", warnings=())

    return build


@pytest.mark.parametrize(
    "changes",
    [
        {"translation": (math.nan, 0.0, 0.0)},
        {"orientation": (0.0, 0.0, 0.0, 1.5)},
        {"covariance": (math.nan,) * 36},
        {"timestamp_ns": 5 * MS},
        {"timestamp_ns": 1_000 * MS},
    ],
)
def test_invalid_estimator_output_fails_instead_of_being_repaired(
    changes: dict[str, object],
) -> None:
    runner = _FakeRunner(_with_pose(1, **changes))

    with pytest.raises(FastLioFailure) as raised:
        FastLioEstimator(_config(), runner).estimate(_request())

    assert raised.value.kind is FastLioFailureKind.INVALID_OUTPUT


def test_a_slightly_denormalized_orientation_is_renormalized_and_recorded() -> None:
    runner = _FakeRunner(_with_pose(0, orientation=(0.0, 0.0, 0.0, 1.0004)))

    pose = FastLioEstimator(_config(), runner).estimate(_request()).trajectory.poses[0]

    assert pose.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert any("renormalized" in note for note in pose.provenance.conversions_applied)


# --- Configuration and reproducibility --------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_frame": FrameId("")},
        {"reference_frame": FrameId("imu")},
        {"fast_lio_ref": ""},
        {"scan_period_ns": 0},
        {"max_gap_ns": 0},
        {"orientation_norm_tolerance": -1.0},
    ],
)
def test_configuration_rejects_impossible_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"frame|fast_lio_ref|scan_period_ns|max_gap_ns|tolerance"):
        _config(**overrides)


def test_the_fingerprint_covers_the_configuration_parameters_and_the_runner() -> None:
    base = FastLioEstimator(_config(), _FakeRunner()).estimator_provenance()

    assert base == FastLioEstimator(_config(), _FakeRunner()).estimator_provenance()
    variants = [
        FastLioEstimator(_config(fast_lio_ref="v2"), _FakeRunner()),
        FastLioEstimator(_config(parameters={"lidar_type": 2}), _FakeRunner()),
        FastLioEstimator(_config(scan_period_ns=50 * MS), _FakeRunner()),
    ]
    fingerprints = {e.estimator_provenance().configuration_fingerprint for e in variants}
    assert base.configuration_fingerprint not in fingerprints
    assert len(fingerprints) == 3


def test_the_backend_carries_no_dataset_specific_knowledge() -> None:
    source = Path(fast_lio_module.__file__).read_text(encoding="utf-8").lower()

    for dataset_specific in ("corridor", "cerberus", "gazebo", "-gt.txt"):
        assert dataset_specific not in source


# --- Run through the application service and the artifact -------------------


def test_a_missing_extrinsic_blocks_the_run_before_the_estimator_starts() -> None:
    runner = _FakeRunner()

    with pytest.raises(GeometryPreflightError) as raised:
        execute_state_estimation(
            FastLioEstimator(_config(), runner), _request(with_calibration=False)
        )

    assert "missing_calibration" in {finding.code for finding in raised.value.report.blockers}
    assert runner.jobs == []


def test_a_fast_lio_run_persists_with_the_calibration_identity_it_used(tmp_path: Path) -> None:
    estimator = FastLioEstimator(_config(), _FakeRunner())
    request = _request()

    outcome = execute_state_estimation(estimator, request)
    manifest = StateEstimationRunWriter(
        workspace_root=tmp_path,
        sequence_name="seq",
        run_id=StateEstimationRunId("run-0001"),
        run_index=1,
        selection_label="full-sequence",
        backend_label="fast-lio",
    ).finalize(outcome, runtime_s=1.5)

    reader = StateEstimationRunReader(
        tmp_path / "runs" / "state-estimation" / "seq" / "run-0001__full-sequence__fast-lio"
    )
    assert manifest.estimator.backend_id == "fast_lio"
    assert manifest.calibration_identity == calibration_identity(request.calibration)
    assert reader.trajectory() == outcome.result.trajectory
