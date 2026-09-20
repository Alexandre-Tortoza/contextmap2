import math

import pytest
from calibration_builders import (
    QUARTER_TURN_Z,
    calibration,
    camera_entry,
    closing_transform,
    image,
    imu,
    lidar,
    rigid,
)
from pose_builders import make_external_pose, make_request

from contextmap.ingestion import CalibrationReferenceId, FrameId, SourceObservation
from contextmap.state_estimation import (
    BODY_ENDPOINT,
    GeometryPreflightError,
    GeometryRequirements,
    PreflightStatus,
    PreflightTolerances,
    StateEstimationRequest,
    StateEstimationResult,
    StaticRelationRequirement,
    calibration_identity,
    execute_state_estimation,
    run_geometry_preflight,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)

LIDAR_IMU = StaticRelationRequirement(from_endpoint="lidar", to_endpoint="imu")


def _fast_lio_like() -> GeometryRequirements:
    """A LiDAR-inertial estimator: needs both streams and the lidar<->imu extrinsic."""
    return GeometryRequirements(
        capability="state_estimation:lidar_inertial",
        modalities=frozenset({"lidar", "imu"}),
        static_relations=(LIDAR_IMU,),
        reference_frame=FrameId("map"),
        body_frame=FrameId("imu"),
    )


def _sensors() -> list[SourceObservation]:
    return [lidar(frame="velodyne"), imu(frame="imu")]


def _codes(findings: tuple[object, ...]) -> set[str]:
    return {finding.code for finding in findings}  # type: ignore[attr-defined]


# --- Ready and blocked runs -------------------------------------------------


def test_an_external_pose_run_is_ready_without_any_calibration() -> None:
    execution = GeometryRequirements(
        capability="state_estimation:external_pose",
        modalities=frozenset({"external_pose"}),
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
    )

    report = run_geometry_preflight(
        observations=[make_external_pose(0)], calibration=None, execution=execution
    )

    assert report.status is PreflightStatus.READY
    assert report.required_inputs == frozenset({"external_pose"})
    assert report.available_inputs == frozenset({"external_pose"})
    assert report.calibration_identity is None
    assert report.blockers == ()


def test_a_lidar_inertial_run_with_a_valid_extrinsic_is_ready() -> None:
    extrinsic = rigid("imu", "velodyne", (0.1, 0.0, 0.2), QUARTER_TURN_Z)

    report = run_geometry_preflight(
        observations=_sensors(), calibration=calibration(extrinsic), execution=_fast_lio_like()
    )

    assert report.status is PreflightStatus.READY
    assert report.blockers == ()
    assert report.frame_graph.reference_frame == FrameId("map")
    assert report.frame_graph.body_frame == FrameId("imu")
    assert report.frame_graph.static_frames == (FrameId("imu"), FrameId("velodyne"))
    assert all(check.passed for check in report.transform_checks)
    assert all(check.passed for check in report.clock_checks)
    assert report.calibration_identity is not None


def test_transform_checks_cover_finiteness_orthonormality_and_the_inverse_round_trip() -> None:
    report = run_geometry_preflight(
        observations=_sensors(),
        calibration=calibration(rigid("imu", "velodyne", (0.1, 0.0, 0.2), QUARTER_TURN_Z)),
        execution=_fast_lio_like(),
    )

    names = {check.check for check in report.transform_checks}

    assert {"finite", "rotation_orthonormal", "inverse_round_trip", "relation_available"} <= names


def test_a_missing_required_input_blocks() -> None:
    report = run_geometry_preflight(
        observations=[lidar(frame="velodyne")],
        calibration=calibration(rigid("imu", "velodyne")),
        execution=_fast_lio_like(),
    )

    assert report.status is PreflightStatus.BLOCKED
    assert "missing_input" in _codes(report.blockers)
    assert any("imu" in blocker.message for blocker in report.blockers)


def test_a_required_relation_without_any_calibration_blocks() -> None:
    report = run_geometry_preflight(
        observations=_sensors(), calibration=None, execution=_fast_lio_like()
    )

    assert report.status is PreflightStatus.BLOCKED
    assert "missing_calibration" in _codes(report.blockers)


def test_a_required_relation_without_a_static_path_blocks() -> None:
    report = run_geometry_preflight(
        observations=_sensors(),
        calibration=calibration(rigid("world", "anchor")),
        execution=_fast_lio_like(),
    )

    assert report.status is PreflightStatus.BLOCKED
    assert "missing_static_transform" in _codes(report.blockers)


@pytest.mark.parametrize(
    ("bad_transform", "failed_check"),
    [
        (rigid("imu", "velodyne", (math.nan, 0.0, 0.0)), "finite"),
        (rigid("imu", "velodyne", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.05)), "rotation_orthonormal"),
    ],
)
def test_an_invalid_static_transform_blocks(bad_transform: object, failed_check: str) -> None:
    report = run_geometry_preflight(
        observations=_sensors(),
        calibration=calibration(bad_transform),  # type: ignore[arg-type]
        execution=_fast_lio_like(),
    )

    assert report.status is PreflightStatus.BLOCKED
    assert "invalid_transform" in _codes(report.blockers)
    assert any(
        check.check == failed_check and not check.passed for check in report.transform_checks
    )


def _inconsistent_loop() -> list:
    body_lidar = rigid("imu", "velodyne", (1.0, 0.0, 0.0))
    body_camera = rigid("imu", "camera", (0.0, 2.0, 0.0))
    closing = closing_transform(body_lidar, body_camera, parent="velodyne", child="camera")
    x, y, z = closing.translation
    wrong = rigid("velodyne", "camera", (x + 0.1, y, z), closing.rotation)
    return [body_lidar, body_camera, wrong]


def test_an_ambiguous_frame_graph_blocks() -> None:
    report = run_geometry_preflight(
        observations=_sensors(),
        calibration=calibration(*_inconsistent_loop()),
        execution=_fast_lio_like(),
    )

    assert report.status is PreflightStatus.BLOCKED
    assert "ambiguous_frame_graph" in _codes(report.blockers)


def test_an_inconsistency_in_an_unrelated_component_only_warns() -> None:
    unrelated = [
        rigid("world", "anchor_a", (1.0, 0.0, 0.0)),
        rigid("world", "anchor_b", (0.0, 1.0, 0.0)),
        rigid("anchor_a", "anchor_b", (9.0, 9.0, 9.0)),
    ]

    report = run_geometry_preflight(
        observations=_sensors(),
        calibration=calibration(rigid("imu", "velodyne"), *unrelated),
        execution=_fast_lio_like(),
    )

    assert report.status is PreflightStatus.READY
    assert "unused_calibration_inconsistent" in _codes(report.warnings)


def test_a_loose_loop_tolerance_can_accept_a_small_inconsistency() -> None:
    report = run_geometry_preflight(
        observations=_sensors(),
        calibration=calibration(*_inconsistent_loop()),
        execution=_fast_lio_like(),
        tolerances=PreflightTolerances(loop_translation_m=0.5),
    )

    assert report.status is PreflightStatus.READY


def test_a_modality_spread_over_several_frames_makes_a_relation_ambiguous() -> None:
    report = run_geometry_preflight(
        observations=[
            lidar("scan-1", frame="velodyne_a"),
            lidar("scan-2", frame="velodyne_b"),
            imu(),
        ],
        calibration=calibration(rigid("imu", "velodyne_a"), rigid("imu", "velodyne_b")),
        execution=_fast_lio_like(),
    )

    assert report.status is PreflightStatus.BLOCKED
    assert "ambiguous_endpoint" in _codes(report.blockers)


def test_modalities_in_different_clock_domains_block() -> None:
    report = run_geometry_preflight(
        observations=[lidar(frame="velodyne", clock_id="clock:a"), imu(clock_id="clock:b")],
        calibration=calibration(rigid("imu", "velodyne")),
        execution=_fast_lio_like(),
    )

    assert report.status is PreflightStatus.BLOCKED
    assert "clock_domain_mismatch" in _codes(report.blockers)
    assert any(not check.passed for check in report.clock_checks)


def test_an_empty_clock_identity_blocks() -> None:
    report = run_geometry_preflight(
        observations=[lidar(frame="velodyne", clock_id=""), imu(clock_id="")],
        calibration=calibration(rigid("imu", "velodyne")),
        execution=_fast_lio_like(),
    )

    assert "missing_clock" in _codes(report.blockers)


# --- Downstream readiness ---------------------------------------------------


def _mapping() -> GeometryRequirements:
    return GeometryRequirements(
        capability="geometric_mapping",
        modalities=frozenset({"lidar"}),
        static_relations=(
            StaticRelationRequirement(from_endpoint="lidar", to_endpoint=BODY_ENDPOINT),
        ),
    )


def _association() -> GeometryRequirements:
    return GeometryRequirements(
        capability="sensor_association",
        modalities=frozenset({"image"}),
        static_relations=(
            StaticRelationRequirement(from_endpoint="image", to_endpoint=BODY_ENDPOINT),
        ),
        camera_model_modalities=frozenset({"image"}),
    )


def test_camera_intrinsics_never_block_a_lidar_inertial_run_but_are_reported_downstream() -> None:
    observations = [*_sensors(), image(frame="camera")]

    report = run_geometry_preflight(
        observations=observations,
        calibration=calibration(rigid("imu", "velodyne")),
        execution=_fast_lio_like(),
        downstream=[_mapping(), _association()],
    )

    readiness = {item.capability: item for item in report.downstream}
    assert report.status is PreflightStatus.READY
    assert readiness["geometric_mapping"].ready
    assert not readiness["sensor_association"].ready
    assert {"missing_static_transform", "missing_camera_model"} <= _codes(
        readiness["sensor_association"].missing
    )
    assert "downstream_not_ready" in _codes(report.warnings)


def test_downstream_capabilities_are_ready_when_their_prerequisites_exist() -> None:
    observations = [*_sensors(), image(frame="camera", calibration_id="cam0")]
    entries = {CalibrationReferenceId("cam0"): camera_entry("cam0")}

    report = run_geometry_preflight(
        observations=observations,
        calibration=calibration(
            rigid("imu", "velodyne"), rigid("imu", "camera", (0.0, 0.1, 0.0)), entries=entries
        ),
        execution=_fast_lio_like(),
        downstream=[_mapping(), _association()],
    )

    assert all(item.ready for item in report.downstream)
    assert "downstream_not_ready" not in _codes(report.warnings)


# --- Calibration identity ---------------------------------------------------


def test_calibration_identity_is_deterministic_and_tracks_the_transforms() -> None:
    first = calibration(rigid("imu", "velodyne", (0.1, 0.0, 0.0)))
    changed = calibration(rigid("imu", "velodyne", (0.2, 0.0, 0.0)))

    assert calibration_identity(first) == calibration_identity(
        calibration(rigid("imu", "velodyne", (0.1, 0.0, 0.0)))
    )
    assert calibration_identity(first) != calibration_identity(changed)
    assert calibration_identity(None) is None


# --- Requirements contract --------------------------------------------------


def test_requirements_reject_unknown_endpoints() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        StaticRelationRequirement(from_endpoint="radar", to_endpoint="imu")


def test_the_executing_capability_must_declare_its_dynamic_frames() -> None:
    execution = GeometryRequirements(capability="x", modalities=frozenset({"imu"}))

    with pytest.raises(ValueError, match="reference_frame and body_frame"):
        run_geometry_preflight(observations=[imu()], calibration=None, execution=execution)


# --- Blocking before the estimator starts -----------------------------------


class _SpyEstimator:
    def __init__(self, requirements: GeometryRequirements) -> None:
        self._requirements = requirements
        self.calls = 0

    def geometry_requirements(self) -> GeometryRequirements:
        return self._requirements

    def estimator_provenance(self):  # type: ignore[no-untyped-def]
        raise AssertionError("must not be called when preflight blocks")

    def estimate(self, request: StateEstimationRequest) -> StateEstimationResult:
        self.calls += 1
        raise AssertionError("must not run when preflight blocks")


def test_a_blocked_preflight_stops_the_run_before_the_estimator_starts() -> None:
    spy = _SpyEstimator(_fast_lio_like())
    request = make_request(_sensors())

    with pytest.raises(GeometryPreflightError) as raised:
        execute_state_estimation(spy, request)  # type: ignore[arg-type]

    assert spy.calls == 0
    assert raised.value.report.status is PreflightStatus.BLOCKED
    assert "missing_calibration" in str(raised.value)


def test_a_ready_run_executes_the_estimator_and_keeps_the_preflight_report() -> None:
    estimator = ExternalPoseEstimator(
        ExternalPoseConfig(reference_frame=FrameId("map"), body_frame=FrameId("body"))
    )
    request = make_request([make_external_pose(index) for index in range(3)])

    outcome = execute_state_estimation(estimator, request)

    assert outcome.preflight.status is PreflightStatus.READY
    assert len(outcome.result.trajectory.poses) == 3
    assert outcome.preflight.frame_graph.reference_frame == FrameId("map")
