import math
from pathlib import Path

import pytest
from pose_builders import (
    make_external_pose,
    make_imu_observation,
    make_request,
)

import contextmap.state_estimation.backends.external_pose as external_pose_module
from contextmap.ingestion import ExternalPoseMeasurement, FrameId, SourceObservation
from contextmap.state_estimation import (
    DiagnosticSeverity,
    LookupPolicy,
    MissingEstimatorInputError,
    PoseEstimate,
    PoseEstimateId,
    PoseValidity,
    ResolvedPose,
    StateEstimationError,
    StateEstimator,
    TrajectoryLookup,
    pose_estimate_id_for,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
    InvalidPoseSampleError,
    InvalidSamplePolicy,
)

MS = 1_000_000


def _config(**overrides: object) -> ExternalPoseConfig:
    values: dict[str, object] = {
        "reference_frame": FrameId("map"),
        "body_frame": FrameId("body"),
    }
    values.update(overrides)
    return ExternalPoseConfig(**values)  # type: ignore[arg-type]


def _only_pose(measurement: ExternalPoseMeasurement) -> PoseEstimate:
    result = ExternalPoseEstimator(_config()).estimate(make_request([measurement]))
    return result.trajectory.poses[0]


# --- Canonical output -------------------------------------------------------


def test_publishes_a_canonical_trajectory_from_external_poses() -> None:
    measurements = [make_external_pose(index) for index in range(4)]

    result = ExternalPoseEstimator(_config()).estimate(make_request(measurements))

    trajectory = result.trajectory
    assert (trajectory.reference_frame, trajectory.body_frame) == (FrameId("map"), FrameId("body"))
    assert len(trajectory.poses) == 4
    assert [pose.translation_m for pose in trajectory.poses] == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    ]
    assert all(pose.validity is PoseValidity.VALID for pose in trajectory.poses)
    assert result.consumed_observation_count == 4
    assert result.rejected_observation_count == 0
    assert result.diagnostics == ()


def test_every_pose_traces_back_to_its_source_measurement() -> None:
    measurements = [make_external_pose(index) for index in range(3)]

    result = ExternalPoseEstimator(_config()).estimate(make_request(measurements))

    for measurement, pose in zip(measurements, result.trajectory.poses, strict=True):
        assert pose.provenance.source_observation_ids == (measurement.observation_id,)
        assert pose.provenance.conversions_applied == ()
        assert pose.provenance.derived_from == ()
        assert pose.timestamp == measurement.timestamp


def test_estimate_ids_and_trajectory_provenance_are_deterministic() -> None:
    estimator = ExternalPoseEstimator(_config())
    request = make_request([make_external_pose(index) for index in range(3)])

    result = estimator.estimate(request)

    assert result == estimator.estimate(request)
    assert result.trajectory.poses[1].estimate_id == pose_estimate_id_for(
        trajectory_id=request.trajectory_id, index=1
    )
    provenance = result.trajectory.provenance
    assert provenance.estimator == estimator.estimator_provenance()
    assert provenance.estimator.backend_id == "external_pose"
    assert provenance.sequence_artifact_id == request.sequence_artifact_id
    assert provenance.selection_id == request.selection_id
    assert provenance.calibration_identity is None


def test_external_pose_is_a_state_estimator_and_feeds_the_common_lookup() -> None:
    estimator = ExternalPoseEstimator(_config())
    result = estimator.estimate(make_request([make_external_pose(index) for index in range(3)]))

    assert isinstance(estimator, StateEstimator)
    looked_up = TrajectoryLookup(result.trajectory).pose_at(
        result.trajectory.poses[1].timestamp, policy=LookupPolicy.exact()
    )
    assert isinstance(looked_up, ResolvedPose)


def test_covariance_is_propagated_only_when_the_source_provides_it() -> None:
    covariance = tuple(float(i) for i in range(36))
    measurements = [make_external_pose(0, covariance=covariance), make_external_pose(1)]

    poses = ExternalPoseEstimator(_config()).estimate(make_request(measurements)).trajectory.poses

    assert poses[0].covariance == covariance
    assert poses[1].covariance is None


def test_non_pose_observations_are_ignored() -> None:
    observations: list[SourceObservation] = [
        make_imu_observation("imu-0001", time_ns=50 * MS),
        make_external_pose(0),
        make_external_pose(1),
    ]

    result = ExternalPoseEstimator(_config()).estimate(make_request(observations))

    assert len(result.trajectory.poses) == 2
    assert result.consumed_observation_count == 2


def test_requires_external_pose_measurements() -> None:
    request = make_request([make_imu_observation("imu-0001", time_ns=0)])

    with pytest.raises(MissingEstimatorInputError, match="ExternalPoseMeasurement"):
        ExternalPoseEstimator(_config()).estimate(request)


# --- Orientation normalization ----------------------------------------------


def test_orientation_within_the_canonical_tolerance_is_kept_untouched() -> None:
    almost_unit = (0.0, 0.0, 0.0, 1.0 + 5e-7)

    pose = _only_pose(make_external_pose(0, orientation=almost_unit))

    assert pose.orientation == almost_unit
    assert pose.provenance.conversions_applied == ()


def test_orientation_within_the_source_tolerance_is_renormalized_and_recorded() -> None:
    slightly_long = (0.0, 0.0, 0.0, 1.0004)

    pose = _only_pose(make_external_pose(0, orientation=slightly_long))

    assert pose.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert any("renormalized" in note for note in pose.provenance.conversions_applied)


# --- Invalid samples: fail by default ---------------------------------------

INVALID_SAMPLES = [
    (
        "external_pose.frame_mismatch",
        make_external_pose(1, parent_frame="odom"),
    ),
    (
        "external_pose.frame_mismatch",
        make_external_pose(1, child_frame="imu"),
    ),
    (
        "external_pose.non_finite_translation",
        make_external_pose(1, translation=(math.nan, 0.0, 0.0)),
    ),
    (
        "external_pose.invalid_orientation",
        make_external_pose(1, orientation=(0.0, 0.0, 0.0, 1.01)),
    ),
    (
        "external_pose.invalid_orientation",
        make_external_pose(1, orientation=(0.0, 0.0, 0.0, 0.0)),
    ),
    (
        "external_pose.invalid_orientation",
        make_external_pose(1, orientation=(math.inf, 0.0, 0.0, 1.0)),
    ),
    (
        "external_pose.invalid_covariance",
        make_external_pose(1, covariance=(math.nan,) * 36),
    ),
    (
        "external_pose.clock_domain_mismatch",
        make_external_pose(1, clock_id="another:clock"),
    ),
    (
        "external_pose.non_increasing_timestamp",
        make_external_pose(1, time_ns=0),
    ),
    (
        "external_pose.non_increasing_timestamp",
        make_external_pose(1, time_ns=-5 * MS),
    ),
]


@pytest.mark.parametrize(("code", "invalid"), INVALID_SAMPLES)
def test_invalid_samples_fail_by_default_with_an_actionable_error(
    code: str, invalid: ExternalPoseMeasurement
) -> None:
    request = make_request([make_external_pose(0), invalid, make_external_pose(2)])

    with pytest.raises(InvalidPoseSampleError) as raised:
        ExternalPoseEstimator(_config()).estimate(request)

    assert raised.value.code == code
    assert raised.value.observation_id == invalid.observation_id
    assert str(invalid.observation_id) in str(raised.value)


# --- Invalid samples: explicit skip policy ----------------------------------


@pytest.mark.parametrize(("code", "invalid"), INVALID_SAMPLES)
def test_skip_policy_drops_the_sample_and_records_why(
    code: str, invalid: ExternalPoseMeasurement
) -> None:
    request = make_request([make_external_pose(0), invalid, make_external_pose(2)])

    result = ExternalPoseEstimator(
        _config(invalid_sample_policy=InvalidSamplePolicy.SKIP)
    ).estimate(request)

    assert [pose.provenance.source_observation_ids[0] for pose in result.trajectory.poses] == [
        make_external_pose(0).observation_id,
        make_external_pose(2).observation_id,
    ]
    assert result.consumed_observation_count == 3
    assert result.rejected_observation_count == 1
    (diagnostic,) = result.diagnostics
    assert diagnostic.code == code
    assert diagnostic.severity is DiagnosticSeverity.WARNING
    assert diagnostic.observation_id == invalid.observation_id
    # Os ids seguem a posição na trajetória publicada, sem buracos.
    assert [pose.estimate_id for pose in result.trajectory.poses] == [
        PoseEstimateId(f"{request.trajectory_id}--pose-000000"),
        PoseEstimateId(f"{request.trajectory_id}--pose-000001"),
    ]


def test_skip_policy_still_fails_when_no_sample_is_valid() -> None:
    request = make_request([make_external_pose(0, parent_frame="odom")])

    with pytest.raises(StateEstimationError, match="no valid"):
        ExternalPoseEstimator(_config(invalid_sample_policy=InvalidSamplePolicy.SKIP)).estimate(
            request
        )


# --- Gaps -------------------------------------------------------------------


def test_gaps_beyond_the_configured_tolerance_are_recorded_not_repaired() -> None:
    measurements = [
        make_external_pose(0),
        make_external_pose(1),
        make_external_pose(2, time_ns=1_000 * MS),
        make_external_pose(3, time_ns=1_100 * MS),
    ]

    result = ExternalPoseEstimator(_config(max_gap_ns=250 * MS)).estimate(
        make_request(measurements)
    )

    poses = result.trajectory.poses
    (gap,) = result.trajectory.gaps
    assert (gap.previous_estimate_id, gap.next_estimate_id) == (
        poses[1].estimate_id,
        poses[2].estimate_id,
    )
    assert gap.duration_ns == 900 * MS
    (diagnostic,) = result.diagnostics
    assert diagnostic.code == "external_pose.timestamp_gap"
    assert diagnostic.severity is DiagnosticSeverity.WARNING
    assert len(poses) == 4


def test_no_gaps_are_recorded_without_a_configured_tolerance() -> None:
    measurements = [make_external_pose(0), make_external_pose(1, time_ns=10_000 * MS)]

    result = ExternalPoseEstimator(_config()).estimate(make_request(measurements))

    assert result.trajectory.gaps == ()
    assert result.diagnostics == ()


# --- Configuration and dataset independence ---------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_frame": FrameId("")},
        {"body_frame": FrameId("")},
        {"reference_frame": FrameId("body")},
        {"max_gap_ns": 0},
        {"orientation_norm_tolerance": -1.0},
        {"orientation_norm_tolerance": math.nan},
    ],
)
def test_configuration_rejects_impossible_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"frame|max_gap_ns|orientation_norm_tolerance"):
        _config(**overrides)


def test_configuration_fingerprint_is_deterministic_and_sensitive_to_every_field() -> None:
    base = _config()

    assert base.fingerprint() == _config().fingerprint()
    variants = [
        _config(reference_frame=FrameId("odom")),
        _config(body_frame=FrameId("imu")),
        _config(invalid_sample_policy=InvalidSamplePolicy.SKIP),
        _config(max_gap_ns=1),
        _config(orientation_norm_tolerance=1e-2),
    ]
    assert len({variant.fingerprint() for variant in variants} | {base.fingerprint()}) == 6


def test_backend_carries_no_dataset_specific_knowledge() -> None:
    source = Path(external_pose_module.__file__).read_text(encoding="utf-8").lower()

    for dataset_specific in ("corridor", "cerberus", "gazebo", "-gt.txt"):
        assert dataset_specific not in source
