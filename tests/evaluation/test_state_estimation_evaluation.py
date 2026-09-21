import json
import math
import os
from pathlib import Path

import pytest
from state_estimation_builders import (
    IDENTITY,
    MS,
    straight_line,
    trajectory_from,
    yaw,
)

from contextmap.evaluation import (
    AlignmentMethod,
    MotionThresholds,
    ReferenceComparisonConfig,
    ReferenceRole,
    ReferenceTrajectory,
    StateEstimationEvaluationError,
    StateEstimationEvaluationReport,
    compare_state_estimation_reports,
    encode_state_estimation_report,
    evaluate_state_estimation,
    trace_transform_chain,
)
from contextmap.ingestion import FrameId
from contextmap.shared import Quaternion, Vector3, quaternion_angle_between, rotate_vector
from contextmap.state_estimation import ResolvedTransform, StateEstimationRunId


def _reference(trajectory, *, role=ReferenceRole.EVALUATION_REFERENCE):  # type: ignore[no-untyped-def]
    return ReferenceTrajectory(
        trajectory=trajectory, reference_id="reference-profile:fixture@1", role=role
    )


def _config(**overrides: object) -> ReferenceComparisonConfig:
    values: dict[str, object] = {
        "alignment": AlignmentMethod.NONE,
        "max_time_difference_ns": 10 * MS,
    }
    values.update(overrides)
    return ReferenceComparisonConfig(**values)  # type: ignore[arg-type]


# --- Structural and identity -------------------------------------------------


def test_the_report_verifies_the_structure_from_the_numbers_themselves() -> None:
    report = evaluate_state_estimation(trajectory=straight_line(5))

    structure = report.structural
    assert structure.pose_count == 5
    assert (structure.reference_frame, structure.body_frame) == (FrameId("map"), FrameId("body"))
    assert structure.clock_id == "fixture:header"
    assert structure.all_values_finite
    assert structure.max_quaternion_norm_error <= 1e-12
    assert structure.min_interval_ns == 100 * MS
    assert structure.gap_count == 0
    assert structure.degraded_pose_count == 0
    assert structure.rejected_observation_count == 0


def test_the_report_preserves_every_identity_needed_to_reproduce_it() -> None:
    trajectory = straight_line(5, backend_id="external_pose", configuration="abc")

    report = evaluate_state_estimation(
        trajectory=trajectory, run_id=StateEstimationRunId("run-0007")
    )

    assert report.evaluator_version
    assert report.sequence_artifact_id == trajectory.provenance.sequence_artifact_id
    assert report.selection_id == "full-sequence"
    assert report.run_id == StateEstimationRunId("run-0007")
    assert report.trajectory_id == trajectory.trajectory_id
    assert report.estimator == trajectory.provenance.estimator
    assert report.calibration_identity == "sha256:calibration"
    assert report.accuracy is None


# --- Motion sanity -----------------------------------------------------------


def test_motion_is_measured_without_any_hardcoded_threshold() -> None:
    report = evaluate_state_estimation(trajectory=straight_line(6, step_m=1.0))

    motion = report.motion
    assert motion.thresholds is None
    assert motion.anomalies == ()
    assert motion.summary.translation_delta_m is not None
    assert motion.summary.translation_delta_m.median == pytest.approx(1.0)
    assert motion.summary.linear_speed_mps is not None
    assert motion.summary.linear_speed_mps.median == pytest.approx(10.0)
    assert motion.interval_ns is not None
    assert motion.interval_ns.median == 100 * MS


def test_jumps_are_reported_only_against_thresholds_the_profile_supplies() -> None:
    samples = [
        (0, (0.0, 0.0, 0.0), IDENTITY),
        (100 * MS, (1.0, 0.0, 0.0), IDENTITY),
        (200 * MS, (9.0, 0.0, 0.0), IDENTITY),
        (300 * MS, (10.0, 0.0, 0.0), yaw(1.0)),
    ]
    thresholds = MotionThresholds(max_translation_delta_m=2.0, max_orientation_delta_rad=0.5)

    report = evaluate_state_estimation(trajectory=trajectory_from(samples), thresholds=thresholds)

    kinds = {(a.kind, a.next_estimate_id.rsplit("-", 1)[-1]) for a in report.motion.anomalies}
    assert kinds == {("translation_delta_m", "000002"), ("orientation_delta_rad", "000003")}
    assert report.motion.thresholds == thresholds
    (translation_jump,) = [a for a in report.motion.anomalies if a.kind == "translation_delta_m"]
    assert translation_jump.value == pytest.approx(8.0)
    assert translation_jump.threshold == 2.0


def test_a_single_pose_has_no_motion_to_judge() -> None:
    report = evaluate_state_estimation(trajectory=straight_line(1))

    assert report.motion.summary.translation_delta_m is None
    assert report.motion.interval_ns is None


# --- Transform trace ---------------------------------------------------------


def test_the_transform_chain_is_reconstructable_and_numerically_consistent() -> None:
    trajectory = trajectory_from(
        [
            (0, (1.0, 2.0, 0.0), yaw(math.pi / 2)),
            (100 * MS, (4.0, 5.0, 1.0), yaw(math.pi / 4)),
        ]
    )
    body_to_lidar = ResolvedTransform(
        parent_frame=FrameId("body"),
        child_frame=FrameId("lidar"),
        translation=(0.5, 0.0, 0.25),
        rotation=yaw(math.pi / 6),
    )

    trace = trace_transform_chain(trajectory, body_to_sensor=body_to_lidar)

    first = trace.records[0]
    assert (first.reference_frame, first.sensor_frame) == (FrameId("map"), FrameId("lidar"))
    expected = rotate_vector(yaw(math.pi / 2), (0.5, 0.0, 0.25))
    assert first.sensor_translation_m == pytest.approx(
        (1.0 + expected[0], 2.0 + expected[1], expected[2])
    )
    assert quaternion_angle_between(first.sensor_rotation, yaw(math.pi / 2 + math.pi / 6)) == (
        pytest.approx(0.0, abs=1e-12)
    )
    assert first.pose_estimate_id == trajectory.poses[0].estimate_id
    assert trace.max_composition_error_m < 1e-12
    assert trace.max_round_trip_error_m < 1e-12
    assert trace.max_round_trip_error_rad < 1e-12
    assert len(trace.records) == 2


def test_the_trace_samples_at_most_the_requested_number_of_poses() -> None:
    body_to_lidar = ResolvedTransform(
        parent_frame=FrameId("body"),
        child_frame=FrameId("lidar"),
        translation=(0.1, 0.0, 0.0),
        rotation=IDENTITY,
    )

    trace = trace_transform_chain(straight_line(50), body_to_sensor=body_to_lidar, max_records=5)

    assert len(trace.records) == 5
    assert trace.records[0].pose_estimate_id.endswith("000000")
    assert trace.records[-1].pose_estimate_id.endswith("000049")


# --- Reference comparison ----------------------------------------------------


def test_only_a_reference_declared_for_evaluation_is_accepted() -> None:
    estimated = straight_line(5)
    measurement = _reference(straight_line(5), role=ReferenceRole.INPUT_MEASUREMENT)

    with pytest.raises(StateEstimationEvaluationError, match="evaluation reference"):
        evaluate_state_estimation(
            trajectory=estimated, reference=measurement, reference_config=_config()
        )


def test_a_reference_requires_an_explicit_comparison_configuration() -> None:
    with pytest.raises(StateEstimationEvaluationError, match="reference_config"):
        evaluate_state_estimation(
            trajectory=straight_line(5), reference=_reference(straight_line(5))
        )


def test_a_constant_offset_is_the_absolute_trajectory_error_without_alignment() -> None:
    estimated = straight_line(6, offset=(0.0, 0.1, 0.0))

    report = evaluate_state_estimation(
        trajectory=estimated, reference=_reference(straight_line(6)), reference_config=_config()
    )

    accuracy = report.accuracy
    assert accuracy is not None
    assert accuracy.reference_id == "reference-profile:fixture@1"
    assert accuracy.reference_role is ReferenceRole.EVALUATION_REFERENCE
    assert accuracy.alignment.method is AlignmentMethod.NONE
    assert accuracy.alignment.translation_m is None
    assert accuracy.association.matched_count == 6
    assert accuracy.ate_translation_m.rmse == pytest.approx(0.1)
    assert accuracy.ate_translation_m.maximum == pytest.approx(0.1)
    assert accuracy.rotation_error_rad.maximum == pytest.approx(0.0, abs=1e-12)


def test_rigid_alignment_removes_a_pure_frame_offset_and_states_what_it_applied() -> None:
    reference = trajectory_from(
        [(i * 100 * MS, (float(i), math.sin(i), 0.1 * i), yaw(0.1 * i)) for i in range(8)]
    )
    rotation = yaw(0.7)
    offset = (5.0, -3.0, 1.0)

    def moved(index: int, position: Vector3) -> tuple[int, Vector3, Quaternion]:
        x, y, z = rotate_vector(rotation, position)
        shifted = (x + offset[0], y + offset[1], z + offset[2])
        return (index * 100 * MS, shifted, yaw(0.1 * index + 0.7))

    estimated = trajectory_from(
        [moved(i, pose.translation_m) for i, pose in enumerate(reference.poses)]
    )

    report = evaluate_state_estimation(
        trajectory=estimated,
        reference=_reference(reference),
        reference_config=_config(alignment=AlignmentMethod.SE3),
    )

    accuracy = report.accuracy
    assert accuracy is not None
    assert accuracy.alignment.method is AlignmentMethod.SE3
    assert accuracy.ate_translation_m.rmse < 1e-9
    assert accuracy.rotation_error_rad.maximum < 1e-9
    assert accuracy.alignment.translation_m is not None
    assert accuracy.alignment.rotation is not None
    # A transformação aplicada leva o frame estimado ao frame de referência (inversa do offset).
    assert quaternion_angle_between(accuracy.alignment.rotation, yaw(-0.7)) < 1e-9


def test_relative_pose_error_exposes_drift_that_alignment_hides() -> None:
    reference = straight_line(6, step_m=1.0)
    drifting = straight_line(6, step_m=1.1)

    report = evaluate_state_estimation(
        trajectory=drifting,
        reference=_reference(reference),
        reference_config=_config(
            alignment=AlignmentMethod.SE3,
            relative_interval_ns=100 * MS,
            relative_interval_tolerance_ns=1 * MS,
        ),
    )

    accuracy = report.accuracy
    assert accuracy is not None
    assert accuracy.rpe is not None
    assert accuracy.rpe.interval_ns == 100 * MS
    assert accuracy.rpe.translation_m.rmse == pytest.approx(0.1)
    assert accuracy.rpe.rotation_rad.maximum == pytest.approx(0.0, abs=1e-12)
    assert accuracy.rpe.pair_count == 5


def test_relative_pose_error_is_anchored_on_time_so_missing_reference_poses_do_not_stretch_it() -> (
    None
):
    # A referência perde toda terceira pose (como o GT real perde ~25% delas): pareando por
    # índice, alguns pares cobririam 300 ms em vez de 200 ms e o erro relativo cresceria.
    estimated = straight_line(13, step_m=1.1)
    reference_poses = straight_line(13, step_m=1.0)
    kept = [pose for index, pose in enumerate(reference_poses.poses) if index % 3 != 2]
    reference = trajectory_from(
        [
            (pose.timestamp.total_nanoseconds(), pose.translation_m, pose.orientation)
            for pose in kept
        ]
    )

    report = evaluate_state_estimation(
        trajectory=estimated,
        reference=_reference(reference),
        reference_config=_config(
            alignment=AlignmentMethod.SE3,
            relative_interval_ns=200 * MS,
            relative_interval_tolerance_ns=5 * MS,
        ),
    )

    accuracy = report.accuracy
    assert accuracy is not None and accuracy.rpe is not None
    assert accuracy.rpe.interval_ns == 200 * MS
    # 0,1 m de deriva por 100 ms: exatamente 0,2 m por 200 ms, sem par de 300 ms.
    assert accuracy.rpe.translation_m.median == pytest.approx(0.2)
    assert accuracy.rpe.translation_m.maximum == pytest.approx(0.2)
    assert accuracy.rpe.pair_count == accuracy.rpe.translation_m.count > 0


def test_no_pair_at_the_interval_means_no_relative_error_instead_of_a_wrong_one() -> None:
    report = evaluate_state_estimation(
        trajectory=straight_line(6),
        reference=_reference(straight_line(6)),
        reference_config=_config(
            relative_interval_ns=10_000 * MS, relative_interval_tolerance_ns=5 * MS
        ),
    )

    assert report.accuracy is not None and report.accuracy.rpe is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"relative_interval_ns": 100 * MS},
        {"relative_interval_tolerance_ns": 1 * MS},
        {"relative_interval_ns": 0, "relative_interval_tolerance_ns": 1 * MS},
        {"relative_interval_ns": 100 * MS, "relative_interval_tolerance_ns": -1},
    ],
)
def test_the_relative_error_protocol_must_state_interval_and_tolerance_together(
    overrides: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="relative_interval"):
        _config(**overrides)


def test_the_report_traces_the_reference_to_the_file_it_came_from() -> None:
    reference = ReferenceTrajectory(
        trajectory=straight_line(6),
        reference_id="reference-profile:fixture@1",
        role=ReferenceRole.EVALUATION_REFERENCE,
        source="datasets/fixture/poses.txt",
        source_sha256="sha256:abc",
    )

    report = evaluate_state_estimation(
        trajectory=straight_line(6), reference=reference, reference_config=_config()
    )

    record = json.loads(json.dumps(encode_state_estimation_report(report)))
    assert record["reference"]["source"] == "datasets/fixture/poses.txt"
    assert record["reference"]["source_sha256"] == "sha256:abc"


def test_a_comparison_rejects_reports_measured_against_different_reference_files() -> None:
    def against(digest: str) -> StateEstimationEvaluationReport:
        reference = ReferenceTrajectory(
            trajectory=straight_line(6),
            reference_id="reference-profile:fixture@1",
            role=ReferenceRole.EVALUATION_REFERENCE,
            source_sha256=digest,
        )
        return evaluate_state_estimation(
            trajectory=straight_line(6), reference=reference, reference_config=_config()
        )

    with pytest.raises(StateEstimationEvaluationError, match="reference"):
        compare_state_estimation_reports([against("sha256:a"), against("sha256:b")])


def test_a_comparison_with_no_associable_pose_is_an_error_not_an_empty_result() -> None:
    estimated = straight_line(4, time_offset_ns=50 * MS)

    with pytest.raises(StateEstimationEvaluationError, match="no pose could be associated"):
        evaluate_state_estimation(
            trajectory=estimated,
            reference=_reference(straight_line(4)),
            reference_config=_config(max_time_difference_ns=20 * MS),
        )


def test_association_reports_matched_and_unmatched_poses() -> None:
    estimated = straight_line(6)
    reference = straight_line(4)  # covers only the first four estimated poses

    report = evaluate_state_estimation(
        trajectory=estimated,
        reference=_reference(reference),
        reference_config=_config(max_time_difference_ns=10 * MS),
    )

    accuracy = report.accuracy
    assert accuracy is not None
    assert (accuracy.association.matched_count, accuracy.association.unmatched_count) == (4, 2)
    assert accuracy.association.max_time_difference_ns == 0


def test_a_reference_in_another_clock_domain_is_never_compared_implicitly() -> None:
    reference = straight_line(4, clock_id="another:clock")

    with pytest.raises(StateEstimationEvaluationError, match="clock"):
        evaluate_state_estimation(
            trajectory=straight_line(4),
            reference=_reference(reference),
            reference_config=_config(),
        )


def test_alignment_needs_enough_pairs_to_be_defined() -> None:
    with pytest.raises(StateEstimationEvaluationError, match="at least three"):
        evaluate_state_estimation(
            trajectory=straight_line(2),
            reference=_reference(straight_line(2)),
            reference_config=_config(alignment=AlignmentMethod.SE3),
        )


# --- Quality and cost stay separate -----------------------------------------


def test_runtime_is_reported_apart_from_the_quality_sections() -> None:
    report = evaluate_state_estimation(trajectory=straight_line(5), runtime_s=2.5)

    assert report.cost.runtime_s == 2.5
    assert report.cost.pose_count == 5
    record = encode_state_estimation_report(report)
    assert "runtime_s" not in json.dumps(
        {key: value for key, value in record.items() if key != "cost"}
    )


# --- Common report schema and comparison ------------------------------------


def test_the_report_encodes_to_plain_json_with_every_identity() -> None:
    report = evaluate_state_estimation(
        trajectory=straight_line(6, backend_id="fast_lio", configuration="xyz"),
        reference=_reference(straight_line(6)),
        reference_config=_config(),
        run_id=StateEstimationRunId("run-0002"),
    )

    record = json.loads(json.dumps(encode_state_estimation_report(report)))

    assert record["evaluator_version"] == report.evaluator_version
    assert record["estimator"]["backend_id"] == "fast_lio"
    assert record["estimator"]["configuration_fingerprint"] == "sha256:xyz"
    assert record["run_id"] == "run-0002"
    assert record["calibration_identity"] == "sha256:calibration"
    assert record["reference"]["reference_id"] == "reference-profile:fixture@1"
    assert record["reference"]["role"] == "evaluation_reference"
    assert record["reference"]["alignment"]["method"] == "none"
    assert set(record) >= {"structural", "motion", "accuracy", "cost", "estimator"}


def test_backends_are_compared_under_one_schema_preserving_their_identities() -> None:
    reference = _reference(straight_line(6))
    common = {"reference": reference, "reference_config": _config()}
    external = evaluate_state_estimation(
        trajectory=straight_line(6, backend_id="external_pose", configuration="a"), **common
    )
    lidar_inertial = evaluate_state_estimation(
        trajectory=straight_line(6, step_m=1.05, backend_id="fast_lio", configuration="b"),
        **common,
    )

    comparison = compare_state_estimation_reports([external, lidar_inertial])

    assert [entry.estimator.backend_id for entry in comparison.entries] == [
        "external_pose",
        "fast_lio",
    ]
    assert comparison.entries[0].ate_rmse_m == pytest.approx(0.0, abs=1e-12)
    assert comparison.entries[1].ate_rmse_m is not None and comparison.entries[1].ate_rmse_m > 0
    assert comparison.sequence_artifact_id == external.sequence_artifact_id
    assert comparison.reference_id == "reference-profile:fixture@1"
    assert [entry.configuration_fingerprint for entry in comparison.entries] == [
        "sha256:a",
        "sha256:b",
    ]


def test_a_backend_that_consumes_no_calibration_is_compared_with_one_that_does() -> None:
    # ExternalPose não consome calibração (identidade None) e o FAST-LIO consome: a comparação
    # entre eles é o caso de uso, então só duas identidades *diferentes* a impedem.
    common = {"reference": _reference(straight_line(6)), "reference_config": _config()}
    external = evaluate_state_estimation(
        trajectory=straight_line(
            6, backend_id="external_pose", configuration="a", calibration_identity=None
        ),
        **common,
    )
    lidar_inertial = evaluate_state_estimation(
        trajectory=straight_line(
            6, backend_id="fast_lio", configuration="b", calibration_identity="sha256:cal"
        ),
        **common,
    )

    comparison = compare_state_estimation_reports([external, lidar_inertial])

    assert comparison.calibration_identity == "sha256:cal"
    assert [entry.calibration_identity for entry in comparison.entries] == [None, "sha256:cal"]


def test_a_comparison_rejects_reports_that_changed_more_than_the_backend() -> None:
    reference = _reference(straight_line(6))
    common = {"reference": reference, "reference_config": _config()}
    baseline = evaluate_state_estimation(trajectory=straight_line(6), **common)
    other_selection = evaluate_state_estimation(
        trajectory=straight_line(6, selection_id="frames-0-10"), **common
    )
    other_calibration = evaluate_state_estimation(
        trajectory=straight_line(6, calibration_identity="sha256:other"), **common
    )
    no_reference = evaluate_state_estimation(trajectory=straight_line(6))

    with pytest.raises(StateEstimationEvaluationError, match="selection"):
        compare_state_estimation_reports([baseline, other_selection])
    with pytest.raises(StateEstimationEvaluationError, match="calibration"):
        compare_state_estimation_reports([baseline, other_calibration])
    with pytest.raises(StateEstimationEvaluationError, match="reference"):
        compare_state_estimation_reports([baseline, no_reference])
    with pytest.raises(StateEstimationEvaluationError, match="at least two"):
        compare_state_estimation_reports([baseline])


# --- Real data (optional) ----------------------------------------------------

_DATASET = Path(
    os.environ.get(
        "CONTEXTMAP_CORRIDOR02_DIR",
        Path(__file__).resolve().parents[2] / "datasets" / "corridor-02",
    )
)


@pytest.mark.skipif(
    not (_DATASET / "corridor-02-gt.txt").is_file(),
    reason="corridor-02 dataset is not available (it is not versioned)",
)
def test_the_baseline_report_over_the_real_external_pose_sequence() -> None:
    from decimal import Decimal

    samples = []
    for line in (_DATASET / "corridor-02-gt.txt").read_text().splitlines():
        t, x, y, z, qx, qy, qz, qw = line.split()
        norm = math.sqrt(sum(float(v) ** 2 for v in (qx, qy, qz, qw)))
        samples.append(
            (
                int(Decimal(t) * 1_000_000_000),
                (float(x), float(y), float(z)),
                tuple(float(v) / norm for v in (qx, qy, qz, qw)),
            )
        )

    report = evaluate_state_estimation(
        trajectory=trajectory_from(samples),  # type: ignore[arg-type]
        thresholds=MotionThresholds(max_translation_delta_m=5.0),
    )

    assert report.structural.pose_count == 5522
    assert report.structural.all_values_finite
    assert report.motion.interval_ns is not None
    assert report.motion.interval_ns.maximum > 500 * MS
