import dataclasses
import hashlib
from pathlib import Path

import pytest
from input_builders import (
    ACCEPT_ALL,
    INTERPOLATED,
    MS,
    SEQUENCE_ID,
    TRAJECTORY_ID,
    assemble_plan,
    make_calibration,
    make_external_pose,
    make_sequence,
    make_trajectory,
    rigid,
)
from lidar_builders import make_scan, timestamp_ns

from contextmap import geometric_mapping
from contextmap.geometric_mapping import (
    GeometryInputError,
    GeometryInputPlan,
    InputRejectionReason,
    MotionCorrectionEvidence,
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
    MotionCorrectionState,
    PointCloudLayout,
    ScanDisposition,
    UnsupportedPointCloudLayoutError,
    assemble_geometry_inputs_from_artifacts,
    resolve_point_cloud_layout,
)
from contextmap.ingestion import (
    CalibrationSet,
    ExplicitIdsSelection,
    FrameId,
    FrameRangeSelection,
    FullSequenceSelection,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SequenceSelection,
    SourceObservation,
    SourceObservationId,
    TimestampRangeSelection,
    selection_identity,
)
from contextmap.state_estimation import (
    LookupOutcome,
    LookupPolicy,
    StateEstimationRequest,
    StateEstimationRunId,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    TrajectoryId,
    calibration_identity,
    execute_state_estimation,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)

REJECT_UNKNOWN = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.REJECT)
_assemble = assemble_plan


def _ids(plan: GeometryInputPlan) -> list[str]:
    return [str(item.observation_id) for item in plan.inputs]


def _rejected(plan: GeometryInputPlan) -> dict[str, InputRejectionReason]:
    return {str(item.observation_id): item.reason for item in plan.rejections}


# --- Selection and ordering ---------------------------------------------------


def test_every_lidar_scan_becomes_an_input_in_the_selections_order() -> None:
    plan = _assemble()

    assert _ids(plan) == [f"scan-{i:04d}" for i in range(5)]
    assert plan.rejections == ()


def test_observations_that_are_not_geometric_are_counted_instead_of_silently_dropped() -> None:
    plan = _assemble([*make_sequence(), make_external_pose(0)])

    assert dict(plan.ignored_observation_counts) == {"image": 1, "imu": 1, "external_pose": 1}


def test_assembly_is_deterministic() -> None:
    assert _assemble() == _assemble()


def test_the_plan_records_the_lineage_and_the_policies_it_was_assembled_under() -> None:
    plan = _assemble(run_id=StateEstimationRunId("run-0001"), pose_lookup=INTERPOLATED)

    assert plan.sequence_artifact_id == SEQUENCE_ID
    assert plan.selection_id == selection_identity(SEQUENCE_ID, FullSequenceSelection())
    assert plan.trajectory_id == TRAJECTORY_ID
    assert plan.state_estimation_run_id == StateEstimationRunId("run-0001")
    assert (plan.map_frame, plan.body_frame) == (FrameId("map"), FrameId("body"))
    assert plan.calibration_identity == make_trajectory().provenance.calibration_identity
    assert plan.pose_lookup == INTERPOLATED
    assert plan.motion_correction_policy == ACCEPT_ALL


def test_the_plan_records_the_selection_itself_and_not_only_its_identity() -> None:
    selection = FrameRangeSelection(start_frame_index=0, end_frame_index=3)

    plan = _assemble(selection=selection)

    assert plan.selection == selection
    assert plan.selection_id == selection_identity(SEQUENCE_ID, selection)


# --- What an input states before any transformation ------------------------


def test_an_input_states_frame_timestamp_payload_pose_and_static_transform() -> None:
    scan = make_scan("scan-0002", time_ns=200 * MS)

    (item,) = _assemble([scan]).inputs

    assert item.observation == scan
    assert item.source_frame == FrameId("lidar")
    assert item.timestamp == timestamp_ns(200 * MS)
    assert item.timestamp.clock_id == "fixture:header"
    assert item.payload_hash == "sha256:" + hashlib.sha256(scan.data).hexdigest()
    assert item.layout == PointCloudLayout(
        x_offset_bytes=0,
        y_offset_bytes=4,
        z_offset_bytes=8,
        point_step_bytes=12,
        scalar=PointFieldDataType.FLOAT32,
    )
    assert item.pose.outcome is LookupOutcome.EXACT
    assert item.pose.query_observation_id == scan.observation_id
    assert item.pose.pose.translation_m == (2.0, 0.0, 0.0)
    assert item.static_transform is not None
    assert (item.static_transform.parent_frame, item.static_transform.child_frame) == (
        FrameId("body"),
        FrameId("lidar"),
    )
    assert item.static_transform.translation == (0.5, 0.0, 0.25)


def test_an_interpolated_pose_keeps_both_estimates_it_was_derived_from() -> None:
    (item,) = _assemble([make_scan("scan-0001", time_ns=150 * MS)]).inputs

    assert item.pose.outcome is LookupOutcome.INTERPOLATED
    assert len(item.pose.source_estimate_ids) == 2


def test_a_scan_already_in_the_body_frame_needs_no_static_transform() -> None:
    (item,) = _assemble(
        [make_scan("scan-0000", frame="body")],
        calibration=None,
        trajectory=make_trajectory(declare_calibration=False),
    ).inputs

    assert item.static_transform is None


def test_the_static_transform_is_composed_along_the_calibration_path() -> None:
    calibration = make_calibration(
        (rigid("body", "imu", (0.0, 0.0, 0.1)), rigid("imu", "lidar", (0.3, 0.0, 0.0)))
    )

    (item,) = _assemble(
        [make_scan()], calibration=calibration, trajectory=make_trajectory(calibration=calibration)
    ).inputs

    assert item.static_transform is not None
    assert item.static_transform.translation == pytest.approx((0.3, 0.0, 0.1))


# --- Point layout ----------------------------------------------------------


def _with_fields(
    scan: LidarObservation, fields: tuple[PointFieldDescriptor, ...]
) -> LidarObservation:
    return dataclasses.replace(scan, fields=fields)


def test_double_precision_coordinates_and_extra_fields_are_supported() -> None:
    layout = resolve_point_cloud_layout(make_scan(double_precision=True, intensity=True))

    assert layout.scalar is PointFieldDataType.FLOAT64
    assert (layout.x_offset_bytes, layout.y_offset_bytes, layout.z_offset_bytes) == (0, 8, 16)
    assert layout.point_step_bytes == 32


def test_the_layout_follows_declared_offsets_not_field_order() -> None:
    scan = make_scan()
    shuffled = _with_fields(
        scan,
        (
            PointFieldDescriptor(name="z", offset_bytes=0, data_type=PointFieldDataType.FLOAT32),
            PointFieldDescriptor(name="y", offset_bytes=4, data_type=PointFieldDataType.FLOAT32),
            PointFieldDescriptor(name="x", offset_bytes=8, data_type=PointFieldDataType.FLOAT32),
        ),
    )

    layout = resolve_point_cloud_layout(shuffled)

    assert (layout.x_offset_bytes, layout.y_offset_bytes, layout.z_offset_bytes) == (8, 4, 0)


def _field(
    name: str, offset: int, kind: PointFieldDataType, count: int = 1
) -> PointFieldDescriptor:
    return PointFieldDescriptor(name=name, offset_bytes=offset, data_type=kind, count=count)


F32 = PointFieldDataType.FLOAT32


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ((_field("x", 0, F32), _field("y", 4, F32)), "'z'"),
        (
            (
                _field("x", 0, PointFieldDataType.INT16),
                _field("y", 4, F32),
                _field("z", 8, F32),
            ),
            "floating-point",
        ),
        (
            (
                _field("x", 0, F32),
                _field("y", 4, F32),
                _field("z", 8, PointFieldDataType.FLOAT64),
            ),
            "share one",
        ),
        ((_field("x", 0, F32, 2), _field("y", 4, F32), _field("z", 8, F32)), "count"),
        ((_field("x", 0, F32), _field("y", 2, F32), _field("z", 8, F32)), "overlap"),
        ((_field("x", 0, F32), _field("y", 4, F32), _field("z", 10, F32)), "point_step_bytes"),
    ],
)
def test_an_unsupported_layout_is_refused_with_a_reason(
    fields: tuple[PointFieldDescriptor, ...], message: str
) -> None:
    with pytest.raises(UnsupportedPointCloudLayoutError, match=message):
        resolve_point_cloud_layout(_with_fields(make_scan(), fields))


def test_a_payload_that_disagrees_with_its_declared_size_is_refused() -> None:
    truncated = dataclasses.replace(make_scan(), data=b"\x00" * 5)

    with pytest.raises(UnsupportedPointCloudLayoutError, match="size"):
        resolve_point_cloud_layout(truncated)


# --- Scans that cannot be used are rejected explicitly ---------------------


def test_a_scan_without_a_payload_is_rejected() -> None:
    plan = _assemble([make_scan("scan-0000"), make_scan("scan-0001", points=())])

    assert _ids(plan) == ["scan-0000"]
    assert _rejected(plan) == {"scan-0001": InputRejectionReason.MISSING_PAYLOAD}


def test_a_scan_without_a_source_frame_is_rejected() -> None:
    scan = dataclasses.replace(make_scan(), frame_id=FrameId(""))

    plan = _assemble([scan])

    assert plan.inputs == ()
    assert _rejected(plan) == {"scan-0001": InputRejectionReason.MISSING_SOURCE_FRAME}


def test_a_scan_whose_timestamp_names_no_clock_is_rejected() -> None:
    scan = dataclasses.replace(make_scan(), timestamp=timestamp_ns(0, clock_id=""))

    plan = _assemble([scan])

    assert _rejected(plan) == {"scan-0001": InputRejectionReason.MISSING_CLOCK_IDENTITY}


def test_a_scan_with_an_unsupported_layout_is_rejected() -> None:
    scan = _with_fields(make_scan(), (_field("x", 0, F32), _field("y", 4, F32)))

    plan = _assemble([scan])

    assert _rejected(plan) == {"scan-0001": InputRejectionReason.UNSUPPORTED_LAYOUT}
    assert "'z'" in plan.rejections[0].detail


def test_a_scan_outside_the_trajectory_is_rejected_with_the_lookup_reason() -> None:
    plan = _assemble([make_scan("scan-0000"), make_scan("scan-0009", time_ns=900 * MS)])

    assert _ids(plan) == ["scan-0000"]
    assert _rejected(plan) == {"scan-0009": InputRejectionReason.POSE_LOOKUP_REJECTED}
    assert "out_of_range" in plan.rejections[0].detail


def test_a_lookup_policy_that_needs_an_exact_stamp_rejects_a_scan_between_poses() -> None:
    plan = _assemble([make_scan(time_ns=150 * MS)], pose_lookup=LookupPolicy.exact())

    assert _rejected(plan) == {"scan-0001": InputRejectionReason.POSE_LOOKUP_REJECTED}
    assert "no_exact_match" in plan.rejections[0].detail


def test_a_scan_in_another_clock_domain_is_rejected_never_compared() -> None:
    plan = _assemble([make_scan(clock_id="other:clock")])

    assert _rejected(plan) == {"scan-0001": InputRejectionReason.CLOCK_DOMAIN_MISMATCH}
    assert "other:clock" in plan.rejections[0].detail


def test_a_sensor_frame_without_a_static_path_to_the_body_is_rejected() -> None:
    plan = _assemble([make_scan(frame="unmounted")])

    assert _rejected(plan) == {"scan-0001": InputRejectionReason.NO_STATIC_TRANSFORM}


def test_without_any_calibration_only_scans_in_the_body_frame_survive() -> None:
    plan = _assemble(
        [make_scan("scan-0000"), make_scan("scan-0001", frame="body", time_ns=100 * MS)],
        calibration=None,
        trajectory=make_trajectory(declare_calibration=False),
    )

    assert _ids(plan) == ["scan-0001"]
    assert _rejected(plan) == {"scan-0000": InputRejectionReason.NO_STATIC_TRANSFORM}
    assert plan.calibration_identity is None


def test_rejected_scans_do_not_stop_the_others_and_keep_the_selections_order() -> None:
    observations: list[SourceObservation] = [
        make_scan("scan-0000"),
        make_scan("scan-0001", points=(), time_ns=100 * MS),
        make_scan("scan-0002", time_ns=200 * MS),
        make_scan("scan-0003", frame="unmounted", time_ns=300 * MS),
    ]

    plan = _assemble(observations)

    assert _ids(plan) == ["scan-0000", "scan-0002"]
    assert [str(item.observation_id) for item in plan.rejections] == ["scan-0001", "scan-0003"]


# --- Inputs that make the whole assembly meaningless raise -----------------


def test_duplicate_observation_ids_in_one_selection_are_an_error() -> None:
    scan = make_scan()

    with pytest.raises(GeometryInputError, match=r"duplicate.*scan-0001"):
        _assemble([scan, dataclasses.replace(scan, timestamp=timestamp_ns(100 * MS))])


def test_a_trajectory_from_another_sequence_is_an_error() -> None:
    other = make_trajectory(sequence_artifact_id=SequenceArtifactId("sequence-9999"))

    with pytest.raises(GeometryInputError, match="sequence-9999"):
        _assemble(trajectory=other)


def test_a_trajectory_estimated_with_another_calibration_is_an_error() -> None:
    other_calibration = make_calibration((rigid("body", "lidar", (9.0, 0.0, 0.0)),))

    with pytest.raises(GeometryInputError, match="calibration"):
        _assemble(trajectory=make_trajectory(calibration=other_calibration))


def test_a_trajectory_that_declares_no_calibration_is_compatible_with_any() -> None:
    plan = _assemble(trajectory=make_trajectory(declare_calibration=False))

    assert len(plan.inputs) == 5


def test_explicitly_selecting_an_observation_that_is_not_a_lidar_scan_is_an_error() -> None:
    selection = ExplicitIdsSelection(
        observation_ids=frozenset(
            {SourceObservationId("scan-0000"), SourceObservationId("imu-0000")}
        )
    )

    with pytest.raises(GeometryInputError, match="imu-0000"):
        _assemble(make_sequence()[:3], selection=selection)


# --- Motion correction -----------------------------------------------------


def test_every_input_carries_an_explicit_correction_state_unknown_by_default() -> None:
    plan = _assemble()

    assert {item.motion_correction.record.state for item in plan.inputs} == {
        MotionCorrectionState.UNKNOWN
    }
    assert all(
        item.motion_correction.record.observation_id == item.observation_id for item in plan.inputs
    )


def test_a_strict_policy_rejects_scans_of_unknown_correction_and_names_them() -> None:
    plan = _assemble(policy=REJECT_UNKNOWN)

    assert plan.inputs == ()
    assert set(_rejected(plan).values()) == {InputRejectionReason.MOTION_CORRECTION_REJECTED}
    assert "scan-0000" in plan.rejections[0].detail
    assert "unknown" in plan.rejections[0].detail


def test_a_warning_policy_keeps_the_scan_and_records_the_message() -> None:
    policy = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN)

    plan = _assemble(policy=policy)

    assert len(plan.inputs) == 5
    assert all(item.motion_correction.disposition is ScanDisposition.WARN for item in plan.inputs)
    assert all(item.motion_correction.message for item in plan.inputs)


def _corrected(scan: LidarObservation) -> MotionCorrectionRecord:
    return MotionCorrectionRecord(
        observation_id=scan.observation_id,
        state=MotionCorrectionState.CORRECTED,
        acquisition_start=timestamp_ns(scan.timestamp.total_nanoseconds() - 50 * MS),
        acquisition_end=timestamp_ns(scan.timestamp.total_nanoseconds() + 50 * MS),
        evidence=MotionCorrectionEvidence(
            producer="dedicated-deskew",
            trajectory_id=TrajectoryId("run-0001--trajectory"),
            payload_hash="sha256:" + "a" * 64,
        ),
    )


def test_a_declared_corrected_scan_passes_a_strict_policy_while_an_undeclared_one_does_not() -> (
    None
):
    corrected = make_scan("scan-0000", time_ns=0)
    undeclared = make_scan("scan-0001", time_ns=100 * MS)

    plan = _assemble(
        [corrected, undeclared],
        policy=REJECT_UNKNOWN,
        motion_correction={corrected.observation_id: _corrected(corrected)},
    )

    assert _ids(plan) == ["scan-0000"]
    assert plan.inputs[0].motion_correction.record.state is MotionCorrectionState.CORRECTED
    assert _rejected(plan) == {"scan-0001": InputRejectionReason.MOTION_CORRECTION_REJECTED}


def test_a_declaration_that_disagrees_with_its_scan_is_rejected() -> None:
    scan = make_scan("scan-0000", time_ns=0)
    elsewhere = dataclasses.replace(_corrected(scan), observation_id=SourceObservationId("scan-7"))

    plan = _assemble([scan], motion_correction={scan.observation_id: elsewhere})

    assert _rejected(plan) == {"scan-0000": InputRejectionReason.INCONSISTENT_MOTION_CORRECTION}


# --- Dataset independence ----------------------------------------------------

# Nomes de dataset, arquivo, tópico ou fabricante pertencem ao Ingestion e à
# proveniência da observação, nunca à lógica de geometria.
_DATASET_SPECIFIC_TOKENS = (
    "corridor",
    ".bag",
    ".pcd",
    "rosbag",
    "velodyne",
    "ouster",
    "livox",
    "hesai",
    "source_topic",
    "source_path",
)


def test_the_capability_code_never_names_a_dataset_a_file_a_topic_or_a_vendor() -> None:
    source_dir = Path(geometric_mapping.__file__).parent

    offenders = [
        f"{path.name}: {token}"
        for path in sorted(source_dir.glob("*.py"))
        for token in _DATASET_SPECIFIC_TOKENS
        if token in path.read_text(encoding="utf-8").lower()
    ]

    assert offenders == []


# --- Real artifacts --------------------------------------------------------


def _write_artifacts(workspace: Path) -> tuple[SequenceArtifactReader, StateEstimationRunReader]:
    """A sequence artifact and a State Estimation run over the same fixture sequence."""
    calibration = make_calibration()
    observations = sorted(
        [*make_sequence(), *(make_external_pose(index) for index in range(5))],
        key=lambda observation: observation.timestamp.total_nanoseconds(),
    )
    with SequenceArtifactWriter(
        workspace_root=workspace, sequence_name="corridor-02", artifact_id=SEQUENCE_ID
    ) as writer:
        writer.set_calibration(calibration)
        for observation in observations:
            writer.add_observation(observation)
        writer.finalize()

    estimator = ExternalPoseEstimator(
        ExternalPoseConfig(reference_frame=FrameId("map"), body_frame=FrameId("body"))
    )
    outcome = execute_state_estimation(
        estimator,
        _request(observations, calibration),
    )
    run_dir = workspace / "state_estimation"
    StateEstimationRunWriter(
        output_dir=run_dir,
        sequence_name="corridor-02",
        run_id=StateEstimationRunId("run-0001"),
        run_index=1,
    ).finalize(outcome)

    return (
        SequenceArtifactReader(workspace / "sequences" / "corridor-02" / str(SEQUENCE_ID)),
        StateEstimationRunReader(run_dir),
    )


def _request(
    observations: list[SourceObservation], calibration: CalibrationSet
) -> StateEstimationRequest:
    return StateEstimationRequest(
        trajectory_id=TRAJECTORY_ID,
        sequence_artifact_id=SEQUENCE_ID,
        selection_id=selection_identity(SEQUENCE_ID, FullSequenceSelection()),
        observations=observations,
        calibration=calibration,
    )


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (
            FullSequenceSelection(),
            ["scan-0000", "scan-0001", "scan-0002", "scan-0003", "scan-0004"],
        ),
        (FrameRangeSelection(start_frame_index=0, end_frame_index=5), ["scan-0000", "scan-0001"]),
        (
            TimestampRangeSelection(
                clock_id="fixture:header", start_seconds=0.15, end_seconds=0.35
            ),
            ["scan-0002", "scan-0003"],
        ),
        (
            ExplicitIdsSelection(
                observation_ids=frozenset(
                    {SourceObservationId("scan-0003"), SourceObservationId("scan-0001")}
                )
            ),
            ["scan-0001", "scan-0003"],
        ),
    ],
    ids=["full", "frame-range", "timestamp-range", "explicit-ids"],
)
def test_every_selection_kind_pairs_the_scans_with_the_run_and_calibration(
    tmp_path: Path, selection: SequenceSelection, expected: list[str]
) -> None:
    sequence, run = _write_artifacts(tmp_path)

    plan = assemble_geometry_inputs_from_artifacts(
        sequence=sequence,
        selection=selection,
        run=run,
        pose_lookup=INTERPOLATED,
        motion_correction_policy=ACCEPT_ALL,
    )

    assert _ids(plan) == expected
    assert plan.selection_id == selection_identity(SEQUENCE_ID, selection)
    assert plan.state_estimation_run_id == StateEstimationRunId("run-0001")
    assert plan.trajectory_id == run.manifest.trajectory_id
    # O backend ExternalPose não usa calibração; a do plano é a que fornece os extrínsecos.
    assert run.manifest.calibration_identity is None
    assert plan.calibration_identity == calibration_identity(make_calibration())
    for item in plan.inputs:
        assert item.static_transform is not None
        assert item.pose.pose.timestamp == item.timestamp
