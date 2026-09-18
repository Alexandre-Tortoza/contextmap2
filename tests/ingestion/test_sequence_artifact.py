import json
from pathlib import Path

import pytest

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    IncompleteSequenceArtifactError,
    LidarObservation,
    PinholeCameraModel,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SequenceArtifactError,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SequenceProvenance,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.shared import SourceTimestamp


def _timestamp(seconds: int = 1) -> SourceTimestamp:
    return SourceTimestamp(seconds=seconds, nanoseconds=0, clock_id="system")


def _provenance(**overrides: object) -> SourceProvenance:
    defaults: dict[str, object] = {"source_type": "dataset", "source_path": "fixtures/example"}
    defaults.update(overrides)
    return SourceProvenance(**defaults)  # type: ignore[arg-type]


def _build_fixture_sequence(writer: SequenceArtifactWriter) -> None:
    """Deterministic multimodal fixture: 2 images, 1 lidar scan, 1 IMU, 1 pose."""
    writer.add_observation(
        ImageObservation(
            observation_id=SourceObservationId("frame-0001"),
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            timestamp=_timestamp(1),
            provenance=_provenance(source_topic="/camera/image_raw"),
            width=2,
            height=1,
            encoding=ImageEncoding.RGB8,
            data=b"\x01\x02\x03\x04\x05\x06",
        )
    )
    writer.add_observation(
        ImageObservation(
            observation_id=SourceObservationId("frame-0002"),
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            timestamp=_timestamp(2),
            provenance=_provenance(source_topic="/camera/image_raw"),
            width=2,
            height=1,
            encoding=ImageEncoding.RGB8,
            data=b"\x07\x08\x09\x0a\x0b\x0c",
        )
    )
    writer.add_observation(
        LidarObservation(
            observation_id=SourceObservationId("scan-0001"),
            sensor_id=SensorId("velodyne_top"),
            frame_id=FrameId("velodyne"),
            timestamp=_timestamp(1),
            provenance=_provenance(source_topic="/velodyne_points"),
            point_count=1,
            point_step_bytes=12,
            fields=(
                PointFieldDescriptor(
                    name="x", offset_bytes=0, data_type=PointFieldDataType.FLOAT32
                ),
                PointFieldDescriptor(
                    name="y", offset_bytes=4, data_type=PointFieldDataType.FLOAT32
                ),
                PointFieldDescriptor(
                    name="z", offset_bytes=8, data_type=PointFieldDataType.FLOAT32
                ),
            ),
            data=b"\x00" * 12,
        )
    )
    writer.add_observation(
        ImuObservation(
            observation_id=SourceObservationId("imu-0001"),
            sensor_id=SensorId("imu0"),
            frame_id=FrameId("imu_link"),
            timestamp=_timestamp(1),
            provenance=_provenance(source_topic="/imu/data"),
            linear_acceleration=(0.0, 0.0, 9.81),
            angular_velocity=(0.0, 0.0, 0.0),
            orientation=None,
        )
    )
    writer.add_observation(
        ExternalPoseMeasurement(
            observation_id=SourceObservationId("odom-0001"),
            sensor_id=SensorId("wheel_odometry"),
            frame_id=FrameId("base_link"),
            timestamp=_timestamp(1),
            provenance=_provenance(source_topic="/odom"),
            parent_frame=FrameId("odom"),
            translation=(1.0, 2.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
    )


def test_finalized_artifact_can_be_reopened_without_the_original_source(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)

    manifest = writer.finalize()

    assert manifest.observation_counts == {"image": 2, "lidar": 1, "imu": 1, "external_pose": 1}

    artifact_dir = tmp_path / "sequences" / "corridor-02" / manifest.artifact_id
    assert artifact_dir.is_dir()
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "index.jsonl").is_file()
    assert (artifact_dir / "rgb" / "frame-0001.bin").is_file()
    assert not any(tmp_path.glob("sequences/corridor-02/.tmp-*"))

    reader = SequenceArtifactReader(artifact_dir)
    observations = reader.list_observations()

    assert [obs.observation_id for obs in observations] == [
        "frame-0001",
        "frame-0002",
        "scan-0001",
        "imu-0001",
        "odom-0001",
    ]
    assert reader.verify_integrity() == []


def test_round_trip_preserves_observation_fields(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()

    reader = SequenceArtifactReader(tmp_path / "sequences" / "corridor-02" / manifest.artifact_id)

    image = reader.get_observation(SourceObservationId("frame-0001"))
    assert isinstance(image, ImageObservation)
    assert image.data == b"\x01\x02\x03\x04\x05\x06"
    assert image.width == 2
    assert image.encoding is ImageEncoding.RGB8

    lidar = reader.get_observation(SourceObservationId("scan-0001"))
    assert isinstance(lidar, LidarObservation)
    assert lidar.fields[0].name == "x"
    assert lidar.data == b"\x00" * 12

    imu = reader.get_observation(SourceObservationId("imu-0001"))
    assert isinstance(imu, ImuObservation)
    assert imu.orientation is None
    assert imu.linear_acceleration == (0.0, 0.0, 9.81)

    pose = reader.get_observation(SourceObservationId("odom-0001"))
    assert isinstance(pose, ExternalPoseMeasurement)
    assert pose.parent_frame == "odom"
    assert pose.translation == (1.0, 2.0, 0.0)


def test_get_observation_raises_for_unknown_id(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()
    reader = SequenceArtifactReader(tmp_path / "sequences" / "corridor-02" / manifest.artifact_id)

    with pytest.raises(SequenceArtifactError, match="not found"):
        reader.get_observation(SourceObservationId("does-not-exist"))


def test_duplicate_observation_id_is_rejected(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)

    with pytest.raises(SequenceArtifactError, match="duplicate"):
        _build_fixture_sequence(writer)


def test_cannot_add_observation_after_finalize(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    writer.finalize()

    with pytest.raises(SequenceArtifactError, match="finalize"):
        writer.add_observation(
            ImuObservation(
                observation_id=SourceObservationId("imu-0002"),
                sensor_id=SensorId("imu0"),
                frame_id=FrameId("imu_link"),
                timestamp=_timestamp(3),
                provenance=_provenance(),
            )
        )


def test_finalize_refuses_to_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    from contextmap.ingestion.sequence_artifact import SequenceArtifactId

    artifact_id = SequenceArtifactId("fixed-id")

    first_writer = SequenceArtifactWriter(
        workspace_root=tmp_path, sequence_name="corridor-02", artifact_id=artifact_id
    )
    _build_fixture_sequence(first_writer)
    first_writer.finalize()

    second_writer = SequenceArtifactWriter(
        workspace_root=tmp_path, sequence_name="corridor-02", artifact_id=artifact_id
    )
    _build_fixture_sequence(second_writer)

    with pytest.raises(SequenceArtifactError, match="already exists"):
        second_writer.finalize()

    # Interrupted/rejected writes must not leave a directory at the final path.
    artifact_dir = tmp_path / "sequences" / "corridor-02" / artifact_id
    assert artifact_dir.is_dir()  # the first, valid artifact remains untouched
    reader = SequenceArtifactReader(artifact_dir)
    assert reader.verify_integrity() == []


def test_opening_a_directory_without_a_manifest_fails(tmp_path: Path) -> None:
    empty_dir = tmp_path / "not-an-artifact"
    empty_dir.mkdir()

    with pytest.raises(IncompleteSequenceArtifactError):
        SequenceArtifactReader(empty_dir)


def test_verify_integrity_detects_a_missing_payload_file(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()
    artifact_dir = tmp_path / "sequences" / "corridor-02" / manifest.artifact_id

    (artifact_dir / "rgb" / "frame-0001.bin").unlink()

    reader = SequenceArtifactReader(artifact_dir)
    problems = reader.verify_integrity()

    assert any("missing file" in problem for problem in problems)


def test_verify_integrity_detects_a_content_hash_mismatch(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()
    artifact_dir = tmp_path / "sequences" / "corridor-02" / manifest.artifact_id

    (artifact_dir / "rgb" / "frame-0001.bin").write_bytes(b"\xff\xff\xff\xff\xff\xff")

    reader = SequenceArtifactReader(artifact_dir)
    problems = reader.verify_integrity()

    assert any("hash mismatch" in problem for problem in problems)


def _calibration_set() -> CalibrationSet:
    model = PinholeCameraModel(width=2, height=1, fx=1.0, fy=1.0, cx=1.0, cy=0.5)
    entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId("front_camera-calib"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        camera_model=model,
        provenance=CalibrationProvenance(source_type="dataset", source_path="fixtures/calib.yaml"),
        content_hash=compute_content_hash(
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            camera_model=model,
        ),
    )
    return CalibrationSet(entries={entry.calibration_id: entry}, static_transforms=())


def test_calibration_round_trips_through_the_artifact(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    writer.set_calibration(_calibration_set())
    manifest = writer.finalize()

    artifact_dir = tmp_path / "sequences" / "corridor-02" / manifest.artifact_id
    assert (artifact_dir / "calibration" / "calibration.json").is_file()

    reader = SequenceArtifactReader(artifact_dir)
    calibration = reader.read_calibration()

    assert calibration is not None
    assert calibration.entries.keys() == {CalibrationReferenceId("front_camera-calib")}
    assert reader.verify_integrity() == []


def test_read_calibration_returns_none_when_never_set(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()

    reader = SequenceArtifactReader(tmp_path / "sequences" / "corridor-02" / manifest.artifact_id)

    assert reader.read_calibration() is None


def _sequence_provenance() -> SequenceProvenance:
    return SequenceProvenance(
        source_type="dataset",
        source_path="fixtures/example",
        source_content_hash="sha256:aaaa",
        configuration_hash="sha256:bbbb",
        adapter_type="dataset",
    )


def test_provenance_round_trips_through_the_artifact(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    writer.set_provenance(_sequence_provenance())
    manifest = writer.finalize()

    artifact_dir = tmp_path / "sequences" / "corridor-02" / manifest.artifact_id
    assert (artifact_dir / "provenance" / "provenance.json").is_file()

    reader = SequenceArtifactReader(artifact_dir)
    provenance = reader.read_provenance()

    assert provenance is not None
    assert provenance.source_type == "dataset"
    assert provenance.configuration_hash == "sha256:bbbb"
    assert reader.verify_integrity() == []


def test_read_provenance_returns_none_when_never_set(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()

    reader = SequenceArtifactReader(tmp_path / "sequences" / "corridor-02" / manifest.artifact_id)

    assert reader.read_provenance() is None


def test_verify_integrity_detects_an_index_payload_cross_reference_error(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()
    artifact_dir = tmp_path / "sequences" / "corridor-02" / manifest.artifact_id

    index_path = artifact_dir / "index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    first_record = json.loads(lines[0])
    first_record["payload_path"] = "rgb/does-not-exist.bin"
    lines[0] = json.dumps(first_record)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reader = SequenceArtifactReader(artifact_dir)
    problems = reader.verify_integrity()

    assert any(
        "payload_path" in problem and "not present in manifest" in problem for problem in problems
    )


def test_diagnostics_round_trip_through_the_artifact(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    writer.set_diagnostics(warnings=["imu topic not available"])
    manifest = writer.finalize()

    artifact_dir = tmp_path / "sequences" / "corridor-02" / manifest.artifact_id
    assert (artifact_dir / "diagnostics" / "summary.json").is_file()
    assert (artifact_dir / "diagnostics" / "warnings.jsonl").is_file()

    reader = SequenceArtifactReader(artifact_dir)
    diagnostics = reader.read_diagnostics()

    assert diagnostics is not None
    assert diagnostics.warnings == ("imu topic not available",)
    assert diagnostics.summary.modality_summaries["image"].count == 2
    assert diagnostics.summary.warning_count == 1
    assert reader.verify_integrity() == []


def test_read_diagnostics_returns_none_when_never_set(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    manifest = writer.finalize()

    reader = SequenceArtifactReader(tmp_path / "sequences" / "corridor-02" / manifest.artifact_id)

    assert reader.read_diagnostics() is None


def test_set_diagnostics_with_no_warnings_still_writes_a_summary(tmp_path: Path) -> None:
    writer = SequenceArtifactWriter(workspace_root=tmp_path, sequence_name="corridor-02")
    _build_fixture_sequence(writer)
    writer.set_diagnostics()
    manifest = writer.finalize()

    reader = SequenceArtifactReader(tmp_path / "sequences" / "corridor-02" / manifest.artifact_id)
    diagnostics = reader.read_diagnostics()

    assert diagnostics is not None
    assert diagnostics.warnings == ()
    assert diagnostics.summary.warning_count == 0
