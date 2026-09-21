import errno
import json
import shutil
from pathlib import Path
from typing import Any

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
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SequenceProvenance,
    SourceObservationId,
    SourceProvenance,
    SynchronizationConfig,
    synchronize,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.shared import SourceTimestamp

_ARTIFACT_ID = "sequence-0001"


def _output_dir(tmp_path: Path) -> Path:
    """Diretório final escolhido pelo chamador; o leitor abre exatamente este caminho."""
    return tmp_path / "ingestion"


def _writer(
    tmp_path: Path, *, sequence_name: str = "corridor-02", artifact_id: str = _ARTIFACT_ID
) -> SequenceArtifactWriter:
    return SequenceArtifactWriter(
        output_dir=_output_dir(tmp_path),
        sequence_name=sequence_name,
        artifact_id=SequenceArtifactId(artifact_id),
    )


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
            linear_acceleration_covariance=tuple(float(value) for value in range(9)),
            angular_velocity_covariance=tuple(float(value) for value in range(10, 19)),
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
            pose_covariance=tuple(float(value) for value in range(36)),
            linear_velocity=(0.1, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.01),
            twist_covariance=tuple(float(value) for value in range(100, 136)),
        )
    )


def test_finalized_artifact_can_be_reopened_without_the_original_source(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)

    manifest = writer.finalize()

    assert manifest.schema_version == "0.2.0"
    assert manifest.observation_counts == {"image": 2, "lidar": 1, "imu": 1, "external_pose": 1}

    artifact_dir = _output_dir(tmp_path)
    assert artifact_dir.is_dir()
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "index.jsonl").is_file()
    assert (artifact_dir / "rgb" / "frame-0001.bin").is_file()

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
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()

    reader = SequenceArtifactReader(_output_dir(tmp_path))

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
    assert imu.linear_acceleration_covariance == tuple(float(value) for value in range(9))
    assert imu.angular_velocity_covariance == tuple(float(value) for value in range(10, 19))

    pose = reader.get_observation(SourceObservationId("odom-0001"))
    assert isinstance(pose, ExternalPoseMeasurement)
    assert pose.parent_frame == "odom"
    assert pose.translation == (1.0, 2.0, 0.0)
    assert pose.pose_covariance == tuple(float(value) for value in range(36))
    assert pose.linear_velocity == (0.1, 0.0, 0.0)
    assert pose.angular_velocity == (0.0, 0.0, 0.01)
    assert pose.twist_covariance == tuple(float(value) for value in range(100, 136))


def test_get_observation_raises_for_unknown_id(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()
    reader = SequenceArtifactReader(_output_dir(tmp_path))

    with pytest.raises(SequenceArtifactError, match="not found"):
        reader.get_observation(SourceObservationId("does-not-exist"))


def test_duplicate_observation_id_is_rejected(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)

    with pytest.raises(SequenceArtifactError, match="duplicate"):
        _build_fixture_sequence(writer)


def test_cannot_add_observation_after_finalize(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
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


def test_the_artifact_appears_exactly_at_output_dir_and_nothing_else_is_created(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)

    writer.finalize()

    assert list(tmp_path.iterdir()) == [_output_dir(tmp_path)]
    assert not (tmp_path / "sequences").exists()
    assert not (tmp_path / "runs.json").exists()


def test_the_artifact_id_is_recorded_as_supplied_and_never_allocated(tmp_path: Path) -> None:
    writer = _writer(tmp_path, artifact_id="chosen-by-the-caller")
    _build_fixture_sequence(writer)

    manifest = writer.finalize()

    assert manifest.artifact_id == "chosen-by-the-caller"
    assert (
        SequenceArtifactReader(_output_dir(tmp_path)).manifest.artifact_id == manifest.artifact_id
    )
    with pytest.raises(TypeError, match="artifact_id"):
        SequenceArtifactWriter(  # type: ignore[call-arg]
            output_dir=tmp_path / "without-identity", sequence_name="corridor-02"
        )


def test_finalize_refuses_a_second_run_at_the_same_directory_without_altering_the_first(
    tmp_path: Path,
) -> None:
    first_writer = _writer(tmp_path)
    _build_fixture_sequence(first_writer)
    first_writer.finalize()
    artifact_dir = _output_dir(tmp_path)
    before = {
        path.relative_to(artifact_dir): path.read_bytes()
        for path in artifact_dir.rglob("*")
        if path.is_file()
    }

    second_writer = _writer(tmp_path, artifact_id="another-run")
    _build_fixture_sequence(second_writer)

    with pytest.raises(SequenceArtifactError, match="already exists"):
        second_writer.finalize()

    after = {
        path.relative_to(artifact_dir): path.read_bytes()
        for path in artifact_dir.rglob("*")
        if path.is_file()
    }
    assert after == before  # the first, valid artifact remains untouched
    assert list(tmp_path.iterdir()) == [artifact_dir]  # and the refused run left nothing behind
    assert SequenceArtifactReader(artifact_dir).verify_integrity() == []


def test_opening_a_directory_without_a_manifest_fails(tmp_path: Path) -> None:
    empty_dir = tmp_path / "not-an-artifact"
    empty_dir.mkdir()

    with pytest.raises(IncompleteSequenceArtifactError):
        SequenceArtifactReader(empty_dir)


def test_verify_integrity_detects_a_missing_payload_file(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()
    artifact_dir = _output_dir(tmp_path)

    (artifact_dir / "rgb" / "frame-0001.bin").unlink()

    reader = SequenceArtifactReader(artifact_dir)
    problems = reader.verify_integrity()

    assert any("missing file" in problem for problem in problems)


def test_verify_integrity_detects_a_content_hash_mismatch(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()
    artifact_dir = _output_dir(tmp_path)

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
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.set_calibration(_calibration_set())
    writer.finalize()

    artifact_dir = _output_dir(tmp_path)
    assert (artifact_dir / "calibration" / "calibration.json").is_file()

    reader = SequenceArtifactReader(artifact_dir)
    calibration = reader.read_calibration()

    assert calibration is not None
    assert calibration.entries.keys() == {CalibrationReferenceId("front_camera-calib")}
    assert reader.verify_integrity() == []


def test_read_calibration_returns_none_when_never_set(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()

    reader = SequenceArtifactReader(_output_dir(tmp_path))

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
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.set_provenance(_sequence_provenance())
    writer.finalize()

    artifact_dir = _output_dir(tmp_path)
    assert (artifact_dir / "provenance" / "provenance.json").is_file()

    reader = SequenceArtifactReader(artifact_dir)
    provenance = reader.read_provenance()

    assert provenance is not None
    assert provenance.source_type == "dataset"
    assert provenance.configuration_hash == "sha256:bbbb"
    assert reader.verify_integrity() == []


def test_read_provenance_returns_none_when_never_set(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()

    reader = SequenceArtifactReader(_output_dir(tmp_path))

    assert reader.read_provenance() is None


def test_verify_integrity_detects_an_index_payload_cross_reference_error(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()
    artifact_dir = _output_dir(tmp_path)

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
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.set_diagnostics(warnings=["imu topic not available"])
    writer.finalize()

    artifact_dir = _output_dir(tmp_path)
    assert (artifact_dir / "diagnostics" / "summary.json").is_file()
    assert (artifact_dir / "diagnostics" / "warnings.jsonl").is_file()

    reader = SequenceArtifactReader(artifact_dir)
    diagnostics = reader.read_diagnostics()

    assert diagnostics is not None
    assert diagnostics.warnings == ("imu topic not available",)
    assert diagnostics.summary.modality_summaries["image"].count == 2
    assert diagnostics.summary.warning_count == 1
    assert reader.verify_integrity() == []


def test_structured_synchronization_and_frame_graph_diagnostics_round_trip(
    tmp_path: Path,
) -> None:
    image = ImageObservation(
        observation_id=SourceObservationId("frame-sync"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=_timestamp(1),
        provenance=_provenance(source_topic="/camera/image_raw"),
        width=1,
        height=1,
        encoding=ImageEncoding.MONO8,
        data=b"\x01",
    )
    imu = ImuObservation(
        observation_id=SourceObservationId("imu-sync"),
        sensor_id=SensorId("imu0"),
        frame_id=FrameId("imu_link"),
        timestamp=_timestamp(1),
        provenance=_provenance(source_topic="/imu/data"),
    )
    _, synchronization = synchronize(
        [image, imu],
        config=SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=0),
    )
    calibration = _calibration_set()
    writer = _writer(tmp_path, sequence_name="structured")
    writer.add_observation(image)
    writer.add_observation(imu)
    writer.set_calibration(calibration)
    writer.set_diagnostics(synchronization=synchronization)

    writer.finalize()
    artifact_dir = _output_dir(tmp_path)

    assert (artifact_dir / "diagnostics" / "synchronization.jsonl").is_file()
    assert (artifact_dir / "diagnostics" / "dropped-events.jsonl").is_file()
    assert (artifact_dir / "diagnostics" / "frame-graph.json").is_file()

    diagnostics = SequenceArtifactReader(artifact_dir).read_diagnostics()

    assert diagnostics is not None
    assert diagnostics.synchronization == synchronization
    assert diagnostics.frame_graph is not None
    assert diagnostics.frame_graph.calibration_ids == ("front_camera-calib",)
    assert diagnostics.summary.synchronization_status_counts["matched"] == 1
    assert diagnostics.summary.calibration_ids == ["front_camera-calib"]


def test_read_diagnostics_returns_none_when_never_set(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.finalize()

    reader = SequenceArtifactReader(_output_dir(tmp_path))

    assert reader.read_diagnostics() is None


def test_set_diagnostics_with_no_warnings_still_writes_a_summary(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.set_diagnostics()
    writer.finalize()

    reader = SequenceArtifactReader(_output_dir(tmp_path))
    diagnostics = reader.read_diagnostics()

    assert diagnostics is not None
    assert diagnostics.warnings == ()
    assert diagnostics.summary.warning_count == 0


def test_read_summary_diagnostics_does_not_decode_observation_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.set_diagnostics()
    writer.finalize()
    reader = SequenceArtifactReader(_output_dir(tmp_path))

    def fail_if_called() -> None:
        raise AssertionError("summary-only diagnostics must not decode observation payloads")

    monkeypatch.setattr(reader, "list_observations", fail_if_called)

    assert reader.read_diagnostics() is not None


def _image(index: int, data: bytes) -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId(f"frame-{index:04d}"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=_timestamp(index),
        provenance=_provenance(source_topic="/camera/image_raw"),
        width=len(data),
        height=1,
        encoding=ImageEncoding.MONO8,
        data=data,
    )


def _tmp_dirs(workspace: Path) -> list[Path]:
    return list(workspace.glob(".tmp-*"))


def _final_dirs(workspace: Path) -> list[Path]:
    return [
        path for path in workspace.iterdir() if path.is_dir() and not path.name.startswith(".tmp-")
    ]


def test_add_observation_persists_the_payload_before_finalize(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    writer.add_observation(_image(1, b"\x01\x02\x03"))

    (tmp_dir,) = _tmp_dirs(tmp_path)
    assert (tmp_dir / "rgb" / "frame-0001.bin").read_bytes() == b"\x01\x02\x03"
    assert _final_dirs(tmp_path) == []


def test_writer_does_not_retain_payload_bytes_in_memory(tmp_path: Path) -> None:
    import tracemalloc

    writer = _writer(tmp_path)
    payload_size = 1_000_000

    tracemalloc.start()
    try:
        for index in range(20):
            writer.add_observation(_image(index, bytes([index]) * payload_size))
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert retained < 5 * payload_size  # 20 payloads queued in memory would be 20 * payload_size
    manifest = writer.finalize()
    reader = SequenceArtifactReader(_output_dir(tmp_path))
    assert reader.verify_integrity() == []
    assert manifest.observation_counts["image"] == 20


def test_streamed_artifact_exposes_summary_diagnostics_without_payloads(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    writer.set_diagnostics()

    writer.finalize()

    reader = SequenceArtifactReader(_output_dir(tmp_path))
    diagnostics = reader.read_diagnostics()
    assert diagnostics is not None
    assert diagnostics.summary.modality_summaries["image"].count == 2
    assert list(diagnostics.summary.image_resolutions) == [(2, 1)]
    assert list(diagnostics.summary.pointcloud_field_names) == [("x", "y", "z")]


def test_abort_removes_the_partial_artifact_and_closes_the_writer(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)

    writer.abort()

    assert _tmp_dirs(tmp_path) == []
    assert _final_dirs(tmp_path) == []
    with pytest.raises(SequenceArtifactError, match="abort"):
        writer.add_observation(_image(9, b"\x00"))
    with pytest.raises(SequenceArtifactError, match="abort"):
        writer.finalize()


def test_writer_context_manager_aborts_when_the_block_raises(tmp_path: Path) -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        _writer(tmp_path) as writer,
    ):
        _build_fixture_sequence(writer)
        raise RuntimeError("boom")

    assert _tmp_dirs(tmp_path) == []


def test_writer_context_manager_keeps_a_finalized_artifact(tmp_path: Path) -> None:
    with _writer(tmp_path) as writer:
        _build_fixture_sequence(writer)
        writer.finalize()

    assert _tmp_dirs(tmp_path) == []
    reader = SequenceArtifactReader(_output_dir(tmp_path))
    assert reader.verify_integrity() == []


def test_rejected_finalize_leaves_no_temporary_directory(tmp_path: Path) -> None:
    for _ in range(2):
        writer = _writer(tmp_path)
        _build_fixture_sequence(writer)
        if _ == 0:
            writer.finalize()

    with pytest.raises(SequenceArtifactError, match="already exists"):
        writer.finalize()

    assert _tmp_dirs(tmp_path) == []


def test_empty_writer_finalizes_to_an_empty_valid_artifact(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    writer.finalize()

    reader = SequenceArtifactReader(_output_dir(tmp_path))
    assert reader.verify_integrity() == []
    assert reader.list_observations() == []


class _FailingIndexHandle:
    """Wraps the real index handle; ``write``/``close`` can be made to fail like a full disk."""

    def __init__(self, real: Any, *, fail_write: bool, fail_close: bool) -> None:
        self._real = real
        self._fail_write = fail_write
        self._fail_close = fail_close

    def write(self, data: bytes) -> int:
        if self._fail_write:
            raise OSError(errno.ENOSPC, "no space left on device (write)")
        return int(self._real.write(data))

    def close(self) -> None:
        self._real.close()  # the descriptor is released even when the flush fails
        if self._fail_close:
            raise OSError(errno.ENOSPC, "no space left on device (flush)")


def _fail_index_io(
    monkeypatch: pytest.MonkeyPatch, *, fail_write: bool = False, fail_close: bool = False
) -> None:
    real_open = Path.open

    def open_with_failures(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        if self.name == "index.jsonl" and args[:1] == ("wb",):
            return _FailingIndexHandle(handle, fail_write=fail_write, fail_close=fail_close)
        return handle

    monkeypatch.setattr(Path, "open", open_with_failures)


def test_abort_removes_the_temporary_directory_even_when_closing_the_index_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_index_io(monkeypatch, fail_close=True)
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)

    writer.abort()  # a flush error on data being discarded is irrelevant and must not escape

    assert _tmp_dirs(tmp_path) == []
    with pytest.raises(SequenceArtifactError, match="abort"):
        writer.add_observation(_image(9, b"\x00"))


def test_failed_write_reports_the_original_error_and_cleans_up_when_close_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_index_io(monkeypatch, fail_write=True, fail_close=True)
    writer = _writer(tmp_path)

    with pytest.raises(OSError, match=r"\(write\)"):
        writer.add_observation(_image(1, b"\x01"))

    assert _tmp_dirs(tmp_path) == []


def test_abort_surfaces_a_failed_removal_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)
    real_rmtree = shutil.rmtree

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError(errno.EACCES, "cannot remove")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with pytest.raises(PermissionError):
        writer.abort()
    assert len(_tmp_dirs(tmp_path)) == 1  # still there, and the caller knows about it

    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    writer.abort()  # retry removes it

    assert _tmp_dirs(tmp_path) == []


def test_failed_cleanup_is_attached_to_the_original_error_instead_of_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    _build_fixture_sequence(writer)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError(errno.EACCES, "cannot remove")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with pytest.raises(RuntimeError, match="boom") as excinfo, writer:
        raise RuntimeError("boom")

    assert any("temporary" in note for note in excinfo.value.__notes__)
