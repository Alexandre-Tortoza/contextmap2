from pathlib import Path

import pytest

from contextmap.ingestion import (
    ExternalPoseMeasurement,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SourceAdapterConfig,
    SourceAdapterWarning,
    SourceObservationId,
    SourceTopicMapping,
)
from contextmap.ingestion.adapters.pose_file import PoseFileConfigError, PoseFileSourceAdapter
from contextmap.ingestion.sequence_provenance import SequenceProvenance, compute_source_content_hash

_TUM_TEXT = (
    "# timestamp tx ty tz qx qy qz qw\n"
    "1645999726.984117 0.000000 0.000000 0.000000 0.002482 -0.023542 0.000058 0.999720\n"
    "1645999727.084925 -0.005613 0.001583 0.003994 0.002486 -0.018791 0.000028 0.999820\n"
    "\n"
    "1645999727.185785 -0.006679 0.002398 0.003451 0.002501 -0.018812 0.000014 0.999820\n"
)


def _pose_file(tmp_path: Path, text: str = _TUM_TEXT) -> Path:
    path = tmp_path / "corridor-02-gt.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _config(path: Path, **extra: object) -> SourceAdapterConfig:
    return SourceAdapterConfig(
        source_type="pose_file",
        path=str(path),
        topics=SourceTopicMapping(),
        extra={
            "format": "tum",
            "parent_frame": "map",
            "body_frame": "epson",
            **extra,
        },
    )


def test_reads_every_non_comment_line_as_an_external_pose_measurement(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    adapter = PoseFileSourceAdapter(_config(path))

    observations = list(adapter.read_observations())

    assert len(observations) == 3
    assert all(isinstance(observation, ExternalPoseMeasurement) for observation in observations)


def test_parses_frame_translation_and_orientation_from_configured_settings(
    tmp_path: Path,
) -> None:
    path = _pose_file(tmp_path)
    adapter = PoseFileSourceAdapter(_config(path))

    first = next(iter(adapter.read_observations()))

    assert isinstance(first, ExternalPoseMeasurement)
    assert first.parent_frame == "map"
    assert first.frame_id == "epson"
    assert first.translation == (0.0, 0.0, 0.0)
    assert first.orientation == (0.002482, -0.023542, 0.000058, 0.99972)


def test_timestamp_is_converted_without_floating_point_rounding(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    adapter = PoseFileSourceAdapter(_config(path))

    first = next(iter(adapter.read_observations()))

    assert first.timestamp.seconds == 1645999726
    assert first.timestamp.nanoseconds == 984117000


def test_provenance_carries_source_path_format_and_content_hash(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    expected_hash = compute_source_content_hash(path)
    adapter = PoseFileSourceAdapter(_config(path))

    first = next(iter(adapter.read_observations()))

    assert first.provenance.source_type == "pose_file"
    assert first.provenance.source_path == str(path)
    assert first.provenance.raw_metadata["format"] == "tum"
    assert first.provenance.raw_metadata["source_content_hash"] == expected_hash


def test_observation_ids_are_stable_and_unique_per_line(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    adapter = PoseFileSourceAdapter(_config(path))

    ids = [str(observation.observation_id) for observation in adapter.read_observations()]

    assert len(ids) == len(set(ids))


def test_capabilities_report_external_pose_only(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    adapter = PoseFileSourceAdapter(_config(path))

    capabilities = adapter.capabilities()

    assert capabilities.external_pose is True
    assert capabilities.rgb is False
    assert capabilities.lidar is False
    assert capabilities.imu is False


def test_read_calibration_returns_none_when_not_configured(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    adapter = PoseFileSourceAdapter(_config(path))

    assert adapter.read_calibration() is None


def test_missing_format_raises_config_error_before_reading_the_file(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    config = SourceAdapterConfig(
        source_type="pose_file",
        path=str(path),
        topics=SourceTopicMapping(),
        extra={"parent_frame": "map", "body_frame": "epson"},
    )

    with pytest.raises(PoseFileConfigError, match="format"):
        PoseFileSourceAdapter(config)


def test_missing_parent_frame_raises_config_error(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    config = SourceAdapterConfig(
        source_type="pose_file",
        path=str(path),
        topics=SourceTopicMapping(),
        extra={"format": "tum", "body_frame": "epson"},
    )

    with pytest.raises(PoseFileConfigError, match="parent_frame"):
        PoseFileSourceAdapter(config)


def test_missing_body_frame_raises_config_error(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)
    config = SourceAdapterConfig(
        source_type="pose_file",
        path=str(path),
        topics=SourceTopicMapping(),
        extra={"format": "tum", "parent_frame": "map"},
    )

    with pytest.raises(PoseFileConfigError, match="body_frame"):
        PoseFileSourceAdapter(config)


def test_unsupported_format_raises_config_error(tmp_path: Path) -> None:
    path = _pose_file(tmp_path)

    with pytest.raises(PoseFileConfigError, match="format"):
        PoseFileSourceAdapter(_config(path, format="csv"))


def test_malformed_line_becomes_a_warning_not_a_crash(tmp_path: Path) -> None:
    path = _pose_file(
        tmp_path,
        text=(
            "1645999726.984117 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
            "not-a-valid-line\n"
            "1645999727.084925 0.1 0.0 0.0 0.0 0.0 0.0 1.0\n"
        ),
    )
    adapter = PoseFileSourceAdapter(_config(path))

    observations = list(adapter.read_observations())

    assert len(observations) == 2
    (warning,) = adapter.warnings()
    assert isinstance(warning, SourceAdapterWarning)
    assert warning.message_index == 2


def test_non_finite_timestamp_becomes_a_warning(tmp_path: Path) -> None:
    path = _pose_file(
        tmp_path,
        text="not-a-number 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n",
    )
    adapter = PoseFileSourceAdapter(_config(path))

    observations = list(adapter.read_observations())

    assert observations == []
    (warning,) = adapter.warnings()
    assert "timestamp" in warning.reason


def test_missing_source_file_raises_file_not_found(tmp_path: Path) -> None:
    adapter = PoseFileSourceAdapter(_config(tmp_path / "does-not-exist.txt"))

    with pytest.raises(FileNotFoundError):
        list(adapter.read_observations())


def test_same_adapter_processes_a_different_file_via_configuration_only(tmp_path: Path) -> None:
    first_path = _pose_file(tmp_path)
    second_path = tmp_path / "second.txt"
    second_path.write_text("1700000000.000000 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n", encoding="utf-8")

    first_adapter = PoseFileSourceAdapter(_config(first_path))
    second_adapter = PoseFileSourceAdapter(_config(second_path))

    assert len(list(first_adapter.read_observations())) == 3
    assert len(list(second_adapter.read_observations())) == 1


def test_real_corridor_02_ground_truth_file_parses_to_5522_measurements() -> None:
    real_path = Path("/home/alexmrtr/Projects/contextmap2/datasets/corridor-02/corridor-02-gt.txt")
    if not real_path.is_file():
        pytest.skip("corridor-02 dataset is not available (it is not versioned)")
    adapter = PoseFileSourceAdapter(_config(real_path))

    observations = list(adapter.read_observations())

    assert len(observations) == 5522
    assert adapter.warnings() == ()
    assert all(isinstance(observation, ExternalPoseMeasurement) for observation in observations)


def test_real_corridor_02_ground_truth_file_ingests_into_a_companion_sequence_artifact(
    tmp_path: Path,
) -> None:
    """Real ingestion, end to end: adapter -> writer -> reader, no reingestion of the bag.

    Builds the "documented, identified companion artifact" the issue's
    acceptance criteria allows as an alternative to merging into the main
    `corridor-02` sequence artifact (which would require reingesting the
    24 GB bag, out of scope here). Every observation, hash and count below
    comes from the real `corridor-02-gt.txt` file.
    """
    real_path = Path("/home/alexmrtr/Projects/contextmap2/datasets/corridor-02/corridor-02-gt.txt")
    if not real_path.is_file():
        pytest.skip("corridor-02 dataset is not available (it is not versioned)")

    adapter = PoseFileSourceAdapter(_config(real_path, sensor_id="corridor-02-gt"))
    output_dir = tmp_path / "corridor-02-external-pose"
    with SequenceArtifactWriter(
        output_dir=output_dir,
        sequence_name="corridor-02-external-pose",
        artifact_id=SequenceArtifactId("corridor-02-external-pose-real-validation"),
    ) as writer:
        for observation in adapter.read_observations():
            writer.add_observation(observation)
        writer.set_provenance(
            SequenceProvenance(
                source_type="pose_file",
                source_path=str(real_path),
                source_content_hash=compute_source_content_hash(real_path),
                adapter_type="pose_file",
            )
        )
        manifest = writer.finalize()

    assert manifest.observation_counts["external_pose"] == 5522
    assert adapter.warnings() == ()

    reader = SequenceArtifactReader(output_dir)
    assert reader.verify_integrity() == []
    provenance = reader.read_provenance()
    assert provenance is not None
    assert provenance.source_content_hash == compute_source_content_hash(real_path)
    first = reader.get_observation(SourceObservationId("corridor-02-gt-000001"))
    assert isinstance(first, ExternalPoseMeasurement)
    assert first.timestamp.seconds == 1645999726
