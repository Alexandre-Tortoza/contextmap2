from fixtures import build_valid_sequence

from contextmap.ingestion import SequenceDiagnostics, summarize_observations
from contextmap.ingestion.diagnostics import decode_diagnostics_summary, encode_diagnostics


def test_summary_counts_observations_per_modality() -> None:
    summary = summarize_observations(build_valid_sequence())

    assert summary.modality_summaries["image"].count == 2
    assert summary.modality_summaries["lidar"].count == 1
    assert summary.modality_summaries["imu"].count == 1
    assert summary.modality_summaries["external_pose"].count == 1


def test_summary_reports_zero_count_for_absent_modality() -> None:
    only_images = [obs for obs in build_valid_sequence() if obs.observation_id.startswith("frame")]

    summary = summarize_observations(only_images)

    assert summary.modality_summaries["lidar"].count == 0
    assert summary.modality_summaries["lidar"].first_timestamp_seconds is None


def test_summary_reports_time_range_per_modality() -> None:
    summary = summarize_observations(build_valid_sequence())

    image_summary = summary.modality_summaries["image"]
    assert image_summary.first_timestamp_seconds == 1.0
    assert image_summary.last_timestamp_seconds == 2.0
    assert image_summary.clock_ids == ["fixture:header"]


def test_summary_collects_distinct_image_resolutions() -> None:
    summary = summarize_observations(build_valid_sequence())

    assert summary.image_resolutions == [(2, 1)]


def test_summary_collects_distinct_pointcloud_field_layouts() -> None:
    summary = summarize_observations(build_valid_sequence())

    assert summary.pointcloud_field_names == [("x", "y", "z")]


def test_summary_records_warning_count() -> None:
    summary = summarize_observations(build_valid_sequence(), warning_count=3)

    assert summary.warning_count == 3


def test_summary_reports_source_and_topic_inventory() -> None:
    summary = summarize_observations(build_valid_sequence())

    assert summary.source_types == ["fixture"]
    assert summary.topic_counts["/camera/image_raw"] == 2
    assert summary.topic_counts["/velodyne_points"] == 1


def test_diagnostics_round_trip_through_encode_decode() -> None:
    summary = summarize_observations(build_valid_sequence(), warning_count=1)
    diagnostics = SequenceDiagnostics(summary=summary, warnings=("missing imu topic",))

    decoded_summary = decode_diagnostics_summary(encode_diagnostics(diagnostics))

    assert decoded_summary == summary
