import pytest

from contextmap.ingestion import (
    FrameId,
    ImageEncoding,
    ImageObservation,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservationId,
    SourceProvenance,
    validate_frame_references,
    validate_image_observation,
    validate_lidar_observation,
    validate_observations,
    validate_timestamp_ordering,
)
from contextmap.shared import SourceTimestamp
from fixtures import (
    build_calibration_set,
    build_image_with_data_size_mismatch,
    build_lidar_with_inconsistent_fields,
    build_non_monotonic_observations,
    build_observation_with_unknown_frame,
    build_valid_sequence,
)


def test_valid_sequence_has_no_problems() -> None:
    observations = build_valid_sequence()

    assert validate_observations(observations, calibration=build_calibration_set()) == []


def test_valid_image_observation_has_no_problems() -> None:
    image = next(obs for obs in build_valid_sequence() if obs.observation_id == "frame-0001")

    assert validate_image_observation(image) == []  # type: ignore[arg-type]


def test_detects_image_data_size_mismatch() -> None:
    problems = validate_image_observation(build_image_with_data_size_mismatch())

    assert any("does not match" in problem for problem in problems)


def test_detects_lidar_data_size_and_field_offset_problems() -> None:
    problems = validate_lidar_observation(build_lidar_with_inconsistent_fields())

    assert any("out of range" in problem for problem in problems)


@pytest.mark.parametrize(
    ("field", "field_end"),
    [
        (
            PointFieldDescriptor(
                name="range", offset_bytes=4, data_type=PointFieldDataType.FLOAT64
            ),
            12,
        ),
        (
            PointFieldDescriptor(
                name="normal", offset_bytes=0, data_type=PointFieldDataType.FLOAT32, count=3
            ),
            12,
        ),
    ],
    ids=["wide_data_type", "element_count"],
)
def test_detects_a_field_that_extends_beyond_point_step(
    field: PointFieldDescriptor, field_end: int
) -> None:
    # O offset está dentro do registro, mas offset + size*count ultrapassa o point_step.
    lidar = LidarObservation(
        observation_id=SourceObservationId("scan-overflow"),
        sensor_id=SensorId("velodyne_top"),
        frame_id=FrameId("velodyne"),
        timestamp=SourceTimestamp(seconds=1, nanoseconds=0, clock_id="fixture:header"),
        provenance=SourceProvenance(source_type="dataset", source_path="fixtures/example"),
        point_count=1,
        point_step_bytes=8,
        fields=(field,),
        data=b"\x00" * 8,
    )

    problems = validate_lidar_observation(lidar)

    assert problems == [
        f"scan-overflow: field {field.name!r} ends at byte {field_end}, beyond point_step_bytes=8"
    ]


def test_a_field_ending_exactly_at_point_step_is_valid() -> None:
    lidar = LidarObservation(
        observation_id=SourceObservationId("scan-tight"),
        sensor_id=SensorId("velodyne_top"),
        frame_id=FrameId("velodyne"),
        timestamp=SourceTimestamp(seconds=1, nanoseconds=0, clock_id="fixture:header"),
        provenance=SourceProvenance(source_type="dataset", source_path="fixtures/example"),
        point_count=1,
        point_step_bytes=8,
        fields=(
            PointFieldDescriptor(
                name="xy", offset_bytes=0, data_type=PointFieldDataType.FLOAT32, count=2
            ),
        ),
        data=b"\x00" * 8,
    )

    assert validate_lidar_observation(lidar) == []


def test_detects_non_monotonic_timestamps_on_the_same_clock() -> None:
    problems = validate_timestamp_ordering(build_non_monotonic_observations())

    assert any("non-monotonic" in problem for problem in problems)


def test_allows_duplicate_timestamps_by_default() -> None:
    observations = build_valid_sequence()
    duplicate = [observations[0], observations[0]]

    assert validate_timestamp_ordering(duplicate) == []


def test_rejects_duplicate_timestamps_when_disallowed() -> None:
    observations = build_valid_sequence()
    duplicate = [observations[0], observations[0]]

    problems = validate_timestamp_ordering(duplicate, allow_duplicates=False)

    assert any("duplicate timestamp" in problem for problem in problems)


def test_timestamps_on_different_clocks_are_never_compared() -> None:
    later_but_different_clock = ImageObservation(
        observation_id=SourceObservationId("frame-other-clock"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=SourceTimestamp(seconds=0, nanoseconds=0, clock_id="other-clock"),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/corridor"),
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00\x00\x00",
    )
    observations = [*build_valid_sequence(), later_but_different_clock]

    assert validate_timestamp_ordering(observations) == []


# Em t ~ 1.7e9 s o ulp de float64 é ~238 ns: estes pares colapsam no mesmo float.
_EPOCH_SECONDS = 1_700_000_000


def _at(observation_id: str, nanoseconds: int) -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=SourceTimestamp(seconds=_EPOCH_SECONDS, nanoseconds=nanoseconds, clock_id="rec"),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/corridor"),
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00\x00\x00",
    )


def test_a_regression_below_float_resolution_is_non_monotonic() -> None:
    # Regressão ING-02: 100 ns -> 0 ns viravam o mesmo float e a regressão sumia.
    problems = validate_timestamp_ordering([_at("first", 100), _at("second", 0)])

    assert len(problems) == 1
    assert "non-monotonic" in problems[0]


def test_distinct_timestamps_below_float_resolution_are_not_duplicates() -> None:
    problems = validate_timestamp_ordering(
        [_at("first", 10), _at("second", 50)], allow_duplicates=False
    )

    assert problems == []


def test_detects_unknown_frame_reference() -> None:
    problems = validate_frame_references(
        [build_observation_with_unknown_frame()], build_calibration_set()
    )

    assert any("not a known calibration frame" in problem for problem in problems)


def test_frame_reference_check_is_a_no_op_without_calibration() -> None:
    assert validate_frame_references([build_observation_with_unknown_frame()], None) == []
