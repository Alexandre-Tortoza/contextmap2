import dataclasses
from pathlib import Path

import pytest

from contextmap.ingestion import (
    ExplicitIdsSelection,
    FrameId,
    FrameRangeSelection,
    FullSequenceSelection,
    ImageEncoding,
    ImageObservation,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SequenceSelectionError,
    SourceObservationId,
    SourceProvenance,
    TimestampRangeSelection,
    decode_selection,
    encode_selection,
    resolve_selection,
    selection_identity,
)
from contextmap.shared import SourceTimestamp

_PROVENANCE = SourceProvenance(source_type="dataset", source_path="fixtures/example")


def _image(observation_id: str, seconds: int, clock_id: str = "clock-a") -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=SourceTimestamp(seconds=seconds, nanoseconds=0, clock_id=clock_id),
        provenance=_PROVENANCE,
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00",
    )


@pytest.fixture
def reader(tmp_path: Path) -> SequenceArtifactReader:
    writer = SequenceArtifactWriter(
        output_dir=tmp_path / "ingestion",
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-0001"),
    )
    for index in range(5):
        writer.add_observation(_image(f"frame-{index:04d}", seconds=index))
    writer.finalize()
    return SequenceArtifactReader(tmp_path / "ingestion")


def test_full_selection_returns_every_observation_in_order(reader: SequenceArtifactReader) -> None:
    result = resolve_selection(reader, FullSequenceSelection())

    assert [obs.observation_id for obs in result.observations] == [
        f"frame-{i:04d}" for i in range(5)
    ]
    assert result.sequence_artifact_id == reader.manifest.artifact_id


def test_frame_range_selection_matches_full_sequence_slice(reader: SequenceArtifactReader) -> None:
    full = resolve_selection(reader, FullSequenceSelection()).observations
    result = resolve_selection(reader, FrameRangeSelection(start_frame_index=1, end_frame_index=3))

    assert list(result.observations) == list(full[1:3])


def test_frame_range_end_beyond_length_is_clipped(reader: SequenceArtifactReader) -> None:
    result = resolve_selection(
        reader, FrameRangeSelection(start_frame_index=3, end_frame_index=100)
    )

    assert [obs.observation_id for obs in result.observations] == ["frame-0003", "frame-0004"]


def test_frame_range_start_beyond_length_raises(reader: SequenceArtifactReader) -> None:
    with pytest.raises(SequenceSelectionError, match="out of range"):
        resolve_selection(reader, FrameRangeSelection(start_frame_index=10, end_frame_index=12))


@pytest.fixture
def empty_reader(tmp_path: Path) -> SequenceArtifactReader:
    SequenceArtifactWriter(
        output_dir=tmp_path / "empty",
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-empty"),
    ).finalize()
    return SequenceArtifactReader(tmp_path / "empty")


def test_frame_range_start_beyond_an_empty_sequence_raises(
    empty_reader: SequenceArtifactReader,
) -> None:
    with pytest.raises(SequenceSelectionError, match="out of range for 0 observations"):
        resolve_selection(empty_reader, FrameRangeSelection(start_frame_index=3, end_frame_index=5))


def test_frame_range_from_zero_over_an_empty_sequence_is_empty(
    empty_reader: SequenceArtifactReader,
) -> None:
    result = resolve_selection(
        empty_reader, FrameRangeSelection(start_frame_index=0, end_frame_index=10)
    )

    assert result.observations == ()


def test_timestamp_range_selects_within_bounds(reader: SequenceArtifactReader) -> None:
    result = resolve_selection(
        reader, TimestampRangeSelection(clock_id="clock-a", start_seconds=1.0, end_seconds=3.0)
    )

    assert [obs.observation_id for obs in result.observations] == ["frame-0001", "frame-0002"]


def test_timestamp_range_bounds_compare_exact_nanoseconds(tmp_path: Path) -> None:
    # Regressão ING-02: em t ~ 1.7e9 s, 200 ns e 300 ns arredondam para o mesmo float (~238 ns);
    # o limite inferior é esse float, então só a entrada de 300 ns está de fato dentro da faixa.
    writer = SequenceArtifactWriter(
        output_dir=tmp_path / "ingestion",
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-0001"),
    )
    for observation_id, nanoseconds in (("below", 200), ("inside", 300)):
        image = _image(observation_id, seconds=1_700_000_000)
        stamp = SourceTimestamp(seconds=1_700_000_000, nanoseconds=nanoseconds, clock_id="clock-a")
        writer.add_observation(dataclasses.replace(image, timestamp=stamp))
    writer.finalize()
    start = 1_700_000_000 + 200e-9
    assert start == 1_700_000_000 + 300e-9  # os dois colapsam no mesmo float

    result = resolve_selection(
        SequenceArtifactReader(tmp_path / "ingestion"),
        TimestampRangeSelection(clock_id="clock-a", start_seconds=start, end_seconds=1.8e9),
    )

    assert [obs.observation_id for obs in result.observations] == ["inside"]


def test_timestamp_range_excludes_mismatched_clock_id(reader: SequenceArtifactReader) -> None:
    result = resolve_selection(
        reader, TimestampRangeSelection(clock_id="clock-b", start_seconds=0.0, end_seconds=10.0)
    )

    assert result.observations == ()


def test_explicit_ids_selection_preserves_sequence_order(reader: SequenceArtifactReader) -> None:
    result = resolve_selection(
        reader,
        ExplicitIdsSelection(
            observation_ids=frozenset(
                {SourceObservationId("frame-0003"), SourceObservationId("frame-0001")}
            )
        ),
    )

    assert [obs.observation_id for obs in result.observations] == ["frame-0001", "frame-0003"]


def test_explicit_ids_selection_raises_for_unknown_id(reader: SequenceArtifactReader) -> None:
    with pytest.raises(SequenceSelectionError, match="not found"):
        resolve_selection(
            reader,
            ExplicitIdsSelection(
                observation_ids=frozenset({SourceObservationId("does-not-exist")})
            ),
        )


def test_empty_frame_range_is_a_valid_empty_result(reader: SequenceArtifactReader) -> None:
    result = resolve_selection(reader, FrameRangeSelection(start_frame_index=2, end_frame_index=2))

    assert result.observations == ()


def test_explicit_ids_selection_rejects_empty_set() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ExplicitIdsSelection(observation_ids=frozenset())


def test_frame_range_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end_frame_index"):
        FrameRangeSelection(start_frame_index=5, end_frame_index=2)


def test_selection_identity_is_deterministic_and_selection_specific() -> None:
    from contextmap.ingestion.sequence_artifact import SequenceArtifactId

    artifact_id = SequenceArtifactId("corridor-02-a1b2c3")
    selection_a = FrameRangeSelection(start_frame_index=0, end_frame_index=5)
    selection_b = FrameRangeSelection(start_frame_index=0, end_frame_index=6)

    assert selection_identity(artifact_id, selection_a) == selection_identity(
        artifact_id, selection_a
    )
    assert selection_identity(artifact_id, selection_a) != selection_identity(
        artifact_id, selection_b
    )


def test_selection_round_trips_through_encode_decode() -> None:
    for selection in (
        FullSequenceSelection(),
        FrameRangeSelection(start_frame_index=1, end_frame_index=3),
        TimestampRangeSelection(clock_id="clock-a", start_seconds=0.0, end_seconds=1.0),
        ExplicitIdsSelection(observation_ids=frozenset({SourceObservationId("frame-0001")})),
    ):
        assert decode_selection(encode_selection(selection)) == selection


def test_repeated_runs_share_selection_id_but_stay_independent(
    reader: SequenceArtifactReader,
) -> None:
    selection = FrameRangeSelection(start_frame_index=0, end_frame_index=3)

    result_a = resolve_selection(reader, selection)
    result_b = resolve_selection(reader, selection)

    assert result_a.selection_id == result_b.selection_id
    assert result_a.observations == result_b.observations
    assert result_a is not result_b
