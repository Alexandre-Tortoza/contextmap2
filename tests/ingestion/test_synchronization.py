from dataclasses import replace

import pytest

from contextmap.ingestion import (
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    ModalityAssociation,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
    SynchronizationConfig,
    synchronize,
)
from contextmap.shared import SourceTimestamp

_PROVENANCE = SourceProvenance(source_type="dataset", source_path="fixtures/example")


def _observation_id(association: ModalityAssociation) -> str:
    assert association.observation is not None
    return str(association.observation.observation_id)


def _image(observation_id: str, seconds: float, clock_id: str = "clock-a") -> ImageObservation:
    whole = int(seconds)
    nanoseconds = round((seconds - whole) * 1_000_000_000)
    return ImageObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=SourceTimestamp(seconds=whole, nanoseconds=nanoseconds, clock_id=clock_id),
        provenance=_PROVENANCE,
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00",
    )


def _imu(observation_id: str, seconds: float, clock_id: str = "clock-a") -> ImuObservation:
    whole = int(seconds)
    nanoseconds = round((seconds - whole) * 1_000_000_000)
    return ImuObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("imu0"),
        frame_id=FrameId("imu_link"),
        timestamp=SourceTimestamp(seconds=whole, nanoseconds=nanoseconds, clock_id=clock_id),
        provenance=_PROVENANCE,
    )


def test_matches_nearest_candidate_within_tolerance() -> None:
    observations: list[SourceObservation] = [
        _image("frame-0001", 1.00),
        _image("frame-0002", 2.00),
        _imu("imu-0001", 0.99),
        _imu("imu-0002", 1.98),
        _imu("imu-0003", 3.50),
    ]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=50_000_000)

    groups, diagnostics = synchronize(observations, config=config)

    assert [group.frame_index for group in groups] == [0, 1]
    assert groups[0].anchor.observation_id == "frame-0001"
    assert _observation_id(groups[0].associations["imu"]) == "imu-0001"
    assert groups[0].associations["imu"].offset_nanoseconds == -10_000_000
    assert _observation_id(groups[1].associations["imu"]) == "imu-0002"

    assert len(diagnostics.dropped_events) == 1
    assert diagnostics.dropped_events[0].observation.observation_id == "imu-0003"
    assert diagnostics.dropped_events[0].reason == "no_anchor_within_tolerance"
    first_imu_decision = next(
        decision
        for decision in diagnostics.decisions
        if decision.frame_index == 0 and decision.modality == "imu"
    )
    assert first_imu_decision.status == "matched"
    assert first_imu_decision.offset_nanoseconds == -10_000_000


def test_missing_modality_is_explicit_none_not_dropped_silently() -> None:
    observations = [_image("frame-0001", 1.00)]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=50_000_000)

    groups, diagnostics = synchronize(observations, config=config)

    assert groups[0].associations["imu"].observation is None
    assert groups[0].associations["imu"].offset_nanoseconds is None
    assert groups[0].associations["lidar"].observation is None
    assert groups[0].associations["external_pose"].observation is None
    assert diagnostics.dropped_events == ()
    assert {decision.status for decision in diagnostics.decisions} == {"no_candidate"}


def test_grouping_is_deterministic_regardless_of_input_order() -> None:
    observations: list[SourceObservation] = [
        _imu("imu-0002", 1.98),
        _image("frame-0002", 2.00),
        _imu("imu-0001", 0.99),
        _image("frame-0001", 1.00),
    ]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=50_000_000)

    groups_a, _ = synchronize(observations, config=config)
    groups_b, _ = synchronize(list(reversed(observations)), config=config)

    assert [g.anchor.observation_id for g in groups_a] == [
        g.anchor.observation_id for g in groups_b
    ]
    assert [_observation_id(g.associations["imu"]) for g in groups_a] == [
        _observation_id(g.associations["imu"]) for g in groups_b
    ]


def test_duplicate_anchor_timestamps_are_ordered_by_observation_id() -> None:
    observations = [_image("frame-b", 1.00), _image("frame-a", 1.00)]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=50_000_000)

    groups, _ = synchronize(observations, config=config)

    assert [g.anchor.observation_id for g in groups] == ["frame-a", "frame-b"]


def test_candidate_may_be_selected_by_more_than_one_anchor() -> None:
    observations: list[SourceObservation] = [
        _image("frame-0001", 1.00),
        _image("frame-0002", 1.02),
        _imu("imu-0001", 1.01),
    ]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=50_000_000)

    groups, diagnostics = synchronize(observations, config=config)

    assert _observation_id(groups[0].associations["imu"]) == "imu-0001"
    assert _observation_id(groups[1].associations["imu"]) == "imu-0001"
    assert diagnostics.dropped_events == ()


def test_mismatched_clock_id_is_never_treated_as_comparable() -> None:
    observations: list[SourceObservation] = [
        _image("frame-0001", 1.00, clock_id="clock-a"),
        _imu("imu-0001", 1.00, clock_id="clock-b"),
    ]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=1_000_000_000)

    groups, diagnostics = synchronize(observations, config=config)

    assert groups[0].associations["imu"].observation is None
    assert len(diagnostics.dropped_events) == 1
    assert diagnostics.dropped_events[0].reason == "clock_id_mismatch"
    imu_decision = next(
        decision for decision in diagnostics.decisions if decision.modality == "imu"
    )
    assert imu_decision.status == "clock_id_mismatch"


def test_out_of_order_source_events_are_still_grouped_correctly() -> None:
    observations = [_image("frame-late", 5.00), _image("frame-early", 1.00)]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=50_000_000)

    groups, _ = synchronize(observations, config=config)

    assert [g.anchor.observation_id for g in groups] == ["frame-early", "frame-late"]
    assert [g.frame_index for g in groups] == [0, 1]


def test_rejects_unknown_reference_modality() -> None:
    with pytest.raises(ValueError, match="reference_modality"):
        SynchronizationConfig(reference_modality="radar", tolerance_nanoseconds=50_000_000)


def test_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance_nanoseconds"):
        SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=-1)


def test_large_epoch_timestamps_are_compared_without_float_precision_loss() -> None:
    anchor = replace(
        _image("frame", 0.0),
        timestamp=SourceTimestamp(
            seconds=1_700_000_000,
            nanoseconds=100,
            clock_id="clock-a",
        ),
    )
    candidate = replace(
        _imu("imu", 0.0),
        timestamp=SourceTimestamp(
            seconds=1_700_000_000,
            nanoseconds=101,
            clock_id="clock-a",
        ),
    )

    groups, _ = synchronize(
        [anchor, candidate],
        config=SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=1),
    )

    assert groups[0].associations["imu"].offset_nanoseconds == 1
