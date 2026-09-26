import random
from collections.abc import Sequence
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


@pytest.mark.parametrize(
    ("winner_seconds", "loser_seconds"),
    [(1.01, 0.99), (1.01, 1.01)],
    ids=["symmetric_offsets", "equal_timestamps"],
)
def test_a_tie_goes_to_the_smallest_observation_id_and_the_loser_is_dropped(
    winner_seconds: float, loser_seconds: float
) -> None:
    # O empate em |offset| é decidido por observation_id, não pela ordem de entrada nem pelo sinal.
    observations: list[SourceObservation] = [
        _imu("imu-b", loser_seconds),
        _image("frame-0001", 1.00),
        _imu("imu-a", winner_seconds),
    ]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=50_000_000)

    groups, diagnostics = synchronize(observations, config=config)

    assert _observation_id(groups[0].associations["imu"]) == "imu-a"
    assert [
        (str(event.observation.observation_id), event.reason)
        for event in diagnostics.dropped_events
    ] == [("imu-b", "no_anchor_within_tolerance")]


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


# --- ING-01: busca por bisseção, com o comportamento da varredura linear --------------------


def _brute_force(
    observations: Sequence[SourceObservation], config: SynchronizationConfig
) -> tuple[list[tuple[object, ...]], list[tuple[str, str]]]:
    """Referência de força bruta: a varredura linear por anchor, decisão a decisão."""
    by_modality: dict[str, list[SourceObservation]] = {"image": [], "imu": []}
    for item in observations:
        by_modality["image" if isinstance(item, ImageObservation) else "imu"].append(item)
    anchors = sorted(
        by_modality["image"],
        key=lambda item: (item.timestamp.total_nanoseconds(), str(item.observation_id)),
    )
    decisions: list[tuple[object, ...]] = []
    selected: set[str] = set()
    for frame_index, anchor in enumerate(anchors):
        anchor_ns = anchor.timestamp.total_nanoseconds()
        comparable = [
            item
            for item in by_modality["imu"]
            if item.timestamp.clock_id == anchor.timestamp.clock_id
            and abs(item.timestamp.total_nanoseconds() - anchor_ns) <= config.tolerance_nanoseconds
        ]
        if comparable:
            best = min(
                comparable,
                key=lambda item: (
                    abs(item.timestamp.total_nanoseconds() - anchor_ns),
                    str(item.observation_id),
                ),
            )
            selected.add(str(best.observation_id))
            offset = best.timestamp.total_nanoseconds() - anchor_ns
            decisions.append(
                (
                    frame_index,
                    str(anchor.observation_id),
                    str(best.observation_id),
                    offset,
                    "matched",
                )
            )
        else:
            if not by_modality["imu"]:
                status = "no_candidate"
            elif all(
                item.timestamp.clock_id != anchor.timestamp.clock_id for item in by_modality["imu"]
            ):
                status = "clock_id_mismatch"
            else:
                status = "outside_tolerance"
            decisions.append((frame_index, str(anchor.observation_id), None, None, status))
    anchor_clocks = {anchor.timestamp.clock_id for anchor in anchors}
    dropped = [
        (
            str(item.observation_id),
            "no_anchor_within_tolerance"
            if item.timestamp.clock_id in anchor_clocks
            else "clock_id_mismatch",
        )
        for item in by_modality["imu"]
        if str(item.observation_id) not in selected
    ]
    return decisions, dropped


def _random_scene(seed: int) -> list[SourceObservation]:
    rng = random.Random(seed)
    ids = rng.sample(range(10_000), 60)
    observations: list[SourceObservation] = []
    for index, number in enumerate(ids):
        # Passos de 10 ms com repetição: timestamps duplicados, empates de offset e eventos fora
        # da tolerância; dois relógios para exercitar o domínio de clock.
        seconds = 1_700_000_000 + rng.randrange(0, 40) * 0.01
        clock = rng.choice(["clock-a", "clock-a", "clock-b"])
        if index < 15:
            observations.append(_image(f"frame-{number:05d}", seconds, clock))
        else:
            observations.append(_imu(f"imu-{number:05d}", seconds, clock))
    rng.shuffle(observations)
    return observations


@pytest.mark.parametrize("seed", range(40))
def test_bisection_selects_exactly_what_the_linear_scan_selects(seed: int) -> None:
    observations = _random_scene(seed)
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=20_000_000)

    _, diagnostics = synchronize(observations, config=config)

    expected_decisions, expected_dropped = _brute_force(observations, config)
    imu_decisions = [
        (
            decision.frame_index,
            decision.anchor_observation_id,
            decision.selected_observation_id,
            decision.offset_nanoseconds,
            decision.status,
        )
        for decision in diagnostics.decisions
        if decision.modality == "imu"
    ]
    assert imu_decisions == expected_decisions
    dropped = [
        (str(event.observation.observation_id), event.reason)
        for event in diagnostics.dropped_events
        if isinstance(event.observation, ImuObservation)
    ]
    assert dropped == expected_dropped


def test_the_work_per_anchor_does_not_grow_with_the_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Gate de escala por contador (sem tempo de parede): cada anchor não pode examinar todos
    # os C candidatos da modalidade.
    calls = 0
    real = SourceTimestamp.total_nanoseconds

    def counting(self: SourceTimestamp) -> int:
        nonlocal calls
        calls += 1
        return real(self)

    monkeypatch.setattr(SourceTimestamp, "total_nanoseconds", counting)
    anchor_count, candidate_count = 50, 2_000
    observations: list[SourceObservation] = [
        _image(f"frame-{index:04d}", 1_000 + index * 0.1) for index in range(anchor_count)
    ]
    observations += [
        _imu(f"imu-{index:05d}", 1_000 + index * 0.0025) for index in range(candidate_count)
    ]
    config = SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=5_000_000)

    synchronize(observations, config=config)

    assert calls <= 4 * (anchor_count + candidate_count)
