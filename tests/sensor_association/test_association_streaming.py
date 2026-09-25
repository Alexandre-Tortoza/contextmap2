"""Bounded-memory execution: a completed frame is persisted and released, not retained.

The invariant these tests hold the service to is that finishing frame ``K`` never needs
the heavy projection and visibility arrays of frames ``0..K-1``. They check it directly,
with weak references: by the time a later frame is accepted, every earlier frame's
resolution must already be collectable, which is only true if neither the service nor the
outcome keeps it alive.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import weakref
from collections.abc import Iterator
from pathlib import Path

import pytest
from run_builders import NATIVE, frame_id, frame_input, make_request

from contextmap.sensor_association import (
    AssociationInputError,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
)
from contextmap.sensor_association.run_artifact import RunArtifactError
from contextmap.sensor_association.service import AssociationFrameInput, FrameAssociation
from contextmap.state_estimation import LookupPolicy

SERVICE = SensorAssociationService()

# Quatro frames dentro da janela da trajetória do fixture, para o invariante valer entre
# vários frames e não só entre dois.
MANY_FRAMES = tuple(range(4))
FRAME_TIMES = (0, 25_000_000, 50_000_000, 100_000_000)


def _many_frames() -> list[object]:
    return [frame_input(index, time_ns=FRAME_TIMES[index]) for index in MANY_FRAMES]


class _Collecting:
    """A sink that keeps every frame, for the assertions that need them all."""

    def __init__(self) -> None:
        self.frames: list[FrameAssociation] = []

    def accept(self, frame: FrameAssociation) -> None:
        self.frames.append(frame)


class _Releasing:
    """A sink that keeps only weak references, plus what it observed about lifetime."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.alive_when_accepted: list[int] = []
        self._resolutions: list[weakref.ref[object]] = []

    def accept(self, frame: FrameAssociation) -> None:
        gc.collect()
        self.alive_when_accepted.append(sum(1 for ref in self._resolutions if ref() is not None))
        self.order.append(str(frame.source_observation_id))
        self._resolutions.append(weakref.ref(frame.resolution))

    def still_alive(self) -> int:
        gc.collect()
        return sum(1 for ref in self._resolutions if ref() is not None)


def _writer(tmp_path: Path, **kwargs: object) -> SensorAssociationRunWriter:
    return SensorAssociationRunWriter(
        output_dir=tmp_path / "association",
        sequence_name="fixture",
        run_id=SensorAssociationRunId("sa-stream"),
        run_index=1,
        **kwargs,  # type: ignore[arg-type]
    )


# --- The sink sees every frame, once, in order -----------------------------------------------


def test_each_completed_frame_reaches_the_sink_in_the_order_it_was_requested() -> None:
    sink = _Collecting()

    outcome = SERVICE.run(make_request(), sink=sink)

    assert [f.source_observation_id for f in sink.frames] == [frame_id(0), frame_id(1)]
    assert outcome.frame_count == 2


def test_the_outcome_carries_run_level_facts_and_no_per_frame_arrays() -> None:
    outcome = SERVICE.run(make_request(), sink=_Collecting())

    assert not hasattr(outcome, "frames")
    assert outcome.frame_count == 2
    assert outcome.rejected == ()


def test_a_rejected_frame_is_reported_without_reaching_the_sink() -> None:
    late = frame_input(1, time_ns=10_000_000_000)
    sink = _Collecting()

    outcome = SERVICE.run(make_request(frames=[frame_input(0), late]), sink=sink)

    assert [f.source_observation_id for f in sink.frames] == [frame_id(0)]
    assert [r.source_observation_id for r in outcome.rejected] == [frame_id(1)]
    assert outcome.frame_count == 1


# --- Nothing heavy survives the frame that produced it ---------------------------------------


def test_finishing_a_frame_never_needs_the_arrays_of_the_frames_before_it() -> None:
    sink = _Releasing()

    SERVICE.run(
        make_request(frames=_many_frames(), pose_policy=LookupPolicy.interpolated()),  # type: ignore[arg-type]
        sink=sink,
    )

    assert sink.order == [str(frame_id(index)) for index in range(4)]
    # Ao aceitar o frame K, nenhuma resolução de 0..K-1 ainda está viva.
    assert sink.alive_when_accepted == [0, 0, 0, 0]
    assert sink.still_alive() == 0


def test_the_service_holds_no_frame_once_the_run_returns() -> None:
    sink = _Releasing()

    outcome = SERVICE.run(make_request(frames=[frame_input(0), frame_input(1)]), sink=sink)

    assert outcome.frame_count == 2
    assert sink.still_alive() == 0


# --- The input side is bounded too -----------------------------------------------------------


def test_a_generator_of_frames_is_consumed_once_and_never_held() -> None:
    """The run must not retain its inputs either, or memory still grows with frame count.

    A weak reference to each `FrameAssociation` proves only that the *output* is released;
    the executor used to materialize every image payload up front, which on corridor-02 was
    2.19 GB alive for the whole run. This holds the service to consuming its frames lazily.
    """
    alive: list[weakref.ref[object]] = []

    def produce() -> Iterator[object]:
        for index in range(4):
            frame = frame_input(index, time_ns=FRAME_TIMES[index])
            alive.append(weakref.ref(frame))
            yield frame
            del frame

    outcome = SERVICE.run(
        make_request(frames=produce(), pose_policy=LookupPolicy.interpolated()),  # type: ignore[arg-type]
        sink=_Releasing(),
    )

    assert outcome.frame_count == 4
    gc.collect()
    # Nenhum AssociationFrameInput sobrevive ao frame que o consumiu.
    assert sum(1 for ref in alive if ref() is not None) == 0


def test_only_one_frame_input_is_alive_while_the_run_advances() -> None:
    """At most one input survives at a time, whatever the frame count."""
    live_counts: list[int] = []

    def produce() -> Iterator[object]:
        for index in range(4):
            gc.collect()
            live_counts.append(
                sum(1 for obj in gc.get_objects() if isinstance(obj, AssociationFrameInput))
            )
            yield frame_input(index, time_ns=FRAME_TIMES[index])

    SERVICE.run(
        make_request(frames=produce(), pose_policy=LookupPolicy.interpolated()),  # type: ignore[arg-type]
        sink=_Releasing(),
    )

    # Ao produzir o frame K, no máximo um input está vivo: o K-1, ainda preso à variável de
    # laço do serviço. Nunca K deles.
    assert live_counts == [0, 1, 1, 1]


def test_a_duplicate_frame_is_still_refused_without_a_second_pass() -> None:
    def produce() -> Iterator[object]:
        yield frame_input(0)
        yield frame_input(0)

    with pytest.raises(AssociationInputError, match="twice"):
        SERVICE.run(make_request(frames=produce()), sink=_Collecting())  # type: ignore[arg-type]


def test_a_frame_missing_a_declared_channel_is_refused_as_it_arrives() -> None:
    def produce() -> Iterator[object]:
        yield frame_input(0, channels=[NATIVE])
        yield dataclasses.replace(frame_input(1, channels=[NATIVE]), dense_maps={})

    with pytest.raises(AssociationInputError, match="dino-native"):
        SERVICE.run(
            make_request(frames=produce(), channels=[NATIVE]),  # type: ignore[arg-type]
            sink=_Collecting(),
        )


# --- The writer transaction is a sink --------------------------------------------------------


def test_a_run_streamed_through_the_writer_reads_back_complete(tmp_path: Path) -> None:
    request = make_request()

    with _writer(tmp_path).transaction() as run:
        outcome = SERVICE.run(request, sink=run)
        manifest = run.finalize(outcome)

    reader = SensorAssociationRunReader(tmp_path / "association")
    assert reader.verify_integrity() == []
    assert manifest.frame_count == 2
    assert manifest.observation_count == len(list(reader.observations()))
    assert reader.read_record("metrics/summary.json")["frame_count"] == 2


def test_the_streamed_frames_are_persisted_in_request_order(tmp_path: Path) -> None:
    with _writer(tmp_path).transaction() as run:
        run.finalize(SERVICE.run(make_request(), sink=run))

    reader = SensorAssociationRunReader(tmp_path / "association")
    projections = reader.read_records("outputs/projection-records.jsonl")
    assert [record["source_observation_id"] for record in projections] == [
        str(frame_id(0)),
        str(frame_id(1)),
    ]


def test_the_transaction_releases_each_frame_as_it_persists_it(tmp_path: Path) -> None:
    seen: list[weakref.ref[object]] = []

    class _Watching:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def accept(self, frame: FrameAssociation) -> None:
            self._inner.accept(frame)  # type: ignore[attr-defined]
            seen.append(weakref.ref(frame.resolution))

    with _writer(tmp_path).transaction() as run:
        outcome = SERVICE.run(
            make_request(frames=_many_frames(), pose_policy=LookupPolicy.interpolated()),  # type: ignore[arg-type]
            sink=_Watching(run),
        )
        run.finalize(outcome)

    gc.collect()
    assert len(seen) == 4
    assert sum(1 for ref in seen if ref() is not None) == 0


# --- An interrupted run is never a published artifact ----------------------------------------


def test_a_failure_part_way_through_publishes_nothing(tmp_path: Path) -> None:
    class _Failing:
        def __init__(self, inner: object) -> None:
            self._inner = inner
            self._accepted = 0

        def accept(self, frame: FrameAssociation) -> None:
            self._accepted += 1
            if self._accepted == 2:
                raise RuntimeError("the frame could not be persisted")
            self._inner.accept(frame)  # type: ignore[attr-defined]

    with (
        pytest.raises(RuntimeError, match="could not be persisted"),
        _writer(tmp_path).transaction() as run,
    ):
        SERVICE.run(make_request(), sink=_Failing(run))

    assert not (tmp_path / "association").exists()
    assert [p.name for p in tmp_path.iterdir() if not p.name.startswith(".")] == []


def test_a_transaction_left_without_finalizing_publishes_nothing(tmp_path: Path) -> None:
    with _writer(tmp_path).transaction() as run:
        SERVICE.run(make_request(), sink=run)

    assert not (tmp_path / "association").exists()


def test_finalizing_twice_is_refused(tmp_path: Path) -> None:
    with _writer(tmp_path).transaction() as run:
        outcome = SERVICE.run(make_request(), sink=run)
        run.finalize(outcome)
        with pytest.raises(RunArtifactError, match="finalized"):
            run.finalize(outcome)


def test_an_outcome_that_counts_other_frames_than_the_sink_received_is_refused(
    tmp_path: Path,
) -> None:
    import dataclasses

    with _writer(tmp_path).transaction() as run:
        outcome = SERVICE.run(make_request(), sink=run)
        with pytest.raises(RunArtifactError, match="frame"):
            run.finalize(dataclasses.replace(outcome, frame_count=99))


# --- Timing is measured, never contractual ---------------------------------------------------


def test_per_frame_timings_are_reported_only_when_the_caller_measured(tmp_path: Path) -> None:
    with _writer(tmp_path).transaction() as run:
        run.finalize(SERVICE.run(make_request(), sink=run), runtime_s=1.5)

    runtime = json.loads((tmp_path / "association" / "metrics" / "runtime.json").read_text())
    assert runtime["runtime_s"] == 1.5
    assert [f["source_observation_id"] for f in runtime["frames"]] == [
        str(frame_id(0)),
        str(frame_id(1)),
    ]
    assert all(f["candidate_query_seconds"] >= 0.0 for f in runtime["frames"])


def test_without_a_measurement_the_run_writes_no_timing_file(tmp_path: Path) -> None:
    with _writer(tmp_path).transaction() as run:
        run.finalize(SERVICE.run(make_request(), sink=run))

    assert not (tmp_path / "association" / "metrics" / "runtime.json").exists()
