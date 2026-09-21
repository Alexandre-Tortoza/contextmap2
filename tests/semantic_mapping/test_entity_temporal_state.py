import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from mapping_builders import (
    make_entity,
    make_evidence_links,
    make_temporal_state,
    timestamp,
)
from mapping_fusion import ClaimSpec, FusionRun, View, entity_from_outcome, fuse, write_fusion_run

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import PhysicalObservationGroup
from contextmap.semantic_mapping import (
    TEMPORAL_SUMMARY_RULE_ID,
    EntityLifecycle,
    EntityTemporalState,
    ObservationRef,
    TemporalEvidenceError,
    TemporalProvenance,
    summarize_temporal_state,
)
from contextmap.semantic_mapping.serialization import decode_entity, encode_entity
from contextmap.sensor_association import SpatialObservationId
from contextmap.visual_perception import PerceptionResultId, PerceptionRunId


def _group(
    frame: str, seconds: int, *, runs: Sequence[str] = ("run-a",), clock: str = "fixture:header"
) -> PhysicalObservationGroup:
    return PhysicalObservationGroup(
        physical_observation_id=SourceObservationId(frame),
        acquisition_timestamp=timestamp(seconds, clock_id=clock),
        spatial_observation_ids=tuple(
            sorted(SpatialObservationId(f"spatial--{run}--{frame}") for run in runs)
        ),
        perception_result_ids=tuple(sorted(PerceptionResultId(f"{run}--{frame}") for run in runs)),
        perception_run_ids=tuple(sorted(PerceptionRunId(run) for run in runs)),
    )


class TestTemporalSummary:
    def test_one_physical_observation_with_several_inference_runs_counts_one_frame(self) -> None:
        state = summarize_temporal_state(
            [_group("frame-0120", 10, runs=("run-a", "run-b", "run-c"))]
        )

        assert state.physical_observation_count == 1
        assert state.inference_result_count == 3
        assert state.first_seen == state.last_seen == timestamp(10)

    def test_several_physical_observations_over_time_span_the_interval(self) -> None:
        state = summarize_temporal_state(
            [_group("frame-0120", 10), _group("frame-0121", 11), _group("frame-0125", 15)]
        )

        assert state.first_seen == timestamp(10)
        assert state.last_seen == timestamp(15)
        assert state.physical_observation_count == 3
        assert state.inference_result_count == 3
        assert state.time_bounds.duration_ns == 5_000_000_000

    def test_first_and_last_seen_are_reproducible_from_the_exact_evidence(self) -> None:
        groups = [_group("frame-0121", 11), _group("frame-0120", 10, runs=("run-a", "run-b"))]

        state = summarize_temporal_state(groups)

        stamps = [group.acquisition_timestamp for group in groups]
        assert state.first_seen == min(stamps, key=lambda item: item.total_nanoseconds())
        assert state.last_seen == max(stamps, key=lambda item: item.total_nanoseconds())
        assert [item.physical_observation_id for item in state.observation_refs] == [
            "frame-0120",
            "frame-0121",
        ]
        assert [item.inference_result_count for item in state.observation_refs] == [2, 1]

    def test_a_non_chronological_selection_is_recorded_not_silently_reordered(self) -> None:
        chronological = summarize_temporal_state(
            [_group("frame-0120", 10), _group("frame-0121", 11)]
        )
        shuffled = summarize_temporal_state([_group("frame-0121", 11), _group("frame-0120", 10)])

        assert chronological.provenance.input_order_chronological is True
        assert shuffled.provenance.input_order_chronological is False
        assert shuffled.observation_refs == chronological.observation_refs
        assert shuffled.provenance.rule_id == TEMPORAL_SUMMARY_RULE_ID

    def test_frames_acquired_at_the_same_instant_are_ordered_by_identity(self) -> None:
        state = summarize_temporal_state([_group("frame-b", 10), _group("frame-a", 10)])

        assert [item.physical_observation_id for item in state.observation_refs] == [
            "frame-a",
            "frame-b",
        ]

    def test_the_baseline_asserts_only_that_the_entity_was_observed(self) -> None:
        assert (
            summarize_temporal_state([_group("frame-0120", 10)]).lifecycle
            is EntityLifecycle.OBSERVED
        )
        assert (
            summarize_temporal_state([_group("frame-0120", 10)], lifecycle=None).lifecycle is None
        )

    def test_it_is_deterministic(self) -> None:
        groups = [_group("frame-0120", 10), _group("frame-0121", 11)]

        assert summarize_temporal_state(groups) == summarize_temporal_state(groups)

    def test_no_evidence_is_an_explicit_error_and_never_a_fabricated_timestamp(self) -> None:
        with pytest.raises(TemporalEvidenceError, match="cannot be derived"):
            summarize_temporal_state([])

    def test_a_duplicated_observation_reference_is_refused(self) -> None:
        with pytest.raises(TemporalEvidenceError, match="repeated"):
            summarize_temporal_state([_group("frame-0120", 10), _group("frame-0120", 11)])

    def test_mixed_clock_domains_are_refused(self) -> None:
        with pytest.raises(TemporalEvidenceError, match="clock domain"):
            summarize_temporal_state(
                [_group("frame-0120", 10), _group("frame-0121", 11, clock="other:clock")]
            )


class TestFromRealFusedEvidence:
    @pytest.fixture
    def run(self, tmp_path: Path) -> FusionRun:
        return write_fusion_run(tmp_path)

    def test_it_agrees_with_the_temporal_summary_and_counts_of_the_fused_evidence(
        self, run: FusionRun
    ) -> None:
        for outcome in run.outcomes:
            evidence = outcome.evidence

            state = summarize_temporal_state(evidence.physical_observation_groups)

            assert state.time_bounds == evidence.temporal_summary
            assert state.physical_observation_count == evidence.physical_observation_count
            assert state.inference_result_count == evidence.inference_result_count

    def test_repeated_inference_over_one_frame_is_not_counted_as_more_frames(
        self, run: FusionRun
    ) -> None:
        contradiction = entity_from_outcome(run.outcomes[0], run).temporal_state

        # run-a e run-b interpretam frame-0120; run-a interpreta frame-0121.
        assert contradiction.physical_observation_count == 2
        assert contradiction.inference_result_count == 3
        assert [item.inference_result_count for item in contradiction.observation_refs] == [2, 1]

    def test_a_single_view_over_one_frame_is_one_and_one(self) -> None:
        _, evidence = fuse([View("run-a", "frame-0120", (ClaimSpec("pallet"),))])

        state = summarize_temporal_state(evidence.physical_observation_groups)

        assert (state.physical_observation_count, state.inference_result_count) == (1, 1)


class TestTemporalStateContract:
    def test_first_and_last_seen_are_ordered(self) -> None:
        state = make_temporal_state()

        with pytest.raises(ValueError, match="must not precede"):
            dataclasses.replace(state, first_seen=timestamp(20))

    def test_the_interval_stays_in_one_clock_domain(self) -> None:
        state = make_temporal_state()

        with pytest.raises(ValueError, match="clock domain"):
            dataclasses.replace(state, last_seen=timestamp(12, clock_id="other:clock"))

    def test_an_entity_was_observed_at_least_once(self) -> None:
        state = make_temporal_state()

        with pytest.raises(ValueError, match="at least 1"):
            dataclasses.replace(state, physical_observation_count=0)

    def test_inference_cannot_be_counted_below_the_frames_it_interpreted(self) -> None:
        state = make_temporal_state()

        with pytest.raises(ValueError, match="cannot be lower"):
            dataclasses.replace(state, inference_result_count=1)

    def test_the_history_cannot_be_invented(self) -> None:
        state = make_temporal_state()

        with pytest.raises(ValueError, match="must not be empty"):
            dataclasses.replace(state, observation_refs=())

    def test_the_history_is_chronological_and_never_repeats_a_frame(self) -> None:
        first, second = make_temporal_state().observation_refs

        with pytest.raises(ValueError, match="sorted chronologically and unique"):
            dataclasses.replace(make_temporal_state(), observation_refs=(second, first))
        with pytest.raises(ValueError, match="sorted chronologically and unique"):
            dataclasses.replace(make_temporal_state(), observation_refs=(first, first))

    def test_the_history_repeating_a_frame_at_another_time_is_refused(self) -> None:
        first, second = make_temporal_state().observation_refs
        again = dataclasses.replace(second, physical_observation_id=first.physical_observation_id)

        with pytest.raises(ValueError, match="must not repeat"):
            dataclasses.replace(make_temporal_state(), observation_refs=(first, again))

    def test_the_interval_must_be_that_of_the_first_and_last_frame(self) -> None:
        with pytest.raises(ValueError, match="first and last frame"):
            dataclasses.replace(make_temporal_state(), first_seen=timestamp(11))

    def test_the_counts_must_agree_with_the_history(self) -> None:
        state = make_temporal_state()

        with pytest.raises(ValueError, match="must equal the 2 frames"):
            dataclasses.replace(state, physical_observation_count=3, inference_result_count=4)
        with pytest.raises(ValueError, match="must equal the 3 inference results"):
            dataclasses.replace(state, inference_result_count=5)

    def test_every_frame_of_the_history_was_interpreted(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ObservationRef(
                physical_observation_id=SourceObservationId("frame-0120"),
                acquisition_timestamp=timestamp(10),
                inference_result_count=0,
            )

    def test_the_history_shares_the_clock_of_the_interval(self) -> None:
        first, second = make_temporal_state().observation_refs
        foreign = dataclasses.replace(second, acquisition_timestamp=timestamp(12, clock_id="other"))

        with pytest.raises(ValueError, match="clock domain"):
            dataclasses.replace(make_temporal_state(), observation_refs=(first, foreign))

    def test_a_rule_is_required(self) -> None:
        with pytest.raises(ValueError, match="rule_id"):
            TemporalProvenance(rule_id=" ", input_order_chronological=True)

    def test_the_history_and_the_evidence_links_must_list_the_same_frames(self) -> None:
        with pytest.raises(ValueError, match="same physical observations"):
            make_entity(evidence=make_evidence_links(physical=("frame-0120", "frame-0122")))

    def test_no_dynamic_tracking_is_part_of_the_contract(self) -> None:
        fields = {field.name for field in dataclasses.fields(EntityTemporalState)}

        assert not {"velocity", "trajectory", "track_id", "predicted_pose"} & fields


class TestSerialization:
    def test_the_temporal_state_survives_persistence_with_history_and_lifecycle(self) -> None:
        entity = make_entity(temporal_state=make_temporal_state())

        decoded = decode_entity(json.loads(json.dumps(encode_entity(entity))))

        assert decoded.temporal_state == entity.temporal_state
        assert decoded.temporal_state.lifecycle is EntityLifecycle.OBSERVED

    def test_a_missing_lifecycle_and_a_non_chronological_input_survive_as_recorded(self) -> None:
        state = summarize_temporal_state(
            [_group("frame-0121", 11), _group("frame-0120", 10)], lifecycle=None
        )
        entity = make_entity(temporal_state=state)

        decoded = decode_entity(json.loads(json.dumps(encode_entity(entity)))).temporal_state

        assert decoded.lifecycle is None
        assert decoded.provenance.input_order_chronological is False

    def test_a_record_with_an_inconsistent_history_is_refused(self) -> None:
        record = encode_entity(make_entity())
        record["temporal_state"]["observation_refs"][0]["inference_result_count"] = 9

        with pytest.raises(ValueError, match="must equal the"):
            decode_entity(record)
