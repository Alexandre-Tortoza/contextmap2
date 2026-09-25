"""Tests for the dataset-scoped timestamp normalization policy (issue #554)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from contextmap.ingestion.models import (
    FrameId,
    ImageEncoding,
    ImageObservation,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.timestamp_policy import (
    DEFAULT_TIMESTAMP_POLICY,
    ClockPlausibilityError,
    ConstantOffsetCorrection,
    TimestampPolicy,
    apply_timestamp_policy,
    decode_timestamp_policy,
    diagnose_source_clock,
    encode_timestamp_policy,
    validate_cross_source_clock_plausibility,
)
from contextmap.shared import SourceTimestamp

_CLOCK = "ros1_bag:datasets/corridor-02/corridor-02.bag:header"

# A 2001 header stamp that should have been 2026: exactly 25 years of drift, in seconds.
_TWENTY_FIVE_YEARS_SECONDS = 789_004_800


def _timestamp(seconds: int, nanoseconds: int = 0, *, clock_id: str = _CLOCK) -> SourceTimestamp:
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)


def _image(
    seconds: int,
    nanoseconds: int = 0,
    *,
    clock_id: str = _CLOCK,
    raw_metadata: dict[str, object] | None = None,
) -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId(f"camera-{seconds}-{nanoseconds:09d}"),
        sensor_id=SensorId("camera"),
        frame_id=FrameId("camera_optical"),
        timestamp=_timestamp(seconds, nanoseconds, clock_id=clock_id),
        provenance=SourceProvenance(
            source_type="ros1_bag",
            source_path="datasets/corridor-02/corridor-02.bag",
            source_topic="/camera/image_raw",
            raw_metadata=raw_metadata or {},
        ),
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00\x00\x00",
    )


class TestTimestampPolicy:
    def test_the_default_policy_applies_no_correction(self) -> None:
        assert DEFAULT_TIMESTAMP_POLICY.correction is None
        assert DEFAULT_TIMESTAMP_POLICY.event_clock == "header_stamp"
        assert DEFAULT_TIMESTAMP_POLICY.window_clock == "recording_time"

    def test_an_unknown_event_clock_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="event_clock"):
            TimestampPolicy(event_clock="system_time")

    def test_an_unknown_window_clock_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="window_clock"):
            TimestampPolicy(window_clock="header_stamp")


class TestConstantOffsetCorrection:
    def test_an_anchor_pair_must_be_given_together(self) -> None:
        with pytest.raises(ValueError, match="together"):
            ConstantOffsetCorrection(offset_nanoseconds=0, anchor_source_time=_timestamp(1))

    def test_from_anchors_derives_the_exact_nanosecond_offset(self) -> None:
        source = _timestamp(1000, 500)
        reference = _timestamp(1000 + _TWENTY_FIVE_YEARS_SECONDS, 500)
        correction = ConstantOffsetCorrection.from_anchors(
            source_time=source, reference_time=reference
        )
        assert correction.offset_nanoseconds == _TWENTY_FIVE_YEARS_SECONDS * 1_000_000_000
        assert correction.anchor_source_time == source
        assert correction.anchor_reference_time == reference

    def test_from_offset_seconds_rounds_to_the_nearest_nanosecond(self) -> None:
        correction = ConstantOffsetCorrection.from_offset_seconds(1.5)
        assert correction.offset_nanoseconds == 1_500_000_000
        assert correction.anchor_source_time is None


class TestApplyTimestampPolicy:
    def test_no_correction_returns_the_same_instance(self) -> None:
        observation = _image(1_700_000_000)
        assert apply_timestamp_policy(observation, DEFAULT_TIMESTAMP_POLICY) is observation

    def test_a_constant_offset_corrects_the_wrong_absolute_epoch_exactly(self) -> None:
        # Scenario 2: correct relative timing, wrong absolute epoch (the corridor-02 case #554
        # names explicitly): three observations one second apart in the wrong-epoch clock.
        wrong_epoch_start = 1_000_000_000  # somewhere in 2001, in the source's own clock
        raw = [_image(wrong_epoch_start + step) for step in range(3)]
        correction = ConstantOffsetCorrection.from_offset_seconds(_TWENTY_FIVE_YEARS_SECONDS)
        policy = TimestampPolicy(correction=correction)

        corrected = [apply_timestamp_policy(observation, policy) for observation in raw]

        assert [obs.timestamp.seconds for obs in corrected] == [
            wrong_epoch_start + _TWENTY_FIVE_YEARS_SECONDS + step for step in range(3)
        ]
        # Relative timing (the whole point of trusting this clock) is preserved exactly.
        deltas = [b.timestamp.seconds - a.timestamp.seconds for a, b in pairwise(corrected)]
        assert deltas == [1, 1]

    def test_the_raw_timestamp_is_preserved_in_provenance_never_silently_overwritten(self) -> None:
        observation = _image(1_000_000_000, 250)
        policy = TimestampPolicy(correction=ConstantOffsetCorrection.from_offset_seconds(10.0))

        corrected = apply_timestamp_policy(observation, policy)

        assert corrected.timestamp.seconds == 1_000_000_010
        assert corrected.timestamp.nanoseconds == 250
        assert corrected.provenance.raw_metadata["source_time_before_correction_seconds"] == (
            1_000_000_000
        )
        assert corrected.provenance.raw_metadata["source_time_before_correction_nanoseconds"] == 250
        # The clock domain identity itself is unchanged: only the value is normalized.
        assert corrected.timestamp.clock_id == observation.timestamp.clock_id

    def test_a_negative_offset_crossing_a_second_boundary_stays_a_valid_timestamp(self) -> None:
        observation = _image(100, 100)
        policy = TimestampPolicy(correction=ConstantOffsetCorrection.from_offset_seconds(-0.5))

        corrected = apply_timestamp_policy(observation, policy)

        # 100.0000001 - 0.5 = 99.5000001s
        assert corrected.timestamp.seconds == 99
        assert corrected.timestamp.nanoseconds == 500_000_100

    def test_another_dataset_processed_without_a_policy_does_not_inherit_the_correction(
        self,
    ) -> None:
        # Scenario 4: two independent requests, only one configured.
        corrected_policy = TimestampPolicy(
            correction=ConstantOffsetCorrection.from_offset_seconds(1_000.0)
        )
        observation = _image(1_000_000_000)

        untouched = apply_timestamp_policy(observation, DEFAULT_TIMESTAMP_POLICY)
        corrected = apply_timestamp_policy(observation, corrected_policy)

        assert untouched.timestamp.seconds == 1_000_000_000
        assert corrected.timestamp.seconds == 1_000_001_000


class TestDiagnoseSourceClock:
    def test_a_clean_monotonic_sequence_has_no_findings(self) -> None:
        observations = [_image(1000 + step) for step in range(5)]
        diagnostics = diagnose_source_clock(observations)
        assert diagnostics.non_monotonic_count == 0
        assert diagnostics.missing_source_timestamp_count == 0
        assert diagnostics.warnings() == ()

    def test_a_zero_timestamp_is_reported_as_missing(self) -> None:
        observations = [_image(1000), _image(0, 0), _image(1002)]
        diagnostics = diagnose_source_clock(observations)
        assert diagnostics.missing_source_timestamp_count == 1
        assert "missing (zero) source timestamp" in diagnostics.warnings()[0]

    def test_a_non_monotonic_reset_is_detected_and_actionable(self) -> None:
        # Scenario 6: the clock resets partway through.
        observations = [_image(1000), _image(1001), _image(500), _image(501)]
        diagnostics = diagnose_source_clock(observations)
        assert diagnostics.non_monotonic_count == 1
        messages = diagnostics.warnings()
        assert any("non-monotonic" in message for message in messages)

    def test_the_recording_minus_source_distribution_is_computed_when_available(self) -> None:
        offset_ns = 5 * 1_000_000_000
        observations = [
            _image(
                1000 + step,
                raw_metadata={
                    "bag_timestamp_nanoseconds": (1000 + step) * 1_000_000_000 + offset_ns
                },
            )
            for step in range(4)
        ]
        diagnostics = diagnose_source_clock(observations)
        assert diagnostics.recording_minus_source_seconds_mean == pytest.approx(5.0)
        assert diagnostics.recording_minus_source_seconds_stdev == pytest.approx(0.0)

    def test_the_residual_around_a_configured_offset_is_reported(self) -> None:
        # recording - source is consistently 5s, but the configured offset is 4.9s: a 0.1s
        # residual that a validator should be able to see.
        observations = [
            _image(
                1000 + step,
                raw_metadata={"bag_timestamp_nanoseconds": (1000 + step + 5) * 1_000_000_000},
            )
            for step in range(3)
        ]
        correction = ConstantOffsetCorrection.from_offset_seconds(4.9)
        diagnostics = diagnose_source_clock(observations, correction=correction)
        assert diagnostics.residual_seconds_mean == pytest.approx(0.1, abs=1e-6)
        assert diagnostics.residual_seconds_max_abs == pytest.approx(0.1, abs=1e-6)
        assert any("residual" in message for message in diagnostics.warnings())

    def test_diagnostics_read_the_raw_timestamp_even_after_correction_was_applied(self) -> None:
        # The function must give the same verdict whether called before or after
        # apply_timestamp_policy: it always diagnoses the raw clock, never the corrected one.
        observations = [_image(1000), _image(1001), _image(500)]
        policy = TimestampPolicy(correction=ConstantOffsetCorrection.from_offset_seconds(100.0))
        corrected = [apply_timestamp_policy(observation, policy) for observation in observations]

        before = diagnose_source_clock(observations)
        after = diagnose_source_clock(corrected)

        assert before.non_monotonic_count == after.non_monotonic_count == 1


class TestEncodeDecodeTimestampPolicy:
    def test_the_default_policy_round_trips(self) -> None:
        document = encode_timestamp_policy(DEFAULT_TIMESTAMP_POLICY)
        assert document == {
            "event_clock": "header_stamp",
            "window_clock": "recording_time",
            "correction": {"type": "none"},
        }
        assert decode_timestamp_policy(document) == DEFAULT_TIMESTAMP_POLICY

    def test_none_document_decodes_to_the_default_policy(self) -> None:
        assert decode_timestamp_policy(None) == DEFAULT_TIMESTAMP_POLICY

    def test_a_directly_supplied_offset_round_trips(self) -> None:
        policy = TimestampPolicy(correction=ConstantOffsetCorrection.from_offset_seconds(12.5))
        document = encode_timestamp_policy(policy)
        rebuilt = decode_timestamp_policy(document)
        assert rebuilt.correction is not None
        assert rebuilt.correction.offset_nanoseconds == 12_500_000_000
        assert rebuilt.correction.anchor_source_time is None

    def test_an_anchor_derived_offset_round_trips_with_its_anchors(self) -> None:
        source = _timestamp(1_000_000_000)
        reference = _timestamp(1_000_000_000 + _TWENTY_FIVE_YEARS_SECONDS)
        policy = TimestampPolicy(
            correction=ConstantOffsetCorrection.from_anchors(
                source_time=source, reference_time=reference
            )
        )

        document = encode_timestamp_policy(policy)
        rebuilt = decode_timestamp_policy(document)

        assert rebuilt.correction is not None
        assert rebuilt.correction.offset_nanoseconds == _TWENTY_FIVE_YEARS_SECONDS * 1_000_000_000
        assert rebuilt.correction.anchor_source_time == source
        assert rebuilt.correction.anchor_reference_time == reference

    def test_an_unsupported_correction_type_is_rejected(self) -> None:
        document = {
            "event_clock": "header_stamp",
            "window_clock": "recording_time",
            "correction": {"type": "affine"},
        }
        with pytest.raises(ValueError, match="affine"):
            decode_timestamp_policy(document)


class TestValidateCrossSourceClockPlausibility:
    """Issue #555: a matching ``clock_id`` string is an assertion, not evidence."""

    def test_overlapping_ranges_on_the_same_clock_pass(self) -> None:
        clock = "corridor-02-header"
        primary = [_image(100, clock_id=clock), _image(200, clock_id=clock)]
        auxiliary = [_image(150, clock_id=clock), _image(250, clock_id=clock)]

        validate_cross_source_clock_plausibility(primary, auxiliary)

    def test_auxiliary_range_contained_in_primary_range_passes(self) -> None:
        clock = "corridor-02-header"
        primary = [_image(0, clock_id=clock), _image(1000, clock_id=clock)]
        auxiliary = [_image(400, clock_id=clock), _image(600, clock_id=clock)]

        validate_cross_source_clock_plausibility(primary, auxiliary)

    def test_different_clock_ids_are_rejected(self) -> None:
        primary = [_image(100, clock_id="corridor-02-header")]
        auxiliary = [_image(100, clock_id="corridor-02-gt:tum-header-stamp")]

        with pytest.raises(ClockPlausibilityError, match="clock"):
            validate_cross_source_clock_plausibility(primary, auxiliary)

    def test_disjoint_ranges_on_the_same_clock_id_are_rejected(self) -> None:
        """The exact shape of bug #554 fixed: a matching clock_id label but a wrong epoch."""
        clock = "corridor-02-header"
        primary = [_image(100, clock_id=clock), _image(200, clock_id=clock)]
        auxiliary = [
            _image(_TWENTY_FIVE_YEARS_SECONDS + 100, clock_id=clock),
            _image(_TWENTY_FIVE_YEARS_SECONDS + 200, clock_id=clock),
        ]

        with pytest.raises(ClockPlausibilityError, match="implausible"):
            validate_cross_source_clock_plausibility(primary, auxiliary)

    def test_a_sequence_using_more_than_one_clock_id_is_rejected(self) -> None:
        primary = [_image(100, clock_id="corridor-02-header")]
        auxiliary = [
            _image(100, clock_id="corridor-02-gt:tum-header-stamp"),
            _image(200, clock_id="some-other-clock"),
        ]

        with pytest.raises(ClockPlausibilityError, match="clock_id"):
            validate_cross_source_clock_plausibility(primary, auxiliary)

    def test_an_empty_sequence_is_rejected(self) -> None:
        with pytest.raises(ClockPlausibilityError, match="empty"):
            validate_cross_source_clock_plausibility([], [_image(100)])
