"""Dataset-scoped timestamp normalization policy (issue #554).

Some recorded sources carry a source/header clock whose relative progression is usable for
synchronization even though its absolute epoch is wrong (e.g. a camera driver whose system clock
was never set, stamping every message somewhere in the year 2001). Silently repairing that from
the source type, or from a heuristic such as "the year looks wrong", would hide a dataset-specific
decision inside generic adapter code — exactly what ``docs/adapters.md`` and this module reject.

This module keeps four notions separate, matching the issue's required model:

- raw recording/container time — a bag's own per-message capture clock, already carried in
  ``SourceProvenance.raw_metadata["bag_timestamp_nanoseconds"]`` for the bag adapters, untouched
  by anything here;
- raw source/sensor timestamp — the value a
  :class:`~contextmap.ingestion.source_adapter.SourceAdapter` decoded onto
  :attr:`~contextmap.ingestion.models._SourceObservationBase.timestamp` before any policy in this
  module ran;
- normalized/derived event time — the same field, after :func:`apply_timestamp_policy` replaced it;
  the raw value it replaced is preserved in provenance, never dropped;
- the clock used to resolve timestamp-based sequence windows — always the source's own recording
  time today (:meth:`~contextmap.ingestion.source_adapter.SourceAdapterConfig.
  resolved_window_clock_id`), named explicitly by :attr:`TimestampPolicy.window_clock` so it
  travels with the rest of the policy in one auditable record instead of being implicit.

A policy is selected per dataset/source configuration (:class:`TimestampPolicy`, carried on
:class:`~contextmap.runtime.ingestion_service.IngestionRequest`), never inferred from the adapter
family: the default applies no correction at all, so an unconfigured or streaming source behaves
exactly as ingestion did before this module existed.

The only supported correction is a constant offset (:class:`ConstantOffsetCorrection`):
``event_time = source_time + offset``. No affine term, drift compensation, or online estimation —
see AGENTS.md's YAGNI rule and the issue itself, which explicitly rejects those without evidence a
constant offset is insufficient.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean, pstdev
from typing import Any

from contextmap.ingestion.models import SourceObservation
from contextmap.shared import SourceTimestamp

EVENT_CLOCK_HEADER_STAMP = "header_stamp"
"""The only supported event clock: the per-message timestamp an adapter already decodes."""

WINDOW_CLOCK_RECORDING_TIME = "recording_time"
"""The only supported window clock: a source's own recording-time clock (``docs/adapters.md``)."""

_KNOWN_EVENT_CLOCKS = frozenset({EVENT_CLOCK_HEADER_STAMP})
_KNOWN_WINDOW_CLOCKS = frozenset({WINDOW_CLOCK_RECORDING_TIME})

_RECORDING_TIME_METADATA_KEY = "bag_timestamp_nanoseconds"
_RAW_SOURCE_SECONDS_KEY = "source_time_before_correction_seconds"
_RAW_SOURCE_NANOSECONDS_KEY = "source_time_before_correction_nanoseconds"

_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, kw_only=True)
class ConstantOffsetCorrection:
    """A constant-offset correction from a source/header clock to normalized event time.

    ``event_time = source_time + offset``, computed with exact integer nanosecond arithmetic so a
    multi-year offset never accumulates floating-point drift.

    Attributes:
        offset_nanoseconds: The constant offset added to every source/header timestamp.
        anchor_source_time: The source-clock reading the offset was derived from, when derived
            from anchors via :meth:`from_anchors` rather than supplied directly. ``None`` when the
            offset was supplied directly.
        anchor_reference_time: The known-correct reference time paired with ``anchor_source_time``.
            ``None`` under the same condition.
    """

    offset_nanoseconds: int
    anchor_source_time: SourceTimestamp | None = None
    anchor_reference_time: SourceTimestamp | None = None

    def __post_init__(self) -> None:
        """Validate that the anchor pair is given together, or not at all.

        Raises:
            ValueError: If exactly one of ``anchor_source_time``/``anchor_reference_time`` is given.
        """
        has_source = self.anchor_source_time is not None
        has_reference = self.anchor_reference_time is not None
        if has_source != has_reference:
            raise ValueError("anchor_source_time and anchor_reference_time must be given together")

    @property
    def offset_seconds(self) -> float:
        """Return the offset as a float number of seconds, for display/config purposes only."""
        return self.offset_nanoseconds / _NANOSECONDS_PER_SECOND

    @classmethod
    def from_offset_seconds(cls, offset_seconds: float) -> ConstantOffsetCorrection:
        """Build a correction from a directly supplied offset, in seconds.

        Args:
            offset_seconds: The constant offset to add to every source/header timestamp.

        Returns:
            The correction, with no anchors recorded.
        """
        return cls(offset_nanoseconds=round(offset_seconds * _NANOSECONDS_PER_SECOND))

    @classmethod
    def from_anchors(
        cls, *, source_time: SourceTimestamp, reference_time: SourceTimestamp
    ) -> ConstantOffsetCorrection:
        """Derive a constant offset from one known-correct (source, reference) time pair.

        Args:
            source_time: A source/header timestamp known to be wrong (e.g. the wrong calendar
                epoch), read directly off the affected clock.
            reference_time: The known-correct time the same physical instant should map to.

        Returns:
            The correction; ``reference_time - source_time``, exactly, in nanoseconds.
        """
        offset_ns = reference_time.total_nanoseconds() - source_time.total_nanoseconds()
        return cls(
            offset_nanoseconds=offset_ns,
            anchor_source_time=source_time,
            anchor_reference_time=reference_time,
        )


@dataclass(frozen=True, kw_only=True)
class TimestampPolicy:
    """Dataset-scoped selection of clocks and correction ingestion applies (issue #554).

    The default is equivalent to no timestamp handling beyond what ingestion already did before
    this module existed: the header/source clock is the event clock, the source's own
    recording-time clock resolves windows, and no correction is applied.

    Attributes:
        event_clock: Name of the clock normalized event time is derived from.
            ``"header_stamp"`` (:data:`EVENT_CLOCK_HEADER_STAMP`) is the only supported value: every
            :data:`~contextmap.ingestion.models.SourceObservation` already carries exactly one
            per-message clock reading, decoded from the source's own header/native timestamp.
        window_clock: Name of the clock explicit temporal windows
            (:class:`~contextmap.ingestion.source_adapter.SourceWindow`) are expressed in.
            ``"recording_time"`` (:data:`WINDOW_CLOCK_RECORDING_TIME`) is the only supported value,
            matching :meth:`~contextmap.ingestion.source_adapter.SourceAdapterConfig.
            resolved_window_clock_id`. Naming it here, alongside ``event_clock`` and
            ``correction``, keeps the dataset's whole timestamp policy in one auditable record
            instead of leaving the window clock implicit.
        correction: The correction applied to every ``event_clock`` reading before it becomes an
            observation's canonical timestamp. ``None`` (the default) applies no correction at all.
    """

    event_clock: str = EVENT_CLOCK_HEADER_STAMP
    window_clock: str = WINDOW_CLOCK_RECORDING_TIME
    correction: ConstantOffsetCorrection | None = None

    def __post_init__(self) -> None:
        """Validate the named clocks are ones this module actually knows how to resolve.

        Raises:
            ValueError: If ``event_clock`` or ``window_clock`` names an unsupported clock.
        """
        if self.event_clock not in _KNOWN_EVENT_CLOCKS:
            raise ValueError(
                f"unknown event_clock {self.event_clock!r}; known: {sorted(_KNOWN_EVENT_CLOCKS)}"
            )
        if self.window_clock not in _KNOWN_WINDOW_CLOCKS:
            raise ValueError(
                f"unknown window_clock {self.window_clock!r}; known: {sorted(_KNOWN_WINDOW_CLOCKS)}"
            )


DEFAULT_TIMESTAMP_POLICY = TimestampPolicy()
"""The policy equivalent to no dataset-specific timestamp handling."""


def apply_timestamp_policy(
    observation: SourceObservation, policy: TimestampPolicy
) -> SourceObservation:
    """Apply a timestamp policy's correction to one observation.

    The source/header timestamp is preserved in ``observation.provenance.raw_metadata`` before
    being replaced, so a normalized event time never silently overwrites the raw value it was
    derived from. When ``policy.correction`` is ``None`` the observation is returned unchanged
    (the same instance, not a copy) — the default is byte-identical to ingestion without a policy.

    Args:
        observation: A canonical observation as decoded by a source adapter, in its own
            (uncorrected) clock.
        policy: The dataset's timestamp policy.

    Returns:
        The observation with its ``timestamp`` replaced by the corrected event time and the
        original preserved in provenance, or the same instance when no correction applies.
    """
    if policy.correction is None:
        return observation
    raw = observation.timestamp
    corrected_ns = raw.total_nanoseconds() + policy.correction.offset_nanoseconds
    corrected_seconds, corrected_nanoseconds = divmod(corrected_ns, _NANOSECONDS_PER_SECOND)
    corrected = SourceTimestamp(
        seconds=corrected_seconds, nanoseconds=corrected_nanoseconds, clock_id=raw.clock_id
    )
    provenance = replace(
        observation.provenance,
        raw_metadata={
            **observation.provenance.raw_metadata,
            _RAW_SOURCE_SECONDS_KEY: raw.seconds,
            _RAW_SOURCE_NANOSECONDS_KEY: raw.nanoseconds,
        },
    )
    return replace(observation, timestamp=corrected, provenance=provenance)


@dataclass(frozen=True, kw_only=True)
class TimestampCorrectionDiagnostics:
    """Diagnostics computed from raw, pre-correction observation timestamps (issue #554).

    None of these change what correction is applied: they exist to make an inappropriate
    constant-offset model visible instead of silently accepted.

    Attributes:
        observation_count: Observations the diagnostics were computed over.
        missing_source_timestamp_count: Observations whose source/header timestamp is the all-zero
            convention for "not set" (``seconds == 0 and nanoseconds == 0``).
        non_monotonic_count: Observations whose raw source timestamp is before the previous
            observation's, within the same ``clock_id``.
        recording_minus_source_seconds_mean: Mean of ``recording_time - source_time`` over
            observations where both are available. ``None`` when none are.
        recording_minus_source_seconds_stdev: Population standard deviation of the same
            distribution. ``None`` when fewer than two observations have both times.
        residual_seconds_mean: Mean of
            ``(recording_time - source_time) - correction.offset_seconds`` over the same
            observations, when a correction was given. ``None`` otherwise.
        residual_seconds_max_abs: Largest absolute value of the same residual. ``None`` otherwise.
    """

    observation_count: int
    missing_source_timestamp_count: int
    non_monotonic_count: int
    recording_minus_source_seconds_mean: float | None = None
    recording_minus_source_seconds_stdev: float | None = None
    residual_seconds_mean: float | None = None
    residual_seconds_max_abs: float | None = None

    def warnings(self) -> tuple[str, ...]:
        """Render these diagnostics as human-readable provenance warnings.

        Returns:
            Zero or more warning strings; empty only when nothing worth reporting was found.
        """
        messages: list[str] = []
        if self.missing_source_timestamp_count:
            messages.append(
                f"timestamp policy: {self.missing_source_timestamp_count} observation(s) with a "
                "missing (zero) source timestamp"
            )
        if self.non_monotonic_count:
            messages.append(
                f"timestamp policy: {self.non_monotonic_count} non-monotonic source timestamp "
                "step(s) detected on the raw (pre-correction) clock"
            )
        if self.recording_minus_source_seconds_mean is not None:
            stdev = self.recording_minus_source_seconds_stdev or 0.0
            messages.append(
                "timestamp policy: recording_time - source_time over "
                f"{self.observation_count} observation(s): mean="
                f"{self.recording_minus_source_seconds_mean:.6f}s stdev={stdev:.6f}s"
            )
        if self.residual_seconds_mean is not None:
            messages.append(
                "timestamp policy: residual around the configured constant_offset: mean="
                f"{self.residual_seconds_mean:.6f}s max_abs={self.residual_seconds_max_abs:.6f}s"
            )
        return tuple(messages)


def diagnose_source_clock(
    observations: Sequence[SourceObservation],
    *,
    correction: ConstantOffsetCorrection | None = None,
) -> TimestampCorrectionDiagnostics:
    """Compute constant-offset correction diagnostics from a sequence of observations.

    Reconstructs each observation's raw, pre-correction source timestamp from the provenance
    :func:`apply_timestamp_policy` stashes when a correction was already applied, falling back to
    ``observation.timestamp`` itself when it was not — so this can run either before or after
    :func:`apply_timestamp_policy`, over the same sequence, with the same result.

    Args:
        observations: Observations to diagnose, in source order.
        correction: The correction under consideration, to compute a residual against. ``None``
            computes only the clock's own diagnostics (monotonicity, missing timestamps, the
            recording/source distribution), with no residual.

    Returns:
        The diagnostics.
    """
    missing = 0
    non_monotonic = 0
    # Monotonicidade em nanossegundos inteiros: o float64 colapsa diferenças de ~238 ns em
    # epochs reais. As deltas continuam em float: são distribuição, não ordem.
    last_by_clock: dict[str, int] = {}
    deltas: list[float] = []
    for observation in observations:
        raw = _raw_source_timestamp(observation)
        if raw.seconds == 0 and raw.nanoseconds == 0:
            missing += 1
        nanoseconds = raw.total_nanoseconds()
        previous = last_by_clock.get(raw.clock_id)
        if previous is not None and nanoseconds < previous:
            non_monotonic += 1
        last_by_clock[raw.clock_id] = nanoseconds
        recording_ns = observation.provenance.raw_metadata.get(_RECORDING_TIME_METADATA_KEY)
        if isinstance(recording_ns, int):
            deltas.append(recording_ns / _NANOSECONDS_PER_SECOND - raw.to_float_seconds())

    mean_delta = fmean(deltas) if deltas else None
    stdev_delta = pstdev(deltas) if len(deltas) >= 2 else None
    residual_mean: float | None = None
    residual_max_abs: float | None = None
    if correction is not None and deltas:
        residuals = [delta - correction.offset_seconds for delta in deltas]
        residual_mean = fmean(residuals)
        residual_max_abs = max(abs(value) for value in residuals)

    return TimestampCorrectionDiagnostics(
        observation_count=len(observations),
        missing_source_timestamp_count=missing,
        non_monotonic_count=non_monotonic,
        recording_minus_source_seconds_mean=mean_delta,
        recording_minus_source_seconds_stdev=stdev_delta,
        residual_seconds_mean=residual_mean,
        residual_seconds_max_abs=residual_max_abs,
    )


def _raw_source_timestamp(observation: SourceObservation) -> SourceTimestamp:
    """Return an observation's timestamp as it was before any correction was applied."""
    raw_metadata = observation.provenance.raw_metadata
    seconds = raw_metadata.get(_RAW_SOURCE_SECONDS_KEY)
    nanoseconds = raw_metadata.get(_RAW_SOURCE_NANOSECONDS_KEY)
    if isinstance(seconds, int) and isinstance(nanoseconds, int):
        return SourceTimestamp(
            seconds=seconds, nanoseconds=nanoseconds, clock_id=observation.timestamp.clock_id
        )
    return observation.timestamp


class ClockPlausibilityError(ValueError):
    """Raised when two observation sequences' clock relationship fails a plausibility check.

    Issue #555.
    """


def validate_cross_source_clock_plausibility(
    primary: Sequence[SourceObservation], auxiliary: Sequence[SourceObservation]
) -> None:
    """Reject an implausible clock relationship before two observation sequences are merged.

    Two sources declaring the same ``timestamp_clock_id`` string is an *assertion*, not
    evidence (issue #555): before a caller -- for example, a runtime executor merging a main
    sequence with an auxiliary pose sequence -- combines their observations, it must at least
    rule out the wrong-epoch defect issue #554 fixed, never trust the label alone. This checks
    each sequence's *published* (already timestamp-policy-corrected) ``observation.timestamp``,
    not the pre-correction raw value :func:`diagnose_source_clock` reconstructs -- merging
    happens on the corrected clock, so that is the relationship that must be checked.

    This is a plausibility/sanity check, not proof of temporal alignment: a matching
    ``clock_id`` with overlapping ranges rules out the specific wrong-epoch defect this
    function was written against, but it does not prove the two sources share the same time
    base. Two independent recordings that happen to overlap in the same epoch, or two sources
    with a real, undetected offset smaller than the sequence's own duration, both pass this
    check without actually being aligned. A caller must not treat a passing call as proof the
    merge is temporally correct, only as evidence the specific defect this function targets is
    absent.

    Args:
        primary: Observations of the main sequence.
        auxiliary: Observations of the auxiliary sequence to be merged with it.

    Raises:
        ClockPlausibilityError: If either sequence is empty (nothing to check against), if
            either sequence itself spans more than one ``clock_id``, if the two sequences use
            different clocks, or if their timestamp ranges in the shared clock do not overlap --
            a matching ``clock_id`` string with disjoint ranges is exactly the wrong-epoch defect
            issue #554 fixed, and is never accepted as plausible.
    """
    if not primary or not auxiliary:
        raise ClockPlausibilityError(
            "cannot check clock plausibility against an empty observation sequence"
        )
    primary_clocks = {observation.timestamp.clock_id for observation in primary}
    auxiliary_clocks = {observation.timestamp.clock_id for observation in auxiliary}
    if len(primary_clocks) != 1 or len(auxiliary_clocks) != 1:
        raise ClockPlausibilityError(
            "cannot check clock plausibility: each sequence must use exactly one clock_id "
            f"(primary uses {sorted(primary_clocks)!r}, "
            f"auxiliary uses {sorted(auxiliary_clocks)!r})"
        )
    (primary_clock,) = primary_clocks
    (auxiliary_clock,) = auxiliary_clocks
    if primary_clock != auxiliary_clock:
        raise ClockPlausibilityError(
            f"primary sequence uses clock {primary_clock!r} but auxiliary sequence uses "
            f"{auxiliary_clock!r} -- merging across different clocks is not supported yet; a "
            "declared offset/transform between them must be explicit, never inferred from naming"
        )
    primary_seconds = [observation.timestamp.to_float_seconds() for observation in primary]
    auxiliary_seconds = [observation.timestamp.to_float_seconds() for observation in auxiliary]
    primary_range = (min(primary_seconds), max(primary_seconds))
    auxiliary_range = (min(auxiliary_seconds), max(auxiliary_seconds))
    if primary_range[1] < auxiliary_range[0] or auxiliary_range[1] < primary_range[0]:
        raise ClockPlausibilityError(
            f"primary sequence spans [{primary_range[0]:.6f}, {primary_range[1]:.6f}]s on clock "
            f"{primary_clock!r} but auxiliary sequence spans [{auxiliary_range[0]:.6f}, "
            f"{auxiliary_range[1]:.6f}]s with no overlap -- sharing a clock_id does not mean the "
            "two sequences were recorded over the same physical timeframe; the relationship is "
            "implausible and the two artifacts cannot be safely merged"
        )


def encode_timestamp_policy(policy: TimestampPolicy) -> dict[str, Any]:
    """Encode a timestamp policy into the JSON-compatible form persisted in provenance.

    Args:
        policy: The policy to encode.

    Returns:
        A dict matching the ``time:`` configuration shape from issue #554.
    """
    correction: dict[str, Any]
    if policy.correction is None:
        correction = {"type": "none"}
    else:
        correction = {
            "type": "constant_offset",
            "offset_seconds": policy.correction.offset_seconds,
            "anchor_source_time": _encode_timestamp(policy.correction.anchor_source_time),
            "anchor_reference_time": _encode_timestamp(policy.correction.anchor_reference_time),
        }
    return {
        "event_clock": policy.event_clock,
        "window_clock": policy.window_clock,
        "correction": correction,
    }


def decode_timestamp_policy(document: dict[str, Any] | None) -> TimestampPolicy:
    """Decode a timestamp policy from its JSON-compatible form.

    Args:
        document: A dict as produced by :func:`encode_timestamp_policy`, or ``None`` for a source
            with no configured policy.

    Returns:
        The decoded policy, or :data:`DEFAULT_TIMESTAMP_POLICY` when ``document`` is ``None``.

    Raises:
        ValueError: If ``document["correction"]["type"]`` is not a supported correction type.
    """
    if document is None:
        return DEFAULT_TIMESTAMP_POLICY
    correction_document = document["correction"]
    correction_type = correction_document["type"]
    correction: ConstantOffsetCorrection | None
    if correction_type == "none":
        correction = None
    elif correction_type == "constant_offset":
        anchor_source = _decode_timestamp(correction_document["anchor_source_time"])
        anchor_reference = _decode_timestamp(correction_document["anchor_reference_time"])
        if anchor_source is not None and anchor_reference is not None:
            correction = ConstantOffsetCorrection.from_anchors(
                source_time=anchor_source, reference_time=anchor_reference
            )
        else:
            correction = ConstantOffsetCorrection.from_offset_seconds(
                correction_document["offset_seconds"]
            )
    else:
        raise ValueError(f"unsupported timestamp correction type: {correction_type!r}")
    return TimestampPolicy(
        event_clock=document["event_clock"],
        window_clock=document["window_clock"],
        correction=correction,
    )


def _encode_timestamp(timestamp: SourceTimestamp | None) -> dict[str, Any] | None:
    return None if timestamp is None else timestamp.to_record()


def _decode_timestamp(record: dict[str, Any] | None) -> SourceTimestamp | None:
    return None if record is None else SourceTimestamp.from_record(record)
