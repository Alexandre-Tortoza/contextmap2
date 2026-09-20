"""Explicit motion-correction (deskew) state of geometric observations.

A LiDAR scan is acquired over a non-zero interval while the platform may be
moving. Treating every point as captured at one timestamp can deform the map,
and a scan must never be assumed deskewed merely because a particular estimator
was used. This module makes the state explicit for every scan:

* ``RAW``: declared as not corrected for platform motion;
* ``CORRECTED``: corrected, with evidence of who corrected it, from which
  trajectory, with which configuration and which payload;
* ``UNKNOWN``: nothing is declared. This is the default and is never inferred
  from a sensor, a source type or an estimator.

Timing is never fabricated: a record states the acquisition interval it knows
or that per-point timing exists, and a correction that names neither is
rejected. Whether a scan may enter a map in each state is decided by an explicit
:class:`MotionCorrectionPolicy` recorded with the run. No corrector exists in
this repository yet, so there is no "correct when supported" disposition.
See ``src/contextmap/geometric_mapping/docs/motion-correction.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from contextmap.ingestion import LidarObservation, SourceObservationId
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import TrajectoryId

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class MotionCorrectionState(Enum):
    """Whether a scan was corrected for platform motion.

    Attributes:
        RAW: Declared as not corrected.
        CORRECTED: Corrected, with supporting evidence.
        UNKNOWN: No declaration; never inferred.
    """

    RAW = "raw"
    CORRECTED = "corrected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True)
class MotionCorrectionEvidence:
    """What supports a claim that a scan was motion corrected.

    Attributes:
        producer: Identity of the backend or implementation that corrected the scan.
        trajectory_id: Trajectory (state source) the correction used.
        payload_hash: ``"sha256:<64 hex digits>"`` of the corrected payload.
        configuration_fingerprint: Hash of the correction configuration, when it has one.
    """

    producer: str
    trajectory_id: TrajectoryId
    payload_hash: str
    configuration_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Validate that the evidence is complete and well formed.

        Raises:
            ValueError: If the producer is empty or ``payload_hash`` is not a SHA-256.
        """
        if not self.producer:
            raise ValueError("motion correction evidence producer must not be empty")
        if not _SHA256_PATTERN.match(self.payload_hash):
            raise ValueError(
                f"payload_hash must be 'sha256:<64 hex digits>', got {self.payload_hash!r}"
            )


@dataclass(frozen=True, kw_only=True)
class MotionCorrectionRecord:
    """The declared correction state of one scan, with what supports it.

    Attributes:
        observation_id: The physical observation this record describes.
        state: Raw, corrected or unknown.
        acquisition_start: Start of the acquisition interval, when known.
        acquisition_end: End of the acquisition interval, when known.
        per_point_timing_available: Whether the payload carries per-point timing.
        evidence: Support for a ``CORRECTED`` claim; absent otherwise.
    """

    observation_id: SourceObservationId
    state: MotionCorrectionState
    acquisition_start: SourceTimestamp | None = None
    acquisition_end: SourceTimestamp | None = None
    per_point_timing_available: bool = False
    evidence: MotionCorrectionEvidence | None = None

    def __post_init__(self) -> None:
        """Validate the interval, the evidence and the timing.

        Raises:
            ValueError: If the interval is incomplete, spans clocks or is inverted;
                evidence accompanies a scan that is not corrected or is missing
                from a corrected one; or a corrected scan states no timing.
        """
        start, end = self.acquisition_start, self.acquisition_end
        if (start is None) != (end is None):
            raise ValueError("acquisition_start and acquisition_end must be given both or neither")
        if start is not None and end is not None:
            if start.clock_id != end.clock_id:
                raise ValueError("acquisition_start and acquisition_end must share one clock")
            if end.total_nanoseconds() < start.total_nanoseconds():
                raise ValueError("acquisition interval end precedes its start")
        if self.state is MotionCorrectionState.CORRECTED:
            if self.evidence is None:
                raise ValueError("a corrected scan needs motion correction evidence")
            if start is None and not self.per_point_timing_available:
                raise ValueError(
                    "a corrected scan must state the timing it used (an acquisition interval or "
                    "per-point timing); timing is never fabricated"
                )
        elif self.evidence is not None:
            raise ValueError("motion correction evidence is only valid for a corrected scan")


def unknown_motion_correction(observation: LidarObservation) -> MotionCorrectionRecord:
    """Declare nothing about a scan: the default, never inferred.

    Args:
        observation: The scan.

    Returns:
        An ``UNKNOWN`` record, whatever the scan's sensor, source or estimator.
    """
    return MotionCorrectionRecord(
        observation_id=observation.observation_id, state=MotionCorrectionState.UNKNOWN
    )


def declared_raw_motion_correction(observation: LidarObservation) -> MotionCorrectionRecord:
    """Declare a scan as not corrected, for sources known to deliver raw scans.

    Args:
        observation: The scan.

    Returns:
        A ``RAW`` record without correction evidence.
    """
    return MotionCorrectionRecord(
        observation_id=observation.observation_id, state=MotionCorrectionState.RAW
    )


def verify_motion_correction(
    record: MotionCorrectionRecord, observation: LidarObservation
) -> list[str]:
    """Check that a record agrees with the scan it describes.

    Args:
        record: The declared state.
        observation: The scan.

    Returns:
        Human-readable problems; empty means the record is consistent with the scan.
    """
    problems: list[str] = []
    if record.observation_id != observation.observation_id:
        problems.append(
            f"record describes observation {record.observation_id!r} but the scan is "
            f"{observation.observation_id!r}"
        )
    start, end = record.acquisition_start, record.acquisition_end
    if start is not None and end is not None:
        stamp = observation.timestamp
        if stamp.clock_id != start.clock_id:
            problems.append(
                f"the scan clock {stamp.clock_id!r} differs from the acquisition interval clock "
                f"{start.clock_id!r}"
            )
        elif not start.total_nanoseconds() <= stamp.total_nanoseconds() <= end.total_nanoseconds():
            problems.append("the scan timestamp lies outside its acquisition interval")
    return problems


class ScanDisposition(Enum):
    """What a mapping run does with a scan in a given correction state.

    Attributes:
        ACCEPT: Use the scan.
        WARN: Use the scan and record a warning.
        REJECT: Leave the scan out and record why.
    """

    ACCEPT = "accept"
    WARN = "warn"
    REJECT = "reject"


@dataclass(frozen=True, kw_only=True)
class MotionCorrectionPolicy:
    """Explicit rule for scans that are not known to be corrected.

    A corrected scan is always accepted. Both uncorrected states are stated, with
    no default, so a run never lets a raw or unknown scan in by omission.

    Attributes:
        raw: Disposition of a scan declared raw.
        unknown: Disposition of a scan whose state is not declared.
    """

    raw: ScanDisposition
    unknown: ScanDisposition

    def disposition_for(self, state: MotionCorrectionState) -> ScanDisposition:
        """Return the disposition of a scan in ``state``."""
        if state is MotionCorrectionState.RAW:
            return self.raw
        if state is MotionCorrectionState.UNKNOWN:
            return self.unknown
        return ScanDisposition.ACCEPT


@dataclass(frozen=True, kw_only=True)
class MotionCorrectionVerdict:
    """The outcome of applying a policy to one scan.

    Attributes:
        record: The declared state that was judged.
        disposition: What the run does with the scan.
        message: Why, when the scan is warned about or rejected; ``None`` when accepted.
    """

    record: MotionCorrectionRecord
    disposition: ScanDisposition
    message: str | None


def apply_motion_correction_policy(
    record: MotionCorrectionRecord, policy: MotionCorrectionPolicy
) -> MotionCorrectionVerdict:
    """Apply a policy to a scan's declared state.

    Args:
        record: The scan's declared state.
        policy: The run's policy.

    Returns:
        The verdict, naming the scan and the state when it warns or rejects.
    """
    disposition = policy.disposition_for(record.state)
    message = None
    if disposition is not ScanDisposition.ACCEPT:
        message = (
            f"scan {record.observation_id} is {record.state.value} for platform motion; "
            f"the policy says {disposition.value}"
        )
    return MotionCorrectionVerdict(record=record, disposition=disposition, message=message)
