"""Deterministic grouping of source observations into processing observations.

Source streams (RGB, LiDAR, IMU, external pose) rarely share a sampling
rate or arrive perfectly aligned. This module groups normalized
:data:`~contextmap.ingestion.models.SourceObservation` events around a
reference modality's events, producing an auditable, deterministic index
that later stages can replay without reopening source-specific APIs. See
``src/contextmap/ingestion/docs/synchronization.md`` for the policy
definition, the clock-domain limitation of v0, and worked examples.

Only one policy is implemented in v0: nearest-within-tolerance, matching
each reference-modality event with the closest same-clock-domain candidate
from every other modality, when one exists inside the configured tolerance
window. No interpolation is performed; a synchronization decision is either
"this candidate, at this offset" or "no candidate", and both are recorded.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contextmap.ingestion.models import MODALITY_NAMES, SourceObservation, observation_modality

_POLICY_NAME = "nearest_within_tolerance"


@dataclass(frozen=True)
class SynchronizationConfig:
    """Configuration for :func:`synchronize`.

    Attributes:
        reference_modality: Modality whose events anchor each processing
            observation. Must be one of
            :data:`~contextmap.ingestion.models.MODALITY_NAMES`.
        tolerance_nanoseconds: Maximum accepted absolute offset, in nanoseconds,
            between a candidate event's normalized timestamp and the
            anchor's normalized timestamp.
    """

    reference_modality: str
    tolerance_nanoseconds: int

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If ``reference_modality`` is not a known modality,
                or ``tolerance_nanoseconds`` is negative.
        """
        if self.reference_modality not in MODALITY_NAMES:
            raise ValueError(f"unknown reference_modality: {self.reference_modality!r}")
        if self.tolerance_nanoseconds < 0:
            raise ValueError(
                f"tolerance_nanoseconds must be >= 0, got {self.tolerance_nanoseconds}"
            )


@dataclass(frozen=True)
class ModalityAssociation:
    """One modality's contribution to a :class:`ProcessingObservation`.

    Attributes:
        observation: The selected source observation for this modality, or
            ``None`` when no candidate satisfied the tolerance window. A
            ``None`` here is an explicit "no match", never a silently
            dropped association.
        offset_nanoseconds: Signed exact offset, in nanoseconds, of ``observation``'s
            normalized timestamp relative to the anchor
            (``observation - anchor``). ``None`` when ``observation`` is
            ``None``.
    """

    observation: SourceObservation | None
    offset_nanoseconds: int | None


@dataclass(frozen=True)
class ProcessingObservation:
    """A deterministic grouping of source observations around one anchor event.

    Attributes:
        frame_index: Zero-based position of this group within the
            synchronized sequence, deterministic for a given input and
            :class:`SynchronizationConfig`.
        anchor: The reference-modality observation this group is centered
            on.
        associations: One :class:`ModalityAssociation` per modality in
            :data:`~contextmap.ingestion.models.MODALITY_NAMES`, including
            the reference modality (whose association always resolves to
            ``anchor`` at zero offset).
        policy: Identity of the synchronization policy that produced this
            group.
        config: The configuration used to produce this group.
    """

    frame_index: int
    anchor: SourceObservation
    associations: Mapping[str, ModalityAssociation]
    policy: str
    config: SynchronizationConfig


@dataclass(frozen=True)
class DroppedEvent:
    """A source event that did not join any processing observation.

    Attributes:
        observation: The event that was not selected.
        reason: ``"clock_id_mismatch"`` when the event never shared a clock
            domain with any anchor event, or ``"no_anchor_within_tolerance"``
            when it shared a clock domain with at least one anchor but no
            anchor selected it: it was outside every anchor's
            ``tolerance_nanoseconds``, another candidate was closer, or it
            tied for the closest and lost the ``observation_id`` tie-break.
            The reason therefore does not claim the event was outside the
            tolerance.
    """

    observation: SourceObservation
    reason: str


@dataclass(frozen=True)
class SynchronizationDecision:
    """One auditable association decision for an anchor and modality."""

    frame_index: int
    anchor_observation_id: str
    modality: str
    selected_observation_id: str | None
    offset_nanoseconds: int | None
    status: str


@dataclass(frozen=True)
class SynchronizationDiagnostics:
    """Auditable record of synchronization decisions and dropped events.

    Attributes:
        dropped_events: Every non-reference-modality event that did not
            join any processing observation, with why.
        decisions: One decision per anchor and non-reference modality,
            including successful and unsuccessful associations.
    """

    dropped_events: Sequence[DroppedEvent]
    decisions: Sequence[SynchronizationDecision]


def synchronize(
    observations: Sequence[SourceObservation],
    *,
    config: SynchronizationConfig,
) -> tuple[Sequence[ProcessingObservation], SynchronizationDiagnostics]:
    """Group source observations into processing observations.

    Events are grouped around ``config.reference_modality``'s events, in
    ascending normalized-timestamp order; ties are broken by
    ``observation_id`` for determinism. For every other modality, the
    closest event sharing the anchor's clock domain is selected, when one
    exists within ``config.tolerance_nanoseconds``. Candidates equally close
    to the anchor (the same absolute offset, before or after it, or equal
    timestamps) are decided by the lexicographically smallest
    ``observation_id``. A candidate may be selected by more than one anchor
    (selection does not consume events); an event is only reported as
    dropped when no anchor ever selects it, so a candidate that only ever
    lost a tie is dropped with ``"no_anchor_within_tolerance"``, the same
    reason as one outside the tolerance.

    Only events sharing exactly the anchor's ``timestamp.clock_id`` are
    considered comparable; see module docs for this v0 limitation.

    Args:
        observations: Source observations to group, in any order.
        config: Synchronization policy configuration.

    Returns:
        The deterministic list of processing observations, in
        ``frame_index`` order, and diagnostics for events that were never
        selected.
    """
    by_modality: dict[str, list[SourceObservation]] = {name: [] for name in MODALITY_NAMES}
    for observation in observations:
        by_modality[observation_modality(observation)].append(observation)

    anchors = sorted(
        by_modality[config.reference_modality],
        key=lambda observation: (
            observation.timestamp.total_nanoseconds(),
            str(observation.observation_id),
        ),
    )

    other_modalities = sorted(MODALITY_NAMES - {config.reference_modality})
    indexes = {name: _CandidateIndex(by_modality[name]) for name in other_modalities}
    selected_ids: dict[str, set[str]] = {name: set() for name in other_modalities}
    processing_observations: list[ProcessingObservation] = []
    decisions: list[SynchronizationDecision] = []
    for frame_index, anchor in enumerate(anchors):
        associations: dict[str, ModalityAssociation] = {
            config.reference_modality: ModalityAssociation(observation=anchor, offset_nanoseconds=0)
        }
        for modality in other_modalities:
            best = indexes[modality].closest_within_tolerance(anchor, config.tolerance_nanoseconds)
            if best is not None:
                candidate, offset_nanoseconds = best
                selected_ids[modality].add(str(candidate.observation_id))
                associations[modality] = ModalityAssociation(
                    observation=candidate, offset_nanoseconds=offset_nanoseconds
                )
                decisions.append(
                    SynchronizationDecision(
                        frame_index=frame_index,
                        anchor_observation_id=str(anchor.observation_id),
                        modality=modality,
                        selected_observation_id=str(candidate.observation_id),
                        offset_nanoseconds=offset_nanoseconds,
                        status="matched",
                    )
                )
            else:
                associations[modality] = ModalityAssociation(
                    observation=None, offset_nanoseconds=None
                )
                decisions.append(
                    SynchronizationDecision(
                        frame_index=frame_index,
                        anchor_observation_id=str(anchor.observation_id),
                        modality=modality,
                        selected_observation_id=None,
                        offset_nanoseconds=None,
                        status=indexes[modality].missing_status(anchor),
                    )
                )

        processing_observations.append(
            ProcessingObservation(
                frame_index=frame_index,
                anchor=anchor,
                associations=associations,
                policy=_POLICY_NAME,
                config=config,
            )
        )

    anchor_clock_ids = {anchor.timestamp.clock_id for anchor in anchors}
    dropped_events: list[DroppedEvent] = []
    for modality in other_modalities:
        for observation in by_modality[modality]:
            if str(observation.observation_id) in selected_ids[modality]:
                continue
            reason = (
                "no_anchor_within_tolerance"
                if observation.timestamp.clock_id in anchor_clock_ids
                else "clock_id_mismatch"
            )
            dropped_events.append(DroppedEvent(observation=observation, reason=reason))

    return processing_observations, SynchronizationDiagnostics(
        dropped_events=tuple(dropped_events),
        decisions=tuple(decisions),
    )


class _CandidateIndex:
    """The candidates of one modality, sorted once per clock domain for bisection lookup.

    Within a clock domain, candidates are ordered by ``(total_nanoseconds, observation_id)``, so
    the first entry of a run of equal timestamps is the one the ``observation_id`` tie-break
    selects. A lookup then examines at most two such runs instead of every candidate.
    """

    def __init__(self, candidates: Sequence[SourceObservation]) -> None:
        by_clock: dict[str, list[tuple[int, str, SourceObservation]]] = {}
        for candidate in candidates:
            by_clock.setdefault(candidate.timestamp.clock_id, []).append(
                (
                    candidate.timestamp.total_nanoseconds(),
                    str(candidate.observation_id),
                    candidate,
                )
            )
        self._is_empty = not candidates
        self._by_clock = {
            clock_id: sorted(entries, key=lambda entry: (entry[0], entry[1]))
            for clock_id, entries in by_clock.items()
        }
        self._nanoseconds = {
            clock_id: [entry[0] for entry in entries]
            for clock_id, entries in self._by_clock.items()
        }

    def closest_within_tolerance(
        self, anchor: SourceObservation, tolerance_nanoseconds: int
    ) -> tuple[SourceObservation, int] | None:
        """Return the closest candidate on the anchor's clock, and its signed offset.

        Ties on the absolute offset go to the smallest ``observation_id``; a candidate farther
        than ``tolerance_nanoseconds`` is never selected.
        """
        clock_id = anchor.timestamp.clock_id
        entries = self._by_clock.get(clock_id)
        if entries is None:
            return None
        nanoseconds = self._nanoseconds[clock_id]
        anchor_nanoseconds = anchor.timestamp.total_nanoseconds()
        after = bisect_left(nanoseconds, anchor_nanoseconds)
        # O primeiro de cada sequência de timestamps iguais é o de menor observation_id.
        neighbours = []
        if after < len(entries):
            neighbours.append(entries[after])
        if after > 0:
            neighbours.append(entries[bisect_left(nanoseconds, nanoseconds[after - 1])])
        stamp, _, candidate = min(
            neighbours, key=lambda entry: (abs(entry[0] - anchor_nanoseconds), entry[1])
        )
        offset_nanoseconds = stamp - anchor_nanoseconds
        if abs(offset_nanoseconds) > tolerance_nanoseconds:
            return None
        return candidate, offset_nanoseconds

    def missing_status(self, anchor: SourceObservation) -> str:
        """Classify why one anchor/modality association has no selected candidate."""
        if self._is_empty:
            return "no_candidate"
        if anchor.timestamp.clock_id not in self._by_clock:
            return "clock_id_mismatch"
        return "outside_tolerance"
