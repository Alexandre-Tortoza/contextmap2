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
            when it shared a clock domain with at least one anchor but was
            never the closest candidate within ``tolerance_nanoseconds``.
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
    exists within ``config.tolerance_nanoseconds``. A candidate may be selected
    by more than one anchor (selection does not consume events); an event is
    only reported as dropped when no anchor ever selects it.

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
    selected_ids: dict[str, set[str]] = {name: set() for name in other_modalities}
    processing_observations: list[ProcessingObservation] = []
    decisions: list[SynchronizationDecision] = []
    for frame_index, anchor in enumerate(anchors):
        associations: dict[str, ModalityAssociation] = {
            config.reference_modality: ModalityAssociation(observation=anchor, offset_nanoseconds=0)
        }
        for modality in other_modalities:
            best = _closest_within_tolerance(
                anchor, by_modality[modality], config.tolerance_nanoseconds
            )
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
                        status=_missing_status(anchor, by_modality[modality]),
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


def _closest_within_tolerance(
    anchor: SourceObservation,
    candidates: Sequence[SourceObservation],
    tolerance_nanoseconds: int,
) -> tuple[SourceObservation, int] | None:
    anchor_clock_id = anchor.timestamp.clock_id
    anchor_nanoseconds = anchor.timestamp.total_nanoseconds()

    best: tuple[SourceObservation, int] | None = None
    best_abs_offset: int | None = None
    for candidate in candidates:
        if candidate.timestamp.clock_id != anchor_clock_id:
            continue
        offset_nanoseconds = candidate.timestamp.total_nanoseconds() - anchor_nanoseconds
        abs_offset = abs(offset_nanoseconds)
        if abs_offset > tolerance_nanoseconds:
            continue
        if (
            best_abs_offset is None
            or abs_offset < best_abs_offset
            or (
                abs_offset == best_abs_offset
                and best is not None
                and str(candidate.observation_id) < str(best[0].observation_id)
            )
        ):
            best = (candidate, offset_nanoseconds)
            best_abs_offset = abs_offset

    return best


def _missing_status(
    anchor: SourceObservation,
    candidates: Sequence[SourceObservation],
) -> str:
    """Classify why one anchor/modality association has no selected candidate."""
    if not candidates:
        return "no_candidate"
    if not any(item.timestamp.clock_id == anchor.timestamp.clock_id for item in candidates):
        return "clock_id_mismatch"
    return "outside_tolerance"
