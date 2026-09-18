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
        tolerance_seconds: Maximum accepted absolute offset, in seconds,
            between a candidate event's normalized timestamp and the
            anchor's normalized timestamp.
    """

    reference_modality: str
    tolerance_seconds: float

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If ``reference_modality`` is not a known modality,
                or ``tolerance_seconds`` is negative.
        """
        if self.reference_modality not in MODALITY_NAMES:
            raise ValueError(f"unknown reference_modality: {self.reference_modality!r}")
        if self.tolerance_seconds < 0:
            raise ValueError(f"tolerance_seconds must be >= 0, got {self.tolerance_seconds}")


@dataclass(frozen=True)
class ModalityAssociation:
    """One modality's contribution to a :class:`ProcessingObservation`.

    Attributes:
        observation: The selected source observation for this modality, or
            ``None`` when no candidate satisfied the tolerance window. A
            ``None`` here is an explicit "no match", never a silently
            dropped association.
        offset_seconds: Signed offset, in seconds, of ``observation``'s
            normalized timestamp relative to the anchor
            (``observation - anchor``). ``None`` when ``observation`` is
            ``None``.
    """

    observation: SourceObservation | None
    offset_seconds: float | None


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
            never the closest candidate within ``tolerance_seconds``.
    """

    observation: SourceObservation
    reason: str


@dataclass(frozen=True)
class SynchronizationDiagnostics:
    """Auditable record of synchronization decisions that were not a match.

    Attributes:
        dropped_events: Every non-reference-modality event that did not
            join any processing observation, with why.
    """

    dropped_events: Sequence[DroppedEvent]


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
    exists within ``config.tolerance_seconds``. A candidate may be selected
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
            observation.timestamp.to_float_seconds(),
            str(observation.observation_id),
        ),
    )

    other_modalities = sorted(MODALITY_NAMES - {config.reference_modality})
    selected_ids: dict[str, set[str]] = {name: set() for name in other_modalities}
    seen_clock_ids: dict[str, set[str]] = {name: set() for name in other_modalities}

    processing_observations: list[ProcessingObservation] = []
    for frame_index, anchor in enumerate(anchors):
        associations: dict[str, ModalityAssociation] = {
            config.reference_modality: ModalityAssociation(observation=anchor, offset_seconds=0.0)
        }
        for modality in other_modalities:
            best = _closest_within_tolerance(
                anchor, by_modality[modality], config.tolerance_seconds
            )
            if best is not None:
                candidate, offset_seconds = best
                selected_ids[modality].add(str(candidate.observation_id))
                seen_clock_ids[modality].add(candidate.timestamp.clock_id)
                associations[modality] = ModalityAssociation(
                    observation=candidate, offset_seconds=offset_seconds
                )
            else:
                associations[modality] = ModalityAssociation(observation=None, offset_seconds=None)

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

    return processing_observations, SynchronizationDiagnostics(dropped_events=tuple(dropped_events))


def _closest_within_tolerance(
    anchor: SourceObservation,
    candidates: Sequence[SourceObservation],
    tolerance_seconds: float,
) -> tuple[SourceObservation, float] | None:
    anchor_clock_id = anchor.timestamp.clock_id
    anchor_seconds = anchor.timestamp.to_float_seconds()

    best: tuple[SourceObservation, float] | None = None
    best_abs_offset = float("inf")
    for candidate in candidates:
        if candidate.timestamp.clock_id != anchor_clock_id:
            continue
        offset_seconds = candidate.timestamp.to_float_seconds() - anchor_seconds
        abs_offset = abs(offset_seconds)
        if abs_offset > tolerance_seconds:
            continue
        if abs_offset < best_abs_offset or (
            abs_offset == best_abs_offset
            and best is not None
            and str(candidate.observation_id) < str(best[0].observation_id)
        ):
            best = (candidate, offset_seconds)
            best_abs_offset = abs_offset

    return best
