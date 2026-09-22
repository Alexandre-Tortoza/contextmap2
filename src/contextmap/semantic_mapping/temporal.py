"""The temporal state of an entity.

Even in a mostly static map an entity needs temporal provenance, so a consumer can tell when and
how often it was observed. Physical observations (frames) and inference results are counted apart,
as in Semantic Fusion: several inference runs over one frame are correlated, never independent
views. ``first_seen`` and ``last_seen`` are derived from the exact contributing frames and can be
reproduced from the observation history, which is a compact index of one small entry per frame.

Tracking moving objects is a different problem and is deliberately absent: there is no velocity,
no trajectory, no re-identification after movement and no prediction, and the lifecycle never
infers motion, disappearance or destruction. Missing or contradictory temporal evidence is an
explicit error, never a fabricated timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import PhysicalObservationGroup
from contextmap.semantic_mapping._checks import require_present
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import TimeBounds

TEMPORAL_SUMMARY_RULE_ID = "physical-observation-temporal-summary-v1"
"""Versioned identity of the rule that derives the temporal state from physical observations."""


class TemporalEvidenceError(ValueError):
    """Raised when the temporal evidence of an entity is missing, duplicated or inconsistent."""


class EntityLifecycle(Enum):
    """A conservative, explicitly defined status of an entity.

    The baseline assigns only ``OBSERVED``. The other values are contract-level statuses a later,
    documented policy may assign; none of them is inferred here, and none says anything about
    motion, disappearance or destruction.

    Attributes:
        OBSERVED: At least one physical observation within the selected evidence contributed to
            the entity. It asserts nothing about persistence or current presence.
        STALE: The latest observation is older than a horizon a policy declared, relative to a
            reference time that policy declared.
        UNCERTAIN: A policy judged the observations too inconsistent to say the entity exists.
    """

    OBSERVED = "observed"
    STALE = "stale"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, kw_only=True)
class ObservationRef:
    """One physical observation in the history of an entity.

    Attributes:
        physical_observation_id: The physical frame.
        acquisition_timestamp: When the frame was acquired.
        inference_result_count: Inference results over that frame; at least one.
    """

    physical_observation_id: SourceObservationId
    acquisition_timestamp: SourceTimestamp
    inference_result_count: int

    def __post_init__(self) -> None:
        """Validate the identity and that the frame was interpreted.

        Raises:
            ValueError: If the identity is empty or no inference result was produced.
        """
        require_present(self, "physical_observation_id")
        if self.inference_result_count < 1:
            raise ValueError("inference_result_count must be at least 1")


@dataclass(frozen=True, kw_only=True)
class TemporalProvenance:
    """How the temporal state was derived.

    Attributes:
        rule_id: Versioned rule that derived it.
        input_order_chronological: Whether the selected evidence arrived in chronological order.
            ``False`` records that it did not, so a non-monotonic selection is explicit instead
            of silently reordered.
    """

    rule_id: str
    input_order_chronological: bool

    def __post_init__(self) -> None:
        """Require the rule.

        Raises:
            ValueError: If the rule identity is empty.
        """
        require_present(self, "rule_id")


@dataclass(frozen=True, kw_only=True)
class EntityTemporalState:
    """When and how often an entity was observed.

    Attributes:
        first_seen: Acquisition time of the earliest physical observation that contributed.
        last_seen: Acquisition time of the latest one; not before ``first_seen``.
        physical_observation_count: Distinct physical frames that contributed.
        inference_result_count: Inference results over those frames, correlated within a frame;
            never fewer than the frames.
        observation_refs: The history index: one entry per physical frame, in chronological
            order and unique. ``first_seen``, ``last_seen`` and both counts must agree with it,
            so they are reproducible from the exact contributing evidence.
        provenance: How the state was derived.
        lifecycle: A conservative status, or ``None`` when none is asserted.
    """

    first_seen: SourceTimestamp
    last_seen: SourceTimestamp
    physical_observation_count: int
    inference_result_count: int
    observation_refs: tuple[ObservationRef, ...]
    provenance: TemporalProvenance
    lifecycle: EntityLifecycle | None = None

    def __post_init__(self) -> None:
        """Validate that the interval, the counts and the history agree.

        Raises:
            ValueError: If the timestamps use different clock domains, ``last_seen`` precedes
                ``first_seen``, there is no physical observation, there are fewer inference
                results than physical observations, the history is not chronological and
                unique, or the interval or the counts disagree with the history.
        """
        if self.first_seen.clock_id != self.last_seen.clock_id:
            raise ValueError(
                f"first_seen and last_seen must share one clock domain, got "
                f"{self.first_seen.clock_id!r} and {self.last_seen.clock_id!r}"
            )
        if self.last_seen.total_nanoseconds() < self.first_seen.total_nanoseconds():
            raise ValueError("last_seen must not precede first_seen")
        if self.physical_observation_count < 1:
            raise ValueError("physical_observation_count must be at least 1")
        if self.inference_result_count < self.physical_observation_count:
            raise ValueError(
                f"inference_result_count {self.inference_result_count} cannot be lower than "
                f"physical_observation_count {self.physical_observation_count}: every "
                f"physical observation was interpreted at least once"
            )
        self._require_history_matches()

    @property
    def time_bounds(self) -> TimeBounds:
        """The closed interval from ``first_seen`` to ``last_seen``."""
        return TimeBounds(start=self.first_seen, end=self.last_seen)

    def _require_history_matches(self) -> None:
        refs = self.observation_refs
        if not refs:
            raise ValueError("observation_refs must not be empty: the history cannot be invented")
        keys = [
            (item.acquisition_timestamp.total_nanoseconds(), item.physical_observation_id)
            for item in refs
        ]
        if any(left >= right for left, right in pairwise(keys)):
            raise ValueError("observation_refs must be sorted chronologically and unique")
        if len({item.physical_observation_id for item in refs}) != len(refs):
            raise ValueError("observation_refs must not repeat a physical observation")
        if any(item.acquisition_timestamp.clock_id != self.first_seen.clock_id for item in refs):
            raise ValueError("observation_refs must share the clock domain of the interval")
        if (
            refs[0].acquisition_timestamp != self.first_seen
            or refs[-1].acquisition_timestamp != self.last_seen
        ):
            raise ValueError("first_seen and last_seen must be those of the first and last frame")
        if self.physical_observation_count != len(refs):
            raise ValueError(
                f"physical_observation_count {self.physical_observation_count} must equal the "
                f"{len(refs)} frames of the history"
            )
        total = sum(item.inference_result_count for item in refs)
        if self.inference_result_count != total:
            raise ValueError(
                f"inference_result_count {self.inference_result_count} must equal the {total} "
                f"inference results of the history"
            )


def summarize_temporal_state(
    groups: Sequence[PhysicalObservationGroup],
    *,
    lifecycle: EntityLifecycle | None = EntityLifecycle.OBSERVED,
) -> EntityTemporalState:
    """Derive the temporal state of an entity from the physical observations that contributed.

    Nothing is fabricated: ``first_seen`` and ``last_seen`` are the earliest and latest frame of
    the evidence, the physical observations and the inference results are counted apart, and an
    input that is not in chronological order is recorded as such.

    Args:
        groups: The physical observations of the fused evidence, in the order received.
        lifecycle: The status to assert; the baseline asserts only that the entity was observed.

    Returns:
        The temporal state, with its history index.

    Raises:
        TemporalEvidenceError: If there is no physical observation, one is repeated, or the
            timestamps span more than one clock domain.
    """
    if not groups:
        raise TemporalEvidenceError(
            "no physical observation contributed: first_seen and last_seen cannot be derived"
        )
    identities = [group.physical_observation_id for group in groups]
    repeated = sorted({item for item in identities if identities.count(item) > 1})
    if repeated:
        raise TemporalEvidenceError(f"physical observations are repeated: {repeated!r}")
    clocks = {group.acquisition_timestamp.clock_id for group in groups}
    if len(clocks) != 1:
        raise TemporalEvidenceError(
            f"the physical observations span more than one clock domain: {sorted(clocks)!r}"
        )
    refs = tuple(
        sorted(
            (
                ObservationRef(
                    physical_observation_id=group.physical_observation_id,
                    acquisition_timestamp=group.acquisition_timestamp,
                    inference_result_count=group.inference_result_count,
                )
                for group in groups
            ),
            key=lambda item: (
                item.acquisition_timestamp.total_nanoseconds(),
                item.physical_observation_id,
            ),
        )
    )
    return EntityTemporalState(
        first_seen=refs[0].acquisition_timestamp,
        last_seen=refs[-1].acquisition_timestamp,
        physical_observation_count=len(refs),
        inference_result_count=sum(item.inference_result_count for item in refs),
        observation_refs=refs,
        provenance=TemporalProvenance(
            rule_id=TEMPORAL_SUMMARY_RULE_ID,
            input_order_chronological=[group.physical_observation_id for group in groups]
            == [item.physical_observation_id for item in refs],
        ),
        lifecycle=lifecycle,
    )
