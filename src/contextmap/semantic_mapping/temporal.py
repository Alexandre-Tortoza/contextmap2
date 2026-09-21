"""The temporal state of an entity.

Even in a mostly static map an entity needs temporal provenance, so a consumer can tell when
and how often it was observed. Physical observations (frames) and inference results are counted
apart, as in Semantic Fusion: several inference runs over one frame are correlated, never
independent views. Tracking moving objects is a different problem and is deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.shared import SourceTimestamp


@dataclass(frozen=True, kw_only=True)
class EntityTemporalState:
    """When and how often an entity was observed.

    Attributes:
        first_seen: Acquisition time of the earliest physical observation that contributed.
        last_seen: Acquisition time of the latest one; not before ``first_seen``.
        physical_observation_count: Distinct physical frames that contributed.
        inference_result_count: Inference results over those frames, correlated within a frame;
            never fewer than the frames.
    """

    first_seen: SourceTimestamp
    last_seen: SourceTimestamp
    physical_observation_count: int
    inference_result_count: int

    def __post_init__(self) -> None:
        """Validate that the interval is ordered in one clock and the counts are coherent.

        Raises:
            ValueError: If the timestamps use different clock domains, ``last_seen`` precedes
                ``first_seen``, there is no physical observation, or there are fewer inference
                results than physical observations.
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
