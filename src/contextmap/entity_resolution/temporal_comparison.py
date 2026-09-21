"""Temporal compatibility of two entities, from their observation histories.

An entity keeps a small history of *physical* observations (frames) and, apart from it, how many
inference results interpreted them: several inference runs over one frame are correlated, never
independent views. This channel exposes that history as a measurement and one explicit rule:

* how the two observation intervals relate (their overlap or their gap, in nanoseconds);
* how many physical observations each entity has, how many inference results, how many frames they
  share and how many distinct frames the pair spans;
* ``static-scene-co-observation``: under a **declared** static-scene assumption, two entities seen
  in the same frame are two objects, because one object cannot appear twice in one frame. The
  assumption is a policy declaration, off unless the profile justifies it, because an
  over-segmented frame can split one object in two regions.

Solution 1 assumes a static or quasi-static scene, so time never infers motion, re-identification
or disappearance: a large gap between the intervals is measured, not judged. Timestamps of
different clock domains are not comparable, so the channel is unavailable rather than negative;
missing timestamps cannot occur, because an entity without a history is not a valid entity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from contextmap.entity_resolution.channels import (
    EvidenceStatus,
    Finding,
    TemporalEvidence,
    TemporalMeasurement,
    Unavailability,
    UnavailableReason,
)
from contextmap.entity_resolution.models import PolicyRef, reference_order
from contextmap.semantic_mapping import Entity

TEMPORAL_COMPATIBILITY_POLICY_ID = "entity-temporal-compatibility-v1"
"""Versioned identity of the temporal measurement and rule described in this module."""


@dataclass(frozen=True, kw_only=True)
class TemporalCompatibilityPolicy:
    """Explicit declaration of the temporal rule.

    Attributes:
        static_scene: Whether the scene is declared static, so that two entities seen in the same
            physical observation are read as two objects. There is no default: it is an assumption
            a profile must justify.
    """

    static_scene: bool

    def fingerprint(self) -> str:
        """Hash the policy identity and declaration, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {"policy_id": TEMPORAL_COMPATIBILITY_POLICY_ID, "static_scene": self.static_scene},
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on the evidence."""
        return PolicyRef(
            policy_id=TEMPORAL_COMPATIBILITY_POLICY_ID, configuration_fingerprint=self.fingerprint()
        )


def compare_temporal(
    entity_a: Entity, entity_b: Entity, policy: TemporalCompatibilityPolicy
) -> TemporalEvidence:
    """Compare when and from which frames two entities were observed.

    The result does not depend on the order of the arguments: the pair is put in canonical order
    and the measurements refer to it.

    Args:
        entity_a: One entity.
        entity_b: The other entity.
        policy: The static-scene declaration.

    Returns:
        The temporal evidence, or an unavailable channel when the histories use different clock
        domains.
    """
    first, second = sorted((entity_a, entity_b), key=lambda item: reference_order(item.reference))
    state_a, state_b = first.temporal_state, second.temporal_state
    clock_a, clock_b = state_a.first_seen.clock_id, state_b.first_seen.clock_id
    if clock_a != clock_b:
        return TemporalEvidence(
            policy=policy.ref(),
            unavailable=Unavailability(
                reason=UnavailableReason.INCOMPATIBLE_DOMAIN,
                detail=(
                    f"the observation histories use different clock domains, {clock_a!r} and "
                    f"{clock_b!r}, so their timestamps are not comparable"
                ),
            ),
        )
    start_a, end_a = state_a.first_seen.total_nanoseconds(), state_a.last_seen.total_nanoseconds()
    start_b, end_b = state_b.first_seen.total_nanoseconds(), state_b.last_seen.total_nanoseconds()
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
    gap = max(0, max(start_a, start_b) - min(end_a, end_b))
    frames_a = {item.physical_observation_id for item in state_a.observation_refs}
    frames_b = {item.physical_observation_id for item in state_b.observation_refs}
    shared = len(frames_a & frames_b)
    measurement = TemporalMeasurement(
        clock_id=clock_a,
        interval_overlap_ns=overlap,
        interval_gap_ns=gap,
        physical_observation_count_a=state_a.physical_observation_count,
        physical_observation_count_b=state_b.physical_observation_count,
        inference_result_count_a=state_a.inference_result_count,
        inference_result_count_b=state_b.inference_result_count,
        shared_physical_observation_count=shared,
        union_physical_observation_count=len(frames_a | frames_b),
    )
    return TemporalEvidence(
        policy=policy.ref(),
        measurement=measurement,
        findings=(_co_observation(shared, policy),),
    )


def _co_observation(shared: int, policy: TemporalCompatibilityPolicy) -> Finding:
    if not policy.static_scene:
        return Finding(
            rule_id="static-scene-co-observation",
            status=EvidenceStatus.NEUTRAL,
            detail=(
                f"{shared} shared physical observation(s); a static scene was not declared, so "
                f"seeing both entities in one frame is not read as a conflict"
            ),
            metric="shared_physical_observation_count",
            observed=float(shared),
        )
    conflicting = shared >= 1
    return Finding(
        rule_id="static-scene-co-observation",
        status=EvidenceStatus.CONFLICTING if conflicting else EvidenceStatus.NEUTRAL,
        detail=(
            f"{shared} shared physical observation(s); in a declared static scene one object "
            f"cannot appear twice in one frame, unless the frame's segmentation split it"
        ),
        metric="shared_physical_observation_count",
        observed=float(shared),
        threshold=1.0,
    )
