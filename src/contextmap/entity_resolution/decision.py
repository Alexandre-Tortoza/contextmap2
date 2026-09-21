"""Resolution decisions: MATCH, DISTINCT or UNRESOLVED, with the rules that produced them.

A :class:`ResolutionDecision` is the verdict of a versioned policy on one comparison. Its decision
space has three members and ``UNRESOLVED`` is a first-class result, never coerced into a match or a
non-match: a policy that lacks the evidence, or finds it materially conflicting, says so and says
why. Every decision points to the comparison evidence it was reached from (``evidence_ref``, the
identity of an :class:`~contextmap.entity_resolution.evidence.EntityMatchEvidence`), lists the rules
that fired in the order they were evaluated and separates the channels that were used from those
that were ignored, so it can be reconstructed from typed evidence and explicit rules without
rerunning anything.

A decision applies to entity references under an explicit resolution context. It never implies
permanent identity across independently rebuilt maps, never mutates an entity, and is not a merge:
grouping the entities of ``MATCH`` decisions is materialization, a separate step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import NewType

from contextmap.entity_resolution._checks import require_present
from contextmap.entity_resolution.channels import MatchChannel
from contextmap.entity_resolution.evidence import (
    ComparisonId,
    comparison_id_for,
    reference_order,
)
from contextmap.semantic_mapping import EntityReference

ResolutionDecisionId = NewType("ResolutionDecisionId", str)
"""Identity of one decision, derived from the comparison and the policy that decided it."""

_CHANNEL_ORDER = {channel: index for index, channel in enumerate(MatchChannel)}


class ResolutionOutcome(Enum):
    """The verdict of a resolution policy on a pair of entities.

    Attributes:
        MATCH: The entities describe the same physical object.
        DISTINCT: The entities describe different physical objects.
        UNRESOLVED: The policy cannot tell. A valid result, never coerced into the other two.
    """

    MATCH = "match"
    DISTINCT = "distinct"
    UNRESOLVED = "unresolved"


class UnresolvedReason(Enum):
    """Why a policy left a pair unresolved.

    Attributes:
        COMPARISON_BLOCKED: A hard validity gate failed, so the pair was never compared.
        INSUFFICIENT_EVIDENCE: The evidence available is not enough for the policy's rules.
        CONFLICTING_EVIDENCE: The channels, or the findings of one channel, disagree materially.
    """

    COMPARISON_BLOCKED = "comparison_blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class PolicyStage(Enum):
    """The stage of a staged resolution policy a rule belongs to.

    Attributes:
        GATE: A hard validity gate.
        ELIGIBILITY: A check that the pair is eligible for comparison.
        EVIDENCE: The collection of channel evidence.
        RULE: A rule or threshold evaluated over the evidence.
        DECISION: The rule that produced the final outcome.
    """

    GATE = "gate"
    ELIGIBILITY = "eligibility"
    EVIDENCE = "evidence"
    RULE = "rule"
    DECISION = "decision"


@dataclass(frozen=True, kw_only=True)
class ResolutionPolicyRef:
    """The versioned policy and configuration that produced a decision.

    Attributes:
        policy_id: Versioned identity of the policy.
        configuration_fingerprint: Hash of the thresholds and options of that execution.
    """

    policy_id: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "policy_id", "configuration_fingerprint")


@dataclass(frozen=True, kw_only=True)
class TriggeredRule:
    """A rule of the policy that fired while evaluating a pair.

    Attributes:
        stage: The policy stage the rule belongs to.
        rule_id: Versioned identity of the rule.
        outcome: What the rule concludes, or ``None`` when it only records a fact.
        detail: A deterministic, human-readable explanation, naming the values it looked at.
        channels: The channels whose evidence the rule used, in canonical order.
    """

    stage: PolicyStage
    rule_id: str
    outcome: ResolutionOutcome | None
    detail: str
    channels: tuple[MatchChannel, ...]

    def __post_init__(self) -> None:
        """Validate the rule and its channels.

        Raises:
            ValueError: If the rule or the detail is empty, or the channels are not in canonical
                order and unique.
        """
        require_present(self, "rule_id", "detail")
        _require_channels_in_order("channels", self.channels)


@dataclass(frozen=True, kw_only=True)
class DecisionProvenance:
    """Where a decision was produced.

    Attributes:
        code_version: Code revision that produced the decision, when known.
    """

    code_version: str | None = None

    def __post_init__(self) -> None:
        """Validate the code version.

        Raises:
            ValueError: If the code version is present but empty.
        """
        if self.code_version is not None:
            require_present(self, "code_version")


def decision_id_for(
    evidence_ref: ComparisonId, policy: ResolutionPolicyRef
) -> ResolutionDecisionId:
    """Compute the deterministic identity of the decision of a policy on a comparison.

    Args:
        evidence_ref: The comparison that was decided.
        policy: The policy and configuration that decided it.

    Returns:
        A pure function of its inputs, so rerunning the same policy on the same evidence gives
        the same identity and another policy or configuration gives another.
    """
    material = "\n".join((evidence_ref, policy.policy_id, policy.configuration_fingerprint))
    return ResolutionDecisionId(f"decision--{hashlib.sha256(material.encode()).hexdigest()[:16]}")


@dataclass(frozen=True, kw_only=True)
class ResolutionDecision:
    """The verdict of a policy on one comparison of two entities.

    Attributes:
        decision_id: Identity of the decision, derived from the evidence and the policy.
        entity_a_ref: The entity whose reference sorts first.
        entity_b_ref: The other entity.
        decision: ``MATCH``, ``DISTINCT`` or ``UNRESOLVED``.
        evidence_ref: The comparison the decision was reached from.
        policy: The policy and configuration that produced it.
        triggered_rules: The rules that fired, in the order they were evaluated, with exactly the
            rule of the ``DECISION`` stage that concluded the outcome.
        channels_used: The channels whose evidence the decision rests on, in canonical order.
        channels_ignored: The channels available but not used, in canonical order.
        unresolved_reason: Why the pair is unresolved; present exactly for ``UNRESOLVED``.
        provenance: Where the decision was produced.
    """

    decision_id: ResolutionDecisionId
    entity_a_ref: EntityReference
    entity_b_ref: EntityReference
    decision: ResolutionOutcome
    evidence_ref: ComparisonId
    policy: ResolutionPolicyRef
    triggered_rules: tuple[TriggeredRule, ...]
    channels_used: tuple[MatchChannel, ...]
    channels_ignored: tuple[MatchChannel, ...]
    unresolved_reason: UnresolvedReason | None
    provenance: DecisionProvenance

    def __post_init__(self) -> None:
        """Validate that the decision is auditable and consistent with its own record.

        Raises:
            ValueError: If the pair is not two different entities in canonical order, the
                evidence reference or the decision id is not the one derived, the unresolved
                reason is not present exactly for ``UNRESOLVED``, no rule of the decision stage
                concludes the outcome, a resolved decision rests on no channel, or the channels
                are unsorted or both used and ignored.
        """
        require_present(self, "decision_id", "evidence_ref")
        if reference_order(self.entity_a_ref) >= reference_order(self.entity_b_ref):
            raise ValueError("the pair must be two different entities in canonical order")
        if self.evidence_ref != comparison_id_for(self.entity_a_ref, self.entity_b_ref):
            raise ValueError("evidence_ref is not the comparison of this pair")
        if self.decision_id != decision_id_for(self.evidence_ref, self.policy):
            raise ValueError("decision_id is not the identity derived from the evidence and policy")
        if (self.unresolved_reason is not None) != (self.decision is ResolutionOutcome.UNRESOLVED):
            raise ValueError(
                "unresolved_reason must be present exactly when the decision is unresolved"
            )
        if not any(
            rule.stage is PolicyStage.DECISION and rule.outcome is self.decision
            for rule in self.triggered_rules
        ):
            raise ValueError(
                f"a {self.decision.value} decision needs a deciding rule of the decision stage "
                f"that concludes {self.decision.value}"
            )
        _require_channels_in_order("channels_used", self.channels_used)
        _require_channels_in_order("channels_ignored", self.channels_ignored)
        if self.decision is not ResolutionOutcome.UNRESOLVED and not self.channels_used:
            raise ValueError("a resolved decision must rest on at least one channel: channels_used")
        both = set(self.channels_used) & set(self.channels_ignored)
        if both:
            raise ValueError(
                f"channels {sorted(item.value for item in both)!r} are both used and ignored"
            )


def _require_channels_in_order(name: str, channels: tuple[MatchChannel, ...]) -> None:
    keys = [_CHANNEL_ORDER[channel] for channel in channels]
    if keys != sorted(set(keys)):
        raise ValueError(f"{name} must be in canonical channel order and unique")
