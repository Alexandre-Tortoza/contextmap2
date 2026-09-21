"""The conservative staged baseline resolution policy.

The policy turns the typed evidence of one comparison into ``MATCH``, ``DISTINCT`` or
``UNRESOLVED`` through explicit, ordered stages, and records every rule that fired:

1. **gates** -- a failed hard validity gate (another geometric map or frame) leaves the pair
   ``UNRESOLVED`` as ``comparison_blocked``: coordinates that cannot be compared say nothing;
2. **eligibility** -- the baseline is anchored in geometry, because in a static scene the same
   object is in the same place. Without available geometric evidence it does not guess: the pair is
   ``UNRESOLVED``. Every other channel is optional;
3. **evidence** -- each channel is *used* (enabled and available), *ignored* (available but not
   enabled, so channels can be ablated one at a time) or *unavailable*. An unavailable channel is
   recorded and casts no vote: missing evidence is never a negative vote;
4. **rules** -- only the *status* of each used channel is read (supporting, conflicting, neutral),
   never a score, and no number is averaged or weighted across channels. The rules run in this
   order and the first that applies decides:

   * ``materially-conflicting``: some used channel supports and another conflicts, so the channels
     disagree and the pair stays ``UNRESOLVED`` (``conflicting_evidence``);
   * ``match-supported``: geometry supports, no used channel conflicts and at least
     ``min_supporting_channels`` channels support, so ``MATCH``. With one channel required the
     geometry-only path is valid; a profile can demand corroboration from another channel;
   * ``distinct-supported``: geometry conflicts and no used channel supports, so ``DISTINCT``.
     A conflict in any *other* channel is never enough (upstream semantics may be wrong), and
     neither is neutral geometry: nearby, indistinct objects stay unresolved;
   * ``insufficient-evidence``: anything else stays ``UNRESOLVED``;

5. **decision** -- the rule that concluded the outcome, with the channels it rests on.

The policy prefers ``UNRESOLVED`` whenever the available evidence is insufficient or materially
conflicting, and it never forces a pair into a binary answer. It has no learned matcher, consults no
external knowledge, tracks no moving object, has no side effect (a ``MATCH`` merges nothing; that is
materialization) and never falls back to another policy. The same evidence and configuration always
give the same decision, and the thresholds that shape each channel's status are recorded on the
channel evidence itself, under their own versioned policies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from contextmap.entity_resolution.channels import EvidenceStatus, MatchChannel
from contextmap.entity_resolution.decision import (
    DecisionProvenance,
    PolicyStage,
    ResolutionDecision,
    ResolutionOutcome,
    TriggeredRule,
    UnresolvedReason,
    decision_id_for,
)
from contextmap.entity_resolution.evidence import EntityMatchEvidence
from contextmap.entity_resolution.models import PolicyRef

CONSERVATIVE_RESOLUTION_POLICY_ID = "conservative-staged-resolution-v1"
"""Versioned identity of the staged rules described in this module."""

_CHANNEL_ORDER = {channel: index for index, channel in enumerate(MatchChannel)}


@dataclass(frozen=True, kw_only=True)
class ConservativeResolutionPolicy:
    """The configuration of the conservative baseline.

    There is no score, weight or threshold to tune here: the thresholds that decide whether a
    channel supports or conflicts belong to each channel's own versioned policy. This
    configuration only says which channels count and how much corroboration a match needs, and both
    are declared, never defaulted.

    Attributes:
        use_channels: The channels whose evidence the decision may rest on, in canonical order and
            unique; must include geometry. A channel that is available but not listed is ignored,
            which makes channel ablations a configuration change.
        min_supporting_channels: How many used channels must support for a ``MATCH``, at least one
            (geometry alone, the geometry-only path) and at most the number of used channels.
    """

    use_channels: tuple[MatchChannel, ...]
    min_supporting_channels: int

    def __post_init__(self) -> None:
        """Validate that the configuration is coherent.

        Raises:
            ValueError: If the channels are not in canonical order and unique, geometry is not
                among them, or the number of supporting channels is outside ``[1, len(channels)]``.
        """
        order = [_CHANNEL_ORDER[channel] for channel in self.use_channels]
        if order != sorted(set(order)):
            raise ValueError("use_channels must be in canonical channel order and unique")
        if MatchChannel.GEOMETRY not in self.use_channels:
            raise ValueError("use_channels must include geometry: the baseline is anchored in it")
        if not 1 <= self.min_supporting_channels <= len(self.use_channels):
            raise ValueError(
                f"min_supporting_channels must be within [1, {len(self.use_channels)}], got "
                f"{self.min_supporting_channels}"
            )

    def fingerprint(self) -> str:
        """Hash the policy identity and configuration, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": CONSERVATIVE_RESOLUTION_POLICY_ID,
                "use_channels": [channel.value for channel in self.use_channels],
                "min_supporting_channels": self.min_supporting_channels,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on every decision."""
        return PolicyRef(
            policy_id=CONSERVATIVE_RESOLUTION_POLICY_ID,
            configuration_fingerprint=self.fingerprint(),
        )


def decide(
    evidence: EntityMatchEvidence,
    policy: ConservativeResolutionPolicy,
    *,
    code_version: str | None = None,
) -> ResolutionDecision:
    """Decide one comparison under the conservative baseline.

    Args:
        evidence: The typed evidence of the comparison.
        policy: Which channels count and how much corroboration a match needs.
        code_version: Code revision that produced the decision, when known.

    Returns:
        The decision, with every rule that fired, the channels it used and ignored and, when it is
        ``UNRESOLVED``, why. The evidence is not modified and nothing is merged.
    """
    rules: list[TriggeredRule] = []
    failed = evidence.failed_gates()
    if failed:
        rules.extend(
            _rule(
                PolicyStage.GATE,
                f"gate-failed-{gate.gate_id}",
                ResolutionOutcome.UNRESOLVED,
                f"{gate.gate_id}: {gate.detail}",
            )
            for gate in failed
        )
        return _conclude(
            evidence,
            policy,
            rules,
            ResolutionOutcome.UNRESOLVED,
            UnresolvedReason.COMPARISON_BLOCKED,
            "comparison-blocked",
            "a hard validity gate failed, so the pair was never compared",
            used=(),
            ignored=(),
            channels=(),
            code_version=code_version,
        )
    rules.append(
        _rule(
            PolicyStage.GATE,
            "gates-passed",
            None,
            f"{len(evidence.gates)} gate(s) passed: {', '.join(g.gate_id for g in evidence.gates)}",
        )
    )
    geometry = evidence.geometry
    unavailable = None if geometry is None else geometry.unavailable
    if geometry is None or unavailable is not None:
        cause = "not evaluated" if unavailable is None else unavailable.reason.value
        rules.append(
            _rule(
                PolicyStage.ELIGIBILITY,
                "geometry-required",
                ResolutionOutcome.UNRESOLVED,
                f"the baseline is anchored in geometry, which is not available ({cause}); the "
                f"other channels alone are not used to guess",
            )
        )
        return _conclude(
            evidence,
            policy,
            rules,
            ResolutionOutcome.UNRESOLVED,
            UnresolvedReason.INSUFFICIENT_EVIDENCE,
            "insufficient-evidence",
            "no geometric evidence is available",
            used=(),
            ignored=(),
            channels=(),
            code_version=code_version,
        )
    rules.append(
        _rule(
            PolicyStage.ELIGIBILITY,
            "geometry-available",
            None,
            "geometric evidence is available",
            (MatchChannel.GEOMETRY,),
        )
    )
    used, ignored = _collect_evidence(evidence, policy, rules)
    statuses = {channel: item.status for channel, item in evidence.channels() if channel in used}
    supporting = tuple(
        channel for channel in used if statuses[channel] is EvidenceStatus.SUPPORTING
    )
    conflicting = tuple(
        channel for channel in used if statuses[channel] is EvidenceStatus.CONFLICTING
    )
    rules.append(
        _rule(
            PolicyStage.RULE,
            "channel-votes",
            None,
            f"supporting: {_names(supporting)}; conflicting: {_names(conflicting)}; "
            f"a match needs {policy.min_supporting_channels} supporting channel(s)",
            tuple(sorted(supporting + conflicting, key=_CHANNEL_ORDER.__getitem__)),
        )
    )
    geometry_status = statuses[MatchChannel.GEOMETRY]
    if supporting and conflicting:
        outcome, reason, rule_id = (
            ResolutionOutcome.UNRESOLVED,
            UnresolvedReason.CONFLICTING_EVIDENCE,
            "materially-conflicting",
        )
        detail = (
            f"{_names(supporting)} support but {_names(conflicting)} conflict: the channels "
            f"disagree"
        )
        channels = tuple(sorted(supporting + conflicting, key=_CHANNEL_ORDER.__getitem__))
    elif geometry_status is EvidenceStatus.SUPPORTING and not conflicting:
        if len(supporting) >= policy.min_supporting_channels:
            outcome, reason, rule_id = ResolutionOutcome.MATCH, None, "match-supported"
            detail = f"geometry supports, nothing conflicts, and {_names(supporting)} support"
            channels = supporting
        else:
            outcome, reason, rule_id = (
                ResolutionOutcome.UNRESOLVED,
                UnresolvedReason.INSUFFICIENT_EVIDENCE,
                "insufficient-evidence",
            )
            detail = (
                f"geometry supports but only {len(supporting)} of the required "
                f"{policy.min_supporting_channels} channel(s) do"
            )
            channels = supporting
    elif geometry_status is EvidenceStatus.CONFLICTING and not supporting:
        outcome, reason, rule_id = ResolutionOutcome.DISTINCT, None, "distinct-supported"
        detail = f"geometry conflicts and nothing supports; conflicting: {_names(conflicting)}"
        channels = conflicting
    else:
        outcome, reason, rule_id = (
            ResolutionOutcome.UNRESOLVED,
            UnresolvedReason.INSUFFICIENT_EVIDENCE,
            "insufficient-evidence",
        )
        detail = f"geometry is {geometry_status.value}: the evidence does not settle the pair"
        channels = ()
    return _conclude(
        evidence,
        policy,
        rules,
        outcome,
        reason,
        rule_id,
        detail,
        used=used,
        ignored=ignored,
        channels=channels,
        code_version=code_version,
    )


def _collect_evidence(
    evidence: EntityMatchEvidence, policy: ConservativeResolutionPolicy, rules: list[TriggeredRule]
) -> tuple[tuple[MatchChannel, ...], tuple[MatchChannel, ...]]:
    """Sort the evaluated channels into used, ignored and unavailable, recording each."""
    used: list[MatchChannel] = []
    ignored: list[MatchChannel] = []
    for channel, item in evidence.channels():
        if item.status is EvidenceStatus.UNAVAILABLE:
            reason = item.unavailable.reason.value if item.unavailable else "unavailable"
            rules.append(
                _rule(
                    PolicyStage.EVIDENCE,
                    f"channel-unavailable-{channel.value}",
                    None,
                    f"{channel.value} is unavailable ({reason}) and casts no vote",
                    (channel,),
                )
            )
        elif channel in policy.use_channels:
            used.append(channel)
            rules.append(
                _rule(
                    PolicyStage.EVIDENCE,
                    f"channel-used-{channel.value}",
                    None,
                    f"{channel.value} is {item.status.value}",
                    (channel,),
                )
            )
        else:
            ignored.append(channel)
            rules.append(
                _rule(
                    PolicyStage.EVIDENCE,
                    f"channel-ignored-{channel.value}",
                    None,
                    f"{channel.value} is {item.status.value} but the policy does not use it",
                    (channel,),
                )
            )
    return tuple(used), tuple(ignored)


def _rule(
    stage: PolicyStage,
    rule_id: str,
    outcome: ResolutionOutcome | None,
    detail: str,
    channels: tuple[MatchChannel, ...] = (),
) -> TriggeredRule:
    return TriggeredRule(
        stage=stage, rule_id=rule_id, outcome=outcome, detail=detail, channels=channels
    )


def _names(channels: tuple[MatchChannel, ...]) -> str:
    return ", ".join(channel.value for channel in channels) or "none"


def _conclude(
    evidence: EntityMatchEvidence,
    policy: ConservativeResolutionPolicy,
    rules: list[TriggeredRule],
    outcome: ResolutionOutcome,
    reason: UnresolvedReason | None,
    rule_id: str,
    detail: str,
    *,
    used: tuple[MatchChannel, ...],
    ignored: tuple[MatchChannel, ...],
    channels: tuple[MatchChannel, ...],
    code_version: str | None,
) -> ResolutionDecision:
    rules.append(_rule(PolicyStage.DECISION, rule_id, outcome, detail, channels))
    return ResolutionDecision(
        decision_id=decision_id_for(evidence.comparison_id, policy.ref()),
        entity_a_ref=evidence.entity_a_ref,
        entity_b_ref=evidence.entity_b_ref,
        decision=outcome,
        evidence_ref=evidence.comparison_id,
        policy=policy.ref(),
        triggered_rules=tuple(rules),
        channels_used=used,
        channels_ignored=ignored,
        unresolved_reason=reason,
        provenance=DecisionProvenance(code_version=code_version),
    )
