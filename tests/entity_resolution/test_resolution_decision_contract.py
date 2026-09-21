"""ResolutionDecision: MATCH, DISTINCT and UNRESOLVED are all representable and auditable."""

from __future__ import annotations

import dataclasses

import pytest
from resolution_builders import REF_A, REF_B, decision, policy_ref, ref, triggered_rule

from contextmap.entity_resolution import (
    DecisionProvenance,
    MatchChannel,
    PolicyRef,
    PolicyStage,
    ResolutionOutcome,
    TriggeredRule,
    UnresolvedReason,
    comparison_id_for,
    decision_id_for,
)


def test_the_decision_space_has_exactly_three_outcomes() -> None:
    assert {outcome.value for outcome in ResolutionOutcome} == {"match", "distinct", "unresolved"}


def test_a_match_is_auditable_from_the_rule_that_decided_it() -> None:
    result = decision(ResolutionOutcome.MATCH)

    assert result.decision is ResolutionOutcome.MATCH
    assert result.unresolved_reason is None
    assert result.evidence_ref == comparison_id_for(REF_A, REF_B)
    assert result.policy == policy_ref()
    assert [rule.rule_id for rule in result.triggered_rules] == ["geometry-only-match"]


def test_a_distinct_decision_is_auditable_too() -> None:
    result = decision(ResolutionOutcome.DISTINCT)

    assert result.decision is ResolutionOutcome.DISTINCT
    assert result.unresolved_reason is None


def test_unresolved_is_a_valid_result_that_says_why() -> None:
    result = decision(
        ResolutionOutcome.UNRESOLVED,
        unresolved_reason=UnresolvedReason.CONFLICTING_EVIDENCE,
        channels_used=(),
        channels_ignored=(),
        triggered_rules=(
            triggered_rule(
                ResolutionOutcome.UNRESOLVED,
                stage=PolicyStage.DECISION,
                rule_id="prefer-unresolved",
            ),
        ),
    )

    assert result.decision is ResolutionOutcome.UNRESOLVED
    assert result.unresolved_reason is UnresolvedReason.CONFLICTING_EVIDENCE


def test_unresolved_is_never_silently_a_match_or_a_non_match() -> None:
    assert ResolutionOutcome.UNRESOLVED not in (ResolutionOutcome.MATCH, ResolutionOutcome.DISTINCT)
    with pytest.raises(ValueError, match="unresolved_reason"):
        decision(ResolutionOutcome.UNRESOLVED, unresolved_reason=None)


@pytest.mark.parametrize("outcome", [ResolutionOutcome.MATCH, ResolutionOutcome.DISTINCT])
def test_only_unresolved_carries_an_unresolved_reason(outcome: ResolutionOutcome) -> None:
    with pytest.raises(ValueError, match="unresolved_reason"):
        decision(outcome, unresolved_reason=UnresolvedReason.INSUFFICIENT_EVIDENCE)


@pytest.mark.parametrize("outcome", [ResolutionOutcome.MATCH, ResolutionOutcome.DISTINCT])
def test_a_resolved_decision_needs_a_deciding_rule_that_agrees_with_it(
    outcome: ResolutionOutcome,
) -> None:
    other = (
        ResolutionOutcome.DISTINCT
        if outcome is ResolutionOutcome.MATCH
        else ResolutionOutcome.MATCH
    )

    with pytest.raises(ValueError, match="deciding rule"):
        decision(outcome, triggered_rules=(triggered_rule(other),))
    with pytest.raises(ValueError, match="deciding rule"):
        decision(outcome, triggered_rules=(triggered_rule(None),))
    with pytest.raises(ValueError, match="deciding rule"):
        decision(outcome, triggered_rules=())


@pytest.mark.parametrize("outcome", [ResolutionOutcome.MATCH, ResolutionOutcome.DISTINCT])
def test_a_resolved_decision_rests_on_at_least_one_channel(outcome: ResolutionOutcome) -> None:
    with pytest.raises(ValueError, match="channels_used"):
        decision(outcome, channels_used=())


def test_a_channel_is_used_or_ignored_never_both() -> None:
    with pytest.raises(ValueError, match="both used and ignored"):
        decision(channels_used=(MatchChannel.GEOMETRY,), channels_ignored=(MatchChannel.GEOMETRY,))


def test_channels_are_listed_in_canonical_order() -> None:
    with pytest.raises(ValueError, match="channels_used"):
        decision(channels_used=(MatchChannel.SEMANTIC, MatchChannel.GEOMETRY))


def test_the_decision_names_the_pair_in_canonical_order() -> None:
    with pytest.raises(ValueError, match="canonical order"):
        decision(entity_a_ref=REF_B, entity_b_ref=REF_A)


def test_the_decision_points_to_the_evidence_of_the_same_pair() -> None:
    with pytest.raises(ValueError, match="evidence_ref"):
        decision(evidence_ref=comparison_id_for(REF_A, ref("entity--support-000009")))


def test_the_decision_id_is_derived_from_the_evidence_and_the_policy() -> None:
    comparison = comparison_id_for(REF_A, REF_B)
    other_policy = PolicyRef(
        policy_id="conservative-staged-resolution-v2", configuration_fingerprint="sha256:policy"
    )

    assert decision_id_for(comparison, policy_ref()) == decision_id_for(comparison, policy_ref())
    assert decision_id_for(comparison, policy_ref()) != decision_id_for(comparison, other_policy)
    with pytest.raises(ValueError, match="decision_id"):
        decision(decision_id="decision--tampered")


def test_the_policy_and_its_configuration_are_recorded() -> None:
    with pytest.raises(ValueError, match="configuration_fingerprint"):
        PolicyRef(policy_id="conservative-staged-resolution-v1", configuration_fingerprint="")
    with pytest.raises(ValueError, match="policy_id"):
        PolicyRef(policy_id=" ", configuration_fingerprint="sha256:policy")


def test_a_triggered_rule_states_its_stage_and_an_explanation() -> None:
    with pytest.raises(ValueError, match="detail"):
        TriggeredRule(
            stage=PolicyStage.GATE,
            rule_id="same-geometric-map",
            outcome=None,
            detail=" ",
            channels=(),
        )
    assert {stage.value for stage in PolicyStage} == {
        "gate",
        "eligibility",
        "evidence",
        "rule",
        "decision",
    }


def test_the_provenance_can_omit_the_code_version_but_not_be_blank() -> None:
    assert DecisionProvenance(code_version=None).code_version is None
    with pytest.raises(ValueError, match="code_version"):
        DecisionProvenance(code_version=" ")


def test_a_decision_is_immutable() -> None:
    result = decision()

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.decision = ResolutionOutcome.DISTINCT  # type: ignore[misc]
