"""The conservative staged baseline: explicit rules over typed channel statuses, no weighted sum."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from resolution_builders import (
    REF_A,
    REF_B,
    appearance_evidence,
    channel_policy,
    geometry_evidence,
    match_evidence,
    passed_gates,
    representation_evidence,
    semantic_evidence,
    temporal_evidence,
    unavailable,
)

from contextmap.entity_resolution import (
    CONSERVATIVE_RESOLUTION_POLICY_ID,
    AppearanceEvidence,
    ConservativeResolutionPolicy,
    EntityMatchEvidence,
    EvidenceStatus,
    GateResult,
    GeometryEvidence,
    MatchChannel,
    PolicyStage,
    ResolutionDecision,
    ResolutionOutcome,
    UnavailableReason,
    UnresolvedReason,
    decide,
)

GEOMETRY_ONLY = ConservativeResolutionPolicy(
    use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
)
ALL_CHANNELS = ConservativeResolutionPolicy(
    use_channels=tuple(MatchChannel), min_supporting_channels=1
)
S, C, N = EvidenceStatus.SUPPORTING, EvidenceStatus.CONFLICTING, EvidenceStatus.NEUTRAL


def evidence_with(**channels: Any) -> EntityMatchEvidence:
    return match_evidence(**channels)


def missing(channel_class: Any) -> Any:
    return channel_class(policy=channel_policy(), unavailable=unavailable())


def rule_ids(decision: ResolutionDecision, stage: PolicyStage) -> list[str]:
    return [item.rule_id for item in decision.triggered_rules if item.stage is stage]


def deciding_rule(decision: ResolutionDecision) -> Any:
    (rule,) = [item for item in decision.triggered_rules if item.stage is PolicyStage.DECISION]
    return rule


# --- MATCH --------------------------------------------------------------------------------------


def test_geometry_alone_can_match_because_the_geometry_only_path_is_valid() -> None:
    decision = decide(evidence_with(geometry=geometry_evidence(S)), GEOMETRY_ONLY)

    assert decision.decision is ResolutionOutcome.MATCH
    assert decision.channels_used == (MatchChannel.GEOMETRY,)
    assert decision.channels_ignored == ()
    assert decision.unresolved_reason is None
    assert deciding_rule(decision).rule_id == "match-supported"


def test_a_match_is_reconstructable_from_the_evidence_and_the_rules() -> None:
    evidence = evidence_with(geometry=geometry_evidence(S), semantic=semantic_evidence(S))

    decision = decide(evidence, ALL_CHANNELS)

    assert decision.evidence_ref == evidence.comparison_id
    assert (decision.entity_a_ref, decision.entity_b_ref) == (REF_A, REF_B)
    assert rule_ids(decision, PolicyStage.GATE) == ["gates-passed"]
    assert rule_ids(decision, PolicyStage.ELIGIBILITY) == ["geometry-available"]
    assert rule_ids(decision, PolicyStage.EVIDENCE) == [
        "channel-used-geometry",
        "channel-used-semantic",
    ]
    assert deciding_rule(decision).channels == (MatchChannel.GEOMETRY, MatchChannel.SEMANTIC)
    # Reaplicar a política à mesma evidência reproduz a decisão.
    assert decide(evidence, ALL_CHANNELS) == decision


def test_missing_optional_channels_are_not_negative_votes() -> None:
    evidence = evidence_with(
        geometry=geometry_evidence(S),
        semantic=missing(type(semantic_evidence())),
        appearance=missing(AppearanceEvidence),
        point_representation=missing(type(representation_evidence())),
    )

    decision = decide(evidence, ALL_CHANNELS)

    assert decision.decision is ResolutionOutcome.MATCH
    assert decision.channels_used == (MatchChannel.GEOMETRY,)
    assert rule_ids(decision, PolicyStage.EVIDENCE) == [
        "channel-used-geometry",
        "channel-unavailable-semantic",
        "channel-unavailable-appearance",
        "channel-unavailable-point_representation",
    ]


def test_neutral_channels_cast_no_vote() -> None:
    evidence = evidence_with(
        geometry=geometry_evidence(S), semantic=semantic_evidence(N), temporal=temporal_evidence(N)
    )

    decision = decide(evidence, ALL_CHANNELS)

    assert decision.decision is ResolutionOutcome.MATCH
    assert deciding_rule(decision).channels == (MatchChannel.GEOMETRY,)


def test_a_policy_can_demand_corroboration_from_another_channel() -> None:
    demanding = ConservativeResolutionPolicy(
        use_channels=tuple(MatchChannel), min_supporting_channels=2
    )

    alone = decide(evidence_with(geometry=geometry_evidence(S)), demanding)
    corroborated = decide(
        evidence_with(geometry=geometry_evidence(S), appearance=appearance_evidence(S)), demanding
    )

    assert alone.decision is ResolutionOutcome.UNRESOLVED
    assert alone.unresolved_reason is UnresolvedReason.INSUFFICIENT_EVIDENCE
    assert corroborated.decision is ResolutionOutcome.MATCH


# --- DISTINCT -----------------------------------------------------------------------------------


def test_geometric_separation_with_no_support_is_distinct() -> None:
    decision = decide(evidence_with(geometry=geometry_evidence(C)), GEOMETRY_ONLY)

    assert decision.decision is ResolutionOutcome.DISTINCT
    assert deciding_rule(decision).rule_id == "distinct-supported"
    assert decision.channels_used == (MatchChannel.GEOMETRY,)


def test_distinct_tolerates_neutral_and_unavailable_channels() -> None:
    evidence = evidence_with(
        geometry=geometry_evidence(C),
        semantic=semantic_evidence(N),
        appearance=missing(AppearanceEvidence),
    )

    assert decide(evidence, ALL_CHANNELS).decision is ResolutionOutcome.DISTINCT


# --- UNRESOLVED, the conservative default -------------------------------------------------------


@pytest.mark.parametrize(
    ("geometry", "other"),
    [(S, ("semantic", C)), (S, ("appearance", C)), (C, ("semantic", S)), (C, ("appearance", S))],
)
def test_materially_conflicting_channels_are_unresolved_not_averaged(
    geometry: EvidenceStatus, other: tuple[str, EvidenceStatus]
) -> None:
    name, status = other
    channel = {"semantic": semantic_evidence, "appearance": appearance_evidence}[name](status)

    decision = decide(
        evidence_with(geometry=geometry_evidence(geometry), **{name: channel}), ALL_CHANNELS
    )

    assert decision.decision is ResolutionOutcome.UNRESOLVED
    assert decision.unresolved_reason is UnresolvedReason.CONFLICTING_EVIDENCE
    assert deciding_rule(decision).rule_id == "materially-conflicting"
    assert set(deciding_rule(decision).channels) == {
        MatchChannel.GEOMETRY,
        MatchChannel[name.upper()],
    }


def test_neutral_geometry_is_insufficient_evidence() -> None:
    decision = decide(evidence_with(geometry=geometry_evidence(N)), GEOMETRY_ONLY)

    assert decision.decision is ResolutionOutcome.UNRESOLVED
    assert decision.unresolved_reason is UnresolvedReason.INSUFFICIENT_EVIDENCE
    assert deciding_rule(decision).rule_id == "insufficient-evidence"


def test_a_semantic_conflict_with_neutral_geometry_is_not_promoted_to_distinct() -> None:
    evidence = evidence_with(geometry=geometry_evidence(N), semantic=semantic_evidence(C))

    assert decide(evidence, ALL_CHANNELS).decision is ResolutionOutcome.UNRESOLVED


def test_without_geometry_the_baseline_does_not_guess() -> None:
    evidence = evidence_with(
        geometry=missing(GeometryEvidence),
        semantic=semantic_evidence(S),
        appearance=appearance_evidence(S),
    )

    decision = decide(evidence, ALL_CHANNELS)

    assert decision.decision is ResolutionOutcome.UNRESOLVED
    assert decision.unresolved_reason is UnresolvedReason.INSUFFICIENT_EVIDENCE
    assert rule_ids(decision, PolicyStage.ELIGIBILITY) == ["geometry-required"]
    assert decision.channels_used == ()


def test_a_geometry_channel_that_was_never_evaluated_is_also_not_enough() -> None:
    decision = decide(evidence_with(geometry=None, semantic=semantic_evidence(S)), ALL_CHANNELS)

    assert decision.decision is ResolutionOutcome.UNRESOLVED
    assert rule_ids(decision, PolicyStage.ELIGIBILITY) == ["geometry-required"]


def test_a_failed_gate_leaves_the_pair_unresolved_and_names_the_gate() -> None:
    failed = tuple(
        GateResult(gate_id=gate.gate_id, passed=False, detail="map-0001 and map-0002")
        if gate.gate_id == "same-geometric-map"
        else gate
        for gate in passed_gates()
    )
    blocked = GeometryEvidence(
        policy=channel_policy(),
        unavailable=unavailable(UnavailableReason.BLOCKED_BY_GATE, "same-geometric-map"),
    )

    decision = decide(evidence_with(gates=failed, geometry=blocked), ALL_CHANNELS)

    assert decision.decision is ResolutionOutcome.UNRESOLVED
    assert decision.unresolved_reason is UnresolvedReason.COMPARISON_BLOCKED
    assert rule_ids(decision, PolicyStage.GATE) == ["gate-failed-same-geometric-map"]
    assert deciding_rule(decision).rule_id == "comparison-blocked"
    assert decision.channels_used == ()


# --- ablation: channels enabled independently ---------------------------------------------------


def test_a_disabled_channel_is_ignored_and_recorded_as_such() -> None:
    evidence = evidence_with(geometry=geometry_evidence(S), semantic=semantic_evidence(C))

    ignoring = decide(evidence, GEOMETRY_ONLY)
    using = decide(evidence, ALL_CHANNELS)

    assert ignoring.decision is ResolutionOutcome.MATCH
    assert ignoring.channels_ignored == (MatchChannel.SEMANTIC,)
    assert "channel-ignored-semantic" in rule_ids(ignoring, PolicyStage.EVIDENCE)
    assert using.decision is ResolutionOutcome.UNRESOLVED


@pytest.mark.parametrize(
    "extra",
    [
        MatchChannel.SEMANTIC,
        MatchChannel.APPEARANCE,
        MatchChannel.TEMPORAL,
        MatchChannel.POINT_REPRESENTATION,
    ],
)
def test_each_channel_can_be_enabled_independently_of_the_others(extra: MatchChannel) -> None:
    policy = ConservativeResolutionPolicy(
        use_channels=(MatchChannel.GEOMETRY, extra), min_supporting_channels=1
    )
    everything = evidence_with(
        geometry=geometry_evidence(S),
        semantic=semantic_evidence(N),
        appearance=appearance_evidence(N),
        temporal=temporal_evidence(N),
        point_representation=representation_evidence(N),
    )

    decision = decide(everything, policy)

    assert decision.channels_used == (MatchChannel.GEOMETRY, extra)
    assert set(decision.channels_ignored) == set(MatchChannel) - {MatchChannel.GEOMETRY, extra}


# --- explicitness, versioning and replaceability ------------------------------------------------


def test_there_is_no_weighted_sum_or_score_in_the_policy_or_the_decision() -> None:
    fields = {field.name for field in dataclasses.fields(ConservativeResolutionPolicy)}
    decision = decide(evidence_with(geometry=geometry_evidence(S)), GEOMETRY_ONLY)

    assert fields == {"use_channels", "min_supporting_channels"}
    assert not any(hasattr(decision, name) for name in ("score", "weight", "similarity"))


def test_the_decision_records_the_policy_identity_and_its_configuration() -> None:
    decision = decide(evidence_with(geometry=geometry_evidence(S)), ALL_CHANNELS)

    assert decision.policy.policy_id == CONSERVATIVE_RESOLUTION_POLICY_ID
    assert decision.policy.configuration_fingerprint == ALL_CHANNELS.fingerprint()


def test_the_same_evidence_and_configuration_give_the_same_decision() -> None:
    evidence = evidence_with(geometry=geometry_evidence(S))

    first = decide(evidence, GEOMETRY_ONLY)
    second = decide(evidence, GEOMETRY_ONLY)
    other = decide(evidence, ALL_CHANNELS)

    assert first == second
    assert first.decision_id != other.decision_id
    assert GEOMETRY_ONLY.fingerprint() != ALL_CHANNELS.fingerprint()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_channels": (), "min_supporting_channels": 1},
        {"use_channels": (MatchChannel.SEMANTIC,), "min_supporting_channels": 1},
        {
            "use_channels": (MatchChannel.SEMANTIC, MatchChannel.GEOMETRY),
            "min_supporting_channels": 1,
        },
        {
            "use_channels": (MatchChannel.GEOMETRY, MatchChannel.GEOMETRY),
            "min_supporting_channels": 1,
        },
        {"use_channels": (MatchChannel.GEOMETRY,), "min_supporting_channels": 0},
        {"use_channels": (MatchChannel.GEOMETRY,), "min_supporting_channels": 2},
    ],
)
def test_the_policy_refuses_an_incoherent_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ConservativeResolutionPolicy(**kwargs)


def test_the_decision_never_merges_or_touches_the_entities() -> None:
    evidence = evidence_with(geometry=geometry_evidence(S))

    decision = decide(evidence, GEOMETRY_ONLY)

    # Só devolve um veredito: a fusão em uma entidade resolvida é outro passo (materialização).
    assert isinstance(decision, ResolutionDecision)
    assert evidence == evidence_with(geometry=geometry_evidence(S))
