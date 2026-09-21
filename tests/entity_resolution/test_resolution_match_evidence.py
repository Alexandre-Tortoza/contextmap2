"""EntityMatchEvidence: one comparison, every channel kept apart, and the gates that block it."""

from __future__ import annotations

import dataclasses

import pytest
from resolution_builders import (
    REF_A,
    REF_B,
    appearance_evidence,
    channel_policy,
    geometry_evidence,
    match_evidence,
    passed_gates,
    ref,
    representation_evidence,
    semantic_evidence,
    temporal_evidence,
    unavailable,
)

from contextmap.entity_resolution import (
    AppearanceEvidence,
    EntityMatchEvidence,
    EvidenceStatus,
    GateResult,
    GeometryEvidence,
    MatchChannel,
    MatchEvidenceProvenance,
    UnavailableReason,
    comparison_id_for,
)


def full_evidence() -> EntityMatchEvidence:
    return match_evidence(
        semantic=semantic_evidence(),
        appearance=appearance_evidence(),
        temporal=temporal_evidence(),
        point_representation=representation_evidence(),
    )


def test_every_channel_coexists_without_a_single_composite_score() -> None:
    evidence = full_evidence()

    assert {channel for channel, _ in evidence.channels()} == set(MatchChannel)
    assert not any(hasattr(evidence, name) for name in ("score", "similarity", "confidence"))
    # Cada canal guarda a própria medição: nenhuma escala é misturada com outra.
    assert evidence.geometry is not None and evidence.geometry.measurement is not None
    assert evidence.appearance is not None and evidence.appearance.measurement is not None
    assert evidence.geometry.measurement.bounds_iou != evidence.appearance.measurement.similarity


def test_the_pair_is_in_canonical_order_and_two_different_entities() -> None:
    with pytest.raises(ValueError, match="canonical order"):
        match_evidence(entity_a_ref=REF_B, entity_b_ref=REF_A)
    with pytest.raises(ValueError, match="itself"):
        match_evidence(entity_a_ref=REF_A, entity_b_ref=REF_A)


def test_entities_are_ordered_by_map_and_then_by_id() -> None:
    other_map = ref("entity--support-000001", semantic_map_id="semantic-map-0002")  # type: ignore[arg-type]

    evidence = match_evidence(
        entity_a_ref=REF_B,
        entity_b_ref=other_map,
        comparison_id=comparison_id_for(REF_B, other_map),
    )

    assert (evidence.entity_a_ref, evidence.entity_b_ref) == (REF_B, other_map)


def test_the_comparison_id_is_derived_from_the_pair() -> None:
    assert comparison_id_for(REF_A, REF_B) == comparison_id_for(REF_A, REF_B)
    assert comparison_id_for(REF_A, REF_B) != comparison_id_for(
        REF_A, ref("entity--support-000003")
    )
    with pytest.raises(ValueError, match="comparison_id"):
        match_evidence(comparison_id="comparison--tampered")


def test_only_channels_that_were_run_are_reported_and_none_means_not_evaluated() -> None:
    evidence = match_evidence()

    assert evidence.appearance is None
    assert [channel for channel, _ in evidence.channels()] == [MatchChannel.GEOMETRY]
    # Um canal não avaliado não vira "indisponível" nem apoio: ele simplesmente não existe.
    assert evidence.unavailable_channels() == ()
    assert evidence.supporting_channels() == (MatchChannel.GEOMETRY,)


def test_supporting_conflicting_and_unavailable_channels_are_reported_separately() -> None:
    evidence = match_evidence(
        geometry=geometry_evidence(EvidenceStatus.SUPPORTING),
        semantic=semantic_evidence(EvidenceStatus.CONFLICTING),
        appearance=AppearanceEvidence(
            policy=channel_policy(),
            unavailable=unavailable(UnavailableReason.MISSING_EVIDENCE, "no features"),
        ),
        temporal=temporal_evidence(EvidenceStatus.NEUTRAL),
    )

    assert evidence.supporting_channels() == (MatchChannel.GEOMETRY,)
    assert evidence.conflicting_channels() == (MatchChannel.SEMANTIC,)
    assert evidence.unavailable_channels() == (MatchChannel.APPEARANCE,)
    assert evidence.channel_status(MatchChannel.TEMPORAL) is EvidenceStatus.NEUTRAL


def test_supporting_and_conflicting_signals_keep_the_exact_finding_and_its_channel() -> None:
    evidence = match_evidence(
        semantic=semantic_evidence(EvidenceStatus.CONFLICTING),
        temporal=temporal_evidence(EvidenceStatus.NEUTRAL),
    )

    signals = {(channel, item.rule_id, item.status) for channel, item in evidence.signals()}

    assert signals == {
        (MatchChannel.GEOMETRY, "geometry-overlap", EvidenceStatus.SUPPORTING),
        (MatchChannel.SEMANTIC, "semantic-refinement", EvidenceStatus.CONFLICTING),
    }


def test_a_failed_gate_names_the_reason_the_comparison_was_blocked() -> None:
    blocked = tuple(
        GateResult(gate_id="same-geometric-map", passed=False, detail="map-0001 vs map-0002")
        if gate.gate_id == "same-geometric-map"
        else gate
        for gate in passed_gates()
    )

    evidence = match_evidence(
        gates=blocked,
        geometry=GeometryEvidence(
            policy=channel_policy(),
            unavailable=unavailable(UnavailableReason.BLOCKED_BY_GATE, "same-geometric-map"),
        ),
    )

    assert evidence.blocked
    assert [gate.gate_id for gate in evidence.failed_gates()] == ["same-geometric-map"]


def test_a_blocked_comparison_cannot_carry_available_evidence() -> None:
    blocked = (GateResult(gate_id="same-map-frame", passed=False, detail="map vs odom"),)

    with pytest.raises(ValueError, match="blocked"):
        match_evidence(gates=blocked, geometry=geometry_evidence())


def test_gates_are_unique_and_sorted() -> None:
    with pytest.raises(ValueError, match="gates"):
        match_evidence(gates=tuple(reversed(passed_gates())))
    with pytest.raises(ValueError, match="gate_id"):
        GateResult(gate_id=" ", passed=True, detail="x")


def test_the_evidence_names_the_gate_policy_that_produced_the_gate_results() -> None:
    with pytest.raises(ValueError, match="gate_policy_id"):
        MatchEvidenceProvenance(gate_policy_id="")


def test_a_channel_must_be_the_class_of_its_own_slot() -> None:
    with pytest.raises(TypeError, match="geometry"):
        match_evidence(geometry=appearance_evidence())


def test_match_evidence_is_immutable() -> None:
    evidence = match_evidence()

    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.geometry = None  # type: ignore[misc]
