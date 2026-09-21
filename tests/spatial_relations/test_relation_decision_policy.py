"""The baseline decision policy: conservative, deterministic and structurally consistent."""

from __future__ import annotations

import dataclasses
import json

import pytest
from relation_builders import (
    CONVENTIONS_FINGERPRINT,
    entity_ref,
    evidence_provenance,
    make_candidate_set,
    make_evidence,
)

from contextmap.geometric_mapping import MapId
from contextmap.spatial_relations import (
    CONSERVATIVE_DECISION_POLICY_ID,
    TAXONOMY_VERSION,
    DecisionRule,
    EvidenceCaveat,
    EvidenceCaveatKind,
    EvidenceUse,
    Relation,
    RelationDecision,
    RelationDecisionResult,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceId,
    RelationEvidenceStatus,
    RelationPredicate,
    RelationState,
    RelationUncertaintyKind,
    decide_relations,
    decision_policy_fingerprint,
    decode_relation_decision,
    encode_relation_decision,
    relation_id_for,
)

SUPPORTS = RelationEvidenceStatus.SUPPORTS
CONFLICTS = RelationEvidenceStatus.CONFLICTS
AMBIGUOUS = RelationEvidenceStatus.AMBIGUOUS
UNAVAILABLE = RelationEvidenceStatus.UNAVAILABLE
GEOMETRY = RelationEvidenceChannel.GEOMETRY
CONTACT = RelationEvidenceChannel.CONTACT
P = RelationPredicate

WITHIN = EvidenceCaveat(kind=EvidenceCaveatKind.WITHIN_TOLERANCE, detail="near the threshold")
MISSING = EvidenceCaveat(kind=EvidenceCaveatKind.MISSING_INPUT, detail="no orientation")


def _evidence(
    subject: int,
    predicate: RelationPredicate,
    obj: int,
    status: RelationEvidenceStatus = SUPPORTS,
    channel: RelationEvidenceChannel = GEOMETRY,
) -> RelationEvidence:
    caveats: tuple[EvidenceCaveat, ...] = ()
    measurements = None
    if status is AMBIGUOUS:
        caveats = (WITHIN,)
    elif status is UNAVAILABLE:
        caveats = (MISSING,)
        measurements = ()
    return make_evidence(
        subject=subject,
        obj=obj,
        predicate=predicate,
        status=status,
        channel=channel,
        caveats=caveats,
        measurements=measurements,
        provenance=evidence_provenance(),
    )


def _decide(
    keys: list[tuple[int, RelationPredicate, int]], evidence: list[RelationEvidence]
) -> RelationDecisionResult:
    return decide_relations(make_candidate_set(*keys), evidence)


def _relation(
    result: RelationDecisionResult, subject: int, predicate: RelationPredicate, obj: int
) -> Relation:
    wanted = relation_id_for(
        subject_entity_ref=entity_ref(subject),
        predicate=predicate,
        object_entity_ref=entity_ref(obj),
    )
    return next(item for item in result.relations if item.relation_id == wanted)


def _decision(result: RelationDecisionResult, relation: Relation) -> RelationDecision:
    return next(item for item in result.decisions if item.relation_id == relation.relation_id)


def _states(result: RelationDecisionResult) -> dict[tuple[int, str, int], RelationState]:
    return {
        (
            int(item.subject_entity_ref.resolved_entity_id[-4:]),
            item.predicate.value,
            int(item.object_entity_ref.resolved_entity_id[-4:]),
        ): item.state
        for item in result.relations
    }


# --- states from evidence ---


def test_supporting_evidence_supports_the_relation_with_a_full_trace() -> None:
    evidence = _evidence(1, P.ABOVE, 2)
    result = _decide([(1, P.ABOVE, 2)], [evidence])
    relation = _relation(result, 1, P.ABOVE, 2)
    assert relation.state is RelationState.SUPPORTED
    assert relation.relation_evidence_refs == (evidence.evidence_id,)
    assert relation.uncertainty == ()
    decision = _decision(result, relation)
    assert decision.rule is DecisionRule.CHANNELS_SUPPORT
    assert decision.deciding_evidence_refs == (evidence.evidence_id,)
    assert decision.ignored == ()


def test_conflicting_evidence_rejects_the_relation_and_stays_distinguishable() -> None:
    evidence = _evidence(1, P.ABOVE, 2, CONFLICTS)
    result = _decide([(1, P.ABOVE, 2)], [evidence])
    relation = _relation(result, 1, P.ABOVE, 2)
    assert relation.state is RelationState.REJECTED
    assert relation.relation_evidence_refs == (evidence.evidence_id,)
    assert relation.uncertainty == ()
    assert _decision(result, relation).rule is DecisionRule.CHANNELS_CONFLICT


def test_undecided_evidence_leaves_the_relation_unresolved_not_rejected() -> None:
    for status in (AMBIGUOUS, UNAVAILABLE):
        evidence = _evidence(1, P.ABOVE, 2, status)
        result = _decide([(1, P.ABOVE, 2)], [evidence])
        relation = _relation(result, 1, P.ABOVE, 2)
        assert relation.state is RelationState.UNRESOLVED, status
        assert relation.uncertainty[0].kind is RelationUncertaintyKind.INSUFFICIENT_EVIDENCE
        assert relation.uncertainty[0].evidence_refs == (evidence.evidence_id,)
        decision = _decision(result, relation)
        assert decision.rule is DecisionRule.NO_DECISIVE_EVIDENCE
        assert [(item.evidence_id, item.status) for item in decision.ignored] == [
            (evidence.evidence_id, status)
        ]


def test_a_candidate_without_any_evidence_is_unresolved_with_no_trace() -> None:
    result = _decide([(1, P.ON_TOP_OF, 2)], [])
    relation = _relation(result, 1, P.ON_TOP_OF, 2)
    assert relation.state is RelationState.UNRESOLVED
    assert relation.relation_evidence_refs == ()
    assert relation.uncertainty[0].kind is RelationUncertaintyKind.INSUFFICIENT_EVIDENCE


def test_channels_that_disagree_keep_the_conflict_visible_and_stay_unresolved() -> None:
    geometry = _evidence(1, P.ON_TOP_OF, 2, SUPPORTS, GEOMETRY)
    contact = _evidence(1, P.ON_TOP_OF, 2, CONFLICTS, CONTACT)
    result = _decide([(1, P.ON_TOP_OF, 2)], [geometry, contact])
    relation = _relation(result, 1, P.ON_TOP_OF, 2)
    assert relation.state is RelationState.UNRESOLVED
    assert set(relation.relation_evidence_refs) == {geometry.evidence_id, contact.evidence_id}
    (uncertainty,) = relation.uncertainty
    assert uncertainty.kind is RelationUncertaintyKind.CONFLICTING_EVIDENCE
    assert set(uncertainty.evidence_refs) == {geometry.evidence_id, contact.evidence_id}
    decision = _decision(result, relation)
    assert decision.rule is DecisionRule.CHANNELS_DISAGREE
    assert set(decision.deciding_evidence_refs) == {geometry.evidence_id, contact.evidence_id}


def test_an_undecided_channel_is_not_a_negative_vote() -> None:
    geometry = _evidence(1, P.ON_TOP_OF, 2, SUPPORTS, GEOMETRY)
    contact = _evidence(1, P.ON_TOP_OF, 2, AMBIGUOUS, CONTACT)
    result = _decide([(1, P.ON_TOP_OF, 2)], [geometry, contact])
    relation = _relation(result, 1, P.ON_TOP_OF, 2)
    assert relation.state is RelationState.SUPPORTED
    assert set(relation.relation_evidence_refs) == {geometry.evidence_id, contact.evidence_id}
    decision = _decision(result, relation)
    assert decision.deciding_evidence_refs == (geometry.evidence_id,)
    assert [item.evidence_id for item in decision.ignored] == [contact.evidence_id]


def test_agreeing_channels_support_and_reject_together() -> None:
    supports = _decide(
        [(1, P.ON_TOP_OF, 2)],
        [
            _evidence(1, P.ON_TOP_OF, 2, SUPPORTS, GEOMETRY),
            _evidence(1, P.ON_TOP_OF, 2, SUPPORTS, CONTACT),
        ],
    )
    rejects = _decide(
        [(1, P.ON_TOP_OF, 2)],
        [
            _evidence(1, P.ON_TOP_OF, 2, CONFLICTS, GEOMETRY),
            _evidence(1, P.ON_TOP_OF, 2, CONFLICTS, CONTACT),
        ],
    )
    assert _relation(supports, 1, P.ON_TOP_OF, 2).state is RelationState.SUPPORTED
    assert _relation(rejects, 1, P.ON_TOP_OF, 2).state is RelationState.REJECTED


# --- symmetry and inverse ---


def test_a_symmetric_relation_is_generated_in_the_other_direction() -> None:
    evidence = _evidence(1, P.NEXT_TO, 2)
    result = _decide([(1, P.NEXT_TO, 2)], [evidence])
    twin = _relation(result, 2, P.NEXT_TO, 1)
    source = _relation(result, 1, P.NEXT_TO, 2)
    assert twin.state is source.state is RelationState.SUPPORTED
    assert twin.derived_from == source.relation_id
    assert source.derived_from is None
    assert twin.relation_evidence_refs == source.relation_evidence_refs
    assert _decision(result, twin).rule is DecisionRule.DERIVED_BY_SYMMETRY


def test_an_inverse_relation_is_generated_with_the_same_state_and_evidence() -> None:
    pairs = {
        P.ABOVE: P.BELOW,
        P.IN_FRONT_OF: P.BEHIND,
        P.INSIDE: P.CONTAINS,
    }
    for predicate, inverse in pairs.items():
        for status, state in (
            (SUPPORTS, RelationState.SUPPORTED),
            (CONFLICTS, RelationState.REJECTED),
            (AMBIGUOUS, RelationState.UNRESOLVED),
        ):
            evidence = _evidence(1, predicate, 2, status)
            result = _decide([(1, predicate, 2)], [evidence])
            derived = _relation(result, 2, inverse, 1)
            source = _relation(result, 1, predicate, 2)
            assert derived.state is source.state is state, (predicate, status)
            assert derived.derived_from == source.relation_id
            assert derived.relation_evidence_refs == source.relation_evidence_refs
            assert derived.uncertainty == source.uncertainty
            assert _decision(result, derived).rule is DecisionRule.DERIVED_FROM_INVERSE


def test_a_directed_predicate_without_an_inverse_generates_nothing() -> None:
    result = _decide([(1, P.ON_TOP_OF, 2)], [_evidence(1, P.ON_TOP_OF, 2)])
    assert [item.predicate for item in result.relations] == [P.ON_TOP_OF]


def test_mutual_support_of_a_directed_predicate_is_a_contradiction_not_two_truths() -> None:
    forward = _evidence(1, P.ABOVE, 2)
    backward = _evidence(2, P.ABOVE, 1)
    result = _decide([(1, P.ABOVE, 2), (2, P.ABOVE, 1)], [forward, backward])
    for subject, obj in ((1, 2), (2, 1)):
        relation = _relation(result, subject, P.ABOVE, obj)
        assert relation.state is RelationState.UNRESOLVED
        (uncertainty,) = relation.uncertainty
        assert uncertainty.kind is RelationUncertaintyKind.INCONSISTENT_STRUCTURE
        assert relation.relation_evidence_refs
        decision = _decision(result, relation)
        assert decision.rule is DecisionRule.STRUCTURAL_INCONSISTENCY
        assert decision.deciding_evidence_refs == relation.relation_evidence_refs
    below = _relation(result, 2, P.BELOW, 1)
    assert below.state is RelationState.UNRESOLVED
    assert "contradict" in _relation(result, 1, P.ABOVE, 2).uncertainty[0].detail


def test_support_in_one_direction_and_rejection_in_the_other_is_consistent() -> None:
    result = _decide(
        [(1, P.ABOVE, 2), (2, P.ABOVE, 1)],
        [_evidence(1, P.ABOVE, 2), _evidence(2, P.ABOVE, 1, CONFLICTS)],
    )
    assert _relation(result, 1, P.ABOVE, 2).state is RelationState.SUPPORTED
    assert _relation(result, 2, P.ABOVE, 1).state is RelationState.REJECTED
    assert _relation(result, 2, P.BELOW, 1).state is RelationState.SUPPORTED
    assert _relation(result, 1, P.BELOW, 2).state is RelationState.REJECTED


def test_a_symmetric_predicate_evaluated_both_ways_must_agree() -> None:
    disagree = _decide(
        [(1, P.NEXT_TO, 2), (2, P.NEXT_TO, 1)],
        [_evidence(1, P.NEXT_TO, 2), _evidence(2, P.NEXT_TO, 1, CONFLICTS)],
    )
    for subject, obj in ((1, 2), (2, 1)):
        relation = _relation(disagree, subject, P.NEXT_TO, obj)
        assert relation.state is RelationState.UNRESOLVED
        assert relation.uncertainty[0].kind is RelationUncertaintyKind.INCONSISTENT_STRUCTURE
    agree = _decide(
        [(1, P.NEXT_TO, 2), (2, P.NEXT_TO, 1)],
        [_evidence(1, P.NEXT_TO, 2), _evidence(2, P.NEXT_TO, 1)],
    )
    assert [item.state for item in agree.relations] == [RelationState.SUPPORTED] * 2
    assert all(item.derived_from is None for item in agree.relations)


# --- determinism and traceability ---


def _scene() -> tuple[list[tuple[int, RelationPredicate, int]], list[RelationEvidence]]:
    keys = [
        (1, P.NEXT_TO, 2),
        (2, P.ABOVE, 1),
        (1, P.ABOVE, 2),
        (3, P.ON_TOP_OF, 1),
        (1, P.INSIDE, 3),
    ]
    evidence = [
        _evidence(1, P.NEXT_TO, 2),
        _evidence(2, P.ABOVE, 1),
        _evidence(1, P.ABOVE, 2, CONFLICTS),
        _evidence(3, P.ON_TOP_OF, 1, SUPPORTS, GEOMETRY),
        _evidence(3, P.ON_TOP_OF, 1, CONFLICTS, CONTACT),
        _evidence(1, P.INSIDE, 3, AMBIGUOUS),
    ]
    return keys, evidence


def test_the_same_inputs_give_the_same_result_whatever_their_order() -> None:
    keys, evidence = _scene()
    first = _decide(keys, evidence)
    second = _decide(list(reversed(keys)), list(reversed(evidence)))
    assert first == second
    assert first == _decide(keys, evidence)


def test_every_supported_relation_traces_to_the_evidence_that_decided_it() -> None:
    keys, evidence = _scene()
    result = _decide(keys, evidence)
    known = {item.evidence_id: item for item in evidence}
    for relation in result.relations:
        assert set(relation.relation_evidence_refs) <= set(known)
        decision = _decision(result, relation)
        if relation.state is RelationState.SUPPORTED:
            assert relation.relation_evidence_refs
            assert decision.deciding_evidence_refs
            assert all(known[ref].status is SUPPORTS for ref in decision.deciding_evidence_refs)
        if relation.state is RelationState.REJECTED:
            assert all(known[ref].status is CONFLICTS for ref in decision.deciding_evidence_refs)


def test_there_is_one_decision_per_relation_in_canonical_order() -> None:
    keys, evidence = _scene()
    result = _decide(keys, evidence)
    assert [item.relation_id for item in result.decisions] == [
        item.relation_id for item in result.relations
    ]
    ids = [str(item.relation_id) for item in result.relations]
    assert len(set(ids)) == len(ids)
    order = [
        (
            item.subject_entity_ref.resolved_entity_id,
            item.predicate.value,
            item.object_entity_ref.resolved_entity_id,
        )
        for item in result.relations
    ]
    assert order == sorted(order)
    with pytest.raises(ValueError, match="decision"):
        RelationDecisionResult(relations=result.relations, decisions=result.decisions[:-1])


def test_the_relations_record_the_policy_and_the_taxonomy() -> None:
    result = _decide([(1, P.NEXT_TO, 2)], [_evidence(1, P.NEXT_TO, 2)])
    provenance = result.relations[0].provenance
    assert provenance.decision_policy_id == CONSERVATIVE_DECISION_POLICY_ID
    assert provenance.taxonomy_version == TAXONOMY_VERSION
    assert provenance.configuration_fingerprint == decision_policy_fingerprint()
    assert decision_policy_fingerprint().startswith("sha256:")


# --- input validation ---


def test_evidence_for_a_pair_that_is_not_a_candidate_is_refused() -> None:
    with pytest.raises(ValueError, match="candidate"):
        _decide([(1, P.NEXT_TO, 2)], [_evidence(1, P.ABOVE, 2)])


def test_the_same_evidence_twice_is_refused() -> None:
    evidence = _evidence(1, P.NEXT_TO, 2)
    with pytest.raises(ValueError, match="twice"):
        _decide([(1, P.NEXT_TO, 2)], [evidence, evidence])


def test_evidence_from_another_map_frame_or_taxonomy_is_refused() -> None:
    candidates = make_candidate_set((1, P.NEXT_TO, 2))
    base = _evidence(1, P.NEXT_TO, 2)
    other_map = dataclasses.replace(
        base,
        provenance=dataclasses.replace(base.provenance, geometric_map_id=MapId("map-0002")),
    )
    other_frame = dataclasses.replace(
        base, provenance=dataclasses.replace(base.provenance, map_frame="odom")
    )
    other_taxonomy = dataclasses.replace(
        base, provenance=dataclasses.replace(base.provenance, taxonomy_version="other-taxonomy")
    )
    other_axes = dataclasses.replace(
        base,
        provenance=dataclasses.replace(
            base.provenance, frame_conventions_fingerprint="sha256:another"
        ),
    )
    for changed in (other_map, other_frame, other_taxonomy, other_axes):
        with pytest.raises(ValueError, match="lineage"):
            decide_relations(candidates, [changed])
    same_axes = dataclasses.replace(
        base,
        provenance=dataclasses.replace(
            base.provenance, frame_conventions_fingerprint=CONVENTIONS_FINGERPRINT
        ),
    )
    assert decide_relations(candidates, [same_axes]).relations


# --- serialization ---


def test_decisions_round_trip_through_json_and_are_revalidated() -> None:
    keys, evidence = _scene()
    result = _decide(keys, evidence)
    for decision in result.decisions:
        record = json.loads(json.dumps(encode_relation_decision(decision), sort_keys=True))
        assert decode_relation_decision(record) == decision
    unresolved = next(item for item in result.decisions if item.ignored)
    record = encode_relation_decision(unresolved)
    record["ignored"] = [
        {
            "evidence_id": "evidence--x",
            "channel": "geometry",
            "status": "supports",
            "contradicts_relation": False,
        }
    ]
    with pytest.raises(ValueError, match="ignored"):
        decode_relation_decision(record)
    record = encode_relation_decision(result.decisions[0])
    del record["rule"]
    with pytest.raises(ValueError, match="rule"):
        decode_relation_decision(record)


def test_evidence_that_decided_cannot_also_be_ignored() -> None:
    decision = _decide([(1, P.ABOVE, 2)], [_evidence(1, P.ABOVE, 2)]).decisions[0]
    overlap = EvidenceUse(
        evidence_id=decision.deciding_evidence_refs[0], channel=GEOMETRY, status=AMBIGUOUS
    )
    with pytest.raises(ValueError, match="both"):
        dataclasses.replace(decision, ignored=(overlap,))
    with pytest.raises(ValueError, match="ignored"):
        EvidenceUse(
            evidence_id=decision.deciding_evidence_refs[0], channel=GEOMETRY, status=SUPPORTS
        )
    with pytest.raises(ValueError, match="contradict"):
        EvidenceUse(
            evidence_id=RelationEvidenceId("evidence--x"),
            channel=GEOMETRY,
            status=AMBIGUOUS,
            contradicts_relation=True,
        )


# --- against the real evaluators ---


def test_the_real_chain_decides_a_crate_on_a_floor() -> None:
    from relation_scene import Scene

    from contextmap.spatial_relations import (
        AxisDirection,
        CandidatePolicy,
        FrameConventions,
        GeometricPredicatePolicy,
        evaluate_geometric_candidates,
        generate_relation_candidates,
    )

    scene = Scene()
    scene.add_box("floor", (0.0, 0.0, 0.0), (4.0, 4.0, 0.1))
    scene.add_box("crate", (1.0, 1.0, 0.1), (2.0, 2.0, 1.1))
    entities = {entity_ref(1): scene.geometry("floor"), entity_ref(2): scene.geometry("crate")}
    conventions = FrameConventions(
        map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
    )
    candidates = generate_relation_candidates(
        entities,
        policy=CandidatePolicy(
            predicates=(P.NEXT_TO, P.ABOVE),
            proximity_radius_m=0.6,
            directional_radius_m=2.0,
        ),
        conventions=conventions,
    )
    evidence = evaluate_geometric_candidates(
        candidates,
        entities=entities,
        policy=GeometricPredicatePolicy(
            boundary_tolerance_m=0.02,
            next_to_max_gap_m=0.5,
            adjacent_penetration_m=0.05,
            containment_slack_m=0.05,
            directional_overlap_fraction=0.5,
        ),
        conventions=conventions,
    )
    result = decide_relations(candidates, evidence)
    states = _states(result)
    assert states[(2, "above", 1)] is RelationState.SUPPORTED
    assert states[(1, "below", 2)] is RelationState.SUPPORTED
    assert states[(1, "next_to", 2)] is RelationState.SUPPORTED
    assert states[(2, "next_to", 1)] is RelationState.SUPPORTED
    assert (1, "above", 2) not in states
