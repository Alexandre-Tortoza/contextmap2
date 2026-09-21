"""Observation-level evidence: upstream statements as one channel that never overrides geometry."""

from __future__ import annotations

import dataclasses
import json

import pytest
from relation_builders import (
    entity_ref,
    evidence_provenance,
    make_candidate_set,
    make_evidence,
)

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.spatial_relations import (
    CONFLICTING_STATEMENTS_CAVEAT,
    OBSERVATION_RULE_ID,
    TAXONOMY_VERSION,
    EndpointLink,
    EvidenceCaveat,
    EvidenceCaveatKind,
    IncompatibleLineageError,
    ObservationRelationStatement,
    Relation,
    RelationDecision,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceStatus,
    RelationPredicate,
    RelationState,
    StatementPolarity,
    UnlinkedEndpointError,
    UpstreamStatementRef,
    canonical_predicate,
    decide_relations,
    decode_relation_evidence,
    encode_relation_evidence,
    observation_evidence_from_statements,
)

SUPPORTS = RelationEvidenceStatus.SUPPORTS
CONFLICTS = RelationEvidenceStatus.CONFLICTS
AMBIGUOUS = RelationEvidenceStatus.AMBIGUOUS
P = RelationPredicate
ASSERTS = StatementPolarity.ASSERTS
DENIES = StatementPolarity.DENIES
ENTITIES = frozenset(entity_ref(number) for number in range(1, 5))


def _link(number: int, upstream: str | None = None) -> EndpointLink:
    return EndpointLink(
        upstream_ref=upstream or f"region-{number:04d}",
        entity_ref=entity_ref(number),
        linked_through=f"fused--support-{number:06d}",
    )


def _statement(
    subject: int,
    text: str,
    obj: int,
    polarity: StatementPolarity = ASSERTS,
    *,
    statement_id: str = "statement-0001",
    observation: str = "frame-0120",
) -> ObservationRelationStatement:
    return ObservationRelationStatement(
        source=UpstreamStatementRef(
            source_run_id="perception-run-0001",
            statement_id=statement_id,
            physical_observation_id=observation,
            producer="qwen-vl/qwen3/prompt-v1",
        ),
        subject=_link(subject),
        predicate_text=text,
        object=_link(obj),
        polarity=polarity,
    )


def _evidence(
    *statements: ObservationRelationStatement,
    entities: frozenset[ResolvedEntityReference] = ENTITIES,
) -> tuple[RelationEvidence, ...]:
    return observation_evidence_from_statements(statements, entities=entities).evidence


# --- from statement to evidence ---


def test_an_asserting_statement_becomes_supporting_observation_evidence() -> None:
    statement = _statement(1, "on top of", 2)
    (evidence,) = _evidence(statement)
    assert evidence.channel is RelationEvidenceChannel.OBSERVATION
    assert evidence.status is SUPPORTS
    assert evidence.subject_entity_ref == entity_ref(1)
    assert evidence.predicate is P.ON_TOP_OF
    assert evidence.object_entity_ref == entity_ref(2)
    assert evidence.statements == (statement,)
    assert evidence.geometry == ()
    counts = {item.name: item.value for item in evidence.measurements}
    assert counts == {"asserting_statements": 1.0, "denying_statements": 0.0}


def test_a_denying_statement_becomes_conflicting_evidence() -> None:
    (evidence,) = _evidence(_statement(1, "on top of", 2, DENIES))
    assert evidence.status is CONFLICTS


def test_the_evidence_traces_to_the_exact_upstream_statement_and_links() -> None:
    (evidence,) = _evidence(_statement(1, "next to", 2, observation="frame-0777"))
    (statement,) = evidence.statements
    assert statement.source.source_run_id == "perception-run-0001"
    assert statement.source.physical_observation_id == "frame-0777"
    assert statement.subject.linked_through == "fused--support-000001"
    assert statement.object.upstream_ref == "region-0002"
    assert evidence.provenance.rule_id == OBSERVATION_RULE_ID
    assert evidence.provenance.taxonomy_version == TAXONOMY_VERSION
    assert evidence.provenance.map_frame is None
    assert evidence.provenance.geometric_map_id is None


def test_a_derived_wording_is_normalized_to_the_evaluated_direction() -> None:
    cases = {
        "below": (P.ABOVE, 2, 1),
        "behind": (P.IN_FRONT_OF, 2, 1),
        "contains": (P.INSIDE, 2, 1),
        "above": (P.ABOVE, 1, 2),
    }
    for text, (predicate, subject, obj) in cases.items():
        (evidence,) = _evidence(_statement(1, text, 2))
        assert (evidence.subject_entity_ref, evidence.predicate, evidence.object_entity_ref) == (
            entity_ref(subject),
            predicate,
            entity_ref(obj),
        ), text
    (denied,) = _evidence(_statement(1, "below", 2, DENIES))
    assert (denied.predicate, denied.status) == (P.ABOVE, CONFLICTS)


def test_a_symmetric_predicate_is_recorded_in_canonical_order_whichever_way_it_was_said() -> None:
    forward = _evidence(_statement(1, "next to", 2))
    backward = _evidence(_statement(2, "next to", 1))
    for (evidence,) in (forward, backward):
        assert evidence.subject_entity_ref == entity_ref(1)
        assert evidence.object_entity_ref == entity_ref(2)


def test_wordings_map_only_to_the_canonical_names_without_synonym_expansion() -> None:
    for text, predicate in {
        "next to": P.NEXT_TO,
        "NEXT_TO": P.NEXT_TO,
        "in-front-of": P.IN_FRONT_OF,
        "  On Top Of ": P.ON_TOP_OF,
        "leaning against": P.LEANING_AGAINST,
        "touching": P.TOUCHING,
        "intersects": P.INTERSECTS,
    }.items():
        assert canonical_predicate(text) is predicate, text
    for text in ("near", "beside", "under", "next", "on", "", "resting on"):
        assert canonical_predicate(text) is None, text


def test_a_statement_with_an_unmapped_wording_is_reported_not_turned_into_a_relation() -> None:
    unmapped = _statement(1, "beside", 2)
    result = observation_evidence_from_statements([unmapped], entities=ENTITIES)
    assert result.evidence == ()
    assert result.unmapped_statements == (unmapped,)


def test_statements_about_the_same_candidate_are_merged_with_their_counts() -> None:
    first = _statement(1, "on top of", 2, statement_id="statement-0001")
    second = _statement(1, "on top of", 2, statement_id="statement-0002", observation="frame-0121")
    (evidence,) = _evidence(second, first)
    assert evidence.status is SUPPORTS
    assert [item.source.statement_id for item in evidence.statements] == [
        "statement-0001",
        "statement-0002",
    ]
    counts = {item.name: item.value for item in evidence.measurements}
    assert counts["asserting_statements"] == 2.0


def test_statements_that_disagree_make_ambiguous_evidence_that_keeps_both() -> None:
    asserting = _statement(1, "on top of", 2, statement_id="statement-0001")
    denying = _statement(1, "on top of", 2, DENIES, statement_id="statement-0002")
    (evidence,) = _evidence(asserting, denying)
    assert evidence.status is AMBIGUOUS
    assert len(evidence.statements) == 2
    assert EvidenceCaveatKind.CONFLICTING_STATEMENTS in {item.kind for item in evidence.caveats}
    assert CONFLICTING_STATEMENTS_CAVEAT in {item.detail for item in evidence.caveats}
    counts = {item.name: item.value for item in evidence.measurements}
    assert (counts["asserting_statements"], counts["denying_statements"]) == (1.0, 1.0)


def test_one_evidence_record_per_candidate_in_canonical_order() -> None:
    evidence = _evidence(
        _statement(2, "above", 3, statement_id="s-b"),
        _statement(1, "next to", 2, statement_id="s-a"),
        _statement(1, "on top of", 3, statement_id="s-c"),
    )
    keys = [(item.subject_entity_ref.resolved_entity_id, item.predicate.value) for item in evidence]
    assert keys == sorted(keys)
    assert len({item.evidence_id for item in evidence}) == 3


def test_the_result_does_not_depend_on_the_order_of_the_statements() -> None:
    statements = [
        _statement(1, "on top of", 2, statement_id="s-1"),
        _statement(2, "below", 3, DENIES, statement_id="s-2"),
        _statement(3, "next to", 1, statement_id="s-3"),
    ]
    forward = observation_evidence_from_statements(statements, entities=ENTITIES)
    backward = observation_evidence_from_statements(reversed(statements), entities=ENTITIES)
    assert forward == backward


# --- linkage and lineage ---


def test_an_end_that_is_not_in_the_selected_entity_set_is_refused() -> None:
    with pytest.raises(UnlinkedEndpointError, match="resolved-0009"):
        _evidence(_statement(1, "on top of", 9))
    outsider = frozenset({entity_ref(1)})
    with pytest.raises(UnlinkedEndpointError, match="resolved-0002"):
        _evidence(_statement(1, "on top of", 2), entities=outsider)


def test_an_end_tied_to_another_resolution_artifact_is_refused() -> None:
    foreign = dataclasses.replace(
        _statement(1, "on top of", 2),
        object=EndpointLink(
            upstream_ref="region-0002",
            entity_ref=entity_ref(2, run="resolution-run-0002"),
            linked_through="fused--support-000002",
        ),
    )
    with pytest.raises(IncompatibleLineageError, match="resolution"):
        _evidence(foreign)


def test_a_link_without_its_evidence_and_a_statement_of_one_entity_are_invalid() -> None:
    with pytest.raises(ValueError, match="linked_through"):
        EndpointLink(upstream_ref="region-1", entity_ref=entity_ref(1), linked_through=" ")
    with pytest.raises(ValueError, match="upstream_ref"):
        EndpointLink(upstream_ref="", entity_ref=entity_ref(1), linked_through="fused--x")
    with pytest.raises(ValueError, match="itself"):
        _statement(1, "next to", 1)
    with pytest.raises(ValueError, match="producer"):
        UpstreamStatementRef(
            source_run_id="run", statement_id="s", physical_observation_id="f", producer=""
        )


# --- the contract of observation evidence ---


def test_observation_evidence_needs_its_statements_and_carries_no_geometry() -> None:
    (evidence,) = _evidence(_statement(1, "on top of", 2))
    with pytest.raises(ValueError, match="statement"):
        dataclasses.replace(evidence, statements=())
    with pytest.raises(ValueError, match="geometry"):
        dataclasses.replace(evidence, geometry=make_evidence().geometry)


def test_geometric_evidence_still_needs_its_geometry_and_carries_no_statements() -> None:
    (observation,) = _evidence(_statement(1, "next to", 2))
    geometric = make_evidence()
    with pytest.raises(ValueError, match="statement"):
        dataclasses.replace(geometric, statements=observation.statements)
    with pytest.raises(ValueError, match="geometry"):
        dataclasses.replace(geometric, geometry=())


def test_observation_evidence_round_trips_through_json() -> None:
    evidence = _evidence(
        _statement(1, "on top of", 2, statement_id="s-1"),
        _statement(1, "on top of", 2, DENIES, statement_id="s-2"),
    )
    for item in evidence:
        record = json.loads(json.dumps(encode_relation_evidence(item), sort_keys=True))
        assert decode_relation_evidence(record) == item


# --- reconciliation: never overriding geometry ---


def _decide(
    predicate: RelationPredicate,
    *,
    geometry: RelationEvidenceStatus | None,
    observation: StatementPolarity | None,
) -> tuple[Relation, RelationDecision]:
    records: list[RelationEvidence] = []
    if geometry is not None:
        caveats: tuple[EvidenceCaveat, ...] = ()
        if geometry is AMBIGUOUS:
            caveats = (EvidenceCaveat(kind=EvidenceCaveatKind.WITHIN_TOLERANCE, detail="close"),)
        records.append(
            make_evidence(
                predicate=predicate,
                status=geometry,
                caveats=caveats,
                provenance=evidence_provenance(),
            )
        )
    if observation is not None:
        records.extend(_evidence(_statement(1, predicate.value, 2, observation)))
    result = decide_relations(make_candidate_set((1, predicate, 2)), records)
    relation = next(
        item
        for item in result.relations
        if (item.subject_entity_ref, item.predicate)
        == (
            entity_ref(1),
            predicate,
        )
    )
    decision = next(item for item in result.decisions if item.relation_id == relation.relation_id)
    return relation, decision


def test_an_observation_alone_never_establishes_a_relation() -> None:
    for polarity in (ASSERTS, DENIES):
        relation, decision = _decide(P.ON_TOP_OF, geometry=None, observation=polarity)
        assert relation.state is RelationState.UNRESOLVED, polarity
        assert decision.deciding_evidence_refs == ()
        (ignored,) = decision.ignored
        assert ignored.channel is RelationEvidenceChannel.OBSERVATION
        assert "observation" in decision.detail


def test_an_agreeing_observation_changes_nothing_and_a_disagreeing_one_is_recorded() -> None:
    plain, _ = _decide(P.NEXT_TO, geometry=SUPPORTS, observation=None)
    agreeing, agree_decision = _decide(P.NEXT_TO, geometry=SUPPORTS, observation=ASSERTS)
    disagreeing, disagree_decision = _decide(P.NEXT_TO, geometry=SUPPORTS, observation=DENIES)
    for relation in (plain, agreeing, disagreeing):
        assert relation.state is RelationState.SUPPORTED
    assert [item.contradicts_relation for item in agree_decision.ignored] == [False]
    assert [item.contradicts_relation for item in disagree_decision.ignored] == [True]
    assert len(disagreeing.relation_evidence_refs) == 2
    assert disagree_decision.deciding_evidence_refs == plain.relation_evidence_refs


def test_a_supporting_observation_cannot_overturn_a_rejection() -> None:
    relation, decision = _decide(P.NEXT_TO, geometry=CONFLICTS, observation=ASSERTS)
    assert relation.state is RelationState.REJECTED
    assert [item.contradicts_relation for item in decision.ignored] == [True]


def test_an_observation_cannot_break_the_tie_of_undecided_geometry() -> None:
    relation, decision = _decide(P.NEXT_TO, geometry=AMBIGUOUS, observation=ASSERTS)
    assert relation.state is RelationState.UNRESOLVED
    assert {item.channel for item in decision.ignored} == {
        RelationEvidenceChannel.GEOMETRY,
        RelationEvidenceChannel.OBSERVATION,
    }


def test_missing_observation_evidence_is_neutral() -> None:
    supported, _ = _decide(P.ABOVE, geometry=SUPPORTS, observation=None)
    rejected, _ = _decide(P.ABOVE, geometry=CONFLICTS, observation=None)
    assert supported.state is RelationState.SUPPORTED
    assert rejected.state is RelationState.REJECTED
