"""Linking upstream statements to resolved entities through Entity Resolution's public API."""

from __future__ import annotations

import dataclasses

import pytest
from relation_resolution_fixture import ER_RUN, build_inputs

from contextmap.entity_resolution import (
    ResolvedEntity,
    ResolvedEntityReference,
    ResolvedEntitySet,
    materialize_resolved_entities,
)
from contextmap.semantic_mapping import Entity
from contextmap.sensor_association import SpatialObservationId
from contextmap.spatial_relations import (
    LinkFailure,
    RelationEvidenceStatus,
    RelationPredicate,
    StatementPolarity,
    UpstreamRelationStatement,
    UpstreamStatementRef,
    link_statements,
    observation_evidence_from_statements,
)

NAMES = "abcdef"


def _spatial(name: str) -> str:
    return f"spatial--run-a--frame-0120--region-{NAMES.index(name):04d}"


def _with_spatial(entity: Entity, spatial: str) -> Entity:
    links = dataclasses.replace(
        entity.evidence, spatial_observation_ids=(SpatialObservationId(spatial),)
    )
    return dataclasses.replace(entity, evidence=links)


def _resolved(shared: dict[str, str] | None = None) -> tuple[ResolvedEntitySet, dict[str, Entity]]:
    """The fixture's resolution, with a distinct spatial observation per source entity."""
    inputs = build_inputs()
    entities = {
        name: _with_spatial(entity, (shared or {}).get(name, _spatial(name)))
        for name, entity in inputs.entities.items()
    }
    materialization = materialize_resolved_entities(
        entities.values(),
        [item.decision for item in inputs.resolutions],
        resolution_run_id=ER_RUN,
    )
    return materialization.resolved, entities


def _statement(
    subject: str,
    text: str,
    obj: str,
    *,
    polarity: StatementPolarity = StatementPolarity.ASSERTS,
    statement_id: str = "statement-0001",
) -> UpstreamRelationStatement:
    return UpstreamRelationStatement(
        source=UpstreamStatementRef(
            source_run_id="perception-run-0001",
            statement_id=statement_id,
            physical_observation_id="frame-0120",
            producer="qwen-vl/prompt-v1",
        ),
        subject_spatial_observation_id=_spatial(subject),
        predicate_text=text,
        object_spatial_observation_id=_spatial(obj),
        polarity=polarity,
    )


def _reference(resolved: ResolvedEntity) -> ResolvedEntityReference:
    return ResolvedEntityReference(
        resolution_run_id=ER_RUN, resolved_entity_id=resolved.resolved_entity_id
    )


# --- a statement reaches the resolved entity through the evidence of its members ---


def test_a_statement_is_linked_to_the_resolved_entities_of_its_spatial_observations() -> None:
    resolved, entities = _resolved()
    result = link_statements([_statement("d", "next to", "e")], resolved=resolved)
    assert result.unlinked == ()
    (linked,) = result.linked
    assert linked.subject.entity_ref == _reference(resolved.resolved_of(entities["d"].reference))
    assert linked.object.entity_ref == _reference(resolved.resolved_of(entities["e"].reference))
    assert linked.subject.upstream_ref == _spatial("d")
    assert linked.predicate_text == "next to"
    assert linked.polarity is StatementPolarity.ASSERTS
    assert linked.source.statement_id == "statement-0001"


def test_the_link_records_the_evidence_it_went_through_and_the_members() -> None:
    resolved, entities = _resolved()
    (linked,) = link_statements([_statement("d", "next to", "e")], resolved=resolved).linked
    merged = resolved.resolved_of(entities["e"].reference)
    assert _spatial("e") in linked.object.linked_through
    assert "(members: e, f)" in linked.object.linked_through
    assert str(merged.resolved_entity_id) in linked.object.linked_through


def test_statements_from_different_members_of_one_merged_entity_share_it() -> None:
    resolved, entities = _resolved()
    merged = resolved.resolved_of(entities["e"].reference)
    assert {str(item.entity_ref.entity_id) for item in merged.members} >= {"e", "f"}
    statements = [
        _statement("e", "next to", "d", statement_id="s-e"),
        _statement("f", "next to", "d", statement_id="s-f"),
    ]
    linked = link_statements(statements, resolved=resolved).linked
    assert {item.subject.entity_ref for item in linked} == {_reference(merged)}
    entity_set = {item.subject.entity_ref for item in linked} | {
        item.object.entity_ref for item in linked
    }
    (evidence,) = observation_evidence_from_statements(linked, entities=entity_set).evidence
    assert evidence.status is RelationEvidenceStatus.SUPPORTS
    assert evidence.predicate is RelationPredicate.NEXT_TO
    assert len(evidence.statements) == 2


# --- what cannot be linked is reported, never approximated ---


def test_a_spatial_observation_no_entity_has_is_not_linked_and_says_why() -> None:
    resolved, _ = _resolved()
    unknown = dataclasses.replace(
        _statement("d", "next to", "e"), object_spatial_observation_id="spatial--nowhere"
    )
    result = link_statements([unknown], resolved=resolved)
    assert result.linked == ()
    (item,) = result.unlinked
    assert item.statement == unknown
    assert item.failures == (LinkFailure.NOT_IN_ANY_ENTITY,)
    assert "spatial--nowhere" in item.detail


def test_a_spatial_observation_in_several_resolved_entities_is_ambiguous() -> None:
    resolved, _ = _resolved(shared={"d": _spatial("e")})
    result = link_statements([_statement("e", "next to", "a")], resolved=resolved)
    assert result.linked == ()
    assert result.unlinked[0].failures == (LinkFailure.IN_SEVERAL_ENTITIES,)


def test_both_ends_in_one_resolved_entity_cannot_be_a_relation() -> None:
    resolved, _ = _resolved()
    result = link_statements([_statement("e", "next to", "f")], resolved=resolved)
    assert result.linked == ()
    assert result.unlinked[0].failures == (LinkFailure.BOTH_ENDS_IN_ONE_ENTITY,)


def test_every_failure_of_a_statement_is_listed_once_in_canonical_order() -> None:
    resolved, _ = _resolved()
    both_unknown = dataclasses.replace(
        _statement("d", "next to", "e"),
        subject_spatial_observation_id="spatial--a",
        object_spatial_observation_id="spatial--b",
    )
    (item,) = link_statements([both_unknown], resolved=resolved).unlinked
    assert item.failures == (LinkFailure.NOT_IN_ANY_ENTITY,)
    assert "spatial--a" in item.detail and "spatial--b" in item.detail


# --- determinism and the contract of the input ---


def test_the_result_does_not_depend_on_the_order_of_the_statements() -> None:
    resolved, _ = _resolved()
    statements = [
        _statement("d", "next to", "e", statement_id="s-1"),
        _statement("a", "above", "d", statement_id="s-2"),
        _statement("e", "next to", "f", statement_id="s-3"),
        dataclasses.replace(
            _statement("d", "next to", "e", statement_id="s-4"),
            object_spatial_observation_id="spatial--nowhere",
        ),
    ]
    forward = link_statements(statements, resolved=resolved)
    backward = link_statements(reversed(statements), resolved=resolved)
    assert forward == backward
    assert [item.source.statement_id for item in forward.linked] == ["s-1", "s-2"]
    assert [item.statement.source.statement_id for item in forward.unlinked] == ["s-3", "s-4"]


def test_an_upstream_statement_names_two_different_observations_and_a_wording() -> None:
    base = _statement("d", "next to", "e")
    with pytest.raises(ValueError, match="itself"):
        dataclasses.replace(base, object_spatial_observation_id=base.subject_spatial_observation_id)
    with pytest.raises(ValueError, match="predicate_text"):
        dataclasses.replace(base, predicate_text=" ")
    with pytest.raises(ValueError, match="subject_spatial_observation_id"):
        dataclasses.replace(base, subject_spatial_observation_id="")
