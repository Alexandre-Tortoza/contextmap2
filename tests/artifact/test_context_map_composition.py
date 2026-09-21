"""Composition of geometry, entities and relations: identity scopes and reference semantics."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest
from context_map_builders import (
    CONTEXT_MAP_ID,
    ENTITY_RESOLUTION_ARTIFACT_ID,
    GEOMETRIC_MAP_ID,
    POINT_COUNT,
    context_map,
    entity,
    entity_capabilities,
    entity_reference,
    geometry_ref,
    hypothesis,
    metadata,
    populated_map,
    relation,
    semantic_state,
    upstream_record,
)

from contextmap.artifact import (
    AmbiguityStatus,
    ContextEntity,
    ContextRelation,
    ForeignContextEntityReferenceError,
    ReferenceIntegrityError,
    RelationState,
    UnknownContextEntityError,
    context_map_from_record,
    context_map_to_record,
)
from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId, geometry_id_for

# --- entities: identity scope and authoritative geometry ---------------------------------------


def test_an_entity_is_reached_through_a_reference_scoped_to_its_map() -> None:
    result = populated_map()

    found = result.entity(entity_reference("entity-0002"))

    assert found.entity_id == "entity-0002"


def test_a_reference_to_another_map_is_refused_even_when_the_id_exists() -> None:
    result = populated_map()
    foreign = entity_reference("entity-0001", context_map_id="context-map--other--0001")

    with pytest.raises(ForeignContextEntityReferenceError, match="other"):
        result.entity(foreign)


def test_an_unknown_entity_is_distinct_from_a_foreign_reference() -> None:
    with pytest.raises(UnknownContextEntityError, match="entity-9999"):
        populated_map().entity(entity_reference("entity-9999"))


def test_entities_are_unique_and_sorted_by_id() -> None:
    first, second = entity("entity-0001"), entity("entity-0002")
    meta = metadata(capabilities=entity_capabilities())

    assert context_map(metadata=meta, entities=(first, second)).entities == (first, second)
    with pytest.raises(ValueError, match="sorted"):
        context_map(metadata=meta, entities=(second, first))
    with pytest.raises(ValueError, match="unique"):
        context_map(metadata=meta, entities=(first, first))


def test_every_entity_needs_authoritative_geometry_support() -> None:
    with pytest.raises(ValueError, match="geometry_refs"):
        entity(geometry=())


def test_geometry_references_are_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        entity(geometry=(2, 1))
    with pytest.raises(ValueError, match="unique"):
        entity(geometry=(1, 1))


def test_entity_geometry_must_belong_to_the_referenced_geometric_map() -> None:
    foreign = geometry_ref(0, map_id=MapId("another-map"))
    stray = entity(geometry_refs=(foreign,))

    with pytest.raises(ReferenceIntegrityError, match="another-map"):
        context_map(metadata=metadata(capabilities=entity_capabilities()), entities=(stray,))


def test_entity_geometry_must_exist_in_the_referenced_map() -> None:
    beyond = entity(geometry=(POINT_COUNT,))

    with pytest.raises(ReferenceIntegrityError, match="exceeds"):
        context_map(metadata=metadata(capabilities=entity_capabilities()), entities=(beyond,))


def test_the_last_geometry_element_is_still_in_range() -> None:
    last = entity(geometry=(POINT_COUNT - 1,))

    assert context_map(
        metadata=metadata(capabilities=entity_capabilities()), entities=(last,)
    ).entities == (last,)


def test_entity_geometry_must_use_the_canonical_identity() -> None:
    forged = GeometryReference(map_id=GEOMETRIC_MAP_ID, geometry_id=GeometryId("not-a-geometry-id"))
    forged_entity = entity(geometry_refs=(forged,))

    with pytest.raises(ReferenceIntegrityError, match="not-a-geometry-id"):
        context_map(
            metadata=metadata(capabilities=entity_capabilities()), entities=(forged_entity,)
        )


def test_entities_hold_references_never_coordinates() -> None:
    names = {field.name for field in fields(ContextEntity)}

    assert not {name for name in names if "coordinate" in name or "xyz" in name or "point" in name}
    assert "geometry_refs" in names


def test_two_entities_may_share_geometry_support() -> None:
    shared = (entity("entity-0001", geometry=(1, 2)), entity("entity-0002", geometry=(2, 3)))

    result = context_map(metadata=metadata(capabilities=entity_capabilities()), entities=shared)

    assert len(result.entities) == 2


# --- source identity mapping -------------------------------------------------------------------


def test_every_entity_states_the_upstream_record_it_maps_to() -> None:
    found = populated_map().entity(entity_reference("entity-0001"))

    assert found.source == upstream_record(ENTITY_RESOLUTION_ARTIFACT_ID, "resolved-entity-0001")


def test_a_source_identity_is_mapped_explicitly_never_rewritten_silently() -> None:
    # O id no mapa pode diferir do id de origem; o mapeamento é o registro `source`.
    renamed = entity(
        "entity-0001", source=upstream_record(ENTITY_RESOLUTION_ARTIFACT_ID, "resolved-entity-77")
    )

    assert renamed.entity_id != renamed.source.record_id
    assert renamed.source.record_id == "resolved-entity-77"


def test_two_entities_cannot_map_to_the_same_upstream_record() -> None:
    twin_source = upstream_record(ENTITY_RESOLUTION_ARTIFACT_ID, "resolved-shared")
    twins = (
        entity("entity-0001", source=twin_source),
        entity("entity-0002", source=twin_source),
    )

    with pytest.raises(ReferenceIntegrityError, match="resolved-shared"):
        context_map(metadata=metadata(capabilities=entity_capabilities()), entities=twins)


@pytest.mark.parametrize("field", ["artifact_id", "record_id"])
def test_an_upstream_record_reference_needs_both_identities(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        upstream_record(**{"artifact_id": "a", "record_id": "r", field: " "})


def test_an_upstream_artifact_identity_is_not_a_path() -> None:
    with pytest.raises(ValueError, match="path"):
        upstream_record("workspace/run-0001/entities", "r")


# --- semantic state: unresolved and conflicting states are preserved --------------------------


def test_an_unambiguous_entity_has_exactly_one_hypothesis() -> None:
    assert [item.label for item in semantic_state().hypotheses] == ["chair"]
    with pytest.raises(ValueError, match="exactly one"):
        semantic_state(AmbiguityStatus.UNAMBIGUOUS, ("box", "table"))


@pytest.mark.parametrize("status", [AmbiguityStatus.AMBIGUOUS, AmbiguityStatus.CONFLICTING])
def test_competing_states_keep_every_hypothesis(status: AmbiguityStatus) -> None:
    state = semantic_state(status, ("box", "table"))

    assert [item.label for item in state.hypotheses] == ["box", "table"]
    with pytest.raises(ValueError, match="at least two"):
        semantic_state(status, ("box",))


def test_insufficient_evidence_asserts_nothing() -> None:
    state = semantic_state(AmbiguityStatus.INSUFFICIENT_EVIDENCE, ())

    assert state.hypotheses == ()
    with pytest.raises(ValueError, match="no hypothesis"):
        semantic_state(AmbiguityStatus.INSUFFICIENT_EVIDENCE, ("chair",))


def test_hypotheses_are_sorted_unique_and_labelled() -> None:
    with pytest.raises(ValueError, match="sorted"):
        semantic_state(AmbiguityStatus.AMBIGUOUS, ("table", "box"))
    with pytest.raises(ValueError, match="unique"):
        semantic_state(AmbiguityStatus.AMBIGUOUS, ("box", "box"))
    with pytest.raises(ValueError, match="label"):
        hypothesis(" ")


def test_a_conflicting_entity_is_kept_not_resolved_by_the_map() -> None:
    conflicting = populated_map().entity(entity_reference("entity-0002"))

    assert conflicting.semantic_state.status is AmbiguityStatus.AMBIGUOUS
    assert len(conflicting.semantic_state.hypotheses) == 2


def test_an_entity_without_evidence_stays_in_the_map() -> None:
    silent = populated_map().entity(entity_reference("entity-0003"))

    assert silent.semantic_state.status is AmbiguityStatus.INSUFFICIENT_EVIDENCE
    assert silent.semantic_state.hypotheses == ()


# --- relations: subject/object reference semantics ---------------------------------------------


def test_a_relation_names_an_ordered_subject_and_object() -> None:
    found = populated_map().relations[0]

    assert found.subject == entity_reference("entity-0001")
    assert found.predicate == "on"
    assert found.object == entity_reference("entity-0002")


def test_every_relation_resolves_to_valid_entities() -> None:
    result = populated_map()

    for item in result.relations:
        assert result.entity(item.subject)
        assert result.entity(item.object)


def test_a_relation_to_an_unknown_entity_is_rejected() -> None:
    dangling = (relation("relation-0001", "entity-0001", "on", "entity-9999"),)

    with pytest.raises(ReferenceIntegrityError, match="entity-9999"):
        populated_map(relations=dangling)


def test_a_relation_to_an_entity_of_another_map_is_rejected() -> None:
    foreign = relation(
        "relation-0001",
        "entity-0001",
        "on",
        "entity-0002",
        object=entity_reference("entity-0002", context_map_id="context-map--other--0001"),
    )

    with pytest.raises(ReferenceIntegrityError, match="other"):
        populated_map(relations=(foreign,))


def test_a_relation_cannot_relate_an_entity_to_itself() -> None:
    with pytest.raises(ValueError, match="itself"):
        relation("relation-0001", "entity-0001", "on", "entity-0001")


def test_a_relation_needs_a_predicate() -> None:
    with pytest.raises(ValueError, match="predicate"):
        relation(predicate=" ")


def test_relations_are_unique_and_sorted_by_id() -> None:
    first = relation("relation-0001", "entity-0001", "on", "entity-0002")
    second = relation("relation-0002", "entity-0002", "next_to", "entity-0003")

    with pytest.raises(ValueError, match="sorted"):
        populated_map(relations=(second, first))
    with pytest.raises(ValueError, match="unique"):
        populated_map(relations=(first, first))


def test_two_relations_cannot_map_to_the_same_upstream_record() -> None:
    same = upstream_record("spatial-relations--run-0001", "source-shared")
    first = relation("relation-0001", "entity-0001", "on", "entity-0002", source=same)
    second = relation("relation-0002", "entity-0002", "next_to", "entity-0003", source=same)

    with pytest.raises(ReferenceIntegrityError, match="source-shared"):
        populated_map(relations=(first, second))


def test_unresolved_and_conflicting_relations_are_preserved_with_their_state() -> None:
    result = populated_map()

    assert result.relations[0].state is RelationState.SUPPORTED
    assert result.relations[1].state is RelationState.UNRESOLVED
    conflicting = relation(
        "relation-0003", "entity-0001", "next_to", "entity-0003", state=RelationState.CONFLICTING
    )
    both = populated_map(relations=(*result.relations, conflicting))
    assert both.relations[2].state is RelationState.CONFLICTING


# --- traversal ---------------------------------------------------------------------------------


def test_relations_are_reached_from_either_endpoint() -> None:
    result = populated_map()

    around_second = result.relations_for(entity_reference("entity-0002"))

    assert [item.relation_id for item in around_second] == ["relation-0001", "relation-0002"]
    assert result.relations_for(entity_reference("entity-0001")) == (result.relations[0],)


def test_an_entity_without_relations_has_an_empty_traversal() -> None:
    lonely = entity("entity-0004", geometry=(30,))
    result = populated_map(entities=(*populated_map().entities, lonely))

    assert result.relations_for(entity_reference("entity-0004")) == ()


def test_traversal_refuses_foreign_and_unknown_references() -> None:
    result = populated_map()

    with pytest.raises(ForeignContextEntityReferenceError):
        result.relations_for(entity_reference("entity-0001", context_map_id="other"))
    with pytest.raises(UnknownContextEntityError):
        result.relations_for(entity_reference("entity-9999"))


def test_a_structure_walk_needs_only_the_public_contracts() -> None:
    result = populated_map()

    relation_ = result.relations[0]
    subject = result.entity(relation_.subject)
    target = result.entity(relation_.object)

    assert subject.geometry_refs[0].map_id == result.geometry_ref.map_id
    assert target.geometry_refs[0].geometry_id == geometry_id_for(map_id=GEOMETRIC_MAP_ID, index=10)


# --- capabilities are checked against the real content -----------------------------------------


def test_entities_require_the_entities_capability() -> None:
    with pytest.raises(ValueError, match="ENTITIES"):
        context_map(entities=(entity(),))


def test_relations_require_the_relations_capability() -> None:
    result = populated_map()

    with pytest.raises(ValueError, match="RELATIONS"):
        populated_map(
            metadata=metadata(capabilities=entity_capabilities()), relations=result.relations
        )


def test_the_declared_predicates_must_match_the_relations_present() -> None:
    with pytest.raises(ValueError, match="relation_predicates"):
        populated_map(metadata=metadata(capabilities=entity_capabilities("on")))
    with pytest.raises(ValueError, match="relation_predicates"):
        populated_map(
            metadata=metadata(capabilities=entity_capabilities("on", "next_to", "inside"))
        )


def test_a_declared_capability_may_be_present_and_empty() -> None:
    empty_but_declared = context_map(metadata=metadata(capabilities=entity_capabilities()))

    assert empty_but_declared.entities == ()
    assert empty_but_declared.relations == ()


def test_a_geometry_only_map_is_valid() -> None:
    assert context_map().entities == ()


# --- records -----------------------------------------------------------------------------------


def test_a_populated_map_round_trips_through_plain_json() -> None:
    original = populated_map()

    record = json.loads(json.dumps(context_map_to_record(original), allow_nan=False))

    assert context_map_from_record(record) == original


def test_the_record_carries_references_and_identity_mappings() -> None:
    record = context_map_to_record(populated_map())

    first = record["entities"][0]
    assert first["source"] == {
        "artifact_id": ENTITY_RESOLUTION_ARTIFACT_ID,
        "record_id": "resolved-entity-0001",
    }
    assert first["geometry_refs"][0]["map_id"] == GEOMETRIC_MAP_ID
    assert record["relations"][0]["subject"] == {
        "context_map_id": CONTEXT_MAP_ID,
        "entity_id": "entity-0001",
    }
    assert record["relations"][1]["state"] == "unresolved"


def test_a_dangling_reference_in_a_record_is_not_repaired_on_decode() -> None:
    record = context_map_to_record(populated_map())
    record["relations"][0]["object"]["entity_id"] = "entity-9999"

    with pytest.raises(ReferenceIntegrityError, match="entity-9999"):
        context_map_from_record(record)


def test_the_composition_types_are_public() -> None:
    assert ContextEntity.__module__.startswith("contextmap.artifact")
    assert ContextRelation.__module__.startswith("contextmap.artifact")
