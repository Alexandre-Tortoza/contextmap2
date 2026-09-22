"""Relation: an explicit, traceable claim about two resolved entities, with a state."""

from __future__ import annotations

import dataclasses

import pytest
from relation_builders import entity_ref, make_relation, make_uncertainty

from contextmap.spatial_relations import (
    TAXONOMY_VERSION,
    RelationId,
    RelationPredicate,
    RelationProvenance,
    RelationState,
    RelationUncertainty,
    RelationUncertaintyKind,
    relation_id_for,
)


def _id(subject: int, obj: int, predicate: RelationPredicate = RelationPredicate.NEXT_TO) -> str:
    return str(
        relation_id_for(
            subject_entity_ref=entity_ref(subject),
            predicate=predicate,
            object_entity_ref=entity_ref(obj),
        )
    )


def test_a_supported_relation_names_subject_predicate_object_and_evidence() -> None:
    relation = make_relation(subject=1, obj=2, predicate=RelationPredicate.ABOVE)
    assert relation.subject_entity_ref == entity_ref(1)
    assert relation.object_entity_ref == entity_ref(2)
    assert relation.predicate is RelationPredicate.ABOVE
    assert relation.state is RelationState.SUPPORTED
    assert len(relation.relation_evidence_refs) == 1
    assert relation.uncertainty == ()
    assert relation.provenance.taxonomy_version == TAXONOMY_VERSION


def test_relations_are_immutable() -> None:
    relation = make_relation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        relation.state = RelationState.REJECTED  # type: ignore[misc]


def test_a_relation_holds_references_never_entities() -> None:
    fields = {field.name for field in dataclasses.fields(make_relation())}
    assert {"subject_entity_ref", "object_entity_ref"} <= fields
    assert not {"subject", "object", "subject_entity", "object_entity"} & fields


def test_an_entity_is_never_related_to_itself() -> None:
    with pytest.raises(ValueError, match="itself"):
        make_relation(subject=3, obj=3)


def test_both_entities_must_come_from_one_resolution_artifact() -> None:
    relation = make_relation()
    with pytest.raises(ValueError, match="resolution"):
        dataclasses.replace(relation, object_entity_ref=entity_ref(2, run="resolution-run-0002"))


def test_a_decided_relation_needs_the_evidence_it_was_decided_on() -> None:
    for state in (RelationState.SUPPORTED, RelationState.REJECTED):
        with pytest.raises(ValueError, match="evidence"):
            make_relation(state=state, evidence_refs=())


def test_a_decided_relation_carries_no_uncertainty() -> None:
    reason = make_uncertainty(evidence_refs=())
    for state in (RelationState.SUPPORTED, RelationState.REJECTED):
        with pytest.raises(ValueError, match="uncertainty"):
            make_relation(state=state, uncertainty=(reason,))


def test_an_unresolved_relation_states_why() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        make_relation(state=RelationState.UNRESOLVED, evidence_refs=())
    relation = make_relation(
        state=RelationState.UNRESOLVED, evidence_refs=(), uncertainty=(make_uncertainty(),)
    )
    assert relation.uncertainty[0].kind is RelationUncertaintyKind.INSUFFICIENT_EVIDENCE


def test_a_rejected_relation_is_distinguishable_from_an_unresolved_one() -> None:
    rejected = make_relation(state=RelationState.REJECTED)
    unresolved = make_relation(
        state=RelationState.UNRESOLVED, evidence_refs=(), uncertainty=(make_uncertainty(),)
    )
    assert rejected.state is not unresolved.state
    assert rejected.uncertainty == ()
    assert unresolved.uncertainty != ()


def test_evidence_references_are_canonical() -> None:
    with pytest.raises(ValueError, match="sorted"):
        make_relation(evidence_refs=("evidence--b", "evidence--a"))
    with pytest.raises(ValueError, match="sorted"):
        make_relation(evidence_refs=("evidence--a", "evidence--a"))


def test_every_uncertainty_points_at_evidence_the_relation_used() -> None:
    reason = make_uncertainty(
        RelationUncertaintyKind.CONFLICTING_EVIDENCE, evidence_refs=("evidence--a", "evidence--z")
    )
    with pytest.raises(ValueError, match="evidence--z"):
        make_relation(
            state=RelationState.UNRESOLVED,
            evidence_refs=("evidence--a", "evidence--b"),
            uncertainty=(reason,),
        )
    relation = make_relation(
        state=RelationState.UNRESOLVED,
        evidence_refs=("evidence--a", "evidence--z"),
        uncertainty=(reason,),
    )
    assert relation.uncertainty == (reason,)


def test_uncertainty_is_canonical() -> None:
    first = make_uncertainty(RelationUncertaintyKind.CONFLICTING_EVIDENCE, detail="channels differ")
    second = make_uncertainty(RelationUncertaintyKind.INSUFFICIENT_EVIDENCE, detail="nothing")
    ordered = tuple(sorted((first, second), key=lambda item: (item.kind.value, item.detail)))
    make_relation(state=RelationState.UNRESOLVED, evidence_refs=(), uncertainty=ordered)
    with pytest.raises(ValueError, match="sorted"):
        make_relation(
            state=RelationState.UNRESOLVED, evidence_refs=(), uncertainty=tuple(reversed(ordered))
        )


def test_uncertainty_needs_an_explanation_and_canonical_references() -> None:
    with pytest.raises(ValueError, match="detail"):
        RelationUncertainty(kind=RelationUncertaintyKind.INSUFFICIENT_EVIDENCE, detail=" ")
    with pytest.raises(ValueError, match="sorted"):
        make_uncertainty(evidence_refs=("evidence--b", "evidence--a"))


def test_a_derived_relation_names_the_relation_it_was_generated_from() -> None:
    source = make_relation(subject=1, obj=2, predicate=RelationPredicate.ABOVE)
    derived = make_relation(
        subject=2,
        obj=1,
        predicate=RelationPredicate.BELOW,
        evidence_refs=tuple(str(item) for item in source.relation_evidence_refs),
        derived_from=source.relation_id,
    )
    assert derived.derived_from == source.relation_id
    with pytest.raises(ValueError, match="itself"):
        dataclasses.replace(derived, derived_from=derived.relation_id)


def test_provenance_records_the_taxonomy_and_the_decision_policy() -> None:
    with pytest.raises(ValueError, match="decision_policy_id"):
        RelationProvenance(
            taxonomy_version=TAXONOMY_VERSION,
            decision_policy_id="",
            configuration_fingerprint="sha256:x",
        )
    with pytest.raises(ValueError, match="taxonomy_version"):
        RelationProvenance(
            taxonomy_version="",
            decision_policy_id="policy",
            configuration_fingerprint="sha256:x",
        )
    with pytest.raises(ValueError, match="configuration_fingerprint"):
        RelationProvenance(
            taxonomy_version=TAXONOMY_VERSION,
            decision_policy_id="policy",
            configuration_fingerprint=" ",
        )


def test_relation_identity_is_deterministic_and_directional() -> None:
    assert _id(1, 2) == _id(1, 2)
    assert len({_id(1, 2), _id(2, 1)}) == 2
    assert _id(1, 2) != _id(1, 2, RelationPredicate.ABOVE)
    assert "above" in _id(1, 2, RelationPredicate.ABOVE)
    assert RelationId(_id(1, 2)) == _id(1, 2)


def test_relation_identity_depends_on_the_resolution_artifact() -> None:
    same_local_ids = relation_id_for(
        subject_entity_ref=entity_ref(1, run="resolution-run-0002"),
        predicate=RelationPredicate.NEXT_TO,
        object_entity_ref=entity_ref(2, run="resolution-run-0002"),
    )
    assert str(same_local_ids) != _id(1, 2)
