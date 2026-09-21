"""The relation taxonomy: direction, symmetry, inverse and frame assumptions per predicate."""

from __future__ import annotations

import pytest

from contextmap.spatial_relations import (
    PREDICATE_SPECS,
    TAXONOMY_VERSION,
    FrameRequirement,
    PredicateFamily,
    RelationPredicate,
    predicate_spec,
)

SYMMETRIC = {
    RelationPredicate.NEXT_TO,
    RelationPredicate.INTERSECTS,
    RelationPredicate.TOUCHING,
}
INVERSE_PAIRS = {
    RelationPredicate.ABOVE: RelationPredicate.BELOW,
    RelationPredicate.IN_FRONT_OF: RelationPredicate.BEHIND,
    RelationPredicate.INSIDE: RelationPredicate.CONTAINS,
}
WITHOUT_CONVERSE = {RelationPredicate.ON_TOP_OF, RelationPredicate.LEANING_AGAINST}


def test_taxonomy_is_versioned() -> None:
    assert TAXONOMY_VERSION == "spatial-relation-taxonomy-v1"


def test_the_vocabulary_is_exactly_the_documented_one() -> None:
    assert {predicate.name for predicate in RelationPredicate} == {
        "NEXT_TO",
        "ABOVE",
        "BELOW",
        "IN_FRONT_OF",
        "BEHIND",
        "INSIDE",
        "CONTAINS",
        "INTERSECTS",
        "TOUCHING",
        "ON_TOP_OF",
        "LEANING_AGAINST",
    }


def test_every_predicate_has_exactly_one_spec() -> None:
    assert set(PREDICATE_SPECS) == set(RelationPredicate)
    for predicate in RelationPredicate:
        assert predicate_spec(predicate).predicate is predicate


def test_symmetric_predicates_are_their_own_inverse() -> None:
    for predicate in RelationPredicate:
        spec = predicate_spec(predicate)
        assert spec.symmetric is (predicate in SYMMETRIC)
        if spec.symmetric:
            assert spec.inverse is predicate


def test_inverse_pairs_are_mutual() -> None:
    for predicate, inverse in INVERSE_PAIRS.items():
        assert predicate_spec(predicate).inverse is inverse
        assert predicate_spec(inverse).inverse is predicate
        assert not predicate_spec(predicate).symmetric


def test_predicates_without_a_converse_in_the_vocabulary_have_no_inverse() -> None:
    for predicate in WITHOUT_CONVERSE:
        spec = predicate_spec(predicate)
        assert spec.inverse is None
        assert not spec.symmetric


def test_inverse_is_an_involution() -> None:
    for spec in PREDICATE_SPECS.values():
        if spec.inverse is not None:
            assert predicate_spec(spec.inverse).inverse is spec.predicate


def test_exactly_one_member_of_each_inverse_pair_is_evaluated_directly() -> None:
    derived = {predicate for predicate in RelationPredicate if predicate_spec(predicate).is_derived}
    assert derived == set(INVERSE_PAIRS.values())
    for predicate, inverse in INVERSE_PAIRS.items():
        assert not predicate_spec(predicate).is_derived
        assert predicate_spec(inverse).is_derived


def test_families_separate_proximity_direction_topology_and_contact() -> None:
    families = {predicate: predicate_spec(predicate).family for predicate in RelationPredicate}
    assert families == {
        RelationPredicate.NEXT_TO: PredicateFamily.PROXIMITY,
        RelationPredicate.ABOVE: PredicateFamily.DIRECTIONAL,
        RelationPredicate.BELOW: PredicateFamily.DIRECTIONAL,
        RelationPredicate.IN_FRONT_OF: PredicateFamily.DIRECTIONAL,
        RelationPredicate.BEHIND: PredicateFamily.DIRECTIONAL,
        RelationPredicate.INSIDE: PredicateFamily.TOPOLOGICAL,
        RelationPredicate.CONTAINS: PredicateFamily.TOPOLOGICAL,
        RelationPredicate.INTERSECTS: PredicateFamily.TOPOLOGICAL,
        RelationPredicate.TOUCHING: PredicateFamily.SUPPORT_CONTACT,
        RelationPredicate.ON_TOP_OF: PredicateFamily.SUPPORT_CONTACT,
        RelationPredicate.LEANING_AGAINST: PredicateFamily.SUPPORT_CONTACT,
    }


def test_frame_assumptions_are_explicit_per_predicate() -> None:
    needs_up = {
        RelationPredicate.ABOVE,
        RelationPredicate.BELOW,
        RelationPredicate.ON_TOP_OF,
        RelationPredicate.LEANING_AGAINST,
    }
    needs_forward = {RelationPredicate.IN_FRONT_OF, RelationPredicate.BEHIND}
    for predicate in RelationPredicate:
        requirement = predicate_spec(predicate).frame_requirement
        if predicate in needs_up:
            assert requirement is FrameRequirement.UP_AXIS
        elif predicate in needs_forward:
            assert requirement is FrameRequirement.UP_AND_FORWARD_AXES
        else:
            assert requirement is FrameRequirement.MAP_FRAME


def test_inverse_pairs_share_their_frame_requirement() -> None:
    for predicate, inverse in INVERSE_PAIRS.items():
        assert (
            predicate_spec(predicate).frame_requirement is predicate_spec(inverse).frame_requirement
        )


def test_every_spec_states_its_meaning() -> None:
    for spec in PREDICATE_SPECS.values():
        assert spec.meaning.strip()


def test_canonical_names_round_trip_through_the_enum_value() -> None:
    for predicate in RelationPredicate:
        assert RelationPredicate(predicate.value) is predicate
    with pytest.raises(ValueError):
        RelationPredicate("near")
