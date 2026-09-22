"""Semantic compatibility: every hypothesis is compared, and unknown never becomes a conflict."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from mapping_builders import claim_signal, make_evidence_item, make_hypothesis
from resolution_entity_builders import attribute, entity_at, hypotheses_of

from contextmap.entity_resolution import (
    BASELINE_REFINEMENT_MODIFIERS,
    SEMANTIC_COMPATIBILITY_POLICY_ID,
    EvidenceStatus,
    LabelRelation,
    SemanticCompatibilityPolicy,
    SemanticEvidence,
    UnavailableReason,
    compare_labels,
    compare_semantics,
    normalize_label,
)
from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.semantic_mapping import (
    CLASS_ATTRIBUTE_DERIVATION_ID,
    AmbiguityState,
    AttributeOrigin,
    Entity,
)

POLICY = SemanticCompatibilityPolicy(refinement_modifiers=BASELINE_REFINEMENT_MODIFIERS)


def labeled(entity_id: str, *labels: str, attributes: Any = ()) -> Entity:
    return entity_at(entity_id, hypotheses=hypotheses_of(*labels), attributes=attributes)


def compare(first: Entity, second: Entity, policy: SemanticCompatibilityPolicy = POLICY) -> Any:
    return compare_semantics(first, second, policy)


def status_of(evidence: SemanticEvidence, rule_id: str) -> EvidenceStatus:
    return next(item.status for item in evidence.findings if item.rule_id == rule_id)


# --- label relations ----------------------------------------------------------------------------


def test_the_same_label_is_compatible() -> None:
    evidence = compare(labeled("a", "pallet"), labeled("b", "pallet"))

    (pair,) = evidence.measurement.label_comparisons
    assert pair.relation is LabelRelation.SAME
    assert status_of(evidence, "label-compatibility") is EvidenceStatus.SUPPORTING
    assert evidence.status is EvidenceStatus.SUPPORTING


def test_a_declared_modifier_makes_a_compatible_refinement() -> None:
    evidence = compare(labeled("a", "pallet"), labeled("b", "wooden pallet"))

    (pair,) = evidence.measurement.label_comparisons
    assert (pair.label_a, pair.label_b) == ("pallet", "wooden pallet")
    assert pair.relation is LabelRelation.REFINEMENT
    assert status_of(evidence, "label-compatibility") is EvidenceStatus.SUPPORTING


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("Wooden  Pallet", "wooden pallet", LabelRelation.SAME),
        ("wooden-pallet", "wooden pallet", LabelRelation.SAME),
        ("wooden_pallet", "Wooden Pallet.", LabelRelation.SAME),
        ("red wooden pallet", "pallet", LabelRelation.REFINEMENT),
        # Um modificador não declarado nunca vira refinamento em silêncio.
        ("toy pallet", "pallet", LabelRelation.DIFFERENT),
        # O núcleo do nome precisa ser o mesmo: "pallet rack" não é um "pallet".
        ("pallet rack", "pallet", LabelRelation.DIFFERENT),
        ("pallet", "crate", LabelRelation.DIFFERENT),
        ("pallet", "person", LabelRelation.DIFFERENT),
    ],
)
def test_label_relations_are_conservative(first: str, second: str, expected: LabelRelation) -> None:
    relation, rule_id = compare_labels(first, second, POLICY)

    assert relation is expected
    assert rule_id.startswith("label-")


def test_related_labels_exist_only_when_the_policy_declares_them() -> None:
    declared = SemanticCompatibilityPolicy(
        refinement_modifiers=BASELINE_REFINEMENT_MODIFIERS,
        related_label_pairs=(("crate", "pallet"),),
    )

    assert compare_labels("pallet", "crate", POLICY)[0] is LabelRelation.DIFFERENT
    assert compare_labels("pallet", "crate", declared)[0] is LabelRelation.RELATED
    assert compare_labels("crate", "wooden pallet", declared)[0] is LabelRelation.DIFFERENT
    evidence = compare(labeled("a", "pallet"), labeled("b", "crate"), declared)
    assert status_of(evidence, "label-compatibility") is EvidenceStatus.NEUTRAL


def test_incompatible_settled_labels_are_a_conflict_but_not_a_decision() -> None:
    evidence = compare(labeled("a", "pallet"), labeled("b", "person"))

    assert status_of(evidence, "label-compatibility") is EvidenceStatus.CONFLICTING
    assert evidence.status is EvidenceStatus.CONFLICTING
    assert not hasattr(evidence, "decision")


def test_normalization_only_folds_case_punctuation_and_spacing() -> None:
    assert normalize_label("  Wooden__Pallet!! ") == "wooden pallet"
    assert normalize_label("pallets") == "pallets"


# --- rich semantic states are not reduced to one label ------------------------------------------


def test_every_hypothesis_of_an_ambiguous_entity_is_compared_and_kept() -> None:
    ambiguous = labeled("a", "pallet", "crate")

    evidence = compare(ambiguous, labeled("b", "crate"))

    measurement = evidence.measurement
    assert measurement.ambiguity_a is AmbiguityState.AMBIGUOUS
    assert measurement.ambiguity_b is AmbiguityState.UNAMBIGUOUS
    assert measurement.hypothesis_count_a == 2
    pairs = {(item.label_a, item.label_b): item.relation for item in measurement.label_comparisons}
    assert pairs == {
        ("crate", "crate"): LabelRelation.SAME,
        ("pallet", "crate"): LabelRelation.DIFFERENT,
    }
    assert status_of(evidence, "label-compatibility") is EvidenceStatus.SUPPORTING


def test_disagreement_involving_an_ambiguous_entity_is_not_a_false_conflict() -> None:
    evidence = compare(labeled("a", "pallet", "crate"), labeled("b", "person"))

    assert status_of(evidence, "label-compatibility") is EvidenceStatus.NEUTRAL
    assert evidence.status is EvidenceStatus.NEUTRAL


def test_an_entity_without_hypotheses_is_unavailable_not_a_conflict() -> None:
    evidence = compare(entity_at("a", hypotheses=()), labeled("b", "pallet"))

    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.measurement is None
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.MISSING_EVIDENCE
    assert "a" in evidence.unavailable.detail


def test_scores_are_neither_read_nor_compared() -> None:
    unscored = make_hypothesis(evidence=(make_evidence_item(signals=()),))
    scored = make_hypothesis(evidence=(make_evidence_item(signals=(claim_signal(0.99),)),))
    low = make_hypothesis(evidence=(make_evidence_item(signals=(claim_signal(0.01),)),))

    def compare_with(hypothesis: Any) -> Any:
        return compare(
            entity_at("a", hypotheses=(unscored,)), entity_at("b", hypotheses=(hypothesis,))
        )

    assert compare_with(scored) == compare_with(low)


def test_the_labels_of_the_entities_are_kept_verbatim_in_the_evidence() -> None:
    evidence = compare(labeled("a", "Wooden Pallet"), labeled("b", "pallet"))

    (pair,) = evidence.measurement.label_comparisons
    assert (pair.label_a, pair.label_b) == ("Wooden Pallet", "pallet")


# --- attributes ---------------------------------------------------------------------------------


def test_a_differing_shared_attribute_conflicts_with_its_provenance() -> None:
    first = labeled("a", "pallet", attributes=(attribute("material", "wood"),))
    second = labeled("b", "pallet", attributes=(attribute("material", "metal"),))

    evidence = compare(first, second)

    (item,) = evidence.measurement.attribute_comparisons
    assert (item.name, item.value_a, item.value_b, item.same) == (
        "material",
        "wood",
        "metal",
        False,
    )
    assert (item.origin_a, item.origin_b) == (AttributeOrigin.OBSERVED, AttributeOrigin.OBSERVED)
    assert status_of(evidence, "attribute-compatibility") is EvidenceStatus.CONFLICTING
    assert evidence.status is EvidenceStatus.CONFLICTING


def test_sharing_an_attribute_value_is_not_evidence_of_identity() -> None:
    first = labeled("a", "pallet", attributes=(attribute("material", "wood"),))
    second = labeled("b", "pallet", attributes=(attribute("material", "Wood"),))

    evidence = compare(first, second)

    assert evidence.measurement.attribute_comparisons[0].same
    assert status_of(evidence, "attribute-compatibility") is EvidenceStatus.NEUTRAL


def test_the_derived_class_attribute_and_external_knowledge_are_not_compared() -> None:
    derived = attribute(
        "class",
        "pallet",
        origin=AttributeOrigin.DERIVED,
        derivation_id=CLASS_ATTRIBUTE_DERIVATION_ID,
    )
    other_class = attribute(
        "class",
        "crate",
        origin=AttributeOrigin.DERIVED,
        derivation_id=CLASS_ATTRIBUTE_DERIVATION_ID,
    )
    imagined = attribute("material", "wood", origin=AttributeOrigin.EXTERNAL_KNOWLEDGE)
    real = attribute("material", "metal")

    evidence = compare(
        labeled("a", "pallet", attributes=(derived, imagined)),
        labeled("b", "pallet", attributes=(other_class, real)),
    )

    assert evidence.measurement.attribute_comparisons == ()
    assert all(item.rule_id != "attribute-compatibility" for item in evidence.findings)


def test_an_attribute_only_one_entity_has_is_not_a_conflict() -> None:
    evidence = compare(
        labeled("a", "pallet", attributes=(attribute("material", "wood"),)), labeled("b", "pallet")
    )

    assert evidence.measurement.attribute_comparisons == ()
    assert evidence.status is EvidenceStatus.SUPPORTING


# --- contract -----------------------------------------------------------------------------------


def test_the_comparison_is_symmetric_and_names_its_policy() -> None:
    first, second = labeled("a", "pallet"), labeled("b", "wooden pallet")

    assert compare(first, second) == compare(second, first)
    evidence = compare(first, second)
    assert evidence.policy.policy_id == SEMANTIC_COMPATIBILITY_POLICY_ID
    assert evidence.policy.configuration_fingerprint == POLICY.fingerprint()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refinement_modifiers": ("Wooden",)},
        {"refinement_modifiers": ("red wooden",)},
        {"refinement_modifiers": ("wooden", "red")},
        {"refinement_modifiers": ("red", "red")},
        {"refinement_modifiers": (), "related_label_pairs": (("pallet", "crate"),)},
        {"refinement_modifiers": (), "related_label_pairs": (("Crate", "pallet"),)},
        {"refinement_modifiers": (), "related_label_pairs": (("crate", "crate"),)},
        {
            "refinement_modifiers": (),
            "related_label_pairs": (("crate", "pallet"), ("box", "crate")),
        },
    ],
)
def test_the_policy_refuses_unnormalized_or_unordered_declarations(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SemanticCompatibilityPolicy(**kwargs)


def test_the_fingerprint_changes_with_every_declaration() -> None:
    variants = [
        SemanticCompatibilityPolicy(refinement_modifiers=()),
        SemanticCompatibilityPolicy(refinement_modifiers=("wooden",)),
        SemanticCompatibilityPolicy(
            refinement_modifiers=("wooden",), related_label_pairs=(("crate", "pallet"),)
        ),
    ]

    assert len({item.fingerprint() for item in variants}) == 3
    assert math.isfinite(len(BASELINE_REFINEMENT_MODIFIERS))


def test_the_evidence_survives_a_json_round_trip_with_attribute_provenance() -> None:
    first = labeled("a", "pallet", "crate", attributes=(attribute("material", "wood"),))
    second = labeled("b", "wooden pallet", attributes=(attribute("material", "metal"),))
    evidence = compare(first, second)

    record = json.loads(json.dumps(to_record(evidence)))

    assert from_record(SemanticEvidence, record) == evidence
    assert record["measurement"]["attribute_comparisons"][0]["origin_a"] == "observed"


def test_labels_without_letters_are_only_the_same_when_verbatim_equal() -> None:
    assert compare_labels("!!!", "!!!", POLICY)[0] is LabelRelation.SAME
    assert compare_labels("!!!", "???", POLICY)[0] is LabelRelation.DIFFERENT
    assert compare_labels("!!!", "pallet", POLICY)[0] is LabelRelation.DIFFERENT


def test_several_values_and_origins_of_one_attribute_are_compared_pairwise_and_deduplicated() -> (
    None
):
    first = labeled(
        "a",
        "pallet",
        attributes=(
            attribute(
                "color", "brown", origin=AttributeOrigin.DERIVED, derivation_id="derived-color-v1"
            ),
            attribute("color", "brown"),
            attribute("color", "red"),
            attribute("weight", "heavy"),
        ),
    )
    second = labeled("b", "pallet", attributes=(attribute("color", "Brown"),))

    evidence = compare(first, second)

    pairs = [
        (item.value_a, item.value_b, item.same, item.origin_a)
        for item in evidence.measurement.attribute_comparisons
    ]
    assert pairs == [
        ("brown", "Brown", True, AttributeOrigin.DERIVED),
        ("red", "Brown", False, AttributeOrigin.OBSERVED),
    ]
    assert status_of(evidence, "attribute-compatibility") is EvidenceStatus.CONFLICTING
