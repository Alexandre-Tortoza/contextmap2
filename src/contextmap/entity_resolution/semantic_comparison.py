"""Semantic compatibility of two entities: hypotheses compared, never collapsed to one label.

Upstream, an entity keeps every hypothesis, ambiguity, conflict and attribute, so the semantic
channel compares *all* of it instead of one label per entity. The baseline is deliberately
conservative and has no ontology:

* **normalization** folds only case, punctuation and spacing (``Wooden-Pallet`` and
  ``wooden pallet`` are the same label); it never stems, lemmatizes or translates;
* **refinement** holds when a label is another one plus leading words that the policy explicitly
  declares to be modifiers (``wooden pallet`` refines ``pallet`` if ``wooden`` is declared), and the
  head of the name is unchanged, so ``pallet rack`` is not a ``pallet`` and an undeclared modifier
  (``toy pallet``) is not a refinement;
* **related** labels exist only when the policy declares the pair; nothing is inferred from
  common sense (``pallet`` and ``crate`` are different unless declared related).

Semantic disagreement is evidence, not a verdict. Two settled, unrelated labels (``pallet`` and
``person``) are a strong ``conflicting`` finding, but an ambiguous entity that shares no hypothesis
with the other is only ``neutral``, because its alternatives are still open and upstream labels may
be wrong. An entity with no supported hypothesis (unknown, abstained) makes the channel
``unavailable``: unknown is neither support nor evidence against.

Hypothesis scores are never read: they are heterogeneous signals with no defined common semantics,
and a hypothesis exists only because at least one claim supports it. Attributes are compared by
name when both entities have the name; the ``class`` attribute (which restates the primary
hypothesis) and external knowledge (never evidence) are excluded, and agreeing on an attribute value
is not evidence of identity, since many objects share a material or a color.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from contextmap.entity_resolution._checks import require_canonical
from contextmap.entity_resolution.channels import (
    AttributeComparison,
    EvidenceStatus,
    Finding,
    LabelComparison,
    LabelRelation,
    SemanticEvidence,
    SemanticMeasurement,
    Unavailability,
    UnavailableReason,
)
from contextmap.entity_resolution.models import PolicyRef, reference_order
from contextmap.semantic_mapping import (
    CLASS_ATTRIBUTE_DERIVATION_ID,
    AmbiguityState,
    AttributeOrigin,
    Entity,
    EntitySemanticState,
)

SEMANTIC_COMPATIBILITY_POLICY_ID = "entity-semantic-compatibility-v1"
"""Versioned identity of the label and attribute rules described in this module."""

BASELINE_REFINEMENT_MODIFIERS = (
    "big",
    "black",
    "blue",
    "brown",
    "cardboard",
    "gray",
    "green",
    "large",
    "little",
    "metal",
    "metallic",
    "orange",
    "plastic",
    "red",
    "small",
    "white",
    "wooden",
    "yellow",
)
"""A starting set of color, material and size modifiers, for a profile that wants one.

It is a convention and not an ontology: the policy takes the modifiers it is given, and this
tuple is used only when a profile chooses to pass it.
"""

_NOT_WORD = re.compile(r"[\W_]+")


def normalize_label(label: str) -> str:
    """Fold case, punctuation and spacing of a label, and nothing else.

    Args:
        label: A hypothesis label, verbatim.

    Returns:
        The label lower-cased, with every run of punctuation or whitespace turned into one space.
    """
    return _NOT_WORD.sub(" ", label.casefold()).strip()


@dataclass(frozen=True, kw_only=True)
class SemanticCompatibilityPolicy:
    """Explicit declarations of the label rules.

    There are no defaults: which words are modifiers and which labels are related are choices a
    profile declares (see :data:`BASELINE_REFINEMENT_MODIFIERS`).

    Attributes:
        refinement_modifiers: Single normalized words that may precede a label without changing
            what it names, sorted and unique.
        related_label_pairs: Pairs of normalized labels declared related, each ordered
            alphabetically, the pairs sorted and unique.
    """

    refinement_modifiers: tuple[str, ...]
    related_label_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate that every declaration is normalized and canonically ordered.

        Raises:
            ValueError: If a modifier is not one normalized word, a pair is not two different
                normalized labels in alphabetical order, or a collection is not sorted and
                unique.
        """
        for modifier in self.refinement_modifiers:
            if not modifier or normalize_label(modifier) != modifier or " " in modifier:
                raise ValueError(f"refinement modifier {modifier!r} must be one normalized word")
        require_canonical("refinement_modifiers", self.refinement_modifiers, lambda item: (item,))
        for first, second in self.related_label_pairs:
            if (
                not first
                or not second
                or normalize_label(first) != first
                or normalize_label(second) != second
            ):
                raise ValueError(f"related pair {(first, second)!r} must hold normalized labels")
            if first >= second:
                raise ValueError(f"related pair {(first, second)!r} must be two labels in order")
        require_canonical("related_label_pairs", self.related_label_pairs, lambda item: item)

    def fingerprint(self) -> str:
        """Hash the policy identity and declarations, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": SEMANTIC_COMPATIBILITY_POLICY_ID,
                "refinement_modifiers": list(self.refinement_modifiers),
                "related_label_pairs": [list(item) for item in self.related_label_pairs],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on the evidence."""
        return PolicyRef(
            policy_id=SEMANTIC_COMPATIBILITY_POLICY_ID, configuration_fingerprint=self.fingerprint()
        )


def compare_labels(
    label_a: str, label_b: str, policy: SemanticCompatibilityPolicy
) -> tuple[LabelRelation, str]:
    """Relate two hypothesis labels under the explicit rules of the policy.

    Args:
        label_a: A label, verbatim.
        label_b: Another label, verbatim.
        policy: The declared modifiers and related pairs.

    Returns:
        The relation and the versioned identity of the rule that decided it.
    """
    first, second = normalize_label(label_a), normalize_label(label_b)
    if label_a == label_b or (first and first == second):
        return LabelRelation.SAME, "label-equal-v1"
    if first and second:
        modifiers = set(policy.refinement_modifiers)
        if _refines(first, second, modifiers) or _refines(second, first, modifiers):
            return LabelRelation.REFINEMENT, "label-refinement-v1"
        if tuple(sorted((first, second))) in policy.related_label_pairs:
            return LabelRelation.RELATED, "label-declared-related-v1"
    return LabelRelation.DIFFERENT, "label-different-v1"


def _refines(longer: str, shorter: str, modifiers: set[str]) -> bool:
    long_words, short_words = longer.split(), shorter.split()
    added = len(long_words) - len(short_words)
    return (
        added > 0
        and long_words[added:] == short_words
        and all(word in modifiers for word in long_words[:added])
    )


def compare_semantics(
    entity_a: Entity, entity_b: Entity, policy: SemanticCompatibilityPolicy
) -> SemanticEvidence:
    """Compare what two entities may be, keeping every hypothesis and attribute.

    The result does not depend on the order of the arguments: the pair is put in canonical order
    and the measurements refer to it.

    Args:
        entity_a: One entity.
        entity_b: The other entity.
        policy: The declared modifiers and related pairs.

    Returns:
        The semantic evidence, or an unavailable channel when an entity has no supported
        hypothesis.
    """
    first, second = sorted((entity_a, entity_b), key=lambda item: reference_order(item.reference))
    state_a, state_b = first.semantic_state, second.semantic_state
    empty = [
        item.entity_id
        for item, state in ((first, state_a), (second, state_b))
        if not state.hypotheses
    ]
    if empty:
        return SemanticEvidence(
            policy=policy.ref(),
            unavailable=Unavailability(
                reason=UnavailableReason.MISSING_EVIDENCE,
                detail=(
                    f"{', '.join(repr(item) for item in empty)} has no supported hypothesis "
                    f"(unknown or abstained), which is neither support nor evidence against"
                ),
            ),
        )
    labels = _label_comparisons(state_a, state_b, policy)
    attributes = _attribute_comparisons(state_a, state_b)
    measurement = SemanticMeasurement(
        ambiguity_a=state_a.ambiguity_state,
        ambiguity_b=state_b.ambiguity_state,
        hypothesis_count_a=len(state_a.hypotheses),
        hypothesis_count_b=len(state_b.hypotheses),
        label_comparisons=labels,
        attribute_comparisons=attributes,
    )
    findings = [_label_finding(measurement)]
    if attributes:
        findings.append(_attribute_finding(attributes))
    return SemanticEvidence(policy=policy.ref(), measurement=measurement, findings=tuple(findings))


def _labels(state: EntitySemanticState) -> dict[str, bool]:
    """Each distinct label, verbatim, and whether it is the primary hypothesis."""
    labels: dict[str, bool] = {}
    for hypothesis in state.hypotheses:
        primary = hypothesis.ref == state.primary_hypothesis
        labels[hypothesis.label] = labels.get(hypothesis.label, False) or primary
    return labels


def _label_comparisons(
    state_a: EntitySemanticState, state_b: EntitySemanticState, policy: SemanticCompatibilityPolicy
) -> tuple[LabelComparison, ...]:
    labels_a, labels_b = _labels(state_a), _labels(state_b)
    comparisons = []
    for label_a in sorted(labels_a):
        for label_b in sorted(labels_b):
            relation, rule_id = compare_labels(label_a, label_b, policy)
            comparisons.append(
                LabelComparison(
                    label_a=label_a,
                    label_b=label_b,
                    primary_a=labels_a[label_a],
                    primary_b=labels_b[label_b],
                    relation=relation,
                    rule_id=rule_id,
                )
            )
    return tuple(comparisons)


def _attributes(state: EntitySemanticState) -> dict[str, dict[str, AttributeOrigin]]:
    """Observed and derived attribute values by name; class and external knowledge are excluded."""
    found: dict[str, dict[str, AttributeOrigin]] = {}
    for item in state.attributes:
        if (
            item.origin is AttributeOrigin.EXTERNAL_KNOWLEDGE
            or item.derivation_id == CLASS_ATTRIBUTE_DERIVATION_ID
        ):
            continue
        values = found.setdefault(item.name, {})
        # O mesmo valor com duas origens vira um só; a origem "derived" precede "observed".
        if item.value not in values or item.origin.value < values[item.value].value:
            values[item.value] = item.origin
    return found


def _attribute_comparisons(
    state_a: EntitySemanticState, state_b: EntitySemanticState
) -> tuple[AttributeComparison, ...]:
    attributes_a, attributes_b = _attributes(state_a), _attributes(state_b)
    return tuple(
        AttributeComparison(
            name=name,
            value_a=value_a,
            value_b=value_b,
            origin_a=attributes_a[name][value_a],
            origin_b=attributes_b[name][value_b],
            same=normalize_label(value_a) == normalize_label(value_b),
        )
        for name in sorted(attributes_a.keys() & attributes_b.keys())
        for value_a in sorted(attributes_a[name])
        for value_b in sorted(attributes_b[name])
    )


def _label_finding(measurement: SemanticMeasurement) -> Finding:
    comparisons = measurement.label_comparisons
    compatible = [
        item
        for item in comparisons
        if item.relation in (LabelRelation.SAME, LabelRelation.REFINEMENT)
    ]
    related = [item for item in comparisons if item.relation is LabelRelation.RELATED]
    if compatible:
        # O par primário com primário é o mais informativo; senão o primeiro par compatível.
        best = max(
            compatible,
            key=lambda item: (item.primary_a and item.primary_b, item.primary_a or item.primary_b),
        )
        return Finding(
            rule_id="label-compatibility",
            status=EvidenceStatus.SUPPORTING,
            detail=(
                f"hypotheses {best.label_a!r} and {best.label_b!r} are compatible "
                f"({best.relation.value}, {best.rule_id})"
            ),
        )
    if related:
        best = related[0]
        return Finding(
            rule_id="label-compatibility",
            status=EvidenceStatus.NEUTRAL,
            detail=(
                f"no compatible hypothesis; {best.label_a!r} and {best.label_b!r} are only "
                f"declared related, which is not enough to support a match"
            ),
        )
    settled = (
        measurement.ambiguity_a is AmbiguityState.UNAMBIGUOUS
        and measurement.ambiguity_b is AmbiguityState.UNAMBIGUOUS
    )
    if settled:
        first = comparisons[0]
        return Finding(
            rule_id="label-compatibility",
            status=EvidenceStatus.CONFLICTING,
            detail=(
                f"settled labels {first.label_a!r} and {first.label_b!r} are unrelated; a "
                f"semantic conflict is not proof of distinct identity"
            ),
        )
    return Finding(
        rule_id="label-compatibility",
        status=EvidenceStatus.NEUTRAL,
        detail=(
            f"no compatible hypothesis, but an entity is {measurement.ambiguity_a.value} or "
            f"{measurement.ambiguity_b.value}: its alternatives stay open, not a conflict"
        ),
    )


def _attribute_finding(attributes: tuple[AttributeComparison, ...]) -> Finding:
    differing = [item for item in attributes if not item.same]
    if differing:
        first = differing[0]
        return Finding(
            rule_id="attribute-compatibility",
            status=EvidenceStatus.CONFLICTING,
            detail=(
                f"attribute {first.name!r} is {first.value_a!r} on one entity and "
                f"{first.value_b!r} on the other"
            ),
        )
    return Finding(
        rule_id="attribute-compatibility",
        status=EvidenceStatus.NEUTRAL,
        detail="the shared attributes agree, which is not evidence of identity",
    )
