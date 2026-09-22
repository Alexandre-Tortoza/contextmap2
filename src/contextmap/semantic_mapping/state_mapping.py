"""Mapping rules from fused evidence to the semantic state of an entity.

Semantic Fusion produces belief in formation: every hypothesis, the evidence behind each and the
conflicts between observations, with no winner. This module turns that into the persistent
semantic state of an entity **without destroying any of it**:

* every hypothesis is kept with all of its evidence, so alternatives, ambiguous and conflicting
  claims, abstentions and unscored signals survive exactly as fused;
* every uncertainty record is kept, so contradictions, ambiguity, near ties and lack of evidence
  stay visible;
* a **primary hypothesis** is exposed only under ``unambiguous-single-hypothesis-v1``: exactly one
  hypothesis and nothing that competes with it. Otherwise there is no primary, and the entity
  is not given a label the evidence does not support;
* the only attribute produced is ``class``, derived from the primary hypothesis by
  ``primary-hypothesis-label-v1`` and citing the claims that support it. Nothing else is
  inferred: no common-sense property, no ontology enrichment and no external knowledge.
"""

from __future__ import annotations

from contextmap.semantic_fusion import EvidenceReference, EvidenceStance, FusedEvidence
from contextmap.semantic_mapping.semantic_state import (
    AmbiguityState,
    AttributeOrigin,
    EntityAttribute,
    EntityHypothesis,
    EntitySemanticState,
    EntityUncertainty,
    SemanticStateProvenance,
    derive_ambiguity_state,
)

SEMANTIC_STATE_MAPPING_RULE_ID = "fused-evidence-semantic-state-v1"
"""Versioned identity of the mapping from fused evidence to semantic state."""

PRIMARY_HYPOTHESIS_POLICY_ID = "unambiguous-single-hypothesis-v1"
"""Versioned identity of the rule that decides when a primary hypothesis is exposed."""

CLASS_ATTRIBUTE_DERIVATION_ID = "primary-hypothesis-label-v1"
"""Versioned identity of the rule that derives the ``class`` attribute."""


def semantic_state_from_fused_evidence(evidence: FusedEvidence) -> EntitySemanticState:
    """Map the evidence fused over one support into the semantic state of an entity.

    Nothing is dropped: the hypotheses, their evidence and the uncertainty records are kept as
    fused. A primary hypothesis and the ``class`` attribute are added only when the state is
    unambiguous.

    Args:
        evidence: The evidence fused over the support the entity is materialized from.

    Returns:
        The semantic state, whose ambiguity state is derived from the same records it keeps.
    """
    hypotheses = tuple(
        EntityHypothesis(
            fused_evidence_id=evidence.fused_evidence_id,
            hypothesis_id=item.hypothesis_id,
            label=item.label,
            evidence=item.evidence,
        )
        for item in evidence.hypotheses
    )
    uncertainty = tuple(
        EntityUncertainty(fused_evidence_id=evidence.fused_evidence_id, record=record)
        for record in evidence.uncertainty
    )
    ambiguity_state = derive_ambiguity_state(hypotheses, uncertainty)
    primary = hypotheses[0] if ambiguity_state is AmbiguityState.UNAMBIGUOUS else None
    return EntitySemanticState(
        hypotheses=hypotheses,
        ambiguity_state=ambiguity_state,
        provenance=SemanticStateProvenance(
            mapping_rule_id=SEMANTIC_STATE_MAPPING_RULE_ID,
            primary_policy_id=PRIMARY_HYPOTHESIS_POLICY_ID,
        ),
        primary_hypothesis=None if primary is None else primary.ref,
        attributes=() if primary is None else (_class_attribute(primary),),
        uncertainty=uncertainty,
    )


def _class_attribute(primary: EntityHypothesis) -> EntityAttribute:
    """The ``class`` attribute, citing exactly the claims that support the primary hypothesis."""
    return EntityAttribute(
        name="class",
        value=primary.label,
        origin=AttributeOrigin.DERIVED,
        derivation_id=CLASS_ATTRIBUTE_DERIVATION_ID,
        evidence=tuple(
            EvidenceReference(contribution_id=item.contribution_id, claim_id=item.claim_id)
            for item in primary.evidence
            if item.stance is EvidenceStance.SUPPORTING
        ),
    )
