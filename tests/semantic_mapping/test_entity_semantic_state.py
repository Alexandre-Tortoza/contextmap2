import dataclasses
import json

import pytest
from mapping_builders import (
    FUSED_EVIDENCE_ID,
    claim_signal,
    make_entity,
    make_evidence_item,
    make_evidence_links,
    make_hypothesis,
    make_semantic_state,
    scorer_signal,
)
from mapping_fusion import ClaimSpec, View, fuse

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceReference,
    EvidenceStance,
    FusedEvidence,
    FusedEvidenceId,
    FusedHypothesisId,
    SupportSignalKind,
    UncertaintyKind,
    UncertaintyRecord,
)
from contextmap.semantic_mapping import (
    CLASS_ATTRIBUTE_DERIVATION_ID,
    PRIMARY_HYPOTHESIS_POLICY_ID,
    SEMANTIC_STATE_MAPPING_RULE_ID,
    AmbiguityState,
    AttributeOrigin,
    EntityAttribute,
    EntityEvidenceLinks,
    EntityHypothesisRef,
    EntitySemanticState,
    EntityUncertainty,
    SemanticStateProvenance,
    derive_ambiguity_state,
    semantic_state_from_fused_evidence,
)
from contextmap.semantic_mapping.serialization import decode_entity, encode_entity
from contextmap.visual_perception import HypothesisRole

UNKNOWN = BaselineAccumulationPolicy(abstention_labels=frozenset({"unknown"}))


def _state(
    views: list[View], policy: BaselineAccumulationPolicy | None = None
) -> EntitySemanticState:
    return semantic_state_from_fused_evidence(fuse(views, policy=policy)[1])


def _fused(views: list[View], policy: BaselineAccumulationPolicy | None = None) -> FusedEvidence:
    return fuse(views, policy=policy)[1]


def _links_for(evidence: FusedEvidence) -> EntityEvidenceLinks:
    return make_evidence_links(
        fused_evidence_id=evidence.fused_evidence_id,
        fusion_support_id=evidence.fusion_support_id,
    )


def _round_trip(state: EntitySemanticState, evidence: FusedEvidence) -> EntitySemanticState:
    entity = make_entity(semantic_state=state, evidence=_links_for(evidence))
    return decode_entity(json.loads(json.dumps(encode_entity(entity)))).semantic_state


class TestMappingFromFusedEvidence:
    def test_a_single_unambiguous_hypothesis_exposes_a_primary_and_a_derived_class(self) -> None:
        state = _state([View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.8),))])

        assert state.ambiguity_state is AmbiguityState.UNAMBIGUOUS
        assert state.primary is not None and state.primary.label == "pallet"
        assert state.alternative_hypotheses == ()
        (attribute,) = state.attributes
        assert (attribute.name, attribute.value) == ("class", "pallet")
        assert attribute.origin is AttributeOrigin.DERIVED
        assert attribute.derivation_id == CLASS_ATTRIBUTE_DERIVATION_ID
        assert state.provenance == SemanticStateProvenance(
            mapping_rule_id=SEMANTIC_STATE_MAPPING_RULE_ID,
            primary_policy_id=PRIMARY_HYPOTHESIS_POLICY_ID,
        )

    def test_the_class_attribute_cites_exactly_the_claims_that_support_the_primary(self) -> None:
        state = _state(
            [
                View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.8),)),
                View("run-a", "frame-0121", (ClaimSpec("pallet", confidence=0.6),)),
            ]
        )

        supporting = {
            EvidenceReference(contribution_id=item.contribution_id, claim_id=item.claim_id)
            for item in state.hypotheses[0].evidence
            if item.stance is EvidenceStance.SUPPORTING
        }
        assert set(state.attributes[0].evidence) == supporting and len(supporting) == 2

    def test_alternatives_are_kept_and_no_primary_is_invented(self) -> None:
        state = _state(
            [
                View(
                    "run-a",
                    "frame-0120",
                    (
                        ClaimSpec("pallet", confidence=0.8),
                        ClaimSpec("wooden crate", role=HypothesisRole.ALTERNATIVE, confidence=0.4),
                    ),
                )
            ]
        )

        assert state.ambiguity_state is AmbiguityState.AMBIGUOUS
        assert state.primary_hypothesis is None and state.primary is None
        assert [item.label for item in state.alternative_hypotheses] == ["pallet", "wooden crate"]
        assert state.attributes == ()

    def test_a_contradiction_between_observations_is_kept_as_a_conflict(self) -> None:
        state = _state(
            [
                View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9),)),
                View("run-a", "frame-0121", (ClaimSpec("cabinet", confidence=0.7),)),
            ]
        )

        assert state.ambiguity_state is AmbiguityState.CONFLICTING
        assert state.primary is None
        (conflict,) = state.conflicts
        assert conflict.record.kind is UncertaintyKind.CONTRADICTION
        assert len(conflict.record.evidence) == 2

    def test_a_near_tie_stays_ambiguous_rather_than_electing_a_winner(self) -> None:
        views = [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-a", "frame-0121", (ClaimSpec("door"),)),
            View("run-a", "frame-0122", (ClaimSpec("cabinet"),)),
        ]

        state = _state(views, BaselineAccumulationPolicy(near_tie_margin=1))

        kinds = {item.record.kind for item in state.uncertainty}
        assert UncertaintyKind.NEAR_TIE in kinds
        assert state.primary is None

    def test_an_abstention_is_neither_a_hypothesis_nor_evidence_against_one(self) -> None:
        state = _state(
            [
                View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9),)),
                View("run-a", "frame-0121", (ClaimSpec("unknown"),)),
            ],
            UNKNOWN,
        )

        assert [item.label for item in state.hypotheses] == ["door"]
        stances = {item.stance for item in state.hypotheses[0].evidence}
        assert EvidenceStance.ABSTAINING in stances
        assert state.ambiguity_state is AmbiguityState.UNAMBIGUOUS

    def test_only_abstentions_leave_no_hypothesis_and_insufficient_evidence(self) -> None:
        fused = _fused([View("run-a", "frame-0120", (ClaimSpec("unknown"),))], UNKNOWN)

        state = semantic_state_from_fused_evidence(fused)

        assert state.hypotheses == ()
        assert state.ambiguity_state is AmbiguityState.INSUFFICIENT_EVIDENCE
        assert state.primary is None and state.attributes == ()
        (item,) = state.uncertainty
        assert item.record.kind is UncertaintyKind.INSUFFICIENT_EVIDENCE
        assert len(item.record.evidence) == 1

    def test_a_view_without_claims_is_insufficient_evidence_not_an_empty_label(self) -> None:
        state = _state([View("run-a", "frame-0120", ())])

        assert state.ambiguity_state is AmbiguityState.INSUFFICIENT_EVIDENCE
        assert state.hypotheses == ()

    def test_an_unscored_claim_stays_distinct_from_a_zero_score(self) -> None:
        state = _state(
            [
                View("run-a", "frame-0120", (ClaimSpec("door", confidence=None),)),
                View("run-a", "frame-0121", (ClaimSpec("door", confidence=0.0),)),
            ]
        )

        values = {
            signal.value
            for item in state.hypotheses[0].evidence
            for signal in item.signals
            if signal.kind is SupportSignalKind.CLAIM_CONFIDENCE
        }
        assert values == {None, 0.0}

    def test_nothing_fused_is_dropped_or_rewritten(self) -> None:
        fused = _fused(
            [
                View(
                    "run-a",
                    "frame-0120",
                    (
                        ClaimSpec("pallet", confidence=0.8),
                        ClaimSpec("wooden crate", role=HypothesisRole.ALTERNATIVE),
                    ),
                ),
                View("run-b", "frame-0120", (ClaimSpec("pallet", confidence=None),)),
                View("run-a", "frame-0121", (ClaimSpec("unknown"),)),
            ],
            UNKNOWN,
        )

        state = semantic_state_from_fused_evidence(fused)

        assert [(item.hypothesis_id, item.label, item.evidence) for item in state.hypotheses] == [
            (item.hypothesis_id, item.label, item.evidence) for item in fused.hypotheses
        ]
        assert [item.record for item in state.uncertainty] == list(fused.uncertainty)
        assert {item.fused_evidence_id for item in state.hypotheses} == {fused.fused_evidence_id}

    def test_the_mapping_is_deterministic(self) -> None:
        fused = _fused([View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.8),))])

        assert semantic_state_from_fused_evidence(fused) == semantic_state_from_fused_evidence(
            fused
        )

    def test_the_state_needs_no_perception_or_model_backend_to_be_inspected(self) -> None:
        state = _state([View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.8),))])

        # Só identidades, textos e números: nada de objetos de backend.
        assert isinstance(state.primary_hypothesis, EntityHypothesisRef)
        assert isinstance(state.hypotheses[0].label, str)

    @pytest.mark.parametrize(
        "views",
        [
            [View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.8),))],
            [
                View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9),)),
                View("run-a", "frame-0121", (ClaimSpec("cabinet", confidence=0.7),)),
            ],
            [View("run-a", "frame-0120", (ClaimSpec("unknown"),))],
        ],
    )
    def test_every_state_survives_persistence_unchanged(self, views: list[View]) -> None:
        fused = _fused(views, UNKNOWN)
        state = semantic_state_from_fused_evidence(fused)

        assert _round_trip(state, fused) == state


class TestAmbiguityDerivation:
    def _uncertainty(self, kind: UncertaintyKind, *ids: str) -> EntityUncertainty:
        return EntityUncertainty(
            fused_evidence_id=FUSED_EVIDENCE_ID,
            record=UncertaintyRecord(
                kind=kind,
                hypothesis_ids=tuple(FusedHypothesisId(item) for item in ids),
                evidence=(),
                rule_id="rule",
            ),
        )

    def test_no_hypothesis_is_insufficient_evidence(self) -> None:
        assert derive_ambiguity_state((), ()) is AmbiguityState.INSUFFICIENT_EVIDENCE

    def test_one_hypothesis_with_nothing_competing_is_unambiguous(self) -> None:
        assert derive_ambiguity_state((make_hypothesis(),), ()) is AmbiguityState.UNAMBIGUOUS

    def test_several_hypotheses_of_one_evidence_compete_even_without_a_record(self) -> None:
        hypotheses = (
            make_hypothesis("hypothesis-0001", "pallet"),
            make_hypothesis("hypothesis-0002", "crate"),
        )

        assert derive_ambiguity_state(hypotheses, ()) is AmbiguityState.AMBIGUOUS

    def test_a_contradiction_dominates_ambiguity(self) -> None:
        hypotheses = (
            make_hypothesis("hypothesis-0001", "pallet"),
            make_hypothesis("hypothesis-0002", "crate"),
        )
        records = (
            self._uncertainty(UncertaintyKind.AMBIGUITY, "hypothesis-0001", "hypothesis-0002"),
            self._uncertainty(UncertaintyKind.CONTRADICTION, "hypothesis-0001", "hypothesis-0002"),
        )

        assert derive_ambiguity_state(hypotheses, records) is AmbiguityState.CONFLICTING

    def test_an_insufficient_evidence_record_marks_the_state_insufficient(self) -> None:
        records = (self._uncertainty(UncertaintyKind.INSUFFICIENT_EVIDENCE),)

        assert derive_ambiguity_state((), records) is AmbiguityState.INSUFFICIENT_EVIDENCE


class TestSemanticStateContract:
    def test_the_ambiguity_state_must_be_the_one_the_records_imply(self) -> None:
        with pytest.raises(ValueError, match="does not match the hypotheses"):
            dataclasses.replace(make_semantic_state(), ambiguity_state=AmbiguityState.CONFLICTING)

    def test_a_primary_cannot_hide_alternatives(self) -> None:
        pallet = make_hypothesis("hypothesis-0001", "pallet")
        crate = make_hypothesis("hypothesis-0002", "crate")

        with pytest.raises(ValueError, match="alternatives would be hidden"):
            make_semantic_state((pallet, crate), primary=pallet.ref)

    def test_a_primary_must_be_a_known_hypothesis(self) -> None:
        stranger = EntityHypothesisRef(
            fused_evidence_id=FUSED_EVIDENCE_ID, hypothesis_id=FusedHypothesisId("hypothesis-9999")
        )

        with pytest.raises(ValueError, match="is not known"):
            make_semantic_state(primary=stranger)

    def test_uncertainty_must_name_known_hypotheses(self) -> None:
        item = EntityUncertainty(
            fused_evidence_id=FUSED_EVIDENCE_ID,
            record=UncertaintyRecord(
                kind=UncertaintyKind.AMBIGUITY,
                hypothesis_ids=(
                    FusedHypothesisId("hypothesis-0001"),
                    FusedHypothesisId("hypothesis-0002"),
                ),
                evidence=(),
                rule_id="rule",
            ),
        )

        with pytest.raises(ValueError, match="unknown hypothesis"):
            make_semantic_state(uncertainty=(item,))

    def test_uncertainty_must_come_from_evidence_the_entity_links_to(self) -> None:
        item = EntityUncertainty(
            fused_evidence_id=FusedEvidenceId("fused--other"),
            record=UncertaintyRecord(
                kind=UncertaintyKind.INSUFFICIENT_EVIDENCE,
                hypothesis_ids=(),
                evidence=(),
                rule_id="rule",
            ),
        )
        state = make_semantic_state((), uncertainty=(item,))

        with pytest.raises(ValueError, match="does not link to"):
            make_entity(semantic_state=state)

    def test_uncertainty_is_canonically_ordered(self) -> None:
        def record(rule: str) -> EntityUncertainty:
            return EntityUncertainty(
                fused_evidence_id=FUSED_EVIDENCE_ID,
                record=UncertaintyRecord(
                    kind=UncertaintyKind.INSUFFICIENT_EVIDENCE,
                    hypothesis_ids=(),
                    evidence=(),
                    rule_id=rule,
                ),
            )

        with pytest.raises(ValueError, match="sorted and unique"):
            make_semantic_state((), uncertainty=(record("b"), record("a")))

    def test_an_attribute_is_not_a_hypothesis(self) -> None:
        attribute = _attribute("material", "wood")

        state = make_semantic_state(attributes=(attribute,))

        assert [item.label for item in state.hypotheses] == ["pallet"]
        assert [(item.name, item.value) for item in state.attributes] == [("material", "wood")]

    def test_attributes_are_canonically_ordered(self) -> None:
        with pytest.raises(ValueError, match="sorted and unique"):
            make_semantic_state(
                attributes=(_attribute("material", "wood"), _attribute("class", "pallet"))
            )


def _attribute(
    name: str,
    value: str,
    *,
    origin: AttributeOrigin = AttributeOrigin.OBSERVED,
    evidence: tuple[EvidenceReference, ...] | None = None,
) -> EntityAttribute:
    item = make_evidence_item()
    return EntityAttribute(
        name=name,
        value=value,
        origin=origin,
        derivation_id="test-rule-v1",
        evidence=(
            (EvidenceReference(contribution_id=item.contribution_id, claim_id=item.claim_id),)
            if evidence is None
            else evidence
        ),
    )


class TestAttributes:
    def test_an_observed_attribute_needs_evidence(self) -> None:
        with pytest.raises(ValueError, match="cites no evidence"):
            _attribute("material", "wood", evidence=())

    def test_a_derived_attribute_needs_evidence(self) -> None:
        with pytest.raises(ValueError, match="cites no evidence"):
            _attribute("class", "pallet", origin=AttributeOrigin.DERIVED, evidence=())

    def test_external_knowledge_is_accepted_without_evidence_but_stays_labeled(self) -> None:
        attribute = _attribute(
            "usual_use", "storage", origin=AttributeOrigin.EXTERNAL_KNOWLEDGE, evidence=()
        )

        assert attribute.origin is AttributeOrigin.EXTERNAL_KNOWLEDGE
        assert attribute.derivation_id == "test-rule-v1"

    def test_a_derivation_is_always_required(self) -> None:
        with pytest.raises(ValueError, match="derivation_id"):
            EntityAttribute(
                name="material",
                value="wood",
                origin=AttributeOrigin.EXTERNAL_KNOWLEDGE,
                derivation_id=" ",
            )

    def test_an_attribute_keeps_its_own_typed_support_and_survives_persistence(self) -> None:
        attribute = dataclasses.replace(
            _attribute("material", "wood"), support=(claim_signal(0.6), scorer_signal(None))
        )
        state = make_semantic_state(attributes=(attribute,))
        entity = make_entity(semantic_state=state)

        decoded = decode_entity(json.loads(json.dumps(encode_entity(entity)))).semantic_state

        assert decoded.attributes[0].support[0].value == 0.6
        assert decoded.attributes[0].support[1].value is None
        assert decoded == state

    def test_the_mapping_adds_no_property_that_was_not_observed(self) -> None:
        state = _state([View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.8),))])

        assert [item.name for item in state.attributes] == ["class"]
        assert all(
            item.origin is not AttributeOrigin.EXTERNAL_KNOWLEDGE for item in state.attributes
        )
