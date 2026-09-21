import dataclasses
import math

import pytest
from fusion_builders import (
    claim_signal,
    contribution_id_for,
    evidence,
    frame_timestamp,
    make_contribution,
    make_fused_evidence,
    make_group,
    make_hypothesis,
    make_interpreter,
    make_point_representation_ref,
    make_scorer,
    make_time_bounds,
    make_uncertainty,
    scorer_signal,
    spatial_id,
)

from contextmap.semantic_fusion import (
    EvidenceContributionId,
    EvidenceReference,
    EvidenceStance,
    FusedEvidence,
    FusedHypothesis,
    FusedHypothesisId,
    ScoreReference,
    SupportSignal,
    SupportSignalKind,
    UncertaintyKind,
)
from contextmap.sensor_association import QualityComponent
from contextmap.visual_perception import ClaimId, HypothesisRole, PerceptionResultId

THREE_RUNS = ("run-a", "run-b", "run-c")
HYPOTHESIS_ID = FusedHypothesisId("hypothesis-0001")


def _three_runs_over_one_frame() -> FusedEvidence:
    contributions = tuple(make_contribution(run=run) for run in THREE_RUNS)
    items = sorted(
        (evidence(run=run) for run in THREE_RUNS),
        key=lambda item: (item.contribution_id, item.claim_id),
    )
    hypothesis = make_hypothesis(items=tuple(items))
    return make_fused_evidence(
        contributions=contributions,
        groups=(make_group(runs=THREE_RUNS),),
        hypotheses=(hypothesis,),
    )


def _two_frames_two_labels() -> FusedEvidence:
    contributions = (
        make_contribution(frame="frame-0120", claim_ids=("claim-0001",)),
        make_contribution(frame="frame-0121", claim_ids=("claim-0001",)),
    )
    door = make_hypothesis(
        "hypothesis-0001",
        "door",
        items=(evidence(frame="frame-0120", signals=(claim_signal(0.9),)),),
    )
    cabinet = make_hypothesis(
        "hypothesis-0002",
        "cabinet",
        items=(evidence(frame="frame-0121", signals=(claim_signal(None),)),),
    )
    return make_fused_evidence(
        contributions=contributions,
        groups=(make_group(frame="frame-0120"), make_group(frame="frame-0121")),
        hypotheses=(door, cabinet),
        temporal_summary=make_time_bounds("frame-0120", "frame-0121"),
    )


def test_three_inference_runs_over_one_frame_remain_one_physical_observation() -> None:
    fused = _three_runs_over_one_frame()

    assert fused.physical_observation_count == 1
    assert fused.inference_result_count == 3
    assert len(fused.contributions) == 3
    assert fused.hypotheses[0].label == "pallet"
    assert fused.supporting_physical_observations(HYPOTHESIS_ID) == ("frame-0120",)


def test_evidence_from_separate_frames_stays_separate_physical_observations() -> None:
    fused = _two_frames_two_labels()

    assert fused.physical_observation_count == 2
    assert fused.inference_result_count == 2
    assert [group.physical_observation_id for group in fused.physical_observation_groups] == [
        "frame-0120",
        "frame-0121",
    ]


def test_supporting_observations_are_counted_once_however_much_geometry_they_see() -> None:
    small = make_fused_evidence(contributions=(make_contribution(geometry_indexes=(0,)),))
    large = make_fused_evidence(contributions=(make_contribution(geometry_indexes=range(500)),))

    assert small.supporting_physical_observations(
        HYPOTHESIS_ID
    ) == large.supporting_physical_observations(HYPOTHESIS_ID)
    assert len(large.contributions[0].geometry_support) == 500


def test_competing_hypotheses_are_kept_without_a_winner() -> None:
    fused = _two_frames_two_labels()
    names = {field.name for field in dataclasses.fields(FusedEvidence)}

    assert [hypothesis.label for hypothesis in fused.hypotheses] == ["door", "cabinet"]
    assert not {"winner", "primary", "primary_hypothesis", "label", "confidence"} & names


def test_unscored_evidence_is_kept_distinct_from_a_zero_score() -> None:
    fused = _two_frames_two_labels()
    door, cabinet = fused.hypotheses

    assert door.evidence[0].signals[0].value == 0.9
    assert cabinet.evidence[0].signals[0].value is None
    assert (
        make_hypothesis(items=(evidence(signals=(claim_signal(0.0),)),))
        .evidence[0]
        .signals[0]
        .value
        == 0.0
    )


def test_every_signal_is_typed_and_tied_to_the_model_that_produced_it() -> None:
    signals = (claim_signal(0.7), scorer_signal(0.31))
    item = evidence(signals=signals)

    assert [signal.kind for signal in item.signals] == [
        SupportSignalKind.CLAIM_CONFIDENCE,
        SupportSignalKind.SCORER_SUPPORT,
    ]
    assert item.signals[0].producer == make_interpreter()
    assert item.signals[1].producer == make_scorer()


def test_observation_quality_and_semantic_signals_have_no_kind_in_common() -> None:
    quality_names = {component.value for component in QualityComponent} | {"observation_quality"}
    signal_names = {kind.value for kind in SupportSignalKind}

    assert not signal_names & quality_names
    assert signal_names == {"claim_confidence", "scorer_support"}


def test_stance_and_role_of_every_claim_are_preserved() -> None:
    conflicting = evidence(
        frame="frame-0121",
        stance=EvidenceStance.CONFLICTING,
        role=HypothesisRole.ALTERNATIVE,
    )

    assert conflicting.stance is EvidenceStance.CONFLICTING
    assert conflicting.role is HypothesisRole.ALTERNATIVE


def test_a_conflict_names_the_exact_observations_and_claims_that_produced_it() -> None:
    fused = _two_frames_two_labels()
    references = (
        EvidenceReference(
            contribution_id=contribution_id_for("run-a", "frame-0120"),
            claim_id=ClaimId("claim-0001"),
        ),
        EvidenceReference(
            contribution_id=contribution_id_for("run-a", "frame-0121"),
            claim_id=ClaimId("claim-0001"),
        ),
    )
    conflict = make_uncertainty(
        UncertaintyKind.CONTRADICTION,
        hypothesis_ids=("hypothesis-0001", "hypothesis-0002"),
        references=references,
    )

    with_conflict = dataclasses.replace(fused, uncertainty=(conflict,))

    assert with_conflict.uncertainty[0].evidence == references
    assert with_conflict.uncertainty[0].kind is UncertaintyKind.CONTRADICTION


def test_insufficient_evidence_points_at_contributions_without_claims() -> None:
    silent = make_contribution(claim_ids=(), with_quality=False)
    insufficient = make_uncertainty(
        UncertaintyKind.INSUFFICIENT_EVIDENCE,
        references=(EvidenceReference(contribution_id=silent.contribution_id, claim_id=None),),
    )

    fused = make_fused_evidence(contributions=(silent,), hypotheses=(), uncertainty=(insufficient,))

    assert fused.hypotheses == ()
    assert fused.uncertainty[0].evidence[0].claim_id is None


def test_static_structural_evidence_is_attached_once_to_the_support() -> None:
    reference = make_point_representation_ref()
    fused = make_fused_evidence(point_representation_refs=(reference,))

    assert fused.point_representation_refs == (reference,)


def test_a_signal_value_is_either_absent_or_a_finite_score_in_the_unit_interval() -> None:
    for value in (None, 0.0, 0.5, 1.0):
        assert claim_signal(value).value == value
    for invalid in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="value"):
            claim_signal(invalid)


def test_a_signal_needs_an_identified_producer() -> None:
    anonymous = dataclasses.replace(make_interpreter(), backend_id="")

    with pytest.raises(ValueError, match="producer"):
        SupportSignal(kind=SupportSignalKind.CLAIM_CONFIDENCE, producer=anonymous, value=0.5)


def test_signals_of_one_evidence_item_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="signals must be sorted and unique"):
        evidence(signals=(claim_signal(0.5), claim_signal(0.5)))
    with pytest.raises(ValueError, match="signals must be sorted and unique"):
        evidence(signals=(scorer_signal(0.4), claim_signal(0.5)))


def test_a_hypothesis_needs_a_label_and_at_least_one_supporting_claim() -> None:
    with pytest.raises(ValueError, match="label"):
        FusedHypothesis(hypothesis_id=HYPOTHESIS_ID, label=" ", evidence=(evidence(),))
    with pytest.raises(ValueError, match="supporting"):
        make_hypothesis(items=(evidence(stance=EvidenceStance.CONFLICTING),))


def test_hypothesis_evidence_must_be_sorted_and_unique() -> None:
    first = evidence(frame="frame-0120")
    second = evidence(frame="frame-0121")

    with pytest.raises(ValueError, match="evidence must be sorted and unique"):
        make_hypothesis(items=(second, first))
    with pytest.raises(ValueError, match="evidence must be sorted and unique"):
        make_hypothesis(items=(first, first))


def test_evidence_that_references_an_unknown_contribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown contribution"):
        make_fused_evidence(hypotheses=(make_hypothesis(items=(evidence(run="run-z"),)),))


def test_evidence_for_a_claim_the_contribution_does_not_carry_is_rejected() -> None:
    with pytest.raises(ValueError, match="claim-0099"):
        make_fused_evidence(hypotheses=(make_hypothesis(items=(evidence(claim_id="claim-0099"),)),))


def test_a_scorer_signal_needs_the_score_reference_of_its_contribution() -> None:
    hypothesis = make_hypothesis(items=(evidence(signals=(scorer_signal(0.3),)),))

    with pytest.raises(ValueError, match="score reference"):
        make_fused_evidence(hypotheses=(hypothesis,))

    scored = make_contribution(
        score_refs=(ScoreReference(claim_id=ClaimId("claim-0001"), scorer=make_scorer()),)
    )
    assert make_fused_evidence(contributions=(scored,), hypotheses=(hypothesis,))


def test_hypothesis_labels_and_identities_are_unique() -> None:
    same_label = (
        make_hypothesis("hypothesis-0001", "pallet"),
        make_hypothesis("hypothesis-0002", "pallet"),
    )
    same_id = (
        make_hypothesis("hypothesis-0001", "pallet"),
        make_hypothesis("hypothesis-0001", "crate"),
    )

    with pytest.raises(ValueError, match="hypotheses must be sorted and unique"):
        make_fused_evidence(hypotheses=same_id)
    with pytest.raises(ValueError, match="label"):
        make_fused_evidence(hypotheses=same_label)


def test_a_frame_without_a_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in any physical observation group"):
        make_fused_evidence(
            contributions=(
                make_contribution(frame="frame-0120"),
                make_contribution(frame="frame-0121"),
            ),
            groups=(make_group(frame="frame-0120"),),
            hypotheses=(),
            temporal_summary=make_time_bounds("frame-0120", "frame-0121"),
        )


def test_a_group_that_omits_a_contribution_of_its_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="spatial observations"):
        make_fused_evidence(
            contributions=(make_contribution(run="run-a"), make_contribution(run="run-b")),
            groups=(make_group(runs=("run-a",)),),
            hypotheses=(),
        )


def test_a_group_that_disagrees_about_the_inference_results_is_rejected() -> None:
    group = dataclasses.replace(
        make_group(runs=("run-a", "run-b")),
        perception_result_ids=(PerceptionResultId("run-a--frame-0120"),),
    )

    with pytest.raises(ValueError, match="perception results"):
        make_fused_evidence(
            contributions=(make_contribution(run="run-a"), make_contribution(run="run-b")),
            groups=(group,),
            hypotheses=(),
        )


def test_a_group_naming_a_spatial_observation_of_another_frame_is_rejected() -> None:
    stray = dataclasses.replace(
        make_group(frame="frame-0120"),
        spatial_observation_ids=(spatial_id("run-a", "frame-0121"),),
    )

    with pytest.raises(ValueError, match="spatial observations"):
        make_fused_evidence(groups=(stray,), hypotheses=())


def test_one_spatial_observation_cannot_contribute_twice() -> None:
    twin = dataclasses.replace(
        make_contribution(), contribution_id=EvidenceContributionId("contribution--twin")
    )

    with pytest.raises(ValueError, match="more than one contribution"):
        make_fused_evidence(contributions=(make_contribution(), twin), hypotheses=())


def test_one_inference_result_cannot_belong_to_two_physical_observations() -> None:
    liar = dataclasses.replace(
        make_contribution(frame="frame-0121"),
        perception_result_id=PerceptionResultId("run-a--frame-0120"),
    )

    with pytest.raises(ValueError, match="perception result"):
        make_fused_evidence(
            contributions=(make_contribution(frame="frame-0120"), liar),
            groups=(make_group(frame="frame-0120"), make_group(frame="frame-0121")),
            hypotheses=(),
            temporal_summary=make_time_bounds("frame-0120", "frame-0121"),
        )


def test_the_temporal_summary_must_span_exactly_the_observed_frames() -> None:
    with pytest.raises(ValueError, match="temporal_summary"):
        make_fused_evidence(temporal_summary=make_time_bounds("frame-0120", "frame-0125"))
    assert make_fused_evidence().temporal_summary.start == frame_timestamp("frame-0120")


def test_uncertainty_must_reference_known_hypotheses_and_evidence() -> None:
    unknown_hypothesis = make_uncertainty(
        UncertaintyKind.AMBIGUITY, hypothesis_ids=("hypothesis-0001", "hypothesis-0009")
    )
    with pytest.raises(ValueError, match="unknown hypothesis"):
        make_fused_evidence(uncertainty=(unknown_hypothesis,))

    stray = make_uncertainty(
        UncertaintyKind.INSUFFICIENT_EVIDENCE,
        references=(
            EvidenceReference(
                contribution_id=contribution_id_for("run-z", "frame-0120"), claim_id=None
            ),
        ),
    )
    with pytest.raises(ValueError, match="unknown contribution"):
        make_fused_evidence(uncertainty=(stray,))


@pytest.mark.parametrize(
    "kind", [UncertaintyKind.CONTRADICTION, UncertaintyKind.AMBIGUITY, UncertaintyKind.NEAR_TIE]
)
def test_competition_needs_at_least_two_hypotheses(kind: UncertaintyKind) -> None:
    with pytest.raises(ValueError, match="at least two hypotheses"):
        make_uncertainty(kind, hypothesis_ids=("hypothesis-0001",))


def test_a_contradiction_needs_more_than_one_physical_observation() -> None:
    fused = _two_frames_two_labels()
    one_frame = make_uncertainty(
        UncertaintyKind.CONTRADICTION,
        hypothesis_ids=("hypothesis-0001", "hypothesis-0002"),
        references=(
            EvidenceReference(
                contribution_id=contribution_id_for("run-a", "frame-0120"),
                claim_id=ClaimId("claim-0001"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="distinct physical observations"):
        dataclasses.replace(fused, uncertainty=(one_frame,))


def test_uncertainty_needs_a_versioned_rule() -> None:
    with pytest.raises(ValueError, match="rule_id"):
        make_uncertainty(UncertaintyKind.INSUFFICIENT_EVIDENCE, rule_id="")


def test_structural_references_are_unique_and_sorted() -> None:
    first = make_point_representation_ref(representation_id="repr-0001")
    second = make_point_representation_ref(representation_id="repr-0002", geometry_index=1)

    with pytest.raises(ValueError, match="point_representation_refs must be sorted and unique"):
        make_fused_evidence(point_representation_refs=(first, first))
    with pytest.raises(ValueError, match="point_representation_refs must be sorted and unique"):
        make_fused_evidence(point_representation_refs=(second, first))


def test_a_fused_evidence_needs_its_provenance_and_at_least_one_contribution() -> None:
    with pytest.raises(ValueError, match="contribution"):
        make_fused_evidence(contributions=(), groups=(), hypotheses=())
    with pytest.raises(ValueError, match="fusion_policy_id"):
        dataclasses.replace(make_fused_evidence().provenance, fusion_policy_id="")
