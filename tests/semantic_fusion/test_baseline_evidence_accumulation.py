import dataclasses
import random
from collections.abc import Mapping, Sequence

import pytest
from evidence_builders import (
    ClaimSpec,
    Scenario,
    View,
    build_scenario,
    claim_id_of,
    make_score,
)
from fusion_builders import (
    make_interpreter,
    make_point_representation_ref,
    make_scorer,
    spatial_id,
)

from contextmap.semantic_fusion import (
    BASELINE_ACCUMULATION_POLICY_ID,
    EvidenceStance,
    FusedEvidence,
    FusedHypothesis,
    ObservationQualityRef,
    PointRepresentationRef,
    ScoreReference,
    SupportSignalKind,
    accumulate_baseline_evidence,
)
from contextmap.sensor_association import SpatialObservationId
from contextmap.visual_perception import (
    ClaimId,
    HypothesisRole,
    PerceptionResultId,
    SemanticSupport,
)

PALLET = ClaimSpec("pallet", confidence=0.8)


def _accumulate(
    scenario: Scenario,
    *,
    scores: Mapping[PerceptionResultId, Sequence[SemanticSupport]] | None = None,
    qualities: Mapping[SpatialObservationId, ObservationQualityRef] | None = None,
    structure: Sequence[PointRepresentationRef] = (),
) -> FusedEvidence:
    return accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        semantic_scores=scores,
        observation_quality_refs=qualities,
        point_representation_refs=structure,
    )


def _by_label(fused: FusedEvidence) -> dict[str, FusedHypothesis]:
    return {hypothesis.label: hypothesis for hypothesis in fused.hypotheses}


def _stances(hypothesis: FusedHypothesis) -> dict[tuple[str, str], EvidenceStance]:
    return {(item.contribution_id, item.claim_id): item.stance for item in hypothesis.evidence}


def test_three_runs_over_one_frame_are_one_physical_observation_with_three_inferences() -> None:
    scenario = build_scenario(
        [View(run, "frame-0120", (PALLET,)) for run in ("run-a", "run-b", "run-c")]
    )

    fused = _accumulate(scenario)

    assert fused.physical_observation_count == 1
    assert fused.inference_result_count == 3
    pallet = _by_label(fused)["pallet"]
    assert len(pallet.evidence) == 3
    assert fused.supporting_physical_observations(pallet.hypothesis_id) == ("frame-0120",)


def test_the_same_label_from_separate_frames_counts_separate_physical_observations() -> None:
    scenario = build_scenario(
        [View("run-a", "frame-0120", (PALLET,)), View("run-a", "frame-0121", (PALLET,))]
    )

    fused = _accumulate(scenario)

    pallet = _by_label(fused)["pallet"]
    assert fused.supporting_physical_observations(pallet.hypothesis_id) == (
        "frame-0120",
        "frame-0121",
    )


def test_one_claim_over_many_geometry_points_is_not_multiplied() -> None:
    small = _accumulate(
        build_scenario([View("run-a", "frame-0120", (PALLET,), geometry=range(20))])
    )
    large = _accumulate(
        build_scenario([View("run-a", "frame-0120", (PALLET,), geometry=range(500))])
    )

    for fused in (small, large):
        pallet = _by_label(fused)["pallet"]
        assert len(pallet.evidence) == 1
        assert fused.supporting_physical_observations(pallet.hypothesis_id) == ("frame-0120",)
    assert len(large.contributions[0].geometry_support) == 500


def test_competing_labels_from_different_frames_are_conflicting_and_neither_wins() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9),)),
            View("run-a", "frame-0121", (ClaimSpec("cabinet", confidence=0.6),)),
        ]
    )

    fused = _accumulate(scenario)

    door, cabinet = _by_label(fused)["door"], _by_label(fused)["cabinet"]
    assert len(fused.hypotheses) == 2
    assert {item.stance for item in door.evidence} == {
        EvidenceStance.SUPPORTING,
        EvidenceStance.CONFLICTING,
    }
    assert {item.stance for item in cabinet.evidence} == {
        EvidenceStance.SUPPORTING,
        EvidenceStance.CONFLICTING,
    }
    assert {item.contribution_id for item in door.evidence} == {
        item.contribution_id for item in cabinet.evidence
    }
    assert fused.supporting_physical_observations(door.hypothesis_id) == ("frame-0120",)
    assert fused.supporting_physical_observations(cabinet.hypothesis_id) == ("frame-0121",)


def test_alternatives_of_one_interpretation_are_ambiguous_and_not_conflicting() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (ClaimSpec("door"), ClaimSpec("cabinet", role=HypothesisRole.ALTERNATIVE)),
    )

    fused = _accumulate(build_scenario([view]))

    door, cabinet = _by_label(fused)["door"], _by_label(fused)["cabinet"]
    assert [item.stance for item in door.evidence] == [
        EvidenceStance.SUPPORTING,
        EvidenceStance.AMBIGUOUS,
    ]
    assert [item.stance for item in cabinet.evidence] == [
        EvidenceStance.AMBIGUOUS,
        EvidenceStance.SUPPORTING,
    ]
    assert cabinet.evidence[1].role is HypothesisRole.ALTERNATIVE


def test_an_alternative_from_another_view_is_ambiguous_and_its_primary_is_conflicting() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-a", "frame-0121", (ClaimSpec("cabinet", role=HypothesisRole.ALTERNATIVE),)),
        ]
    )

    fused = _accumulate(scenario)

    door, cabinet = _by_label(fused)["door"], _by_label(fused)["cabinet"]
    assert [item.stance for item in door.evidence] == [
        EvidenceStance.SUPPORTING,
        EvidenceStance.AMBIGUOUS,
    ]
    assert [item.stance for item in cabinet.evidence] == [
        EvidenceStance.CONFLICTING,
        EvidenceStance.SUPPORTING,
    ]


def test_two_runs_that_disagree_over_one_frame_conflict_but_stay_one_physical_observation() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-b", "frame-0120", (ClaimSpec("cabinet"),)),
        ]
    )

    fused = _accumulate(scenario)

    assert fused.physical_observation_count == 1
    assert fused.inference_result_count == 2
    assert set(_stances(_by_label(fused)["door"]).values()) == {
        EvidenceStance.SUPPORTING,
        EvidenceStance.CONFLICTING,
    }


def test_unscored_claims_survive_with_an_explicit_absence_and_never_a_zero() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door", confidence=None),)),
            View("run-a", "frame-0121", (ClaimSpec("door", confidence=0.0),)),
        ]
    )

    door = _by_label(_accumulate(scenario))["door"]

    values = [item.signals[0].value for item in door.evidence]
    assert values == [None, 0.0]
    assert all(item.signals[0].kind is SupportSignalKind.CLAIM_CONFIDENCE for item in door.evidence)


def test_claim_confidence_and_scorer_support_stay_separate_typed_signals() -> None:
    view = View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.7),))
    scenario = build_scenario([view])
    scores = {scenario.grouping.groups[0].perception_result_ids[0]: [make_score(view, 0, 0.31)]}

    fused = _accumulate(scenario, scores=scores)

    signals = _by_label(fused)["pallet"].evidence[0].signals
    assert [(signal.kind, signal.value) for signal in signals] == [
        (SupportSignalKind.CLAIM_CONFIDENCE, 0.7),
        (SupportSignalKind.SCORER_SUPPORT, 0.31),
    ]
    assert signals[0].producer == make_interpreter()
    assert signals[1].producer == make_scorer()
    assert fused.contributions[0].score_refs == (
        ScoreReference(claim_id=claim_id_of(view, 0), scorer=make_scorer()),
    )


def test_two_scorers_of_one_claim_are_both_kept() -> None:
    view = View("run-a", "frame-0120", (ClaimSpec("pallet"),))
    scenario = build_scenario([view])
    key = scenario.grouping.groups[0].perception_result_ids[0]
    scores = {
        key: [make_score(view, 0, 0.2, scorer="clip_b"), make_score(view, 0, 0.9, scorer="clip_a")]
    }

    signals = _by_label(_accumulate(scenario, scores=scores))["pallet"].evidence[0].signals

    scorer_signals = [s for s in signals if s.kind is SupportSignalKind.SCORER_SUPPORT]
    assert [(s.producer.backend_id, s.value) for s in scorer_signals] == [
        ("clip_a", 0.9),
        ("clip_b", 0.2),
    ]


def test_the_same_scorer_twice_for_one_claim_is_rejected() -> None:
    view = View("run-a", "frame-0120", (ClaimSpec("pallet"),))
    scenario = build_scenario([view])
    key = scenario.grouping.groups[0].perception_result_ids[0]
    scores = {key: [make_score(view, 0, 0.2), make_score(view, 0, 0.9)]}

    with pytest.raises(ValueError, match="more than one score"):
        _accumulate(scenario, scores=scores)


def test_observation_quality_is_kept_by_reference_and_never_changes_the_hypotheses() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9),)),
            View("run-a", "frame-0121", (ClaimSpec("door", confidence=0.9),)),
        ]
    )
    refs = {
        observation_id: ObservationQualityRef(
            spatial_observation_id=observation_id, definitions_version="observation-quality-v1"
        )
        for observation_id in scenario.observations
    }

    plain = _accumulate(scenario)
    with_quality = _accumulate(scenario, qualities=refs)

    assert with_quality.hypotheses == plain.hypotheses
    assert [c.observation_quality for c in plain.contributions] == [None, None]
    assert [c.observation_quality for c in with_quality.contributions] == [
        refs[c.spatial_observation_id] for c in with_quality.contributions
    ]


def test_point_representation_is_attached_once_and_only_where_it_is_anchored() -> None:
    scenario = build_scenario(
        [View("run-a", "frame-0120", (PALLET,)), View("run-a", "frame-0121", (PALLET,))]
    )
    inside = make_point_representation_ref(representation_id="repr-0001", geometry_index=3)
    outside = make_point_representation_ref(representation_id="repr-0002", geometry_index=500)

    fused = _accumulate(scenario, structure=[inside, outside])

    assert fused.point_representation_refs == (inside,)
    assert all(
        not hasattr(contribution, "point_representation_refs")
        for contribution in fused.contributions
    )


def test_without_structure_the_baseline_uses_none() -> None:
    fused = _accumulate(build_scenario([View("run-a", "frame-0120", (PALLET,))]))

    assert fused.point_representation_refs == ()


def test_spelling_differences_are_typographic_and_synonyms_are_not_assumed() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("pallet"),)),
            View("run-a", "frame-0121", (ClaimSpec("pallet "),)),
            View("run-a", "frame-0122", (ClaimSpec("  Pallet"),)),
            View("run-a", "frame-0123", (ClaimSpec("wooden pallet"),)),
        ]
    )

    fused = _accumulate(scenario)

    assert [hypothesis.label for hypothesis in fused.hypotheses] == ["pallet", "wooden pallet"]
    supporting = [i for i in fused.hypotheses[0].evidence if i.stance is EvidenceStance.SUPPORTING]
    assert len(supporting) == 3


def test_the_label_is_the_most_common_spelling() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("Pallet"),)),
            View("run-a", "frame-0121", (ClaimSpec("pallet"),)),
            View("run-a", "frame-0122", (ClaimSpec("pallet"),)),
        ]
    )

    assert [h.label for h in _accumulate(scenario).hypotheses] == ["pallet"]


def test_hypotheses_are_ordered_by_label_and_never_by_support() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("box"),)),
            View("run-a", "frame-0121", (ClaimSpec("box"),)),
            View("run-a", "frame-0122", (ClaimSpec("box"),)),
            View("run-a", "frame-0123", (ClaimSpec("ax"),)),
        ]
    )

    fused = _accumulate(scenario)

    assert [(h.hypothesis_id, h.label) for h in fused.hypotheses] == [
        ("hypothesis-0001", "ax"),
        ("hypothesis-0002", "box"),
    ]


def test_views_without_claims_stay_as_contributions_without_creating_a_hypothesis() -> None:
    fused = _accumulate(build_scenario([View("run-a", "frame-0120", ())]))

    assert fused.hypotheses == ()
    assert len(fused.contributions) == 1
    assert fused.contributions[0].claim_refs == ()
    assert fused.physical_observation_count == 1


def test_every_hypothesis_is_traceable_to_its_exact_contributions() -> None:
    scenario = build_scenario(
        [
            View(
                "run-a",
                "frame-0120",
                (ClaimSpec("door"), ClaimSpec("cabinet", role=HypothesisRole.ALTERNATIVE)),
            ),
            View("run-b", "frame-0120", (ClaimSpec("door"),)),
        ]
    )

    fused = _accumulate(scenario)

    contributions = {c.contribution_id: c for c in fused.contributions}
    for hypothesis in fused.hypotheses:
        for item in hypothesis.evidence:
            claims = {ref.claim_id for ref in contributions[item.contribution_id].claim_refs}
            assert item.claim_id in claims


def test_the_result_does_not_depend_on_the_order_of_the_evidence() -> None:
    views = [
        View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9),)),
        View(
            "run-b",
            "frame-0120",
            (ClaimSpec("door"), ClaimSpec("frame", role=HypothesisRole.ALTERNATIVE)),
        ),
        View("run-a", "frame-0121", (ClaimSpec("cabinet", confidence=0.4),)),
    ]
    scenario = build_scenario(views)
    expected = _accumulate(scenario)

    for seed in range(5):
        rng = random.Random(seed)
        observations = list(scenario.observations.items())
        results = list(scenario.results.items())
        rng.shuffle(observations)
        rng.shuffle(results)
        shuffled = dataclasses.replace(
            scenario, observations=dict(observations), results=dict(results)
        )
        assert _accumulate(shuffled) == expected


def test_provenance_names_the_policy_and_the_grouping() -> None:
    scenario = build_scenario([View("run-a", "frame-0120", (PALLET,))])
    fused = _accumulate(scenario)

    assert fused.provenance.fusion_policy_id == BASELINE_ACCUMULATION_POLICY_ID
    assert fused.provenance.grouping_policy_id == "physical-observation-grouping-v1"
    assert fused.fusion_support_id == scenario.support.fusion_support_id


def test_the_temporal_summary_and_groups_follow_the_grouping() -> None:
    scenario = build_scenario(
        [View("run-a", "frame-0120", (PALLET,)), View("run-b", "frame-0122", (PALLET,))]
    )

    fused = _accumulate(scenario)

    assert [g.physical_observation_id for g in fused.physical_observation_groups] == [
        "frame-0120",
        "frame-0122",
    ]
    assert (
        fused.temporal_summary.start == fused.physical_observation_groups[0].acquisition_timestamp
    )
    assert fused.temporal_summary.end == fused.physical_observation_groups[1].acquisition_timestamp


def test_an_observation_missing_from_the_inputs_is_rejected() -> None:
    scenario = build_scenario(
        [View("run-a", "frame-0120", (PALLET,)), View("run-a", "frame-0121", (PALLET,))]
    )
    partial = dict(scenario.observations)
    partial.pop(spatial_id("run-a", "frame-0121"))

    with pytest.raises(ValueError, match="no spatial observation"):
        _accumulate(dataclasses.replace(scenario, observations=partial))


def test_a_missing_perception_result_is_rejected() -> None:
    scenario = build_scenario([View("run-a", "frame-0120", (PALLET,))])

    with pytest.raises(ValueError, match="no perception result"):
        _accumulate(dataclasses.replace(scenario, results={}))


def test_a_claim_the_result_does_not_contain_is_rejected() -> None:
    scenario = build_scenario([View("run-a", "frame-0120", (PALLET,))])
    key = next(iter(scenario.results))
    emptied = dataclasses.replace(scenario.results[key], claims=())

    with pytest.raises(ValueError, match="does not contain"):
        _accumulate(dataclasses.replace(scenario, results={key: emptied}))


def test_a_grouping_that_does_not_cover_the_support_is_rejected() -> None:
    scenario = build_scenario(
        [View("run-a", "frame-0120", (PALLET,)), View("run-a", "frame-0121", (PALLET,))]
    )
    trimmed = dataclasses.replace(scenario.grouping, groups=scenario.grouping.groups[:1])

    with pytest.raises(ValueError, match="grouping does not cover"):
        _accumulate(dataclasses.replace(scenario, grouping=trimmed))


def test_an_observation_the_grouping_did_not_place_in_its_frame_is_rejected() -> None:
    scenario = build_scenario(
        [View("run-a", "frame-0120", (PALLET,)), View("run-b", "frame-0120", (PALLET,))]
    )
    group = scenario.grouping.groups[0]
    stripped = dataclasses.replace(group, spatial_observation_ids=group.spatial_observation_ids[:1])
    grouping = dataclasses.replace(scenario.grouping, groups=(stripped,))

    with pytest.raises(ValueError, match="does not list spatial observation"):
        _accumulate(dataclasses.replace(scenario, grouping=grouping))


def test_a_score_for_a_claim_of_another_region_is_ignored() -> None:
    view = View("run-a", "frame-0120", (ClaimSpec("pallet"),))
    scenario = build_scenario([view])
    key = scenario.grouping.groups[0].perception_result_ids[0]
    stray = SemanticSupport(
        claim_id=ClaimId("claim-of-another-region"), support_score=0.5, provenance=make_scorer()
    )

    fused = _accumulate(scenario, scores={key: [stray]})

    assert fused.contributions[0].score_refs == ()


def test_claims_and_features_are_put_in_canonical_order_whatever_the_upstream_order() -> None:
    view = View("run-a", "frame-0120", (ClaimSpec("door"), ClaimSpec("cabinet")))
    scenario = build_scenario([view])
    expected = _accumulate(scenario)
    key, observation = next(iter(scenario.observations.items()))
    unsorted = dataclasses.replace(
        observation, semantic_claim_refs=tuple(reversed(observation.semantic_claim_refs))
    )

    fused = _accumulate(dataclasses.replace(scenario, observations={key: unsorted}))

    assert fused == expected
