import dataclasses
import random
from collections.abc import Mapping

import pytest
from evidence_builders import ClaimSpec, Scenario, View, build_scenario, reference_to
from fusion_builders import evidence, make_hypothesis

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceStance,
    FusedEvidence,
    ObservationQualityRef,
    UncertaintyKind,
    UncertaintyRecord,
    accumulate_baseline_evidence,
)
from contextmap.sensor_association import SpatialObservationId
from contextmap.visual_perception import HypothesisRole

ABSTAINS = BaselineAccumulationPolicy(abstention_labels=frozenset({"unknown"}))


def _accumulate(
    scenario: Scenario,
    policy: BaselineAccumulationPolicy | None = None,
    *,
    qualities: Mapping[SpatialObservationId, ObservationQualityRef] | None = None,
) -> FusedEvidence:
    return accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        observation_quality_refs=qualities,
        policy=policy,
    )


def _records(fused: FusedEvidence, kind: UncertaintyKind) -> list[UncertaintyRecord]:
    return [record for record in fused.uncertainty if record.kind is kind]


def _labels(fused: FusedEvidence) -> list[str]:
    return [hypothesis.label for hypothesis in fused.hypotheses]


def test_contradictory_frames_report_the_exact_claims_that_disagree() -> None:
    door = View("run-a", "frame-0120", (ClaimSpec("door"),))
    cabinet = View("run-a", "frame-0121", (ClaimSpec("cabinet"),))
    scenario = build_scenario([door, cabinet])

    fused = _accumulate(scenario)

    (contradiction,) = _records(fused, UncertaintyKind.CONTRADICTION)
    assert [h for h in contradiction.hypothesis_ids] == [h.hypothesis_id for h in fused.hypotheses]
    assert contradiction.evidence == (
        reference_to(scenario, door),
        reference_to(scenario, cabinet),
    )
    assert contradiction.rule_id.startswith("baseline-evidence-accumulation-v1")


def test_a_majority_does_not_hide_the_contradiction() -> None:
    views = [
        View("run-a", "frame-0120", (ClaimSpec("door"),)),
        View("run-a", "frame-0121", (ClaimSpec("door"),)),
        View("run-a", "frame-0122", (ClaimSpec("cabinet"),)),
    ]
    scenario = build_scenario(views)

    fused = _accumulate(scenario)

    (contradiction,) = _records(fused, UncertaintyKind.CONTRADICTION)
    assert contradiction.evidence == tuple(reference_to(scenario, view) for view in views)
    assert _records(fused, UncertaintyKind.NEAR_TIE) == []


def test_a_margin_makes_a_narrow_lead_a_near_tie() -> None:
    views = [
        View("run-a", "frame-0120", (ClaimSpec("door"),)),
        View("run-a", "frame-0121", (ClaimSpec("door"),)),
        View("run-a", "frame-0122", (ClaimSpec("cabinet"),)),
    ]
    scenario = build_scenario(views)

    fused = _accumulate(scenario, BaselineAccumulationPolicy(near_tie_margin=1))

    (tie,) = _records(fused, UncertaintyKind.NEAR_TIE)
    assert len(tie.hypothesis_ids) == 2
    assert tie.rule_id.endswith("margin-1")
    assert tie.evidence == tuple(reference_to(scenario, view) for view in views)


def test_an_exact_tie_is_reported_with_the_claims_behind_it() -> None:
    door = View("run-a", "frame-0120", (ClaimSpec("door"),))
    cabinet = View("run-a", "frame-0121", (ClaimSpec("cabinet"),))
    scenario = build_scenario([door, cabinet])

    fused = _accumulate(scenario)

    (tie,) = _records(fused, UncertaintyKind.NEAR_TIE)
    assert tie.evidence == (reference_to(scenario, door), reference_to(scenario, cabinet))
    assert tie.rule_id.endswith("margin-0")


def test_a_tie_never_compares_scores() -> None:
    scored = View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.99),))
    unscored = View("run-a", "frame-0121", (ClaimSpec("cabinet", confidence=None),))

    fused = _accumulate(build_scenario([scored, unscored]))

    assert len(_records(fused, UncertaintyKind.NEAR_TIE)) == 1
    values = {
        item.signals[0].value for hypothesis in fused.hypotheses for item in hypothesis.evidence
    }
    assert values == {0.99, None}


def test_disagreement_between_runs_over_one_frame_is_ambiguity_and_not_contradiction() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-b", "frame-0120", (ClaimSpec("cabinet"),)),
        ]
    )

    fused = _accumulate(scenario)

    assert _records(fused, UncertaintyKind.CONTRADICTION) == []
    (ambiguity,) = _records(fused, UncertaintyKind.AMBIGUITY)
    assert len(ambiguity.hypothesis_ids) == 2
    assert fused.physical_observation_count == 1


def test_alternatives_of_one_interpretation_are_reported_as_ambiguity() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (ClaimSpec("door"), ClaimSpec("cabinet", role=HypothesisRole.ALTERNATIVE)),
    )
    scenario = build_scenario([view])

    fused = _accumulate(scenario)

    (ambiguity,) = _records(fused, UncertaintyKind.AMBIGUITY)
    assert ambiguity.evidence == (reference_to(scenario, view, 0), reference_to(scenario, view, 1))
    assert _records(fused, UncertaintyKind.CONTRADICTION) == []
    assert _records(fused, UncertaintyKind.NEAR_TIE) == []


def test_one_hypothesis_from_agreeing_views_has_no_uncertainty() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-b", "frame-0121", (ClaimSpec("door"),)),
        ]
    )

    assert _accumulate(scenario).uncertainty == ()


def test_abstention_is_not_a_hypothesis_and_not_negative_evidence() -> None:
    pallet_a = View("run-a", "frame-0120", (ClaimSpec("pallet"),))
    unknown = View("run-a", "frame-0121", (ClaimSpec("unknown"),))
    pallet_c = View("run-a", "frame-0122", (ClaimSpec("pallet"),))
    scenario = build_scenario([pallet_a, unknown, pallet_c])

    fused = _accumulate(scenario, ABSTAINS)

    assert _labels(fused) == ["pallet"]
    stances = [item.stance for item in fused.hypotheses[0].evidence]
    assert stances == [
        EvidenceStance.SUPPORTING,
        EvidenceStance.ABSTAINING,
        EvidenceStance.SUPPORTING,
    ]
    assert fused.supporting_physical_observations(fused.hypotheses[0].hypothesis_id) == (
        "frame-0120",
        "frame-0122",
    )
    assert fused.uncertainty == ()


def test_without_configured_abstention_labels_unknown_is_an_ordinary_label() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("pallet"),)),
            View("run-a", "frame-0121", (ClaimSpec("unknown"),)),
        ]
    )

    fused = _accumulate(scenario)

    assert _labels(fused) == ["pallet", "unknown"]


def test_abstention_labels_match_by_the_label_key() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("pallet"),)),
            View("run-a", "frame-0121", (ClaimSpec("  UNKNOWN "),)),
        ]
    )

    fused = _accumulate(scenario, ABSTAINS)

    assert _labels(fused) == ["pallet"]


def test_views_that_only_abstain_or_stay_silent_are_insufficient_evidence() -> None:
    abstaining = View("run-a", "frame-0120", (ClaimSpec("unknown"),))
    silent = View("run-a", "frame-0121", ())
    scenario = build_scenario([abstaining, silent])

    fused = _accumulate(scenario, ABSTAINS)

    assert fused.hypotheses == ()
    (insufficient,) = fused.uncertainty
    assert insufficient.kind is UncertaintyKind.INSUFFICIENT_EVIDENCE
    assert insufficient.hypothesis_ids == ()
    assert insufficient.evidence == (
        reference_to(scenario, abstaining, 0),
        reference_to(scenario, silent, None),
    )


def test_a_support_without_any_claim_is_insufficient_evidence() -> None:
    view = View("run-a", "frame-0120", ())
    scenario = build_scenario([view])

    (insufficient,) = _accumulate(scenario).uncertainty

    assert insufficient.kind is UncertaintyKind.INSUFFICIENT_EVIDENCE
    assert insufficient.evidence == (reference_to(scenario, view, None),)


def test_insufficient_evidence_is_not_reported_when_a_hypothesis_exists() -> None:
    scenario = build_scenario(
        [View("run-a", "frame-0120", (ClaimSpec("pallet"),)), View("run-a", "frame-0121", ())]
    )

    assert _accumulate(scenario).uncertainty == ()


def test_uncertainty_does_not_depend_on_the_order_of_the_evidence() -> None:
    views = [
        View("run-a", "frame-0120", (ClaimSpec("door"),)),
        View("run-b", "frame-0120", (ClaimSpec("cabinet"), ClaimSpec("unknown"))),
        View(
            "run-a",
            "frame-0121",
            (ClaimSpec("door"), ClaimSpec("frame", role=HypothesisRole.ALTERNATIVE)),
        ),
    ]
    scenario = build_scenario(views)
    expected = _accumulate(scenario, ABSTAINS)

    for seed in range(5):
        rng = random.Random(seed)
        observations = list(scenario.observations.items())
        results = list(scenario.results.items())
        rng.shuffle(observations)
        rng.shuffle(results)
        shuffled = dataclasses.replace(
            scenario, observations=dict(observations), results=dict(results)
        )
        assert _accumulate(shuffled, ABSTAINS) == expected
    assert expected.uncertainty


def test_observation_quality_never_changes_the_uncertainty() -> None:
    scenario = build_scenario(
        [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-a", "frame-0121", (ClaimSpec("cabinet"),)),
        ]
    )
    refs = {
        observation_id: ObservationQualityRef(
            spatial_observation_id=observation_id, definitions_version="observation-quality-v1"
        )
        for observation_id in scenario.observations
    }

    assert _accumulate(scenario, qualities=refs).uncertainty == _accumulate(scenario).uncertainty


def test_the_policy_is_recorded_in_the_provenance() -> None:
    scenario = build_scenario([View("run-a", "frame-0120", (ClaimSpec("door"),))])
    other = BaselineAccumulationPolicy(abstention_labels=frozenset({"unknown", "n/a"}))

    fused = _accumulate(scenario, ABSTAINS)

    assert fused.provenance.configuration_fingerprint == ABSTAINS.fingerprint()
    assert other.fingerprint() != ABSTAINS.fingerprint()
    assert BaselineAccumulationPolicy().fingerprint() != ABSTAINS.fingerprint()
    assert BaselineAccumulationPolicy(abstention_labels=frozenset({"Unknown"})).fingerprint() == (
        ABSTAINS.fingerprint()
    )


def test_an_impossible_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="near_tie_margin"):
        BaselineAccumulationPolicy(near_tie_margin=-1)
    with pytest.raises(ValueError, match="abstention label"):
        BaselineAccumulationPolicy(abstention_labels=frozenset({"  "}))


def test_a_hypothesis_still_needs_a_supporting_claim_when_abstentions_exist() -> None:
    abstaining = evidence(stance=EvidenceStance.ABSTAINING)

    with pytest.raises(ValueError, match="supporting"):
        make_hypothesis(items=(abstaining,))
