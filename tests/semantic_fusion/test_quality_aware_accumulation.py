import dataclasses
import math
import random
import typing
from collections.abc import Mapping

import pytest
from evidence_builders import ClaimSpec, Scenario, View, build_scenario, make_score
from fusion_builders import make_point_representation_ref, spatial_id
from quality_builders import make_quality

from contextmap.semantic_fusion import (
    QUALITY_AWARE_ACCUMULATION_POLICY_ID,
    BaselineAccumulationPolicy,
    ComponentTreatment,
    EvidenceChannel,
    FusedEvidence,
    QualityAwareAccumulationPolicy,
    QualityInput,
    QualityRamp,
    UncertaintyKind,
    accumulate_baseline_evidence,
    accumulate_quality_aware_evidence,
)
from contextmap.sensor_association import ObservationQuality, SpatialObservationId

DEPTH = QualityRamp(quality_input=QualityInput.SUPPORT_DEPTH_MEDIAN_M, good=5.0, bad=25.0)
SHARE = QualityRamp(quality_input=QualityInput.VISIBLE_SHARE, good=0.9, bad=0.3)
POLICY = QualityAwareAccumulationPolicy(
    definitions_version="observation-quality-v1", ramps=(DEPTH,), neutral_factor=0.5
)


def _qualities(
    views: list[View], **per_frame: dict[str, typing.Any]
) -> dict[SpatialObservationId, ObservationQuality]:
    """Quality for every view; ``per_frame`` maps ``frame-0120`` style names to overrides."""
    result = {}
    for view in views:
        observation_id = spatial_id(view.run, view.frame, view.region)
        overrides = per_frame.get(view.frame.replace("-", "_"), {})
        result[observation_id] = make_quality(observation_id, frame=view.frame, **overrides)
    return result


def _run(
    scenario: Scenario,
    qualities: Mapping[SpatialObservationId, ObservationQuality],
    policy: QualityAwareAccumulationPolicy = POLICY,
) -> FusedEvidence:
    return accumulate_quality_aware_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        observation_qualities=qualities,
        policy=policy,
    )


def _baseline(scenario: Scenario) -> FusedEvidence:
    return accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
    )


def _door(*frames: str, run: str = "run-a", **kwargs: object) -> list[View]:
    return [View(run, frame, (ClaimSpec("door", confidence=0.8),), **kwargs) for frame in frames]  # type: ignore[arg-type]


def test_a_distant_view_weighs_less_and_the_counts_stay_apart() -> None:
    views = _door("frame-0120", "frame-0121")
    scenario = build_scenario(views)
    qualities = _qualities(
        views, frame_0120={"depth_median_m": 3.0}, frame_0121={"depth_median_m": 15.0}
    )

    fused = _run(scenario, qualities)

    assert fused.weighting is not None
    (support,) = fused.weighting.hypotheses
    assert support.supporting_physical_observations == 2
    assert [(o.physical_observation_id, o.factor) for o in support.observation_factors] == [
        ("frame-0120", 1.0),
        ("frame-0121", 0.5),
    ]
    assert support.weighted_support == pytest.approx(1.5)


def test_the_same_inputs_run_through_the_baseline_and_the_quality_aware_policy() -> None:
    views = _door("frame-0120", "frame-0121")
    scenario = build_scenario(views)

    baseline = _baseline(scenario)
    weighted = _run(scenario, _qualities(views))

    assert baseline.weighting is None
    assert weighted.hypotheses == baseline.hypotheses
    assert weighted.uncertainty == baseline.uncertainty
    assert weighted.physical_observation_groups == baseline.physical_observation_groups
    assert baseline.provenance.fusion_policy_id == "baseline-evidence-accumulation-v1"
    assert weighted.provenance.fusion_policy_id == QUALITY_AWARE_ACCUMULATION_POLICY_ID


def test_a_low_weight_never_discards_the_evidence() -> None:
    views = _door("frame-0120")
    scenario = build_scenario(views)

    fused = _run(scenario, _qualities(views, frame_0120={"depth_median_m": 40.0}))

    assert fused.weighting is not None
    (support,) = fused.weighting.hypotheses
    assert support.weighted_support == 0.0
    assert support.supporting_physical_observations == 1
    assert len(fused.hypotheses[0].evidence) == 1
    assert len(fused.contributions) == 1


def test_repeated_inference_over_one_frame_is_one_weighted_observation() -> None:
    views = [
        *_door("frame-0120", run="run-a"),
        *_door("frame-0120", run="run-b"),
        *_door("frame-0120", run="run-c"),
    ]
    scenario = build_scenario(views)
    qualities = {
        spatial_id("run-a", "frame-0120"): make_quality(
            spatial_id("run-a", "frame-0120"), depth_median_m=3.0
        ),
        spatial_id("run-b", "frame-0120"): make_quality(
            spatial_id("run-b", "frame-0120"), depth_median_m=15.0
        ),
        spatial_id("run-c", "frame-0120"): make_quality(
            spatial_id("run-c", "frame-0120"), depth_median_m=15.0
        ),
    }

    fused = _run(scenario, qualities)

    assert fused.weighting is not None
    assert len(fused.weighting.contributions) == 3
    (support,) = fused.weighting.hypotheses
    assert support.supporting_physical_observations == 1
    assert support.weighted_support == pytest.approx(2 / 3)
    assert fused.physical_observation_count == 1


def test_geometry_point_count_does_not_multiply_the_weight() -> None:
    small = _door("frame-0120", geometry=range(20))
    large = _door("frame-0120", geometry=range(500))

    weights = []
    for views in (small, large):
        fused = _run(build_scenario(views), _qualities(views))
        assert fused.weighting is not None
        weights.append(fused.weighting.hypotheses[0].weighted_support)

    assert weights[0] == weights[1]


def test_a_missing_component_uses_the_recorded_neutral_factor_and_never_zero() -> None:
    views = _door("frame-0120")
    scenario = build_scenario(views)

    fused = _run(scenario, _qualities(views, frame_0120={"depth_median_m": None}))

    assert fused.weighting is not None
    (weight,) = fused.weighting.contributions
    (component,) = weight.components
    assert component.treatment is ComponentTreatment.NEUTRAL_FALLBACK
    assert component.measured_value is None
    assert component.factor == 0.5
    assert component.unavailable_reason == "fixture: not measurable"
    assert weight.factor == 0.5


def test_a_view_without_quality_is_neutral_and_says_why() -> None:
    views = _door("frame-0120", "frame-0121")
    scenario = build_scenario(views)
    qualities = _qualities(views)
    qualities.pop(spatial_id("run-a", "frame-0121"))

    fused = _run(scenario, qualities)

    assert fused.weighting is not None
    without = fused.weighting.contributions[1]
    assert without.components[0].treatment is ComponentTreatment.NEUTRAL_FALLBACK
    assert "no observation quality" in (without.components[0].unavailable_reason or "")
    assert without.factor == 0.5
    assert fused.contributions[1].observation_quality is None
    assert fused.contributions[0].observation_quality is not None


def test_the_combined_factor_is_the_weakest_declared_component() -> None:
    views = _door("frame-0120")
    scenario = build_scenario(views)
    policy = dataclasses.replace(POLICY, ramps=(DEPTH, SHARE))

    fused = _run(
        scenario,
        _qualities(views, frame_0120={"depth_median_m": 5.0, "visible_share": 0.6}),
        policy,
    )

    assert fused.weighting is not None
    (weight,) = fused.weighting.contributions
    assert [(c.component, c.measured_value, c.factor) for c in weight.components] == [
        ("support_depth_median_m", 5.0, 1.0),
        ("visible_share", 0.6, pytest.approx(0.5)),
    ]
    assert weight.factor == pytest.approx(0.5)


def test_claim_confidence_and_scores_never_enter_the_weight() -> None:
    def factors(confidence: float | None, score: float) -> list[float]:
        views = [View("run-a", "frame-0120", (ClaimSpec("door", confidence=confidence),))]
        scenario = build_scenario(views)
        scores = {next(iter(scenario.results)): [make_score(views[0], 0, score)]}
        fused = accumulate_quality_aware_evidence(
            scenario.support,
            observations=scenario.observations,
            grouping=scenario.grouping,
            perception_results=scenario.results,
            observation_qualities=_qualities(views),
            semantic_scores=scores,
            policy=dataclasses.replace(
                POLICY,
                baseline=BaselineAccumulationPolicy(
                    channels=frozenset(
                        {EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.SEMANTIC_SCORES}
                    )
                ),
            ),
        )
        assert fused.weighting is not None
        return [w.factor for w in fused.weighting.contributions]

    assert factors(0.99, 0.9) == factors(None, 0.01) == factors(0.0, 0.5)


def test_the_weights_are_reproducible_and_do_not_depend_on_order() -> None:
    views = [
        *_door("frame-0120", "frame-0121", run="run-a"),
        *_door("frame-0120", run="run-b"),
    ]
    scenario = build_scenario(views)
    qualities = _qualities(views, frame_0121={"depth_median_m": 12.0})
    expected = _run(scenario, qualities)

    for seed in range(5):
        rng = random.Random(seed)
        observations = list(scenario.observations.items())
        shuffled_qualities = list(qualities.items())
        rng.shuffle(observations)
        rng.shuffle(shuffled_qualities)
        shuffled = dataclasses.replace(scenario, observations=dict(observations))
        assert _run(shuffled, dict(shuffled_qualities)) == expected
    assert _run(scenario, qualities) == expected


def test_an_unsupported_definitions_version_fails_clearly() -> None:
    views = _door("frame-0120")
    scenario = build_scenario(views)
    qualities = {
        spatial_id("run-a", "frame-0120"): make_quality(
            spatial_id("run-a", "frame-0120"), definitions_version="observation-quality-v2"
        )
    }

    with pytest.raises(ValueError, match="observation-quality-v2"):
        _run(scenario, qualities)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ramps": ()}, "at least one ramp"),
        ({"ramps": (DEPTH, DEPTH)}, "declared more than once"),
        ({"neutral_factor": 1.5}, "neutral_factor"),
        ({"neutral_factor": math.nan}, "neutral_factor"),
        ({"definitions_version": " "}, "definitions_version"),
    ],
)
def test_an_impossible_policy_is_rejected(kwargs: dict[str, typing.Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(POLICY, **kwargs)


@pytest.mark.parametrize(
    "ramp",
    [
        {"good": 5.0, "bad": 5.0},
        {"good": math.inf, "bad": 5.0},
        {"good": 5.0, "bad": math.nan},
    ],
)
def test_an_impossible_ramp_is_rejected(ramp: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="ramp"):
        QualityRamp(quality_input=QualityInput.SUPPORT_DEPTH_MEDIAN_M, **ramp)


def test_the_provenance_names_the_policy_the_channel_and_the_configuration() -> None:
    views = _door("frame-0120")
    scenario = build_scenario(views)
    other = dataclasses.replace(POLICY, ramps=(DEPTH, SHARE))

    fused = _run(scenario, _qualities(views))

    assert fused.provenance.configuration_fingerprint == POLICY.fingerprint()
    assert other.fingerprint() != POLICY.fingerprint()
    assert dataclasses.replace(POLICY, neutral_factor=1.0).fingerprint() != POLICY.fingerprint()
    channel = next(c for c in fused.channels if c.channel is EvidenceChannel.OBSERVATION_QUALITY)
    assert channel.identities == ("observation-quality-v1",)


def test_the_weighting_records_the_formula_that_produced_it() -> None:
    views = _door("frame-0120")

    fused = _run(build_scenario(views), _qualities(views))

    assert fused.weighting is not None
    assert fused.weighting.policy_id == QUALITY_AWARE_ACCUMULATION_POLICY_ID
    assert fused.weighting.combination_rule == "minimum-of-component-factors"
    assert fused.weighting.observation_rule == "mean-of-supporting-contribution-factors"
    assert fused.weighting.definitions_version == "observation-quality-v1"


def test_static_structure_is_listed_once_and_never_weighted() -> None:
    views = _door("frame-0120", "frame-0121")
    scenario = build_scenario(views)
    reference = make_point_representation_ref(geometry_index=3)
    policy = dataclasses.replace(
        POLICY,
        baseline=BaselineAccumulationPolicy(
            channels=frozenset(
                {EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.POINT_REPRESENTATION}
            )
        ),
    )

    fused = accumulate_quality_aware_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        observation_qualities=_qualities(views),
        point_representation_refs=[reference],
        policy=policy,
    )

    assert fused.point_representation_refs == (reference,)
    assert fused.weighting is not None
    names = {f.name for f in dataclasses.fields(fused.weighting)}
    assert not {name for name in names if "representation" in name or "structure" in name}


def test_uncertainty_and_abstention_follow_the_baseline_settings() -> None:
    views = [
        View("run-a", "frame-0120", (ClaimSpec("door"),)),
        View("run-a", "frame-0121", (ClaimSpec("cabinet"),)),
        View("run-a", "frame-0122", (ClaimSpec("unknown"),)),
    ]
    scenario = build_scenario(views)
    baseline_policy = BaselineAccumulationPolicy(abstention_labels=frozenset({"unknown"}))
    policy = dataclasses.replace(POLICY, baseline=baseline_policy)

    fused = _run(scenario, _qualities(views), policy)

    assert {r.kind for r in fused.uncertainty} == {
        UncertaintyKind.CONTRADICTION,
        UncertaintyKind.NEAR_TIE,
    }
    assert [h.label for h in fused.hypotheses] == ["cabinet", "door"]
    assert fused.weighting is not None
    assert [s.supporting_physical_observations for s in fused.weighting.hypotheses] == [1, 1]


def test_a_view_that_produced_no_claim_still_gets_a_weight() -> None:
    views = [View("run-a", "frame-0120", ())]

    fused = _run(build_scenario(views), _qualities(views))

    assert fused.weighting is not None
    assert len(fused.weighting.contributions) == 1
    assert fused.weighting.hypotheses == ()
