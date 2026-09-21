import dataclasses

import pytest
from evidence_builders import ClaimSpec, Scenario, View, build_scenario, make_score
from fusion_builders import (
    evidence,
    make_contribution,
    make_fused_evidence,
    make_hypothesis,
    make_point_representation_ref,
    make_region_feature,
    make_scorer,
    scorer_signal,
)

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    ChannelProvenance,
    EvidenceChannel,
    FusedEvidence,
    ObservationQualityRef,
    ScoreReference,
    SupportSignalKind,
    accumulate_baseline_evidence,
)
from contextmap.visual_perception import ClaimId

CLAIMS = EvidenceChannel.SEMANTIC_CLAIMS
SCORES = EvidenceChannel.SEMANTIC_SCORES
FEATURES = EvidenceChannel.VISUAL_FEATURES
QUALITY = EvidenceChannel.OBSERVATION_QUALITY
GEOMETRY = EvidenceChannel.GEOMETRY_SUPPORT
STRUCTURE = EvidenceChannel.POINT_REPRESENTATION

OPTIONAL = (SCORES, FEATURES, QUALITY, STRUCTURE)


def _views() -> list[View]:
    features = (make_region_feature(feature="feature-0001"),)
    return [
        View("run-a", "frame-0120", (ClaimSpec("pallet", confidence=0.7),), features=features),
        View("run-a", "frame-0121", (ClaimSpec("pallet", confidence=0.5),), features=features),
    ]


def _run(
    scenario: Scenario, channels: frozenset[EvidenceChannel], *, views: list[View] | None = None
) -> FusedEvidence:
    """Offer every kind of evidence, so that only the declared channels decide what is used."""
    used = _views() if views is None else views
    scores = {
        scenario.grouping.groups[index].perception_result_ids[0]: [make_score(view, 0, 0.4)]
        for index, view in enumerate(used)
    }
    qualities = {
        observation_id: ObservationQualityRef(
            spatial_observation_id=observation_id, definitions_version="observation-quality-v1"
        )
        for observation_id in scenario.observations
    }
    return accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        semantic_scores=scores,
        observation_quality_refs=qualities,
        point_representation_refs=[make_point_representation_ref(geometry_index=3)],
        policy=BaselineAccumulationPolicy(channels=channels),
    )


def _data_of(fused: FusedEvidence) -> dict[EvidenceChannel, bool]:
    signals = [s for h in fused.hypotheses for item in h.evidence for s in item.signals]
    return {
        SCORES: any(c.score_refs for c in fused.contributions)
        or any(s.kind is SupportSignalKind.SCORER_SUPPORT for s in signals),
        FEATURES: any(c.visual_feature_refs for c in fused.contributions),
        QUALITY: any(c.observation_quality is not None for c in fused.contributions),
        STRUCTURE: bool(fused.point_representation_refs),
    }


def test_semantic_claims_only_is_the_default_and_ignores_every_other_offered_channel() -> None:
    scenario = build_scenario(_views())

    fused = accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        semantic_scores={key: [] for key in scenario.results},
        observation_quality_refs={},
        point_representation_refs=[make_point_representation_ref(geometry_index=3)],
    )

    assert _data_of(fused) == {SCORES: False, FEATURES: False, QUALITY: False, STRUCTURE: False}
    assert [c.channel for c in fused.channels] == [GEOMETRY, CLAIMS]


@pytest.mark.parametrize("channel", OPTIONAL)
def test_a_declared_channel_takes_part_and_the_others_stay_out(channel: EvidenceChannel) -> None:
    scenario = build_scenario(_views())

    fused = _run(scenario, frozenset({CLAIMS, channel}))

    expected = {other: other is channel for other in OPTIONAL}
    assert _data_of(fused) == expected
    assert {c.channel for c in fused.channels} == {CLAIMS, GEOMETRY, channel}


def test_semantic_only_and_richer_runs_share_the_same_upstream_inputs() -> None:
    scenario = build_scenario(_views())

    plain = _run(scenario, frozenset({CLAIMS}))
    rich = _run(scenario, frozenset({CLAIMS, *OPTIONAL}))

    def shape(fused: FusedEvidence) -> list[tuple[str, list[tuple[str, str, str]]]]:
        return [
            (h.label, [(i.contribution_id, i.claim_id, i.stance.value) for i in h.evidence])
            for h in fused.hypotheses
        ]

    assert shape(plain) == shape(rich)
    assert plain.contributions != rich.contributions
    assert _data_of(rich) == {channel: True for channel in OPTIONAL}


def test_channel_provenance_names_the_identities_that_fed_each_channel() -> None:
    scenario = build_scenario(_views())

    fused = _run(scenario, frozenset({CLAIMS, *OPTIONAL}))

    identities = {c.channel: c.identities for c in fused.channels}
    assert identities[CLAIMS] == ("qwen_vl/qwen3-vl/1",)
    assert identities[SCORES] == ("clip_scorer/clip-vit-l14/1",)
    assert identities[FEATURES] == ("dinov3-vit-b16",)
    assert identities[QUALITY] == ("observation-quality-v1",)
    assert identities[STRUCTURE] == ("sha256:space-geometric-descriptor",)
    assert identities[GEOMETRY] == ("map:map-0001", "support-policy:geometry-jaccard-support-v1")


def test_a_declared_channel_that_finds_nothing_is_active_with_no_identities() -> None:
    scenario = build_scenario([View("run-a", "frame-0120", (ClaimSpec("pallet"),))])

    fused = accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        semantic_scores={},
        policy=BaselineAccumulationPolicy(channels=frozenset({CLAIMS, SCORES})),
    )

    scores = next(c for c in fused.channels if c.channel is SCORES)
    assert scores.identities == ()


@pytest.mark.parametrize(
    ("channel", "message"),
    [
        (SCORES, "semantic_scores"),
        (QUALITY, "observation_quality"),
        (STRUCTURE, "point_representation"),
    ],
)
def test_a_declared_channel_without_any_input_fails_clearly(
    channel: EvidenceChannel, message: str
) -> None:
    scenario = build_scenario([View("run-a", "frame-0120", (ClaimSpec("pallet"),))])

    with pytest.raises(ValueError, match=f"channel {message} is declared but"):
        accumulate_baseline_evidence(
            scenario.support,
            observations=scenario.observations,
            grouping=scenario.grouping,
            perception_results=scenario.results,
            policy=BaselineAccumulationPolicy(channels=frozenset({CLAIMS, channel})),
        )


def test_semantic_claims_are_required() -> None:
    with pytest.raises(ValueError, match="semantic_claims"):
        BaselineAccumulationPolicy(channels=frozenset({SCORES}))
    with pytest.raises(ValueError, match="semantic_claims"):
        BaselineAccumulationPolicy(channels=frozenset())


def test_geometry_support_is_intrinsic_and_declaring_it_changes_nothing() -> None:
    explicit = BaselineAccumulationPolicy(channels=frozenset({CLAIMS, GEOMETRY}))
    implicit = BaselineAccumulationPolicy(channels=frozenset({CLAIMS}))
    scenario = build_scenario(_views())

    assert explicit.fingerprint() == implicit.fingerprint()
    assert _run(scenario, explicit.channels) == _run(scenario, implicit.channels)


def test_the_fingerprint_follows_the_declared_channels() -> None:
    semantic = BaselineAccumulationPolicy()
    scored = BaselineAccumulationPolicy(channels=frozenset({CLAIMS, SCORES}))

    assert semantic.fingerprint() != scored.fingerprint()
    assert (
        scored.fingerprint()
        == BaselineAccumulationPolicy(channels=frozenset({SCORES, CLAIMS})).fingerprint()
    )


def test_the_provenance_fingerprint_records_the_channel_configuration() -> None:
    scenario = build_scenario(_views())
    policy = BaselineAccumulationPolicy(channels=frozenset({CLAIMS, SCORES}))

    fused = _run(scenario, policy.channels)

    assert fused.provenance.configuration_fingerprint == policy.fingerprint()


def test_observation_quality_never_changes_the_hypotheses_or_the_uncertainty() -> None:
    scenario = build_scenario(_views())

    plain = _run(scenario, frozenset({CLAIMS}))
    with_quality = _run(scenario, frozenset({CLAIMS, QUALITY}))

    assert with_quality.hypotheses == plain.hypotheses
    assert with_quality.uncertainty == plain.uncertainty


def test_a_fused_evidence_cannot_carry_data_of_a_channel_it_did_not_declare() -> None:
    reference = ScoreReference(claim_id=ClaimId("claim-0001"), scorer=make_scorer())
    scored = make_contribution(with_quality=False, score_refs=(reference,))
    featured = make_contribution(with_quality=False, feature_refs=(make_region_feature(),))
    plain = make_contribution(with_quality=False)
    signalled = make_hypothesis(items=(evidence(signals=(scorer_signal(0.3),)),))
    base = (ChannelProvenance(channel=GEOMETRY), ChannelProvenance(channel=CLAIMS))
    with_scores = (*base, ChannelProvenance(channel=SCORES))

    with pytest.raises(ValueError, match="semantic_scores"):
        make_fused_evidence(contributions=(scored,), channels=base)
    with pytest.raises(ValueError, match="semantic_scores"):
        make_fused_evidence(contributions=(scored,), hypotheses=(signalled,), channels=base)
    with pytest.raises(ValueError, match="visual_features"):
        make_fused_evidence(contributions=(featured,), channels=with_scores)
    with pytest.raises(ValueError, match="observation_quality"):
        make_fused_evidence(contributions=(make_contribution(),), channels=with_scores)
    with pytest.raises(ValueError, match="point_representation"):
        make_fused_evidence(
            contributions=(plain,),
            point_representation_refs=(make_point_representation_ref(),),
            channels=with_scores,
        )


def test_a_fused_evidence_must_declare_claims_and_geometry() -> None:
    plain = make_contribution(with_quality=False)

    with pytest.raises(ValueError, match="semantic_claims"):
        make_fused_evidence(contributions=(plain,), channels=(ChannelProvenance(channel=GEOMETRY),))
    with pytest.raises(ValueError, match="geometry_support"):
        make_fused_evidence(contributions=(plain,), channels=(ChannelProvenance(channel=CLAIMS),))


def test_channels_must_be_sorted_and_unique() -> None:
    claims = ChannelProvenance(channel=CLAIMS)
    geometry = ChannelProvenance(channel=GEOMETRY)
    plain = make_contribution(with_quality=False)

    with pytest.raises(ValueError, match="channels must be sorted and unique"):
        make_fused_evidence(contributions=(plain,), channels=(claims, geometry))
    with pytest.raises(ValueError, match="channels must be sorted and unique"):
        make_fused_evidence(contributions=(plain,), channels=(geometry, geometry, claims))


def test_channel_identities_are_sorted_unique_and_named() -> None:
    with pytest.raises(ValueError, match="identities must be sorted and unique"):
        ChannelProvenance(channel=CLAIMS, identities=("b", "a"))
    with pytest.raises(ValueError, match="identity must not be empty"):
        ChannelProvenance(channel=CLAIMS, identities=(" ",))
    assert ChannelProvenance(channel=CLAIMS).identities == ()


def test_the_channel_names_are_the_ones_the_issue_lists() -> None:
    assert {channel.value for channel in EvidenceChannel} == {
        "semantic_claims",
        "semantic_scores",
        "visual_features",
        "observation_quality",
        "geometry_support",
        "point_representation",
    }


def test_provenance_replace_keeps_the_contract_for_the_default_builder() -> None:
    fused = make_fused_evidence()

    assert dataclasses.replace(fused, channels=fused.channels) == fused
