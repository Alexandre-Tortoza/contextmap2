import dataclasses
import math

import pytest
from fusion_builders import (
    contribution_id_for,
    make_contribution,
    make_fused_evidence,
)

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    ChannelProvenance,
    ComponentFactor,
    ComponentTreatment,
    ContributionWeight,
    EvidenceChannel,
    FusedHypothesisId,
    HypothesisSupport,
    ObservationFactor,
    QualityWeighting,
)

FRAME = SourceObservationId("frame-0120")


def _component(**overrides: object) -> ComponentFactor:
    values: dict[str, object] = {
        "component": "support_depth_median_m",
        "treatment": ComponentTreatment.MEASURED,
        "measured_value": 3.0,
        "factor": 1.0,
        "unavailable_reason": None,
    }
    values.update(overrides)
    return ComponentFactor(**values)  # type: ignore[arg-type]


def _weight(factor: float = 1.0) -> ContributionWeight:
    return ContributionWeight(
        contribution_id=contribution_id_for("run-a", "frame-0120"),
        components=(_component(factor=factor),),
        factor=factor,
    )


def _support(**overrides: object) -> HypothesisSupport:
    values: dict[str, object] = {
        "hypothesis_id": FusedHypothesisId("hypothesis-0001"),
        "supporting_physical_observations": 1,
        "observation_factors": (ObservationFactor(physical_observation_id=FRAME, factor=1.0),),
        "weighted_support": 1.0,
    }
    values.update(overrides)
    return HypothesisSupport(**values)  # type: ignore[arg-type]


def _weighting(**overrides: object) -> QualityWeighting:
    values: dict[str, object] = {
        "policy_id": "quality-aware-evidence-accumulation-v1",
        "combination_rule": "minimum-of-component-factors",
        "observation_rule": "mean-of-supporting-contribution-factors",
        "definitions_version": "observation-quality-v1",
        "contributions": (_weight(),),
        "hypotheses": (_support(),),
    }
    values.update(overrides)
    return QualityWeighting(**values)  # type: ignore[arg-type]


def test_a_coherent_weighting_is_accepted_beside_the_evidence_it_weighs() -> None:
    fused = make_fused_evidence()

    weighted = dataclasses.replace(fused, weighting=_weighting())

    assert weighted.weighting is not None
    assert weighted.weighting.hypotheses[0].weighted_support == 1.0


@pytest.mark.parametrize("factor", [-0.1, 1.1, math.nan, math.inf])
def test_every_factor_is_a_finite_number_in_the_unit_interval(factor: float) -> None:
    with pytest.raises(ValueError, match="factor"):
        _component(factor=factor)
    with pytest.raises(ValueError, match="factor"):
        ContributionWeight(
            contribution_id=contribution_id_for("run-a", "frame-0120"),
            components=(_component(),),
            factor=factor,
        )
    with pytest.raises(ValueError, match="factor"):
        ObservationFactor(physical_observation_id=FRAME, factor=factor)


def test_a_measured_component_carries_its_value_and_no_reason() -> None:
    with pytest.raises(ValueError, match="measured_value"):
        _component(measured_value=None)
    with pytest.raises(ValueError, match="unavailable_reason"):
        _component(unavailable_reason="because")


def test_a_neutral_fallback_carries_a_reason_and_no_value() -> None:
    fallback = {"treatment": ComponentTreatment.NEUTRAL_FALLBACK, "factor": 0.5}

    assert _component(measured_value=None, unavailable_reason="no quality", **fallback)
    with pytest.raises(ValueError, match="measured_value"):
        _component(measured_value=3.0, unavailable_reason="no quality", **fallback)
    with pytest.raises(ValueError, match="unavailable_reason"):
        _component(measured_value=None, unavailable_reason=None, **fallback)


def test_components_are_sorted_unique_and_present() -> None:
    first = _component(component="a_component")
    second = _component(component="b_component")

    with pytest.raises(ValueError, match="components must be sorted and unique"):
        ContributionWeight(
            contribution_id=contribution_id_for("run-a", "frame-0120"),
            components=(second, first),
            factor=1.0,
        )
    with pytest.raises(ValueError, match="at least one component"):
        ContributionWeight(
            contribution_id=contribution_id_for("run-a", "frame-0120"), components=(), factor=1.0
        )


def test_the_weighted_support_is_the_sum_of_the_observation_factors() -> None:
    factors = (
        ObservationFactor(physical_observation_id=SourceObservationId("frame-0120"), factor=1.0),
        ObservationFactor(physical_observation_id=SourceObservationId("frame-0121"), factor=0.5),
    )

    assert _support(
        supporting_physical_observations=2, observation_factors=factors, weighted_support=1.5
    )
    with pytest.raises(ValueError, match="weighted_support"):
        _support(
            supporting_physical_observations=2, observation_factors=factors, weighted_support=2.0
        )
    with pytest.raises(ValueError, match="supporting_physical_observations"):
        _support(
            supporting_physical_observations=3, observation_factors=factors, weighted_support=1.5
        )
    with pytest.raises(ValueError, match="observation_factors must be sorted and unique"):
        _support(
            supporting_physical_observations=2,
            observation_factors=tuple(reversed(factors)),
            weighted_support=1.5,
        )


def test_a_weighting_needs_the_quality_channel_to_be_declared() -> None:
    plain = make_contribution(with_quality=False)
    base = (
        ChannelProvenance(channel=EvidenceChannel.GEOMETRY_SUPPORT),
        ChannelProvenance(channel=EvidenceChannel.SEMANTIC_CLAIMS),
    )
    without_quality = make_fused_evidence(contributions=(plain,), channels=base)

    with pytest.raises(ValueError, match="observation_quality"):
        dataclasses.replace(without_quality, weighting=_weighting())


def test_the_weights_must_cover_exactly_the_contributions() -> None:
    fused = make_fused_evidence()

    with pytest.raises(ValueError, match="contributions"):
        dataclasses.replace(fused, weighting=_weighting(contributions=()))
    stray = dataclasses.replace(
        _weight(), contribution_id=contribution_id_for("run-z", "frame-0120")
    )
    with pytest.raises(ValueError, match="contribution"):
        dataclasses.replace(fused, weighting=_weighting(contributions=(_weight(), stray)))


def test_the_supports_must_cover_exactly_the_hypotheses_and_their_observations() -> None:
    fused = make_fused_evidence()

    with pytest.raises(ValueError, match="hypotheses"):
        dataclasses.replace(fused, weighting=_weighting(hypotheses=()))
    wrong_count = _support(
        supporting_physical_observations=2,
        observation_factors=(
            ObservationFactor(physical_observation_id=FRAME, factor=1.0),
            ObservationFactor(
                physical_observation_id=SourceObservationId("frame-0121"), factor=1.0
            ),
        ),
        weighted_support=2.0,
    )
    with pytest.raises(ValueError, match="supporting physical observations"):
        dataclasses.replace(fused, weighting=_weighting(hypotheses=(wrong_count,)))


def test_a_weighting_never_calls_itself_a_confidence_or_a_probability() -> None:
    names = {
        field.name
        for cls in (QualityWeighting, ContributionWeight, HypothesisSupport, ObservationFactor)
        for field in dataclasses.fields(cls)
    }

    assert not {"confidence", "probability", "score", "similarity", "label"} & names


def test_a_weighting_names_its_rules_and_definitions() -> None:
    for field in ("policy_id", "combination_rule", "observation_rule", "definitions_version"):
        with pytest.raises(ValueError, match=field):
            _weighting(**{field: " "})
