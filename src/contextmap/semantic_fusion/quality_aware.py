"""The optional quality-aware multi-view evidence policy.

``quality-aware-evidence-accumulation-v1`` runs the *same* accumulation as the baseline and
adds an inspectable **fusion contribution weight** derived from measurable observation
quality. The baseline stays unchanged and remains the control arm: this policy is chosen
explicitly and is not the default.

The weight is not a semantic confidence, a scorer support or a probability, and nothing here
reads any of them: only the declared quality components. It has a fixed, versioned form:

* every **ramp** maps one measured quality quantity to a factor in ``[0, 1]``: ``1`` at or
  beyond ``good``, ``0`` at or beyond ``bad``, linear in between, so the direction (higher or
  lower is better) is the order of ``good`` and ``bad``;
* the factor of a contribution is the **minimum** of its declared components' factors: the
  weakest declared condition limits it (``minimum-of-component-factors``);
* a component that cannot be measured, or an observation with no quality at all, takes the
  policy's declared **neutral factor** and the reason is recorded; it is never silently zero;
* the factor of a physical observation for a hypothesis is the **mean** of the factors of its
  contributions that support the hypothesis (``mean-of-supporting-contribution-factors``), so
  N inference runs over one frame are one observation whose factor cannot exceed one, and a
  claim over many geometry points is not multiplied;
* the *weighted support* of a hypothesis is the sum of its observation factors, reported next
  to the unweighted count of distinct physical observations, so the effect of the policy is
  visible before and after.

Weak observations are never discarded: hypotheses, contributions and uncertainty are exactly
those of the baseline, and a factor of zero only means the observation adds no weighted
support. Static point representations are listed once on the support and are never weighted.
There is no learned weighting; every rule and number is versioned in the policy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion.accumulation import (
    BaselineAccumulationPolicy,
    accumulate_baseline_evidence,
)
from contextmap.semantic_fusion.grouping import PhysicalObservationGrouping
from contextmap.semantic_fusion.models import (
    ComponentFactor,
    ComponentTreatment,
    ContributionWeight,
    EvidenceChannel,
    EvidenceContributionId,
    EvidenceStance,
    FusedEvidence,
    FusionSupport,
    HypothesisSupport,
    ObservationFactor,
    ObservationQualityRef,
    PointRepresentationRef,
    QualityWeighting,
)
from contextmap.sensor_association import (
    ObservationQuality,
    QualityComponent,
    SpatialObservation,
    SpatialObservationId,
)
from contextmap.visual_perception import PerceptionResult, PerceptionResultId, SemanticSupport

QUALITY_AWARE_ACCUMULATION_POLICY_ID = "quality-aware-evidence-accumulation-v1"
"""Versioned identity of the quality-aware policy, including its combination rules."""

COMBINATION_RULE = "minimum-of-component-factors"
OBSERVATION_RULE = "mean-of-supporting-contribution-factors"


class QualityInput(Enum):
    """The measured quality quantities a ramp can be declared over.

    Each is read from :class:`ObservationQuality` as a single number, in the units of the
    member name, and is available or not exactly as that contract says.
    """

    SUPPORT_DEPTH_MEDIAN_M = "support_depth_median_m"
    SUPPORT_OFF_AXIS_ANGLE_MEDIAN_RAD = "support_off_axis_angle_median_rad"
    BORDER_DISTANCE_MEDIAN_PX = "border_distance_median_px"
    VISIBLE_SHARE = "visible_share"
    OCCLUDED_FRACTION = "occluded_fraction"
    OUTSIDE_VALID_SUPPORT_FRACTION = "outside_valid_support_fraction"
    REPROJECTION_MEDIAN_PX = "reprojection_median_px"
    SUPPORT_DENSITY_PER_MASK_PIXEL = "support_density_per_mask_pixel"
    ASSOCIATED_COUNT = "associated_count"
    TEMPORAL_OFFSET_NS = "temporal_offset_ns"


@dataclass(frozen=True, kw_only=True)
class QualityRamp:
    """One declared component: how a measured quantity becomes a factor.

    Attributes:
        quality_input: The measured quantity.
        good: The value at or beyond which the factor is ``1``.
        bad: The value at or beyond which the factor is ``0``. Its order relative to
            ``good`` says whether higher or lower is better.
    """

    quality_input: QualityInput
    good: float
    bad: float

    def __post_init__(self) -> None:
        """Validate the ramp.

        Raises:
            ValueError: If a threshold is not finite or the two are equal.
        """
        if not (math.isfinite(self.good) and math.isfinite(self.bad)) or self.good == self.bad:
            raise ValueError(
                f"a quality ramp needs finite, different good and bad values, got "
                f"good={self.good!r} and bad={self.bad!r}"
            )

    def factor_of(self, value: float) -> float:
        """Map a measured value to its factor.

        Args:
            value: The measured value, in the units of the input.

        Returns:
            A factor in ``[0, 1]``.
        """
        return min(1.0, max(0.0, (value - self.bad) / (self.good - self.bad)))


@dataclass(frozen=True, kw_only=True)
class QualityAwareAccumulationPolicy:
    """The versioned configuration of the quality-aware policy.

    There are no defaults for the ramps, the neutral factor or the definitions version: they
    are scientific choices a profile declares.

    Attributes:
        definitions_version: The observation-quality definitions this policy understands;
            quality read under another version is refused.
        ramps: The declared components, one per quantity.
        neutral_factor: The factor a component takes when it cannot be measured, in ``[0, 1]``.
        baseline: The baseline settings the accumulation runs under (abstention labels, tie
            margin and channels). The quality channel is always added.
    """

    definitions_version: str
    ramps: tuple[QualityRamp, ...]
    neutral_factor: float
    baseline: BaselineAccumulationPolicy = field(default_factory=BaselineAccumulationPolicy)

    def __post_init__(self) -> None:
        """Validate the version, the ramps and the neutral factor.

        Raises:
            ValueError: If the version is empty, there is no ramp, a quantity is declared
                twice, or the neutral factor is not a number in ``[0, 1]``.
        """
        if not self.definitions_version.strip():
            raise ValueError("definitions_version must not be empty")
        if not self.ramps:
            raise ValueError("a quality-aware policy needs at least one ramp")
        inputs = [ramp.quality_input for ramp in self.ramps]
        for quality_input in inputs:
            if inputs.count(quality_input) > 1:
                raise ValueError(f"quality input {quality_input.value} is declared more than once")
        if not (math.isfinite(self.neutral_factor) and 0.0 <= self.neutral_factor <= 1.0):
            raise ValueError(f"neutral_factor must be within [0, 1], got {self.neutral_factor!r}")

    def fingerprint(self) -> str:
        """Hash the policy identity and configuration, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration, which includes
            the baseline settings.
        """
        canonical = json.dumps(
            {
                "policy_id": QUALITY_AWARE_ACCUMULATION_POLICY_ID,
                "combination_rule": COMBINATION_RULE,
                "observation_rule": OBSERVATION_RULE,
                "definitions_version": self.definitions_version,
                "ramps": sorted(
                    (ramp.quality_input.value, ramp.good, ramp.bad) for ramp in self.ramps
                ),
                "neutral_factor": self.neutral_factor,
                "baseline": self.baseline.fingerprint(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def accumulate_quality_aware_evidence(
    support: FusionSupport,
    *,
    observations: Mapping[SpatialObservationId, SpatialObservation],
    grouping: PhysicalObservationGrouping,
    perception_results: Mapping[PerceptionResultId, PerceptionResult],
    observation_qualities: Mapping[SpatialObservationId, ObservationQuality],
    policy: QualityAwareAccumulationPolicy,
    semantic_scores: Mapping[PerceptionResultId, Sequence[SemanticSupport]] | None = None,
    point_representation_refs: Iterable[PointRepresentationRef] | None = None,
    code_version: str | None = None,
) -> FusedEvidence:
    """Accumulate the evidence of one support and weight it by observation quality.

    The inputs are those of :func:`accumulate_baseline_evidence`, so the same upstream
    artifacts can run through both policies, plus the measured quality of the observations.

    Args:
        support: The support to accumulate over.
        observations: The spatial observations, by identity.
        grouping: The physical-observation grouping of the same observations.
        perception_results: The perception results that own the claims.
        observation_qualities: The measured quality of the observations, by identity. An
            observation without quality is weighted with the neutral factor and says so.
        policy: The quality-aware configuration.
        semantic_scores: Scorer outputs, needed if the baseline settings declare that channel.
        point_representation_refs: Static structure, needed if the baseline settings declare
            that channel; never weighted.
        code_version: Code revision to record in the provenance, when known.

    Returns:
        The evidence the baseline would produce, with ``weighting`` filled in and the
        provenance naming this policy.

    Raises:
        ValueError: If a quality was read under definitions this policy does not understand,
            or the baseline accumulation rejects its inputs.
    """
    for contribution_id in support.spatial_observation_ids:
        quality = observation_qualities.get(contribution_id)
        if quality is not None and quality.definitions_version != policy.definitions_version:
            raise ValueError(
                f"observation quality of {contribution_id!r} was read under definitions "
                f"{quality.definitions_version!r}, but the policy understands "
                f"{policy.definitions_version!r}"
            )
    references = {
        observation_id: ObservationQualityRef(
            spatial_observation_id=observation_id, definitions_version=quality.definitions_version
        )
        for observation_id, quality in observation_qualities.items()
    }
    baseline = dataclasses.replace(
        policy.baseline,
        channels=policy.baseline.channels | {EvidenceChannel.OBSERVATION_QUALITY},
    )
    fused = accumulate_baseline_evidence(
        support,
        observations=observations,
        grouping=grouping,
        perception_results=perception_results,
        semantic_scores=semantic_scores,
        observation_quality_refs=references,
        point_representation_refs=point_representation_refs,
        policy=baseline,
        code_version=code_version,
    )
    return dataclasses.replace(
        fused,
        weighting=_weighting(fused, observation_qualities, policy),
        provenance=dataclasses.replace(
            fused.provenance,
            fusion_policy_id=QUALITY_AWARE_ACCUMULATION_POLICY_ID,
            configuration_fingerprint=policy.fingerprint(),
        ),
    )


def _weighting(
    fused: FusedEvidence,
    qualities: Mapping[SpatialObservationId, ObservationQuality],
    policy: QualityAwareAccumulationPolicy,
) -> QualityWeighting:
    weights = [
        _contribution_weight(
            item.contribution_id, qualities.get(item.spatial_observation_id), policy
        )
        for item in fused.contributions
    ]
    factor_of = {weight.contribution_id: weight.factor for weight in weights}
    observation_of = {
        item.contribution_id: item.physical_observation_id for item in fused.contributions
    }

    supports: list[HypothesisSupport] = []
    for hypothesis in fused.hypotheses:
        supporting = {
            item.contribution_id
            for item in hypothesis.evidence
            if item.stance is EvidenceStance.SUPPORTING
        }
        by_observation: dict[SourceObservationId, list[float]] = {}
        for contribution_id in supporting:
            by_observation.setdefault(observation_of[contribution_id], []).append(
                factor_of[contribution_id]
            )
        factors = tuple(
            ObservationFactor(
                physical_observation_id=observation_id,
                factor=math.fsum(values) / len(values),
            )
            for observation_id, values in sorted(by_observation.items())
        )
        supports.append(
            HypothesisSupport(
                hypothesis_id=hypothesis.hypothesis_id,
                supporting_physical_observations=len(factors),
                observation_factors=factors,
                weighted_support=math.fsum(item.factor for item in factors),
            )
        )
    return QualityWeighting(
        policy_id=QUALITY_AWARE_ACCUMULATION_POLICY_ID,
        combination_rule=COMBINATION_RULE,
        observation_rule=OBSERVATION_RULE,
        definitions_version=policy.definitions_version,
        contributions=tuple(weights),
        hypotheses=tuple(supports),
    )


def _contribution_weight(
    contribution_id: EvidenceContributionId,
    quality: ObservationQuality | None,
    policy: QualityAwareAccumulationPolicy,
) -> ContributionWeight:
    components = tuple(
        _component_factor(ramp, quality, policy.neutral_factor)
        for ramp in sorted(policy.ramps, key=lambda item: item.quality_input.value)
    )
    return ContributionWeight(
        contribution_id=contribution_id,
        components=components,
        factor=min(component.factor for component in components),
    )


def _component_factor(
    ramp: QualityRamp, quality: ObservationQuality | None, neutral_factor: float
) -> ComponentFactor:
    name = ramp.quality_input.value
    if quality is None:
        return ComponentFactor(
            component=name,
            treatment=ComponentTreatment.NEUTRAL_FALLBACK,
            measured_value=None,
            factor=neutral_factor,
            unavailable_reason="no observation quality was provided for this observation",
        )
    value, reason = _measure(quality, ramp.quality_input)
    if value is None:
        return ComponentFactor(
            component=name,
            treatment=ComponentTreatment.NEUTRAL_FALLBACK,
            measured_value=None,
            factor=neutral_factor,
            unavailable_reason=reason,
        )
    return ComponentFactor(
        component=name,
        treatment=ComponentTreatment.MEASURED,
        measured_value=value,
        factor=ramp.factor_of(value),
        unavailable_reason=None,
    )


def _measure(
    quality: ObservationQuality, quality_input: QualityInput
) -> tuple[float | None, str | None]:
    """Read one quantity from the quality, with the reason it is unavailable when it is."""
    optional: dict[QualityInput, tuple[QualityComponent, float | None]] = {
        QualityInput.SUPPORT_DEPTH_MEDIAN_M: (
            QualityComponent.SUPPORT_DEPTH,
            None if quality.support_depth_m is None else quality.support_depth_m.median,
        ),
        QualityInput.SUPPORT_OFF_AXIS_ANGLE_MEDIAN_RAD: (
            QualityComponent.SUPPORT_OFF_AXIS_ANGLE,
            None
            if quality.support_off_axis_angle_rad is None
            else quality.support_off_axis_angle_rad.median,
        ),
        QualityInput.BORDER_DISTANCE_MEDIAN_PX: (
            QualityComponent.BORDER_DISTANCE,
            None if quality.border_distance_px is None else quality.border_distance_px.median,
        ),
        QualityInput.VISIBLE_SHARE: (QualityComponent.VISIBLE_SHARE, quality.visible_share),
        QualityInput.OCCLUDED_FRACTION: (
            QualityComponent.OCCLUDED_FRACTION,
            quality.occluded_fraction,
        ),
        QualityInput.OUTSIDE_VALID_SUPPORT_FRACTION: (
            QualityComponent.OUTSIDE_VALID_SUPPORT_FRACTION,
            quality.outside_valid_support_fraction,
        ),
        QualityInput.REPROJECTION_MEDIAN_PX: (
            QualityComponent.REPROJECTION,
            None if quality.reprojection is None else quality.reprojection.median_px,
        ),
    }
    if quality_input in optional:
        component, value = optional[quality_input]
        return value, None if value is not None else quality.unavailable[component]
    always = {
        QualityInput.SUPPORT_DENSITY_PER_MASK_PIXEL: quality.support_density_per_mask_pixel,
        QualityInput.ASSOCIATED_COUNT: float(quality.associated_count),
        QualityInput.TEMPORAL_OFFSET_NS: float(quality.temporal_offset_ns),
    }
    return always[quality_input], None
