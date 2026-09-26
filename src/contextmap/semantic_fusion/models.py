"""Canonical contracts for Semantic Fusion.

Semantic Fusion accumulates evidence from several views over a shared spatial
support **without** deciding what object that support is. Its output,
:class:`FusedEvidence`, is *belief in formation*: it keeps every hypothesis, the
exact evidence behind it, the conflicts between physical observations and what
is simply unknown. It does not elect a winner, it creates no entity identity and
it never averages heterogeneous scores into one number.

Four ideas shape the contracts:

* a :class:`FusionSupport` is only *where* evidence is accumulated -- geometry and
  the spatial observations that see it. It asserts no label and no object identity;
* an :class:`EvidenceContribution` is *one view*: one region of one physical frame
  as interpreted by one inference run. Several runs over the same frame are
  correlated inference, so contributions carry the physical observation identity
  and a :class:`PhysicalObservationGroup` counts frames apart from inferences;
* signals stay typed and unmixed: a claim's own confidence, a scorer's support and
  the measurable observation quality are three different things. Quality is only
  *referenced* by a contribution; it is never a :class:`SupportSignal`. A signal
  whose value is ``None`` is unscored -- never a zero;
* static 3D structure (:class:`PointRepresentationRef`) belongs to the support and
  is listed once, so it cannot be counted once per camera view.

Everything upstream is referenced by identity; nothing here copies XYZ,
embeddings, claims or quality components. See
``src/contextmap/semantic_fusion/docs/contracts.md`` for the field reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import NewType, TypeVar

from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId
from contextmap.ingestion import SourceObservationId
from contextmap.point_representation import PointRepresentationId, PointRepresentationRunId
from contextmap.sensor_association import SemanticClaimRef, SpatialObservationId, VisualFeatureRef
from contextmap.shared import SourceTimestamp, Vector3
from contextmap.state_estimation import TimeBounds
from contextmap.visual_perception import (
    BackendProvenance,
    ClaimId,
    FeatureScope,
    HypothesisRole,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)

FusionSupportId = NewType("FusionSupportId", str)
"""Identity of one spatial support, local to a Semantic Fusion run."""

EvidenceContributionId = NewType("EvidenceContributionId", str)
"""Identity of one view's contribution, local to a Semantic Fusion run."""

FusedEvidenceId = NewType("FusedEvidenceId", str)
"""Identity of the evidence fused over one support, local to a Semantic Fusion run."""

FusedHypothesisId = NewType("FusedHypothesisId", str)
"""Identity of one hypothesis, local to its :class:`FusedEvidence`."""

_Item = TypeVar("_Item")


def evidence_contribution_id_for(
    *, fusion_support_id: FusionSupportId, spatial_observation_id: SpatialObservationId
) -> EvidenceContributionId:
    """Compute the deterministic identity of one observation's contribution to one support.

    Args:
        fusion_support_id: The support the evidence is accumulated over.
        spatial_observation_id: The spatial observation the view comes from.

    Returns:
        A pure function of the inputs, so a contribution can be referenced without a registry.
    """
    return EvidenceContributionId(f"contribution--{fusion_support_id}--{spatial_observation_id}")


def fused_evidence_id_for(*, fusion_support_id: FusionSupportId) -> FusedEvidenceId:
    """Compute the deterministic identity of the evidence fused over one support."""
    return FusedEvidenceId(f"fused--{fusion_support_id}")


def _require_unit_factor(name: str, value: float) -> None:
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be within [0, 1], got {value!r}")


def _require_present(owner: object, *names: str) -> None:
    for name in names:
        if not str(getattr(owner, name)).strip():
            raise ValueError(f"{name} must not be empty")


def _require_canonical(
    name: str, items: Sequence[_Item], key: Callable[[_Item], tuple[str, ...]], *, detail: str = ""
) -> None:
    """Require strictly increasing keys, so equivalent evidence encodes identically."""
    keys = [key(item) for item in items]
    if any(left >= right for left, right in pairwise(keys)):
        raise ValueError(f"{name} must be sorted {detail}and unique")


def _time_bounds_of(stamps: Sequence[SourceTimestamp], *, owner: str) -> TimeBounds:
    """Span the given acquisition timestamps, which must share one clock domain."""
    clocks = {stamp.clock_id for stamp in stamps}
    if len(clocks) != 1:
        raise ValueError(f"{owner} span more than one clock domain: {sorted(clocks)!r}")
    return TimeBounds(
        start=min(stamps, key=SourceTimestamp.total_nanoseconds),
        end=max(stamps, key=SourceTimestamp.total_nanoseconds),
    )


def _uncertainty_key(record: UncertaintyRecord) -> tuple[str, ...]:
    return (
        record.kind.value,
        record.rule_id,
        "|".join(record.hypothesis_ids),
        "|".join(f"{ref.contribution_id}/{ref.claim_id or ''}" for ref in record.evidence),
    )


def _producer_key(producer: BackendProvenance) -> tuple[str, ...]:
    return (
        producer.backend_id,
        producer.capability,
        producer.provider,
        producer.model,
        producer.version,
        producer.configuration_fingerprint or "",
    )


def _require_identified(name: str, producer: BackendProvenance) -> None:
    if not (producer.backend_id.strip() and producer.model.strip() and producer.version.strip()):
        raise ValueError(f"{name} must identify its backend_id, model and version")


def _require_geometry_support(name: str, references: Sequence[GeometryReference]) -> None:
    if not references:
        raise ValueError(f"{name} must not be empty")
    _require_canonical(
        name, references, lambda reference: (reference.geometry_id,), detail="by geometry_id "
    )


class EvidenceChannel(Enum):
    """The typed evidence channels a fusion policy can declare.

    Each channel keeps its own semantics, scale and provenance. None is renamed, converted
    into or averaged with another, and a channel takes part only when the policy declares
    it: data being available never activates it.

    Attributes:
        SEMANTIC_CLAIMS: Hypotheses, alternatives and abstentions from an interpreter.
        SEMANTIC_SCORES: Scorer outputs, each specific to its scorer unless calibrated.
        VISUAL_FEATURES: References to visual features, with their embedding space.
        OBSERVATION_QUALITY: References to the measurable view quality.
        GEOMETRY_SUPPORT: The geometry an observation sees. It is intrinsic to every
            contribution, so it is always active and cannot be ablated.
        POINT_REPRESENTATION: References to static 3D structure, attached to the support.
    """

    SEMANTIC_CLAIMS = "semantic_claims"
    SEMANTIC_SCORES = "semantic_scores"
    VISUAL_FEATURES = "visual_features"
    OBSERVATION_QUALITY = "observation_quality"
    GEOMETRY_SUPPORT = "geometry_support"
    POINT_REPRESENTATION = "point_representation"


class ComponentTreatment(Enum):
    """How a quality component entered a contribution factor.

    Attributes:
        MEASURED: The component was measured and turned into a factor by its ramp.
        NEUTRAL_FALLBACK: The component could not be measured, so the policy's declared
            neutral factor was used, and the reason is recorded. Never a silent zero.
    """

    MEASURED = "measured"
    NEUTRAL_FALLBACK = "neutral_fallback"


class SupportSignalKind(Enum):
    """The typed origin of a :class:`SupportSignal`.

    The two kinds are different quantities that are never combined here. Observation
    quality is deliberately **not** a kind: it is measured upstream and only referenced.

    Attributes:
        CLAIM_CONFIDENCE: The confidence an interpreter reported for its own claim.
        SCORER_SUPPORT: How well visual evidence supports a claim according to a scorer.
    """

    CLAIM_CONFIDENCE = "claim_confidence"
    SCORER_SUPPORT = "scorer_support"


class EvidenceStance(Enum):
    """How one claim relates to the hypothesis that lists it.

    Attributes:
        SUPPORTING: The claim proposes this hypothesis.
        CONFLICTING: The claim proposes an incompatible hypothesis for the same support.
        AMBIGUOUS: The claim is retained but its relation to this hypothesis is undecided.
        ABSTAINING: The claim is an abstention (``unknown``): it is neither support for the
            hypothesis nor evidence against it.
    """

    SUPPORTING = "supporting"
    CONFLICTING = "conflicting"
    AMBIGUOUS = "ambiguous"
    ABSTAINING = "abstaining"


class UncertaintyKind(Enum):
    """Why a fused evidence does not settle on one interpretation.

    Attributes:
        CONTRADICTION: Distinct physical observations propose incompatible hypotheses.
        AMBIGUITY: Competing hypotheses that no contradiction between observations explains,
            for example alternatives proposed by one interpretation.
        NEAR_TIE: Hypotheses indistinguishable under the rule that declared the tie.
        INSUFFICIENT_EVIDENCE: Too little semantic evidence to support any hypothesis.
    """

    CONTRADICTION = "contradiction"
    AMBIGUITY = "ambiguity"
    NEAR_TIE = "near_tie"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


_COMPETITIVE_KINDS = frozenset(
    {UncertaintyKind.CONTRADICTION, UncertaintyKind.AMBIGUITY, UncertaintyKind.NEAR_TIE}
)


@dataclass(frozen=True, kw_only=True)
class FusionSupportProvenance:
    """How a spatial support was built.

    Attributes:
        support_policy_id: Versioned rule that decided which observations share a support.
        configuration_fingerprint: Hash of the support configuration, when configurable.
        code_version: Code revision that produced the support, when known.
    """

    support_policy_id: str
    configuration_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require the policy identity.

        Raises:
            ValueError: If ``support_policy_id`` is empty.
        """
        _require_present(self, "support_policy_id")


@dataclass(frozen=True, kw_only=True)
class FusionSupport:
    """A spatial support over which evidence can be accumulated.

    It states only that its spatial observations see compatible geometry under the
    support policy. It asserts no label, no class and no object identity, and it is
    not stable across map versions: entity identity belongs to later capabilities.

    Attributes:
        fusion_support_id: Identity of this support.
        geometric_map_id: The immutable map the geometry belongs to.
        geometry_support: References to the geometry, sorted by ``geometry_id`` and unique.
        spatial_observation_ids: The spatial observations accumulated here, sorted and unique.
        bounds: Extent of the geometry, expressed in the frame it declares.
        centroid_m: Centroid of the geometry, in meters, in the frame of ``bounds``.
        time_bounds: Interval spanned by the acquisition of the observations.
        provenance: How the support was built.
    """

    fusion_support_id: FusionSupportId
    geometric_map_id: MapId
    geometry_support: tuple[GeometryReference, ...]
    spatial_observation_ids: tuple[SpatialObservationId, ...]
    bounds: Bounds3D
    centroid_m: Vector3
    time_bounds: TimeBounds
    provenance: FusionSupportProvenance

    def __post_init__(self) -> None:
        """Validate identities, geometry, observations and the centroid.

        Raises:
            ValueError: If an identity or the support is empty, the geometry spans another
                map or is not sorted and unique, the observations are not sorted and unique,
                or the centroid is not finite or lies outside ``bounds``.
        """
        _require_present(self, "fusion_support_id", "geometric_map_id")
        _require_geometry_support("geometry_support", self.geometry_support)
        foreign = {reference.map_id for reference in self.geometry_support} - {
            self.geometric_map_id
        }
        if foreign:
            raise ValueError(
                f"geometry_support must reference only the support map "
                f"{self.geometric_map_id!r}, got maps {sorted(foreign)!r}"
            )
        if not self.spatial_observation_ids:
            raise ValueError("spatial_observation_ids must not be empty")
        _require_canonical(
            "spatial_observation_ids", self.spatial_observation_ids, lambda item: (item,)
        )
        if not all(math.isfinite(value) for value in self.centroid_m):
            raise ValueError(f"centroid_m must be finite, got {self.centroid_m!r}")
        if not self.bounds.contains(self.centroid_m, frame_id=self.bounds.frame_id):
            raise ValueError(
                f"centroid_m {self.centroid_m!r} must lie inside the support bounds "
                f"{self.bounds.minimum_m!r}..{self.bounds.maximum_m!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ScoreReference:
    """Reference to the support a scorer assigned to one claim.

    Attributes:
        claim_id: The scored claim, local to the contribution's perception result.
        scorer: The scorer that produced the score.
    """

    claim_id: ClaimId
    scorer: BackendProvenance

    def __post_init__(self) -> None:
        """Require the claim and the scorer identity.

        Raises:
            ValueError: If the claim is empty or the scorer is not identified.
        """
        _require_present(self, "claim_id")
        _require_identified("scorer", self.scorer)


@dataclass(frozen=True, kw_only=True)
class ObservationQualityRef:
    """Reference to the measurable quality of one spatial observation.

    The quality itself stays with Sensor Association as separate typed components;
    a reference is all a contribution carries, so quality cannot be mistaken for a
    semantic score.

    Attributes:
        spatial_observation_id: The observation the quality describes.
        definitions_version: Version of the quality component definitions.
    """

    spatial_observation_id: SpatialObservationId
    definitions_version: str

    def __post_init__(self) -> None:
        """Require the observation and the definitions version.

        Raises:
            ValueError: If either is empty.
        """
        _require_present(self, "spatial_observation_id", "definitions_version")


@dataclass(frozen=True, kw_only=True)
class EvidenceContribution:
    """One view's evidence about a support: one region of one frame, one inference run.

    ``physical_observation_id`` is the correlation key: contributions that share it
    are correlated inference over the same sensor data, never independent views.
    A claim appears once however many geometry elements the region sees.

    Attributes:
        contribution_id: Identity of this contribution.
        physical_observation_id: The physical frame that was interpreted.
        perception_result_id: The inference result that produced the claims.
        perception_run_id: The run that owns that result.
        spatial_observation_id: The spatial observation this view comes from.
        region_id: The observed region.
        geometry_support: The geometry the region sees, sorted by ``geometry_id`` and
            unique, all from one map.
        claim_refs: Claims about the region, sorted by claim and unique; empty when the
            inference produced none.
        score_refs: Scorer outputs for those claims, sorted and unique.
        visual_feature_refs: Features describing the region, sorted by feature and unique.
        observation_quality: Reference to the measurable view quality, when available.
    """

    contribution_id: EvidenceContributionId
    physical_observation_id: SourceObservationId
    perception_result_id: PerceptionResultId
    perception_run_id: PerceptionRunId
    spatial_observation_id: SpatialObservationId
    region_id: RegionId
    geometry_support: tuple[GeometryReference, ...]
    claim_refs: tuple[SemanticClaimRef, ...]
    score_refs: tuple[ScoreReference, ...]
    visual_feature_refs: tuple[VisualFeatureRef, ...]
    observation_quality: ObservationQualityRef | None

    def __post_init__(self) -> None:
        """Validate identities, geometry and the internal consistency of the references.

        Raises:
            ValueError: If an identity or the geometry is empty, the geometry spans more than
                one map, a collection is not sorted and unique, a score names a claim the
                contribution does not carry, a region-scoped feature belongs to another
                region, or the quality reference describes another observation.
        """
        _require_present(
            self,
            "contribution_id",
            "physical_observation_id",
            "perception_result_id",
            "perception_run_id",
            "spatial_observation_id",
            "region_id",
        )
        _require_geometry_support("geometry_support", self.geometry_support)
        if len({reference.map_id for reference in self.geometry_support}) > 1:
            raise ValueError("geometry_support must belong to one map")
        _require_canonical("claim_refs", self.claim_refs, lambda ref: (ref.claim_id,))
        claim_ids = {ref.claim_id for ref in self.claim_refs}
        for score in self.score_refs:
            if score.claim_id not in claim_ids:
                raise ValueError(
                    f"score reference names claim {score.claim_id!r}, which the "
                    f"contribution does not carry"
                )
        _require_canonical(
            "score_refs", self.score_refs, lambda ref: (ref.claim_id, *_producer_key(ref.scorer))
        )
        _require_canonical(
            "visual_feature_refs", self.visual_feature_refs, lambda ref: (ref.feature_id,)
        )
        for feature in self.visual_feature_refs:
            if feature.scope is FeatureScope.REGION and feature.region_id != self.region_id:
                raise ValueError(
                    f"region-scoped feature {feature.feature_id!r} belongs to region "
                    f"{feature.region_id!r}, not the contribution region {self.region_id!r}"
                )
        quality = self.observation_quality
        if quality is not None and quality.spatial_observation_id != self.spatial_observation_id:
            raise ValueError(
                f"observation_quality describes {quality.spatial_observation_id!r}, not the "
                f"contribution observation {self.spatial_observation_id!r}"
            )


@dataclass(frozen=True, kw_only=True)
class PhysicalObservationGroup:
    """Everything inferred from one physical frame, kept apart from the frame itself.

    Attributes:
        physical_observation_id: The physical frame.
        acquisition_timestamp: When the frame was acquired.
        spatial_observation_ids: Spatial observations of this frame, sorted and unique.
        perception_result_ids: The correlated inference results over this frame.
        perception_run_ids: The runs that produced those results.
    """

    physical_observation_id: SourceObservationId
    acquisition_timestamp: SourceTimestamp
    spatial_observation_ids: tuple[SpatialObservationId, ...]
    perception_result_ids: tuple[PerceptionResultId, ...]
    perception_run_ids: tuple[PerceptionRunId, ...]

    def __post_init__(self) -> None:
        """Validate the identity and that each collection is present, sorted and unique.

        Raises:
            ValueError: If the identity or a collection is empty or not sorted and unique.
        """
        _require_present(self, "physical_observation_id")
        for name in ("spatial_observation_ids", "perception_result_ids", "perception_run_ids"):
            items: tuple[str, ...] = getattr(self, name)
            if not items:
                raise ValueError(f"{name} must not be empty")
            _require_canonical(name, items, lambda item: (item,))

    @property
    def inference_result_count(self) -> int:
        """Number of inference results over this one physical observation."""
        return len(self.perception_result_ids)


@dataclass(frozen=True, kw_only=True)
class SupportSignal:
    """A typed score attached to one claim, with the model whose semantics it has.

    ``value`` is ``None`` when the producer emitted no score. That is *unscored*, not
    zero: a low score and a missing score are different evidence.

    Attributes:
        kind: Which quantity this is.
        producer: The interpreter or scorer that produced the value.
        value: A score in ``[0, 1]`` under the producer's own semantics, or ``None``.
    """

    kind: SupportSignalKind
    producer: BackendProvenance
    value: float | None

    def __post_init__(self) -> None:
        """Validate the producer and the value.

        Raises:
            ValueError: If the producer is not identified or the value is neither ``None``
                nor a finite number in ``[0, 1]``.
        """
        _require_identified("producer", self.producer)
        if self.value is not None and not (math.isfinite(self.value) and 0.0 <= self.value <= 1.0):
            raise ValueError(f"signal value must be None or within [0, 1], got {self.value!r}")


@dataclass(frozen=True, kw_only=True)
class HypothesisEvidence:
    """One claim of one contribution, as it relates to one hypothesis.

    Attributes:
        contribution_id: The contribution that carries the claim.
        claim_id: The claim, local to that contribution's perception result.
        stance: How the claim relates to the hypothesis.
        role: Whether the interpreter proposed the claim as its primary or an alternative.
        signals: Typed scores of the claim, sorted by kind and producer and unique.
    """

    contribution_id: EvidenceContributionId
    claim_id: ClaimId
    stance: EvidenceStance
    role: HypothesisRole
    signals: tuple[SupportSignal, ...] = ()

    def __post_init__(self) -> None:
        """Validate the references and the order of the signals.

        Raises:
            ValueError: If an identity is empty or the signals are not sorted and unique.
        """
        _require_present(self, "contribution_id", "claim_id")
        _require_canonical(
            "signals",
            self.signals,
            lambda signal: (signal.kind.value, *_producer_key(signal.producer)),
        )


@dataclass(frozen=True, kw_only=True)
class FusedHypothesis:
    """A semantic candidate for a support, with every piece of evidence behind it.

    The label is the open-vocabulary text as proposed; equivalence between labels is a
    policy decision, never assumed here. There is no probability and no single
    confidence: the typed signals of each evidence item stay separate.

    Attributes:
        hypothesis_id: Identity local to its :class:`FusedEvidence`.
        label: The hypothesis text, verbatim.
        evidence: Claims that support, conflict with or are ambiguous about the hypothesis,
            sorted by contribution and claim and unique.
    """

    hypothesis_id: FusedHypothesisId
    label: str
    evidence: tuple[HypothesisEvidence, ...]

    def __post_init__(self) -> None:
        """Validate the label and the evidence.

        Raises:
            ValueError: If the identity or label is empty, the evidence is not sorted and
                unique, or no evidence item supports the hypothesis.
        """
        _require_present(self, "hypothesis_id", "label")
        _require_canonical(
            "evidence", self.evidence, lambda item: (item.contribution_id, item.claim_id)
        )
        if not any(item.stance is EvidenceStance.SUPPORTING for item in self.evidence):
            raise ValueError(
                f"hypothesis {self.label!r} needs at least one supporting evidence item"
            )


@dataclass(frozen=True, kw_only=True)
class EvidenceReference:
    """Reference to a contribution, and to one of its claims when the evidence is a claim.

    Attributes:
        contribution_id: The contribution.
        claim_id: The claim, or ``None`` when the evidence is the contribution itself, for
            example a view that produced no claim.
    """

    contribution_id: EvidenceContributionId
    claim_id: ClaimId | None = None

    def __post_init__(self) -> None:
        """Require the contribution and, when present, the claim.

        Raises:
            ValueError: If the contribution is empty or a given claim is empty.
        """
        _require_present(self, "contribution_id")
        if self.claim_id is not None:
            _require_present(self, "claim_id")


@dataclass(frozen=True, kw_only=True)
class UncertaintyRecord:
    """A conflict, ambiguity or lack of evidence, with the exact evidence behind it.

    Attributes:
        kind: What kind of uncertainty this is.
        hypothesis_ids: The hypotheses involved, sorted and unique.
        evidence: The contributions and claims that produced it, sorted and unique.
        rule_id: The versioned rule that declared it.
    """

    kind: UncertaintyKind
    hypothesis_ids: tuple[FusedHypothesisId, ...]
    evidence: tuple[EvidenceReference, ...]
    rule_id: str

    def __post_init__(self) -> None:
        """Validate the rule, the ordering and that competition names competitors.

        Raises:
            ValueError: If the rule is empty, a collection is not sorted and unique, or a
                contradiction, ambiguity or tie names fewer than two hypotheses.
        """
        _require_present(self, "rule_id")
        _require_canonical("hypothesis_ids", self.hypothesis_ids, lambda item: (item,))
        _require_canonical(
            "evidence", self.evidence, lambda ref: (ref.contribution_id, ref.claim_id or "")
        )
        if self.kind in _COMPETITIVE_KINDS and len(self.hypothesis_ids) < 2:
            raise ValueError(f"a {self.kind.value} needs at least two hypotheses")


@dataclass(frozen=True, kw_only=True)
class PointRepresentationRef:
    """Reference to a 3D representation anchored to the support's geometry.

    Static structure of the map, not a view: it is listed once on the
    :class:`FusedEvidence`, so it is never counted once per camera observation.

    Attributes:
        representation_id: The representation, local to its run.
        run_id: The Point Representation run that owns it.
        representation_space_id: Fingerprint of its representation space; representations
            are comparable only within one space.
        geometry_reference: The geometry element the representation is centered on.
    """

    representation_id: PointRepresentationId
    run_id: PointRepresentationRunId
    representation_space_id: str
    geometry_reference: GeometryReference

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        _require_present(self, "representation_id", "run_id", "representation_space_id")


@dataclass(frozen=True, kw_only=True)
class ChannelProvenance:
    """An active evidence channel and the identities of what fed it.

    A channel can be active and empty: a declared channel that found no data for a support
    has no identities. Identities are strings whose form depends on the channel: an
    interpreter or scorer as ``backend_id/model/version``, an embedding space, a quality
    definitions version, a representation space, or the map and support policy.

    Attributes:
        channel: The active channel.
        identities: What fed it, sorted and unique.
    """

    channel: EvidenceChannel
    identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the identities.

        Raises:
            ValueError: If an identity is empty or the identities are not sorted and unique.
        """
        if any(not identity.strip() for identity in self.identities):
            raise ValueError("a channel identity must not be empty")
        _require_canonical("identities", self.identities, lambda identity: (identity,))


@dataclass(frozen=True, kw_only=True)
class ComponentFactor:
    """The factor one declared quality component gave one contribution.

    Attributes:
        component: The measured quantity, named by the policy that declared it.
        treatment: Whether it was measured or replaced by the neutral factor.
        measured_value: The measured value; ``None`` for a neutral fallback.
        factor: The factor in ``[0, 1]``.
        unavailable_reason: Why the component could not be measured; ``None`` when measured.
    """

    component: str
    treatment: ComponentTreatment
    measured_value: float | None
    factor: float
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        """Validate the factor and that the treatment matches the value and the reason.

        Raises:
            ValueError: If the component is empty, the factor is outside ``[0, 1]``, a
                measured component has no finite value or has a reason, or a neutral
                fallback has a value or no reason.
        """
        _require_present(self, "component")
        _require_unit_factor("factor", self.factor)
        if self.treatment is ComponentTreatment.MEASURED:
            if self.measured_value is None or not math.isfinite(self.measured_value):
                raise ValueError("a measured component needs a finite measured_value")
            if self.unavailable_reason is not None:
                raise ValueError("unavailable_reason must be None for a measured component")
        else:
            if self.measured_value is not None:
                raise ValueError("measured_value must be None for a neutral fallback")
            if not (self.unavailable_reason or "").strip():
                raise ValueError("unavailable_reason is required for a neutral fallback")


@dataclass(frozen=True, kw_only=True)
class ContributionWeight:
    """The fusion factor of one contribution, with every component behind it.

    The factor is a fusion contribution weight. It is not a semantic confidence, a scorer
    support or a probability, and it never replaces the raw quality it came from.

    Attributes:
        contribution_id: The weighted contribution.
        components: The declared components and their factors, sorted by component.
        factor: The contribution factor in ``[0, 1]``, combined by the weighting's rule.
    """

    contribution_id: EvidenceContributionId
    components: tuple[ComponentFactor, ...]
    factor: float

    def __post_init__(self) -> None:
        """Validate the identity, the components and the factor.

        Raises:
            ValueError: If the identity is empty, there is no component, the components are
                not sorted and unique, or the factor is outside ``[0, 1]``.
        """
        _require_present(self, "contribution_id")
        if not self.components:
            raise ValueError("a contribution weight needs at least one component")
        _require_canonical("components", self.components, lambda item: (item.component,))
        _require_unit_factor("factor", self.factor)


@dataclass(frozen=True, kw_only=True)
class ObservationFactor:
    """The factor of one physical observation for one hypothesis.

    Attributes:
        physical_observation_id: The physical frame.
        factor: Its factor in ``[0, 1]``, however many inference runs interpreted it.
    """

    physical_observation_id: SourceObservationId
    factor: float

    def __post_init__(self) -> None:
        """Validate the identity and the factor.

        Raises:
            ValueError: If the identity is empty or the factor is outside ``[0, 1]``.
        """
        _require_present(self, "physical_observation_id")
        _require_unit_factor("factor", self.factor)


@dataclass(frozen=True, kw_only=True)
class HypothesisSupport:
    """The support of one hypothesis before and after the weighting is applied.

    Attributes:
        hypothesis_id: The hypothesis.
        supporting_physical_observations: Distinct supporting physical observations: the
            support *before* weighting, the same count the baseline uses.
        observation_factors: The factor of each of those observations, sorted by observation.
        weighted_support: The sum of those factors, in units of weighted physical
            observations: the support *after* weighting. It is not a probability.
    """

    hypothesis_id: FusedHypothesisId
    supporting_physical_observations: int
    observation_factors: tuple[ObservationFactor, ...]
    weighted_support: float

    def __post_init__(self) -> None:
        """Validate the counts and that the weighted support is the sum of the factors.

        Raises:
            ValueError: If the identity is empty, the count differs from the factors listed,
                the factors are not sorted and unique, or the weighted support is not the
                sum of the factors.
        """
        _require_present(self, "hypothesis_id")
        _require_canonical(
            "observation_factors",
            self.observation_factors,
            lambda item: (item.physical_observation_id,),
        )
        if self.supporting_physical_observations != len(self.observation_factors):
            raise ValueError(
                f"supporting_physical_observations {self.supporting_physical_observations} "
                f"must equal the {len(self.observation_factors)} observation_factors listed"
            )
        total = math.fsum(item.factor for item in self.observation_factors)
        if not math.isclose(self.weighted_support, total, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"weighted_support {self.weighted_support!r} must equal the sum of the "
                f"observation factors, {total!r}"
            )


@dataclass(frozen=True, kw_only=True)
class QualityWeighting:
    """How a quality-aware policy weighted the evidence, for inspection and reproduction.

    The evidence itself is untouched: hypotheses, contributions and uncertainty are exactly
    what the baseline would have produced. This only adds the derived factors.

    Attributes:
        policy_id: The versioned quality-aware policy.
        combination_rule: How component factors combine into a contribution factor.
        observation_rule: How the factors of several contributions of one physical
            observation combine, so repeated inference is never counted as several views.
        definitions_version: The quality definitions the components were read under.
        contributions: The factor of every contribution, sorted by contribution.
        hypotheses: The support of every hypothesis before and after weighting, sorted.
    """

    policy_id: str
    combination_rule: str
    observation_rule: str
    definitions_version: str
    contributions: tuple[ContributionWeight, ...]
    hypotheses: tuple[HypothesisSupport, ...]

    def __post_init__(self) -> None:
        """Validate the identities and the ordering.

        Raises:
            ValueError: If a policy, rule or version is empty, or a collection is not sorted
                and unique.
        """
        _require_present(
            self, "policy_id", "combination_rule", "observation_rule", "definitions_version"
        )
        _require_canonical(
            "contributions", self.contributions, lambda item: (item.contribution_id,)
        )
        _require_canonical("hypotheses", self.hypotheses, lambda item: (item.hypothesis_id,))


@dataclass(frozen=True, kw_only=True)
class FusedEvidenceProvenance:
    """How evidence was grouped and fused.

    Attributes:
        grouping_policy_id: Versioned rule that grouped inference by physical observation.
        fusion_policy_id: Versioned rule that accumulated the evidence.
        configuration_fingerprint: Hash of the fusion configuration, when configurable.
        code_version: Code revision that produced the evidence, when known.
    """

    grouping_policy_id: str
    fusion_policy_id: str
    configuration_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require both policy identities.

        Raises:
            ValueError: If a policy identity is empty.
        """
        _require_present(self, "grouping_policy_id", "fusion_policy_id")


@dataclass(frozen=True, kw_only=True)
class FusedEvidence:
    """The evidence accumulated over one support, with nothing collapsed.

    There is no winner, no primary hypothesis and no combined confidence: competing
    hypotheses, contradictions, ambiguity and missing evidence are all represented.
    Persistent entity identity is created by Semantic Mapping, not here.

    Attributes:
        fused_evidence_id: Identity of this fused evidence.
        fusion_support_id: The support the evidence was accumulated over.
        physical_observation_groups: One group per physical frame, sorted by frame.
        contributions: Every view's contribution, sorted by contribution and unique.
        hypotheses: The candidate interpretations, sorted by hypothesis and unique; empty
            when no view produced a claim.
        point_representation_refs: Static 3D structure of the support, listed once each.
        uncertainty: Conflicts, ambiguities and lack of evidence.
        temporal_summary: Interval spanned by the acquisition of the physical observations.
        provenance: How the evidence was grouped and fused.
        channels: The evidence channels the policy declared, sorted by channel and unique,
            each with the identities that fed it. Semantic claims and geometry support are
            always present; data of any other channel exists only if that channel is here,
            so an effect can be attributed to a channel.
        weighting: The factors a quality-aware policy derived, or ``None`` for the baseline.
    """

    fused_evidence_id: FusedEvidenceId
    fusion_support_id: FusionSupportId
    physical_observation_groups: tuple[PhysicalObservationGroup, ...]
    contributions: tuple[EvidenceContribution, ...]
    hypotheses: tuple[FusedHypothesis, ...]
    temporal_summary: TimeBounds
    provenance: FusedEvidenceProvenance
    channels: tuple[ChannelProvenance, ...]
    point_representation_refs: tuple[PointRepresentationRef, ...] = ()
    uncertainty: tuple[UncertaintyRecord, ...] = ()
    weighting: QualityWeighting | None = None

    def __post_init__(self) -> None:
        """Validate that groups, hypotheses and uncertainty agree with the contributions.

        Raises:
            ValueError: If an identity is empty, there is no contribution, a collection is
                not sorted and unique, a group disagrees with its contributions, one
                inference result is claimed by two physical observations, the temporal
                summary does not span the groups, a hypothesis or uncertainty references
                evidence that does not exist, a scorer signal has no score reference, or a
                hypothesis label is repeated.
        """
        _require_present(self, "fused_evidence_id", "fusion_support_id")
        if not self.contributions:
            raise ValueError("a fused evidence needs at least one contribution")
        self._require_canonical_order()
        self._require_channels_match()
        contributions = {item.contribution_id: item for item in self.contributions}
        self._require_groups_match(contributions)
        self._require_temporal_summary()
        self._require_hypotheses_resolve(contributions)
        self._require_uncertainty_resolves(contributions)
        self._require_weighting_matches()

    @property
    def physical_observation_count(self) -> int:
        """Number of distinct physical frames, however many inferences interpreted each."""
        return len(self.physical_observation_groups)

    @property
    def inference_result_count(self) -> int:
        """Number of inference results, correlated within each physical observation."""
        return sum(group.inference_result_count for group in self.physical_observation_groups)

    def supporting_physical_observations(
        self, hypothesis_id: FusedHypothesisId
    ) -> tuple[SourceObservationId, ...]:
        """List the distinct physical frames whose claims support a hypothesis.

        Repeated inference over one frame and one claim over many geometry points each
        count once, so the result is safe to use as an independent-evidence count.

        Args:
            hypothesis_id: A hypothesis of this fused evidence.

        Returns:
            The physical observation identities, sorted.

        Raises:
            KeyError: If the hypothesis does not belong to this fused evidence.
        """
        hypothesis = next(
            (item for item in self.hypotheses if item.hypothesis_id == hypothesis_id), None
        )
        if hypothesis is None:
            raise KeyError(hypothesis_id)
        contributions = {item.contribution_id: item for item in self.contributions}
        return tuple(
            sorted(
                {
                    contributions[item.contribution_id].physical_observation_id
                    for item in hypothesis.evidence
                    if item.stance is EvidenceStance.SUPPORTING
                }
            )
        )

    def _require_weighting_matches(self) -> None:
        weighting = self.weighting
        if weighting is None:
            return
        if EvidenceChannel.OBSERVATION_QUALITY not in {item.channel for item in self.channels}:
            raise ValueError(
                "a quality weighting needs the observation_quality channel to be declared"
            )
        if [item.contribution_id for item in weighting.contributions] != [
            item.contribution_id for item in self.contributions
        ]:
            raise ValueError("the weighting must cover exactly the contributions of the evidence")
        if [item.hypothesis_id for item in weighting.hypotheses] != [
            item.hypothesis_id for item in self.hypotheses
        ]:
            raise ValueError("the weighting must cover exactly the hypotheses of the evidence")
        for support in weighting.hypotheses:
            expected = self.supporting_physical_observations(support.hypothesis_id)
            listed = tuple(item.physical_observation_id for item in support.observation_factors)
            if listed != expected:
                raise ValueError(
                    f"hypothesis {support.hypothesis_id!r} has supporting physical observations "
                    f"{list(expected)!r}, but its weighting lists {list(listed)!r}"
                )

    def _require_channels_match(self) -> None:
        active = {item.channel for item in self.channels}
        for required in (EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.GEOMETRY_SUPPORT):
            if required not in active:
                raise ValueError(
                    f"channel {required.value} must be declared: it is intrinsic to every "
                    f"fused evidence"
                )
        signals = [
            signal
            for hypothesis in self.hypotheses
            for item in hypothesis.evidence
            for signal in item.signals
        ]
        carried = {
            EvidenceChannel.SEMANTIC_SCORES: any(item.score_refs for item in self.contributions)
            or any(signal.kind is SupportSignalKind.SCORER_SUPPORT for signal in signals),
            EvidenceChannel.VISUAL_FEATURES: any(
                item.visual_feature_refs for item in self.contributions
            ),
            EvidenceChannel.OBSERVATION_QUALITY: any(
                item.observation_quality is not None for item in self.contributions
            ),
            EvidenceChannel.POINT_REPRESENTATION: bool(self.point_representation_refs),
        }
        for channel, has_data in carried.items():
            if has_data and channel not in active:
                raise ValueError(
                    f"data of channel {channel.value} is present, but the channel was not declared"
                )

    def _require_canonical_order(self) -> None:
        _require_canonical("channels", self.channels, lambda item: (item.channel.value,))
        _require_canonical(
            "contributions", self.contributions, lambda item: (item.contribution_id,)
        )
        _require_canonical(
            "physical_observation_groups",
            self.physical_observation_groups,
            lambda group: (group.physical_observation_id,),
        )
        _require_canonical("hypotheses", self.hypotheses, lambda item: (item.hypothesis_id,))
        _require_canonical(
            "point_representation_refs",
            self.point_representation_refs,
            lambda ref: (ref.run_id, ref.representation_id),
        )
        _require_canonical("uncertainty", self.uncertainty, _uncertainty_key)

    def _require_groups_match(
        self, contributions: dict[EvidenceContributionId, EvidenceContribution]
    ) -> None:
        contributed: set[SpatialObservationId] = set()
        observation_of_result: dict[PerceptionResultId, SourceObservationId] = {}
        for contribution in self.contributions:
            if contribution.spatial_observation_id in contributed:
                raise ValueError(
                    f"spatial observation {contribution.spatial_observation_id!r} appears in "
                    f"more than one contribution"
                )
            contributed.add(contribution.spatial_observation_id)
            known = observation_of_result.setdefault(
                contribution.perception_result_id, contribution.physical_observation_id
            )
            if known != contribution.physical_observation_id:
                raise ValueError(
                    f"perception result {contribution.perception_result_id!r} is claimed by "
                    f"physical observations {known!r} and "
                    f"{contribution.physical_observation_id!r}"
                )
        groups = {
            group.physical_observation_id: group for group in self.physical_observation_groups
        }
        for observation_id in sorted(
            {item.physical_observation_id for item in contributions.values()}
        ):
            if observation_id not in groups:
                raise ValueError(
                    f"contributions of physical observation {observation_id!r} are not in any "
                    f"physical observation group"
                )
        for group in self.physical_observation_groups:
            members = [
                item
                for item in self.contributions
                if item.physical_observation_id == group.physical_observation_id
            ]
            if not members:
                raise ValueError(
                    f"physical observation group {group.physical_observation_id!r} has no "
                    f"contribution"
                )
            expected = (
                tuple(sorted(item.spatial_observation_id for item in members)),
                tuple(sorted({item.perception_result_id for item in members})),
                tuple(sorted({item.perception_run_id for item in members})),
            )
            found = (
                group.spatial_observation_ids,
                group.perception_result_ids,
                group.perception_run_ids,
            )
            for label, want, got in zip(
                ("spatial observations", "perception results", "perception runs"),
                expected,
                found,
                strict=True,
            ):
                if want != got:
                    raise ValueError(
                        f"physical observation group {group.physical_observation_id!r} lists "
                        f"{label} {list(got)!r}, but its contributions have {list(want)!r}"
                    )

    def _require_temporal_summary(self) -> None:
        expected = _time_bounds_of(
            [group.acquisition_timestamp for group in self.physical_observation_groups],
            owner="the physical observations",
        )
        if self.temporal_summary != expected:
            raise ValueError(
                f"temporal_summary must span exactly the acquisition of the physical "
                f"observations, expected {expected!r}, got {self.temporal_summary!r}"
            )

    def _require_hypotheses_resolve(
        self, contributions: dict[EvidenceContributionId, EvidenceContribution]
    ) -> None:
        labels: dict[str, FusedHypothesisId] = {}
        for hypothesis in self.hypotheses:
            other = labels.setdefault(hypothesis.label, hypothesis.hypothesis_id)
            if other != hypothesis.hypothesis_id:
                raise ValueError(
                    f"hypothesis label {hypothesis.label!r} is used by both {other!r} and "
                    f"{hypothesis.hypothesis_id!r}"
                )
            for item in hypothesis.evidence:
                contribution = contributions.get(item.contribution_id)
                if contribution is None:
                    raise ValueError(
                        f"hypothesis {hypothesis.hypothesis_id!r} references unknown "
                        f"contribution {item.contribution_id!r}"
                    )
                if item.claim_id not in {ref.claim_id for ref in contribution.claim_refs}:
                    raise ValueError(
                        f"hypothesis {hypothesis.hypothesis_id!r} lists claim {item.claim_id!r}, "
                        f"which contribution {item.contribution_id!r} does not carry"
                    )
                scorers = {
                    _producer_key(ref.scorer)
                    for ref in contribution.score_refs
                    if ref.claim_id == item.claim_id
                }
                for signal in item.signals:
                    if (
                        signal.kind is SupportSignalKind.SCORER_SUPPORT
                        and _producer_key(signal.producer) not in scorers
                    ):
                        raise ValueError(
                            f"scorer signal of claim {item.claim_id!r} has no score reference "
                            f"in contribution {item.contribution_id!r}"
                        )

    def _require_uncertainty_resolves(
        self, contributions: dict[EvidenceContributionId, EvidenceContribution]
    ) -> None:
        hypothesis_ids = {item.hypothesis_id for item in self.hypotheses}
        for record in self.uncertainty:
            for hypothesis_id in record.hypothesis_ids:
                if hypothesis_id not in hypothesis_ids:
                    raise ValueError(f"uncertainty names unknown hypothesis {hypothesis_id!r}")
            observations: set[SourceObservationId] = set()
            for reference in record.evidence:
                contribution = contributions.get(reference.contribution_id)
                if contribution is None:
                    raise ValueError(
                        f"uncertainty references unknown contribution {reference.contribution_id!r}"
                    )
                if reference.claim_id is not None and reference.claim_id not in {
                    ref.claim_id for ref in contribution.claim_refs
                }:
                    raise ValueError(
                        f"uncertainty references unknown claim {reference.claim_id!r} of "
                        f"contribution {reference.contribution_id!r}"
                    )
                observations.add(contribution.physical_observation_id)
            if record.kind is UncertaintyKind.CONTRADICTION and len(observations) < 2:
                raise ValueError(
                    "a contradiction needs evidence from at least two distinct physical "
                    "observations"
                )
