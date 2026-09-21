"""Typed evidence of each comparison channel.

Two entities are compared through independent channels: geometry, semantics, appearance, time and,
optionally, a 3D point representation. Each channel keeps *its own* measurement, in its own units
and semantics, and never a shared score: a metric distance in meters, a cosine between visual
embeddings and a label relation cannot be added, averaged or thresholded together without a
documented and testable rule, so no contract here offers a combined number.

A channel is either **measured** or **unavailable**, never both and never neither. An unavailable
channel says why (missing evidence, incompatible embedding or representation spaces, a hard gate
that blocked the comparison) and carries no measurement and no findings: missing evidence is never
a zero, and never a vote for ``DISTINCT``. A channel that was not evaluated at all is simply absent
from the comparison (see :class:`~contextmap.entity_resolution.evidence.EntityMatchEvidence`).

A measured channel interprets its measurement through explicit, versioned rules and records each
outcome as a :class:`Finding`. The channel status follows the findings: a conflict inside the
channel is never diluted by the findings that support, so a materially conflicting channel reaches
the resolution policy as such.

Every measurement refers to the pair in canonical order: ``a`` is the entity whose reference sorts
first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from contextmap.entity_resolution._checks import (
    require_canonical,
    require_finite,
    require_fraction,
    require_non_negative,
    require_present,
)
from contextmap.geometric_mapping import GeometryReference, MapId
from contextmap.point_representation import PointRepresentationId, PointRepresentationRunId
from contextmap.semantic_mapping import AmbiguityState, EntityFeatureRef


class MatchChannel(Enum):
    """A source of evidence about whether two entities are the same object.

    The definition order is the canonical order of channels everywhere in a record.

    Attributes:
        GEOMETRY: Where the 3D support of the entities is and how much of it they share.
        SEMANTIC: Whether what the entities may be is compatible.
        APPEARANCE: Whether their visual features are alike, within one embedding space.
        TEMPORAL: When and from which physical observations they were seen.
        POINT_REPRESENTATION: Whether their local 3D structure is alike, within one
            representation space. Optional.
    """

    GEOMETRY = "geometry"
    SEMANTIC = "semantic"
    APPEARANCE = "appearance"
    TEMPORAL = "temporal"
    POINT_REPRESENTATION = "point_representation"


class EvidenceStatus(Enum):
    """What one channel says about the pair.

    Attributes:
        SUPPORTING: The channel's findings support the same-object hypothesis.
        CONFLICTING: At least one finding argues against it. Not proof of distinct identity.
        NEUTRAL: The channel was measured and neither supports nor argues against.
        UNAVAILABLE: The channel could not compare the pair. Not evidence against, and never
            interpreted as a zero.
    """

    SUPPORTING = "supporting"
    CONFLICTING = "conflicting"
    NEUTRAL = "neutral"
    UNAVAILABLE = "unavailable"


class UnavailableReason(Enum):
    """Why a channel could not compare the pair.

    Attributes:
        MISSING_EVIDENCE: One of the entities has nothing this channel could use.
        INCOMPATIBLE_DOMAIN: The sides live in different embedding spaces, representation spaces
            or clock domains, so a numerical comparison would be meaningless.
        BLOCKED_BY_GATE: A hard validity gate failed, so the comparison was never attempted.
        INSUFFICIENT_EVIDENCE: There is evidence, but too little for the policy to measure.
    """

    MISSING_EVIDENCE = "missing_evidence"
    INCOMPATIBLE_DOMAIN = "incompatible_domain"
    BLOCKED_BY_GATE = "blocked_by_gate"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, kw_only=True)
class ChannelPolicyRef:
    """The versioned policy and configuration a channel was measured under.

    Attributes:
        policy_id: Versioned identity of the comparison and interpretation rules.
        configuration_fingerprint: Hash of the thresholds and options of that execution.
    """

    policy_id: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "policy_id", "configuration_fingerprint")


@dataclass(frozen=True, kw_only=True)
class Unavailability:
    """Why a channel carries no measurement.

    Attributes:
        reason: The kind of obstacle.
        detail: A deterministic, human-readable explanation.
    """

    reason: UnavailableReason
    detail: str

    def __post_init__(self) -> None:
        """Require an explanation.

        Raises:
            ValueError: If the detail is empty.
        """
        require_present(self, "detail")


@dataclass(frozen=True, kw_only=True)
class Finding:
    """The outcome of one explicit rule of a channel.

    Attributes:
        rule_id: Versioned identity of the rule.
        status: What the rule says; never ``UNAVAILABLE``, which belongs to the whole channel.
        detail: A deterministic, human-readable explanation.
        metric: The measured quantity the rule looked at, when it thresholds one.
        observed: Its value; required exactly when ``metric`` is named.
        threshold: The threshold the rule applied to it, when it applies one.
    """

    rule_id: str
    status: EvidenceStatus
    detail: str
    metric: str | None = None
    observed: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate that the finding is explained and coherent.

        Raises:
            ValueError: If the rule or the detail is empty, the status is ``UNAVAILABLE``, a
                metric has no observed value, a value has no metric, or a value is not finite.
        """
        require_present(self, "rule_id", "detail")
        if self.status is EvidenceStatus.UNAVAILABLE:
            raise ValueError(
                "a finding cannot be unavailable: unavailability belongs to the channel"
            )
        if self.metric is None:
            if self.observed is not None or self.threshold is not None:
                raise ValueError("observed and threshold need a metric")
            return
        require_present(self, "metric")
        if self.observed is None:
            raise ValueError("a finding that names a metric must record the observed value")
        require_finite(self, "observed")
        if self.threshold is not None:
            require_finite(self, "threshold")


@dataclass(frozen=True, kw_only=True)
class ChannelEvidence:
    """What one channel measured about the pair, or why it could not.

    Concrete channels add a ``measurement`` field of their own type. This class owns the rules
    every channel shares.

    Attributes:
        policy: The policy and configuration the channel was measured under.
        findings: The outcome of each rule applied to the measurement, empty when unavailable.
        unavailable: Why the channel could not compare the pair; ``None`` when it was measured.
    """

    policy: ChannelPolicyRef
    findings: tuple[Finding, ...] = ()
    unavailable: Unavailability | None = None

    channel: ClassVar[MatchChannel]

    def __post_init__(self) -> None:
        """Validate that the channel is measured or unavailable, exactly one of them.

        Raises:
            ValueError: If the channel has both a measurement and an unavailability or neither,
                or an unavailable channel reports findings.
        """
        measured = getattr(self, "measurement", None) is not None
        if measured == (self.unavailable is not None):
            raise ValueError(
                "a channel carries either a measurement or an unavailability, never both or neither"
            )
        if self.unavailable is not None and self.findings:
            raise ValueError("an unavailable channel reports no findings")

    @property
    def status(self) -> EvidenceStatus:
        """What the channel says: a conflict dominates, then support, otherwise neutral."""
        if self.unavailable is not None:
            return EvidenceStatus.UNAVAILABLE
        statuses = {finding.status for finding in self.findings}
        if EvidenceStatus.CONFLICTING in statuses:
            return EvidenceStatus.CONFLICTING
        if EvidenceStatus.SUPPORTING in statuses:
            return EvidenceStatus.SUPPORTING
        return EvidenceStatus.NEUTRAL


# --- geometry -----------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SupportDistance:
    """Distances between the points of the two supports, when their geometry was resolved.

    Attributes:
        a_to_b_mean_m: Mean distance from each point of ``a`` to its nearest point of ``b``.
        b_to_a_mean_m: The same from ``b`` to ``a``.
        hausdorff_m: The largest of the nearest-point distances in either direction.
        points_a: Points of ``a`` used.
        points_b: Points of ``b`` used.
    """

    a_to_b_mean_m: float
    b_to_a_mean_m: float
    hausdorff_m: float
    points_a: int
    points_b: int

    def __post_init__(self) -> None:
        """Validate that the statistics are coherent with each other.

        Raises:
            ValueError: If a distance is not finite and non-negative, the Hausdorff distance is
                below a mean, or a point count is not positive.
        """
        require_non_negative(self, "a_to_b_mean_m", "b_to_a_mean_m", "hausdorff_m")
        if self.hausdorff_m < max(self.a_to_b_mean_m, self.b_to_a_mean_m):
            raise ValueError("hausdorff_m cannot be lower than the mean nearest-point distances")
        if self.points_a < 1 or self.points_b < 1:
            raise ValueError("points_a and points_b must be at least 1")


@dataclass(frozen=True, kw_only=True)
class GeometryMeasurement:
    """Geometric comparison of two entities, in one common map frame.

    Attributes:
        geometric_map_id: The geometric map both supports belong to.
        map_frame: The frame every metric is expressed in.
        centroid_distance_m: Distance between the centroids, in meters.
        bounds_gap_m: Distance between the two axis-aligned boxes, in meters; ``0`` when they
            overlap or touch.
        bounds_iou: Intersection over union of the two boxes' volumes; ``None`` when the union has
            no volume (flat boxes), which is not a zero overlap.
        bounds_overlap_fraction_a: Share of the volume of ``a``'s box inside ``b``'s; ``None`` when
            that box is flat.
        bounds_overlap_fraction_b: The same for ``b``.
        support_count_a: Geometry elements supporting ``a``.
        support_count_b: Geometry elements supporting ``b``.
        shared_support_count: Geometry elements in both supports.
        support_jaccard: ``shared / (a + b - shared)``.
        extent_ratio: Smallest ratio, over the three axes, between the smaller and the larger
            extent of the two boxes, in ``(0, 1]``; ``None`` when an extent is flat in one entity
            and not in the other.
        support_distance: Nearest-point statistics between the supports; ``None`` when the
            policy did not resolve the geometry.
        orientation_angle_rad: Angle between the principal axes of the two supports, in
            ``[0, pi / 2]``; ``None`` when either orientation is not defined.
        caveats: Diagnostics of the entity summaries that qualify these figures, prefixed by side
            (``a:sparse_support``), sorted and unique.
    """

    geometric_map_id: MapId
    map_frame: str
    centroid_distance_m: float
    bounds_gap_m: float
    bounds_iou: float | None
    bounds_overlap_fraction_a: float | None
    bounds_overlap_fraction_b: float | None
    support_count_a: int
    support_count_b: int
    shared_support_count: int
    support_jaccard: float
    extent_ratio: float | None
    support_distance: SupportDistance | None
    orientation_angle_rad: float | None
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate ranges and that the support figures agree with each other.

        Raises:
            ValueError: If an identity is empty, a figure is out of its range, the shared support
                exceeds either support, the Jaccard index does not match the counts, or the
                caveats are not sorted and unique.
        """
        require_present(self, "geometric_map_id", "map_frame")
        require_non_negative(self, "centroid_distance_m", "bounds_gap_m")
        require_fraction(
            self,
            "bounds_iou",
            "bounds_overlap_fraction_a",
            "bounds_overlap_fraction_b",
            "support_jaccard",
        )
        if self.extent_ratio is not None and not (
            math.isfinite(self.extent_ratio) and 0.0 < self.extent_ratio <= 1.0
        ):
            raise ValueError(f"extent_ratio must be within (0, 1], got {self.extent_ratio!r}")
        if self.support_count_a < 1 or self.support_count_b < 1:
            raise ValueError("support_count_a and support_count_b must be at least 1")
        if not 0 <= self.shared_support_count <= min(self.support_count_a, self.support_count_b):
            raise ValueError("shared_support_count cannot exceed either support")
        union = self.support_count_a + self.support_count_b - self.shared_support_count
        if not math.isclose(self.support_jaccard, self.shared_support_count / union, abs_tol=1e-9):
            raise ValueError("support_jaccard must equal shared / (a + b - shared)")
        if self.orientation_angle_rad is not None and not (
            math.isfinite(self.orientation_angle_rad)
            and 0.0 <= self.orientation_angle_rad <= math.pi / 2 + 1e-9
        ):
            raise ValueError(
                f"orientation_angle_rad must be within [0, pi / 2], got "
                f"{self.orientation_angle_rad!r}"
            )
        require_canonical("caveats", self.caveats, lambda item: (item,))


@dataclass(frozen=True, kw_only=True)
class GeometryEvidence(ChannelEvidence):
    """Geometric evidence about the pair.

    Attributes:
        measurement: The geometric comparison; ``None`` when the channel is unavailable.
    """

    channel: ClassVar[MatchChannel] = MatchChannel.GEOMETRY
    measurement: GeometryMeasurement | None = None


# --- semantics ----------------------------------------------------------------------------------


class LabelRelation(Enum):
    """How two hypothesis labels relate under an explicit, versioned rule.

    Attributes:
        SAME: Equal after conservative normalization.
        REFINEMENT: One label refines the other (``wooden pallet`` refines ``pallet``).
        RELATED: The policy explicitly declares the two related; never inferred.
        DIFFERENT: No declared relation. Different open-vocabulary labels are not proof of
            different objects.
    """

    SAME = "same"
    REFINEMENT = "refinement"
    RELATED = "related"
    DIFFERENT = "different"


@dataclass(frozen=True, kw_only=True)
class LabelComparison:
    """The relation between one hypothesis label of each entity.

    Attributes:
        label_a: A hypothesis label of ``a``, verbatim.
        label_b: A hypothesis label of ``b``, verbatim.
        primary_a: Whether ``label_a`` is the primary hypothesis of ``a``.
        primary_b: Whether ``label_b`` is the primary hypothesis of ``b``.
        relation: How the labels relate.
        rule_id: Versioned rule that decided the relation.
    """

    label_a: str
    label_b: str
    primary_a: bool
    primary_b: bool
    relation: LabelRelation
    rule_id: str

    def __post_init__(self) -> None:
        """Require both labels and the rule.

        Raises:
            ValueError: If a label or the rule is empty.
        """
        require_present(self, "label_a", "label_b", "rule_id")


@dataclass(frozen=True, kw_only=True)
class AttributeComparison:
    """The values of one attribute name on each entity.

    Attributes:
        name: The attribute, for example ``material``.
        value_a: Its value on ``a``, verbatim.
        value_b: Its value on ``b``, verbatim.
        same: Whether the values are equal.
    """

    name: str
    value_a: str
    value_b: str
    same: bool

    def __post_init__(self) -> None:
        """Validate the name and that ``same`` agrees with the values.

        Raises:
            ValueError: If the name is empty or ``same`` does not match the values.
        """
        require_present(self, "name")
        if self.same != (self.value_a == self.value_b):
            raise ValueError("same must agree with the two values")


@dataclass(frozen=True, kw_only=True)
class SemanticMeasurement:
    """Semantic comparison of two entities, keeping every hypothesis instead of one label.

    Attributes:
        ambiguity_a: How settled the semantic state of ``a`` is.
        ambiguity_b: How settled the semantic state of ``b`` is.
        hypothesis_count_a: Hypotheses of ``a``.
        hypothesis_count_b: Hypotheses of ``b``.
        label_comparisons: The relation of each compared label pair, sorted by labels and unique.
        attribute_comparisons: The values of the attribute names both entities have, sorted by
            name and unique.
    """

    ambiguity_a: AmbiguityState
    ambiguity_b: AmbiguityState
    hypothesis_count_a: int
    hypothesis_count_b: int
    label_comparisons: tuple[LabelComparison, ...]
    attribute_comparisons: tuple[AttributeComparison, ...]

    def __post_init__(self) -> None:
        """Validate counts and canonical ordering.

        Raises:
            ValueError: If a count is negative or a collection is not sorted and unique.
        """
        if self.hypothesis_count_a < 0 or self.hypothesis_count_b < 0:
            raise ValueError("hypothesis counts must not be negative")
        require_canonical(
            "label_comparisons",
            self.label_comparisons,
            lambda item: (item.label_a, item.label_b),
        )
        require_canonical(
            "attribute_comparisons", self.attribute_comparisons, lambda item: (item.name,)
        )


@dataclass(frozen=True, kw_only=True)
class SemanticEvidence(ChannelEvidence):
    """Semantic compatibility evidence about the pair.

    Attributes:
        measurement: The semantic comparison; ``None`` when the channel is unavailable.
    """

    channel: ClassVar[MatchChannel] = MatchChannel.SEMANTIC
    measurement: SemanticMeasurement | None = None


# --- appearance ---------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FeatureContribution:
    """The visual features one physical observation contributed to an entity.

    Several features of one physical observation (repeated inference over the same frame) are
    correlated, never independent views, so they are grouped under that observation.

    Attributes:
        physical_observation_id: The physical frame the features were extracted from.
        feature_refs: The features, sorted by result and feature and unique; never empty.
    """

    physical_observation_id: str
    feature_refs: tuple[EntityFeatureRef, ...]

    def __post_init__(self) -> None:
        """Validate the identity and the features.

        Raises:
            ValueError: If the observation is empty or the features are empty, unsorted or
                repeated.
        """
        require_present(self, "physical_observation_id")
        if not self.feature_refs:
            raise ValueError("feature_refs must not be empty")
        require_canonical(
            "feature_refs",
            self.feature_refs,
            lambda ref: (ref.perception_result_id, ref.feature_id),
        )


@dataclass(frozen=True, kw_only=True)
class AppearanceMeasurement:
    """Visual comparison of two entities inside one embedding space.

    Attributes:
        embedding_space_id: The fingerprint of the embedding space every feature lives in;
            features of another space are never compared.
        metric: The similarity metric, for example ``cosine-similarity``.
        aggregation_id: Versioned policy that turned many features into one comparison.
        similarity: The similarity under ``metric`` and ``aggregation_id``. Not a probability and
            not comparable with any other channel.
        pair_similarity_min: Smallest similarity between one observation of ``a`` and one of ``b``.
        pair_similarity_max: Largest such similarity.
        contributions_a: The physical observations of ``a`` that contributed, sorted and unique;
            never empty.
        contributions_b: The same for ``b``.
    """

    embedding_space_id: str
    metric: str
    aggregation_id: str
    similarity: float
    pair_similarity_min: float
    pair_similarity_max: float
    contributions_a: tuple[FeatureContribution, ...]
    contributions_b: tuple[FeatureContribution, ...]

    def __post_init__(self) -> None:
        """Validate identities, figures and that every feature lives in the declared space.

        Raises:
            ValueError: If an identity is empty, a figure is not finite, the pair range is
                inverted, a side has no contribution or repeats an observation, or a feature
                belongs to another embedding space.
        """
        require_present(self, "embedding_space_id", "metric", "aggregation_id")
        require_finite(self, "similarity", "pair_similarity_min", "pair_similarity_max")
        if self.pair_similarity_min > self.pair_similarity_max:
            raise ValueError("pair_similarity_min cannot exceed pair_similarity_max")
        for name in ("contributions_a", "contributions_b"):
            contributions: tuple[FeatureContribution, ...] = getattr(self, name)
            if not contributions:
                raise ValueError(f"{name} must not be empty")
            require_canonical(name, contributions, lambda item: (item.physical_observation_id,))
            for contribution in contributions:
                for feature in contribution.feature_refs:
                    if feature.embedding_space_id != self.embedding_space_id:
                        raise ValueError(
                            f"feature {feature.feature_id!r} lives in embedding space "
                            f"{feature.embedding_space_id!r}, not {self.embedding_space_id!r}"
                        )


@dataclass(frozen=True, kw_only=True)
class AppearanceEvidence(ChannelEvidence):
    """Visual appearance evidence about the pair.

    Attributes:
        measurement: The visual comparison; ``None`` when the channel is unavailable.
    """

    channel: ClassVar[MatchChannel] = MatchChannel.APPEARANCE
    measurement: AppearanceMeasurement | None = None


# --- temporal -----------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TemporalMeasurement:
    """Temporal comparison of two entities from their observation histories.

    Physical observations (frames) and inference results are counted apart, as in Semantic Fusion:
    several inference runs over one frame are correlated, never independent views.

    Attributes:
        clock_id: The clock domain both histories share.
        interval_overlap_ns: Length of the overlap of the two observation intervals, in
            nanoseconds; ``0`` when they do not overlap.
        interval_gap_ns: Length of the gap between the intervals, in nanoseconds; ``0`` when they
            overlap or touch.
        physical_observation_count_a: Distinct physical observations of ``a``.
        physical_observation_count_b: The same for ``b``.
        inference_result_count_a: Inference results over those observations for ``a``.
        inference_result_count_b: The same for ``b``.
        shared_physical_observation_count: Physical observations that contributed to both.
        union_physical_observation_count: Distinct physical observations of the pair.
    """

    clock_id: str
    interval_overlap_ns: int
    interval_gap_ns: int
    physical_observation_count_a: int
    physical_observation_count_b: int
    inference_result_count_a: int
    inference_result_count_b: int
    shared_physical_observation_count: int
    union_physical_observation_count: int

    def __post_init__(self) -> None:
        """Validate that the intervals and the counts agree with each other.

        Raises:
            ValueError: If the clock is empty, an interval is negative or both an overlap and a
                gap are stated, a count is not positive, there are fewer inference results than
                physical observations, or the shared and union counts do not add up.
        """
        require_present(self, "clock_id")
        if self.interval_overlap_ns < 0 or self.interval_gap_ns < 0:
            raise ValueError("interval_overlap_ns and interval_gap_ns must not be negative")
        if self.interval_overlap_ns > 0 and self.interval_gap_ns > 0:
            raise ValueError("intervals cannot both overlap and have a gap")
        if self.physical_observation_count_a < 1 or self.physical_observation_count_b < 1:
            raise ValueError("physical observation counts must be at least 1")
        if (
            self.inference_result_count_a < self.physical_observation_count_a
            or self.inference_result_count_b < self.physical_observation_count_b
        ):
            raise ValueError("inference results cannot be fewer than physical observations")
        if (
            not 0
            <= self.shared_physical_observation_count
            <= min(self.physical_observation_count_a, self.physical_observation_count_b)
        ):
            raise ValueError("shared_physical_observation_count cannot exceed either side")
        expected_union = (
            self.physical_observation_count_a
            + self.physical_observation_count_b
            - self.shared_physical_observation_count
        )
        if self.union_physical_observation_count != expected_union:
            raise ValueError(
                "union_physical_observation_count must equal a + b - shared, "
                f"expected {expected_union}"
            )


@dataclass(frozen=True, kw_only=True)
class TemporalEvidence(ChannelEvidence):
    """Temporal compatibility evidence about the pair.

    Attributes:
        measurement: The temporal comparison; ``None`` when the channel is unavailable.
    """

    channel: ClassVar[MatchChannel] = MatchChannel.TEMPORAL
    measurement: TemporalMeasurement | None = None


# --- point representation -----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RepresentationRef:
    """A 3D point representation that contributed to an entity's comparison.

    Attributes:
        run_id: The Point Representation run that owns it.
        representation_id: The representation, local to its run.
        representation_space_id: Fingerprint of its representation space.
        geometry_reference: The geometry element it is anchored to.
    """

    run_id: PointRepresentationRunId
    representation_id: PointRepresentationId
    representation_space_id: str
    geometry_reference: GeometryReference

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "run_id", "representation_id", "representation_space_id")


@dataclass(frozen=True, kw_only=True)
class RepresentationMeasurement:
    """3D structural comparison of two entities inside one representation space.

    Attributes:
        representation_space_id: The fingerprint of the representation space every vector lives
            in; vectors of another space, or another dimension, are never compared.
        metric: The similarity metric, for example ``cosine-similarity``.
        aggregation_id: Versioned policy that turned many representations into one comparison.
        similarity: The similarity under ``metric`` and ``aggregation_id``. Not a probability and
            not comparable with any other channel.
        pair_similarity_min: Smallest similarity between one representation of ``a`` and one of
            ``b``.
        pair_similarity_max: Largest such similarity.
        representations_a: The representations of ``a`` that contributed, sorted and unique; never
            empty.
        representations_b: The same for ``b``.
    """

    representation_space_id: str
    metric: str
    aggregation_id: str
    similarity: float
    pair_similarity_min: float
    pair_similarity_max: float
    representations_a: tuple[RepresentationRef, ...]
    representations_b: tuple[RepresentationRef, ...]

    def __post_init__(self) -> None:
        """Validate identities, figures and that every representation lives in the declared space.

        Raises:
            ValueError: If an identity is empty, a figure is not finite, the pair range is
                inverted, a side has no representation, or one belongs to another space.
        """
        require_present(self, "representation_space_id", "metric", "aggregation_id")
        require_finite(self, "similarity", "pair_similarity_min", "pair_similarity_max")
        if self.pair_similarity_min > self.pair_similarity_max:
            raise ValueError("pair_similarity_min cannot exceed pair_similarity_max")
        for name in ("representations_a", "representations_b"):
            representations: tuple[RepresentationRef, ...] = getattr(self, name)
            if not representations:
                raise ValueError(f"{name} must not be empty")
            require_canonical(
                name, representations, lambda item: (item.run_id, item.representation_id)
            )
            for representation in representations:
                if representation.representation_space_id != self.representation_space_id:
                    raise ValueError(
                        f"representation {representation.representation_id!r} lives in "
                        f"representation space {representation.representation_space_id!r}, not "
                        f"{self.representation_space_id!r}"
                    )


@dataclass(frozen=True, kw_only=True)
class PointRepresentationEvidence(ChannelEvidence):
    """Optional 3D structural evidence about the pair.

    Attributes:
        measurement: The structural comparison; ``None`` when the channel is unavailable.
    """

    channel: ClassVar[MatchChannel] = MatchChannel.POINT_REPRESENTATION
    measurement: RepresentationMeasurement | None = None
