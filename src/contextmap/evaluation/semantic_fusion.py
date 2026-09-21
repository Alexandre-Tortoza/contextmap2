"""Deterministic, stratified evaluation of Semantic Fusion runs and policy ablations.

Semantic Fusion is validated as an evidence-accumulation stage, and its failures must stay
visible: this module reads persisted runs through the public reader and reports
*correlation handling* (repeated inference and geometry never multiply evidence), *uncertainty
retention* (contradiction, ambiguity, near ties, abstention and unscored evidence survive),
*channel provenance*, the *quality weighting* of a quality-aware arm, and, where annotated,
whether the reference label was recovered. Every quantity is reported separately, with its
denominator, and is never merged into one score: there is no ranking, no winner and no
composite. Cost is reported apart from every quality measure.

Nothing here alters a run and ``debug/`` is never read. Strata come from an explicit
:class:`FusionStratificationProfile` with no defaults, and a factor that could not be measured
goes to an explicit ``unavailable`` stratum. A missing annotation is *not applicable*, never a
negative label. Observation quality is only a stratification factor: it is never treated as a
semantic confidence.

Several arms are compared only when everything but the fusion configuration is identical: the
same sequence, map, selected upstream runs, support policy, exact evidence base (supports,
contributions and physical observations), annotations and profile. That is what lets the
uniform baseline and a quality-aware policy, or different channel subsets, be measured on the
same physical observations and inference artifacts, and lets a change be attributed to a
condition instead of a global average.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Any

from contextmap.semantic_fusion import (
    EvidenceChannel,
    EvidenceStance,
    FusedEvidence,
    FusionOutcome,
    SemanticFusionRunReader,
    SupportSignalKind,
    UncertaintyKind,
    label_key,
)
from contextmap.sensor_association import ObservationQuality, SpatialObservationId

EVALUATOR_VERSION = "2"
"""Bumped whenever a metric's definition changes, so reports stay comparable."""

_TOLERANCE = 1e-9

_UNCERTAINTY_LEVELS = ("none", "ambiguity_or_near_tie", "contradiction", "insufficient_evidence")


class SemanticFusionEvaluationError(ValueError):
    """Raised when an evaluation invariant cannot be satisfied."""


class FusionArmRole(Enum):
    """What an arm is in a controlled comparison.

    Attributes:
        BASELINE_CONTROL: The uniform baseline; exactly one per comparison.
        QUALITY_AWARE: A quality-aware policy under evaluation.
        CHANNEL_ABLATION: The baseline with a different subset of evidence channels.
    """

    BASELINE_CONTROL = "baseline_control"
    QUALITY_AWARE = "quality_aware"
    CHANNEL_ABLATION = "channel_ablation"


@dataclass(frozen=True, kw_only=True)
class FusionStratificationProfile:
    """The band edges of every stratification. There are no defaults.

    Each tuple of edges must be strictly increasing and finite; ``n`` edges give ``n + 1``
    bands. Quality-based factors are the median over the observations of a support.

    Attributes:
        range_edges_m: Edges of the median depth of the support's observations, in meters.
        visible_share_edges: Edges of the median visible share.
        support_density_edges: Edges of the median associated points per mask pixel.
        border_distance_edges_px: Edges of the median smallest distance to the prepared image
            border, in pixels (valid-region and fisheye-border strata).
        physical_observation_edges: Edges of the number of distinct physical observations.
    """

    range_edges_m: tuple[float, ...]
    visible_share_edges: tuple[float, ...]
    support_density_edges: tuple[float, ...]
    border_distance_edges_px: tuple[float, ...]
    physical_observation_edges: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate every edge list.

        Raises:
            ValueError: If an edge is not finite or the edges are not strictly increasing.
        """
        for name in (
            "range_edges_m",
            "visible_share_edges",
            "support_density_edges",
            "border_distance_edges_px",
            "physical_observation_edges",
        ):
            edges = getattr(self, name)
            if not all(math.isfinite(edge) for edge in edges):
                raise ValueError(f"{name} must be finite, got {edges!r}")
            if any(low >= high for low, high in pairwise(edges)):
                raise ValueError(f"{name} must be strictly increasing, got {edges!r}")


@dataclass(frozen=True, kw_only=True)
class ReferenceAnnotation:
    """A reference label for one spatial observation.

    Attributes:
        spatial_observation_id: The annotated observation.
        label: The reference label; compared with hypotheses by label key.
    """

    spatial_observation_id: SpatialObservationId
    label: str

    def __post_init__(self) -> None:
        """Require a label.

        Raises:
            ValueError: If the label is empty.
        """
        if not label_key(self.label):
            raise ValueError("a reference annotation needs a label")


@dataclass(frozen=True, kw_only=True)
class FusionLineage:
    """Everything that identifies a fusion run, copied from its manifest.

    Attributes:
        run_id: The fusion run.
        run_index: Its monotonic index.
        sequence_name: The sequence it processed.
        sequence_artifact_id: The canonical sequence everything was built from.
        geometric_map_id: The geometric map of the geometry.
        association_run_ids: The selected Sensor Association runs.
        perception_run_ids: The selected perception runs.
        point_representation_run_ids: The Point Representation runs referenced.
        grouping_policy_id: The physical-observation grouping policy.
        support_policy_id: The support construction policy.
        support_configuration_fingerprint: Hash of the support configuration.
        fusion_policy_id: The fusion policy.
        fusion_configuration_fingerprint: Hash of the fusion configuration.
        code_version: Code revision of the run.
        schema_version: Run artifact schema version.
    """

    run_id: str
    run_index: int
    sequence_name: str
    sequence_artifact_id: str
    geometric_map_id: str
    association_run_ids: tuple[str, ...]
    perception_run_ids: tuple[str, ...]
    point_representation_run_ids: tuple[str, ...]
    grouping_policy_id: str | None
    support_policy_id: str | None
    support_configuration_fingerprint: str | None
    fusion_policy_id: str | None
    fusion_configuration_fingerprint: str | None
    code_version: str
    schema_version: str


@dataclass(frozen=True, kw_only=True)
class FusionCorrelationReport:
    """How correlated inference and shared geometry were handled.

    The last four counts are invariants that a valid run keeps at zero; they are reported so a
    regression cannot hide behind a stage-level metric.

    Attributes:
        supports_with_repeated_inference: Supports with more inference results than physical
            observations.
        max_inference_results_per_physical_observation: The most results over one frame.
        hypotheses_with_repeated_inference_support: Hypotheses supported by more inference
            results than physical observations.
        supporting_exceeds_physical: Hypotheses that count more supporting physical
            observations than the support has.
        duplicated_evidence_items: Hypotheses that list the same claim twice.
        evidence_items_exceeding_claims: Hypotheses that list more evidence than the support
            has claims (geometry multiplying evidence).
        duplicated_structure_refs: Structural references listed more than once in a support.
    """

    supports_with_repeated_inference: int
    max_inference_results_per_physical_observation: int
    hypotheses_with_repeated_inference_support: int
    supporting_exceeds_physical: int
    duplicated_evidence_items: int
    evidence_items_exceeding_claims: int
    duplicated_structure_refs: int


@dataclass(frozen=True, kw_only=True)
class FusionUncertaintyReport:
    """Whether uncertainty survived fusion, by kind and with its denominators.

    Attributes:
        supports_with_hypotheses: Supports with at least one hypothesis.
        supports_with_multiple_hypotheses: Supports with two or more.
        unreported_competition: Supports with competing hypotheses and neither a contradiction
            nor an ambiguity record; zero in a valid run.
        supports_by_kind: Supports carrying each kind of uncertainty record.
        abstaining_claims: Distinct claims that abstained.
        scored_claims: Distinct claims with a confidence, among those under some hypothesis.
        unscored_claims: Distinct claims without a confidence, among those under a hypothesis.
        claims_without_hypothesis: Claims that appear under no hypothesis (abstentions only).
    """

    supports_with_hypotheses: int
    supports_with_multiple_hypotheses: int
    unreported_competition: int
    supports_by_kind: Mapping[str, int]
    abstaining_claims: int
    scored_claims: int
    unscored_claims: int
    claims_without_hypothesis: int


@dataclass(frozen=True, kw_only=True)
class FusionChannelReport:
    """One evidence channel: whether it was active, what fed it and how much.

    Attributes:
        channel: The channel.
        active: Whether the policy declared it.
        identities: What fed it (interpreters, scorers, spaces, versions, map).
        data_items: How many items of it the run holds.
    """

    channel: str
    active: bool
    identities: tuple[str, ...]
    data_items: int


@dataclass(frozen=True, kw_only=True)
class FusionValueSummary:
    """Order statistics of a measured quantity.

    Attributes:
        count: Number of values.
        minimum: Smallest value.
        median: Median value.
        maximum: Largest value.
    """

    count: int
    minimum: float
    median: float
    maximum: float


@dataclass(frozen=True, kw_only=True)
class FusionWeightingReport:
    """What a quality-aware policy did to the weight of the evidence.

    Attributes:
        contributions_weighted: Contributions that got a factor.
        zero_factor_contributions: Of those, the ones whose factor is exactly zero.
        neutral_fallback_components: Components that fell back to the neutral factor, counted
            per component; empty when every component was measured.
        factor: Distribution of the contribution factors, ``None`` without contributions.
        supports_where_leading_changes: Supports whose leading hypotheses differ before and
            after weighting.
    """

    contributions_weighted: int
    zero_factor_contributions: int
    neutral_fallback_components: Mapping[str, int]
    factor: FusionValueSummary | None
    supports_where_leading_changes: int


@dataclass(frozen=True, kw_only=True)
class FusionReferenceStatus:
    """How the annotated supports stand against their reference, for one notion of leader.

    Every annotated support is in exactly one of the four outcomes.

    Attributes:
        leading_matches_reference: The reference hypothesis is the single leader.
        tied_with_reference: It is one of several tied leaders.
        retained_not_leading: It is present but not among the leaders.
        reference_missing: No hypothesis carries the reference label.
    """

    leading_matches_reference: int
    tied_with_reference: int
    retained_not_leading: int
    reference_missing: int


@dataclass(frozen=True, kw_only=True)
class FusionAnnotationReport:
    """Reference recovery where annotations exist; everything else is not applicable.

    Attributes:
        annotated_supports: Supports with one consistent reference label.
        unannotated_supports: Supports with no annotation: not applicable, never a negative.
        ambiguous_reference: Supports whose annotations disagree, left out of correctness.
        unmatched_annotations: Annotations of observations that are in no support.
        reference_recovered: Annotated supports with a hypothesis carrying the reference.
        reference_missing: Annotated supports with no such hypothesis.
        leading_matches_reference: Leaders by distinct supporting physical observations.
        tied_with_reference: The same, when the reference is among several tied leaders.
        retained_not_leading: The reference is present but not a leader.
        weighted: The same outcomes with leaders by weighted support, for a weighted arm.
    """

    annotated_supports: int
    unannotated_supports: int
    ambiguous_reference: int
    unmatched_annotations: int
    reference_recovered: int
    reference_missing: int
    leading_matches_reference: int
    tied_with_reference: int
    retained_not_leading: int
    weighted: FusionReferenceStatus | None


@dataclass(frozen=True, kw_only=True)
class FusionStratumReport:
    """The supports that fall in one band of one stratification.

    Attributes:
        band: The band, e.g. ``"[5.0, 15.0)"``, a category, or ``"unavailable"``.
        supports: Supports in the band.
        with_uncertainty: Of those, the ones carrying an uncertainty record.
        annotated: Of those, the ones with a consistent reference.
        reference_recovered: Of the annotated, the ones whose reference is a hypothesis.
        leading_matches_reference: Of the annotated, the ones with the reference as the
            single leader by supporting physical observations.
        weighted_leading_matches_reference: The same by weighted support; ``None`` for an
            arm without weighting.
    """

    band: str
    supports: int
    with_uncertainty: int
    annotated: int
    reference_recovered: int
    leading_matches_reference: int
    weighted_leading_matches_reference: int | None


@dataclass(frozen=True, kw_only=True)
class FusionStratification:
    """Supports partitioned by one factor.

    The bands and the ``unavailable`` stratum together hold every support exactly once.

    Attributes:
        dimension: What is stratified, e.g. ``"range"``.
        source: Where the factor comes from.
        unit: The unit of the values and the edges, when it has one.
        edges: The band edges used.
        strata: One report per band, in order.
        unavailable: Supports whose factor could not be measured.
    """

    dimension: str
    source: str
    unit: str
    edges: tuple[float, ...]
    strata: tuple[FusionStratumReport, ...]
    unavailable: FusionStratumReport


@dataclass(frozen=True, kw_only=True)
class FusionCostReport:
    """Cost, kept apart from every quality measure.

    Attributes:
        runtime: Time and memory figures, only when the run's author measured them.
        payload_bytes: Size of each contractual file.
        total_payload_bytes: Their sum.
    """

    runtime: Mapping[str, float | int | None] | None
    payload_bytes: Mapping[str, int]
    total_payload_bytes: int


@dataclass(frozen=True, kw_only=True)
class FusionSupportLeaders:
    """The leading hypotheses of one support, before and after weighting.

    Attributes:
        support_id: The support.
        unweighted: Labels tied for the most distinct supporting physical observations.
        weighted: Labels tied for the most weighted support; ``None`` without weighting.
    """

    support_id: str
    unweighted: tuple[str, ...]
    weighted: tuple[str, ...] | None


@dataclass(frozen=True, kw_only=True)
class SemanticFusionEvaluationReport:
    """The evaluation of one fusion arm, in separate sections.

    Attributes:
        evaluator_version: The version of the metric definitions.
        arm_id: The arm's name in a comparison.
        arm_role: What the arm is.
        lineage: The complete lineage of the run.
        profile: The stratification profile used.
        support_count: Supports in the run.
        physical_observation_count: Distinct physical observations in the run.
        inference_result_count: Distinct inference results in the run: a result reaches every
            support that one of its regions falls in, and is counted once, like the physical
            observations.
        evidence_base_id: Hash of the supports, contributions and physical observations.
        hypothesis_labels_id: Hash of the hypotheses' labels per support.
        hypothesis_stances_id: Hash of every claim's stance under every hypothesis.
        annotations_id: Hash of the annotations used, ``None`` without annotations.
        leaders: The leading hypotheses of each support.
        correlation: Correlation handling.
        uncertainty: Uncertainty retention.
        channels: Every evidence channel.
        weighting: The weighting, for a quality-aware arm.
        annotations: Reference recovery, when annotations were given.
        strata: The stratified views.
        cost: Cost.
    """

    evaluator_version: str
    arm_id: str
    arm_role: FusionArmRole
    lineage: FusionLineage
    profile: FusionStratificationProfile
    support_count: int
    physical_observation_count: int
    inference_result_count: int
    evidence_base_id: str
    hypothesis_labels_id: str
    hypothesis_stances_id: str
    annotations_id: str | None
    leaders: tuple[FusionSupportLeaders, ...]
    correlation: FusionCorrelationReport
    uncertainty: FusionUncertaintyReport
    channels: tuple[FusionChannelReport, ...]
    weighting: FusionWeightingReport | None
    annotations: FusionAnnotationReport | None
    strata: tuple[FusionStratification, ...]
    cost: FusionCostReport


@dataclass(frozen=True, kw_only=True)
class FusionComparisonEntry:
    """One arm of a comparison, next to the control.

    Attributes:
        arm_id: The arm.
        arm_role: What it is.
        run_id: Its fusion run.
        fusion_policy_id: Its fusion policy.
        fusion_configuration_fingerprint: Its configuration hash.
        active_channels: Its declared channels.
        hypotheses_match_control: Whether it has exactly the control's hypotheses.
        stances_match_control: Whether every claim has the same stance under every hypothesis.
        supports_with_changed_leader: Supports whose leaders differ from the control's
            unweighted leaders.
    """

    arm_id: str
    arm_role: FusionArmRole
    run_id: str
    fusion_policy_id: str | None
    fusion_configuration_fingerprint: str | None
    active_channels: tuple[str, ...]
    hypotheses_match_control: bool
    stances_match_control: bool
    supports_with_changed_leader: int


@dataclass(frozen=True, kw_only=True)
class SemanticFusionComparison:
    """Arms compared on identical evidence, with no winner.

    Attributes:
        lineage: The lineage every arm shares (of the control).
        control_arm_id: The baseline control arm.
        entries: One entry per arm, in the order given.
    """

    lineage: FusionLineage
    control_arm_id: str
    entries: tuple[FusionComparisonEntry, ...]


@dataclass
class _Facts:
    """What the evaluation needs to know about one support."""

    support_id: str
    physical: int
    labels: tuple[str, ...]
    leaders: tuple[str, ...]
    weighted_leaders: tuple[str, ...] | None
    kinds: frozenset[str]
    level: str
    reference: str | None
    reference_ambiguous: bool
    quality: dict[str, float | None]


def evaluate_semantic_fusion(
    reader: SemanticFusionRunReader,
    *,
    arm_id: str,
    arm_role: FusionArmRole,
    profile: FusionStratificationProfile,
    annotations: Iterable[ReferenceAnnotation] | None = None,
    qualities: Mapping[SpatialObservationId, ObservationQuality] | None = None,
) -> SemanticFusionEvaluationReport:
    """Evaluate a persisted Semantic Fusion run.

    Args:
        reader: The run, opened through its public reader.
        arm_id: The name of the arm, unique in a comparison.
        arm_role: What the arm is.
        profile: The band edges of the stratifications.
        annotations: Reference labels per spatial observation, if any exist. Without them the
            annotation section is absent; with them, unannotated supports are not applicable.
        qualities: The measured quality of the observations, used only to stratify. Without
            it the quality-based strata are ``unavailable``.

    Returns:
        The report, with the run's complete lineage.

    Raises:
        SemanticFusionEvaluationError: If the run fails its own integrity check.
    """
    problems = reader.verify_integrity()
    if problems:
        raise SemanticFusionEvaluationError(f"the run is not intact: {problems}")
    annotation_list = None if annotations is None else list(annotations)
    reference_of = _reference_index(annotation_list)
    used_annotations: set[SpatialObservationId] = set()

    tally = _Tally()
    for outcome in reader.iter_outcomes():
        facts = _facts(outcome, reference_of, qualities, used_annotations)
        tally.add(outcome, facts)

    manifest = reader.manifest
    annotation_report = (
        None
        if annotation_list is None
        else tally.annotation_report(
            unmatched=sum(
                1 for item in annotation_list if item.spatial_observation_id not in used_annotations
            )
        )
    )
    return SemanticFusionEvaluationReport(
        evaluator_version=EVALUATOR_VERSION,
        arm_id=arm_id,
        arm_role=arm_role,
        lineage=FusionLineage(
            run_id=str(manifest.run_id),
            run_index=manifest.run_index,
            sequence_name=manifest.sequence_name,
            sequence_artifact_id=manifest.lineage.sequence_artifact_id,
            geometric_map_id=str(manifest.lineage.geometric_map_id),
            association_run_ids=manifest.lineage.association_run_ids,
            perception_run_ids=tuple(str(item) for item in manifest.lineage.perception_run_ids),
            point_representation_run_ids=tuple(
                str(item) for item in manifest.lineage.point_representation_run_ids
            ),
            grouping_policy_id=manifest.grouping_policy_id,
            support_policy_id=manifest.support_policy_id,
            support_configuration_fingerprint=manifest.support_configuration_fingerprint,
            fusion_policy_id=manifest.fusion_policy_id,
            fusion_configuration_fingerprint=manifest.fusion_configuration_fingerprint,
            code_version=manifest.code_version,
            schema_version=manifest.schema_version,
        ),
        profile=profile,
        support_count=len(tally.facts),
        physical_observation_count=len(tally.physical_observations),
        inference_result_count=len(tally.inference_results),
        evidence_base_id=tally.evidence_base_id(),
        hypothesis_labels_id=_digest(tally.label_records),
        hypothesis_stances_id=_digest(tally.stance_records),
        annotations_id=None if annotation_list is None else _annotations_id(reference_of),
        leaders=tuple(
            FusionSupportLeaders(
                support_id=item.support_id,
                unweighted=item.leaders,
                weighted=item.weighted_leaders,
            )
            for item in tally.facts
        ),
        correlation=tally.correlation_report(),
        uncertainty=tally.uncertainty_report(),
        channels=tally.channel_reports(),
        weighting=tally.weighting_report(),
        annotations=annotation_report,
        strata=tally.strata(profile, annotated=annotation_list is not None),
        cost=_cost(reader),
    )


def compare_semantic_fusion_reports(
    reports: Sequence[SemanticFusionEvaluationReport],
) -> SemanticFusionComparison:
    """Compare arms under one control, rejecting anything but the fusion configuration changing.

    Args:
        reports: One report per arm, exactly one of them the baseline control.

    Returns:
        The comparison, preserving each arm's identity, configuration and channels. It has no
        score and no winner: the differences are shown by condition in the reports.

    Raises:
        SemanticFusionEvaluationError: If there are fewer than two reports, an ``arm_id``
            repeats, there is not exactly one baseline control, or the evaluator version,
            stratification profile, annotations, evidence base, sequence, map, selected
            upstream runs or support policy differ.
    """
    if len(reports) < 2:
        raise SemanticFusionEvaluationError("a comparison needs at least two reports")
    ids = [report.arm_id for report in reports]
    if len(set(ids)) != len(ids):
        raise SemanticFusionEvaluationError(f"arm_id must be unique in a comparison, got {ids!r}")
    controls = [r for r in reports if r.arm_role is FusionArmRole.BASELINE_CONTROL]
    if len(controls) != 1:
        raise SemanticFusionEvaluationError(
            f"a comparison needs exactly one baseline control, got {len(controls)}"
        )
    control = controls[0]
    for report in reports:
        if report.evaluator_version != control.evaluator_version:
            raise SemanticFusionEvaluationError("reports use different evaluator versions")
        if report.profile != control.profile:
            raise SemanticFusionEvaluationError("reports use different stratification profiles")
        if report.annotations_id != control.annotations_id:
            raise SemanticFusionEvaluationError("reports use different annotations")
        if report.evidence_base_id != control.evidence_base_id:
            raise SemanticFusionEvaluationError(
                "reports do not share one evidence base: the supports, contributions or "
                "physical observations differ"
            )
        drift = _lineage_drift(control.lineage, report.lineage)
        if drift:
            raise SemanticFusionEvaluationError(
                f"reports differ in something other than the fusion configuration: {drift}"
            )
    control_leaders = {item.support_id: item.unweighted for item in control.leaders}
    return SemanticFusionComparison(
        lineage=control.lineage,
        control_arm_id=control.arm_id,
        entries=tuple(
            FusionComparisonEntry(
                arm_id=report.arm_id,
                arm_role=report.arm_role,
                run_id=report.lineage.run_id,
                fusion_policy_id=report.lineage.fusion_policy_id,
                fusion_configuration_fingerprint=report.lineage.fusion_configuration_fingerprint,
                active_channels=tuple(c.channel for c in report.channels if c.active),
                hypotheses_match_control=report.hypothesis_labels_id
                == control.hypothesis_labels_id,
                stances_match_control=report.hypothesis_stances_id == control.hypothesis_stances_id,
                supports_with_changed_leader=sum(
                    1
                    for item in report.leaders
                    if (item.weighted if item.weighted is not None else item.unweighted)
                    != control_leaders[item.support_id]
                ),
            )
            for report in reports
        ),
    )


def encode_semantic_fusion_report(report: SemanticFusionEvaluationReport) -> dict[str, Any]:
    """Return the report as JSON primitives, with every identity."""
    return {
        "evaluator_version": report.evaluator_version,
        "arm_id": report.arm_id,
        "arm_role": report.arm_role.value,
        "lineage": _encode_lineage(report.lineage),
        "profile": {
            "range_edges_m": list(report.profile.range_edges_m),
            "visible_share_edges": list(report.profile.visible_share_edges),
            "support_density_edges": list(report.profile.support_density_edges),
            "border_distance_edges_px": list(report.profile.border_distance_edges_px),
            "physical_observation_edges": list(report.profile.physical_observation_edges),
        },
        "support_count": report.support_count,
        "physical_observation_count": report.physical_observation_count,
        "inference_result_count": report.inference_result_count,
        "evidence_base_id": report.evidence_base_id,
        "hypothesis_labels_id": report.hypothesis_labels_id,
        "hypothesis_stances_id": report.hypothesis_stances_id,
        "annotations_id": report.annotations_id,
        "leaders": [
            {
                "support_id": item.support_id,
                "unweighted": list(item.unweighted),
                "weighted": None if item.weighted is None else list(item.weighted),
            }
            for item in report.leaders
        ],
        "correlation": _plain(report.correlation),
        "uncertainty": {
            **_plain(report.uncertainty),
            "supports_by_kind": dict(sorted(report.uncertainty.supports_by_kind.items())),
        },
        "channels": [
            {
                "channel": item.channel,
                "active": item.active,
                "identities": list(item.identities),
                "data_items": item.data_items,
            }
            for item in report.channels
        ],
        "weighting": None
        if report.weighting is None
        else {
            **_plain(report.weighting),
            "neutral_fallback_components": dict(
                sorted(report.weighting.neutral_fallback_components.items())
            ),
            "factor": None if report.weighting.factor is None else _plain(report.weighting.factor),
        },
        "annotations": None
        if report.annotations is None
        else {
            **_plain(report.annotations),
            "weighted": None
            if report.annotations.weighted is None
            else _plain(report.annotations.weighted),
        },
        "strata": [
            {
                "dimension": item.dimension,
                "source": item.source,
                "unit": item.unit,
                "edges": list(item.edges),
                "strata": [_plain(band) for band in item.strata],
                "unavailable": _plain(item.unavailable),
            }
            for item in report.strata
        ],
        "cost": {
            "runtime": None if report.cost.runtime is None else dict(report.cost.runtime),
            "payload_bytes": dict(sorted(report.cost.payload_bytes.items())),
            "total_payload_bytes": report.cost.total_payload_bytes,
        },
    }


def encode_semantic_fusion_comparison(comparison: SemanticFusionComparison) -> dict[str, Any]:
    """Return the comparison as JSON primitives, with every arm's identity."""
    return {
        "lineage": _encode_lineage(comparison.lineage),
        "control_arm_id": comparison.control_arm_id,
        "entries": [
            {
                "arm_id": entry.arm_id,
                "arm_role": entry.arm_role.value,
                "run_id": entry.run_id,
                "fusion_policy_id": entry.fusion_policy_id,
                "fusion_configuration_fingerprint": entry.fusion_configuration_fingerprint,
                "active_channels": list(entry.active_channels),
                "hypotheses_match_control": entry.hypotheses_match_control,
                "stances_match_control": entry.stances_match_control,
                "supports_with_changed_leader": entry.supports_with_changed_leader,
            }
            for entry in comparison.entries
        ],
    }


class _Tally:
    """Accumulates the sections of a report while the outcomes stream."""

    def __init__(self) -> None:
        self.facts: list[_Facts] = []
        self.physical_observations: set[str] = set()
        self.inference_results: set[str] = set()
        self.label_records: list[Any] = []
        self.stance_records: list[Any] = []
        self._base: list[Any] = []
        self._repeated = 0
        self._max_results = 0
        self._hyp_repeated = 0
        self._exceeds = 0
        self._duplicated = 0
        self._over_claims = 0
        self._dup_structure = 0
        self._with_hypotheses = 0
        self._multiple = 0
        self._unreported = 0
        self._kinds: Counter[str] = Counter()
        self._abstaining: set[tuple[str, str]] = set()
        self._claims_total = 0
        self._claims_seen: dict[tuple[str, str], float | None] = {}
        self._channels: dict[str, tuple[bool, set[str], int]] = {
            channel.value: (False, set(), 0) for channel in EvidenceChannel
        }
        self._weighted_contributions = 0
        self._zero_factor = 0
        self._factors: list[float] = []
        self._neutral: Counter[str] = Counter()
        self._has_weighting = False
        self._changes = 0

    def add(self, outcome: FusionOutcome, facts: _Facts) -> None:
        self.facts.append(facts)
        evidence = outcome.evidence
        self._base.append(
            (
                str(outcome.support.fusion_support_id),
                [str(i) for i in outcome.support.spatial_observation_ids],
                [str(c.contribution_id) for c in evidence.contributions],
                [
                    (str(g.physical_observation_id), [str(r) for r in g.perception_result_ids])
                    for g in evidence.physical_observation_groups
                ],
            )
        )
        self.physical_observations.update(
            str(g.physical_observation_id) for g in evidence.physical_observation_groups
        )
        self.inference_results.update(
            str(result)
            for g in evidence.physical_observation_groups
            for result in g.perception_result_ids
        )
        self._add_correlation(evidence)
        self._add_uncertainty(evidence, facts)
        self._add_channels(evidence)
        self._add_weighting(evidence, facts)
        for hypothesis in evidence.hypotheses:
            self.label_records.append((facts.support_id, hypothesis.label))
            for item in hypothesis.evidence:
                self.stance_records.append(
                    (
                        facts.support_id,
                        hypothesis.label,
                        str(item.contribution_id),
                        str(item.claim_id),
                        item.stance.value,
                    )
                )

    def _add_correlation(self, evidence: FusedEvidence) -> None:
        results = [len(g.perception_result_ids) for g in evidence.physical_observation_groups]
        if evidence.inference_result_count > evidence.physical_observation_count:
            self._repeated += 1
        self._max_results = max([self._max_results, *results])
        contributions = {item.contribution_id: item for item in evidence.contributions}
        total_claims = sum(len(item.claim_refs) for item in evidence.contributions)
        structure = [
            (ref.run_id, ref.representation_id) for ref in evidence.point_representation_refs
        ]
        self._dup_structure += len(structure) - len(set(structure))
        for hypothesis in evidence.hypotheses:
            supporting = [
                item for item in hypothesis.evidence if item.stance is EvidenceStance.SUPPORTING
            ]
            physical = {
                contributions[i.contribution_id].physical_observation_id for i in supporting
            }
            inference = {contributions[i.contribution_id].perception_result_id for i in supporting}
            if len(inference) > len(physical):
                self._hyp_repeated += 1
            if len(physical) > evidence.physical_observation_count:
                self._exceeds += 1
            keys = [(i.contribution_id, i.claim_id) for i in hypothesis.evidence]
            if len(keys) != len(set(keys)):
                self._duplicated += 1
            if len(hypothesis.evidence) > total_claims:
                self._over_claims += 1

    def _add_uncertainty(self, evidence: FusedEvidence, facts: _Facts) -> None:
        for kind in facts.kinds:
            self._kinds[kind] += 1
        if evidence.hypotheses:
            self._with_hypotheses += 1
        if len(evidence.hypotheses) >= 2:
            self._multiple += 1
            if not facts.kinds & {"contradiction", "ambiguity"}:
                self._unreported += 1
        self._claims_total += sum(len(item.claim_refs) for item in evidence.contributions)
        for hypothesis in evidence.hypotheses:
            for item in hypothesis.evidence:
                key = (str(item.contribution_id), str(item.claim_id))
                confidence = next(
                    (s.value for s in item.signals if s.kind is SupportSignalKind.CLAIM_CONFIDENCE),
                    None,
                )
                self._claims_seen.setdefault(key, confidence)
                if item.stance is EvidenceStance.ABSTAINING:
                    self._abstaining.add(key)
        for record in evidence.uncertainty:
            if record.kind is UncertaintyKind.INSUFFICIENT_EVIDENCE:
                self._abstaining.update(
                    (str(ref.contribution_id), str(ref.claim_id))
                    for ref in record.evidence
                    if ref.claim_id is not None
                )

    def _add_channels(self, evidence: FusedEvidence) -> None:
        items = {
            "semantic_claims": sum(len(c.claim_refs) for c in evidence.contributions),
            "semantic_scores": sum(len(c.score_refs) for c in evidence.contributions),
            "visual_features": sum(len(c.visual_feature_refs) for c in evidence.contributions),
            "observation_quality": sum(
                1 for c in evidence.contributions if c.observation_quality is not None
            ),
            "geometry_support": len(evidence.contributions),
            "point_representation": len(evidence.point_representation_refs),
        }
        for channel in evidence.channels:
            name = channel.channel.value
            _, identities, count = self._channels[name]
            self._channels[name] = (True, identities | set(channel.identities), count)
        for name, amount in items.items():
            active, identities, count = self._channels[name]
            self._channels[name] = (active, identities, count + amount)

    def _add_weighting(self, evidence: FusedEvidence, facts: _Facts) -> None:
        weighting = evidence.weighting
        if weighting is None:
            return
        self._has_weighting = True
        for item in weighting.contributions:
            self._weighted_contributions += 1
            self._factors.append(item.factor)
            if item.factor == 0.0:
                self._zero_factor += 1
            for component in item.components:
                if component.treatment.value == "neutral_fallback":
                    self._neutral[component.component] += 1
        if facts.weighted_leaders is not None and facts.weighted_leaders != facts.leaders:
            self._changes += 1

    def evidence_base_id(self) -> str:
        return _digest(self._base)

    def correlation_report(self) -> FusionCorrelationReport:
        return FusionCorrelationReport(
            supports_with_repeated_inference=self._repeated,
            max_inference_results_per_physical_observation=self._max_results,
            hypotheses_with_repeated_inference_support=self._hyp_repeated,
            supporting_exceeds_physical=self._exceeds,
            duplicated_evidence_items=self._duplicated,
            evidence_items_exceeding_claims=self._over_claims,
            duplicated_structure_refs=self._dup_structure,
        )

    def uncertainty_report(self) -> FusionUncertaintyReport:
        scored = sum(1 for value in self._claims_seen.values() if value is not None)
        return FusionUncertaintyReport(
            supports_with_hypotheses=self._with_hypotheses,
            supports_with_multiple_hypotheses=self._multiple,
            unreported_competition=self._unreported,
            supports_by_kind={
                kind.value: self._kinds.get(kind.value, 0)
                for kind in (
                    UncertaintyKind.AMBIGUITY,
                    UncertaintyKind.CONTRADICTION,
                    UncertaintyKind.INSUFFICIENT_EVIDENCE,
                    UncertaintyKind.NEAR_TIE,
                )
            },
            abstaining_claims=len(self._abstaining),
            scored_claims=scored,
            unscored_claims=len(self._claims_seen) - scored,
            claims_without_hypothesis=self._claims_total - len(self._claims_seen),
        )

    def channel_reports(self) -> tuple[FusionChannelReport, ...]:
        return tuple(
            FusionChannelReport(
                channel=name,
                active=active,
                identities=tuple(sorted(identities)),
                data_items=count,
            )
            for name, (active, identities, count) in sorted(self._channels.items())
        )

    def weighting_report(self) -> FusionWeightingReport | None:
        if not self._has_weighting:
            return None
        return FusionWeightingReport(
            contributions_weighted=self._weighted_contributions,
            zero_factor_contributions=self._zero_factor,
            neutral_fallback_components=dict(sorted(self._neutral.items())),
            factor=_summary(self._factors),
            supports_where_leading_changes=self._changes,
        )

    def annotation_report(self, *, unmatched: int) -> FusionAnnotationReport:
        annotated = [item for item in self.facts if item.reference is not None]
        weighted = self._has_weighting
        plain = Counter(_status(item, weighted=False) for item in annotated)
        heavy = Counter(_status(item, weighted=True) for item in annotated) if weighted else None
        recovered = len(annotated) - plain["missing"]
        return FusionAnnotationReport(
            annotated_supports=len(annotated),
            unannotated_supports=sum(
                1 for item in self.facts if item.reference is None and not item.reference_ambiguous
            ),
            ambiguous_reference=sum(1 for item in self.facts if item.reference_ambiguous),
            unmatched_annotations=unmatched,
            reference_recovered=recovered,
            reference_missing=plain["missing"],
            leading_matches_reference=plain["leading"],
            tied_with_reference=plain["tied"],
            retained_not_leading=plain["retained"],
            weighted=None
            if heavy is None
            else FusionReferenceStatus(
                leading_matches_reference=heavy["leading"],
                tied_with_reference=heavy["tied"],
                retained_not_leading=heavy["retained"],
                reference_missing=heavy["missing"],
            ),
        )

    def strata(
        self, profile: FusionStratificationProfile, *, annotated: bool
    ) -> tuple[FusionStratification, ...]:
        weighted = self._has_weighting
        return (
            _banded(
                "range",
                "observation_quality.support_depth",
                "m",
                profile.range_edges_m,
                self.facts,
                lambda f: f.quality["range"],
                weighted,
            ),
            _banded(
                "visibility",
                "observation_quality.visible_share",
                "ratio",
                profile.visible_share_edges,
                self.facts,
                lambda f: f.quality["visibility"],
                weighted,
            ),
            _banded(
                "support_density",
                "observation_quality.support_density_per_mask_pixel",
                "points_per_mask_pixel",
                profile.support_density_edges,
                self.facts,
                lambda f: f.quality["density"],
                weighted,
            ),
            _banded(
                "image_border",
                "observation_quality.border_distance",
                "px",
                profile.border_distance_edges_px,
                self.facts,
                lambda f: f.quality["border"],
                weighted,
            ),
            _banded(
                "physical_observations",
                "fused_evidence.physical_observation_groups",
                "count",
                profile.physical_observation_edges,
                self.facts,
                lambda f: float(f.physical),
                weighted,
            ),
            _categorical("uncertainty_level", "fused_evidence.uncertainty", self.facts, weighted),
        )


def _facts(
    outcome: FusionOutcome,
    reference_of: Mapping[SpatialObservationId, str],
    qualities: Mapping[SpatialObservationId, ObservationQuality] | None,
    used: set[SpatialObservationId],
) -> _Facts:
    evidence = outcome.evidence
    hypotheses = evidence.hypotheses
    counts = {
        h.hypothesis_id: len(evidence.supporting_physical_observations(h.hypothesis_id))
        for h in hypotheses
    }
    leaders = _leaders([(h.label, float(counts[h.hypothesis_id])) for h in hypotheses])
    weighted_leaders = None
    if evidence.weighting is not None:
        support_of = {s.hypothesis_id: s.weighted_support for s in evidence.weighting.hypotheses}
        weighted_leaders = _leaders([(h.label, support_of[h.hypothesis_id]) for h in hypotheses])
    kinds = frozenset(record.kind.value for record in evidence.uncertainty)
    labels = {reference_of[i] for i in outcome.support.spatial_observation_ids if i in reference_of}
    used.update(i for i in outcome.support.spatial_observation_ids if i in reference_of)
    return _Facts(
        support_id=str(outcome.support.fusion_support_id),
        physical=evidence.physical_observation_count,
        labels=tuple(sorted(h.label for h in hypotheses)),
        leaders=leaders,
        weighted_leaders=weighted_leaders,
        kinds=kinds,
        level=_level(kinds),
        reference=next(iter(labels)) if len(labels) == 1 else None,
        reference_ambiguous=len(labels) > 1,
        quality=_quality(outcome, qualities),
    )


def _leaders(candidates: Sequence[tuple[str, float]]) -> tuple[str, ...]:
    if not candidates:
        return ()
    best = max(value for _, value in candidates)
    return tuple(sorted(label for label, value in candidates if best - value <= _TOLERANCE))


def _level(kinds: frozenset[str]) -> str:
    if "insufficient_evidence" in kinds:
        return "insufficient_evidence"
    if "contradiction" in kinds:
        return "contradiction"
    if kinds & {"ambiguity", "near_tie"}:
        return "ambiguity_or_near_tie"
    return "none"


def _quality(
    outcome: FusionOutcome, qualities: Mapping[SpatialObservationId, ObservationQuality] | None
) -> dict[str, float | None]:
    values: dict[str, list[float]] = {"range": [], "visibility": [], "density": [], "border": []}
    for observation_id in outcome.support.spatial_observation_ids:
        quality = None if qualities is None else qualities.get(observation_id)
        if quality is None:
            continue
        if quality.support_depth_m is not None:
            values["range"].append(quality.support_depth_m.median)
        if quality.visible_share is not None:
            values["visibility"].append(quality.visible_share)
        values["density"].append(quality.support_density_per_mask_pixel)
        if quality.border_distance_px is not None:
            values["border"].append(quality.border_distance_px.minimum)
    return {name: statistics.median(found) if found else None for name, found in values.items()}


def _status(facts: _Facts, *, weighted: bool) -> str:
    reference = facts.reference
    leaders = (
        facts.weighted_leaders if weighted and facts.weighted_leaders is not None else facts.leaders
    )
    keys = {label_key(label) for label in facts.labels}
    if reference is None or reference not in keys:
        return "missing"
    leading = {label_key(label) for label in leaders}
    if leading == {reference}:
        return "leading"
    return "tied" if reference in leading else "retained"


def _banded(
    dimension: str,
    source: str,
    unit: str,
    edges: tuple[float, ...],
    facts: Sequence[_Facts],
    value_of: Any,
    weighted: bool,
) -> FusionStratification:
    groups: dict[int, list[_Facts]] = {band: [] for band in range(len(edges) + 1)}
    missing: list[_Facts] = []
    for item in facts:
        value = value_of(item)
        if value is None:
            missing.append(item)
        else:
            groups[bisect_right(edges, value)].append(item)
    return FusionStratification(
        dimension=dimension,
        source=source,
        unit=unit,
        edges=edges,
        strata=tuple(
            _stratum(_band_label(edges, band), groups[band], weighted)
            for band in range(len(edges) + 1)
        ),
        unavailable=_stratum("unavailable", missing, weighted),
    )


def _categorical(
    dimension: str, source: str, facts: Sequence[_Facts], weighted: bool
) -> FusionStratification:
    return FusionStratification(
        dimension=dimension,
        source=source,
        unit="category",
        edges=(),
        strata=tuple(
            _stratum(level, [f for f in facts if f.level == level], weighted)
            for level in _UNCERTAINTY_LEVELS
        ),
        unavailable=_stratum("unavailable", [], weighted),
    )


def _stratum(band: str, members: Sequence[_Facts], weighted: bool) -> FusionStratumReport:
    annotated = [item for item in members if item.reference is not None]
    return FusionStratumReport(
        band=band,
        supports=len(members),
        with_uncertainty=sum(1 for item in members if item.kinds),
        annotated=len(annotated),
        reference_recovered=sum(
            1 for item in annotated if _status(item, weighted=False) != "missing"
        ),
        leading_matches_reference=sum(
            1 for item in annotated if _status(item, weighted=False) == "leading"
        ),
        weighted_leading_matches_reference=(
            sum(1 for item in annotated if _status(item, weighted=True) == "leading")
            if weighted
            else None
        ),
    )


def _band_label(edges: tuple[float, ...], band: int) -> str:
    if not edges:
        return "(-inf, +inf)"
    if band == 0:
        return f"(-inf, {edges[0]})"
    if band == len(edges):
        return f"[{edges[-1]}, +inf)"
    return f"[{edges[band - 1]}, {edges[band]})"


def _reference_index(
    annotations: Sequence[ReferenceAnnotation] | None,
) -> dict[SpatialObservationId, str]:
    if annotations is None:
        return {}
    index: dict[SpatialObservationId, str] = {}
    for item in annotations:
        key = label_key(item.label)
        if index.setdefault(item.spatial_observation_id, key) != key:
            raise SemanticFusionEvaluationError(
                f"observation {item.spatial_observation_id!r} has two different reference labels"
            )
    return index


def _annotations_id(reference_of: Mapping[SpatialObservationId, str]) -> str:
    return _digest(sorted(reference_of.items()))


def _cost(reader: SemanticFusionRunReader) -> FusionCostReport:
    payload = reader.read_record("metrics/payload.json")
    try:
        runtime: Mapping[str, float | int | None] | None = reader.read_record(
            "metrics/runtime.json"
        )
    except FileNotFoundError:
        runtime = None
    return FusionCostReport(
        runtime=runtime,
        payload_bytes=dict(payload["files"]),
        total_payload_bytes=payload["total_bytes"],
    )


def _summary(values: Sequence[float]) -> FusionValueSummary | None:
    if not values:
        return None
    return FusionValueSummary(
        count=len(values),
        minimum=min(values),
        median=statistics.median(values),
        maximum=max(values),
    )


def _lineage_drift(first: FusionLineage, other: FusionLineage) -> str:
    shared = (
        "sequence_name",
        "sequence_artifact_id",
        "geometric_map_id",
        "association_run_ids",
        "perception_run_ids",
        "point_representation_run_ids",
        "grouping_policy_id",
        "support_policy_id",
        "support_configuration_fingerprint",
    )
    return ", ".join(name for name in shared if getattr(first, name) != getattr(other, name))


def _digest(records: Any) -> str:
    text = json.dumps(records, sort_keys=True, separators=(",", ":"), default=list)
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def _encode_lineage(lineage: FusionLineage) -> dict[str, Any]:
    return {
        "run_id": lineage.run_id,
        "run_index": lineage.run_index,
        "sequence_name": lineage.sequence_name,
        "sequence_artifact_id": lineage.sequence_artifact_id,
        "geometric_map_id": lineage.geometric_map_id,
        "association_run_ids": list(lineage.association_run_ids),
        "perception_run_ids": list(lineage.perception_run_ids),
        "point_representation_run_ids": list(lineage.point_representation_run_ids),
        "grouping_policy_id": lineage.grouping_policy_id,
        "support_policy_id": lineage.support_policy_id,
        "support_configuration_fingerprint": lineage.support_configuration_fingerprint,
        "fusion_policy_id": lineage.fusion_policy_id,
        "fusion_configuration_fingerprint": lineage.fusion_configuration_fingerprint,
        "code_version": lineage.code_version,
        "schema_version": lineage.schema_version,
    }


def _plain(section: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(section).items()
        if not isinstance(value, (Mapping, tuple)) and not hasattr(value, "__dataclass_fields__")
    }
