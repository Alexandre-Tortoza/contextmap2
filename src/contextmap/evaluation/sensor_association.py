"""Deterministic, stratified evaluation reports for Sensor Association runs.

Sensor Association is validated as an evidence-anchoring stage: how far the supporting
geometry is, how much of each region was visible, how densely it is supported, where it sits
in the image, how well the pose aligns in time and which dense feature path was sampled. This
module reads a persisted run through the public reader and reports those measurable factors
per stratum, keeping complete lineage, and without ever turning them into one probability or
mixing them with semantic confidence.

Nothing here alters a run. Strata come from an explicit :class:`StratificationProfile` with
no defaults (a band that suits one map and camera does not suit another); a measurement that
was unavailable goes to an explicit ``unavailable`` stratum and is never counted as zero.
Observations are regions, so each stratum also counts the **physical frames** they belong to:
several regions or repeated inference over one physical frame are not several physical
observations.

Two runs are compared only when everything but the dense feature path is identical, which
lets a native and an enhanced feature path be measured while geometry, calibration,
perception and association configuration stay constant.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from contextmap.sensor_association import (
    SensorAssociationRunManifest,
    SensorAssociationRunReader,
    ValueSummary,
)

EVALUATOR_VERSION = "2"
"""Bumped whenever a metric's definition changes, so reports stay comparable."""


class SensorAssociationEvaluationError(ValueError):
    """Raised when an evaluation invariant cannot be satisfied."""


@dataclass(frozen=True, kw_only=True)
class StratificationProfile:
    """The band edges of every stratification. There are no defaults.

    Each tuple of edges must be strictly increasing and finite; ``n`` edges give ``n + 1``
    bands, ``(-inf, e0)``, ``[e0, e1)``, ..., ``[e_last, +inf)``. An empty tuple gives a
    single band.

    Attributes:
        range_edges_m: Edges of the median depth of a region's support, in meters.
        visible_share_edges: Edges of the visible share of a region's footprint.
        support_density_edges: Edges of the associated points per mask pixel.
        border_distance_edges_px: Edges of the smallest distance of the support to the prepared
            image border, in pixels (valid-region and fisheye-border strata).
        viewing_angle_edges_rad: Edges of the median angle of the support from the optical
            axis, in radians.
    """

    range_edges_m: tuple[float, ...]
    visible_share_edges: tuple[float, ...]
    support_density_edges: tuple[float, ...]
    border_distance_edges_px: tuple[float, ...]
    viewing_angle_edges_rad: tuple[float, ...]

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
            "viewing_angle_edges_rad",
        ):
            edges = getattr(self, name)
            if not all(math.isfinite(edge) for edge in edges):
                raise ValueError(f"{name} must be finite, got {edges!r}")
            if any(low >= high for low, high in pairwise(edges)):
                raise ValueError(f"{name} must be strictly increasing, got {edges!r}")


@dataclass(frozen=True, kw_only=True)
class StratumReport:
    """The observations that fall in one band of one stratification.

    Attributes:
        band: The band, e.g. ``"[3.0, 8.0)"``, or ``"unavailable"``.
        observation_count: Regions in the band, across all frames.
        physical_observation_count: Distinct physical camera frames those regions belong to.
        associated_count: Visible geometry inside the regions' masks, summed.
        footprint_count: Projected geometry of any state inside the masks, summed; the
            denominator of :attr:`pooled_visible_share`.
    """

    band: str
    observation_count: int
    physical_observation_count: int
    associated_count: int
    footprint_count: int

    @property
    def pooled_visible_share(self) -> float | None:
        """``associated_count / footprint_count``; ``None`` when nothing was projected."""
        if self.footprint_count == 0:
            return None
        return self.associated_count / self.footprint_count


@dataclass(frozen=True, kw_only=True)
class StratificationReport:
    """Observations partitioned by one measurable factor.

    The bands and the ``unavailable`` stratum together hold every observation exactly once.

    Attributes:
        dimension: What is stratified, e.g. ``"range"``.
        component: The quality component the observations are assigned by.
        unit: The unit of the values and the edges.
        edges: The band edges used.
        strata: One report per band, in order.
        unavailable: Observations whose component could not be measured.
    """

    dimension: str
    component: str
    unit: str
    edges: tuple[float, ...]
    strata: tuple[StratumReport, ...]
    unavailable: StratumReport


@dataclass(frozen=True, kw_only=True)
class FeaturePathReport:
    """How one dense-feature channel was sampled over the run.

    Attributes:
        channel_id: The evidence channel.
        interpolation: The interpolation policy used.
        feature_sources: The exact feature sources the channel consumed, from the manifest.
        frame_count: Frames sampled.
        eligible_point_count: Visible points that were eligible, summed over frames.
        sampled_point_count: Of those, the ones the feature grid served.
    """

    channel_id: str
    interpolation: str
    feature_sources: tuple[Mapping[str, Any], ...]
    frame_count: int
    eligible_point_count: int
    sampled_point_count: int

    @property
    def out_of_support_count(self) -> int:
        """Eligible points the feature grid could not serve."""
        return self.eligible_point_count - self.sampled_point_count

    @property
    def sampled_ratio(self) -> float | None:
        """``sampled_point_count / eligible_point_count``; ``None`` without eligible points."""
        if self.eligible_point_count == 0:
            return None
        return self.sampled_point_count / self.eligible_point_count


@dataclass(frozen=True, kw_only=True)
class TimingReport:
    """Temporal alignment of the poses used, per physical frame.

    Attributes:
        pose_time_delta_ns: Distance from the frame timestamp to the nearest contributing pose.
        lookup_outcomes: Frames by pose lookup outcome.
    """

    pose_time_delta_ns: ValueSummary | None
    lookup_outcomes: Mapping[str, int]


@dataclass(frozen=True, kw_only=True)
class ReprojectionReport:
    """Reprojection residuals, only where a trusted reference existed.

    A missing residual is not one fact but three, and they are counted apart: a frame may have
    had no reference at all, a reference whose geometry its candidate policy never evaluated,
    or a reference it evaluated and could not project. Collapsing them lost both the
    distinction and the correspondence counts.

    Attributes:
        frames_measured: Frames whose residual was actually measured.
        frames_without_reference: Frames with no trusted reference; none is invented for them.
        frames_with_reference_not_evaluated: Frames whose candidate policy evaluated none of
            the reference geometry. A run whose reference lies outside its candidate range
            lands here instead of looking well-calibrated.
        frames_without_projectable_reference: Frames that evaluated the reference geometry and
            could project none of it. This one is a camera or calibration problem.
        correspondence_count: Reference correspondences declared, summed over every frame that
            had a reference, measured or not.
        invalid_correspondence_count: Of the evaluated ones, those the camera model could not
            project.
        unevaluated_correspondence_count: Correspondences whose geometry the frames' candidate
            policy never evaluated. They are not projection failures and are excluded from
            :attr:`invalid_rate`; a run whose reference lies outside its candidate range reports
            them here instead of looking well-calibrated.
        frame_median_px: Distribution, over frames, of each frame's median residual.
        frame_p95_px: Distribution, over frames, of each frame's 95th percentile residual.
    """

    frames_measured: int
    frames_without_reference: int
    frames_with_reference_not_evaluated: int
    frames_without_projectable_reference: int
    correspondence_count: int
    invalid_correspondence_count: int
    unevaluated_correspondence_count: int
    frame_median_px: ValueSummary | None
    frame_p95_px: ValueSummary | None

    @property
    def frames_with_reference(self) -> int:
        """Frames a trusted reference was supplied for, whatever came of it."""
        return (
            self.frames_measured
            + self.frames_with_reference_not_evaluated
            + self.frames_without_projectable_reference
        )

    @property
    def evaluated_correspondence_count(self) -> int:
        """Correspondences the frames actually evaluated: the population the rate is over."""
        return self.correspondence_count - self.unevaluated_correspondence_count

    @property
    def invalid_rate(self) -> float | None:
        """Share of the evaluated correspondences that could not be projected.

        ``None`` when nothing was evaluated, which is not the same as a rate of zero.
        """
        evaluated = self.evaluated_correspondence_count
        return None if evaluated == 0 else self.invalid_correspondence_count / evaluated


@dataclass(frozen=True, kw_only=True)
class SensorAssociationLineage:
    """The complete lineage of the evaluated run, copied from its manifest.

    Attributes:
        run_id: The evaluated run.
        sequence_artifact_id: The canonical sequence.
        selection_id: The sequence selection.
        geometric_map_id: The geometry the observations reference.
        trajectory_id: The trajectory that placed the camera.
        state_estimation_run_id: The state-estimation run it came from, when there is one.
        perception_run_ids: The perception runs the evidence came from.
        calibration_identity: The exact calibration used.
        candidate_policy: The candidate policy and its fingerprint: which map geometry each frame
            evaluated before projection. Two runs that differ here evaluated different
            populations, so comparing them as if only the feature path changed would be wrong.
        visibility_policy: The occlusion policy and its fingerprint.
        membership_policy_id: The mask-membership rule.
        definitions: Versions of the coverage, quality, diagnostics and sampling definitions.
        pose_policy: The pose lookup policy.
        tolerances: The diagnostic tolerances.
        configuration_fingerprint: Hash of the effective configuration.
        code_version: Code revision that produced the run, when known.
    """

    run_id: str
    sequence_artifact_id: str
    selection_id: str
    geometric_map_id: str
    trajectory_id: str
    state_estimation_run_id: str | None
    perception_run_ids: tuple[str, ...]
    calibration_identity: str
    candidate_policy: Mapping[str, Any]
    visibility_policy: Mapping[str, Any]
    membership_policy_id: str
    definitions: Mapping[str, str]
    pose_policy: Mapping[str, Any]
    tolerances: Mapping[str, Any]
    configuration_fingerprint: str
    code_version: str | None


@dataclass(frozen=True, kw_only=True)
class SensorAssociationEvaluationReport:
    """The stratified evaluation of one Sensor Association run.

    Attributes:
        evaluator_version: Version of the metric definitions.
        lineage: The complete lineage of the run.
        profile: The band edges the strata use.
        frame_count: Physical frames associated.
        rejected_frame_count: Frames whose pose was rejected.
        observation_count: Regions evaluated, across frames.
        physical_observation_count: Distinct physical frames they belong to.
        observations_without_support: Regions with no associated geometry.
        state_counts: Every projected point by visibility state, summed over frames.
        strata: Range, visibility, support density, image-border and viewing-angle strata.
        feature_paths: One report per dense-feature channel.
        timing: Temporal alignment of the poses.
        reprojection: Residuals against trusted references.
        finding_counts: Diagnostic findings by code.
    """

    evaluator_version: str
    lineage: SensorAssociationLineage
    profile: StratificationProfile
    frame_count: int
    rejected_frame_count: int
    observation_count: int
    physical_observation_count: int
    observations_without_support: int
    state_counts: Mapping[str, int]
    strata: tuple[StratificationReport, ...]
    feature_paths: tuple[FeaturePathReport, ...]
    timing: TimingReport
    reprojection: ReprojectionReport
    finding_counts: Mapping[str, int]


@dataclass(frozen=True, kw_only=True)
class SensorAssociationComparisonEntry:
    """One run of a comparison, with its own identity and feature paths.

    Attributes:
        run_id: The run.
        configuration_fingerprint: Its effective configuration.
        feature_paths: How each of its dense channels was sampled.
    """

    run_id: str
    configuration_fingerprint: str
    feature_paths: tuple[FeaturePathReport, ...]


@dataclass(frozen=True, kw_only=True)
class SensorAssociationComparison:
    """A controlled comparison of runs that differ only in their dense feature path.

    Attributes:
        lineage: The lineage every run shares.
        entries: One entry per run, preserving its identity and feature paths.
    """

    lineage: SensorAssociationLineage
    entries: tuple[SensorAssociationComparisonEntry, ...]


def evaluate_sensor_association(
    reader: SensorAssociationRunReader, *, profile: StratificationProfile
) -> SensorAssociationEvaluationReport:
    """Evaluate a persisted Sensor Association run.

    Args:
        reader: The run, opened through its public reader.
        profile: The band edges of the stratifications.

    Returns:
        The stratified report, with the run's complete lineage.

    Raises:
        SensorAssociationEvaluationError: If the run fails its own integrity check.
    """
    problems = reader.verify_integrity()
    if problems:
        raise SensorAssociationEvaluationError(f"the run is not intact: {problems}")
    manifest = reader.manifest
    observations = list(reader.observations())
    qualities = [reader.quality(o.spatial_observation_id) for o in observations]
    frame_ids = [r["source_observation_id"] for r in reader.read_records(_PROJECTION)]
    summary = reader.read_record("metrics/summary.json")

    return SensorAssociationEvaluationReport(
        evaluator_version=EVALUATOR_VERSION,
        lineage=_lineage(manifest),
        profile=profile,
        frame_count=manifest.frame_count,
        rejected_frame_count=manifest.rejected_frame_count,
        observation_count=len(observations),
        physical_observation_count=len({str(o.source_observation_id) for o in observations}),
        observations_without_support=summary["observations_without_support"],
        state_counts=dict(summary["state_counts"]),
        strata=(
            _stratify(
                "range",
                "support_depth",
                "m",
                profile.range_edges_m,
                observations,
                qualities,
                lambda q: None if q.support_depth_m is None else q.support_depth_m.median,
            ),
            _stratify(
                "visibility",
                "visible_share",
                "ratio",
                profile.visible_share_edges,
                observations,
                qualities,
                lambda q: q.visible_share,
            ),
            _stratify(
                "support_density",
                "support_density_per_mask_pixel",
                "points_per_mask_pixel",
                profile.support_density_edges,
                observations,
                qualities,
                lambda q: q.support_density_per_mask_pixel,
            ),
            _stratify(
                "image_border",
                "border_distance",
                "px",
                profile.border_distance_edges_px,
                observations,
                qualities,
                lambda q: None if q.border_distance_px is None else q.border_distance_px.minimum,
            ),
            _stratify(
                "viewing_angle",
                "support_off_axis_angle",
                "rad",
                profile.viewing_angle_edges_rad,
                observations,
                qualities,
                lambda q: (
                    None
                    if q.support_off_axis_angle_rad is None
                    else q.support_off_axis_angle_rad.median
                ),
            ),
        ),
        feature_paths=_feature_paths(reader, manifest, frame_ids),
        timing=_timing(reader.read_records(_PROJECTION)),
        reprojection=_reprojection(reader.read_records(_DIAGNOSTICS)),
        finding_counts=dict(manifest.finding_counts),
    )


def compare_sensor_association_reports(
    reports: Sequence[SensorAssociationEvaluationReport],
) -> SensorAssociationComparison:
    """Compare runs under one protocol, rejecting anything but the dense feature path changing.

    Args:
        reports: One report per run.

    Returns:
        The comparison, preserving each run's identity, configuration and feature paths.

    Raises:
        SensorAssociationEvaluationError: If there are fewer than two reports, or the sequence,
            selection, geometry, trajectory, calibration, perception, association
            configuration, stratification profile, evaluator version or any geometry-side
            measurement differs.
    """
    if len(reports) < 2:
        raise SensorAssociationEvaluationError("a comparison needs at least two reports")
    first = reports[0]
    for report in reports[1:]:
        if report.evaluator_version != first.evaluator_version:
            raise SensorAssociationEvaluationError("reports use different evaluator versions")
        if report.profile != first.profile:
            raise SensorAssociationEvaluationError("reports use different stratification profiles")
        drift = _shared_lineage_drift(first.lineage, report.lineage)
        if drift:
            raise SensorAssociationEvaluationError(
                f"reports differ in something other than the feature path: {drift}"
            )
        if _geometry_side(report) != _geometry_side(first):
            raise SensorAssociationEvaluationError(
                "reports differ in geometry-side measurements: the runs are not comparable"
            )
    return SensorAssociationComparison(
        lineage=first.lineage,
        entries=tuple(
            SensorAssociationComparisonEntry(
                run_id=report.lineage.run_id,
                configuration_fingerprint=report.lineage.configuration_fingerprint,
                feature_paths=report.feature_paths,
            )
            for report in reports
        ),
    )


def encode_sensor_association_report(report: SensorAssociationEvaluationReport) -> dict[str, Any]:
    """Return the report as JSON primitives, with every identity."""
    return {
        "evaluator_version": report.evaluator_version,
        "lineage": _encode_lineage(report.lineage),
        "profile": {
            "range_edges_m": list(report.profile.range_edges_m),
            "visible_share_edges": list(report.profile.visible_share_edges),
            "support_density_edges": list(report.profile.support_density_edges),
            "border_distance_edges_px": list(report.profile.border_distance_edges_px),
            "viewing_angle_edges_rad": list(report.profile.viewing_angle_edges_rad),
        },
        "frame_count": report.frame_count,
        "rejected_frame_count": report.rejected_frame_count,
        "observation_count": report.observation_count,
        "physical_observation_count": report.physical_observation_count,
        "observations_without_support": report.observations_without_support,
        "state_counts": dict(report.state_counts),
        "strata": [
            {
                "dimension": stratification.dimension,
                "component": stratification.component,
                "unit": stratification.unit,
                "edges": list(stratification.edges),
                "strata": [_encode_stratum(s) for s in stratification.strata],
                "unavailable": _encode_stratum(stratification.unavailable),
            }
            for stratification in report.strata
        ],
        "feature_paths": [_encode_feature_path(p) for p in report.feature_paths],
        "timing": {
            "pose_time_delta_ns": _encode_summary(report.timing.pose_time_delta_ns),
            "lookup_outcomes": dict(report.timing.lookup_outcomes),
        },
        "reprojection": {
            "frames_measured": report.reprojection.frames_measured,
            "frames_with_reference": report.reprojection.frames_with_reference,
            "frames_without_reference": report.reprojection.frames_without_reference,
            "frames_with_reference_not_evaluated": (
                report.reprojection.frames_with_reference_not_evaluated
            ),
            "frames_without_projectable_reference": (
                report.reprojection.frames_without_projectable_reference
            ),
            "correspondence_count": report.reprojection.correspondence_count,
            "invalid_correspondence_count": report.reprojection.invalid_correspondence_count,
            "unevaluated_correspondence_count": (
                report.reprojection.unevaluated_correspondence_count
            ),
            "evaluated_correspondence_count": report.reprojection.evaluated_correspondence_count,
            "invalid_rate": report.reprojection.invalid_rate,
            "frame_median_px": _encode_summary(report.reprojection.frame_median_px),
            "frame_p95_px": _encode_summary(report.reprojection.frame_p95_px),
        },
        "finding_counts": dict(report.finding_counts),
    }


def encode_sensor_association_comparison(comparison: SensorAssociationComparison) -> dict[str, Any]:
    """Return the comparison as JSON primitives, keeping each run's identity."""
    return {
        "lineage": _encode_lineage(comparison.lineage),
        "entries": [
            {
                "run_id": entry.run_id,
                "configuration_fingerprint": entry.configuration_fingerprint,
                "feature_paths": [_encode_feature_path(p) for p in entry.feature_paths],
            }
            for entry in comparison.entries
        ],
    }


_PROJECTION = "outputs/projection-records.jsonl"
_DIAGNOSTICS = "metrics/frame-diagnostics.jsonl"


def _lineage(manifest: SensorAssociationRunManifest) -> SensorAssociationLineage:
    return SensorAssociationLineage(
        run_id=str(manifest.run_id),
        sequence_artifact_id=str(manifest.sequence_artifact_id),
        selection_id=manifest.selection_id,
        geometric_map_id=str(manifest.geometric_map_id),
        trajectory_id=manifest.trajectory_id,
        state_estimation_run_id=manifest.state_estimation_run_id,
        perception_run_ids=tuple(manifest.perception_run_ids),
        calibration_identity=manifest.calibration_identity,
        candidate_policy=dict(manifest.candidate_policy),
        visibility_policy=dict(manifest.visibility_policy),
        membership_policy_id=manifest.membership_policy_id,
        definitions=dict(manifest.definitions),
        pose_policy=dict(manifest.pose_policy),
        tolerances=dict(manifest.tolerances),
        configuration_fingerprint=manifest.configuration_fingerprint,
        code_version=manifest.code_version,
    )


def _stratify(
    dimension: str,
    component: str,
    unit: str,
    edges: tuple[float, ...],
    observations: Sequence[Any],
    qualities: Sequence[Any],
    value_of: Any,
) -> StratificationReport:
    band_count = len(edges) + 1
    members: list[list[int]] = [[] for _ in range(band_count)]
    missing: list[int] = []
    for index, quality in enumerate(qualities):
        value = value_of(quality)
        if value is None:
            missing.append(index)
        else:
            members[bisect_right(edges, value)].append(index)

    def report(band: str, indices: list[int]) -> StratumReport:
        return StratumReport(
            band=band,
            observation_count=len(indices),
            physical_observation_count=len(
                {str(observations[i].source_observation_id) for i in indices}
            ),
            associated_count=sum(qualities[i].associated_count for i in indices),
            footprint_count=sum(qualities[i].footprint_count for i in indices),
        )

    return StratificationReport(
        dimension=dimension,
        component=component,
        unit=unit,
        edges=edges,
        strata=tuple(report(_band_label(edges, band), members[band]) for band in range(band_count)),
        unavailable=report("unavailable", missing),
    )


def _band_label(edges: tuple[float, ...], band: int) -> str:
    if not edges:
        return "(-inf, +inf)"
    if band == 0:
        return f"(-inf, {edges[0]})"
    if band == len(edges):
        return f"[{edges[-1]}, +inf)"
    return f"[{edges[band - 1]}, {edges[band]})"


def _feature_paths(
    reader: SensorAssociationRunReader,
    manifest: SensorAssociationRunManifest,
    frame_ids: Sequence[str],
) -> tuple[FeaturePathReport, ...]:
    paths: list[FeaturePathReport] = []
    for channel in manifest.dense_channels:
        eligible = sampled = 0
        for frame_id in frame_ids:
            record = reader.dense_association(frame_id, channel["channel_id"])
            eligible += len(record.eligible_indices)
            sampled += sum(record.sampled)
        paths.append(
            FeaturePathReport(
                channel_id=channel["channel_id"],
                interpolation=channel["interpolation"],
                feature_sources=tuple(channel["feature_sources"]),
                frame_count=len(frame_ids),
                eligible_point_count=eligible,
                sampled_point_count=sampled,
            )
        )
    return tuple(paths)


def _timing(projection_records: Sequence[Mapping[str, Any]]) -> TimingReport:
    deltas = [float(r["pose_ref"]["time_delta_ns"]) for r in projection_records]
    outcomes: dict[str, int] = {}
    for record in projection_records:
        outcome = record["pose_ref"]["lookup_outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return TimingReport(pose_time_delta_ns=_summary(deltas), lookup_outcomes=outcomes)


def _reprojection(diagnostics: Sequence[Mapping[str, Any]]) -> ReprojectionReport:
    medians: list[float] = []
    p95s: list[float] = []
    correspondences = invalid = unevaluated = 0
    without_reference = not_evaluated = none_projectable = 0
    for record in diagnostics:
        # A tentativa está sempre no registro; o residual, só quando algo projetou. Ler o
        # residual e seguir em frente descartava as contagens dos frames sem residual.
        attempt = record["reprojection_attempt"]
        outcome = attempt["outcome"]
        if outcome == "no_reference":
            without_reference += 1
            continue
        correspondences += attempt["correspondence_count"]
        invalid += attempt["invalid_count"]
        unevaluated += attempt["unevaluated_count"]
        if outcome == "not_evaluated":
            not_evaluated += 1
            continue
        if outcome == "none_projectable":
            none_projectable += 1
            continue
        reprojection = record["reprojection"]
        medians.append(reprojection["median_px"])
        p95s.append(reprojection["p95_px"])
    return ReprojectionReport(
        frames_measured=len(medians),
        frames_without_reference=without_reference,
        frames_with_reference_not_evaluated=not_evaluated,
        frames_without_projectable_reference=none_projectable,
        correspondence_count=correspondences,
        invalid_correspondence_count=invalid,
        unevaluated_correspondence_count=unevaluated,
        frame_median_px=_summary(medians),
        frame_p95_px=_summary(p95s),
    )


def _summary(values: Sequence[float]) -> ValueSummary | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return ValueSummary(count=len(ordered), minimum=ordered[0], median=median, maximum=ordered[-1])


def _shared_lineage_drift(first: SensorAssociationLineage, other: SensorAssociationLineage) -> str:
    for name in (
        "sequence_artifact_id",
        "selection_id",
        "geometric_map_id",
        "trajectory_id",
        "state_estimation_run_id",
        "perception_run_ids",
        "calibration_identity",
        "candidate_policy",
        "visibility_policy",
        "membership_policy_id",
        "definitions",
        "pose_policy",
        "tolerances",
    ):
        if getattr(first, name) != getattr(other, name):
            return name
    return ""


def _geometry_side(report: SensorAssociationEvaluationReport) -> tuple[Any, ...]:
    return (
        report.frame_count,
        report.rejected_frame_count,
        report.observation_count,
        report.physical_observation_count,
        report.observations_without_support,
        dict(report.state_counts),
        report.strata,
        report.timing,
        report.reprojection,
    )


def _encode_lineage(lineage: SensorAssociationLineage) -> dict[str, Any]:
    return {
        "run_id": lineage.run_id,
        "sequence_artifact_id": lineage.sequence_artifact_id,
        "selection_id": lineage.selection_id,
        "geometric_map_id": lineage.geometric_map_id,
        "trajectory_id": lineage.trajectory_id,
        "state_estimation_run_id": lineage.state_estimation_run_id,
        "perception_run_ids": list(lineage.perception_run_ids),
        "calibration_identity": lineage.calibration_identity,
        "candidate_policy": dict(lineage.candidate_policy),
        "visibility_policy": dict(lineage.visibility_policy),
        "membership_policy_id": lineage.membership_policy_id,
        "definitions": dict(lineage.definitions),
        "pose_policy": dict(lineage.pose_policy),
        "tolerances": dict(lineage.tolerances),
        "configuration_fingerprint": lineage.configuration_fingerprint,
        "code_version": lineage.code_version,
    }


def _encode_stratum(stratum: StratumReport) -> dict[str, Any]:
    return {
        "band": stratum.band,
        "observation_count": stratum.observation_count,
        "physical_observation_count": stratum.physical_observation_count,
        "associated_count": stratum.associated_count,
        "footprint_count": stratum.footprint_count,
        "pooled_visible_share": stratum.pooled_visible_share,
    }


def _encode_feature_path(path: FeaturePathReport) -> dict[str, Any]:
    return {
        "channel_id": path.channel_id,
        "interpolation": path.interpolation,
        "feature_sources": [dict(source) for source in path.feature_sources],
        "frame_count": path.frame_count,
        "eligible_point_count": path.eligible_point_count,
        "sampled_point_count": path.sampled_point_count,
        "out_of_support_count": path.out_of_support_count,
        "sampled_ratio": path.sampled_ratio,
    }


def _encode_summary(summary: ValueSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "median": summary.median,
        "maximum": summary.maximum,
    }
