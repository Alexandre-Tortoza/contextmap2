"""Deterministic validation reports for Geometric Mapping.

A map is not accepted because it was persisted: its points must be numerically
valid, the chain that placed them must reproduce their coordinates, overlapping
scans must agree with each other, and reopening the artifact must not change a
coordinate or a reference. This module builds one report for a persisted
:class:`~contextmap.geometric_mapping.GeometricMapArtifactReader`, so geometry
errors are visible *before* RGB-to-3D association can make them look like perception
failures.

Nothing here alters an artifact. Every scientific threshold comes from an explicit
:class:`GeometricMappingProtocol` with no defaults, so nothing is inherited from one
corridor or dataset. Quality and cost stay in separate sections. A point cloud is a
reference only when its profile explicitly declares it one (a ``.pcd`` name never
does), it must already be in the map frame, and no alignment is ever applied: the
report states ``alignment: none`` because aligning would hide the pose or frame error
being measured. Camera projection and semantics are outside this module.

NumPy is imported only inside the nearest-neighbour searches, so importing
:mod:`contextmap.evaluation` adds no base runtime dependency.
See ``src/contextmap/evaluation/docs/geometric_mapping.md``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMapArtifactManifest,
    GeometricMapArtifactReader,
    GeometricMapRunId,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
    MapId,
    MotionCorrectionState,
    PackedGeometry,
    PointOrigin,
    geometry_id_for,
    transform_trace_residual_m,
)
from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import Vector3, invert_rigid, rotate_vector
from contextmap.state_estimation import (
    DistributionSummary,
    StateEstimationRunId,
    TrajectoryId,
    summarize_distribution,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

EVALUATOR_VERSION = "2"
"""Bumped whenever a metric's definition changes, so reports stay comparable."""

# Amostragem para tornar a comparação com a referência tratável; as contagens são reportadas.
_MAX_REFERENCE_SAMPLES = 5_000
# Memória máxima (em elementos de distância) por bloco da busca de vizinho mais próximo.
_NEIGHBOUR_BLOCK_ELEMENTS = 4_000_000


class GeometricMappingEvaluationError(ValueError):
    """Raised when an evaluation invariant cannot be satisfied."""


def _require_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value}")


@dataclass(frozen=True, kw_only=True)
class GeometricMappingProtocol:
    """Everything an evaluation measures with, supplied by the reference profile.

    No field has a default, so no threshold is inherited from a dataset.

    Attributes:
        structure_point_stride: Check every ``k``-th persisted point (``1`` checks all).
            Density and range are computed over the checked points.
        trace_sample_count: Points traced end to end, evenly spread over the map.
        transform_tolerance_m: Largest accepted disagreement of the chain that
            placed a point, in meters.
        adjacent_scan_lag: Scan ``i`` is compared with scan ``i + lag``.
        coherence_pair_count: Adjacent pairs measured, evenly spread; ``0`` skips
            the overlap measurements.
        coherence_points_per_scan: Query points sampled from the earlier scan of
            each pair.
        plane_neighbour_count: Nearest points of the later scan a local plane is
            fitted through (at least three).
        max_plane_curvature: Largest ``λ0 / (λ0 + λ1 + λ2)`` of that neighbourhood's
            covariance for it to count as a plane; it excludes edges and corners.
        max_correspondence_distance_m: A query point whose plane neighbours are not all
            within this distance has no counterpart, and contributes no residual.
        ghost_residual_m: A residual above this is an inconsistency indicator (a
            duplicate or ghost surface), not a proof.
        density_voxel_m: Edge of the occupancy voxels.
        max_plausible_range_m: Sensor range beyond which a point is implausible.
        expected_bounds: Map-frame box the profile expects the geometry inside.
    """

    structure_point_stride: int
    trace_sample_count: int
    transform_tolerance_m: float
    adjacent_scan_lag: int
    coherence_pair_count: int
    coherence_points_per_scan: int
    plane_neighbour_count: int
    max_plane_curvature: float
    max_correspondence_distance_m: float
    ghost_residual_m: float
    density_voxel_m: float
    max_plausible_range_m: float | None = None
    expected_bounds: Bounds3D | None = None

    def __post_init__(self) -> None:
        """Validate the protocol.

        Raises:
            ValueError: If a count is not positive (or negative for the pair
                count), or a threshold is not positive and finite.
        """
        for name in ("structure_point_stride", "trace_sample_count", "adjacent_scan_lag"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.coherence_pair_count < 0:
            raise ValueError("coherence_pair_count must not be negative")
        if self.coherence_points_per_scan < 1:
            raise ValueError("coherence_points_per_scan must be at least 1")
        if self.plane_neighbour_count < 3:
            raise ValueError("plane_neighbour_count must be at least 3")
        if not 0 < self.max_plane_curvature <= 1 / 3:
            raise ValueError("max_plane_curvature must be in (0, 1/3]")
        for name in (
            "transform_tolerance_m",
            "max_correspondence_distance_m",
            "ghost_residual_m",
            "density_voxel_m",
        ):
            _require_positive(name, getattr(self, name))
        if self.max_plausible_range_m is not None:
            _require_positive("max_plausible_range_m", self.max_plausible_range_m)


@dataclass(frozen=True, kw_only=True)
class ExpectedPoint:
    """A source point whose global position is known independently of the mapping.

    Attributes:
        source_observation_id: The scan the point is in.
        source_point_index: Index of the point within that scan.
        expected_map_coordinates_m: Where it must land, in the map frame, in meters.
    """

    source_observation_id: SourceObservationId
    source_point_index: int
    expected_map_coordinates_m: Vector3


@dataclass(frozen=True, kw_only=True)
class ExpectedPointCheck:
    """An expected point against the coordinate the map produced.

    Attributes:
        source_observation_id: The scan the point is in.
        source_point_index: Index of the point within that scan.
        expected_map_coordinates_m: Where it had to land.
        produced_map_coordinates_m: Where the persisted map put it.
        error_m: Distance between the two.
        within_tolerance: Whether ``error_m`` is within ``transform_tolerance_m``.
    """

    source_observation_id: SourceObservationId
    source_point_index: int
    expected_map_coordinates_m: Vector3
    produced_map_coordinates_m: Vector3
    error_m: float
    within_tolerance: bool


@dataclass(frozen=True, kw_only=True)
class GeometricMappingStructureReport:
    """Structure of the persisted map, re-verified from its points.

    Counts by origin and the finiteness and frame checks cover the *checked* points
    (every ``structure_point_stride``-th); the rest are totals from the map.

    Attributes:
        point_count: Geometry elements in the map.
        scan_count: Scans accumulated.
        scans_without_geometry: Accumulated scans that contributed no point.
        rejected_scan_count: Selected scans the assembly left out.
        map_frame: The global frame.
        source_frames: Sensor frames of the checked points.
        points_checked: Points the checks covered.
        all_map_coordinates_finite: Whether every checked map coordinate is finite.
        all_source_coordinates_finite: Whether every checked source coordinate is finite.
        points_in_other_frame_count: Checked points not in the map frame.
        measured_point_count: Checked points that are one raw measurement.
        aggregated_point_count: Checked points that merge measurements.
        lineage_problem_count: Scans whose transform chain does not connect the map
            frame to the sensor frame.
        dropped_non_finite_count: Points dropped for a non-finite source coordinate.
        integrity_problems: Inventory and derived-index problems of the artifact.
    """

    point_count: int
    scan_count: int
    scans_without_geometry: int
    rejected_scan_count: int
    map_frame: FrameId
    source_frames: tuple[FrameId, ...]
    points_checked: int
    all_map_coordinates_finite: bool
    all_source_coordinates_finite: bool
    points_in_other_frame_count: int
    measured_point_count: int
    aggregated_point_count: int
    lineage_problem_count: int
    dropped_non_finite_count: int
    integrity_problems: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class GeometricMappingTraceEntry:
    """One traced point and how consistently its chain reproduces it.

    Attributes:
        source_observation_id: The scan the point is in.
        source_point_index: Index of the point within that scan; ``None`` for an
            aggregated point.
        pose_reference: The pose that placed it.
        calibration_reference: The calibration behind the static extrinsic, if any.
        composition_error_m: Distance between the recorded map coordinate and the
            source coordinate carried through the chain.
        round_trip_error_m: Distance between the source coordinate and the map
            coordinate carried back through the inverse chain.
    """

    source_observation_id: SourceObservationId
    source_point_index: int | None
    pose_reference: str
    calibration_reference: str | None
    composition_error_m: float
    round_trip_error_m: float


@dataclass(frozen=True, kw_only=True)
class GeometricMappingTraceReport:
    """Traced points and the worst numerical inconsistency among them.

    Attributes:
        entries: The traced points.
        tolerance_m: The tolerance they were judged against.
        max_composition_error_m: Worst composition error.
        max_round_trip_error_m: Worst inverse round-trip error.
        within_tolerance: Whether both maxima are within the tolerance.
    """

    entries: tuple[GeometricMappingTraceEntry, ...]
    tolerance_m: float
    max_composition_error_m: float
    max_round_trip_error_m: float
    within_tolerance: bool


@dataclass(frozen=True, kw_only=True)
class MapDensityReport:
    """Occupancy of the map.

    Attributes:
        voxel_m: Edge of the occupancy voxels.
        occupied_voxel_count: Voxels holding at least one checked point.
        points_per_voxel: Distribution of checked points per occupied voxel.
        points_per_scan: Distribution of persisted points per contributing scan.
    """

    voxel_m: float
    occupied_voxel_count: int
    points_per_voxel: DistributionSummary | None
    points_per_scan: DistributionSummary | None


@dataclass(frozen=True, kw_only=True)
class MapRangeReport:
    """Range of the checked points from their sensor.

    Attributes:
        distribution: Distance from the sensor origin, from the original coordinates.
        max_plausible_range_m: The profile's limit, if any.
        beyond_max_plausible_range_count: Checked points beyond it; ``None`` without a limit.
    """

    distribution: DistributionSummary | None
    max_plausible_range_m: float | None
    beyond_max_plausible_range_count: int | None


@dataclass(frozen=True, kw_only=True)
class MapBoundsReport:
    """Bounds of the map and points outside the box the profile expects.

    Attributes:
        bounds: Tight bounds of the persisted geometry.
        extent_m: Size of the bounds on each axis.
        expected_bounds: The profile's box, if any.
        outside_expected_count: Checked points outside it; ``None`` without a box.
    """

    bounds: Bounds3D
    extent_m: Vector3
    expected_bounds: Bounds3D | None
    outside_expected_count: int | None


@dataclass(frozen=True, kw_only=True)
class ScanOverlapPair:
    """Agreement between two temporally adjacent scans.

    Attributes:
        earlier_observation_id: The scan the query points were sampled from.
        later_observation_id: The scan the planes are fitted in.
        motion_correction: Correction state of the earlier scan.
        sample_count: Query points sampled.
        correspondence_count: Query points whose neighbours form a plane within the radius.
        overlap_fraction: ``correspondence_count / sample_count``.
        median_residual_m: Median point-to-plane distance, or ``None``.
        p95_residual_m: 95th percentile, or ``None``.
        max_residual_m: Largest, or ``None``.
        inconsistent_fraction: Share of correspondences farther than the ghost
            threshold, or ``None`` without correspondences.
    """

    earlier_observation_id: SourceObservationId
    later_observation_id: SourceObservationId
    motion_correction: MotionCorrectionState
    sample_count: int
    correspondence_count: int
    overlap_fraction: float
    median_residual_m: float | None
    p95_residual_m: float | None
    max_residual_m: float | None
    inconsistent_fraction: float | None


@dataclass(frozen=True, kw_only=True)
class CorrectionGroupOverlap:
    """Overlap residuals of the pairs whose earlier scan has one correction state.

    Attributes:
        state: The correction state.
        pair_count: Pairs in the group.
        correspondence_count: Correspondences in the group.
        median_residual_m: Median residual, or ``None``.
        p95_residual_m: 95th percentile, or ``None``.
    """

    state: MotionCorrectionState
    pair_count: int
    correspondence_count: int
    median_residual_m: float | None
    p95_residual_m: float | None


@dataclass(frozen=True, kw_only=True)
class ScanOverlapReport:
    """Agreement between adjacent scans, an internal-consistency measure.

    The residual is a point-to-plane distance: a query point of the earlier scan to a
    plane fitted through its nearest points in the later scan. It is blind to sliding
    along a surface, so an error that moves every scan alike, or that points along a
    corridor, is not visible here; expected points and a trusted reference catch it.
    Read the residual against another run of the same sequence, not as an absolute
    accuracy.

    Attributes:
        lag: Scan ``i`` is compared with scan ``i + lag``.
        max_correspondence_distance_m: The radius that defines a counterpart.
        ghost_residual_m: The residual above which a correspondence is flagged.
        pairs: The measured pairs.
        pooled_residual_m: Distribution of every correspondence's residual.
        overlap_fraction: Share of all query points with a counterpart, or ``None`` when
            no query point was sampled (no candidate pair): nothing measured is not an
            overlap of zero.
        inconsistent_fraction: Share of all correspondences above the ghost threshold.
        by_motion_correction: The residuals split by the earlier scan's correction state.
    """

    lag: int
    max_correspondence_distance_m: float
    ghost_residual_m: float
    pairs: tuple[ScanOverlapPair, ...]
    pooled_residual_m: DistributionSummary | None
    overlap_fraction: float | None
    inconsistent_fraction: float | None
    by_motion_correction: tuple[CorrectionGroupOverlap, ...]


class GeometryReferenceRole(Enum):
    """What a point cloud was declared to be by the reference profile.

    Attributes:
        EVALUATION_REFERENCE: Explicitly declared suitable to evaluate against.
        UNVERIFIED_PRODUCT: Some product (a map file, a ``.pcd``) with no declared
            provenance or role; whatever its name or format, it is not reference data.
    """

    EVALUATION_REFERENCE = "evaluation_reference"
    UNVERIFIED_PRODUCT = "unverified_product"


@dataclass(frozen=True, kw_only=True)
class ReferenceGeometry:
    """A point cloud together with the role, identity and frame its profile gave it.

    Attributes:
        reference_id: Identity of the reference profile that declares its role.
        role: The declared role; only ``EVALUATION_REFERENCE`` can be compared against.
        frame_id: The frame the coordinates are expressed in.
        coordinates_m: Points ``(x, y, z)`` in meters.
    """

    reference_id: str
    role: GeometryReferenceRole
    frame_id: FrameId
    coordinates_m: Sequence[Vector3]

    def __post_init__(self) -> None:
        """Require the profile identity and some geometry.

        Raises:
            ValueError: If ``reference_id`` is empty or there are no points.
        """
        if not self.reference_id:
            raise ValueError("reference_id must not be empty")
        if not self.coordinates_m:
            raise ValueError("a reference geometry needs at least one point")


@dataclass(frozen=True, kw_only=True)
class ReferenceGeometryReport:
    """Closeness of the map to a declared reference, without any alignment.

    Attributes:
        reference_id: Identity of the reference profile.
        role: The role that profile declared.
        frame_id: The frame both clouds are in.
        alignment: Always ``"none"``: the error includes any frame or pose offset.
        max_correspondence_distance_m: The radius that defines a counterpart.
        map_sample_count: Map points sampled.
        reference_sample_count: Reference points sampled.
        accuracy_m: Distance from each map sample to the nearest reference point,
            over the samples with one within the radius.
        matched_fraction: Share of map samples with a reference point within the radius.
        completeness: Share of reference samples with a map sample within the radius.
    """

    reference_id: str
    role: GeometryReferenceRole
    frame_id: FrameId
    alignment: str
    max_correspondence_distance_m: float
    map_sample_count: int
    reference_sample_count: int
    accuracy_m: DistributionSummary | None
    matched_fraction: float
    completeness: float


@dataclass(frozen=True, kw_only=True)
class GeometricMappingCostReport:
    """Execution cost, kept apart from every quality measure.

    Attributes:
        point_count: Geometry elements persisted.
        geometry_size_bytes: Size of the geometry payload.
        runtime_s: Wall-clock time of the run, when measured.
        peak_memory_bytes: Peak memory of the run, when measured.
    """

    point_count: int
    geometry_size_bytes: int
    runtime_s: float | None
    peak_memory_bytes: int | None


@dataclass(frozen=True, kw_only=True)
class GeometricMappingEvaluationReport:
    """Common validation report of one persisted map.

    Attributes:
        evaluator_version: Version of the metric definitions.
        run_id: The mapping run.
        map_id: The map, part of every geometry reference.
        sequence_artifact_id: Canonical sequence the map was built from.
        selection_id: Deterministic identity of the selection.
        trajectory_id: Trajectory whose poses placed the geometry.
        state_estimation_run_id: State Estimation run behind the trajectory, if known.
        calibration_identity: Hash of the calibration the extrinsics came from.
        configuration_fingerprint: Hash of the effective mapping configuration.
        code_version: Code revision that produced the map.
        protocol: The protocol the measurements were made with.
        structure: Re-verified structure and integrity.
        transform_trace: Traced points and their numerical consistency.
        expected_points: Independent expected coordinates against the produced ones.
        density: Occupancy.
        sensor_range: Range from the sensor.
        bounds: Map bounds and expected-box outliers.
        overlap: Adjacent-scan agreement, or ``None`` when no pair was requested.
        reference: Comparison against a declared reference, or ``None``.
        cost: Execution cost.
    """

    evaluator_version: str
    run_id: GeometricMapRunId
    map_id: MapId
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    calibration_identity: str | None
    configuration_fingerprint: str
    code_version: str | None
    protocol: GeometricMappingProtocol
    structure: GeometricMappingStructureReport
    transform_trace: GeometricMappingTraceReport
    expected_points: tuple[ExpectedPointCheck, ...]
    density: MapDensityReport
    sensor_range: MapRangeReport
    bounds: MapBoundsReport
    overlap: ScanOverlapReport | None
    reference: ReferenceGeometryReport | None
    cost: GeometricMappingCostReport


def evaluate_geometric_mapping(
    *,
    reader: GeometricMapArtifactReader,
    protocol: GeometricMappingProtocol,
    expected_points: Sequence[ExpectedPoint] = (),
    reference: ReferenceGeometry | None = None,
) -> GeometricMappingEvaluationReport:
    """Validate one persisted map under an explicit protocol.

    Args:
        reader: The opened mapping run; the caller closes it.
        protocol: Every threshold and sampling decision, from the reference profile.
        expected_points: Source points with independently known global positions.
        reference: A point cloud the profile declared an evaluation reference.

    Returns:
        The report.

    Raises:
        GeometricMappingEvaluationError: If an expected point is not a raw
            measurement of the map, or a reference is not declared for evaluation
            or is in another frame than the map.
    """
    manifest = reader.manifest
    geometry = reader.geometry()
    if reference is not None:
        _require_usable_reference(reference, manifest.map_frame)

    integrity = tuple(reader.verify_integrity())
    scan = _scan_pass(geometry, manifest, protocol)
    trace = _trace_report(geometry, protocol)
    checks = tuple(_check_expected(geometry, item, protocol) for item in expected_points)
    overlap = _overlap_report(geometry, protocol) if protocol.coherence_pair_count else None

    mapping = reader.read_record("metrics/mapping.json")
    runtime = (
        reader.read_record("metrics/runtime.json")
        if any(entry.path == "metrics/runtime.json" for entry in manifest.file_inventory)
        else {}
    )
    return GeometricMappingEvaluationReport(
        evaluator_version=EVALUATOR_VERSION,
        run_id=manifest.run_id,
        map_id=manifest.map_id,
        sequence_artifact_id=manifest.sequence_artifact_id,
        selection_id=manifest.selection_id,
        trajectory_id=manifest.trajectory_id,
        state_estimation_run_id=manifest.state_estimation_run_id,
        calibration_identity=manifest.calibration_identity,
        configuration_fingerprint=manifest.configuration_fingerprint,
        code_version=manifest.code_version,
        protocol=protocol,
        structure=GeometricMappingStructureReport(
            point_count=manifest.point_count,
            scan_count=manifest.scan_count,
            scans_without_geometry=sum(1 for item in geometry.scans if item.geometry_count == 0),
            rejected_scan_count=manifest.rejected_scan_count,
            map_frame=manifest.map_frame,
            source_frames=tuple(sorted(scan.source_frames)),
            points_checked=scan.points_checked,
            all_map_coordinates_finite=scan.map_finite,
            all_source_coordinates_finite=scan.source_finite,
            points_in_other_frame_count=scan.other_frame,
            measured_point_count=scan.measured,
            aggregated_point_count=scan.aggregated,
            lineage_problem_count=sum(
                1
                for item in geometry.scans
                if not item.transform_lineage.connects(manifest.map_frame, item.source_frame)
            ),
            dropped_non_finite_count=sum(item.dropped_non_finite_count for item in geometry.scans),
            integrity_problems=integrity,
        ),
        transform_trace=trace,
        expected_points=checks,
        density=MapDensityReport(
            voxel_m=protocol.density_voxel_m,
            occupied_voxel_count=len(scan.voxels),
            points_per_voxel=summarize_distribution(list(scan.voxels.values())),
            points_per_scan=summarize_distribution(
                [item.geometry_count for item in geometry.scans if item.geometry_count > 0]
            ),
        ),
        sensor_range=MapRangeReport(
            distribution=summarize_distribution(scan.ranges),
            max_plausible_range_m=protocol.max_plausible_range_m,
            beyond_max_plausible_range_count=scan.beyond_range,
        ),
        bounds=MapBoundsReport(
            bounds=geometry.geometric_map.bounds,
            extent_m=_extent(geometry.geometric_map.bounds),
            expected_bounds=protocol.expected_bounds,
            outside_expected_count=scan.outside_expected,
        ),
        overlap=overlap,
        reference=None
        if reference is None
        else _reference_report(geometry, reference, protocol, manifest.map_frame),
        cost=GeometricMappingCostReport(
            point_count=manifest.point_count,
            geometry_size_bytes=mapping["geometry_size_bytes"],
            runtime_s=runtime.get("runtime_s"),
            peak_memory_bytes=runtime.get("peak_memory_bytes"),
        ),
    )


@dataclass
class _ScanPass:
    """Accumulators of the single pass over the checked points."""

    points_checked: int = 0
    map_finite: bool = True
    source_finite: bool = True
    other_frame: int = 0
    measured: int = 0
    aggregated: int = 0
    beyond_range: int | None = None
    outside_expected: int | None = None

    def __post_init__(self) -> None:
        self.source_frames: set[FrameId] = set()
        self.voxels: dict[tuple[int, int, int], int] = defaultdict(int)
        self.ranges: list[float] = []


def _scan_pass(
    geometry: PackedGeometry,
    manifest: GeometricMapArtifactManifest,
    protocol: GeometricMappingProtocol,
) -> _ScanPass:
    result = _ScanPass()
    if protocol.max_plausible_range_m is not None:
        result.beyond_range = 0
    if protocol.expected_bounds is not None:
        result.outside_expected = 0
    voxel = protocol.density_voxel_m
    limit = protocol.max_plausible_range_m
    box = protocol.expected_bounds
    for point in _checked_points(geometry, protocol.structure_point_stride):
        result.points_checked += 1
        result.source_frames.add(point.source_frame)
        if point.map_frame != manifest.map_frame:
            result.other_frame += 1
        x, y, z = point.coordinates_m
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            result.map_finite = False
        sx, sy, sz = point.source_coordinates_m
        if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(sz)):
            result.source_finite = False
        if point.provenance.origin is PointOrigin.AGGREGATED:
            result.aggregated += 1
        else:
            result.measured += 1
        result.voxels[(math.floor(x / voxel), math.floor(y / voxel), math.floor(z / voxel))] += 1
        distance = math.sqrt(sx * sx + sy * sy + sz * sz)
        result.ranges.append(distance)
        if limit is not None and distance > limit:
            result.beyond_range = (result.beyond_range or 0) + 1
        if box is not None and not (
            box.minimum_m[0] <= x <= box.maximum_m[0]
            and box.minimum_m[1] <= y <= box.maximum_m[1]
            and box.minimum_m[2] <= z <= box.maximum_m[2]
        ):
            result.outside_expected = (result.outside_expected or 0) + 1
    return result


def _checked_points(geometry: PackedGeometry, stride: int) -> Iterator[GeometryPoint]:
    """Every ``stride``-th point in index order; a stride does not read the skipped points."""
    if stride == 1:
        yield from geometry.iter_geometry()
        return
    map_id = geometry.geometric_map.map_id
    for index in range(0, geometry.geometric_map.point_count, stride):
        yield geometry.get(
            GeometryReference(
                map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
            )
        )


def _extent(bounds: Bounds3D) -> Vector3:
    return (
        bounds.maximum_m[0] - bounds.minimum_m[0],
        bounds.maximum_m[1] - bounds.minimum_m[1],
        bounds.maximum_m[2] - bounds.minimum_m[2],
    )


def _evenly_spread(count: int, wanted: int) -> list[int]:
    """Deterministic, evenly spread, distinct positions in ``range(count)``."""
    if count <= 0:
        return []
    chosen = min(count, wanted)
    if chosen == 1:
        return [0]
    return sorted({round(i * (count - 1) / (chosen - 1)) for i in range(chosen)})


def _trace_report(
    geometry: PackedGeometry, protocol: GeometricMappingProtocol
) -> GeometricMappingTraceReport:
    map_id = geometry.geometric_map.map_id
    entries = []
    for index in _evenly_spread(geometry.geometric_map.point_count, protocol.trace_sample_count):
        trace = geometry.trace(
            GeometryReference(
                map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
            )
        )
        transforms = trace.transforms
        entries.append(
            GeometricMappingTraceEntry(
                source_observation_id=trace.source_observation_id,
                source_point_index=trace.source_point_index,
                pose_reference=transforms[0].step.reference,
                calibration_reference=transforms[1].step.reference if len(transforms) > 1 else None,
                composition_error_m=transform_trace_residual_m(trace),
                round_trip_error_m=_round_trip_error_m(trace.map_coordinates_m, trace),
            )
        )
    composition = max((entry.composition_error_m for entry in entries), default=0.0)
    round_trip = max((entry.round_trip_error_m for entry in entries), default=0.0)
    return GeometricMappingTraceReport(
        entries=tuple(entries),
        tolerance_m=protocol.transform_tolerance_m,
        max_composition_error_m=composition,
        max_round_trip_error_m=round_trip,
        within_tolerance=max(composition, round_trip) <= protocol.transform_tolerance_m,
    )


def _round_trip_error_m(map_point: Vector3, trace: Any) -> float:
    """Carry a map coordinate back through the inverse chain and compare with the source."""
    point = map_point
    for transform in trace.transforms:
        translation, rotation = invert_rigid(
            translation=transform.translation_m, rotation=transform.rotation
        )
        rotated = rotate_vector(rotation, point)
        point = (
            rotated[0] + translation[0],
            rotated[1] + translation[1],
            rotated[2] + translation[2],
        )
    return math.dist(point, trace.source_coordinates_m)


def _check_expected(
    geometry: PackedGeometry, expected: ExpectedPoint, protocol: GeometricMappingProtocol
) -> ExpectedPointCheck:
    try:
        geometry.scan_record(expected.source_observation_id)
    except KeyError as error:
        raise GeometricMappingEvaluationError(
            f"expected point names scan {expected.source_observation_id!r}, which is not in the map"
        ) from error
    for reference in geometry.references_for(expected.source_observation_id):
        point = geometry.get(reference)
        if point.source_point_index == expected.source_point_index:
            error_m = math.dist(point.coordinates_m, expected.expected_map_coordinates_m)
            return ExpectedPointCheck(
                source_observation_id=expected.source_observation_id,
                source_point_index=expected.source_point_index,
                expected_map_coordinates_m=expected.expected_map_coordinates_m,
                produced_map_coordinates_m=point.coordinates_m,
                error_m=error_m,
                within_tolerance=error_m <= protocol.transform_tolerance_m,
            )
    raise GeometricMappingEvaluationError(
        f"scan {expected.source_observation_id!r} has no raw measurement of point "
        f"{expected.source_point_index} in the map (dropped as non-finite, or aggregated)"
    )


def _nearest_distances(
    queries: NDArray[np.float64], targets: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distance from each query to its nearest target, exact and in bounded memory."""
    import numpy as np

    block = max(1, _NEIGHBOUR_BLOCK_ELEMENTS // max(len(targets), 1))
    nearest = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), block):
        difference = queries[start : start + block, None, :] - targets[None, :, :]
        nearest[start : start + block] = np.sqrt((difference * difference).sum(axis=2).min(axis=1))
    return nearest


def _plane_residuals(
    queries: NDArray[np.float64],
    targets: NDArray[np.float64],
    protocol: GeometricMappingProtocol,
) -> NDArray[np.float64]:
    """Point-to-plane distance from each query to a plane fitted through its neighbours.

    The plane is fitted through the ``plane_neighbour_count`` nearest target points.
    A query has no residual when those neighbours are not all within the correspondence
    radius, or do not form a plane. Point-to-plane, unlike point-to-point, does not
    depend on how sparsely the surface was sampled, which dominates the nearest-neighbour
    distance between two LiDAR scans.
    """
    import numpy as np

    neighbours = protocol.plane_neighbour_count
    if len(targets) < neighbours:
        return np.empty(0, dtype=np.float64)
    radius_squared = protocol.max_correspondence_distance_m**2
    block = max(1, _NEIGHBOUR_BLOCK_ELEMENTS // max(len(targets), 1))
    residuals: list[NDArray[np.float64]] = []
    for start in range(0, len(queries), block):
        chunk = queries[start : start + block]
        difference = chunk[:, None, :] - targets[None, :, :]
        squared = np.einsum("bmi,bmi->bm", difference, difference)
        nearest = np.argpartition(squared, neighbours - 1, axis=1)[:, :neighbours]
        within = np.take_along_axis(squared, nearest, axis=1).max(axis=1) <= radius_squared
        points = targets[nearest]
        centroid = points.mean(axis=1)
        centred = points - centroid[:, None, :]
        covariance = np.einsum("bki,bkj->bij", centred, centred) / neighbours
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        spread = eigenvalues.sum(axis=1)
        curvature = eigenvalues[:, 0] / np.where(spread > 0, spread, 1.0)
        distance = np.abs(np.einsum("bi,bi->b", eigenvectors[:, :, 0], chunk - centroid))
        usable = within & (spread > 0) & (curvature <= protocol.max_plane_curvature)
        residuals.append(distance[usable])
    return np.concatenate(residuals) if residuals else np.empty(0, dtype=np.float64)


def _as_points(flat: Iterable[float]) -> NDArray[np.float64]:
    import numpy as np

    return np.asarray(list(flat), dtype=np.float64).reshape(-1, 3)


def _sample(points: NDArray[np.float64], wanted: int) -> NDArray[np.float64]:
    stride = max(1, len(points) // wanted)
    return points[::stride][:wanted]


def _overlap_report(
    geometry: PackedGeometry, protocol: GeometricMappingProtocol
) -> ScanOverlapReport:
    import numpy as np

    contributing = [scan for scan in geometry.scans if scan.geometry_count > 0]
    lag = protocol.adjacent_scan_lag
    candidates = list(range(max(0, len(contributing) - lag)))
    pairs: list[ScanOverlapPair] = []
    residuals: list[float] = []
    by_state: dict[MotionCorrectionState, list[float]] = defaultdict(list)
    pair_states: dict[MotionCorrectionState, int] = defaultdict(int)
    samples_total = 0
    for position in _evenly_spread(len(candidates), protocol.coherence_pair_count):
        earlier, later = (
            contributing[candidates[position]],
            contributing[candidates[position] + lag],
        )
        queries = _sample(
            _as_points(geometry.scan_map_coordinates(earlier.observation_id)),
            protocol.coherence_points_per_scan,
        )
        targets = _as_points(geometry.scan_map_coordinates(later.observation_id))
        pair_residuals = [float(value) for value in _plane_residuals(queries, targets, protocol)]
        summary = summarize_distribution(pair_residuals)
        pairs.append(
            ScanOverlapPair(
                earlier_observation_id=earlier.observation_id,
                later_observation_id=later.observation_id,
                motion_correction=earlier.motion_correction,
                sample_count=len(queries),
                correspondence_count=len(pair_residuals),
                overlap_fraction=len(pair_residuals) / len(queries),
                median_residual_m=None if summary is None else summary.median,
                p95_residual_m=None if summary is None else summary.p95,
                max_residual_m=None if summary is None else summary.maximum,
                inconsistent_fraction=None
                if not pair_residuals
                else sum(1 for value in pair_residuals if value > protocol.ghost_residual_m)
                / len(pair_residuals),
            )
        )
        samples_total += len(queries)
        residuals.extend(pair_residuals)
        by_state[earlier.motion_correction].extend(pair_residuals)
        pair_states[earlier.motion_correction] += 1
    del np
    groups = []
    for state in sorted(pair_states, key=lambda item: item.value):
        summary = summarize_distribution(by_state[state])
        groups.append(
            CorrectionGroupOverlap(
                state=state,
                pair_count=pair_states[state],
                correspondence_count=len(by_state[state]),
                median_residual_m=None if summary is None else summary.median,
                p95_residual_m=None if summary is None else summary.p95,
            )
        )
    return ScanOverlapReport(
        lag=lag,
        max_correspondence_distance_m=protocol.max_correspondence_distance_m,
        ghost_residual_m=protocol.ghost_residual_m,
        pairs=tuple(pairs),
        pooled_residual_m=summarize_distribution(residuals),
        overlap_fraction=len(residuals) / samples_total if samples_total else None,
        inconsistent_fraction=None
        if not residuals
        else sum(1 for value in residuals if value > protocol.ghost_residual_m) / len(residuals),
        by_motion_correction=tuple(groups),
    )


def _require_usable_reference(reference: ReferenceGeometry, map_frame: FrameId) -> None:
    if reference.role is not GeometryReferenceRole.EVALUATION_REFERENCE:
        raise GeometricMappingEvaluationError(
            f"reference {reference.reference_id!r} was not declared an evaluation reference "
            f"(role {reference.role.value!r}); a point cloud is never a reference because of "
            "its file name or format"
        )
    if reference.frame_id != map_frame:
        raise GeometricMappingEvaluationError(
            f"reference {reference.reference_id!r} is in frame {reference.frame_id!r} but the map "
            f"is in {map_frame!r}; no frame is assumed aligned and no alignment is applied"
        )


def _reference_report(
    geometry: PackedGeometry,
    reference: ReferenceGeometry,
    protocol: GeometricMappingProtocol,
    map_frame: FrameId,
) -> ReferenceGeometryReport:
    import numpy as np

    stride = max(1, geometry.geometric_map.point_count // _MAX_REFERENCE_SAMPLES)
    map_points = _as_points(
        value
        for index, point in enumerate(geometry.iter_geometry())
        if index % stride == 0
        for value in point.coordinates_m
    )
    reference_points = _sample(
        np.asarray(reference.coordinates_m, dtype=np.float64), _MAX_REFERENCE_SAMPLES
    )
    radius = protocol.max_correspondence_distance_m
    accuracy = _nearest_distances(map_points, reference_points)
    coverage = _nearest_distances(reference_points, map_points)
    matched = accuracy[accuracy <= radius]
    return ReferenceGeometryReport(
        reference_id=reference.reference_id,
        role=reference.role,
        frame_id=map_frame,
        alignment="none",
        max_correspondence_distance_m=radius,
        map_sample_count=len(map_points),
        reference_sample_count=len(reference_points),
        accuracy_m=summarize_distribution([float(value) for value in matched]),
        matched_fraction=len(matched) / len(map_points),
        completeness=float(np.count_nonzero(coverage <= radius)) / len(reference_points),
    )


@dataclass(frozen=True, kw_only=True)
class RoundTripReport:
    """Whether reopening a persisted map preserves its geometry.

    Attributes:
        points_compared: Points compared, in index order.
        coordinate_mismatch_count: Points whose map or source coordinates differ.
        provenance_mismatch_count: Points equal in coordinates but not in identity,
            lineage or provenance.
        map_metadata_equal: Whether the two maps' metadata are equal.
        bounds_equal: Whether the two maps' bounds are equal.
        queries_compared: Probe boxes compared.
        query_mismatch_count: Probe boxes whose result differs.
    """

    points_compared: int
    coordinate_mismatch_count: int
    provenance_mismatch_count: int
    map_metadata_equal: bool
    bounds_equal: bool
    queries_compared: int
    query_mismatch_count: int


def evaluate_round_trip(
    *,
    expected: GeometrySource,
    reopened: GeometrySource,
    probe_boxes: Sequence[Bounds3D],
) -> RoundTripReport:
    """Compare a map that was built with the same map read back from its artifact.

    Args:
        expected: The geometry as it was built in memory.
        reopened: The geometry read back from the persisted artifact.
        probe_boxes: Boxes to query on both; their results must match.

    Returns:
        The counts of coordinate, provenance and query differences.
    """
    coordinates = provenance = compared = 0
    for built, read in zip(expected.iter_geometry(), reopened.iter_geometry(), strict=False):
        compared += 1
        if _coordinates(built) != _coordinates(read):
            coordinates += 1
        elif built != read:
            provenance += 1
    query_mismatches = sum(
        1
        for box in probe_boxes
        if [p.reference for p in expected.query_bounds(box)]
        != [p.reference for p in reopened.query_bounds(box)]
    )
    return RoundTripReport(
        points_compared=compared,
        coordinate_mismatch_count=coordinates,
        provenance_mismatch_count=provenance,
        map_metadata_equal=expected.geometric_map == reopened.geometric_map,
        bounds_equal=expected.geometric_map.bounds == reopened.geometric_map.bounds,
        queries_compared=len(probe_boxes),
        query_mismatch_count=query_mismatches,
    )


def _coordinates(point: GeometryPoint) -> tuple[Vector3, Vector3]:
    return point.coordinates_m, point.source_coordinates_m


@dataclass(frozen=True, kw_only=True)
class ReproducibilityReport:
    """Whether two runs produced the same contractual files.

    Attributes:
        identical: Whether every contractual file has the same hash in both runs.
        differing_paths: Contractual files that differ or exist in only one run.
    """

    identical: bool
    differing_paths: tuple[str, ...]


def compare_contractual_inventories(
    first: GeometricMapArtifactManifest, second: GeometricMapArtifactManifest
) -> ReproducibilityReport:
    """Compare the contractual files of two runs by content hash.

    Identity that legitimately differs between runs (the run id, the creation time)
    is not part of the inventory, so identical inputs and configuration compare equal.

    Args:
        first: One run's manifest.
        second: The other run's manifest.
    """
    left = {entry.path: entry.content_hash for entry in first.file_inventory}
    right = {entry.path: entry.content_hash for entry in second.file_inventory}
    differing = tuple(
        sorted(path for path in left.keys() | right.keys() if left.get(path) != right.get(path))
    )
    return ReproducibilityReport(identical=not differing, differing_paths=differing)


@dataclass(frozen=True, kw_only=True)
class GeometricMappingComparisonEntry:
    """One report's identity and headline numbers, side by side with the others.

    Attributes:
        run_id: The mapping run.
        map_id: The map.
        trajectory_id: The trajectory that placed the geometry.
        configuration_fingerprint: The mapping configuration.
        code_version: Code revision.
        point_count: Geometry elements persisted.
        median_overlap_residual_m: Median adjacent-scan residual, or ``None``.
        inconsistent_fraction: Share of correspondences above the ghost threshold, or ``None``.
        max_expected_point_error_m: Worst expected-point error, or ``None`` without any.
        max_composition_error_m: Worst chain composition error.
        runtime_s: Run time, apart from quality.
    """

    run_id: GeometricMapRunId
    map_id: MapId
    trajectory_id: TrajectoryId
    configuration_fingerprint: str
    code_version: str | None
    point_count: int
    median_overlap_residual_m: float | None
    inconsistent_fraction: float | None
    max_expected_point_error_m: float | None
    max_composition_error_m: float
    runtime_s: float | None


@dataclass(frozen=True, kw_only=True)
class GeometricMappingComparison:
    """A controlled comparison: everything is fixed except the mapping under test.

    Attributes:
        evaluator_version: The shared evaluator version.
        sequence_artifact_id: The shared sequence.
        selection_id: The shared selection.
        calibration_identity: The shared calibration.
        protocol: The shared protocol.
        entries: One entry per report, in the order given.
    """

    evaluator_version: str
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    calibration_identity: str | None
    protocol: GeometricMappingProtocol
    entries: tuple[GeometricMappingComparisonEntry, ...]


def compare_geometric_mapping_reports(
    reports: Sequence[GeometricMappingEvaluationReport],
) -> GeometricMappingComparison:
    """Compare reports that differ only in the mapping under test.

    Args:
        reports: At least two reports.

    Returns:
        The comparison, preserving each run's identity and configuration.

    Raises:
        GeometricMappingEvaluationError: If fewer than two reports are given or they
            differ in evaluator version, sequence, selection, calibration or protocol.
    """
    if len(reports) < 2:
        raise GeometricMappingEvaluationError("a comparison needs at least two reports")
    first = reports[0]
    for name in (
        "evaluator_version",
        "sequence_artifact_id",
        "selection_id",
        "calibration_identity",
        "protocol",
    ):
        if any(getattr(report, name) != getattr(first, name) for report in reports[1:]):
            raise GeometricMappingEvaluationError(
                f"reports differ in {name}; a controlled comparison changes one variable at a time"
            )
    return GeometricMappingComparison(
        evaluator_version=first.evaluator_version,
        sequence_artifact_id=first.sequence_artifact_id,
        selection_id=first.selection_id,
        calibration_identity=first.calibration_identity,
        protocol=first.protocol,
        entries=tuple(
            GeometricMappingComparisonEntry(
                run_id=report.run_id,
                map_id=report.map_id,
                trajectory_id=report.trajectory_id,
                configuration_fingerprint=report.configuration_fingerprint,
                code_version=report.code_version,
                point_count=report.structure.point_count,
                median_overlap_residual_m=None
                if report.overlap is None or report.overlap.pooled_residual_m is None
                else report.overlap.pooled_residual_m.median,
                inconsistent_fraction=None
                if report.overlap is None
                else report.overlap.inconsistent_fraction,
                max_expected_point_error_m=max(
                    (check.error_m for check in report.expected_points), default=None
                ),
                max_composition_error_m=report.transform_trace.max_composition_error_m,
                runtime_s=report.cost.runtime_s,
            )
            for report in reports
        ),
    )


def _encode_summary(summary: DistributionSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "median": summary.median,
        "p95": summary.p95,
        "maximum": summary.maximum,
    }


def _encode_bounds(bounds: Bounds3D | None) -> dict[str, Any] | None:
    if bounds is None:
        return None
    return {
        "frame_id": str(bounds.frame_id),
        "minimum_m": list(bounds.minimum_m),
        "maximum_m": list(bounds.maximum_m),
    }


def encode_geometric_mapping_report(report: GeometricMappingEvaluationReport) -> dict[str, Any]:
    """Encode a report as plain JSON primitives, keeping every identity.

    Args:
        report: The report.

    Returns:
        A JSON-compatible mapping.
    """
    protocol = report.protocol
    structure = report.structure
    overlap = report.overlap
    reference = report.reference
    return {
        "evaluator_version": report.evaluator_version,
        "run_id": str(report.run_id),
        "map_id": str(report.map_id),
        "sequence_artifact_id": str(report.sequence_artifact_id),
        "selection_id": report.selection_id,
        "trajectory_id": str(report.trajectory_id),
        "state_estimation_run_id": None
        if report.state_estimation_run_id is None
        else str(report.state_estimation_run_id),
        "calibration_identity": report.calibration_identity,
        "configuration_fingerprint": report.configuration_fingerprint,
        "code_version": report.code_version,
        "protocol": {
            "structure_point_stride": protocol.structure_point_stride,
            "trace_sample_count": protocol.trace_sample_count,
            "transform_tolerance_m": protocol.transform_tolerance_m,
            "adjacent_scan_lag": protocol.adjacent_scan_lag,
            "coherence_pair_count": protocol.coherence_pair_count,
            "coherence_points_per_scan": protocol.coherence_points_per_scan,
            "plane_neighbour_count": protocol.plane_neighbour_count,
            "max_plane_curvature": protocol.max_plane_curvature,
            "max_correspondence_distance_m": protocol.max_correspondence_distance_m,
            "ghost_residual_m": protocol.ghost_residual_m,
            "density_voxel_m": protocol.density_voxel_m,
            "max_plausible_range_m": protocol.max_plausible_range_m,
            "expected_bounds": _encode_bounds(protocol.expected_bounds),
        },
        "structure": {
            "point_count": structure.point_count,
            "scan_count": structure.scan_count,
            "scans_without_geometry": structure.scans_without_geometry,
            "rejected_scan_count": structure.rejected_scan_count,
            "map_frame": str(structure.map_frame),
            "source_frames": [str(frame) for frame in structure.source_frames],
            "points_checked": structure.points_checked,
            "all_map_coordinates_finite": structure.all_map_coordinates_finite,
            "all_source_coordinates_finite": structure.all_source_coordinates_finite,
            "points_in_other_frame_count": structure.points_in_other_frame_count,
            "measured_point_count": structure.measured_point_count,
            "aggregated_point_count": structure.aggregated_point_count,
            "lineage_problem_count": structure.lineage_problem_count,
            "dropped_non_finite_count": structure.dropped_non_finite_count,
            "integrity_problems": list(structure.integrity_problems),
        },
        "transform_trace": {
            "tolerance_m": report.transform_trace.tolerance_m,
            "max_composition_error_m": report.transform_trace.max_composition_error_m,
            "max_round_trip_error_m": report.transform_trace.max_round_trip_error_m,
            "within_tolerance": report.transform_trace.within_tolerance,
            "entries": [
                {
                    "source_observation_id": str(entry.source_observation_id),
                    "source_point_index": entry.source_point_index,
                    "pose_reference": entry.pose_reference,
                    "calibration_reference": entry.calibration_reference,
                    "composition_error_m": entry.composition_error_m,
                    "round_trip_error_m": entry.round_trip_error_m,
                }
                for entry in report.transform_trace.entries
            ],
        },
        "expected_points": [
            {
                "source_observation_id": str(check.source_observation_id),
                "source_point_index": check.source_point_index,
                "expected_map_coordinates_m": list(check.expected_map_coordinates_m),
                "produced_map_coordinates_m": list(check.produced_map_coordinates_m),
                "error_m": check.error_m,
                "within_tolerance": check.within_tolerance,
            }
            for check in report.expected_points
        ],
        "density": {
            "voxel_m": report.density.voxel_m,
            "occupied_voxel_count": report.density.occupied_voxel_count,
            "points_per_voxel": _encode_summary(report.density.points_per_voxel),
            "points_per_scan": _encode_summary(report.density.points_per_scan),
        },
        "sensor_range": {
            "distribution": _encode_summary(report.sensor_range.distribution),
            "max_plausible_range_m": report.sensor_range.max_plausible_range_m,
            "beyond_max_plausible_range_count": (
                report.sensor_range.beyond_max_plausible_range_count
            ),
        },
        "bounds": {
            "bounds": _encode_bounds(report.bounds.bounds),
            "extent_m": list(report.bounds.extent_m),
            "expected_bounds": _encode_bounds(report.bounds.expected_bounds),
            "outside_expected_count": report.bounds.outside_expected_count,
        },
        "overlap": None
        if overlap is None
        else {
            "lag": overlap.lag,
            "max_correspondence_distance_m": overlap.max_correspondence_distance_m,
            "ghost_residual_m": overlap.ghost_residual_m,
            "overlap_fraction": overlap.overlap_fraction,
            "inconsistent_fraction": overlap.inconsistent_fraction,
            "pooled_residual_m": _encode_summary(overlap.pooled_residual_m),
            "pairs": [
                {
                    "earlier_observation_id": str(pair.earlier_observation_id),
                    "later_observation_id": str(pair.later_observation_id),
                    "motion_correction": pair.motion_correction.value,
                    "sample_count": pair.sample_count,
                    "correspondence_count": pair.correspondence_count,
                    "overlap_fraction": pair.overlap_fraction,
                    "median_residual_m": pair.median_residual_m,
                    "p95_residual_m": pair.p95_residual_m,
                    "max_residual_m": pair.max_residual_m,
                    "inconsistent_fraction": pair.inconsistent_fraction,
                }
                for pair in overlap.pairs
            ],
            "by_motion_correction": [
                {
                    "state": group.state.value,
                    "pair_count": group.pair_count,
                    "correspondence_count": group.correspondence_count,
                    "median_residual_m": group.median_residual_m,
                    "p95_residual_m": group.p95_residual_m,
                }
                for group in overlap.by_motion_correction
            ],
        },
        "reference": None
        if reference is None
        else {
            "reference_id": reference.reference_id,
            "role": reference.role.value,
            "frame_id": str(reference.frame_id),
            "alignment": reference.alignment,
            "max_correspondence_distance_m": reference.max_correspondence_distance_m,
            "map_sample_count": reference.map_sample_count,
            "reference_sample_count": reference.reference_sample_count,
            "accuracy_m": _encode_summary(reference.accuracy_m),
            "matched_fraction": reference.matched_fraction,
            "completeness": reference.completeness,
        },
        "cost": {
            "point_count": report.cost.point_count,
            "geometry_size_bytes": report.cost.geometry_size_bytes,
            "runtime_s": report.cost.runtime_s,
            "peak_memory_bytes": report.cost.peak_memory_bytes,
        },
    }
