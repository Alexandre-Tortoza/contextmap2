"""Paired evaluation of a raw map against its inter-scan voxel aggregation (#624).

Two questions, answered on the same raw map and never by visual inspection:

* **Geometric fidelity** — how far each raw point is from the representative of the
  aggregate that summarizes it, overall and per range to the sensor that measured it,
  and whether the map's extent survives the aggregation.
* **2D↔3D association stability** — for the same camera frames, regions and policies,
  whether the regions the raw map supports are still supported by the aggregates, on
  which voxels the two agree, and how the geometry a consumer would build from the
  support (its centroid, its box) moves.

Every definition is fixed here, before any real run is looked at; the experiment only
supplies inputs and thresholds. Nothing alters an artifact. See
``src/contextmap/evaluation/docs/voxel_aggregation.md``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping import (
    DEFAULT_BLOCK_POINTS,
    AggregatedGeometry,
    Bounds3D,
    GeometryReference,
    GeometrySource,
    MapId,
    PackedGeometry,
    ScanRecord,
    VoxelAggregation,
    geometry_index_of,
)
from contextmap.sensor_association import SpatialObservation
from contextmap.shared import Vector3
from contextmap.state_estimation import DistributionSummary

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class RangeBandFidelity:
    """Distance to the representative for the raw points in one band of sensor range.

    Attributes:
        minimum_range_m: Inclusive lower edge of the band.
        maximum_range_m: Exclusive upper edge; ``None`` for the last, open band.
        point_count: Raw points whose range falls in the band.
        distance_m: Order statistics of their distance to their representative.
        mean_distance_m: Mean of that distance; ``None`` when the band is empty.
    """

    minimum_range_m: float
    maximum_range_m: float | None
    point_count: int
    distance_m: DistributionSummary | None
    mean_distance_m: float | None


@dataclass(frozen=True, kw_only=True)
class AggregationFidelityReport:
    """How much geometry an aggregation moved, measured on every raw point.

    Attributes:
        aggregation_rule: The rule evaluated.
        policy_fingerprint: Identity of the policy, grid included.
        cell_m: The voxel edge length.
        source_point_count: Raw points evaluated; all of them.
        aggregate_count: Aggregates they collapse into.
        reduction_ratio: ``aggregate_count / source_point_count``.
        distance_m: Distance from each raw point to its aggregate's centroid.
        mean_distance_m: Mean of that distance.
        by_range: The same distance, per band of range to the measuring sensor.
        raw_bounds: The raw map's bounds.
        centroid_bounds: The envelope of the centroids, what a consumer of the
            representatives sees.
        statistics_bounds_preserved: Whether the envelope of the per-aggregate
            minima and maxima is exactly the raw bounds.
        max_bounds_shrinkage_m: Largest distance between a face of the raw bounds and
            the same face of the centroid envelope.
    """

    aggregation_rule: str
    policy_fingerprint: str
    cell_m: float
    source_point_count: int
    aggregate_count: int
    reduction_ratio: float
    distance_m: DistributionSummary
    mean_distance_m: float
    by_range: tuple[RangeBandFidelity, ...]
    raw_bounds: Bounds3D
    centroid_bounds: Bounds3D
    statistics_bounds_preserved: bool
    max_bounds_shrinkage_m: float


def evaluate_aggregation_fidelity(
    raw: PackedGeometry,
    aggregation: VoxelAggregation,
    *,
    range_edges_m: Sequence[float],
    block_points: int = DEFAULT_BLOCK_POINTS,
) -> AggregationFidelityReport:
    """Measure the distance from every raw point to the representative that replaces it.

    The range of a point is its distance to the origin of the sensor that measured it,
    placed in the map by the scan's own transform chain; the chain is rigid, so this is
    the range the sensor measured.

    Args:
        raw: The raw map the aggregation was derived from.
        aggregation: The aggregation to evaluate.
        range_edges_m: Increasing band edges starting at ``0``; the last band is open.
        block_points: Rows read per block.

    Returns:
        The fidelity report.

    Raises:
        ValueError: If the edges are not increasing from zero, or the aggregation was not
            derived from ``raw``.
    """
    import numpy as np

    edges = _validated_edges(range_edges_m)
    if raw.geometric_map.map_id != aggregation.source_map_id:
        raise ValueError(
            f"the aggregation summarizes map {aggregation.source_map_id!r}, not "
            f"{raw.geometric_map.map_id!r}"
        )
    grid = aggregation.policy.grid
    count = raw.geometric_map.point_count
    origins = np.array([_sensor_origin_m(scan) for scan in raw.scans], dtype=np.float64)
    distances = np.empty(count, dtype=np.float64)
    ranges = np.empty(count, dtype=np.float64)
    for block in raw.iter_blocks(block_points=block_points):
        members = aggregation.index_of(grid.keys_of(block.coordinates_m))
        if (members < 0).any():
            raise ValueError("a raw point falls in no aggregate: the aggregation is of another map")
        coordinates = block.coordinates_m
        distances[block.indices] = np.linalg.norm(
            coordinates - aggregation.centroids_m[members], axis=1
        )
        scans = _scan_ordinals(raw.scans, block.indices)
        ranges[block.indices] = np.linalg.norm(coordinates - origins[scans], axis=1)

    bands = np.searchsorted(np.array(edges), ranges, side="right") - 1
    by_range = tuple(_band(edges, band, distances[bands == band]) for band in range(len(edges)))
    raw_bounds = raw.geometric_map.bounds
    centroid_bounds = _envelope(aggregation.centroids_m, raw_bounds)
    statistics = Bounds3D(
        frame_id=raw_bounds.frame_id,
        minimum_m=_triple(aggregation.minimum_m.min(axis=0)),
        maximum_m=_triple(aggregation.maximum_m.max(axis=0)),
    )
    summary = _summary(distances)
    if summary is None:
        raise ValueError("the raw map has no point to evaluate")
    return AggregationFidelityReport(
        aggregation_rule=aggregation.policy.rule,
        policy_fingerprint=aggregation.policy.fingerprint(),
        cell_m=grid.cell_m,
        source_point_count=count,
        aggregate_count=len(aggregation),
        reduction_ratio=len(aggregation) / count,
        distance_m=summary,
        mean_distance_m=float(distances.mean()),
        by_range=by_range,
        raw_bounds=raw_bounds,
        centroid_bounds=centroid_bounds,
        statistics_bounds_preserved=statistics == raw_bounds,
        max_bounds_shrinkage_m=max(
            max(
                abs(raw_bounds.minimum_m[axis] - centroid_bounds.minimum_m[axis]),
                abs(raw_bounds.maximum_m[axis] - centroid_bounds.maximum_m[axis]),
            )
            for axis in range(3)
        ),
    )


@dataclass(frozen=True, kw_only=True, order=True)
class PairedRegion:
    """The identity of one region of one frame, shared by the raw and aggregate runs.

    Attributes:
        source_observation_id: The camera frame.
        perception_result_id: The perception result that owns the region.
        region_id: The region.
    """

    source_observation_id: str
    perception_result_id: str
    region_id: str


def region_supports(
    observations: Iterable[SpatialObservation],
) -> dict[PairedRegion, tuple[GeometryReference, ...]]:
    """Index the geometry support of spatial observations by the region they describe.

    Raises:
        ValueError: If one region appears twice, which pairing could not resolve.
    """
    supports: dict[PairedRegion, tuple[GeometryReference, ...]] = {}
    for observation in observations:
        region = PairedRegion(
            source_observation_id=str(observation.source_observation_id),
            perception_result_id=str(observation.perception_result_id),
            region_id=str(observation.region_id),
        )
        if region in supports:
            raise ValueError(f"region {region} appears twice in one association run")
        supports[region] = observation.geometry_support
    return supports


@dataclass(frozen=True, kw_only=True)
class RegionAssociationComparison:
    """One region's support on the raw map and on the aggregates.

    Voxel sets are compared: the raw support is mapped to the aggregates holding its
    points (``m(R)``) and compared with the aggregate support (``A``).

    Attributes:
        region: The paired region.
        raw_support_count: ``|R|``, raw points supporting the region.
        mapped_raw_support_count: ``|m(R)|``, aggregates holding those raw points.
        aggregate_support_count: ``|A|``, aggregates supporting the region.
        shared_aggregate_count: ``|m(R) & A|``, aggregates in both.
        support_retention: ``|m(R) & A| / |m(R)|``; ``None`` without raw support.
        disagreement: ``1 - |m(R) & A| / |m(R) | A|``, one minus the Jaccard index;
            ``None`` when both are empty.
        support_centroid_shift_m: Distance between the mean of the raw support points
            and the mean of the supporting centroids; ``None`` unless both exist.
        support_extent_change_m: Largest per-axis change of the support's box extent.
        support_bounds_iou: Volume IoU of the two support boxes; ``None`` unless both
            boxes have a positive volume.
    """

    region: PairedRegion
    raw_support_count: int
    mapped_raw_support_count: int
    aggregate_support_count: int
    shared_aggregate_count: int
    support_retention: float | None
    disagreement: float | None
    support_centroid_shift_m: float | None
    support_extent_change_m: float | None
    support_bounds_iou: float | None


@dataclass(frozen=True, kw_only=True)
class AssociationStabilityReport:
    """Whether 2D↔3D association survives replacing raw points by aggregates.

    Attributes:
        thin_support_max_points: Raw support size at or below which a region is thin.
        paired_region_count: Regions present in both runs; the only ones compared.
        raw_only_region_count: Regions only the raw run produced.
        aggregate_only_region_count: Regions only the aggregate run produced.
        raw_supported_region_count: Paired regions with raw support.
        retained_region_count: Of those, the ones with aggregate support.
        lost_region_count: Of those, the ones without aggregate support.
        gained_region_count: Paired regions supported only on the aggregates.
        region_retention_rate: ``retained / raw_supported``; ``None`` without any.
        thin_region_count: Raw-supported regions that are thin.
        thin_retained_region_count: Of those, the ones retained.
        support_retention: Distribution of the per-region support retention.
        disagreement: Distribution of the per-region disagreement.
        raw_support_points: Distribution of ``|R|`` over paired regions.
        aggregate_support_points: Distribution of ``|A|`` over paired regions.
        support_centroid_shift_m: Distribution of the per-region centroid shift.
        support_extent_change_m: Distribution of the per-region extent change.
        support_bounds_iou: Distribution of the per-region box IoU.
        regions: Every paired region, in order.
    """

    thin_support_max_points: int
    paired_region_count: int
    raw_only_region_count: int
    aggregate_only_region_count: int
    raw_supported_region_count: int
    retained_region_count: int
    lost_region_count: int
    gained_region_count: int
    region_retention_rate: float | None
    thin_region_count: int
    thin_retained_region_count: int
    support_retention: DistributionSummary | None
    disagreement: DistributionSummary | None
    raw_support_points: DistributionSummary | None
    aggregate_support_points: DistributionSummary | None
    support_centroid_shift_m: DistributionSummary | None
    support_extent_change_m: DistributionSummary | None
    support_bounds_iou: DistributionSummary | None
    regions: tuple[RegionAssociationComparison, ...]


def compare_paired_associations(
    *,
    raw_supports: Mapping[PairedRegion, Sequence[GeometryReference]],
    aggregate_supports: Mapping[PairedRegion, Sequence[GeometryReference]],
    raw_geometry: GeometrySource,
    aggregated: AggregatedGeometry,
    membership: NDArray[Any],
    thin_support_max_points: int,
) -> AssociationStabilityReport:
    """Compare the support of the same regions associated on the raw map and on aggregates.

    Args:
        raw_supports: Support per region of the association over the raw map.
        aggregate_supports: Support per region of the association over the aggregates,
            same frames, regions and policies.
        raw_geometry: The raw map.
        aggregated: The aggregates the second association ran on.
        membership: Aggregate index of every raw point, from
            :meth:`~contextmap.geometric_mapping.VoxelAggregation.membership`.
        thin_support_max_points: Raw support size at or below which a region is thin.

    Returns:
        The stability report.

    Raises:
        ValueError: If a support names another map, the membership is not the raw map's,
            or the thin threshold is negative.
    """
    import numpy as np

    raw_map = raw_geometry.geometric_map
    aggregation = aggregated.aggregation
    if aggregation.source_map_id != raw_map.map_id:
        raise ValueError(
            f"the aggregates summarize map {aggregation.source_map_id!r}, not {raw_map.map_id!r}"
        )
    if (
        membership.shape != (raw_map.point_count,)
        or (membership < 0).any()
        or (membership >= len(aggregation)).any()
    ):
        raise ValueError("the membership does not give one aggregate for every raw point")
    if thin_support_max_points < 0:
        raise ValueError(f"thin_support_max_points must not be negative: {thin_support_max_points}")

    paired = sorted(set(raw_supports) & set(aggregate_supports))
    regions: list[RegionAssociationComparison] = []
    for region in paired:
        raw_indices = _indices(raw_supports[region], raw_map.map_id)
        aggregate_indices = _indices(aggregate_supports[region], aggregated.geometric_map.map_id)
        mapped = set(membership[raw_indices].tolist())
        supported = set(aggregate_indices.tolist())
        shared = len(mapped & supported)
        union = len(mapped | supported)
        raw_points = np.array(
            [raw_geometry.get(reference).coordinates_m for reference in raw_supports[region]],
            dtype=np.float64,
        ).reshape(-1, 3)
        centroids = aggregation.centroids_m[aggregate_indices]
        regions.append(
            RegionAssociationComparison(
                region=region,
                raw_support_count=len(raw_indices),
                mapped_raw_support_count=len(mapped),
                aggregate_support_count=len(supported),
                shared_aggregate_count=shared,
                support_retention=shared / len(mapped) if mapped else None,
                disagreement=1.0 - shared / union if union else None,
                **_support_geometry(raw_points, centroids),
            )
        )

    raw_supported = [r for r in regions if r.raw_support_count > 0]
    retained = [r for r in raw_supported if r.aggregate_support_count > 0]
    thin = [r for r in raw_supported if r.raw_support_count <= thin_support_max_points]
    return AssociationStabilityReport(
        thin_support_max_points=thin_support_max_points,
        paired_region_count=len(regions),
        raw_only_region_count=len(set(raw_supports) - set(aggregate_supports)),
        aggregate_only_region_count=len(set(aggregate_supports) - set(raw_supports)),
        raw_supported_region_count=len(raw_supported),
        retained_region_count=len(retained),
        lost_region_count=len(raw_supported) - len(retained),
        gained_region_count=sum(
            1 for r in regions if r.raw_support_count == 0 and r.aggregate_support_count > 0
        ),
        region_retention_rate=len(retained) / len(raw_supported) if raw_supported else None,
        thin_region_count=len(thin),
        thin_retained_region_count=sum(1 for r in thin if r.aggregate_support_count > 0),
        support_retention=_summary_of(r.support_retention for r in regions),
        disagreement=_summary_of(r.disagreement for r in regions),
        raw_support_points=_summary_of(float(r.raw_support_count) for r in regions),
        aggregate_support_points=_summary_of(float(r.aggregate_support_count) for r in regions),
        support_centroid_shift_m=_summary_of(r.support_centroid_shift_m for r in regions),
        support_extent_change_m=_summary_of(r.support_extent_change_m for r in regions),
        support_bounds_iou=_summary_of(r.support_bounds_iou for r in regions),
        regions=tuple(regions),
    )


def encode_aggregation_fidelity_report(report: AggregationFidelityReport) -> dict[str, Any]:
    """Serialize a fidelity report to a JSON-compatible record."""
    return {
        "aggregation_rule": report.aggregation_rule,
        "policy_fingerprint": report.policy_fingerprint,
        "cell_m": report.cell_m,
        "source_point_count": report.source_point_count,
        "aggregate_count": report.aggregate_count,
        "reduction_ratio": report.reduction_ratio,
        "distance_m": _encode_summary(report.distance_m),
        "mean_distance_m": report.mean_distance_m,
        "by_range": [
            {
                "minimum_range_m": band.minimum_range_m,
                "maximum_range_m": band.maximum_range_m,
                "point_count": band.point_count,
                "distance_m": _encode_summary(band.distance_m),
                "mean_distance_m": band.mean_distance_m,
            }
            for band in report.by_range
        ],
        "raw_bounds": _encode_bounds(report.raw_bounds),
        "centroid_bounds": _encode_bounds(report.centroid_bounds),
        "statistics_bounds_preserved": report.statistics_bounds_preserved,
        "max_bounds_shrinkage_m": report.max_bounds_shrinkage_m,
    }


def encode_association_stability_report(report: AssociationStabilityReport) -> dict[str, Any]:
    """Serialize a stability report, per-region comparisons included."""
    record: dict[str, Any] = {
        field.name: getattr(report, field.name)
        for field in dataclasses.fields(report)
        if field.name != "regions"
    }
    for name, value in record.items():
        if isinstance(value, DistributionSummary):
            record[name] = _encode_summary(value)
    record["regions"] = [
        {
            **dataclasses.asdict(comparison.region),
            **{
                field.name: getattr(comparison, field.name)
                for field in dataclasses.fields(comparison)
                if field.name != "region"
            },
        }
        for comparison in report.regions
    ]
    return record


def _support_geometry(raw_points: NDArray[Any], centroids: NDArray[Any]) -> dict[str, Any]:
    """Centroid shift, extent change and box IoU of a raw support against its aggregates."""
    import numpy as np

    if not len(raw_points) or not len(centroids):
        return {
            "support_centroid_shift_m": None,
            "support_extent_change_m": None,
            "support_bounds_iou": None,
        }
    raw_low, raw_high = raw_points.min(axis=0), raw_points.max(axis=0)
    low, high = centroids.min(axis=0), centroids.max(axis=0)
    raw_volume = float(np.prod(raw_high - raw_low))
    volume = float(np.prod(high - low))
    iou: float | None = None
    if raw_volume > 0 and volume > 0:
        overlap = float(
            np.prod(np.clip(np.minimum(raw_high, high) - np.maximum(raw_low, low), 0, None))
        )
        iou = overlap / (raw_volume + volume - overlap)
    return {
        "support_centroid_shift_m": float(
            np.linalg.norm(raw_points.mean(axis=0) - centroids.mean(axis=0))
        ),
        "support_extent_change_m": float(np.abs((raw_high - raw_low) - (high - low)).max()),
        "support_bounds_iou": iou,
    }


def _indices(references: Sequence[GeometryReference], map_id: MapId) -> NDArray[Any]:
    import numpy as np

    foreign = {str(reference.map_id) for reference in references} - {str(map_id)}
    if foreign:
        raise ValueError(f"a support references map {sorted(foreign)}, not {map_id!r}")
    return np.array(
        [
            geometry_index_of(map_id=map_id, geometry_id=reference.geometry_id)
            for reference in references
        ],
        dtype=np.int64,
    )


def _validated_edges(range_edges_m: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(float(edge) for edge in range_edges_m)
    if (
        not edges
        or edges[0] != 0.0
        or not all(math.isfinite(edge) for edge in edges)
        or any(high <= low for low, high in itertools.pairwise(edges))
    ):
        raise ValueError(
            f"range_edges_m must be finite, strictly increasing and start at 0, got {edges}"
        )
    return edges


def _band(edges: tuple[float, ...], band: int, distances: NDArray[Any]) -> RangeBandFidelity:
    return RangeBandFidelity(
        minimum_range_m=edges[band],
        maximum_range_m=edges[band + 1] if band + 1 < len(edges) else None,
        point_count=len(distances),
        distance_m=_summary(distances),
        mean_distance_m=float(distances.mean()) if len(distances) else None,
    )


def _sensor_origin_m(scan: ScanRecord) -> Vector3:
    """The sensor's origin in the map, through the scan's own chain, innermost first."""
    point: Vector3 = (0.0, 0.0, 0.0)
    for transform in reversed(scan.transform_chain):
        point = transform.apply(point)
    return point


def _scan_ordinals(scans: Sequence[ScanRecord], indices: NDArray[Any]) -> NDArray[Any]:
    import numpy as np

    holding = [scan for scan in scans if scan.geometry_count > 0]
    firsts = np.array([scan.first_geometry_index for scan in holding], dtype=np.int64)
    ordinals = np.array([scan.ordinal for scan in holding], dtype=np.int64)
    result: NDArray[Any] = ordinals[np.searchsorted(firsts, indices, side="right") - 1]
    return result


def _summary(values: NDArray[Any]) -> DistributionSummary | None:
    """Order statistics with the same p95 (nearest rank) as the rest of the evaluation."""
    import numpy as np

    if not len(values):
        return None
    ordered = np.sort(values)
    rank = math.ceil(0.95 * len(ordered))
    return DistributionSummary(
        count=len(ordered),
        minimum=float(ordered[0]),
        median=float(np.median(ordered)),
        p95=float(ordered[rank - 1]),
        maximum=float(ordered[-1]),
    )


def _summary_of(values: Iterable[float | None]) -> DistributionSummary | None:
    import numpy as np

    return _summary(np.array([value for value in values if value is not None], dtype=np.float64))


def _envelope(points: NDArray[Any], like: Bounds3D) -> Bounds3D:
    return Bounds3D(
        frame_id=like.frame_id,
        minimum_m=_triple(points.min(axis=0)),
        maximum_m=_triple(points.max(axis=0)),
    )


def _triple(values: Any) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _encode_summary(summary: DistributionSummary | None) -> dict[str, Any] | None:
    return None if summary is None else dataclasses.asdict(summary)


def _encode_bounds(bounds: Bounds3D) -> dict[str, Any]:
    return {
        "frame_id": str(bounds.frame_id),
        "minimum_m": list(bounds.minimum_m),
        "maximum_m": list(bounds.maximum_m),
    }
