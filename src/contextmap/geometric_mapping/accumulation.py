"""Accumulation of transformed scans into one persistent, referenceable map.

A :class:`MapAccumulator` takes transformed scans one at a time and streams their
geometry, packed, to a sink, so a long run never holds the map in memory. It keeps
what a map has to remember: the source index (one entry per scan, with its
lineage), the bounds, the time range and the point counts.

The map stays in **one global frame**. A scan expressed in another frame is
refused: processing only part of a sequence still yields coordinates in the same
frame, and a segment or submap frame may only ever be a derived view.

Geometry is only ever measured or *explicitly* aggregated. By default every
transformed point is persisted as one raw measurement. The optional
:class:`ScanVoxelPolicy` merges the points of **one scan** that fall in the same
voxel into their centroid; the merged point declares the rule and how many
measurements are behind it and carries no source point index. Points of different
scans are never merged, so no aggregation collapses source observations.
See ``src/contextmap/geometric_mapping/docs/accumulation.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

from contextmap.geometric_mapping.geometry_storage import (
    AGGREGATED_SOURCE_INDEX,
    PACKED_POINT,
    ScanRecord,
)
from contextmap.geometric_mapping.inputs import GeometryInputPlan
from contextmap.geometric_mapping.models import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    MapId,
    SpatialIndexMetadata,
)
from contextmap.geometric_mapping.transformation import TransformedScan, transform_scans
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import TimeBounds

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

PACKED_DTYPE_FIELDS = [
    ("map", "<f8", (3,)),
    ("source", "<f8", (3,)),
    ("scan", "<u4"),
    ("index", "<i8"),
    ("count", "<u4"),
]
"""NumPy spelling of :data:`~contextmap.geometric_mapping.geometry_storage.PACKED_POINT`."""

_SCAN_BOUNDS_INDEX = SpatialIndexMetadata(kind="scan_bounds", parameters={}, is_derived=True)


class AccumulationError(ValueError):
    """Raised when a scan or a map cannot be accumulated without breaking an invariant."""


@dataclass(frozen=True, kw_only=True)
class ScanVoxelPolicy:
    """Merge the points of one scan that share a voxel into their centroid.

    The grid is aligned to the map frame's origin, so the rule is a pure function
    of the coordinates. A voxel with a single point stays a raw measurement.

    Attributes:
        cell_m: Voxel edge length, in meters.
    """

    cell_m: float

    def __post_init__(self) -> None:
        """Require a usable voxel size.

        Raises:
            ValueError: If ``cell_m`` is not positive and finite.
        """
        if not (math.isfinite(self.cell_m) and self.cell_m > 0):
            raise ValueError(f"cell_m must be positive and finite, got {self.cell_m}")

    @property
    def rule(self) -> str:
        """The name persisted with every aggregated point, exact in the voxel size."""
        return f"scan-voxel-centroid-{self.cell_m!r}m"


@dataclass(frozen=True, kw_only=True)
class AccumulatedMap:
    """The outcome of accumulating scans; the geometry itself went to the sink.

    Attributes:
        geometric_map: The map's metadata.
        scans: The source index, one entry per accumulated scan.
        source_point_count: Finite points measured before any aggregation; with
            ``geometric_map.point_count`` it gives the reduction ratio.
    """

    geometric_map: GeometricMap
    scans: tuple[ScanRecord, ...]
    source_point_count: int


def map_provenance_from_plan(
    plan: GeometryInputPlan,
    *,
    configuration_fingerprint: str | None,
    code_version: str | None,
) -> GeometricMapProvenance:
    """Describe the run that assembled ``plan`` for the map built from it.

    Args:
        plan: The assembled geometry inputs.
        configuration_fingerprint: Hash of the effective mapping configuration.
        code_version: Code revision that builds the map.
    """
    return GeometricMapProvenance(
        sequence_artifact_id=plan.sequence_artifact_id,
        selection_id=plan.selection_id,
        trajectory_id=plan.trajectory_id,
        state_estimation_run_id=plan.state_estimation_run_id,
        calibration_identity=plan.calibration_identity,
        pose_lookup=plan.pose_lookup,
        configuration_fingerprint=configuration_fingerprint,
        code_version=code_version,
    )


def accumulate_plan(
    plan: GeometryInputPlan,
    *,
    map_id: MapId,
    sink: BinaryIO,
    aggregation: ScanVoxelPolicy | None,
    configuration_fingerprint: str | None,
    code_version: str | None,
) -> AccumulatedMap:
    """Transform and accumulate every scan of a plan, one at a time.

    Args:
        plan: The assembled geometry inputs.
        map_id: Identity of the map being built.
        sink: Where the packed geometry is written.
        aggregation: The explicit aggregation rule, or ``None`` to persist every
            point as a raw measurement.
        configuration_fingerprint: Hash of the effective mapping configuration.
        code_version: Code revision that builds the map.

    Returns:
        The map's metadata and source index.

    Raises:
        GeometryTransformError: If a scan cannot be transformed.
        AccumulationError: If the scans cannot form one map.
    """
    accumulator = MapAccumulator(
        map_id=map_id, map_frame=plan.map_frame, sink=sink, aggregation=aggregation
    )
    for scan in transform_scans(plan):
        accumulator.add_scan(scan)
    return accumulator.finish(
        provenance=map_provenance_from_plan(
            plan, configuration_fingerprint=configuration_fingerprint, code_version=code_version
        )
    )


class MapAccumulator:
    """Streams transformed scans into a packed geometry payload."""

    def __init__(
        self,
        *,
        map_id: MapId,
        map_frame: FrameId,
        sink: BinaryIO,
        aggregation: ScanVoxelPolicy | None,
    ) -> None:
        """Start a map.

        Args:
            map_id: Identity of the map; part of every geometry reference.
            map_frame: The one global frame every scan must already be in.
            sink: Writable binary stream that receives the packed geometry.
            aggregation: The explicit aggregation rule, or ``None`` to persist
                every point as a raw measurement.
        """
        self._map_id = map_id
        self._map_frame = map_frame
        self._sink = sink
        self._aggregation = aggregation
        self._scans: list[ScanRecord] = []
        self._seen: set[SourceObservationId] = set()
        self._geometry_count = 0
        self._source_point_count = 0
        self._minimum: list[float] | None = None
        self._maximum: list[float] | None = None
        self._earliest: SourceTimestamp | None = None
        self._latest: SourceTimestamp | None = None
        self._clock_id: str | None = None
        self._finished = False

    def add_scan(self, scan: TransformedScan) -> ScanRecord:
        """Append one scan's geometry and index entry.

        Args:
            scan: A scan already expressed in the map frame.

        Returns:
            The scan's source-index entry.

        Raises:
            AccumulationError: If the accumulator is finished, the scan is in
                another frame or clock domain, or it was already accumulated.
        """
        if self._finished:
            raise AccumulationError("the accumulator is finished and accepts no more scans")
        if scan.map_frame != self._map_frame:
            raise AccumulationError(
                f"scan {scan.observation_id!r} is in map frame {scan.map_frame!r} but this map "
                f"is in {self._map_frame!r}; a second global coordinate system is never created"
            )
        if scan.observation_id in self._seen:
            raise AccumulationError(f"scan {scan.observation_id!r} was already accumulated")
        clock_id = scan.acquisition_timestamp.clock_id
        if self._clock_id is not None and clock_id != self._clock_id:
            raise AccumulationError(
                f"scan {scan.observation_id!r} is in clock domain {clock_id!r} but the map's "
                f"scans are in {self._clock_id!r}; clock domains are never mixed"
            )

        ordinal = len(self._scans)
        geometry_count, bounds = self._write(scan, ordinal)
        record = ScanRecord(
            ordinal=ordinal,
            observation_id=scan.observation_id,
            source_frame=scan.source_frame,
            acquisition_timestamp=scan.acquisition_timestamp,
            payload_hash=scan.payload_hash,
            motion_correction=scan.motion_correction,
            transform_chain=scan.transform_chain,
            source_point_count=scan.source_point_count,
            dropped_non_finite_count=scan.dropped_non_finite_count,
            first_geometry_index=self._geometry_count,
            geometry_count=geometry_count,
            bounds=bounds,
        )
        self._scans.append(record)
        self._seen.add(scan.observation_id)
        self._clock_id = clock_id
        self._geometry_count += geometry_count
        self._source_point_count += scan.point_count
        if bounds is not None:
            self._extend(bounds, scan.acquisition_timestamp)
        return record

    def finish(self, *, provenance: GeometricMapProvenance) -> AccumulatedMap:
        """Close the map and describe it.

        Args:
            provenance: Run-level traceability of the map.

        Returns:
            The map's metadata and source index.

        Raises:
            AccumulationError: If no scan contributed any geometry.
        """
        if self._finished:
            raise AccumulationError("the accumulator is already finished")
        if (
            self._geometry_count == 0
            or self._minimum is None
            or self._maximum is None
            or self._earliest is None
            or self._latest is None
        ):
            raise AccumulationError("the scans produced no geometry, so there is no map to persist")
        self._finished = True
        geometric_map = GeometricMap(
            map_id=self._map_id,
            frame_id=self._map_frame,
            point_count=self._geometry_count,
            bounds=Bounds3D(
                frame_id=self._map_frame,
                minimum_m=(self._minimum[0], self._minimum[1], self._minimum[2]),
                maximum_m=(self._maximum[0], self._maximum[1], self._maximum[2]),
            ),
            source_observation_ids=tuple(
                scan.observation_id for scan in self._scans if scan.geometry_count > 0
            ),
            time_bounds=TimeBounds(start=self._earliest, end=self._latest),
            spatial_index=_SCAN_BOUNDS_INDEX,
            provenance=provenance,
            aggregation_rule=None if self._aggregation is None else self._aggregation.rule,
        )
        return AccumulatedMap(
            geometric_map=geometric_map,
            scans=tuple(self._scans),
            source_point_count=self._source_point_count,
        )

    def _extend(self, bounds: Bounds3D, timestamp: SourceTimestamp) -> None:
        low, high = list(bounds.minimum_m), list(bounds.maximum_m)
        if self._minimum is None or self._maximum is None:
            self._minimum, self._maximum = low, high
        else:
            self._minimum = [min(a, b) for a, b in zip(self._minimum, low, strict=True)]
            self._maximum = [max(a, b) for a, b in zip(self._maximum, high, strict=True)]
        if (
            self._earliest is None
            or timestamp.total_nanoseconds() < self._earliest.total_nanoseconds()
        ):
            self._earliest = timestamp
        if self._latest is None or timestamp.total_nanoseconds() > self._latest.total_nanoseconds():
            self._latest = timestamp

    def _write(self, scan: TransformedScan, ordinal: int) -> tuple[int, Bounds3D | None]:
        if scan.point_count == 0:
            return 0, None
        import numpy as np

        map_xyz = np.frombuffer(scan.map_coordinates_m, dtype=np.float64).reshape(-1, 3)
        source_xyz = np.frombuffer(scan.source_coordinates_m, dtype=np.float64).reshape(-1, 3)
        indices = np.frombuffer(scan.source_point_indices, dtype=np.int64)
        counts = np.ones(len(indices), dtype=np.uint32)
        if self._aggregation is not None:
            map_xyz, source_xyz, indices, counts = _voxel_centroids(
                map_xyz, source_xyz, indices, self._aggregation.cell_m
            )

        records = np.empty(len(map_xyz), dtype=np.dtype(PACKED_DTYPE_FIELDS))
        records["map"] = map_xyz
        records["source"] = source_xyz
        records["scan"] = ordinal
        records["index"] = indices
        records["count"] = counts
        payload = records.tobytes()
        if len(payload) != len(map_xyz) * PACKED_POINT.size:
            raise AccumulationError("the packed record layout disagrees with the documented one")
        self._sink.write(payload)

        low, high = map_xyz.min(axis=0), map_xyz.max(axis=0)
        bounds = Bounds3D(
            frame_id=self._map_frame,
            minimum_m=(float(low[0]), float(low[1]), float(low[2])),
            maximum_m=(float(high[0]), float(high[1]), float(high[2])),
        )
        return len(map_xyz), bounds


def _voxel_centroids(
    map_xyz: NDArray[np.float64],
    source_xyz: NDArray[np.float64],
    indices: NDArray[np.int64],
    cell_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.uint32]]:
    """Merge the points of one scan by voxel, in a deterministic voxel order.

    Returns the centroids of the map and of the source coordinates, the source
    point index of a voxel that kept one point (``-1`` for a merged one) and the
    number of points in each voxel. The rigid chain is affine, so the centroid of
    the source points maps to the centroid of the map points.
    """
    import numpy as np

    voxels = np.floor(map_xyz / cell_m).astype(np.int64)
    # lexsort é estável: dentro de um voxel os pontos seguem a ordem original do scan.
    order = np.lexsort((voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    sorted_voxels = voxels[order]
    starts_new_voxel = np.empty(len(order), dtype=bool)
    starts_new_voxel[0] = True
    starts_new_voxel[1:] = (sorted_voxels[1:] != sorted_voxels[:-1]).any(axis=1)
    group_of_sorted = np.cumsum(starts_new_voxel) - 1
    group = np.empty(len(order), dtype=np.int64)
    group[order] = group_of_sorted
    first_of_group = order[np.flatnonzero(starts_new_voxel)]
    counts = np.bincount(group)

    def centroid(values: NDArray[np.float64]) -> NDArray[np.float64]:
        sums = [np.bincount(group, weights=values[:, axis]) for axis in range(3)]
        return np.column_stack(sums) / counts[:, None]

    kept_index = np.where(counts == 1, indices[first_of_group], AGGREGATED_SOURCE_INDEX)
    return (
        centroid(map_xyz),
        centroid(source_xyz),
        kept_index.astype(np.int64),
        counts.astype(np.uint32),
    )
