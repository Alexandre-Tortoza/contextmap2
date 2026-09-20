"""Deterministic transformed scans and accumulated maps for Geometric Mapping tests."""

from __future__ import annotations

import io
from collections.abc import Sequence

from input_builders import MS, assemble_plan, make_calibration, make_trajectory, rigid
from lidar_builders import DEFAULT_POINTS, make_scan

from contextmap.geometric_mapping import (
    AccumulatedMap,
    GeometricMapProvenance,
    MapAccumulator,
    MapId,
    PackedGeometry,
    ScanVoxelPolicy,
    TransformedScan,
    map_provenance_from_plan,
    transform_scans,
)
from contextmap.ingestion import FrameId, SourceObservation
from contextmap.shared import Vector3

MAP_ID = MapId("map-0001")


def make_scans(
    count: int = 3,
    *,
    points: Sequence[Vector3] = DEFAULT_POINTS,
    first_scan: int = 0,
) -> list[TransformedScan]:
    """Transform ``count`` scans, one per pose, 100 ms apart.

    Scan ``k`` is at ``k * 100 ms``, where the pose translates ``k`` meters along
    +x, and the static extrinsic adds ``(0.5, 0, 0.25)``, so the point
    ``(x, y, z)`` lands at ``(x + 0.5 + k, y, z + 0.25)``.
    """
    calibration = make_calibration((rigid("body", "lidar", (0.5, 0.0, 0.25)),))
    scans: list[SourceObservation] = [
        make_scan(f"scan-{index:04d}", time_ns=index * 100 * MS, points=points)
        for index in range(first_scan, first_scan + count)
    ]
    plan = assemble_plan(
        scans,
        calibration=calibration,
        trajectory=make_trajectory(count=max(5, first_scan + count), calibration=calibration),
    )
    return list(transform_scans(plan))


def make_provenance() -> GeometricMapProvenance:
    calibration = make_calibration((rigid("body", "lidar", (0.5, 0.0, 0.25)),))
    plan = assemble_plan(
        [make_scan()], calibration=calibration, trajectory=make_trajectory(calibration=calibration)
    )
    return map_provenance_from_plan(
        plan, configuration_fingerprint="sha256:mapping-config", code_version="test"
    )


def accumulate(
    scans: Sequence[TransformedScan],
    *,
    map_id: MapId = MAP_ID,
    map_frame: str = "map",
    aggregation: ScanVoxelPolicy | None = None,
) -> tuple[AccumulatedMap, bytes]:
    """Accumulate ``scans`` into an in-memory sink; return the result and the packed bytes."""
    sink = io.BytesIO()
    accumulator = MapAccumulator(
        map_id=map_id, map_frame=FrameId(map_frame), sink=sink, aggregation=aggregation
    )
    for scan in scans:
        accumulator.add_scan(scan)
    return accumulator.finish(provenance=make_provenance()), sink.getvalue()


def open_geometry(accumulated: AccumulatedMap, packed: bytes) -> PackedGeometry:
    return PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=accumulated.scans, records=packed
    )
