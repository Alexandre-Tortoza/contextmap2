"""Raw maps whose scans revisit the same place, for inter-scan aggregation tests."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

from input_builders import MS, assemble_plan, make_calibration, make_trajectory, rigid
from lidar_builders import make_scan
from map_builders import MAP_ID, make_provenance

from contextmap.geometric_mapping import (
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    GeometryInputPlan,
    InterScanVoxelPolicy,
    MapAccumulator,
    PackedGeometry,
    VoxelGridSpec,
    transform_scans,
)
from contextmap.ingestion import FrameId, SourceObservation
from contextmap.shared import Vector3

EXTRINSIC_M: Vector3 = (0.5, 0.0, 0.25)
"""The static body→lidar translation of :func:`revisit_map`."""


def unit_grid(
    *, cell_m: float = 1.0, origin_m: Vector3 = (0.0, 0.0, 0.0), frame: str = "map"
) -> VoxelGridSpec:
    return VoxelGridSpec(frame_id=FrameId(frame), origin_m=origin_m, cell_m=cell_m)


def unit_policy(
    *, cell_m: float = 1.0, origin_m: Vector3 = (0.0, 0.0, 0.0), frame: str = "map"
) -> InterScanVoxelPolicy:
    return InterScanVoxelPolicy(grid=unit_grid(cell_m=cell_m, origin_m=origin_m, frame=frame))


def revisit_plan(map_points_by_scan: Sequence[Sequence[Vector3]]) -> GeometryInputPlan:
    """Assemble scans where scan ``k`` persists (about) ``map_points_by_scan[k]``.

    Scan ``k`` is acquired at ``k * 100 ms`` from a pose translated ``k`` meters along +x,
    so its source points are placed back by ``(k, 0, 0)`` and the extrinsic: several scans
    can then hit the same voxel, which is the revisit an inter-scan aggregation merges.
    Coordinates go through the real transform chain in float64, so tests read the
    persisted coordinates back instead of trusting the requested ones.
    """
    calibration = make_calibration((rigid("body", "lidar", EXTRINSIC_M),))
    observations: list[SourceObservation] = [
        make_scan(
            f"scan-{index:04d}",
            time_ns=index * 100 * MS,
            points=[
                (x - EXTRINSIC_M[0] - index, y - EXTRINSIC_M[1], z - EXTRINSIC_M[2])
                for x, y, z in points
            ],
            double_precision=True,
        )
        for index, points in enumerate(map_points_by_scan)
    ]
    return assemble_plan(
        observations,
        calibration=calibration,
        trajectory=make_trajectory(count=max(5, len(map_points_by_scan)), calibration=calibration),
    )


def revisit_map(map_points_by_scan: Sequence[Sequence[Vector3]]) -> PackedGeometry:
    """Accumulate :func:`revisit_plan` into an in-memory raw map."""
    sink = io.BytesIO()
    accumulator = MapAccumulator(
        map_id=MAP_ID, map_frame=FrameId("map"), sink=sink, aggregation=None
    )
    for scan in transform_scans(revisit_plan(map_points_by_scan)):
        accumulator.add_scan(scan)
    accumulated = accumulator.finish(provenance=make_provenance())
    return PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=accumulated.scans, records=sink.getvalue()
    )


def write_raw_artifact(directory: Path, map_points_by_scan: Sequence[Sequence[Vector3]]) -> Path:
    """Persist :func:`revisit_plan` as a raw Geometric Mapping artifact at ``directory``."""
    GeometricMapArtifactWriter(
        output_dir=directory,
        sequence_name="revisit",
        run_id=GeometricMapRunId("map-run-0001"),
        run_index=1,
    ).finalize(plan=revisit_plan(map_points_by_scan), aggregation=None, code_version="test")
    return directory
