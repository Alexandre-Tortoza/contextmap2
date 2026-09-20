"""Deterministic LiDAR scan builders for Geometric Mapping tests."""

from __future__ import annotations

import struct
from collections.abc import Sequence

from contextmap.ingestion import (
    FrameId,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp, Vector3

CLOCK_ID = "fixture:header"
DEFAULT_POINTS: tuple[Vector3, ...] = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.5))


def timestamp_ns(total_nanoseconds: int, *, clock_id: str = CLOCK_ID) -> SourceTimestamp:
    seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)


def make_scan(
    observation_id: str = "scan-0001",
    *,
    time_ns: int = 0,
    frame: str = "lidar",
    sensor: str = "velodyne",
    points: Sequence[Vector3] = DEFAULT_POINTS,
    clock_id: str = CLOCK_ID,
    intensity: bool = False,
    double_precision: bool = False,
    is_dense: bool = True,
) -> LidarObservation:
    """Build a scan of ``points`` (x, y, z) packed little-endian, optionally with intensity."""
    scalar = PointFieldDataType.FLOAT64 if double_precision else PointFieldDataType.FLOAT32
    code, size = ("d", 8) if double_precision else ("f", 4)
    names = ["x", "y", "z"] + (["intensity"] if intensity else [])
    fields = tuple(
        PointFieldDescriptor(name=name, offset_bytes=i * size, data_type=scalar)
        for i, name in enumerate(names)
    )
    rows = []
    for x, y, z in points:
        values = [x, y, z] + ([100.0] if intensity else [])
        rows.append(struct.pack(f"<{len(values)}{code}", *values))
    return LidarObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId(sensor),
        frame_id=FrameId(frame),
        timestamp=timestamp_ns(time_ns, clock_id=clock_id),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/lidar"),
        point_count=len(points),
        point_step_bytes=len(names) * size,
        fields=fields,
        data=b"".join(rows),
        is_dense=is_dense,
    )
