"""Deterministic builders for Geometric Mapping tests."""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryId,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    MapId,
    PointOrigin,
    SpatialIndexMetadata,
    TransformKind,
    TransformLineage,
    TransformStep,
    geometry_id_for,
)
from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import SourceTimestamp, Vector3
from contextmap.state_estimation import (
    LookupPolicy,
    PoseEstimateId,
    StateEstimationRunId,
    TimeBounds,
    TrajectoryId,
)

MAP_ID = MapId("map-0001")
CLOCK_ID = "fixture:header"


def timestamp(total_nanoseconds: int) -> SourceTimestamp:
    seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=CLOCK_ID)


def make_lineage(
    *,
    map_frame: str = "map",
    body_frame: str = "body",
    source_frame: str = "lidar",
    pose_ids: Sequence[str] = ("traj--pose-000007",),
) -> TransformLineage:
    """``T_map_body(t)`` (dynamic) followed by ``T_body_lidar`` (static)."""
    return TransformLineage(
        steps=(
            TransformStep(
                kind=TransformKind.DYNAMIC_POSE,
                parent_frame=FrameId(map_frame),
                child_frame=FrameId(body_frame),
                reference="traj--pose-000007",
                source_estimate_ids=tuple(PoseEstimateId(p) for p in pose_ids),
            ),
            TransformStep(
                kind=TransformKind.STATIC_CALIBRATION,
                parent_frame=FrameId(body_frame),
                child_frame=FrameId(source_frame),
                reference="sha256:calibration",
            ),
        )
    )


def make_point(
    index: int = 0,
    *,
    map_id: MapId = MAP_ID,
    coordinates_m: Vector3 = (18.41, 3.82, 1.24),
    source_coordinates_m: Vector3 = (4.21, -0.71, 0.32),
    map_frame: str = "map",
    source_frame: str = "lidar",
    source_point_index: int | None = 713,
    lineage: TransformLineage | None = None,
    provenance: GeometryPointProvenance | None = None,
) -> GeometryPoint:
    return GeometryPoint(
        geometry_id=geometry_id_for(map_id=map_id, index=index),
        map_id=map_id,
        map_frame=FrameId(map_frame),
        coordinates_m=coordinates_m,
        source_frame=FrameId(source_frame),
        source_coordinates_m=source_coordinates_m,
        source_observation_id=SourceObservationId("lidar-frame-01824"),
        source_point_index=source_point_index,
        acquisition_timestamp=timestamp(1_500_000_000),
        transform_lineage=lineage if lineage is not None else make_lineage(),
        provenance=provenance if provenance is not None else GeometryPointProvenance(),
    )


def make_bounds(
    minimum_m: Vector3 = (0.0, 0.0, 0.0),
    maximum_m: Vector3 = (10.0, 5.0, 3.0),
    *,
    frame: str = "map",
) -> Bounds3D:
    return Bounds3D(frame_id=FrameId(frame), minimum_m=minimum_m, maximum_m=maximum_m)


def make_map_provenance() -> GeometricMapProvenance:
    return GeometricMapProvenance(
        sequence_artifact_id=SequenceArtifactId("sequence-0001"),
        selection_id="full-sequence",
        trajectory_id=TrajectoryId("run-0001--trajectory"),
        state_estimation_run_id=StateEstimationRunId("run-0001"),
        calibration_identity="sha256:calibration",
        pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=250_000_000),
        configuration_fingerprint="sha256:mapping-config",
        code_version="test",
    )


def make_map(
    *,
    map_id: MapId = MAP_ID,
    frame: str = "map",
    point_count: int = 1200,
    bounds: Bounds3D | None = None,
    source_observation_ids: Sequence[str] = ("lidar-frame-01824", "lidar-frame-01825"),
    spatial_index: SpatialIndexMetadata | None = None,
    aggregation_rule: str | None = None,
) -> GeometricMap:
    return GeometricMap(
        map_id=map_id,
        frame_id=FrameId(frame),
        point_count=point_count,
        bounds=bounds if bounds is not None else make_bounds(frame=frame),
        source_observation_ids=tuple(SourceObservationId(i) for i in source_observation_ids),
        time_bounds=TimeBounds(start=timestamp(1_000_000_000), end=timestamp(2_000_000_000)),
        spatial_index=spatial_index,
        provenance=make_map_provenance(),
        aggregation_rule=aggregation_rule,
    )


__all__ = [
    "GeometryId",
    "GeometryReference",
    "PointOrigin",
    "make_bounds",
    "make_lineage",
    "make_map",
    "make_map_provenance",
    "make_point",
    "timestamp",
]
