"""An in-memory GeometrySource: a lightweight stand-in for the persisted map."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    MapId,
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

_CLOCK = "fixture:header"
_STAMP = SourceTimestamp(seconds=1, nanoseconds=0, clock_id=_CLOCK)


def _lineage(map_frame: str) -> TransformLineage:
    return TransformLineage(
        steps=(
            TransformStep(
                kind=TransformKind.DYNAMIC_POSE,
                parent_frame=FrameId(map_frame),
                child_frame=FrameId("body"),
                reference="traj--pose-000007",
                source_estimate_ids=(PoseEstimateId("traj--pose-000007"),),
            ),
            TransformStep(
                kind=TransformKind.STATIC_CALIBRATION,
                parent_frame=FrameId("body"),
                child_frame=FrameId("lidar"),
                reference="sha256:calibration",
            ),
        )
    )


class InMemoryGeometrySource:
    """Serves the points of one map, keyed by their index in the map."""

    def __init__(
        self,
        map_id: MapId,
        coordinates_by_index: Mapping[int, Vector3],
        *,
        frame: str = "map",
        source_observation: str = "lidar-frame-01824",
    ) -> None:
        self._points = {
            geometry_id_for(map_id=map_id, index=index): GeometryPoint(
                geometry_id=geometry_id_for(map_id=map_id, index=index),
                map_id=map_id,
                map_frame=FrameId(frame),
                coordinates_m=coordinates,
                source_frame=FrameId("lidar"),
                source_coordinates_m=(0.0, 0.0, 0.0),
                source_observation_id=SourceObservationId(source_observation),
                source_point_index=index,
                acquisition_timestamp=_STAMP,
                transform_lineage=_lineage(frame),
                provenance=GeometryPointProvenance(),
            )
            for index, coordinates in sorted(coordinates_by_index.items())
        }
        self._map = GeometricMap(
            map_id=map_id,
            frame_id=FrameId(frame),
            point_count=len(self._points),
            bounds=Bounds3D.enclosing(coordinates_by_index.values(), frame_id=FrameId(frame)),
            source_observation_ids=(SourceObservationId(source_observation),),
            time_bounds=TimeBounds(start=_STAMP, end=_STAMP),
            spatial_index=None,
            provenance=GeometricMapProvenance(
                sequence_artifact_id=SequenceArtifactId("sequence-0001"),
                selection_id="full-sequence",
                trajectory_id=TrajectoryId("traj"),
                state_estimation_run_id=StateEstimationRunId("run-0001"),
                calibration_identity="sha256:calibration",
                pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=250_000_000),
                configuration_fingerprint="sha256:mapping-config",
                code_version="test",
            ),
        )

    @property
    def geometric_map(self) -> GeometricMap:
        return self._map

    def get(self, reference: GeometryReference) -> GeometryPoint:
        if reference.map_id != self._map.map_id:
            raise KeyError(reference)
        return self._points[reference.geometry_id]

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        return iter(self._points.values())

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        return (
            point
            for point in self._points.values()
            if bounds.contains(point.coordinates_m, frame_id=bounds.frame_id)
        )
