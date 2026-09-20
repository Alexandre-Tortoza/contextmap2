import dataclasses
import math
import subprocess
import sys
from collections.abc import Iterator

import pytest
from geometry_builders import (
    MAP_ID,
    make_bounds,
    make_lineage,
    make_map,
    make_point,
)

import contextmap.geometric_mapping as geometric_mapping
from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    GeometrySource,
    MapId,
    PointOrigin,
    SpatialIndexMetadata,
    TransformKind,
    TransformLineage,
    geometry_id_for,
)
from contextmap.ingestion import FrameId
from contextmap.state_estimation import PoseEstimateId


def test_public_api_exports_resolve() -> None:
    for name in geometric_mapping.__all__:
        assert hasattr(geometric_mapping, name), name


def test_the_contracts_are_readable_without_robotics_or_model_libraries() -> None:
    code = (
        "import sys, contextmap.geometric_mapping;"
        "bad = [m for m in ('numpy', 'rosbags', 'torch', 'open3d') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


# --- Bounds3D ---------------------------------------------------------------


def test_bounds_declare_the_frame_they_are_expressed_in() -> None:
    bounds = make_bounds(frame="map")

    assert bounds.frame_id == FrameId("map")


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (1.0, -1.0, 1.0)),
        ((math.nan, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (math.inf, 1.0, 1.0)),
    ],
)
def test_bounds_reject_inverted_or_non_finite_extents(
    minimum: tuple[float, float, float], maximum: tuple[float, float, float]
) -> None:
    with pytest.raises(ValueError, match="bounds"):
        make_bounds(minimum, maximum)


def test_bounds_reject_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="frame"):
        make_bounds(frame="")


def test_containment_is_inclusive_on_every_face() -> None:
    bounds = make_bounds((0.0, 0.0, 0.0), (10.0, 5.0, 3.0))
    frame = FrameId("map")

    assert bounds.contains((0.0, 0.0, 0.0), frame_id=frame)
    assert bounds.contains((10.0, 5.0, 3.0), frame_id=frame)
    assert bounds.contains((5.0, 2.5, 1.5), frame_id=frame)
    assert not bounds.contains((10.0001, 2.0, 1.0), frame_id=frame)
    assert not bounds.contains((5.0, -0.0001, 1.0), frame_id=frame)


def test_containment_never_infers_the_frame() -> None:
    with pytest.raises(ValueError, match="frame"):
        make_bounds().contains((1.0, 1.0, 1.0), frame_id=FrameId("lidar"))


def test_intersection_is_inclusive_and_needs_the_same_frame() -> None:
    box = make_bounds((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))

    assert box.intersects(make_bounds((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)))
    assert not box.intersects(make_bounds((2.1, 0.0, 0.0), (3.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="frame"):
        box.intersects(make_bounds(frame="lidar"))


def test_enclosing_bounds_are_the_tight_axis_aligned_box() -> None:
    bounds = Bounds3D.enclosing(
        [(1.0, 5.0, -2.0), (4.0, -1.0, 0.0), (2.0, 2.0, 9.0)], frame_id=FrameId("map")
    )

    assert bounds.minimum_m == (1.0, -1.0, -2.0)
    assert bounds.maximum_m == (4.0, 5.0, 9.0)
    with pytest.raises(ValueError, match="at least one"):
        Bounds3D.enclosing([], frame_id=FrameId("map"))


# --- Identity and references ------------------------------------------------


def test_geometry_ids_are_deterministic_and_scoped_to_their_map() -> None:
    first = geometry_id_for(map_id=MAP_ID, index=7)

    assert first == geometry_id_for(map_id=MAP_ID, index=7)
    assert first != geometry_id_for(map_id=MAP_ID, index=8)
    assert first != geometry_id_for(map_id=MapId("other-map"), index=7)


def test_a_point_is_referenced_by_map_and_geometry_identity_alone() -> None:
    point = make_point(3)

    reference = point.reference

    assert reference == GeometryReference(map_id=MAP_ID, geometry_id=point.geometry_id)
    assert {reference: "x"}[GeometryReference(map_id=MAP_ID, geometry_id=point.geometry_id)] == "x"
    assert {field.name for field in dataclasses.fields(GeometryReference)} == {
        "map_id",
        "geometry_id",
    }


# --- GeometryPoint ----------------------------------------------------------


def test_map_and_source_coordinates_are_distinct_and_carry_their_frames() -> None:
    point = make_point(
        coordinates_m=(18.41, 3.82, 1.24),
        source_coordinates_m=(4.21, -0.71, 0.32),
        map_frame="map",
        source_frame="lidar",
    )

    assert point.coordinates_m == (18.41, 3.82, 1.24)
    assert point.source_coordinates_m == (4.21, -0.71, 0.32)
    assert (point.map_frame, point.source_frame) == (FrameId("map"), FrameId("lidar"))
    assert point.source_observation_id == "lidar-frame-01824"
    assert point.source_point_index == 713


def test_the_source_point_index_is_optional_but_never_negative() -> None:
    assert make_point(source_point_index=None).source_point_index is None
    with pytest.raises(ValueError, match="source_point_index"):
        make_point(source_point_index=-1)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_coordinates_must_be_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="coordinates_m"):
        make_point(coordinates_m=(bad, 0.0, 0.0))
    with pytest.raises(ValueError, match="source_coordinates_m"):
        make_point(source_coordinates_m=(0.0, bad, 0.0))


def test_identities_and_frames_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="frame"):
        make_point(map_frame="")
    with pytest.raises(ValueError, match="map_id"):
        make_point(map_id=MapId(""))


def test_the_lineage_must_connect_the_map_frame_to_the_source_frame() -> None:
    with pytest.raises(ValueError, match="lineage"):
        make_point(lineage=make_lineage(map_frame="odom"))
    with pytest.raises(ValueError, match="lineage"):
        make_point(lineage=make_lineage(source_frame="camera"))


def test_a_point_already_in_the_map_frame_needs_no_transform_steps() -> None:
    point = make_point(map_frame="map", source_frame="map", lineage=TransformLineage(steps=()))

    assert point.transform_lineage.steps == ()
    with pytest.raises(ValueError, match="lineage"):
        make_point(map_frame="map", source_frame="lidar", lineage=TransformLineage(steps=()))


def test_the_transform_chain_must_be_contiguous() -> None:
    steps = make_lineage().steps
    broken = dataclasses.replace(steps[1], parent_frame=FrameId("imu"))

    with pytest.raises(ValueError, match="contiguous"):
        TransformLineage(steps=(steps[0], broken))


def test_a_dynamic_step_names_the_poses_it_came_from_and_a_static_one_names_none() -> None:
    dynamic, static = make_lineage().steps

    assert dynamic.kind is TransformKind.DYNAMIC_POSE
    assert dynamic.source_estimate_ids == (PoseEstimateId("traj--pose-000007"),)
    with pytest.raises(ValueError, match="source_estimate_ids"):
        dataclasses.replace(dynamic, source_estimate_ids=())
    with pytest.raises(ValueError, match="source_estimate_ids"):
        dataclasses.replace(static, source_estimate_ids=(PoseEstimateId("x"),))


def test_an_interpolated_pose_keeps_both_of_its_source_estimates() -> None:
    lineage = make_lineage(pose_ids=("traj--pose-000007", "traj--pose-000008"))

    assert len(lineage.steps[0].source_estimate_ids) == 2


def test_a_step_needs_distinct_frames_and_a_reference() -> None:
    _, static = make_lineage().steps

    with pytest.raises(ValueError, match="frame"):
        dataclasses.replace(static, child_frame=static.parent_frame)
    with pytest.raises(ValueError, match="reference"):
        dataclasses.replace(static, reference="")


def test_a_measured_point_is_the_default_and_names_no_aggregation() -> None:
    provenance = make_point().provenance

    assert provenance.origin is PointOrigin.MEASURED
    assert provenance.aggregation_rule is None
    assert provenance.contributing_point_count == 1


def test_an_aggregated_point_declares_its_rule_and_is_not_a_single_raw_measurement() -> None:
    aggregated = GeometryPointProvenance(
        origin=PointOrigin.AGGREGATED,
        aggregation_rule="voxel-centroid-0.05m",
        contributing_point_count=12,
    )

    point = make_point(source_point_index=None, provenance=aggregated)

    assert point.provenance.origin is PointOrigin.AGGREGATED
    with pytest.raises(ValueError, match="aggregated"):
        make_point(source_point_index=713, provenance=aggregated)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "origin": PointOrigin.MEASURED,
            "aggregation_rule": "voxel",
            "contributing_point_count": 1,
        },
        {"origin": PointOrigin.MEASURED, "contributing_point_count": 3},
        {"origin": PointOrigin.AGGREGATED, "contributing_point_count": 3},
        {
            "origin": PointOrigin.AGGREGATED,
            "aggregation_rule": "voxel",
            "contributing_point_count": 1,
        },
    ],
)
def test_provenance_rejects_a_point_that_misdescribes_how_it_was_produced(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"provenance|aggregation|contributing"):
        GeometryPointProvenance(**kwargs)  # type: ignore[arg-type]


def test_geometry_carries_no_semantic_state() -> None:
    forbidden = {
        "label",
        "labels",
        "claim",
        "claims",
        "entity",
        "entity_id",
        "embedding",
        "feature",
    }

    for contract in (GeometryPoint, GeometricMap, GeometryReference):
        assert not forbidden & {field.name for field in dataclasses.fields(contract)}


def test_a_point_is_immutable() -> None:
    point = make_point()

    with pytest.raises(dataclasses.FrozenInstanceError):
        point.coordinates_m = (0.0, 0.0, 0.0)  # type: ignore[misc]


# --- GeometricMap -----------------------------------------------------------


def test_a_map_declares_its_frame_bounds_sources_and_provenance() -> None:
    geometric_map = make_map()

    assert geometric_map.frame_id == FrameId("map")
    assert geometric_map.bounds.frame_id == geometric_map.frame_id
    assert geometric_map.point_count == 1200
    assert geometric_map.time_bounds.duration_ns == 1_000_000_000
    assert geometric_map.provenance.trajectory_id == "run-0001--trajectory"
    assert geometric_map.provenance.calibration_identity == "sha256:calibration"
    assert geometric_map.spatial_index is None


def test_the_map_bounds_must_be_in_the_global_map_frame() -> None:
    with pytest.raises(ValueError, match="frame"):
        make_map(bounds=make_bounds(frame="lidar"))


@pytest.mark.parametrize("count", [0, -1])
def test_a_map_needs_at_least_one_point(count: int) -> None:
    with pytest.raises(ValueError, match="point_count"):
        make_map(point_count=count)


def test_a_map_needs_unique_source_observations() -> None:
    with pytest.raises(ValueError, match="source_observation_ids"):
        make_map(source_observation_ids=())
    with pytest.raises(ValueError, match="source_observation_ids"):
        make_map(source_observation_ids=("scan-1", "scan-1"))


def test_a_derived_spatial_index_is_distinguished_from_authoritative_geometry() -> None:
    index = SpatialIndexMetadata(kind="voxel_hash", parameters={"cell_m": 0.5}, is_derived=True)

    geometric_map = make_map(spatial_index=index)

    assert geometric_map.spatial_index is not None
    assert geometric_map.spatial_index.is_derived
    with pytest.raises(ValueError, match="kind"):
        SpatialIndexMetadata(kind="", parameters={}, is_derived=True)


def test_a_map_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_map().point_count = 2  # type: ignore[misc]


# --- GeometrySource (the read boundary other capabilities depend on) --------


class _ListGeometrySource:
    def __init__(self, geometric_map: GeometricMap, points: list[GeometryPoint]) -> None:
        self._map = geometric_map
        self._points = points

    @property
    def geometric_map(self) -> GeometricMap:
        return self._map

    def get(self, reference: GeometryReference) -> GeometryPoint:
        return next(p for p in self._points if p.reference == reference)

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        return iter(self._points)

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        return (p for p in self._points if bounds.contains(p.coordinates_m, frame_id=p.map_frame))


def test_an_implementation_of_the_read_boundary_is_recognized() -> None:
    points = [make_point(0), make_point(1, coordinates_m=(100.0, 0.0, 0.0))]
    source = _ListGeometrySource(make_map(point_count=2), points)

    assert isinstance(source, GeometrySource)
    assert source.get(points[1].reference) == points[1]
    assert list(source.query_bounds(make_bounds((0.0, 0.0, 0.0), (50.0, 50.0, 50.0)))) == [
        points[0]
    ]
    assert not isinstance(object(), GeometrySource)


def test_an_aggregation_rule_that_is_set_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="aggregation_rule"):
        make_map(aggregation_rule="")
