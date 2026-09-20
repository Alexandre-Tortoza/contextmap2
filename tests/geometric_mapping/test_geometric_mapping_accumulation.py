import dataclasses
import io
import math

import pytest
from input_builders import MS, assemble_plan, make_calibration, make_trajectory, rigid
from lidar_builders import make_scan, timestamp_ns
from map_builders import MAP_ID, accumulate, make_provenance, make_scans, open_geometry

from contextmap.geometric_mapping import (
    AccumulationError,
    Bounds3D,
    MapAccumulator,
    MotionCorrectionState,
    PointOrigin,
    ScanVoxelPolicy,
    accumulate_plan,
    geometry_id_for,
    transform_scans,
)
from contextmap.geometric_mapping.geometry_storage import PACKED_POINT
from contextmap.ingestion import FrameId

TOLERANCE_M = 1e-6
NAN = float("nan")


# --- One immutable map with stable references -------------------------------------


def test_scans_accumulate_into_one_map_with_sequential_stable_references() -> None:
    scans = make_scans(3)
    accumulated, packed = accumulate(scans)
    geometry = open_geometry(accumulated, packed)

    assert accumulated.geometric_map.point_count == 6
    for scan_index, scan in enumerate(scans):
        for position in range(scan.point_count):
            geometry_index = 2 * scan_index + position
            expected = scan.geometry_point(position, map_id=MAP_ID, geometry_index=geometry_index)
            reference = expected.reference
            assert reference.geometry_id == geometry_id_for(map_id=MAP_ID, index=geometry_index)
            assert geometry.get(reference) == expected


def test_the_same_inputs_reproduce_the_same_bytes_and_references() -> None:
    first, first_bytes = accumulate(make_scans(3))
    second, second_bytes = accumulate(make_scans(3))

    assert first_bytes == second_bytes
    assert first == second
    assert list(open_geometry(first, first_bytes).iter_geometry()) == list(
        open_geometry(second, second_bytes).iter_geometry()
    )


def test_a_packed_point_is_64_little_endian_bytes_in_a_documented_order() -> None:
    _, packed = accumulate(make_scans(1))

    assert PACKED_POINT.size == 64
    assert len(packed) == 2 * 64
    # mapa xyz, origem xyz, ordinal do scan, índice do ponto no scan, pontos contribuintes.
    assert PACKED_POINT.unpack_from(packed, 0) == pytest.approx(
        (1.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0, 0, 1)
    )
    assert PACKED_POINT.unpack_from(packed, 64) == pytest.approx(
        (0.5, 2.0, 0.75, 0.0, 2.0, 0.5, 0, 1, 1)
    )


def test_the_numpy_record_and_the_documented_struct_agree_on_the_layout() -> None:
    import numpy as np

    from contextmap.geometric_mapping.accumulation import PACKED_DTYPE_FIELDS

    assert np.dtype(PACKED_DTYPE_FIELDS).itemsize == PACKED_POINT.size


# --- What the map records ----------------------------------------------------------


def test_the_map_records_bounds_time_sources_and_provenance() -> None:
    accumulated, _ = accumulate(make_scans(3))
    geometric_map = accumulated.geometric_map

    assert geometric_map.map_id == MAP_ID and geometric_map.frame_id == FrameId("map")
    assert geometric_map.bounds == Bounds3D(
        frame_id=FrameId("map"), minimum_m=(0.5, 0.0, 0.25), maximum_m=(3.5, 2.0, 0.75)
    )
    assert [str(item) for item in geometric_map.source_observation_ids] == [
        "scan-0000",
        "scan-0001",
        "scan-0002",
    ]
    assert geometric_map.time_bounds.start == timestamp_ns(0)
    assert geometric_map.time_bounds.end == timestamp_ns(200 * MS)
    assert geometric_map.provenance == make_provenance()
    assert geometric_map.aggregation_rule is None
    assert accumulated.source_point_count == 6


def test_the_provenance_comes_from_the_plan_that_was_mapped() -> None:
    provenance = make_provenance()

    assert str(provenance.sequence_artifact_id) == "sequence-0001"
    assert str(provenance.trajectory_id) == "run-0001--trajectory"
    assert provenance.calibration_identity is not None
    assert provenance.configuration_fingerprint == "sha256:mapping-config"
    assert provenance.code_version == "test"


def test_every_scan_gets_a_source_index_entry() -> None:
    accumulated, _ = accumulate(make_scans(3))

    first, second, third = accumulated.scans
    assert [record.ordinal for record in accumulated.scans] == [0, 1, 2]
    assert (first.first_geometry_index, first.geometry_count) == (0, 2)
    assert (second.first_geometry_index, second.geometry_count) == (2, 2)
    assert (third.first_geometry_index, third.geometry_count) == (4, 2)
    assert str(second.observation_id) == "scan-0001"
    assert second.bounds == Bounds3D(
        frame_id=FrameId("map"), minimum_m=(1.5, 0.0, 0.25), maximum_m=(2.5, 2.0, 0.75)
    )
    assert second.motion_correction is MotionCorrectionState.UNKNOWN
    assert second.transform_lineage == make_scans(3)[1].transform_lineage


def test_non_finite_points_are_counted_in_the_source_index_not_persisted() -> None:
    points = ((1.0, 0.0, 0.0), (NAN, 0.0, 0.0), (0.0, 2.0, 0.5))
    accumulated, packed = accumulate(make_scans(1, points=points))

    (record,) = accumulated.scans
    assert (record.source_point_count, record.dropped_non_finite_count) == (3, 1)
    assert record.geometry_count == 2
    geometry = open_geometry(accumulated, packed)
    kept = [point.source_point_index for point in geometry.iter_geometry()]
    assert kept == [0, 2]


def test_a_scan_without_finite_points_is_indexed_but_contributes_no_geometry() -> None:
    empty = make_scans(1, points=((NAN, NAN, NAN),))[0]
    real = make_scans(1, first_scan=1)[0]

    accumulated, _ = accumulate([empty, real])

    assert [str(item) for item in accumulated.geometric_map.source_observation_ids] == ["scan-0001"]
    first, second = accumulated.scans
    assert (first.geometry_count, first.bounds) == (0, None)
    assert second.first_geometry_index == 0
    assert accumulated.geometric_map.point_count == 2


def test_the_map_time_range_uses_the_earliest_and_latest_contributing_scans() -> None:
    accumulated, _ = accumulate([make_scans(1, first_scan=2)[0], make_scans(1, first_scan=0)[0]])

    assert accumulated.geometric_map.time_bounds.start == timestamp_ns(0)
    assert accumulated.geometric_map.time_bounds.end == timestamp_ns(200 * MS)


# --- One global frame ----------------------------------------------------------------


def test_a_scan_in_another_map_frame_would_create_a_second_coordinate_system_and_is_refused() -> (
    None
):
    (scan,) = make_scans(1)
    elsewhere = dataclasses.replace(scan, map_frame=FrameId("odom"))

    with pytest.raises(AccumulationError, match="frame"):
        accumulate([elsewhere])


def test_processing_a_segment_keeps_the_same_global_coordinates_as_the_full_run() -> None:
    scans = make_scans(4)
    full, full_bytes = accumulate(scans)
    segment, segment_bytes = accumulate(scans[2:4])

    full_points = {
        (str(p.source_observation_id), p.source_point_index): p.coordinates_m
        for p in open_geometry(full, full_bytes).iter_geometry()
    }
    for point in open_geometry(segment, segment_bytes).iter_geometry():
        key = (str(point.source_observation_id), point.source_point_index)
        assert point.map_frame == FrameId("map")
        assert point.coordinates_m == full_points[key]


def test_the_same_scan_twice_is_refused() -> None:
    (scan,) = make_scans(1)

    with pytest.raises(AccumulationError, match="scan-0000"):
        accumulate([scan, scan])


def test_scans_in_different_clock_domains_are_refused() -> None:
    (scan,) = make_scans(1)
    other = dataclasses.replace(
        make_scans(1, first_scan=1)[0], acquisition_timestamp=timestamp_ns(0, clock_id="other")
    )

    with pytest.raises(AccumulationError, match="clock"):
        accumulate([scan, other])


def test_a_map_without_any_geometry_is_refused() -> None:
    empty = make_scans(1, points=((NAN, NAN, NAN),))[0]

    with pytest.raises(AccumulationError, match="no geometry"):
        accumulate([empty])


def test_a_finished_accumulator_accepts_no_more_scans() -> None:
    (scan,) = make_scans(1)
    accumulator = MapAccumulator(
        map_id=MAP_ID, map_frame=FrameId("map"), sink=io.BytesIO(), aggregation=None
    )
    accumulator.add_scan(scan)
    accumulator.finish(provenance=make_provenance())

    with pytest.raises(AccumulationError, match="finished"):
        accumulator.add_scan(make_scans(1, first_scan=1)[0])


def test_a_plan_is_accumulated_scan_by_scan() -> None:
    calibration = make_calibration((rigid("body", "lidar", (0.5, 0.0, 0.25)),))
    plan = assemble_plan(
        [make_scan(f"scan-{i:04d}", time_ns=i * 100 * MS) for i in range(3)],
        calibration=calibration,
        trajectory=make_trajectory(calibration=calibration),
    )
    sink = io.BytesIO()

    accumulated = accumulate_plan(
        plan,
        map_id=MAP_ID,
        sink=sink,
        aggregation=None,
        configuration_fingerprint="sha256:mapping-config",
        code_version="test",
    )

    assert accumulated.geometric_map.point_count == 6
    assert accumulated.geometric_map.provenance.trajectory_id == plan.trajectory_id
    assert accumulated.geometric_map.frame_id == plan.map_frame
    assert len(sink.getvalue()) == 6 * PACKED_POINT.size
    assert next(iter(transform_scans(plan))).point_count == 2


# --- Explicit aggregation ---------------------------------------------------------


CLUSTER = (
    (1.1, 0.0, 0.0),  # mapa: (1.6, 0, 0.25)  -> voxel (3, 0, 0) com célula de 0,5 m
    (1.2, 0.1, 0.0),  # mapa: (1.7, 0.1, 0.25) -> mesmo voxel
    (1.3, 0.0, 0.0),  # mapa: (1.8, 0, 0.25)  -> mesmo voxel
    (3.1, 0.0, 0.0),  # mapa: (3.6, 0, 0.25)  -> voxel (7, 0, 0), sozinho
)


def test_points_of_one_scan_in_the_same_voxel_become_one_declared_aggregate() -> None:
    accumulated, packed = accumulate(
        make_scans(1, points=CLUSTER), aggregation=ScanVoxelPolicy(cell_m=0.5)
    )
    geometry = open_geometry(accumulated, packed)

    points = list(geometry.iter_geometry())
    assert accumulated.source_point_count == 4
    assert accumulated.geometric_map.point_count == 2
    aggregated = next(p for p in points if p.provenance.origin is PointOrigin.AGGREGATED)
    assert aggregated.provenance.aggregation_rule == "scan-voxel-centroid-0.5m"
    assert aggregated.provenance.contributing_point_count == 3
    assert aggregated.source_point_index is None
    assert aggregated.coordinates_m == pytest.approx((1.7, 1 / 30, 0.25), abs=TOLERANCE_M)
    assert aggregated.source_coordinates_m == pytest.approx((1.2, 1 / 30, 0.0), abs=TOLERANCE_M)


def test_a_voxel_with_one_point_stays_a_raw_measurement_with_its_index() -> None:
    accumulated, packed = accumulate(
        make_scans(1, points=CLUSTER), aggregation=ScanVoxelPolicy(cell_m=0.5)
    )

    measured = next(
        p
        for p in open_geometry(accumulated, packed).iter_geometry()
        if p.provenance.origin is PointOrigin.MEASURED
    )

    assert measured.source_point_index == 3
    assert measured.provenance.aggregation_rule is None
    assert measured.coordinates_m == pytest.approx((3.6, 0.0, 0.25), abs=TOLERANCE_M)


def test_the_map_declares_the_aggregation_rule_and_the_reduction() -> None:
    accumulated, _ = accumulate(
        make_scans(1, points=CLUSTER), aggregation=ScanVoxelPolicy(cell_m=0.5)
    )

    assert accumulated.geometric_map.aggregation_rule == "scan-voxel-centroid-0.5m"
    assert accumulated.source_point_count == 4
    assert accumulated.geometric_map.point_count == 2


def test_aggregation_never_merges_points_of_different_scans() -> None:
    # Duas poses distintas (1 m de diferença) com pontos na mesma região: cada scan agrega o seu.
    same_region = ((1.1, 0.0, 0.0), (1.2, 0.0, 0.0))
    scans = make_scans(2, points=same_region)

    accumulated, packed = accumulate(scans, aggregation=ScanVoxelPolicy(cell_m=5.0))

    points = list(open_geometry(accumulated, packed).iter_geometry())
    assert accumulated.geometric_map.point_count == 2
    assert {str(p.source_observation_id) for p in points} == {"scan-0000", "scan-0001"}
    assert all(p.provenance.contributing_point_count == 2 for p in points)


def test_aggregated_points_are_ordered_deterministically_by_voxel() -> None:
    first, first_bytes = accumulate(
        make_scans(1, points=CLUSTER), aggregation=ScanVoxelPolicy(cell_m=0.5)
    )
    second, second_bytes = accumulate(
        make_scans(1, points=tuple(reversed(CLUSTER))), aggregation=ScanVoxelPolicy(cell_m=0.5)
    )

    assert [p.coordinates_m for p in open_geometry(first, first_bytes).iter_geometry()] == [
        pytest.approx(p.coordinates_m, abs=TOLERANCE_M)
        for p in open_geometry(second, second_bytes).iter_geometry()
    ]


@pytest.mark.parametrize("cell_m", [0.0, -0.1, math.nan, math.inf])
def test_a_voxel_size_that_is_not_positive_and_finite_is_refused(cell_m: float) -> None:
    with pytest.raises(ValueError, match="cell_m"):
        ScanVoxelPolicy(cell_m=cell_m)


def test_the_rule_name_states_the_voxel_size() -> None:
    assert ScanVoxelPolicy(cell_m=0.05).rule == "scan-voxel-centroid-0.05m"
