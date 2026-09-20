import dataclasses
import json
import math
import os
import subprocess
import sys
from array import array
from collections.abc import Iterator
from pathlib import Path

import pytest
from input_builders import (
    MS,
    QUARTER_TURN_Z,
    assemble_plan,
    make_calibration,
    make_sequence,
    make_trajectory,
    rigid,
)
from lidar_builders import make_scan, timestamp_ns

from contextmap import geometric_mapping
from contextmap.geometric_mapping import (
    GeometryInput,
    GeometryTransformError,
    MapId,
    MotionCorrectionRecord,
    MotionCorrectionState,
    PointOrigin,
    TransformedScan,
    TransformKind,
    geometry_id_for,
    transform_scan,
    transform_scans,
    verify_transform_trace,
)
from contextmap.geometric_mapping.serialization import (
    decode_transform_trace,
    encode_transform_trace,
)
from contextmap.ingestion import (
    FrameId,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
)
from contextmap.shared import (
    Quaternion,
    Vector3,
    invert_rigid,
    normalize_quaternion,
    rotate_vector,
)
from contextmap.state_estimation import (
    LookupPolicy,
    ResolvedTransform,
    calibration_identity,
)

MAP_ID = MapId("map-0001")
CALIBRATION = make_calibration()
CALIBRATION_IDENTITY = calibration_identity(CALIBRATION)
TOLERANCE_M = 1e-9


def _axis_angle(axis: Vector3, angle_rad: float) -> Quaternion:
    norm = math.sqrt(sum(component * component for component in axis))
    sin = math.sin(angle_rad / 2.0)
    unit = (axis[0] / norm, axis[1] / norm, axis[2] / norm)
    return normalize_quaternion(
        (unit[0] * sin, unit[1] * sin, unit[2] * sin, math.cos(angle_rad / 2))
    )


def _transform(
    scan: LidarObservation,
    *,
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    static: tuple[Vector3, Quaternion] = ((0.5, 0.0, 0.25), (0.0, 0.0, 0.0, 1.0)),
) -> tuple[TransformedScan, GeometryInput]:
    """Transform ``scan`` under a trajectory rotated by ``orientation`` and extrinsic ``static``."""
    calibration = make_calibration((rigid("body", "lidar", static[0], static[1]),))
    plan = assemble_plan(
        [scan],
        calibration=calibration,
        trajectory=make_trajectory(calibration=calibration, orientation=orientation),
    )
    (item,) = plan.inputs
    return transform_scan(item, calibration_identity=plan.calibration_identity), item


def _points(scan: TransformedScan) -> list[Vector3]:
    return [scan.map_point(position) for position in range(scan.point_count)]


# --- A known point through a known chain -------------------------------------


def test_a_known_point_lands_where_the_hand_computed_chain_puts_it() -> None:
    # T_map_body: 90 graus em z e translação (2, 0, 0); T_body_lidar: translação (0.5, 0, 0.25).
    # (1, 0, 0) -> corpo (1.5, 0, 0.25) -> girado (0, 1.5, 0.25) -> mapa (2, 1.5, 0.25).
    scan, _ = _transform(
        make_scan(time_ns=200 * MS, points=((1.0, 0.0, 0.0), (0.0, 2.0, 0.5))),
        orientation=QUARTER_TURN_Z,
    )

    assert _points(scan)[0] == pytest.approx((2.0, 1.5, 0.25), abs=TOLERANCE_M)
    assert _points(scan)[1] == pytest.approx((0.0, 0.5, 0.75), abs=TOLERANCE_M)


def test_a_rotation_in_both_the_pose_and_the_extrinsic_composes_in_the_right_order() -> None:
    # Estático (90 graus): (1, 0, 0) -> (0, 1, 0) + (0.5, 0, 0.25) = (0.5, 1, 0.25).
    # Pose (90 graus): -> (-1, 0.5, 0.25) + (2, 0, 0) = (1, 0.5, 0.25).
    scan, _ = _transform(
        make_scan(time_ns=200 * MS, points=((1.0, 0.0, 0.0),)),
        orientation=QUARTER_TURN_Z,
        static=((0.5, 0.0, 0.25), QUARTER_TURN_Z),
    )

    assert _points(scan)[0] == pytest.approx((1.0, 0.5, 0.25), abs=TOLERANCE_M)


def _apply(transform: tuple[Vector3, Quaternion], point: Vector3) -> Vector3:
    """Oráculo escalar de ``p_parent = R · p_child + t``, sem NumPy."""
    translation, rotation = transform
    rotated = rotate_vector(rotation, point)
    return (
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    )


def _sequential(
    point: Vector3, pose: tuple[Vector3, Quaternion], static: tuple[Vector3, Quaternion]
) -> Vector3:
    """Apply ``T_body_source`` and then ``T_map_body`` one after the other."""
    return _apply(pose, _apply(static, point))


GENERAL_POSE_ROTATION = _axis_angle((1.0, 2.0, 3.0), 0.7)
GENERAL_STATIC = ((0.31, -0.12, 0.85), _axis_angle((-2.0, 1.0, 0.5), 1.9))
CLOUD: tuple[Vector3, ...] = tuple(
    (math.sin(i) * 20.0, math.cos(2 * i) * 3.0, i * 0.1 - 2.0) for i in range(50)
)


def test_the_composed_transform_matches_applying_the_chain_step_by_step() -> None:
    scan, item = _transform(
        make_scan(time_ns=300 * MS, points=CLOUD),
        orientation=GENERAL_POSE_ROTATION,
        static=GENERAL_STATIC,
    )

    pose = (item.pose.pose.translation_m, item.pose.pose.orientation)
    assert scan.point_count == len(CLOUD)
    for position in range(scan.point_count):
        expected = _sequential(scan.source_point(position), pose, GENERAL_STATIC)
        assert scan.map_point(position) == pytest.approx(expected, abs=TOLERANCE_M)


def test_the_inverse_chain_takes_every_map_point_back_to_its_source_point() -> None:
    scan, item = _transform(
        make_scan(time_ns=300 * MS, points=CLOUD),
        orientation=GENERAL_POSE_ROTATION,
        static=GENERAL_STATIC,
    )
    inverse_pose = invert_rigid(
        translation=item.pose.pose.translation_m, rotation=item.pose.pose.orientation
    )
    inverse_static = invert_rigid(translation=GENERAL_STATIC[0], rotation=GENERAL_STATIC[1])

    for position in range(scan.point_count):
        in_body = _apply(inverse_pose, scan.map_point(position))
        back = _apply(inverse_static, in_body)
        assert back == pytest.approx(scan.source_point(position), abs=TOLERANCE_M)


def test_double_precision_payloads_keep_their_precision() -> None:
    scan, _ = _transform(make_scan(points=((123456.78901234567, 0.0, 0.0),), double_precision=True))

    assert scan.source_point(0) == (123456.78901234567, 0.0, 0.0)
    assert scan.map_point(0)[0] == pytest.approx(123456.78901234567 + 0.5, abs=TOLERANCE_M)


def test_the_layout_decides_which_bytes_are_x_y_and_z() -> None:
    packed = make_scan(points=((1.0, 0.0, 0.0), (0.0, 2.0, 0.5)), intensity=True)
    kind = PointFieldDataType.FLOAT32
    reordered = dataclasses.replace(
        packed,
        fields=(
            PointFieldDescriptor(name="z", offset_bytes=0, data_type=kind),
            PointFieldDescriptor(name="y", offset_bytes=4, data_type=kind),
            PointFieldDescriptor(name="x", offset_bytes=8, data_type=kind),
            PointFieldDescriptor(name="intensity", offset_bytes=12, data_type=kind),
        ),
    )

    scan, _ = _transform(reordered)

    assert [scan.source_point(0), scan.source_point(1)] == [(0.0, 0.0, 1.0), (0.5, 2.0, 0.0)]


# --- What is preserved -------------------------------------------------------


def test_the_original_source_coordinates_and_identity_are_preserved() -> None:
    source = make_scan("scan-0002", time_ns=200 * MS, points=((4.21, -0.71, 0.32),))

    scan, item = _transform(source)

    assert scan.observation_id == source.observation_id
    assert scan.source_frame == FrameId("lidar") and scan.map_frame == FrameId("map")
    assert scan.acquisition_timestamp == source.timestamp
    assert scan.payload_hash == item.payload_hash
    # O valor de origem é o float32 do payload, promovido sem perda; nunca recalculado do mapa.
    assert scan.source_point(0) == (
        pytest.approx(4.21, abs=1e-6),
        pytest.approx(-0.71, abs=1e-6),
        pytest.approx(0.32, abs=1e-6),
    )
    assert scan.source_point_indices[0] == 0


def test_non_finite_points_are_dropped_counted_and_keep_their_original_index() -> None:
    nan, inf = float("nan"), float("inf")
    source = make_scan(
        points=(
            (1.0, 0.0, 0.0),
            (nan, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, inf, 0.0),
            (3.0, 0.0, 0.0),
        ),
        is_dense=False,
    )

    scan, _ = _transform(source)

    assert scan.point_count == 3
    assert scan.source_point_count == 5
    assert scan.dropped_non_finite_count == 2
    assert list(scan.source_point_indices) == [0, 2, 4]
    assert [point[0] for point in _points(scan)] == pytest.approx([1.5, 2.5, 3.5])


def test_a_scan_whose_points_are_all_non_finite_yields_an_empty_result() -> None:
    nan = float("nan")

    scan, _ = _transform(make_scan(points=((nan, nan, nan),), is_dense=False))

    assert (scan.point_count, scan.dropped_non_finite_count) == (0, 1)


def test_the_correction_state_of_the_scan_travels_with_its_points() -> None:
    source = make_scan()
    declared = MotionCorrectionRecord(
        observation_id=source.observation_id, state=MotionCorrectionState.RAW
    )
    calibration = make_calibration()
    plan = assemble_plan(
        [source], calibration=calibration, motion_correction={source.observation_id: declared}
    )

    scan = transform_scan(plan.inputs[0], calibration_identity=plan.calibration_identity)

    assert scan.motion_correction is MotionCorrectionState.RAW
    assert scan.geometry_point(0, map_id=MAP_ID, geometry_index=0).provenance.motion_correction is (
        MotionCorrectionState.RAW
    )


# --- Lineage -------------------------------------------------------------------


def test_the_lineage_names_the_pose_and_the_calibration_that_were_used() -> None:
    scan, item = _transform(make_scan(time_ns=200 * MS))

    dynamic, static = scan.transform_lineage.steps
    assert dynamic.kind is TransformKind.DYNAMIC_POSE
    assert (dynamic.parent_frame, dynamic.child_frame) == (FrameId("map"), FrameId("body"))
    assert dynamic.reference == str(item.pose.pose.estimate_id)
    assert dynamic.source_estimate_ids == item.pose.source_estimate_ids
    assert static.kind is TransformKind.STATIC_CALIBRATION
    assert (static.parent_frame, static.child_frame) == (FrameId("body"), FrameId("lidar"))
    assert static.reference == calibration_identity(
        make_calibration((rigid("body", "lidar", (0.5, 0.0, 0.25)),))
    )


def test_an_interpolated_pose_is_traced_to_both_estimates_it_came_from() -> None:
    scan, item = _transform(make_scan(time_ns=150 * MS))

    dynamic = scan.transform_lineage.steps[0]
    assert len(dynamic.source_estimate_ids) == 2
    assert dynamic.reference == str(item.pose.pose.estimate_id)
    assert dynamic.reference not in {str(identity) for identity in dynamic.source_estimate_ids}


def test_a_scan_in_the_body_frame_has_only_the_dynamic_step() -> None:
    scan, _ = _transform(make_scan(frame="body"))

    (dynamic,) = scan.transform_lineage.steps
    assert dynamic.kind is TransformKind.DYNAMIC_POSE
    assert scan.geometry_point(0, map_id=MAP_ID, geometry_index=0).source_frame == FrameId("body")


def test_a_geometry_point_carries_authoritative_map_coordinates_and_the_original_ones() -> None:
    source = make_scan(
        "scan-0002",
        time_ns=200 * MS,
        points=((1.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), (0.0, 2.0, 0.5)),
        is_dense=False,
    )
    scan, _ = _transform(source, orientation=QUARTER_TURN_Z)

    point = scan.geometry_point(1, map_id=MAP_ID, geometry_index=7)

    assert point.geometry_id == geometry_id_for(map_id=MAP_ID, index=7)
    assert point.map_id == MAP_ID and point.map_frame == FrameId("map")
    assert point.coordinates_m == pytest.approx((0.0, 0.5, 0.75), abs=TOLERANCE_M)
    assert point.source_coordinates_m == (0.0, 2.0, 0.5)
    assert point.source_frame == FrameId("lidar")
    assert point.source_observation_id == source.observation_id
    assert point.source_point_index == 2
    assert point.acquisition_timestamp == source.timestamp
    assert point.transform_lineage == scan.transform_lineage
    assert point.provenance.origin is PointOrigin.MEASURED


# --- Validation ----------------------------------------------------------------


def _with_static(
    item: GeometryInput,
    *,
    parent: str = "body",
    child: str = "lidar",
    translation: Vector3 = (0.5, 0.0, 0.25),
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
) -> GeometryInput:
    return dataclasses.replace(
        item,
        static_transform=ResolvedTransform(
            parent_frame=FrameId(parent),
            child_frame=FrameId(child),
            translation=translation,
            rotation=rotation,
        ),
    )


def test_a_static_rotation_that_is_not_a_rotation_is_refused() -> None:
    _, item = _transform(make_scan())

    with pytest.raises(GeometryTransformError, match="rotation"):
        transform_scan(
            _with_static(item, rotation=(0.0, 0.0, 0.0, 1.05)),
            calibration_identity=CALIBRATION_IDENTITY,
        )


def test_a_static_translation_that_is_not_finite_is_refused() -> None:
    _, item = _transform(make_scan())

    with pytest.raises(GeometryTransformError, match="finite"):
        transform_scan(
            _with_static(item, translation=(float("nan"), 0.0, 0.0)),
            calibration_identity=CALIBRATION_IDENTITY,
        )


@pytest.mark.parametrize(("parent", "child"), [("other", "lidar"), ("body", "other")])
def test_a_chain_whose_frames_do_not_connect_is_refused(parent: str, child: str) -> None:
    _, item = _transform(make_scan())

    with pytest.raises(GeometryTransformError, match="frame"):
        transform_scan(
            _with_static(item, parent=parent, child=child),
            calibration_identity=CALIBRATION_IDENTITY,
        )


def test_a_scan_in_another_frame_than_the_body_needs_a_static_transform() -> None:
    _, item = _transform(make_scan())

    with pytest.raises(GeometryTransformError, match="frame"):
        transform_scan(
            dataclasses.replace(item, static_transform=None),
            calibration_identity=CALIBRATION_IDENTITY,
        )


def test_a_static_transform_without_the_calibration_it_came_from_is_refused() -> None:
    _, item = _transform(make_scan())

    with pytest.raises(GeometryTransformError, match="calibration"):
        transform_scan(item, calibration_identity=None)


# --- Batches and determinism -----------------------------------------------------


def test_the_plan_is_transformed_one_scan_at_a_time_in_order() -> None:
    plan = assemble_plan()

    results = transform_scans(plan)

    assert isinstance(results, Iterator)
    scans = list(results)
    assert [str(scan.observation_id) for scan in scans] == [
        str(item.observation_id) for item in plan.inputs
    ]
    assert all(scan.point_count == 2 for scan in scans)


def test_the_transformation_is_deterministic() -> None:
    plan = assemble_plan()

    assert list(transform_scans(plan)) == list(transform_scans(plan))


def test_the_public_result_holds_plain_arrays_not_numpy_objects() -> None:
    scan, _ = _transform(make_scan())

    assert isinstance(scan.map_coordinates_m, array)
    assert isinstance(scan.source_coordinates_m, array)
    assert isinstance(scan.source_point_indices, array)


def test_importing_the_capability_does_not_import_numpy() -> None:
    source_dir = Path(geometric_mapping.__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, contextmap.geometric_mapping; print('numpy' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(source_dir)},
    )

    assert result.stdout.strip() == "False"


# --- Transform traces --------------------------------------------------------------


def test_a_trace_reconstructs_the_chain_used_for_one_audited_point() -> None:
    source = make_scan("lidar-frame-01824", time_ns=200 * MS, points=((4.21, -0.71, 0.32),))
    scan, item = _transform(source, orientation=QUARTER_TURN_Z)

    trace = scan.trace_point(0)

    assert str(trace.source_observation_id) == "lidar-frame-01824"
    assert trace.source_point_index == 0
    assert trace.source_coordinates_m == scan.source_point(0)
    assert trace.map_coordinates_m == scan.map_point(0)
    dynamic, static = trace.transforms
    assert dynamic.translation_m == item.pose.pose.translation_m
    assert dynamic.rotation == item.pose.pose.orientation
    assert dynamic.step.reference == str(item.pose.pose.estimate_id)
    assert static.translation_m == (0.5, 0.0, 0.25)
    assert static.step.reference == scan.transform_lineage.steps[1].reference
    assert verify_transform_trace(trace) == []


def test_a_trace_whose_map_coordinates_disagree_with_its_chain_is_reported() -> None:
    scan, _ = _transform(make_scan(time_ns=200 * MS))
    trace = scan.trace_point(0)
    tampered = dataclasses.replace(trace, map_coordinates_m=(99.0, 0.0, 0.0))

    problems = verify_transform_trace(tampered)

    assert len(problems) == 1 and "map" in problems[0]


def test_a_trace_is_persistable_without_the_scan() -> None:
    scan, _ = _transform(make_scan(time_ns=200 * MS), orientation=QUARTER_TURN_Z)
    trace = scan.trace_point(1)

    record = json.loads(json.dumps(encode_transform_trace(trace)))

    assert decode_transform_trace(record) == trace
    assert verify_transform_trace(decode_transform_trace(record)) == []
    # As matrizes ficam uma vez por cadeia, não por ponto: o registro tem só os dois passos.
    assert len(record["transforms"]) == 2


def test_the_default_sequence_transforms_end_to_end() -> None:
    plan = assemble_plan(make_sequence(), pose_lookup=LookupPolicy.interpolated())

    scans = list(transform_scans(plan))

    assert len(scans) == 5
    assert scans[0].map_point(0) == pytest.approx((1.5, 0.0, 0.25), abs=TOLERANCE_M)
    assert scans[4].map_point(0) == pytest.approx((5.5, 0.0, 0.25), abs=TOLERANCE_M)
    assert timestamp_ns(400 * MS) == scans[4].acquisition_timestamp
