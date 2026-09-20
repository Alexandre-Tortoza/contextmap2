import math

import pytest
from calibration_builders import (
    IDENTITY,
    QUARTER_TURN_Z,
    calibration,
    closing_transform,
    rigid,
)

from contextmap.ingestion import FrameId, RigidTransform
from contextmap.shared import quaternion_angle_between
from contextmap.state_estimation import FrameGraphError, StaticFrameGraph


def test_resolves_a_direct_transform_and_its_inverse() -> None:
    graph = StaticFrameGraph([rigid("body", "lidar", (0.0, 0.0, 0.5))])

    forward = graph.resolve(FrameId("body"), FrameId("lidar"))
    backward = graph.resolve(FrameId("lidar"), FrameId("body"))

    assert (forward.parent_frame, forward.child_frame) == (FrameId("body"), FrameId("lidar"))
    assert forward.translation == pytest.approx((0.0, 0.0, 0.5))
    assert (backward.parent_frame, backward.child_frame) == (FrameId("lidar"), FrameId("body"))
    assert backward.translation == pytest.approx((0.0, 0.0, -0.5))


def test_resolves_a_chain_with_an_explicit_frame_direction() -> None:
    graph = StaticFrameGraph(
        [
            rigid("body", "lidar", (1.0, 0.0, 0.0), QUARTER_TURN_Z),
            rigid("lidar", "camera", (1.0, 0.0, 0.0)),
        ]
    )

    resolved = graph.resolve(FrameId("body"), FrameId("camera"))

    assert resolved.translation == pytest.approx((1.0, 1.0, 0.0), abs=1e-12)
    assert resolved.rotation == pytest.approx(QUARTER_TURN_Z)


def test_a_frame_resolved_against_itself_is_the_identity() -> None:
    resolved = StaticFrameGraph([rigid("body", "lidar")]).resolve(FrameId("body"), FrameId("body"))

    assert resolved.translation == (0.0, 0.0, 0.0)
    assert resolved.rotation == IDENTITY


def test_resolution_is_reproducible_and_inverse_consistent() -> None:
    graph = StaticFrameGraph(
        [
            rigid("body", "lidar", (1.0, 2.0, 3.0), QUARTER_TURN_Z),
            rigid("lidar", "camera", (0.1, 0.0, -0.2)),
        ]
    )

    forward = graph.resolve(FrameId("body"), FrameId("camera"))
    backward = graph.resolve(FrameId("camera"), FrameId("body"))

    assert forward == graph.resolve(FrameId("body"), FrameId("camera"))
    assert quaternion_angle_between(
        forward.rotation,
        (-backward.rotation[0], -backward.rotation[1], -backward.rotation[2], backward.rotation[3]),
    ) == pytest.approx(0.0, abs=1e-9)


def test_an_unknown_frame_is_an_error() -> None:
    with pytest.raises(FrameGraphError, match="unknown frame"):
        StaticFrameGraph([rigid("body", "lidar")]).resolve(FrameId("body"), FrameId("radar"))


def test_frames_without_a_static_path_are_an_error() -> None:
    graph = StaticFrameGraph([rigid("body", "lidar"), rigid("world", "anchor")])

    with pytest.raises(FrameGraphError, match="no static path"):
        graph.resolve(FrameId("body"), FrameId("anchor"))


def test_components_group_connected_frames() -> None:
    graph = StaticFrameGraph([rigid("body", "lidar"), rigid("world", "anchor")])

    assert graph.component_of(FrameId("lidar")) == frozenset({FrameId("body"), FrameId("lidar")})
    assert graph.frames == frozenset(
        {FrameId("body"), FrameId("lidar"), FrameId("world"), FrameId("anchor")}
    )


def _loop(inconsistent_by: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[RigidTransform]:
    body_lidar = rigid("body", "lidar", (1.0, 0.0, 0.0), QUARTER_TURN_Z)
    body_camera = rigid("body", "camera", (0.0, 2.0, 0.5))
    lidar_camera = closing_transform(body_lidar, body_camera, parent="lidar", child="camera")
    x, y, z = lidar_camera.translation
    dx, dy, dz = inconsistent_by
    return [
        body_lidar,
        body_camera,
        rigid("lidar", "camera", (x + dx, y + dy, z + dz), lidar_camera.rotation),
    ]


def test_a_consistent_loop_has_no_inconsistency() -> None:
    graph = StaticFrameGraph(_loop())

    assert (
        graph.loop_inconsistencies(translation_tolerance_m=1e-6, rotation_tolerance_rad=1e-6) == ()
    )


def test_an_inconsistent_loop_reports_the_size_of_the_disagreement() -> None:
    graph = StaticFrameGraph(_loop(inconsistent_by=(0.05, 0.0, 0.0)))

    (finding,) = graph.loop_inconsistencies(
        translation_tolerance_m=1e-3, rotation_tolerance_rad=1e-3
    )

    assert finding.translation_error_m == pytest.approx(0.05, abs=1e-9)
    assert finding.rotation_error_rad == pytest.approx(0.0, abs=1e-9)
    assert {FrameId("lidar"), FrameId("camera")} <= set(finding.frames)


def test_a_loop_error_within_the_tolerance_is_accepted() -> None:
    graph = StaticFrameGraph(_loop(inconsistent_by=(0.0005, 0.0, 0.0)))

    assert (
        graph.loop_inconsistencies(translation_tolerance_m=1e-3, rotation_tolerance_rad=1e-3) == ()
    )


def test_a_rotation_disagreement_in_a_loop_is_reported() -> None:
    body_lidar = rigid("body", "lidar")
    body_camera = rigid("body", "camera")
    wrong = rigid("lidar", "camera", (0.0, 0.0, 0.0), QUARTER_TURN_Z)

    (finding,) = StaticFrameGraph([body_lidar, body_camera, wrong]).loop_inconsistencies(
        translation_tolerance_m=1e-3, rotation_tolerance_rad=1e-3
    )

    assert finding.rotation_error_rad == pytest.approx(math.pi / 2)


def test_a_reversed_duplicate_is_consistent_only_when_it_is_the_inverse() -> None:
    forward = rigid("body", "lidar", (1.0, 0.0, 0.0))
    correct_reverse = rigid("lidar", "body", (-1.0, 0.0, 0.0))
    same_direction = rigid("lidar", "body", (1.0, 0.0, 0.0))
    tolerance = {"translation_tolerance_m": 1e-6, "rotation_tolerance_rad": 1e-6}

    assert StaticFrameGraph([forward, correct_reverse]).loop_inconsistencies(**tolerance) == ()
    (finding,) = StaticFrameGraph([forward, same_direction]).loop_inconsistencies(**tolerance)
    assert finding.translation_error_m == pytest.approx(2.0)


def test_calibration_sets_build_the_same_graph() -> None:
    transforms = [rigid("body", "lidar", (0.0, 0.0, 0.5))]

    graph = StaticFrameGraph.from_calibration(calibration(*transforms))

    assert graph.frames == frozenset({FrameId("body"), FrameId("lidar")})
