import dataclasses
import math

import numpy as np
import pytest
from projection_builders import (
    ArrayGeometrySource,
    artifact,
    make_calibration,
    make_camera_observation,
    make_lookup,
    make_prepared_image,
    make_trajectory,
    map_point_for_camera_point,
    map_point_for_pixel,
    mask_with,
)

from contextmap.ingestion import CalibrationSet, MeiCameraModel
from contextmap.sensor_association import DepthMetric, VisibilityState
from contextmap.sensor_association.frame_projection import (
    FrameProjection,
    FrameProjector,
    ProjectionStage,
)
from contextmap.sensor_association.geometry_cloud import GeometryCloud
from contextmap.sensor_association.visibility import (
    OcclusionPolicy,
    depth_metric_for,
    resolve_visibility,
)
from contextmap.shared import Vector3
from contextmap.state_estimation import LookupPolicy
from contextmap.visual_perception import (
    ExclusionRegion,
    PreparedImage,
    ResizeOperation,
)

# Vizinhança de 5 x 5 células de 4 px em torno da célula do ponto (cerca de +-10 px).
POLICY = OcclusionPolicy(
    cell_size_px=4,
    neighborhood_radius_cells=2,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)
EXACT_PIXEL = OcclusionPolicy(
    cell_size_px=1,
    neighborhood_radius_cells=0,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)


def _frame(
    points: list[Vector3],
    *,
    prepared: PreparedImage | None = None,
    calibration: CalibrationSet | None = None,
) -> FrameProjection:
    calibration = calibration if calibration is not None else make_calibration()
    source = ArrayGeometrySource(points, calibration=calibration)
    projector = FrameProjector(
        cloud=GeometryCloud.from_source(source),
        trajectory=make_lookup(make_trajectory(calibration)),
        pose_policy=LookupPolicy.exact(),
        calibration=calibration,
    )
    frame = projector.project(
        make_camera_observation(0), prepared if prepared is not None else make_prepared_image()
    )
    assert isinstance(frame, FrameProjection)
    return frame


def _scene(*pixels: tuple[float, float, float], **options: object) -> FrameProjection:
    return _frame([map_point_for_pixel(u, v, z) for u, v, z in pixels], **options)  # type: ignore[arg-type]


# --- Front and back surfaces ------------------------------------------------


def test_a_point_behind_a_nearer_surface_does_not_inherit_foreground_evidence() -> None:
    frame = _scene((320, 240, 2.0), (320, 240, 6.0), (100, 100, 6.0))

    resolution = resolve_visibility(frame, POLICY)

    assert resolution.visible.tolist() == [True, False, True]
    assert resolution.occluded.tolist() == [False, True, False]
    np.testing.assert_allclose(resolution.support_depth_m, [2.0, 2.0, 6.0])


def test_a_point_near_the_surface_within_the_depth_margin_is_the_same_surface() -> None:
    # Margem: max(0.1 m, 2 % do apoio). Perto de 2 m vale 0,1 m; perto de 20 m, 0,4 m.
    frame = _scene(
        (100, 100, 2.0),
        (100, 100, 2.05),
        (100, 100, 2.2),
        (500, 400, 20.0),
        (500, 400, 20.3),
        (500, 400, 20.6),
    )

    resolution = resolve_visibility(frame, POLICY)

    assert resolution.occluded.tolist() == [False, False, True, False, False, True]


def test_the_depth_margin_can_be_widened_or_removed_by_policy() -> None:
    frame = _scene((100, 100, 2.0), (100, 100, 2.2))

    strict = dataclasses.replace(POLICY, depth_margin_m=0.0, depth_margin_ratio=0.0)
    lenient = dataclasses.replace(POLICY, depth_margin_m=0.5)

    assert resolve_visibility(frame, strict).occluded.tolist() == [False, True]
    assert resolve_visibility(frame, lenient).occluded.tolist() == [False, False]


# --- Sparse maps ------------------------------------------------------------


def _sparse_wall_scene() -> tuple[list[tuple[float, float, float]], int]:
    # Parede a 2 m só a cada 12 px, com fundo a 6 m projetado nas lacunas.
    wall = [(float(u), float(v), 2.0) for u in range(300, 361, 12) for v in range(224, 261, 12)]
    background = [
        (float(u + 6), float(v + 6), 6.0) for u in range(300, 349, 12) for v in range(224, 249, 12)
    ]
    return wall + background, len(background)


def test_the_gaps_of_a_sparse_front_surface_do_not_let_the_background_leak() -> None:
    pixels, background_count = _sparse_wall_scene()
    frame = _scene(*pixels)
    wall_count = len(pixels) - background_count

    resolution = resolve_visibility(frame, POLICY)

    assert resolution.visible[:wall_count].all()
    assert resolution.occluded[wall_count:].all()


def test_an_exact_pixel_depth_buffer_would_leak_the_background_through_those_gaps() -> None:
    pixels, background_count = _sparse_wall_scene()
    frame = _scene(*pixels)
    wall_count = len(pixels) - background_count

    leaky = resolve_visibility(frame, EXACT_PIXEL)

    assert leaky.visible[wall_count:].all()


def test_the_diagnostics_count_the_points_only_the_neighborhood_occluded() -> None:
    pixels, background_count = _sparse_wall_scene()
    frame = _scene(*pixels)

    resolution = resolve_visibility(frame, POLICY)

    assert resolution.neighborhood_only_occlusions == background_count
    assert resolve_visibility(frame, EXACT_PIXEL).neighborhood_only_occlusions == 0


def test_background_far_from_any_front_surface_stays_visible() -> None:
    frame = _scene((320, 240, 2.0), (100, 400, 6.0), (560, 60, 6.0))

    assert resolve_visibility(frame, POLICY).visible.tolist() == [True, True, True]


# --- The neighborhood is policy, not a constant -----------------------------


def test_the_neighborhood_radius_decides_how_far_a_surface_hides_what_is_behind_it() -> None:
    # A frente em (320, 240) e o fundo 8 px ao lado: duas células de distância.
    frame = _scene((320, 240, 2.0), (328, 240, 6.0))

    def occluded(radius: int) -> bool:
        policy = dataclasses.replace(POLICY, neighborhood_radius_cells=radius)
        return bool(resolve_visibility(frame, policy).occluded[1])

    assert [occluded(0), occluded(1), occluded(2)] == [False, False, True]


def test_the_cell_size_is_measured_in_prepared_image_pixels() -> None:
    # Com a imagem preparada à metade, os 16 px crus entre os pontos viram 8 px preparados.
    resized = make_prepared_image(
        (
            ResizeOperation(
                width=320, height=240, output_image=artifact("half"), provenance_source="cfg"
            ),
        )
    )
    raw_frame = _scene((100, 100, 2.0), (116, 100, 6.0))
    half_frame = _scene((100, 100, 2.0), (116, 100, 6.0), prepared=resized)

    assert not resolve_visibility(raw_frame, POLICY).occluded[1]
    assert resolve_visibility(half_frame, POLICY).occluded[1]


def test_a_cell_size_that_does_not_divide_the_image_still_covers_its_border() -> None:
    policy = dataclasses.replace(POLICY, cell_size_px=7)
    frame = _scene((639, 479, 2.0), (639, 479, 6.0), (0, 0, 2.0), (0, 0, 6.0))

    assert resolve_visibility(frame, policy).occluded.tolist() == [False, True, False, True]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("cell_size_px", 0),
        ("neighborhood_radius_cells", -1),
        ("depth_margin_m", -0.1),
        ("depth_margin_m", float("nan")),
        ("depth_margin_ratio", -0.01),
        ("depth_margin_ratio", float("inf")),
    ],
)
def test_the_policy_is_validated(field_name: str, value: float) -> None:
    with pytest.raises(ValueError, match=field_name):
        dataclasses.replace(POLICY, **{field_name: value})  # type: ignore[arg-type]


def test_the_policy_has_a_versioned_identity_and_a_deterministic_fingerprint() -> None:
    assert POLICY.policy_id == "conservative-depth-support-v1"
    assert POLICY.fingerprint() == dataclasses.replace(POLICY).fingerprint()
    assert POLICY.fingerprint().startswith("sha256:")
    assert POLICY.fingerprint() != dataclasses.replace(POLICY, cell_size_px=5).fingerprint()
    assert POLICY.fingerprint() != EXACT_PIXEL.fingerprint()


# --- Depth semantics --------------------------------------------------------


def test_the_depth_metric_follows_the_camera_model() -> None:
    assert depth_metric_for("pinhole") is DepthMetric.OPTICAL_AXIS
    assert depth_metric_for("fisheye") is DepthMetric.RAY_RANGE
    assert depth_metric_for("mei") is DepthMetric.RAY_RANGE
    with pytest.raises(ValueError, match="hologram"):
        depth_metric_for("hologram")


def test_a_perspective_camera_measures_depth_along_the_optical_axis_not_the_ray() -> None:
    frame = _scene((600, 240, 3.0))

    resolution = resolve_visibility(frame, POLICY)

    assert resolution.depth_metric is DepthMetric.OPTICAL_AXIS
    np.testing.assert_allclose(resolution.depth_m, [3.0])
    assert frame.camera_range_m[0] > 3.4


def test_a_fronto_parallel_wall_is_one_surface_to_a_perspective_camera() -> None:
    # Dois pontos na mesma parede (z = 3) em ângulos diferentes: o alcance difere, a
    # profundidade óptica não, então nenhum oculta o outro.
    frame = _scene((320, 240, 3.0), (330, 240, 3.0), (310, 240, 3.0))

    nearly_strict = dataclasses.replace(POLICY, depth_margin_m=1e-6, depth_margin_ratio=0.0)

    assert resolve_visibility(frame, EXACT_PIXEL).visible.all()
    assert resolve_visibility(frame, nearly_strict).visible.all()


def test_an_omnidirectional_camera_measures_depth_along_the_ray_even_beyond_ninety_degrees() -> (
    None
):
    wide = MeiCameraModel(
        width=640,
        height=480,
        fx=150.0,
        fy=150.0,
        cx=320.0,
        cy=240.0,
        xi=1.563,
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0),
    )
    theta = math.radians(100.0)
    behind_the_plane = (3.0 * math.sin(theta), 0.0, 3.0 * math.cos(theta))
    frame = _frame(
        [map_point_for_camera_point(behind_the_plane)], calibration=make_calibration(model=wide)
    )

    resolution = resolve_visibility(frame, POLICY)

    assert frame.camera_depth_m[0] < 0
    assert resolution.depth_metric is DepthMetric.RAY_RANGE
    np.testing.assert_allclose(resolution.depth_m, [3.0])
    assert resolution.visible.tolist() == [True]


# --- Valid support ----------------------------------------------------------


def test_a_point_outside_the_valid_support_is_never_visible_though_it_still_occludes() -> None:
    prepared = make_prepared_image(
        exclusion_regions=(
            ExclusionRegion(
                name="own_body",
                mask=mask_with(640, 480, (320, 240)),
                reason="fixture",
                source="test",
            ),
        )
    )
    frame = _scene((320, 240, 2.0), (322, 240, 6.0), prepared=prepared)

    resolution = resolve_visibility(frame, POLICY)

    assert frame.audit(0).stage is ProjectionStage.OUTSIDE_VALID_SUPPORT
    assert resolution.outside_valid_support.tolist() == [True, False]
    assert resolution.visible.tolist() == [False, False]
    assert resolution.occluded.tolist() == [False, True]


def test_points_without_a_pixel_do_not_take_part_in_occlusion() -> None:
    frame = _frame([(-3.0, 0.0, 0.0), map_point_for_pixel(320, 240, 6.0), (0.5, -10.0, 0.0)])

    resolution = resolve_visibility(frame, POLICY)

    assert resolution.behind_camera.tolist() == [True, False, False]
    assert resolution.outside_image.tolist() == [False, False, True]
    assert resolution.visible.tolist() == [False, True, False]
    assert np.isnan(resolution.support_depth_m[[0, 2]]).all()


def test_the_states_partition_every_point() -> None:
    frame = _frame(
        [
            map_point_for_pixel(320, 240, 2.0),
            map_point_for_pixel(320, 240, 6.0),
            (-3.0, 0.0, 0.0),
            (0.5, -10.0, 0.0),
            map_point_for_pixel(100, 100, 4.0),
        ]
    )

    resolution = resolve_visibility(frame, POLICY)

    assert resolution.state_counts() == {
        VisibilityState.BEHIND_CAMERA: 1,
        VisibilityState.OUTSIDE_IMAGE: 1,
        VisibilityState.OUTSIDE_VALID_SUPPORT: 0,
        VisibilityState.OCCLUDED: 1,
    }
    assert resolution.visible_count == 2
    assert sum(resolution.state_counts().values()) + resolution.visible_count == 5


def test_a_frame_with_no_point_in_the_image_resolves_to_nothing_visible() -> None:
    frame = _frame([(-3.0, 0.0, 0.0), (-1.0, 2.0, 0.0)])

    resolution = resolve_visibility(frame, POLICY)

    assert not resolution.visible.any()
    assert resolution.visible_count == 0
    assert resolution.neighborhood_only_occlusions == 0


def test_the_result_does_not_depend_on_the_order_of_the_points() -> None:
    pixels, _ = _sparse_wall_scene()
    forward = resolve_visibility(_scene(*pixels), POLICY)
    order = np.random.default_rng(7).permutation(len(pixels))
    shuffled = resolve_visibility(_scene(*[pixels[i] for i in order]), POLICY)

    assert shuffled.visible.tolist() == forward.visible[order].tolist()
    assert shuffled.occluded.tolist() == forward.occluded[order].tolist()


# --- Auditable decisions ----------------------------------------------------


def test_every_decision_keeps_the_pixel_the_depth_the_support_and_the_reason() -> None:
    frame = _frame(
        [
            map_point_for_pixel(320, 240, 2.0),
            map_point_for_pixel(320, 240, 6.0),
            (-3.0, 0.0, 0.0),
            (0.5, -10.0, 0.0),
        ]
    )
    resolution = resolve_visibility(frame, POLICY)

    front = resolution.correspondence(0, visible_state=VisibilityState.ASSOCIATED)
    hidden = resolution.correspondence(1, visible_state=VisibilityState.ASSOCIATED)
    behind = resolution.correspondence(2, visible_state=VisibilityState.ASSOCIATED)
    beside = resolution.correspondence(3, visible_state=VisibilityState.ASSOCIATED)

    assert front.visibility is VisibilityState.ASSOCIATED
    assert front.geometry == frame.map_reference(0)
    assert front.prepared_pixel == pytest.approx((320.0, 240.0))
    assert (front.camera_depth_m, front.support_depth_m) == pytest.approx((2.0, 2.0))
    assert hidden.visibility is VisibilityState.OCCLUDED
    assert (hidden.camera_depth_m, hidden.support_depth_m) == pytest.approx((6.0, 2.0))
    assert behind.visibility is VisibilityState.BEHIND_CAMERA
    assert behind.prepared_pixel is None
    assert beside.visibility is VisibilityState.OUTSIDE_IMAGE
    assert beside.support_depth_m is None
    assert resolution.policy == POLICY
    assert resolution.frame is frame


def test_a_visible_point_takes_the_state_the_membership_step_gives_it() -> None:
    frame = _scene((320, 240, 2.0))
    resolution = resolve_visibility(frame, POLICY)

    unassigned = resolution.correspondence(0, visible_state=VisibilityState.VISIBLE_UNASSIGNED)

    assert unassigned.visibility is VisibilityState.VISIBLE_UNASSIGNED
    with pytest.raises(ValueError, match="visible_state"):
        resolution.correspondence(0, visible_state=VisibilityState.OCCLUDED)


def test_the_resolution_arrays_are_consistent() -> None:
    resolution = resolve_visibility(_scene((320, 240, 2.0), (320, 240, 6.0)), POLICY)

    with pytest.raises(ValueError, match="visible"):
        dataclasses.replace(resolution, visible=np.array([True, True]))
    with pytest.raises(ValueError, match="shape"):
        dataclasses.replace(resolution, depth_m=np.zeros(3))
