import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import contextmap.sensor_association.camera_models as camera_models
from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationError,
    CalibrationProvenance,
    CalibrationReferenceId,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    FrameId,
    MeiCameraModel,
    PinholeCameraModel,
    SensorId,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.sensor_association import (
    CameraIdentity,
    CameraProjection,
    PixelProjection,
    camera_projection_for,
)
from contextmap.sensor_association.camera_models import (
    FisheyeProjection,
    MeiProjection,
    PinholeProjection,
)

FX, FY, CX, CY = 500.0, 480.0, 320.0, 240.0
WIDTH, HEIGHT = 640, 480
PLUMB_BOB = (-0.28, 0.07, 0.0002, -0.0001, 0.0)
RATIONAL = (-0.3, 0.09, 0.0003, -0.0002, -0.01, 0.02, 0.005, -0.001)
FISHEYE_K = (0.01, -0.004, 0.0007, -0.0001)
MEI_K = (-0.0708, 0.3095, 0.00046, 0.000157)
MEI_XI = 1.563


def _entry(model: CameraModel | None, *, name: str = "camera") -> CalibrationEntry:
    return CalibrationEntry(
        calibration_id=CalibrationReferenceId(f"{name}-calib"),
        sensor_id=SensorId(name),
        frame_id=FrameId(f"{name}_optical"),
        camera_model=model,
        provenance=CalibrationProvenance(source_type="dataset", source_path="fixtures/cal.yaml"),
        content_hash=compute_content_hash(
            sensor_id=SensorId(name), frame_id=FrameId(f"{name}_optical"), camera_model=model
        ),
    )


def _pinhole_model(
    model: DistortionModel = DistortionModel.NONE, coefficients: tuple[float, ...] = ()
) -> PinholeCameraModel:
    return PinholeCameraModel(
        width=WIDTH,
        height=HEIGHT,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        distortion_model=model,
        distortion_coefficients=coefficients,
    )


def _fisheye_model(k: tuple[float, float, float, float] = FISHEYE_K) -> FisheyeCameraModel:
    return FisheyeCameraModel(
        width=WIDTH, height=HEIGHT, fx=FX, fy=FY, cx=CX, cy=CY, distortion_coefficients=k
    )


def _mei_model(xi: float = MEI_XI, k: tuple[float, float, float, float] = MEI_K) -> MeiCameraModel:
    return MeiCameraModel(
        width=WIDTH, height=HEIGHT, fx=FX, fy=FY, cx=CX, cy=CY, xi=xi, distortion_coefficients=k
    )


def _points(*rows: tuple[float, float, float]) -> NDArray[np.float64]:
    return np.array(rows, dtype=np.float64)


def _at_angle(angle_deg: float, *, range_m: float = 2.0) -> tuple[float, float, float]:
    """A point ``angle_deg`` off the optical axis, in the x-z plane."""
    theta = math.radians(angle_deg)
    return (range_m * math.sin(theta), 0.0, range_m * math.cos(theta))


# --- Trusted formulas, written scalar and independent of the implementation ---


def _reference_pinhole(
    p: tuple[float, float, float], k: tuple[float, ...] = (0.0,) * 8
) -> tuple[float, float]:
    """OpenCV plumb_bob / rational_polynomial: (k1, k2, p1, p2, k3[, k4, k5, k6])."""
    padded = (*k, *([0.0] * (8 - len(k))))
    k1, k2, p1, p2, k3, k4, k5, k6 = padded
    x, y = p[0] / p[2], p[1] / p[2]
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return FX * xd + CX, FY * yd + CY


def _reference_fisheye(
    p: tuple[float, float, float], k: tuple[float, float, float, float]
) -> tuple[float, float]:
    """OpenCV equidistant model, valid for any direction (theta from the axis)."""
    rho = math.hypot(p[0], p[1])
    theta = math.atan2(rho, p[2])
    theta_d = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)
    if rho == 0:
        return CX, CY
    return FX * theta_d * p[0] / rho + CX, FY * theta_d * p[1] / rho + CY


def _reference_mei(
    p: tuple[float, float, float], xi: float, k: tuple[float, float, float, float]
) -> tuple[float, float]:
    """CamOdoCal CataCamera::spaceToPlane."""
    z = p[2] + xi * math.sqrt(sum(c * c for c in p))
    x, y = p[0] / z, p[1] / z
    k1, k2, p1, p2 = k
    r2 = x * x + y * y
    radial = k1 * r2 + k2 * r2 * r2
    xd = x + x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y + y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    return FX * xd + CX, FY * yd + CY


# --- Exact, hand-checkable cases --------------------------------------------


def test_an_ideal_pinhole_projects_by_similar_triangles() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    result = projection.project(_points((0.5, -0.25, 2.0), (0.0, 0.0, 3.0)))

    np.testing.assert_allclose(result.pixels, [[445.0, 180.0], [320.0, 240.0]], atol=1e-12)
    assert result.projectable.tolist() == [True, True]
    np.testing.assert_allclose(result.depth_m, [2.0, 3.0])
    np.testing.assert_allclose(result.range_m, [math.sqrt(4.3125), 3.0])


def test_an_ideal_equidistant_camera_maps_the_angle_linearly_to_the_radius() -> None:
    projection = camera_projection_for(_entry(_fisheye_model((0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)))

    np.testing.assert_allclose(result.pixels[0], [FX * math.pi / 4 + CX, CY], atol=1e-9)
    np.testing.assert_allclose(result.pixels[1], [CX, FY * math.pi / 2 + CY], atol=1e-9)


def test_a_unified_model_with_no_mirror_shift_is_a_pinhole() -> None:
    unified = camera_projection_for(_entry(_mei_model(xi=0.0, k=(0.0, 0.0, 0.0, 0.0))))
    pinhole = camera_projection_for(_entry(_pinhole_model()))
    points = _points((0.5, -0.25, 2.0), (-1.0, 0.7, 4.0))

    np.testing.assert_allclose(unified.project(points).pixels, pinhole.project(points).pixels)


def test_a_unified_model_with_a_unit_mirror_shift_projects_stereographically() -> None:
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)))

    tan_eighth = math.tan(math.pi / 8)
    np.testing.assert_allclose(
        result.pixels,
        [[FX + CX, CY], [FX * tan_eighth + CX, CY], [CX, CY]],
        atol=1e-9,
    )


# --- Agreement with the published formulas ----------------------------------

_FRONT_POINTS = [
    (0.3, -0.2, 1.5),
    (-1.1, 0.6, 2.4),
    (0.9, 0.8, 1.2),
    (-0.05, 0.02, 0.7),
    (2.0, -1.5, 3.0),
]


@pytest.mark.parametrize(
    ("model", "coefficients"),
    [
        (DistortionModel.NONE, ()),
        (DistortionModel.PLUMB_BOB, PLUMB_BOB),
        (DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL),
    ],
)
def test_the_pinhole_matches_the_opencv_distortion_formulas(
    model: DistortionModel, coefficients: tuple[float, ...]
) -> None:
    projection = camera_projection_for(_entry(_pinhole_model(model, coefficients)))

    result = projection.project(_points(*_FRONT_POINTS))

    expected = [_reference_pinhole(p, coefficients) for p in _FRONT_POINTS]
    np.testing.assert_allclose(result.pixels, expected, atol=1e-9)


def test_the_fisheye_matches_the_opencv_equidistant_formula_even_beyond_ninety_degrees() -> None:
    projection = camera_projection_for(_entry(_fisheye_model()))
    points = [*_FRONT_POINTS, _at_angle(95.0), _at_angle(-110.0)]

    result = projection.project(_points(*points))

    expected = [_reference_fisheye(p, FISHEYE_K) for p in points]
    np.testing.assert_allclose(result.pixels, expected, atol=1e-9)
    assert result.projectable.all()


def test_the_unified_model_matches_the_camodocal_formula_beyond_ninety_degrees() -> None:
    projection = camera_projection_for(_entry(_mei_model()))
    points = [*_FRONT_POINTS, _at_angle(95.0), _at_angle(-120.0)]

    result = projection.project(_points(*points))

    expected = [_reference_mei(p, MEI_XI, MEI_K) for p in points]
    np.testing.assert_allclose(result.pixels, expected, atol=1e-9)
    assert result.projectable.all()


# --- The viewing domain is explicit, never clipped --------------------------


def test_a_pinhole_cannot_project_what_is_beside_or_behind_the_camera() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    result = projection.project(
        _points(
            (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1e-12), (1.0, 0.0, 1e-3)
        )
    )

    assert result.projectable.tolist() == [False, False, False, False, True]
    assert np.isnan(result.pixels[:4]).all()
    assert np.isfinite(result.pixels[4]).all()
    np.testing.assert_allclose(result.depth_m, [-1.0, 0.0, 0.0, 1e-12, 1e-3])


def test_a_distorted_pinhole_stops_where_the_distorted_radius_stops_growing() -> None:
    # d(r (1 + k1 r^2))/dr = 1 + 3 k1 r^2 se anula em r = sqrt(1 / (-3 k1)), cerca de 47,5 graus.
    k1 = -0.28
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (k1, 0.0, 0.0, 0.0, 0.0)))
    )
    limit_deg = math.degrees(math.atan(math.sqrt(1 / (-3 * k1))))

    result = projection.project(_points(_at_angle(limit_deg - 1.0), _at_angle(limit_deg + 1.0)))

    assert result.projectable.tolist() == [True, False]
    assert np.isnan(result.pixels[1]).all()


def test_a_point_beyond_the_fold_never_lands_on_the_principal_point() -> None:
    """The audit's reproduction: past the fold the radius shrinks back to zero.

    With ``k1 = -0.28`` alone, ``r (1 + k1 r^2)`` returns to zero at ``r = sqrt(1 / 0.28)``,
    about 62.1 degrees off the axis, so the unguarded polynomial sent that point exactly to the
    principal point, inside the image.
    """
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.0, 0.0, 0.0, 0.0)))
    )
    folded_deg = math.degrees(math.atan(math.sqrt(1 / 0.28)))

    result = projection.project(_points(_at_angle(folded_deg)))

    assert not result.projectable[0]
    assert np.isnan(result.pixels).all()
    assert not projection.in_image(result.pixels)[0]


def test_a_distorted_pinhole_gives_no_ray_beyond_its_fold() -> None:
    """``unproject`` must stay inside the same domain ``project`` accepts.

    With ``k2 > 0`` the distorted radius grows again past the fold, so a pixel beyond the
    fold's peak has a preimage only on that outer branch; returning it would hand out a ray
    that ``project`` refuses.
    """
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.03, 0.0, 0.0, 0.0)))
    )

    with pytest.raises(ValueError, match="no viewing ray"):
        projection.unproject(np.array([[CX + 0.9 * FX, CY]]))


def test_a_fisheye_stops_where_the_radius_stops_growing_with_the_angle() -> None:
    # d(theta_d)/d(theta) = 1 + 3 k1 theta^2 se anula em theta = sqrt(1 / (-3 k1)).
    projection = camera_projection_for(_entry(_fisheye_model((-0.3, 0.0, 0.0, 0.0))))
    limit_deg = math.degrees(math.sqrt(1 / 0.9))

    result = projection.project(_points(_at_angle(limit_deg - 1.0), _at_angle(limit_deg + 1.0)))

    assert result.projectable.tolist() == [True, False]
    assert np.isnan(result.pixels[1]).all()


def test_an_undistorted_fisheye_sees_everything_except_the_exact_opposite_direction() -> None:
    projection = camera_projection_for(_entry(_fisheye_model((0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points(_at_angle(179.0), (0.0, 0.0, -1.0)))

    assert result.projectable.tolist() == [True, False]


@pytest.mark.parametrize(
    ("xi", "inside_deg", "outside_deg"),
    [
        # xi > 1: a projeção dobra de volta além de acos(-1 / xi), cerca de 129,8 graus.
        (MEI_XI, 128.8, 130.8),
        # xi < 1: o denominador de normalização se anula em acos(-xi).
        (0.5, 119.0, 121.0),
        (0.0, 89.0, 91.0),
    ],
)
def test_a_unified_model_stops_at_its_horizon(
    xi: float, inside_deg: float, outside_deg: float
) -> None:
    projection = camera_projection_for(_entry(_mei_model(xi=xi, k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points(_at_angle(inside_deg), _at_angle(outside_deg)))

    assert result.projectable.tolist() == [True, False]
    assert np.isfinite(result.pixels[0]).all()
    assert np.isnan(result.pixels[1]).all()


def test_a_unit_mirror_shift_sees_everything_except_the_exact_opposite_direction() -> None:
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points(_at_angle(179.0), (0.0, 0.0, -1.0)))

    assert result.projectable.tolist() == [True, False]


def test_the_point_on_the_negative_axis_never_folds_back_to_the_principal_point() -> None:
    projection = camera_projection_for(_entry(_mei_model(k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points((0.0, 0.0, -1.0)))

    assert not result.projectable[0]
    assert np.isnan(result.pixels).all()


def test_the_input_must_be_finite_camera_frame_points() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    with pytest.raises(ValueError, match="finite"):
        projection.project(_points((0.0, float("nan"), 1.0)))
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        projection.project(np.zeros((4, 2)))
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        projection.project(np.zeros(3))


# --- Round trip through unproject -------------------------------------------


def _pixel_grid() -> NDArray[np.float64]:
    us = np.linspace(0.0, WIDTH - 1.0, 9)
    vs = np.linspace(0.0, HEIGHT - 1.0, 7)
    grid = np.array([(u, v) for u in us for v in vs], dtype=np.float64)
    return grid


@pytest.mark.parametrize(
    "model",
    [
        _pinhole_model(),
        _pinhole_model(DistortionModel.PLUMB_BOB, PLUMB_BOB),
        _pinhole_model(DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL),
        _fisheye_model(),
        _mei_model(),
        _mei_model(xi=0.6),
    ],
    ids=["pinhole", "plumb_bob", "rational", "fisheye", "mei", "mei-small-xi"],
)
def test_unprojecting_a_pixel_and_projecting_the_ray_returns_the_same_pixel(
    model: CameraModel,
) -> None:
    projection = camera_projection_for(_entry(model))
    pixels = _pixel_grid()

    rays = projection.unproject(pixels)
    back = projection.project(rays * 3.0)

    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)
    assert back.projectable.all()
    np.testing.assert_allclose(back.pixels, pixels, atol=1e-6)


def test_a_pixel_beyond_the_horizon_of_the_model_has_no_ray() -> None:
    # O raio normalizado 1 / sqrt(xi^2 - 1) é o horizonte do modelo unificado.
    xi = 1.5
    projection = camera_projection_for(
        _entry(_mei_model(xi=xi, k=(0.0, 0.0, 0.0, 0.0))),
    )
    horizon_px = FX / math.sqrt(xi * xi - 1)

    with pytest.raises(ValueError, match="no viewing ray"):
        projection.unproject(np.array([[CX + horizon_px * 1.01, CY]]))
    np.testing.assert_allclose(
        projection.unproject(np.array([[CX, CY]])), [[0.0, 0.0, 1.0]], atol=1e-12
    )


def test_unproject_rejects_pixels_that_are_not_finite() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    with pytest.raises(ValueError, match="finite"):
        projection.unproject(np.array([[float("nan"), 1.0]]))


# --- Bound of the ray angle over an image box -------------------------------

_FULL_IMAGE = {"u_bounds": (-0.5, WIDTH - 0.5), "v_bounds": (-0.5, HEIGHT - 0.5)}


def _ray_angle(rays: NDArray[np.float64]) -> NDArray[np.float64]:
    angles: NDArray[np.float64] = np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2])
    return angles


def _image_border(step_px: float = 0.5) -> NDArray[np.float64]:
    """Pixel centers along the four edges of the image, corners included."""
    us = np.arange(0.0, WIDTH - 1.0 + step_px / 2, step_px)
    vs = np.arange(0.0, HEIGHT - 1.0 + step_px / 2, step_px)
    edges = [
        np.column_stack((us, np.zeros_like(us))),
        np.column_stack((us, np.full_like(us, HEIGHT - 1.0))),
        np.column_stack((np.zeros_like(vs), vs)),
        np.column_stack((np.full_like(vs, WIDTH - 1.0), vs)),
    ]
    return np.vstack(edges)


def test_the_ray_angle_bound_of_an_ideal_pinhole_is_its_farthest_corner() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    corner = math.atan(math.hypot(320.5 / FX, 240.5 / FY))
    assert corner <= bound <= corner + 1e-12
    # O canto, não a metade do campo de visão horizontal.
    assert bound > math.atan(320.5 / FX) + 0.1


def test_a_radially_distorted_pinhole_is_bounded_by_the_ray_of_its_farthest_corner() -> None:
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.07, 0.0, 0.0, 0.0)))
    )

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    farthest = float(_ray_angle(projection.unproject(np.array([[-0.5, -0.5]])))[0])
    assert farthest - 1e-12 <= bound <= farthest + 1e-9


def test_a_pinhole_whose_fold_lies_inside_the_image_is_bounded_by_the_fold() -> None:
    # Com k1 = -0,28 sozinho o raio distorcido chega no máximo a 0,727 antes de dobrar,
    # e o canto está a 0,81: a imagem alcança a dobra, então o limite é o domínio.
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.0, 0.0, 0.0, 0.0)))
    )

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    assert bound == pytest.approx(math.atan(math.sqrt(1 / 0.84)), abs=1e-9)


def test_a_fisheye_whose_valid_circle_ends_inside_the_image_is_bounded_by_its_domain() -> None:
    # theta_d cresce até sqrt(1 / 0,9) rad, onde vale 0,703; o canto está a 0,81.
    projection = camera_projection_for(_entry(_fisheye_model((-0.3, 0.0, 0.0, 0.0))))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    assert bound == pytest.approx(math.sqrt(1 / 0.9), abs=1e-9)


def test_a_unified_model_whose_horizon_is_inside_the_image_is_bounded_by_the_horizon() -> None:
    wide = MeiCameraModel(
        width=WIDTH,
        height=HEIGHT,
        fx=200.0,
        fy=200.0,
        cx=CX,
        cy=CY,
        xi=MEI_XI,
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0),
    )
    projection = camera_projection_for(_entry(wide))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    assert bound == pytest.approx(math.acos(-1 / MEI_XI), abs=1e-9)
    assert bound > math.pi / 2


@pytest.mark.parametrize(
    "model",
    [
        _pinhole_model(),
        _pinhole_model(DistortionModel.PLUMB_BOB, PLUMB_BOB),
        _pinhole_model(DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL),
        _fisheye_model(),
        _mei_model(),
        _mei_model(xi=0.6),
    ],
    ids=["pinhole", "plumb_bob", "rational", "fisheye", "mei", "mei-small-xi"],
)
def test_no_ray_the_model_sends_into_the_image_exceeds_the_bound(model: CameraModel) -> None:
    """Sound for every model, tangential terms included, and still close to the border."""
    projection = camera_projection_for(_entry(model))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    border = _ray_angle(projection.unproject(_image_border()))
    assert border.max() <= bound
    assert bound - border.max() < 0.01


def test_the_box_must_be_finite_and_ordered() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    with pytest.raises(ValueError, match="finite and ordered"):
        projection.max_ray_angle_rad(u_bounds=(10.0, 0.0), v_bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="finite and ordered"):
        projection.max_ray_angle_rad(u_bounds=(0.0, 1.0), v_bounds=(0.0, float("nan")))


def test_the_bound_narrows_with_the_box() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    full = projection.max_ray_angle_rad(**_FULL_IMAGE)
    central = projection.max_ray_angle_rad(u_bounds=(219.5, 419.5), v_bounds=(139.5, 339.5))

    assert central == pytest.approx(math.atan(math.hypot(100.5 / FX, 100.5 / FY)), abs=1e-12)
    assert central < full


# --- Pixel domain -----------------------------------------------------------


def test_a_pixel_is_inside_when_its_center_is_within_the_image_extent() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))
    pixels = np.array(
        [
            [0.0, 0.0],
            [-0.5, -0.5],
            [WIDTH - 1.0, HEIGHT - 1.0],
            [WIDTH - 0.5, 10.0],
            [10.0, HEIGHT - 0.5],
            [-0.6, 5.0],
            [float("nan"), 5.0],
        ]
    )

    assert projection.in_image(pixels).tolist() == [True, True, True, False, False, False, False]


# --- Selection and provenance from canonical calibration only ---------------


@pytest.mark.parametrize(
    ("model", "expected_type", "kind"),
    [
        (_pinhole_model(), PinholeProjection, "pinhole"),
        (_fisheye_model(), FisheyeProjection, "fisheye"),
        (_mei_model(), MeiProjection, "mei"),
    ],
)
def test_the_projection_is_selected_solely_by_the_canonical_model_kind(
    model: CameraModel, expected_type: type, kind: str
) -> None:
    entry = _entry(model)

    projection = camera_projection_for(entry)

    assert isinstance(projection, expected_type)
    assert isinstance(projection, CameraProjection)
    assert projection.identity == CameraIdentity(
        calibration_id=entry.calibration_id,
        content_hash=entry.content_hash,
        camera_frame=entry.frame_id,
        camera_model_kind=kind,
        image_size=(WIDTH, HEIGHT),
    )


def test_every_projection_result_names_the_calibration_that_produced_it() -> None:
    projection = camera_projection_for(_entry(_mei_model()))

    result = projection.project(_points((0.0, 0.0, 1.0)))

    assert isinstance(result, PixelProjection)
    assert result.camera == projection.identity


def test_an_entry_without_a_camera_model_cannot_be_projected() -> None:
    with pytest.raises(ValueError, match="camera model"):
        camera_projection_for(_entry(None, name="lidar"))


def test_an_invalid_calibration_is_rejected_instead_of_projected() -> None:
    broken = dataclasses.replace(_pinhole_model(), fx=float("nan"))

    with pytest.raises(CalibrationError, match="fx"):
        camera_projection_for(_entry(broken))


def test_the_projection_result_keeps_its_arrays_consistent() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))
    good = projection.project(_points((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))

    with pytest.raises(ValueError, match="projectable"):
        dataclasses.replace(good, projectable=np.array([True, True]))
    with pytest.raises(ValueError, match="shape"):
        dataclasses.replace(good, depth_m=np.zeros(3))


def test_the_camera_models_know_no_dataset_by_name() -> None:
    source = Path(camera_models.__file__).read_text(encoding="utf-8").lower()

    assert "corridor" not in source
    assert "kalibr" not in source
