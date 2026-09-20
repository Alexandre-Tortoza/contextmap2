"""Calibrated camera projection behind one contract.

Sensor Association needs physically meaningful projection from a point in the
camera optical frame to an image pixel, and back. The camera model comes only
from the canonical calibration (:class:`~contextmap.ingestion.CalibrationEntry`):
:func:`camera_projection_for` selects pinhole, equidistant fisheye or unified
omnidirectional (MEI) math from the declared model kind and never substitutes
one for another. There is no dataset knowledge and no source-file parsing here.

Conventions
    Points are in the camera optical frame of the calibration entry: ``x`` to
    the right, ``y`` down, ``z`` forward, meters. Pixels are continuous ``(u, v)``
    with the pixel *center* at the integer, so the image covers
    ``[-0.5, width - 0.5) x [-0.5, height - 0.5)``.

Viewing domain
    Every model can project only part of the sphere of directions. A point the
    model cannot project is reported as not ``projectable`` with ``NaN`` pixels;
    it is never clipped into the image. The limit is where the projection stops
    being injective: the plane ``z = 0`` for a pinhole, the angle where the
    radius stops growing for a fisheye, and the horizon of the sphere model for
    MEI (``acos(-1 / xi)`` for ``xi > 1``, ``acos(-xi)`` otherwise). Distortion
    polynomials are trusted inside that domain; a calibration whose polynomial
    folds earlier is a calibration problem that reprojection diagnostics expose.

NumPy is imported lazily so that importing the capability contracts stays cheap.
See ``src/contextmap/sensor_association/docs/camera_models.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationError,
    CalibrationReferenceId,
    CalibrationSet,
    DistortionModel,
    FisheyeCameraModel,
    FrameId,
    MeiCameraModel,
    PinholeCameraModel,
    camera_model_kind,
    validate_calibration_set,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_COSINE_MARGIN = 1e-9
"""Margin, in cosine of the angle to the optical axis, kept from the domain limit.

It keeps the projected coordinates finite next to the boundary, where the
denominators of the models vanish.
"""

_UNDISTORT_ITERATIONS = 100
_UNDISTORT_STEP_TOLERANCE = 1e-13
_UNDISTORT_RESIDUAL_TOLERANCE = 1e-9
"""Normalized-plane residual above which an undistorted pixel is rejected as ray-less."""

_RADIUS_SCAN_STEPS = 31416
"""Angular resolution (about 1e-4 rad) of the scan for the fisheye radius limit."""

_BISECTION_STEPS = 64


@dataclass(frozen=True, kw_only=True)
class CameraIdentity:
    """Which calibrated camera produced a projection.

    Attributes:
        calibration_id: The calibration entry the camera was built from.
        content_hash: Hash of that entry's canonical values.
        camera_frame: The optical frame points must be expressed in.
        camera_model_kind: ``"pinhole"``, ``"fisheye"`` or ``"mei"``.
        image_size: ``(width, height)`` of the raw image, in pixels.
    """

    calibration_id: CalibrationReferenceId
    content_hash: str
    camera_frame: FrameId
    camera_model_kind: str
    image_size: tuple[int, int]


@dataclass(frozen=True, kw_only=True, eq=False)
class PixelProjection:
    """The projection of ``N`` camera-frame points, with the camera that made it.

    Attributes:
        camera: The calibrated camera that produced the projection.
        pixels: ``(N, 2)`` raw-image pixels; ``NaN`` where a point is not projectable.
        projectable: ``(N,)`` ``True`` where the point lies in the model's viewing domain.
        depth_m: ``(N,)`` signed ``z`` of each point in the camera frame, in meters.
        range_m: ``(N,)`` distance from the optical center to each point, in meters.
    """

    camera: CameraIdentity
    pixels: NDArray[Any]
    projectable: NDArray[Any]
    depth_m: NDArray[Any]
    range_m: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate that the arrays describe the same points.

        Raises:
            ValueError: If the shapes disagree or ``projectable`` is not true
                exactly where the pixel is finite.
        """
        import numpy as np

        if self.projectable.ndim != 1:
            raise ValueError(f"projectable must have shape (N,), got {self.projectable.shape}")
        count = self.projectable.shape[0]
        if self.pixels.shape != (count, 2):
            raise ValueError(f"pixels must have shape ({count}, 2), got {self.pixels.shape}")
        if self.depth_m.shape != (count,) or self.range_m.shape != (count,):
            raise ValueError(f"depth_m and range_m must have shape ({count},)")
        if not np.array_equal(np.isfinite(self.pixels).all(axis=1), self.projectable):
            raise ValueError("projectable must be true exactly where the pixel is finite")


@runtime_checkable
class CameraProjection(Protocol):
    """Capability port: project camera-frame points to pixels and back."""

    @property
    def identity(self) -> CameraIdentity:
        """The calibrated camera this projection was built from."""
        ...

    def project(self, points_camera_m: NDArray[Any]) -> PixelProjection:
        """Project points expressed in the camera optical frame.

        Args:
            points_camera_m: ``(N, 3)`` finite points, in meters.

        Returns:
            The pixels, the depth and range of each point, and which points the
            model could project.

        Raises:
            ValueError: If the input is not a finite ``(N, 3)`` array.
        """
        ...

    def unproject(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Return the unit viewing ray of each pixel, in the camera optical frame.

        Args:
            pixels: ``(N, 2)`` finite raw-image pixels.

        Returns:
            ``(N, 3)`` unit vectors.

        Raises:
            ValueError: If the input is not a finite ``(N, 2)`` array or a pixel
                has no viewing ray under the model.
        """
        ...

    def in_image(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Tell which pixels fall inside the raw image extent.

        Args:
            pixels: ``(N, 2)`` pixels; ``NaN`` is never inside.

        Returns:
            ``(N,)`` booleans.
        """
        ...


def _numpy() -> Any:
    import numpy as np

    return np


def _as_points(value: NDArray[Any]) -> NDArray[Any]:
    np = _numpy()
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera_m must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_camera_m must be finite")
    return points  # type: ignore[no-any-return]


def _as_pixels(value: NDArray[Any], *, finite: bool) -> NDArray[Any]:
    np = _numpy()
    pixels = np.asarray(value, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2), got {pixels.shape}")
    if finite and not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite")
    return pixels  # type: ignore[no-any-return]


def _undistort(
    distort: Callable[[NDArray[Any], NDArray[Any]], tuple[NDArray[Any], NDArray[Any]]],
    xd: NDArray[Any],
    yd: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Invert a normalized-plane distortion by fixed-point iteration.

    Returns:
        The undistorted coordinates and a mask of the pixels whose residual is
        within tolerance; the others have no trustworthy inverse.
    """
    np = _numpy()
    x, y = xd.copy(), yd.copy()
    for _ in range(_UNDISTORT_ITERATIONS):
        fx_, fy_ = distort(x, y)
        # x <- x - (f(x) - x_d): passo com Jacobiano identidade, que contrai enquanto
        # a distorção varia devagar dentro do domínio calibrado.
        step_x, step_y = fx_ - xd, fy_ - yd
        x, y = x - step_x, y - step_y
        largest_step = max(float(np.abs(step_x).max()), float(np.abs(step_y).max()))
        if largest_step < _UNDISTORT_STEP_TOLERANCE:
            break
    fx_, fy_ = distort(x, y)
    converged = (np.abs(fx_ - xd) < _UNDISTORT_RESIDUAL_TOLERANCE) & (
        np.abs(fy_ - yd) < _UNDISTORT_RESIDUAL_TOLERANCE
    )
    return x, y, converged


class _CalibratedProjection:
    """Shared plumbing: identity, validation, pixel extent and result assembly.

    Subclasses own the model math: the viewing domain, the forward projection of
    in-domain points and the inverse ray of a pixel.
    """

    def __init__(
        self,
        entry: CalibrationEntry,
        model: PinholeCameraModel | FisheyeCameraModel | MeiCameraModel,
    ) -> None:
        self._identity = CameraIdentity(
            calibration_id=entry.calibration_id,
            content_hash=entry.content_hash,
            camera_frame=entry.frame_id,
            camera_model_kind=camera_model_kind(model),
            image_size=(model.width, model.height),
        )
        self._fx, self._fy, self._cx, self._cy = model.fx, model.fy, model.cx, model.cy

    @property
    def identity(self) -> CameraIdentity:
        """The calibrated camera this projection was built from."""
        return self._identity

    def project(self, points_camera_m: NDArray[Any]) -> PixelProjection:
        """Project points, reporting the ones outside the viewing domain."""
        np = _numpy()
        points = _as_points(points_camera_m)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        range_m = np.sqrt(x * x + y * y + z * z)
        projectable = self._in_domain(x, y, z, range_m)
        pixels = np.full((points.shape[0], 2), np.nan)
        if projectable.any():
            pixels[projectable] = self._to_pixels(
                x[projectable], y[projectable], z[projectable], range_m[projectable]
            )
        return PixelProjection(
            camera=self._identity,
            pixels=pixels,
            projectable=projectable,
            depth_m=z.copy(),
            range_m=range_m,
        )

    def unproject(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Return unit viewing rays; a pixel with no ray is an error, not a guess."""
        np = _numpy()
        array = _as_pixels(pixels, finite=True)
        xd = (array[:, 0] - self._cx) / self._fx
        yd = (array[:, 1] - self._cy) / self._fy
        rays, has_ray = self._to_rays(xd, yd)
        if not has_ray.all():
            first = array[int(np.argmin(has_ray))]
            raise ValueError(
                f"{int((~has_ray).sum())} pixel(s) have no viewing ray under the "
                f"{self._identity.camera_model_kind} model, e.g. {tuple(float(v) for v in first)}"
            )
        return rays  # type: ignore[no-any-return]

    def in_image(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Tell which pixels fall inside ``[-0.5, size - 0.5)`` on both axes."""
        array = _as_pixels(pixels, finite=False)
        width, height = self._identity.image_size
        u, v = array[:, 0], array[:, 1]
        return (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)  # type: ignore[no-any-return]

    def _apply_intrinsics(self, xd: NDArray[Any], yd: NDArray[Any]) -> NDArray[Any]:
        """Apply focal lengths and principal point to normalized-plane coordinates."""
        np = _numpy()
        pixels: NDArray[Any] = np.column_stack((self._fx * xd + self._cx, self._fy * yd + self._cy))
        return pixels

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        raise NotImplementedError

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        raise NotImplementedError

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        raise NotImplementedError


def _pinhole_coefficients(model: PinholeCameraModel) -> tuple[float, ...]:
    """Pad to OpenCV's ``(k1, k2, p1, p2, k3, k4, k5, k6)``."""
    coefficients = model.distortion_coefficients
    return (*coefficients, *([0.0] * (8 - len(coefficients))))


class PinholeProjection(_CalibratedProjection):
    """Pinhole with optional OpenCV radial/tangential (plumb_bob or rational) distortion."""

    def __init__(self, entry: CalibrationEntry, model: PinholeCameraModel) -> None:
        """Build the projection of a validated pinhole calibration entry."""
        super().__init__(entry, model)
        self._distorted = model.distortion_model is not DistortionModel.NONE
        self._k = _pinhole_coefficients(model)

    def _distort(self, x: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        k1, k2, p1, p2, k3, k4, k5, k6 = self._k
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        # z > margem * alcance também exclui a origem (0 > 0 é falso).
        return z > _COSINE_MARGIN * range_m  # type: ignore[no-any-return]

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        xn, yn = x / z, y / z
        if self._distorted:
            xn, yn = self._distort(xn, yn)
        return self._apply_intrinsics(xn, yn)

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        np = _numpy()
        if self._distorted:
            xn, yn, has_ray = _undistort(self._distort, xd, yd)
        else:
            xn, yn, has_ray = xd, yd, np.ones(xd.shape, dtype=bool)
        rays = np.column_stack((xn, yn, np.ones_like(xn)))
        return rays / np.linalg.norm(rays, axis=1, keepdims=True), has_ray


class FisheyeProjection(_CalibratedProjection):
    """Equidistant (Kannala-Brandt) fisheye.

    The radius is ``theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8)`` for the
    angle ``theta`` to the optical axis.
    """

    def __init__(self, entry: CalibrationEntry, model: FisheyeCameraModel) -> None:
        """Build the projection of a validated fisheye calibration entry."""
        super().__init__(entry, model)
        self._k = model.distortion_coefficients
        self._theta_max = self._radius_growth_limit()
        self._theta_d_max = float(self._theta_d(_numpy().array([self._theta_max]))[0])

    def _theta_d(self, theta: NDArray[Any]) -> NDArray[Any]:
        k1, k2, k3, k4 = self._k
        t2 = theta * theta
        return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))  # type: ignore[no-any-return]

    def _radius_growth_limit(self) -> float:
        """Largest angle up to which the radius keeps growing with the angle.

        ``d(theta_d)/d(theta) = 1 + 3 k1 t^2 + 5 k2 t^4 + 7 k3 t^6 + 9 k4 t^8``; past its
        first zero two angles share a radius, so the projection is no longer injective.
        Falls back to ``pi`` when it never vanishes in ``(0, pi)``.
        """
        np = _numpy()
        k1, k2, k3, k4 = self._k

        def growth(theta: NDArray[Any]) -> NDArray[Any]:
            t2 = theta * theta
            return 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))  # type: ignore[no-any-return]

        theta = np.linspace(0.0, math.pi, _RADIUS_SCAN_STEPS)
        stalled = np.flatnonzero(growth(theta) <= 0)
        if stalled.size == 0:
            return math.pi
        low, high = float(theta[stalled[0] - 1]), float(theta[stalled[0]])
        for _ in range(_BISECTION_STEPS):
            middle = (low + high) / 2
            if float(growth(np.array([middle]))[0]) > 0:
                low = middle
            else:
                high = middle
        return low

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        np = _numpy()
        theta = np.arctan2(np.hypot(x, y), z)
        return (range_m > 0) & (theta < self._theta_max)  # type: ignore[no-any-return]

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        np = _numpy()
        rho = np.hypot(x, y)
        theta_d = self._theta_d(np.arctan2(rho, z))
        # Sobre o eixo (rho = 0) a direção é indefinida e o raio do pixel é zero.
        scale = np.divide(theta_d, rho, out=np.zeros_like(rho), where=rho > 0)
        return self._apply_intrinsics(scale * x, scale * y)

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        np = _numpy()
        radius = np.hypot(xd, yd)
        has_ray = radius <= self._theta_d_max
        # theta_d é monótona em [0, theta_max]: a bisseção é robusta perto do limite,
        # onde o Newton divergiria (derivada tendendo a zero).
        low = np.zeros_like(radius)
        high = np.full_like(radius, self._theta_max)
        for _ in range(_BISECTION_STEPS):
            middle = (low + high) / 2
            below = self._theta_d(middle) < radius
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        theta = (low + high) / 2
        scale = np.divide(np.sin(theta), radius, out=np.zeros_like(radius), where=radius > 0)
        return np.column_stack((scale * xd, scale * yd, np.cos(theta))), has_ray


class MeiProjection(_CalibratedProjection):
    """Unified omnidirectional model as parameterized by CamOdoCal (Geyer-Daniilidis / Mei)."""

    def __init__(self, entry: CalibrationEntry, model: MeiCameraModel) -> None:
        """Build the projection of a validated MEI calibration entry."""
        super().__init__(entry, model)
        self._xi = model.xi
        self._k1, self._k2, self._p1, self._p2 = model.distortion_coefficients
        # Cosseno mínimo do ângulo com o eixo: acima do horizonte da esfera (xi > 1) a
        # projeção dobra de volta; para xi < 1 o denominador z + xi * alcance zera antes.
        self._cosine_limit = -min(self._xi, 1 / self._xi) if self._xi > 0 else 0.0

    def _distort(self, x: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        r2 = x * x + y * y
        radial = self._k1 * r2 + self._k2 * r2 * r2
        dx = x * radial + 2 * self._p1 * x * y + self._p2 * (r2 + 2 * x * x)
        dy = y * radial + 2 * self._p2 * x * y + self._p1 * (r2 + 2 * y * y)
        return x + dx, y + dy

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        return z > (self._cosine_limit + _COSINE_MARGIN) * range_m  # type: ignore[no-any-return]

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        shifted = z + self._xi * range_m
        xd, yd = self._distort(x / shifted, y / shifted)
        return self._apply_intrinsics(xd, yd)

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        np = _numpy()
        x, y, converged = _undistort(self._distort, xd, yd)
        rho2 = x * x + y * y
        discriminant = 1 + (1 - self._xi * self._xi) * rho2
        # discriminante < 0: o pixel está além do horizonte da esfera e não tem raio.
        has_ray = converged & (discriminant >= 0)
        scale = (self._xi + np.sqrt(np.maximum(discriminant, 0.0))) / (1 + rho2)
        return np.column_stack((scale * x, scale * y, scale - self._xi)), has_ray


def camera_projection_for(entry: CalibrationEntry) -> CameraProjection:
    """Build the projection that the canonical calibration declares.

    Args:
        entry: A calibration entry of a camera.

    Returns:
        A pinhole, fisheye or MEI projection according to the entry's camera model.

    Raises:
        ValueError: If the entry has no camera model.
        CalibrationError: If the calibration is invalid (non-finite or non-positive
            parameters, wrong coefficient counts, negative ``xi``, ...).
    """
    model = entry.camera_model
    if model is None:
        raise ValueError(
            f"calibration {entry.calibration_id!r} has no camera model to project with"
        )
    problems = validate_calibration_set(
        CalibrationSet(entries={entry.calibration_id: entry}, static_transforms=())
    )
    if problems:
        raise CalibrationError(f"cannot project with an invalid calibration: {problems}")
    if isinstance(model, PinholeCameraModel):
        return PinholeProjection(entry, model)
    if isinstance(model, FisheyeCameraModel):
        return FisheyeProjection(entry, model)
    return MeiProjection(entry, model)
