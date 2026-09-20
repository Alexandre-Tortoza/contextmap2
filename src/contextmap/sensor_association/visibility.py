"""Visibility and occlusion of projected persistent geometry.

A valid projection does not prove that a point was visible: points behind a nearer
surface land on the same or neighboring pixels and must not inherit foreground
evidence. Persistent maps are sparse, so an exact-pixel depth buffer lets background
points leak through the gaps of a front surface. This module resolves visibility with
a **conservative local depth support**:

1. the prepared image is divided into square cells of ``cell_size_px`` pixels;
2. each cell keeps the nearest depth among the points that projected into it;
3. a point is compared with the nearest depth in the window of
   ``(2 * neighborhood_radius_cells + 1)`` cells around its own cell;
4. it is occluded when it is farther than that support by more than the depth margin
   ``max(depth_margin_m, depth_margin_ratio * support)``.

Every parameter is explicit policy (:class:`OcclusionPolicy`) with no default: the
neighborhood is not a universal constant, and it is measured in prepared-image pixels,
the space regions live in. Occluders are all the points that landed inside the prepared
image, including those outside the valid region: occlusion is physical, whereas the valid
region only limits which points may become evidence.

Depth is measured under a :class:`~contextmap.sensor_association.DepthMetric` chosen from
the camera model (:func:`depth_metric_for`): along the optical axis for a perspective
camera, along the viewing ray for a model whose field of view can exceed a hemisphere.

The result keeps, per point, the projected pixel, the depth, the nearest supporting depth
and the reason, so each decision can be audited. It does **not** choose a region: a visible
point is only a candidate until mask membership assigns it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.models import DepthMetric, PointCorrespondence, VisibilityState

if TYPE_CHECKING:
    from numpy.typing import NDArray

_ASSIGNED_VISIBLE_STATES = (VisibilityState.ASSOCIATED, VisibilityState.VISIBLE_UNASSIGNED)


def depth_metric_for(camera_model_kind: str) -> DepthMetric:
    """Choose how depth is measured for a camera model.

    Args:
        camera_model_kind: ``"pinhole"``, ``"fisheye"`` or ``"mei"``.

    Returns:
        ``OPTICAL_AXIS`` for a perspective camera, where a fronto-parallel surface has
        constant depth; ``RAY_RANGE`` for fisheye and MEI, whose field of view can exceed
        a hemisphere so that ``z`` is undefined or non-positive at the periphery.

    Raises:
        ValueError: If the camera model kind is unknown.
    """
    if camera_model_kind == "pinhole":
        return DepthMetric.OPTICAL_AXIS
    if camera_model_kind in ("fisheye", "mei"):
        return DepthMetric.RAY_RANGE
    raise ValueError(f"no depth metric is defined for the camera model kind {camera_model_kind!r}")


@dataclass(frozen=True, kw_only=True)
class OcclusionPolicy:
    """The parameters of the conservative depth-support occlusion rule.

    There are no defaults: a neighborhood or margin that suits one map density and
    camera does not suit another, so it is chosen and recorded per run.

    Attributes:
        cell_size_px: Side of a support cell, in prepared-image pixels.
        neighborhood_radius_cells: Radius, in cells, of the window whose nearest depth
            supports a point; ``0`` compares a point only with its own cell.
        depth_margin_m: Absolute depth tolerance, in meters.
        depth_margin_ratio: Depth tolerance as a fraction of the supporting depth.
    """

    policy_id: ClassVar[str] = "conservative-depth-support-v1"

    cell_size_px: int
    neighborhood_radius_cells: int
    depth_margin_m: float
    depth_margin_ratio: float

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a size is not positive, the radius is negative, or a margin is
                negative or not finite.
        """
        if self.cell_size_px < 1:
            raise ValueError(f"cell_size_px must be at least 1, got {self.cell_size_px}")
        if self.neighborhood_radius_cells < 0:
            raise ValueError(
                "neighborhood_radius_cells must not be negative, "
                f"got {self.neighborhood_radius_cells}"
            )
        for name in ("depth_margin_m", "depth_margin_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and not negative, got {value!r}")

    def to_record(self) -> dict[str, Any]:
        """Return the policy as JSON primitives, with its versioned identity."""
        return {
            "policy_id": self.policy_id,
            "cell_size_px": self.cell_size_px,
            "neighborhood_radius_cells": self.neighborhood_radius_cells,
            "depth_margin_m": self.depth_margin_m,
            "depth_margin_ratio": self.depth_margin_ratio,
        }

    def fingerprint(self) -> str:
        """Return ``"sha256:<hex>"`` over the policy identity and parameters."""
        payload = json.dumps(self.to_record(), sort_keys=True).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, kw_only=True, eq=False)
class VisibilityResolution:
    """The visibility of every projected point of one frame.

    The five masks partition the points: a point is exactly one of behind the camera,
    outside the image, outside the valid support, occluded, or visible. A visible point is
    still only a candidate; mask membership assigns it later.

    Attributes:
        frame: The projection the decisions were made on.
        policy: The occlusion policy applied.
        depth_metric: How depth was measured.
        depth_m: ``(N,)`` depth of each point under ``depth_metric``, in meters.
        support_depth_m: ``(N,)`` nearest supporting depth around each point that landed in
            the prepared image, in meters; ``NaN`` for the others.
        behind_camera: ``(N,)`` the camera model cannot project the point.
        outside_image: ``(N,)`` it projects outside the prepared image.
        outside_valid_support: ``(N,)`` it lands outside the valid region or inside an
            exclusion region.
        occluded: ``(N,)`` a nearer supported surface hides it.
        visible: ``(N,)`` it is supported and not occluded.
        neighborhood_only_occlusions: Points the neighborhood occluded that an
            exact-cell depth buffer would have let through: the leakage the policy stops.
    """

    frame: FrameProjection
    policy: OcclusionPolicy
    depth_metric: DepthMetric
    depth_m: NDArray[Any]
    support_depth_m: NDArray[Any]
    behind_camera: NDArray[Any]
    outside_image: NDArray[Any]
    outside_valid_support: NDArray[Any]
    occluded: NDArray[Any]
    visible: NDArray[Any]
    neighborhood_only_occlusions: int

    def __post_init__(self) -> None:
        """Validate that the arrays describe the frame's points and partition them.

        Raises:
            ValueError: If a shape disagrees with the frame or the masks do not assign each
                point to exactly one state.
        """
        import numpy as np

        count = len(self.frame.projectable)
        masks = (
            self.behind_camera,
            self.outside_image,
            self.outside_valid_support,
            self.occluded,
            self.visible,
        )
        if (
            self.depth_m.shape != (count,)
            or self.support_depth_m.shape != (count,)
            or any(mask.shape != (count,) for mask in masks)
        ):
            raise ValueError(f"per-point arrays must have shape ({count},)")
        states_per_point = sum(mask.astype(int) for mask in masks)
        if not np.array_equal(states_per_point, np.ones(count, dtype=int)):
            raise ValueError(
                "behind_camera, outside_image, outside_valid_support, occluded and visible "
                "must assign each point to exactly one state"
            )
        if not np.array_equal(self.visible, self.frame.in_valid_support & ~self.occluded):
            raise ValueError("visible must be the supported points that are not occluded")

    @property
    def visible_count(self) -> int:
        """Number of visible points, still unassigned to any region."""
        return int(self.visible.sum())

    def state_counts(self) -> dict[VisibilityState, int]:
        """Count the points that ended in each decided state; every state is present.

        Visible points are not counted here: they become ``ASSOCIATED`` or
        ``VISIBLE_UNASSIGNED`` only once mask membership is known (:attr:`visible_count`).
        """
        return {
            VisibilityState.BEHIND_CAMERA: int(self.behind_camera.sum()),
            VisibilityState.OUTSIDE_IMAGE: int(self.outside_image.sum()),
            VisibilityState.OUTSIDE_VALID_SUPPORT: int(self.outside_valid_support.sum()),
            VisibilityState.OCCLUDED: int(self.occluded.sum()),
        }

    def correspondence(self, index: int, *, visible_state: VisibilityState) -> PointCorrespondence:
        """Return one point's outcome, with its pixel, depth, support and reason.

        Args:
            index: Position of the point in the frame.
            visible_state: The state a visible point takes, decided by mask membership:
                ``ASSOCIATED`` or ``VISIBLE_UNASSIGNED``. Ignored for any other point.

        Returns:
            The point's record.

        Raises:
            ValueError: If ``visible_state`` is not one of the two visible states.
        """
        if visible_state not in _ASSIGNED_VISIBLE_STATES:
            raise ValueError(
                f"visible_state must be one of {[s.value for s in _ASSIGNED_VISIBLE_STATES]}, "
                f"got {visible_state.value!r}"
            )
        audit = self.frame.audit(index)
        if self.behind_camera[index]:
            state = VisibilityState.BEHIND_CAMERA
        elif self.outside_image[index]:
            state = VisibilityState.OUTSIDE_IMAGE
        elif self.outside_valid_support[index]:
            state = VisibilityState.OUTSIDE_VALID_SUPPORT
        elif self.occluded[index]:
            state = VisibilityState.OCCLUDED
        else:
            state = visible_state
        support = float(self.support_depth_m[index])
        return PointCorrespondence(
            geometry=audit.geometry,
            visibility=state,
            camera_depth_m=float(self.depth_m[index]),
            raw_pixel=None if state is VisibilityState.BEHIND_CAMERA else audit.raw_pixel,
            prepared_pixel=None if state is VisibilityState.BEHIND_CAMERA else audit.prepared_pixel,
            support_depth_m=None if math.isnan(support) else support,
        )


def resolve_visibility(frame: FrameProjection, policy: OcclusionPolicy) -> VisibilityResolution:
    """Decide which projected points are occluded and which remain visible.

    Args:
        frame: The projection of a map into one camera frame.
        policy: The conservative depth-support parameters.

    Returns:
        The per-point decisions with their depth and supporting depth.
    """
    import numpy as np

    metric = depth_metric_for(frame.camera.camera_model_kind)
    depth = frame.camera_depth_m if metric is DepthMetric.OPTICAL_AXIS else frame.camera_range_m
    count = len(depth)
    support = np.full(count, np.nan)
    occluded = np.zeros(count, dtype=bool)
    neighborhood_only = 0

    in_image = np.flatnonzero(frame.in_prepared_image)
    if in_image.size:
        width, height = frame.image_transform.prepared_size
        columns = -(-width // policy.cell_size_px)
        rows = -(-height // policy.cell_size_px)
        # As bordas do pixel de centro c estão em c +- 0.5; a célula é floor(borda / lado).
        edges = frame.prepared_pixels[in_image] + 0.5
        cell_x = np.minimum((edges[:, 0] // policy.cell_size_px).astype(np.int64), columns - 1)
        cell_y = np.minimum((edges[:, 1] // policy.cell_size_px).astype(np.int64), rows - 1)
        point_depth = depth[in_image]

        flat_grid = np.full(rows * columns, np.inf)
        np.minimum.at(flat_grid, cell_y * columns + cell_x, point_depth)
        grid = flat_grid.reshape(rows, columns)
        window = _minimum_filter(grid, policy.neighborhood_radius_cells)

        supported = window[cell_y, cell_x]
        own_cell = grid[cell_y, cell_x]
        hidden = point_depth > supported + _margin(policy, supported)
        hidden_by_own_cell = point_depth > own_cell + _margin(policy, own_cell)
        candidates = frame.in_valid_support[in_image]
        support[in_image] = supported
        occluded[in_image] = hidden & candidates
        neighborhood_only = int((hidden & ~hidden_by_own_cell & candidates).sum())

    projectable = frame.projectable
    return VisibilityResolution(
        frame=frame,
        policy=policy,
        depth_metric=metric,
        depth_m=depth,
        support_depth_m=support,
        behind_camera=~projectable,
        outside_image=projectable & ~frame.in_prepared_image,
        outside_valid_support=frame.in_prepared_image & ~frame.in_valid_support,
        occluded=occluded,
        visible=frame.in_valid_support & ~occluded,
        neighborhood_only_occlusions=neighborhood_only,
    )


def _margin(policy: OcclusionPolicy, support: NDArray[Any]) -> NDArray[Any]:
    import numpy as np

    margin: NDArray[Any] = np.maximum(policy.depth_margin_m, policy.depth_margin_ratio * support)
    return margin


def _minimum_filter(grid: NDArray[Any], radius: int) -> NDArray[Any]:
    """Minimum over a ``(2 * radius + 1)`` square window, truncated at the grid border."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    if radius == 0:
        return grid
    padded = np.pad(grid, radius, constant_values=np.inf)
    size = 2 * radius + 1
    rows: NDArray[Any] = sliding_window_view(padded, size, axis=0).min(axis=-1)
    filtered: NDArray[Any] = sliding_window_view(rows, size, axis=1).min(axis=-1)
    return filtered
