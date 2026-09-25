"""Canonical observation-quality metadata for spatial visual evidence.

Sensor Association measures conditions that affect how useful one spatial visual
observation is: how far the supporting geometry is, how much of the region was visible,
how densely it is supported, where it sits in the image, how well the pose aligns in time
and, when a trusted reference exists, the reprojection residual. :class:`ObservationQuality`
keeps those **measurements** as separate typed components with their units and provenance.

It is **not** a semantic confidence, a CLIP or VLM score, a probability that a label is
right, or a fusion weight, and it has no combined scalar: a later fusion policy may derive a
contribution from a declared subset of components, but that policy is not defined here.
Quality is attached to a spatial observation by identity and never changes a semantic claim.

A component that could not be measured is ``None`` and is listed in ``unavailable`` with the
reason; it is never coerced to zero. See
``src/contextmap/sensor_association/docs/quality.md`` for the definitions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import SourceObservationId
from contextmap.sensor_association.models import (
    CalibrationRef,
    DepthMetric,
    PoseRef,
    SpatialObservationId,
)

QUALITY_DEFINITIONS_VERSION = "observation-quality-v2"
"""Versioned identity of the component definitions in :class:`ObservationQuality`.

``v2`` (#562): :class:`ReprojectionStatistics` gained ``unevaluated_count`` and now defines its
rates over the **evaluated** population through :attr:`ReprojectionStatistics.evaluated_count`
and :attr:`ReprojectionStatistics.invalid_rate`, and it refuses to exist without an evaluated
correspondence that projects. A residual measured over a population the candidate policy
narrowed is not the same quantity as one measured over the whole map, so it does not keep the
same identity.
"""


class QualityComponent(Enum):
    """The optional components of an :class:`ObservationQuality`.

    The value of each member is the name used for the component in ``unavailable`` and in
    serialized records.
    """

    SUPPORT_DEPTH = "support_depth"
    SUPPORT_OFF_AXIS_ANGLE = "support_off_axis_angle"
    BORDER_DISTANCE = "border_distance"
    VISIBLE_SHARE = "visible_share"
    OCCLUDED_FRACTION = "occluded_fraction"
    OUTSIDE_VALID_SUPPORT_FRACTION = "outside_valid_support_fraction"
    REPROJECTION = "reprojection"


@dataclass(frozen=True, kw_only=True)
class ValueSummary:
    """Order statistics of a measured quantity over the points of one support.

    Attributes:
        count: Number of measured points, at least one.
        minimum: Smallest value.
        median: Median value.
        maximum: Largest value.
    """

    count: int
    minimum: float
    median: float
    maximum: float

    def __post_init__(self) -> None:
        """Validate the summary.

        Raises:
            ValueError: If ``count`` is not positive, a value is not finite, or the values
                are not in order.
        """
        if self.count < 1:
            raise ValueError(f"count must be at least 1, got {self.count}")
        values = (self.minimum, self.median, self.maximum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"summary values must be finite, got {values!r}")
        if not self.minimum <= self.median <= self.maximum:
            raise ValueError(f"summary values must be in order, got {values!r}")


@dataclass(frozen=True, kw_only=True)
class ReprojectionStatistics:
    """Reprojection residual against a **trusted** set of reference correspondences.

    It exists only when real reference correspondences exist; it is never estimated from the
    association itself.

    Attributes:
        reference_id: Identity of the trusted correspondence set.
        correspondence_count: Number of reference correspondences evaluated.
        invalid_count: Of those, the ones that could not be projected (behind the camera or
            outside the image), which contribute no residual.
        unevaluated_count: Correspondences whose geometry the frame's candidate policy did
            not evaluate at all, so the frame says nothing about them. They are reported
            rather than counted as invalid: not projected and not selected are different
            facts, and a residual over a shrinking population must say so.
        mean_px: Mean residual over the valid correspondences, in pixels.
        median_px: Median residual, in pixels.
        p95_px: 95th percentile residual, in pixels.
        max_px: Largest residual, in pixels.
    """

    reference_id: str
    correspondence_count: int
    invalid_count: int
    unevaluated_count: int
    mean_px: float
    median_px: float
    p95_px: float
    max_px: float

    def __post_init__(self) -> None:
        """Validate the statistics.

        Raises:
            ValueError: If the reference is unnamed, the counts are inconsistent, a value is
                negative or not finite, or the quantiles are not in order.
        """
        if not self.reference_id:
            raise ValueError("reference_id must not be empty")
        if self.correspondence_count < 1:
            raise ValueError("correspondence_count must be at least 1")
        if not 0 <= self.invalid_count <= self.correspondence_count:
            raise ValueError(
                f"invalid_count {self.invalid_count} must be between 0 and the "
                f"correspondence_count {self.correspondence_count}"
            )
        if not 0 <= self.unevaluated_count <= self.correspondence_count:
            raise ValueError(
                f"unevaluated_count {self.unevaluated_count} must be between 0 and the "
                f"correspondence_count {self.correspondence_count}"
            )
        if self.invalid_count + self.unevaluated_count > self.correspondence_count:
            raise ValueError(
                f"invalid_count {self.invalid_count} and unevaluated_count "
                f"{self.unevaluated_count} cannot together exceed the correspondence_count "
                f"{self.correspondence_count}"
            )
        if self.invalid_count >= self.correspondence_count - self.unevaluated_count:
            raise ValueError(
                "residual statistics need at least one evaluated correspondence the camera model "
                f"could project, but {self.invalid_count} of {self.evaluated_count} evaluated were "
                "invalid; a frame with none has no statistics at all"
            )
        values = (self.mean_px, self.median_px, self.p95_px, self.max_px)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"residuals must be finite and not negative, got {values!r}")
        if not self.median_px <= self.p95_px <= self.max_px:
            raise ValueError(
                "residual quantiles must be in order (median <= p95 <= max), got "
                f"{(self.median_px, self.p95_px, self.max_px)!r}"
            )

    @property
    def evaluated_count(self) -> int:
        """Correspondences the frame actually evaluated: the population every rate is over.

        A correspondence the candidate policy excluded says nothing about the camera, so it
        belongs in neither the numerator nor the denominator of an invalid rate.
        """
        return self.correspondence_count - self.unevaluated_count

    @property
    def invalid_rate(self) -> float:
        """Share of the **evaluated** correspondences the camera model could not project.

        Never diluted by the unevaluated ones: with 100 references of which 90 fell outside the
        candidate policy, 1 invalid and 9 valid, this is ``0.1`` and not ``0.01``.
        """
        return self.invalid_count / self.evaluated_count


@dataclass(frozen=True, kw_only=True)
class ObservationQuality:
    """Measurable quality of one spatial observation, as separate components.

    Attributes:
        definitions_version: Version of the component definitions.
        spatial_observation_id: The observation this quality describes.
        source_observation_id: The physical camera frame; regions of one frame share it, so
            repeated inference over one physical frame is not a new view-quality sample.
        geometric_map_id: The geometry artifact the support belongs to.
        calibration_ref: The exact calibration and camera used.
        pose_ref: The exact pose used; ``pose_ref.time_delta_ns`` is the temporal alignment.
        image_transform_id: The raw-to-prepared image chain the pixels went through.
        visibility_policy_id: The occlusion rule applied.
        depth_metric: How ``support_depth_m`` is measured.
        associated_count: Visible geometry inside the region mask.
        footprint_count: Projected geometry of any state inside the mask.
        mask_area_px: Foreground pixels of the region mask.
        support_density_per_mask_pixel: ``associated_count / mask_area_px``.
        support_pixel_coverage: Share of mask pixels holding supported geometry.
        support_depth_m: Depth of the associated geometry, in meters, under ``depth_metric``.
        support_off_axis_angle_rad: Angle of the associated geometry from the optical axis,
            in radians; defined for every camera model.
        border_distance_px: Distance of the associated geometry's prepared-image pixel to the
            nearest border of the prepared image, in pixels.
        visible_share: ``associated_count / footprint_count``.
        occluded_fraction: Share of the footprint hidden by a nearer surface.
        outside_valid_support_fraction: Share of the footprint outside the valid support.
        reprojection: Residual against trusted reference correspondences, when they exist.
        unavailable: The reason for every component that is ``None``.
    """

    definitions_version: str
    spatial_observation_id: SpatialObservationId
    source_observation_id: SourceObservationId
    geometric_map_id: MapId
    calibration_ref: CalibrationRef
    pose_ref: PoseRef
    image_transform_id: str
    visibility_policy_id: str
    depth_metric: DepthMetric
    associated_count: int
    footprint_count: int
    mask_area_px: int
    support_density_per_mask_pixel: float
    support_pixel_coverage: float
    support_depth_m: ValueSummary | None
    support_off_axis_angle_rad: ValueSummary | None
    border_distance_px: ValueSummary | None
    visible_share: float | None
    occluded_fraction: float | None
    outside_valid_support_fraction: float | None
    reprojection: ReprojectionStatistics | None
    unavailable: Mapping[QualityComponent, str]

    def __post_init__(self) -> None:
        """Validate identities, bounds and the availability of every component.

        Raises:
            ValueError: If an identity is empty, a count or fraction is out of bounds, or the
                ``unavailable`` reasons do not match exactly the components that are ``None``.
        """
        for name in (
            "definitions_version",
            "spatial_observation_id",
            "source_observation_id",
            "geometric_map_id",
            "image_transform_id",
            "visibility_policy_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.associated_count < 0 or self.associated_count > self.footprint_count:
            raise ValueError(
                f"associated_count {self.associated_count} must be between 0 and the "
                f"footprint_count {self.footprint_count}"
            )
        if self.mask_area_px < 1:
            raise ValueError("mask_area_px must be at least 1")
        if not math.isfinite(self.support_density_per_mask_pixel) or (
            self.support_density_per_mask_pixel < 0
        ):
            raise ValueError("support_density_per_mask_pixel must be finite and not negative")
        for name in (
            "support_pixel_coverage",
            "visible_share",
            "occluded_fraction",
            "outside_valid_support_fraction",
        ):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be within [0, 1], got {value!r}")
        missing = {component for component in QualityComponent if self._is_missing(component)}
        if set(self.unavailable) != missing:
            raise ValueError(
                "unavailable must give a reason for exactly the missing components: missing "
                f"{sorted(c.value for c in missing)}, with reasons for "
                f"{sorted(c.value for c in self.unavailable)}"
            )
        if any(not reason for reason in self.unavailable.values()):
            raise ValueError("every unavailable component needs a non-empty reason")
        canonical = {c: self.unavailable[c] for c in QualityComponent if c in self.unavailable}
        object.__setattr__(self, "unavailable", canonical)

    @property
    def temporal_offset_ns(self) -> int:
        """Distance in nanoseconds from the frame timestamp to the nearest contributing pose."""
        return self.pose_ref.time_delta_ns

    def _is_missing(self, component: QualityComponent) -> bool:
        values = {
            QualityComponent.SUPPORT_DEPTH: self.support_depth_m,
            QualityComponent.SUPPORT_OFF_AXIS_ANGLE: self.support_off_axis_angle_rad,
            QualityComponent.BORDER_DISTANCE: self.border_distance_px,
            QualityComponent.VISIBLE_SHARE: self.visible_share,
            QualityComponent.OCCLUDED_FRACTION: self.occluded_fraction,
            QualityComponent.OUTSIDE_VALID_SUPPORT_FRACTION: self.outside_valid_support_fraction,
            QualityComponent.REPROJECTION: self.reprojection,
        }
        return values[component] is None
