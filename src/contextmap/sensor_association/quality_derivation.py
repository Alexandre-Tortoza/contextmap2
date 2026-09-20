"""Derivation of :class:`~contextmap.sensor_association.ObservationQuality` from association data.

Every component is computed from the projection, visibility and membership that produced the
spatial observations, so the same inputs always give the same quality. Nothing is estimated
from the semantic evidence, and a reprojection residual is attached only when trusted
reference correspondences supplied one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.membership import FrameMembership, RegionMembership
from contextmap.sensor_association.models import SpatialObservation
from contextmap.sensor_association.quality import (
    QUALITY_DEFINITIONS_VERSION,
    ObservationQuality,
    QualityComponent,
    ReprojectionStatistics,
    ValueSummary,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_NO_SUPPORT = "no visible geometry is associated with the region"
_NO_FOOTPRINT = "no projected geometry falls inside the region mask"
_NO_REFERENCE = "no trusted reference correspondences were provided"


def derive_observation_quality(
    membership: FrameMembership,
    observations: Sequence[SpatialObservation],
    *,
    reprojection: ReprojectionStatistics | None = None,
) -> tuple[ObservationQuality, ...]:
    """Derive the quality of each spatial observation of one frame.

    Args:
        membership: The membership the observations were built from.
        observations: The observations of that membership, in the same order as its regions
            (as :func:`~contextmap.sensor_association.membership.build_spatial_observations`
            returns them).
        reprojection: Residual statistics against trusted reference correspondences of this
            frame's camera, when they exist; ``None`` leaves the component unavailable.

    Returns:
        One quality per observation.

    Raises:
        AssociationInputError: If the observations are not those of the membership's regions.
    """
    regions = membership.regions
    if len(observations) != len(regions) or any(
        observation.region_id != region.region_id
        for observation, region in zip(observations, regions, strict=True)
    ):
        raise AssociationInputError(
            "the observations must be those of the membership's regions, in the same order"
        )
    return tuple(
        _quality(membership, region, observation, reprojection)
        for region, observation in zip(regions, observations, strict=True)
    )


def _quality(
    membership: FrameMembership,
    region: RegionMembership,
    observation: SpatialObservation,
    reprojection: ReprojectionStatistics | None,
) -> ObservationQuality:
    import numpy as np

    resolution = membership.resolution
    frame = membership.frame
    indices = region.associated_indices
    unavailable: dict[QualityComponent, str] = {}

    depth = angle = border = None
    if indices.size:
        depth = _summary(resolution.depth_m[indices])
        # cos(theta) = z / alcance: o ângulo com o eixo óptico existe em qualquer modelo.
        cosine = frame.camera_depth_m[indices] / frame.camera_range_m[indices]
        angle = _summary(np.arccos(np.clip(cosine, -1.0, 1.0)))
        width, height = frame.image_transform.prepared_size
        edges = frame.prepared_pixels[indices] + 0.5
        distance = np.minimum.reduce(
            [edges[:, 0], width - edges[:, 0], edges[:, 1], height - edges[:, 1]]
        )
        border = _summary(distance)
    else:
        for component in (
            QualityComponent.SUPPORT_DEPTH,
            QualityComponent.SUPPORT_OFF_AXIS_ANGLE,
            QualityComponent.BORDER_DISTANCE,
        ):
            unavailable[component] = _NO_SUPPORT

    footprint = region.footprint_point_count
    visible_share = region.visible_share
    occluded_fraction = outside_fraction = None
    if footprint:
        occluded_fraction = region.occluded_count / footprint
        outside_fraction = region.outside_valid_support_count / footprint
    else:
        for component in (
            QualityComponent.VISIBLE_SHARE,
            QualityComponent.OCCLUDED_FRACTION,
            QualityComponent.OUTSIDE_VALID_SUPPORT_FRACTION,
        ):
            unavailable[component] = _NO_FOOTPRINT
    if reprojection is None:
        unavailable[QualityComponent.REPROJECTION] = _NO_REFERENCE

    return ObservationQuality(
        definitions_version=QUALITY_DEFINITIONS_VERSION,
        spatial_observation_id=observation.spatial_observation_id,
        source_observation_id=observation.source_observation_id,
        geometric_map_id=observation.provenance.geometric_map_id,
        calibration_ref=observation.calibration_ref,
        pose_ref=observation.pose_ref,
        image_transform_id=observation.projection_summary.image_transform_id,
        visibility_policy_id=observation.provenance.visibility_policy_id,
        depth_metric=resolution.depth_metric,
        associated_count=region.associated_count,
        footprint_count=footprint,
        mask_area_px=region.mask_area_px,
        support_density_per_mask_pixel=region.associated_count / region.mask_area_px,
        support_pixel_coverage=region.support_pixel_coverage,
        support_depth_m=depth,
        support_off_axis_angle_rad=angle,
        border_distance_px=border,
        visible_share=visible_share,
        occluded_fraction=occluded_fraction,
        outside_valid_support_fraction=outside_fraction,
        reprojection=reprojection,
        unavailable=unavailable,
    )


def _summary(values: NDArray[Any]) -> ValueSummary:
    import numpy as np

    return ValueSummary(
        count=int(values.shape[0]),
        minimum=float(values.min()),
        median=float(np.median(values)),
        maximum=float(values.max()),
    )
