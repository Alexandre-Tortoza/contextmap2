"""Mask membership: which frozen ``Region2D`` masks support each visible geometry.

Once geometry is projected into the prepared image and filtered for visibility, this
step asks, for every visible point, which region masks contain its prepared pixel. The
answer is zero, one or several regions and is kept as such: **overlapping regions are
valid and no semantic winner is chosen**. A region is evidence about the image; the
geometry it covers is a set of references, never a copy of a label or a payload, and
nothing here assigns a final class, fuses observations or reruns region discovery.

Regions are read as frozen: masks are only read, region identities are preserved, and a
mask that is not expressed in the prepared image the geometry was projected into is
rejected rather than resampled, so raw and prepared coordinates cannot be confused.

Two indexes are kept in compact form: region to geometry (the associated positions of
each region) and geometry to region (sorted ``(point, region)`` pairs). Coverage and
support statistics are defined in :data:`COVERAGE_DEFINITIONS_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.ingestion import SequenceArtifactId, SourceObservationId
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.models import (
    AssociationProvenance,
    ProjectionSummary,
    SemanticClaimRef,
    SpatialObservation,
    VisibilityDiagnostics,
    VisibilityState,
    VisualFeatureRef,
    spatial_observation_id_for,
)
from contextmap.sensor_association.visibility import VisibilityResolution
from contextmap.visual_perception import FeatureScope, InlineMask, PerceptionResult, RegionId

if TYPE_CHECKING:
    from numpy.typing import NDArray

MEMBERSHIP_POLICY_ID = "mask-membership-v1"
"""Versioned identity of the membership rule: a visible point belongs to every accepted
region whose inline mask is foreground at the pixel whose center is nearest to the
point's prepared-image pixel."""

COVERAGE_DEFINITIONS_VERSION = "region-support-metrics-v1"
"""Versioned identity of the coverage and support definitions documented on
:class:`RegionMembership` and :class:`MembershipStatistics`."""


class SkipReason(Enum):
    """Why a region of a perception result was not evaluated.

    Attributes:
        REJECTED: The candidate was rejected by normalization, so it is not part of the
            frozen canonical region set.
        NO_INLINE_MASK: The region has no mask at all: a box-only region, or a region
            whose mask was persisted by reference (#378) without a ``mask_loader``
            supplied to :func:`associate_regions` to resolve it.
        EMPTY_MASK: The accepted region's mask has no foreground pixel, so it has no
            footprint to support geometry and its per-pixel statistics are undefined.
    """

    REJECTED = "rejected"
    NO_INLINE_MASK = "no_inline_mask"
    EMPTY_MASK = "empty_mask"


class RegionMaskLoader(Protocol):
    """Adapter boundary for resolving a frozen region's mask kept by reference.

    A region's mask is no longer always inlined in its
    :class:`~contextmap.visual_perception.models.Region2D.mask` field
    once a run is persisted and reopened (#378): it may instead be a
    ``mask_reference`` pointing into a lazily-loaded store. Mask
    membership only needs *some* way to resolve that reference back into
    pixels for the owning observation — never a particular storage
    backend — so this is a minimal, sensor-association-local Protocol.
    :meth:`~contextmap.visual_perception.mask_store.MaskStoreReader.load`
    already satisfies this shape and is the composition root's usual
    choice.
    """

    def load(self, source_observation_id: SourceObservationId, region_id: RegionId) -> InlineMask:
        """Load and hash-verify one region's full-image mask."""
        ...


@dataclass(frozen=True, kw_only=True)
class SkippedRegion:
    """A region that was not evaluated, kept so its absence is explainable.

    Attributes:
        region_id: The region.
        reason: Why it was skipped.
    """

    region_id: RegionId
    reason: SkipReason


@dataclass(frozen=True, kw_only=True, eq=False)
class RegionMembership:
    """The geometry one frozen region's mask supports, with its support statistics.

    Definitions (version :data:`COVERAGE_DEFINITIONS_VERSION`):

    * *footprint*: the points whose prepared pixel falls inside the mask, whatever their
      visibility state (behind-camera and outside-image points have no pixel and are never
      in it);
    * ``associated_count``: footprint points that are visible;
    * ``occluded_count``, ``outside_valid_support_count``: footprint points that are hidden
      or unsupported;
    * ``covered_pixel_count``: distinct mask pixels holding at least one associated point;
    * ``support_pixel_coverage``: ``covered_pixel_count / mask_area_px``, the share of the
      region's pixels that have supported 3D geometry (low on a sparse map);
    * ``visible_share``: ``associated_count / footprint_point_count``, the share of the
      footprint that was visible; ``None`` for an empty footprint.

    Attributes:
        region_id: The region, as frozen by Visual Perception.
        mask_area_px: Number of foreground pixels of the mask.
        associated_indices: Sorted positions, in the frame, of the associated geometry.
        occluded_count: Footprint points hidden by a nearer surface.
        outside_valid_support_count: Footprint points outside the valid support.
        covered_pixel_count: Distinct mask pixels holding an associated point.
    """

    region_id: RegionId
    mask_area_px: int
    associated_indices: NDArray[Any]
    occluded_count: int
    outside_valid_support_count: int
    covered_pixel_count: int

    @property
    def associated_count(self) -> int:
        """Number of visible geometry elements inside the mask."""
        return int(self.associated_indices.shape[0])

    @property
    def footprint_point_count(self) -> int:
        """Number of points, of any state, whose prepared pixel is inside the mask."""
        return self.associated_count + self.occluded_count + self.outside_valid_support_count

    @property
    def support_pixel_coverage(self) -> float:
        """Share of mask pixels that hold at least one associated point."""
        return self.covered_pixel_count / self.mask_area_px if self.mask_area_px else 0.0

    @property
    def visible_share(self) -> float | None:
        """Share of the footprint that was visible; ``None`` when the footprint is empty."""
        footprint = self.footprint_point_count
        return None if footprint == 0 else self.associated_count / footprint


@dataclass(frozen=True, kw_only=True)
class MembershipStatistics:
    """Auditable counts for one frame (version :data:`COVERAGE_DEFINITIONS_VERSION`).

    The five state counts partition ``point_count``; ``visible_count`` splits into
    ``associated_count`` and ``visible_unassigned_count``.

    Attributes:
        definitions_version: The version of these definitions.
        point_count: Map points evaluated for the frame.
        behind_camera_count: Points the camera model cannot project.
        outside_image_count: Points that project outside the prepared image.
        outside_valid_support_count: Points outside the valid region or inside an exclusion.
        occluded_count: Points hidden by a nearer surface.
        visible_count: Points that are supported and not occluded.
        associated_count: Visible points inside at least one evaluated mask.
        visible_unassigned_count: Visible points inside no evaluated mask.
        overlap_point_count: Visible points inside two or more masks.
        membership_pair_count: ``(point, region)`` memberships.
        points_by_region_count: For each number of regions ``k``, how many visible points
            belong to exactly ``k`` regions; ``0`` counts the unassigned ones.
    """

    definitions_version: str
    point_count: int
    behind_camera_count: int
    outside_image_count: int
    outside_valid_support_count: int
    occluded_count: int
    visible_count: int
    associated_count: int
    visible_unassigned_count: int
    overlap_point_count: int
    membership_pair_count: int
    points_by_region_count: dict[int, int]


@dataclass(frozen=True, kw_only=True, eq=False)
class FrameMembership:
    """Which evaluated regions support each visible point of one frame.

    Attributes:
        resolution: The visibility the membership was computed on.
        perception_result: The frozen perception result the regions came from.
        regions: Evaluated regions, in the order of the perception result.
        skipped: Regions that were not evaluated, with the reason.
        point_indices: ``(M,)`` frame position of each membership, sorted by point then
            region.
        point_region_slots: ``(M,)`` index into ``regions`` of each membership.
        visible_unassigned: ``(N,)`` visible points that are inside no evaluated mask.
    """

    resolution: VisibilityResolution
    perception_result: PerceptionResult
    regions: tuple[RegionMembership, ...]
    skipped: tuple[SkippedRegion, ...]
    point_indices: NDArray[Any]
    point_region_slots: NDArray[Any]
    visible_unassigned: NDArray[Any]

    @property
    def frame(self) -> FrameProjection:
        """The projection the memberships refer to."""
        return self.resolution.frame

    def points_of(self, region_id: RegionId) -> NDArray[Any]:
        """Return the sorted frame positions of the geometry a region supports.

        Raises:
            KeyError: If the region was not evaluated.
        """
        for region in self.regions:
            if region.region_id == region_id:
                return region.associated_indices
        raise KeyError(region_id)

    def regions_of(self, point_index: int) -> tuple[RegionId, ...]:
        """Return every evaluated region that contains a point, in result order.

        An empty tuple means the point is inside no mask; overlaps keep all regions.
        """
        import numpy as np

        low = int(np.searchsorted(self.point_indices, point_index, side="left"))
        high = int(np.searchsorted(self.point_indices, point_index, side="right"))
        return tuple(
            self.regions[int(slot)].region_id for slot in self.point_region_slots[low:high]
        )

    def statistics(self) -> MembershipStatistics:
        """Summarize the frame: every state, the association split and the overlaps."""
        import numpy as np

        counts = self.resolution.state_counts()
        visible = self.resolution.visible_count
        unassigned = int(self.visible_unassigned.sum())
        per_point = np.bincount(self.point_indices, minlength=len(self.visible_unassigned))
        overlap = int((per_point >= 2).sum())
        histogram: dict[int, int] = {0: unassigned}
        for regions_per_point in per_point[per_point > 0].tolist():
            histogram[regions_per_point] = histogram.get(regions_per_point, 0) + 1
        return MembershipStatistics(
            definitions_version=COVERAGE_DEFINITIONS_VERSION,
            point_count=len(self.visible_unassigned),
            behind_camera_count=counts[VisibilityState.BEHIND_CAMERA],
            outside_image_count=counts[VisibilityState.OUTSIDE_IMAGE],
            outside_valid_support_count=counts[VisibilityState.OUTSIDE_VALID_SUPPORT],
            occluded_count=counts[VisibilityState.OCCLUDED],
            visible_count=visible,
            associated_count=visible - unassigned,
            visible_unassigned_count=unassigned,
            overlap_point_count=overlap,
            membership_pair_count=int(self.point_indices.shape[0]),
            points_by_region_count={k: histogram[k] for k in sorted(histogram)},
        )


def associate_regions(
    resolution: VisibilityResolution,
    perception_result: PerceptionResult,
    *,
    mask_loader: RegionMaskLoader | None = None,
) -> FrameMembership:
    """Find which frozen region masks contain each visible projected point.

    Args:
        resolution: The visibility of the frame's points.
        perception_result: The perception result of the same observation.
        mask_loader: Resolves a region's mask when it was persisted by
            reference instead of being inlined on ``Region2D.mask``
            (#378), for example a reopened run's
            ``PerceptionRunReader.mask_store()``. ``None`` (the default)
            keeps the previous behavior: a region without an inline mask
            is skipped with :attr:`SkipReason.NO_INLINE_MASK`.

    An accepted region whose mask has no foreground pixel is skipped with
    :attr:`SkipReason.EMPTY_MASK`: every :class:`RegionMembership` has ``mask_area_px >= 1``,
    which the per-pixel statistics and the observation quality require.

    Returns:
        The region-to-geometry and geometry-to-region indexes, the support statistics and
        the regions that were skipped.

    Raises:
        AssociationInputError: If the perception result belongs to another observation, or
            an evaluated region's mask is not in the prepared image's coordinate space.
    """
    import numpy as np

    frame = resolution.frame
    if perception_result.source_observation_id != frame.source_observation_id:
        raise AssociationInputError(
            f"the perception result {perception_result.result_id!r} is of observation "
            f"{perception_result.source_observation_id!r}, not of {frame.source_observation_id!r}"
        )
    width, height = frame.image_transform.prepared_size
    in_image = np.flatnonzero(frame.in_prepared_image)
    # O pixel de índice i cobre [i - 0.5, i + 0.5): o índice do centro c é floor(c + 0.5).
    columns = np.floor(frame.prepared_pixels[in_image, 0] + 0.5).astype(np.int64)
    rows = np.floor(frame.prepared_pixels[in_image, 1] + 0.5).astype(np.int64)
    visible = resolution.visible[in_image]
    occluded = resolution.occluded[in_image]
    unsupported = resolution.outside_valid_support[in_image]

    regions: list[RegionMembership] = []
    skipped: list[SkippedRegion] = []
    pairs_points: list[NDArray[Any]] = []
    pairs_slots: list[NDArray[Any]] = []
    for region in perception_result.regions:
        if not region.is_accepted:
            skipped.append(SkippedRegion(region_id=region.region_id, reason=SkipReason.REJECTED))
            continue
        region_mask = region.mask
        if region_mask is None and region.mask_reference is not None and mask_loader is not None:
            region_mask = mask_loader.load(
                perception_result.source_observation_id, region.region_id
            )
        if region_mask is None:
            skipped.append(
                SkippedRegion(region_id=region.region_id, reason=SkipReason.NO_INLINE_MASK)
            )
            continue
        if (region_mask.width, region_mask.height) != (width, height):
            raise AssociationInputError(
                f"the mask of region {region.region_id!r} is {region_mask.width}x"
                f"{region_mask.height} but the prepared image is {width}x{height}: regions must be "
                f"expressed in the prepared image the geometry was projected into"
            )
        mask = region_mask.as_array()
        mask_area_px = region_mask.area
        if mask_area_px == 0:
            skipped.append(SkippedRegion(region_id=region.region_id, reason=SkipReason.EMPTY_MASK))
            continue
        inside = mask[rows, columns]
        associated = inside & visible
        associated_indices = in_image[associated]
        covered = np.unique(rows[associated] * width + columns[associated]).shape[0]
        slot = len(regions)
        regions.append(
            RegionMembership(
                region_id=region.region_id,
                mask_area_px=mask_area_px,
                associated_indices=associated_indices,
                occluded_count=int((inside & occluded).sum()),
                outside_valid_support_count=int((inside & unsupported).sum()),
                covered_pixel_count=int(covered),
            )
        )
        pairs_points.append(associated_indices)
        pairs_slots.append(np.full(associated_indices.shape, slot, dtype=np.int64))

    if pairs_points:
        points = np.concatenate(pairs_points)
        slots = np.concatenate(pairs_slots)
    else:
        points = np.empty(0, dtype=np.int64)
        slots = np.empty(0, dtype=np.int64)
    order = np.lexsort((slots, points))
    points, slots = points[order], slots[order]
    assigned = np.zeros(len(resolution.visible), dtype=bool)
    assigned[points] = True
    return FrameMembership(
        resolution=resolution,
        perception_result=perception_result,
        regions=tuple(regions),
        skipped=tuple(skipped),
        point_indices=points,
        point_region_slots=slots,
        visible_unassigned=resolution.visible & ~assigned,
    )


def build_spatial_observations(
    membership: FrameMembership,
    *,
    configuration_fingerprint: str | None,
    code_version: str | None,
) -> tuple[SpatialObservation, ...]:
    """Turn the evaluated regions into canonical spatial observations.

    Each observation carries references to the geometry, the region's features and its
    claims, plus the calibration, pose and policies used; nothing is copied into it.

    Features a region refers to are its own region-scoped features and the dense feature
    maps of the same result (which can be sampled over the region); global features and
    scene-level claims describe the whole image, not the region, and are left out.

    Args:
        membership: The membership of one frame.
        configuration_fingerprint: Hash of the association configuration, when known.
        code_version: Code revision that produced the observations, when known.

    Returns:
        One observation per evaluated region, in the perception result's order.
    """
    resolution = membership.resolution
    frame = membership.frame
    result = membership.perception_result
    summary = ProjectionSummary(
        camera_model_kind=frame.camera.camera_model_kind,
        depth_metric=resolution.depth_metric,
        image_transform_id=frame.image_transform.transform_id,
        prepared_image_size=frame.image_transform.prepared_size,
        considered_count=len(resolution.visible),
        visible_count=resolution.visible_count,
    )
    provenance = AssociationProvenance(
        geometric_map_id=frame.map_id,
        perception_run_id=result.run_id,
        sequence_artifact_id=SequenceArtifactId(result.sequence_artifact_id),
        visibility_policy_id=resolution.policy.policy_id,
        membership_policy_id=MEMBERSHIP_POLICY_ID,
        configuration_fingerprint=configuration_fingerprint,
        code_version=code_version,
    )
    observations: list[SpatialObservation] = []
    for region in membership.regions:
        counts = {
            VisibilityState.ASSOCIATED: region.associated_count,
            VisibilityState.OCCLUDED: region.occluded_count,
            VisibilityState.OUTSIDE_VALID_SUPPORT: region.outside_valid_support_count,
        }
        observations.append(
            SpatialObservation(
                spatial_observation_id=spatial_observation_id_for(
                    perception_result_id=result.result_id, region_id=region.region_id
                ),
                source_observation_id=frame.source_observation_id,
                perception_result_id=result.result_id,
                region_id=region.region_id,
                geometry_support=tuple(
                    frame.map_reference(int(index)) for index in region.associated_indices
                ),
                projection_summary=summary,
                visual_feature_refs=_feature_refs(result, region.region_id),
                semantic_claim_refs=tuple(
                    SemanticClaimRef(claim_id=claim.claim_id)
                    for claim in result.claims
                    if claim.region_id == region.region_id
                ),
                calibration_ref=frame.calibration_ref,
                pose_ref=frame.pose_ref,
                visibility=VisibilityDiagnostics(counts=counts),
                provenance=provenance,
            )
        )
    return tuple(observations)


def _feature_refs(result: PerceptionResult, region_id: RegionId) -> tuple[VisualFeatureRef, ...]:
    return tuple(
        VisualFeatureRef(
            feature_id=feature.feature_id,
            embedding_space_id=feature.embedding_space_id,
            scope=feature.scope,
            region_id=feature.region_id,
        )
        for feature in result.features
        if (feature.scope is FeatureScope.REGION and feature.region_id == region_id)
        or feature.scope is FeatureScope.DENSE
    )
