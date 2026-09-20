"""Canonical contracts for Sensor Association.

Sensor Association is the bridge between 2D visual evidence and persistent 3D
geometry. It answers one question per region of one image: *which geometry of
the map does this region see?* The answer is a :class:`SpatialObservation`. It is
**evidence**, not belief and not knowledge: it never fixes a label, an identity
or an entity, it does not pick a winner between overlapping regions, and it does
not copy XYZ, embeddings or claims. Everything it points to is reached by
reference.

Two ideas are kept apart on purpose:

* geometry support -- the :class:`GeometryReference` of every map element that is
  visible and falls inside the region;
* visibility diagnostics -- why the other candidate elements were **not** taken
  (behind the camera, outside the image, occluded, ...), so that a missing point
  is explainable instead of silently absent.

Ranges are distances along the viewing ray from the camera optical center, in
meters. They are used instead of ``z`` depth because a camera model with a field
of view above 180 degrees can see points with ``z <= 0``.

No ROS, NumPy, Torch or point-cloud type appears here: a persisted observation is
readable with the standard library alone. See
``src/contextmap/sensor_association/docs/contracts.md`` for the field reference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import NewType

from contextmap.geometric_mapping import GeometryReference, MapId
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.state_estimation import (
    LookupOutcome,
    PoseEstimateId,
    StateEstimationRunId,
    TrajectoryId,
)
from contextmap.visual_perception import (
    ClaimId,
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)

SpatialObservationId = NewType("SpatialObservationId", str)
"""Identity of one region's spatial observation."""

PixelCoordinate = tuple[float, float]
"""``(u, v)`` in pixels, continuous, with the pixel *center* at the integer."""


def spatial_observation_id_for(
    *, perception_result_id: PerceptionResultId, region_id: RegionId
) -> SpatialObservationId:
    """Compute the deterministic identity of a region's spatial observation.

    Args:
        perception_result_id: The per-frame perception result that owns the region.
        region_id: The region within that result.

    Returns:
        A pure function of the inputs, so the identity survives an artifact
        round-trip without a registry.
    """
    return SpatialObservationId(f"spatial--{perception_result_id}--{region_id}")


class VisibilityState(Enum):
    """Why a candidate geometry element is, or is not, evidence for a region.

    Attributes:
        BEHIND_CAMERA: The camera model cannot project the point.
        OUTSIDE_IMAGE: It projects outside the prepared image.
        OUTSIDE_VALID_SUPPORT: It lands inside the prepared image but outside the
            valid support or inside an exclusion region.
        OCCLUDED: A nearer surface hides it along the same ray.
        VISIBLE_UNASSIGNED: It is visible but falls in no region's mask.
        ASSOCIATED: It is visible and inside the region's mask.
    """

    BEHIND_CAMERA = "behind_camera"
    OUTSIDE_IMAGE = "outside_image"
    OUTSIDE_VALID_SUPPORT = "outside_valid_support"
    OCCLUDED = "occluded"
    VISIBLE_UNASSIGNED = "visible_unassigned"
    ASSOCIATED = "associated"


_PIXEL_REQUIRED = frozenset(
    {
        VisibilityState.OUTSIDE_VALID_SUPPORT,
        VisibilityState.OCCLUDED,
        VisibilityState.VISIBLE_UNASSIGNED,
        VisibilityState.ASSOCIATED,
    }
)


def _require_finite_pair(name: str, pixel: PixelCoordinate | None) -> None:
    if pixel is not None and not all(math.isfinite(value) for value in pixel):
        raise ValueError(f"{name} must be finite, got {pixel!r}")


def _require_range(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")


@dataclass(frozen=True, kw_only=True)
class PointCorrespondence:
    """The 2D outcome of one map element for one camera frame.

    This is the lower-level record behind the region-level summary: it says where
    one geometry element landed and what became of it. It exists for diagnostics
    and reprojection checks; downstream stages consume
    :class:`SpatialObservation`.

    Attributes:
        geometry: The map element.
        visibility: What became of it.
        camera_range_m: Distance from the camera optical center, in meters.
        raw_pixel: Pixel in the raw image, when the model produced one.
        prepared_pixel: Pixel in the prepared image, when the point reached it.
        support_range_m: Range of the nearest surface seen at that pixel, in
            meters; the occluder's range for an ``OCCLUDED`` point.
    """

    geometry: GeometryReference
    visibility: VisibilityState
    camera_range_m: float
    raw_pixel: PixelCoordinate | None
    prepared_pixel: PixelCoordinate | None
    support_range_m: float | None

    def __post_init__(self) -> None:
        """Validate the record against its visibility state.

        Raises:
            ValueError: If a value is not finite, a range is negative, a pixel is
                missing for a point that reached the prepared image, a pixel or
                support is present for a point the model cannot project, or an
                occluded point is not hidden by a strictly nearer surface.
        """
        _require_range("camera_range_m", self.camera_range_m)
        if self.support_range_m is not None:
            _require_range("support_range_m", self.support_range_m)
        _require_finite_pair("raw_pixel", self.raw_pixel)
        _require_finite_pair("prepared_pixel", self.prepared_pixel)
        state = self.visibility
        if state in _PIXEL_REQUIRED and self.prepared_pixel is None:
            raise ValueError(f"a point in state {state.value} must carry its prepared_pixel")
        if state is VisibilityState.BEHIND_CAMERA and (
            self.raw_pixel is not None
            or self.prepared_pixel is not None
            or self.support_range_m is not None
        ):
            raise ValueError(
                "a behind_camera point has no raw_pixel, prepared_pixel or support_range_m"
            )
        if state is VisibilityState.OUTSIDE_IMAGE and self.support_range_m is not None:
            raise ValueError("an outside_image point has no support_range_m")
        if state is VisibilityState.OCCLUDED and (
            self.support_range_m is None or self.support_range_m >= self.camera_range_m
        ):
            raise ValueError(
                "an occluded point needs a support_range_m nearer than its camera_range_m, "
                f"got {self.support_range_m!r} and {self.camera_range_m!r}"
            )


@dataclass(frozen=True, kw_only=True)
class VisibilityDiagnostics:
    """How many candidate elements ended in each visibility state.

    Only positive counts are kept, in :class:`VisibilityState` declaration order,
    so two equal diagnostics always encode to the same record.

    Attributes:
        counts: Number of elements per state.
    """

    counts: Mapping[VisibilityState, int]

    def __post_init__(self) -> None:
        """Validate and canonicalize the counts.

        Raises:
            ValueError: If a count is negative.
        """
        for state, count in self.counts.items():
            if count < 0:
                raise ValueError(f"visibility count for {state.value} must not be negative")
        canonical = {
            state: self.counts[state] for state in VisibilityState if self.counts.get(state, 0) > 0
        }
        object.__setattr__(self, "counts", canonical)

    @property
    def total(self) -> int:
        """Number of elements accounted for across every state."""
        return sum(self.counts.values())

    def count(self, state: VisibilityState) -> int:
        """Return the number of elements in ``state``."""
        return self.counts.get(state, 0)


@dataclass(frozen=True, kw_only=True)
class ProjectionSummary:
    """Frame-level facts about the projection a region's evidence came from.

    Attributes:
        camera_model_kind: Camera model used, e.g. ``"pinhole"`` or ``"mei"``.
        image_transform_id: Identity of the raw-to-prepared image transform the
            pixels are expressed after.
        prepared_image_size: ``(width, height)`` of the prepared image, in pixels.
        considered_count: Geometry elements evaluated for this frame.
        visible_count: Of those, elements that were in front of the camera,
            inside the valid prepared image and not occluded.
    """

    camera_model_kind: str
    image_transform_id: str
    prepared_image_size: tuple[int, int]
    considered_count: int
    visible_count: int

    def __post_init__(self) -> None:
        """Validate the summary.

        Raises:
            ValueError: If an identity is empty, the image size is not positive,
                a count is negative, or ``visible_count`` exceeds
                ``considered_count``.
        """
        if not self.camera_model_kind:
            raise ValueError("camera_model_kind must not be empty")
        if not self.image_transform_id:
            raise ValueError("image_transform_id must not be empty")
        if any(side < 1 for side in self.prepared_image_size):
            raise ValueError(
                f"prepared_image_size must be positive, got {self.prepared_image_size!r}"
            )
        if self.considered_count < 0 or self.visible_count < 0:
            raise ValueError("projection counts must not be negative")
        if self.visible_count > self.considered_count:
            raise ValueError(
                f"visible_count {self.visible_count} must not exceed considered_count "
                f"{self.considered_count}"
            )


@dataclass(frozen=True, kw_only=True)
class VisualFeatureRef:
    """Reference to a visual feature of the same perception result.

    Attributes:
        feature_id: The feature, owned by Visual Perception.
        embedding_space_id: The embedding space its vector lives in, so features
            of different spaces are never mixed by accident.
        scope: Whether the feature is dense, global or per-region.
        region_id: The region a ``REGION``-scoped feature belongs to; ``None`` for
            ``DENSE`` and ``GLOBAL``.
    """

    feature_id: FeatureId
    embedding_space_id: str
    scope: FeatureScope
    region_id: RegionId | None

    def __post_init__(self) -> None:
        """Validate the reference.

        Raises:
            ValueError: If an identity is empty or ``region_id`` does not match
                ``scope``.
        """
        if not self.feature_id:
            raise ValueError("feature_id must not be empty")
        if not self.embedding_space_id:
            raise ValueError("embedding_space_id must not be empty")
        if (self.scope is FeatureScope.REGION) != (self.region_id is not None):
            raise ValueError(
                f"region_id is required for, and only for, region-scoped features; got "
                f"scope {self.scope.value} and region_id {self.region_id!r}"
            )


@dataclass(frozen=True, kw_only=True)
class SemanticClaimRef:
    """Reference to a semantic claim of the same perception result.

    Attributes:
        claim_id: The claim, owned by Visual Perception.
    """

    claim_id: ClaimId

    def __post_init__(self) -> None:
        """Require the claim identity.

        Raises:
            ValueError: If ``claim_id`` is empty.
        """
        if not self.claim_id:
            raise ValueError("claim_id must not be empty")


@dataclass(frozen=True, kw_only=True)
class CalibrationRef:
    """Which calibration and camera turned geometry into pixels.

    Attributes:
        calibration_identity: Hash of the calibration set used.
        camera_calibration_id: The camera entry within that set.
        camera_model_kind: Camera model of that entry.
        camera_frame: The camera optical frame the projection was done in.
    """

    calibration_identity: str
    camera_calibration_id: CalibrationReferenceId
    camera_model_kind: str
    camera_frame: FrameId

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        for name in (
            "calibration_identity",
            "camera_calibration_id",
            "camera_model_kind",
            "camera_frame",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, kw_only=True)
class PoseRef:
    """Which pose placed the camera in the map frame for this observation.

    Mirrors the evidence of a state-estimation lookup without embedding the pose.

    Attributes:
        trajectory_id: The trajectory the pose was read from.
        state_estimation_run_id: The persisted run it came from, when there is one.
        source_estimate_ids: The estimated pose, or the two neighbors of an
            interpolation.
        lookup_outcome: How the pose was resolved.
        time_delta_ns: Distance in nanoseconds from the frame timestamp to the
            nearest contributing pose.
        interpolation_fraction: Position in ``[0, 1]`` between the two neighbors
            for an interpolated pose; ``None`` otherwise.
    """

    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    source_estimate_ids: tuple[PoseEstimateId, ...]
    lookup_outcome: LookupOutcome
    time_delta_ns: int
    interpolation_fraction: float | None

    def __post_init__(self) -> None:
        """Validate that the reference is coherent with how the pose was resolved.

        Raises:
            ValueError: If the trajectory or estimates are missing, the time delta
                is negative, or the estimates and fraction do not match the
                lookup outcome.
        """
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        if not self.source_estimate_ids:
            raise ValueError("source_estimate_ids must name at least one pose estimate")
        if self.time_delta_ns < 0:
            raise ValueError(f"time_delta_ns must not be negative, got {self.time_delta_ns}")
        if self.lookup_outcome is LookupOutcome.INTERPOLATED:
            if len(self.source_estimate_ids) != 2:
                raise ValueError("an interpolated pose reference needs exactly two estimates")
            fraction = self.interpolation_fraction
            if fraction is None or not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"an interpolated pose reference needs an interpolation_fraction in "
                    f"[0, 1], got {fraction!r}"
                )
        else:
            if len(self.source_estimate_ids) != 1:
                raise ValueError("a direct pose reference names exactly one source estimate")
            if self.interpolation_fraction is not None:
                raise ValueError("a direct pose reference has no interpolation_fraction")


@dataclass(frozen=True, kw_only=True)
class AssociationProvenance:
    """Run-level traceability of a spatial observation.

    Attributes:
        geometric_map_id: The immutable map the geometry references belong to.
        perception_run_id: The perception run the region and evidence came from.
        sequence_artifact_id: The canonical sequence both were built from.
        visibility_policy_id: Versioned rule that decides visibility and occlusion.
        membership_policy_id: Versioned rule that decides mask membership.
        configuration_fingerprint: Hash of the association configuration, when
            there is something configurable.
        code_version: Code revision that produced the observation, when known.
    """

    geometric_map_id: MapId
    perception_run_id: PerceptionRunId
    sequence_artifact_id: SequenceArtifactId
    visibility_policy_id: str
    membership_policy_id: str
    configuration_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require every identity and policy.

        Raises:
            ValueError: If an identity or policy is empty.
        """
        for name in (
            "geometric_map_id",
            "perception_run_id",
            "sequence_artifact_id",
            "visibility_policy_id",
            "membership_policy_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, kw_only=True)
class SpatialObservation:
    """The map geometry one region of one image sees.

    A spatial observation is evidence produced in one execution. Regions may
    overlap, so the same geometry can support several observations; nothing here
    decides which region owns it. It carries references to the geometry, the
    visual features and the claims, never copies of them.

    Attributes:
        spatial_observation_id: Identity of this observation.
        source_observation_id: The camera frame the region was observed in.
        perception_result_id: The perception result that owns the region.
        region_id: The observed region.
        geometry_support: References to the visible geometry inside the region,
            sorted by ``geometry_id`` and unique, all from one map.
        projection_summary: Frame-level projection facts.
        visual_feature_refs: Features of the same result that describe the region.
        semantic_claim_refs: Claims of the same result about the region.
        calibration_ref: The calibration and camera used.
        pose_ref: The pose used.
        visibility: Why candidate elements were or were not taken.
        provenance: Run-level traceability.
    """

    spatial_observation_id: SpatialObservationId
    source_observation_id: SourceObservationId
    perception_result_id: PerceptionResultId
    region_id: RegionId
    geometry_support: tuple[GeometryReference, ...]
    projection_summary: ProjectionSummary
    visual_feature_refs: tuple[VisualFeatureRef, ...]
    semantic_claim_refs: tuple[SemanticClaimRef, ...]
    calibration_ref: CalibrationRef
    pose_ref: PoseRef
    visibility: VisibilityDiagnostics
    provenance: AssociationProvenance

    def __post_init__(self) -> None:
        """Validate identities, support and cross-field consistency.

        Raises:
            ValueError: If an identity is empty; the support spans more than one
                map, is not the provenance map, or is not sorted and unique; the
                associated count differs from the support size; the diagnostics
                exceed what the projection considered or kept visible; or a
                region-scoped feature belongs to another region.
        """
        for name in (
            "spatial_observation_id",
            "source_observation_id",
            "perception_result_id",
            "region_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        maps = {reference.map_id for reference in self.geometry_support}
        if maps - {self.provenance.geometric_map_id}:
            raise ValueError(
                f"geometry_support must reference only the provenance map "
                f"{self.provenance.geometric_map_id!r}, got maps {sorted(maps)!r}"
            )
        identities = [reference.geometry_id for reference in self.geometry_support]
        if any(left >= right for left, right in pairwise(identities)):
            raise ValueError("geometry_support must be sorted by geometry_id and unique")
        associated = self.visibility.count(VisibilityState.ASSOCIATED)
        if associated != len(self.geometry_support):
            raise ValueError(
                f"the associated count {associated} must equal the geometry_support size "
                f"{len(self.geometry_support)}"
            )
        summary = self.projection_summary
        if self.visibility.total > summary.considered_count:
            raise ValueError(
                f"visibility counts {self.visibility.total} exceed the "
                f"considered_count {summary.considered_count}"
            )
        if len(self.geometry_support) > summary.visible_count:
            raise ValueError(
                f"geometry_support size {len(self.geometry_support)} exceeds the "
                f"visible_count {summary.visible_count}"
            )
        for feature in self.visual_feature_refs:
            if feature.scope is FeatureScope.REGION and feature.region_id != self.region_id:
                raise ValueError(
                    f"region-scoped feature {feature.feature_id!r} belongs to region "
                    f"{feature.region_id!r}, not the observed region {self.region_id!r}"
                )
