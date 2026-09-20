"""The deterministic path from map geometry to a prepared-image pixel.

For one RGB observation the projector runs the whole chain::

    P_map -> P_camera at the RGB timestamp -> raw camera pixel -> prepared-image pixel

The 3D half resolves ``T_map_body(t_rgb)`` from the selected trajectory, the static
``T_body_camera`` from the canonical calibration, composes
``T_map_camera = T_map_body * T_body_camera`` and moves the authoritative map
coordinates into the camera optical frame before the calibrated camera model
projects them. The 2D half maps the raw pixel through the exact preparation steps
recorded by Visual Perception and checks the valid region and exclusion masks in the
prepared space.

This stage only *places* points. It does not decide occlusion, does not pick a region
and does not attach any label: a point that lands in the supported prepared image is
a candidate for the visibility and membership steps that follow.

A frame whose pose the lookup policy rejects is returned as data
(:class:`RejectedProjection`) so skipped frames can be counted and audited. Everything
else that does not fit together (frames, lineage, calibration, image chain) raises
:class:`~contextmap.sensor_association.errors.AssociationInputError`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationReferenceId,
    CalibrationSet,
    CameraModel,
    FrameId,
    ImageObservation,
    SourceObservationId,
)
from contextmap.sensor_association.camera_models import (
    CameraIdentity,
    CameraProjection,
    camera_projection_for,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.geometry_cloud import GeometryCloud
from contextmap.sensor_association.image_transform import (
    RawToPreparedTransform,
    raw_to_prepared_transform,
)
from contextmap.sensor_association.models import CalibrationRef, PixelCoordinate, PoseRef
from contextmap.shared import SourceTimestamp, compose_rigid, quaternion_to_rotation_matrix
from contextmap.state_estimation import (
    FrameGraphError,
    LookupPolicy,
    LookupRejection,
    RejectedLookup,
    StateEstimationRunId,
    StaticFrameGraph,
    TimeBounds,
    TrajectoryLookup,
    calibration_identity,
)
from contextmap.visual_perception import InlineMask, PreparedImage

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ProjectionStage(Enum):
    """How far a point got along the projection chain.

    Attributes:
        BEHIND_CAMERA: The camera model cannot project the point.
        OUTSIDE_IMAGE: It projects, but outside the prepared image.
        OUTSIDE_VALID_SUPPORT: It lands inside the prepared image but outside the valid
            region or inside an exclusion region.
        IN_SUPPORT: It lands on supported prepared-image pixels; whether it is visible
            or occluded is decided later.
    """

    BEHIND_CAMERA = "behind_camera"
    OUTSIDE_IMAGE = "outside_image"
    OUTSIDE_VALID_SUPPORT = "outside_valid_support"
    IN_SUPPORT = "in_support"


@dataclass(frozen=True, kw_only=True)
class ExtrinsicRef:
    """The static extrinsic that placed the camera relative to the body.

    Attributes:
        calibration_identity: Hash of the calibration set the transform came from.
        parent_frame: The body frame.
        child_frame: The camera optical frame.
    """

    calibration_identity: str
    parent_frame: FrameId
    child_frame: FrameId


@dataclass(frozen=True, kw_only=True)
class RejectedProjection:
    """A frame that could not be projected because no pose was accepted for it.

    Attributes:
        source_observation_id: The camera frame that was skipped.
        rejection: Why the pose lookup was rejected.
        detail: The numbers behind the decision.
    """

    source_observation_id: SourceObservationId
    rejection: LookupRejection
    detail: str


@dataclass(frozen=True, kw_only=True)
class AuditedProjection:
    """One point's projection, reconstructable from every source it depended on.

    ``geometry -> pose -> extrinsic -> camera model -> raw pixel -> image transform
    -> prepared pixel``.

    Attributes:
        geometry: The map element.
        stage: How far the point got.
        camera_range_m: Distance from the camera optical center, in meters.
        raw_pixel: Raw-image pixel, when the camera model produced one.
        prepared_pixel: Prepared-image pixel, when the point was projected.
        pose_ref: The pose that placed the camera.
        extrinsic: The static body-to-camera extrinsic.
        camera: The calibrated camera.
        calibration_ref: The calibration and camera entry.
        image_transform: The raw-to-prepared chain the pixel went through.
    """

    geometry: GeometryReference
    stage: ProjectionStage
    camera_range_m: float
    raw_pixel: PixelCoordinate | None
    prepared_pixel: PixelCoordinate | None
    pose_ref: PoseRef
    extrinsic: ExtrinsicRef
    camera: CameraIdentity
    calibration_ref: CalibrationRef
    image_transform: RawToPreparedTransform


@dataclass(frozen=True, kw_only=True, eq=False)
class FrameProjection:
    """The projection of every map element for one camera frame.

    Arrays are indexed like the :class:`GeometryCloud` they were computed from.

    Attributes:
        source_observation_id: The camera frame.
        image_timestamp: When the frame was acquired, in the trajectory's clock domain.
        map_id: The map the geometry belongs to.
        map_time_bounds: The acquisition window of the geometry the map was built from.
        camera: The calibrated camera used.
        calibration_ref: The calibration and camera entry used.
        pose_ref: The pose that placed the camera.
        extrinsic: The static body-to-camera extrinsic used.
        image_transform: The raw-to-prepared chain used.
        camera_depth_m: ``(N,)`` signed ``z`` in the camera optical frame, in meters.
        camera_range_m: ``(N,)`` distance from the optical center, in meters.
        raw_pixels: ``(N, 2)`` raw-image pixels; ``NaN`` where the model cannot project.
        prepared_pixels: ``(N, 2)`` prepared-image pixels; ``NaN`` where not projected.
        projectable: ``(N,)`` the camera model could project the point.
        in_prepared_image: ``(N,)`` it projects inside the prepared image.
        in_valid_support: ``(N,)`` it also lies on supported pixels (valid region,
            outside every exclusion region).
    """

    source_observation_id: SourceObservationId
    image_timestamp: SourceTimestamp
    map_id: MapId
    map_time_bounds: TimeBounds
    camera: CameraIdentity
    calibration_ref: CalibrationRef
    pose_ref: PoseRef
    extrinsic: ExtrinsicRef
    image_transform: RawToPreparedTransform
    camera_depth_m: NDArray[Any]
    camera_range_m: NDArray[Any]
    raw_pixels: NDArray[Any]
    prepared_pixels: NDArray[Any]
    projectable: NDArray[Any]
    in_prepared_image: NDArray[Any]
    in_valid_support: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate that the arrays describe the same points, stage by stage.

        Raises:
            ValueError: If the shapes disagree, or a later stage holds a point an
                earlier stage did not (a supported point must be in the image, and an
                in-image point must be projectable).
        """
        if self.projectable.ndim != 1:
            raise ValueError(f"projectable must have shape (N,), got {self.projectable.shape}")
        count = self.projectable.shape[0]
        vector_shape = (count,)
        pixel_shape = (count, 2)
        if (
            self.camera_depth_m.shape != vector_shape
            or self.camera_range_m.shape != vector_shape
            or self.in_prepared_image.shape != vector_shape
            or self.in_valid_support.shape != vector_shape
        ):
            raise ValueError(f"per-point arrays must have shape {vector_shape}")
        if self.raw_pixels.shape != pixel_shape or self.prepared_pixels.shape != pixel_shape:
            raise ValueError(f"pixel arrays must have shape {pixel_shape}")
        if (self.in_prepared_image & ~self.projectable).any():
            raise ValueError("in_prepared_image must imply projectable")
        if (self.in_valid_support & ~self.in_prepared_image).any():
            raise ValueError("in_valid_support must imply in_prepared_image")

    @property
    def support_indices(self) -> NDArray[Any]:
        """Positions of the points that landed on supported prepared-image pixels."""
        import numpy as np

        indices: NDArray[Any] = np.flatnonzero(self.in_valid_support)
        return indices

    def map_reference(self, index: int) -> GeometryReference:
        """Return the stable reference of the ``index``-th map element."""
        return GeometryReference(
            map_id=self.map_id, geometry_id=geometry_id_for(map_id=self.map_id, index=index)
        )

    def stage_counts(self) -> dict[ProjectionStage, int]:
        """Count the points that ended at each stage; every stage is present."""
        supported = int(self.in_valid_support.sum())
        in_image = int(self.in_prepared_image.sum())
        projectable = int(self.projectable.sum())
        return {
            ProjectionStage.BEHIND_CAMERA: len(self.projectable) - projectable,
            ProjectionStage.OUTSIDE_IMAGE: projectable - in_image,
            ProjectionStage.OUTSIDE_VALID_SUPPORT: in_image - supported,
            ProjectionStage.IN_SUPPORT: supported,
        }

    def audit(self, index: int) -> AuditedProjection:
        """Reconstruct one point's whole chain, for diagnostics and audits."""
        if self.in_valid_support[index]:
            stage = ProjectionStage.IN_SUPPORT
        elif self.in_prepared_image[index]:
            stage = ProjectionStage.OUTSIDE_VALID_SUPPORT
        elif self.projectable[index]:
            stage = ProjectionStage.OUTSIDE_IMAGE
        else:
            stage = ProjectionStage.BEHIND_CAMERA
        return AuditedProjection(
            geometry=self.map_reference(index),
            stage=stage,
            camera_range_m=float(self.camera_range_m[index]),
            raw_pixel=_pixel_or_none(self.raw_pixels[index]),
            prepared_pixel=_pixel_or_none(self.prepared_pixels[index]),
            pose_ref=self.pose_ref,
            extrinsic=self.extrinsic,
            camera=self.camera,
            calibration_ref=self.calibration_ref,
            image_transform=self.image_transform,
        )


def _pixel_or_none(pixel: NDArray[Any]) -> PixelCoordinate | None:
    import numpy as np

    if not np.isfinite(pixel).all():
        return None
    return (float(pixel[0]), float(pixel[1]))


class FrameProjector:
    """Projects the geometry of one map into the camera frames of one sequence.

    The constructor fixes what is shared by every frame (map, trajectory, calibration)
    and rejects a combination whose lineage does not fit; :meth:`project` then handles
    one observation at a time.
    """

    def __init__(
        self,
        *,
        cloud: GeometryCloud,
        trajectory: TrajectoryLookup,
        pose_policy: LookupPolicy,
        calibration: CalibrationSet,
        state_estimation_run_id: StateEstimationRunId | None = None,
    ) -> None:
        """Bind a map, a trajectory and a calibration, validating that they belong together.

        Args:
            cloud: The map geometry.
            trajectory: The pose lookup of the selected state-estimation trajectory.
            pose_policy: Which pose lookups are acceptable for an image timestamp.
            calibration: The canonical calibration; it must be the one the map and the
                trajectory were built with.
            state_estimation_run_id: The persisted state-estimation run the trajectory
                came from, recorded in every pose reference.

        Raises:
            AssociationInputError: If the trajectory is in another frame than the map, or
                the map, trajectory and calibration do not share the same trajectory,
                sequence and calibration identity.
        """
        geometric_map = cloud.geometric_map
        source = trajectory.trajectory
        if source.reference_frame != geometric_map.frame_id:
            raise AssociationInputError(
                f"the trajectory is expressed in frame {source.reference_frame!r} but the map "
                f"is in frame {geometric_map.frame_id!r}"
            )
        if geometric_map.provenance.trajectory_id != source.trajectory_id:
            raise AssociationInputError(
                f"the map was built from trajectory {geometric_map.provenance.trajectory_id!r} "
                f"but the association uses trajectory {source.trajectory_id!r}"
            )
        if geometric_map.provenance.sequence_artifact_id != source.provenance.sequence_artifact_id:
            raise AssociationInputError(
                f"the map comes from sequence {geometric_map.provenance.sequence_artifact_id!r} "
                f"but the trajectory from {source.provenance.sequence_artifact_id!r}"
            )
        identity = calibration_identity(calibration)
        if identity is None:
            raise AssociationInputError("the calibration set has no identity")
        for owner, recorded in (
            ("map", geometric_map.provenance.calibration_identity),
            ("trajectory", source.provenance.calibration_identity),
        ):
            if recorded is not None and recorded != identity:
                raise AssociationInputError(
                    f"the calibration differs from the one the {owner} used: {recorded} recorded, "
                    f"{identity} given"
                )
        self._cloud = cloud
        self._trajectory = trajectory
        self._policy = pose_policy
        self._calibration = calibration
        self._calibration_identity = identity
        self._run_id = state_estimation_run_id
        self._graph = StaticFrameGraph.from_calibration(calibration)
        self._cameras: dict[CalibrationReferenceId, CameraProjection] = {}

    def restricted_to(self, indices: NDArray[Any]) -> FrameProjector:
        """Return a projector over a subset of the map, for pixel-only diagnostics.

        Positions in the result follow ``indices``, not the map, so the geometry references
        of its projections do not identify map elements; use it only where the pixels matter.

        Args:
            indices: Positions of the map elements to keep, in the order to keep them.

        Returns:
            A projector with the same trajectory, policy and calibration.
        """
        return FrameProjector(
            cloud=dataclasses.replace(
                self._cloud, coordinates_m=self._cloud.coordinates_m[indices]
            ),
            trajectory=self._trajectory,
            pose_policy=self._policy,
            calibration=self._calibration,
            state_estimation_run_id=self._run_id,
        )

    def project(
        self, observation: ImageObservation, prepared_image: PreparedImage
    ) -> FrameProjection | RejectedProjection:
        """Project the whole map into one camera frame.

        Args:
            observation: The RGB observation; its timestamp selects the pose.
            prepared_image: The image Visual Perception prepared from that observation.

        Returns:
            The projection, or a :class:`RejectedProjection` when the pose lookup policy
            accepts no pose for the observation timestamp.

        Raises:
            AssociationInputError: If the observation has no matching calibration, its
                frame or size disagrees with the calibration, the prepared image belongs
                to another observation, no static extrinsic connects the body to the
                camera, or the image transformation chain cannot be reproduced.
            ClockDomainMismatchError: If the observation is in another clock domain than
                the trajectory.
        """
        import numpy as np

        entry, model = self._camera_entry(observation)
        if prepared_image.source_observation_id != observation.observation_id:
            raise AssociationInputError(
                f"the prepared image comes from source_observation_id "
                f"{prepared_image.source_observation_id!r}, not from {observation.observation_id!r}"
            )
        transform = raw_to_prepared_transform(
            raw_size=(model.width, model.height), prepared_image=prepared_image
        )
        resolved = self._trajectory.pose_for_observation(observation, policy=self._policy)
        if isinstance(resolved, RejectedLookup):
            return RejectedProjection(
                source_observation_id=observation.observation_id,
                rejection=resolved.rejection,
                detail=resolved.detail,
            )
        pose = resolved.pose
        try:
            body_to_camera = self._graph.resolve(pose.child_frame, entry.frame_id)
        except FrameGraphError as error:
            raise AssociationInputError(
                f"no static extrinsic connects the body frame {pose.child_frame!r} to the camera "
                f"frame {entry.frame_id!r}: {error}"
            ) from error
        translation, rotation = compose_rigid(
            outer_translation=pose.translation_m,
            outer_rotation=pose.orientation,
            inner_translation=body_to_camera.translation,
            inner_rotation=body_to_camera.rotation,
        )
        # P_camera = R_map_camera^T (P_map - t_map_camera); em linhas, (P - t) @ R.
        points_camera = (self._cloud.coordinates_m - np.array(translation)) @ np.array(
            quaternion_to_rotation_matrix(rotation)
        )
        camera = self._camera(entry)
        projected = camera.project(points_camera)
        prepared_pixels = transform.map_pixels(projected.pixels)
        in_image = projected.projectable & transform.in_prepared_image(prepared_pixels)
        in_support = in_image & _supported(prepared_image, prepared_pixels, in_image)
        return FrameProjection(
            source_observation_id=observation.observation_id,
            image_timestamp=observation.timestamp,
            map_id=self._cloud.geometric_map.map_id,
            map_time_bounds=self._cloud.geometric_map.time_bounds,
            camera=camera.identity,
            calibration_ref=CalibrationRef(
                calibration_identity=self._calibration_identity,
                camera_calibration_id=entry.calibration_id,
                camera_model_kind=camera.identity.camera_model_kind,
                camera_frame=entry.frame_id,
            ),
            pose_ref=PoseRef(
                trajectory_id=self._trajectory.trajectory.trajectory_id,
                state_estimation_run_id=self._run_id,
                source_estimate_ids=resolved.source_estimate_ids,
                lookup_outcome=resolved.outcome,
                time_delta_ns=resolved.time_delta_ns,
                interpolation_fraction=resolved.interpolation_fraction,
            ),
            extrinsic=ExtrinsicRef(
                calibration_identity=self._calibration_identity,
                parent_frame=pose.child_frame,
                child_frame=entry.frame_id,
            ),
            image_transform=transform,
            camera_depth_m=projected.depth_m,
            camera_range_m=projected.range_m,
            raw_pixels=projected.pixels,
            prepared_pixels=prepared_pixels,
            projectable=projected.projectable,
            in_prepared_image=in_image,
            in_valid_support=in_support,
        )

    def _camera_entry(self, observation: ImageObservation) -> tuple[CalibrationEntry, CameraModel]:
        if observation.calibration_id is None:
            raise AssociationInputError(
                f"observation {observation.observation_id!r} names no calibration"
            )
        entry = self._calibration.entries.get(observation.calibration_id)
        model = None if entry is None else entry.camera_model
        if entry is None or model is None:
            raise AssociationInputError(
                f"the calibration has no camera model for {observation.calibration_id!r}"
            )
        if observation.frame_id != entry.frame_id:
            raise AssociationInputError(
                f"observation {observation.observation_id!r} is in frame {observation.frame_id!r} "
                f"but its calibration is in frame {entry.frame_id!r}"
            )
        if (observation.width, observation.height) != (model.width, model.height):
            raise AssociationInputError(
                f"observation {observation.observation_id!r} has size "
                f"{(observation.width, observation.height)} but its calibration describes "
                f"{(model.width, model.height)}"
            )
        return entry, model

    def _camera(self, entry: CalibrationEntry) -> CameraProjection:
        camera = self._cameras.get(entry.calibration_id)
        if camera is None:
            camera = camera_projection_for(entry)
            self._cameras[entry.calibration_id] = camera
        return camera


def _supported(
    prepared_image: PreparedImage, prepared_pixels: NDArray[Any], in_image: NDArray[Any]
) -> NDArray[Any]:
    """Tell which prepared pixels are inside the valid region and outside every exclusion.

    Only rows already inside the prepared image are looked up; the others stay ``False``.
    """
    import numpy as np

    supported = np.zeros(in_image.shape, dtype=bool)
    if not in_image.any():
        return supported
    # O pixel de índice i cobre [i - 0.5, i + 0.5): o índice do centro c é floor(c + 0.5).
    columns = np.floor(prepared_pixels[in_image, 0] + 0.5).astype(np.int64)
    rows = np.floor(prepared_pixels[in_image, 1] + 0.5).astype(np.int64)
    allowed = np.ones(columns.shape, dtype=bool)
    if prepared_image.valid_region is not None:
        allowed &= _mask_array(prepared_image.valid_region.mask)[rows, columns]
    for exclusion in prepared_image.exclusion_regions:
        allowed &= ~_mask_array(exclusion.mask)[rows, columns]
    supported[in_image] = allowed
    return supported


def _mask_array(mask: InlineMask) -> NDArray[Any]:
    import numpy as np

    array: NDArray[Any] = np.array(mask.data, dtype=bool).reshape(mask.height, mask.width)
    return array
