"""Deterministic builders for the map -> camera -> prepared-image projection tests.

The scene is small and every expected pixel in the tests can be checked by hand:
a body frame with x forward, y left, z up, and a camera optical frame (x right,
y down, z forward) rigidly attached to it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    MapId,
    TransformKind,
    TransformLineage,
    TransformStep,
    geometry_id_for,
)
from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    CameraModel,
    FrameId,
    ImageEncoding,
    ImageObservation,
    PinholeCameraModel,
    RigidTransform,
    SensorId,
    SequenceArtifactId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.shared import SourceTimestamp, Vector3
from contextmap.state_estimation import (
    EstimatorProvenance,
    LookupPolicy,
    PoseEstimate,
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
    TimeBounds,
    Trajectory,
    TrajectoryId,
    TrajectoryLookup,
    TrajectoryProvenance,
    calibration_identity,
    pose_estimate_id_for,
)
from contextmap.visual_perception import (
    ArtifactReference,
    ExclusionRegion,
    InlineMask,
    PreparedImage,
    SourceImage,
    ValidRegion,
    prepare_image,
)
from contextmap.visual_perception.image_preparation import PreparationOperation

CLOCK_ID = "fixture:header"
MAP_ID = MapId("map-0001")
TRAJECTORY_ID = TrajectoryId("run-0001--trajectory")
SEQUENCE_ID = SequenceArtifactId("sequence-0001")
CAMERA_CALIBRATION_ID = CalibrationReferenceId("camera-calib")
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
FOCAL_PX = 500.0
PRINCIPAL_X, PRINCIPAL_Y = 320.0, 240.0
# Camera optical axes expressed in the body frame: x_cam = -y_body, y_cam = -z_body,
# z_cam = +x_body, so the camera looks along the body's forward axis.
BODY_TO_CAMERA_ROTATION = (-0.5, 0.5, -0.5, 0.5)
IDENTITY = (0.0, 0.0, 0.0, 1.0)
YAW_90 = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)


def timestamp_ns(total_nanoseconds: int, *, clock_id: str = CLOCK_ID) -> SourceTimestamp:
    seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)


def pinhole_model() -> PinholeCameraModel:
    return PinholeCameraModel(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        fx=FOCAL_PX,
        fy=FOCAL_PX,
        cx=PRINCIPAL_X,
        cy=PRINCIPAL_Y,
    )


def make_calibration(
    *,
    model: CameraModel | None = None,
    camera_frame: str = "camera_optical",
    extrinsic_translation: Vector3 = (0.0, 0.0, 0.0),
    with_extrinsic: bool = True,
) -> CalibrationSet:
    camera_model = model if model is not None else pinhole_model()
    entry = CalibrationEntry(
        calibration_id=CAMERA_CALIBRATION_ID,
        sensor_id=SensorId("camera"),
        frame_id=FrameId(camera_frame),
        camera_model=camera_model,
        provenance=CalibrationProvenance(source_type="fixture", source_path="fixtures/cal"),
        content_hash=compute_content_hash(
            sensor_id=SensorId("camera"),
            frame_id=FrameId(camera_frame),
            camera_model=camera_model,
        ),
    )
    transforms = (
        (
            RigidTransform(
                parent_frame=FrameId("body"),
                child_frame=FrameId(camera_frame),
                translation=extrinsic_translation,
                rotation=BODY_TO_CAMERA_ROTATION,
            ),
        )
        if with_extrinsic
        else ()
    )
    return CalibrationSet(entries={CAMERA_CALIBRATION_ID: entry}, static_transforms=transforms)


def make_trajectory(
    calibration: CalibrationSet,
    poses: Sequence[tuple[int, Vector3, tuple[float, float, float, float]]] = (
        (0, (0.0, 0.0, 0.0), IDENTITY),
        (100_000_000, (0.0, 0.0, 0.0), IDENTITY),
    ),
    *,
    reference_frame: str = "map",
    trajectory_id: TrajectoryId = TRAJECTORY_ID,
) -> Trajectory:
    estimates = tuple(
        PoseEstimate(
            estimate_id=pose_estimate_id_for(trajectory_id=trajectory_id, index=index),
            timestamp=timestamp_ns(time_ns),
            parent_frame=FrameId(reference_frame),
            child_frame=FrameId("body"),
            translation_m=translation,
            orientation=orientation,
            validity=PoseValidity.VALID,
            provenance=PoseProvenance(
                source_observation_ids=(SourceObservationId(f"pose-{index:04d}"),)
            ),
        )
        for index, (time_ns, translation, orientation) in enumerate(poses)
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        reference_frame=FrameId(reference_frame),
        body_frame=FrameId("body"),
        poses=estimates,
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(
                backend_id="fake_estimator",
                backend_version="0",
                configuration_fingerprint="sha256:cfg",
            ),
            sequence_artifact_id=SEQUENCE_ID,
            selection_id="full-sequence",
            calibration_identity=calibration_identity(calibration),
            code_version="test",
        ),
    )


class ArrayGeometrySource:
    """A ``GeometrySource`` over an explicit list of map-frame points."""

    def __init__(
        self,
        coordinates_m: Sequence[Vector3],
        *,
        calibration: CalibrationSet,
        map_id: MapId = MAP_ID,
        frame: str = "map",
        trajectory_id: TrajectoryId = TRAJECTORY_ID,
        sequence_artifact_id: SequenceArtifactId = SEQUENCE_ID,
        positional_ids: bool = True,
    ) -> None:
        self._points = tuple(
            _geometry_point(
                index,
                coordinates,
                map_id=map_id,
                frame=frame,
                positional_ids=positional_ids,
            )
            for index, coordinates in enumerate(coordinates_m)
        )
        self._map = GeometricMap(
            map_id=map_id,
            frame_id=FrameId(frame),
            point_count=len(self._points),
            bounds=Bounds3D.enclosing(list(coordinates_m), frame_id=FrameId(frame)),
            source_observation_ids=(SourceObservationId("lidar-0001"),),
            time_bounds=TimeBounds(start=timestamp_ns(0), end=timestamp_ns(100_000_000)),
            spatial_index=None,
            provenance=GeometricMapProvenance(
                sequence_artifact_id=sequence_artifact_id,
                selection_id="full-sequence",
                trajectory_id=trajectory_id,
                state_estimation_run_id=None,
                calibration_identity=calibration_identity(calibration),
                pose_lookup=LookupPolicy.interpolated(),
            ),
        )

    @property
    def geometric_map(self) -> GeometricMap:
        return self._map

    def get(self, reference: GeometryReference) -> GeometryPoint:
        for point in self._points:
            if point.geometry_id == reference.geometry_id:
                return point
        raise KeyError(reference)

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        return iter(self._points)

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        raise NotImplementedError


def _geometry_point(
    index: int, coordinates: Vector3, *, map_id: MapId, frame: str, positional_ids: bool
) -> GeometryPoint:
    geometry_id = (
        geometry_id_for(map_id=map_id, index=index)
        if positional_ids
        else geometry_id_for(map_id=map_id, index=index + 1000)
    )
    return GeometryPoint(
        geometry_id=geometry_id,
        map_id=map_id,
        map_frame=FrameId(frame),
        coordinates_m=coordinates,
        source_frame=FrameId("lidar"),
        source_coordinates_m=(0.0, 0.0, 0.0),
        source_observation_id=SourceObservationId("lidar-0001"),
        source_point_index=index,
        acquisition_timestamp=timestamp_ns(0),
        transform_lineage=TransformLineage(
            steps=(
                TransformStep(
                    kind=TransformKind.DYNAMIC_POSE,
                    parent_frame=FrameId(frame),
                    child_frame=FrameId("body"),
                    reference=str(pose_estimate_id_for(trajectory_id=TRAJECTORY_ID, index=0)),
                    source_estimate_ids=(PoseEstimateId("traj--pose-000000"),),
                ),
                TransformStep(
                    kind=TransformKind.STATIC_CALIBRATION,
                    parent_frame=FrameId("body"),
                    child_frame=FrameId("lidar"),
                    reference="sha256:calibration",
                ),
            )
        ),
        provenance=GeometryPointProvenance(),
    )


def make_lookup(trajectory: Trajectory) -> TrajectoryLookup:
    return TrajectoryLookup(trajectory)


def make_camera_observation(
    time_ns: int,
    *,
    observation_id: str = "frame-0001",
    calibration_id: CalibrationReferenceId | None = CAMERA_CALIBRATION_ID,
    frame: str = "camera_optical",
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    clock_id: str = CLOCK_ID,
) -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("camera"),
        frame_id=FrameId(frame),
        timestamp=timestamp_ns(time_ns, clock_id=clock_id),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/camera"),
        calibration_id=calibration_id,
        width=width,
        height=height,
        encoding=ImageEncoding.MONO8,
        data=bytes(width * height),
    )


def artifact(name: str) -> ArtifactReference:
    return ArtifactReference(uri=f"prepared/{name}.png", sha256="0" * 64, media_type="image/png")


def make_prepared_image(
    operations: tuple[PreparationOperation, ...] = (),
    *,
    source_observation_id: str = "frame-0001",
    raw_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    valid_region: ValidRegion | None = None,
    exclusion_regions: tuple[ExclusionRegion, ...] = (),
) -> PreparedImage:
    return prepare_image(
        SourceImage(
            source_observation_id=source_observation_id,
            image=artifact("raw"),
            width=raw_size[0],
            height=raw_size[1],
        ),
        operations=operations,
        valid_region=valid_region,
        exclusion_regions=exclusion_regions,
    )


def mask_with(width: int, height: int, *cells: tuple[int, int]) -> InlineMask:
    """A binary mask that is foreground exactly at the given ``(x, y)`` pixels."""
    marked = set(cells)
    return InlineMask(
        width=width,
        height=height,
        data=tuple((x, y) in marked for y in range(height) for x in range(width)),
    )


def full_mask(width: int, height: int, *, hole: tuple[int, int] | None = None) -> InlineMask:
    """A mask that is foreground everywhere, except optionally at one pixel."""
    return InlineMask(
        width=width,
        height=height,
        data=tuple((x, y) != hole for y in range(height) for x in range(width)),
    )


def map_point_for_pixel(
    u: float,
    v: float,
    depth_z: float,
    *,
    focal_px: float = FOCAL_PX,
    principal: tuple[float, float] = (PRINCIPAL_X, PRINCIPAL_Y),
) -> Vector3:
    """The map-frame point a pinhole camera sees at raw pixel ``(u, v)`` and depth ``depth_z``.

    Assumes the body sits at the map origin with identity orientation and the camera
    at the body origin, so body ``x`` is camera ``z``, body ``y`` is camera ``-x`` and
    body ``z`` is camera ``-y``.
    """
    x_camera = (u - principal[0]) / focal_px * depth_z
    y_camera = (v - principal[1]) / focal_px * depth_z
    return (depth_z, -x_camera, -y_camera)


def map_point_for_camera_point(camera_point: Vector3) -> Vector3:
    """The map-frame point at a camera-frame position, under the same placement."""
    return (camera_point[2], -camera_point[0], -camera_point[1])
