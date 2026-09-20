import dataclasses
import math

import numpy as np
import pytest
from projection_builders import (
    BODY_TO_CAMERA_ROTATION,
    CAMERA_CALIBRATION_ID,
    IDENTITY,
    YAW_90,
    ArrayGeometrySource,
    artifact,
    full_mask,
    make_calibration,
    make_camera_observation,
    make_lookup,
    make_prepared_image,
    make_trajectory,
    mask_with,
)

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import (
    CalibrationReferenceId,
    CalibrationSet,
    FisheyeCameraModel,
    FrameId,
    PinholeCameraModel,
    SequenceArtifactId,
)
from contextmap.sensor_association import CameraProjection, camera_projection_for
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.frame_projection import (
    ExtrinsicRef,
    FrameProjection,
    FrameProjector,
    ProjectionStage,
    RejectedProjection,
)
from contextmap.sensor_association.geometry_cloud import GeometryCloud
from contextmap.sensor_association.image_transform import raw_to_prepared_transform
from contextmap.shared import Quaternion, Vector3, rotate_vector
from contextmap.state_estimation import (
    ClockDomainMismatchError,
    LookupOutcome,
    LookupPolicy,
    LookupRejection,
    TrajectoryId,
    calibration_identity,
)
from contextmap.visual_perception import (
    BoundingBox,
    CropOperation,
    ExclusionRegion,
    PreparedImage,
    RectifyOperation,
    ResizeOperation,
    ValidRegion,
)

# Ponto visto pela câmera em (x, y, z) = (0.5, -0.3, 2.0): com o corpo na origem e sem
# rotação, o corpo o vê em (2.0, -0.5, 0.3). Pixel cru esperado: (445, 165).
CAMERA_POINT: Vector3 = (0.5, -0.3, 2.0)
BASE_MAP_POINT: Vector3 = (2.0, -0.5, 0.3)
RAW_PIXEL = (445.0, 165.0)
EXACT = LookupPolicy.exact()


def _projector(
    points: list[Vector3] | None = None,
    *,
    calibration: CalibrationSet | None = None,
    poses: list | None = None,  # type: ignore[type-arg]
    policy: LookupPolicy = EXACT,
    **map_options: object,
) -> FrameProjector:
    calibration = calibration if calibration is not None else make_calibration()
    trajectory = (
        make_trajectory(calibration) if poses is None else make_trajectory(calibration, poses)
    )
    source = ArrayGeometrySource(
        points if points is not None else [BASE_MAP_POINT],
        calibration=calibration,
        **map_options,  # type: ignore[arg-type]
    )
    return FrameProjector(
        cloud=GeometryCloud.from_source(source),
        trajectory=make_lookup(trajectory),
        pose_policy=policy,
        calibration=calibration,
    )


def _project(
    projector: FrameProjector,
    prepared: PreparedImage | None = None,
    *,
    time_ns: int = 0,
    **observation_options: object,
) -> FrameProjection:
    result = projector.project(
        make_camera_observation(time_ns, **observation_options),  # type: ignore[arg-type]
        prepared if prepared is not None else make_prepared_image(),
    )
    assert isinstance(result, FrameProjection)
    return result


def _resize(width: int, height: int) -> ResizeOperation:
    return ResizeOperation(
        width=width,
        height=height,
        output_image=artifact(f"resize-{width}"),
        provenance_source="cfg",
    )


def _crop(x_min: float, y_min: float, x_max: float, y_max: float) -> CropOperation:
    return CropOperation(
        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        output_image=artifact("crop"),
        provenance_source="cfg",
    )


CROP_THEN_RESIZE = (_crop(100, 50, 500, 350), _resize(200, 150))
# A câmera vê o ponto base em (445, 165); o corte o leva a (345, 115) e a metade da escala
# a ((345.5 * 0.5) - 0.5, (115.5 * 0.5) - 0.5) = (172.25, 57.25), o pixel de índice (172, 57).
PREPARED_PIXEL = (172.25, 57.25)


# --- The 3D chain -----------------------------------------------------------


def test_the_extrinsic_fixture_points_the_camera_along_the_body_forward_axis() -> None:
    forward = rotate_vector(BODY_TO_CAMERA_ROTATION, (0.0, 0.0, 1.0))

    np.testing.assert_allclose(forward, (1.0, 0.0, 0.0), atol=1e-12)


def test_a_known_map_point_reaches_the_expected_prepared_pixel() -> None:
    frame = _project(_projector())

    np.testing.assert_allclose(frame.raw_pixels, [RAW_PIXEL])
    np.testing.assert_allclose(frame.prepared_pixels, [RAW_PIXEL])
    np.testing.assert_allclose(frame.camera_range_m, [math.sqrt(0.25 + 0.09 + 4.0)])
    assert frame.projectable.tolist() == [True]
    assert frame.in_prepared_image.tolist() == [True]
    assert frame.in_valid_support.tolist() == [True]


def _map_point_seen_at(
    camera_point: Vector3,
    *,
    body_translation: Vector3,
    body_orientation: Quaternion,
    extrinsic_translation: Vector3,
) -> Vector3:
    """Build the map-frame point by composing the chain forward, camera -> body -> map."""
    in_body = rotate_vector(BODY_TO_CAMERA_ROTATION, camera_point)
    in_body = tuple(a + b for a, b in zip(in_body, extrinsic_translation, strict=True))  # type: ignore[assignment]
    in_map = rotate_vector(body_orientation, in_body)  # type: ignore[arg-type]
    return tuple(a + b for a, b in zip(in_map, body_translation, strict=True))  # type: ignore[return-value]


def test_the_camera_is_placed_by_the_body_pose_and_the_static_extrinsic() -> None:
    body_translation = (1.0, 2.0, 0.5)
    extrinsic_translation = (0.2, -0.1, 0.3)
    point = _map_point_seen_at(
        CAMERA_POINT,
        body_translation=body_translation,
        body_orientation=YAW_90,
        extrinsic_translation=extrinsic_translation,
    )
    calibration = make_calibration(extrinsic_translation=extrinsic_translation)
    projector = _projector(
        [point],
        calibration=calibration,
        poses=[(0, body_translation, YAW_90), (100_000_000, body_translation, YAW_90)],
    )

    frame = _project(projector)

    np.testing.assert_allclose(frame.prepared_pixels, [RAW_PIXEL], atol=1e-9)
    np.testing.assert_allclose(frame.camera_range_m, [np.linalg.norm(CAMERA_POINT)], atol=1e-12)


def test_the_pose_is_interpolated_to_the_image_timestamp_and_the_interpolation_is_recorded() -> (
    None
):
    # O corpo avança 1 m em 100 ms; aos 25 ms ele está em x = 0.25.
    point = _map_point_seen_at(
        CAMERA_POINT,
        body_translation=(0.25, 0.0, 0.0),
        body_orientation=IDENTITY,
        extrinsic_translation=(0.0, 0.0, 0.0),
    )
    projector = _projector(
        [point],
        poses=[(0, (0.0, 0.0, 0.0), IDENTITY), (100_000_000, (1.0, 0.0, 0.0), IDENTITY)],
        policy=LookupPolicy.interpolated(),
    )

    frame = _project(projector, time_ns=25_000_000)

    np.testing.assert_allclose(frame.prepared_pixels, [RAW_PIXEL], atol=1e-9)
    assert frame.pose_ref.lookup_outcome is LookupOutcome.INTERPOLATED
    assert len(frame.pose_ref.source_estimate_ids) == 2
    assert frame.pose_ref.interpolation_fraction == pytest.approx(0.25)
    assert frame.pose_ref.time_delta_ns == 25_000_000


def test_a_point_behind_the_camera_is_reported_and_never_clipped() -> None:
    frame = _project(_projector([(-3.0, 0.0, 0.0)]))

    assert frame.projectable.tolist() == [False]
    assert np.isnan(frame.raw_pixels).all()
    assert np.isnan(frame.prepared_pixels).all()
    assert frame.audit(0).stage is ProjectionStage.BEHIND_CAMERA


def test_a_point_beside_the_camera_projects_but_falls_outside_the_image() -> None:
    frame = _project(_projector([(0.5, -10.0, 0.0)]))

    assert frame.projectable.tolist() == [True]
    assert frame.in_prepared_image.tolist() == [False]
    assert frame.in_valid_support.tolist() == [False]
    assert frame.audit(0).stage is ProjectionStage.OUTSIDE_IMAGE


# --- The 2D chain -----------------------------------------------------------


def test_crop_and_resize_carry_the_pixel_to_the_prepared_image() -> None:
    frame = _project(_projector(), make_prepared_image(CROP_THEN_RESIZE))

    np.testing.assert_allclose(frame.raw_pixels, [RAW_PIXEL])
    np.testing.assert_allclose(frame.prepared_pixels, [PREPARED_PIXEL])
    assert frame.image_transform.prepared_size == (200, 150)


def test_a_point_cropped_away_is_outside_the_prepared_image_though_inside_the_raw_one() -> None:
    # Câmera em (-0.6, 0, 1): pixel cru (20, 240), dentro da imagem crua; o corte a partir
    # de x = 100 o deixa em x = -80 na imagem preparada.
    projector = _projector([(1.0, 0.6, 0.0)])
    frame = _project(projector, make_prepared_image((_crop(100, 50, 500, 350),)))

    np.testing.assert_allclose(frame.raw_pixels, [(20.0, 240.0)], atol=1e-9)
    assert projector_camera_in_image(frame.raw_pixels)
    assert frame.in_prepared_image.tolist() == [False]
    assert frame.audit(0).stage is ProjectionStage.OUTSIDE_IMAGE


def projector_camera_in_image(raw_pixels: np.ndarray) -> bool:  # type: ignore[type-arg]
    camera: CameraProjection = camera_projection_for(
        make_calibration().entries[CAMERA_CALIBRATION_ID]
    )
    return bool(camera.in_image(raw_pixels).all())


def test_mask_membership_follows_the_prepared_pixel_and_not_the_raw_one() -> None:
    # Só o pixel (172, 57) da imagem preparada está excluído. Um mapeamento que ignorasse o
    # corte ou a escala cairia em outro pixel e deixaria o ponto passar.
    prepared = make_prepared_image(
        CROP_THEN_RESIZE,
        exclusion_regions=(
            ExclusionRegion(
                name="marked", mask=mask_with(200, 150, (172, 57)), reason="fixture", source="test"
            ),
        ),
    )

    frame = _project(_projector(), prepared)

    assert frame.in_prepared_image.tolist() == [True]
    assert frame.in_valid_support.tolist() == [False]
    assert frame.audit(0).stage is ProjectionStage.OUTSIDE_VALID_SUPPORT


def test_the_mask_pixel_is_the_one_whose_center_is_nearest_to_the_prepared_pixel() -> None:
    # Câmera em (0.504, -0.296, 2): pixel cru (446, 166) e preparado (172.75, 57.75), mais perto
    # do centro do pixel (173, 58) que do (172, 57): truncar em vez de arredondar erra meio pixel.
    point = (2.0, -0.504, 0.296)

    def in_support(marked: tuple[int, int]) -> bool:
        prepared = make_prepared_image(
            CROP_THEN_RESIZE,
            exclusion_regions=(
                ExclusionRegion(
                    name="marked", mask=mask_with(200, 150, marked), reason="fixture", source="test"
                ),
            ),
        )
        frame = _project(_projector([point]), prepared)
        np.testing.assert_allclose(frame.prepared_pixels, [(172.75, 57.75)], atol=1e-9)
        return bool(frame.in_valid_support[0])

    assert not in_support((173, 58))
    assert in_support((172, 57))


def test_a_neighboring_exclusion_pixel_does_not_remove_the_point() -> None:
    prepared = make_prepared_image(
        CROP_THEN_RESIZE,
        exclusion_regions=(
            ExclusionRegion(
                name="neighbors",
                mask=mask_with(200, 150, (171, 57), (173, 57), (172, 56), (172, 58)),
                reason="fixture",
                source="test",
            ),
        ),
    )

    assert _project(_projector(), prepared).in_valid_support.tolist() == [True]


def test_the_valid_region_must_contain_the_prepared_pixel() -> None:
    inside = ValidRegion(mask=full_mask(200, 150), reason="fixture", source="test")
    hole = ValidRegion(mask=full_mask(200, 150, hole=(172, 57)), reason="fixture", source="test")

    assert _project(
        _projector(), make_prepared_image(CROP_THEN_RESIZE, valid_region=inside)
    ).in_valid_support.tolist() == [True]
    assert _project(
        _projector(), make_prepared_image(CROP_THEN_RESIZE, valid_region=hole)
    ).in_valid_support.tolist() == [False]


def test_the_stage_counts_partition_every_point() -> None:
    points = [BASE_MAP_POINT, (-3.0, 0.0, 0.0), (0.5, -10.0, 0.0), (1.0, 0.6, 0.0)]
    frame = _project(_projector(points), make_prepared_image((_crop(100, 50, 500, 350),)))

    counts = frame.stage_counts()

    assert counts == {
        ProjectionStage.BEHIND_CAMERA: 1,
        ProjectionStage.OUTSIDE_IMAGE: 2,
        ProjectionStage.OUTSIDE_VALID_SUPPORT: 0,
        ProjectionStage.IN_SUPPORT: 1,
    }
    assert sum(counts.values()) == 4
    assert frame.support_indices.tolist() == [0]


# --- Provenance -------------------------------------------------------------


def test_every_audited_projection_names_its_whole_chain() -> None:
    calibration = make_calibration()
    identity = calibration_identity(calibration)
    assert identity is not None
    projector = _projector(calibration=calibration)
    prepared = make_prepared_image(CROP_THEN_RESIZE)

    frame = _project(projector, prepared)
    audit = frame.audit(0)

    entry = calibration.entries[CAMERA_CALIBRATION_ID]
    assert audit.geometry == frame.map_reference(0)
    assert audit.geometry.map_id == MapId("map-0001")
    assert audit.pose_ref.trajectory_id == TrajectoryId("run-0001--trajectory")
    assert audit.pose_ref.lookup_outcome is LookupOutcome.EXACT
    assert audit.extrinsic == ExtrinsicRef(
        calibration_identity=identity,
        parent_frame=FrameId("body"),
        child_frame=FrameId("camera_optical"),
    )
    assert audit.camera == camera_projection_for(entry).identity
    assert audit.calibration_ref.calibration_identity == identity
    assert audit.calibration_ref.camera_calibration_id == CAMERA_CALIBRATION_ID
    assert audit.calibration_ref.camera_model_kind == "pinhole"
    assert audit.raw_pixel == pytest.approx(RAW_PIXEL)
    assert audit.image_transform == raw_to_prepared_transform(
        raw_size=(640, 480), prepared_image=prepared
    )
    assert audit.prepared_pixel == pytest.approx(PREPARED_PIXEL)
    assert audit.stage is ProjectionStage.IN_SUPPORT
    assert frame.source_observation_id == "frame-0001"
    assert frame.image_timestamp == make_camera_observation(0).timestamp
    assert frame.map_time_bounds.start.total_nanoseconds() == 0
    assert frame.map_time_bounds.end.total_nanoseconds() == 100_000_000


# --- Rejections that are data, not errors -----------------------------------


def test_a_pose_the_policy_does_not_accept_is_a_counted_rejection() -> None:
    projector = _projector()
    observation = make_camera_observation(10_000_000_000)

    result = projector.project(observation, make_prepared_image())

    expected = make_lookup(make_trajectory(make_calibration())).pose_at(
        observation.timestamp, policy=EXACT
    )
    assert isinstance(result, RejectedProjection)
    assert result.source_observation_id == observation.observation_id
    assert result.rejection is expected.rejection  # type: ignore[union-attr]
    assert result.rejection is LookupRejection.OUT_OF_RANGE


def test_an_observation_in_another_clock_domain_is_an_error_not_a_rejection() -> None:
    projector = _projector()

    with pytest.raises(ClockDomainMismatchError):
        projector.project(make_camera_observation(0, clock_id="other:clock"), make_prepared_image())


# --- Incompatible inputs fail early -----------------------------------------


def test_the_trajectory_must_be_in_the_frame_of_the_map() -> None:
    calibration = make_calibration()
    source = ArrayGeometrySource([BASE_MAP_POINT], calibration=calibration)

    with pytest.raises(AssociationInputError, match="frame"):
        FrameProjector(
            cloud=GeometryCloud.from_source(source),
            trajectory=make_lookup(make_trajectory(calibration, reference_frame="odom")),
            pose_policy=EXACT,
            calibration=calibration,
        )


def test_the_map_and_the_trajectory_must_come_from_the_same_trajectory() -> None:
    with pytest.raises(AssociationInputError, match="trajectory"):
        _projector(trajectory_id=TrajectoryId("other-trajectory"))


def test_the_map_and_the_trajectory_must_come_from_the_same_sequence() -> None:
    with pytest.raises(AssociationInputError, match="sequence"):
        _projector(sequence_artifact_id=SequenceArtifactId("sequence-other"))


def test_the_calibration_must_be_the_one_the_map_and_the_trajectory_used() -> None:
    calibration = make_calibration()
    other = make_calibration(extrinsic_translation=(1.0, 0.0, 0.0))
    source = ArrayGeometrySource([BASE_MAP_POINT], calibration=calibration)

    with pytest.raises(AssociationInputError, match="calibration"):
        FrameProjector(
            cloud=GeometryCloud.from_source(source),
            trajectory=make_lookup(make_trajectory(calibration)),
            pose_policy=EXACT,
            calibration=other,
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"calibration_id": None}, "calibration"),
        ({"calibration_id": CalibrationReferenceId("unknown-calib")}, "calibration"),
        ({"frame": "another_optical"}, "frame"),
        ({"width": 320, "height": 240}, "size"),
    ],
)
def test_an_observation_that_disagrees_with_its_calibration_is_rejected(
    options: dict[str, object], message: str
) -> None:
    projector = _projector()

    with pytest.raises(AssociationInputError, match=message):
        projector.project(make_camera_observation(0, **options), make_prepared_image())  # type: ignore[arg-type]


def test_a_camera_without_a_static_extrinsic_to_the_body_is_rejected() -> None:
    projector = _projector(calibration=make_calibration(with_extrinsic=False))

    with pytest.raises(AssociationInputError, match="extrinsic"):
        projector.project(make_camera_observation(0), make_prepared_image())


def test_a_prepared_image_of_another_observation_is_rejected() -> None:
    projector = _projector()

    with pytest.raises(AssociationInputError, match="source_observation_id"):
        projector.project(
            make_camera_observation(0), make_prepared_image(source_observation_id="frame-9999")
        )


def test_an_unsupported_image_transform_fails_the_projection_instead_of_guessing() -> None:
    operation = RectifyOperation(
        calibration_id="camera-calib", output_image=artifact("rect"), provenance_source="cfg"
    )
    projector = _projector()

    with pytest.raises(AssociationInputError, match="rectif"):
        projector.project(make_camera_observation(0), make_prepared_image((operation,)))


def test_the_camera_model_declared_by_the_calibration_is_the_one_used() -> None:
    # Um pinhole e um fisheye com os mesmos parâmetros projetam diferente: o modelo declarado manda.
    fisheye = FisheyeCameraModel(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        distortion_coefficients=(0, 0, 0, 0),
    )
    pinhole_frame = _project(_projector())
    fisheye_frame = _project(_projector(calibration=make_calibration(model=fisheye)))

    assert not np.allclose(pinhole_frame.raw_pixels, fisheye_frame.raw_pixels)
    assert fisheye_frame.calibration_ref.camera_model_kind == "fisheye"
    assert isinstance(
        make_calibration().entries[CAMERA_CALIBRATION_ID].camera_model, PinholeCameraModel
    )


def test_the_result_arrays_are_consistent() -> None:
    frame = _project(_projector([BASE_MAP_POINT, (-3.0, 0.0, 0.0)]))

    with pytest.raises(ValueError, match="shape"):
        dataclasses.replace(frame, camera_range_m=np.zeros(3))
    with pytest.raises(ValueError, match="in_prepared_image"):
        dataclasses.replace(frame, in_prepared_image=np.array([True, True]))
