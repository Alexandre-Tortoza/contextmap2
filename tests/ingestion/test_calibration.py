import pytest

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationError,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    DistortionModel,
    FisheyeCameraModel,
    FrameId,
    PinholeCameraModel,
    RigidTransform,
    SensorId,
    camera_model_kind,
    validate_calibration_set,
)
from contextmap.ingestion.calibration import (
    compute_content_hash,
    decode_calibration_set,
    encode_calibration_set,
    ensure_valid_calibration_set,
)

_PROVENANCE = CalibrationProvenance(source_type="dataset", source_path="fixtures/calibration.yaml")


def _pinhole_entry() -> CalibrationEntry:
    model = PinholeCameraModel(
        width=1280,
        height=720,
        fx=600.0,
        fy=600.0,
        cx=640.0,
        cy=360.0,
        distortion_model=DistortionModel.PLUMB_BOB,
        distortion_coefficients=(0.1, -0.05, 0.0, 0.0, 0.0),
    )
    return CalibrationEntry(
        calibration_id=CalibrationReferenceId("front_camera-calib"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        camera_model=model,
        provenance=_PROVENANCE,
        content_hash=compute_content_hash(
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            camera_model=model,
        ),
    )


def _fisheye_entry() -> CalibrationEntry:
    model = FisheyeCameraModel(
        width=1280,
        height=800,
        fx=400.0,
        fy=400.0,
        cx=640.0,
        cy=400.0,
        distortion_coefficients=(0.01, 0.002, 0.0003, 0.00004),
    )
    return CalibrationEntry(
        calibration_id=CalibrationReferenceId("fisheye_camera-calib"),
        sensor_id=SensorId("fisheye_camera"),
        frame_id=FrameId("fisheye_camera_optical"),
        camera_model=model,
        provenance=_PROVENANCE,
        content_hash=compute_content_hash(
            sensor_id=SensorId("fisheye_camera"),
            frame_id=FrameId("fisheye_camera_optical"),
            camera_model=model,
        ),
    )


def _lidar_entry() -> CalibrationEntry:
    return CalibrationEntry(
        calibration_id=CalibrationReferenceId("velodyne_top-calib"),
        sensor_id=SensorId("velodyne_top"),
        frame_id=FrameId("velodyne"),
        camera_model=None,
        provenance=_PROVENANCE,
        content_hash=compute_content_hash(
            sensor_id=SensorId("velodyne_top"), frame_id=FrameId("velodyne"), camera_model=None
        ),
    )


def test_pinhole_and_fisheye_coexist_without_lossy_conversion() -> None:
    calibration_set = CalibrationSet(
        entries={
            CalibrationReferenceId("front_camera-calib"): _pinhole_entry(),
            CalibrationReferenceId("fisheye_camera-calib"): _fisheye_entry(),
        },
        static_transforms=(),
    )

    pinhole = calibration_set.entries[CalibrationReferenceId("front_camera-calib")].camera_model
    fisheye = calibration_set.entries[CalibrationReferenceId("fisheye_camera-calib")].camera_model
    assert pinhole is not None
    assert fisheye is not None

    assert camera_model_kind(pinhole) == "pinhole"
    assert camera_model_kind(fisheye) == "fisheye"
    assert isinstance(pinhole, PinholeCameraModel)
    assert isinstance(fisheye, FisheyeCameraModel)
    assert validate_calibration_set(calibration_set) == []


def test_lidar_entry_has_no_camera_model() -> None:
    entry = _lidar_entry()
    assert entry.camera_model is None


def test_static_transform_follows_parent_child_convention() -> None:
    transform = RigidTransform(
        parent_frame=FrameId("base_link"),
        child_frame=FrameId("front_camera_optical"),
        translation=(0.1, 0.0, 0.2),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    calibration_set = CalibrationSet(
        entries={CalibrationReferenceId("front_camera-calib"): _pinhole_entry()},
        static_transforms=(transform,),
    )

    assert validate_calibration_set(calibration_set) == []
    assert calibration_set.static_transforms[0].parent_frame == "base_link"


def test_round_trip_preserves_camera_models_and_transforms() -> None:
    calibration_set = CalibrationSet(
        entries={
            CalibrationReferenceId("front_camera-calib"): _pinhole_entry(),
            CalibrationReferenceId("fisheye_camera-calib"): _fisheye_entry(),
            CalibrationReferenceId("velodyne_top-calib"): _lidar_entry(),
        },
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("base_link"),
                child_frame=FrameId("front_camera_optical"),
                translation=(0.1, 0.0, 0.2),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )

    decoded = decode_calibration_set(encode_calibration_set(calibration_set))

    assert decoded.entries.keys() == calibration_set.entries.keys()
    assert decoded.entries[CalibrationReferenceId("front_camera-calib")].camera_model == (
        calibration_set.entries[CalibrationReferenceId("front_camera-calib")].camera_model
    )
    assert decoded.entries[CalibrationReferenceId("velodyne_top-calib")].camera_model is None
    assert decoded.static_transforms == calibration_set.static_transforms


def test_rejects_wrong_distortion_coefficient_count() -> None:
    model = PinholeCameraModel(
        width=100,
        height=100,
        fx=1.0,
        fy=1.0,
        cx=50.0,
        cy=50.0,
        distortion_model=DistortionModel.PLUMB_BOB,
        distortion_coefficients=(0.1,),
    )
    entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId("bad-calib"),
        sensor_id=SensorId("cam"),
        frame_id=FrameId("cam_optical"),
        camera_model=model,
        provenance=_PROVENANCE,
        content_hash="sha256:0",
    )
    calibration_set = CalibrationSet(
        entries={CalibrationReferenceId("bad-calib"): entry}, static_transforms=()
    )

    problems = validate_calibration_set(calibration_set)
    assert any("distortion coefficients" in problem for problem in problems)


def test_rejects_non_unit_quaternion_transform() -> None:
    transform = RigidTransform(
        parent_frame=FrameId("base_link"),
        child_frame=FrameId("front_camera_optical"),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 2.0),
    )
    calibration_set = CalibrationSet(entries={}, static_transforms=(transform,))

    problems = validate_calibration_set(calibration_set)
    assert any("unit quaternion" in problem for problem in problems)


def test_rejects_self_referential_transform() -> None:
    transform = RigidTransform(
        parent_frame=FrameId("base_link"),
        child_frame=FrameId("base_link"),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    calibration_set = CalibrationSet(entries={}, static_transforms=(transform,))

    problems = validate_calibration_set(calibration_set)
    assert any("identical parent_frame and child_frame" in problem for problem in problems)


def test_rejects_duplicate_transform() -> None:
    transform = RigidTransform(
        parent_frame=FrameId("base_link"),
        child_frame=FrameId("front_camera_optical"),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    calibration_set = CalibrationSet(entries={}, static_transforms=(transform, transform))

    problems = validate_calibration_set(calibration_set)
    assert any("duplicate static transform" in problem for problem in problems)


def test_ensure_valid_raises_calibration_error() -> None:
    transform = RigidTransform(
        parent_frame=FrameId("base_link"),
        child_frame=FrameId("base_link"),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    calibration_set = CalibrationSet(entries={}, static_transforms=(transform,))

    with pytest.raises(CalibrationError):
        ensure_valid_calibration_set(calibration_set)


def test_content_hash_changes_when_intrinsics_change() -> None:
    base = compute_content_hash(
        sensor_id=SensorId("cam"),
        frame_id=FrameId("cam_optical"),
        camera_model=PinholeCameraModel(width=1, height=1, fx=1.0, fy=1.0, cx=0.5, cy=0.5),
    )
    changed = compute_content_hash(
        sensor_id=SensorId("cam"),
        frame_id=FrameId("cam_optical"),
        camera_model=PinholeCameraModel(width=1, height=1, fx=2.0, fy=1.0, cx=0.5, cy=0.5),
    )

    assert base != changed
