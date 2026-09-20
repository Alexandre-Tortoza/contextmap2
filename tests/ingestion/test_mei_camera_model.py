import dataclasses
import json

import pytest

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    FrameId,
    MeiCameraModel,
    PinholeCameraModel,
    SensorId,
    camera_model_kind,
    validate_calibration_set,
)
from contextmap.ingestion.calibration import (
    compute_content_hash,
    decode_calibration_set,
    encode_calibration_set,
)

_PROVENANCE = CalibrationProvenance(source_type="dataset", source_path="fixtures/omni.yaml")


def _mei() -> MeiCameraModel:
    return MeiCameraModel(
        width=640,
        height=480,
        fx=761.679,
        fy=678.768,
        cx=317.740,
        cy=233.502,
        xi=1.563,
        distortion_coefficients=(-0.0708, 0.3095, 0.00046, 0.000157),
    )


def _calibration_set(model: CameraModel) -> CalibrationSet:
    calibration_id = CalibrationReferenceId("camera-calib")
    entry = CalibrationEntry(
        calibration_id=calibration_id,
        sensor_id=SensorId("camera"),
        frame_id=FrameId("camera_optical"),
        camera_model=model,
        provenance=_PROVENANCE,
        content_hash=compute_content_hash(
            sensor_id=SensorId("camera"), frame_id=FrameId("camera_optical"), camera_model=model
        ),
    )
    return CalibrationSet(entries={calibration_id: entry}, static_transforms=())


def test_the_unified_omnidirectional_model_is_a_distinct_canonical_kind() -> None:
    model = _mei()

    assert camera_model_kind(model) == "mei"
    assert not isinstance(model, PinholeCameraModel | FisheyeCameraModel)
    assert validate_calibration_set(_calibration_set(model)) == []


def test_the_mirror_parameter_and_the_distortion_survive_a_json_round_trip() -> None:
    calibration_set = _calibration_set(_mei())

    record = json.loads(json.dumps(encode_calibration_set(calibration_set)))
    decoded = decode_calibration_set(record)

    entry_record = record["entries"][0]["camera_model"]
    assert entry_record["kind"] == "mei"
    assert entry_record["xi"] == 1.563
    (entry,) = decoded.entries.values()
    assert entry.camera_model == _mei()
    assert entry.content_hash == next(iter(calibration_set.entries.values())).content_hash


def test_the_content_hash_tracks_the_mirror_parameter() -> None:
    def hash_for(model: MeiCameraModel) -> str:
        return compute_content_hash(
            sensor_id=SensorId("camera"), frame_id=FrameId("camera_optical"), camera_model=model
        )

    assert hash_for(_mei()) == hash_for(_mei())
    assert hash_for(_mei()) != hash_for(dataclasses.replace(_mei(), xi=1.2))


@pytest.mark.parametrize("xi", [-0.1, float("nan"), float("inf")])
def test_the_mirror_parameter_must_be_finite_and_not_negative(xi: float) -> None:
    problems = validate_calibration_set(_calibration_set(dataclasses.replace(_mei(), xi=xi)))

    assert any("xi" in problem for problem in problems)


def test_the_unified_model_needs_exactly_four_distortion_coefficients() -> None:
    model = dataclasses.replace(_mei(), distortion_coefficients=(0.1, 0.2, 0.0))  # type: ignore[arg-type]

    problems = validate_calibration_set(_calibration_set(model))

    assert any("mei" in problem and "4" in problem for problem in problems)


@pytest.mark.parametrize("field_name", ["fx", "fy", "cx", "cy"])
def test_intrinsic_parameters_must_be_finite_for_every_model(field_name: str) -> None:
    pinhole = PinholeCameraModel(width=64, height=48, fx=50.0, fy=50.0, cx=32.0, cy=24.0)

    for model in (_mei(), pinhole):
        broken = dataclasses.replace(model, **{field_name: float("nan")})  # type: ignore[arg-type]
        problems = validate_calibration_set(_calibration_set(broken))
        assert any(field_name in problem and "finite" in problem for problem in problems)


def test_distortion_coefficients_must_be_finite_for_every_model() -> None:
    fisheye = FisheyeCameraModel(
        width=64,
        height=48,
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=24.0,
        distortion_coefficients=(0.0, float("inf"), 0.0, 0.0),
    )
    pinhole = PinholeCameraModel(
        width=64,
        height=48,
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=24.0,
        distortion_model=DistortionModel.PLUMB_BOB,
        distortion_coefficients=(0.0, float("nan"), 0.0, 0.0, 0.0),
    )
    mei = dataclasses.replace(_mei(), distortion_coefficients=(0.0, 0.0, float("nan"), 0.0))

    for model in (fisheye, pinhole, mei):
        problems = validate_calibration_set(_calibration_set(model))
        assert any("distortion" in problem and "finite" in problem for problem in problems)
