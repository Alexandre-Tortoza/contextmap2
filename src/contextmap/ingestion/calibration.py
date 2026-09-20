"""Canonical calibration and coordinate-frame metadata contract.

Ingestion normalizes and owns calibration data (camera intrinsics/model,
static sensor extrinsics, frame identity). Applying calibration to perform
2D/3D projection is Sensor Association's responsibility, not this module's;
see ``docs/shared-primitives.md``. See
``src/contextmap/ingestion/docs/calibration.md`` for coordinate-frame
conventions (transform direction, quaternion order, handedness) and worked
examples.

Pinhole, fisheye and unified omnidirectional (MEI) cameras are represented
by distinct, non-lossy types (:class:`PinholeCameraModel`,
:class:`FisheyeCameraModel`, :class:`MeiCameraModel`) instead of forcing one
distortion model into another's record. Dynamic transforms
(a moving robot's pose over time) never belong here: only *static* rigid
transforms between sensor/body frames are calibration; a moving frame's
pose over time is an :class:`~contextmap.ingestion.models.ExternalPoseMeasurement`
in the temporal observation stream.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contextmap.ingestion.models import CalibrationReferenceId, FrameId, SensorId

SCHEMA_VERSION = "0.1.0"
"""Calibration contract schema version."""

_QUATERNION_NORM_TOLERANCE = 1e-3


class DistortionModel(Enum):
    """Lens distortion model for :class:`PinholeCameraModel`.

    Coefficient order follows the OpenCV convention for each model.
    """

    NONE = "none"
    PLUMB_BOB = "plumb_bob"
    RATIONAL_POLYNOMIAL = "rational_polynomial"


_PLUMB_BOB_COEFFICIENT_COUNTS: Mapping[DistortionModel, int] = {
    DistortionModel.NONE: 0,
    DistortionModel.PLUMB_BOB: 5,
    DistortionModel.RATIONAL_POLYNOMIAL: 8,
}


class CalibrationError(Exception):
    """Raised when a calibration set fails validation."""


@dataclass(frozen=True, kw_only=True)
class PinholeCameraModel:
    """Standard pinhole camera intrinsics, with optional radial/tangential distortion.

    Attributes:
        width: Image width in pixels this calibration applies to.
        height: Image height in pixels this calibration applies to.
        fx: Focal length along x, in pixels.
        fy: Focal length along y, in pixels.
        cx: Principal point x, in pixels.
        cy: Principal point y, in pixels.
        distortion_model: Distortion model used by ``distortion_coefficients``.
        distortion_coefficients: Coefficients in the order OpenCV documents
            for ``distortion_model`` (empty for
            :attr:`DistortionModel.NONE`).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: DistortionModel = DistortionModel.NONE
    distortion_coefficients: tuple[float, ...] = ()


@dataclass(frozen=True, kw_only=True)
class FisheyeCameraModel:
    """Fisheye camera intrinsics using the equidistant (Kannala-Brandt) model.

    Attributes:
        width: Image width in pixels this calibration applies to.
        height: Image height in pixels this calibration applies to.
        fx: Focal length along x, in pixels.
        fy: Focal length along y, in pixels.
        cx: Principal point x, in pixels.
        cy: Principal point y, in pixels.
        distortion_coefficients: Equidistant model coefficients
            ``(k1, k2, k3, k4)``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_coefficients: tuple[float, float, float, float]


@dataclass(frozen=True, kw_only=True)
class MeiCameraModel:
    """Unified omnidirectional camera intrinsics (Geyer-Daniilidis / Mei).

    The model first projects a point through a unit sphere whose center is
    shifted by ``xi`` along the optical axis, then applies radial and
    tangential distortion on the normalized plane, then the generalized
    projection matrix. This is the parameterization used by CamOdoCal and by
    calibrations declared with ``model_type: MEI``.

    Attributes:
        width: Image width in pixels this calibration applies to.
        height: Image height in pixels this calibration applies to.
        fx: Generalized focal length along x (``gamma1``), in pixels.
        fy: Generalized focal length along y (``gamma2``), in pixels.
        cx: Principal point x (``u0``), in pixels.
        cy: Principal point y (``v0``), in pixels.
        xi: Mirror parameter, dimensionless and not negative. ``0`` degenerates
            to a distorted pinhole; values above ``1`` describe a field of view
            wider than a hemisphere.
        distortion_coefficients: ``(k1, k2, p1, p2)``: two radial and two
            tangential coefficients, applied on the normalized plane.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    distortion_coefficients: tuple[float, float, float, float]


CameraModel = PinholeCameraModel | FisheyeCameraModel | MeiCameraModel
"""Any camera model kind; no lossy conversion exists between them."""


def camera_model_kind(model: CameraModel) -> str:
    """Return ``"pinhole"``, ``"fisheye"`` or ``"mei"`` for a camera model.

    Args:
        model: A pinhole, fisheye or unified omnidirectional camera model.

    Returns:
        ``"pinhole"``, ``"fisheye"`` or ``"mei"``.

    Raises:
        TypeError: If ``model`` is not a recognized camera model type.
    """
    if isinstance(model, PinholeCameraModel):
        return "pinhole"
    if isinstance(model, FisheyeCameraModel):
        return "fisheye"
    if isinstance(model, MeiCameraModel):
        return "mei"
    raise TypeError(f"unsupported camera model type: {type(model)!r}")


@dataclass(frozen=True, kw_only=True)
class RigidTransform:
    """A static rigid-body transform between two coordinate frames.

    Follows the ``T_parent_child`` convention: this transform takes
    coordinates expressed in ``child_frame`` into ``parent_frame``, i.e.
    ``p_parent = rotate(rotation, p_child) + translation``. ``rotation`` is
    a unit quaternion ordered ``(x, y, z, w)``.

    A ``RigidTransform`` is only valid for a relationship that does not
    change during the sequence (e.g. a camera rigidly mounted on a robot
    body). A frame whose pose changes over time (e.g. the robot body in the
    world) is not a calibration transform; it is reported per-timestamp as
    an :class:`~contextmap.ingestion.models.ExternalPoseMeasurement`.

    Attributes:
        parent_frame: Frame this transform expresses coordinates in.
        child_frame: Frame whose coordinates are being transformed.
        translation: ``(x, y, z)`` in meters.
        rotation: Unit quaternion ``(x, y, z, w)`` rotating ``child_frame``'s
            axes into ``parent_frame``'s axes.
    """

    parent_frame: FrameId
    child_frame: FrameId
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True, kw_only=True)
class CalibrationProvenance:
    """Traceability metadata for one calibration entry.

    Attributes:
        source_type: Origin of the calibration, e.g. ``"ros1_bag"``,
            ``"ros2_bag"``, ``"dataset"``, ``"manual"``.
        source_path: Path or identity of the calibration source.
        original_values: Source-native calibration values preserved as
            primitives before normalization (e.g. the raw distortion array
            or frame name as given by the source). Empty when the source
            already matched the canonical representation exactly.
        conversions_applied: Human-readable notes of any normalization
            performed, e.g. ``"reordered quaternion from wxyz to xyzw"``.
            An empty sequence asserts no conversion was necessary.
    """

    source_type: str
    source_path: str
    original_values: Mapping[str, object] = field(default_factory=dict)
    conversions_applied: Sequence[str] = ()


@dataclass(frozen=True, kw_only=True)
class CalibrationEntry:
    """One sensor's canonical, inspectable calibration.

    Attributes:
        calibration_id: Identity referenced by
            ``SourceObservation.calibration_id``.
        sensor_id: Sensor this calibration applies to.
        frame_id: Coordinate frame ``camera_model`` (when present) is
            expressed in, e.g. the camera's optical frame.
        camera_model: Intrinsics/distortion, when ``sensor_id`` is a
            camera. ``None`` for non-camera sensors (e.g. LiDAR, which has
            no intrinsic camera model but may still have a
            ``CalibrationEntry`` for its frame identity/provenance).
        provenance: Traceability back to the calibration source.
        content_hash: ``"sha256:<hex digest>"`` of this entry's canonical
            values, for change detection.
    """

    calibration_id: CalibrationReferenceId
    sensor_id: SensorId
    frame_id: FrameId
    camera_model: CameraModel | None
    provenance: CalibrationProvenance
    content_hash: str


@dataclass(frozen=True, kw_only=True)
class CalibrationSet:
    """The full, inspectable calibration and coordinate-frame inventory of a sequence.

    Attributes:
        entries: One :class:`CalibrationEntry` per calibrated sensor, keyed
            by its ``calibration_id``.
        static_transforms: Rigid transforms between sensor/body frames that
            do not change during the sequence. Dynamic transforms belong in
            the temporal observation stream, not here.
        schema_version: Calibration contract schema version.
    """

    entries: Mapping[CalibrationReferenceId, CalibrationEntry]
    static_transforms: Sequence[RigidTransform]
    schema_version: str = SCHEMA_VERSION


def compute_content_hash(
    *,
    sensor_id: SensorId,
    frame_id: FrameId,
    camera_model: CameraModel | None,
) -> str:
    """Compute the deterministic content hash for a calibration entry.

    Only the canonical values a downstream consumer relies on are hashed
    (sensor/frame identity and camera model); provenance/free-text fields
    are intentionally excluded so hash changes track calibration content,
    not bookkeeping metadata.

    Args:
        sensor_id: Sensor the calibration applies to.
        frame_id: Coordinate frame of the camera model, if any.
        camera_model: Camera intrinsics/distortion, or ``None``.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    payload = json.dumps(
        {
            "sensor_id": str(sensor_id),
            "frame_id": str(frame_id),
            "camera_model": _encode_camera_model(camera_model),
        },
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_calibration_set(calibration_set: CalibrationSet) -> list[str]:
    """Check a calibration set for missing, malformed, or inconsistent data.

    Args:
        calibration_set: The calibration set to validate.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    problems: list[str] = []

    for key, entry in calibration_set.entries.items():
        if entry.calibration_id != key:
            problems.append(
                f"entries key {key!r} does not match entry.calibration_id {entry.calibration_id!r}"
            )
        if entry.camera_model is not None:
            problems.extend(_validate_camera_model(entry.calibration_id, entry.camera_model))

    seen_pairs: set[tuple[str, str]] = set()
    for transform in calibration_set.static_transforms:
        pair = (str(transform.parent_frame), str(transform.child_frame))
        if transform.parent_frame == transform.child_frame:
            problems.append(f"transform has identical parent_frame and child_frame: {pair[0]!r}")
        if pair in seen_pairs:
            problems.append(f"duplicate static transform for parent={pair[0]!r} child={pair[1]!r}")
        seen_pairs.add(pair)

        norm = math.sqrt(sum(component**2 for component in transform.rotation))
        if not math.isclose(norm, 1.0, abs_tol=_QUATERNION_NORM_TOLERANCE):
            problems.append(
                f"transform parent={pair[0]!r} child={pair[1]!r} rotation is not a unit "
                f"quaternion (norm={norm:.6f})"
            )

    return problems


def ensure_valid_calibration_set(calibration_set: CalibrationSet) -> None:
    """Validate a calibration set and raise if it has any problem.

    Args:
        calibration_set: The calibration set to validate.

    Raises:
        CalibrationError: If :func:`validate_calibration_set` finds any
            problem.
    """
    problems = validate_calibration_set(calibration_set)
    if problems:
        raise CalibrationError(f"invalid calibration set: {problems}")


def _validate_camera_model(calibration_id: CalibrationReferenceId, model: CameraModel) -> list[str]:
    problems: list[str] = []
    if model.width <= 0 or model.height <= 0:
        problems.append(f"{calibration_id}: width and height must be positive")
    for name in ("fx", "fy", "cx", "cy"):
        if not math.isfinite(getattr(model, name)):
            problems.append(f"{calibration_id}: {name} must be finite")
    if model.fx <= 0 or model.fy <= 0:
        problems.append(f"{calibration_id}: fx and fy must be positive")
    if not all(math.isfinite(value) for value in model.distortion_coefficients):
        problems.append(f"{calibration_id}: distortion coefficients must be finite")

    if isinstance(model, PinholeCameraModel):
        expected = _PLUMB_BOB_COEFFICIENT_COUNTS[model.distortion_model]
        actual = len(model.distortion_coefficients)
        if actual != expected:
            problems.append(
                f"{calibration_id}: {model.distortion_model.value} expects {expected} "
                f"distortion coefficients, got {actual}"
            )
    elif isinstance(model, FisheyeCameraModel) and len(model.distortion_coefficients) != 4:
        problems.append(
            f"{calibration_id}: fisheye model requires exactly 4 distortion coefficients, "
            f"got {len(model.distortion_coefficients)}"
        )
    elif isinstance(model, MeiCameraModel):
        if len(model.distortion_coefficients) != 4:
            problems.append(
                f"{calibration_id}: mei model requires exactly 4 distortion coefficients "
                f"(k1, k2, p1, p2), got {len(model.distortion_coefficients)}"
            )
        if not math.isfinite(model.xi) or model.xi < 0:
            problems.append(
                f"{calibration_id}: mei mirror parameter xi must be finite and not negative, "
                f"got {model.xi!r}"
            )

    return problems


def _encode_camera_model(model: CameraModel | None) -> dict[str, Any] | None:
    if model is None:
        return None
    record: dict[str, Any] = {
        "kind": camera_model_kind(model),
        "width": model.width,
        "height": model.height,
        "fx": model.fx,
        "fy": model.fy,
        "cx": model.cx,
        "cy": model.cy,
        "distortion_coefficients": list(model.distortion_coefficients),
    }
    if isinstance(model, PinholeCameraModel):
        record["distortion_model"] = model.distortion_model.value
    if isinstance(model, MeiCameraModel):
        record["xi"] = model.xi
    return record


def _decode_camera_model(record: dict[str, Any] | None) -> CameraModel | None:
    if record is None:
        return None
    if record["kind"] == "pinhole":
        return PinholeCameraModel(
            width=record["width"],
            height=record["height"],
            fx=record["fx"],
            fy=record["fy"],
            cx=record["cx"],
            cy=record["cy"],
            distortion_model=DistortionModel(record["distortion_model"]),
            distortion_coefficients=tuple(record["distortion_coefficients"]),
        )
    if record["kind"] == "fisheye":
        x, y, z, w = record["distortion_coefficients"]
        return FisheyeCameraModel(
            width=record["width"],
            height=record["height"],
            fx=record["fx"],
            fy=record["fy"],
            cx=record["cx"],
            cy=record["cy"],
            distortion_coefficients=(x, y, z, w),
        )
    if record["kind"] == "mei":
        k1, k2, p1, p2 = record["distortion_coefficients"]
        return MeiCameraModel(
            width=record["width"],
            height=record["height"],
            fx=record["fx"],
            fy=record["fy"],
            cx=record["cx"],
            cy=record["cy"],
            xi=record["xi"],
            distortion_coefficients=(k1, k2, p1, p2),
        )
    raise CalibrationError(f"unknown camera model kind: {record['kind']!r}")


def encode_calibration_set(calibration_set: CalibrationSet) -> dict[str, Any]:
    """Encode a calibration set into a JSON-serializable dict.

    Args:
        calibration_set: The calibration set to encode.

    Returns:
        A dict suitable for ``json.dumps``.
    """
    return {
        "schema_version": calibration_set.schema_version,
        "entries": [
            {
                "calibration_id": str(entry.calibration_id),
                "sensor_id": str(entry.sensor_id),
                "frame_id": str(entry.frame_id),
                "camera_model": _encode_camera_model(entry.camera_model),
                "provenance": {
                    "source_type": entry.provenance.source_type,
                    "source_path": entry.provenance.source_path,
                    "original_values": dict(entry.provenance.original_values),
                    "conversions_applied": list(entry.provenance.conversions_applied),
                },
                "content_hash": entry.content_hash,
            }
            for entry in calibration_set.entries.values()
        ],
        "static_transforms": [
            {
                "parent_frame": str(transform.parent_frame),
                "child_frame": str(transform.child_frame),
                "translation": list(transform.translation),
                "rotation": list(transform.rotation),
            }
            for transform in calibration_set.static_transforms
        ],
    }


def decode_calibration_set(record: dict[str, Any]) -> CalibrationSet:
    """Decode a calibration set from a dict produced by :func:`encode_calibration_set`.

    Args:
        record: A dict as produced by :func:`encode_calibration_set`.

    Returns:
        The decoded calibration set.
    """
    entries = {
        CalibrationReferenceId(item["calibration_id"]): CalibrationEntry(
            calibration_id=CalibrationReferenceId(item["calibration_id"]),
            sensor_id=SensorId(item["sensor_id"]),
            frame_id=FrameId(item["frame_id"]),
            camera_model=_decode_camera_model(item["camera_model"]),
            provenance=CalibrationProvenance(
                source_type=item["provenance"]["source_type"],
                source_path=item["provenance"]["source_path"],
                original_values=dict(item["provenance"]["original_values"]),
                conversions_applied=list(item["provenance"]["conversions_applied"]),
            ),
            content_hash=item["content_hash"],
        )
        for item in record["entries"]
    }
    static_transforms = tuple(
        RigidTransform(
            parent_frame=FrameId(item["parent_frame"]),
            child_frame=FrameId(item["child_frame"]),
            translation=tuple(item["translation"]),
            rotation=tuple(item["rotation"]),
        )
        for item in record["static_transforms"]
    )
    return CalibrationSet(
        entries=entries,
        static_transforms=static_transforms,
        schema_version=record["schema_version"],
    )
