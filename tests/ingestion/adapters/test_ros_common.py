import random
from types import SimpleNamespace

import numpy as np
import pytest

from contextmap.ingestion import (
    FrameId,
    ImageEncoding,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.adapters import _ros_common
from contextmap.shared import SourceTimestamp

_OBSERVATION_ID = SourceObservationId("observation-1")
_SENSOR_ID = SensorId("sensor-1")
_TIMESTAMP = SourceTimestamp(seconds=1, nanoseconds=0, clock_id="ros-clock")
_PROVENANCE = SourceProvenance(source_type="ros1_bag", source_path="fixture.bag")


def _header() -> SimpleNamespace:
    return SimpleNamespace(frame_id="sensor-frame")


def test_decode_image_removes_row_padding() -> None:
    message = SimpleNamespace(
        header=_header(),
        encoding="mono8",
        width=2,
        height=2,
        step=3,
        is_bigendian=0,
        data=np.array([1, 2, 99, 3, 4, 99], dtype=np.uint8),
    )

    observation = _ros_common.decode_image(
        message,
        observation_id=_OBSERVATION_ID,
        sensor_id=_SENSOR_ID,
        timestamp=_TIMESTAMP,
        provenance=_PROVENANCE,
    )

    assert observation.encoding is ImageEncoding.MONO8
    assert observation.data == b"\x01\x02\x03\x04"
    assert observation.provenance.raw_metadata["source_step_bytes"] == 3
    assert observation.provenance.raw_metadata["row_padding_removed"] is True


def test_decode_image_normalizes_big_endian_mono16_to_little_endian() -> None:
    message = SimpleNamespace(
        header=_header(),
        encoding="mono16",
        width=2,
        height=1,
        step=4,
        is_bigendian=1,
        data=np.array([1, 2, 3, 4], dtype=np.uint8),
    )

    observation = _ros_common.decode_image(
        message,
        observation_id=_OBSERVATION_ID,
        sensor_id=_SENSOR_ID,
        timestamp=_TIMESTAMP,
        provenance=_PROVENANCE,
    )

    assert observation.data == b"\x02\x01\x04\x03"
    assert observation.provenance.raw_metadata["source_is_bigendian"] is True
    assert observation.provenance.raw_metadata["byte_order_normalized"] is True


def test_decode_image_rejects_stride_smaller_than_row_payload() -> None:
    message = SimpleNamespace(
        header=_header(),
        encoding="rgb8",
        width=2,
        height=1,
        step=5,
        is_bigendian=0,
        data=np.zeros(6, dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="step"):
        _ros_common.decode_image(
            message,
            observation_id=_OBSERVATION_ID,
            sensor_id=_SENSOR_ID,
            timestamp=_TIMESTAMP,
            provenance=_PROVENANCE,
        )


def test_decode_lidar_removes_row_padding_and_normalizes_field_byte_order() -> None:
    field = SimpleNamespace(name="range", offset=0, datatype=4, count=1)
    message = SimpleNamespace(
        header=_header(),
        width=1,
        height=2,
        point_step=4,
        row_step=6,
        fields=[field],
        is_bigendian=True,
        is_dense=True,
        data=np.array([1, 2, 0, 0, 99, 99, 3, 4, 0, 0, 99, 99], dtype=np.uint8),
    )

    observation = _ros_common.decode_lidar(
        message,
        observation_id=_OBSERVATION_ID,
        sensor_id=_SENSOR_ID,
        timestamp=_TIMESTAMP,
        provenance=_PROVENANCE,
    )

    assert observation.data == b"\x02\x01\x00\x00\x04\x03\x00\x00"
    assert observation.provenance.raw_metadata["source_row_step_bytes"] == 6
    assert observation.provenance.raw_metadata["byte_order_normalized"] is True


def test_decode_lidar_rejects_unknown_point_field_datatype_with_semantic_error() -> None:
    message = SimpleNamespace(
        header=_header(),
        width=1,
        height=1,
        point_step=8,
        row_step=8,
        fields=[
            SimpleNamespace(name="x", offset=0, datatype=7, count=1),
            SimpleNamespace(name="intensity", offset=4, datatype=9, count=1),
        ],
        is_bigendian=False,
        is_dense=True,
        data=np.zeros(8, dtype=np.uint8),
    )

    with pytest.raises(ValueError) as excinfo:
        _ros_common.decode_lidar(
            message,
            observation_id=_OBSERVATION_ID,
            sensor_id=_SENSOR_ID,
            timestamp=_TIMESTAMP,
            provenance=_PROVENANCE,
        )

    reason = str(excinfo.value)
    assert "'intensity'" in reason
    assert "datatype 9" in reason
    assert "1-8" in reason


def test_decode_imu_preserves_each_available_covariance() -> None:
    vector = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    quaternion = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    message = SimpleNamespace(
        header=_header(),
        linear_acceleration=vector,
        linear_acceleration_covariance=np.arange(9, dtype=np.float64),
        angular_velocity=vector,
        angular_velocity_covariance=np.arange(9, dtype=np.float64) + 10,
        orientation=quaternion,
        orientation_covariance=np.arange(9, dtype=np.float64) + 20,
    )

    observation = _ros_common.decode_imu(
        message,
        observation_id=_OBSERVATION_ID,
        sensor_id=_SENSOR_ID,
        timestamp=_TIMESTAMP,
        provenance=_PROVENANCE,
    )

    assert observation.linear_acceleration_covariance == tuple(float(i) for i in range(9))
    assert observation.angular_velocity_covariance == tuple(float(i) for i in range(10, 19))
    assert observation.orientation_covariance == tuple(float(i) for i in range(20, 29))


def test_decode_pose_preserves_twist_and_covariances() -> None:
    vector = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    quaternion = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id="odom"),
        child_frame_id="base_link",
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=vector, orientation=quaternion),
            covariance=np.arange(36, dtype=np.float64),
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(linear=vector, angular=vector),
            covariance=np.arange(36, dtype=np.float64) + 100,
        ),
    )

    observation = _ros_common.decode_pose(
        message,
        observation_id=_OBSERVATION_ID,
        sensor_id=_SENSOR_ID,
        timestamp=_TIMESTAMP,
        provenance=_PROVENANCE,
    )

    assert observation.frame_id == FrameId("base_link")
    assert observation.pose_covariance == tuple(float(i) for i in range(36))
    assert observation.linear_velocity == (1.0, 2.0, 3.0)
    assert observation.twist_covariance == tuple(float(i) for i in range(100, 136))


def test_remove_row_padding_returns_an_unpadded_payload_without_copying() -> None:
    data = bytes(range(6))

    packed = _ros_common._remove_row_padding(
        data, height=2, row_step=3, row_payload_size=3, payload_name="image"
    )

    assert packed is data


def _reference_point_field_byte_order(
    data: bytes, *, point_step: int, fields: tuple[PointFieldDescriptor, ...]
) -> bytes:
    """Laço escalar original, ponto a ponto e elemento a elemento: a referência de equivalência."""
    normalized = bytearray(data)
    for field in fields:
        value_size = _ros_common.POINTFIELD_SIZE_BYTES[field.data_type]
        if value_size == 1:
            continue
        for point_offset in range(0, len(normalized), point_step):
            for element_index in range(field.count):
                start = point_offset + field.offset_bytes + element_index * value_size
                end = start + value_size
                normalized[start:end] = normalized[start:end][::-1]
    return bytes(normalized)


@pytest.mark.parametrize("seed", range(20))
def test_point_field_byte_order_matches_the_scalar_reference_byte_for_byte(seed: int) -> None:
    rng = random.Random(seed)
    data_types = list(_ros_common.POINTFIELD_SIZE_BYTES)
    fields: list[PointFieldDescriptor] = []
    offset = rng.randrange(0, 3)
    for index in range(rng.randrange(1, 6)):
        data_type = rng.choice(data_types)
        count = rng.randrange(1, 4)
        fields.append(
            PointFieldDescriptor(
                name=f"field-{index}", offset_bytes=offset, data_type=data_type, count=count
            )
        )
        offset += _ros_common.POINTFIELD_SIZE_BYTES[data_type] * count + rng.randrange(0, 3)
    point_step = offset + rng.randrange(0, 4)
    point_count = rng.randrange(1, 40)
    data = bytes(rng.randrange(256) for _ in range(point_step * point_count))

    normalized = _ros_common._normalize_point_field_byte_order(
        data, point_step=point_step, fields=tuple(fields)
    )

    assert normalized == _reference_point_field_byte_order(
        data, point_step=point_step, fields=tuple(fields)
    )


def test_point_field_byte_order_swaps_a_field_that_fills_the_whole_record() -> None:
    # Campo contíguo ao registro inteiro: a view de origem e o destino são a mesma memória.
    field = PointFieldDescriptor(
        name="xyz", offset_bytes=0, data_type=PointFieldDataType.UINT16, count=3
    )
    data = bytes(range(12))

    normalized = _ros_common._normalize_point_field_byte_order(data, point_step=6, fields=(field,))

    assert normalized == bytes([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])


def test_point_field_byte_order_still_rejects_a_field_beyond_point_step() -> None:
    field = PointFieldDescriptor(name="range", offset_bytes=4, data_type=PointFieldDataType.FLOAT64)

    with pytest.raises(ValueError, match="exceeds point_step=8"):
        _ros_common._normalize_point_field_byte_order(bytes(16), point_step=8, fields=(field,))


_K_MATRIX = (600.0, 0.0, 640.0, 0.0, 600.0, 360.0, 0.0, 0.0, 1.0)


def test_build_camera_model_reports_no_conversion_for_four_equidistant_coefficients() -> None:
    model, conversions = _ros_common.build_camera_model(
        width=1280,
        height=720,
        k_matrix=_K_MATRIX,
        distortion_model_name="equidistant",
        distortion_coefficients=(0.1, 0.2, 0.3, 0.4),
    )

    assert model.distortion_coefficients == (0.1, 0.2, 0.3, 0.4)
    assert conversions == ()


@pytest.mark.parametrize(
    ("coefficients", "expected", "note"),
    [
        ((0.1, 0.2, 0.3, 0.4, 0.5), (0.1, 0.2, 0.3, 0.4), "dropped 1 extra coefficient(s) [0.5]"),
        ((0.1, 0.2), (0.1, 0.2, 0.0, 0.0), "zero-filled the missing 2"),
    ],
    ids=["truncated", "zero_filled"],
)
def test_build_camera_model_records_how_equidistant_coefficients_were_normalized(
    coefficients: tuple[float, ...], expected: tuple[float, ...], note: str
) -> None:
    model, conversions = _ros_common.build_camera_model(
        width=1280,
        height=720,
        k_matrix=_K_MATRIX,
        distortion_model_name="equidistant",
        distortion_coefficients=coefficients,
    )

    assert model.distortion_coefficients == expected
    (conversion,) = conversions
    assert f"got {len(coefficients)}" in conversion
    assert note in conversion
