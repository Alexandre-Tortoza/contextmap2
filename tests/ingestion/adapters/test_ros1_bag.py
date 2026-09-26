from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationError,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    ExternalPoseMeasurement,
    FrameId,
    ImageObservation,
    ImuObservation,
    InvalidSourceWindowError,
    LidarObservation,
    MissingRequiredTopicError,
    RigidTransform,
    SensorId,
    SourceAdapterConfig,
    SourceTopicMapping,
    SourceWindow,
    SynchronizationConfig,
    synchronize,
)
from contextmap.ingestion.adapters import _ros_common
from contextmap.ingestion.adapters.ros1_bag import Ros1BagSourceAdapter
from contextmap.ingestion.calibration import (
    FisheyeCameraModel,
    PinholeCameraModel,
    compute_content_hash,
)

_TOPICS = SourceTopicMapping(
    rgb="/camera/image_raw",
    camera_info="/camera/camera_info",
    lidar="/velodyne_points",
    imu="/imu/data",
    pose="/odom",
)

_TS = get_typestore(Stores.ROS1_NOETIC)


def _time(seconds: int, nanoseconds: int) -> Any:
    return _TS.types["builtin_interfaces/msg/Time"](sec=seconds, nanosec=nanoseconds)


def _header(seconds: int, frame_id: str) -> Any:
    return _TS.types["std_msgs/msg/Header"](seq=0, stamp=_time(seconds, 0), frame_id=frame_id)


def _write(connection: Any, writer: Writer, msg: Any, bag_timestamp_ns: int) -> None:
    raw = _TS.serialize_ros1(msg, connection.msgtype)
    writer.write(connection, bag_timestamp_ns, raw)


def _build_bag(
    path: Path,
    *,
    distortion_model: str = "plumb_bob",
    imu_orientation_available: bool = True,
    include_bad_image: bool = False,
    include_changed_camera_info: bool = False,
    include_unsupported_point_field: bool = False,
) -> None:
    types = _TS.types
    with Writer(path) as writer:
        image_conn = writer.add_connection(
            "/camera/image_raw", types["sensor_msgs/msg/Image"].__msgtype__, typestore=_TS
        )
        cam_info_conn = writer.add_connection(
            "/camera/camera_info", types["sensor_msgs/msg/CameraInfo"].__msgtype__, typestore=_TS
        )
        lidar_conn = writer.add_connection(
            "/velodyne_points", types["sensor_msgs/msg/PointCloud2"].__msgtype__, typestore=_TS
        )
        imu_conn = writer.add_connection(
            "/imu/data", types["sensor_msgs/msg/Imu"].__msgtype__, typestore=_TS
        )
        odom_conn = writer.add_connection(
            "/odom", types["nav_msgs/msg/Odometry"].__msgtype__, typestore=_TS
        )

        image_msg = types["sensor_msgs/msg/Image"](
            header=_header(1, "front_camera_optical"),
            height=1,
            width=2,
            encoding="rgb8",
            is_bigendian=0,
            step=6,
            data=np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8),
        )
        _write(image_conn, writer, image_msg, 1_000_000_000)

        if include_bad_image:
            bad_image_msg = types["sensor_msgs/msg/Image"](
                header=_header(2, "front_camera_optical"),
                height=1,
                width=1,
                encoding="not_a_real_encoding",
                is_bigendian=0,
                step=1,
                data=np.array([9], dtype=np.uint8),
            )
            _write(image_conn, writer, bad_image_msg, 2_000_000_000)

        distortion_coefficients = (
            np.array([0.01, 0.002, 0.0003, 0.00004], dtype=np.float64)
            if distortion_model == "equidistant"
            else np.array([0.1, -0.05, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        camera_info_msg: Any = types["sensor_msgs/msg/CameraInfo"](
            header=_header(1, "front_camera_optical"),
            height=720,
            width=1280,
            distortion_model=distortion_model,
            D=distortion_coefficients,
            K=np.array([600.0, 0.0, 640.0, 0.0, 600.0, 360.0, 0.0, 0.0, 1.0], dtype=np.float64),
            R=np.zeros(9, dtype=np.float64),
            P=np.zeros(12, dtype=np.float64),
            binning_x=0,
            binning_y=0,
            roi=types["sensor_msgs/msg/RegionOfInterest"](
                x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
            ),
        )
        _write(cam_info_conn, writer, camera_info_msg, 1_000_000_000)
        if include_changed_camera_info:
            camera_info_msg.K[0] = 601.0
            camera_info_msg.header = _header(2, "front_camera_optical")
            _write(cam_info_conn, writer, camera_info_msg, 2_000_000_000)

        point_field = types["sensor_msgs/msg/PointField"]
        pointcloud_msg = types["sensor_msgs/msg/PointCloud2"](
            header=_header(1, "velodyne"),
            height=1,
            width=1,
            fields=[
                point_field(name="x", offset=0, datatype=7, count=1),
                point_field(name="y", offset=4, datatype=7, count=1),
                point_field(name="z", offset=8, datatype=7, count=1),
            ],
            is_bigendian=False,
            point_step=12,
            row_step=12,
            data=np.zeros(12, dtype=np.uint8),
            is_dense=True,
        )
        _write(lidar_conn, writer, pointcloud_msg, 1_000_000_000)

        if include_unsupported_point_field:
            unsupported_pointcloud_msg = types["sensor_msgs/msg/PointCloud2"](
                header=_header(2, "velodyne"),
                height=1,
                width=1,
                fields=[
                    point_field(name="x", offset=0, datatype=7, count=1),
                    point_field(name="intensity", offset=4, datatype=9, count=1),
                ],
                is_bigendian=False,
                point_step=8,
                row_step=8,
                data=np.zeros(8, dtype=np.uint8),
                is_dense=True,
            )
            _write(lidar_conn, writer, unsupported_pointcloud_msg, 2_000_000_000)

        orientation_covariance: Any = np.zeros(9, dtype=np.float64)
        if not imu_orientation_available:
            orientation_covariance[0] = -1.0
        imu_msg = types["sensor_msgs/msg/Imu"](
            header=_header(1, "imu_link"),
            orientation=types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=orientation_covariance,
            angular_velocity=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.01),
            angular_velocity_covariance=np.zeros(9, dtype=np.float64),
            linear_acceleration=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=9.81),
            linear_acceleration_covariance=np.zeros(9, dtype=np.float64),
        )
        _write(imu_conn, writer, imu_msg, 1_000_000_000)

        odom_msg = types["nav_msgs/msg/Odometry"](
            header=_header(1, "odom"),
            child_frame_id="base_link",
            pose=types["geometry_msgs/msg/PoseWithCovariance"](
                pose=types["geometry_msgs/msg/Pose"](
                    position=types["geometry_msgs/msg/Point"](x=1.0, y=2.0, z=0.0),
                    orientation=types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                covariance=np.zeros(36, dtype=np.float64),
            ),
            twist=types["geometry_msgs/msg/TwistWithCovariance"](
                twist=types["geometry_msgs/msg/Twist"](
                    linear=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0),
                    angular=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0),
                ),
                covariance=np.zeros(36, dtype=np.float64),
            ),
        )
        _write(odom_conn, writer, odom_msg, 1_000_000_000)


@pytest.fixture
def bag_path(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.bag"
    _build_bag(path)
    return path


def test_capabilities_reflect_topics_present_in_the_bag(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    capabilities = adapter.capabilities()

    assert capabilities.rgb is True
    assert capabilities.lidar is True
    assert capabilities.imu is True
    assert capabilities.external_pose is True
    assert capabilities.calibration is True


def test_reads_one_observation_per_modality(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    observations = list(adapter.read_observations())
    by_type = {type(obs).__name__ for obs in observations}

    assert by_type == {
        "ImageObservation",
        "LidarObservation",
        "ImuObservation",
        "ExternalPoseMeasurement",
    }
    assert adapter.warnings() == ()


def test_image_observation_is_decoded_without_ros_types_leaking(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    image = next(obs for obs in adapter.read_observations() if isinstance(obs, ImageObservation))

    assert image.width == 2
    assert image.height == 1
    assert image.data == b"\x01\x02\x03\x04\x05\x06"
    assert image.frame_id == "front_camera_optical"
    assert image.timestamp.clock_id == config.resolved_timestamp_clock_id()
    assert image.timestamp.seconds == 1
    assert image.provenance.source_topic == "/camera/image_raw"
    assert image.provenance.source_message_index == 0
    assert image.provenance.raw_metadata["bag_timestamp_nanoseconds"] == 1_000_000_000
    # No rosbags/ROS-native object anywhere in the canonical observation.
    for value in (image.data, image.width, image.height, image.encoding):
        assert not type(value).__module__.startswith("rosbags")


def test_lidar_observation_preserves_field_layout(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    lidar = next(obs for obs in adapter.read_observations() if isinstance(obs, LidarObservation))

    assert [field.name for field in lidar.fields] == ["x", "y", "z"]
    assert lidar.point_step_bytes == 12
    assert lidar.data == bytes(12)


def test_imu_orientation_absent_when_covariance_marks_it_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "no_orientation.bag"
    _build_bag(path, imu_orientation_available=False)
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    imu = next(obs for obs in adapter.read_observations() if isinstance(obs, ImuObservation))

    assert imu.orientation is None
    assert imu.linear_acceleration == (0.0, 0.0, 9.81)


def test_external_pose_uses_header_and_child_frame_id_convention(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    pose = next(
        obs for obs in adapter.read_observations() if isinstance(obs, ExternalPoseMeasurement)
    )

    assert pose.parent_frame == "odom"
    assert pose.frame_id == "base_link"
    assert pose.translation == (1.0, 2.0, 0.0)


def test_read_calibration_decodes_pinhole_model(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    calibration = adapter.read_calibration()

    assert calibration is not None
    (entry,) = calibration.entries.values()
    assert isinstance(entry.camera_model, PinholeCameraModel)
    assert entry.camera_model.fx == 600.0
    assert entry.camera_model.cx == 640.0


def test_read_calibration_decodes_fisheye_model_without_lossy_conversion(tmp_path: Path) -> None:
    path = tmp_path / "fisheye.bag"
    _build_bag(path, distortion_model="equidistant")
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    calibration = adapter.read_calibration()

    assert calibration is not None
    (entry,) = calibration.entries.values()
    assert isinstance(entry.camera_model, FisheyeCameraModel)
    assert entry.camera_model.distortion_coefficients == (0.01, 0.002, 0.0003, 0.00004)


def test_read_calibration_rejects_intrinsics_that_change_during_sequence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "changing-calibration.bag"
    _build_bag(path, include_changed_camera_info=True)
    adapter = Ros1BagSourceAdapter(
        SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_TOPICS)
    )

    with pytest.raises(CalibrationError, match="conflicting calibration"):
        adapter.read_calibration()


def test_read_calibration_scans_the_bag_only_once_per_adapter_instance(
    bag_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regressão do #506: calibração é lida uma única vez, não uma por chamada.

    ``read_observations()`` já chama ``read_calibration()`` internamente
    (para resolver ``calibration_id`` de cada observação); o runtime
    (``ingestion_service._read``) chama ``read_calibration()`` de novo
    depois de exaurir as observações. Sem cache, isso decodifica o mesmo
    ``camera_info`` duas vezes — um scan completo do bag cada vez.
    """
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)
    calls: list[Any] = []
    real_entry = Ros1BagSourceAdapter._camera_calibration_entry

    def counting_entry(self: Ros1BagSourceAdapter, message: Any, topic: str) -> CalibrationEntry:
        calls.append(message)
        return real_entry(self, message, topic)

    monkeypatch.setattr(Ros1BagSourceAdapter, "_camera_calibration_entry", counting_entry)

    list(adapter.read_observations())  # calls read_calibration() internally
    second_call_result = adapter.read_calibration()  # mirrors the runtime's second call

    assert len(calls) == 1, (
        f"camera_info must be decoded once per adapter instance, not {len(calls)} times"
    )
    assert second_call_result is not None


def test_missing_required_topic_raises_before_any_observation(bag_path: Path) -> None:
    config = SourceAdapterConfig(
        source_type="ros1_bag",
        path=str(bag_path),
        topics=SourceTopicMapping(rgb="/camera/image_raw", lidar="/does_not_exist"),
        required_topics=frozenset({"lidar"}),
    )
    adapter = Ros1BagSourceAdapter(config)

    with pytest.raises(MissingRequiredTopicError):
        list(adapter.read_observations())


def test_unsupported_encoding_becomes_a_warning_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "with_bad_image.bag"
    _build_bag(path, include_bad_image=True)
    config = SourceAdapterConfig(
        source_type="ros1_bag", path=str(path), topics=SourceTopicMapping(rgb="/camera/image_raw")
    )
    adapter = Ros1BagSourceAdapter(config)

    observations = list(adapter.read_observations())

    assert len(observations) == 1
    assert len(adapter.warnings()) == 1
    assert "encoding" in adapter.warnings()[0].reason


def test_unsupported_point_field_datatype_becomes_a_readable_warning(tmp_path: Path) -> None:
    path = tmp_path / "with_unsupported_point_field.bag"
    _build_bag(path, include_unsupported_point_field=True)
    config = SourceAdapterConfig(
        source_type="ros1_bag", path=str(path), topics=SourceTopicMapping(lidar="/velodyne_points")
    )
    adapter = Ros1BagSourceAdapter(config)

    observations = list(adapter.read_observations())

    assert len(observations) == 1
    assert len(adapter.warnings()) == 1
    warning = adapter.warnings()[0]
    assert warning.topic == "/velodyne_points"
    assert warning.message_index == 1
    assert "'intensity'" in warning.reason
    assert "datatype 9" in warning.reason


@pytest.mark.parametrize(
    "structural_error",
    [
        AttributeError("'PointCloud2' object has no attribute 'fields'"),
        KeyError("fields"),
    ],
    ids=["attribute_error", "key_error"],
)
def test_structural_decode_error_propagates_instead_of_becoming_a_warning(
    bag_path: Path, monkeypatch: pytest.MonkeyPatch, structural_error: Exception
) -> None:
    def raise_structural_error(*args: Any, **kwargs: Any) -> Any:
        raise structural_error

    monkeypatch.setattr(_ros_common, "decode_lidar", raise_structural_error)
    config = SourceAdapterConfig(
        source_type="ros1_bag",
        path=str(bag_path),
        topics=SourceTopicMapping(lidar="/velodyne_points"),
    )
    adapter = Ros1BagSourceAdapter(config)

    with pytest.raises(type(structural_error)) as excinfo:
        list(adapter.read_observations())

    assert excinfo.value is structural_error
    assert adapter.warnings() == ()


def test_same_adapter_processes_a_different_bag_via_configuration_only(tmp_path: Path) -> None:
    first_path = tmp_path / "a.bag"
    second_path = tmp_path / "b.bag"
    _build_bag(first_path)
    _build_bag(second_path, distortion_model="equidistant")

    first = Ros1BagSourceAdapter(
        SourceAdapterConfig(source_type="ros1_bag", path=str(first_path), topics=_TOPICS)
    )
    second = Ros1BagSourceAdapter(
        SourceAdapterConfig(source_type="ros1_bag", path=str(second_path), topics=_TOPICS)
    )

    assert len(list(first.read_observations())) == len(list(second.read_observations()))
    first_calibration = first.read_calibration()
    second_calibration = second.read_calibration()
    assert first_calibration is not None
    assert second_calibration is not None


def test_adapter_output_synchronizes_modalities_on_shared_header_clock(bag_path: Path) -> None:
    config = SourceAdapterConfig(
        source_type="ros1_bag",
        path=str(bag_path),
        topics=_TOPICS,
        timestamp_clock_id="robot-header-clock",
    )
    observations = list(Ros1BagSourceAdapter(config).read_observations())

    groups, diagnostics = synchronize(
        observations,
        config=SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=0),
    )

    assert len(groups) == 1
    assert all(
        association.observation is not None for association in groups[0].associations.values()
    )
    assert diagnostics.dropped_events == ()


def test_provided_calibration_supplies_extrinsics_and_observation_reference(
    bag_path: Path,
) -> None:
    lidar_sensor_id = SensorId("velodyne_points")
    lidar_frame_id = FrameId("velodyne")
    lidar_calibration_id = CalibrationReferenceId("lidar-calibration")
    lidar_entry = CalibrationEntry(
        calibration_id=lidar_calibration_id,
        sensor_id=lidar_sensor_id,
        frame_id=lidar_frame_id,
        camera_model=None,
        provenance=CalibrationProvenance(
            source_type="file", source_path="fixtures/extrinsics.yaml"
        ),
        content_hash=compute_content_hash(
            sensor_id=lidar_sensor_id,
            frame_id=lidar_frame_id,
            camera_model=None,
        ),
    )
    transform = RigidTransform(
        parent_frame=FrameId("front_camera_optical"),
        child_frame=lidar_frame_id,
        translation=(0.1, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    config = SourceAdapterConfig(
        source_type="ros1_bag",
        path=str(bag_path),
        topics=_TOPICS,
        calibration=CalibrationSet(
            entries={lidar_calibration_id: lidar_entry},
            static_transforms=(transform,),
        ),
    )
    adapter = Ros1BagSourceAdapter(config)

    calibration = adapter.read_calibration()
    observations = list(adapter.read_observations())
    image = next(item for item in observations if isinstance(item, ImageObservation))
    lidar = next(item for item in observations if isinstance(item, LidarObservation))

    assert calibration is not None
    assert len(calibration.entries) == 2
    assert calibration.static_transforms == (transform,)
    assert image.calibration_id is not None
    assert lidar.calibration_id == lidar_calibration_id


_WINDOW_TOPICS = SourceTopicMapping(rgb="/camera/image_raw")


def _build_windowed_bag(path: Path, *, image_bag_timestamps_ns: list[int]) -> None:
    """N image messages at distinct bag-recording timestamps.

    Header stamps are assigned from a deliberately unrelated numbering
    (seconds 1000, 1001, ...) so a header-time-based filter would select a
    different subset than the bag-recording-time-based filter the window
    actually uses -- proving which clock domain the window operates on.
    """
    types = _TS.types
    with Writer(path) as writer:
        image_conn = writer.add_connection(
            "/camera/image_raw", types["sensor_msgs/msg/Image"].__msgtype__, typestore=_TS
        )
        for index, bag_ts in enumerate(image_bag_timestamps_ns):
            msg = types["sensor_msgs/msg/Image"](
                header=_header(1000 + index, "front_camera_optical"),
                height=1,
                width=1,
                encoding="mono8",
                is_bigendian=0,
                step=1,
                data=np.array([index], dtype=np.uint8),
            )
            _write(image_conn, writer, msg, bag_ts)


def test_window_selects_only_messages_within_the_recording_time_range(tmp_path: Path) -> None:
    path = tmp_path / "windowed.bag"
    _build_windowed_bag(
        path,
        image_bag_timestamps_ns=[
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
            4_000_000_000,
        ],
    )
    base_config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_WINDOW_TOPICS)
    window = SourceWindow(
        clock_id=base_config.resolved_window_clock_id(), start_seconds=2.0, end_seconds=4.0
    )
    adapter = Ros1BagSourceAdapter(replace(base_config, window=window))

    observations = list(adapter.read_observations())

    assert [obs.provenance.raw_metadata["bag_timestamp_nanoseconds"] for obs in observations] == [
        2_000_000_000,
        3_000_000_000,
    ]


def test_observation_id_counter_restarts_at_zero_relative_to_the_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "windowed.bag"
    _build_windowed_bag(path, image_bag_timestamps_ns=[1_000_000_000, 2_000_000_000, 3_000_000_000])
    base_config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_WINDOW_TOPICS)
    window = SourceWindow(
        clock_id=base_config.resolved_window_clock_id(), start_seconds=2.0, end_seconds=4.0
    )
    adapter = Ros1BagSourceAdapter(replace(base_config, window=window))

    observations = list(adapter.read_observations())

    assert [str(obs.observation_id) for obs in observations] == [
        "camera_image_raw-000000",
        "camera_image_raw-000001",
    ]


def test_windowed_and_full_ingestion_agree_on_content_for_a_prefix_window(
    tmp_path: Path,
) -> None:
    """Contract test: a window covering the source's own start never changes content/ids."""
    path = tmp_path / "windowed.bag"
    _build_windowed_bag(
        path,
        image_bag_timestamps_ns=[
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
            4_000_000_000,
        ],
    )
    base_config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_WINDOW_TOPICS)

    full_adapter = Ros1BagSourceAdapter(base_config)
    full_observations = list(full_adapter.read_observations())[:2]

    window = SourceWindow(
        clock_id=base_config.resolved_window_clock_id(), start_seconds=0.5, end_seconds=2.5
    )
    windowed_adapter = Ros1BagSourceAdapter(replace(base_config, window=window))
    windowed_observations = list(windowed_adapter.read_observations())

    assert windowed_observations == full_observations


def test_window_clock_id_mismatch_is_rejected_before_reading_anything(
    tmp_path: Path,
) -> None:
    path = tmp_path / "windowed.bag"
    _build_windowed_bag(path, image_bag_timestamps_ns=[1_000_000_000])
    base_config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_WINDOW_TOPICS)
    window = SourceWindow(clock_id="some-other-clock", start_seconds=0.0, end_seconds=5.0)
    adapter = Ros1BagSourceAdapter(replace(base_config, window=window))

    with pytest.raises(InvalidSourceWindowError, match="clock"):
        list(adapter.read_observations())


def test_window_with_no_overlap_is_rejected_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "windowed.bag"
    _build_windowed_bag(path, image_bag_timestamps_ns=[1_000_000_000, 2_000_000_000])
    base_config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_WINDOW_TOPICS)
    window = SourceWindow(
        clock_id=base_config.resolved_window_clock_id(), start_seconds=10.0, end_seconds=20.0
    )
    adapter = Ros1BagSourceAdapter(replace(base_config, window=window))

    with pytest.raises(InvalidSourceWindowError, match="does not overlap"):
        list(adapter.read_observations())


def test_content_hash_covers_only_the_window_not_the_whole_source(tmp_path: Path) -> None:
    path = tmp_path / "windowed.bag"
    _build_windowed_bag(
        path,
        image_bag_timestamps_ns=[
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
            4_000_000_000,
        ],
    )
    base_config = SourceAdapterConfig(source_type="ros1_bag", path=str(path), topics=_WINDOW_TOPICS)

    full_adapter = Ros1BagSourceAdapter(base_config)
    assert full_adapter.content_hash() is None
    list(full_adapter.read_observations())
    full_hash = full_adapter.content_hash()

    window = SourceWindow(
        clock_id=base_config.resolved_window_clock_id(), start_seconds=1.0, end_seconds=3.0
    )
    windowed_config = replace(base_config, window=window)
    windowed_adapter = Ros1BagSourceAdapter(windowed_config)
    list(windowed_adapter.read_observations())
    windowed_hash = windowed_adapter.content_hash()

    assert windowed_hash is not None
    assert windowed_hash != full_hash

    repeat_adapter = Ros1BagSourceAdapter(windowed_config)
    list(repeat_adapter.read_observations())
    assert repeat_adapter.content_hash() == windowed_hash


def test_no_window_configured_behaves_exactly_as_before(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros1_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros1BagSourceAdapter(config)

    observations = list(adapter.read_observations())

    assert len(observations) == 4
    assert adapter.content_hash() is not None
