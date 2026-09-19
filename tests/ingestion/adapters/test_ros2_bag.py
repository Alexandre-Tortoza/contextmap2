from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from contextmap.ingestion import (
    ExternalPoseMeasurement,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    MissingRequiredTopicError,
    SourceAdapterConfig,
    SourceTopicMapping,
    SynchronizationConfig,
    synchronize,
)
from contextmap.ingestion.adapters.ros2_bag import Ros2BagSourceAdapter
from contextmap.ingestion.calibration import FisheyeCameraModel, PinholeCameraModel

_TOPICS = SourceTopicMapping(
    rgb="/camera/color/image",
    camera_info="/camera/color/camera_info",
    lidar="/ouster/points",
    imu="/imu/data",
    pose="/odom",
)

_TS = get_typestore(Stores.ROS2_HUMBLE)


def _time(seconds: int, nanoseconds: int) -> Any:
    return _TS.types["builtin_interfaces/msg/Time"](sec=seconds, nanosec=nanoseconds)


def _header(seconds: int, frame_id: str) -> Any:
    return _TS.types["std_msgs/msg/Header"](stamp=_time(seconds, 0), frame_id=frame_id)


def _write(connection: Any, writer: Writer, msg: Any, bag_timestamp_ns: int) -> None:
    raw = _TS.serialize_cdr(msg, connection.msgtype)
    writer.write(connection, bag_timestamp_ns, raw)


def _build_bag(
    path: Path,
    *,
    distortion_model: str = "plumb_bob",
    imu_orientation_available: bool = True,
    include_bad_image: bool = False,
) -> None:
    types = _TS.types
    with Writer(path, version=9) as writer:
        image_conn = writer.add_connection(
            "/camera/color/image", types["sensor_msgs/msg/Image"].__msgtype__, typestore=_TS
        )
        cam_info_conn = writer.add_connection(
            "/camera/color/camera_info",
            types["sensor_msgs/msg/CameraInfo"].__msgtype__,
            typestore=_TS,
        )
        lidar_conn = writer.add_connection(
            "/ouster/points", types["sensor_msgs/msg/PointCloud2"].__msgtype__, typestore=_TS
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
        camera_info_msg = types["sensor_msgs/msg/CameraInfo"](
            header=_header(1, "front_camera_optical"),
            height=720,
            width=1280,
            distortion_model=distortion_model,
            d=distortion_coefficients,
            k=np.array([600.0, 0.0, 640.0, 0.0, 600.0, 360.0, 0.0, 0.0, 1.0], dtype=np.float64),
            r=np.zeros(9, dtype=np.float64),
            p=np.zeros(12, dtype=np.float64),
            binning_x=0,
            binning_y=0,
            roi=types["sensor_msgs/msg/RegionOfInterest"](
                x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
            ),
        )
        _write(cam_info_conn, writer, camera_info_msg, 1_000_000_000)

        point_field = types["sensor_msgs/msg/PointField"]
        pointcloud_msg = types["sensor_msgs/msg/PointCloud2"](
            header=_header(1, "ouster"),
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
    path = tmp_path / "fixture_bag"
    _build_bag(path)
    return path


def test_capabilities_reflect_topics_present_in_the_bag(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    capabilities = adapter.capabilities()

    assert capabilities.rgb is True
    assert capabilities.lidar is True
    assert capabilities.imu is True
    assert capabilities.external_pose is True
    assert capabilities.calibration is True


def test_reads_one_observation_per_modality(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    observations = list(adapter.read_observations())
    by_type = {type(obs).__name__ for obs in observations}

    assert by_type == {
        "ImageObservation",
        "LidarObservation",
        "ImuObservation",
        "ExternalPoseMeasurement",
    }
    assert adapter.warnings() == ()


def test_image_observation_matches_ros1_adapter_contract_shape(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    image = next(obs for obs in adapter.read_observations() if isinstance(obs, ImageObservation))

    assert image.width == 2
    assert image.height == 1
    assert image.data == b"\x01\x02\x03\x04\x05\x06"
    assert image.frame_id == "front_camera_optical"
    assert image.timestamp.clock_id == config.resolved_timestamp_clock_id()
    assert image.provenance.raw_metadata["bag_timestamp_nanoseconds"] == 1_000_000_000
    assert image.provenance.source_topic == "/camera/color/image"


def test_lidar_observation_preserves_field_layout(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    lidar = next(obs for obs in adapter.read_observations() if isinstance(obs, LidarObservation))

    assert [field.name for field in lidar.fields] == ["x", "y", "z"]
    assert lidar.point_step_bytes == 12


def test_imu_orientation_absent_when_covariance_marks_it_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "no_orientation_bag"
    _build_bag(path, imu_orientation_available=False)
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    imu = next(obs for obs in adapter.read_observations() if isinstance(obs, ImuObservation))

    assert imu.orientation is None
    assert imu.linear_acceleration == (0.0, 0.0, 9.81)


def test_external_pose_uses_header_and_child_frame_id_convention(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    pose = next(
        obs for obs in adapter.read_observations() if isinstance(obs, ExternalPoseMeasurement)
    )

    assert pose.parent_frame == "odom"
    assert pose.frame_id == "base_link"
    assert pose.translation == (1.0, 2.0, 0.0)


def test_read_calibration_decodes_pinhole_model_from_lowercase_fields(bag_path: Path) -> None:
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(bag_path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    calibration = adapter.read_calibration()

    assert calibration is not None
    (entry,) = calibration.entries.values()
    assert isinstance(entry.camera_model, PinholeCameraModel)
    assert entry.camera_model.fx == 600.0
    assert entry.camera_model.cx == 640.0


def test_read_calibration_decodes_fisheye_model_without_lossy_conversion(tmp_path: Path) -> None:
    path = tmp_path / "fisheye_bag"
    _build_bag(path, distortion_model="equidistant")
    config = SourceAdapterConfig(source_type="ros2_bag", path=str(path), topics=_TOPICS)
    adapter = Ros2BagSourceAdapter(config)

    calibration = adapter.read_calibration()

    assert calibration is not None
    (entry,) = calibration.entries.values()
    assert isinstance(entry.camera_model, FisheyeCameraModel)
    assert entry.camera_model.distortion_coefficients == (0.01, 0.002, 0.0003, 0.00004)


def test_missing_required_topic_raises_before_any_observation(bag_path: Path) -> None:
    config = SourceAdapterConfig(
        source_type="ros2_bag",
        path=str(bag_path),
        topics=SourceTopicMapping(rgb="/camera/color/image", lidar="/does_not_exist"),
        required_topics=frozenset({"lidar"}),
    )
    adapter = Ros2BagSourceAdapter(config)

    with pytest.raises(MissingRequiredTopicError):
        list(adapter.read_observations())


def test_unsupported_encoding_becomes_a_warning_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "with_bad_image_bag"
    _build_bag(path, include_bad_image=True)
    config = SourceAdapterConfig(
        source_type="ros2_bag",
        path=str(path),
        topics=SourceTopicMapping(rgb="/camera/color/image"),
    )
    adapter = Ros2BagSourceAdapter(config)

    observations = list(adapter.read_observations())

    assert len(observations) == 1
    assert len(adapter.warnings()) == 1
    assert "encoding" in adapter.warnings()[0].reason


def test_same_adapter_processes_a_different_bag_via_configuration_only(tmp_path: Path) -> None:
    first_path = tmp_path / "a_bag"
    second_path = tmp_path / "b_bag"
    _build_bag(first_path)
    _build_bag(second_path, distortion_model="equidistant")

    first = Ros2BagSourceAdapter(
        SourceAdapterConfig(source_type="ros2_bag", path=str(first_path), topics=_TOPICS)
    )
    second = Ros2BagSourceAdapter(
        SourceAdapterConfig(source_type="ros2_bag", path=str(second_path), topics=_TOPICS)
    )

    assert len(list(first.read_observations())) == len(list(second.read_observations()))
    first_calibration = first.read_calibration()
    second_calibration = second.read_calibration()
    assert first_calibration is not None
    assert second_calibration is not None


def test_ros1_and_ros2_adapters_produce_the_same_canonical_shape(tmp_path: Path) -> None:
    """Downstream code never needs to know which ROS version produced a sequence."""
    from rosbags.rosbag1 import Writer as Ros1Writer
    from rosbags.typesys import Stores as Ros1Stores
    from rosbags.typesys import get_typestore as get_ros1_typestore

    from contextmap.ingestion.adapters.ros1_bag import Ros1BagSourceAdapter

    ros1_path = tmp_path / "ros1.bag"
    ros1_ts = get_ros1_typestore(Ros1Stores.ROS1_NOETIC)
    types = ros1_ts.types
    with Ros1Writer(ros1_path) as writer:
        conn = writer.add_connection(
            "/camera/image_raw", types["sensor_msgs/msg/Image"].__msgtype__, typestore=ros1_ts
        )
        msg = types["sensor_msgs/msg/Image"](
            header=types["std_msgs/msg/Header"](
                seq=0,
                stamp=types["builtin_interfaces/msg/Time"](sec=1, nanosec=0),
                frame_id="front_camera_optical",
            ),
            height=1,
            width=2,
            encoding="rgb8",
            is_bigendian=0,
            step=6,
            data=np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8),
        )
        writer.write(conn, 1_000_000_000, ros1_ts.serialize_ros1(msg, conn.msgtype))

    ros2_path = tmp_path / "ros2_bag"
    _build_bag(ros2_path)

    ros1_adapter = Ros1BagSourceAdapter(
        SourceAdapterConfig(
            source_type="ros1_bag",
            path=str(ros1_path),
            topics=SourceTopicMapping(rgb="/camera/image_raw"),
        )
    )
    ros2_adapter = Ros2BagSourceAdapter(
        SourceAdapterConfig(
            source_type="ros2_bag",
            path=str(ros2_path),
            topics=SourceTopicMapping(rgb="/camera/color/image"),
        )
    )

    def consume(adapter: Any) -> tuple[int, int, bytes]:
        (image,) = list(adapter.read_observations())
        return image.width, image.height, image.data

    assert consume(ros1_adapter) == consume(ros2_adapter)


def test_adapter_output_synchronizes_modalities_on_shared_header_clock(bag_path: Path) -> None:
    config = SourceAdapterConfig(
        source_type="ros2_bag",
        path=str(bag_path),
        topics=_TOPICS,
        timestamp_clock_id="robot-header-clock",
    )
    observations = list(Ros2BagSourceAdapter(config).read_observations())

    groups, diagnostics = synchronize(
        observations,
        config=SynchronizationConfig(reference_modality="image", tolerance_nanoseconds=0),
    )

    assert len(groups) == 1
    assert all(
        association.observation is not None for association in groups[0].associations.values()
    )
    assert diagnostics.dropped_events == ()
