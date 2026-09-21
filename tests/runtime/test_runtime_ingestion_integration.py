"""Canonical ingestion through the CLI with the real, composed ROS 1 adapter and a real bag.

The synthetic bag is built at test time with ``rosbags`` (the same approach the adapter tests
use), so nothing binary is checked in. Skipped when the optional ROS 1 support is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import selected_document
from runtime_worlds import run_cli

np = pytest.importorskip("numpy")
rosbags_rosbag1 = pytest.importorskip("rosbags.rosbag1")
rosbags_typesys = pytest.importorskip("rosbags.typesys")

from contextmap.ingestion import (  # noqa: E402
    SequenceArtifactReader,
    compute_source_content_hash,
)

Writer = rosbags_rosbag1.Writer
_TS = rosbags_typesys.get_typestore(rosbags_typesys.Stores.ROS1_NOETIC)


def _stamp(seconds: int) -> Any:
    return _TS.types["builtin_interfaces/msg/Time"](sec=seconds, nanosec=0)


def _header(seconds: int, frame: str) -> Any:
    return _TS.types["std_msgs/msg/Header"](seq=0, stamp=_stamp(seconds), frame_id=frame)


def _build_bag(path: Path, *, frames: int = 3) -> None:
    types = _TS.types
    with Writer(path) as writer:
        image = writer.add_connection(
            "/camera/image_raw", types["sensor_msgs/msg/Image"].__msgtype__, typestore=_TS
        )
        imu = writer.add_connection(
            "/imu/data", types["sensor_msgs/msg/Imu"].__msgtype__, typestore=_TS
        )
        for second in range(1, frames + 1):
            image_message = types["sensor_msgs/msg/Image"](
                header=_header(second, "front_camera_optical"),
                height=1,
                width=2,
                encoding="rgb8",
                is_bigendian=0,
                step=6,
                data=np.arange(6, dtype=np.uint8) + second,
            )
            writer.write(
                image,
                second * 1_000_000_000,
                _TS.serialize_ros1(image_message, image.msgtype),
            )
            imu_message = types["sensor_msgs/msg/Imu"](
                header=_header(second, "imu_link"),
                orientation=types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.zeros(9, dtype=np.float64),
                angular_velocity=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.01),
                angular_velocity_covariance=np.zeros(9, dtype=np.float64),
                linear_acceleration=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=9.81),
                linear_acceleration_covariance=np.zeros(9, dtype=np.float64),
            )
            writer.write(imu, second * 1_000_000_000, _TS.serialize_ros1(imu_message, imu.msgtype))


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(selected_document()), encoding="utf-8")
    return path


def _ingest(tmp_path: Path, bag: Path, *extra: str) -> tuple[int, str, str]:
    return run_cli(
        "ingest",
        "-c",
        str(_config(tmp_path)),
        "--source",
        str(bag),
        "--sequence-name",
        "corridor-02",
        "--topic",
        "rgb=/camera/image_raw",
        "--topic",
        "imu=/imu/data",
        "--sync-reference",
        "image",
        "--sync-tolerance-ns",
        "100000000",
        "--workspace",
        str(tmp_path / "ws"),
        "--json",
        *extra,
        environ={},
    )


def test_a_real_ros1_bag_becomes_a_reopenable_sequence_artifact(tmp_path: Path) -> None:
    bag = tmp_path / "recording.bag"
    _build_bag(bag)

    code, out, err = _ingest(tmp_path, bag)

    result = json.loads(out)
    assert code == 0, out + err
    assert result["status"] == "completed"
    assert result["observation_counts"] == {"external_pose": 0, "image": 3, "imu": 3, "lidar": 0}
    reader = SequenceArtifactReader(Path(result["artifact_path"]))
    assert reader.verify_integrity() == []
    provenance = reader.read_provenance()
    assert provenance is not None
    assert provenance.source_type == "ros1_bag"
    assert provenance.source_content_hash == compute_source_content_hash(bag)
    assert provenance.synchronization_policy == "nearest_within_tolerance"
    assert result["diagnostics"]["processing_observations"] == 3


def test_ingesting_the_same_bag_twice_gives_the_same_request_identity(tmp_path: Path) -> None:
    bag = tmp_path / "recording.bag"
    _build_bag(bag)

    first = json.loads(_ingest(tmp_path, bag)[1])
    second = json.loads(_ingest(tmp_path, bag)[1])

    assert first["request_identity"] == second["request_identity"]
    assert first["artifact_id"] != second["artifact_id"]


def test_the_preflight_finds_a_required_topic_the_real_bag_lacks(tmp_path: Path) -> None:
    bag = tmp_path / "recording.bag"
    _build_bag(bag)

    code, out, err = _ingest(
        tmp_path, bag, "--preflight", "--topic", "lidar=/velodyne_points", "--required", "lidar"
    )

    report = json.loads(out)
    assert code == 1, out + err
    assert any(p["path"] == "source.required_topics" for p in report["problems"])


def test_a_file_that_is_not_a_bag_is_a_preflight_problem_naming_the_error(tmp_path: Path) -> None:
    bag = tmp_path / "not-a-bag.bag"
    bag.write_bytes(b"this is not a ros bag")

    code, out, err = _ingest(tmp_path, bag, "--preflight")

    report = json.loads(out)
    assert code == 1, out + err
    assert any(p["path"] == "source.read" for p in report["problems"])
    assert not (tmp_path / "ws").exists()


def test_a_missing_source_fails_before_touching_the_workspace(tmp_path: Path) -> None:
    code, out, err = _ingest(tmp_path, tmp_path / "nowhere.bag")

    result = json.loads(out)
    assert code == 1, out + err
    assert result["status"] == "failed" and result["failure"]["category"] == "source"
    assert not (tmp_path / "ws").exists()
