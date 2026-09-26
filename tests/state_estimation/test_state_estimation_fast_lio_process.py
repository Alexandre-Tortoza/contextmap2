import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from calibration_builders import QUARTER_TURN_Z, sensor_run

from contextmap.ingestion import (
    FrameId,
    ImuObservation,
    LidarObservation,
    SourceAdapterConfig,
    SourceTopicMapping,
)
from contextmap.ingestion.adapters.ros1_bag import Ros1BagSourceAdapter
from contextmap.state_estimation.backends.fast_lio import (
    FastLioFailure,
    FastLioFailureKind,
    FastLioJob,
)
from contextmap.state_estimation.backends.fast_lio_process import (
    IMU_TOPIC,
    LIDAR_TOPIC,
    SubprocessFastLioRunner,
    build_job_record,
    parse_trajectory,
    write_input_bag,
)

MS = 1_000_000


def _job() -> FastLioJob:
    observations = sensor_run()
    return FastLioJob(
        lidar=tuple(o for o in observations if isinstance(o, LidarObservation)),
        imu=tuple(o for o in observations if isinstance(o, ImuObservation)),
        lidar_frame=FrameId("velodyne"),
        imu_frame=FrameId("imu"),
        lidar_in_imu_translation=(0.1, 0.0, 0.2),
        lidar_in_imu_rotation=QUARTER_TURN_Z,
        clock_id="fixture:header",
        scan_period_ns=100 * MS,
        parameters={"lidar_type": 2, "blind_m": 2.0},
    )


# --- Input bag and job description ------------------------------------------


def test_the_input_bag_carries_the_canonical_observations_unchanged(tmp_path: Path) -> None:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    job = _job()
    bag = tmp_path / "input.bag"

    write_input_bag(bag, job)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        counts = {c.topic: c.msgcount for c in reader.connections}
        assert counts == {LIDAR_TOPIC: len(job.lidar), IMU_TOPIC: len(job.imu)}
        clouds: list[tuple[int, Any]] = [
            (timestamp, typestore.deserialize_ros1(raw, connection.msgtype))
            for connection, timestamp, raw in reader.messages()
            if connection.topic == LIDAR_TOPIC
        ]
        imus: list[Any] = [
            typestore.deserialize_ros1(raw, connection.msgtype)
            for connection, _, raw in reader.messages()
            if connection.topic == IMU_TOPIC
        ]
    scan = job.lidar[1]
    bag_time, cloud = clouds[1]
    assert bag_time == scan.timestamp.total_nanoseconds()
    assert cloud.header.frame_id == "velodyne"
    assert (cloud.header.stamp.sec, cloud.header.stamp.nanosec) == (
        scan.timestamp.seconds,
        scan.timestamp.nanoseconds,
    )
    assert (cloud.width, cloud.height, cloud.point_step) == (2, 1, 12)
    assert [(f.name, f.offset, f.datatype) for f in cloud.fields] == [
        ("x", 0, 7),
        ("y", 4, 7),
        ("z", 8, 7),
    ]
    assert bytes(cloud.data) == scan.data
    assert imus[0].angular_velocity.x == 0.01
    # Sem orientação na fonte, o ROS a marca como ausente pelo primeiro termo da covariância.
    assert imus[0].orientation_covariance[0] == -1.0


def test_the_input_bag_reads_back_as_the_same_canonical_observations(tmp_path: Path) -> None:
    job = _job()
    bag = tmp_path / "input.bag"
    write_input_bag(bag, job)
    adapter = Ros1BagSourceAdapter(
        SourceAdapterConfig(
            source_type="ros1_bag",
            path=str(bag),
            topics=SourceTopicMapping(lidar=LIDAR_TOPIC, imu=IMU_TOPIC),
            timestamp_clock_id=job.clock_id,
        )
    )

    observations = list(adapter.read_observations())

    scans = [o for o in observations if isinstance(o, LidarObservation)]
    samples = [o for o in observations if isinstance(o, ImuObservation)]
    assert len(scans) == len(job.lidar) and len(samples) == len(job.imu)
    for original, read_back in zip(job.lidar, scans, strict=True):
        assert (read_back.point_count, read_back.point_step_bytes, read_back.data) == (
            original.point_count,
            original.point_step_bytes,
            original.data,
        )
        assert list(read_back.fields) == list(original.fields)
        assert (read_back.frame_id, read_back.timestamp) == (original.frame_id, original.timestamp)
    for original_sample, read_sample in zip(job.imu, samples, strict=True):
        assert read_sample.timestamp == original_sample.timestamp
        assert read_sample.angular_velocity == original_sample.angular_velocity
        assert read_sample.linear_acceleration == original_sample.linear_acceleration
        assert read_sample.orientation is None


def test_the_job_record_states_frames_extrinsic_direction_clock_and_parameters() -> None:
    record = build_job_record(_job())

    assert record["topics"] == {"lidar": LIDAR_TOPIC, "imu": IMU_TOPIC}
    assert record["frames"] == {"lidar": "velodyne", "imu": "imu"}
    assert record["extrinsic_T_imu_lidar"] == {
        "translation_m": [0.1, 0.0, 0.2],
        "rotation_xyzw": list(QUARTER_TURN_Z),
    }
    assert record["clock_id"] == "fixture:header"
    assert record["scan_period_ns"] == 100 * MS
    assert record["parameters"] == {"blind_m": 2.0, "lidar_type": 2}
    assert record["counts"] == {"lidar": 3, "imu": 32}
    json.dumps(record)


# --- Trajectory exchange format ---------------------------------------------


def test_the_trajectory_is_parsed_with_nanosecond_precision() -> None:
    text = "# comment\n\n1645999726.984117123 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n"

    (pose,) = parse_trajectory(text)

    assert pose.timestamp_ns == 1_645_999_726_984_117_123
    assert pose.translation == (1.0, 2.0, 3.0)
    assert pose.orientation == (0.0, 0.0, 0.0, 1.0)
    assert pose.covariance is None


def test_an_optional_row_major_covariance_follows_the_pose() -> None:
    covariance = " ".join(str(float(i)) for i in range(36))

    (pose,) = parse_trajectory(f"1.5 0 0 0 0 0 0 1 {covariance}\n")

    assert pose.covariance == tuple(float(i) for i in range(36))


@pytest.mark.parametrize(
    "text",
    ["1.0 2.0 3.0\n", "abc 1 2 3 0 0 0 1\n", "1.0 1 2 3 0 0 0 1 5\n", "1.0 1 2 x 0 0 0 1\n"],
)
def test_a_malformed_trajectory_line_is_reported_with_its_line_number(text: str) -> None:
    with pytest.raises(ValueError, match="line 1"):
        parse_trajectory(text)


# --- Process isolation, exercised with a stand-in for the FAST-LIO process --

FAKE_FAST_LIO = r"""
import json
import sys
import time
from pathlib import Path

from rosbags.rosbag1 import Reader

args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
mode = args["--mode"]
job = json.loads(Path(args["--job"]).read_text())
out = Path(args["--out"])

if mode == "hang":
    time.sleep(30)
if mode == "hang_with_child":
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    # O work dir some com o timeout; o PID do neto fica no work_root do runner.
    (out.parents[1] / "grandchild.pid").write_text(str(child.pid))
    time.sleep(30)
if mode == "fail":
    print("optimizer exploded", file=sys.stderr)
    sys.exit(3)
if mode == "nooutput":
    sys.exit(0)

scans = []
with Reader(args["--bag"]) as reader:
    for connection, timestamp, _ in reader.messages():
        if connection.topic == job["topics"]["lidar"]:
            scans.append(timestamp)

status = {"status": "ok", "fast_lio_ref": "v1.0.0", "warnings": []}
if mode == "diverged":
    status = {"status": "diverged", "message": "state covariance exploded"}
elif mode == "init":
    status = {"status": "initialization_failed", "message": "not enough IMU excitation"}
elif mode == "wrongref":
    status["fast_lio_ref"] = "v9.9.9"
elif mode == "warn":
    status["warnings"] = ["low feature count"]
(out / "status.json").write_text(json.dumps(status))

if mode == "badline":
    (out / "trajectory.tum").write_text("not a pose\n")
elif mode == "empty":
    (out / "trajectory.tum").write_text("# nothing\n")
else:
    lines = [
        f"{(start + 50_000_000) // 10**9}.{(start + 50_000_000) % 10**9:09d} {i} 0 0 0 0 0 1"
        for i, start in enumerate(scans)
    ]
    (out / "trajectory.tum").write_text("\n".join(lines) + "\n")
"""


def _runner(tmp_path: Path, mode: str, *, timeout_s: float = 60.0) -> SubprocessFastLioRunner:
    script = tmp_path / "fake_fast_lio.py"
    script.write_text(FAKE_FAST_LIO, encoding="utf-8")
    return SubprocessFastLioRunner(
        command=[
            sys.executable,
            str(script),
            "--job",
            "{job}",
            "--bag",
            "{input_bag}",
            "--out",
            "{output_dir}",
            "--mode",
            mode,
        ],
        timeout_s=timeout_s,
        work_root=tmp_path,
    )


def test_the_process_receives_the_bag_and_the_job_and_its_trajectory_is_read_back(
    tmp_path: Path,
) -> None:
    output = _runner(tmp_path, "ok").run(_job())

    assert [pose.translation for pose in output.poses] == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    assert [pose.timestamp_ns for pose in output.poses] == [
        50 * MS,
        150 * MS,
        250 * MS,
    ]
    assert output.reported_ref == "v1.0.0"
    assert output.warnings == ()


def test_the_runner_leaves_no_working_files_behind(tmp_path: Path) -> None:
    _runner(tmp_path, "ok").run(_job())

    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("fast-lio-")] == []


def test_warnings_reported_by_the_process_are_preserved(tmp_path: Path) -> None:
    output = _runner(tmp_path, "warn").run(_job())

    assert output.warnings == ("low feature count",)


@pytest.mark.parametrize(
    ("mode", "kind"),
    [
        ("fail", FastLioFailureKind.PROCESS_FAILED),
        ("nooutput", FastLioFailureKind.MISSING_OUTPUT),
        ("badline", FastLioFailureKind.INVALID_OUTPUT),
        ("diverged", FastLioFailureKind.DIVERGED),
        ("init", FastLioFailureKind.INITIALIZATION_FAILED),
    ],
)
def test_process_failures_are_classified(
    tmp_path: Path, mode: str, kind: FastLioFailureKind
) -> None:
    with pytest.raises(FastLioFailure) as raised:
        _runner(tmp_path, mode).run(_job())

    assert raised.value.kind is kind


def test_a_failing_process_keeps_the_tail_of_its_log(tmp_path: Path) -> None:
    with pytest.raises(FastLioFailure) as raised:
        _runner(tmp_path, "fail").run(_job())

    assert "exit code 3" in str(raised.value)
    assert "optimizer exploded" in raised.value.stderr_tail


def test_a_hung_process_is_stopped_at_the_timeout(tmp_path: Path) -> None:
    with pytest.raises(FastLioFailure) as raised:
        _runner(tmp_path, "hang", timeout_s=2.0).run(_job())

    assert raised.value.kind is FastLioFailureKind.TIMEOUT


def _is_running(pid: int) -> bool:
    """Whether ``pid`` names a live process; a zombie awaiting its reaper is not live."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_a_timeout_stops_the_whole_process_tree(tmp_path: Path) -> None:
    # #597: `docker run` e afins geram netos; matar só o filho direto os deixa vivos.
    started = time.monotonic()
    with pytest.raises(FastLioFailure) as raised:
        _runner(tmp_path, "hang_with_child", timeout_s=2.0).run(_job())
    elapsed = time.monotonic() - started

    assert raised.value.kind is FastLioFailureKind.TIMEOUT
    assert elapsed < 15.0
    grandchild = int((tmp_path / "grandchild.pid").read_text())
    deadline = time.monotonic() + 5.0
    try:
        while _is_running(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _is_running(grandchild)
    finally:
        if _is_running(grandchild):
            os.kill(grandchild, signal.SIGKILL)


def test_an_executable_that_cannot_start_is_a_process_failure(tmp_path: Path) -> None:
    runner = SubprocessFastLioRunner(
        command=[str(tmp_path / "does-not-exist"), "{input_bag}", "{job}", "{output_dir}"],
        timeout_s=5.0,
        work_root=tmp_path,
    )

    with pytest.raises(FastLioFailure) as raised:
        runner.run(_job())

    assert raised.value.kind is FastLioFailureKind.PROCESS_FAILED


def test_the_command_is_never_run_through_a_shell(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    runner = SubprocessFastLioRunner(
        command=[
            sys.executable,
            "-c",
            "pass",
            f"; touch {marker}",
            "{input_bag}",
            "{job}",
            "{output_dir}",
        ],
        timeout_s=5.0,
        work_root=tmp_path,
    )

    with pytest.raises(FastLioFailure):
        runner.run(_job())

    assert not marker.exists()


@pytest.mark.parametrize(
    ("command", "timeout"),
    [
        ([], 5.0),
        (["run", "{input_bag}", "{job}"], 5.0),
        (["run", "{input_bag}", "{job}", "{output_dir}"], 0.0),
    ],
)
def test_the_runner_configuration_must_be_complete(command: list[str], timeout: float) -> None:
    with pytest.raises(ValueError, match=r"command|placeholder|timeout"):
        SubprocessFastLioRunner(command=command, timeout_s=timeout)


def test_the_runner_describes_its_configuration_for_the_fingerprint() -> None:
    runner = SubprocessFastLioRunner(
        command=["run", "{input_bag}", "{job}", "{output_dir}"], timeout_s=30.0
    )

    assert runner.describe() == {
        "command": ["run", "{input_bag}", "{job}", "{output_dir}"],
        "timeout_s": 30.0,
    }
