"""Runs FAST-LIO as an external process behind :class:`FastLioRunner`.

This is the only place where ROS messages and the FAST-LIO process exist. It
turns a canonical :class:`~contextmap.state_estimation.backends.fast_lio.FastLioJob`
into files a FAST-LIO deployment can consume, runs the deployment's command
without a shell and inside a timeout, and reads the trajectory back.

The exchange contract with the deployment (a wrapper around FAST-LIO that is
installed where FAST-LIO is) is deliberately small and independent of FAST-LIO's
own configuration keys:

* ``{input_bag}``: a ROS 1 bag with the LiDAR scans (``/lidar``,
  ``sensor_msgs/PointCloud2``) and IMU samples (``/imu``,
  ``sensor_msgs/Imu``), stamped with the canonical timestamps;
* ``{job}``: ``job.json`` with the topics, frames, the LiDAR-to-IMU extrinsic
  ``T_imu_lidar``, the clock domain, the scan period and the estimator
  parameters, which the wrapper maps to FAST-LIO's configuration;
* ``{output_dir}``: where the wrapper writes ``trajectory.tum`` (one pose per
  line: ``timestamp tx ty tz qx qy qz qw`` with the timestamp as a decimal
  number of seconds to nanosecond precision, optionally followed by the 36
  covariance values row-major) and, optionally, ``status.json``
  (``{"status": "ok" | "initialization_failed" | "diverged", "message": ...,
  "fast_lio_ref": ..., "warnings": [...]}``).

Nothing here has been run against a real FAST-LIO installation. The bag, the
job description, the process handling and the trajectory parsing are tested
with a stand-in process; a reference execution with FAST-LIO installed is still
required to validate the wrapper contract. See
``src/contextmap/state_estimation/docs/backends.md``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

from contextmap.state_estimation.backends.fast_lio import (
    FastLioFailure,
    FastLioFailureKind,
    FastLioJob,
    FastLioRawPose,
    FastLioRunOutput,
)

LIDAR_TOPIC = "/lidar"
IMU_TOPIC = "/imu"
_TRAJECTORY_FILE = "trajectory.tum"
_STATUS_FILE = "status.json"
_PLACEHOLDERS = ("{input_bag}", "{job}", "{output_dir}")
_POSE_COLUMNS = 8
_COVARIANCE_COLUMNS = 36
_STDERR_TAIL_CHARS = 2000
_NANOSECONDS_PER_SECOND = 1_000_000_000

# Códigos de tipo de dado do sensor_msgs/PointField.
_POINT_FIELD_DATATYPES = {
    "int8": 1,
    "uint8": 2,
    "int16": 3,
    "uint16": 4,
    "int32": 5,
    "uint32": 6,
    "float32": 7,
    "float64": 8,
}


def write_input_bag(path: Path, job: FastLioJob) -> None:
    """Write the job's canonical observations as a ROS 1 bag.

    Args:
        path: Where to write the bag.
        job: The canonical run description.
    """
    import numpy as np
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    types = typestore.types
    time_type = types["builtin_interfaces/msg/Time"]
    header_type = types["std_msgs/msg/Header"]
    vector_type = types["geometry_msgs/msg/Vector3"]
    quaternion_type = types["geometry_msgs/msg/Quaternion"]
    field_type = types["sensor_msgs/msg/PointField"]

    def header(timestamp_seconds: int, timestamp_nanoseconds: int, frame: str) -> Any:
        return header_type(
            seq=0,
            stamp=time_type(sec=timestamp_seconds, nanosec=timestamp_nanoseconds),
            frame_id=frame,
        )

    def covariance(values: Sequence[float] | None, *, present: bool) -> Any:
        if values is not None:
            return np.array(values, dtype=np.float64)
        # Convenção do ROS: o primeiro termo -1 marca a grandeza como ausente na fonte.
        marked = np.zeros(9, dtype=np.float64)
        if not present:
            marked[0] = -1.0
        return marked

    messages: list[tuple[int, str, str, Any]] = []
    for scan in job.lidar:
        cloud = types["sensor_msgs/msg/PointCloud2"](
            header=header(scan.timestamp.seconds, scan.timestamp.nanoseconds, str(scan.frame_id)),
            height=1,
            width=scan.point_count,
            fields=[
                field_type(
                    name=descriptor.name,
                    offset=descriptor.offset_bytes,
                    datatype=_POINT_FIELD_DATATYPES[descriptor.data_type.value],
                    count=descriptor.count,
                )
                for descriptor in scan.fields
            ],
            is_bigendian=False,
            point_step=scan.point_step_bytes,
            row_step=scan.point_step_bytes * scan.point_count,
            data=np.frombuffer(scan.data, dtype=np.uint8),
            is_dense=scan.is_dense,
        )
        messages.append(
            (scan.timestamp.total_nanoseconds(), LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2", cloud)
        )
    for sample in job.imu:
        x, y, z, w = sample.orientation if sample.orientation is not None else (0.0, 0.0, 0.0, 1.0)
        angular = sample.angular_velocity if sample.angular_velocity is not None else (0.0,) * 3
        linear = (
            sample.linear_acceleration if sample.linear_acceleration is not None else (0.0,) * 3
        )
        imu = types["sensor_msgs/msg/Imu"](
            header=header(
                sample.timestamp.seconds, sample.timestamp.nanoseconds, str(sample.frame_id)
            ),
            orientation=quaternion_type(x=x, y=y, z=z, w=w),
            orientation_covariance=covariance(
                sample.orientation_covariance, present=sample.orientation is not None
            ),
            angular_velocity=vector_type(x=angular[0], y=angular[1], z=angular[2]),
            angular_velocity_covariance=covariance(
                sample.angular_velocity_covariance, present=sample.angular_velocity is not None
            ),
            linear_acceleration=vector_type(x=linear[0], y=linear[1], z=linear[2]),
            linear_acceleration_covariance=covariance(
                sample.linear_acceleration_covariance,
                present=sample.linear_acceleration is not None,
            ),
        )
        messages.append(
            (sample.timestamp.total_nanoseconds(), IMU_TOPIC, "sensor_msgs/msg/Imu", imu)
        )

    with Writer(path) as writer:
        connections = {
            topic: writer.add_connection(topic, msgtype, typestore=typestore)
            for topic, msgtype in (
                (LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2"),
                (IMU_TOPIC, "sensor_msgs/msg/Imu"),
            )
        }
        for timestamp_ns, topic, msgtype, message in sorted(messages, key=lambda item: item[0]):
            writer.write(
                connections[topic], timestamp_ns, typestore.serialize_ros1(message, msgtype)
            )


def build_job_record(job: FastLioJob) -> dict[str, Any]:
    """Describe a job as the JSON the deployment wrapper reads.

    Args:
        job: The canonical run description.

    Returns:
        A JSON-serializable record: topics, frames, the extrinsic
        ``T_imu_lidar`` (``p_imu = R * p_lidar + t``), clock domain, scan period,
        time bounds, counts and the estimator parameters.
    """
    times = [item.timestamp.total_nanoseconds() for item in (*job.lidar, *job.imu)]
    return {
        "schema": "contextmap.fast_lio_job/1",
        "topics": {"lidar": LIDAR_TOPIC, "imu": IMU_TOPIC},
        "frames": {"lidar": str(job.lidar_frame), "imu": str(job.imu_frame)},
        "extrinsic_T_imu_lidar": {
            "translation_m": list(job.lidar_in_imu_translation),
            "rotation_xyzw": list(job.lidar_in_imu_rotation),
        },
        "clock_id": job.clock_id,
        "scan_period_ns": job.scan_period_ns,
        "time_bounds_ns": {"start": min(times), "end": max(times)},
        "counts": {"lidar": len(job.lidar), "imu": len(job.imu)},
        "parameters": dict(job.parameters),
    }


def parse_trajectory(text: str) -> tuple[FastLioRawPose, ...]:
    """Parse the TUM-style trajectory a deployment writes.

    Args:
        text: File content: one pose per line, ``#`` comments and blank lines allowed.

    Returns:
        The poses in file order.

    Raises:
        ValueError: If a line is malformed; the message names the line number.
    """
    poses: list[FastLioRawPose] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) not in (_POSE_COLUMNS, _POSE_COLUMNS + _COVARIANCE_COLUMNS):
            raise ValueError(
                f"line {number}: expected {_POSE_COLUMNS} or "
                f"{_POSE_COLUMNS + _COVARIANCE_COLUMNS} columns, got {len(tokens)}"
            )
        try:
            timestamp_ns = int(
                (Decimal(tokens[0]) * _NANOSECONDS_PER_SECOND).to_integral_value(ROUND_HALF_EVEN)
            )
            values = [float(token) for token in tokens[1:]]
        except (ValueError, ArithmeticError) as error:
            raise ValueError(f"line {number}: {error}") from error
        poses.append(
            FastLioRawPose(
                timestamp_ns=timestamp_ns,
                translation=(values[0], values[1], values[2]),
                orientation=(values[3], values[4], values[5], values[6]),
                covariance=tuple(values[7:]) or None,
            )
        )
    return tuple(poses)


class SubprocessFastLioRunner:
    """Runs a deployment's FAST-LIO command as a subprocess for each job."""

    def __init__(
        self, *, command: Sequence[str], timeout_s: float, work_root: Path | None = None
    ) -> None:
        """Create the runner.

        Args:
            command: Command and arguments, run without a shell. Together they
                must contain the placeholders ``{input_bag}``, ``{job}`` and
                ``{output_dir}``, replaced by the paths of each run.
            timeout_s: Seconds after which the process is stopped.
            work_root: Directory for the per-run working directories; the
                system temporary directory when omitted.

        Raises:
            ValueError: If the command is empty or lacks a placeholder, or the
                timeout is not positive.
        """
        if not command:
            raise ValueError("command must not be empty")
        missing = [item for item in _PLACEHOLDERS if not any(item in arg for arg in command)]
        if missing:
            raise ValueError(f"command is missing the placeholders {missing}")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._command = tuple(command)
        self._timeout_s = timeout_s
        self._work_root = work_root

    def describe(self) -> Mapping[str, object]:
        """Report the command template and timeout, which influence the run."""
        return {"command": list(self._command), "timeout_s": self._timeout_s}

    def run(self, job: FastLioJob) -> FastLioRunOutput:
        """Write the job's files, run the command and read the trajectory back.

        Args:
            job: The canonical run description.

        Returns:
            The poses the process wrote, with the version and warnings it reported.

        Raises:
            FastLioFailure: If the process cannot start, fails, times out, reports
                an initialization failure or divergence, or writes no valid trajectory.
        """
        with tempfile.TemporaryDirectory(prefix="fast-lio-", dir=self._work_root) as work:
            work_dir = Path(work)
            output_dir = work_dir / "output"
            output_dir.mkdir()
            bag_path = work_dir / "input.bag"
            job_path = work_dir / "job.json"
            write_input_bag(bag_path, job)
            job_path.write_text(
                json.dumps(build_job_record(job), indent=2, sort_keys=True), encoding="utf-8"
            )
            arguments = [
                arg.replace("{input_bag}", str(bag_path))
                .replace("{job}", str(job_path))
                .replace("{output_dir}", str(output_dir))
                for arg in self._command
            ]
            self._execute(arguments)
            return self._read_output(output_dir)

    def _execute(self, arguments: list[str]) -> None:
        try:
            completed = subprocess.run(
                arguments,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self._timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise FastLioFailure(
                kind=FastLioFailureKind.TIMEOUT,
                detail=f"the process did not finish within {self._timeout_s} s",
            ) from error
        except OSError as error:
            raise FastLioFailure(
                kind=FastLioFailureKind.PROCESS_FAILED,
                detail=f"cannot start {arguments[0]!r}: {error}",
            ) from error
        if completed.returncode != 0:
            raise FastLioFailure(
                kind=FastLioFailureKind.PROCESS_FAILED,
                detail=f"the process exited with exit code {completed.returncode}",
                stderr_tail=completed.stderr[-_STDERR_TAIL_CHARS:],
            )

    def _read_output(self, output_dir: Path) -> FastLioRunOutput:
        reported_ref: str | None = None
        warnings: tuple[str, ...] = ()
        status_path = output_dir / _STATUS_FILE
        if status_path.is_file():
            try:
                record = json.loads(status_path.read_text(encoding="utf-8"))
                status = record.get("status", "ok")
                reported_ref = record.get("fast_lio_ref")
                warnings = tuple(str(item) for item in record.get("warnings", ()))
                message = str(record.get("message", ""))
            except (ValueError, AttributeError) as error:
                raise FastLioFailure(
                    kind=FastLioFailureKind.INVALID_OUTPUT,
                    detail=f"{_STATUS_FILE} is not a valid status record: {error}",
                ) from error
            if status == "initialization_failed":
                raise FastLioFailure(
                    kind=FastLioFailureKind.INITIALIZATION_FAILED, detail=message or status
                )
            if status == "diverged":
                raise FastLioFailure(kind=FastLioFailureKind.DIVERGED, detail=message or status)
            if status != "ok":
                raise FastLioFailure(
                    kind=FastLioFailureKind.INVALID_OUTPUT,
                    detail=f"unknown status {status!r} in {_STATUS_FILE}",
                )

        trajectory_path = output_dir / _TRAJECTORY_FILE
        if not trajectory_path.is_file():
            raise FastLioFailure(
                kind=FastLioFailureKind.MISSING_OUTPUT,
                detail=f"the process finished without writing {_TRAJECTORY_FILE}",
            )
        try:
            poses = parse_trajectory(trajectory_path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise FastLioFailure(
                kind=FastLioFailureKind.INVALID_OUTPUT,
                detail=f"{_TRAJECTORY_FILE}: {error}",
            ) from error
        return FastLioRunOutput(poses=poses, reported_ref=reported_ref, warnings=warnings)
