r"""Deployment wrapper that runs FAST-LIO inside its ROS 1 container.

This is the component the host-side ``SubprocessFastLioRunner`` executes. It lives
where FAST-LIO is installed and speaks the exchange contract of
:mod:`contextmap.state_estimation.backends.fast_lio_process`: it reads ``job.json``
and the input bag, runs the FAST-LIO node over the bag and writes ``trajectory.tum``
and ``status.json``. Everything ROS-specific about a run (the master, the parameter
server, the node, the bag player and the odometry topic) stays here.

The module is deliberately standalone: it imports only the standard library, plus
``rospy`` and ``nav_msgs`` on demand inside the container, and it is written for the
Python 3.8 of ROS Noetic. It must not import ``contextmap``, which is not installed
in the container. It is mounted into the container as a file::

    python3 fast_lio_wrapper.py --input-bag /data/input.bag --job /data/job.json \
        --output-dir /data/output

What the trajectory means: FAST-LIO publishes the pose of the IMU in a frame
anchored at its first pose, stamped with the end time of each scan. The wrapper
exports no covariance: this FAST-LIO revision fills the covariance of its odometry
message *after* publishing it (so a subscriber sees the previous scan's) and orders
it rotation-then-position, so it is not the pose covariance the canonical contract
defines. Poses are therefore exported without one rather than with a wrong one.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

JOB_SCHEMA = "contextmap.fast_lio_job/1"
ODOMETRY_TOPIC = "/Odometry"

_TRAJECTORY_FILE = "trajectory.tum"
_STATUS_FILE = "status.json"
_NANOSECONDS_PER_SECOND = 1_000_000_000
_LOG_TAIL_CHARS = 2000

# Parâmetros que o wrapper define a partir do job. Deixá-los para a configuração
# criaria uma segunda cópia da calibração ou ligaria saídas que a implantação não usa.
_OWNED_PARAMETERS = frozenset(
    {
        "common/lid_topic",
        "common/imu_topic",
        "mapping/extrinsic_T",
        "mapping/extrinsic_R",
        "mapping/extrinsic_est_en",
        "pcd_save/pcd_save_en",
        "pcd_save/interval",
        "publish/path_en",
        "publish/scan_publish_en",
        "publish/dense_publish_en",
        "publish/scan_bodyframe_pub_en",
    }
)

# Parâmetros do FAST-LIO que a configuração pode fornecer, com o tipo que o nó lê.
# O roscpp ignora em silêncio um parâmetro de tipo incompatível e usa o padrão do
# código, então tipo e nome são verificados aqui: nome errado ou tipo errado
# produziria uma execução diferente da que o fingerprint declara.
_ESTIMATOR_PARAMETERS: dict[str, type] = {
    "max_iteration": int,
    "filter_size_corner": float,
    "filter_size_surf": float,
    "filter_size_map": float,
    "cube_side_length": float,
    "common/time_sync_en": bool,
    "common/time_offset_lidar_to_imu": float,
    "mapping/det_range": float,
    "mapping/fov_degree": float,
    "mapping/gyr_cov": float,
    "mapping/acc_cov": float,
    "mapping/b_gyr_cov": float,
    "mapping/b_acc_cov": float,
    "preprocess/blind": float,
    "preprocess/lidar_type": int,
    "preprocess/scan_line": int,
    "preprocess/timestamp_unit": int,
    "preprocess/scan_rate": int,
    "point_filter_num": int,
    "feature_extract_enable": bool,
    "runtime_pos_log_enable": bool,
}

# O padrão do FAST-LIO é o tipo Livox: sem este parâmetro uma nuvem Velodyne seria
# lida pelo tratador errado.
_REQUIRED_PARAMETERS = ("preprocess/lidar_type",)


class JobError(ValueError):
    """Raised when ``job.json`` cannot be used to run FAST-LIO."""


class RuntimeFailure(RuntimeError):
    """Raised when the ROS runtime cannot complete a run (process crash, timeout, ...).

    Attributes:
        log_tail: The end of the log of the process involved, when there is one.
    """

    def __init__(self, message: str, *, log_tail: str = "") -> None:
        """Build the failure.

        Args:
            message: The reason, without the log.
            log_tail: The end of the log of the process involved.
        """
        super().__init__(message)
        self.log_tail = log_tail


@dataclass(frozen=True)
class Job:
    """The parts of ``job.json`` the wrapper needs.

    Attributes:
        lidar_topic: Topic carrying the LiDAR scans in the input bag.
        imu_topic: Topic carrying the IMU samples in the input bag.
        translation_m: Translation of ``T_imu_lidar`` in meters.
        rotation_xyzw: Rotation of ``T_imu_lidar`` as a quaternion ``(x, y, z, w)``.
        parameters: Estimator parameters as given by the configuration.
    """

    lidar_topic: str
    imu_topic: str
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    parameters: dict[str, Any]


@dataclass(frozen=True)
class OdometryPose:
    """One pose FAST-LIO published.

    Attributes:
        timestamp_ns: Header stamp of the odometry message in nanoseconds; FAST-LIO
            stamps it with the end time of the scan the pose was estimated from.
        position: ``(x, y, z)`` of the IMU in the frame anchored at the first pose, in meters.
        orientation_xyzw: Orientation as a quaternion ``(x, y, z, w)``.
    """

    timestamp_ns: int
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


class FastLioRuntime(Protocol):
    """Runs the FAST-LIO node over a bag and returns what it published."""

    def fast_lio_ref(self) -> str | None:
        """Report the git revision of the FAST-LIO that will run, when known."""
        ...

    def run(self, parameters: Mapping[str, Any], input_bag: Path) -> list[OdometryPose]:
        """Run FAST-LIO with the given parameters over the bag.

        Raises:
            RuntimeFailure: If a process cannot start, crashes or exceeds its deadline.
        """
        ...


# --- Job and parameters ------------------------------------------------------


def load_job(path: Path) -> Job:
    """Read and validate ``job.json``.

    Args:
        path: The job file the host runner wrote.

    Returns:
        The job.

    Raises:
        JobError: If the file cannot be read, is not JSON, has another schema or
            lacks a field.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise JobError(f"cannot read the job {path}: {error}") from error
    except ValueError as error:
        raise JobError(f"the job {path} is not valid JSON: {error}") from error
    if not isinstance(record, dict) or record.get("schema") != JOB_SCHEMA:
        found = record.get("schema") if isinstance(record, dict) else None
        raise JobError(f"unsupported job schema {found!r}: this wrapper reads {JOB_SCHEMA}")

    topics = _field(record, "topics")
    extrinsic = _field(record, "extrinsic_T_imu_lidar")
    translation = _numbers(_field(extrinsic, "translation_m", parent="extrinsic_T_imu_lidar"), 3)
    rotation = _numbers(_field(extrinsic, "rotation_xyzw", parent="extrinsic_T_imu_lidar"), 4)
    if math.sqrt(sum(value * value for value in rotation)) == 0.0:
        raise JobError("the extrinsic rotation is a zero quaternion")
    parameters = _field(record, "parameters")
    if not isinstance(parameters, dict):
        raise JobError("the job field 'parameters' must be an object")
    return Job(
        lidar_topic=str(_field(topics, "lidar", parent="topics")),
        imu_topic=str(_field(topics, "imu", parent="topics")),
        translation_m=(translation[0], translation[1], translation[2]),
        rotation_xyzw=(rotation[0], rotation[1], rotation[2], rotation[3]),
        parameters=dict(parameters),
    )


def _field(record: Any, name: str, *, parent: str | None = None) -> Any:
    if not isinstance(record, dict) or name not in record:
        where = name if parent is None else f"{parent}.{name}"
        raise JobError(f"the job field {where!r} is missing")
    return record[name]


def _numbers(values: Any, count: int) -> list[float]:
    valid = (
        isinstance(values, list)
        and len(values) == count
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        and all(math.isfinite(v) for v in values)
    )
    if not valid:
        raise JobError(f"expected {count} finite numbers, got {values!r}")
    return [float(v) for v in values]


def build_parameters(job: Job) -> dict[str, Any]:
    """Map a job to FAST-LIO's ROS parameters.

    The topics and the LiDAR-to-IMU extrinsic come from the job (which took the
    extrinsic from the canonical calibration), and the online extrinsic estimation
    is switched off so the calibration is never silently replaced. The publishing
    and PCD-saving outputs are switched off: only the odometry is read. The
    estimator parameters are forwarded verbatim after being checked.

    Args:
        job: The validated job.

    Returns:
        Parameter names (relative to the ROS root) mapped to values.

    Raises:
        JobError: If an estimator parameter is unknown, owned by the wrapper,
            of the wrong type, or a required one is missing.
    """
    estimator: dict[str, Any] = {}
    for name, value in job.parameters.items():
        if name in _OWNED_PARAMETERS:
            raise JobError(
                f"the parameter {name!r} is set by the wrapper from the job and cannot be "
                "overridden"
            )
        expected = _ESTIMATOR_PARAMETERS.get(name)
        if expected is None:
            raise JobError(
                f"unknown FAST-LIO parameter {name!r}; it would be silently ignored. "
                f"Known parameters: {sorted(_ESTIMATOR_PARAMETERS)}"
            )
        estimator[name] = _typed(name, value, expected)
    for name in _REQUIRED_PARAMETERS:
        if name not in estimator:
            raise JobError(f"the parameter {name!r} is required and was not given")
    return {
        "common/lid_topic": job.lidar_topic,
        "common/imu_topic": job.imu_topic,
        "mapping/extrinsic_est_en": False,
        "mapping/extrinsic_T": list(job.translation_m),
        "mapping/extrinsic_R": quaternion_to_row_major_matrix(job.rotation_xyzw),
        "pcd_save/pcd_save_en": False,
        "publish/path_en": False,
        "publish/scan_publish_en": False,
        "publish/dense_publish_en": False,
        "publish/scan_bodyframe_pub_en": False,
        **estimator,
    }


def _typed(name: str, value: Any, expected: type) -> Any:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is bool:
        valid = isinstance(value, bool)
    elif expected is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected is float:
        valid = is_number and math.isfinite(value)
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise JobError(
            f"the parameter {name!r} must be a {expected.__name__}, got {value!r}: FAST-LIO "
            "ignores a parameter of another type and would fall back to its built-in default"
        )
    return float(value) if expected is float else value


def quaternion_to_row_major_matrix(
    quaternion_xyzw: tuple[float, float, float, float],
) -> list[float]:
    """Convert a quaternion ``(x, y, z, w)`` into a row-major 3x3 rotation matrix.

    Args:
        quaternion_xyzw: The rotation; it is normalized before conversion.

    Returns:
        The nine matrix entries, row by row.
    """
    x, y, z, w = quaternion_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ]


# --- Outputs -----------------------------------------------------------------


def format_trajectory_line(pose: OdometryPose) -> str:
    """Format a pose as one ``trajectory.tum`` line: ``timestamp tx ty tz qx qy qz qw``.

    Args:
        pose: The pose.

    Returns:
        The line, with the timestamp as decimal seconds to nanosecond precision.
    """
    seconds, nanoseconds = divmod(pose.timestamp_ns, _NANOSECONDS_PER_SECOND)
    values = " ".join(repr(value) for value in (*pose.position, *pose.orientation_xyzw))
    return f"{seconds}.{nanoseconds:09d} {values}"


def read_fast_lio_ref(source_dir: Path) -> str | None:
    """Read the git revision of a FAST-LIO source tree without running git.

    Args:
        source_dir: The FAST-LIO checkout.

    Returns:
        The commit the checkout is at, or ``None`` when it has no readable git metadata.
    """
    git_dir = source_dir / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            head = (git_dir / head[len("ref: ") :]).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return head or None


def _is_finite(pose: OdometryPose) -> bool:
    return all(math.isfinite(value) for value in (*pose.position, *pose.orientation_xyzw))


def _write_status(
    output_dir: Path,
    status: str,
    *,
    ref: str | None,
    message: str = "",
    warnings: Sequence[str] = (),
) -> None:
    record: dict[str, Any] = {"status": status, "fast_lio_ref": ref, "warnings": list(warnings)}
    if message:
        record["message"] = message
    (output_dir / _STATUS_FILE).write_text(json.dumps(record), encoding="utf-8")


def run_wrapper(
    *, job_path: Path, input_bag: Path, output_dir: Path, runtime: FastLioRuntime
) -> int:
    """Run one job and write its outputs.

    Args:
        job_path: ``job.json``.
        input_bag: The ROS 1 bag with the LiDAR scans and IMU samples.
        output_dir: Where ``trajectory.tum`` and ``status.json`` are written.
        runtime: Runs FAST-LIO.

    Returns:
        ``0`` when the run finished, including a FAST-LIO initialization failure or
        divergence, which are reported in ``status.json``; ``1`` when the runtime
        failed; ``2`` when the job is invalid. A failure is described on stderr.
    """
    try:
        parameters = build_parameters(load_job(job_path))
    except JobError as error:
        print(f"fast_lio_wrapper: invalid job: {error}", file=sys.stderr)
        return 2
    try:
        ref = runtime.fast_lio_ref()
        poses = runtime.run(parameters, input_bag)
    except RuntimeFailure as error:
        # O runner do host guarda só o final do stderr: o motivo vem depois do log.
        if error.log_tail:
            print(error.log_tail.rstrip(), file=sys.stderr)
        print(f"fast_lio_wrapper: {error}", file=sys.stderr)
        return 1

    if not poses:
        _write_status(
            output_dir,
            "initialization_failed",
            ref=ref,
            message="FAST-LIO published no odometry: it never initialized or skipped every scan",
        )
        return 0
    bad = next((index for index, pose in enumerate(poses) if not _is_finite(pose)), None)
    if bad is not None:
        _write_status(
            output_dir,
            "diverged",
            ref=ref,
            message=f"FAST-LIO published a non-finite pose (message {bad} of {len(poses)})",
        )
        return 0

    (output_dir / _TRAJECTORY_FILE).write_text(
        "".join(format_trajectory_line(pose) + "\n" for pose in poses), encoding="utf-8"
    )
    warnings = (
        []
        if ref is not None
        else ["the FAST-LIO source has no readable git metadata: its revision is unverified"]
    )
    _write_status(output_dir, "ok", ref=ref, warnings=warnings)
    print(f"fast_lio_wrapper: {len(poses)} poses written", file=sys.stdout)
    return 0


# --- The ROS runtime (only exists inside the container) ---------------------


class _Process:
    """A child process whose output goes to a log file and which is stopped as a group."""

    def __init__(
        self, name: str, command: Sequence[str], log_dir: Path, environment: Mapping[str, str]
    ) -> None:
        self.name = name
        self._log_path = log_dir / f"{name}.log"
        self._log = self._log_path.open("wb")
        try:
            self._popen = subprocess.Popen(
                list(command),
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=dict(environment),
                start_new_session=True,
            )
        except OSError as error:
            self._log.close()
            raise RuntimeFailure(f"cannot start {name} ({command[0]!r}): {error}") from error

    def exit_code(self) -> int | None:
        return self._popen.poll()

    def tail(self) -> str:
        self._log.flush()
        return self._log_path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:]

    def stop(self, timeout_s: float = 15.0) -> None:
        if self._popen.poll() is None:
            for signal_number, wait_s in ((signal.SIGINT, timeout_s), (signal.SIGKILL, 5.0)):
                try:
                    os.killpg(self._popen.pid, signal_number)
                except ProcessLookupError:
                    break
                try:
                    self._popen.wait(timeout=wait_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self._log.close()


class RosRuntime:
    """Runs the FAST-LIO node from an install in a ROS 1 container."""

    def __init__(
        self,
        *,
        fast_lio_root: Path,
        play_rate: float = 1.0,
        idle_timeout_s: float = 5.0,
        deadline_s: float = 3600.0,
    ) -> None:
        """Create the runtime.

        Args:
            fast_lio_root: The catkin workspace holding the built FAST-LIO.
            play_rate: Playback rate of the bag; ``1.0`` is real time.
            idle_timeout_s: Seconds without odometry, after playback ends, that
                mean FAST-LIO has processed everything it will.
            deadline_s: Seconds after which the whole run is aborted.
        """
        self._root = fast_lio_root
        self._play_rate = play_rate
        self._idle_timeout_s = idle_timeout_s
        self._deadline_s = deadline_s

    def fast_lio_ref(self) -> str | None:
        """Report the revision of the FAST-LIO checkout in the workspace."""
        return read_fast_lio_ref(self._root / "src" / "FAST_LIO")

    def run(self, parameters: Mapping[str, Any], input_bag: Path) -> list[OdometryPose]:
        """Start the ROS master and FAST-LIO, play the bag and collect the odometry.

        Args:
            parameters: FAST-LIO's ROS parameters.
            input_bag: The bag to play.

        Returns:
            The odometry poses in publication order.

        Raises:
            RuntimeFailure: If a process cannot start, exits early or the deadline passes.
        """
        work_dir = Path(tempfile.mkdtemp(prefix="fast-lio-wrapper-"))
        port = _free_port()
        environment = {
            **os.environ,
            "ROS_MASTER_URI": f"http://127.0.0.1:{port}",
            "ROS_HOSTNAME": "127.0.0.1",
            "ROS_HOME": str(work_dir / "ros"),
            "ROS_LOG_DIR": str(work_dir / "ros" / "log"),
        }
        # O rospy deste processo lê o ambiente do próprio processo.
        os.environ.update({k: environment[k] for k in environment if k.startswith("ROS_")})
        deadline = time.monotonic() + self._deadline_s
        started: list[_Process] = []
        recorder: _OdometryRecorder | None = None
        try:
            master = self._start(
                started, "roscore", ["roscore", "-p", str(port)], work_dir, environment
            )
            _wait_for_port(port, master, deadline)
            recorder = _OdometryRecorder(parameters)
            node = self._start(
                started,
                "fastlio_mapping",
                [str(self._root / "devel" / "lib" / "fast_lio" / "fastlio_mapping")],
                work_dir,
                environment,
            )
            player = self._start(
                started,
                "rosbag_play",
                [
                    "rosbag",
                    "play",
                    "--wait-for-subscribers",
                    "--rate",
                    str(self._play_rate),
                    str(input_bag),
                ],
                work_dir,
                environment,
            )
            _wait_for_playback(player, node, deadline)
            recorder.mark_playback_finished()
            self._wait_until_idle(recorder, node, deadline)
            return recorder.poses()
        finally:
            for process in reversed(started):
                process.stop()
            if recorder is not None:
                recorder.shutdown()
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _start(
        started: list[_Process],
        name: str,
        command: Sequence[str],
        work_dir: Path,
        environment: Mapping[str, str],
    ) -> _Process:
        log_dir = work_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        process = _Process(name, command, log_dir, environment)
        started.append(process)
        return process

    def _wait_until_idle(
        self, recorder: _OdometryRecorder, node: _Process, deadline: float
    ) -> None:
        while recorder.seconds_since_last_message() < self._idle_timeout_s:
            _check_alive(node, "fastlio_mapping")
            _check_deadline(deadline, node)
            time.sleep(0.25)


class _OdometryRecorder:
    """Sets FAST-LIO's parameters and records its odometry through rospy."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        rospy = importlib.import_module("rospy")
        odometry = importlib.import_module("nav_msgs.msg").Odometry
        self._rospy = rospy
        self._lock = threading.Lock()
        self._poses: list[OdometryPose] = []
        self._last_message_at = time.monotonic()
        rospy.init_node("contextmap_fast_lio_recorder", anonymous=True, disable_signals=True)
        for name, value in parameters.items():
            rospy.set_param("/" + name, value)
        self._subscriber = rospy.Subscriber(
            ODOMETRY_TOPIC, odometry, self._on_message, queue_size=100_000
        )

    def _on_message(self, message: Any) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        pose = OdometryPose(
            timestamp_ns=int(message.header.stamp.to_nsec()),
            position=(position.x, position.y, position.z),
            orientation_xyzw=(orientation.x, orientation.y, orientation.z, orientation.w),
        )
        with self._lock:
            self._poses.append(pose)
            self._last_message_at = time.monotonic()

    def mark_playback_finished(self) -> None:
        """Start counting idle time from now, whatever arrived while playing."""
        with self._lock:
            self._last_message_at = max(self._last_message_at, time.monotonic())

    def seconds_since_last_message(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_message_at

    def poses(self) -> list[OdometryPose]:
        with self._lock:
            return list(self._poses)

    def shutdown(self) -> None:
        self._subscriber.unregister()
        self._rospy.signal_shutdown("run finished")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _check_alive(process: _Process, what: str) -> None:
    code = process.exit_code()
    if code is not None:
        raise RuntimeFailure(f"{what} exited with code {code}", log_tail=process.tail())


def _check_deadline(deadline: float, process: _Process) -> None:
    if time.monotonic() > deadline:
        raise RuntimeFailure("the run exceeded its deadline", log_tail=process.tail())


def _wait_for_port(port: int, master: _Process, deadline: float) -> None:
    while True:
        _check_alive(master, "roscore")
        _check_deadline(deadline, master)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)


def _wait_for_playback(player: _Process, node: _Process, deadline: float) -> None:
    while player.exit_code() is None:
        _check_alive(node, "fastlio_mapping")
        _check_deadline(deadline, player)
        time.sleep(0.25)
    if player.exit_code() != 0:
        raise RuntimeFailure(
            f"rosbag play exited with code {player.exit_code()}", log_tail=player.tail()
        )


# --- Command line ------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wrapper from the command line.

    Args:
        argv: Arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        The process exit code (see :func:`run_wrapper`).
    """
    parser = argparse.ArgumentParser(description="Run FAST-LIO over a job's input bag.")
    parser.add_argument("--input-bag", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fast-lio-root", type=Path, default=Path("/opt/fast-lio"))
    parser.add_argument("--play-rate", type=float, default=1.0)
    parser.add_argument("--idle-timeout-s", type=float, default=5.0)
    parser.add_argument("--deadline-s", type=float, default=3600.0)
    arguments = parser.parse_args(argv)
    runtime = RosRuntime(
        fast_lio_root=arguments.fast_lio_root,
        play_rate=arguments.play_rate,
        idle_timeout_s=arguments.idle_timeout_s,
        deadline_s=arguments.deadline_s,
    )
    return run_wrapper(
        job_path=arguments.job,
        input_bag=arguments.input_bag,
        output_dir=arguments.output_dir,
        runtime=runtime,
    )


if __name__ == "__main__":
    sys.exit(main())
