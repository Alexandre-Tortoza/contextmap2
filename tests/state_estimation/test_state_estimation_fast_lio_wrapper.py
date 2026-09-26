"""Tests of the deployment wrapper that runs FAST-LIO inside its ROS 1 container.

The wrapper is exercised here with a stand-in for the ROS runtime: what these tests
protect is the exchange contract with the host runner and the mapping from the job to
FAST-LIO's parameters. Running the real FAST-LIO is a validation, not a unit test.
"""

import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from calibration_builders import QUARTER_TURN_Z, sensor_run

from contextmap.ingestion import FrameId, ImuObservation, LidarObservation
from contextmap.state_estimation.backends import fast_lio_wrapper
from contextmap.state_estimation.backends.fast_lio import (
    FastLioFailure,
    FastLioFailureKind,
    FastLioJob,
)
from contextmap.state_estimation.backends.fast_lio_process import (
    SubprocessFastLioRunner,
    build_job_record,
    parse_trajectory,
)
from contextmap.state_estimation.backends.fast_lio_wrapper import (
    JOB_SCHEMA,
    JobError,
    OdometryPose,
    RuntimeFailure,
    build_parameters,
    format_trajectory_line,
    load_job,
    read_fast_lio_ref,
    run_wrapper,
)

MS = 1_000_000
LIDAR_PARAMETERS: dict[str, Any] = {"preprocess/lidar_type": 2}

# O runner do host escreve o bag de entrada do FAST-LIO, e escrever um bag ROS 1 exige o extra
# ``ros1``. Só os dois testes do runner chegam lá; o resto do módulo roda na instalação base.
_needs_rosbags = pytest.mark.skipif(
    importlib.util.find_spec("rosbags") is None,
    reason="writing the FAST-LIO input bag needs the ros1 extra",
)


def _record(parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The job description exactly as the host runner writes it."""
    observations = sensor_run()
    job = FastLioJob(
        lidar=tuple(o for o in observations if isinstance(o, LidarObservation)),
        imu=tuple(o for o in observations if isinstance(o, ImuObservation)),
        lidar_frame=FrameId("velodyne"),
        imu_frame=FrameId("imu"),
        lidar_in_imu_translation=(0.1, -0.2, 0.3),
        lidar_in_imu_rotation=QUARTER_TURN_Z,
        clock_id="fixture:header",
        scan_period_ns=100 * MS,
        parameters=dict(LIDAR_PARAMETERS if parameters is None else parameters),
    )
    return build_job_record(job)


def _write_job(tmp_path: Path, record: Mapping[str, Any]) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _pose(index: int, **changes: Any) -> OdometryPose:
    values: dict[str, Any] = {
        "timestamp_ns": 1_645_999_726_000_000_000 + index * 100 * MS + 100_837_141,
        "position": (float(index), 0.5 * index, -0.25),
        "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
    }
    values.update(changes)
    return OdometryPose(**values)


class _FakeRuntime:
    def __init__(
        self, poses: list[OdometryPose] | Exception, *, ref: str | None = "abc123"
    ) -> None:
        self._poses = poses
        self._ref = ref
        self.calls: list[tuple[dict[str, Any], Path]] = []

    def fast_lio_ref(self) -> str | None:
        return self._ref

    def run(self, parameters: Mapping[str, Any], input_bag: Path) -> list[OdometryPose]:
        self.calls.append((dict(parameters), input_bag))
        if isinstance(self._poses, Exception):
            raise self._poses
        return self._poses


# --- The job the host runner writes ------------------------------------------


def test_the_wrapper_reads_the_job_description_the_runner_writes(tmp_path: Path) -> None:
    job = load_job(_write_job(tmp_path, _record({"preprocess/lidar_type": 2})))

    assert job.lidar_topic == "/lidar" and job.imu_topic == "/imu"
    assert job.translation_m == (0.1, -0.2, 0.3)
    assert job.rotation_xyzw == QUARTER_TURN_Z
    assert job.parameters == {"preprocess/lidar_type": 2}


def test_a_job_of_another_schema_is_refused(tmp_path: Path) -> None:
    record = _record()
    record["schema"] = "contextmap.fast_lio_job/2"

    with pytest.raises(JobError, match=JOB_SCHEMA):
        load_job(_write_job(tmp_path, record))


def test_a_job_missing_a_field_names_it(tmp_path: Path) -> None:
    record = _record()
    del record["extrinsic_T_imu_lidar"]

    with pytest.raises(JobError, match="extrinsic_T_imu_lidar"):
        load_job(_write_job(tmp_path, record))


def test_a_job_that_is_not_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(JobError, match="not valid JSON"):
        load_job(path)


# --- From the job to FAST-LIO's parameters -----------------------------------


def test_topics_and_the_extrinsic_are_mapped_in_the_direction_fast_lio_expects(
    tmp_path: Path,
) -> None:
    parameters = build_parameters(load_job(_write_job(tmp_path, _record())))

    assert parameters["common/lid_topic"] == "/lidar"
    assert parameters["common/imu_topic"] == "/imu"
    # T_imu_lidar: p_imu = R * p_lidar + t, com R uma volta de 90 graus em torno de z.
    assert parameters["mapping/extrinsic_T"] == [0.1, -0.2, 0.3]
    expected = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert parameters["mapping/extrinsic_R"] == pytest.approx(expected, abs=1e-12)


def test_the_extrinsic_is_never_estimated_online_and_nothing_is_published_or_saved(
    tmp_path: Path,
) -> None:
    parameters = build_parameters(load_job(_write_job(tmp_path, _record())))

    assert parameters["mapping/extrinsic_est_en"] is False
    assert parameters["pcd_save/pcd_save_en"] is False
    assert parameters["publish/scan_publish_en"] is False
    assert parameters["publish/path_en"] is False
    assert parameters["publish/dense_publish_en"] is False
    assert parameters["publish/scan_bodyframe_pub_en"] is False


def test_the_estimator_parameters_are_forwarded_verbatim(tmp_path: Path) -> None:
    given = {
        "preprocess/lidar_type": 2,
        "preprocess/scan_line": 16,
        "preprocess/timestamp_unit": 0,
        "preprocess/blind": 2.0,
        "mapping/acc_cov": 0.08,
        "point_filter_num": 4,
        "feature_extract_enable": False,
    }

    parameters = build_parameters(load_job(_write_job(tmp_path, _record(given))))

    assert {name: parameters[name] for name in given} == given


def test_an_integer_is_accepted_for_a_floating_point_parameter_and_sent_as_a_float(
    tmp_path: Path,
) -> None:
    given = {"preprocess/lidar_type": 2, "preprocess/blind": 2}

    parameters = build_parameters(load_job(_write_job(tmp_path, _record(given))))

    assert parameters["preprocess/blind"] == 2.0
    assert isinstance(parameters["preprocess/blind"], float)


def test_the_lidar_type_is_required_because_the_default_would_parse_the_wrong_sensor(
    tmp_path: Path,
) -> None:
    with pytest.raises(JobError, match="preprocess/lidar_type"):
        build_parameters(load_job(_write_job(tmp_path, _record({}))))


@pytest.mark.parametrize(
    "name",
    [
        "common/lid_topic",
        "common/imu_topic",
        "mapping/extrinsic_T",
        "mapping/extrinsic_R",
        "mapping/extrinsic_est_en",
        "pcd_save/pcd_save_en",
        "publish/path_en",
    ],
)
def test_a_parameter_the_wrapper_owns_cannot_be_overridden(tmp_path: Path, name: str) -> None:
    given = {**LIDAR_PARAMETERS, name: "x"}

    with pytest.raises(JobError, match=name):
        build_parameters(load_job(_write_job(tmp_path, _record(given))))


def test_an_unknown_parameter_is_refused_instead_of_being_silently_ignored(
    tmp_path: Path,
) -> None:
    given = {**LIDAR_PARAMETERS, "mapping/acc_covariance": 0.1}

    with pytest.raises(JobError, match="mapping/acc_covariance"):
        build_parameters(load_job(_write_job(tmp_path, _record(given))))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("preprocess/scan_line", 16.0),
        ("preprocess/scan_line", True),
        ("mapping/acc_cov", "0.1"),
        ("feature_extract_enable", 1),
    ],
)
def test_a_parameter_of_the_wrong_type_is_refused_because_fast_lio_would_ignore_it(
    tmp_path: Path, name: str, value: object
) -> None:
    given = {**LIDAR_PARAMETERS, name: value}

    with pytest.raises(JobError, match=name):
        build_parameters(load_job(_write_job(tmp_path, _record(given))))


# --- The trajectory exchange format ------------------------------------------


def test_a_pose_is_written_with_nanosecond_precision_and_read_back_by_the_runner() -> None:
    pose = _pose(0, timestamp_ns=1_645_999_726_984_117_123, position=(1.5, -2.25, 1e-9))

    line = format_trajectory_line(pose)
    (read_back,) = parse_trajectory(line)

    assert read_back.timestamp_ns == 1_645_999_726_984_117_123
    assert read_back.translation == (1.5, -2.25, 1e-9)
    assert read_back.orientation == (0.0, 0.0, 0.0, 1.0)
    assert read_back.covariance is None


# --- Which FAST-LIO the deployment ran ---------------------------------------


def test_the_reference_is_read_from_the_detached_head_of_the_source(tmp_path: Path) -> None:
    source = tmp_path / "FAST_LIO"
    (source / ".git").mkdir(parents=True)
    sha = "7cc4175de6f8ba2edf34bab02a42195b141027e9"
    (source / ".git" / "HEAD").write_text(sha + "\n", encoding="utf-8")

    assert read_fast_lio_ref(source) == sha


def test_the_reference_follows_a_symbolic_head(tmp_path: Path) -> None:
    source = tmp_path / "FAST_LIO"
    (source / ".git" / "refs" / "heads").mkdir(parents=True)
    sha = "0123456789abcdef0123456789abcdef01234567"
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (source / ".git" / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")

    assert read_fast_lio_ref(source) == sha


def test_a_source_without_a_repository_reports_no_reference(tmp_path: Path) -> None:
    assert read_fast_lio_ref(tmp_path) is None


# --- The wrapper end to end, with a stand-in for the ROS runtime -------------


def _run(tmp_path: Path, runtime: _FakeRuntime, record: Mapping[str, Any] | None = None) -> int:
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    return run_wrapper(
        job_path=_write_job(tmp_path, _record() if record is None else record),
        input_bag=tmp_path / "input.bag",
        output_dir=output,
        runtime=runtime,
    )


def test_a_successful_run_writes_the_trajectory_and_the_reference_it_ran(tmp_path: Path) -> None:
    runtime = _FakeRuntime([_pose(0), _pose(1), _pose(2)], ref="7cc4175")

    assert _run(tmp_path, runtime) == 0

    output = tmp_path / "output"
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status == {"status": "ok", "fast_lio_ref": "7cc4175", "warnings": []}
    poses = parse_trajectory((output / "trajectory.tum").read_text(encoding="utf-8"))
    assert [p.timestamp_ns for p in poses] == [
        1_645_999_726_100_837_141,
        1_645_999_726_200_837_141,
        1_645_999_726_300_837_141,
    ]
    assert [p.translation for p in poses] == [
        (0.0, 0.0, -0.25),
        (1.0, 0.5, -0.25),
        (2.0, 1.0, -0.25),
    ]


def test_an_unverifiable_source_revision_is_reported_as_a_warning_not_hidden(
    tmp_path: Path,
) -> None:
    assert _run(tmp_path, _FakeRuntime([_pose(0)], ref=None)) == 0

    status = json.loads((tmp_path / "output" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    assert status["fast_lio_ref"] is None
    assert "unverified" in status["warnings"][0]


def test_the_runtime_receives_the_bag_and_the_parameters_derived_from_the_job(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime([_pose(0)])

    _run(tmp_path, runtime)

    ((parameters, bag),) = runtime.calls
    assert bag == tmp_path / "input.bag"
    assert parameters["preprocess/lidar_type"] == 2
    assert parameters["mapping/extrinsic_T"] == [0.1, -0.2, 0.3]


def test_a_run_that_publishes_no_odometry_is_an_initialization_failure(tmp_path: Path) -> None:
    assert _run(tmp_path, _FakeRuntime([])) == 0

    output = tmp_path / "output"
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "initialization_failed"
    assert "no odometry" in status["message"]
    assert not (output / "trajectory.tum").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"position": (math.nan, 0.0, 0.0)},
        {"position": (0.0, math.inf, 0.0)},
        {"orientation_xyzw": (0.0, 0.0, math.nan, 1.0)},
    ],
)
def test_a_non_finite_pose_is_a_divergence_and_no_trajectory_is_written(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    assert _run(tmp_path, _FakeRuntime([_pose(0), _pose(1, **change), _pose(2)])) == 0

    output = tmp_path / "output"
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "diverged"
    assert "non-finite" in status["message"]
    assert not (output / "trajectory.tum").exists()


def test_a_runtime_failure_exits_with_an_error_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _FakeRuntime(RuntimeFailure("fastlio_mapping exited with code -11"))

    assert _run(tmp_path, runtime) == 1

    assert "fastlio_mapping exited with code -11" in capsys.readouterr().err
    assert not (tmp_path / "output" / "status.json").exists()


def test_the_reason_of_a_runtime_failure_comes_after_the_log_so_the_host_tail_keeps_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    noisy_log = "\n".join(f"[RUNNING] Bag Time: {i}" for i in range(500))
    runtime = _FakeRuntime(RuntimeFailure("the run exceeded its deadline", log_tail=noisy_log))

    assert _run(tmp_path, runtime) == 1

    error = capsys.readouterr().err
    assert "[RUNNING] Bag Time: 499" in error
    assert error.rstrip().endswith("the run exceeded its deadline")


def test_an_invalid_job_never_starts_the_runtime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record({"mapping/acc_covariance": 0.1, **LIDAR_PARAMETERS})
    runtime = _FakeRuntime([_pose(0)])

    assert _run(tmp_path, runtime, record) == 2

    assert runtime.calls == []
    assert "mapping/acc_covariance" in capsys.readouterr().err


# --- The wrapper and the host runner agree -----------------------------------

_STAND_IN = r"""
import sys
from pathlib import Path

from contextmap.state_estimation.backends.fast_lio_wrapper import OdometryPose, run_wrapper

job, bag, out, mode = sys.argv[1:5]


class Runtime:
    def fast_lio_ref(self):
        return "7cc4175"

    def run(self, parameters, input_bag):
        assert Path(input_bag).is_file()
        if mode == "silent":
            return []
        return [
            OdometryPose(
                timestamp_ns=1_645_999_726_000_000_000 + i * 100_000_000 + 50_000_000,
                position=(float(i), 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
            for i in range(3)
        ]


sys.exit(
    run_wrapper(job_path=Path(job), input_bag=Path(bag), output_dir=Path(out), runtime=Runtime())
)
"""


def _host_runner(tmp_path: Path, mode: str) -> SubprocessFastLioRunner:
    script = tmp_path / "stand_in.py"
    script.write_text(_STAND_IN, encoding="utf-8")
    return SubprocessFastLioRunner(
        command=[sys.executable, str(script), "{job}", "{input_bag}", "{output_dir}", mode],
        timeout_s=120.0,
        work_root=tmp_path,
    )


def _host_job() -> FastLioJob:
    observations = sensor_run()
    return FastLioJob(
        lidar=tuple(o for o in observations if isinstance(o, LidarObservation)),
        imu=tuple(o for o in observations if isinstance(o, ImuObservation)),
        lidar_frame=FrameId("velodyne"),
        imu_frame=FrameId("imu"),
        lidar_in_imu_translation=(0.0, 0.0, 0.0),
        lidar_in_imu_rotation=(0.0, 0.0, 0.0, 1.0),
        clock_id="fixture:header",
        scan_period_ns=100 * MS,
        parameters=dict(LIDAR_PARAMETERS),
    )


@_needs_rosbags
def test_the_host_runner_reads_what_the_wrapper_writes_for_the_job_it_wrote(
    tmp_path: Path,
) -> None:
    output = _host_runner(tmp_path, "ok").run(_host_job())

    assert output.reported_ref == "7cc4175"
    assert [pose.translation for pose in output.poses] == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    assert [pose.timestamp_ns for pose in output.poses] == [
        1_645_999_726_050_000_000,
        1_645_999_726_150_000_000,
        1_645_999_726_250_000_000,
    ]


@_needs_rosbags
def test_the_host_runner_classifies_a_silent_run_as_an_initialization_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(FastLioFailure) as failure:
        _host_runner(tmp_path, "silent").run(_host_job())

    assert failure.value.kind is FastLioFailureKind.INITIALIZATION_FAILED


def test_the_wrapper_module_is_standalone_so_it_can_run_where_contextmap_is_not_installed() -> None:
    import ast

    source = Path(fast_lio_wrapper.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert "contextmap" not in imported
    assert imported <= set(sys.stdlib_module_names) | {"rospy", "nav_msgs"}
