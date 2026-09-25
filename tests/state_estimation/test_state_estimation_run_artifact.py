import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from pose_builders import make_external_pose, make_request, timestamp_ns

from contextmap.ingestion import FrameId
from contextmap.state_estimation import (
    IncompleteRunArtifactError,
    PoseEstimateId,
    PreflightFinding,
    PreflightStatus,
    RunArtifactError,
    StateEstimationDebugLevel,
    StateEstimationOutcome,
    StateEstimationRunId,
    StateEstimationRunManifest,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    execute_state_estimation,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
    InvalidSamplePolicy,
)

MS = 1_000_000
SEQUENCE = "corridor-02"


def _outcome(measurements: list | None = None, **config: object) -> StateEstimationOutcome:  # type: ignore[type-arg]
    estimator = ExternalPoseEstimator(
        ExternalPoseConfig(
            reference_frame=FrameId("map"),
            body_frame=FrameId("body"),
            **config,  # type: ignore[arg-type]
        )
    )
    observations = measurements or [make_external_pose(index) for index in range(5)]
    return execute_state_estimation(estimator, make_request(observations))


def _writer(
    workspace: Path,
    *,
    index: int = 1,
    debug_level: StateEstimationDebugLevel = StateEstimationDebugLevel.NONE,
) -> StateEstimationRunWriter:
    return StateEstimationRunWriter(
        output_dir=_run_dir(workspace, index),
        sequence_name=SEQUENCE,
        run_id=StateEstimationRunId(f"run-{index:04d}"),
        run_index=index,
        debug_level=debug_level,
    )


def _run_dir(workspace: Path, index: int = 1) -> Path:
    """Onde o writer grava: o chamador decide o diretório final, o writer não calcula caminho."""
    return workspace / f"run-{index:04d}"


def _write(
    workspace: Path,
    outcome: StateEstimationOutcome | None = None,
    **writer_options: object,
) -> tuple[StateEstimationRunManifest, StateEstimationOutcome]:
    outcome = outcome or _outcome()
    manifest = _writer(workspace, **writer_options).finalize(outcome, runtime_s=0.5)  # type: ignore[arg-type]
    return manifest, outcome


# --- Layout and round trip --------------------------------------------------


def test_a_run_persists_contractual_outputs_metrics_and_a_manifest(tmp_path: Path) -> None:
    _write(tmp_path)

    run_dir = _run_dir(tmp_path)
    for relative in (
        "manifest.json",
        "README.md",
        "outputs/trajectory.json",
        "outputs/poses.jsonl",
        "outputs/pose-index.jsonl",
        "outputs/frame-summary.json",
        "outputs/quality.json",
        "metrics/preflight.json",
        "metrics/motion.json",
        "metrics/runtime.json",
    ):
        assert (run_dir / relative).is_file(), relative


def test_the_persisted_trajectory_round_trips(tmp_path: Path) -> None:
    _, outcome = _write(tmp_path)

    reader = StateEstimationRunReader(_run_dir(tmp_path))

    assert reader.trajectory() == outcome.result.trajectory


def test_the_manifest_identifies_inputs_backend_calibration_and_code(tmp_path: Path) -> None:
    manifest, outcome = _write(tmp_path)

    trajectory = outcome.result.trajectory
    assert manifest.run_id == StateEstimationRunId("run-0001")
    assert manifest.run_index == 1
    assert manifest.sequence_name == SEQUENCE
    assert manifest.sequence_artifact_id == trajectory.provenance.sequence_artifact_id
    assert manifest.selection_id == trajectory.provenance.selection_id
    assert manifest.trajectory_id == trajectory.trajectory_id
    assert manifest.estimator == trajectory.provenance.estimator
    assert manifest.calibration_identity == trajectory.provenance.calibration_identity
    assert manifest.code_version == trajectory.provenance.code_version
    assert (manifest.reference_frame, manifest.body_frame) == (FrameId("map"), FrameId("body"))
    assert manifest.clock_id == trajectory.poses[0].timestamp.clock_id
    assert manifest.start_time_ns == 0
    assert manifest.end_time_ns == 400 * MS
    assert manifest.pose_count == 5
    assert manifest.consumed_observation_count == 5
    assert manifest.rejected_observation_count == 0
    assert manifest.preflight_status == "ready"
    assert manifest.interpolation == {"translation": "linear", "orientation": "slerp_shortest_arc"}
    assert StateEstimationRunReader(_run_dir(tmp_path)).manifest == manifest


# --- Pose retrieval ---------------------------------------------------------


def test_a_pose_is_retrieved_by_identity_or_time_without_any_debug_data(tmp_path: Path) -> None:
    _, outcome = _write(tmp_path, debug_level=StateEstimationDebugLevel.FULL)
    shutil.rmtree(_run_dir(tmp_path) / "debug")
    reader = StateEstimationRunReader(_run_dir(tmp_path))
    target = outcome.result.trajectory.poses[3]

    assert reader.pose(target.estimate_id) == target
    assert reader.pose_at(target.timestamp) == target
    assert reader.pose_at(timestamp_ns(7)) is None
    assert reader.verify_integrity() == []
    with pytest.raises(RunArtifactError, match="unknown"):
        reader.pose(PoseEstimateId(f"{target.estimate_id}-missing"))


def test_a_single_pose_is_read_without_loading_the_other_poses(tmp_path: Path) -> None:
    _, outcome = _write(tmp_path)
    poses_path = _run_dir(tmp_path) / "outputs" / "poses.jsonl"
    lines = poses_path.read_bytes().split(b"\n")
    lines[0] = lines[0].replace(b"map", b"MAP")  # corrupts pose 0, same length
    poses_path.write_bytes(b"\n".join(lines))
    reader = StateEstimationRunReader(_run_dir(tmp_path))

    intact = outcome.result.trajectory.poses[2]

    assert reader.pose(intact.estimate_id) == intact
    assert any("content hash mismatch" in problem for problem in reader.verify_integrity())


# --- Debug levels -----------------------------------------------------------


def _debug_files(run_dir: Path) -> set[str]:
    debug = run_dir / "debug"
    if not debug.exists():
        return set()
    return {str(path.relative_to(debug)) for path in debug.rglob("*") if path.is_file()}


def test_debug_levels_add_human_evidence_without_changing_the_outputs(tmp_path: Path) -> None:
    outcome = _outcome(
        [make_external_pose(0), make_external_pose(1), make_external_pose(2, time_ns=900 * MS)],
        max_gap_ns=250 * MS,
    )
    poses_by_level: dict[str, bytes] = {}
    for index, level in enumerate(StateEstimationDebugLevel, start=1):
        _write(tmp_path, outcome, index=index, debug_level=level)
        run_dir = _run_dir(tmp_path, index)
        poses_by_level[level.value] = (run_dir / "outputs" / "poses.jsonl").read_bytes()

    none = _debug_files(_run_dir(tmp_path, 1))
    standard = _debug_files(_run_dir(tmp_path, 2))
    full = _debug_files(_run_dir(tmp_path, 3))

    assert none == set()
    assert {
        "pose-deltas.jsonl",
        "timestamp-gaps.jsonl",
        "trajectory-xy.csv",
        "trajectory-xz.csv",
    } <= standard
    assert standard < full
    assert "backend-diagnostics/events.jsonl" in full
    assert len(set(poses_by_level.values())) == 1


def test_disabling_debug_keeps_everything_a_downstream_stage_needs(tmp_path: Path) -> None:
    _, outcome = _write(tmp_path, debug_level=StateEstimationDebugLevel.NONE)

    reader = StateEstimationRunReader(_run_dir(tmp_path))

    assert reader.manifest.debug_level == "none"
    assert reader.trajectory() == outcome.result.trajectory
    assert (
        reader.manifest.calibration_identity
        == outcome.result.trajectory.provenance.calibration_identity
    )


# --- Integrity, immutability, identity --------------------------------------


def test_corruption_and_missing_payloads_are_detected(tmp_path: Path) -> None:
    _write(tmp_path)
    run_dir = _run_dir(tmp_path)
    (run_dir / "outputs" / "poses.jsonl").write_bytes(b"tampered")
    (run_dir / "outputs" / "quality.json").unlink()

    problems = StateEstimationRunReader(run_dir).verify_integrity()

    assert any("poses.jsonl" in problem and "size mismatch" in problem for problem in problems)
    assert any("quality.json" in problem and "missing" in problem for problem in problems)


def test_a_directory_without_a_manifest_is_not_a_run(tmp_path: Path) -> None:
    (tmp_path / "half-written").mkdir()

    with pytest.raises(IncompleteRunArtifactError):
        StateEstimationRunReader(tmp_path / "half-written")


def test_an_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path)
    manifest_path = _run_dir(tmp_path) / "manifest.json"
    record = json.loads(manifest_path.read_text())
    record["schema_version"] = "99.0.0"
    manifest_path.write_text(json.dumps(record))

    with pytest.raises(RunArtifactError, match="schema_version"):
        StateEstimationRunReader(_run_dir(tmp_path))


def test_a_finalized_run_is_never_overwritten_or_finalized_twice(tmp_path: Path) -> None:
    _write(tmp_path)
    before = (_run_dir(tmp_path) / "outputs" / "poses.jsonl").read_bytes()

    with pytest.raises(RunArtifactError, match="already exists"):
        _write(tmp_path)

    writer = _writer(tmp_path, index=2)
    writer.finalize(_outcome())
    with pytest.raises(RunArtifactError, match="already finalized"):
        writer.finalize(_outcome())
    assert (_run_dir(tmp_path) / "outputs" / "poses.jsonl").read_bytes() == before


def test_a_run_whose_preflight_was_blocked_is_not_persisted(tmp_path: Path) -> None:
    outcome = _outcome()
    blocked = dataclasses.replace(
        outcome.preflight,
        blockers=(PreflightFinding(code="missing_input", message="no lidar"),),
    )
    assert blocked.status is PreflightStatus.BLOCKED

    with pytest.raises(RunArtifactError, match="blocked"):
        _writer(tmp_path).finalize(StateEstimationOutcome(preflight=blocked, result=outcome.result))

    assert not _run_dir(tmp_path).exists()


def test_the_run_opens_from_its_own_directory_alone(tmp_path: Path) -> None:
    _, outcome = _write(tmp_path)
    moved = tmp_path / "somewhere" / "else"
    shutil.copytree(_run_dir(tmp_path), moved)

    assert StateEstimationRunReader(moved).trajectory() == outcome.result.trajectory


# --- Metrics for investigating anomalies without rerunning ------------------


def test_gaps_rejections_and_motion_can_be_investigated_from_the_persisted_run(
    tmp_path: Path,
) -> None:
    measurements = [
        make_external_pose(0),
        make_external_pose(1, parent_frame="odom"),
        make_external_pose(2, time_ns=100 * MS),
        make_external_pose(3, time_ns=1_100 * MS),
    ]
    outcome = _outcome(
        measurements, invalid_sample_policy=InvalidSamplePolicy.SKIP, max_gap_ns=500 * MS
    )
    manifest, _ = _write(tmp_path, outcome)
    reader = StateEstimationRunReader(_run_dir(tmp_path))

    quality = reader.read_record("outputs/quality.json")
    motion = reader.read_record("metrics/motion.json")

    assert manifest.rejected_observation_count == 1
    assert manifest.gap_count == 1
    assert manifest.diagnostic_counts == {
        "external_pose.frame_mismatch": 1,
        "external_pose.timestamp_gap": 1,
    }
    assert quality["pose_count"] == 3
    assert quality["gaps"][0]["duration_ns"] == 1_000 * MS
    assert quality["sample_rate_hz"] == pytest.approx(2 / 1.1)
    assert motion["translation_delta_m"]["count"] == 2
    events = [
        json.loads(line)
        for line in (_run_dir(tmp_path) / "metrics" / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert {event["code"] for event in events} == set(manifest.diagnostic_counts)


def test_the_preflight_report_and_frame_summary_are_persisted(tmp_path: Path) -> None:
    _write(tmp_path)
    reader = StateEstimationRunReader(_run_dir(tmp_path))

    preflight = reader.read_record("metrics/preflight.json")
    frames = reader.read_record("outputs/frame-summary.json")

    assert preflight["status"] == "ready"
    assert preflight["required_inputs"] == ["external_pose"]
    assert frames["dynamic"] == {"reference_frame": "map", "body_frame": "body"}
    assert frames["conventions"]["transform"] == "T_parent_child"


def test_runtime_is_recorded_separately_and_only_when_measured(tmp_path: Path) -> None:
    _writer(tmp_path).finalize(_outcome())

    assert not (_run_dir(tmp_path) / "metrics" / "runtime.json").exists()


# --- Run identity -----------------------------------------------------------


def test_the_run_is_written_exactly_where_the_caller_says_and_nothing_else_is_created(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "corridor-02" / "run-0001" / "state_estimation"

    StateEstimationRunWriter(
        output_dir=target,
        sequence_name=SEQUENCE,
        run_id=StateEstimationRunId("state-run"),
        run_index=1,
    ).finalize(_outcome())

    assert StateEstimationRunReader(target).manifest.run_id == StateEstimationRunId("state-run")
    # Sem registro `runs.json` e sem `runs/<capability>/<sequência>/`: só o diretório do artifact.
    assert sorted(path.name for path in target.parent.iterdir()) == ["state_estimation"]
    assert sorted(path.name for path in (tmp_path / "ws").iterdir()) == ["corridor-02"]


def test_the_run_id_and_index_are_recorded_as_supplied_and_never_allocated(
    tmp_path: Path,
) -> None:
    manifest, _ = _write(tmp_path, index=7)

    assert (manifest.run_id, manifest.run_index) == (StateEstimationRunId("run-0007"), 7)
    assert not (tmp_path / "run-0001").exists()


def test_a_second_run_at_the_same_output_directory_is_refused_and_leaves_the_first_intact(
    tmp_path: Path,
) -> None:
    _write(tmp_path)
    before = (_run_dir(tmp_path) / "manifest.json").read_bytes()

    with pytest.raises(RunArtifactError):
        _write(tmp_path)

    assert (_run_dir(tmp_path) / "manifest.json").read_bytes() == before
