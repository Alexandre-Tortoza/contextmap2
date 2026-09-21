import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from input_builders import (
    ACCEPT_ALL,
    MS,
    SEQUENCE_ID,
    assemble_plan,
    make_calibration,
    make_image,
    make_imu,
    make_trajectory,
    rigid,
)
from lidar_builders import make_scan

from contextmap.geometric_mapping import (
    AccumulationError,
    GeometricMapArtifactManifest,
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    GeometryInputPlan,
    GeometryPoint,
    GeometryReference,
    IncompleteMapArtifactError,
    MapArtifactError,
    MapDebugLevel,
    MapId,
    MotionCorrectionPolicy,
    ScanDisposition,
    ScanVoxelPolicy,
    geometry_id_for,
    mapping_configuration_fingerprint,
    transform_scans,
    verify_transform_trace,
)
from contextmap.geometric_mapping.serialization import decode_transform_trace
from contextmap.ingestion import FrameId, FullSequenceSelection, SourceObservation, decode_selection
from contextmap.state_estimation import StateEstimationRunId

SEQUENCE = "corridor-02"
CALIBRATION = make_calibration((rigid("body", "lidar", (0.5, 0.0, 0.25)),))
DEFAULT_RUN_ID = StateEstimationRunId("run-0001")
CLOUD = tuple((math.sin(i) * 3.0, math.cos(i) * 2.0, (i % 5) * 0.1) for i in range(30))


def _observations(*, extra: tuple[SourceObservation, ...] = ()) -> list[SourceObservation]:
    scans: list[SourceObservation] = [
        make_scan(f"scan-{i:04d}", time_ns=i * 100 * MS, points=CLOUD) for i in range(5)
    ]
    return [*scans, make_image(), make_imu(), *extra]


def _plan(
    observations: list[SourceObservation] | None = None,
    *,
    policy: MotionCorrectionPolicy = ACCEPT_ALL,
    run_id: StateEstimationRunId | None = DEFAULT_RUN_ID,
) -> GeometryInputPlan:
    return assemble_plan(
        observations if observations is not None else _observations(),
        calibration=CALIBRATION,
        trajectory=make_trajectory(calibration=CALIBRATION),
        policy=policy,
        run_id=run_id,
    )


def _run_dir(workspace: Path, index: int = 1) -> Path:
    """Onde o writer grava: o chamador decide o diretório final, o writer não calcula caminho."""
    return workspace / f"run-{index:04d}"


def _writer(
    workspace: Path,
    *,
    index: int = 1,
    debug_level: MapDebugLevel = MapDebugLevel.NONE,
) -> GeometricMapArtifactWriter:
    return GeometricMapArtifactWriter(
        output_dir=_run_dir(workspace, index),
        sequence_name=SEQUENCE,
        run_id=GeometricMapRunId(f"run-{index:04d}"),
        run_index=index,
        debug_level=debug_level,
    )


def _write(
    workspace: Path,
    plan: GeometryInputPlan | None = None,
    *,
    aggregation: ScanVoxelPolicy | None = None,
    runtime_s: float | None = None,
    peak_memory_bytes: int | None = None,
    **writer_options: Any,
) -> GeometricMapArtifactManifest:
    return _writer(workspace, **writer_options).finalize(
        plan=plan if plan is not None else _plan(),
        aggregation=aggregation,
        code_version="test",
        runtime_s=runtime_s,
        peak_memory_bytes=peak_memory_bytes,
    )


def _record(run_dir: Path, relative_path: str) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((run_dir / relative_path).read_text(encoding="utf-8"))
    return record


def _lines(run_dir: Path, relative_path: str) -> list[dict[str, Any]]:
    text = (run_dir / relative_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


# --- Layout and round trip ---------------------------------------------------------------------


def test_a_run_persists_contractual_outputs_lineage_metrics_and_a_manifest(tmp_path: Path) -> None:
    _write(tmp_path)

    run_dir = _run_dir(tmp_path)
    for relative in (
        "manifest.json",
        "README.md",
        "lineage.json",
        "config.json",
        "environment.json",
        "outputs/geometry.bin",
        "outputs/source-index.jsonl",
        "outputs/map-metadata.json",
        "metrics/input-plan.json",
        "metrics/mapping.json",
    ):
        assert (run_dir / relative).is_file(), relative
    assert not (run_dir / "debug").exists()


def test_a_reference_resolves_after_close_and_reopen_to_the_same_xyz_and_provenance(
    tmp_path: Path,
) -> None:
    plan = _plan()
    manifest = _write(tmp_path, plan)
    map_id = manifest.map_id
    points: list[GeometryPoint] = []
    for scan in transform_scans(plan):
        for position in range(scan.point_count):
            points.append(scan.geometry_point(position, map_id=map_id, geometry_index=len(points)))

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        geometry = reader.geometry()
        assert geometry.geometric_map.point_count == len(points)
        for point in points:
            assert geometry.get(point.reference) == point


def test_the_map_identity_is_derived_from_the_sequence_and_the_run(tmp_path: Path) -> None:
    manifest = _write(tmp_path)

    assert manifest.map_id == MapId("corridor-02--run-0001")
    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert reader.geometry().geometric_map.map_id == manifest.map_id


def test_the_manifest_identifies_inputs_configuration_code_and_counts(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _write(tmp_path, plan)

    assert manifest.run_id == GeometricMapRunId("run-0001") and manifest.run_index == 1
    assert manifest.sequence_name == SEQUENCE
    assert manifest.sequence_artifact_id == SEQUENCE_ID
    assert manifest.selection_id == plan.selection_id
    assert manifest.trajectory_id == plan.trajectory_id
    assert manifest.state_estimation_run_id == StateEstimationRunId("run-0001")
    assert manifest.calibration_identity == plan.calibration_identity
    assert manifest.code_version == "test"
    assert manifest.map_frame == FrameId("map")
    assert manifest.configuration_fingerprint == mapping_configuration_fingerprint(plan, None)
    assert (manifest.point_count, manifest.source_point_count) == (150, 150)
    assert (manifest.scan_count, manifest.rejected_scan_count) == (5, 0)
    assert manifest.aggregation_rule is None
    assert manifest.spatial_index_kind == "scan_bounds"
    assert manifest.clock_id == "fixture:header"
    assert (manifest.start_time_ns, manifest.end_time_ns) == (0, 400 * MS)
    assert manifest.debug_level == "none"
    assert manifest.schema_version
    assert {entry.path for entry in manifest.file_inventory} >= {
        "outputs/geometry.bin",
        "outputs/source-index.jsonl",
        "outputs/map-metadata.json",
        "lineage.json",
        "config.json",
    }
    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert reader.manifest == manifest


# --- Lineage, configuration, metrics -----------------------------------------------------------


def test_the_lineage_names_every_input_the_map_was_built_from(tmp_path: Path) -> None:
    plan = _plan(_observations(extra=(make_scan("scan-late", time_ns=900 * MS),)))
    _write(tmp_path, plan)

    lineage = _record(_run_dir(tmp_path), "lineage.json")

    assert lineage["sequence_artifact_id"] == "sequence-0001"
    assert decode_selection(lineage["selection"]) == FullSequenceSelection()
    assert lineage["selection_id"] == plan.selection_id
    assert lineage["state_estimation_run_id"] == "run-0001"
    assert lineage["trajectory_id"] == str(plan.trajectory_id)
    assert lineage["calibration_identity"] == plan.calibration_identity
    assert (lineage["map_frame"], lineage["body_frame"]) == ("map", "body")
    assert lineage["code_version"] == "test"
    assert lineage["source_observation_ids"] == [f"scan-{i:04d}" for i in range(5)]
    assert lineage["rejected_observation_ids"] == ["scan-late"]


def test_the_effective_configuration_and_its_fingerprint_are_recorded(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _write(tmp_path, plan, aggregation=ScanVoxelPolicy(cell_m=0.5))

    config = _record(_run_dir(tmp_path), "config.json")

    assert config["pose_lookup"] == {
        "mode": "interpolated",
        "max_time_delta_ns": None,
        "max_interpolation_gap_ns": None,
    }
    assert config["motion_correction_policy"] == {"raw": "accept", "unknown": "accept"}
    assert config["aggregation"] == {
        "kind": "scan_voxel_centroid",
        "cell_m": 0.5,
        "rule": "scan-voxel-centroid-0.5m",
    }
    assert config["spatial_index"]["kind"] == "scan_bounds"
    assert config["configuration_fingerprint"] == manifest.configuration_fingerprint
    assert manifest.configuration_fingerprint == mapping_configuration_fingerprint(
        plan, ScanVoxelPolicy(cell_m=0.5)
    )


def test_the_fingerprint_changes_with_any_mapping_parameter() -> None:
    plan = _plan()
    strict = _plan(
        policy=MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN)
    )

    base = mapping_configuration_fingerprint(plan, None)

    assert base == mapping_configuration_fingerprint(_plan(), None)
    assert base != mapping_configuration_fingerprint(plan, ScanVoxelPolicy(cell_m=0.5))
    assert base != mapping_configuration_fingerprint(plan, ScanVoxelPolicy(cell_m=0.25))
    assert base != mapping_configuration_fingerprint(strict, None)


def test_selected_accepted_rejected_and_ignored_inputs_are_recorded(tmp_path: Path) -> None:
    plan = _plan(_observations(extra=(make_scan("scan-late", time_ns=900 * MS),)))
    _write(tmp_path, plan)

    record = _record(_run_dir(tmp_path), "metrics/input-plan.json")

    assert (record["selected_scan_count"], record["accepted_scan_count"]) == (6, 5)
    assert record["rejected_scan_count"] == 1
    assert record["rejections"] == [
        {
            "observation_id": "scan-late",
            "reason": "pose_lookup_rejected",
            "detail": record["rejections"][0]["detail"],
        }
    ]
    assert "out_of_range" in record["rejections"][0]["detail"]
    assert record["ignored_observation_counts"] == {"image": 1, "imu": 1}
    assert record["pose_lookup"]["lookup_count"] == 5
    assert record["pose_lookup"]["exact_count"] == 5
    assert record["pose_lookup"]["max_time_delta_ns"] == 0
    assert record["pose_lookup_rejected_count"] == 1


def test_motion_correction_states_the_policy_and_its_warnings_are_recorded(tmp_path: Path) -> None:
    warn = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN)
    _write(tmp_path, _plan(policy=warn))

    record = _record(_run_dir(tmp_path), "metrics/input-plan.json")["motion_correction"]

    assert record["policy"] == {"raw": "accept", "unknown": "warn"}
    assert record["state_counts"] == {"unknown": 5}
    assert record["disposition_counts"] == {"warn": 5}
    assert len(record["warnings"]) == 5 and "scan-0000" in record["warnings"][0]


def test_the_mapping_metrics_report_counts_bounds_time_and_the_reduction(tmp_path: Path) -> None:
    manifest = _write(tmp_path, aggregation=ScanVoxelPolicy(cell_m=1.0))

    record = _record(_run_dir(tmp_path), "metrics/mapping.json")

    assert record["map_id"] == "corridor-02--run-0001"
    assert record["scan_count"] == 5 and record["scans_without_geometry"] == 0
    assert record["source_point_count"] == 150
    assert record["point_count"] == manifest.point_count < 150
    assert record["dropped_non_finite_count"] == 0
    assert record["aggregation_rule"] == "scan-voxel-centroid-1.0m"
    assert record["reduction_ratio"] == pytest.approx(manifest.point_count / 150)
    assert record["bounds"]["frame_id"] == "map"
    assert record["geometry_size_bytes"] == manifest.point_count * 64


def test_runtime_and_memory_are_metrics_apart_from_the_geometry(tmp_path: Path) -> None:
    _write(tmp_path, runtime_s=1.5, peak_memory_bytes=123_456)
    _write(tmp_path, index=2)

    measured = _record(_run_dir(tmp_path), "metrics/runtime.json")
    assert measured == {"runtime_s": 1.5, "peak_memory_bytes": 123_456}
    assert not (_run_dir(tmp_path, 2) / "metrics" / "runtime.json").exists()
    assert "runtime_s" not in _record(_run_dir(tmp_path), "metrics/mapping.json")


def test_the_environment_is_recorded_without_importing_the_libraries_it_describes(
    tmp_path: Path,
) -> None:
    _write(tmp_path)

    environment = _record(_run_dir(tmp_path), "environment.json")

    assert environment["python"].count(".") >= 2
    assert environment["platform"]
    assert environment["code_version"] == "test"


# --- Determinism -------------------------------------------------------------------------------


def test_identical_inputs_reproduce_the_same_geometry_index_and_lineage(tmp_path: Path) -> None:
    first = _write(tmp_path / "a")
    second = _write(tmp_path / "b")

    def hashes(manifest: GeometricMapArtifactManifest) -> dict[str, str]:
        return {entry.path: entry.content_hash for entry in manifest.file_inventory}

    assert hashes(first) == hashes(second)


# --- Integrity ---------------------------------------------------------------------------------


def test_an_intact_run_verifies_clean(tmp_path: Path) -> None:
    _write(tmp_path)

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert reader.verify_integrity() == []


def test_a_flipped_byte_in_the_geometry_is_detected(tmp_path: Path) -> None:
    _write(tmp_path)
    payload = _run_dir(tmp_path) / "outputs" / "geometry.bin"
    corrupted = bytearray(payload.read_bytes())
    corrupted[10] ^= 0xFF
    payload.write_bytes(bytes(corrupted))

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        problems = reader.verify_integrity()

    assert any("geometry.bin" in problem and "hash" in problem for problem in problems)


def test_a_missing_or_truncated_geometry_payload_is_detected_explicitly(tmp_path: Path) -> None:
    _write(tmp_path)
    payload = _run_dir(tmp_path) / "outputs" / "geometry.bin"
    payload.write_bytes(payload.read_bytes()[:-64])

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert any("size" in problem for problem in reader.verify_integrity())
        with pytest.raises(MapArtifactError, match="bytes"):
            reader.geometry()

    payload.unlink()
    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert any("missing" in problem for problem in reader.verify_integrity())
        with pytest.raises(MapArtifactError, match=r"geometry\.bin"):
            reader.geometry()


def test_a_tampered_source_index_is_detected(tmp_path: Path) -> None:
    _write(tmp_path)
    index = _run_dir(tmp_path) / "outputs" / "source-index.jsonl"
    index.write_text(index.read_text().replace("scan-0001", "scan-9999"))

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert any("source-index" in problem for problem in reader.verify_integrity())


def test_a_stale_derived_index_is_reported_by_the_integrity_check(tmp_path: Path) -> None:
    _write(tmp_path)
    run_dir = _run_dir(tmp_path)
    lines = _lines(run_dir, "outputs/source-index.jsonl")
    lines[2]["bounds"]["minimum_m"] = [90.0, 90.0, 90.0]
    lines[2]["bounds"]["maximum_m"] = [91.0, 91.0, 91.0]
    (run_dir / "outputs" / "source-index.jsonl").write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    )

    with GeometricMapArtifactReader(run_dir) as reader:
        problems = reader.verify_integrity()

    # O inventário acusa o arquivo alterado e a verificação do índice acusa os limites do scan.
    assert any("source-index" in problem for problem in problems)
    assert any("scan 2" in problem for problem in problems)


def test_an_incomplete_directory_is_not_a_run(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()

    with pytest.raises(IncompleteMapArtifactError, match="manifest"):
        GeometricMapArtifactReader(tmp_path / "run")


def test_an_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    _write(tmp_path)
    manifest = _run_dir(tmp_path) / "manifest.json"
    record = json.loads(manifest.read_text())
    record["schema_version"] = "99.0.0"
    manifest.write_text(json.dumps(record))

    with pytest.raises(MapArtifactError, match="schema_version"):
        GeometricMapArtifactReader(_run_dir(tmp_path))


# --- Standalone reading ------------------------------------------------------------------------


def test_a_persisted_map_opens_without_numpy_ros_or_a_model_library(tmp_path: Path) -> None:
    _write(tmp_path)
    source_dir = Path(sys.modules["contextmap"].__file__ or "").parents[1]
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from contextmap.geometric_mapping import GeometricMapArtifactReader\n"
        f"reader = GeometricMapArtifactReader(Path({str(_run_dir(tmp_path))!r}))\n"
        "geometry = reader.geometry()\n"
        "points = list(geometry.iter_geometry())\n"
        "assert reader.verify_integrity() == []\n"
        "reader.close()\n"
        "print(len(points), 'numpy' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(source_dir)},
    )

    assert result.stdout.split() == ["150", "False"]


def test_a_researcher_can_trace_a_persisted_point_without_rerunning_the_mapping(
    tmp_path: Path,
) -> None:
    manifest = _write(tmp_path)
    reference = _reference(manifest.map_id, 37)

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        trace = reader.geometry().trace(reference)

    assert verify_transform_trace(trace) == []
    assert str(trace.source_observation_id) == "scan-0001"
    assert [transform.step.kind.value for transform in trace.transforms] == [
        "dynamic_pose",
        "static_calibration",
    ]


def _reference(map_id: MapId, index: int) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


def test_only_contractual_json_records_can_be_read(tmp_path: Path) -> None:
    _write(tmp_path, debug_level=MapDebugLevel.STANDARD)

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert reader.read_record("metrics/mapping.json")["scan_count"] == 5
        assert reader.read_record("lineage.json")["trajectory_id"]
        for forbidden in ("debug/bounds-summary.json", "outputs/geometry.bin", "../x.json"):
            with pytest.raises(MapArtifactError, match="contractual"):
                reader.read_record(forbidden)


# --- Debug evidence ----------------------------------------------------------------------------


def test_debug_none_persists_no_debug_evidence(tmp_path: Path) -> None:
    _write(tmp_path, debug_level=MapDebugLevel.NONE)

    assert not (_run_dir(tmp_path) / "debug").exists()


def test_standard_debug_persists_summaries_and_transform_samples(tmp_path: Path) -> None:
    _write(tmp_path, debug_level=MapDebugLevel.STANDARD)

    run_dir = _run_dir(tmp_path)
    for relative in (
        "debug/scans-by-time.jsonl",
        "debug/selected-transform-traces.jsonl",
        "debug/trajectory-over-map.csv",
        "debug/bounds-summary.json",
        "debug/warnings.jsonl",
    ):
        assert (run_dir / relative).is_file(), relative
    assert not (run_dir / "debug" / "trace-residuals.jsonl").exists()

    scans = _lines(run_dir, "debug/scans-by-time.jsonl")
    assert [scan["observation_id"] for scan in scans] == [f"scan-{i:04d}" for i in range(5)]
    traces = [
        decode_transform_trace(line)
        for line in _lines(run_dir, "debug/selected-transform-traces.jsonl")
    ]
    assert traces and all(verify_transform_trace(trace) == [] for trace in traces)
    csv = (run_dir / "debug" / "trajectory-over-map.csv").read_text().splitlines()
    assert csv[0] == "ordinal,timestamp_ns,x_m,y_m,z_m" and len(csv) == 6


def test_full_debug_adds_a_denser_sample_and_the_reconstruction_residuals(tmp_path: Path) -> None:
    _write(tmp_path / "standard", debug_level=MapDebugLevel.STANDARD)
    _write(tmp_path / "full", debug_level=MapDebugLevel.FULL)

    standard = _lines(_run_dir(tmp_path / "standard"), "debug/selected-transform-traces.jsonl")
    full_dir = _run_dir(tmp_path / "full")
    full = _lines(full_dir, "debug/selected-transform-traces.jsonl")
    residuals = _lines(full_dir, "debug/trace-residuals.jsonl")

    assert len(full) > len(standard)
    assert len(residuals) == len(full)
    assert all(row["residual_m"] <= 1e-9 and row["problems"] == [] for row in residuals)


def test_debug_is_never_part_of_the_inventory_and_removing_it_keeps_the_map_usable(
    tmp_path: Path,
) -> None:
    manifest = _write(tmp_path, debug_level=MapDebugLevel.FULL)
    assert not any(entry.path.startswith("debug/") for entry in manifest.file_inventory)

    shutil.rmtree(_run_dir(tmp_path) / "debug")

    with GeometricMapArtifactReader(_run_dir(tmp_path)) as reader:
        assert reader.verify_integrity() == []
        assert reader.geometry().geometric_map.point_count == 150


def test_warnings_list_rejected_scans_and_correction_warnings(tmp_path: Path) -> None:
    warn = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN)
    plan = _plan(_observations(extra=(make_scan("scan-late", time_ns=900 * MS),)), policy=warn)
    _write(tmp_path, plan, debug_level=MapDebugLevel.STANDARD)

    warnings = _lines(_run_dir(tmp_path), "debug/warnings.jsonl")

    kinds = [warning["kind"] for warning in warnings]
    assert kinds.count("motion_correction") == 5 and kinds.count("rejected_scan") == 1


# --- Immutability, atomicity and run identity ---------------------------------------------------


def test_a_finished_run_is_never_overwritten(tmp_path: Path) -> None:
    _write(tmp_path)
    before = (_run_dir(tmp_path) / "manifest.json").read_bytes()

    with pytest.raises(MapArtifactError, match="already exists"):
        _write(tmp_path)

    assert (_run_dir(tmp_path) / "manifest.json").read_bytes() == before


def test_a_writer_finalizes_once(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.finalize(plan=_plan(), aggregation=None, code_version="test")

    with pytest.raises(MapArtifactError, match="already finalized"):
        writer.finalize(plan=_plan(), aggregation=None, code_version="test")


def test_a_failed_run_leaves_no_run_and_no_temporary_directory(tmp_path: Path) -> None:
    empty_plan = _plan([make_scan("scan-late", time_ns=900 * MS)])

    with pytest.raises(AccumulationError, match="no geometry"):
        _write(tmp_path, empty_plan)

    assert not _run_dir(tmp_path).exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_the_run_is_written_exactly_where_the_caller_says_and_nothing_else_is_created(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "corridor-02" / "run-0001" / "geometric_mapping"

    GeometricMapArtifactWriter(
        output_dir=target,
        sequence_name=SEQUENCE,
        run_id=GeometricMapRunId("map-run"),
        run_index=1,
    ).finalize(plan=_plan(), aggregation=None, code_version="test")

    with GeometricMapArtifactReader(target) as reader:
        assert reader.manifest.run_id == GeometricMapRunId("map-run")
    # Sem registro `runs.json` e sem `runs/<capability>/<sequência>/`: só o diretório do artifact.
    assert sorted(path.name for path in target.parent.iterdir()) == ["geometric_mapping"]
    assert sorted(path.name for path in (tmp_path / "ws").iterdir()) == ["corridor-02"]


def test_the_run_id_and_index_are_recorded_as_supplied_and_never_allocated(
    tmp_path: Path,
) -> None:
    manifest = _write(tmp_path, index=7)

    assert (manifest.run_id, manifest.run_index) == (GeometricMapRunId("run-0007"), 7)
    assert not _run_dir(tmp_path, 1).exists()


def test_the_manifest_describes_a_map_with_no_state_estimation_run_id(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _plan(run_id=None))

    assert manifest.state_estimation_run_id is None
    assert dataclasses.is_dataclass(manifest)
