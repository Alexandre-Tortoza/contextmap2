import dataclasses
import io
import json
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from geometric_mapping_builders import (
    SCAN_COUNT,
    SEQUENCE,
    SEQUENCE_ID,
    TRUE_STATIC,
    WORLD,
    make_plan,
    sensor_view,
    write_run,
)

from contextmap.evaluation import (
    ExpectedPoint,
    GeometricMappingEvaluationError,
    GeometricMappingEvaluationReport,
    GeometricMappingProtocol,
    GeometryReferenceRole,
    ReferenceGeometry,
    compare_contractual_inventories,
    compare_geometric_mapping_reports,
    encode_geometric_mapping_report,
    evaluate_geometric_mapping,
    evaluate_round_trip,
)
from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMapArtifactReader,
    GeometricMapRunId,
    GeometryInputPlan,
    MapId,
    MotionCorrectionRecord,
    MotionCorrectionState,
    PackedGeometry,
    ScanVoxelPolicy,
    accumulate_plan,
    mapping_configuration_fingerprint,
)
from contextmap.ingestion import FrameId, SourceObservation, SourceObservationId
from contextmap.shared import Vector3, invert_rigid, normalize_quaternion

MS = 1_000_000
PROTOCOL = GeometricMappingProtocol(
    structure_point_stride=1,
    trace_sample_count=8,
    transform_tolerance_m=1e-9,
    adjacent_scan_lag=1,
    coherence_pair_count=5,
    coherence_points_per_scan=200,
    plane_neighbour_count=8,
    max_plane_curvature=0.02,
    max_correspondence_distance_m=1.0,
    ghost_residual_m=0.02,
    density_voxel_m=0.3,
)
EXPECTED = tuple(
    ExpectedPoint(
        source_observation_id=SourceObservationId(f"scan-{scan:04d}"),
        source_point_index=index,
        expected_map_coordinates_m=WORLD[index],
    )
    for scan, index in ((0, 5), (3, 400), (6, 900), (9, 942))
)
MAP = FrameId("map")


def _evaluate(
    reader: GeometricMapArtifactReader,
    *,
    protocol: GeometricMappingProtocol = PROTOCOL,
    expected_points: Sequence[ExpectedPoint] = EXPECTED,
    reference: ReferenceGeometry | None = None,
) -> GeometricMappingEvaluationReport:
    with reader:
        return evaluate_geometric_mapping(
            reader=reader,
            protocol=protocol,
            expected_points=expected_points,
            reference=reference,
        )


def _correct(tmp_path: Path) -> GeometricMapArtifactReader:
    return write_run(tmp_path, make_plan())


def _all_points(reader: GeometricMapArtifactReader) -> list[Vector3]:
    return [point.coordinates_m for point in reader.geometry().iter_geometry()]


# --- Protocol ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"structure_point_stride": 0},
        {"trace_sample_count": 0},
        {"transform_tolerance_m": 0.0},
        {"adjacent_scan_lag": 0},
        {"coherence_pair_count": -1},
        {"coherence_points_per_scan": 0},
        {"plane_neighbour_count": 2},
        {"max_plane_curvature": 0.5},
        {"max_correspondence_distance_m": math.nan},
        {"ghost_residual_m": -0.1},
        {"density_voxel_m": 0.0},
        {"max_plausible_range_m": 0.0},
    ],
)
def test_an_impossible_protocol_is_refused(change: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=next(iter(change))):
        dataclasses.replace(PROTOCOL, **change)  # type: ignore[arg-type]


def test_the_protocol_has_no_defaults_so_no_threshold_is_inherited_from_a_dataset() -> None:
    with pytest.raises(TypeError):
        GeometricMappingProtocol()  # type: ignore[call-arg]


# --- A correct mapping passes every layer ------------------------------------------------------


def test_a_correct_mapping_is_numerically_exact_and_globally_consistent(tmp_path: Path) -> None:
    report = _evaluate(_correct(tmp_path))

    assert report.structure.integrity_problems == ()
    assert report.transform_trace.within_tolerance
    assert report.transform_trace.max_composition_error_m <= 1e-9
    assert report.transform_trace.max_round_trip_error_m <= 1e-9
    assert all(check.within_tolerance for check in report.expected_points)
    assert max(check.error_m for check in report.expected_points) <= 1e-9
    assert report.overlap is not None
    assert report.overlap.pooled_residual_m is not None
    assert report.overlap.pooled_residual_m.maximum <= 1e-9
    assert report.overlap.inconsistent_fraction == 0.0
    # Bordas e cantos não formam um plano local e ficam sem correspondência.
    assert 0.6 < report.overlap.overlap_fraction < 1.0


def test_the_structure_is_reverified_from_the_persisted_geometry(tmp_path: Path) -> None:
    report = _evaluate(_correct(tmp_path))
    structure = report.structure

    assert structure.point_count == SCAN_COUNT * len(WORLD)
    assert (structure.scan_count, structure.scans_without_geometry) == (SCAN_COUNT, 0)
    assert structure.rejected_scan_count == 0
    assert structure.map_frame == MAP
    assert structure.source_frames == (FrameId("lidar"),)
    assert structure.points_checked == structure.point_count
    assert structure.all_map_coordinates_finite and structure.all_source_coordinates_finite
    assert structure.points_in_other_frame_count == 0
    assert (structure.measured_point_count, structure.aggregated_point_count) == (
        structure.point_count,
        0,
    )
    assert structure.lineage_problem_count == 0
    assert structure.dropped_non_finite_count == 0


def test_a_stride_checks_a_deterministic_subset_of_the_points(tmp_path: Path) -> None:
    protocol = dataclasses.replace(PROTOCOL, structure_point_stride=10)

    report = _evaluate(_correct(tmp_path), protocol=protocol)

    assert report.structure.points_checked == math.ceil(report.structure.point_count / 10)
    assert report.structure.point_count == SCAN_COUNT * len(WORLD)


def test_an_aggregated_map_reports_its_measured_and_aggregated_points(tmp_path: Path) -> None:
    reader = write_run(tmp_path, make_plan(), aggregation=ScanVoxelPolicy(cell_m=1.0))

    report = _evaluate(reader, expected_points=())

    assert report.structure.aggregated_point_count > 0
    assert (
        report.structure.measured_point_count + report.structure.aggregated_point_count
        == report.structure.point_count
    )


# --- Coordinate trace validation ---------------------------------------------------------------


def test_the_traced_points_are_spread_across_the_map_with_their_references(tmp_path: Path) -> None:
    report = _evaluate(_correct(tmp_path))

    entries = report.transform_trace.entries
    assert len(entries) == PROTOCOL.trace_sample_count
    assert {str(entry.source_observation_id) for entry in entries} >= {"scan-0000", "scan-0009"}
    assert all(entry.pose_reference and entry.calibration_reference for entry in entries)


def test_expected_and_produced_global_coordinates_are_reported_with_their_error(
    tmp_path: Path,
) -> None:
    report = _evaluate(_correct(tmp_path))

    check = report.expected_points[1]
    assert str(check.source_observation_id) == "scan-0003" and check.source_point_index == 400
    assert check.expected_map_coordinates_m == WORLD[400]
    assert check.produced_map_coordinates_m == pytest.approx(WORLD[400], abs=1e-9)
    assert check.error_m <= 1e-9 and check.within_tolerance


def test_an_expected_point_that_is_not_in_the_map_is_an_error(tmp_path: Path) -> None:
    missing = ExpectedPoint(
        source_observation_id=SourceObservationId("scan-0042"),
        source_point_index=0,
        expected_map_coordinates_m=(0.0, 0.0, 0.0),
    )

    with pytest.raises(GeometricMappingEvaluationError, match="scan-0042"):
        _evaluate(_correct(tmp_path), expected_points=(missing,))


def _run_dir(workspace: Path) -> Path:
    """Diretório onde `write_run` grava o primeiro run: o chamador o escolhe, sem glob."""
    return workspace / "run-0001"


def test_a_tampered_chain_is_seen_by_the_trace_and_by_the_integrity_check(tmp_path: Path) -> None:
    _correct(tmp_path).close()
    source_index = _run_dir(tmp_path) / "outputs" / "source-index.jsonl"
    lines = [json.loads(line) for line in source_index.read_text().splitlines()]
    lines[0]["transform_chain"][1]["translation_m"][0] += 0.25
    source_index.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines))

    report = _evaluate(GeometricMapArtifactReader(_run_dir(tmp_path)))

    assert not report.transform_trace.within_tolerance
    assert report.transform_trace.max_composition_error_m > 0.2
    assert any("source-index" in problem for problem in report.structure.integrity_problems)


# --- Regression tests for the common mistakes --------------------------------------------------


def _mistakes() -> dict[str, GeometryInputPlan]:
    inverted_extrinsic = invert_rigid(translation=TRUE_STATIC[0], rotation=TRUE_STATIC[1])
    return {
        "inverted_extrinsic": make_plan(static=inverted_extrinsic),
        "inverted_poses": make_plan(invert_poses=True),
        "pose_timing": make_plan(time_offset_ns=-30 * MS),
    }


@pytest.mark.parametrize("mistake", ["inverted_extrinsic", "inverted_poses", "pose_timing"])
def test_common_transform_and_timing_mistakes_are_caught(tmp_path: Path, mistake: str) -> None:
    good = _evaluate(_correct(tmp_path / "good"))
    bad = _evaluate(write_run(tmp_path / "bad", _mistakes()[mistake]))

    assert not all(check.within_tolerance for check in bad.expected_points)
    assert max(check.error_m for check in bad.expected_points) > 0.05
    assert good.overlap is not None and bad.overlap is not None
    assert good.overlap.pooled_residual_m is not None
    assert bad.overlap.pooled_residual_m is not None
    good_median = max(good.overlap.pooled_residual_m.median, 1e-12)
    assert bad.overlap.pooled_residual_m.median > 100 * good_median


@pytest.mark.parametrize("mistake", ["inverted_extrinsic", "inverted_poses"])
def test_a_wrong_transform_direction_also_breaks_the_agreement_between_scans(
    tmp_path: Path, mistake: str
) -> None:
    bad = _evaluate(write_run(tmp_path, _mistakes()[mistake]))

    assert bad.overlap is not None and bad.overlap.inconsistent_fraction is not None
    assert bad.overlap.inconsistent_fraction > 0.5


def test_a_time_error_that_moves_every_scan_alike_is_seen_by_the_expected_points_first(
    tmp_path: Path,
) -> None:
    # Com velocidade quase constante o erro de tempo desloca todos os scans do mesmo modo:
    # a consistência interna quase não muda, mas os pontos esperados acusam o erro absoluto.
    bad = _evaluate(write_run(tmp_path, make_plan(time_offset_ns=-30 * MS)))

    assert bad.overlap is not None and bad.overlap.inconsistent_fraction == 0.0
    assert max(check.error_m for check in bad.expected_points) > 0.05


def test_the_lookup_that_rejects_a_scan_is_visible_as_a_rejected_scan(tmp_path: Path) -> None:
    # Com a trajetória adiantada em 100 ms, o scan de t = 0 fica antes da primeira pose.
    reader = write_run(tmp_path, make_plan(time_offset_ns=100 * MS))

    report = _evaluate(reader, expected_points=())

    assert report.structure.rejected_scan_count == 1
    assert report.structure.scan_count == SCAN_COUNT - 1


# --- Scan/map coherence ------------------------------------------------------------------------


def test_density_is_the_distribution_of_points_per_occupied_voxel(tmp_path: Path) -> None:
    reader = _correct(tmp_path)
    voxel = PROTOCOL.density_voxel_m
    counts: dict[tuple[int, int, int], int] = {}
    for x, y, z in _all_points(reader):
        key = (math.floor(x / voxel), math.floor(y / voxel), math.floor(z / voxel))
        counts[key] = counts.get(key, 0) + 1

    report = _evaluate(reader)

    assert report.density.voxel_m == voxel
    assert report.density.occupied_voxel_count == len(counts)
    assert report.density.points_per_voxel is not None
    assert report.density.points_per_voxel.maximum == max(counts.values())
    assert report.density.points_per_voxel.count == len(counts)
    assert report.density.points_per_scan is not None
    assert report.density.points_per_scan.median == len(WORLD)


def test_the_sensor_range_distribution_comes_from_the_original_coordinates(
    tmp_path: Path,
) -> None:
    ranges = [math.hypot(*point) for index in range(SCAN_COUNT) for point in sensor_view(index)]
    protocol = dataclasses.replace(PROTOCOL, max_plausible_range_m=sorted(ranges)[len(ranges) // 2])

    report = _evaluate(_correct(tmp_path), protocol=protocol)

    assert report.sensor_range.distribution is not None
    assert report.sensor_range.distribution.maximum == pytest.approx(max(ranges), abs=1e-9)
    assert report.sensor_range.distribution.minimum == pytest.approx(min(ranges), abs=1e-9)
    limit = protocol.max_plausible_range_m
    assert limit is not None
    assert report.sensor_range.beyond_max_plausible_range_count == sum(
        1 for value in ranges if value > limit
    )


def test_bounds_and_the_points_outside_an_expected_box_are_reported(tmp_path: Path) -> None:
    reader = _correct(tmp_path)
    points = _all_points(reader)
    expected = Bounds3D(frame_id=MAP, minimum_m=(0.0, -1.0, -0.5), maximum_m=(10.0, 1.0, 3.0))
    protocol = dataclasses.replace(PROTOCOL, expected_bounds=expected)

    report = _evaluate(reader, protocol=protocol)

    assert report.bounds.bounds == Bounds3D.enclosing(points, frame_id=MAP)
    assert report.bounds.extent_m == pytest.approx(
        tuple(
            high - low
            for low, high in zip(
                report.bounds.bounds.minimum_m, report.bounds.bounds.maximum_m, strict=True
            )
        )
    )
    assert report.bounds.outside_expected_count == sum(
        1 for point in points if not expected.contains(point, frame_id=MAP)
    )


def test_coherence_is_off_when_no_pair_is_requested(tmp_path: Path) -> None:
    protocol = dataclasses.replace(PROTOCOL, coherence_pair_count=0)

    assert _evaluate(_correct(tmp_path), protocol=protocol).overlap is None


def test_overlap_is_measured_between_temporally_adjacent_scans(tmp_path: Path) -> None:
    report = _evaluate(_correct(tmp_path))

    assert report.overlap is not None
    assert report.overlap.lag == 1 and len(report.overlap.pairs) == PROTOCOL.coherence_pair_count
    first = report.overlap.pairs[0]
    assert (str(first.earlier_observation_id), str(first.later_observation_id)) == (
        "scan-0000",
        "scan-0001",
    )
    assert first.sample_count == PROTOCOL.coherence_points_per_scan
    assert 0.6 * first.sample_count < first.correspondence_count < first.sample_count
    assert first.motion_correction is MotionCorrectionState.UNKNOWN


def test_points_beyond_the_correspondence_radius_are_not_counted_as_residuals(
    tmp_path: Path,
) -> None:
    protocol = dataclasses.replace(PROTOCOL, max_correspondence_distance_m=0.01)
    reader = write_run(tmp_path, make_plan(invert_poses=True))

    report = _evaluate(reader, protocol=protocol, expected_points=())

    assert report.overlap is not None
    assert report.overlap.overlap_fraction < 0.5


def test_the_residuals_are_grouped_by_the_motion_correction_state_of_the_scan(
    tmp_path: Path,
) -> None:
    corrected = MotionCorrectionRecord(
        observation_id=SourceObservationId("scan-0000"), state=MotionCorrectionState.RAW
    )
    reader = write_run(
        tmp_path, make_plan(motion_correction={SourceObservationId("scan-0000"): corrected})
    )

    report = _evaluate(reader)

    assert report.overlap is not None
    groups = {group.state: group for group in report.overlap.by_motion_correction}
    assert set(groups) == {MotionCorrectionState.RAW, MotionCorrectionState.UNKNOWN}
    assert groups[MotionCorrectionState.RAW].pair_count == 1
    assert groups[MotionCorrectionState.UNKNOWN].pair_count == PROTOCOL.coherence_pair_count - 1


# --- Reference geometry ------------------------------------------------------------------------


def _reference(
    *, role: GeometryReferenceRole, shift: Vector3 = (0.0, 0.0, 0.0), frame: str = "map"
) -> ReferenceGeometry:
    return ReferenceGeometry(
        reference_id="reference-profile:corridor@1",
        role=role,
        frame_id=FrameId(frame),
        coordinates_m=tuple((x + shift[0], y + shift[1], z + shift[2]) for x, y, z in WORLD),
    )


def test_a_reference_that_matches_the_map_reports_no_distance_and_full_coverage(
    tmp_path: Path,
) -> None:
    report = _evaluate(
        _correct(tmp_path), reference=_reference(role=GeometryReferenceRole.EVALUATION_REFERENCE)
    )

    assert report.reference is not None
    assert report.reference.alignment == "none"
    assert report.reference.accuracy_m is not None and report.reference.accuracy_m.maximum <= 1e-9
    assert report.reference.completeness == 1.0


def test_a_reference_offset_from_the_map_shows_the_offset_and_is_never_aligned_away(
    tmp_path: Path,
) -> None:
    report = _evaluate(
        _correct(tmp_path),
        reference=_reference(
            role=GeometryReferenceRole.EVALUATION_REFERENCE, shift=(0.0, 0.3, 0.0)
        ),
    )

    assert report.reference is not None and report.reference.accuracy_m is not None
    assert report.reference.alignment == "none"
    assert 0.1 < report.reference.accuracy_m.median <= 0.3 + 1e-9


def test_a_point_cloud_is_not_a_reference_because_of_its_file_name_or_format(
    tmp_path: Path,
) -> None:
    with pytest.raises(GeometricMappingEvaluationError, match="declared"):
        _evaluate(
            _correct(tmp_path), reference=_reference(role=GeometryReferenceRole.UNVERIFIED_PRODUCT)
        )


def test_a_reference_in_another_frame_is_refused_instead_of_assumed_aligned(
    tmp_path: Path,
) -> None:
    with pytest.raises(GeometricMappingEvaluationError, match="frame"):
        _evaluate(
            _correct(tmp_path),
            reference=_reference(role=GeometryReferenceRole.EVALUATION_REFERENCE, frame="odom"),
        )


def test_a_reference_needs_an_identity() -> None:
    with pytest.raises(ValueError, match="reference_id"):
        ReferenceGeometry(
            reference_id="",
            role=GeometryReferenceRole.EVALUATION_REFERENCE,
            frame_id=MAP,
            coordinates_m=((0.0, 0.0, 0.0),),
        )


# --- Reproducibility and persistence -----------------------------------------------------------


def _in_memory(plan: GeometryInputPlan) -> PackedGeometry:
    sink = io.BytesIO()
    accumulated = accumulate_plan(
        plan,
        map_id=MapId(f"{SEQUENCE}--run-0001"),
        sink=sink,
        aggregation=None,
        configuration_fingerprint=mapping_configuration_fingerprint(plan, None),
        code_version="test",
    )
    return PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=accumulated.scans, records=sink.getvalue()
    )


def test_a_persisted_map_reopens_to_the_geometry_that_was_built(tmp_path: Path) -> None:
    plan = make_plan()
    reader = write_run(tmp_path, plan)
    probes = [
        Bounds3D(frame_id=MAP, minimum_m=(0.0, -2.0, 0.0), maximum_m=(3.0, 0.0, 1.0)),
        Bounds3D(frame_id=MAP, minimum_m=(5.0, 0.0, 0.0), maximum_m=(9.0, 2.0, 3.0)),
    ]

    with reader:
        report = evaluate_round_trip(
            expected=_in_memory(plan), reopened=reader.geometry(), probe_boxes=probes
        )

    assert report.points_compared == SCAN_COUNT * len(WORLD)
    assert (report.coordinate_mismatch_count, report.provenance_mismatch_count) == (0, 0)
    assert report.map_metadata_equal and report.bounds_equal
    assert (report.queries_compared, report.query_mismatch_count) == (2, 0)


def test_a_round_trip_that_drifts_is_counted_not_ignored(tmp_path: Path) -> None:
    reader = write_run(tmp_path, make_plan())
    drifted = _in_memory(make_plan(time_offset_ns=-30 * MS))

    with reader:
        report = evaluate_round_trip(expected=drifted, reopened=reader.geometry(), probe_boxes=[])

    assert report.coordinate_mismatch_count > 0 and not report.map_metadata_equal


def test_identical_inputs_reproduce_identical_contractual_files(tmp_path: Path) -> None:
    first = write_run(tmp_path / "a", make_plan())
    second = write_run(tmp_path / "b", make_plan())

    with first, second:
        report = compare_contractual_inventories(first.manifest, second.manifest)

    assert report.identical and report.differing_paths == ()


def test_a_changed_configuration_changes_the_contractual_files(tmp_path: Path) -> None:
    first = write_run(tmp_path / "a", make_plan())
    second = write_run(tmp_path / "b", make_plan(), aggregation=ScanVoxelPolicy(cell_m=1.0))

    with first, second:
        report = compare_contractual_inventories(first.manifest, second.manifest)

    assert not report.identical
    assert (
        "outputs/geometry.bin" in report.differing_paths and "config.json" in report.differing_paths
    )


def test_the_evaluation_never_needs_the_debug_directory(tmp_path: Path) -> None:
    reader = write_run(tmp_path, make_plan())

    report = _evaluate(reader)

    assert not any(entry.path.startswith("debug/") for entry in reader.manifest.file_inventory)
    assert report.structure.integrity_problems == ()


# --- Identity, cost and encoding ---------------------------------------------------------------


def test_the_report_carries_the_identities_needed_to_reproduce_it(tmp_path: Path) -> None:
    plan = make_plan()
    report = _evaluate(write_run(tmp_path, plan))

    assert report.evaluator_version
    assert report.run_id == "run-0001" and report.map_id == f"{SEQUENCE}--run-0001"
    assert report.sequence_artifact_id == SEQUENCE_ID
    assert report.selection_id == plan.selection_id
    assert report.trajectory_id == plan.trajectory_id
    assert report.state_estimation_run_id == "run-0001"
    assert report.calibration_identity == plan.calibration_identity
    assert report.configuration_fingerprint == mapping_configuration_fingerprint(plan, None)
    assert report.protocol == PROTOCOL


def test_runtime_is_reported_apart_from_geometric_quality(tmp_path: Path) -> None:
    report = _evaluate(write_run(tmp_path, make_plan(), runtime_s=2.5))
    encoded = encode_geometric_mapping_report(report)

    assert report.cost.runtime_s == 2.5 and report.cost.point_count == SCAN_COUNT * len(WORLD)
    assert "runtime_s" not in json.dumps({k: v for k, v in encoded.items() if k != "cost"})


def test_the_encoded_report_is_plain_json_and_deterministic(tmp_path: Path) -> None:
    reader = _correct(tmp_path)
    first = _evaluate(reader)
    second = _evaluate(GeometricMapArtifactReader(_run_dir(tmp_path)))

    assert first == second
    encoded = encode_geometric_mapping_report(first)
    assert json.loads(json.dumps(encoded)) == encoded
    assert encoded["structure"]["point_count"] == first.structure.point_count
    assert encoded["protocol"]["density_voxel_m"] == PROTOCOL.density_voxel_m


# --- Controlled comparison ---------------------------------------------------------------------


def test_reports_of_the_same_inputs_compare_while_keeping_each_identity(tmp_path: Path) -> None:
    good = _evaluate(_correct(tmp_path / "good"))
    bad = _evaluate(write_run(tmp_path / "bad", make_plan(invert_poses=True)))

    comparison = compare_geometric_mapping_reports([good, bad])

    assert [entry.configuration_fingerprint for entry in comparison.entries] == [
        good.configuration_fingerprint,
        bad.configuration_fingerprint,
    ]
    good_entry, bad_entry = comparison.entries
    assert good_entry.median_overlap_residual_m is not None
    assert bad_entry.median_overlap_residual_m is not None
    assert good_entry.median_overlap_residual_m < bad_entry.median_overlap_residual_m
    assert good_entry.max_expected_point_error_m is not None
    assert bad_entry.max_expected_point_error_m is not None
    assert good_entry.max_expected_point_error_m < bad_entry.max_expected_point_error_m


def test_a_comparison_rejects_drift_in_anything_but_the_mapping_configuration(
    tmp_path: Path,
) -> None:
    good = _evaluate(_correct(tmp_path))

    for change in (
        {"selection_id": "another-selection"},
        {"calibration_identity": "sha256:other"},
        {"protocol": dataclasses.replace(PROTOCOL, density_voxel_m=0.7)},
        {"evaluator_version": "0"},
    ):
        with pytest.raises(GeometricMappingEvaluationError, match="differ"):
            compare_geometric_mapping_reports([good, dataclasses.replace(good, **change)])
    with pytest.raises(GeometricMappingEvaluationError, match="at least two"):
        compare_geometric_mapping_reports([good])


# --- Real data (optional) --------------------------------------------------------------------

_DATASET = Path(
    os.environ.get(
        "CONTEXTMAP_CORRIDOR02_DIR",
        Path(__file__).resolve().parents[2] / "datasets" / "corridor-02",
    )
)
_REAL_CLOCK = "corridor-02:header"
_REAL_SEGMENT_START_S = 170.0
_REAL_SCAN_COUNT = 30


def _quaternion_from_rotation(matrix: list[list[float]]) -> tuple[float, float, float, float]:
    """Unit quaternion of the upper-left 3x3 block of a rigid matrix (trace-based)."""
    r = matrix
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = ((r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s)
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        q = (0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s)
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        q = ((r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s)
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        q = ((r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s, (r[1][0] - r[0][1]) / s)
    return normalize_quaternion(q)


@pytest.mark.skipif(
    not (_DATASET / "corridor-02.bag").is_file(),
    reason="corridor-02 dataset is not available (it is not versioned)",
)
def test_a_real_segment_maps_persists_and_validates_end_to_end(tmp_path: Path) -> None:
    from decimal import Decimal

    from contextmap.geometric_mapping import (
        GeometricMapArtifactWriter,
        MotionCorrectionPolicy,
        ScanDisposition,
        assemble_geometry_inputs,
    )
    from contextmap.ingestion import (
        CalibrationSet,
        ExternalPoseMeasurement,
        FullSequenceSelection,
        RigidTransform,
        SensorId,
        SequenceArtifactId,
        SequenceSelectionResult,
        SourceAdapterConfig,
        SourceProvenance,
        SourceTopicMapping,
        selection_identity,
    )
    from contextmap.ingestion.adapters.ros1_bag import Ros1BagSourceAdapter
    from contextmap.shared import SourceTimestamp
    from contextmap.state_estimation import (
        LookupPolicy,
        StateEstimationRequest,
        StateEstimationRunId,
        TrajectoryId,
    )
    from contextmap.state_estimation.backends.external_pose import (
        ExternalPoseConfig,
        ExternalPoseEstimator,
    )

    gt_lines = (_DATASET / "corridor-02-gt.txt").read_text().splitlines()
    gt_start_ns = int(Decimal(gt_lines[0].split()[0]) * 1_000_000_000)
    adapter = Ros1BagSourceAdapter(
        SourceAdapterConfig(
            source_type="ros1_bag",
            path=str(_DATASET / "corridor-02.bag"),
            topics=SourceTopicMapping(lidar="/velodyne_points"),
            timestamp_clock_id=_REAL_CLOCK,
        )
    )
    scans: list[SourceObservation] = []
    for observation in adapter.read_observations():
        if (observation.timestamp.total_nanoseconds() - gt_start_ns) / 1e9 >= _REAL_SEGMENT_START_S:
            scans.append(observation)
            if len(scans) == _REAL_SCAN_COUNT:
                break
    assert len(scans) == _REAL_SCAN_COUNT

    # laser_to_imu é T_imu_laser; o corpo do GT é o IMU.
    text = (_DATASET / "corridor-02-extrinsics.yaml").read_text()
    block = text.split("laser_to_imu:")[1].split("data:")[1].split("]")[0]
    numbers = [
        float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", block.replace("[", ""))
    ]
    rows = [numbers[0:4], numbers[4:8], numbers[8:12]]
    body, lidar = FrameId("epson"), scans[0].frame_id
    calibration = CalibrationSet(
        entries={},
        static_transforms=(
            RigidTransform(
                parent_frame=body,
                child_frame=lidar,
                translation=(rows[0][3], rows[1][3], rows[2][3]),
                rotation=_quaternion_from_rotation([row[:3] for row in rows]),
            ),
        ),
    )
    first_ns, last_ns = (
        scans[0].timestamp.total_nanoseconds(),
        scans[-1].timestamp.total_nanoseconds(),
    )
    poses = []
    for index, line in enumerate(gt_lines):
        stamp, x, y, z, qx, qy, qz, qw = line.split()
        total_ns = int(Decimal(stamp) * 1_000_000_000)
        if not first_ns - 3_000_000_000 <= total_ns <= last_ns + 3_000_000_000:
            continue
        seconds, nanoseconds = divmod(total_ns, 1_000_000_000)
        poses.append(
            ExternalPoseMeasurement(
                observation_id=SourceObservationId(f"gt-{index:05d}"),
                sensor_id=SensorId("gt"),
                frame_id=body,
                timestamp=SourceTimestamp(
                    seconds=seconds, nanoseconds=nanoseconds, clock_id=_REAL_CLOCK
                ),
                provenance=SourceProvenance(
                    source_type="dataset", source_path="corridor-02-gt.txt"
                ),
                parent_frame=MAP,
                translation=(float(x), float(y), float(z)),
                orientation=normalize_quaternion((float(qx), float(qy), float(qz), float(qw))),
            )
        )
    sequence_id = SequenceArtifactId("corridor-02-segment")
    trajectory = (
        ExternalPoseEstimator(
            ExternalPoseConfig(reference_frame=MAP, body_frame=body, max_gap_ns=500_000_000)
        )
        .estimate(
            StateEstimationRequest(
                trajectory_id=TrajectoryId("gt-window"),
                sequence_artifact_id=sequence_id,
                selection_id="segment",
                observations=tuple(poses),
                calibration=calibration,
            )
        )
        .trajectory
    )
    selection = FullSequenceSelection()
    plan = assemble_geometry_inputs(
        sequence=SequenceSelectionResult(
            sequence_artifact_id=sequence_id,
            selection=selection,
            selection_id=selection_identity(sequence_id, selection),
            observations=tuple(scans),
        ),
        calibration=calibration,
        trajectory=trajectory,
        pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=400_000_000),
        motion_correction_policy=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN
        ),
        state_estimation_run_id=StateEstimationRunId("gt-window"),
    )
    assert plan.inputs

    run_dir = tmp_path / "geometric_mapping"
    GeometricMapArtifactWriter(
        output_dir=run_dir,
        sequence_name="corridor-02-segment",
        run_id=GeometricMapRunId("run-0001"),
        run_index=1,
    ).finalize(plan=plan, aggregation=None, code_version="test")
    protocol = dataclasses.replace(PROTOCOL, structure_point_stride=25, max_plausible_range_m=150.0)
    with GeometricMapArtifactReader(run_dir) as reader:
        report = evaluate_geometric_mapping(reader=reader, protocol=protocol)

    assert report.structure.integrity_problems == ()
    assert (
        report.structure.all_map_coordinates_finite
        and report.structure.all_source_coordinates_finite
    )
    assert report.structure.lineage_problem_count == 0
    assert report.transform_trace.max_composition_error_m <= 1e-6
    assert report.overlap is not None and report.overlap.pooled_residual_m is not None
    assert report.sensor_range.beyond_max_plausible_range_count == 0
