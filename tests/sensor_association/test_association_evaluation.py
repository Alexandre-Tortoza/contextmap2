import dataclasses
import json
from pathlib import Path

import pytest
from run_builders import (
    ENHANCED,
    NATIVE,
    OCCLUSION,
    frame_input,
    make_request,
    make_strata_request,
)

from contextmap.evaluation.sensor_association import (
    EVALUATOR_VERSION,
    SensorAssociationEvaluationError,
    SensorAssociationEvaluationReport,
    StratificationProfile,
    compare_sensor_association_reports,
    encode_sensor_association_comparison,
    encode_sensor_association_report,
    evaluate_sensor_association,
)
from contextmap.sensor_association import (
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
)

PROFILE = StratificationProfile(
    range_edges_m=(3.0, 8.0),
    visible_share_edges=(0.99,),
    support_density_edges=(0.001,),
    border_distance_edges_px=(50.0,),
    viewing_angle_edges_rad=(0.5,),
)


def _run(
    root: Path, request: SensorAssociationRequest, *, index: int = 1
) -> SensorAssociationRunReader:
    """Grava em ``root/run-NNNN``: o chamador decide o diretório final, o writer não calcula."""
    outcome = SensorAssociationService().run(request)
    run_dir = root / f"run-{index:04d}"
    SensorAssociationRunWriter(
        output_dir=run_dir,
        sequence_name="fixture",
        run_id=SensorAssociationRunId(f"assoc-run-{index:04d}"),
        run_index=index,
    ).finalize(outcome)
    return SensorAssociationRunReader(run_dir)


def _report(root: Path, *channels: object) -> SensorAssociationEvaluationReport:
    reader = _run(root, make_strata_request(channels=list(channels)))  # type: ignore[arg-type]
    return evaluate_sensor_association(reader, profile=PROFILE)


def _counts(
    report: SensorAssociationEvaluationReport, dimension: str
) -> dict[str, tuple[int, int]]:
    (stratification,) = [s for s in report.strata if s.dimension == dimension]
    counts = {
        s.band: (s.observation_count, s.physical_observation_count) for s in stratification.strata
    }
    counts["unavailable"] = (
        stratification.unavailable.observation_count,
        stratification.unavailable.physical_observation_count,
    )
    return counts


# --- Lineage ----------------------------------------------------------------


def test_the_report_keeps_the_complete_lineage_of_the_run(tmp_path: Path) -> None:
    report = _report(tmp_path, NATIVE, ENHANCED)
    lineage = report.lineage

    assert report.evaluator_version == EVALUATOR_VERSION == "1"
    assert lineage.run_id == "assoc-run-0001"
    assert lineage.sequence_artifact_id == "sequence-0001"
    assert lineage.selection_id == "full-sequence"
    assert lineage.geometric_map_id == "map-0001"
    assert lineage.calibration_identity.startswith("sha256:")
    assert lineage.perception_run_ids == ("run-0001",)
    assert lineage.visibility_policy["fingerprint"] == OCCLUSION.fingerprint()
    assert lineage.membership_policy_id == "mask-membership-v1"
    assert lineage.configuration_fingerprint.startswith("sha256:")
    assert lineage.code_version == "test"
    assert lineage.pose_policy["mode"] == "exact"
    assert lineage.tolerances["max_pose_time_delta_ns"] == 20_000_000
    assert {path.channel_id for path in report.feature_paths} == {"dino-native", "dino-enhanced"}
    assert all(path.feature_sources for path in report.feature_paths)


# --- Stratification ---------------------------------------------------------


def test_the_range_strata_partition_every_observation_and_count_physical_frames(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)

    counts = _counts(report, "range")
    # Cada frame tem R1 e R1b (perto), R2 (meio), R3 (longe) e R4 sem geometria.
    assert counts == {
        "(-inf, 3.0)": (4, 2),
        "[3.0, 8.0)": (2, 2),
        "[8.0, +inf)": (2, 2),
        "unavailable": (2, 2),
    }
    assert sum(o for o, _ in counts.values()) == report.observation_count == 10
    assert report.physical_observation_count == 2


def test_several_regions_of_one_physical_frame_are_not_several_physical_observations(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)

    (range_report,) = [s for s in report.strata if s.dimension == "range"]
    near = range_report.strata[0]
    assert near.observation_count == 4 and near.physical_observation_count == 2
    assert report.observation_count == 5 * report.physical_observation_count


def test_the_visibility_strata_use_explicit_denominators(tmp_path: Path) -> None:
    report = _report(tmp_path)

    (visibility,) = [s for s in report.strata if s.dimension == "visibility"]
    low, high = visibility.strata
    assert (low.observation_count, low.associated_count, low.footprint_count) == (2, 4, 6)
    assert low.pooled_visible_share == pytest.approx(4 / 6)
    assert (high.observation_count, high.associated_count, high.footprint_count) == (6, 14, 14)
    assert high.pooled_visible_share == 1.0
    assert visibility.unavailable.observation_count == 2
    assert visibility.unavailable.pooled_visible_share is None


def test_a_measured_zero_density_is_a_band_and_never_an_unavailable_measurement(
    tmp_path: Path,
) -> None:
    counts = _counts(_report(tmp_path), "support_density")

    # R4 não tem geometria associada: densidade 0.0 medida, na faixa mais baixa com R2.
    assert counts == {"(-inf, 0.001)": (4, 2), "[0.001, +inf)": (6, 2), "unavailable": (0, 0)}


def test_the_image_border_and_viewing_angle_strata_separate_the_periphery(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert _counts(report, "image_border") == {
        "(-inf, 50.0)": (2, 2),
        "[50.0, +inf)": (6, 2),
        "unavailable": (2, 2),
    }
    assert _counts(report, "viewing_angle") == {
        "(-inf, 0.5)": (6, 2),
        "[0.5, +inf)": (2, 2),
        "unavailable": (2, 2),
    }


def test_a_band_edge_belongs_to_the_upper_band(tmp_path: Path) -> None:
    profile = dataclasses.replace(PROFILE, range_edges_m=(2.2,))
    reader = _run(tmp_path, make_strata_request())

    report = evaluate_sensor_association(reader, profile=profile)

    # A mediana de R1 e R1b é exatamente 2.2 (a do meio das profundidades 2.0, 2.2, 2.4).
    (range_report,) = [s for s in report.strata if s.dimension == "range"]
    assert [s.observation_count for s in range_report.strata] == [0, 8]


def test_an_empty_edge_list_is_a_single_band(tmp_path: Path) -> None:
    profile = StratificationProfile(
        range_edges_m=(),
        visible_share_edges=(),
        support_density_edges=(),
        border_distance_edges_px=(),
        viewing_angle_edges_rad=(),
    )

    report = evaluate_sensor_association(_run(tmp_path, make_strata_request()), profile=profile)

    (range_report,) = [s for s in report.strata if s.dimension == "range"]
    assert [s.band for s in range_report.strata] == ["(-inf, +inf)"]
    assert range_report.strata[0].observation_count == 8


def test_the_profile_is_validated() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        dataclasses.replace(PROFILE, range_edges_m=(8.0, 3.0))
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(PROFILE, visible_share_edges=(float("nan"),))


# --- Feature paths, timing and reprojection ---------------------------------


def test_native_and_enhanced_feature_paths_are_reported_side_by_side(tmp_path: Path) -> None:
    report = _report(tmp_path, NATIVE, ENHANCED)

    native, enhanced = {p.channel_id: p for p in report.feature_paths}.values()
    assert (native.eligible_point_count, native.sampled_point_count) == (14, 14)
    assert native.sampled_ratio == 1.0
    # A bilinear no canto da imagem fica fora do domínio: o ponto (3, 3) de cada frame.
    assert (enhanced.eligible_point_count, enhanced.sampled_point_count) == (14, 12)
    assert enhanced.out_of_support_count == 2
    assert enhanced.sampled_ratio == pytest.approx(6 / 7)
    assert native.interpolation == "nearest" and enhanced.interpolation == "bilinear"


def test_the_timing_and_the_state_counts_are_reported(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report.timing.pose_time_delta_ns is not None
    assert report.timing.pose_time_delta_ns.maximum == 0
    assert dict(report.timing.lookup_outcomes) == {"exact": 2}
    assert report.state_counts["visible"] == 14
    assert report.state_counts["occluded"] == 2


def test_the_reprojection_is_reported_only_where_a_trusted_reference_exists(tmp_path: Path) -> None:
    without = evaluate_sensor_association(_run(tmp_path, make_request(), index=1), profile=PROFILE)
    request = make_request(frames=[frame_input(0, with_reference=True), frame_input(1)])
    with_reference = evaluate_sensor_association(_run(tmp_path, request, index=2), profile=PROFILE)

    assert (without.reprojection.frames_with_reference, without.reprojection.frame_median_px) == (
        0,
        None,
    )
    assert without.reprojection.frames_without_reference == 2
    residual = with_reference.reprojection
    assert (residual.frames_with_reference, residual.frames_without_reference) == (1, 1)
    assert residual.correspondence_count == 4 and residual.invalid_correspondence_count == 0
    assert residual.frame_median_px is not None
    assert residual.frame_median_px.median == pytest.approx(1.0)


# --- Determinism and separation from semantic confidence --------------------


def test_the_report_is_deterministic_and_json(tmp_path: Path) -> None:
    reader = _run(tmp_path, make_strata_request(channels=[NATIVE]))

    first = evaluate_sensor_association(reader, profile=PROFILE)
    second = evaluate_sensor_association(reader, profile=PROFILE)
    record = json.loads(json.dumps(encode_sensor_association_report(first)))

    assert first == second
    assert record["evaluator_version"] == "1"
    assert record["lineage"]["configuration_fingerprint"] == first.lineage.configuration_fingerprint
    assert record["physical_observation_count"] == 2
    assert record["strata"][0]["dimension"] == "range"
    assert record["feature_paths"][0]["sampled_ratio"] == 1.0


def test_the_report_has_no_semantic_confidence_or_combined_score(tmp_path: Path) -> None:
    record = encode_sensor_association_report(_report(tmp_path, NATIVE))
    names: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                names.add(str(key))
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(record)

    assert not names & {"confidence", "score", "probability", "weight", "similarity", "label"}
    assert not any("confidence" in name or "score" in name for name in names)


def test_evaluating_a_run_never_modifies_it(tmp_path: Path) -> None:
    reader = _run(tmp_path, make_strata_request(channels=[NATIVE]))
    before = {e.path: e.content_hash for e in reader.manifest.file_inventory}

    evaluate_sensor_association(reader, profile=PROFILE)

    assert reader.verify_integrity() == []
    assert {e.path: e.content_hash for e in reader.manifest.file_inventory} == before


def test_a_corrupted_run_is_not_evaluated(tmp_path: Path) -> None:
    reader = _run(tmp_path, make_strata_request())
    run_dir = tmp_path / "run-0001"
    (run_dir / "outputs/geometry-support.u32").write_bytes(b"")

    with pytest.raises(SensorAssociationEvaluationError, match="not intact"):
        evaluate_sensor_association(reader, profile=PROFILE)


# --- Controlled comparison --------------------------------------------------


def test_native_and_enhanced_runs_compare_while_geometry_and_calibration_stay_constant(
    tmp_path: Path,
) -> None:
    native = evaluate_sensor_association(
        _run(tmp_path, make_strata_request(channels=[NATIVE]), index=1),
        profile=PROFILE,
    )
    enhanced = evaluate_sensor_association(
        _run(tmp_path, make_strata_request(channels=[ENHANCED]), index=2),
        profile=PROFILE,
    )

    comparison = compare_sensor_association_reports([native, enhanced])

    first, second = comparison.entries
    assert (first.run_id, second.run_id) == ("assoc-run-0001", "assoc-run-0002")
    assert first.configuration_fingerprint != second.configuration_fingerprint
    assert first.feature_paths[0].sampled_ratio == 1.0
    assert second.feature_paths[0].sampled_ratio == pytest.approx(6 / 7)
    assert comparison.lineage.geometric_map_id == "map-0001"
    record = json.loads(json.dumps(encode_sensor_association_comparison(comparison)))
    assert [e["run_id"] for e in record["entries"]] == ["assoc-run-0001", "assoc-run-0002"]


def test_a_comparison_rejects_anything_but_the_feature_path_changing(tmp_path: Path) -> None:
    base = evaluate_sensor_association(
        _run(tmp_path, make_strata_request(channels=[NATIVE]), index=1),
        profile=PROFILE,
    )
    wider = dataclasses.replace(OCCLUSION, neighborhood_radius_cells=3)
    other_policy = evaluate_sensor_association(
        _run(
            tmp_path,
            dataclasses.replace(make_strata_request(channels=[NATIVE]), occlusion_policy=wider),
            index=2,
        ),
        profile=PROFILE,
    )
    other_scene = evaluate_sensor_association(
        _run(tmp_path, make_request(channels=[NATIVE]), index=3), profile=PROFILE
    )
    other_profile = evaluate_sensor_association(
        _run(tmp_path, make_strata_request(channels=[NATIVE]), index=4),
        profile=dataclasses.replace(PROFILE, range_edges_m=(4.0,)),
    )

    with pytest.raises(SensorAssociationEvaluationError, match="visibility_policy"):
        compare_sensor_association_reports([base, other_policy])
    with pytest.raises(SensorAssociationEvaluationError, match="geometry-side"):
        compare_sensor_association_reports([base, other_scene])
    with pytest.raises(SensorAssociationEvaluationError, match="profile"):
        compare_sensor_association_reports([base, other_profile])
    with pytest.raises(SensorAssociationEvaluationError, match="at least two"):
        compare_sensor_association_reports([base])
