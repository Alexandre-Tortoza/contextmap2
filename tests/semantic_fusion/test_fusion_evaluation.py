import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fusion_eval_fixtures import PROFILE, EvalFixture, make_eval_fixture

from contextmap.evaluation import (
    FusionArmRole,
    FusionStratificationProfile,
    ReferenceAnnotation,
    SemanticFusionEvaluationError,
    SemanticFusionEvaluationReport,
    compare_semantic_fusion_reports,
    encode_semantic_fusion_comparison,
    encode_semantic_fusion_report,
    evaluate_semantic_fusion,
)
from contextmap.semantic_fusion import SemanticFusionRunReader

FORBIDDEN_KEYS = {
    "score",
    "winner",
    "best",
    "rank",
    "ranking",
    "overall",
    "composite",
    "probability",
    "confidence",
}


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> EvalFixture:
    return make_eval_fixture(tmp_path_factory.mktemp("fusion-eval"))


def _report(
    fixture: EvalFixture,
    arm: str,
    *,
    annotations: bool = True,
    qualities: bool = True,
    profile: FusionStratificationProfile = PROFILE,
) -> SemanticFusionEvaluationReport:
    return evaluate_semantic_fusion(
        SemanticFusionRunReader(fixture.runs[arm]),
        arm_id=arm,
        arm_role=fixture.roles[arm],
        profile=profile,
        annotations=fixture.annotations if annotations else None,
        qualities=fixture.qualities if qualities else None,
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _stratum(report: SemanticFusionEvaluationReport, dimension: str, band: str) -> Any:
    stratification = next(s for s in report.strata if s.dimension == dimension)
    return next(s for s in (*stratification.strata, stratification.unavailable) if s.band == band)


def test_the_report_carries_the_complete_lineage_of_the_run(fixture: EvalFixture) -> None:
    report = _report(fixture, "quality_aware")

    lineage = report.lineage
    assert report.arm_id == "quality_aware"
    assert report.arm_role is FusionArmRole.QUALITY_AWARE
    assert lineage.run_id == "fusion-run-0002"
    assert lineage.sequence_name == "sequence-0001"
    assert lineage.geometric_map_id == "map-0001"
    assert lineage.perception_run_ids == ("run-a", "run-b")
    assert lineage.association_run_ids == ("association-run-0001",)
    assert lineage.support_policy_id == "geometry-jaccard-support-v1"
    assert lineage.fusion_policy_id == "quality-aware-evidence-accumulation-v1"
    assert (lineage.fusion_configuration_fingerprint or "").startswith("sha256:")
    assert report.evaluator_version == "1"
    assert report.profile == PROFILE


def test_repeated_inference_is_reported_apart_from_physical_observations(
    fixture: EvalFixture,
) -> None:
    report = _report(fixture, "baseline")

    assert report.support_count == 4
    assert report.physical_observation_count == 8
    assert report.inference_result_count == 9
    correlation = report.correlation
    assert correlation.supports_with_repeated_inference == 1
    assert correlation.max_inference_results_per_physical_observation == 2
    assert correlation.hypotheses_with_repeated_inference_support == 1


def test_no_correlation_or_duplication_violation_exists_in_a_valid_run(
    fixture: EvalFixture,
) -> None:
    correlation = _report(fixture, "baseline").correlation

    assert correlation.supporting_exceeds_physical == 0
    assert correlation.duplicated_evidence_items == 0
    assert correlation.evidence_items_exceeding_claims == 0
    assert correlation.duplicated_structure_refs == 0


def test_static_structure_is_counted_once_and_never_per_view(fixture: EvalFixture) -> None:
    report = _report(fixture, "with_visual_and_3d")

    structure = next(c for c in report.channels if c.channel == "point_representation")
    assert structure.active
    assert structure.data_items == 2
    assert report.correlation.duplicated_structure_refs == 0


def test_uncertainty_is_retained_and_reported_by_kind(fixture: EvalFixture) -> None:
    uncertainty = _report(fixture, "baseline").uncertainty

    assert uncertainty.supports_with_hypotheses == 3
    assert uncertainty.supports_with_multiple_hypotheses == 2
    assert uncertainty.unreported_competition == 0
    assert dict(uncertainty.supports_by_kind) == {
        "ambiguity": 1,
        "contradiction": 1,
        "insufficient_evidence": 1,
        "near_tie": 1,
    }
    assert uncertainty.abstaining_claims == 1


def test_scored_unscored_and_hypothesis_less_claims_are_counted_separately(
    fixture: EvalFixture,
) -> None:
    uncertainty = _report(fixture, "baseline").uncertainty

    assert uncertainty.scored_claims == 5
    assert uncertainty.unscored_claims == 4
    assert uncertainty.claims_without_hypothesis == 1


def test_each_channel_reports_whether_it_was_active_and_what_fed_it(fixture: EvalFixture) -> None:
    baseline = {c.channel: c for c in _report(fixture, "baseline").channels}
    scores = {c.channel: c for c in _report(fixture, "with_scores").channels}
    visual = {c.channel: c for c in _report(fixture, "with_visual").channels}
    quality = {c.channel: c for c in _report(fixture, "quality_aware").channels}

    assert baseline["semantic_scores"].active is False
    assert baseline["semantic_claims"].data_items == 10
    assert scores["semantic_scores"].active and scores["semantic_scores"].data_items == 1
    assert scores["semantic_scores"].identities == ("clip_scorer/clip-vit-l14/1",)
    assert visual["visual_features"].data_items == 12
    assert visual["visual_features"].identities == ("clip-vit-l14", "dinov3-vit-b16")
    assert quality["observation_quality"].data_items == 9
    assert quality["observation_quality"].identities == ("observation-quality-v1",)


def test_the_weighting_is_reported_only_for_a_quality_aware_arm(fixture: EvalFixture) -> None:
    assert _report(fixture, "baseline").weighting is None
    assert _report(fixture, "with_scores").weighting is None

    weighting = _report(fixture, "quality_aware").weighting

    assert weighting is not None
    assert weighting.contributions_weighted == 9
    assert weighting.zero_factor_contributions == 2
    assert weighting.neutral_fallback_components == {}
    assert weighting.factor is not None
    assert weighting.factor.minimum == 0.0
    assert weighting.factor.maximum == 1.0
    assert weighting.supports_where_leading_changes == 2


def test_reference_recovery_is_reported_with_an_explicit_status(fixture: EvalFixture) -> None:
    annotations = _report(fixture, "baseline").annotations

    assert annotations is not None
    assert annotations.annotated_supports == 3
    assert annotations.unannotated_supports == 1
    assert annotations.ambiguous_reference == 0
    assert annotations.reference_recovered == 3
    assert annotations.reference_missing == 0
    assert annotations.leading_matches_reference == 1
    assert annotations.tied_with_reference == 1
    assert annotations.retained_not_leading == 1
    assert annotations.weighted is None


def test_a_weighted_arm_reports_the_weighted_leader_beside_the_unweighted_one(
    fixture: EvalFixture,
) -> None:
    annotations = _report(fixture, "quality_aware").annotations

    assert annotations is not None
    assert annotations.leading_matches_reference == 1
    assert annotations.tied_with_reference == 1
    assert annotations.weighted is not None
    assert annotations.weighted.leading_matches_reference == 2
    assert annotations.weighted.tied_with_reference == 1
    assert annotations.weighted.retained_not_leading == 0


def test_a_missing_annotation_is_not_applicable_and_never_a_negative(fixture: EvalFixture) -> None:
    report = _report(fixture, "baseline")
    annotations = report.annotations

    assert annotations is not None
    assert annotations.annotated_supports + annotations.unannotated_supports == report.support_count
    assert _report(fixture, "baseline", annotations=False).annotations is None


def test_annotations_that_disagree_inside_one_support_are_not_a_reference(
    fixture: EvalFixture,
) -> None:
    from fusion_builders import spatial_id

    conflicting = (
        *fixture.annotations,
        ReferenceAnnotation(
            spatial_observation_id=spatial_id("run-a", "frame-0121"), label="wardrobe"
        ),
    )

    report = evaluate_semantic_fusion(
        SemanticFusionRunReader(fixture.runs["baseline"]),
        arm_id="baseline",
        arm_role=FusionArmRole.BASELINE_CONTROL,
        profile=PROFILE,
        annotations=conflicting,
        qualities=fixture.qualities,
    )

    assert report.annotations is not None
    assert report.annotations.ambiguous_reference == 1
    assert report.annotations.annotated_supports == 2
    assert report.annotations.reference_recovered == 2


def test_an_annotation_of_an_observation_in_no_support_is_reported_as_unmatched(
    fixture: EvalFixture,
) -> None:
    from contextmap.sensor_association import SpatialObservationId

    stray = ReferenceAnnotation(
        spatial_observation_id=SpatialObservationId("spatial--elsewhere--region-0001"), label="x"
    )

    report = evaluate_semantic_fusion(
        SemanticFusionRunReader(fixture.runs["baseline"]),
        arm_id="baseline",
        arm_role=FusionArmRole.BASELINE_CONTROL,
        profile=PROFILE,
        annotations=(*fixture.annotations, stray),
        qualities=None,
    )

    assert report.annotations is not None
    assert report.annotations.unmatched_annotations == 1


def test_every_support_lands_in_exactly_one_band_of_each_stratification(
    fixture: EvalFixture,
) -> None:
    report = _report(fixture, "baseline")

    assert {s.dimension for s in report.strata} == {
        "range",
        "visibility",
        "support_density",
        "image_border",
        "physical_observations",
        "uncertainty_level",
    }
    for stratification in report.strata:
        total = sum(s.supports for s in stratification.strata) + stratification.unavailable.supports
        assert total == report.support_count, stratification.dimension


def test_range_strata_group_supports_by_the_median_depth_of_their_observations(
    fixture: EvalFixture,
) -> None:
    report = _report(fixture, "quality_aware")

    assert _stratum(report, "range", "(-inf, 5.0)").supports == 1
    assert _stratum(report, "range", "[5.0, 15.0)").supports == 2
    assert _stratum(report, "range", "[15.0, +inf)").supports == 1
    assert _stratum(report, "range", "unavailable").supports == 0


def test_gains_and_regressions_are_visible_by_condition_and_not_only_globally(
    fixture: EvalFixture,
) -> None:
    baseline = _report(fixture, "baseline")
    weighted = _report(fixture, "quality_aware")

    near = "(-inf, 5.0)"
    assert _stratum(baseline, "range", near).leading_matches_reference == 0
    assert _stratum(baseline, "range", near).weighted_leading_matches_reference is None
    assert _stratum(weighted, "range", near).weighted_leading_matches_reference == 1
    far = "[15.0, +inf)"
    assert _stratum(weighted, "range", far).weighted_leading_matches_reference == 0


def test_the_number_of_physical_observations_and_the_uncertainty_level_are_strata(
    fixture: EvalFixture,
) -> None:
    report = _report(fixture, "baseline")

    assert _stratum(report, "physical_observations", "(-inf, 2.0)").supports == 1
    assert _stratum(report, "physical_observations", "[2.0, 3.0)").supports == 2
    assert _stratum(report, "physical_observations", "[3.0, +inf)").supports == 1
    for level in ("none", "ambiguity_or_near_tie", "contradiction", "insufficient_evidence"):
        assert _stratum(report, "uncertainty_level", level).supports == 1
    assert _stratum(report, "uncertainty_level", "contradiction").with_uncertainty == 1
    assert _stratum(report, "uncertainty_level", "none").with_uncertainty == 0


def test_without_quality_every_quality_stratum_is_unavailable_and_never_zero(
    fixture: EvalFixture,
) -> None:
    report = _report(fixture, "baseline", qualities=False)

    for dimension in ("range", "visibility", "support_density", "image_border"):
        assert _stratum(report, dimension, "unavailable").supports == report.support_count
    assert _stratum(report, "physical_observations", "[2.0, 3.0)").supports == 2


def test_cost_is_reported_apart_from_every_quality_measure(fixture: EvalFixture) -> None:
    cost = _report(fixture, "baseline").cost

    assert cost.runtime is None
    assert cost.total_payload_bytes == sum(cost.payload_bytes.values())
    assert cost.payload_bytes["outputs/fused-evidence.jsonl"] > 0


def test_the_report_is_deterministic_json_without_scores_winners_or_composites(
    fixture: EvalFixture,
) -> None:
    first = encode_semantic_fusion_report(_report(fixture, "quality_aware"))
    second = encode_semantic_fusion_report(_report(fixture, "quality_aware"))

    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert not FORBIDDEN_KEYS & _all_keys(first)
    assert first["lineage"]["run_id"] == "fusion-run-0002"
    assert first["profile"]["range_edges_m"] == [5.0, 15.0]


def test_evaluating_never_modifies_the_run(fixture: EvalFixture) -> None:
    def digest(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = digest(fixture.runs["quality_aware"])

    _report(fixture, "quality_aware")

    assert digest(fixture.runs["quality_aware"]) == before


def test_debug_evidence_is_never_read(tmp_path: Path) -> None:
    fixture = make_eval_fixture(tmp_path)
    run = fixture.runs["baseline"]
    (run / "debug").mkdir()
    (run / "debug" / "stray.json").write_text("not json at all")

    with_debug = _report(fixture, "baseline")
    shutil.rmtree(run / "debug")

    assert _report(fixture, "baseline") == with_debug


def test_a_run_that_fails_its_integrity_check_is_refused(tmp_path: Path) -> None:
    fixture = make_eval_fixture(tmp_path)
    target = fixture.runs["baseline"] / "outputs" / "fused-evidence.jsonl"
    target.write_bytes(target.read_bytes().replace(b"door", b"dooX", 1))

    with pytest.raises(SemanticFusionEvaluationError, match="not intact"):
        _report(fixture, "baseline")


def test_the_arms_compare_under_one_control_without_a_winner(fixture: EvalFixture) -> None:
    reports = [_report(fixture, arm) for arm in fixture.runs]

    comparison = compare_semantic_fusion_reports(reports)

    assert [e.arm_id for e in comparison.entries] == list(fixture.runs)
    assert comparison.control_arm_id == "baseline"
    by_arm = {e.arm_id: e for e in comparison.entries}
    assert by_arm["quality_aware"].hypotheses_match_control
    assert by_arm["quality_aware"].stances_match_control
    assert by_arm["quality_aware"].supports_with_changed_leader == 2
    assert by_arm["with_scores"].supports_with_changed_leader == 0
    assert by_arm["with_visual_and_3d"].active_channels == (
        "geometry_support",
        "point_representation",
        "semantic_claims",
        "visual_features",
    )
    assert by_arm["baseline"].fusion_configuration_fingerprint != (
        by_arm["quality_aware"].fusion_configuration_fingerprint
    )
    encoded = encode_semantic_fusion_comparison(comparison)
    assert not FORBIDDEN_KEYS & _all_keys(encoded)


def test_a_comparison_needs_two_arms_one_control_and_unique_identities(
    fixture: EvalFixture,
) -> None:
    baseline = _report(fixture, "baseline")
    aware = _report(fixture, "quality_aware")

    with pytest.raises(SemanticFusionEvaluationError, match="at least two"):
        compare_semantic_fusion_reports([baseline])
    with pytest.raises(SemanticFusionEvaluationError, match="arm_id"):
        compare_semantic_fusion_reports([baseline, dataclasses.replace(aware, arm_id="baseline")])
    with pytest.raises(SemanticFusionEvaluationError, match="control"):
        compare_semantic_fusion_reports(
            [aware, dataclasses.replace(_report(fixture, "with_scores"))]
        )
    with pytest.raises(SemanticFusionEvaluationError, match="control"):
        compare_semantic_fusion_reports(
            [
                baseline,
                dataclasses.replace(aware, arm_role=FusionArmRole.BASELINE_CONTROL),
            ]
        )


def test_a_comparison_rejects_anything_but_the_fusion_configuration_changing(
    fixture: EvalFixture, tmp_path: Path
) -> None:
    baseline = _report(fixture, "baseline")
    aware = _report(fixture, "quality_aware")
    other_base = make_eval_fixture(tmp_path, extra_frame=True)

    with pytest.raises(SemanticFusionEvaluationError, match="evidence base"):
        compare_semantic_fusion_reports([baseline, _report(other_base, "quality_aware")])
    with pytest.raises(SemanticFusionEvaluationError, match="profile"):
        compare_semantic_fusion_reports(
            [
                baseline,
                dataclasses.replace(
                    aware, profile=dataclasses.replace(PROFILE, range_edges_m=(9.0,))
                ),
            ]
        )
    with pytest.raises(SemanticFusionEvaluationError, match="annotations"):
        compare_semantic_fusion_reports(
            [baseline, _report(fixture, "quality_aware", annotations=False)]
        )
    drifted = dataclasses.replace(
        aware, lineage=dataclasses.replace(aware.lineage, sequence_name="sequence-9999")
    )
    with pytest.raises(SemanticFusionEvaluationError, match="sequence_name"):
        compare_semantic_fusion_reports([baseline, drifted])
    other_map = dataclasses.replace(
        aware, lineage=dataclasses.replace(aware.lineage, support_policy_id="another-policy-v1")
    )
    with pytest.raises(SemanticFusionEvaluationError, match="support_policy_id"):
        compare_semantic_fusion_reports([baseline, other_map])


def test_the_stratification_profile_has_no_defaults_and_rejects_impossible_edges() -> None:
    with pytest.raises(TypeError):
        FusionStratificationProfile()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="strictly increasing"):
        dataclasses.replace(PROFILE, range_edges_m=(5.0, 5.0))
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(PROFILE, visible_share_edges=(float("nan"),))


def test_visual_and_structural_spaces_stay_separate_and_are_never_compared(
    fixture: EvalFixture,
) -> None:
    report = _report(fixture, "with_visual_and_3d")

    channels = {c.channel: c for c in report.channels}
    assert channels["visual_features"].identities == ("clip-vit-l14", "dinov3-vit-b16")
    assert channels["point_representation"].identities == (
        "sha256:space-geometric-descriptor",
        "sha256:space-ptv3",
    )
    encoded = encode_semantic_fusion_report(report)
    assert not {"similarity", "cosine", "distance", "concatenated"} & _all_keys(encoded)


def test_rebuilding_the_same_inputs_gives_the_same_supports_and_evidence_base(
    tmp_path: Path,
) -> None:
    first = make_eval_fixture(tmp_path / "first")
    second = make_eval_fixture(tmp_path / "second")

    a = _report(first, "quality_aware")
    b = _report(second, "quality_aware")

    assert a.evidence_base_id == b.evidence_base_id
    assert a.leaders == b.leaders
    assert a.hypothesis_stances_id == b.hypothesis_stances_id
