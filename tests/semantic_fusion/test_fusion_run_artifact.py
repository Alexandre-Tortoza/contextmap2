import dataclasses
import json
import shutil
import subprocess
import sys
import textwrap
import typing
from collections.abc import Iterator
from pathlib import Path

import pytest
from fusion_run_fixtures import LINEAGE, RunFixture, make_run_fixture

from contextmap.geometric_mapping import MapId
from contextmap.semantic_fusion import (
    EvidenceContributionId,
    ExcludedObservation,
    FusionOutcome,
    FusionRunArtifactError,
    FusionRunLineage,
    FusionSupportId,
    IncompleteFusionRunArtifactError,
    SemanticFusionDebugLevel,
    SemanticFusionRunId,
    SemanticFusionRunReader,
    SemanticFusionRunWriter,
    UncertaintyKind,
)
from contextmap.sensor_association import SpatialObservationId
from contextmap.visual_perception import PerceptionRunId


@pytest.fixture(scope="module")
def fixture() -> RunFixture:
    return make_run_fixture()


def _run_dir(tmp_path: Path, run_index: int = 1) -> Path:
    """Onde o writer grava: o chamador decide o diretório final, o writer não calcula caminho."""
    return tmp_path / f"run-{run_index:04d}"


def _writer(
    tmp_path: Path,
    *,
    run_index: int = 1,
    debug: SemanticFusionDebugLevel = SemanticFusionDebugLevel.NONE,
    lineage: FusionRunLineage = LINEAGE,
) -> SemanticFusionRunWriter:
    return SemanticFusionRunWriter(
        output_dir=_run_dir(tmp_path, run_index),
        sequence_name="sequence-0001",
        run_id=SemanticFusionRunId(f"fusion-run-{run_index:04d}"),
        run_index=run_index,
        lineage=lineage,
        code_version="test",
        debug_level=debug,
    )


def _write(
    tmp_path: Path,
    fixture: RunFixture,
    *,
    debug: SemanticFusionDebugLevel = SemanticFusionDebugLevel.NONE,
    run_index: int = 1,
) -> Path:
    _writer(tmp_path, run_index=run_index, debug=debug).write(
        fixture.outcomes, excluded=fixture.excluded, warnings=("one warning",)
    )
    return _run_dir(tmp_path, run_index)


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_the_fixture_has_three_supports_with_the_situations_that_matter(
    fixture: RunFixture,
) -> None:
    kinds = [{u.kind for u in outcome.evidence.uncertainty} for outcome in fixture.outcomes]

    assert len(fixture.outcomes) == 3
    assert kinds[0] == {UncertaintyKind.CONTRADICTION, UncertaintyKind.NEAR_TIE}
    assert kinds[1] == set()
    assert kinds[2] == {UncertaintyKind.INSUFFICIENT_EVIDENCE}


def test_a_run_is_persisted_with_the_documented_layout(tmp_path: Path, fixture: RunFixture) -> None:
    run_dir = _write(tmp_path, fixture)

    for name in (
        "manifest.json",
        "README.md",
        "outputs/fusion-supports.jsonl",
        "outputs/fused-evidence.jsonl",
        "outputs/support-observation-index.jsonl",
        "outputs/hypothesis-evidence-index.jsonl",
        "outputs/physical-observation-groups.jsonl",
        "outputs/contribution-index.jsonl",
        "outputs/excluded-observations.jsonl",
        "metrics/counts.json",
        "metrics/distributions.json",
        "metrics/payload.json",
    ):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "debug").exists()
    assert not (run_dir / "metrics" / "runtime.json").exists()


def test_a_reopened_run_returns_every_support_and_fused_evidence_intact(
    tmp_path: Path, fixture: RunFixture
) -> None:
    reader = SemanticFusionRunReader(_write(tmp_path, fixture))

    assert list(reader.iter_outcomes()) == list(fixture.outcomes)
    assert reader.verify_integrity() == []


def test_supports_are_read_at_random_by_identity_and_by_observation(
    tmp_path: Path, fixture: RunFixture
) -> None:
    reader = SemanticFusionRunReader(_write(tmp_path, fixture))
    second = fixture.outcomes[1]

    assert reader.support_ids() == [o.support.fusion_support_id for o in fixture.outcomes]
    assert reader.support(second.support.fusion_support_id) == second.support
    assert reader.fused_evidence(second.support.fusion_support_id) == second.evidence
    observation_id = second.support.spatial_observation_ids[0]
    assert reader.support_of_observation(observation_id) == second.support.fusion_support_id
    assert reader.support_of_observation(SpatialObservationId("never-seen")) is None


def test_every_hypothesis_resolves_back_to_exact_upstream_evidence(
    tmp_path: Path, fixture: RunFixture
) -> None:
    reader = SemanticFusionRunReader(_write(tmp_path, fixture))
    evidence = reader.fused_evidence(fixture.outcomes[0].support.fusion_support_id)

    for hypothesis in evidence.hypotheses:
        for item in hypothesis.evidence:
            contribution = reader.contribution(item.contribution_id)
            assert item.claim_id in {ref.claim_id for ref in contribution.claim_refs}
            assert contribution.perception_result_id
            assert contribution.perception_run_id in LINEAGE.perception_run_ids
            assert contribution.spatial_observation_id
    assert reader.manifest.lineage.perception_run_ids == LINEAGE.perception_run_ids


def test_the_hypothesis_index_names_the_exact_evidence_behind_each_row(
    tmp_path: Path, fixture: RunFixture
) -> None:
    run_dir = _write(tmp_path, fixture)
    rows = _lines(run_dir / "outputs" / "hypothesis-evidence-index.jsonl")
    first = fixture.outcomes[0].evidence
    expected = sum(len(h.evidence) for o in fixture.outcomes for h in o.evidence.hypotheses)

    assert len(rows) == expected
    row = next(r for r in rows if r["fusion_support_id"] == first.fusion_support_id)
    assert {
        "fused_evidence_id",
        "hypothesis_id",
        "label",
        "contribution_id",
        "claim_id",
        "stance",
        "role",
        "physical_observation_id",
        "perception_result_id",
        "perception_run_id",
        "spatial_observation_id",
    } <= set(row)


def test_physical_observations_stay_distinct_from_inference_results(
    tmp_path: Path, fixture: RunFixture
) -> None:
    run_dir = _write(tmp_path, fixture)
    groups = _lines(run_dir / "outputs" / "physical-observation-groups.jsonl")
    counts = json.loads((run_dir / "metrics" / "counts.json").read_text())

    first = [
        g for g in groups if g["fusion_support_id"] == fixture.outcomes[0].support.fusion_support_id
    ]
    assert len(first) == 2
    assert sorted(len(typing.cast(list[str], g["perception_result_ids"])) for g in first) == [1, 2]
    assert counts["physical_observations"] == 5
    assert counts["inference_results"] == 6
    distributions = json.loads((run_dir / "metrics" / "distributions.json").read_text())
    assert distributions["physical_observations_per_support"]["max"] == 2
    assert distributions["inference_results_per_support"]["max"] == 3


def test_ambiguity_conflict_abstention_and_unscored_evidence_survive_the_round_trip(
    tmp_path: Path, fixture: RunFixture
) -> None:
    reader = SemanticFusionRunReader(_write(tmp_path, fixture))

    evidence = reader.fused_evidence(fixture.outcomes[0].support.fusion_support_id)

    assert {u.kind for u in evidence.uncertainty} == {
        UncertaintyKind.CONTRADICTION,
        UncertaintyKind.NEAR_TIE,
    }
    assert None in {s.value for h in evidence.hypotheses for i in h.evidence for s in i.signals}
    only_abstention = reader.fused_evidence(fixture.outcomes[2].support.fusion_support_id)
    assert only_abstention.hypotheses == ()
    assert only_abstention.uncertainty[0].kind is UncertaintyKind.INSUFFICIENT_EVIDENCE


def test_the_manifest_records_lineage_policies_identities_and_counts(
    tmp_path: Path, fixture: RunFixture
) -> None:
    manifest = SemanticFusionRunReader(_write(tmp_path, fixture)).manifest

    assert manifest.lineage == LINEAGE
    assert manifest.grouping_policy_id == "physical-observation-grouping-v1"
    assert manifest.support_policy_id == "geometry-jaccard-support-v1"
    assert (manifest.support_configuration_fingerprint or "").startswith("sha256:")
    assert manifest.fusion_policy_id == "quality-aware-evidence-accumulation-v1"
    assert (manifest.fusion_configuration_fingerprint or "").startswith("sha256:")
    assert manifest.code_version == "test"
    assert manifest.evidence_identities["semantic_scores"] == ("clip_scorer/clip-vit-l14/1",)
    assert manifest.evidence_identities["semantic_claims"] == ("qwen_vl/qwen3-vl/1",)
    assert manifest.support_count == 3
    assert manifest.excluded_count == len(fixture.excluded)
    assert manifest.warnings == ("one warning",)
    assert manifest.schema_version == "0.1.0"


def test_metrics_report_each_quantity_on_its_own(tmp_path: Path, fixture: RunFixture) -> None:
    run_dir = _write(tmp_path, fixture)

    counts = json.loads((run_dir / "metrics" / "counts.json").read_text())

    assert counts["supports"] == 3
    assert counts["fused_evidence"] == 3
    assert counts["contributions"] == 6
    assert counts["hypotheses"] == 4
    assert counts["uncertainty"] == {
        "contradiction": 1,
        "near_tie": 1,
        "ambiguity": 0,
        "insufficient_evidence": 1,
    }
    assert counts["evidence_stances"]["abstaining"] >= 1
    assert counts["claims"]["unscored"] >= 1
    assert counts["claims"]["scored"] >= 1
    assert counts["claims"]["without_hypothesis"] == 1
    payload = json.loads((run_dir / "metrics" / "payload.json").read_text())
    assert payload["files"]["outputs/fused-evidence.jsonl"] > 0
    assert payload["total_bytes"] == sum(payload["files"].values())


def test_runtime_is_recorded_apart_from_quality_and_only_when_measured(
    tmp_path: Path, fixture: RunFixture
) -> None:
    _writer(tmp_path).write(
        fixture.outcomes, runtime={"fusion_seconds": 1.5, "peak_memory_bytes": 1024}
    )
    run_dir = _run_dir(tmp_path)

    runtime = json.loads((run_dir / "metrics" / "runtime.json").read_text())

    assert runtime == {"fusion_seconds": 1.5, "peak_memory_bytes": 1024}
    assert "fusion_seconds" not in (run_dir / "metrics" / "counts.json").read_text()


def test_skipped_evidence_and_warnings_are_persisted_explicitly(tmp_path: Path) -> None:
    fixture = make_run_fixture()
    skipped = ExcludedObservation(
        spatial_observation_id=fixture.outcomes[0].support.spatial_observation_ids[0],
        geometry_count=1,
        minimum_geometry_count=3,
    )
    manifest = _writer(tmp_path).write(fixture.outcomes, excluded=(skipped,), warnings=("w1", "w2"))
    reader = SemanticFusionRunReader(_run_dir(tmp_path))

    assert manifest.excluded_count == 1
    assert reader.excluded_observations() == [skipped]
    assert reader.manifest.warnings == ("w1", "w2")


def test_a_run_without_supports_is_a_valid_explicit_run(tmp_path: Path) -> None:
    manifest = _writer(tmp_path).write((), excluded=())
    reader = SemanticFusionRunReader(_run_dir(tmp_path))

    assert manifest.support_count == 0
    assert list(reader.iter_outcomes()) == []
    assert reader.verify_integrity() == []
    assert manifest.fusion_policy_id is None


def test_integrity_detects_a_missing_altered_or_truncated_file(
    tmp_path: Path, fixture: RunFixture
) -> None:
    run_dir = _write(tmp_path, fixture)
    reader = SemanticFusionRunReader(run_dir)
    target = run_dir / "outputs" / "fused-evidence.jsonl"
    original = target.read_bytes()

    target.write_bytes(original[:-5])
    assert any("fused-evidence.jsonl" in problem for problem in reader.verify_integrity())
    target.write_bytes(original.replace(b"door", b"dooX", 1))
    assert any("hash" in problem for problem in reader.verify_integrity())
    target.unlink()
    assert any("missing" in problem for problem in reader.verify_integrity())


def test_the_run_is_written_exactly_where_the_caller_says_and_nothing_else_is_created(
    tmp_path: Path, fixture: RunFixture
) -> None:
    target = tmp_path / "ws" / "corridor-02" / "run-0001" / "semantic_fusion"

    SemanticFusionRunWriter(
        output_dir=target,
        sequence_name="sequence-0001",
        run_id=SemanticFusionRunId("fusion-run"),
        run_index=1,
        lineage=LINEAGE,
        code_version="test",
    ).write(fixture.outcomes, excluded=fixture.excluded)

    assert SemanticFusionRunReader(target).manifest.run_id == SemanticFusionRunId("fusion-run")
    # Sem registro `runs.json` e sem `runs/<capability>/<sequência>/`: só o diretório do artifact.
    assert sorted(path.name for path in target.parent.iterdir()) == ["semantic_fusion"]
    assert sorted(path.name for path in (tmp_path / "ws").iterdir()) == ["corridor-02"]


def test_the_run_id_and_index_are_recorded_as_supplied_and_never_allocated(
    tmp_path: Path, fixture: RunFixture
) -> None:
    run_dir = _write(tmp_path, fixture, run_index=7)

    manifest = SemanticFusionRunReader(run_dir).manifest
    assert (manifest.run_id, manifest.run_index) == (SemanticFusionRunId("fusion-run-0007"), 7)
    assert not _run_dir(tmp_path, 1).exists()


def test_a_second_run_at_the_same_output_directory_is_refused_and_leaves_the_first_intact(
    tmp_path: Path, fixture: RunFixture
) -> None:
    first = _write(tmp_path, fixture)
    before = (first / "manifest.json").read_bytes()

    with pytest.raises(FusionRunArtifactError, match="already exists"):
        _writer(tmp_path, run_index=1).write(fixture.outcomes)

    assert (first / "manifest.json").read_bytes() == before
    assert SemanticFusionRunReader(first).verify_integrity() == []


def test_an_interrupted_write_never_looks_like_a_finished_run(
    tmp_path: Path, fixture: RunFixture
) -> None:
    def failing() -> Iterator[FusionOutcome]:
        yield fixture.outcomes[0]
        raise RuntimeError("the fusion stopped")

    with pytest.raises(RuntimeError, match="stopped"):
        _writer(tmp_path).write(failing())

    assert list(tmp_path.iterdir()) == []


def test_a_directory_without_a_manifest_is_incomplete(tmp_path: Path) -> None:
    with pytest.raises(IncompleteFusionRunArtifactError):
        SemanticFusionRunReader(tmp_path)


def test_an_unknown_schema_version_is_refused(tmp_path: Path, fixture: RunFixture) -> None:
    run_dir = _write(tmp_path, fixture)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["schema_version"] = "9.9.9"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(FusionRunArtifactError, match="schema_version"):
        SemanticFusionRunReader(run_dir)


def test_reading_one_support_does_not_read_the_others(tmp_path: Path, fixture: RunFixture) -> None:
    run_dir = _write(tmp_path, fixture)
    reader = SemanticFusionRunReader(run_dir)
    last = fixture.outcomes[-1]
    rows = _lines(run_dir / "outputs" / "support-observation-index.jsonl")
    first_row = rows[0]
    target = run_dir / "outputs" / "fused-evidence.jsonl"
    data = bytearray(target.read_bytes())
    start = int(first_row["evidence_offset"])  # type: ignore[call-overload]
    data[start : start + 5] = b"#####"
    target.write_bytes(bytes(data))

    assert reader.fused_evidence(last.support.fusion_support_id) == last.evidence
    with pytest.raises(FusionRunArtifactError, match="malformed"):
        reader.fused_evidence(fixture.outcomes[0].support.fusion_support_id)


def test_an_unknown_support_or_contribution_is_an_explicit_error(
    tmp_path: Path, fixture: RunFixture
) -> None:
    reader = SemanticFusionRunReader(_write(tmp_path, fixture))

    with pytest.raises(FusionRunArtifactError, match="unknown"):
        reader.support(FusionSupportId("support-999999"))
    with pytest.raises(FusionRunArtifactError, match="unknown"):
        reader.contribution(EvidenceContributionId("nope"))


def test_debug_is_off_by_default_and_never_needed_by_a_reader(
    tmp_path: Path, fixture: RunFixture
) -> None:
    with_debug = _write(tmp_path, fixture, debug=SemanticFusionDebugLevel.FULL)
    reader = SemanticFusionRunReader(with_debug)
    before = list(reader.iter_outcomes())

    shutil.rmtree(with_debug / "debug")

    assert list(reader.iter_outcomes()) == before
    assert reader.verify_integrity() == []
    with pytest.raises(FusionRunArtifactError, match="contractual"):
        reader.read_record("debug/supports/support-000001/summary.json")


def test_standard_debug_answers_the_questions_a_reviewer_asks(
    tmp_path: Path, fixture: RunFixture
) -> None:
    run_dir = _write(tmp_path, fixture, debug=SemanticFusionDebugLevel.STANDARD)
    support_dir = run_dir / "debug" / "supports" / fixture.outcomes[0].support.fusion_support_id

    summary = json.loads((support_dir / "summary.json").read_text())

    assert summary["physical_observation_count"] == 2
    assert summary["inference_result_count"] == 3
    door = next(h for h in summary["hypotheses"] if h["label"] == "door")
    assert door["supporting_physical_observations"] == ["frame-0120"]
    assert len(door["supporting_inference_results"]) == 2
    assert door["stances"]["supporting"] >= 1
    assert "semantic_scores" in summary["active_channels"]
    assert {u["kind"] for u in summary["uncertainty"]} == {"contradiction", "near_tie"}
    for name in ("hypotheses.json", "conflicts.json", "physical-observations.json"):
        assert (support_dir / name).is_file()
    assert not (support_dir / "contribution-trace.jsonl").exists()


def test_full_debug_adds_the_contribution_trace_and_the_geometry_summary(
    tmp_path: Path, fixture: RunFixture
) -> None:
    run_dir = _write(tmp_path, fixture, debug=SemanticFusionDebugLevel.FULL)
    support_dir = run_dir / "debug" / "supports" / fixture.outcomes[0].support.fusion_support_id

    trace = _lines(support_dir / "contribution-trace.jsonl")
    geometry = json.loads((support_dir / "geometry-summary.json").read_text())

    assert len(trace) == 3
    assert geometry["geometry_count"] == 20
    assert len(list((run_dir / "debug" / "supports").iterdir())) == 3


def test_debug_is_never_part_of_the_inventory(tmp_path: Path, fixture: RunFixture) -> None:
    run_dir = _write(tmp_path, fixture, debug=SemanticFusionDebugLevel.FULL)

    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert not any(entry["path"].startswith("debug/") for entry in manifest["file_inventory"])


@pytest.mark.parametrize(
    "mutation",
    [
        "unsorted_supports",
        "support_evidence_mismatch",
        "duplicate_support",
        "other_map",
        "unlisted_perception_run",
        "unlisted_representation_run",
        "mixed_fusion_policy",
    ],
)
def test_inconsistent_inputs_are_refused_and_leave_no_run(
    tmp_path: Path, fixture: RunFixture, mutation: str
) -> None:
    outcomes = list(fixture.outcomes)
    lineage = LINEAGE
    if mutation == "unsorted_supports":
        outcomes = outcomes[::-1]
    elif mutation == "support_evidence_mismatch":
        wrong = dataclasses.replace(
            outcomes[0].support, spatial_observation_ids=outcomes[1].support.spatial_observation_ids
        )
        outcomes[0] = FusionOutcome(support=wrong, evidence=outcomes[0].evidence)
    elif mutation == "duplicate_support":
        outcomes = [outcomes[0], outcomes[0]]
    elif mutation == "other_map":
        lineage = dataclasses.replace(LINEAGE, geometric_map_id=MapId("map-9999"))
    elif mutation == "unlisted_perception_run":
        lineage = dataclasses.replace(LINEAGE, perception_run_ids=(PerceptionRunId("run-a"),))
    elif mutation == "unlisted_representation_run":
        lineage = dataclasses.replace(LINEAGE, point_representation_run_ids=())
    elif mutation == "mixed_fusion_policy":
        odd = dataclasses.replace(
            outcomes[1].evidence,
            provenance=dataclasses.replace(
                outcomes[1].evidence.provenance, fusion_policy_id="other-v1"
            ),
        )
        outcomes[1] = FusionOutcome(support=outcomes[1].support, evidence=odd)

    with pytest.raises(FusionRunArtifactError):
        _writer(tmp_path, lineage=lineage).write(outcomes)

    assert list(tmp_path.iterdir()) == []


def test_an_outcome_must_pair_a_support_with_its_own_evidence(fixture: RunFixture) -> None:
    with pytest.raises(ValueError, match="fusion_support_id"):
        FusionOutcome(support=fixture.outcomes[0].support, evidence=fixture.outcomes[1].evidence)


def test_the_lineage_needs_its_upstream_identities() -> None:
    with pytest.raises(ValueError, match="perception_run_ids"):
        dataclasses.replace(LINEAGE, perception_run_ids=())
    with pytest.raises(ValueError, match="association_run_ids"):
        dataclasses.replace(LINEAGE, association_run_ids=())
    with pytest.raises(ValueError, match="sorted and unique"):
        dataclasses.replace(
            LINEAGE, perception_run_ids=(PerceptionRunId("run-b"), PerceptionRunId("run-a"))
        )


def test_a_persisted_run_reopens_without_numpy_or_any_model_runtime(
    tmp_path: Path, fixture: RunFixture
) -> None:
    run_dir = _write(tmp_path, fixture)
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from contextmap.semantic_fusion import SemanticFusionRunReader
        reader = SemanticFusionRunReader(Path({str(run_dir)!r}))
        outcomes = list(reader.iter_outcomes())
        assert len(outcomes) == 3, len(outcomes)
        assert reader.verify_integrity() == []
        for heavy in ("numpy", "torch", "rosbags", "open3d", "transformers"):
            assert heavy not in sys.modules, heavy
        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
