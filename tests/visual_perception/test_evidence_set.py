import json
from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    BackendProvenance,
    BoundingBox2D,
    EvidenceSetError,
    PerceptionEvidenceSet,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunWriter,
    Region2D,
    RegionId,
)

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0.1"
)


def _result(
    observation_id: str, run_id: str, sequence_artifact_id: str = "corridor-02-a1b2c3"
) -> PerceptionResult:
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
        provenance=_PROVENANCE,
    )
    return PerceptionResult(
        result_id=PerceptionResultId(f"{run_id}--{observation_id}"),
        source_observation_id=SourceObservationId(observation_id),
        run_id=PerceptionRunId(run_id),
        sequence_artifact_id=sequence_artifact_id,
        created_at="2026-01-01T00:00:00+00:00",
        regions=(region,),
    )


def _write_run(
    tmp_path: Path,
    *,
    run_index: int,
    observation_ids: list[str],
    sequence_artifact_id: str = "corridor-02-a1b2c3",
    run_id: str | None = None,
) -> Path:
    effective_run_id = run_id or f"run-{run_index:04d}"
    run_dir = tmp_path / f"run-{run_index:04d}"
    writer = PerceptionRunWriter(
        output_dir=run_dir,
        sequence_name="corridor-02",
        run_id=PerceptionRunId(effective_run_id),
        run_index=run_index,
        sequence_artifact_id=sequence_artifact_id,
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_discovery"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    )
    for observation_id in observation_ids:
        writer.add_result(_result(observation_id, effective_run_id, sequence_artifact_id))
    writer.finalize()
    return run_dir


def test_overlapping_selections_group_evidence_by_observation(tmp_path: Path) -> None:
    run_0001 = _write_run(tmp_path, run_index=1, observation_ids=["frame-0200", "frame-0450"])
    run_0005 = _write_run(tmp_path, run_index=5, observation_ids=["frame-0450"])
    run_0007 = _write_run(tmp_path, run_index=7, observation_ids=["frame-0450"])

    evidence_set = PerceptionEvidenceSet.open([run_0001, run_0005, run_0007])

    only_run_0001 = evidence_set.evidence_for(SourceObservationId("frame-0200"))
    assert set(only_run_0001.results_by_run.keys()) == {PerceptionRunId("run-0001")}

    all_three = evidence_set.evidence_for(SourceObservationId("frame-0450"))
    assert set(all_three.results_by_run.keys()) == {
        PerceptionRunId("run-0001"),
        PerceptionRunId("run-0005"),
        PerceptionRunId("run-0007"),
    }


def test_excluding_a_run_removes_its_evidence_without_touching_its_artifact(tmp_path: Path) -> None:
    run_0001 = _write_run(tmp_path, run_index=1, observation_ids=["frame-0450"])
    run_0002 = _write_run(tmp_path, run_index=2, observation_ids=["frame-0450"])

    evidence_set = PerceptionEvidenceSet.open([run_0001])  # run-0002 excluded

    evidence = evidence_set.evidence_for(SourceObservationId("frame-0450"))
    assert set(evidence.results_by_run.keys()) == {PerceptionRunId("run-0001")}
    # The excluded run's artifact is untouched and still independently valid.
    assert (run_0002 / "manifest.json").is_file()


def test_view_performs_no_fusion_evidence_stays_per_run() -> None:
    """Two runs' results for the same observation remain two distinct objects."""
    result_a = _result("frame-0124", "run-0001")
    result_b = _result("frame-0124", "run-0002")

    assert result_a.result_id != result_b.result_id
    assert result_a is not result_b


def test_incompatible_sequence_artifacts_are_rejected(tmp_path: Path) -> None:
    run_a = _write_run(
        tmp_path, run_index=1, observation_ids=["frame-0001"], sequence_artifact_id="seq-a"
    )
    run_b = _write_run(
        tmp_path, run_index=2, observation_ids=["frame-0001"], sequence_artifact_id="seq-b"
    )

    with pytest.raises(EvidenceSetError, match="incompatible"):
        PerceptionEvidenceSet.open([run_a, run_b])


def test_duplicate_selected_run_ids_are_rejected(tmp_path: Path) -> None:
    run_a = _write_run(tmp_path, run_index=1, observation_ids=["frame-0001"], run_id="same-run")
    run_b = _write_run(tmp_path, run_index=2, observation_ids=["frame-0002"], run_id="same-run")

    with pytest.raises(EvidenceSetError, match="duplicate run_id"):
        PerceptionEvidenceSet.open([run_a, run_b])


def test_result_run_id_must_match_owning_manifest(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, run_index=1, observation_ids=["frame-0001"])
    results_path = run_dir / "outputs" / "results.jsonl"
    record = json.loads(results_path.read_text(encoding="utf-8"))
    record["run_id"] = "another-run"
    results_path.write_text(f"{json.dumps(record)}\n", encoding="utf-8")

    with pytest.raises(EvidenceSetError, match="result run_id"):
        PerceptionEvidenceSet.open([run_dir])


def test_open_with_no_runs_is_rejected() -> None:
    with pytest.raises(EvidenceSetError, match="at least one run"):
        PerceptionEvidenceSet.open([])


def test_evidence_for_unknown_observation_returns_empty_mapping(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, run_index=1, observation_ids=["frame-0001"])
    evidence_set = PerceptionEvidenceSet.open([run_dir])

    evidence = evidence_set.evidence_for(SourceObservationId("frame-does-not-exist"))

    assert evidence.results_by_run == {}
