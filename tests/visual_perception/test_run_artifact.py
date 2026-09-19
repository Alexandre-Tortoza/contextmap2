import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    BackendProvenance,
    BoundingBox2D,
    FeatureId,
    FeatureScope,
    IncompleteRunArtifactError,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    Region2D,
    RegionId,
    RunArtifactError,
    StageDefinition,
    StageStatus,
    VisualFeature,
    allocate_run_index,
    execute_stage_graph,
    rebuild_run_registry,
)

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0.1"
)
_RUN_DIR_NAME = "run-0001__frames-0000-0010__fake"


def _run_dir(tmp_path: Path, sequence_name: str = "corridor-02", run_index: int = 1) -> Path:
    run_dir_name = f"run-{run_index:04d}__frames-0000-0010__fake"
    return tmp_path / "runs" / "visual-perception" / sequence_name / run_dir_name


def _result(
    observation_id: str,
    run_id: str,
    *,
    features: tuple[VisualFeature, ...] = (),
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
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(region,),
        features=features,
    )


def _write_run(
    tmp_path: Path, *, run_index: int = 1, sequence_name: str = "corridor-02"
) -> PerceptionRunWriter:
    writer = PerceptionRunWriter(
        workspace_root=tmp_path,
        sequence_name=sequence_name,
        run_id=PerceptionRunId(f"run-{run_index:04d}"),
        run_index=run_index,
        sequence_artifact_id="corridor-02-a1b2c3",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_discovery"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
        selection_label="frames-0000-0010",
        profile_label="fake",
    )
    return writer


def test_finalized_run_can_be_reopened_in_isolation(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    manifest = writer.finalize()

    run_dir = _run_dir(tmp_path)
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "README.md").is_file()
    assert (run_dir / "outputs" / "results.jsonl").is_file()

    reader = PerceptionRunReader(run_dir)
    assert reader.manifest.run_index == manifest.run_index
    assert reader.manifest.result_count == 1
    assert reader.verify_integrity() == []


def test_result_lookup_by_source_observation_id(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_result(_result("frame-0002", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    reader = PerceptionRunReader(run_dir)

    result = reader.result(SourceObservationId("frame-0002"))
    assert result.source_observation_id == "frame-0002"


def test_result_lookup_raises_for_unknown_observation(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    reader = PerceptionRunReader(run_dir)

    with pytest.raises(RunArtifactError, match="no result"):
        reader.result(SourceObservationId("does-not-exist"))


def test_writer_rejects_result_owned_by_another_run(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)

    with pytest.raises(RunArtifactError, match="run_id"):
        writer.add_result(_result("frame-0001", "run-9999"))


def test_writer_rejects_result_from_another_sequence_artifact(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    result = _result("frame-0001", "run-0001")
    foreign_result = PerceptionResult(
        result_id=result.result_id,
        source_observation_id=result.source_observation_id,
        run_id=result.run_id,
        sequence_artifact_id="another-sequence-artifact",
        created_at=result.created_at,
        regions=result.regions,
    )

    with pytest.raises(RunArtifactError, match="sequence_artifact_id"):
        writer.add_result(foreign_result)


def test_writer_rejects_duplicate_source_observation(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))

    with pytest.raises(RunArtifactError, match="source_observation_id"):
        writer.add_result(_result("frame-0001", "run-0001"))


def test_stage_outcomes_are_persisted_as_metrics(tmp_path: Path) -> None:
    outcomes = execute_stage_graph(
        [
            StageDefinition(
                stage_id="region_discovery", capability="region_discovery", run=lambda ctx: ()
            )
        ]
    )

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_stage_outcomes(outcomes)
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    metrics_lines = (
        (run_dir / "metrics" / "stage-timings.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(metrics_lines) == 1
    record = json.loads(metrics_lines[0])
    assert record["stage_id"] == "region_discovery"
    assert record["status"] == StageStatus.SUCCEEDED.value


def test_readme_summarizes_the_run(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    readme = (run_dir / "README.md").read_text(encoding="utf-8")

    assert "corridor-02" in readme
    assert "region_discovery" in readme
    assert "Regions: 1" in readme


def test_finalize_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    first = _write_run(tmp_path)
    first.add_result(_result("frame-0001", "run-0001"))
    first.finalize()

    second = _write_run(tmp_path)
    second.add_result(_result("frame-0002", "run-0001"))

    with pytest.raises(RunArtifactError, match="already exists"):
        second.finalize()


def test_allocate_run_index_starts_at_one_and_increments(tmp_path: Path) -> None:
    assert allocate_run_index(workspace_root=tmp_path, sequence_name="corridor-02") == 1

    writer = _write_run(tmp_path, run_index=1)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    assert allocate_run_index(workspace_root=tmp_path, sequence_name="corridor-02") == 2


def test_allocate_run_index_ignores_interrupted_tmp_directories(tmp_path: Path) -> None:
    writer = _write_run(tmp_path, run_index=1)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    # Simulate an interrupted write: a stray .tmp- directory with no manifest.
    stray = tmp_path / "runs" / "visual-perception" / "corridor-02" / ".tmp-run-0002__x__y-deadbeef"
    stray.mkdir(parents=True)

    assert allocate_run_index(workspace_root=tmp_path, sequence_name="corridor-02") == 2


def test_allocate_run_index_ignores_finalized_run_with_missing_output(tmp_path: Path) -> None:
    writer = _write_run(tmp_path, run_index=9)
    writer.add_result(_result("frame-0001", "run-0009"))
    writer.finalize()
    (_run_dir(tmp_path, run_index=9) / "outputs" / "results.jsonl").unlink()

    assert allocate_run_index(workspace_root=tmp_path, sequence_name="corridor-02") == 1


def test_registry_excludes_finalized_run_with_corrupt_output(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()
    results_path = _run_dir(tmp_path) / "outputs" / "results.jsonl"
    results_path.write_text("corrupt\n", encoding="utf-8")

    rebuild_run_registry(tmp_path, "corridor-02")

    registry_path = tmp_path / "runs" / "visual-perception" / "corridor-02" / "runs.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["runs"] == []


def test_registry_is_rebuilt_after_finalize_and_can_be_rebuilt_independently(
    tmp_path: Path,
) -> None:
    writer = _write_run(tmp_path, run_index=1)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    registry_path = tmp_path / "runs" / "visual-perception" / "corridor-02" / "runs.json"
    assert registry_path.is_file()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(registry["runs"]) == 1

    registry_path.unlink()
    rebuild_run_registry(tmp_path, "corridor-02")

    assert registry_path.is_file()
    rebuilt = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(rebuilt["runs"]) == 1


def test_opening_a_directory_without_a_manifest_fails(tmp_path: Path) -> None:
    empty_dir = tmp_path / "not-a-run"
    empty_dir.mkdir()

    with pytest.raises(IncompleteRunArtifactError):
        PerceptionRunReader(empty_dir)


def test_verify_integrity_detects_a_missing_output_file(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    (run_dir / "outputs" / "results.jsonl").unlink()

    reader = PerceptionRunReader(run_dir)
    assert any("missing file" in problem for problem in reader.verify_integrity())


def _dense_feature(feature_id: str = "feature-dense-0000") -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        scope=FeatureScope.DENSE,
        embedding_space_id="fake-dense-space",
        shape=(2, 2),
        dtype="float32",
        payload_reference=f"frame-0001/{feature_id}.npy",
        provenance=_PROVENANCE,
    )


def test_feature_payload_is_persisted_and_lazily_loadable(tmp_path: Path) -> None:
    feature = _dense_feature()
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    writer.add_feature_payload(feature, SourceObservationId("frame-0001"), array)
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    reader = PerceptionRunReader(run_dir)
    assert reader.verify_integrity() == []

    store = reader.feature_store()
    observation_id = SourceObservationId("frame-0001")
    assert store.feature_keys() == ((observation_id, feature.feature_id),)
    # Metadata is readable without loading the array.
    entry = store.entry(observation_id, feature.feature_id)
    assert entry.shape == (2, 2)

    loaded = store.load(observation_id, feature.feature_id)
    np.testing.assert_array_equal(loaded, array)


def test_finalize_rejects_payload_absent_from_result(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_feature_payload(
        feature,
        SourceObservationId("frame-0001"),
        np.zeros((2, 2), dtype="float32"),
    )

    with pytest.raises(RunArtifactError, match="does not resolve to exactly one result feature"):
        writer.finalize()


def test_finalize_rejects_payload_for_wrong_observation(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    writer.add_feature_payload(
        feature,
        SourceObservationId("frame-0002"),
        np.zeros((2, 2), dtype="float32"),
    )

    with pytest.raises(RunArtifactError, match="does not resolve to exactly one result feature"):
        writer.finalize()


def test_finalize_rejects_payload_metadata_that_disagrees_with_result(tmp_path: Path) -> None:
    feature = _dense_feature()
    contradictory_feature = replace(feature, normalization="l2")
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    writer.add_feature_payload(
        contradictory_feature,
        SourceObservationId("frame-0001"),
        np.zeros((2, 2), dtype="float32"),
    )

    with pytest.raises(RunArtifactError, match="normalization"):
        writer.finalize()


def test_a_run_with_no_feature_payloads_has_an_empty_feature_store(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    assert reader.feature_store().feature_keys() == ()
