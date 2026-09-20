import hashlib
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
    ClaimId,
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
    SceneContext,
    SemanticBackendDiagnostics,
    SemanticClaim,
    SemanticConfidencePolicy,
    SemanticDebugLevel,
    SemanticEvidenceReference,
    SemanticFeatureReference,
    SemanticInferenceProvenance,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticPromptTemplate,
    SemanticRequestId,
    SemanticVisualView,
    StageDefinition,
    StageOutcome,
    StageStatus,
    VisualFeature,
    VisualViewKind,
    allocate_run_index,
    execute_stage_graph,
    parse_semantic_response,
    rebuild_run_registry,
    redact_semantic_secrets,
    render_semantic_prompt,
    write_semantic_audit,
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
    claims: tuple[SemanticClaim, ...] = (),
    scene_context: SceneContext | None = None,
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
        claims=claims,
        scene_context=scene_context,
    )


def _write_run(
    tmp_path: Path,
    *,
    run_index: int = 1,
    sequence_name: str = "corridor-02",
    semantic_debug_level: SemanticDebugLevel = SemanticDebugLevel.FULL,
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
        semantic_debug_level=semantic_debug_level,
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


_SEMANTIC_VIEW_PAYLOAD = b"exact semantic view pixels"


def _semantic_execution() -> SemanticInterpretationExecution:
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("region-request-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("run-0001--frame-0001"),
        mode=SemanticInterpretationMode.REGION,
        region_id=RegionId("region-0001"),
        visual_views=(
            SemanticVisualView(
                view_id="crop-0001",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/semantic-views/crop-0001.jpg",
                source_observation_id=SourceObservationId("frame-0001"),
                region_id=RegionId("region-0001"),
                sha256=hashlib.sha256(_SEMANTIC_VIEW_PAYLOAD).hexdigest(),
            ),
        ),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )
    raw_response = json.dumps(
        {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": "pallet",
                    "role": "primary",
                    "category": None,
                    "region_kind": "thing",
                    "attributes": {},
                    "confidence": None,
                }
            ],
            "scene_context": None,
        }
    )
    provenance = SemanticInferenceProvenance(
        backend=replace(
            _PROVENANCE,
            backend_id="fake-semantic",
            capability="semantic_interpreter",
            configuration_fingerprint="sha256:config",
        ),
        task_identity="fake-region",
        prompt_template_id="region/v1",
        output_schema_version="semantic-response/1",
        raw_response_reference=(
            "debug/40-semantic-interpretation/region-request-0001/raw-response.txt"
        ),
    )
    return SemanticInterpretationExecution(
        request=request,
        rendered_prompt=render_semantic_prompt(
            request,
            SemanticPromptTemplate.default_for(request.mode),
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        raw_response=raw_response,
        parsed=parse_semantic_response(
            raw_response,
            request,
            provenance,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        diagnostics=SemanticBackendDiagnostics(latency_ms=3.5, input_tokens=10),
        effective_configuration={"temperature": 0.0},
    )


def _abstained_semantic_execution() -> tuple[
    SemanticInterpretationExecution, VisualFeature, SceneContext
]:
    execution = _semantic_execution()
    feature = VisualFeature(
        feature_id=FeatureId("semantic-feature-0001"),
        scope=FeatureScope.REGION,
        embedding_space_id="measured-space-v1",
        shape=(2,),
        dtype="float32",
        payload_reference="frame-0001/semantic-feature-0001.npy",
        provenance=replace(_PROVENANCE, capability="feature_extractor"),
        region_id=RegionId("region-0001"),
    )
    provenance = execution.parsed.claims[0].provenance
    scene_context = SceneContext(
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("run-0001--frame-0001"),
        provenance=provenance,
        scene_type="warehouse",
    )
    request = replace(
        execution.request,
        visual_features=(
            SemanticFeatureReference(
                feature_id=feature.feature_id,
                embedding_space_id=feature.embedding_space_id,
                scope=feature.scope,
                region_id=feature.region_id,
            ),
        ),
        scene_context_reference=SemanticEvidenceReference(
            evidence_type="scene_context",
            evidence_id=str(scene_context.perception_result_id),
        ),
    )
    raw_response = json.dumps({"abstained": True, "claims": [], "scene_context": None})
    return (
        replace(
            execution,
            request=request,
            rendered_prompt=render_semantic_prompt(
                request,
                SemanticPromptTemplate.default_for(request.mode),
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            ),
            raw_response=raw_response,
            parsed=parse_semantic_response(
                raw_response,
                request,
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            ),
        ),
        feature,
        scene_context,
    )


def _add_semantic_outcome(
    writer: PerceptionRunWriter, execution: SemanticInterpretationExecution
) -> None:
    writer.add_stage_outcomes(
        (
            StageOutcome(
                stage_id="semantic_interpretation",
                status=StageStatus.SUCCEEDED,
                output=execution,
                duration_ms=4.0,
            ),
        )
    )


def test_semantic_debug_none_writes_no_human_diagnostics(tmp_path: Path) -> None:
    paths = write_semantic_audit(
        run_root=tmp_path,
        execution=_semantic_execution(),
        debug_level=SemanticDebugLevel.NONE,
    )
    assert paths == ()
    assert not (tmp_path / "debug").exists()


def test_semantic_full_debug_layout_is_machine_and_human_readable(tmp_path: Path) -> None:
    execution = _semantic_execution()
    paths = write_semantic_audit(
        run_root=tmp_path,
        execution=execution,
        debug_level=SemanticDebugLevel.FULL,
    )
    root = tmp_path / "debug/40-semantic-interpretation/region-request-0001"
    assert (root / "prompt.txt").read_text() == execution.rendered_prompt.text
    assert (root / "raw-response.txt").read_text() == execution.raw_response
    parsed = json.loads((root / "parsed-response.json").read_text())
    assert parsed["claims"][0]["hypothesis"] == "pallet"
    assert "debug/40-semantic-interpretation/region-request-0001/request.json" in paths


def test_semantic_secret_redaction_is_recursive_without_redacting_usage_tokens() -> None:
    redacted = redact_semantic_secrets(
        {
            "api_key": "do-not-persist",
            "headers": {"Authorization": "Bearer secret"},
            "output_tokens": 42,
            "model": "gemini",
        }
    )
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["headers"]["Authorization"] == "[REDACTED]"
    assert redacted["output_tokens"] == 42


def test_semantic_execution_view_and_raw_response_are_persisted_and_reopened(
    tmp_path: Path,
) -> None:
    execution = _semantic_execution()
    request = execution.request
    provenance = execution.parsed.claims[0].provenance

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", claims=execution.parsed.claims))
    writer.add_semantic_view_payload(request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    writer.add_stage_outcomes(
        (
            StageOutcome(
                stage_id="semantic_interpretation",
                status=StageStatus.SUCCEEDED,
                output=execution,
                duration_ms=4.0,
            ),
        )
    )
    manifest = writer.finalize()

    assert manifest.schema_version == "0.4.0"
    run_dir = _run_dir(tmp_path)
    assert provenance.raw_response_reference is not None
    raw_path = run_dir / provenance.raw_response_reference
    assert raw_path.read_text(encoding="utf-8") == execution.raw_response
    view_path = run_dir / request.visual_views[0].payload_reference
    assert view_path.read_bytes() == _SEMANTIC_VIEW_PAYLOAD
    view_entry = next(
        entry
        for entry in manifest.file_inventory
        if entry.path == request.visual_views[0].payload_reference
    )
    assert view_entry.content_hash == f"sha256:{request.visual_views[0].sha256}"
    reopened = PerceptionRunReader(run_dir).list_semantic_executions()
    assert reopened == [execution]
    assert PerceptionRunReader(run_dir).verify_integrity() == []


def test_semantic_canonical_output_remains_reopenable_when_debug_is_disabled(
    tmp_path: Path,
) -> None:
    execution = _semantic_execution()
    writer = _write_run(tmp_path, semantic_debug_level=SemanticDebugLevel.NONE)
    writer.add_result(_result("frame-0001", "run-0001", claims=execution.parsed.claims))
    writer.add_semantic_view_payload(execution.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    _add_semantic_outcome(writer, execution)
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    assert not (run_dir / "debug/40-semantic-interpretation").exists()
    assert PerceptionRunReader(run_dir).list_semantic_executions() == [execution]
    assert PerceptionRunReader(run_dir).verify_integrity() == []


def test_finalize_rejects_semantic_execution_without_owning_result(tmp_path: Path) -> None:
    execution = _semantic_execution()
    writer = _write_run(tmp_path)
    writer.add_semantic_view_payload(execution.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    writer.add_stage_outcomes(
        (
            StageOutcome(
                stage_id="semantic_interpretation",
                status=StageStatus.SUCCEEDED,
                output=execution,
                duration_ms=4.0,
            ),
        )
    )

    with pytest.raises(RunArtifactError, match="does not resolve to exactly one result"):
        writer.finalize()


def test_finalize_rejects_unmaterialized_semantic_claims(tmp_path: Path) -> None:
    execution = _semantic_execution()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_semantic_view_payload(execution.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    writer.add_stage_outcomes(
        (
            StageOutcome(
                stage_id="semantic_interpretation",
                status=StageStatus.SUCCEEDED,
                output=execution,
                duration_ms=4.0,
            ),
        )
    )

    with pytest.raises(RunArtifactError, match="claims were not materialized"):
        writer.finalize()


def test_finalize_rejects_missing_semantic_view_payload(tmp_path: Path) -> None:
    execution = _semantic_execution()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", claims=execution.parsed.claims))
    writer.add_stage_outcomes(
        (
            StageOutcome(
                stage_id="semantic_interpretation",
                status=StageStatus.SUCCEEDED,
                output=execution,
                duration_ms=4.0,
            ),
        )
    )

    with pytest.raises(RunArtifactError, match="missing semantic view payload"):
        writer.finalize()


def test_writer_rejects_semantic_view_payload_with_wrong_hash(tmp_path: Path) -> None:
    execution = _semantic_execution()
    contradictory_view = replace(execution.request.visual_views[0], sha256="0" * 64)
    writer = _write_run(tmp_path)

    with pytest.raises(RunArtifactError, match="hash does not match"):
        writer.add_semantic_view_payload(contradictory_view, _SEMANTIC_VIEW_PAYLOAD)


def test_abstained_execution_persists_all_referenced_input_evidence(tmp_path: Path) -> None:
    execution, feature, scene_context = _abstained_semantic_execution()
    feature_payload = np.array([1.0, 2.0], dtype="float32")
    writer = _write_run(tmp_path)
    writer.add_result(
        _result(
            "frame-0001",
            "run-0001",
            features=(feature,),
            scene_context=scene_context,
        )
    )
    writer.add_feature_payload(feature, SourceObservationId("frame-0001"), feature_payload)
    writer.add_semantic_view_payload(execution.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    _add_semantic_outcome(writer, execution)
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    assert reader.list_semantic_executions()[0] == execution
    np.testing.assert_array_equal(
        reader.feature_store().load(SourceObservationId("frame-0001"), feature.feature_id),
        feature_payload,
    )
    assert reader.list_results()[0].scene_context == scene_context


@pytest.mark.parametrize(
    ("missing", "message"),
    (
        ("region", "region_id does not resolve"),
        ("feature_payload", "feature payload was not persisted"),
        ("scene_context", "scene_context_reference does not resolve"),
    ),
)
def test_finalize_rejects_unresolved_abstained_execution_inputs(
    tmp_path: Path, missing: str, message: str
) -> None:
    execution, feature, scene_context = _abstained_semantic_execution()
    result = _result(
        "frame-0001",
        "run-0001",
        features=(feature,),
        scene_context=scene_context,
    )
    if missing == "region":
        result = replace(result, regions=(), features=())
    elif missing == "scene_context":
        result = replace(result, scene_context=None)
    writer = _write_run(tmp_path)
    writer.add_result(result)
    if missing not in {"feature_payload", "region"}:
        writer.add_feature_payload(
            feature,
            SourceObservationId("frame-0001"),
            np.array([1.0, 2.0], dtype="float32"),
        )
    writer.add_semantic_view_payload(execution.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    _add_semantic_outcome(writer, execution)

    with pytest.raises(RunArtifactError, match=message):
        writer.finalize()


def test_manifest_claim_count_includes_scene_context_claims(tmp_path: Path) -> None:
    execution = _semantic_execution()
    scene_claim = replace(
        execution.parsed.claims[0],
        claim_id=ClaimId("scene-claim-0001"),
        region_id=None,
    )
    scene_context = SceneContext(
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId("run-0001--frame-0001"),
        provenance=scene_claim.provenance,
        claims=(scene_claim,),
    )
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", scene_context=scene_context))

    manifest = writer.finalize()

    assert manifest.claim_count == 1
    assert "Claims: 1" in (_run_dir(tmp_path) / "README.md").read_text(encoding="utf-8")


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


def test_reader_rejects_pre_semantic_contract_schema_before_reading_results(
    tmp_path: Path,
) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()
    manifest_path = _run_dir(tmp_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "0.3.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunArtifactError, match=r"unsupported.*0\.3\.0"):
        PerceptionRunReader(_run_dir(tmp_path))


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
