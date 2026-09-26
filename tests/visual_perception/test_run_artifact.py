import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    BackendDiagnostics,
    BackendProvenance,
    BoundingBox,
    BoundingBox2D,
    ClaimId,
    DiscoveryPass,
    FailedSemanticInterpretation,
    FeatureId,
    FeatureScope,
    IncompleteRunArtifactError,
    MergeDecision,
    MergeKind,
    NormalizationConfig,
    PassKind,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    Region2D,
    RegionDiscoveryAudit,
    RegionId,
    RejectedRegionCandidate,
    RejectionReason,
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
    execute_stage_graph,
    parse_semantic_response,
    redact_semantic_secrets,
    render_semantic_prompt,
    write_semantic_audit,
)
from contextmap.visual_perception import run_artifact as run_artifact_module
from contextmap.visual_perception.region_models import InlineMask

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0.1"
)


def _run_dir(tmp_path: Path, run_index: int = 1) -> Path:
    """Diretório final escolhido pelo chamador; o leitor abre exatamente este caminho."""
    return tmp_path / f"run-{run_index:04d}"


def _current_rss_bytes() -> int:
    """RSS corrente do processo, não o pico (`ru_maxrss` é monotônico e esconderia a liberação)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    pytest.skip("VmRSS unavailable on this platform")


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
        output_dir=_run_dir(tmp_path, run_index),
        sequence_name=sequence_name,
        run_id=PerceptionRunId(f"run-{run_index:04d}"),
        run_index=run_index,
        sequence_artifact_id="corridor-02-a1b2c3",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_discovery"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
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


def test_a_failed_stage_is_persisted_with_its_exception_type_and_traceback(
    tmp_path: Path,
) -> None:
    """VP-11: the metrics kept only ``str(error)``, empty for ``ValueError()``."""

    def failing(context: object) -> None:
        raise ValueError()

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_stage_outcomes(
        execute_stage_graph(
            [
                StageDefinition(
                    stage_id="region_discovery", capability="region_discovery", run=failing
                )
            ]
        )
    )
    writer.finalize()

    metrics_path = _run_dir(tmp_path) / "metrics" / "stage-timings.jsonl"
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert record["status"] == StageStatus.FAILED.value
    assert record["error"] == ""
    assert record["error_type"] == "ValueError"
    assert record["error_traceback"].endswith("ValueError\n")


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

    assert manifest.schema_version == "0.6.0"
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


def test_the_run_appears_exactly_at_output_dir_and_nothing_else_is_created(
    tmp_path: Path,
) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))

    writer.finalize()

    assert list(tmp_path.iterdir()) == [_run_dir(tmp_path)]
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "runs.json").exists()


def test_run_id_and_run_index_are_recorded_as_supplied(tmp_path: Path) -> None:
    writer = _write_run(tmp_path, run_index=7)
    writer.add_result(_result("frame-0001", "run-0007"))

    manifest = writer.finalize()

    reopened = PerceptionRunReader(_run_dir(tmp_path, 7)).manifest
    assert (manifest.run_id, manifest.run_index) == ("run-0007", 7)
    assert (reopened.run_id, reopened.run_index) == ("run-0007", 7)


def test_finalize_refuses_a_second_run_at_the_same_directory_without_altering_the_first(
    tmp_path: Path,
) -> None:
    first = _write_run(tmp_path)
    first.add_result(_result("frame-0001", "run-0001"))
    first.finalize()
    run_dir = _run_dir(tmp_path)
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    second = _write_run(tmp_path)
    second.add_result(_result("frame-0002", "run-0001"))

    with pytest.raises(RunArtifactError, match="already exists"):
        second.finalize()

    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert list(tmp_path.iterdir()) == [run_dir]  # e o run recusado não deixou nada ao redor
    assert PerceptionRunReader(run_dir).verify_integrity() == []


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


def test_the_inventory_is_checked_without_reading_whole_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VP-07: finalize() and verify_integrity() read every inventoried file whole to hash it.

    A dense feature or a mask is a ``.npy`` payload that can be large, so the check must
    stream it in chunks like ``FeatureStoreReader.load`` does since #518.
    """
    feature = _dense_feature()
    read_whole: list[Path] = []
    real_read_bytes = Path.read_bytes

    def spying_read_bytes(path: Path) -> bytes:
        read_whole.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", spying_read_bytes)
    writer = _write_run(tmp_path)
    writer.add_result(
        replace(
            _masked_result("frame-0001", _full_frame_mask(width=8, height=6)), features=(feature,)
        )
    )
    writer.add_feature_payload(
        feature, SourceObservationId("frame-0001"), np.zeros((2, 2), dtype="float32")
    )
    writer.finalize()
    reader = PerceptionRunReader(_run_dir(tmp_path))
    assert reader.verify_integrity() == []

    assert [path for path in read_whole if path.suffix == ".npy"] == []


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


def _full_frame_mask(width: int = 640, height: int = 480) -> InlineMask:
    data = tuple((x // 40 + y // 40) % 2 == 0 for y in range(height) for x in range(width))
    return InlineMask(np.array(data, dtype=bool).reshape(height, width))


def test_full_frame_mask_is_persisted_compactly_and_lazily_loadable(tmp_path: Path) -> None:
    """Regression test for #378: masks move out of results.jsonl into a compact lazy store."""
    mask = _full_frame_mask()
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=0, y=0, width=640, height=480),
        provenance=_PROVENANCE,
        source_observation_id=SourceObservationId("frame-0001"),
        image_width=640,
        image_height=480,
        area_pixels=float(mask.area),
        mask=mask,
    )
    result = PerceptionResult(
        result_id=PerceptionResultId("run-0001--frame-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(region,),
    )

    writer = _write_run(tmp_path)
    writer.add_result(result)
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    results_path = run_dir / "outputs" / "results.jsonl"
    # Was ~923 KB for a single full-frame mask before the fix (#378).
    assert results_path.stat().st_size < 5_000

    reader = PerceptionRunReader(run_dir)
    assert reader.verify_integrity() == []
    reopened_region = reader.list_results()[0].regions[0]
    assert reopened_region.mask is None
    assert reopened_region.mask_reference is not None
    assert reopened_region.mask_reference.startswith("outputs/masks/")

    loaded_mask = reader.mask_store().load(
        SourceObservationId("frame-0001"), RegionId("region-0001")
    )
    assert loaded_mask == mask


def test_a_run_with_no_masks_has_an_empty_mask_store(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    assert reader.mask_store().region_keys() == ()


def test_iter_results_streams_instead_of_materializing_every_result(tmp_path: Path) -> None:
    """A reader must be able to walk a run without holding all of it (#517)."""
    writer = _write_run(tmp_path)
    for index in range(5):
        writer.add_result(_result(f"frame-{index:04d}", "run-0001"))
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    streamed = reader.iter_results()
    assert not isinstance(streamed, list), "iter_results() must be lazy, not a built list"
    assert [r.source_observation_id for r in streamed] == [
        r.source_observation_id for r in reader.list_results()
    ]


def test_result_lookup_stops_at_the_match_instead_of_decoding_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """result() used to call list_results(), so one lookup cost the entire run (#517)."""
    writer = _write_run(tmp_path)
    for index in range(20):
        writer.add_result(_result(f"frame-{index:04d}", "run-0001"))
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    decoded = 0
    real_decode = run_artifact_module.decode_perception_result

    def counting_decode(payload: dict[str, object]) -> PerceptionResult:
        nonlocal decoded
        decoded += 1
        return real_decode(payload)

    monkeypatch.setattr(run_artifact_module, "decode_perception_result", counting_decode)
    found = reader.result(SourceObservationId("frame-0000"))

    assert found.source_observation_id == SourceObservationId("frame-0000")
    assert decoded == 1, f"looking up the first result decoded {decoded} of 20 results"


def test_iter_semantic_executions_streams_instead_of_materializing(tmp_path: Path) -> None:
    """Each execution carries its full raw_response text, so eager reads scale badly."""
    execution = _semantic_execution()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", claims=execution.parsed.claims))
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
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    streamed = reader.iter_semantic_executions()
    assert not isinstance(streamed, list), "iter_semantic_executions() must be lazy"
    assert list(streamed) == [execution]


def _without_raw_response_reference(line: str) -> str:
    record = json.loads(line)
    del record["raw_response_reference"]
    return json.dumps(record)


@pytest.mark.parametrize(
    "malform",
    [
        pytest.param(_without_raw_response_reference, id="missing-raw-response-reference"),
        pytest.param(lambda line: "not json", id="not-json"),
    ],
)
def test_a_malformed_semantic_execution_record_is_an_artifact_error(
    tmp_path: Path, malform: Callable[[str], str]
) -> None:
    """VP-10: a malformed line escaped as a raw KeyError or JSONDecodeError."""
    execution = _semantic_execution()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", claims=execution.parsed.claims))
    writer.add_semantic_view_payload(execution.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    _add_semantic_outcome(writer, execution)
    writer.finalize()
    executions_path = _run_dir(tmp_path) / "outputs" / "semantic-interpretations.jsonl"
    executions_path.write_text(f"{malform(executions_path.read_text().strip())}\n")

    with pytest.raises(
        RunArtifactError,
        match=r"invalid semantic execution record at outputs/semantic-interpretations\.jsonl:1",
    ):
        list(PerceptionRunReader(_run_dir(tmp_path)).iter_semantic_executions())


def _masked_result(observation_id: str, mask: InlineMask) -> PerceptionResult:
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=0, y=0, width=mask.width, height=mask.height),
        provenance=_PROVENANCE,
        source_observation_id=SourceObservationId(observation_id),
        image_width=mask.width,
        image_height=mask.height,
        area_pixels=float(mask.area),
        mask=mask,
    )
    return PerceptionResult(
        result_id=PerceptionResultId(f"run-0001--{observation_id}"),
        source_observation_id=SourceObservationId(observation_id),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(region,),
    )


def _staging_dir(tmp_path: Path, run_index: int = 1) -> Path:
    staging = [p for p in tmp_path.iterdir() if p.name.startswith(f".tmp-run-{run_index:04d}-")]
    assert len(staging) == 1, f"expected exactly one staging directory, found {staging}"
    return staging[0]


def test_mask_pixels_reach_disk_when_the_result_is_added_not_at_finalize(
    tmp_path: Path,
) -> None:
    """A mask must not sit in RAM until finalize().

    Two real 360-frame runs died holding every frame's mask: a 640x480 InlineMask is a
    tuple of 307200 pointers (~2.36 MB measured), so 7828 regions cost ~18 GB before
    finalize() ever ran.
    """
    writer = _write_run(tmp_path)
    writer.add_result(_masked_result("frame-0001", _full_frame_mask()))

    masks = list((_staging_dir(tmp_path) / "outputs" / "masks").rglob("*.npy"))
    assert masks, "add_result() must persist the mask instead of buffering its pixels"


def test_writer_memory_does_not_grow_with_the_number_of_masked_results(
    tmp_path: Path,
) -> None:
    """Peak RSS must not scale with frame count; this is the property the campaign needs."""
    frames = 40
    retained_bytes_if_buffered = frames * 640 * 480 * 8  # ~98 MB of pointers

    writer = _write_run(tmp_path)
    before = _current_rss_bytes()
    for index in range(frames):
        writer.add_result(_masked_result(f"frame-{index:04d}", _full_frame_mask()))
    growth = _current_rss_bytes() - before

    assert growth < retained_bytes_if_buffered // 4, (
        f"writer grew {growth / 2**20:.1f} MB over {frames} masked results; "
        f"buffering them all would cost ~{retained_bytes_if_buffered / 2**20:.0f} MB"
    )


def test_feature_arrays_reach_disk_when_added_not_at_finalize(tmp_path: Path) -> None:
    """Dense DINOv2 payloads are megabytes per frame; they must not queue up in RAM."""
    feature = _dense_feature()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    writer.add_feature_payload(
        feature,
        SourceObservationId("frame-0001"),
        np.zeros((2, 2), dtype="float32"),
    )

    payloads = list((_staging_dir(tmp_path) / "outputs" / "features").rglob("*.npy"))
    assert payloads, "add_feature_payload() must persist the array instead of buffering it"


def test_writer_memory_does_not_grow_with_the_number_of_feature_payloads(
    tmp_path: Path,
) -> None:
    """A real run queues one dense array per frame; 1500 frames must not cost 1500 arrays."""
    frames = 30
    rows = cols = 768  # 2.25 MB per array, the order of a real DINOv2 dense payload
    retained_bytes_if_buffered = frames * rows * cols * 4

    writer = _write_run(tmp_path)
    before = _current_rss_bytes()
    for index in range(frames):
        feature = replace(
            _dense_feature(f"feature-dense-{index:04d}"),
            shape=(rows, cols),
            payload_reference=f"frame-{index:04d}/feature-dense-{index:04d}.npy",
        )
        writer.add_feature_payload(
            feature,
            SourceObservationId(f"frame-{index:04d}"),
            # np.zeros() would not fault its pages in, so it would not show up in RSS at all.
            np.full((rows, cols), float(index), dtype="float32"),
        )
    growth = _current_rss_bytes() - before

    assert growth < retained_bytes_if_buffered // 4, (
        f"writer grew {growth / 2**20:.1f} MB over {frames} dense payloads; "
        f"buffering them all would cost ~{retained_bytes_if_buffered / 2**20:.0f} MB"
    )


def test_semantic_view_payloads_reach_disk_when_added_not_at_finalize(tmp_path: Path) -> None:
    """The exact pixels sent to the VLM are megabytes per frame; they must not be buffered."""
    view = _semantic_execution().request.visual_views[0]
    writer = _write_run(tmp_path)
    writer.add_semantic_view_payload(view, _SEMANTIC_VIEW_PAYLOAD)

    written = (_staging_dir(tmp_path) / view.payload_reference).is_file()
    assert written, "add_semantic_view_payload() must persist the bytes instead of buffering them"


def test_writer_memory_does_not_grow_with_the_number_of_view_payloads(tmp_path: Path) -> None:
    """A real frame sends ~6 views; 1500 frames must not hold 9000 encoded images."""
    views = 60
    payload_bytes = 1 << 20  # 1 MB, the order of one encoded 640x480 frame
    writer = _write_run(tmp_path)

    before = _current_rss_bytes()
    for index in range(views):
        payload = bytes((index % 251,)) * payload_bytes
        view = SemanticVisualView(
            view_id=f"view-{index:04d}",
            kind=VisualViewKind.FULL_FRAME,
            payload_reference=f"outputs/semantic-views/view-{index:04d}.jpg",
            source_observation_id=SourceObservationId(f"frame-{index:04d}"),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        writer.add_semantic_view_payload(view, payload)
    growth = _current_rss_bytes() - before

    assert growth < (views * payload_bytes) // 4, (
        f"writer grew {growth / 2**20:.1f} MB over {views} view payloads; "
        f"buffering them all would cost ~{views * payload_bytes / 2**20:.0f} MB"
    )


def test_writer_memory_does_not_grow_with_retained_stage_outputs(tmp_path: Path) -> None:
    """The real driver hands the region_discovery outcome straight to add_stage_outcomes().

    Its ``output`` is the same tuple of mask-bearing Region2D that add_result() receives, so
    retaining the StageOutcome retains the pixels a second time — and stage timings only ever
    serialize stage_id/status/duration_ms/error, never the output.
    """
    frames = 40
    retained_bytes_if_buffered = frames * 640 * 480 * 8

    writer = _write_run(tmp_path)
    before = _current_rss_bytes()
    for index in range(frames):
        observation_id = f"frame-{index:04d}"
        result = _masked_result(observation_id, _full_frame_mask())
        writer.add_stage_outcomes(
            (
                StageOutcome(
                    stage_id="region_discovery",
                    status=StageStatus.SUCCEEDED,
                    output=result.regions,
                    duration_ms=1.0,
                ),
            )
        )
    growth = _current_rss_bytes() - before

    assert growth < retained_bytes_if_buffered // 4, (
        f"writer grew {growth / 2**20:.1f} MB over {frames} stage outcomes; "
        f"retaining their outputs would cost ~{retained_bytes_if_buffered / 2**20:.0f} MB"
    )


def test_masks_added_one_by_one_are_all_readable_after_finalize(tmp_path: Path) -> None:
    """Streaming the masks out must not lose or mix up any of them."""
    masks = {
        f"frame-{index:04d}": _full_frame_mask(width=8, height=4 + index) for index in range(5)
    }
    writer = _write_run(tmp_path)
    for observation_id, mask in masks.items():
        writer.add_result(_masked_result(observation_id, mask))
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    assert reader.verify_integrity() == []
    store = reader.mask_store()
    for observation_id, mask in masks.items():
        loaded = store.load(SourceObservationId(observation_id), RegionId("region-0001"))
        assert loaded == mask


def _failed_semantic_interpretation() -> FailedSemanticInterpretation:
    """A real backend call whose response was observed but rejected by the parser."""
    from contextmap.visual_perception import SemanticParseFailure

    execution = _semantic_execution()
    raw = json.dumps({"abstained": False, "claims": [{"hypothesis": "a door"}]})
    return FailedSemanticInterpretation(
        request=replace(execution.request, request_id=SemanticRequestId("region-request-0002")),
        rendered_prompt=execution.rendered_prompt,
        raw_response=raw,
        raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        provenance=execution.parsed.claims[0].provenance,
        diagnostics=execution.diagnostics,
        failure=SemanticParseFailure(
            kind="SemanticResponseParseError",
            message="claim[0] is missing required fields: ['confidence']",
        ),
        effective_configuration={"backend": "fake"},
        occurred_at="2026-01-01T00:00:00+00:00",
    )


def test_failed_semantic_interpretations_are_a_first_class_output(tmp_path: Path) -> None:
    """PR #438 review: an invalid response is still observed evidence, not a lost log line.

    It gets its own contractual stream so the 0.5.0 success contract, and every artifact
    already frozen under it, stay readable.
    """
    failed = _failed_semantic_interpretation()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    # The view that produced the rejected response is evidence too, so it must be persisted.
    writer.add_semantic_view_payload(failed.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    writer.add_failed_semantic_interpretation(failed)
    manifest = writer.finalize()

    assert manifest.schema_version == "0.6.0", "the success contract must not break"
    run_dir = _run_dir(tmp_path)
    reader = PerceptionRunReader(run_dir)
    assert reader.verify_integrity() == []

    reopened = reader.list_failed_semantic_interpretations()
    assert reopened == [failed]
    # Contractual: inventoried, so losing it is detected.
    assert any(
        entry.path == "outputs/semantic-interpretation-failures.jsonl"
        for entry in manifest.file_inventory
    )


def test_failed_interpretations_share_the_attempt_identity_with_successes(
    tmp_path: Path,
) -> None:
    """attempted = parsed + parse_failed must be reconcilable by request_id, not guessed."""
    execution = _semantic_execution()
    failed = _failed_semantic_interpretation()

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", claims=execution.parsed.claims))
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
    writer.add_failed_semantic_interpretation(failed)
    writer.finalize()

    reader = PerceptionRunReader(_run_dir(tmp_path))
    parsed_ids = {str(e.request.request_id) for e in reader.iter_semantic_executions()}
    failed_ids = {str(f.request.request_id) for f in reader.iter_failed_semantic_interpretations()}

    assert parsed_ids == {"region-request-0001"}
    assert failed_ids == {"region-request-0002"}
    assert not (parsed_ids & failed_ids), "one attempt must not appear in both streams"


def test_a_failure_view_reference_without_its_payload_is_refused(tmp_path: Path) -> None:
    """P1 of the PR #438 re-review: a dangling view reference must not pass as integral.

    The failed interpretation names a view by payload_reference and SHA-256. If those bytes
    never reach the artifact they stay in the executor's scratch, which is deleted, leaving
    the artifact citing visual evidence nobody can recover -- exactly the evidence that
    produced the rejected response.
    """
    failed = _failed_semantic_interpretation()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_failed_semantic_interpretation(failed)

    with pytest.raises(RunArtifactError, match="missing semantic view payload"):
        writer.finalize()


def test_a_failure_view_payload_is_inventoried_even_without_any_success(tmp_path: Path) -> None:
    """The view of a failed call is contractual evidence on its own."""
    failed = _failed_semantic_interpretation()
    view = failed.request.visual_views[0]

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_semantic_view_payload(view, _SEMANTIC_VIEW_PAYLOAD)
    writer.add_failed_semantic_interpretation(failed)
    manifest = writer.finalize()

    run_dir = _run_dir(tmp_path)
    reader = PerceptionRunReader(run_dir)
    assert reader.verify_integrity() == []
    assert (run_dir / view.payload_reference).read_bytes() == _SEMANTIC_VIEW_PAYLOAD
    assert any(entry.path == view.payload_reference for entry in manifest.file_inventory)


def test_a_failed_interpretation_gets_the_same_per_request_debug_directory(
    tmp_path: Path,
) -> None:
    """Item 2 of the PR #438 re-review: debug/<request-id>/ must exist for both outcomes.

    Non-canonical, but it is where a human goes to diagnose a rejected response, and it was
    part of the agreed shape. One generalized writer, not a second implementation that can
    drift from the successful one.
    """
    failed = _failed_semantic_interpretation()
    written = write_semantic_audit(
        run_root=tmp_path,
        execution=failed,
        debug_level=SemanticDebugLevel.FULL,
    )

    root = tmp_path / "debug" / "40-semantic-interpretation" / str(failed.request.request_id)
    assert root.is_dir()
    assert {path.name for path in root.iterdir()} == {
        "request.json",
        "prompt.txt",
        "diagnostics.json",
        "parse-failure.json",
        "raw-response.txt",
    }
    # The rejected response is readable verbatim, which is the whole point.
    assert (root / "raw-response.txt").read_text(encoding="utf-8") == failed.raw_response
    recorded = json.loads((root / "parse-failure.json").read_text(encoding="utf-8"))
    assert recorded["kind"] == "SemanticResponseParseError"
    assert recorded["message"] == failed.failure.message
    assert recorded["raw_response_sha256"] == failed.raw_response_sha256
    assert all(path.startswith("debug/40-semantic-interpretation/") for path in written)


def test_a_failed_interpretation_writes_no_debug_when_disabled(tmp_path: Path) -> None:
    """SemanticDebugLevel.NONE stays authoritative for failures too."""
    written = write_semantic_audit(
        run_root=tmp_path,
        execution=_failed_semantic_interpretation(),
        debug_level=SemanticDebugLevel.NONE,
    )

    assert written == ()
    assert not (tmp_path / "debug").exists()


def test_the_run_writes_the_debug_directory_for_a_failed_interpretation(tmp_path: Path) -> None:
    """The per-request debug dir must appear in a finalized run, not only when called directly."""
    failed = _failed_semantic_interpretation()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_semantic_view_payload(failed.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    writer.add_failed_semantic_interpretation(failed)
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    root = run_dir / "debug" / "40-semantic-interpretation" / str(failed.request.request_id)
    assert (root / "parse-failure.json").is_file()
    assert (root / "raw-response.txt").read_text(encoding="utf-8") == failed.raw_response
    assert PerceptionRunReader(run_dir).verify_integrity() == []


def test_a_run_with_semantic_attempts_always_writes_the_failures_stream(tmp_path: Path) -> None:
    """An absent failures file must mean "predates the stream", never "zero failures".

    Writing it only when a failure occurred made those two indistinguishable, so a run could
    report a 0% parse-failure rate simply because it never tracked them.
    """
    execution = _semantic_execution()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", claims=execution.parsed.claims))
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
    manifest = writer.finalize()

    run_dir = _run_dir(tmp_path)
    failures_file = run_dir / "outputs" / "semantic-interpretation-failures.jsonl"
    assert failures_file.is_file(), "a run that attempted interpretation must declare its failures"
    assert failures_file.read_text(encoding="utf-8") == "", "zero failures is an empty file"
    assert any(
        entry.path == "outputs/semantic-interpretation-failures.jsonl"
        for entry in manifest.file_inventory
    )
    assert PerceptionRunReader(run_dir).list_failed_semantic_interpretations() == []


def test_every_run_declares_its_failure_tracking_even_with_no_attempt(tmp_path: Path) -> None:
    """Absence of the stream must mean exactly one thing: the artifact predates tracking.

    Emitting it only for runs that attempted interpretation left a hole: a legacy artifact
    whose every attempt failed also has zero executions and no stream, so it looked complete.
    """
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    assert (run_dir / "outputs" / "semantic-interpretation-failures.jsonl").is_file()
    assert PerceptionRunReader(run_dir).tracks_semantic_failures() is True


def test_tracking_is_decided_by_the_manifest_not_by_the_file_on_disk(tmp_path: Path) -> None:
    """A corrupted new artifact must not be silently reclassified as a legacy one."""
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    (run_dir / "outputs" / "semantic-interpretation-failures.jsonl").unlink()

    reader = PerceptionRunReader(run_dir)
    with pytest.raises(RunArtifactError, match="inventoried"):
        reader.tracks_semantic_failures()


def _characterized_run(tmp_path: Path) -> Path:
    """Write a run touching every pre-audit output: masks, features, semantics and timings."""
    execution = _semantic_execution()
    failed = _failed_semantic_interpretation()
    feature = _dense_feature("feature-dense-0001")
    masked = _masked_result("frame-0001", _full_frame_mask(width=4, height=3))
    writer = _write_run(tmp_path)
    writer.add_result(replace(masked, features=(feature,), claims=execution.parsed.claims))
    writer.add_result(_result("frame-0002", "run-0001"))
    writer.add_feature_payload(
        feature,
        SourceObservationId("frame-0001"),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
    )
    writer.add_semantic_view_payload(execution.request.visual_views[0], _SEMANTIC_VIEW_PAYLOAD)
    _add_semantic_outcome(writer, execution)
    writer.add_failed_semantic_interpretation(failed)
    writer.finalize()
    return _run_dir(tmp_path)


def _pre_audit_contents(run_dir: Path) -> dict[str, str]:
    """SHA-256 of every file a pre-audit run wrote, with the manifest reduced to its stable part.

    ``created_at`` is wall-clock, so it leaves the manifest and the README; ``schema_version`` and
    the inventory entry of ``outputs/region-discovery-audit.jsonl`` are what #611 adds on purpose.
    """
    audit_path = "outputs/region-discovery-audit.jsonl"
    contents = {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and str(path.relative_to(run_dir)) not in {"manifest.json", "README.md", audit_path}
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    del manifest["created_at"], manifest["schema_version"]
    manifest["file_inventory"] = [
        entry for entry in manifest["file_inventory"] if entry["path"] != audit_path
    ]
    contents["manifest.json (stable part)"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    readme = (run_dir / "README.md").read_text(encoding="utf-8").splitlines()
    contents["README.md (stable part)"] = hashlib.sha256(
        "\n".join(line for line in readme if not line.startswith("- Created at:")).encode("utf-8")
    ).hexdigest()
    return contents


def test_every_pre_audit_output_stays_byte_identical(tmp_path: Path) -> None:
    """#611 only adds the audit table: every other file must keep its exact bytes.

    The digests were recorded from the 0.5.0 writer, before the audit existed, except for
    ``metrics/stage-timings.jsonl`` and the manifest entry that inventories it: since #619 (VP-11)
    every stage record also carries ``error_type`` and ``error_traceback``, ``null`` for the
    succeeded stage recorded here. Nothing else in the manifest changed.
    """
    contents = _pre_audit_contents(_characterized_run(tmp_path))

    semantic_debug = "debug/40-semantic-interpretation"
    assert contents == {
        "README.md (stable part)": (
            "0863c5f37eef9a97f6624a6b75d3ed5ddeb4a46d1bf55d80859c9014c07608fd"
        ),
        f"{semantic_debug}/region-request-0001/diagnostics.json": (
            "8803b582fcc7b66400a0079331c99dcaf98e2350bcec3501ebc64057be4d2fcd"
        ),
        f"{semantic_debug}/region-request-0001/parsed-response.json": (
            "4da42836be9c59a62f74563b2199a72895c13fdb39c2e4be9b096127a53639e4"
        ),
        f"{semantic_debug}/region-request-0001/prompt.txt": (
            "1e3b2cd9f6037b2719bdb84c9784c291f999e0ab7b967d0aeb6a75329cd26ad0"
        ),
        f"{semantic_debug}/region-request-0001/raw-response.txt": (
            "f0b5f83cde14bedd4abdc7cab53e8c51441720c14ba4f9726c6dab383157b76c"
        ),
        f"{semantic_debug}/region-request-0001/request.json": (
            "d08d77c010881319e950d09842e1c147832a9c48dcc991da8abe8d285d10e58a"
        ),
        f"{semantic_debug}/region-request-0001/semantic-claims.json": (
            "3a76aab90efea82b060d402400603f27c7e3bc452bf514956153d7f2daa6d52e"
        ),
        f"{semantic_debug}/region-request-0002/diagnostics.json": (
            "773266608cbf30e971fc200ebc35ac212ada80a55d3ec0259ba14a3a1c371600"
        ),
        f"{semantic_debug}/region-request-0002/parse-failure.json": (
            "a3031ea5d6becbcb0527dbd6e623f7a1b8199b2b30faabeb3814d29832672ddb"
        ),
        f"{semantic_debug}/region-request-0002/prompt.txt": (
            "1e3b2cd9f6037b2719bdb84c9784c291f999e0ab7b967d0aeb6a75329cd26ad0"
        ),
        f"{semantic_debug}/region-request-0002/raw-response.txt": (
            "f61a27316cdea5e281d848b87d9a077d59a6b2998dea0c52773137f98773741f"
        ),
        f"{semantic_debug}/region-request-0002/request.json": (
            "079016ab6ddf9054b6f82f0452d8ab3fb381d720123b4c06dd3796e4738f23e5"
        ),
        "manifest.json (stable part)": (
            "f47b79899eb493c616a080a387448cbc727ce8004d67e4e387b061b09af60a2f"
        ),
        "metrics/stage-timings.jsonl": (
            "ff75fc1139460aceccc7cc6d72b2f35e9d79f5ec467ce8e4983001ac4d8d3766"
        ),
        "outputs/features/feature-index.jsonl": (
            "8793dc07b59cd8b3f3976e1f8e122c0eebc1b606849f998e1464394d4ed5f93c"
        ),
        "outputs/features/frame-0001/feature-dense-0001.npy": (
            "e8072b61f5d81a3cc4dc59b9d5e14187b20b5d8a3ddd8e6d0bc5128bda5f27aa"
        ),
        "outputs/masks/frame-0001/region-0001.npy": (
            "4d7f924ebd7fdcc67ba12c379b69285c0df6eece071a00cbc299f590bc86cd2e"
        ),
        "outputs/masks/mask-index.jsonl": (
            "a83f82a445bc79e66cf0f99167afc9e99713a4c230d5b63a969e99c0748a2072"
        ),
        "outputs/results.jsonl": (
            "1e58cc8f229f14e56b149b62da2ced55f82de2e9f6c459f4e055ec4e02080c0e"
        ),
        "outputs/semantic-interpretation-failures.jsonl": (
            "672b54900b35a40dd8d68dd4bc5473c0c137c873e4f2e100d1abcdabbb2daf1e"
        ),
        "outputs/semantic-interpretations.jsonl": (
            "09c77d8efe71fbbe42482a8a3cbc01d402959e9e97b7bf6e95983b5a66cb40b6"
        ),
        "outputs/semantic-views/crop-0001.jpg": (
            "d19bbf3d505b6b0023eea97255d7e44d52b831e5f2c426e7670e502af0d19334"
        ),
    }


_AUDIT_TABLE = "outputs/region-discovery-audit.jsonl"


def _discovery_audit(observation_id: str = "frame-0001") -> RegionDiscoveryAudit:
    """One frame's audit: a pass, a rejection before and after normalization, and a merge."""
    return RegionDiscoveryAudit(
        source_observation_id=SourceObservationId(observation_id),
        backend=_PROVENANCE,
        passes=(
            DiscoveryPass(
                pass_id="full-frame", kind=PassKind.FULL_FRAME, window=BoundingBox(0, 0, 10, 10)
            ),
        ),
        diagnostics=(
            BackendDiagnostics(
                duration_ms=4.5,
                proposal_count=3,
                warnings=("slow pass",),
                metadata=(("raw_proposal_count", 5),),
            ),
        ),
        pass_rejections=(
            RejectedRegionCandidate(
                candidate_id="full-frame/p-3",
                reason=RejectionReason.REGION_BUDGET_EXCEEDED,
                detail="candidate exceeded configured per-pass budget",
                discovery_pass_id="full-frame",
            ),
        ),
        normalization_config_digest=NormalizationConfig().digest,
        normalization_rejections=(
            RejectedRegionCandidate(
                candidate_id="full-frame/p-2",
                reason=RejectionReason.MERGED_DUPLICATE,
                detail="merged into full-frame/p-1",
                discovery_pass_id="full-frame",
            ),
        ),
        merge_decisions=(
            MergeDecision(
                representative_candidate_id="full-frame/p-1",
                merged_candidate_id="full-frame/p-2",
                kind=MergeKind.IOU_DUPLICATE,
                iou=0.9,
                containment_fraction=1.0,
            ),
        ),
    )


def test_the_region_discovery_audit_is_a_contractual_output_read_by_observation(
    tmp_path: Path,
) -> None:
    """#611: rejections and merges of each frame are recoverable from the finalized run."""
    first, second = _discovery_audit("frame-0001"), _discovery_audit("frame-0002")
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_result(_result("frame-0002", "run-0001"))
    writer.add_region_discovery_audit(first)
    writer.add_region_discovery_audit(second)
    manifest = writer.finalize()

    assert _AUDIT_TABLE in {entry.path for entry in manifest.file_inventory}
    reader = PerceptionRunReader(_run_dir(tmp_path))
    assert reader.verify_integrity() == []
    assert reader.records_region_discovery_audit() is True
    assert list(reader.iter_region_discovery_audits()) == [first, second]
    assert reader.region_discovery_audit(SourceObservationId("frame-0002")) == second


def test_every_run_writes_the_audit_table_even_when_nothing_was_audited(tmp_path: Path) -> None:
    """An empty table means "recorded, nothing to report", never "not recorded"."""
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    manifest = writer.finalize()

    run_dir = _run_dir(tmp_path)
    assert (run_dir / _AUDIT_TABLE).read_text(encoding="utf-8") == ""
    assert _AUDIT_TABLE in {entry.path for entry in manifest.file_inventory}
    reader = PerceptionRunReader(run_dir)
    assert reader.records_region_discovery_audit() is True
    assert list(reader.iter_region_discovery_audits()) == []


def test_a_frame_without_an_audit_is_reported_not_invented(tmp_path: Path) -> None:
    """A frame whose discovery failed has a result but no audit; the lookup says so."""
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    with pytest.raises(RunArtifactError, match="no region discovery audit"):
        PerceptionRunReader(_run_dir(tmp_path)).region_discovery_audit(
            SourceObservationId("frame-0001")
        )


def test_a_second_audit_for_the_same_observation_is_refused(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_region_discovery_audit(_discovery_audit())

    with pytest.raises(RunArtifactError, match="duplicate region discovery audit"):
        writer.add_region_discovery_audit(_discovery_audit())


def test_an_audit_without_a_result_of_the_run_is_refused(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_region_discovery_audit(_discovery_audit("frame-0002"))

    with pytest.raises(RunArtifactError, match="does not resolve to a result"):
        writer.finalize()
    assert list(tmp_path.iterdir()) == []


def test_audits_cannot_be_added_after_finalize(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    with pytest.raises(RunArtifactError, match="after finalize"):
        writer.add_region_discovery_audit(_discovery_audit())


def _as_pre_audit_run(run_dir: Path) -> None:
    """Rewrite a finalized run as the 0.5.0 writer would have left it: no audit table at all."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "0.5.0"
    manifest["file_inventory"] = [
        entry for entry in manifest["file_inventory"] if entry["path"] != _AUDIT_TABLE
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / _AUDIT_TABLE).unlink()


def test_a_0_5_0_run_still_opens_and_reports_its_audit_as_not_recorded(tmp_path: Path) -> None:
    """v0.1.0 shipped 0.5.0 runs: they stay readable, and their missing audit is explicit."""
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()
    run_dir = _run_dir(tmp_path)
    _as_pre_audit_run(run_dir)

    reader = PerceptionRunReader(run_dir)

    assert reader.manifest.schema_version == "0.5.0"
    assert reader.verify_integrity() == []
    assert [str(result.source_observation_id) for result in reader.iter_results()] == ["frame-0001"]
    assert reader.records_region_discovery_audit() is False
    with pytest.raises(RunArtifactError, match="not recorded"):
        reader.iter_region_discovery_audits()
    with pytest.raises(RunArtifactError, match="not recorded"):
        reader.region_discovery_audit(SourceObservationId("frame-0001"))


def test_a_run_that_lost_its_audit_table_is_corrupt_not_a_0_5_0_run(tmp_path: Path) -> None:
    """Only the schema version says "not recorded"; a missing table is never reclassified."""
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()
    run_dir = _run_dir(tmp_path)
    (run_dir / _AUDIT_TABLE).unlink()

    with pytest.raises(RunArtifactError, match="missing"):
        PerceptionRunReader(run_dir).records_region_discovery_audit()

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_inventory"] = [
        entry for entry in manifest["file_inventory"] if entry["path"] != _AUDIT_TABLE
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunArtifactError, match="does not inventory"):
        PerceptionRunReader(run_dir).records_region_discovery_audit()


def test_an_invalid_audit_record_is_reported_as_an_artifact_error(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_region_discovery_audit(_discovery_audit())
    writer.finalize()
    run_dir = _run_dir(tmp_path)
    (run_dir / _AUDIT_TABLE).write_text('{"source_observation_id": "frame-0001"}\n')

    with pytest.raises(RunArtifactError, match="invalid region discovery audit record"):
        list(PerceptionRunReader(run_dir).iter_region_discovery_audits())
