"""Attempt accounting for Semantic Interpretation: attempted / parsed / parse_failed.

Counting only materialized executions made a stage with many parse failures look better than
it is. The numbers are derived from the two contractual streams by request_id, never from
StageOutcome and never inferred.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from contextmap.evaluation import (
    SemanticAttemptAccounting,
    SemanticMeasurementStatus,
    account_semantic_attempts,
    aggregate_semantic_attempts,
)
from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    SEMANTIC_PROMPT_TEMPLATES,
    BackendProvenance,
    BoundingBox2D,
    FailedSemanticInterpretation,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    Region2D,
    RegionId,
    RenderedSemanticPrompt,
    SemanticBackendDiagnostics,
    SemanticConfidencePolicy,
    SemanticInferenceProvenance,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticParseFailure,
    SemanticRequestId,
    SemanticVisualView,
    StageOutcome,
    StageStatus,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
)

_PAYLOAD = b"exact semantic view pixels"
_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0.1"
)


def _run_dir(tmp_path: Path, index: int = 1) -> Path:
    return tmp_path / f"run-{index:04d}"


def _writer(tmp_path: Path, index: int = 1) -> PerceptionRunWriter:
    return PerceptionRunWriter(
        output_dir=_run_dir(tmp_path, index),
        sequence_name="corridor-02",
        run_id=PerceptionRunId(f"run-{index:04d}"),
        run_index=index,
        sequence_artifact_id="corridor-02-a1b2c3",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_discovery"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    )


def _result(run_id: str) -> PerceptionResult:
    return PerceptionResult(
        result_id=PerceptionResultId(f"{run_id}--frame-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        run_id=PerceptionRunId(run_id),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(
            Region2D(
                region_id=RegionId("region-0001"),
                bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
                provenance=_PROVENANCE,
            ),
        ),
    )


def _request(run_id: str, request_id: str, view_name: str) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId(request_id),
        source_observation_id=SourceObservationId("frame-0001"),
        perception_result_id=PerceptionResultId(f"{run_id}--frame-0001"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(
            SemanticVisualView(
                view_id=f"v-{request_id}",
                kind=VisualViewKind.FULL_FRAME,
                payload_reference=f"outputs/semantic-views/{view_name}.png",
                source_observation_id=SourceObservationId("frame-0001"),
                sha256=hashlib.sha256(_PAYLOAD).hexdigest(),
            ),
        ),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )


def _rendered(request: SemanticInterpretationRequest) -> RenderedSemanticPrompt:
    return render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )


def _inference_provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="qwen",
            capability="semantic_interpreter",
            provider="alibaba",
            model="Qwen3-VL-4B",
            version="1",
        ),
        task_identity="scene",
        prompt_template_id="scene/v1",
        output_schema_version="semantic-response/1",
    )


def _execution(run_id: str, request_id: str, view_name: str) -> SemanticInterpretationExecution:
    request = _request(run_id, request_id, view_name)
    raw = json.dumps({"abstained": True, "claims": [], "scene_context": None})
    return SemanticInterpretationExecution(
        request=request,
        rendered_prompt=_rendered(request),
        raw_response=raw,
        parsed=parse_semantic_response(
            raw,
            request,
            _inference_provenance(),
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
        effective_configuration={"backend": "qwen"},
    )


def _failure(run_id: str, request_id: str, view_name: str) -> FailedSemanticInterpretation:
    request = _request(run_id, request_id, view_name)
    raw = json.dumps({"abstained": False, "claims": [{"hypothesis": "a door"}]})
    return FailedSemanticInterpretation(
        request=request,
        rendered_prompt=_rendered(request),
        raw_response=raw,
        raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        provenance=_inference_provenance(),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
        failure=SemanticParseFailure(
            kind="SemanticResponseParseError",
            message="claim[0] is missing required fields: ['confidence']",
        ),
        effective_configuration={"backend": "qwen"},
        occurred_at="2026-01-01T00:00:00+00:00",
    )


def _finalize(
    writer: PerceptionRunWriter,
    *,
    executions: Sequence[SemanticInterpretationExecution] = (),
    failures: Sequence[FailedSemanticInterpretation] = (),
) -> None:
    writer.add_result(_result(str(writer._run_id)))
    for execution in executions:
        writer.add_semantic_view_payload(execution.request.visual_views[0], _PAYLOAD)
        writer.add_stage_outcomes(
            (
                StageOutcome(
                    stage_id="scene_interpretation",
                    status=StageStatus.SUCCEEDED,
                    output=execution,
                    duration_ms=1.0,
                ),
            )
        )
    for failed in failures:
        writer.add_semantic_view_payload(failed.request.visual_views[0], _PAYLOAD)
        writer.add_failed_semantic_interpretation(failed)
    writer.finalize()


def test_a_run_with_only_successes_reports_no_parse_failures(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _finalize(writer, executions=[_execution("run-0001", "req-a", "a")])

    accounting = account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path)))

    assert accounting == SemanticAttemptAccounting(attempted=1, parsed=1, parse_failed=0)
    assert accounting.parse_failure_rate == 0.0


def test_a_mixed_run_counts_both_streams(tmp_path: Path) -> None:
    """The number that matters: 1 of 3 attempts never became a claim."""
    writer = _writer(tmp_path)
    _finalize(
        writer,
        executions=[_execution("run-0001", "req-a", "a"), _execution("run-0001", "req-b", "b")],
        failures=[_failure("run-0001", "req-c", "c")],
    )

    accounting = account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path)))

    assert accounting == SemanticAttemptAccounting(attempted=3, parsed=2, parse_failed=1)
    assert accounting.parse_failure_rate == pytest.approx(1 / 3)


def test_an_artifact_written_before_the_failures_stream_still_accounts(tmp_path: Path) -> None:
    """Runs frozen under the same schema version have no failures file; that is not an error."""
    writer = _writer(tmp_path)
    _finalize(writer, executions=[_execution("run-0001", "req-a", "a")])
    _make_legacy(_run_dir(tmp_path))
    run_dir = _run_dir(tmp_path)
    # A real artifact from before the stream, not a corrupted one: it must still verify.
    assert PerceptionRunReader(run_dir).verify_integrity() == []

    accounting = account_semantic_attempts(PerceptionRunReader(run_dir))

    assert accounting.attempted == 1
    assert accounting.parsed == 1
    assert accounting.parse_failed == 0
    # The count is a floor, not a measurement: such a run never tracked its failures, so the
    # rate must not read as a perfect 0.0.
    assert accounting.measurement_status is SemanticMeasurementStatus.LEGACY_LOWER_BOUND
    assert accounting.parse_failure_rate is None


def _make_legacy(run_dir: Path) -> None:
    """Turn a freshly written run into a valid artifact from before the failures stream.

    Deleting the file alone would leave manifest.json citing it, which is a corrupt artifact
    that verify_integrity() rejects -- not the contract an older run actually had.
    """
    failures = "outputs/semantic-interpretation-failures.jsonl"
    (run_dir / failures).unlink()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_inventory"] = [
        entry for entry in manifest["file_inventory"] if entry["path"] != failures
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_a_complete_run_declares_its_measurement_complete(tmp_path: Path) -> None:
    """A run that tracked failures can assert a rate; one that never tracked them cannot."""
    writer = _writer(tmp_path)
    _finalize(writer, executions=[_execution("run-0001", "req-a", "a")])

    accounting = account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path)))

    assert accounting.measurement_status is SemanticMeasurementStatus.COMPLETE
    assert accounting.parse_failure_rate == 0.0


def test_one_legacy_run_makes_the_whole_aggregate_inexact(tmp_path: Path) -> None:
    """A campaign cannot claim an exact rate when part of its evidence never tracked failures."""
    _finalize(_writer(tmp_path, 1), executions=[_execution("run-0001", "req-a", "a")])
    _finalize(
        _writer(tmp_path, 2),
        executions=[_execution("run-0002", "req-a", "a")],
        failures=[_failure("run-0002", "req-b", "b")],
    )
    _make_legacy(_run_dir(tmp_path, 1))

    total = aggregate_semantic_attempts(
        [
            account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path, 1))),
            account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path, 2))),
        ]
    )

    assert total.attempted == 3
    assert total.parse_failed == 1
    assert total.measurement_status is SemanticMeasurementStatus.LEGACY_LOWER_BOUND
    assert total.parse_failure_rate is None


def test_a_run_with_no_semantic_attempt_has_an_undefined_failure_rate(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _finalize(writer)

    accounting = account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path)))

    assert accounting == SemanticAttemptAccounting(attempted=0, parsed=0, parse_failed=0)
    assert accounting.parse_failure_rate is None, "0/0 is undefined, not zero"


def test_the_same_request_id_in_both_streams_is_refused() -> None:
    """attempted = |parsed_ids union failed_ids| only means something if they are disjoint."""
    with pytest.raises(ValueError, match="both streams"):
        SemanticAttemptAccounting.from_request_ids(
            parsed_ids=frozenset({"req-a", "req-b"}),
            failed_ids=frozenset({"req-b"}),
        )


def test_aggregating_runs_reconciles_each_one_before_summing(tmp_path: Path) -> None:
    """Two runs may legitimately reuse a request_id; that must not read as a collision."""
    _finalize(_writer(tmp_path, 1), executions=[_execution("run-0001", "req-a", "a")])
    _finalize(
        _writer(tmp_path, 2),
        executions=[_execution("run-0002", "req-a", "a")],
        failures=[_failure("run-0002", "req-b", "b")],
    )

    total = aggregate_semantic_attempts(
        [
            account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path, 1))),
            account_semantic_attempts(PerceptionRunReader(_run_dir(tmp_path, 2))),
        ]
    )

    assert total == SemanticAttemptAccounting(attempted=3, parsed=2, parse_failed=1)
    assert total.parse_failure_rate == pytest.approx(1 / 3)
