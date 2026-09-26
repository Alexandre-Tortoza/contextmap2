"""Grounding executions persisted with, and linked to, the perception run artifact (#566)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    ArtifactReference,
    BackendProvenance,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingRejectionReason,
    GroundingTask,
    PerceptionResult,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RejectedGroundingOutput,
    RunArtifactError,
    perception_result_id_for,
    with_grounded_regions,
)

RUN_ID = PerceptionRunId("run-0001")
SOURCE_ID = SourceObservationId("frame-0124")
RESULT_ID = perception_result_id_for(run_id=RUN_ID, source_observation_id=SOURCE_ID)
FINGERPRINT = "sha256:grounding-config"
RAW = "<ref>chair</ref><box><100><200><300><400></box><box><500><500></box><box><9></box>"


def _request(text: str = "the red chair") -> RegionGroundingRequest:
    return RegionGroundingRequest(
        perception_result_id=RESULT_ID,
        image=PreparedImage(
            source_observation_id=SOURCE_ID,
            payload_reference="frame-0124.png",
            payload_artifact=ArtifactReference(
                uri="frame-0124.png", sha256="c" * 64, media_type="image/png"
            ),
            width=640,
            height=480,
        ),
        query=GroundingQuery(
            task=GroundingTask.PHRASE_GROUNDING,
            policy_id="fake.phrase-grounding/1",
            geometry=GroundingGeometry.BOX,
            text=text,
        ),
        configuration_fingerprint=FINGERPRINT,
    )


def _execution(text: str = "the red chair", *, raw: str = RAW) -> RegionGroundingExecution:
    return RegionGroundingExecution(
        request=_request(text),
        provenance=BackendProvenance(
            backend_id="fake_region_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake-grounder",
            version="1",
            configuration_fingerprint=FINGERPRINT,
        ),
        rendered_prompt=f"Locate all the instances that match the following description: {text}.",
        raw_response=raw,
        outputs=(
            GroundingOutput(
                output_index=0,
                native_text="<box><100><200><300><400></box>",
                label="chair",
                box=BoundingBox2D(x=64.0, y=96.0, width=128.0, height=96.0),
                native_diagnostics=(("decode_mode", "mtp"),),
            ),
            GroundingOutput(
                output_index=1,
                native_text="<box><500><500></box>",
                label="chair",
                point=GroundingPoint(x=320.0, y=240.0),
            ),
        ),
        rejected_outputs=(
            RejectedGroundingOutput(
                output_index=2,
                native_text="<box><9></box>",
                reason=GroundingRejectionReason.MALFORMED_GEOMETRY,
                detail="a box needs four coordinates and a point two",
                label="chair",
            ),
        ),
        diagnostics=GroundingDiagnostics(
            latency_ms=41.0, native=(("stats.switch_to_ar", 1), ("stats.tps", 88.25))
        ),
        effective_configuration=MappingProxyType({"generation_mode": "hybrid"}),
        runtime_identity=MappingProxyType({"transformers": "4.57.1"}),
    )


def _result(*executions: RegionGroundingExecution) -> PerceptionResult:
    result = PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=SOURCE_ID,
        run_id=RUN_ID,
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
    )
    return with_grounded_regions(result, executions)


def _writer(tmp_path: Path) -> PerceptionRunWriter:
    return PerceptionRunWriter(
        output_dir=tmp_path / "run-0001",
        sequence_name="corridor-02",
        run_id=RUN_ID,
        run_index=1,
        sequence_artifact_id="sequence-0001",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_discovery", "region_grounding"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    )


def test_raw_response_and_parsed_evidence_survive_the_run_artifact(tmp_path: Path) -> None:
    execution = _execution()
    writer = _writer(tmp_path)
    writer.add_result(_result(execution))
    writer.add_region_grounding(execution)
    manifest = writer.finalize()

    reader = PerceptionRunReader(tmp_path / "run-0001")
    (reopened,) = reader.list_region_groundings()

    assert reopened == execution
    assert reopened.request.query.text == "the red chair"
    assert reader.result(SOURCE_ID).regions == execution.regions
    assert reader.verify_integrity() == []
    paths = {entry.path for entry in manifest.file_inventory}
    raw_path = f"outputs/region-grounding-raw/{execution.request_id}.txt"
    assert {"outputs/region-grounding.jsonl", raw_path} <= paths
    assert (tmp_path / "run-0001" / raw_path).read_bytes() == RAW.encode("utf-8")
    record = json.loads((tmp_path / "run-0001" / "outputs/region-grounding.jsonl").read_text())
    assert "raw_response" not in record


def test_a_run_without_grounding_has_no_grounding_stream(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_result(_result())
    manifest = writer.finalize()

    reader = PerceptionRunReader(tmp_path / "run-0001")

    assert reader.list_region_groundings() == []
    assert all("region-grounding" not in entry.path for entry in manifest.file_inventory)


def test_finalize_rejects_grounding_without_its_owning_result(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_region_grounding(_execution())

    with pytest.raises(RunArtifactError, match="result"):
        writer.finalize()
    assert not (tmp_path / "run-0001").exists()


def test_finalize_rejects_grounded_regions_missing_from_the_result(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_result(_result())
    writer.add_region_grounding(_execution())

    with pytest.raises(RunArtifactError, match="materialized"):
        writer.finalize()


def test_writer_rejects_the_same_grounding_request_twice(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_region_grounding(_execution())

    with pytest.raises(RunArtifactError, match="duplicate grounding request"):
        writer.add_region_grounding(_execution())


def test_distinct_queries_on_one_observation_are_separate_records(tmp_path: Path) -> None:
    first, second = _execution("the red chair"), _execution("the table")
    writer = _writer(tmp_path)
    writer.add_result(_result(first, second))
    writer.add_region_grounding(first)
    writer.add_region_grounding(second)
    writer.finalize()

    reopened = PerceptionRunReader(tmp_path / "run-0001").list_region_groundings()

    assert [item.request.query.text for item in reopened] == ["the red chair", "the table"]


def test_reader_refuses_a_raw_response_changed_after_publication(tmp_path: Path) -> None:
    execution = _execution()
    writer = _writer(tmp_path)
    writer.add_result(_result(execution))
    writer.add_region_grounding(execution)
    writer.finalize()
    raw_path = tmp_path / "run-0001" / f"outputs/region-grounding-raw/{execution.request_id}.txt"
    raw_path.write_text("<box>none</box>", encoding="utf-8")

    with pytest.raises(RunArtifactError, match="grounding"):
        PerceptionRunReader(tmp_path / "run-0001").list_region_groundings()


def test_writer_refuses_grounding_after_finalize(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_result(_result())
    writer.finalize()

    with pytest.raises(RunArtifactError, match="finalize"):
        writer.add_region_grounding(replace(_execution()))
