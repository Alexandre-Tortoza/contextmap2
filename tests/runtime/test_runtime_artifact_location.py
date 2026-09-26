"""Tests for the artifact locator: a reused artifact is referenced, and its directory is openable.

An ``ArtifactRef`` carries the directory of the artifact relative to the workspace. A stage that is
reused from an earlier run keeps that location: the runtime never copies the artifact, and a
downstream executor opens it through the request it receives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.runtime import (
    ArtifactRef,
    ExecutionPlan,
    FileArtifactStore,
    ReusePolicy,
    RunJournal,
    StageExecutionError,
    StageRequest,
    resolve_plan,
    run_plan,
)

DATASET = "corridor-02"
STAGES = ["ingestion", "visual_perception", "state_estimation"]
PROVIDED = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)


def _scope(tmp_path: Path) -> tuple[Any, ExecutionPlan]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    document["inputs"] = {"sequence": DATASET}
    effective = effective_from(tmp_path, document)
    return effective, resolve_plan(effective).scope(targets=["state_estimation"])


SOURCE = {"ingestion": {"source": "recording-A"}}
"""The source the fake ingestion reads, declared for reuse: its executor does not name it."""


class _Writer:
    """An executor that writes a marker file into the directory it is given, like a real writer."""

    def __init__(self, stage_id: str, contract: str, log: list[str], seen: dict[str, Any]) -> None:
        self.stage_id, self.contract, self.log, self.seen = stage_id, contract, log, seen

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        self.log.append(self.stage_id)
        upstream = {
            name: request.directory_of(ref) for name, refs in request.inputs.items() for ref in refs
        }
        self.seen[self.stage_id] = upstream
        request.output_dir.mkdir(parents=True)
        (request.output_dir / "manifest.json").write_text(self.stage_id, encoding="utf-8")
        return ArtifactRef(
            stage_id=self.stage_id,
            contract=self.contract,
            artifact_id=f"{self.stage_id}--{request.output_dir.parent.name}",
            content_hash=f"sha256:{self.stage_id}",
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


def _executors(execution: ExecutionPlan, log: list[str], seen: dict[str, Any]) -> dict[str, Any]:
    return {
        stage.stage_id: _Writer(stage.stage_id, stage.output or "", log, seen)
        for stage in execution.plan.stages
    }


def _run(execution: ExecutionPlan, executors: dict[str, Any], **options: Any) -> Any:
    return run_plan(
        execution,
        executors,
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        **options,
    )


def test_a_reference_round_trips_with_its_location_and_omits_it_when_there_is_none() -> None:
    located = ArtifactRef(
        stage_id="geometric_mapping",
        contract="GeometricMapArtifact",
        artifact_id="map",
        content_hash="sha256:a",
        location="corridor-02/run-0001/geometric_mapping",
    )
    bare = ArtifactRef(stage_id="s", contract="c", artifact_id="a")

    assert ArtifactRef.from_document(located.to_document()) == located
    assert "location" not in bare.to_document()
    assert ArtifactRef.from_document(bare.to_document()) == bare


@pytest.mark.parametrize("location", ["/etc/passwd", "../outside", "a/../../b", ""])
def test_a_location_that_leaves_the_workspace_is_refused(tmp_path: Path, location: str) -> None:
    ref = ArtifactRef(stage_id="s", contract="c", artifact_id="a", location=location)
    request = StageRequest(
        stage_id="t", inputs={}, components={}, config_digest="d", workspace=tmp_path
    )

    with pytest.raises(ValueError, match="location"):
        request.directory_of(ref)


def test_a_reference_without_a_location_cannot_be_opened(tmp_path: Path) -> None:
    ref = ArtifactRef(stage_id="s", contract="c", artifact_id="a")
    request = StageRequest(
        stage_id="t", inputs={}, components={}, config_digest="d", workspace=tmp_path
    )

    with pytest.raises(ValueError, match="no location"):
        request.directory_of(ref)


def test_a_request_without_a_workspace_cannot_open_an_artifact() -> None:
    ref = ArtifactRef(stage_id="s", contract="c", artifact_id="a", location="x/y")
    request = StageRequest(stage_id="t", inputs={}, components={}, config_digest="d")

    with pytest.raises(ValueError, match="workspace"):
        request.directory_of(ref)


def test_the_request_workspace_is_the_root_of_the_journal_and_none_without_one(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    journal = RunJournal.create(tmp_path / "ws", effective, execution)
    seen: dict[str, Any] = {}

    _run(execution, _executors(execution, [], seen), journal=journal)

    # Cada executor abriu a pasta do artifact anterior dentro do workspace.
    assert seen["state_estimation"]["sequence"] == journal.directory / "ingestion"
    assert seen["state_estimation"]["sequence"].is_dir()


def test_an_artifact_reused_from_an_earlier_run_is_opened_where_it_was_written(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    store = FileArtifactStore(tmp_path / "index", verify=lambda _ref: True)
    policy = ReusePolicy(store=store, code_identity="code-1", identities=SOURCE)
    first = RunJournal.create(tmp_path / "ws", effective, execution)
    _run(execution, _executors(execution, [], {}), journal=first, reuse=policy)

    log: list[str] = []
    seen: dict[str, Any] = {}
    second = RunJournal.create(tmp_path / "ws", effective, execution)
    record = _run(execution, _executors(execution, log, seen), journal=second, reuse=policy)

    # Tudo foi reutilizado: nenhum executor rodou e a segunda run não tem pastas de estágio.
    assert log == []
    assert not any((second.directory / stage).exists() for stage in STAGES)
    reused = {stage.stage_id: stage.output for stage in record.stages}
    assert reused["state_estimation"].location == (
        f"{DATASET}/{first.directory.name}/state_estimation"
    )
    assert (tmp_path / "ws" / str(reused["state_estimation"].location)).is_dir()


def test_a_downstream_stage_opens_the_reused_upstream_directory_of_the_earlier_run(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    store = FileArtifactStore(tmp_path / "index", verify=lambda _ref: True)
    policy = ReusePolicy(store=store, code_identity="code-1", identities=SOURCE)
    first = RunJournal.create(tmp_path / "ws", effective, execution)
    _run(execution, _executors(execution, [], {}), journal=first, reuse=policy)

    seen: dict[str, Any] = {}
    second = RunJournal.create(tmp_path / "ws", effective, execution)
    recompute = ReusePolicy(
        store=store,
        code_identity="code-1",
        force_recompute=frozenset({"state_estimation"}),
        identities=SOURCE,
    )
    _run(execution, _executors(execution, [], seen), journal=second, reuse=recompute)

    # A entrada reutilizada aponta para a pasta da run anterior: referenciada, nunca copiada.
    assert seen["state_estimation"]["sequence"] == first.directory / "ingestion"
    assert not (second.directory / "ingestion").exists()


def test_an_executor_that_reports_another_location_than_the_one_it_was_given_is_refused(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    journal = RunJournal.create(tmp_path / "ws", effective, execution)

    class Elsewhere:
        def execute(self, request: StageRequest) -> ArtifactRef:
            return ArtifactRef(
                stage_id=request.stage_id,
                contract="SequenceArtifact",
                artifact_id="x",
                content_hash="sha256:x",
                location="somewhere/else",
            )

    only_ingestion = resolve_plan(effective).scope(targets=["ingestion"])
    with pytest.raises(StageExecutionError, match="location"):
        _run(only_ingestion, {"ingestion": Elsewhere()}, journal=journal)
