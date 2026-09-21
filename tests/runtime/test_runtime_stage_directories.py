"""Tests for the workspace layout: ``<workspace>/<dataset>/<run-id>/<stage>/``.

The run directory holds the journal at its root and one directory per stage. The runtime only
decides *where* a stage writes; the executor's writer creates and finalizes that directory, so
an executor never computes a path of its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_worlds import World

from contextmap.runtime import (
    ArtifactRef,
    EffectiveConfig,
    ExecutionPlan,
    ReusePolicy,
    RunJournal,
    StageRequest,
    resolve_plan,
    run_plan,
)

DATASET = "corridor-02"
STAGES = [
    "ingestion",
    "visual_perception",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "semantic_fusion",
]
PROVIDED = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)


def _document(sequence: str | None = DATASET) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    document["inputs"] = {"sequence": sequence}
    return document


def _scope(tmp_path: Path, sequence: str | None = DATASET) -> tuple[EffectiveConfig, ExecutionPlan]:
    effective = effective_from(tmp_path, _document(sequence))
    return effective, resolve_plan(effective).scope(targets=["semantic_fusion"])


class _Recording:
    """An executor that records the directory it was given and writes nothing itself."""

    def __init__(self, world: World, stage_id: str, contract: str) -> None:
        self._inner = world.executor(stage_id, contract)
        self.directories: dict[str, Path | None] = {}

    def execute(self, request: StageRequest) -> ArtifactRef:
        self.directories[request.stage_id] = request.output_dir
        ref: ArtifactRef = self._inner.execute(request)
        return ref


def _executors(execution: ExecutionPlan, world: World) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded: dict[str, Path | None] = {}
    executors: dict[str, Any] = {}
    for stage in execution.plan.stages:
        recording = _Recording(world, stage.stage_id, stage.output or "")
        recording.directories = recorded
        executors[stage.stage_id] = recording
    return executors, recorded


def _run(execution: ExecutionPlan, executors: dict[str, Any], **options: Any) -> None:
    run_plan(
        execution,
        executors,
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        **options,
    )


def test_a_run_lives_under_the_dataset_of_its_configuration(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path)

    first = RunJournal.create(tmp_path / "ws", effective, execution)
    second = RunJournal.create(tmp_path / "ws", effective, execution)

    assert first.directory == tmp_path / "ws" / DATASET / "run-0001"
    assert second.directory == tmp_path / "ws" / DATASET / "run-0002"
    assert (first.directory / "status.json").is_file()  # o diário fica na raiz do run
    assert (first.directory / "effective_config.json").is_file()


def test_runs_of_different_datasets_do_not_share_a_directory_or_a_counter(
    tmp_path: Path,
) -> None:
    (tmp_path / "other").mkdir()
    effective, execution = _scope(tmp_path)
    other, other_execution = _scope(tmp_path / "other", "corridor-03")

    one = RunJournal.create(tmp_path / "ws", effective, execution)
    two = RunJournal.create(tmp_path / "ws", other, other_execution)

    assert one.directory == tmp_path / "ws" / DATASET / "run-0001"
    assert two.directory == tmp_path / "ws" / "corridor-03" / "run-0001"


def test_a_run_without_a_dataset_is_refused_before_anything_is_created(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path, None)

    with pytest.raises(ValueError, match=r"inputs\.sequence"):
        RunJournal.create(tmp_path / "ws", effective, execution)

    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize("dataset", ["../escape", "a/b", "..", "."])
def test_a_dataset_that_is_not_one_path_component_is_refused(tmp_path: Path, dataset: str) -> None:
    effective, execution = _scope(tmp_path, dataset)

    with pytest.raises(ValueError, match="dataset"):
        RunJournal.create(tmp_path / "ws", effective, execution)

    assert not (tmp_path / "ws").exists()


def test_each_stage_receives_its_own_directory_under_the_run_and_none_is_precreated(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    journal = RunJournal.create(tmp_path / "ws", effective, execution)
    executors, recorded = _executors(execution, World())

    _run(execution, executors, journal=journal)

    assert recorded == {stage: journal.directory / stage for stage in STAGES}
    # Quem cria e finaliza a pasta da etapa é o writer do executor, de forma atômica.
    assert not any((journal.directory / stage).exists() for stage in STAGES)


def test_a_run_without_a_journal_has_no_output_directory(tmp_path: Path) -> None:
    _, execution = _scope(tmp_path)
    executors, recorded = _executors(execution, World())

    _run(execution, executors)

    assert recorded == dict.fromkeys(STAGES)


def test_a_reused_stage_is_referenced_and_gets_no_request_or_directory(tmp_path: Path) -> None:
    world = World()
    effective, execution = _scope(tmp_path)
    policy = ReusePolicy(store=world.store(tmp_path / "index"), code_identity="code-1")
    first_journal = RunJournal.create(tmp_path / "ws", effective, execution)
    first_executors, _ = _executors(execution, world)
    _run(execution, first_executors, journal=first_journal, reuse=policy)

    second_journal = RunJournal.create(tmp_path / "ws", effective, execution)
    second_executors, recorded = _executors(execution, world)
    _run(execution, second_executors, journal=second_journal, reuse=policy)

    assert recorded == {}  # tudo reutilizado: nenhum executor foi chamado
    assert not any((second_journal.directory / stage).exists() for stage in STAGES)


def test_the_cli_refuses_a_real_run_without_a_dataset_and_creates_nothing(tmp_path: Path) -> None:
    from runtime_worlds import run_cli, world_executors

    config = tmp_path / "experiment.json"
    config.write_text(json.dumps(_document(None)), encoding="utf-8")

    code, _, err = run_cli(
        "run",
        "-c",
        str(config),
        "--stage",
        "ingestion",
        "--workspace",
        str(tmp_path / "ws"),
        executors=world_executors(World()),
    )

    assert code == 2
    assert "inputs.sequence" in err
    assert not (tmp_path / "ws").exists()
