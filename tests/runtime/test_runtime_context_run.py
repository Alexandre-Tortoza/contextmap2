"""ContextRun: one execution that adds context evidence over a spatial foundation (issue #495)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from runtime_context import FRAMES, PROVIDED_RUNTIMES, ContextWorkspace, Located, foundation, ref

from contextmap.runtime import (
    ContextRunError,
    ExecutionRecord,
    RunJournal,
    StageRecord,
    context_scope,
    publish_context_run,
    read_context_run,
    run_plan,
)


class TestTheContextScope:
    def test_only_the_context_stages_run_over_the_providedfoundation(self, tmp_path: Path) -> None:
        _, plan = ContextWorkspace(tmp_path).plan()

        execution = context_scope(plan, foundation())

        assert [stage.stage_id for stage in execution.stages] == [
            "visual_perception",
            "sensor_association",
        ]
        assert execution.reused == {
            "ingestion": (foundation().sequence,),
            "state_estimation": (foundation().state_estimation,),
            "geometric_mapping": (foundation().geometry,),
        }


class TestTheRecord:
    def test_it_names_each_context_artifact_and_whether_it_was_produced(
        self, tmp_path: Path
    ) -> None:
        context = ContextWorkspace(tmp_path).run()

        assert context.identity.startswith("sha256:")
        assert context.foundation == foundation()
        assert context.observation_selection is None
        assert [(item.ref.stage_id, item.disposition) for item in context.artifacts] == [
            ("sensor_association", "produced"),
            ("visual_perception", "produced"),
        ]
        assert context.run == "S1/run-0001"

    def test_the_same_evidence_is_the_same_context_run_whatever_run_holds_it(
        self, tmp_path: Path
    ) -> None:
        workspace = ContextWorkspace(tmp_path)
        first = workspace.run()

        second = workspace.run()

        assert {item.disposition for item in second.artifacts} == {"reused"}
        assert second.run == "S1/run-0002"
        assert second.identity == first.identity

    def test_another_selection_is_another_context_run(self, tmp_path: Path) -> None:
        workspace = ContextWorkspace(tmp_path)

        whole = workspace.run()
        framed = workspace.run(selection=FRAMES)

        assert framed.observation_selection == FRAMES
        assert framed.identity != whole.identity

    def test_another_foundation_is_another_context_run(self, tmp_path: Path) -> None:
        workspace = ContextWorkspace(tmp_path)

        first = workspace.run()
        other = workspace.run(over=foundation(geometry="map-B"))

        assert other.identity != first.identity

    def test_it_is_published_in_its_run_and_reads_back_identically(self, tmp_path: Path) -> None:
        context = ContextWorkspace(tmp_path).run(selection=FRAMES)
        directory = tmp_path / "ws" / "S1" / "run-0001"

        assert (directory / "context_run.json").is_file()
        assert read_context_run(directory, workspace=tmp_path / "ws") == context

    def test_moving_the_workspace_keeps_it_readable_and_identical(self, tmp_path: Path) -> None:
        context = ContextWorkspace(tmp_path).run()
        shutil.copytree(tmp_path / "ws", tmp_path / "moved")

        moved = read_context_run(
            tmp_path / "moved" / "S1" / "run-0001", workspace=tmp_path / "moved"
        )

        assert moved == context

    def test_a_run_has_one_record_only(self, tmp_path: Path) -> None:
        workspace = ContextWorkspace(tmp_path)
        workspace.run()
        last = workspace.last

        with pytest.raises(FileExistsError):
            publish_context_run(
                last["run_directory"],
                workspace=last["workspace"],
                foundation=last["foundation"],
                execution=last["execution"],
                record=last["record"],
            )


class TestAnInconsistentRecord:
    def test_an_altered_record_is_refused(self, tmp_path: Path) -> None:
        ContextWorkspace(tmp_path).run()
        path = tmp_path / "ws" / "S1" / "run-0001" / "context_run.json"
        document = json.loads(path.read_text("utf-8"))
        document["artifacts"][0]["content_hash"] = "sha256:other"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ContextRunError, match="identity"):
            read_context_run(path.parent, workspace=tmp_path / "ws")

    def test_a_record_copied_into_another_run_is_refused(self, tmp_path: Path) -> None:
        ContextWorkspace(tmp_path).run()
        source = tmp_path / "ws" / "S1" / "run-0001" / "context_run.json"
        other = tmp_path / "ws" / "S1" / "run-0009"
        other.mkdir()
        shutil.copy(source, other / "context_run.json")

        with pytest.raises(ContextRunError, match="run-0001"):
            read_context_run(other, workspace=tmp_path / "ws")

    def test_evidence_built_on_another_map_is_refused(self, tmp_path: Path) -> None:
        workspace = ContextWorkspace(tmp_path)
        effective, plan = workspace.plan()
        execution = context_scope(plan, foundation())
        journal = RunJournal.create(tmp_path / "ws", effective, execution)
        record = run_plan(
            execution,
            {
                stage.stage_id: Located(
                    workspace.world.executor(stage.stage_id, stage.output or "")
                )
                for stage in plan.stages
            },
            environ={},
            module_available=lambda _name: True,
            provided_runtimes=PROVIDED_RUNTIMES,
            journal=journal,
        )

        with pytest.raises(ContextRunError, match="map-B"):
            publish_context_run(
                journal.directory,
                workspace=tmp_path / "ws",
                foundation=foundation(geometry="map-B"),
                execution=execution,
                record=record,
            )
        assert not (journal.directory / "context_run.json").exists()

    def test_a_stage_outside_the_context_is_refused(self, tmp_path: Path) -> None:
        workspace = ContextWorkspace(tmp_path)
        _, plan = workspace.plan()
        execution = context_scope(plan, foundation())
        fusion = ref("semantic_fusion", "SemanticFusionRunArtifact", "fusion-1")
        record = ExecutionRecord(
            plan_digest=plan.digest,
            order=("semantic_fusion",),
            stages=(StageRecord(stage_id="semantic_fusion", inputs={}, output=fusion),),
            reused={},
        )
        directory = tmp_path / "ws" / "S1" / "run-0001"
        directory.mkdir(parents=True)

        with pytest.raises(ContextRunError, match="semantic_fusion"):
            publish_context_run(
                directory,
                workspace=tmp_path / "ws",
                foundation=foundation(),
                execution=execution,
                record=record,
            )
