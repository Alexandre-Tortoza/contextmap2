"""ContextRun: one execution that adds context evidence over a spatial foundation (issue #495)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_worlds import World

from contextmap.runtime import (
    ArtifactRef,
    ContextRun,
    ContextRunError,
    ExecutionRecord,
    PipelinePlan,
    ReusePolicy,
    RunJournal,
    SpatialFoundation,
    SpatialFoundationId,
    StageRecord,
    StageRequest,
    context_scope,
    publish_context_run,
    read_context_run,
    resolve_plan,
    run_plan,
)

_FRAMES = {"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 10}
_PROVIDED = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)


def _ref(stage_id: str, contract: str, name: str) -> ArtifactRef:
    return ArtifactRef(
        stage_id=stage_id,
        contract=contract,
        artifact_id=name,
        content_hash=f"sha256:{name}",
        location=f"S1/run-0001/{stage_id}",
    )


def _foundation(geometry: str = "map-A") -> SpatialFoundation:
    return SpatialFoundation(
        identity=SpatialFoundationId(f"sha256:foundation-{geometry}"),
        sequence=_ref("ingestion", "SequenceArtifact", "seq-A"),
        state_estimation=_ref("state_estimation", "StateEstimationRunArtifact", "se-A"),
        geometry=_ref("geometric_mapping", "GeometricMapArtifact", geometry),
    )


def _document(selection: dict[str, Any] | None = None) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    if selection is not None:
        document["inputs"]["observation_selection"] = selection
    return document


class _Located:
    """A World stage that also reports where it wrote, as a real executor does."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def execute(self, request: StageRequest) -> ArtifactRef:
        produced: ArtifactRef = self._inner.execute(request)
        assert request.output_dir is not None and request.workspace is not None
        location = request.output_dir.relative_to(request.workspace).as_posix()
        return dataclasses.replace(produced, location=location)


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.last: dict[str, Any] = {}
        self.root = root
        self.world = World()
        self.index = self.world.store(root / "index")

    def plan(self, selection: dict[str, Any] | None = None) -> tuple[Any, PipelinePlan]:
        effective = effective_from(self.root, _document(selection))
        return effective, resolve_plan(effective)

    def run(
        self,
        foundation: SpatialFoundation | None = None,
        selection: dict[str, Any] | None = None,
    ) -> ContextRun:
        foundation = foundation or _foundation()
        effective, plan = self.plan(selection)
        execution = context_scope(plan, foundation)
        journal = RunJournal.create(self.root / "ws", effective, execution)
        record = run_plan(
            execution,
            {
                stage.stage_id: _Located(self.world.executor(stage.stage_id, stage.output or ""))
                for stage in plan.stages
            },
            environ={},
            module_available=lambda _name: True,
            provided_runtimes=_PROVIDED,
            reuse=ReusePolicy(store=self.index, code_identity="code-1"),
            journal=journal,
        )
        self.last = {
            "run_directory": journal.directory,
            "workspace": self.root / "ws",
            "foundation": foundation,
            "execution": execution,
            "record": record,
        }
        return publish_context_run(
            journal.directory,
            workspace=self.root / "ws",
            foundation=foundation,
            execution=execution,
            record=record,
        )


class TestTheContextScope:
    def test_only_the_context_stages_run_over_the_provided_foundation(self, tmp_path: Path) -> None:
        _, plan = _Workspace(tmp_path).plan()

        execution = context_scope(plan, _foundation())

        assert [stage.stage_id for stage in execution.stages] == [
            "visual_perception",
            "sensor_association",
        ]
        assert execution.reused == {
            "ingestion": (_foundation().sequence,),
            "state_estimation": (_foundation().state_estimation,),
            "geometric_mapping": (_foundation().geometry,),
        }


class TestTheRecord:
    def test_it_names_each_context_artifact_and_whether_it_was_produced(
        self, tmp_path: Path
    ) -> None:
        context = _Workspace(tmp_path).run()

        assert context.identity.startswith("sha256:")
        assert context.foundation == _foundation()
        assert context.observation_selection is None
        assert [(item.ref.stage_id, item.disposition) for item in context.artifacts] == [
            ("sensor_association", "produced"),
            ("visual_perception", "produced"),
        ]
        assert context.run == "S1/run-0001"

    def test_the_same_evidence_is_the_same_context_run_whatever_run_holds_it(
        self, tmp_path: Path
    ) -> None:
        workspace = _Workspace(tmp_path)
        first = workspace.run()

        second = workspace.run()

        assert {item.disposition for item in second.artifacts} == {"reused"}
        assert second.run == "S1/run-0002"
        assert second.identity == first.identity

    def test_another_selection_is_another_context_run(self, tmp_path: Path) -> None:
        workspace = _Workspace(tmp_path)

        whole = workspace.run()
        framed = workspace.run(selection=_FRAMES)

        assert framed.observation_selection == _FRAMES
        assert framed.identity != whole.identity

    def test_another_foundation_is_another_context_run(self, tmp_path: Path) -> None:
        workspace = _Workspace(tmp_path)

        first = workspace.run()
        other = workspace.run(foundation=_foundation(geometry="map-B"))

        assert other.identity != first.identity

    def test_it_is_published_in_its_run_and_reads_back_identically(self, tmp_path: Path) -> None:
        context = _Workspace(tmp_path).run(selection=_FRAMES)
        directory = tmp_path / "ws" / "S1" / "run-0001"

        assert (directory / "context_run.json").is_file()
        assert read_context_run(directory, workspace=tmp_path / "ws") == context

    def test_moving_the_workspace_keeps_it_readable_and_identical(self, tmp_path: Path) -> None:
        context = _Workspace(tmp_path).run()
        shutil.copytree(tmp_path / "ws", tmp_path / "moved")

        moved = read_context_run(
            tmp_path / "moved" / "S1" / "run-0001", workspace=tmp_path / "moved"
        )

        assert moved == context

    def test_a_run_has_one_record_only(self, tmp_path: Path) -> None:
        workspace = _Workspace(tmp_path)
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
        _Workspace(tmp_path).run()
        path = tmp_path / "ws" / "S1" / "run-0001" / "context_run.json"
        document = json.loads(path.read_text("utf-8"))
        document["artifacts"][0]["content_hash"] = "sha256:other"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ContextRunError, match="identity"):
            read_context_run(path.parent, workspace=tmp_path / "ws")

    def test_a_record_copied_into_another_run_is_refused(self, tmp_path: Path) -> None:
        _Workspace(tmp_path).run()
        source = tmp_path / "ws" / "S1" / "run-0001" / "context_run.json"
        other = tmp_path / "ws" / "S1" / "run-0009"
        other.mkdir()
        shutil.copy(source, other / "context_run.json")

        with pytest.raises(ContextRunError, match="run-0001"):
            read_context_run(other, workspace=tmp_path / "ws")

    def test_evidence_built_on_another_map_is_refused(self, tmp_path: Path) -> None:
        workspace = _Workspace(tmp_path)
        effective, plan = workspace.plan()
        execution = context_scope(plan, _foundation())
        journal = RunJournal.create(tmp_path / "ws", effective, execution)
        record = run_plan(
            execution,
            {
                stage.stage_id: _Located(
                    workspace.world.executor(stage.stage_id, stage.output or "")
                )
                for stage in plan.stages
            },
            environ={},
            module_available=lambda _name: True,
            provided_runtimes=_PROVIDED,
            journal=journal,
        )

        with pytest.raises(ContextRunError, match="map-B"):
            publish_context_run(
                journal.directory,
                workspace=tmp_path / "ws",
                foundation=_foundation(geometry="map-B"),
                execution=execution,
                record=record,
            )
        assert not (journal.directory / "context_run.json").exists()

    def test_a_stage_outside_the_context_is_refused(self, tmp_path: Path) -> None:
        workspace = _Workspace(tmp_path)
        _, plan = workspace.plan()
        execution = context_scope(plan, _foundation())
        fusion = _ref("semantic_fusion", "SemanticFusionRunArtifact", "fusion-1")
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
                foundation=_foundation(),
                execution=execution,
                record=record,
            )
