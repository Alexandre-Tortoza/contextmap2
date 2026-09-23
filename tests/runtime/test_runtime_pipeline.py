"""Tests for the config-derived stage DAG: resolution, preflight, scoping and execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_fixtures import unavailable_future_stage  # noqa: F401

from contextmap.runtime import (
    ArtifactRef,
    EffectiveConfig,
    PipelinePlan,
    PlanDocumentError,
    PreflightError,
    RuntimePreset,
    StageDeclaration,
    StageExecutionError,
    StageRequest,
    preflight,
    read_plan_document,
    resolve_plan,
    run_plan,
    write_plan,
)
from contextmap.runtime.catalog import (
    PRESETS,
    Interception,
    StageInput,
)

CANONICAL_ORDER = [
    "ingestion",
    "visual_perception",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "semantic_fusion",
    "semantic_mapping",
    "entity_resolution",
    "spatial_relations",
    "context_map",
]
IMPLEMENTED = CANONICAL_ORDER


def _ready(_name: str) -> bool:
    return True


def _document(*, point_representation: bool = False) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = point_representation
    return document


class FakeStage:
    """A lightweight stage executor: no model, no GPU, just an artifact per call."""

    def __init__(self, stage: str, contract: str, log: list[str]) -> None:
        self.stage = stage
        self.contract = contract
        self.log = log
        self.requests: list[StageRequest] = []

    def execute(self, request: StageRequest) -> ArtifactRef:
        self.requests.append(request)
        self.log.append(self.stage)
        return ArtifactRef(
            stage_id=self.stage, contract=self.contract, artifact_id=f"{self.stage}#1"
        )


def _executors(plan: PipelinePlan, log: list[str]) -> dict[str, FakeStage]:
    return {
        stage.stage_id: FakeStage(stage.stage_id, stage.output or "", log) for stage in plan.stages
    }


class TestCanonicalDag:
    def test_resolves_one_deterministic_validated_dag(self, tmp_path: Path) -> None:
        first = resolve_plan(effective_from(tmp_path, _document()))
        second = resolve_plan(effective_from(tmp_path, _document()))

        assert first.order == second.order == tuple(CANONICAL_ORDER)
        assert first.digest == second.digest
        assert first.problems == ()

    def test_edges_carry_the_producer_and_the_consumed_contract(self, tmp_path: Path) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))

        association = plan.stage("sensor_association")
        wiring = {item.name: (item.source, item.contract) for item in association.inputs}

        assert wiring == {
            "sequence": ("ingestion", "SequenceArtifact"),
            "perception": ("visual_perception", "PerceptionRunArtifact"),
            "trajectory": ("state_estimation", "StateEstimationRunArtifact"),
            "geometry": ("geometric_mapping", "GeometricMapArtifact"),
        }
        assert association.output == "SensorAssociationRunArtifact"

    def test_the_optional_branch_joins_only_when_selected(self, tmp_path: Path) -> None:
        without = resolve_plan(effective_from(tmp_path, _document()))
        with_stage = resolve_plan(effective_from(tmp_path, _document(point_representation=True)))

        assert without.order is not None and with_stage.order is not None
        assert "point_representation" not in without.order
        assert {item.name for item in without.stage("semantic_fusion").inputs} == {
            "sequence",
            "association",
            "perception",
            "geometry",
        }
        assert "point_representation" in with_stage.order
        assert with_stage.order.index("sensor_association") < with_stage.order.index(
            "point_representation"
        )
        assert with_stage.order.index("point_representation") < with_stage.order.index(
            "semantic_fusion"
        )
        assert {item.name: item.source for item in with_stage.stage("semantic_fusion").inputs}[
            "representation"
        ] == "point_representation"

    def test_a_stage_digest_follows_only_its_own_configuration(self, tmp_path: Path) -> None:
        base = resolve_plan(effective_from(tmp_path, _document()))
        document = _document()
        document["components"]["visual_perception"]["region_discovery"]["sam3"]["prompt"] = "table"

        changed = resolve_plan(effective_from(tmp_path, document))

        assert (
            changed.stage("visual_perception").config_digest
            != base.stage("visual_perception").config_digest
        )
        assert (
            changed.stage("state_estimation").config_digest
            == base.stage("state_estimation").config_digest
        )
        assert changed.digest != base.digest

    @pytest.mark.usefixtures("unavailable_future_stage")
    def test_unavailable_stages_stay_in_the_topology_with_their_reason(
        self, tmp_path: Path
    ) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))

        stage = plan.stage("scene_graph")

        assert not stage.available
        assert "milestone" in stage.unavailable_reason
        assert [item.source for item in stage.inputs] == ["geometric_mapping"]


class TestPersistedTopology:
    def test_the_plan_document_records_the_topology_identities_and_order(
        self, tmp_path: Path
    ) -> None:
        effective = effective_from(tmp_path, _document())
        plan = resolve_plan(effective)

        path = write_plan(plan, tmp_path / "run")
        document = read_plan_document(path)

        assert path.name == "plan.json"
        assert document["preset"] == "canonical/1"
        assert document["config_digest"] == effective.digest
        assert [stage["stage_id"] for stage in document["stages"]] == CANONICAL_ORDER
        region = document["stages"][1]["components"]["visual_perception.region_discovery"]
        assert region == "sam3"
        assert (
            document["stages"][1]["config_digest"] == plan.stage("visual_perception").config_digest
        )

    def test_publishing_is_idempotent_and_never_replaces_a_different_plan(
        self, tmp_path: Path
    ) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        first = write_plan(plan, tmp_path / "run")
        assert write_plan(plan, tmp_path / "run") == first

        document = _document()
        document["components"]["state_estimation"]["estimator"]["external_pose"]["body_frame"] = "b"
        other = resolve_plan(effective_from(tmp_path, document))
        with pytest.raises(PlanDocumentError, match="already"):
            write_plan(other, tmp_path / "run")

    def test_a_tampered_plan_is_detected(self, tmp_path: Path) -> None:
        path = write_plan(resolve_plan(effective_from(tmp_path, _document())), tmp_path / "run")
        document = json.loads(path.read_text("utf-8"))
        document["document"]["stages"][0]["capability"] = "other"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(PlanDocumentError, match="digest"):
            read_plan_document(path)


def _register(monkeypatch: pytest.MonkeyPatch, preset: RuntimePreset) -> None:
    monkeypatch.setitem(PRESETS, preset.preset_id, preset)


def _with_enhancement(enhancement_output: str = "DenseMap") -> RuntimePreset:
    """A preset whose optional `enhance` stage sits between `dense` and `associate`."""
    return RuntimePreset(
        preset_id="test/enhance",
        stages=(
            StageDeclaration(stage_id="source", capability="fake", output="Frames"),
            StageDeclaration(
                stage_id="dense",
                capability="fake",
                inputs=(StageInput(name="frames", contract="Frames", source="source"),),
                output="DenseMap",
            ),
            StageDeclaration(
                stage_id="enhance",
                capability="fake",
                optional=True,
                default_enabled=False,
                inputs=(StageInput(name="dense", contract="DenseMap", source="dense"),),
                output=enhancement_output,
                intercepts=Interception(consumer="associate", input_name="dense"),
            ),
            StageDeclaration(
                stage_id="associate",
                capability="fake",
                inputs=(StageInput(name="dense", contract="DenseMap", source="dense"),),
                output="Associated",
            ),
        ),
    )


def _fake_effective(tmp_path: Path, profile: str, **stages: bool) -> EffectiveConfig:
    from contextmap.runtime import resolve_effective_config

    file = tmp_path / "fake.json"
    file.write_text(json.dumps({"pipeline": {"stages": stages}}), encoding="utf-8")
    return resolve_effective_config(profile=profile, files=[file])


class TestOptionalStageInsertion:
    def test_inserting_a_compatible_stage_rewires_the_consumer_without_editing_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(monkeypatch, _with_enhancement())

        native = resolve_plan(_fake_effective(tmp_path, "test/enhance"))
        enhanced = resolve_plan(_fake_effective(tmp_path, "test/enhance", enhance=True))

        assert native.order == ("source", "dense", "associate")
        assert enhanced.order == ("source", "dense", "enhance", "associate")
        assert [(i.name, i.source) for i in native.stage("associate").inputs] == [
            ("dense", "dense")
        ]
        assert [(i.name, i.source) for i in enhanced.stage("associate").inputs] == [
            ("dense", "enhance")
        ]
        assert [(i.name, i.source) for i in enhanced.stage("enhance").inputs] == [
            ("dense", "dense")
        ]
        assert native.problems == () and enhanced.problems == ()

    def test_an_incompatible_stage_fails_preflight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(monkeypatch, _with_enhancement(enhancement_output="Other"))

        plan = resolve_plan(_fake_effective(tmp_path, "test/enhance", enhance=True))

        assert any("Other" in problem.message for problem in plan.problems)
        assert not preflight(plan.scope()).ok

    def test_both_arms_share_the_identical_upstream_stage_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(monkeypatch, _with_enhancement())

        native = resolve_plan(_fake_effective(tmp_path, "test/enhance"))
        enhanced = resolve_plan(_fake_effective(tmp_path, "test/enhance", enhance=True))

        for stage_id in ("source", "dense"):
            assert native.stage(stage_id).config_digest == enhanced.stage(stage_id).config_digest


class TestStructuralPreflight:
    def test_a_cycle_fails_preflight_and_names_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        preset = RuntimePreset(
            preset_id="test/cycle",
            stages=(
                StageDeclaration(
                    stage_id="a",
                    capability="fake",
                    inputs=(StageInput(name="x", contract="B", source="b"),),
                    output="A",
                ),
                StageDeclaration(
                    stage_id="b",
                    capability="fake",
                    inputs=(StageInput(name="x", contract="A", source="a"),),
                    output="B",
                ),
            ),
        )
        _register(monkeypatch, preset)

        plan = resolve_plan(_fake_effective(tmp_path, "test/cycle"))
        report = preflight(plan.scope())

        assert plan.order is None
        assert not report.ok
        assert any("cycle" in problem.message for problem in report.problems)

    def test_a_missing_dependency_fails_preflight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        preset = RuntimePreset(
            preset_id="test/missing",
            stages=(
                StageDeclaration(
                    stage_id="a",
                    capability="fake",
                    inputs=(StageInput(name="x", contract="Z", source="ghost"),),
                    output="A",
                ),
                StageDeclaration(
                    stage_id="off",
                    capability="fake",
                    optional=True,
                    default_enabled=False,
                    output="Off",
                ),
                StageDeclaration(
                    stage_id="b",
                    capability="fake",
                    inputs=(StageInput(name="y", contract="Off", source="off"),),
                    output="B",
                ),
            ),
        )
        _register(monkeypatch, preset)

        report = preflight(resolve_plan(_fake_effective(tmp_path, "test/missing")).scope())

        text = " ".join(problem.message for problem in report.problems)
        assert "ghost" in text
        assert "off" in text

    def test_a_contract_mismatch_between_producer_and_consumer_fails_preflight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        preset = RuntimePreset(
            preset_id="test/contract",
            stages=(
                StageDeclaration(stage_id="a", capability="fake", output="Frames"),
                StageDeclaration(
                    stage_id="b",
                    capability="fake",
                    inputs=(StageInput(name="x", contract="Maps", source="a"),),
                    output="B",
                ),
            ),
        )
        _register(monkeypatch, preset)

        report = preflight(resolve_plan(_fake_effective(tmp_path, "test/contract")).scope())

        assert any(
            "Maps" in problem.message and "Frames" in problem.message for problem in report.problems
        )


class TestScopeAndExecution:
    def test_runs_the_complete_pipeline_in_dependency_order_with_fake_stages(
        self, tmp_path: Path
    ) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        log: list[str] = []
        executors = _executors(plan, log)
        scope = plan.scope(targets=["context_map"])

        record = run_plan(
            scope, executors, environ={}, module_available=_ready, provided_runtimes=_ALL_PROVIDED
        )

        assert log == IMPLEMENTED
        assert record.order == tuple(IMPLEMENTED)
        fusion = executors["semantic_fusion"].requests[0]
        assert {
            name: [ref.artifact_id for ref in refs] for name, refs in fusion.inputs.items()
        } == {
            "sequence": ["ingestion#1"],
            "association": ["sensor_association#1"],
            "perception": ["visual_perception#1"],
            "geometry": ["geometric_mapping#1"],
        }

    def test_an_explicit_subgraph_reuses_the_provided_upstream_artifacts(
        self, tmp_path: Path
    ) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        provided = {
            "ingestion": ArtifactRef(
                stage_id="ingestion", contract="SequenceArtifact", artifact_id="seq-7"
            ),
            "state_estimation": ArtifactRef(
                stage_id="state_estimation",
                contract="StateEstimationRunArtifact",
                artifact_id="traj-3",
            ),
        }
        log: list[str] = []
        executors = _executors(plan, log)

        record = run_plan(
            plan.scope(targets=["geometric_mapping"], provided=provided),
            executors,
            environ={},
            module_available=_ready,
            provided_runtimes=_ALL_PROVIDED,
        )

        assert log == ["geometric_mapping"]
        assert record.reused == {stage: (ref,) for stage, ref in provided.items()}
        request = executors["geometric_mapping"].requests[0]
        assert request.inputs["sequence"][0].artifact_id == "seq-7"
        assert request.inputs["trajectory"][0].artifact_id == "traj-3"

    def test_a_subgraph_whose_upstream_is_not_provided_pulls_it_in(self, tmp_path: Path) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))

        scope = plan.scope(targets=["geometric_mapping"])

        assert [stage.stage_id for stage in scope.stages] == [
            "ingestion",
            "state_estimation",
            "geometric_mapping",
        ]

    def test_a_provided_artifact_of_the_wrong_contract_is_refused(self, tmp_path: Path) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        wrong = {
            "ingestion": ArtifactRef(
                stage_id="ingestion", contract="PerceptionRunArtifact", artifact_id="x"
            )
        }

        report = preflight(plan.scope(targets=["state_estimation"], provided=wrong))

        assert not report.ok
        assert any("SequenceArtifact" in problem.message for problem in report.problems)

    def test_an_unknown_target_is_reported(self, tmp_path: Path) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))

        report = preflight(plan.scope(targets=["nope"]))

        assert any("nope" in problem.message for problem in report.problems)

    @pytest.mark.usefixtures("unavailable_future_stage")
    def test_preflight_blocks_before_any_stage_runs(self, tmp_path: Path) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        log: list[str] = []
        executors = _executors(plan, log)

        with pytest.raises(PreflightError) as error:
            run_plan(
                plan.scope(targets=["scene_graph"]),
                executors,
                environ={},
                module_available=_ready,
                provided_runtimes=_ALL_PROVIDED,
            )

        assert log == []
        assert any("scene_graph" in problem.path for problem in error.value.report.problems)

    def test_preflight_reports_missing_executors_backends_and_secrets_together(
        self, tmp_path: Path
    ) -> None:
        document = _document()
        document["components"]["visual_perception"]["semantic_interpretation"] = {
            "backend": "gemini",
            "gemini": {"model": "g", "timeout_s": 1, "max_retries": 1, "temperature": 0.0},
        }
        plan = resolve_plan(effective_from(tmp_path, document))
        executors = _executors(plan, [])
        del executors["state_estimation"]

        report = preflight(
            plan.scope(targets=["semantic_fusion"]),
            executors=executors,
            environ={},
            module_available=lambda name: name != "torch",
            provided_runtimes=(),
        )

        text = " ".join(f"{problem.path} {problem.message}" for problem in report.problems)
        assert "state_estimation" in text and "executor" in text
        assert "GEMINI_API_KEY" in text
        assert "torch" in text

    def test_an_incomplete_backend_selection_blocks_only_the_stages_in_scope(
        self, tmp_path: Path
    ) -> None:
        document = _document()
        document["components"]["state_estimation"]["estimator"] = {"backend": None}
        plan = resolve_plan(effective_from(tmp_path, document))

        blocked = preflight(
            plan.scope(targets=["state_estimation"]),
            environ={},
            module_available=_ready,
            provided_runtimes=_ALL_PROVIDED,
        )
        free = preflight(
            plan.scope(targets=["visual_perception"]),
            environ={},
            module_available=_ready,
            provided_runtimes=_ALL_PROVIDED,
        )

        assert any("state_estimation.estimator" in p.path for p in blocked.problems)
        assert free.ok

    def test_a_failing_stage_stops_the_run_and_keeps_what_completed(self, tmp_path: Path) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        log: list[str] = []
        executors: dict[str, Any] = _executors(plan, log)

        class Boom:
            def execute(self, request: StageRequest) -> ArtifactRef:
                raise RuntimeError("out of memory")

        executors["state_estimation"] = Boom()

        with pytest.raises(StageExecutionError, match="state_estimation") as error:
            run_plan(
                plan.scope(targets=["semantic_fusion"]),
                executors,
                environ={},
                module_available=_ready,
                provided_runtimes=_ALL_PROVIDED,
            )

        assert log == ["ingestion", "visual_perception"]
        assert error.value.stage_id == "state_estimation"
        assert error.value.completed == ("ingestion", "visual_perception")
        assert isinstance(error.value.__cause__, RuntimeError)

    def test_an_artifact_of_the_wrong_contract_is_not_accepted_from_a_stage(
        self, tmp_path: Path
    ) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        executors: dict[str, Any] = _executors(plan, [])
        executors["ingestion"] = FakeStage("ingestion", "PerceptionRunArtifact", [])

        with pytest.raises(StageExecutionError, match="SequenceArtifact"):
            run_plan(
                plan.scope(targets=["ingestion"]),
                executors,
                environ={},
                module_available=_ready,
                provided_runtimes=_ALL_PROVIDED,
            )

    def test_the_execution_record_persists_exact_inputs_outputs_and_order(
        self, tmp_path: Path
    ) -> None:
        plan = resolve_plan(effective_from(tmp_path, _document()))
        record = run_plan(
            plan.scope(targets=["state_estimation"]),
            _executors(plan, []),
            environ={},
            module_available=_ready,
            provided_runtimes=_ALL_PROVIDED,
        )

        document = record.to_document()

        assert document["plan_digest"] == plan.digest
        assert document["order"] == ["ingestion", "state_estimation"]
        assert document["stages"][1]["inputs"] == {
            "sequence": [
                {
                    "stage_id": "ingestion",
                    "contract": "SequenceArtifact",
                    "artifact_id": "ingestion#1",
                    "content_hash": None,
                }
            ]
        }
        assert document["stages"][1]["output"]["artifact_id"] == "state_estimation#1"

    def test_equivalent_runs_produce_equivalent_records(self, tmp_path: Path) -> None:
        records = []
        for _ in range(2):
            plan = resolve_plan(effective_from(tmp_path, _document()))
            records.append(
                run_plan(
                    plan.scope(targets=["geometric_mapping"]),
                    _executors(plan, []),
                    environ={},
                    module_available=_ready,
                    provided_runtimes=_ALL_PROVIDED,
                ).to_document()
            )

        assert records[0] == records[1]


_ALL_PROVIDED = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)
