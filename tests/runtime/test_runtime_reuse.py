"""Tests for immutable stage reuse, cache identity and selective recomputation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.runtime import (
    ArtifactRef,
    FileArtifactStore,
    PipelinePlan,
    PreflightError,
    ReusePolicy,
    RuntimePreset,
    StageDeclaration,
    StageExecutionError,
    StageRequest,
    predict_reuse,
    preflight,
    resolve_effective_config,
    resolve_plan,
    run_plan,
)
from contextmap.runtime.catalog import PRESETS, Interception, StageInput

IMPLEMENTED = [
    "ingestion",
    "visual_perception",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "semantic_fusion",
]
CODE = "code-1"


def _ready(_name: str) -> bool:
    return True


def _missing(_name: str) -> bool:
    return False


class World:
    """Fake stages that behave like pure functions of their inputs and configuration.

    Content is derived from what a stage consumed and how it is configured, so identical
    identity gives identical content; each execution is a distinct run with its own
    artifact id. ``ignore_config`` makes content independent of configuration, to model a
    configuration change that does not change what the stage produces.
    """

    def __init__(self, *, ignore_config: bool = False) -> None:
        self.ignore_config = ignore_config
        self.runs: list[str] = []
        self.existing: set[str] = set()
        self.fail_at: str | None = None
        self._count = 0

    def executor(self, stage_id: str, contract: str) -> Any:
        world = self

        class Executor:
            def execute(self, request: StageRequest) -> ArtifactRef:
                if world.fail_at == stage_id:
                    raise RuntimeError("out of memory")
                world._count += 1
                world.runs.append(stage_id)
                parts = [stage_id]
                if not world.ignore_config:
                    parts.append(request.config_digest)
                parts.extend(
                    f"{name}={ref.content_hash}"
                    for name, refs in sorted(request.inputs.items())
                    for ref in refs
                )
                content = "sha256:" + hashlib.sha256("|".join(parts).encode()).hexdigest()
                artifact_id = f"{stage_id}-run{world._count}"
                world.existing.add(artifact_id)
                return ArtifactRef(
                    stage_id=stage_id,
                    contract=contract,
                    artifact_id=artifact_id,
                    content_hash=content,
                )

        return Executor()

    def executors(self, plan: PipelinePlan) -> dict[str, Any]:
        return {
            stage.stage_id: self.executor(stage.stage_id, stage.output or "")
            for stage in plan.stages
        }

    def store(self, root: Path) -> FileArtifactStore:
        return FileArtifactStore(root, verify=lambda ref: ref.artifact_id in self.existing)


def _document(**fusion_changes: Any) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    if fusion_changes:
        document["components"]["semantic_fusion"]["support"]["geometry-jaccard-support-v1"].update(
            fusion_changes
        )
    return document


def _run(
    tmp_path: Path,
    world: World,
    document: dict[str, Any],
    *,
    store: FileArtifactStore | None = None,
    code: str = CODE,
    force: frozenset[str] = frozenset(),
    identities: dict[str, dict[str, str]] | None = None,
    provided: dict[str, ArtifactRef] | None = None,
    targets: list[str] | None = None,
) -> Any:
    plan = resolve_plan(effective_from(tmp_path, document))
    policy = ReusePolicy(
        store=store or world.store(tmp_path / "index"),
        code_identity=code,
        force_recompute=force,
        identities=identities or {},
    )
    return run_plan(
        plan.scope(targets=targets or ["semantic_fusion"], provided=provided),
        world.executors(plan),
        environ={},
        module_available=_ready,
        provided_runtimes=_PROVIDED,
        reuse=policy,
    )


_PROVIDED = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)


def _kinds(record: Any) -> dict[str, str]:
    return {stage.stage_id: stage.decision.kind for stage in record.stages}


class TestReuseOfIdenticalWork:
    def test_a_second_identical_run_reuses_every_stage_without_executing_any(
        self, tmp_path: Path
    ) -> None:
        world = World()
        first = _run(tmp_path, world, _document())
        assert world.runs == IMPLEMENTED

        second = _run(tmp_path, world, _document())

        assert world.runs == IMPLEMENTED  # nada novo foi executado
        assert set(_kinds(second).values()) == {"reused"}
        for before, after in zip(first.stages, second.stages, strict=True):
            assert after.output == before.output

    def test_a_reuse_decision_names_the_exact_prior_artifact(self, tmp_path: Path) -> None:
        world = World()
        first = _run(tmp_path, world, _document())
        second = _run(tmp_path, world, _document())

        decision = second.stages[0].decision

        assert decision.reused_from == first.stages[0].output
        assert decision.reused_from is not None
        assert decision.reused_from.artifact_id == "ingestion-run1"
        assert decision.key_digest is not None and decision.key_digest.startswith("sha256:")

    def test_a_fully_reused_run_needs_no_executor_and_no_optional_module(
        self, tmp_path: Path
    ) -> None:
        world = World()
        _run(tmp_path, world, _document())
        plan = resolve_plan(effective_from(tmp_path, _document()))
        policy = ReusePolicy(store=world.store(tmp_path / "index"), code_identity=CODE)

        record = run_plan(
            plan.scope(targets=["semantic_fusion"]),
            {},
            environ={},
            module_available=_missing,
            provided_runtimes=(),
            reuse=policy,
        )

        assert set(_kinds(record).values()) == {"reused"}

    def test_decisions_are_persisted_in_the_execution_record(self, tmp_path: Path) -> None:
        world = World()
        _run(tmp_path, world, _document())
        second = _run(tmp_path, world, _document())

        stage = second.to_document()["stages"][0]

        assert stage["decision"]["kind"] == "reused"
        assert stage["decision"]["reused_from"]["artifact_id"] == "ingestion-run1"
        assert stage["decision"]["reason"]


class TestSelectiveRecomputation:
    def test_changing_a_downstream_policy_recomputes_only_that_stage(self, tmp_path: Path) -> None:
        world = World()
        _run(tmp_path, world, _document())
        world.runs.clear()

        record = _run(tmp_path, world, _document(min_overlap=0.6))

        assert world.runs == ["semantic_fusion"]
        assert _kinds(record) == {
            "ingestion": "reused",
            "visual_perception": "reused",
            "state_estimation": "reused",
            "geometric_mapping": "reused",
            "sensor_association": "reused",
            "semantic_fusion": "recomputed",
        }

    def test_changing_an_upstream_stage_recomputes_its_true_dependents_only(
        self, tmp_path: Path
    ) -> None:
        world = World()
        _run(tmp_path, world, _document())
        world.runs.clear()
        document = _document()
        document["components"]["visual_perception"]["region_discovery"]["sam3"]["prompt"] = "x"

        record = _run(tmp_path, world, document)

        assert world.runs == ["visual_perception", "sensor_association", "semantic_fusion"]
        assert _kinds(record)["state_estimation"] == "reused"
        assert _kinds(record)["geometric_mapping"] == "reused"

    def test_reuse_follows_content_not_configuration_when_the_content_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        world = World(ignore_config=True)
        _run(tmp_path, world, _document())
        world.runs.clear()
        document = _document()
        document["components"]["visual_perception"]["region_discovery"]["sam3"]["prompt"] = "x"

        record = _run(tmp_path, world, document)

        # A etapa mudou de configuração, foi recomputada e produziu o mesmo conteúdo:
        # os dependentes continuam reutilizáveis.
        assert world.runs == ["visual_perception"]
        assert _kinds(record)["visual_perception"] == "recomputed"
        assert _kinds(record)["sensor_association"] == "reused"

    def test_a_different_code_identity_invalidates_everything(self, tmp_path: Path) -> None:
        world = World()
        _run(tmp_path, world, _document())
        world.runs.clear()

        _run(tmp_path, world, _document(), code="code-2")

        assert world.runs == IMPLEMENTED

    def test_an_extra_identity_invalidates_the_stage_it_is_declared_for(
        self, tmp_path: Path
    ) -> None:
        world = World()
        _run(tmp_path, world, _document(), identities={"ingestion": {"calibration": "cal-A"}})
        world.runs.clear()

        record = _run(
            tmp_path, world, _document(), identities={"ingestion": {"calibration": "cal-B"}}
        )

        # A identidade declarada entra só na chave da etapa: ela é recomputada e, como o
        # conteúdo produzido é o mesmo, as dependentes continuam reutilizáveis.
        assert world.runs == ["ingestion"]
        assert _kinds(record)["ingestion"] == "recomputed"
        assert _kinds(record)["visual_perception"] == "reused"

    def test_a_forced_recomputation_runs_the_stage_and_says_so(self, tmp_path: Path) -> None:
        world = World()
        first = _run(tmp_path, world, _document())
        world.runs.clear()

        record = _run(tmp_path, world, _document(), force=frozenset({"state_estimation"}))

        assert world.runs == ["state_estimation"]
        decision = {stage.stage_id: stage.decision for stage in record.stages}["state_estimation"]
        assert decision.kind == "recomputed"
        assert "forced" in decision.reason
        assert record.stages[2].output.artifact_id != first.stages[2].output.artifact_id

    def test_a_forced_run_never_replaces_the_indexed_artifact(self, tmp_path: Path) -> None:
        world = World()
        first = _run(tmp_path, world, _document())
        _run(tmp_path, world, _document(), force=frozenset({"state_estimation"}))

        later = _run(tmp_path, world, _document())

        assert {s.stage_id: s.output for s in later.stages}["state_estimation"] == {
            s.stage_id: s.output for s in first.stages
        }["state_estimation"]


class TestAlternativeDagsShareUpstream:
    def _preset(self) -> RuntimePreset:
        return RuntimePreset(
            preset_id="test/arms",
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
                    output="DenseMap",
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

    def _arm(
        self,
        tmp_path: Path,
        world: World,
        monkeypatch: pytest.MonkeyPatch,
        *,
        enhanced: bool,
    ) -> Any:
        monkeypatch.setitem(PRESETS, "test/arms", self._preset())
        file = tmp_path / f"arm-{enhanced}.json"
        file.write_text(json.dumps({"pipeline": {"stages": {"enhance": enhanced}}}), "utf-8")
        plan = resolve_plan(resolve_effective_config(profile="test/arms", files=[file]))
        return run_plan(
            plan.scope(targets=["associate"]),
            world.executors(plan),
            environ={},
            module_available=_ready,
            reuse=ReusePolicy(store=world.store(tmp_path / "index"), code_identity=CODE),
        )

    def test_the_enhanced_arm_reuses_the_native_arms_upstream_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world = World()
        native = self._arm(tmp_path, world, monkeypatch, enhanced=False)
        world.runs.clear()

        enhanced = self._arm(tmp_path, world, monkeypatch, enhanced=True)

        assert world.runs == ["enhance", "associate"]
        native_outputs = {s.stage_id: s.output for s in native.stages}
        enhanced_outputs = {s.stage_id: s.output for s in enhanced.stages}
        assert enhanced_outputs["source"] == native_outputs["source"]
        assert enhanced_outputs["dense"] == native_outputs["dense"]
        assert enhanced_outputs["associate"] != native_outputs["associate"]

    def test_the_native_arm_stays_reusable_after_the_enhanced_one_ran(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world = World()
        self._arm(tmp_path, world, monkeypatch, enhanced=False)
        self._arm(tmp_path, world, monkeypatch, enhanced=True)
        world.runs.clear()

        again = self._arm(tmp_path, world, monkeypatch, enhanced=False)

        assert world.runs == []
        assert set(_kinds(again).values()) == {"reused"}


class TestNothingUnsafeIsReused:
    def test_a_failed_stage_leaves_nothing_reusable_and_earlier_work_stays_reusable(
        self, tmp_path: Path
    ) -> None:
        world = World()
        world.fail_at = "geometric_mapping"
        with pytest.raises(StageExecutionError):
            _run(tmp_path, world, _document())
        world.fail_at = None
        world.runs.clear()

        record = _run(tmp_path, world, _document())

        assert world.runs == ["geometric_mapping", "sensor_association", "semantic_fusion"]
        assert _kinds(record)["ingestion"] == "reused"
        assert _kinds(record)["geometric_mapping"] == "recomputed"

    def test_an_indexed_artifact_that_fails_verification_is_not_reused(
        self, tmp_path: Path
    ) -> None:
        world = World()
        _run(tmp_path, world, _document())
        world.existing.discard("state_estimation-run3")
        world.runs.clear()

        record = _run(tmp_path, world, _document())

        assert "state_estimation" in world.runs
        decision = {s.stage_id: s.decision for s in record.stages}["state_estimation"]
        assert "verification" in decision.reason

    def test_an_unreadable_or_mismatched_index_entry_is_treated_as_no_prior_artifact(
        self, tmp_path: Path
    ) -> None:
        world = World()
        _run(tmp_path, world, _document())
        entries = sorted((tmp_path / "index").glob("sha256-*.json"))
        entries[0].write_text("{not json", encoding="utf-8")
        entry = json.loads(entries[1].read_text("utf-8"))
        entry["key"]["code_identity"] = "forged"
        entries[1].write_text(json.dumps(entry), encoding="utf-8")
        world.runs.clear()

        record = _run(tmp_path, world, _document())

        recomputed = {stage_id for stage_id, kind in _kinds(record).items() if kind == "recomputed"}
        assert len(recomputed) >= 2
        assert set(world.runs) >= recomputed

    def test_temporary_files_in_the_index_are_never_entries(self, tmp_path: Path) -> None:
        world = World()
        (tmp_path / "index").mkdir()
        (tmp_path / "index" / ".tmp-abc.json").write_text("{}", encoding="utf-8")

        record = _run(tmp_path, world, _document())

        assert set(_kinds(record).values()) == {"recomputed"}

    def test_an_input_without_a_content_hash_cannot_be_reused_downstream(
        self, tmp_path: Path
    ) -> None:
        world = World()
        provided = {
            "ingestion": ArtifactRef(
                stage_id="ingestion", contract="SequenceArtifact", artifact_id="seq-9"
            )
        }
        _run(tmp_path, world, _document(), provided=provided)
        world.runs.clear()

        record = _run(tmp_path, world, _document(), provided=provided)

        decision = {s.stage_id: s.decision for s in record.stages}["visual_perception"]
        assert decision.kind == "recomputed"
        assert "content hash" in decision.reason
        assert decision.key_digest is None

    def test_an_output_without_a_content_hash_is_not_indexed(self, tmp_path: Path) -> None:
        world = World()

        class Bare:
            def execute(self, request: StageRequest) -> ArtifactRef:
                return ArtifactRef(
                    stage_id="ingestion", contract="SequenceArtifact", artifact_id="seq-1"
                )

        plan = resolve_plan(effective_from(tmp_path, _document()))
        executors = world.executors(plan)
        executors["ingestion"] = Bare()
        run_plan(
            plan.scope(targets=["ingestion"]),
            executors,
            environ={},
            module_available=_ready,
            provided_runtimes=_PROVIDED,
            reuse=ReusePolicy(store=world.store(tmp_path / "index"), code_identity=CODE),
        )

        assert not list((tmp_path / "index").glob("sha256-*.json"))


class TestPreflightAndPrediction:
    def test_a_forced_stage_outside_the_execution_is_a_problem(self, tmp_path: Path) -> None:
        world = World()
        plan = resolve_plan(effective_from(tmp_path, _document()))
        policy = ReusePolicy(
            store=world.store(tmp_path / "index"),
            code_identity=CODE,
            force_recompute=frozenset({"semantic_mapping"}),
        )

        report = preflight(plan.scope(targets=["ingestion"]), reuse=policy)

        assert any("semantic_mapping" in problem.message for problem in report.problems)

    def test_the_prediction_says_what_would_be_reused_before_anything_runs(
        self, tmp_path: Path
    ) -> None:
        world = World()
        _run(tmp_path, world, _document())
        changed = _document(min_overlap=0.6)
        plan = resolve_plan(effective_from(tmp_path, changed))
        policy = ReusePolicy(store=world.store(tmp_path / "index"), code_identity=CODE)

        predicted = predict_reuse(plan.scope(targets=["semantic_fusion"]), policy)

        assert {stage_id: d.kind for stage_id, d in predicted.items()} == {
            "ingestion": "reused",
            "visual_perception": "reused",
            "state_estimation": "reused",
            "geometric_mapping": "reused",
            "sensor_association": "reused",
            "semantic_fusion": "recomputed",
        }

    def test_the_prediction_is_conservative_downstream_of_a_stage_that_will_recompute(
        self, tmp_path: Path
    ) -> None:
        world = World()
        _run(tmp_path, world, _document())
        document = _document()
        document["components"]["visual_perception"]["region_discovery"]["sam3"]["prompt"] = "x"
        plan = resolve_plan(effective_from(tmp_path, document))
        policy = ReusePolicy(store=world.store(tmp_path / "index"), code_identity=CODE)

        predicted = predict_reuse(plan.scope(targets=["semantic_fusion"]), policy)

        assert predicted["visual_perception"].kind == "recomputed"
        assert predicted["sensor_association"].kind == "recomputed"
        assert "upstream" in predicted["sensor_association"].reason
        assert predicted["state_estimation"].kind == "reused"

    def test_a_run_without_a_reuse_policy_records_no_decision(self, tmp_path: Path) -> None:
        world = World()
        plan = resolve_plan(effective_from(tmp_path, _document()))

        record = run_plan(
            plan.scope(targets=["ingestion"]),
            world.executors(plan),
            environ={},
            module_available=_ready,
            provided_runtimes=_PROVIDED,
        )

        assert record.stages[0].decision is None
        assert record.to_document()["stages"][0]["decision"] is None

    def test_a_missing_executor_is_still_reported_when_the_stage_cannot_be_reused(
        self, tmp_path: Path
    ) -> None:
        world = World()
        plan = resolve_plan(effective_from(tmp_path, _document()))
        policy = ReusePolicy(store=world.store(tmp_path / "index"), code_identity=CODE)

        with pytest.raises(PreflightError, match="executor"):
            run_plan(
                plan.scope(targets=["ingestion"]),
                {},
                environ={},
                module_available=_ready,
                provided_runtimes=_PROVIDED,
                reuse=policy,
            )


class TestReuseKey:
    def _key(self, **changes: Any) -> Any:
        from contextmap.runtime import ReuseKey

        values: dict[str, Any] = {
            "stage_id": "s",
            "contract": "Out",
            "stage_config_digest": "sha256:a",
            "inputs": {"x": ("In", "sha256:same")},
            "code_identity": CODE,
            "identities": {},
        }
        values.update(changes)
        return ReuseKey(**values)

    def test_it_is_stable_and_uses_content_never_a_name(self) -> None:
        assert self._key().digest == self._key().digest
        assert self._key().digest.startswith("sha256:")

    @pytest.mark.parametrize(
        "change",
        [
            {"stage_id": "t"},
            {"contract": "Other"},
            {"stage_config_digest": "sha256:b"},
            {"inputs": {"x": ("In", "sha256:other")}},
            {"inputs": {"x": ("Other", "sha256:same")}},
            {"inputs": {"y": ("In", "sha256:same")}},
            {"code_identity": "code-2"},
            {"identities": {"calibration": "cal"}},
        ],
    )
    def test_every_relevant_identity_changes_it(self, change: dict[str, Any]) -> None:
        assert self._key(**change).digest != self._key().digest
