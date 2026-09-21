"""Tests for the run lifecycle: events, status, failure records, cancellation and resume."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_worlds import World

from contextmap.runtime import (
    ArtifactRef,
    CancellationToken,
    EffectiveConfig,
    ExecutionEvent,
    ExecutionPlan,
    PreflightError,
    ResolvedSecrets,
    ResumeError,
    ReusePolicy,
    RunCancelledError,
    RunJournal,
    RunRecordError,
    RunStatus,
    StageExecutionError,
    StageFailure,
    StageRequest,
    capture_environment,
    categorize_failure,
    read_run,
    resolve_plan,
    resume_plan,
    run_plan,
)

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


def _ready(_name: str) -> bool:
    return True


def _clock() -> Any:
    ticks = itertools.count(1)
    return lambda: f"2026-01-01T00:00:{next(ticks):02d}.000+00:00"


def _document(**changes: Any) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    document.update(changes)
    return document


def _scope(
    tmp_path: Path, document: dict[str, Any] | None = None, targets: list[str] | None = None
) -> tuple[EffectiveConfig, ExecutionPlan]:
    effective = effective_from(tmp_path, document or _document())
    plan = resolve_plan(effective)
    return effective, plan.scope(targets=targets or ["semantic_fusion"])


def _run(
    tmp_path: Path,
    world: World,
    *,
    document: dict[str, Any] | None = None,
    targets: list[str] | None = None,
    reuse: ReusePolicy | None = None,
    journal: bool = True,
    workspace: str = "ws",
    **options: Any,
) -> tuple[RunJournal | None, Any]:
    effective, execution = _scope(tmp_path, document, targets)
    run_journal = (
        RunJournal.create(tmp_path / workspace, effective, execution, clock=_clock())
        if journal
        else None
    )
    options.setdefault("environ", {})
    record = run_plan(
        execution,
        world.executors(execution.plan),
        module_available=_ready,
        provided_runtimes=PROVIDED,
        reuse=reuse,
        journal=run_journal,
        clock=_clock(),
        **options,
    )
    return run_journal, record


def _policy(tmp_path: Path, world: World, **changes: Any) -> ReusePolicy:
    return ReusePolicy(store=world.store(tmp_path / "index"), code_identity="code-1", **changes)


def _kinds(journal: RunJournal) -> list[str]:
    return [event.kind for event in read_run(journal.directory).events]


def _fail(tmp_path: Path, world: World, **options: Any) -> RunJournal:
    """Run until the stage the world fails at, and return the journal of the failed run."""
    effective, execution = _scope(tmp_path)
    journal = RunJournal.create(tmp_path / "ws", effective, execution, clock=_clock())
    with pytest.raises(StageExecutionError):
        run_plan(
            execution,
            world.executors(execution.plan),
            environ={},
            module_available=_ready,
            provided_runtimes=PROVIDED,
            journal=journal,
            clock=_clock(),
            **options,
        )
    return journal


class TestSuccessfulRun:
    def test_events_are_numbered_ordered_and_timestamped(self, tmp_path: Path) -> None:
        journal, record = _run(tmp_path, World())
        assert journal is not None

        summary = read_run(journal.directory)

        kinds = [event.kind for event in summary.events]
        assert kinds[:2] == ["run_planned", "run_started"]
        assert kinds[-1] == "run_completed"
        assert kinds.count("stage_started") == kinds.count("stage_completed") == len(STAGES)
        assert [event.sequence for event in summary.events] == list(
            range(1, len(summary.events) + 1)
        )
        assert summary.events[0].time == "2026-01-01T00:00:01.000+00:00"
        assert record.order == tuple(STAGES)

    def test_the_status_execution_record_and_lock_reflect_a_completed_run(
        self, tmp_path: Path
    ) -> None:
        journal, _ = _run(tmp_path, World())
        assert journal is not None

        summary = read_run(journal.directory)

        assert summary.status is RunStatus.COMPLETED
        assert summary.completed_stages == tuple(STAGES)
        assert summary.failure is None and summary.notes == ()
        assert (journal.directory / "execution.json").is_file()
        assert not (journal.directory / "run.lock").exists()
        assert journal.directory.name == "run-0001"

    def test_stage_events_carry_the_exact_artifacts_and_inputs(self, tmp_path: Path) -> None:
        journal, _ = _run(tmp_path, World())
        assert journal is not None

        events = read_run(journal.directory).events
        started = next(
            e for e in events if e.kind == "stage_started" and e.stage_id == "geometric_mapping"
        )
        completed = next(
            e for e in events if e.kind == "stage_completed" and e.stage_id == "geometric_mapping"
        )

        assert started.data["inputs"] == {
            "sequence": ["ingestion-run1"],
            "trajectory": ["state_estimation-run3"],
        }
        assert completed.data["artifact"]["artifact_id"] == "geometric_mapping-run4"
        assert completed.data["elapsed_s"] >= 0

    def test_a_reused_stage_is_an_event_naming_the_prior_artifact(self, tmp_path: Path) -> None:
        world = World()
        _run(tmp_path, world, reuse=_policy(tmp_path, world), workspace="ws1")

        journal, _ = _run(tmp_path, world, reuse=_policy(tmp_path, world), workspace="ws2")
        assert journal is not None

        events = read_run(journal.directory).events
        reused = [e for e in events if e.kind == "stage_reused"]
        assert [e.stage_id for e in reused] == STAGES
        assert reused[0].data["artifact"]["artifact_id"] == "ingestion-run1"
        assert "stage_started" not in [e.kind for e in events]

    def test_an_in_memory_sink_receives_the_same_events_as_the_journal(
        self, tmp_path: Path
    ) -> None:
        received: list[ExecutionEvent] = []

        class Sink:
            def emit(self, event: ExecutionEvent) -> None:
                received.append(event)

        journal, _ = _run(tmp_path, World(), events=Sink())
        assert journal is not None

        assert received == list(read_run(journal.directory).events)

    def test_no_journal_is_needed(self, tmp_path: Path) -> None:
        _, record = _run(tmp_path, World(), journal=False)

        assert record.order == tuple(STAGES)

    def test_the_environment_is_recorded_without_loading_a_sdk_or_naming_the_host(
        self, tmp_path: Path
    ) -> None:
        loaded_before = "torch" in sys.modules
        journal, _ = _run(tmp_path, World())
        assert journal is not None

        environment = read_run(journal.directory).environment

        assert {"python", "platform", "machine", "contextmap", "cpu_count"} <= environment.keys()
        assert environment["device"] == "cpu"
        assert set(environment["packages"]) >= {"torch", "transformers", "PIL"}
        assert ("torch" in sys.modules) == loaded_before  # só consulta metadados
        import platform

        assert platform.node() not in [str(value) for value in environment.values()]

    def test_capture_environment_without_a_configuration_is_still_useful(self) -> None:
        environment = capture_environment()

        assert environment["python"] and "packages" not in environment


class TestFailureRecords:
    def test_a_failed_run_has_an_inspectable_failure_record(self, tmp_path: Path) -> None:
        world = World()
        world.fail_at = "geometric_mapping"

        journal = _fail(tmp_path, world)
        summary = read_run(journal.directory)

        assert summary.status is RunStatus.FAILED
        assert summary.failure is not None
        assert summary.failure.stage_id == "geometric_mapping"
        assert summary.failure.category == "execution"
        assert summary.failure.message == "out of memory"
        assert summary.failure.exception_type == "RuntimeError"
        assert summary.failure.completed == ("ingestion", "visual_perception", "state_estimation")
        assert not (journal.directory / "execution.json").exists()
        assert [e.kind for e in summary.events][-2:] == ["stage_failed", "run_failed"]

    @pytest.mark.parametrize(
        ("error", "category"),
        [
            (ImportError("no module named torch"), "dependency"),
            (MemoryError(), "resource"),
            (
                StageFailure("bad calibration", category="calibration_blocker"),
                "calibration_blocker",
            ),
            (StageFailure("plain"), "execution"),
            (ValueError("anything else"), "execution"),
        ],
    )
    def test_the_failure_category_follows_the_exception_not_the_message(
        self, tmp_path: Path, error: BaseException, category: str
    ) -> None:
        world = World()
        world.fail_at = "ingestion"
        world.fail_with = error

        summary = read_run(_fail(tmp_path, world).directory)

        assert summary.failure is not None and summary.failure.category == category
        assert categorize_failure(error) == category

    def test_a_contract_violation_is_its_own_category(self, tmp_path: Path) -> None:
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution, clock=_clock())
        executors = World().executors(execution.plan)

        class Wrong:
            def execute(self, request: StageRequest) -> ArtifactRef:
                return ArtifactRef(stage_id="ingestion", contract="Other", artifact_id="x")

        executors["ingestion"] = Wrong()

        with pytest.raises(StageExecutionError, match="Other"):
            run_plan(
                execution,
                executors,
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                journal=journal,
            )

        failure = read_run(journal.directory).failure
        assert failure is not None and failure.category == "contract"

    def test_a_stage_is_executed_once_and_never_retried(self, tmp_path: Path) -> None:
        calls: list[str] = []
        world = World()
        world.fail_at = "ingestion"
        original = world.executor

        def counting(stage_id: str, contract: str) -> Any:
            calls.append(stage_id)
            return original(stage_id, contract)

        world.executor = counting  # type: ignore[method-assign]

        _fail(tmp_path, world)

        assert calls.count("ingestion") == 1

    def test_an_unexpected_runner_failure_is_recorded_and_re_raised(self, tmp_path: Path) -> None:
        world = World()

        class BrokenStore:
            def find(self, key: Any) -> Any:
                raise RuntimeError("index unreadable")

            def record(self, key: Any, output: Any) -> bool:  # pragma: no cover
                return False

        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution)

        with pytest.raises(RuntimeError, match="index unreadable"):
            run_plan(
                execution,
                world.executors(execution.plan),
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                reuse=ReusePolicy(store=BrokenStore(), code_identity="c"),
                journal=journal,
            )

        summary = read_run(journal.directory)
        assert summary.status is RunStatus.FAILED
        assert summary.failure is not None and summary.failure.category == "unexpected"
        assert summary.failure.stage_id is None


class TestBlockedRun:
    def test_a_preflight_refusal_is_a_blocked_run_and_nothing_runs(self, tmp_path: Path) -> None:
        world = World()
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution, clock=_clock())

        with pytest.raises(PreflightError):
            run_plan(
                execution,
                world.executors(execution.plan),
                environ={},
                module_available=lambda name: name != "torch",
                provided_runtimes=(),
                journal=journal,
            )

        summary = read_run(journal.directory)
        assert summary.status is RunStatus.BLOCKED
        assert any("torch" in problem["message"] for problem in summary.blocked_problems)
        assert [e.kind for e in summary.events] == ["run_planned", "run_blocked"]
        assert world.runs == []
        assert not (journal.directory / "run.lock").exists()

    def test_a_plan_with_structural_problems_is_recorded_without_a_plan_file(
        self, tmp_path: Path
    ) -> None:
        effective = effective_from(tmp_path, _document())
        plan = resolve_plan(effective)
        execution = plan.scope(targets=["semantic_mapping"])  # capability inexistente
        journal = RunJournal.create(tmp_path / "ws", effective, execution)

        with pytest.raises(PreflightError):
            run_plan(execution, {}, environ={}, module_available=_ready, journal=journal)

        assert read_run(journal.directory).status is RunStatus.BLOCKED


class TestCancellation:
    def test_cancellation_stops_at_a_stage_boundary_with_every_artifact_complete(
        self, tmp_path: Path
    ) -> None:
        world = World()
        token = CancellationToken()
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution)
        executors = world.executors(execution.plan)
        real = executors["visual_perception"]

        class Cancelling:
            def execute(self, request: StageRequest) -> ArtifactRef:
                artifact: ArtifactRef = real.execute(request)
                token.cancel("operator asked")
                return artifact

        executors["visual_perception"] = Cancelling()

        with pytest.raises(RunCancelledError, match="operator asked") as error:
            run_plan(
                execution,
                executors,
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                journal=journal,
                cancellation=token,
            )

        assert world.runs == ["ingestion", "visual_perception"]
        assert error.value.stage_id == "state_estimation"
        summary = read_run(journal.directory)
        assert summary.status is RunStatus.CANCELLED
        assert summary.completed_stages == ("ingestion", "visual_perception")
        assert summary.events[-1].data["reason"] == "operator asked"

    def test_a_token_cancelled_before_the_run_starts_no_stage(self, tmp_path: Path) -> None:
        world = World()
        token = CancellationToken()
        token.cancel()

        with pytest.raises(RunCancelledError):
            _run(tmp_path, world, cancellation=token)

        assert world.runs == []

    def test_an_interrupt_inside_a_stage_is_recorded_as_a_cancellation_and_propagates(
        self, tmp_path: Path
    ) -> None:
        world = World()
        world.fail_at = "state_estimation"
        world.fail_with = KeyboardInterrupt()
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution)

        with pytest.raises(KeyboardInterrupt):
            run_plan(
                execution,
                world.executors(execution.plan),
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                journal=journal,
            )

        summary = read_run(journal.directory)
        assert summary.status is RunStatus.CANCELLED
        assert summary.events[-1].data["reason"] == "interrupted"


class TestSecrets:
    SECRET = "s3cr3t-token-value"

    def _gemini(self) -> dict[str, Any]:
        document = _document()
        document["components"]["visual_perception"]["semantic_interpretation"] = {
            "backend": "gemini",
            "gemini": {"model": "g", "timeout_s": 1, "max_retries": 1, "temperature": 0.0},
        }
        return document

    def test_a_secret_in_an_error_message_never_reaches_the_events_or_the_status(
        self, tmp_path: Path
    ) -> None:
        document = self._gemini()
        effective = effective_from(tmp_path, document)
        from contextmap.runtime import resolve_secrets

        secrets = resolve_secrets(effective.config, environ={"GEMINI_API_KEY": self.SECRET})
        world = World()
        world.fail_at = "visual_perception"
        world.fail_with = RuntimeError(f"401 for key {self.SECRET} on request")
        execution = resolve_plan(effective).scope(targets=["semantic_fusion"])
        journal = RunJournal.create(tmp_path / "ws", effective, execution)

        with pytest.raises(StageExecutionError) as error:
            run_plan(
                execution,
                world.executors(execution.plan),
                environ={"GEMINI_API_KEY": self.SECRET},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                journal=journal,
                redact=secrets.redact,
            )

        assert self.SECRET not in str(error.value)
        for path in journal.directory.iterdir():
            assert self.SECRET not in path.read_text(encoding="utf-8"), path.name
        failure = read_run(journal.directory).failure
        assert failure is not None and "***" in failure.message

    def test_redaction_replaces_every_held_value_and_never_renders_one(self) -> None:
        secrets = ResolvedSecrets(["A", "B"], {"A": "alpha-1", "B": "alpha-12"})

        assert secrets.redact("x alpha-12 y alpha-1 z") == "x *** y *** z"
        assert "alpha" not in repr(secrets)


class TestInterruptedRuns:
    def _interrupt(self, tmp_path: Path) -> RunJournal:
        """A run whose process died mid-stage: stage_started, no completion, a dead owner."""
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution, clock=_clock())
        from contextmap.runtime.lifecycle import EventEmitter

        emitter = EventEmitter([journal], clock=_clock())
        emitter.emit("run_planned", plan_digest="p", config_digest="c", stages=STAGES, provided={})
        emitter.emit("run_started")
        emitter.emit("stage_started", stage_id="ingestion", inputs={}, config_digest="d")
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (journal.directory / "run.lock").write_text(f"{dead.pid}\n", encoding="ascii")
        return journal

    def test_a_killed_run_reads_back_as_running_and_interrupted(self, tmp_path: Path) -> None:
        summary = read_run(self._interrupt(tmp_path).directory)

        assert summary.status is RunStatus.RUNNING
        assert summary.interrupted is True
        assert [e.kind for e in summary.events][-1] == "stage_started"

    def test_a_live_run_is_not_reported_as_interrupted(self, tmp_path: Path) -> None:
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution)

        assert read_run(journal.directory).interrupted is False  # o lock é deste processo

    def test_a_partial_last_line_is_ignored_and_reported(self, tmp_path: Path) -> None:
        journal = self._interrupt(tmp_path)
        with (journal.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"sequence":4,"time":"x","kind":"stage_comp')

        summary = read_run(journal.directory)

        assert summary.truncated is True
        assert len(summary.events) == 3

    def test_a_corrupt_line_in_the_middle_is_an_error(self, tmp_path: Path) -> None:
        journal = self._interrupt(tmp_path)
        path = journal.directory / "events.jsonl"
        lines = path.read_text("utf-8").splitlines()
        lines[1] = "{not json"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(RunRecordError, match="line 2"):
            read_run(journal.directory)

    def test_a_gap_in_the_numbering_is_an_error(self, tmp_path: Path) -> None:
        journal = self._interrupt(tmp_path)
        path = journal.directory / "events.jsonl"
        lines = path.read_text("utf-8").splitlines()
        del lines[1]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(RunRecordError, match="gap"):
            read_run(journal.directory)

    def test_unreadable_or_foreign_records_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RunRecordError, match=r"status\.json"):
            read_run(tmp_path)
        journal = self._interrupt(tmp_path)
        status = journal.directory / "status.json"
        document = json.loads(status.read_text("utf-8"))
        document["schema_version"] = "9.9.9"
        status.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(RunRecordError, match="schema version"):
            read_run(journal.directory)

    def test_inconsistencies_are_reported_as_notes(self, tmp_path: Path) -> None:
        journal, _ = _run(tmp_path, World())
        assert journal is not None
        (journal.directory / "execution.json").unlink()

        summary = read_run(journal.directory)

        assert any("execution.json" in note for note in summary.notes)


class TestResume:
    def _failed_then_fixed(
        self, tmp_path: Path, *, fail_at: str = "geometric_mapping"
    ) -> tuple[World, ReusePolicy, RunJournal]:
        world = World()
        policy = _policy(tmp_path, world)
        world.fail_at = fail_at
        journal = _fail(tmp_path, world, reuse=policy)
        world.fail_at = None
        world.runs.clear()
        return world, policy, journal

    def _resume(
        self,
        tmp_path: Path,
        world: World,
        policy: ReusePolicy,
        previous: RunJournal,
        *,
        document: dict[str, Any] | None = None,
        targets: list[str] | None = None,
    ) -> tuple[RunJournal, Any]:
        effective, execution = _scope(tmp_path, document, targets)
        journal = RunJournal.create(tmp_path / "ws", effective, execution, clock=_clock())
        record = resume_plan(
            previous.directory,
            execution,
            world.executors(execution.plan),
            reuse=policy,
            environ={},
            module_available=_ready,
            provided_runtimes=PROVIDED,
            journal=journal,
            clock=_clock(),
        )
        return journal, record

    def test_resume_reuses_the_completed_stages_and_continues_at_the_failed_one(
        self, tmp_path: Path
    ) -> None:
        world, policy, previous = self._failed_then_fixed(tmp_path)

        journal, record = self._resume(tmp_path, world, policy, previous)

        assert world.runs == ["geometric_mapping", "sensor_association", "semantic_fusion"]
        assert record.resume == {
            "from": "run-0001",
            "previous_status": "failed",
            "reused": ["ingestion", "visual_perception", "state_estimation"],
            "recomputed": [],
        }
        summary = read_run(journal.directory)
        assert summary.status is RunStatus.COMPLETED
        assert summary.resumed_from == "run-0001"
        kinds = [e.kind for e in summary.events]
        assert kinds[:3] == ["run_planned", "run_resumed", "run_started"]
        assert kinds.count("stage_reused") == 3

    def test_the_previous_run_is_history_and_is_not_modified(self, tmp_path: Path) -> None:
        world, policy, previous = self._failed_then_fixed(tmp_path)
        before = {p.name: p.read_bytes() for p in previous.directory.iterdir()}

        self._resume(tmp_path, world, policy, previous)

        assert {p.name: p.read_bytes() for p in previous.directory.iterdir()} == before

    def test_a_completed_stage_that_no_longer_verifies_is_recomputed_and_reported(
        self, tmp_path: Path
    ) -> None:
        world, policy, previous = self._failed_then_fixed(tmp_path)
        world.existing.discard("state_estimation-run3")

        _, record = self._resume(tmp_path, world, policy, previous)

        assert "state_estimation" in world.runs
        assert record.resume is not None
        assert record.resume["recomputed"] == ["state_estimation"]
        assert record.resume["reused"] == ["ingestion", "visual_perception"]

    def test_a_partial_output_of_the_interrupted_stage_is_never_reused(
        self, tmp_path: Path
    ) -> None:
        world = World()
        policy = _policy(tmp_path, world)
        world.fail_at = "state_estimation"
        world.fail_with = KeyboardInterrupt()
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution, clock=_clock())
        with pytest.raises(KeyboardInterrupt):
            run_plan(
                execution,
                world.executors(execution.plan),
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                journal=journal,
                reuse=policy,
            )
        world.fail_at = None
        world.runs.clear()

        _, record = self._resume(tmp_path, world, policy, journal)

        assert world.runs[0] == "state_estimation"
        assert record.resume is not None and record.resume["reused"] == [
            "ingestion",
            "visual_perception",
        ]

    def test_a_killed_run_can_be_resumed(self, tmp_path: Path) -> None:
        world, policy, previous = self._failed_then_fixed(tmp_path)
        status_path = previous.directory / "status.json"
        document = json.loads(status_path.read_text("utf-8"))
        document["status"] = "running"  # o processo morreu sem registrar o fim
        status_path.write_text(json.dumps(document), encoding="utf-8")
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (previous.directory / "run.lock").write_text(f"{dead.pid}\n", encoding="ascii")

        _, record = self._resume(tmp_path, world, policy, previous)

        assert record.resume is not None and record.resume["previous_status"] == "running"

    def test_a_cancelled_run_can_be_resumed(self, tmp_path: Path) -> None:
        world = World()
        policy = _policy(tmp_path, world)
        token = CancellationToken()
        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution)
        executors = world.executors(execution.plan)
        real = executors["ingestion"]

        class Cancelling:
            def execute(self, request: StageRequest) -> ArtifactRef:
                artifact: ArtifactRef = real.execute(request)
                token.cancel()
                return artifact

        executors["ingestion"] = Cancelling()
        with pytest.raises(RunCancelledError):
            run_plan(
                execution,
                executors,
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                journal=journal,
                reuse=policy,
                cancellation=token,
            )
        world.runs.clear()

        _, record = self._resume(tmp_path, world, policy, journal)

        assert record.resume is not None and record.resume["reused"] == ["ingestion"]

    @pytest.mark.parametrize(
        ("what", "change"),
        [
            ("effective configuration", {"resources": {"device": "cuda"}}),
        ],
    )
    def test_resuming_under_a_changed_request_is_refused(
        self, tmp_path: Path, what: str, change: dict[str, Any]
    ) -> None:
        world, policy, previous = self._failed_then_fixed(tmp_path)

        with pytest.raises(ResumeError, match=what):
            self._resume(tmp_path, world, policy, previous, document=_document(**change))

        assert world.runs == []

    def test_resuming_with_other_targets_is_refused(self, tmp_path: Path) -> None:
        world, policy, previous = self._failed_then_fixed(tmp_path)

        with pytest.raises(ResumeError, match="stages to run"):
            self._resume(tmp_path, world, policy, previous, targets=["geometric_mapping"])

    def test_supplied_upstream_artifacts_are_part_of_the_request(self, tmp_path: Path) -> None:
        world, policy, previous = self._failed_then_fixed(tmp_path)
        effective = effective_from(tmp_path, _document())
        execution = resolve_plan(effective).scope(
            targets=["semantic_fusion"],
            provided={
                "ingestion": ArtifactRef(
                    stage_id="ingestion",
                    contract="SequenceArtifact",
                    artifact_id="seq-9",
                    content_hash="sha256:9",
                )
            },
        )

        with pytest.raises(ResumeError, match="upstream"):
            resume_plan(
                previous.directory,
                execution,
                world.executors(execution.plan),
                reuse=policy,
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
            )

    def test_a_completed_blocked_or_live_run_cannot_be_resumed(self, tmp_path: Path) -> None:
        world = World()
        policy = _policy(tmp_path, world)
        done, _ = _run(tmp_path, world, reuse=policy, workspace="done")
        assert done is not None
        with pytest.raises(ResumeError, match="nothing to resume"):
            self._resume(tmp_path, world, policy, done)

        effective, execution = _scope(tmp_path)
        live = RunJournal.create(tmp_path / "live", effective, execution)
        live.emit(ExecutionEvent(sequence=1, time="t", kind="run_started", stage_id=None, data={}))
        with pytest.raises(ResumeError, match="in progress"):
            self._resume(tmp_path, world, policy, live)

        planned = RunJournal.create(tmp_path / "planned", effective, execution)
        with pytest.raises(ResumeError, match="nothing ran"):
            self._resume(tmp_path, world, policy, planned)

    def test_resuming_requires_a_reuse_policy(self, tmp_path: Path) -> None:
        world, _, previous = self._failed_then_fixed(tmp_path)
        _, execution = _scope(tmp_path)

        with pytest.raises(ValueError, match="reuse policy"):
            run_plan(
                execution,
                world.executors(execution.plan),
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                resume_from=read_run(previous.directory),
            )

    def test_a_failed_rerun_does_not_disturb_the_artifacts_of_an_earlier_success(
        self, tmp_path: Path
    ) -> None:
        world = World()
        policy = _policy(tmp_path, world)
        _run(tmp_path, world, reuse=policy, workspace="ws1")
        world.runs.clear()
        world.fail_at = "ingestion"
        forced = _policy(tmp_path, world, force_recompute=frozenset({"ingestion"}))

        effective, execution = _scope(tmp_path)
        journal = RunJournal.create(tmp_path / "ws2", effective, execution)
        with pytest.raises(StageExecutionError):
            run_plan(
                execution,
                world.executors(execution.plan),
                environ={},
                module_available=_ready,
                provided_runtimes=PROVIDED,
                reuse=forced,
                journal=journal,
            )
        world.fail_at = None

        _, record = _run(tmp_path, world, reuse=policy, workspace="ws3")

        assert world.runs == []  # o índice anterior segue intacto e reutilizável
        assert record.stages[0].output.artifact_id == "ingestion-run1"

    def test_the_same_failure_and_resume_produce_equivalent_records(self, tmp_path: Path) -> None:
        records = []
        for name in ("a", "b"):
            root = tmp_path / name
            root.mkdir()
            world, policy, previous = self._failed_then_fixed(root)
            _, record = self._resume(root, world, policy, previous)
            records.append(record.to_document())

        assert records[0] == records[1]
