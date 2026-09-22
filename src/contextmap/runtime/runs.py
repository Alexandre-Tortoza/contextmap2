"""The persistent run journal, run inspection and resume.

A run directory, ``<workspace>/<dataset>/run-NNNN/``, is the durable trail of one execution.
The dataset is the physical sequence the configuration is about (``inputs.sequence``). The
journal files below live at the run root; every stage the run executes writes its artifact in
its own directory next to them, ``<run>/<stage>/``:

- ``effective_config.json`` and ``plan.json``: what was requested and how it was resolved;
- ``events.jsonl``: the structured events, append-only, one JSON object per line;
- ``status.json``: the current lifecycle state, replaced atomically at every transition,
  with the failure record when the run failed;
- ``execution.json``: the exact inputs and outputs, written only when the run completes;
- ``run.lock``: present while the process that owns the run is alive.

A failed or cancelled run is history: it is never modified afterwards. Resuming starts a
**new** run that reuses the completed, still valid stage artifacts of the old one through
the normal reuse checks and continues from the first stage that has to run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextmap.runtime._files import replace_text
from contextmap.runtime.config import EffectiveConfig, write_effective_config
from contextmap.runtime.errors import ResumeError, RunRecordError
from contextmap.runtime.lifecycle import (
    RUN_SCHEMA_VERSION,
    TERMINAL_EVENTS,
    TERMINAL_STATUSES,
    CancellationToken,
    EventSink,
    ExecutionEvent,
    FailureRecord,
    RunStatus,
    capture_environment,
    utc_now,
)
from contextmap.runtime.pipeline import (
    ExecutionPlan,
    ExecutionRecord,
    StageExecutor,
    run_plan,
    write_execution_record,
    write_plan,
)
from contextmap.runtime.reuse import ReusePolicy

STATUS_FILENAME = "status.json"
EVENTS_FILENAME = "events.jsonl"
LOCK_FILENAME = "run.lock"


def dataset_directory(workspace: str | os.PathLike[str], dataset: str | None) -> Path:
    """Return ``<workspace>/<dataset>``, the directory that holds the runs of one dataset.

    Args:
        workspace: The workspace root.
        dataset: The physical sequence the runs are about (``inputs.sequence``).

    Returns:
        The dataset directory; it may not exist yet.

    Raises:
        ValueError: If ``dataset`` is missing, or is not a single path component (a name that
            could point outside the workspace is refused, never sanitized).
    """
    if not dataset:
        raise ValueError(
            "a run lives under <workspace>/<dataset>: set inputs.sequence to the dataset name"
        )
    if dataset in (".", "..") or Path(dataset).name != dataset:
        raise ValueError(f"the dataset {dataset!r} must be a single path component")
    return Path(workspace) / dataset


@dataclass(kw_only=True)
class _State:
    """The mutable part of a run's status, rebuilt from events as they arrive."""

    status: RunStatus
    plan_digest: str
    config_digest: str
    targets: list[str]
    provided: dict[str, list[str]]
    completed: list[str] = field(default_factory=list)
    failure: FailureRecord | None = None
    blocked: list[dict[str, str]] = field(default_factory=list)
    resumed_from: str | None = None
    created_at: str = ""
    updated_at: str = ""


class RunJournal:
    """Persists one run's lifecycle: status, events and the execution record.

    The journal is an :class:`~contextmap.runtime.lifecycle.EventSink`: the runner emits
    events into it, and it appends each to the event log and updates the run status
    atomically, so a process that dies at any point leaves a consistent, inspectable trail.
    """

    def __init__(
        self,
        directory: Path,
        *,
        state: _State,
        environment: Mapping[str, Any],
        code_identity: str | None,
    ) -> None:
        """Bind a journal to an existing run directory; use :meth:`create` instead."""
        self._directory = directory
        self._state = state
        self._environment = dict(environment)
        self._code_identity = code_identity

    @classmethod
    def create(
        cls,
        workspace: str | os.PathLike[str],
        effective: EffectiveConfig,
        execution: ExecutionPlan,
        *,
        code_identity: str | None = None,
        clock: Callable[[], str] | None = None,
    ) -> RunJournal:
        """Allocate a new run directory and record that the run is planned.

        The directory is created atomically, so concurrent runs never share one. The plan is
        written when it has no structural problem; a plan that cannot run is still recorded,
        as blocked, once the runner refuses it.

        Args:
            workspace: The workspace; a run lives under ``<workspace>/<dataset>``, where the
                dataset is ``effective.config.inputs.sequence``.
            effective: The effective configuration, persisted as ``effective_config.json``.
            execution: The scoped execution the run will perform.
            code_identity: Identity of the code producing the results, recorded for
                reproduction.
            clock: Returns timestamps; defaults to the current UTC time.

        Returns:
            The journal of the new run.

        Raises:
            ValueError: If the configuration names no dataset (``inputs.sequence``) or the
                name is not a single path component.
        """
        base = dataset_directory(workspace, effective.config.inputs.sequence)
        base.mkdir(parents=True, exist_ok=True)
        taken = [
            int(path.name.removeprefix("run-"))
            for path in base.iterdir()
            if path.name.startswith("run-") and path.name.removeprefix("run-").isdigit()
        ]
        index = max(taken, default=0) + 1
        while True:
            directory = base / f"run-{index:04d}"
            try:
                directory.mkdir()
            except FileExistsError:
                index += 1  # outro processo alocou este índice primeiro
                continue
            break
        descriptor = os.open(directory / LOCK_FILENAME, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n")
        write_effective_config(effective, directory)
        if not execution.plan.problems:
            write_plan(execution.plan, directory)
        now = (clock or utc_now)()
        state = _State(
            status=RunStatus.PLANNED,
            plan_digest=execution.plan.digest,
            config_digest=execution.plan.config_digest,
            targets=[stage.stage_id for stage in execution.stages],
            provided={
                stage: [ref.artifact_id for ref in refs] for stage, refs in execution.reused.items()
            },
            created_at=now,
            updated_at=now,
        )
        journal = cls(
            directory,
            state=state,
            environment=capture_environment(effective.config),
            code_identity=code_identity,
        )
        journal._write_status()
        return journal

    @property
    def directory(self) -> Path:
        """The run directory."""
        return self._directory

    @property
    def run_id(self) -> str:
        """The run identity, the name of its directory."""
        return self._directory.name

    @property
    def status(self) -> RunStatus:
        """The current lifecycle state."""
        return self._state.status

    def emit(self, event: ExecutionEvent) -> None:
        """Append an event to the log, then update the run status.

        Each event is one line, flushed to disk before the status changes, so the status
        never gets ahead of the trail.

        Args:
            event: The event, already numbered and redacted.
        """
        line = json.dumps(event.to_document(), sort_keys=True, separators=(",", ":")) + "\n"
        with (self._directory / EVENTS_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(event)
        self._write_status()
        if event.kind in TERMINAL_EVENTS:
            (self._directory / LOCK_FILENAME).unlink(missing_ok=True)

    def record_execution(self, record: ExecutionRecord) -> None:
        """Write ``execution.json``; it exists only for a run that completed.

        Args:
            record: The execution record of the run.
        """
        write_execution_record(record, self._directory)

    def _apply(self, event: ExecutionEvent) -> None:
        state, data = self._state, event.data
        state.updated_at = event.time
        if event.kind == "run_resumed":
            state.resumed_from = data.get("resumed_from")
        elif event.kind == "run_started":
            state.status = RunStatus.RUNNING
        elif event.kind in {"stage_completed", "stage_reused"} and event.stage_id is not None:
            state.completed.append(event.stage_id)
        elif event.kind == "run_failed":
            state.status = RunStatus.FAILED
            state.failure = FailureRecord(
                stage_id=event.stage_id,
                category=data["category"],
                message=data["message"],
                exception_type=data["exception_type"],
                completed=tuple(data["completed"]),
            )
        elif event.kind == "run_blocked":
            state.status = RunStatus.BLOCKED
            state.blocked = list(data["problems"])
        elif event.kind == "run_cancelled":
            state.status = RunStatus.CANCELLED
        elif event.kind == "run_completed":
            state.status = RunStatus.COMPLETED

    def _write_status(self) -> None:
        state = self._state
        document = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": state.status.value,
            "plan_digest": state.plan_digest,
            "config_digest": state.config_digest,
            "targets": state.targets,
            "provided": state.provided,
            "completed_stages": state.completed,
            "failure": None if state.failure is None else state.failure.to_document(),
            "blocked_problems": state.blocked,
            "resumed_from": state.resumed_from,
            "code_identity": self._code_identity,
            "environment": self._environment,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }
        replace_text(
            self._directory, STATUS_FILENAME, json.dumps(document, indent=2, sort_keys=True) + "\n"
        )


@dataclass(frozen=True, kw_only=True)
class RunSummary:
    """What a run directory says about its run, read back and cross-checked.

    Attributes:
        run_id: The run identity.
        directory: The run directory.
        status: The lifecycle state recorded in ``status.json``.
        interrupted: Whether the run never reached a terminal state and no live process owns
            it any more. It is what a process killed mid-run looks like.
        plan_digest: Digest of the topology the run executed.
        config_digest: Digest of the effective configuration.
        targets: The stages the run had to execute.
        provided: The upstream artifacts that were supplied, by stage.
        completed_stages: Stages that completed (or were reused), in order.
        failure: The failure record, when the run failed.
        blocked_problems: Why preflight refused to start, when the run was blocked.
        resumed_from: The run this one resumed, if any.
        code_identity: The code identity recorded for reproduction.
        environment: The environment recorded for reproduction.
        created_at: When the run was planned.
        updated_at: The time of the last recorded event.
        events: The events read back, in order.
        truncated: Whether a partial last line of the event log was ignored.
        notes: Inconsistencies found between the status, the events and the files.
    """

    run_id: str
    directory: Path
    status: RunStatus
    interrupted: bool
    plan_digest: str
    config_digest: str
    targets: tuple[str, ...]
    provided: Mapping[str, tuple[str, ...]]
    completed_stages: tuple[str, ...]
    failure: FailureRecord | None
    blocked_problems: tuple[Mapping[str, str], ...]
    resumed_from: str | None
    code_identity: str | None
    environment: Mapping[str, Any]
    created_at: str
    updated_at: str
    events: tuple[ExecutionEvent, ...]
    truncated: bool
    notes: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form of the summary, events included."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "interrupted": self.interrupted,
            "plan_digest": self.plan_digest,
            "config_digest": self.config_digest,
            "targets": list(self.targets),
            "provided": {stage: list(ids) for stage, ids in self.provided.items()},
            "completed_stages": list(self.completed_stages),
            "failure": None if self.failure is None else self.failure.to_document(),
            "blocked_problems": [dict(problem) for problem in self.blocked_problems],
            "resumed_from": self.resumed_from,
            "code_identity": self.code_identity,
            "environment": dict(self.environment),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "events": [event.to_document() for event in self.events],
            "truncated": self.truncated,
            "notes": list(self.notes),
        }


def read_run(path: str | os.PathLike[str]) -> RunSummary:
    """Read a run directory back and check it is coherent.

    The status is authoritative and the event log is the trail behind it. A partial last
    line of the log (the process died while appending) is ignored and reported; a corrupt
    line anywhere else, a gap in the numbering, an unreadable status or another schema
    version is an error. Coherence problems that do not prevent reading, such as a
    completed run without ``execution.json``, are returned as notes.

    Args:
        path: The run directory.

    Returns:
        The summary.

    Raises:
        RunRecordError: If the directory is not a readable run record.
    """
    directory = Path(path)
    status_path = directory / STATUS_FILENAME
    try:
        raw = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RunRecordError(
            f"{directory} is not a run record: cannot read status.json: {error}"
        ) from error
    if not isinstance(raw, dict) or raw.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RunRecordError(
            f"{status_path} follows another schema version; this runtime reads "
            f"{RUN_SCHEMA_VERSION!r}"
        )
    try:
        status = RunStatus(raw["status"])
        failure = None if raw["failure"] is None else FailureRecord.from_document(raw["failure"])
        summary_fields: dict[str, Any] = {
            "run_id": raw["run_id"],
            "plan_digest": raw["plan_digest"],
            "config_digest": raw["config_digest"],
            "targets": tuple(raw["targets"]),
            "provided": {stage: tuple(ids) for stage, ids in raw["provided"].items()},
            "completed_stages": tuple(raw["completed_stages"]),
            "blocked_problems": tuple(raw["blocked_problems"]),
            "resumed_from": raw["resumed_from"],
            "code_identity": raw["code_identity"],
            "environment": raw["environment"],
            "created_at": raw["created_at"],
            "updated_at": raw["updated_at"],
        }
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise RunRecordError(f"{status_path} is not a valid run status: {error}") from error

    events, truncated = _read_events(directory / EVENTS_FILENAME)
    notes: list[str] = []
    terminal_event = next((e for e in events if e.kind in TERMINAL_EVENTS), None)
    if terminal_event is not None and status not in TERMINAL_STATUSES:
        notes.append(
            f"the event log ends with {terminal_event.kind} but the status is {status.value}"
        )
    if terminal_event is None and status in TERMINAL_STATUSES and events:
        notes.append(f"the status is {status.value} but the event log has no terminal event")
    if status is RunStatus.COMPLETED and not (directory / "execution.json").is_file():
        notes.append("the run is completed but execution.json is missing")
    if status is not RunStatus.COMPLETED and (directory / "execution.json").is_file():
        notes.append(f"execution.json exists but the status is {status.value}")
    interrupted = status not in TERMINAL_STATUSES and not _lock_alive(directory)
    return RunSummary(
        directory=directory,
        status=status,
        interrupted=interrupted,
        failure=failure,
        events=events,
        truncated=truncated,
        notes=tuple(notes),
        **summary_fields,
    )


def _read_events(path: Path) -> tuple[tuple[ExecutionEvent, ...], bool]:
    if not path.is_file():
        return (), False
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line]
    events: list[ExecutionEvent] = []
    truncated = False
    for position, line in enumerate(lines):
        try:
            event = ExecutionEvent.from_document(json.loads(line))
        except ValueError as error:
            if position == len(lines) - 1:
                truncated = True  # a última linha foi cortada por um processo que morreu
                break
            raise RunRecordError(
                f"{path}: line {position + 1} is not a valid event: {error}"
            ) from error
        if event.sequence != position + 1:
            raise RunRecordError(
                f"{path}: event {position + 1} is numbered {event.sequence}; the log has a gap"
            )
        events.append(event)
    return tuple(events), truncated


def _lock_alive(directory: Path) -> bool:
    """Tell whether the process that owns a run is still alive."""
    try:
        pid = int((directory / LOCK_FILENAME).read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_resumable(summary: RunSummary, execution: ExecutionPlan) -> None:
    """Check that a run can be resumed by this execution.

    Only a failed, cancelled or interrupted run can be resumed, and only by the very same
    request: the same topology, the same effective configuration, the same stages to run
    and the same supplied upstream artifacts. Anything else is a new run, which still
    reuses whatever is unchanged through the ordinary reuse checks.

    Args:
        summary: The run to resume.
        execution: The execution that would resume it.

    Raises:
        ResumeError: If the run cannot be resumed as requested.
    """
    if summary.status is RunStatus.COMPLETED:
        raise ResumeError(f"{summary.run_id} already completed; there is nothing to resume")
    if summary.status in {RunStatus.PLANNED, RunStatus.BLOCKED}:
        raise ResumeError(
            f"{summary.run_id} is {summary.status.value}: nothing ran, start a new run instead"
        )
    if summary.status is RunStatus.RUNNING and not summary.interrupted:
        raise ResumeError(f"{summary.run_id} is still in progress; it cannot be resumed")
    changed = []
    if summary.plan_digest != execution.plan.digest:
        changed.append("the topology")
    if summary.config_digest != execution.plan.config_digest:
        changed.append("the effective configuration")
    if list(summary.targets) != [stage.stage_id for stage in execution.stages]:
        changed.append("the stages to run")
    provided = {
        stage: tuple(r.artifact_id for r in refs) for stage, refs in execution.reused.items()
    }
    if dict(summary.provided) != provided:
        changed.append("the supplied upstream artifacts")
    if changed:
        raise ResumeError(
            f"{summary.run_id} cannot be resumed: {', '.join(changed)} changed since it ran; "
            "start a new run (stages whose identity is unchanged are still reused)"
        )


def resume_plan(
    previous: str | os.PathLike[str],
    execution: ExecutionPlan,
    executors: Mapping[str, StageExecutor],
    *,
    reuse: ReusePolicy,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    provided_runtimes: Collection[str] = (),
    journal: RunJournal | None = None,
    events: EventSink | None = None,
    cancellation: CancellationToken | None = None,
    redact: Callable[[str], str] | None = None,
    clock: Callable[[], str] | None = None,
) -> ExecutionRecord:
    """Resume a failed, cancelled or interrupted run as a new run.

    Completed stages are reused **only** when the ordinary reuse checks pass: an artifact
    with the same identity is indexed and still verifies. Whatever fails a check is
    recomputed and reported as such; a partial or temporary output is never indexed, so it
    can never be mistaken for a completed stage. The execution then continues from the
    first stage that has to run. The previous run is not modified.

    Args:
        previous: The run directory to resume.
        execution: The execution that resumes it; it must be the same request.
        executors: One executor per stage that may run.
        reuse: The reuse policy; resuming has no meaning without one.
        environ: Environment to look secrets up in.
        module_available: Predicate telling whether an optional module is installed.
        provided_runtimes: Component identities whose model runtime the caller supplies.
        journal: The journal of the new run.
        events: An extra receiver of the run's events.
        cancellation: A cooperative cancellation handle.
        redact: Replaces secret values inside a string.
        clock: Returns event timestamps.

    Returns:
        The execution record of the new run, with a ``resume`` section saying which
        previously completed stages were reused and which were recomputed.

    Raises:
        ResumeError: If the run cannot be resumed as requested.
        RunRecordError: If the previous run record is unreadable.
    """
    summary = read_run(previous)
    check_resumable(summary, execution)
    return run_plan(
        execution,
        executors,
        environ=environ,
        module_available=module_available,
        provided_runtimes=provided_runtimes,
        reuse=reuse,
        journal=journal,
        events=events,
        cancellation=cancellation,
        redact=redact,
        clock=clock,
        resume_from=summary,
    )
