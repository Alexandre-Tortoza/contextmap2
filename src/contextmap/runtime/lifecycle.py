"""Run lifecycle vocabulary: status, structured events, failure records and cancellation.

A run moves through explicit states and leaves a structured trail. This module holds the
parts of that trail that do not touch the filesystem: the states, the events the runner
emits, the failure categories, the cooperative cancellation handle, the environment capture
useful to reproduce a run, and the redaction that keeps secrets out of every event. The
persistent journal that stores them, and the resume operation, are in
:mod:`contextmap.runtime.runs`.

Nothing here retries, substitutes a backend or swaps a device: a failure is recorded and
raised, never repaired.
"""

from __future__ import annotations

import os
import platform
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from typing import Any, Protocol

from contextmap import __version__
from contextmap.runtime.catalog import COMPONENTS
from contextmap.runtime.config import RuntimeConfig

RUN_SCHEMA_VERSION = "0.1.0"
"""Version of the status, event and failure documents."""

REDACTION = "***"
"""What replaces a secret value wherever it would have been recorded."""


class RunStatus(StrEnum):
    """Where a run is in its lifecycle.

    Attributes:
        PLANNED: The run exists and has a plan, but nothing has started.
        RUNNING: Stages are executing (or the process died without recording an end; see
            ``RunSummary.interrupted``).
        COMPLETED: Every stage finished and the execution record was written.
        FAILED: A stage failed or the runner did; the failure record says where and why.
        BLOCKED: Preflight refused to start; nothing ran.
        CANCELLED: Cooperative cancellation or an interrupt stopped the run.
    """

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.BLOCKED, RunStatus.CANCELLED}
)

EVENT_KINDS = frozenset(
    {
        "run_planned",
        "run_resumed",
        "run_blocked",
        "run_started",
        "stage_started",
        "stage_reused",
        "stage_completed",
        "stage_failed",
        "run_failed",
        "run_cancelled",
        "run_completed",
        "ingestion.planned",
        "ingestion.reading-source",
        "ingestion.progress",
        "ingestion.validating",
        "ingestion.synchronizing",
        "ingestion.writing-artifact",
        "ingestion.completed",
        "ingestion.failed",
        "ingestion.cancelled",
    }
)
TERMINAL_EVENTS = frozenset({"run_blocked", "run_failed", "run_cancelled", "run_completed"})


class FailureCategory(StrEnum):
    """Why a stage or the runner failed, coarsely enough to act on.

    Attributes:
        DEPENDENCY: An optional module or model dependency could not be imported.
        RESOURCE: The machine ran out of a resource, for example memory.
        CONTRACT: A stage returned something its declaration forbids.
        EXECUTION: The stage raised while doing its work.
        UNEXPECTED: The runner itself failed outside a stage.
    """

    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    CONTRACT = "contract"
    EXECUTION = "execution"
    UNEXPECTED = "unexpected"


class StageFailure(Exception):
    """A failure an executor classifies itself, with a category of its own.

    Executors raise this to say more than "it raised": for example a calibration blocker
    or a corrupt input, which the runtime records verbatim.

    Attributes:
        category: The category recorded in the failure record.
    """

    def __init__(self, message: str, *, category: str = FailureCategory.EXECUTION.value) -> None:
        """Build the failure with its category."""
        super().__init__(message)
        self.category = category


def categorize_failure(error: BaseException) -> str:
    """Classify an exception raised by a stage.

    Args:
        error: What the executor raised.

    Returns:
        The executor's own category for a :class:`StageFailure`, ``"dependency"`` for an
        import failure, ``"resource"`` for :class:`MemoryError` and ``"execution"``
        otherwise. The message is never inspected: a category is not guessed from text.
    """
    if isinstance(error, StageFailure):
        return error.category
    if isinstance(error, ImportError):
        return FailureCategory.DEPENDENCY.value
    if isinstance(error, MemoryError):
        return FailureCategory.RESOURCE.value
    return FailureCategory.EXECUTION.value


@dataclass(frozen=True, kw_only=True)
class ExecutionEvent:
    """One structured fact about a run, in the order it happened.

    Attributes:
        sequence: Position in the run's trail, starting at 1 with no gaps.
        time: ISO-8601 UTC timestamp.
        kind: One of :data:`EVENT_KINDS`.
        stage_id: The stage the event is about, or ``None`` for the run as a whole.
        data: Kind-specific, JSON-compatible facts (artifact ids, decisions, failure).
    """

    sequence: int
    time: str
    kind: str
    stage_id: str | None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form persisted in the event log."""
        return {
            "sequence": self.sequence,
            "time": self.time,
            "kind": self.kind,
            "stage_id": self.stage_id,
            "data": dict(self.data),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ExecutionEvent:
        """Rebuild an event from its persisted form.

        Args:
            document: A mapping produced by :meth:`to_document`.

        Returns:
            The event.

        Raises:
            ValueError: If a field is missing, of the wrong type or the kind is unknown.
        """
        try:
            sequence, time, kind = document["sequence"], document["time"], document["kind"]
            stage_id, data = document["stage_id"], document["data"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"an event lacks {error}") from error
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not isinstance(time, str)
            or kind not in EVENT_KINDS
            or not (stage_id is None or isinstance(stage_id, str))
            or not isinstance(data, dict)
        ):
            raise ValueError(f"invalid event: {dict(document)!r}")
        return cls(sequence=sequence, time=time, kind=kind, stage_id=stage_id, data=data)


class EventSink(Protocol):
    """Receives the events of a run, in order; a sink must not change what the run does."""

    def emit(self, event: ExecutionEvent) -> None:
        """Receive one event."""
        ...


@dataclass(frozen=True, kw_only=True)
class FailureRecord:
    """Where and why a run failed.

    Attributes:
        stage_id: The stage that failed, or ``None`` when the runner failed outside one.
        category: A :class:`FailureCategory` value or an executor's own category.
        message: What went wrong, with secrets redacted.
        exception_type: Name of the exception class, for diagnosis.
        completed: Stages that had completed before the failure, in order.
    """

    stage_id: str | None
    category: str
    message: str
    exception_type: str
    completed: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form persisted in the run status."""
        return {
            "stage_id": self.stage_id,
            "category": self.category,
            "message": self.message,
            "exception_type": self.exception_type,
            "completed": list(self.completed),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> FailureRecord:
        """Rebuild a failure record from its persisted form.

        Args:
            document: A mapping produced by :meth:`to_document`.

        Returns:
            The record.

        Raises:
            ValueError: If a field is missing or of the wrong type.
        """
        try:
            stage_id = document["stage_id"]
            category, message = document["category"], document["message"]
            exception_type, completed = document["exception_type"], document["completed"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"a failure record lacks {error}") from error
        if (
            not (stage_id is None or isinstance(stage_id, str))
            or not all(isinstance(v, str) for v in (category, message, exception_type))
            or not isinstance(completed, list)
        ):
            raise ValueError(f"invalid failure record: {dict(document)!r}")
        return cls(
            stage_id=stage_id,
            category=category,
            message=message,
            exception_type=exception_type,
            completed=tuple(completed),
        )


class CancellationToken:
    """A cooperative cancellation handle.

    The runner checks it between stages: it never interrupts a stage in the middle, so a
    cancelled run stops at a boundary where every artifact is complete.
    """

    def __init__(self) -> None:
        """Create a token that is not cancelled."""
        self._event = threading.Event()
        self._reason = "cancelled"

    def cancel(self, reason: str = "cancelled") -> None:
        """Ask the run to stop before its next stage.

        Args:
            reason: Recorded in the run's cancellation event.
        """
        self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Why cancellation was requested."""
        return self._reason


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class EventEmitter:
    """Numbers, timestamps and redacts events, then hands them to every sink in order."""

    def __init__(
        self,
        sinks: Sequence[EventSink],
        *,
        clock: Callable[[], str] | None = None,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        """Create an emitter.

        Args:
            sinks: Receivers of every event, in order.
            clock: Returns the timestamp of an event; defaults to the current UTC time.
            redact: Replaces secret values inside a string; defaults to no redaction.
        """
        self._sinks = list(sinks)
        self._clock = clock or utc_now
        self._redact = redact
        self._sequence = 0
        self.terminal = False

    def emit(self, kind: str, stage_id: str | None = None, **data: Any) -> ExecutionEvent:
        """Emit one event.

        Args:
            kind: One of :data:`EVENT_KINDS`.
            stage_id: The stage the event is about, if any.
            **data: Kind-specific facts; every string in them is redacted.

        Returns:
            The event that was emitted.
        """
        self._sequence += 1
        cleaned = data if self._redact is None else redact_document(data, self._redact)
        event = ExecutionEvent(
            sequence=self._sequence, time=self._clock(), kind=kind, stage_id=stage_id, data=cleaned
        )
        for sink in self._sinks:
            sink.emit(event)
        if kind in TERMINAL_EVENTS:
            self.terminal = True
        return event


def redact_document(value: Any, redact: Callable[[str], str]) -> Any:
    """Apply a string redaction to every string inside a JSON-compatible value.

    Args:
        value: Any JSON-compatible value.
        redact: Replaces secret values inside one string.

    Returns:
        A copy in which every string, key or value, has been redacted.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {redact(str(key)): redact_document(item, redact) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact_document(item, redact) for item in value]
    return value


def capture_environment(config: RuntimeConfig | None = None) -> dict[str, Any]:
    """Describe the machine and software a run executed on, for reproduction.

    Only facts that are safe to persist: no host name, no user, no environment variable.
    Package versions are looked up in the installed metadata of the modules the selected
    backends need; nothing is imported, so no model SDK is loaded.

    Args:
        config: The resolved configuration, to report the device and the versions of the
            optional packages its selected backends require.

    Returns:
        A JSON-compatible mapping.
    """
    environment: dict[str, Any] = {
        "contextmap": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    if config is None:
        return environment
    environment["device"] = config.resources.device
    modules = sorted(
        {
            module
            for component_id, component in config.components.items()
            if component.backend is not None
            for module in COMPONENTS[component_id].backends[component.backend].requires
        }
    )
    if modules:
        distributions = metadata.packages_distributions()
        environment["packages"] = {
            module: _installed_version(distributions.get(module)) for module in modules
        }
    return environment


def _installed_version(distributions: Sequence[str] | None) -> str | None:
    for name in distributions or ():
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None
