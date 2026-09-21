"""Errors raised while composing implementations and while planning or running the stage DAG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from contextmap.runtime.config import ConfigProblem

if TYPE_CHECKING:
    from contextmap.runtime.pipeline import PreflightReport


class CompositionError(Exception):
    """Base class of the errors the composition root raises."""


class BackendConfigurationError(CompositionError):
    """Raised when a capability rejects the parameters configured for its backend.

    Attributes:
        component_id: Variation point, ``"<capability>.<slot>"``.
        backend: Selected backend identity.
        problems: What the capability rejected, each prefixed by the parameter path.
    """

    def __init__(self, component_id: str, backend: str, problems: Sequence[str]) -> None:
        """Build the error from every parameter problem found."""
        self.component_id = component_id
        self.backend = backend
        self.problems = tuple(problems)
        detail = "; ".join(self.problems)
        super().__init__(f"{component_id}: backend {backend!r} rejected its parameters: {detail}")


class BackendUnavailableError(CompositionError):
    """Raised when a selected backend needs a module or a secret the environment lacks.

    Attributes:
        problems: One entry per missing module or secret, with how to obtain it.
    """

    def __init__(self, problems: Sequence[ConfigProblem]) -> None:
        """Build the error from every missing requirement found."""
        self.problems = tuple(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"a selected backend is unavailable:\n{lines}")


class BackendRuntimeMissingError(CompositionError):
    """Raised when a backend has no bundled model loader and the caller supplied none.

    Attributes:
        component_id: Variation point, ``"<capability>.<slot>"``.
        backend: Selected backend identity.
    """

    def __init__(self, component_id: str, backend: str, protocol: str) -> None:
        """Explain what the caller has to provide."""
        self.component_id = component_id
        self.backend = backend
        super().__init__(
            f"{component_id}: backend {backend!r} has no bundled model loader; pass a provider "
            f"for {component_id!r} in compose(providers=...) that returns a {protocol} for the "
            "built configuration"
        )


class StageUnavailableError(CompositionError):
    """Raised when a stage is requested whose capability is not implemented.

    Attributes:
        stage_id: The requested stage.
        reason: Why it cannot run.
    """

    def __init__(self, stage_id: str, reason: str) -> None:
        """Explain why the stage cannot be composed."""
        self.stage_id = stage_id
        self.reason = reason
        super().__init__(f"stage {stage_id!r} is unavailable: {reason}")


class PipelineError(Exception):
    """Base class of the errors the stage DAG raises."""


class PreflightError(PipelineError):
    """Raised when a plan cannot run: preflight found problems and nothing was executed.

    Attributes:
        report: Everything preflight found, so a user fixes it in one pass.
    """

    def __init__(self, report: PreflightReport) -> None:
        """Build the error from the preflight report."""
        self.report = report
        lines = "\n".join(f"  - {problem}" for problem in report.problems)
        super().__init__(f"the plan cannot run:\n{lines}")


class PlanDocumentError(PipelineError):
    """Raised when a persisted plan or execution record cannot be written or trusted."""


class StageExecutionError(PipelineError):
    """Raised when a stage fails or returns something its declaration forbids.

    The run stops at that stage: nothing later is executed and nothing is substituted.

    Attributes:
        stage_id: The stage that failed.
        completed: Stages that had completed before it, in execution order.
    """

    def __init__(self, stage_id: str, completed: Sequence[str], reason: str) -> None:
        """Explain which stage failed and what had completed."""
        self.stage_id = stage_id
        self.completed = tuple(completed)
        done = ", ".join(self.completed) or "none"
        super().__init__(f"stage {stage_id!r} failed: {reason} (completed before it: {done})")


class ReuseError(PipelineError):
    """Raised when an artifact cannot be indexed under the identity it was produced for."""


class RunCancelledError(PipelineError):
    """Raised when a run stops because cancellation was requested.

    The run stops at a stage boundary, so every artifact produced before it is complete.

    Attributes:
        stage_id: The stage that was about to start.
        completed: Stages that had completed before it, in order.
        reason: Why cancellation was requested.
    """

    def __init__(self, stage_id: str, completed: Sequence[str], reason: str) -> None:
        """Explain where the run stopped."""
        self.stage_id = stage_id
        self.completed = tuple(completed)
        self.reason = reason
        done = ", ".join(self.completed) or "none"
        super().__init__(
            f"run cancelled before stage {stage_id!r}: {reason} (completed before it: {done})"
        )


class RunRecordError(PipelineError):
    """Raised when a run directory is not a readable, coherent run record."""


class ResumeError(PipelineError):
    """Raised when a run cannot be resumed as requested."""
