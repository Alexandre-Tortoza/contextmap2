"""Errors raised while turning a resolved configuration into concrete implementations."""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.runtime.config import ConfigProblem


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
