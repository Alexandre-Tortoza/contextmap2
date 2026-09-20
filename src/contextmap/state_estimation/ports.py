"""Port and request/result contracts for State Estimation backends.

A backend (an external pose source, FAST-LIO, ...) is an adapter that
satisfies :class:`StateEstimator`. Callers hand it canonical observations and
receive a canonical :class:`~contextmap.state_estimation.Trajectory`, so
downstream geometry never sees a ROS message, an estimator's native object or
a dataset's pose format. The port never constructs a concrete backend: that is
the composition root's job, and there is no implicit fallback from one backend
to another when the selected one fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from contextmap.ingestion import (
    CalibrationSet,
    SequenceArtifactId,
    SourceObservation,
    SourceObservationId,
)
from contextmap.state_estimation.models import EstimatorProvenance, Trajectory, TrajectoryId


class StateEstimationError(Exception):
    """Base class for failures a State Estimation backend reports."""


class MissingEstimatorInputError(StateEstimationError):
    """Raised when the request lacks an input the backend cannot run without."""


class DiagnosticSeverity(Enum):
    """How serious an :class:`EstimationDiagnostic` is.

    Attributes:
        INFO: Worth recording, no action needed.
        WARNING: Something was rejected or looks wrong, and the run continued.
    """

    INFO = "info"
    WARNING = "warning"


@dataclass(frozen=True, kw_only=True)
class EstimationDiagnostic:
    """One auditable event a backend recorded while estimating.

    Attributes:
        severity: How serious the event is.
        code: Stable identifier ``"<backend_id>.<condition>"``, suitable for
            counting events across runs.
        message: Human-readable explanation with the numbers behind it.
        observation_id: Observation the event refers to, when there is one.
    """

    severity: DiagnosticSeverity
    code: str
    message: str
    observation_id: SourceObservationId | None = None


@dataclass(frozen=True, kw_only=True)
class StateEstimationRequest:
    """Canonical inputs of one State Estimation run.

    Attributes:
        trajectory_id: Identity the produced trajectory must carry.
        sequence_artifact_id: Canonical sequence the observations come from.
        selection_id: Deterministic identity of the sequence selection.
        observations: The selected canonical observations, in sequence
            order. A backend uses the modalities it needs and ignores the rest.
        calibration: Canonical calibration of the sequence, or ``None`` when
            the run has none. Backends never keep a second, hidden copy.
    """

    trajectory_id: TrajectoryId
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    observations: Sequence[SourceObservation]
    calibration: CalibrationSet | None


@dataclass(frozen=True, kw_only=True)
class StateEstimationResult:
    """What a backend produced for one request.

    Attributes:
        trajectory: The canonical trajectory.
        diagnostics: Events recorded while estimating, in the order they occurred.
        consumed_observation_count: Observations of the backend's input
            modality that it considered.
        rejected_observation_count: Consumed observations the backend
            rejected instead of publishing.
    """

    trajectory: Trajectory
    diagnostics: tuple[EstimationDiagnostic, ...]
    consumed_observation_count: int
    rejected_observation_count: int


@runtime_checkable
class StateEstimator(Protocol):
    """Capability port: turn canonical observations into a canonical trajectory."""

    def estimator_provenance(self) -> EstimatorProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance attached to every trajectory this backend produces.
        """
        ...

    def estimate(self, request: StateEstimationRequest) -> StateEstimationResult:
        """Estimate the trajectory of the body for the requested observations.

        Args:
            request: Canonical inputs of the run.

        Returns:
            The canonical trajectory and the events recorded on the way.

        Raises:
            MissingEstimatorInputError: If an input the backend requires is absent.
            StateEstimationError: If the backend cannot produce a trajectory.
        """
        ...
