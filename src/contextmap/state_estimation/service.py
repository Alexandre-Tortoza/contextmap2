"""Application service: run a State Estimation backend behind the geometry preflight.

The service is the only place that decides a run may start. It evaluates the
geometry preflight for the selected backend, stops with the full report when
the run is blocked (the estimator is never started), and otherwise runs the
backend and checks that what came back answers the request. It selects no
backend and never falls back to another one; constructing the backend is the
composition root's job.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from contextmap.state_estimation.ports import (
    StateEstimationError,
    StateEstimationRequest,
    StateEstimationResult,
    StateEstimator,
)
from contextmap.state_estimation.preflight import (
    GeometryPreflightReport,
    GeometryRequirements,
    PreflightStatus,
    PreflightTolerances,
    run_geometry_preflight,
)


class GeometryPreflightError(StateEstimationError):
    """Raised when the geometry preflight blocks a run before the estimator starts.

    Attributes:
        report: The full preflight report, including every blocker.
    """

    def __init__(self, report: GeometryPreflightReport) -> None:
        """Build the error from a blocked report.

        Args:
            report: The blocked preflight report.
        """
        summary = "; ".join(f"{item.code}: {item.message}" for item in report.blockers)
        super().__init__(f"geometry preflight blocked the run: {summary}")
        self.report = report


@dataclass(frozen=True, kw_only=True)
class StateEstimationOutcome:
    """A completed run: what was checked before it and what it produced.

    Attributes:
        preflight: The geometry preflight report; status is ``READY``.
        result: What the backend produced.
    """

    preflight: GeometryPreflightReport
    result: StateEstimationResult


def execute_state_estimation(
    estimator: StateEstimator,
    request: StateEstimationRequest,
    *,
    downstream: Sequence[GeometryRequirements] = (),
    tolerances: PreflightTolerances | None = None,
) -> StateEstimationOutcome:
    """Run a backend after the geometry preflight passes.

    Args:
        estimator: The selected backend.
        request: Canonical inputs of the run.
        downstream: Requirements of later capabilities, reported as readiness
            without blocking this run.
        tolerances: Preflight tolerances; defaults are documented on
            :class:`PreflightTolerances`.

    Returns:
        The preflight report and the backend's result.

    Raises:
        GeometryPreflightError: If the preflight blocks the run; the backend
            is not started.
        StateEstimationError: If the backend's trajectory does not answer the
            request (other identity, other frames, other provenance).
    """
    requirements = estimator.geometry_requirements()
    report = run_geometry_preflight(
        observations=request.observations,
        calibration=request.calibration,
        execution=requirements,
        downstream=downstream,
        tolerances=tolerances,
    )
    if report.status is PreflightStatus.BLOCKED:
        raise GeometryPreflightError(report)

    result = estimator.estimate(request)
    _require_result_answers_request(estimator, request, requirements, result)
    result = _with_auxiliary_lineage(request, result)
    return StateEstimationOutcome(preflight=report, result=result)


def _with_auxiliary_lineage(
    request: StateEstimationRequest, result: StateEstimationResult
) -> StateEstimationResult:
    """Copy the request's auxiliary sequence identity onto the published trajectory's provenance.

    Issue #555's auxiliary pose bridge merges an auxiliary sequence's observations into
    ``request.observations`` before any backend runs; no backend is aware of the merge or of
    which artifact the merged observations came from, so this -- not any individual backend --
    is where every backend's ``TrajectoryProvenance`` is made to name that artifact, whichever
    backend answered the request.

    Args:
        request: The request a backend just answered.
        result: What the backend returned, already checked against the request.

    Returns:
        ``result`` unchanged when ``request.auxiliary_sequence_artifact_id`` is ``None``;
        otherwise the same result with the trajectory's provenance carrying the auxiliary
        artifact and selection identity.
    """
    if request.auxiliary_sequence_artifact_id is None:
        return result
    return replace(
        result,
        trajectory=replace(
            result.trajectory,
            provenance=replace(
                result.trajectory.provenance,
                auxiliary_sequence_artifact_id=request.auxiliary_sequence_artifact_id,
                auxiliary_selection_id=request.auxiliary_selection_id,
            ),
        ),
    )


def _require_result_answers_request(
    estimator: StateEstimator,
    request: StateEstimationRequest,
    requirements: GeometryRequirements,
    result: StateEstimationResult,
) -> None:
    trajectory = result.trajectory
    provenance = trajectory.provenance
    problems = []
    if trajectory.trajectory_id != request.trajectory_id:
        problems.append(f"trajectory_id {trajectory.trajectory_id!r}")
    if provenance.sequence_artifact_id != request.sequence_artifact_id:
        problems.append(f"sequence_artifact_id {provenance.sequence_artifact_id!r}")
    if provenance.selection_id != request.selection_id:
        problems.append(f"selection_id {provenance.selection_id!r}")
    if provenance.estimator != estimator.estimator_provenance():
        problems.append("estimator provenance")
    if (trajectory.reference_frame, trajectory.body_frame) != (
        requirements.reference_frame,
        requirements.body_frame,
    ):
        problems.append(f"frames {trajectory.reference_frame!r} -> {trajectory.body_frame!r}")
    if problems:
        raise StateEstimationError(
            "the backend returned a trajectory that does not answer the request: "
            + ", ".join(problems)
        )
