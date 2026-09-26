"""Backend-agnostic perception orchestration service.

Executes a resolved stage graph — not a hardcoded sequence of model calls
— and assembles one :class:`~contextmap.visual_perception.models.PerceptionResult`
per processed source observation. This module never constructs a concrete
model backend and never imports a model SDK: it only calls the
:class:`~contextmap.visual_perception.ports.StageRunner` callables it is
given, which close over whichever capability port implementation the
composition root selected.

v0 accepts an already-resolved :class:`StageDefinition` sequence built by
the caller. The declarative preset/config compiler that produces such a
graph from a versioned pipeline configuration is
:mod:`contextmap.visual_perception.pipeline` (a separate issue in this
milestone); this module only executes a graph, however it was produced.

Stage failures are isolated: a failing stage never crashes the run or
other independent branches. Any stage that depends, directly or
transitively, on a stage that did not succeed is marked ``SKIPPED``
rather than executed, so unrelated branches (e.g. dense feature
extraction when region discovery fails) still produce their evidence —
see ``src/contextmap/visual_perception/docs/service.md`` for the failure
policy and worked examples.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import (
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    Region2D,
    SceneContext,
    SemanticClaim,
    SemanticScore,
    VisualFeature,
)
from contextmap.visual_perception.semantic_backend import SemanticInterpretationExecution

STAGE_TRACEBACK_MAX_CHARS = 4000
"""Characters of a failed stage's traceback kept in its :class:`StageOutcome`.

A run records one outcome per stage per frame, so a failure repeated over thousands of frames must
not grow the persisted metrics without bound. The innermost part is kept: it holds the frame that
raised and the exception itself.
"""

StageRunner = Callable[[Mapping[str, object]], object]
"""A stage's execution logic.

Receives a mapping of already-completed upstream stage_id -> that
stage's output, and returns this stage's own output. The orchestration
service treats this value opaquely; only the caller assembling the
canonical pipeline needs to know its real type at each stage_id.
"""


class StageGraphError(Exception):
    """Raised when a stage graph is structurally invalid."""


class StageStatus(Enum):
    """The outcome of attempting to execute one stage."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, kw_only=True)
class StageDefinition:
    """One node in a resolved perception stage graph.

    Attributes:
        stage_id: Unique identity of this stage within the graph.
        capability: Capability port this stage exercises, e.g.
            ``"region_discovery"``. Used for diagnostics/provenance only
            — the executor calls ``run``, never a capability-specific
            method.
        run: This stage's execution logic.
        depends_on: ``stage_id``s that must succeed before this stage
            runs.
    """

    stage_id: str
    capability: str
    run: StageRunner
    depends_on: frozenset[str] = frozenset()


@dataclass(frozen=True, kw_only=True)
class StageOutcome:
    """The result of attempting to execute one stage.

    Attributes:
        stage_id: Which stage this outcome is for.
        status: Whether the stage succeeded, failed, or was skipped
            because an upstream dependency did not succeed.
        output: The stage's produced value. ``None`` unless ``status`` is
            ``SUCCEEDED``.
        error: Human-readable explanation. Set when ``status`` is
            ``FAILED`` (the raised exception's message, which may be empty)
            or ``SKIPPED`` (which dependency did not succeed).
        duration_ms: Wall-clock duration of this stage's own execution.
            ``0.0`` for a skipped stage.
        error_type: The raised exception's type name. Set only when
            ``status`` is ``FAILED``.
        error_traceback: The raised exception's formatted traceback, keeping
            only its last :data:`STAGE_TRACEBACK_MAX_CHARS` characters behind a
            ``[traceback truncated: ...]`` line when it is longer. Set only when
            ``status`` is ``FAILED``.
    """

    stage_id: str
    status: StageStatus
    output: object | None = None
    error: str | None = None
    duration_ms: float = 0.0
    error_type: str | None = None
    error_traceback: str | None = None


def execute_stage_graph(stages: Sequence[StageDefinition]) -> Sequence[StageOutcome]:
    """Execute a resolved stage graph, isolating failures per independent branch.

    Args:
        stages: The stages to execute. Order does not need to already be
            dependency-sorted; this function computes a deterministic
            topological order itself.

    Returns:
        One :class:`StageOutcome` per stage, in the deterministic
        execution order this function computed.

    Raises:
        StageGraphError: If ``stages`` has a duplicate ``stage_id``, a
            dependency on an unknown ``stage_id``, or a cycle.
    """
    ordered = _topological_order(stages)
    outcomes: dict[str, StageOutcome] = {}
    context: dict[str, object] = {}

    for stage in ordered:
        unmet = sorted(
            dep for dep in stage.depends_on if outcomes[dep].status is not StageStatus.SUCCEEDED
        )
        if unmet:
            outcomes[stage.stage_id] = StageOutcome(
                stage_id=stage.stage_id,
                status=StageStatus.SKIPPED,
                error=f"upstream stage(s) did not succeed: {unmet}",
            )
            continue

        start = time.monotonic()
        try:
            output = stage.run(context)
        except Exception as error:
            # A stage's own failure must never crash the run or sibling
            # branches — only stages that depend on this one are skipped.
            outcomes[stage.stage_id] = StageOutcome(
                stage_id=stage.stage_id,
                status=StageStatus.FAILED,
                error=str(error),
                duration_ms=(time.monotonic() - start) * 1000,
                error_type=type(error).__name__,
                error_traceback=_truncated_traceback(error),
            )
            continue

        context[stage.stage_id] = output
        outcomes[stage.stage_id] = StageOutcome(
            stage_id=stage.stage_id,
            status=StageStatus.SUCCEEDED,
            output=output,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    return tuple(outcomes[stage.stage_id] for stage in ordered)


def _truncated_traceback(error: BaseException) -> str:
    """Format ``error``'s traceback, keeping its innermost :data:`STAGE_TRACEBACK_MAX_CHARS`."""
    formatted = "".join(traceback.format_exception(error))
    if len(formatted) <= STAGE_TRACEBACK_MAX_CHARS:
        return formatted
    dropped = len(formatted) - STAGE_TRACEBACK_MAX_CHARS
    return (
        f"[traceback truncated: {dropped} leading characters dropped]\n"
        f"{formatted[-STAGE_TRACEBACK_MAX_CHARS:]}"
    )


def assemble_perception_result(
    *,
    result_id: PerceptionResultId,
    source_observation_id: SourceObservationId,
    run_id: PerceptionRunId,
    sequence_artifact_id: str,
    created_at: str,
    outcomes: Sequence[StageOutcome],
    region_stage_id: str | None = None,
    feature_stage_ids: Sequence[str] = (),
    claim_stage_ids: Sequence[str] = (),
    scene_context_stage_id: str | None = None,
    semantic_execution_stage_ids: Sequence[str] = (),
    semantic_score_stage_ids: Sequence[str] = (),
) -> PerceptionResult:
    """Assemble a PerceptionResult from succeeded stage outcomes.

    A stage that failed or was skipped simply contributes nothing to the
    result — this is what preserves partial evidence from independent
    branches instead of failing the whole result.

    Args:
        result_id: Identity to assign to the assembled result.
        source_observation_id: The physical observation processed.
        run_id: The run this result belongs to.
        sequence_artifact_id: The sequence the observation belongs to.
        created_at: ISO 8601 UTC timestamp for the result.
        outcomes: Stage outcomes, as returned by :func:`execute_stage_graph`.
        region_stage_id: ``stage_id`` whose output is
            ``Sequence[Region2D]``, when region discovery was part of
            this graph.
        feature_stage_ids: ``stage_id``s whose output is
            ``Sequence[VisualFeature]``.
        claim_stage_ids: ``stage_id``s whose output is
            ``Sequence[SemanticClaim]``.
        scene_context_stage_id: ``stage_id`` whose output is
            ``SceneContext | None``.
        semantic_execution_stage_ids: ``stage_id``s whose output is a
            :class:`SemanticInterpretationExecution`. Their parsed claims
            and scene context are materialized into the canonical result.
        semantic_score_stage_ids: ``stage_id``s whose output is
            ``Sequence[SemanticScore]``.

    Returns:
        The assembled result.
    """
    by_id = {outcome.stage_id: outcome for outcome in outcomes}

    regions: Sequence[Region2D] = ()
    if region_stage_id is not None:
        outcome = by_id.get(region_stage_id)
        if outcome is not None and outcome.status is StageStatus.SUCCEEDED:
            regions = outcome.output  # type: ignore[assignment]

    features: list[VisualFeature] = []
    for stage_id in feature_stage_ids:
        outcome = by_id.get(stage_id)
        if outcome is not None and outcome.status is StageStatus.SUCCEEDED:
            features.extend(outcome.output)  # type: ignore[arg-type]

    claims: list[SemanticClaim] = []
    for stage_id in claim_stage_ids:
        outcome = by_id.get(stage_id)
        if outcome is not None and outcome.status is StageStatus.SUCCEEDED:
            claims.extend(outcome.output)  # type: ignore[arg-type]

    scene_context: SceneContext | None = None
    if scene_context_stage_id is not None:
        outcome = by_id.get(scene_context_stage_id)
        if outcome is not None and outcome.status is StageStatus.SUCCEEDED:
            scene_context = outcome.output  # type: ignore[assignment]

    for stage_id in semantic_execution_stage_ids:
        outcome = by_id.get(stage_id)
        if outcome is None or outcome.status is not StageStatus.SUCCEEDED:
            continue
        execution = outcome.output
        if not isinstance(execution, SemanticInterpretationExecution):
            raise TypeError(
                f"semantic execution stage {stage_id!r} did not produce "
                "SemanticInterpretationExecution"
            )
        if execution.request.source_observation_id != source_observation_id:
            raise ValueError(
                f"semantic execution stage {stage_id!r} belongs to another source observation"
            )
        if execution.request.perception_result_id != result_id:
            raise ValueError(
                f"semantic execution stage {stage_id!r} belongs to another perception result"
            )
        claims.extend(execution.parsed.claims)
        parsed_scene_context = execution.parsed.scene_context
        if parsed_scene_context is not None:
            if scene_context is not None:
                raise ValueError("multiple scene contexts cannot be assembled into one result")
            scene_context = parsed_scene_context

    semantic_scores: list[SemanticScore] = []
    for stage_id in semantic_score_stage_ids:
        outcome = by_id.get(stage_id)
        if outcome is not None and outcome.status is StageStatus.SUCCEEDED:
            semantic_scores.extend(outcome.output)  # type: ignore[arg-type]

    return PerceptionResult(
        result_id=result_id,
        source_observation_id=source_observation_id,
        run_id=run_id,
        sequence_artifact_id=sequence_artifact_id,
        created_at=created_at,
        regions=regions,
        features=tuple(features),
        claims=tuple(claims),
        scene_context=scene_context,
        semantic_scores=tuple(semantic_scores),
    )


def _topological_order(stages: Sequence[StageDefinition]) -> list[StageDefinition]:
    by_id = {stage.stage_id: stage for stage in stages}
    if len(by_id) != len(stages):
        raise StageGraphError("duplicate stage_id in graph")

    for stage in stages:
        unknown = stage.depends_on - by_id.keys()
        if unknown:
            raise StageGraphError(
                f"stage {stage.stage_id!r} depends on unknown stages: {sorted(unknown)}"
            )

    resolved: list[StageDefinition] = []
    resolved_ids: set[str] = set()
    remaining = {stage.stage_id: stage for stage in stages}
    while remaining:
        ready = sorted(
            (stage for stage in remaining.values() if stage.depends_on <= resolved_ids),
            key=lambda stage: stage.stage_id,
        )
        if not ready:
            raise StageGraphError(f"cycle detected among stages: {sorted(remaining.keys())}")
        for stage in ready:
            resolved.append(stage)
            resolved_ids.add(stage.stage_id)
            del remaining[stage.stage_id]

    return resolved
