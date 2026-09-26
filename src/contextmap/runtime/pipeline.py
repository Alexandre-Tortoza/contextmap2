"""The config-derived stage DAG: resolve, validate, scope and execute.

The runtime does not hardcode the canonical pipeline. A preset declares the stages, the
artifact kind each one produces and the typed inputs each one consumes, wired to their
producers; the configuration then selects which optional stages take part. From that,
:func:`resolve_plan` derives one deterministic topology, :func:`preflight` validates it
before anything heavy runs, and :func:`run_plan` executes it in dependency order.

This is a small DAG runner, not a workflow engine: no scheduler, no retries, no dynamic
discovery. A stage exists here only when it has independent scientific meaning, cost and
output; the scientific work happens in the executors, which belong to the capabilities'
composition, never to this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.runtime._files import publish_text
from contextmap.runtime.artifacts import ArtifactRef, artifact_directory
from contextmap.runtime.catalog import PRESETS, RuntimePreset
from contextmap.runtime.config import (
    ComponentConfig,
    ConfigProblem,
    EffectiveConfig,
    check_component_availability,
    check_component_selection,
)
from contextmap.runtime.errors import (
    PlanDocumentError,
    PreflightError,
    RunCancelledError,
    StageExecutionError,
)
from contextmap.runtime.lifecycle import (
    CancellationToken,
    EventEmitter,
    EventSink,
    FailureCategory,
    categorize_failure,
)
from contextmap.runtime.reuse import ReuseDecision, ReuseKey, ReusePolicy

if TYPE_CHECKING:
    from contextmap.runtime.runs import RunJournal, RunSummary
    from contextmap.runtime.selection import ResolvedSelections

PLAN_SCHEMA_VERSION = "0.1.0"
"""Version of the persisted plan and execution-record documents."""

PLAN_FILENAME = "plan.json"
EXECUTION_FILENAME = "execution.json"


@dataclass(frozen=True, kw_only=True)
class PlannedInput:
    """A typed input of a planned stage and the stage that feeds it.

    Attributes:
        name: Input name.
        contract: Artifact kind the input consumes.
        source: Stage that produces it in this plan. It can differ from the preset's base
            wiring when an optional stage was inserted in between.
        optional: Whether the stage would also run without it.
        multiple: Whether the input accepts several runs of its source stage at once.
    """

    name: str
    contract: str
    source: str
    optional: bool
    multiple: bool = False


@dataclass(frozen=True, kw_only=True)
class PlannedStage:
    """One stage of a resolved topology.

    Attributes:
        stage_id: Identity of the stage.
        capability: Owner capability package name.
        available: Whether the owner capability is implemented.
        unavailable_reason: Why the stage cannot run, when it is unavailable.
        inputs: Typed inputs, wired to their producers in this plan.
        output: Artifact kind the stage produces.
        components: Selected backend per variation point of the stage.
        component_configs: The resolved configuration of those variation points.
        config_digest: Identity of the stage's own configuration: its declared contract
            and the backend and parameters of each of its variation points. Two stages
            with the same digest are configured identically, whatever else changed.
        observation_selection: For an observation-scoped stage, the encoded selection of the
            observations it processes; ``None`` for the whole sequence and for every other
            stage. It is part of ``config_digest``.
    """

    stage_id: str
    capability: str
    available: bool
    unavailable_reason: str
    inputs: tuple[PlannedInput, ...]
    output: str | None
    components: Mapping[str, str | None]
    component_configs: Mapping[str, ComponentConfig]
    config_digest: str
    observation_selection: Mapping[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class PipelinePlan:
    """The deterministic topology one effective configuration resolves to.

    Attributes:
        preset_id: Preset the topology derives from.
        config_digest: Digest of the effective configuration it was resolved from.
        stages: Stages in execution order, or in declaration order when there is a cycle.
        order: Stage identities in execution order, or ``None`` when the graph has a cycle.
        problems: Structural problems found while resolving; preflight reports them.
    """

    preset_id: str
    config_digest: str
    stages: tuple[PlannedStage, ...]
    order: tuple[str, ...] | None
    problems: tuple[ConfigProblem, ...]

    def stage(self, stage_id: str) -> PlannedStage:
        """Return one planned stage.

        Args:
            stage_id: Identity of the stage.

        Returns:
            The planned stage.

        Raises:
            KeyError: If the stage does not take part in this plan.
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible topology, with identities and execution order."""
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "preset": self.preset_id,
            "config_digest": self.config_digest,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "capability": stage.capability,
                    "available": stage.available,
                    "components": dict(stage.components),
                    "config_digest": stage.config_digest,
                    "inputs": [
                        {
                            "name": item.name,
                            "contract": item.contract,
                            "source": item.source,
                            "optional": item.optional,
                            "multiple": item.multiple,
                        }
                        for item in stage.inputs
                    ],
                    "output": stage.output,
                    **(
                        {}
                        if stage.observation_selection is None
                        else {"observation_selection": _thaw(stage.observation_selection)}
                    ),
                }
                for stage in self.stages
            ],
        }

    @property
    def digest(self) -> str:
        """Return the digest of the topology document."""
        return _digest(self.to_document())

    def scope(
        self,
        *,
        targets: Iterable[str] | None = None,
        provided: Mapping[str, ArtifactRef] | None = None,
        selections: ResolvedSelections | None = None,
    ) -> ExecutionPlan:
        """Select what one execution runs.

        A target pulls in the stages it transitively depends on, except those whose
        artifacts are explicitly supplied: an existing immutable artifact is used instead
        of recomputed. Supply them either as ``provided`` (one exact artifact per stage) or
        as ``selections`` (the resolved run selection, which can hold several runs of a
        stage). Nothing is inferred; a supplied artifact that is not needed, or of the wrong
        kind, and every problem of the selection is reported by preflight.

        Args:
            targets: Stages to produce, or ``None`` for the complete pipeline.
            provided: Existing artifacts by the stage that produced them.
            selections: The resolved run selection, with its lineage checked.

        Returns:
            The execution scope; its problems are reported by :func:`preflight`.

        Raises:
            ValueError: If both ``provided`` and ``selections`` are given.
        """
        if provided is not None and selections is not None:
            raise ValueError("pass either provided artifacts or selections, not both")
        supplied: dict[str, tuple[ArtifactRef, ...]] = {}
        problems = list(self.problems)
        origin = "provided"
        if selections is not None:
            supplied = {stage_id: tuple(refs) for stage_id, refs in selections.provided.items()}
            problems.extend(selections.problems)
            origin = "selections"
        elif provided is not None:
            supplied = {stage_id: (ref,) for stage_id, ref in provided.items()}
        by_id = {stage.stage_id: stage for stage in self.stages}
        valid: dict[str, tuple[ArtifactRef, ...]] = {}
        for stage_id, refs in supplied.items():
            stage = by_id.get(stage_id)
            path = f"{origin}.{stage_id}"
            if stage is None:
                problems.append(
                    ConfigProblem(path=path, message=f"{stage_id!r} is not a stage of this plan")
                )
                continue
            wrong = [
                ref for ref in refs if ref.stage_id != stage_id or ref.contract != stage.output
            ]
            if wrong:
                ref = wrong[0]
                problems.append(
                    ConfigProblem(
                        path=path,
                        message=(
                            f"artifact {ref.artifact_id!r} is a {ref.contract!r} from stage "
                            f"{ref.stage_id!r}; stage {stage_id!r} produces {stage.output!r}"
                        ),
                    )
                )
            else:
                valid[stage_id] = refs
        if self.order is None:
            return ExecutionPlan(
                plan=self, stages=(), reused={}, problems=tuple(problems), selections=selections
            )

        wanted = list(self.order if targets is None else targets)
        stack = []
        for target in wanted:
            if target in by_id:
                stack.append(target)
            else:
                problems.append(
                    ConfigProblem(
                        path=f"targets.{target}",
                        message=f"{target!r} is not a stage of this plan",
                    )
                )
        need: set[str] = set()
        reused_used: set[str] = set()
        while stack:
            stage_id = stack.pop()
            if stage_id in valid:
                reused_used.add(stage_id)
                continue
            if stage_id in need:
                continue
            need.add(stage_id)
            stack.extend(item.source for item in by_id[stage_id].inputs)
        for stage_id in valid:
            if stage_id not in reused_used:
                problems.append(
                    ConfigProblem(
                        path=f"{origin}.{stage_id}",
                        message=(
                            f"{stage_id!r} is supplied but not needed by the targets, and "
                            "an explicit selection is never silently ignored"
                        ),
                    )
                )
        return ExecutionPlan(
            plan=self,
            stages=tuple(by_id[stage_id] for stage_id in self.order if stage_id in need),
            reused={
                stage_id: valid[stage_id] for stage_id in self.order if stage_id in reused_used
            },
            problems=tuple(problems),
            selections=selections,
        )


@dataclass(frozen=True, kw_only=True)
class ExecutionPlan:
    """The part of a plan one execution runs, and what it reuses.

    Attributes:
        plan: The full topology.
        stages: Stages to run, in dependency order.
        reused: Existing artifacts fed to the stages that consume them, by producing stage;
            more than one run of a stage only when the selection asked for it.
        problems: Problems found while scoping.
        selections: The resolved run selection the execution starts from, if any.
    """

    plan: PipelinePlan
    stages: tuple[PlannedStage, ...]
    reused: Mapping[str, tuple[ArtifactRef, ...]]
    problems: tuple[ConfigProblem, ...]
    selections: ResolvedSelections | None = None


@dataclass(frozen=True, kw_only=True)
class PreflightReport:
    """What preflight found before any stage ran.

    Attributes:
        problems: Everything that blocks the execution.
        stages: Stages the execution would run, in order.
        reused: Stages whose existing artifact the execution would reuse.
    """

    problems: tuple[ConfigProblem, ...]
    stages: tuple[str, ...]
    reused: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether nothing blocks the execution."""
        return not self.problems


@dataclass(frozen=True, kw_only=True)
class StageRequest:
    """What a stage executor receives.

    Attributes:
        stage_id: The stage being executed.
        inputs: The exact artifacts it consumes, by input name. Every input is a tuple of
            runs: one run for an ordinary input, several distinct runs (kept as separate
            evidence, in deterministic order) for an input that accepts them.
        components: Resolved configuration of the stage's variation points.
        config_digest: Identity of the stage's own configuration.
        output_dir: The final directory of the stage's artifact, ``<run>/<stage_id>``. The
            executor hands it, unchanged, to the capability's writer, which creates and
            finalizes it atomically: the directory does not exist yet, and an executor never
            computes a path of its own. ``None`` only when the plan runs without a journal
            (an in-memory execution), where an executor that persists cannot run.
        workspace: The workspace root the run lives in, or ``None`` without a journal. An
            executor opens an input through :meth:`directory_of`, which resolves the location of
            the artifact (possibly written by an earlier run) inside this workspace.
        observation_selection: The encoded selection of the observations an
            observation-scoped stage processes, from its plan; ``None`` for the whole sequence.
            The executor decodes it through the Ingestion, which owns the selection kinds.
    """

    stage_id: str
    inputs: Mapping[str, tuple[ArtifactRef, ...]]
    components: Mapping[str, ComponentConfig]
    config_digest: str
    output_dir: Path | None = None
    workspace: Path | None = None
    observation_selection: Mapping[str, Any] | None = None

    def identity(self) -> str:
        """Return the identity of this execution: what a writer records as its run id.

        It combines the stage, the stage's own configuration and the exact content hash of every
        input, so an identical execution gets an identical identity (and therefore identical
        content, which keeps reuse valid across runs) while any change of configuration or input
        gets another. It is a function of the request only, never of the run that asks.

        Returns:
            32 hexadecimal characters.

        Raises:
            ValueError: If an input has no content hash: it cannot take part in an identity.
        """
        parts = [self.stage_id, self.config_digest]
        for name in sorted(self.inputs):
            for ref in self.inputs[name]:
                if ref.content_hash is None:
                    raise ValueError(
                        f"input {name!r} ({ref.artifact_id!r}) has no content hash: an execution "
                        "identity cannot be derived from it"
                    )
                parts.append(f"{name}={ref.content_hash}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]

    def run_number(self) -> int:
        """Return the number of the run this request belongs to (``run-0003`` gives ``3``).

        Returns:
            The number, or ``0`` when the request has no output directory or the run is not
            named ``run-NNNN``. It is the ordinal a writer records as its ``run_index``.
        """
        if self.output_dir is None:
            return 0
        name = self.output_dir.parent.name
        return int(name.removeprefix("run-")) if name.removeprefix("run-").isdigit() else 0

    def directory_of(self, ref: ArtifactRef) -> Path:
        """Return the directory of an input artifact.

        Args:
            ref: One of the request's input handles.

        Returns:
            The directory of the artifact, wherever the run that wrote it lives.

        Raises:
            ValueError: If the request has no workspace, or the handle has no valid location.
        """
        if self.workspace is None:
            raise ValueError("this request has no workspace: run the plan with a journal")
        return artifact_directory(self.workspace, ref)


class StageExecutor(Protocol):
    """Runs one stage and returns the artifact it produced.

    The scientific work belongs to the executor's capability; the runner only orders the
    calls, hands over the exact input artifacts and records what came back.
    """

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Execute the stage for one request."""
        ...


@dataclass(frozen=True, kw_only=True)
class StageRecord:
    """What one executed stage consumed and produced.

    Attributes:
        stage_id: The stage.
        inputs: The exact artifacts it consumed, by input name (a tuple of runs each).
        output: The artifact it produced, or the prior artifact it reused.
        decision: Whether it was reused or recomputed and why, when the execution had a
            reuse policy.
    """

    stage_id: str
    inputs: Mapping[str, tuple[ArtifactRef, ...]]
    output: ArtifactRef
    decision: ReuseDecision | None = None


@dataclass(frozen=True, kw_only=True)
class ExecutionRecord:
    """The outcome of one execution: order, exact inputs and outputs, reused artifacts.

    Attributes:
        plan_digest: Digest of the topology that was executed.
        order: Stages in the order they ran.
        stages: What each stage consumed and produced.
        reused: Existing artifacts that fed the stages, by producing stage.
        selections: The resolved run selection the execution started from: the exact
            artifact ids, how each was chosen and the lineage each declared.
        resume: When the execution resumed an earlier run: which run, and which of its
            completed stages were reused and which had to be recomputed.
    """

    plan_digest: str
    order: tuple[str, ...]
    stages: tuple[StageRecord, ...]
    reused: Mapping[str, tuple[ArtifactRef, ...]]
    selections: Mapping[str, Any] | None = None
    resume: Mapping[str, Any] | None = None

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible record of the execution."""
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_digest": self.plan_digest,
            "order": list(self.order),
            "reused": {
                stage_id: [ref.to_document() for ref in refs]
                for stage_id, refs in self.reused.items()
            },
            "selections": None if self.selections is None else dict(self.selections),
            "resume": None if self.resume is None else dict(self.resume),
            "stages": [
                {
                    "stage_id": record.stage_id,
                    "inputs": {
                        name: [ref.to_document() for ref in refs]
                        for name, refs in record.inputs.items()
                    },
                    "output": record.output.to_document(),
                    "decision": None if record.decision is None else record.decision.to_document(),
                }
                for record in self.stages
            ],
        }


def resolve_plan(
    effective: EffectiveConfig, *, preset: RuntimePreset | None = None
) -> PipelinePlan:
    """Derive the topology an effective configuration selects.

    Enabled stages take part; an optional stage that declares an interception is wired
    between its producer and the consumer it intercepts, and the consumer is not edited.
    Structural problems (an unknown or disabled source, a contract mismatch, a cycle) do
    not raise: they are recorded and reported by :func:`preflight`.

    Args:
        effective: The resolved configuration.
        preset: The preset to use instead of the one the configuration names.

    Returns:
        The plan, in deterministic execution order when the graph is acyclic.
    """
    config = effective.config
    preset = preset or PRESETS[config.pipeline.preset]
    enabled = {
        stage.stage_id: config.pipeline.stages.get(stage.stage_id, stage.default_enabled)
        for stage in preset.stages
    }
    active = [stage for stage in preset.stages if enabled[stage.stage_id]]
    active_ids = {stage.stage_id for stage in active}
    known_ids = {stage.stage_id for stage in preset.stages}
    problems: list[ConfigProblem] = []

    redirect: dict[tuple[str, str], str] = {}
    for stage in active:
        if stage.intercepts is None:
            continue
        key = (stage.intercepts.consumer, stage.intercepts.input_name)
        if key[0] not in active_ids:
            continue
        if key in redirect:
            problems.append(
                ConfigProblem(
                    path=f"stages.{stage.stage_id}",
                    message=(
                        f"stages {redirect[key]!r} and {stage.stage_id!r} both intercept "
                        f"input {key[1]!r} of {key[0]!r}"
                    ),
                )
            )
        else:
            redirect[key] = stage.stage_id

    produces = {stage.stage_id: stage.output for stage in active}
    planned: list[PlannedStage] = []
    for stage in active:
        inputs: list[PlannedInput] = []
        for item in stage.inputs:
            source = redirect.get((stage.stage_id, item.name), item.source)
            path = f"stages.{stage.stage_id}.{item.name}"
            if source not in active_ids:
                if item.optional:
                    continue  # ramo opcional ausente: a entrada some, sem substituto.
                reason = "is disabled" if source in known_ids else "is not part of the preset"
                problems.append(
                    ConfigProblem(
                        path=path, message=f"needs stage {source!r} for its input, which {reason}"
                    )
                )
                continue
            if produces[source] != item.contract:
                problems.append(
                    ConfigProblem(
                        path=path,
                        message=(
                            f"consumes {item.contract!r} but stage {source!r} produces "
                            f"{produces[source]!r}"
                        ),
                    )
                )
            inputs.append(
                PlannedInput(
                    name=item.name,
                    contract=item.contract,
                    source=source,
                    optional=item.optional,
                    multiple=item.multiple,
                )
            )
        component_configs = {
            component_id: config.components[component_id]
            for component_id in stage.components
            if component_id in config.components
        }
        selection = config.inputs.observation_selection if stage.observation_scoped else None
        planned.append(
            PlannedStage(
                stage_id=stage.stage_id,
                capability=stage.capability,
                available=stage.available,
                unavailable_reason=stage.unavailable_reason,
                inputs=tuple(inputs),
                output=stage.output,
                components={cid: cfg.backend for cid, cfg in component_configs.items()},
                component_configs=component_configs,
                config_digest=_stage_digest(
                    stage.stage_id,
                    stage.capability,
                    inputs,
                    stage.output,
                    component_configs,
                    selection,
                ),
                observation_selection=selection,
            )
        )

    order, cycle = _topological_order(planned)
    if cycle is not None:
        problems.append(
            ConfigProblem(path="stages", message=f"the topology has a cycle: {' -> '.join(cycle)}")
        )
    by_id = {stage.stage_id: stage for stage in planned}
    stages = tuple(by_id[stage_id] for stage_id in order) if order is not None else tuple(planned)
    return PipelinePlan(
        preset_id=preset.preset_id,
        config_digest=effective.digest,
        stages=stages,
        order=tuple(order) if order is not None else None,
        problems=tuple(problems),
    )


def preflight(
    execution: ExecutionPlan,
    *,
    executors: Mapping[str, StageExecutor] | None = None,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    provided_runtimes: Collection[str] = (),
    reuse: ReusePolicy | None = None,
) -> PreflightReport:
    """Validate an execution before any stage runs or any model loads.

    Checks the structure found while resolving and scoping (cycles, missing
    dependencies, contract mismatches, wrong or unneeded provided artifacts), that every
    stage to run is implemented, that each of its variation points has a backend and that
    the backend's optional modules and secrets are present, and, when executors are
    given, that each stage has one. Nothing is imported or loaded.

    With a reuse policy, a stage that will certainly be reused (its exact inputs are
    known and an identical, still valid artifact is indexed) needs neither an executor
    nor its optional modules and secrets: nothing of it will run. Its configuration must
    still be complete. A forced stage that is not part of the execution is a problem.

    Args:
        execution: The scoped execution.
        executors: The executors that would run, to check coverage.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
        provided_runtimes: Component identities whose model runtime the caller supplies,
            so their bundled modules are not required.
        reuse: The reuse policy of the execution, if any.

    Returns:
        Every problem found, all at once.
    """
    problems = list(execution.problems)
    predicted = {} if reuse is None else predict_reuse(execution, reuse)
    if reuse is not None:
        in_scope = {stage.stage_id for stage in execution.stages}
        for stage_id in sorted(reuse.force_recompute - in_scope):
            problems.append(
                ConfigProblem(
                    path=f"reuse.force_recompute.{stage_id}",
                    message=f"{stage_id!r} is not a stage of this execution",
                )
            )
    for stage in execution.stages:
        if not stage.available:
            problems.append(
                ConfigProblem(path=f"stages.{stage.stage_id}", message=stage.unavailable_reason)
            )
            continue
        will_run = stage.stage_id not in predicted or predicted[stage.stage_id].kind != "reused"
        for component_id, component in stage.component_configs.items():
            missing = check_component_selection(component_id, component)
            if missing is not None:
                problems.append(missing)
                continue
            if will_run:
                problems.extend(
                    check_component_availability(
                        component_id,
                        component,
                        environ=environ,
                        module_available=module_available,
                        check_modules=component_id not in provided_runtimes,
                    )
                )
        if will_run and executors is not None and stage.stage_id not in executors:
            problems.append(
                ConfigProblem(
                    path=f"stages.{stage.stage_id}", message="no executor is registered for it"
                )
            )
    return PreflightReport(
        problems=tuple(problems),
        stages=tuple(stage.stage_id for stage in execution.stages),
        reused=tuple(execution.reused),
    )


def predict_reuse(execution: ExecutionPlan, reuse: ReusePolicy) -> dict[str, ReuseDecision]:
    """Predict, without running anything, which stages would be reused.

    Stages are visited in dependency order and looked up in the index. The prediction is
    conservative: a stage downstream of one that will be recomputed is reported as
    recomputed, because its inputs are not known yet, although the run may still reuse it
    when the recomputation reproduces the same content. The execution record is the
    authority.

    Args:
        execution: The scoped execution.
        reuse: The reuse policy.

    Returns:
        A decision per stage of the execution.
    """
    known: dict[str, tuple[ArtifactRef, ...] | None] = dict(execution.reused)
    decisions: dict[str, ReuseDecision] = {}
    for stage in execution.stages:
        inputs: dict[str, tuple[ArtifactRef, ...]] = {}
        unknown: set[str] = set()
        for item in stage.inputs:
            refs = known.get(item.source)
            if refs is None:
                unknown.add(item.source)
            else:
                inputs[item.name] = refs
        if unknown:
            decisions[stage.stage_id] = ReuseDecision(
                kind="recomputed",
                reason=(
                    f"upstream stage(s) {', '.join(sorted(unknown))} will be recomputed, so this "
                    "stage's inputs are not known yet"
                ),
            )
            known[stage.stage_id] = None
            continue
        decision, hit, _ = _decide(stage, inputs, reuse)
        decisions[stage.stage_id] = decision
        known[stage.stage_id] = None if hit is None else (hit,)
    return decisions


def _decide(
    stage: PlannedStage, inputs: Mapping[str, tuple[ArtifactRef, ...]], reuse: ReusePolicy
) -> tuple[ReuseDecision, ArtifactRef | None, ReuseKey | None]:
    """Decide between reusing an indexed artifact and recomputing one stage.

    Returns:
        The decision, the artifact to reuse (or ``None``) and the reuse key (or ``None``
        when the inputs cannot be keyed).
    """
    hashes: dict[str, tuple[str, str]] = {}
    missing = []
    for name, refs in sorted(inputs.items()):
        content_hashes = [ref.content_hash for ref in refs]
        if any(content_hash is None for content_hash in content_hashes):
            missing.append(name)
        else:
            hashes[name] = (refs[0].contract, _combined_hash(content_hashes))
    if missing:
        reason = (
            f"input {', '.join(missing)} has no content hash, so its identity cannot be checked"
        )
        return ReuseDecision(kind="recomputed", reason=reason), None, None
    key = ReuseKey(
        stage_id=stage.stage_id,
        contract=stage.output or "",
        stage_config_digest=stage.config_digest,
        inputs=hashes,
        code_identity=reuse.code_identity,
        identities=dict(reuse.identities.get(stage.stage_id, {})),
    )
    if stage.stage_id in reuse.force_recompute:
        decision = ReuseDecision(
            kind="recomputed", reason="forced recomputation requested", key_digest=key.digest
        )
        return decision, None, key
    found = reuse.store.find(key)
    if found.artifact is None:
        decision = ReuseDecision(kind="recomputed", reason=found.reason, key_digest=key.digest)
        return decision, None, key
    decision = ReuseDecision(
        kind="reused", reason=found.reason, key_digest=key.digest, reused_from=found.artifact
    )
    return decision, found.artifact, key


def run_plan(
    execution: ExecutionPlan,
    executors: Mapping[str, StageExecutor],
    *,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    provided_runtimes: Collection[str] = (),
    provider_overrides: Sequence[str] = (),
    reuse: ReusePolicy | None = None,
    journal: RunJournal | None = None,
    events: EventSink | None = None,
    cancellation: CancellationToken | None = None,
    redact: Callable[[str], str] | None = None,
    clock: Callable[[], str] | None = None,
    resume_from: RunSummary | None = None,
) -> ExecutionRecord:
    """Execute a scoped plan in dependency order.

    Preflight runs first and blocks everything when it finds a problem. Each stage then
    receives the exact artifacts of the stages that feed it, reused or just produced. The
    first failure, or an output that contradicts the stage's declared contract, stops the
    run: nothing later runs, nothing is retried and nothing is substituted.

    With a reuse policy, each stage is decided in dependency order against the index of
    completed artifacts: identical identity reuses the exact prior artifact without
    running the stage; anything else runs it and, once it completes, indexes its output.
    A stage that fails leaves nothing indexed. Every decision is recorded with its reason.

    Every step emits a structured event (planned, blocked, started, stage started, reused,
    completed or failed, cancelled, completed) to the journal and to ``events``. The events
    carry no secret: every string in them goes through ``redact``.

    Args:
        execution: The scoped execution.
        executors: One executor per stage to run.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
        provided_runtimes: Component identities whose model runtime the caller supplies.
        provider_overrides: Component identities whose caller-supplied provider won over a
            ``resources.providers`` target the effective configuration also declared for it
            (see ``contextmap.runtime.composition.compose``'s ``on_provider_override``);
            recorded on the ``run_planned`` event so the override is part of the run's own
            trail, not just silently applied. Empty on every ordinary run.
        reuse: How to decide between reusing and recomputing, or ``None`` to always run.
        journal: Persists the run's lifecycle, status and execution record. Its directory is
            the run root: each stage receives ``<run>/<stage_id>`` as its output directory.
        events: An extra receiver of the run's events.
        cancellation: Cooperative cancellation, checked before each stage.
        redact: Replaces secret values inside a string.
        clock: Returns event timestamps; defaults to the current UTC time.
        resume_from: The run this execution resumes; :func:`resume_plan` validates it.

    Returns:
        The execution record: order, exact inputs and outputs, decisions and reused
        artifacts.

    Raises:
        ValueError: If ``resume_from`` is given without a reuse policy.
        PreflightError: If preflight found problems; nothing was executed.
        StageExecutionError: If a stage failed or returned an artifact of another kind.
        RunCancelledError: If cancellation was requested before a stage started.
    """
    if resume_from is not None and reuse is None:
        raise ValueError("resuming a run needs a reuse policy: completed stages are reused")
    sinks = [sink for sink in (journal, events) if sink is not None]
    emitter = EventEmitter(sinks, clock=clock, redact=redact)
    emitter.emit(
        "run_planned",
        plan_digest=execution.plan.digest,
        config_digest=execution.plan.config_digest,
        stages=[stage.stage_id for stage in execution.stages],
        provided={
            stage: [ref.artifact_id for ref in refs] for stage, refs in execution.reused.items()
        },
        provider_overrides=list(provider_overrides),
    )
    if resume_from is not None:
        emitter.emit(
            "run_resumed",
            resumed_from=resume_from.run_id,
            previous_status=resume_from.status.value,
            previously_completed=list(resume_from.completed_stages),
        )
    started = time.monotonic()
    progress: list[str] = []
    try:
        report = preflight(
            execution,
            executors=executors,
            environ=environ,
            module_available=module_available,
            provided_runtimes=provided_runtimes,
            reuse=reuse,
        )
        if not report.ok:
            emitter.emit(
                "run_blocked",
                problems=[{"path": p.path, "message": p.message} for p in report.problems],
            )
            raise PreflightError(report)
        emitter.emit("run_started")
        records = _run_stages(
            execution,
            executors,
            reuse,
            emitter,
            cancellation,
            redact,
            progress,
            None if journal is None else journal.directory,
        )
    except (PreflightError, StageExecutionError, RunCancelledError):
        raise
    except KeyboardInterrupt:
        if not emitter.terminal:
            emitter.emit("run_cancelled", reason="interrupted", completed=list(progress))
        raise
    except Exception as error:
        if not emitter.terminal:
            emitter.emit(
                "run_failed",
                category=FailureCategory.UNEXPECTED.value,
                message=str(error),
                exception_type=type(error).__name__,
                completed=list(progress),
            )
        raise
    record = ExecutionRecord(
        plan_digest=execution.plan.digest,
        order=tuple(record.stage_id for record in records),
        stages=tuple(records),
        reused=dict(execution.reused),
        selections=None if execution.selections is None else execution.selections.to_document(),
        resume=_resume_report(resume_from, records),
    )
    if journal is not None:
        journal.record_execution(record)
    emitter.emit(
        "run_completed",
        order=list(record.order),
        elapsed_s=round(time.monotonic() - started, 6),
        resume=None if record.resume is None else dict(record.resume),
    )
    return record


def _run_stages(
    execution: ExecutionPlan,
    executors: Mapping[str, StageExecutor],
    reuse: ReusePolicy | None,
    emitter: EventEmitter,
    cancellation: CancellationToken | None,
    redact: Callable[[str], str] | None,
    progress: list[str],
    run_directory: Path | None,
) -> list[StageRecord]:
    """Run the stages in order, emitting an event for each step and stopping at a failure."""
    outputs: dict[str, tuple[ArtifactRef, ...]] = dict(execution.reused)
    records: list[StageRecord] = []
    for stage in execution.stages:
        if cancellation is not None and cancellation.cancelled:
            emitter.emit(
                "run_cancelled",
                stage_id=stage.stage_id,
                reason=cancellation.reason,
                completed=list(progress),
            )
            raise RunCancelledError(stage.stage_id, progress, cancellation.reason)
        inputs = {item.name: outputs[item.source] for item in stage.inputs}
        decision: ReuseDecision | None = None
        key: ReuseKey | None = None
        produced: ArtifactRef | None = None
        if reuse is not None:
            decision, produced, key = _decide(stage, inputs, reuse)
        if produced is not None:
            emitter.emit(
                "stage_reused",
                stage_id=stage.stage_id,
                artifact=produced.to_document(),
                decision=None if decision is None else decision.to_document(),
            )
        else:
            executor = executors.get(stage.stage_id)
            if executor is None:
                raise _stage_failed(
                    emitter,
                    stage.stage_id,
                    progress,
                    redact,
                    category=FailureCategory.EXECUTION.value,
                    exception_type="LookupError",
                    message="no executor is registered",
                )
            emitter.emit(
                "stage_started",
                stage_id=stage.stage_id,
                inputs={name: [ref.artifact_id for ref in refs] for name, refs in inputs.items()},
                config_digest=stage.config_digest,
            )
            began = time.monotonic()
            request = StageRequest(
                stage_id=stage.stage_id,
                inputs=inputs,
                components=stage.component_configs,
                config_digest=stage.config_digest,
                output_dir=None if run_directory is None else run_directory / stage.stage_id,
                workspace=None if run_directory is None else run_directory.parent.parent,
                observation_selection=stage.observation_selection,
            )
            try:
                produced = executor.execute(request)
            except Exception as error:
                raise _stage_failed(
                    emitter,
                    stage.stage_id,
                    progress,
                    redact,
                    category=categorize_failure(error),
                    exception_type=type(error).__name__,
                    message=str(error),
                    elapsed=time.monotonic() - began,
                ) from error
            expected = (
                None
                if request.output_dir is None or request.workspace is None
                else request.output_dir.relative_to(request.workspace).as_posix()
            )
            if expected is not None and produced.location not in (None, expected):
                raise _stage_failed(
                    emitter,
                    stage.stage_id,
                    progress,
                    redact,
                    category=FailureCategory.CONTRACT.value,
                    exception_type="ContractViolation",
                    message=(
                        f"reported the location {produced.location!r}; the stage was given "
                        f"{expected!r}"
                    ),
                    elapsed=time.monotonic() - began,
                )
            if produced.contract != stage.output or produced.stage_id != stage.stage_id:
                raise _stage_failed(
                    emitter,
                    stage.stage_id,
                    progress,
                    redact,
                    category=FailureCategory.CONTRACT.value,
                    exception_type="ContractViolation",
                    message=(
                        f"returned a {produced.contract!r} artifact from stage "
                        f"{produced.stage_id!r}; the stage declares {stage.output!r}"
                    ),
                    elapsed=time.monotonic() - began,
                )
            if reuse is not None and decision is not None and key is not None:
                decision = _index(reuse, key, produced, decision)
            emitter.emit(
                "stage_completed",
                stage_id=stage.stage_id,
                artifact=produced.to_document(),
                decision=None if decision is None else decision.to_document(),
                elapsed_s=round(time.monotonic() - began, 6),
            )
        outputs[stage.stage_id] = (produced,)
        progress.append(stage.stage_id)
        records.append(
            StageRecord(stage_id=stage.stage_id, inputs=inputs, output=produced, decision=decision)
        )
    return records


def _stage_failed(
    emitter: EventEmitter,
    stage_id: str,
    progress: Sequence[str],
    redact: Callable[[str], str] | None,
    *,
    category: str,
    exception_type: str,
    message: str,
    elapsed: float | None = None,
) -> StageExecutionError:
    """Record a stage failure as events and build the error that stops the run."""
    clean = message if redact is None else redact(message)
    emitter.emit(
        "stage_failed",
        stage_id=stage_id,
        category=category,
        message=clean,
        exception_type=exception_type,
        elapsed_s=None if elapsed is None else round(elapsed, 6),
    )
    emitter.emit(
        "run_failed",
        stage_id=stage_id,
        category=category,
        message=clean,
        exception_type=exception_type,
        completed=list(progress),
    )
    return StageExecutionError(stage_id, list(progress), clean)


def _resume_report(
    resume_from: RunSummary | None, records: Sequence[StageRecord]
) -> dict[str, Any] | None:
    """Say which stages the resumed run had completed, and how this run treated each."""
    if resume_from is None:
        return None
    by_stage = {record.stage_id: record for record in records}
    reused, recomputed = [], []
    for stage_id in resume_from.completed_stages:
        record = by_stage.get(stage_id)
        if record is None:
            continue  # o estágio não faz parte desta execução (foi fornecido)
        if record.decision is not None and record.decision.kind == "reused":
            reused.append(stage_id)
        else:
            recomputed.append(stage_id)
    return {
        "from": resume_from.run_id,
        "previous_status": resume_from.status.value,
        "reused": reused,
        "recomputed": recomputed,
    }


def _combined_hash(content_hashes: Sequence[str | None]) -> str:
    """Identify a set of input runs by their content.

    One run is identified by its own hash, so a single-run key does not change. Several
    runs are an unordered set of evidence: their hashes are sorted before being combined.
    """
    hashes = sorted(h for h in content_hashes if h is not None)
    if len(hashes) == 1:
        return hashes[0]
    return "sha256:" + hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


def _index(
    reuse: ReusePolicy, key: ReuseKey, produced: ArtifactRef, decision: ReuseDecision
) -> ReuseDecision:
    """Index a completed output and say in the decision if it could not become the entry."""
    if produced.content_hash is None:
        return replace(
            decision, reason=f"{decision.reason}; the output has no content hash and is not indexed"
        )
    if not reuse.store.record(key, produced):
        return replace(
            decision,
            reason=f"{decision.reason}; an earlier artifact of this identity stays indexed",
        )
    return decision


def write_plan(plan: PipelinePlan, directory: str | os.PathLike[str]) -> Path:
    """Persist the resolved topology as ``plan.json``.

    The write is atomic and never replaces an existing file: publishing the same plan
    again is a no-op, a different one is refused.

    Args:
        plan: A plan without structural problems.
        directory: Directory that receives the file; created if missing.

    Returns:
        Path of the persisted file.

    Raises:
        PlanDocumentError: If the plan has problems or the directory holds another plan.
    """
    if plan.problems:
        raise PlanDocumentError(
            "a plan with structural problems is never persisted: "
            + "; ".join(str(problem) for problem in plan.problems)
        )
    return _publish(Path(directory), PLAN_FILENAME, _document_text(plan.to_document(), plan.digest))


def write_execution_record(record: ExecutionRecord, directory: str | os.PathLike[str]) -> Path:
    """Persist an execution record as ``execution.json``, atomically and once.

    Args:
        record: The record to persist.
        directory: Directory that receives the file; created if missing.

    Returns:
        Path of the persisted file.

    Raises:
        PlanDocumentError: If the directory already holds a different record.
    """
    document = record.to_document()
    return _publish(
        Path(directory), EXECUTION_FILENAME, _document_text(document, _digest(document))
    )


def read_plan_document(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a persisted plan or execution record and verify its digest.

    Args:
        path: Path of ``plan.json`` or ``execution.json``.

    Returns:
        The document.

    Raises:
        PlanDocumentError: If the file is unreadable, follows another schema version or
            was altered after it was written.
    """
    source = Path(path)
    try:
        stored = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PlanDocumentError(f"cannot read {source}: {error}") from error
    if not isinstance(stored, dict) or not {"digest", "document"} <= stored.keys():
        raise PlanDocumentError(f"{source} is not a persisted plan document")
    document = stored["document"]
    if document.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanDocumentError(
            f"{source} follows schema_version {document.get('schema_version')!r}, this runtime "
            f"reads {PLAN_SCHEMA_VERSION!r}"
        )
    if _digest(document) != stored["digest"]:
        raise PlanDocumentError(f"{source} does not match its digest; it was altered")
    result: dict[str, Any] = document
    return result


def _publish(directory: Path, filename: str, text: str) -> Path:
    try:
        return publish_text(directory, filename, text)
    except FileExistsError:
        final = directory / filename
        if final.read_text(encoding="utf-8") == text:
            return final
        raise PlanDocumentError(
            f"{final} already holds a different document; a published run is never rewritten"
        ) from None


def _document_text(document: Mapping[str, Any], digest: str) -> str:
    return json.dumps({"digest": digest, "document": document}, indent=2, sort_keys=True) + "\n"


def _topological_order(stages: Sequence[PlannedStage]) -> tuple[list[str] | None, list[str] | None]:
    """Order stages so each follows its producers, keeping declaration order on ties.

    Returns:
        ``(order, None)`` when the graph is acyclic, otherwise ``(None, cycle)`` with one
        cycle as a list of stage identities that starts and ends on the same stage.
    """
    dependencies = {stage.stage_id: {item.source for item in stage.inputs} for stage in stages}
    done: list[str] = []
    remaining = [stage.stage_id for stage in stages]
    while remaining:
        ready = next(
            (stage_id for stage_id in remaining if dependencies[stage_id] <= set(done)), None
        )
        if ready is None:
            return None, _find_cycle(remaining, dependencies)
        done.append(ready)
        remaining.remove(ready)
    return done, None


def _find_cycle(remaining: Sequence[str], dependencies: Mapping[str, set[str]]) -> list[str]:
    """Walk producer edges from any stuck stage until a stage repeats."""
    stuck = set(remaining)
    current = remaining[0]
    path: list[str] = []
    while current not in path:
        path.append(current)
        current = next(source for source in sorted(dependencies[current]) if source in stuck)
    return [*path[path.index(current) :], current]


def _stage_digest(
    stage_id: str,
    capability: str,
    inputs: Sequence[PlannedInput],
    output: str | None,
    components: Mapping[str, ComponentConfig],
    observation_selection: Mapping[str, Any] | None,
) -> str:
    """Identify a stage's own configuration, not its position in the topology.

    The observation selection enters only when there is one, so the identity of a stage over
    the whole sequence is the one it had before selections existed.
    """
    document: dict[str, Any] = {
        "stage_id": stage_id,
        "capability": capability,
        "contract": {"inputs": {item.name: item.contract for item in inputs}, "output": output},
        "components": {
            component_id: {
                "backend": component.backend,
                "parameters": _thaw(component.parameters),
            }
            for component_id, component in components.items()
        },
    }
    if observation_selection is not None:
        document["observation_selection"] = _thaw(observation_selection)
    return _digest(document)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _digest(document: object) -> str:
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
