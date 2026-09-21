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
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from contextmap.runtime._files import publish_text
from contextmap.runtime.catalog import PRESETS, RuntimePreset
from contextmap.runtime.config import (
    ComponentConfig,
    ConfigProblem,
    EffectiveConfig,
    check_component_availability,
    check_component_selection,
)
from contextmap.runtime.errors import PlanDocumentError, PreflightError, StageExecutionError

PLAN_SCHEMA_VERSION = "0.1.0"
"""Version of the persisted plan and execution-record documents."""

PLAN_FILENAME = "plan.json"
EXECUTION_FILENAME = "execution.json"


@dataclass(frozen=True, kw_only=True)
class ArtifactRef:
    """A handle to one immutable stage artifact.

    Attributes:
        stage_id: Stage that produced it.
        contract: Artifact kind, for example ``"SequenceArtifact"``.
        artifact_id: Identity of the exact artifact or run, never a directory name.
    """

    stage_id: str
    contract: str
    artifact_id: str

    def to_document(self) -> dict[str, str]:
        """Return the JSON-compatible form persisted in execution records."""
        return {
            "stage_id": self.stage_id,
            "contract": self.contract,
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True, kw_only=True)
class PlannedInput:
    """A typed input of a planned stage and the stage that feeds it.

    Attributes:
        name: Input name.
        contract: Artifact kind the input consumes.
        source: Stage that produces it in this plan. It can differ from the preset's base
            wiring when an optional stage was inserted in between.
        optional: Whether the stage would also run without it.
    """

    name: str
    contract: str
    source: str
    optional: bool


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
                        }
                        for item in stage.inputs
                    ],
                    "output": stage.output,
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
    ) -> ExecutionPlan:
        """Select what one execution runs.

        A target pulls in the stages it transitively depends on, except those whose
        artifact is explicitly ``provided``: an existing immutable artifact is reused
        instead of recomputed. Nothing is inferred; an artifact that is provided but not
        needed, or of the wrong kind, is reported by preflight.

        Args:
            targets: Stages to produce, or ``None`` for the complete pipeline.
            provided: Existing artifacts by the stage that produced them.

        Returns:
            The execution scope; its problems are reported by :func:`preflight`.
        """
        provided = dict(provided or {})
        problems = list(self.problems)
        by_id = {stage.stage_id: stage for stage in self.stages}
        valid: dict[str, ArtifactRef] = {}
        for stage_id, ref in provided.items():
            stage = by_id.get(stage_id)
            path = f"provided.{stage_id}"
            if stage is None:
                problems.append(
                    ConfigProblem(path=path, message=f"{stage_id!r} is not a stage of this plan")
                )
            elif ref.stage_id != stage_id or ref.contract != stage.output:
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
                valid[stage_id] = ref
        if self.order is None:
            return ExecutionPlan(plan=self, stages=(), reused={}, problems=tuple(problems))

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
                        path=f"provided.{stage_id}",
                        message=(
                            f"{stage_id!r} is provided but not needed by the targets, and "
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
        )


@dataclass(frozen=True, kw_only=True)
class ExecutionPlan:
    """The part of a plan one execution runs, and what it reuses.

    Attributes:
        plan: The full topology.
        stages: Stages to run, in dependency order.
        reused: Existing artifacts fed to the stages that consume them, by producing stage.
        problems: Problems found while scoping.
    """

    plan: PipelinePlan
    stages: tuple[PlannedStage, ...]
    reused: Mapping[str, ArtifactRef]
    problems: tuple[ConfigProblem, ...]


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
        inputs: The exact artifacts it consumes, by input name.
        components: Resolved configuration of the stage's variation points.
        config_digest: Identity of the stage's own configuration.
    """

    stage_id: str
    inputs: Mapping[str, ArtifactRef]
    components: Mapping[str, ComponentConfig]
    config_digest: str


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
        inputs: The exact artifacts it consumed, by input name.
        output: The artifact it produced.
    """

    stage_id: str
    inputs: Mapping[str, ArtifactRef]
    output: ArtifactRef


@dataclass(frozen=True, kw_only=True)
class ExecutionRecord:
    """The outcome of one execution: order, exact inputs and outputs, reused artifacts.

    Attributes:
        plan_digest: Digest of the topology that was executed.
        order: Stages in the order they ran.
        stages: What each stage consumed and produced.
        reused: Existing artifacts that fed the stages, by producing stage.
    """

    plan_digest: str
    order: tuple[str, ...]
    stages: tuple[StageRecord, ...]
    reused: Mapping[str, ArtifactRef]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible record of the execution."""
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_digest": self.plan_digest,
            "order": list(self.order),
            "reused": {stage_id: ref.to_document() for stage_id, ref in self.reused.items()},
            "stages": [
                {
                    "stage_id": record.stage_id,
                    "inputs": {name: ref.to_document() for name, ref in record.inputs.items()},
                    "output": record.output.to_document(),
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
                    name=item.name, contract=item.contract, source=source, optional=item.optional
                )
            )
        component_configs = {
            component_id: config.components[component_id]
            for component_id in stage.components
            if component_id in config.components
        }
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
                    stage.stage_id, stage.capability, inputs, stage.output, component_configs
                ),
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
) -> PreflightReport:
    """Validate an execution before any stage runs or any model loads.

    Checks the structure found while resolving and scoping (cycles, missing
    dependencies, contract mismatches, wrong or unneeded provided artifacts), that every
    stage to run is implemented, that each of its variation points has a backend and that
    the backend's optional modules and secrets are present, and, when executors are
    given, that each stage has one. Nothing is imported or loaded.

    Args:
        execution: The scoped execution.
        executors: The executors that would run, to check coverage.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
        provided_runtimes: Component identities whose model runtime the caller supplies,
            so their bundled modules are not required.

    Returns:
        Every problem found, all at once.
    """
    problems = list(execution.problems)
    for stage in execution.stages:
        if not stage.available:
            problems.append(
                ConfigProblem(path=f"stages.{stage.stage_id}", message=stage.unavailable_reason)
            )
            continue
        for component_id, component in stage.component_configs.items():
            missing = check_component_selection(component_id, component)
            if missing is not None:
                problems.append(missing)
                continue
            problems.extend(
                check_component_availability(
                    component_id,
                    component,
                    environ=environ,
                    module_available=module_available,
                    check_modules=component_id not in provided_runtimes,
                )
            )
        if executors is not None and stage.stage_id not in executors:
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


def run_plan(
    execution: ExecutionPlan,
    executors: Mapping[str, StageExecutor],
    *,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    provided_runtimes: Collection[str] = (),
) -> ExecutionRecord:
    """Execute a scoped plan in dependency order.

    Preflight runs first and blocks everything when it finds a problem. Each stage then
    receives the exact artifacts of the stages that feed it, reused or just produced. The
    first failure, or an output that contradicts the stage's declared contract, stops the
    run: nothing later runs and nothing is substituted.

    Args:
        execution: The scoped execution.
        executors: One executor per stage to run.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
        provided_runtimes: Component identities whose model runtime the caller supplies.

    Returns:
        The execution record: order, exact inputs and outputs, and reused artifacts.

    Raises:
        PreflightError: If preflight found problems; nothing was executed.
        StageExecutionError: If a stage failed or returned an artifact of another kind.
    """
    report = preflight(
        execution,
        executors=executors,
        environ=environ,
        module_available=module_available,
        provided_runtimes=provided_runtimes,
    )
    if not report.ok:
        raise PreflightError(report)

    outputs: dict[str, ArtifactRef] = dict(execution.reused)
    records: list[StageRecord] = []
    for stage in execution.stages:
        inputs = {item.name: outputs[item.source] for item in stage.inputs}
        request = StageRequest(
            stage_id=stage.stage_id,
            inputs=inputs,
            components=stage.component_configs,
            config_digest=stage.config_digest,
        )
        completed = [record.stage_id for record in records]
        try:
            produced = executors[stage.stage_id].execute(request)
        except Exception as error:
            raise StageExecutionError(stage.stage_id, completed, str(error)) from error
        if produced.contract != stage.output or produced.stage_id != stage.stage_id:
            raise StageExecutionError(
                stage.stage_id,
                completed,
                f"returned a {produced.contract!r} artifact from stage {produced.stage_id!r}; "
                f"the stage declares {stage.output!r}",
            )
        outputs[stage.stage_id] = produced
        records.append(StageRecord(stage_id=stage.stage_id, inputs=inputs, output=produced))
    return ExecutionRecord(
        plan_digest=execution.plan.digest,
        order=tuple(record.stage_id for record in records),
        stages=tuple(records),
        reused=dict(execution.reused),
    )


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
) -> str:
    """Identify a stage's own configuration, not its position in the topology."""
    return _digest(
        {
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
    )


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
