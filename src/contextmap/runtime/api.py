"""The public, frontend-neutral runtime application API.

A CLI, a TUI or any other frontend needs the same few use cases: discover what the runtime can
do, resolve a configuration and its topology, preflight, run, and inspect what ran.
:class:`Runtime` exposes exactly those, so a frontend never imports the internal modules
(configuration, DAG, reuse, selection, lifecycle) and never grows a second description of the
pipeline.

It is a facade, not a new engine. Every operation delegates to the service that owns it:

- configuration and topology to :mod:`~contextmap.runtime.config` and
  :mod:`~contextmap.runtime.pipeline`;
- execution, reuse, selection and resume to the same runner, reuse index and selection
  resolution the DAG uses;
- lifecycle and lineage to the persisted run record, which stays authoritative.

The contracts returned here are small dataclasses with a ``to_document()``. They carry no
backend class, no ROS object and no UI-framework type. Discovery only looks modules up, so none
of it loads a model SDK. There is no global registry: a :class:`Runtime` is an ordinary object
that is given its executors, verifier and environment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from contextmap import __version__
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.catalog import (
    CANONICAL_PROFILE_ID,
    COMPONENTS,
    PRESETS,
    StageDeclaration,
)
from contextmap.runtime.composition import RuntimeProvider, compose, compose_executors
from contextmap.runtime.config import (
    CONFIG_SCHEMA_VERSION,
    DEBUG_LEVELS,
    ComponentConfig,
    ConfigProblem,
    ConfigurationError,
    EffectiveConfig,
    check_component_availability,
    read_effective_config,
    resolve_effective_config,
    resolve_secrets,
)
from contextmap.runtime.errors import (
    PlanDocumentError,
    PreflightError,
    RunCancelledError,
    RunRecordError,
    StageExecutionError,
)
from contextmap.runtime.ingestion_service import IngestionService, SourceAdapterFactory
from contextmap.runtime.lifecycle import (
    RUN_SCHEMA_VERSION,
    CancellationToken,
    EventSink,
    ExecutionEvent,
)
from contextmap.runtime.pipeline import (
    EXECUTION_FILENAME,
    PLAN_SCHEMA_VERSION,
    ExecutionPlan,
    PipelinePlan,
    StageExecutor,
    predict_reuse,
    preflight,
    read_plan_document,
    resolve_plan,
    run_plan,
)
from contextmap.runtime.reuse import REUSE_SCHEMA_VERSION, FileArtifactStore, ReusePolicy
from contextmap.runtime.runs import (
    RunJournal,
    RunSummary,
    check_resumable,
    read_run,
    resume_plan,
)
from contextmap.runtime.selection import (
    CATALOG_SCHEMA_VERSION,
    ArtifactCatalog,
    ResolvedSelections,
    resolve_selections,
)

RuntimeExecutionEvent = ExecutionEvent
"""The structured event a frontend receives while a run executes; a consumer must not alter it."""

_RECORDED_OUTCOMES = (PreflightError, StageExecutionError, RunCancelledError)
"""Errors the runner raises after it has already recorded the outcome in the run record."""


@dataclass(frozen=True, kw_only=True)
class RuntimeBackend:
    """One selectable backend of a component, and whether it can be used here.

    Attributes:
        backend_id: Identity used in configuration.
        available: Whether nothing it needs is missing on this machine.
        reasons: Why it is not available: the optional modules that are not installed and the
            environment secrets that are not set, the latter by name and never by value.
        requires: Optional modules its bundled code path imports.
        secrets: Names of the environment variables it needs.
        install_hint: How to install what it requires, when known.
    """

    backend_id: str
    available: bool
    reasons: tuple[str, ...]
    requires: tuple[str, ...]
    secrets: tuple[str, ...]
    install_hint: str

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "backend_id": self.backend_id,
            "available": self.available,
            "reasons": list(self.reasons),
            "requires": list(self.requires),
            "secrets": list(self.secrets),
            "install_hint": self.install_hint,
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeComponent:
    """A variation point of a capability and the backends that can fill it.

    Attributes:
        component_id: ``"<capability>.<slot>"``.
        backends: The selectable backends.
    """

    component_id: str
    backends: tuple[RuntimeBackend, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "component_id": self.component_id,
            "backends": [backend.to_document() for backend in self.backends],
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeCapability:
    """A stage of the topology, its implementation state and its variation points.

    Attributes:
        stage_id: The stage.
        capability: The capability package that owns it.
        implemented: Whether the capability exists in this build.
        reason: Why it does not, when it does not.
        optional: Whether a configuration may turn the stage off.
        default_enabled: Whether it takes part when the configuration is silent.
        components: Its variation points.
    """

    stage_id: str
    capability: str
    implemented: bool
    reason: str
    optional: bool
    default_enabled: bool
    components: tuple[RuntimeComponent, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "stage_id": self.stage_id,
            "capability": self.capability,
            "implemented": self.implemented,
            "reason": self.reason,
            "optional": self.optional,
            "default_enabled": self.default_enabled,
            "components": [component.to_document() for component in self.components],
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeStatus:
    """What this runtime is and what it is wired to.

    Attributes:
        contextmap_version: Installed package version.
        schemas: Versions of the persisted document schemas this runtime reads and writes.
        profiles: Known profile and preset identities.
        workspace: The workspace runs are persisted under, when one is set.
        executors: Stages this runtime was explicitly given an executor for, at construction.
            It does not include the stages :meth:`Runtime.preflight` and :meth:`Runtime.run`
            compose automatically from a specific configuration (composing needs a resolved
            ``EffectiveConfig``, which this discovery call does not take): call
            :meth:`Runtime.preflight` with a configuration and read its ``missing_executors``
            for the accurate, config-specific answer.
        verifier_configured: Whether reuse can verify indexed artifacts.
    """

    contextmap_version: str
    schemas: Mapping[str, str]
    profiles: tuple[str, ...]
    workspace: str | None
    executors: tuple[str, ...]
    verifier_configured: bool

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "contextmap_version": self.contextmap_version,
            "schemas": dict(self.schemas),
            "profiles": list(self.profiles),
            "workspace": self.workspace,
            "executors": list(self.executors),
            "verifier_configured": self.verifier_configured,
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeEdit:
    """One setting a frontend may change, and what it accepts.

    Feeding ``"<path>=<json value>"`` to :meth:`Runtime.resolve_config` as an override applies
    the edit.

    Attributes:
        path: The dotted override path.
        kind: ``"toggle"``, ``"choice"``, ``"text"`` or ``"selection"``.
        allowed: The accepted values, or ``None`` when any value of the kind is.
        current: The value in the effective configuration.
    """

    path: str
    kind: str
    allowed: tuple[Any, ...] | None
    current: Any

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "path": self.path,
            "kind": self.kind,
            "allowed": None if self.allowed is None else list(self.allowed),
            "current": self.current,
        }


@dataclass(frozen=True, kw_only=True)
class RuntimePlanInput:
    """A typed input of a planned stage and the stage that feeds it.

    Attributes:
        name: Input name.
        contract: Artifact kind it consumes.
        source: The stage that produces it in this plan.
        optional: Whether the stage runs without it.
        multiple: Whether it accepts several runs of the source stage.
    """

    name: str
    contract: str
    source: str
    optional: bool
    multiple: bool

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "name": self.name,
            "contract": self.contract,
            "source": self.source,
            "optional": self.optional,
            "multiple": self.multiple,
        }


@dataclass(frozen=True, kw_only=True)
class RuntimePlanStage:
    """One stage of a resolved topology.

    Attributes:
        stage_id: The stage.
        capability: The capability package that owns it.
        optional: Whether a configuration may turn it off.
        available: Whether its capability is implemented.
        unavailable_reason: Why it is not, when it is not.
        backends: The selected backend of each variation point (``None`` while unselected).
        inputs: Its typed inputs, wired to their producers.
        output: The artifact kind it produces.
        config_digest: Identity of the stage's own configuration.
        in_scope: Whether the execution runs it.
        provided: Ids of the upstream artifacts supplied instead of running the stage.
    """

    stage_id: str
    capability: str
    optional: bool
    available: bool
    unavailable_reason: str
    backends: Mapping[str, str | None]
    inputs: tuple[RuntimePlanInput, ...]
    output: str | None
    config_digest: str
    in_scope: bool
    provided: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "stage_id": self.stage_id,
            "capability": self.capability,
            "optional": self.optional,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "backends": dict(self.backends),
            "inputs": [item.to_document() for item in self.inputs],
            "output": self.output,
            "config_digest": self.config_digest,
            "in_scope": self.in_scope,
            "provided": list(self.provided),
        }


@dataclass(frozen=True, kw_only=True)
class ResolvedPipelinePlan:
    """The topology one configuration resolves to, and what an execution would run.

    Attributes:
        preset: The topology preset.
        config_digest: Digest of the effective configuration.
        plan_digest: Digest of the topology.
        order: Stage identities in execution order, or ``None`` when the graph has a cycle.
        stages: The stages that take part, in execution order.
        disabled_stages: Optional stages the configuration turned off.
        run_stages: The stages the execution would run.
        selections: The resolved run selection with each run's origin and lineage, when the
            configuration selects upstream runs.
        problems: Structural problems: a cycle, a contract mismatch, an unknown target.
        editable: The settings a frontend may change.
    """

    preset: str
    config_digest: str
    plan_digest: str
    order: tuple[str, ...] | None
    stages: tuple[RuntimePlanStage, ...]
    disabled_stages: tuple[str, ...]
    run_stages: tuple[str, ...]
    selections: Mapping[str, Any] | None
    problems: tuple[ConfigProblem, ...]
    editable: tuple[RuntimeEdit, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "preset": self.preset,
            "config_digest": self.config_digest,
            "plan_digest": self.plan_digest,
            "order": None if self.order is None else list(self.order),
            "stages": [stage.to_document() for stage in self.stages],
            "disabled_stages": list(self.disabled_stages),
            "run_stages": list(self.run_stages),
            "selections": None if self.selections is None else dict(self.selections),
            "problems": _problem_documents(self.problems),
            "editable": [edit.to_document() for edit in self.editable],
        }


@dataclass(frozen=True, kw_only=True)
class RuntimePreflightReport:
    """What preflight found, and what the execution would consume and produce.

    Attributes:
        ok: Whether nothing blocks the execution.
        problems: Everything that blocks it, all at once.
        warnings: What does not block but deserves attention.
        config_digest: Digest of the effective configuration.
        plan_digest: Digest of the topology.
        stages: The stages the execution would run, in order.
        provided: The upstream artifact ids it would start from, by stage.
        outputs: The artifact kind each stage to run would produce.
        predicted_reuse: When a reuse policy is given, what would be reused or recomputed.
        missing_executors: Stages that would run and have no executor.
    """

    ok: bool
    problems: tuple[ConfigProblem, ...]
    warnings: tuple[str, ...]
    config_digest: str
    plan_digest: str
    stages: tuple[str, ...]
    provided: Mapping[str, tuple[str, ...]]
    outputs: Mapping[str, str | None]
    predicted_reuse: Mapping[str, Mapping[str, Any]]
    missing_executors: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "ok": self.ok,
            "problems": _problem_documents(self.problems),
            "warnings": list(self.warnings),
            "config_digest": self.config_digest,
            "plan_digest": self.plan_digest,
            "stages": list(self.stages),
            "provided": {stage: list(ids) for stage, ids in self.provided.items()},
            "outputs": dict(self.outputs),
            "predicted_reuse": {stage: dict(item) for stage, item in self.predicted_reuse.items()},
            "missing_executors": list(self.missing_executors),
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeRunStage:
    """What the run record says about one stage; nothing here is inferred.

    Attributes:
        stage_id: The stage.
        outcome: ``"pending"`` (nothing recorded), ``"started"`` (no end recorded),
            ``"reused"``, ``"completed"`` or ``"failed"``.
        inputs: The exact upstream artifact ids per input, or ``None`` when the record does not
            say. A stage that was reused by a run that did not complete has no recorded inputs.
        output: The artifact it produced or reused, when recorded.
        decision: The reuse decision (kind, reason, key and the exact prior artifact), when the
            run had a reuse policy.
        elapsed_s: Wall time, when recorded.
    """

    stage_id: str
    outcome: str
    inputs: Mapping[str, tuple[str, ...]] | None
    output: Mapping[str, Any] | None
    decision: Mapping[str, Any] | None
    elapsed_s: float | None

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "stage_id": self.stage_id,
            "outcome": self.outcome,
            "inputs": None
            if self.inputs is None
            else {name: list(ids) for name, ids in self.inputs.items()},
            "output": None if self.output is None else dict(self.output),
            "decision": None if self.decision is None else dict(self.decision),
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeRunRecord:
    """A persisted run, read back: its lifecycle, lineage and failure.

    The persisted record is authoritative: what it does not hold is ``None`` or empty and is
    reported in ``notes``, never reconstructed.

    Attributes:
        run_id: The run identity.
        directory: The run directory.
        status: ``planned``, ``running``, ``completed``, ``failed``, ``blocked`` or ``cancelled``.
        interrupted: Whether the run never ended and no live process owns it.
        config_digest: Digest of the effective configuration.
        plan_digest: Digest of the topology.
        backends: The backend selected for each variation point, from the persisted
            configuration; ``None`` when that file is missing or unreadable.
        targets: The stages the run had to execute, in order.
        provided: The upstream artifact ids it was supplied, by stage.
        stages: What is recorded for each stage of ``targets``.
        selections: The resolved run selection, when the run completed and had one.
        resumed_from: The run this one resumed.
        resume: For a resumed run that completed, which stages were reused and recomputed.
        failure: The failure record.
        blocked_problems: Why preflight refused to start.
        code_identity: The code identity recorded for reproduction.
        environment: The environment recorded for reproduction.
        created_at: When the run was planned.
        updated_at: The time of the last recorded event.
        events: Every recorded event, in order.
        notes: Inconsistencies found in the record.
    """

    run_id: str
    directory: str
    status: str
    interrupted: bool
    config_digest: str
    plan_digest: str
    backends: Mapping[str, str | None] | None
    targets: tuple[str, ...]
    provided: Mapping[str, tuple[str, ...]]
    stages: tuple[RuntimeRunStage, ...]
    selections: Mapping[str, Any] | None
    resumed_from: str | None
    resume: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None
    blocked_problems: tuple[ConfigProblem, ...]
    code_identity: str | None
    environment: Mapping[str, Any]
    created_at: str
    updated_at: str
    events: tuple[ExecutionEvent, ...]
    notes: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "run_id": self.run_id,
            "directory": self.directory,
            "status": self.status,
            "interrupted": self.interrupted,
            "config_digest": self.config_digest,
            "plan_digest": self.plan_digest,
            "backends": None if self.backends is None else dict(self.backends),
            "targets": list(self.targets),
            "provided": {stage: list(ids) for stage, ids in self.provided.items()},
            "stages": [stage.to_document() for stage in self.stages],
            "selections": None if self.selections is None else dict(self.selections),
            "resumed_from": self.resumed_from,
            "resume": None if self.resume is None else dict(self.resume),
            "failure": None if self.failure is None else dict(self.failure),
            "blocked_problems": _problem_documents(self.blocked_problems),
            "code_identity": self.code_identity,
            "environment": dict(self.environment),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "events": [event.to_document() for event in self.events],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeRunSummary:
    """One line of the run list.

    Attributes:
        run_id: The run identity, unique only inside its dataset.
        dataset: The dataset the run belongs to, the directory it lives under.
        readable: Whether the record could be read.
        status: The lifecycle state, when readable.
        interrupted: Whether the run was killed mid-way, when readable.
        created_at: When the run was planned, when readable.
        updated_at: The time of the last recorded event, when readable.
        resumed_from: The run this one resumed, when readable.
        failure_category: The failure category, when the run failed.
        error: Why the record could not be read.
    """

    run_id: str
    dataset: str
    readable: bool
    status: str | None
    interrupted: bool | None
    created_at: str | None
    updated_at: str | None
    resumed_from: str | None
    failure_category: str | None
    error: str | None

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "run_id": self.run_id,
            "readable": self.readable,
            "status": self.status,
            "interrupted": self.interrupted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resumed_from": self.resumed_from,
            "failure_category": self.failure_category,
            "error": self.error,
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeExecutionResult:
    """The outcome of one execution, read back from its persisted run record.

    :meth:`Runtime.run` returns this for every expected outcome (completed, failed, blocked,
    cancelled) instead of raising, so a frontend has one shape to render; the record it wraps is
    the authority on lineage.

    Attributes:
        record: The persisted run record.
        event_errors: What the frontend's event sink raised. The run ignores it: a sink never
            changes what a run does.
    """

    record: RuntimeRunRecord
    event_errors: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        """The run's lifecycle state."""
        return self.record.status

    @property
    def run_id(self) -> str:
        """The run identity."""
        return self.record.run_id

    @property
    def ok(self) -> bool:
        """Whether the run completed."""
        return self.record.status == "completed"

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "status": self.status,
            "run_id": self.run_id,
            "record": self.record.to_document(),
            "event_errors": list(self.event_errors),
        }


@dataclass(frozen=True, kw_only=True)
class _Scoped:
    """A configuration resolved into its topology and the scope of one execution."""

    plan: PipelinePlan
    execution: ExecutionPlan
    resolved: ResolvedSelections | None


class _GuardedSink:
    """Hands events to a frontend without letting the frontend change the run.

    A sink that raises would otherwise fail the runner mid-way and be recorded as an unexpected
    failure of the pipeline. Here its error is kept and reported instead.
    """

    def __init__(self, sink: EventSink, redact: Callable[[str], str]) -> None:
        self._sink = sink
        self._redact = redact
        self.errors: list[str] = []

    def emit(self, event: ExecutionEvent) -> None:
        try:
            self._sink.emit(event)
        except Exception as error:  # qualquer erro do consumidor é isolado, nunca propagado
            message = f"event {event.sequence} ({event.kind}): {type(error).__name__}: {error}"
            self.errors.append(self._redact(message))


class Runtime:
    """The public entry point of the runtime for any frontend.

    Args:
        workspace: Where runs are persisted. It becomes ``resources.workspace`` of every
            configuration :meth:`resolve_config` resolves, exactly as the CLI's ``--workspace``.
        executors: Executors to use in addition to, and in preference over, the ones
            :func:`~contextmap.runtime.composition.compose_executors` builds automatically from
            each call's ``EffectiveConfig`` (today: ``state_estimation``, ``geometric_mapping``,
            ``sensor_association`` and ``semantic_fusion``). Pass an entry here to override a
            composed stage (a test double, for example) or to supply one composition cannot
            build on its own, such as ``ingestion``'s ``IngestionStageExecutor`` (it needs a
            concrete request that is never part of a configuration). A stage with neither a
            composed nor a supplied executor is blocked by preflight.
        providers: Model runtimes or clients for a backend with no bundled loader (SAM2, SAM3,
            Qwen, Gemini, Florence-2 and every other backend ``compose_executors`` builds
            through a ``RuntimeProvider``), keyed by component identity
            (``"<capability>.<slot>"``), in the exact shape
            :func:`~contextmap.runtime.composition.compose_executors` already expects. Without
            it, a stage whose selected backends need one (today, ``visual_perception`` unless
            every one of its four backends bundles its own loader) is composed by neither this
            runtime nor a frontend that never builds a whole executor by hand.
        verifier: Tells whether an indexed artifact still exists and is intact. Reuse and resume
            need it, and only the owner of the executors can provide it.
        adapter_factory: Builds the source adapter for ingestion; composed from the
            configuration when omitted.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed; defaults
            to a metadata lookup that never imports the module.
        clock: Returns event timestamps.
    """

    def __init__(
        self,
        *,
        workspace: str | os.PathLike[str] | None = None,
        executors: Mapping[str, StageExecutor] | None = None,
        providers: Mapping[str, RuntimeProvider] | None = None,
        verifier: Callable[[ArtifactRef], bool] | None = None,
        adapter_factory: SourceAdapterFactory | None = None,
        environ: Mapping[str, str] | None = None,
        module_available: Callable[[str], bool] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        """Create a runtime; nothing is loaded and nothing is discovered."""
        self._workspace = None if workspace is None else Path(workspace)
        self._executors = dict(executors or {})
        self._providers = dict(providers or {})
        self._verifier = verifier
        self._adapter_factory = adapter_factory
        self._environ = environ
        self._module_available = module_available
        self._clock = clock

    # --- discovery --------------------------------------------------------------------

    def status(self) -> RuntimeStatus:
        """Describe this runtime: version, schemas, profiles and what it is wired to.

        Returns:
            The status. It reads nothing from disk.
        """
        return RuntimeStatus(
            contextmap_version=__version__,
            schemas={
                "configuration": CONFIG_SCHEMA_VERSION,
                "plan": PLAN_SCHEMA_VERSION,
                "run": RUN_SCHEMA_VERSION,
                "reuse": REUSE_SCHEMA_VERSION,
                "catalog": CATALOG_SCHEMA_VERSION,
            },
            profiles=tuple(sorted(PRESETS)),
            workspace=None if self._workspace is None else str(self._workspace),
            executors=tuple(sorted(self._executors)),
            verifier_configured=self._verifier is not None,
        )

    def capabilities(self, *, profile: str = CANONICAL_PROFILE_ID) -> tuple[RuntimeCapability, ...]:
        """List every stage, its variation points and whether each backend can be used here.

        Availability is decided by looking modules up in the installed metadata and secrets up
        in the environment: **no module is imported and no model is loaded**. A backend that
        cannot be used says why.

        Args:
            profile: The topology preset to describe.

        Returns:
            One entry per stage, in the preset's documentation order.

        Raises:
            ConfigurationError: If the profile is unknown.
        """
        preset = PRESETS.get(profile)
        if preset is None:
            raise ConfigurationError.single(
                f"unknown profile {profile!r}; known profiles: {', '.join(sorted(PRESETS))}"
            )
        return tuple(self._capability(stage) for stage in preset.stages)

    # --- configuration and topology -----------------------------------------------------

    def resolve_config(
        self,
        *,
        profile: str = CANONICAL_PROFILE_ID,
        files: Sequence[str | os.PathLike[str]] = (),
        overrides: Sequence[str] = (),
    ) -> EffectiveConfig:
        """Resolve the effective configuration from a profile, files and overrides.

        Args:
            profile: The base profile.
            files: ``.json`` or ``.toml`` configuration files; later ones win.
            overrides: ``"dotted.path=value"`` overrides; they win over every file. Without an
                explicit ``resources.workspace`` the runtime's workspace is used.

        Returns:
            The effective configuration, with its versioned digest and the layers that made it.

        Raises:
            ConfigurationError: If it is invalid or selects an unsupported stage or backend.
        """
        layers = list(overrides)
        if self._workspace is not None and not any(
            layer.startswith("resources.workspace=") for layer in layers
        ):
            layers.insert(0, f"resources.workspace={json.dumps(str(self._workspace))}")
        return resolve_effective_config(profile=profile, files=files, overrides=layers)

    def resolve_plan(
        self,
        config: EffectiveConfig,
        *,
        targets: Iterable[str] | None = None,
        provided: Mapping[str, ArtifactRef] | None = None,
        catalog: ArtifactCatalog | None = None,
    ) -> ResolvedPipelinePlan:
        """Resolve the topology and the scope of one execution for a configuration.

        Args:
            config: The effective configuration.
            targets: Stages to produce, or ``None`` for the complete pipeline.
            provided: Exact upstream artifacts to start from instead of running their stages.
            catalog: The runs available to select from, when the configuration selects any.

        Returns:
            The plan, in deterministic order, with its problems and the settings a frontend may
            edit. Structural problems are returned, not raised.

        Raises:
            ValueError: If ``provided`` is given while the configuration selects upstream runs.
        """
        return self._describe(config, self._scope(config, targets, provided, catalog))

    def preflight(
        self,
        config: EffectiveConfig,
        *,
        targets: Iterable[str] | None = None,
        provided: Mapping[str, ArtifactRef] | None = None,
        catalog: ArtifactCatalog | None = None,
        reuse: ReusePolicy | None = None,
    ) -> RuntimePreflightReport:
        """Check what a run would hit, before any model loads.

        It runs the same checks :meth:`run` runs first (topology, contracts, selections,
        backend selection, optional modules, secrets, executors) and reports every problem at
        once. It looks up and imports nothing, so it is cheap enough for an interactive
        frontend. ``missing_executors`` accounts for both the executors this runtime was
        constructed with and the ones :func:`~contextmap.runtime.composition.compose_executors`
        can build from ``config`` alone; a composition failure (an invalid backend parameter, a
        missing module or secret) is reported the same way, never raised.

        Args:
            config: The effective configuration.
            targets: Stages to produce, or ``None`` for the complete pipeline.
            provided: Exact upstream artifacts to start from.
            catalog: The runs available to select from.
            reuse: A reuse policy, to predict what would be reused.

        Returns:
            The report.

        Raises:
            ValueError: If ``provided`` is given while the configuration selects upstream runs.
        """
        scoped = self._scope(config, targets, provided, catalog)
        execution = scoped.execution
        executors = self._executors_for(config)
        report = preflight(
            execution,
            executors=executors,
            environ=self._environ,
            module_available=self._module_available,
            reuse=reuse,
        )
        predicted = {} if reuse is None else predict_reuse(execution, reuse)
        return RuntimePreflightReport(
            ok=report.ok,
            problems=report.problems,
            warnings=self._warnings(config, scoped),
            config_digest=config.digest,
            plan_digest=scoped.plan.digest,
            stages=report.stages,
            provided={
                stage: tuple(ref.artifact_id for ref in refs)
                for stage, refs in execution.reused.items()
            },
            outputs={stage.stage_id: stage.output for stage in execution.stages},
            predicted_reuse={stage: item.to_document() for stage, item in predicted.items()},
            missing_executors=tuple(
                stage.stage_id
                for stage in execution.stages
                if stage.available
                and stage.stage_id not in executors
                and (stage.stage_id not in predicted or predicted[stage.stage_id].kind != "reused")
            ),
        )

    # --- execution --------------------------------------------------------------------

    def run(
        self,
        config: EffectiveConfig,
        *,
        targets: Iterable[str] | None = None,
        provided: Mapping[str, ArtifactRef] | None = None,
        catalog: ArtifactCatalog | None = None,
        reuse: ReusePolicy | None = None,
        resume: str | os.PathLike[str] | None = None,
        events: EventSink | None = None,
        cancellation: CancellationToken | None = None,
        redact: Callable[[str], str] | None = None,
        code_identity: str | None = None,
    ) -> RuntimeExecutionResult:
        """Execute the pipeline, or a subgraph of it, and persist the run.

        The path is the canonical one: preflight, then the DAG runner, journaled into a fresh run
        directory. Every expected outcome (completed, failed, blocked, cancelled) comes back as
        a result read from the persisted record. Only misuse and unexpected bugs raise, and a
        keyboard interrupt is recorded as a cancellation and re-raised. Events reach ``events``
        in order, after they are persisted; what the sink does, or raises, never changes the run.

        Args:
            config: The effective configuration.
            targets: Stages to produce, or ``None`` for the complete pipeline.
            provided: Exact upstream artifacts to start from.
            catalog: The runs available to select from.
            reuse: Reuse policy: a stage with an identical identity reuses the exact prior
                artifact instead of running.
            resume: A failed, cancelled or interrupted run to resume as a new run (a run id or a
                run directory); it needs ``reuse``.
            events: Receives every event of the run.
            cancellation: Cooperative cancellation, checked before each stage.
            redact: Replaces secret values inside a string; by default the secrets the selected
                backends declare are redacted from the environment.
            code_identity: Identity of the code producing the results, recorded; defaults to the
                reuse policy's.

        Returns:
            The result, wrapping the persisted run record.

        Raises:
            ValueError: If there is no workspace, it differs from the configuration's, ``resume``
                comes without a reuse policy, or ``provided`` conflicts with the configured
                selections.
            ResumeError: If the run cannot be resumed as requested; no run is created.
            RunRecordError: If the run to resume cannot be read.
        """
        workspace = self._workspace_for(config)
        scoped = self._scope(config, targets, provided, catalog)
        executors = self._executors_for(config)
        previous: Path | None = None
        if resume is not None:
            if reuse is None:
                raise ValueError("resuming a run reuses its completed stages: pass a reuse policy")
            previous = self._run_directory(resume)
            check_resumable(read_run(previous), scoped.execution)  # recusa antes de criar um run
        if code_identity is None and reuse is not None:
            code_identity = reuse.code_identity
        cleaner = redact or resolve_secrets(config.config, environ=self._environ).redact
        guard = None if events is None else _GuardedSink(events, cleaner)
        journal = RunJournal.create(
            workspace, config, scoped.execution, code_identity=code_identity, clock=self._clock
        )
        try:
            if previous is not None and reuse is not None:
                resume_plan(
                    previous,
                    scoped.execution,
                    executors,
                    reuse=reuse,
                    environ=self._environ,
                    module_available=self._module_available,
                    journal=journal,
                    events=guard,
                    cancellation=cancellation,
                    redact=cleaner,
                    clock=self._clock,
                )
            else:
                run_plan(
                    scoped.execution,
                    executors,
                    environ=self._environ,
                    module_available=self._module_available,
                    reuse=reuse,
                    journal=journal,
                    events=guard,
                    cancellation=cancellation,
                    redact=cleaner,
                    clock=self._clock,
                )
        except _RECORDED_OUTCOMES:
            pass  # o run já registrou o desfecho: o resultado é lido do registro persistido
        return RuntimeExecutionResult(
            record=self.inspect_run(journal.directory),
            event_errors=() if guard is None else tuple(guard.errors),
        )

    def reuse_policy(
        self,
        index: str | os.PathLike[str],
        *,
        code_identity: str,
        force: Collection[str] = (),
        identities: Mapping[str, Mapping[str, str]] | None = None,
    ) -> ReusePolicy:
        """Build a reuse policy over an artifact index directory.

        Args:
            index: The directory of the index.
            code_identity: Identity of the code producing the results; reuse across code
                versions must be a decision.
            force: Stages that always run.
            identities: Extra identities per stage, folded into that stage's key.

        Returns:
            The policy.

        Raises:
            ValueError: If this runtime has no verifier, or the code identity is empty.
        """
        if self._verifier is None:
            raise ValueError(
                "reuse needs an artifact verifier: pass verifier= to Runtime(...); only the "
                "owner of the stage executors can tell whether an indexed artifact still exists"
            )
        return ReusePolicy(
            store=FileArtifactStore(index, verify=self._verifier),
            code_identity=code_identity,
            force_recompute=frozenset(force),
            identities=dict(identities or {}),
        )

    def ingestion(self, config: EffectiveConfig) -> IngestionService:
        """Return the public ingestion service for a configuration.

        The source adapter is the one the configuration selects, built by the composition root:
        there is no fallback to another family.

        Args:
            config: The effective configuration.

        Returns:
            The service; its ``preflight`` and ``run`` are the use cases.

        Raises:
            ConfigurationError: If no source adapter is selected.
            BackendUnavailableError: If the selected adapter needs a module that is missing.
        """
        factory = self._adapter_factory
        if factory is None:
            composed = compose(
                config,
                stages=["ingestion"],
                environ=self._environ,
                module_available=self._module_available,
            )
            assert composed.source_adapter is not None  # o estágio `ingestion` foi composto
            factory = composed.source_adapter
        return IngestionService(factory, clock=self._clock)

    # --- runs --------------------------------------------------------------------------

    def list_runs(self) -> tuple[RuntimeRunSummary, ...]:
        """List the persisted runs of the workspace, ``<workspace>/<dataset>/run-NNNN``.

        Returns:
            One summary per run directory, ordered by dataset and then by run number. A
            record that cannot be read is listed with the reason, never skipped.

        Raises:
            ValueError: If this runtime has no workspace.
        """
        if self._workspace is None:
            raise ValueError("this runtime has no workspace: pass workspace= to Runtime(...)")
        if not self._workspace.is_dir():
            return ()
        numbered = sorted(
            (dataset.name, int(path.name.removeprefix("run-")), path)
            for dataset in self._workspace.iterdir()
            if dataset.is_dir()
            for path in dataset.iterdir()
            if path.is_dir()
            and path.name.startswith("run-")
            and path.name.removeprefix("run-").isdigit()
        )
        return tuple(_summary(path) for _, _, path in numbered)

    def inspect_run(self, run: str | os.PathLike[str]) -> RuntimeRunRecord:
        """Read one persisted run: lifecycle, exact lineage, decisions and failure.

        The persisted record is the authority: what it does not hold stays ``None`` and is
        reported in ``notes``.

        Args:
            run: A run id under the workspace, or a run directory.

        Returns:
            The record.

        Raises:
            RunRecordError: If the run does not exist or its record is not readable.
        """
        return _record(read_run(self._run_directory(run)))

    # --- internals ---------------------------------------------------------------------

    def _capability(self, stage: StageDeclaration) -> RuntimeCapability:
        return RuntimeCapability(
            stage_id=stage.stage_id,
            capability=stage.capability,
            implemented=stage.available,
            reason=stage.unavailable_reason,
            optional=stage.optional,
            default_enabled=stage.default_enabled,
            components=tuple(
                RuntimeComponent(
                    component_id=component_id,
                    backends=tuple(
                        self._backend(component_id, backend_id)
                        for backend_id in sorted(COMPONENTS[component_id].backends)
                    ),
                )
                for component_id in stage.components
            ),
        )

    def _backend(self, component_id: str, backend_id: str) -> RuntimeBackend:
        spec = COMPONENTS[component_id].backends[backend_id]
        problems = check_component_availability(
            component_id,
            ComponentConfig(backend=backend_id, parameters={}),
            environ=self._environ,
            module_available=self._module_available,
        )
        return RuntimeBackend(
            backend_id=backend_id,
            available=not problems,
            reasons=tuple(problem.message for problem in problems),
            requires=spec.requires,
            secrets=spec.secrets,
            install_hint=spec.install_hint,
        )

    def _scope(
        self,
        config: EffectiveConfig,
        targets: Iterable[str] | None,
        provided: Mapping[str, ArtifactRef] | None,
        catalog: ArtifactCatalog | None,
    ) -> _Scoped:
        plan = resolve_plan(config)
        inputs = config.config.inputs
        if not inputs.selections:
            return _Scoped(
                plan=plan,
                execution=plan.scope(targets=targets, provided=provided),
                resolved=None,
            )
        if provided is not None:
            raise ValueError(
                "the configuration selects upstream runs (inputs.selections); pass either "
                "provided artifacts or selections, not both"
            )
        if catalog is None:
            execution = plan.scope(targets=targets)
            missing = ConfigProblem(
                path="inputs.selections",
                message=(
                    "the configuration selects upstream runs but no catalog listing the runs "
                    "available to select from was given"
                ),
            )
            return _Scoped(
                plan=plan,
                execution=replace(execution, problems=(*execution.problems, missing)),
                resolved=None,
            )
        resolved = resolve_selections(plan, inputs, catalog)
        return _Scoped(
            plan=plan,
            execution=plan.scope(targets=targets, selections=resolved),
            resolved=resolved,
        )

    def _describe(self, config: EffectiveConfig, scoped: _Scoped) -> ResolvedPipelinePlan:
        plan, execution = scoped.plan, scoped.execution
        preset = PRESETS[plan.preset_id]
        running = {stage.stage_id for stage in execution.stages}
        stages = tuple(
            RuntimePlanStage(
                stage_id=stage.stage_id,
                capability=stage.capability,
                optional=preset.stage(stage.stage_id).optional,
                available=stage.available,
                unavailable_reason=stage.unavailable_reason,
                backends=dict(stage.components),
                inputs=tuple(
                    RuntimePlanInput(
                        name=item.name,
                        contract=item.contract,
                        source=item.source,
                        optional=item.optional,
                        multiple=item.multiple,
                    )
                    for item in stage.inputs
                ),
                output=stage.output,
                config_digest=stage.config_digest,
                in_scope=stage.stage_id in running,
                provided=tuple(ref.artifact_id for ref in execution.reused.get(stage.stage_id, ())),
            )
            for stage in plan.stages
        )
        planned = {stage.stage_id for stage in plan.stages}
        return ResolvedPipelinePlan(
            preset=plan.preset_id,
            config_digest=config.digest,
            plan_digest=plan.digest,
            order=plan.order,
            stages=stages,
            disabled_stages=tuple(
                stage.stage_id for stage in preset.stages if stage.stage_id not in planned
            ),
            run_stages=tuple(stage.stage_id for stage in execution.stages),
            selections=None if scoped.resolved is None else scoped.resolved.to_document(),
            problems=execution.problems,
            editable=_editable(config, preset.stages, planned),
        )

    def _warnings(self, config: EffectiveConfig, scoped: _Scoped) -> tuple[str, ...]:
        warnings = []
        if self._workspace is None and config.config.resources.workspace is None:
            warnings.append("no workspace is configured: a run needs one to persist its record")
        if scoped.resolved is not None:
            for stage, runs in scoped.resolved.runs.items():
                warnings.extend(
                    f"stage {stage!r}: 'latest' resolved to {run.entry.ref.artifact_id!r}"
                    for run in runs
                    if run.origin == "latest"
                )
        return tuple(warnings)

    def _executors_for(self, config: EffectiveConfig) -> Mapping[str, StageExecutor]:
        """Merge the executors composed from ``config`` with the ones given at construction.

        ``compose_executors`` builds every stage it genuinely can (today: ``state_estimation``,
        ``geometric_mapping``, ``sensor_association``, ``semantic_fusion`` and, once
        ``self._providers`` supplies a runtime for every backend that needs one,
        ``visual_perception``) from ``config`` alone; a stage it cannot build for any reason is
        simply absent, never raised (see its own docstring). Whatever this runtime was
        constructed with in ``executors`` (a test double, a stage composition cannot build such
        as ``ingestion``, or an explicit override) is layered on top and always wins.
        """
        composed = compose_executors(
            config,
            providers=self._providers,
            environ=self._environ,
            module_available=self._module_available,
        )
        return {**composed, **self._executors}

    def _workspace_for(self, config: EffectiveConfig) -> Path:
        configured = config.config.resources.workspace
        if configured is None:
            if self._workspace is None:
                raise ValueError(
                    "a run persists its record: pass workspace= to Runtime(...) or set "
                    "resources.workspace"
                )
            return self._workspace
        if self._workspace is not None and Path(configured) != self._workspace:
            raise ValueError(
                f"the configuration's workspace {configured!r} is not this runtime's "
                f"workspace {str(self._workspace)!r}: runs of one runtime live in one workspace"
            )
        return Path(configured)

    def _run_directory(self, run: str | os.PathLike[str]) -> Path:
        """Resolve a run id under any dataset of the workspace, or a run directory."""
        reference = os.fspath(run)
        if (
            self._workspace is not None
            and self._workspace.is_dir()
            and Path(reference).name == reference
        ):
            found = sorted(
                dataset / reference
                for dataset in self._workspace.iterdir()
                if (dataset / reference).is_dir()
            )
            if len(found) > 1:
                raise RunRecordError(
                    f"run id {reference!r} exists in several datasets "
                    f"({', '.join(path.parent.name for path in found)}): pass the run directory"
                )
            if found:
                return found[0]
        if Path(reference).is_dir():
            return Path(reference)
        where = "" if self._workspace is None else f" under {self._workspace}"
        raise RunRecordError(f"no run record for {reference!r}{where}")


def _problem_documents(problems: Iterable[ConfigProblem]) -> list[dict[str, str]]:
    return [{"path": problem.path, "message": problem.message} for problem in problems]


def _editable(
    config: EffectiveConfig, stages: Sequence[StageDeclaration], planned: Collection[str]
) -> tuple[RuntimeEdit, ...]:
    """List what a frontend may change, each as a path that ``resolve_config`` accepts."""
    edits = [
        RuntimeEdit(
            path=f"pipeline.stages.{stage.stage_id}",
            kind="toggle",
            allowed=(True, False),
            current=stage.stage_id in planned,
        )
        for stage in stages
        if stage.optional
    ]
    for stage in stages:
        if stage.stage_id not in planned:
            continue
        for component_id in stage.components:
            component = config.config.components[component_id]
            edits.append(
                RuntimeEdit(
                    path=f"components.{component_id}.backend",
                    kind="choice",
                    allowed=tuple(sorted(COMPONENTS[component_id].backends)),
                    current=component.backend,
                )
            )
    edits.append(
        RuntimeEdit(
            path="resources.device",
            kind="text",
            allowed=None,
            current=config.config.resources.device,
        )
    )
    edits.append(
        RuntimeEdit(
            path="policies.debug_level",
            kind="choice",
            allowed=DEBUG_LEVELS,
            current=config.config.policies.debug_level,
        )
    )
    edits.append(
        RuntimeEdit(
            path="inputs.sequence",
            kind="text",
            allowed=None,
            current=config.config.inputs.sequence,
        )
    )
    edits.extend(
        RuntimeEdit(
            path=f"inputs.selections.{stage.stage_id}",
            kind="selection",
            allowed=None,
            current=list(config.config.inputs.selections.get(stage.stage_id, ())) or None,
        )
        for stage in stages
        if stage.stage_id in planned and stage.available and stage.output is not None
    )
    return tuple(edits)


def _summary(directory: Path) -> RuntimeRunSummary:
    try:
        run = read_run(directory)
    except RunRecordError as error:
        return RuntimeRunSummary(
            run_id=directory.name,
            dataset=directory.parent.name,
            readable=False,
            status=None,
            interrupted=None,
            created_at=None,
            updated_at=None,
            resumed_from=None,
            failure_category=None,
            error=str(error),
        )
    return RuntimeRunSummary(
        run_id=run.run_id,
        dataset=directory.parent.name,
        readable=True,
        status=run.status.value,
        interrupted=run.interrupted,
        created_at=run.created_at,
        updated_at=run.updated_at,
        resumed_from=run.resumed_from,
        failure_category=None if run.failure is None else run.failure.category,
        error=None,
    )


def _record(summary: RunSummary) -> RuntimeRunRecord:
    """Read a run record into the public contract, adding nothing the record does not hold."""
    directory = summary.directory
    notes = list(summary.notes)
    backends: dict[str, str | None] | None = None
    try:
        persisted = read_effective_config(directory / "effective_config.json")
        backends = {
            component_id: component.backend
            for component_id, component in persisted.config.components.items()
        }
    except ConfigurationError as error:
        notes.append(f"the persisted configuration is unreadable, so backends are unknown: {error}")
    execution: dict[str, Any] | None = None
    if (directory / EXECUTION_FILENAME).is_file():
        try:
            execution = read_plan_document(directory / EXECUTION_FILENAME)
        except PlanDocumentError as error:
            notes.append(f"the execution record is unreadable, so its lineage is unknown: {error}")
    return RuntimeRunRecord(
        run_id=summary.run_id,
        directory=str(directory),
        status=summary.status.value,
        interrupted=summary.interrupted,
        config_digest=summary.config_digest,
        plan_digest=summary.plan_digest,
        backends=backends,
        targets=summary.targets,
        provided=dict(summary.provided),
        stages=_run_stages(summary, execution),
        selections=None if execution is None else execution["selections"],
        resumed_from=summary.resumed_from,
        resume=None if execution is None else execution["resume"],
        failure=None if summary.failure is None else summary.failure.to_document(),
        blocked_problems=tuple(
            ConfigProblem(path=problem["path"], message=problem["message"])
            for problem in summary.blocked_problems
        ),
        code_identity=summary.code_identity,
        environment=dict(summary.environment),
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        events=summary.events,
        notes=tuple(notes),
    )


def _run_stages(
    summary: RunSummary, execution: Mapping[str, Any] | None
) -> tuple[RuntimeRunStage, ...]:
    """Collect what is recorded per stage: from the events, refined by the execution record."""
    recorded: dict[str, dict[str, Any]] = {
        stage_id: {"outcome": "pending", "inputs": None, "output": None, "decision": None}
        for stage_id in summary.targets
    }
    for event in summary.events:
        if event.stage_id is None or not event.kind.startswith("stage_"):
            continue
        entry = recorded.setdefault(
            event.stage_id, {"outcome": "pending", "inputs": None, "output": None, "decision": None}
        )
        data = event.data
        if event.kind == "stage_started":
            entry["outcome"] = "started"
            entry["inputs"] = {name: tuple(ids) for name, ids in data["inputs"].items()}
        elif event.kind == "stage_reused":
            entry.update(outcome="reused", output=data["artifact"], decision=data["decision"])
        elif event.kind == "stage_completed":
            entry.update(
                outcome="completed",
                output=data["artifact"],
                decision=data["decision"],
                elapsed_s=data["elapsed_s"],
            )
        elif event.kind == "stage_failed":
            entry.update(outcome="failed", elapsed_s=data["elapsed_s"])
    if execution is not None:
        for stage in execution["stages"]:
            known = recorded.get(stage["stage_id"])
            if known is not None:
                known["inputs"] = {
                    name: tuple(ref["artifact_id"] for ref in refs)
                    for name, refs in stage["inputs"].items()
                }
    return tuple(
        RuntimeRunStage(
            stage_id=stage_id,
            outcome=entry["outcome"],
            inputs=entry["inputs"],
            output=entry["output"],
            decision=entry["decision"],
            elapsed_s=entry.get("elapsed_s"),
        )
        for stage_id, entry in recorded.items()
    )


__all__ = [
    "ResolvedPipelinePlan",
    "Runtime",
    "RuntimeBackend",
    "RuntimeCapability",
    "RuntimeComponent",
    "RuntimeEdit",
    "RuntimeExecutionEvent",
    "RuntimeExecutionResult",
    "RuntimePlanInput",
    "RuntimePlanStage",
    "RuntimePreflightReport",
    "RuntimeRunRecord",
    "RuntimeRunStage",
    "RuntimeRunSummary",
    "RuntimeStatus",
]
