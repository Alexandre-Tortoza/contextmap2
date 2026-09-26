"""Command-line interface: a thin adapter over the runtime services.

The CLI parses arguments, calls the runtime and prints what came back. It holds no
backend-specific branch and no scientific rule: configuration is resolved by
:func:`~contextmap.runtime.config.resolve_effective_config`, the topology by
:func:`~contextmap.runtime.pipeline.resolve_plan`, selections by
:func:`~contextmap.runtime.selection.resolve_selections` and execution by
:func:`~contextmap.runtime.pipeline.run_plan`. Every flag is translated into a
configuration override, so the CLI never becomes a second place that decides what runs.

Stage executors: for every command that runs or previews a plan, the executors that
:func:`~contextmap.runtime.composition.compose_executors` can build from the resolved
configuration (``state_estimation``, ``geometric_mapping``, ``sensor_association``,
``semantic_fusion``, ``entity_resolution`` and ``spatial_relations`` unconditionally;
``visual_perception`` once a runtime is available for every one of its selected backends
that has no bundled loader -- SAM2, SAM3, Qwen, Gemini and Florence-2 today) are composed
automatically, so the installed ``contextmap`` binary executes them with **no Python
wrapper**: a runtime provider can be supplied either through ``main(providers=...)`` (a
Python embedder only) or, for the installed binary itself, declared in configuration as a
``resources.providers`` target (a ``"module:attribute"`` string, resolved by
:func:`~contextmap.runtime.composition.resolve_provider`; see ``docs/composition.md`` and
``docs/configuration.md``). The same mechanism covers any other optional component that
needs a model runtime or client the repository does not load itself (for example
``entity_resolution.appearance``), keyed by component identity (``"<capability>.<slot>"``).
Executors supplied by the caller of :func:`main` (tests, or a future embedder) are merged
on top and always win, so an explicit injection can override or extend what was composed --
including ``ingestion``, whose
:class:`~contextmap.runtime.ingestion_service.IngestionStageExecutor` needs a concrete
request that is never part of a configuration (see ``contextmap ingest``).
``point_representation`` has no real executor yet: without an injection, a real run of that
stage is blocked by preflight with an explicit message; a dry run needs none.

Exit codes: ``0`` success, ``1`` the request was understood but cannot be satisfied
(invalid configuration, blocked preflight, failed stage, failed integrity check), ``2``
misuse of the command line.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from contextmap import __version__
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.catalog import CANONICAL_PROFILE_ID
from contextmap.runtime.composition import RuntimeProvider, compose, compose_executors
from contextmap.runtime.config import (
    DEBUG_LEVELS,
    ConfigProblem,
    ConfigurationError,
    EffectiveConfig,
    read_effective_config,
    resolve_effective_config,
    resolve_secrets,
)
from contextmap.runtime.errors import (
    BackendUnavailableError,
    CompositionError,
    PipelineError,
    PreflightError,
    RunRecordError,
)
from contextmap.runtime.ingestion_service import (
    IngestionRequest,
    IngestionResult,
    IngestionService,
    SourceAdapterFactory,
)
from contextmap.runtime.lifecycle import ExecutionEvent
from contextmap.runtime.pipeline import (
    ExecutionPlan,
    PipelinePlan,
    PreflightReport,
    StageExecutor,
    predict_reuse,
    preflight,
    read_plan_document,
    resolve_plan,
    run_plan,
    with_source_identities,
)
from contextmap.runtime.reuse import FileArtifactStore, ReusePolicy
from contextmap.runtime.runs import (
    RunJournal,
    RunSummary,
    check_resumable,
    dataset_directory,
    read_run,
    resume_plan,
)
from contextmap.runtime.selection import ResolvedSelections, load_catalog, resolve_selections
from contextmap.shared import FileEntry, check_file_inventory

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

_MANIFEST = "manifest.json"
_SUMMARY_KEYS = (
    "artifact_id",
    "run_id",
    "sequence_name",
    "schema_version",
    "created_at",
    "debug_level",
    "code_version",
)


class _UsageError(Exception):
    """Raised when the command line asks for something that cannot be done as written."""


class _Failure(Exception):
    """Raised for an expected, explainable failure of a well-formed request."""


@dataclass(frozen=True)
class _Session:
    """Everything a command needs besides its arguments."""

    args: argparse.Namespace
    executors: Mapping[str, StageExecutor]
    providers: Mapping[str, RuntimeProvider]
    environ: Mapping[str, str] | None
    module_available: Callable[[str], bool] | None
    verifier: Callable[[ArtifactRef], bool] | None
    adapter_factory: SourceAdapterFactory | None
    out: TextIO
    err: TextIO

    def emit(self, document: Mapping[str, Any], lines: Sequence[str]) -> None:
        """Print a document as JSON, or its human-readable lines."""
        if self.args.json:
            print(json.dumps(document, indent=2, sort_keys=True), file=self.out)
        else:
            print("\n".join(lines), file=self.out)

    def fail(
        self,
        message: str,
        problems: Sequence[ConfigProblem] = (),
        *,
        run_directory: Path | None = None,
    ) -> int:
        """Report a failure, as JSON on stdout or as text on stderr."""
        if self.args.json:
            document: dict[str, Any] = {
                "ok": False,
                "error": message,
                "problems": [{"path": p.path, "message": p.message} for p in problems],
            }
            if run_directory is not None:
                document["run_directory"] = str(run_directory)
            print(json.dumps(document, indent=2, sort_keys=True), file=self.out)
        else:
            print(f"error: {message}", file=self.err)
            for problem in problems:
                print(f"  - {problem}", file=self.err)
            if run_directory is not None:
                print(f"run record: {run_directory}", file=self.err)
        return EXIT_FAILED


def main(
    argv: Sequence[str] | None = None,
    *,
    executors: Mapping[str, StageExecutor] | None = None,
    providers: Mapping[str, RuntimeProvider] | None = None,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    verifier: Callable[[ArtifactRef], bool] | None = None,
    adapter_factory: SourceAdapterFactory | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one CLI command.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.
        executors: Executors to use in addition to, and in preference over, the ones
            :func:`~contextmap.runtime.composition.compose_executors` builds automatically
            from the resolved configuration for ``run``, ``stage`` and their dry runs. Pass
            an entry here to override a composed stage (for example with a test double) or
            to supply one composition cannot build on its own, such as ``ingestion``'s
            :class:`~contextmap.runtime.ingestion_service.IngestionStageExecutor`. A stage
            with neither a composed nor a supplied executor is blocked by preflight.
        providers: Model runtimes or clients for a backend with no bundled loader (SAM2, SAM3,
            Qwen, Gemini, Florence-2 and every other backend built through
            :meth:`~contextmap.runtime.composition._Context.runtime`, for example
            ``entity_resolution.appearance`` or ``visual_perception``'s four backends), keyed
            by component identity (``"<capability>.<slot>"``) in the exact shape
            :func:`~contextmap.runtime.composition.compose_executors` already expects. This is
            the Python-embedding path; the installed binary instead declares a
            ``resources.providers`` target in configuration for the same component (see the
            module docstring) -- an entry given here for a component that also has one
            declared still wins, and that override is recorded on the run's ``run_planned``
            event. Without either, a component that needs one composes as absent, exactly
            like an incomplete selection, never with a substitute.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
        verifier: Tells whether an indexed artifact still exists and is intact; the reuse
            flags need it, and only the owner of the executors can provide it.
        adapter_factory: Builds the source adapter for ``ingest``; by default it is composed
            from the configuration's selected adapter backend.
        stdout: Stream for results; defaults to ``sys.stdout``.
        stderr: Stream for errors; defaults to ``sys.stderr``.

    Returns:
        The process exit code.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            args = _build_parser().parse_args(argv)
        except SystemExit as stop:
            return stop.code if isinstance(stop.code, int) else EXIT_USAGE
        session = _Session(
            args=args,
            executors=executors or {},
            providers=providers or {},
            environ=environ,
            module_available=module_available,
            verifier=verifier,
            adapter_factory=adapter_factory,
            out=out,
            err=err,
        )
        try:
            return int(args.handler(session))
        except _UsageError as error:
            print(f"error: {error}", file=err)
            return EXIT_USAGE
        except ConfigurationError as error:
            return session.fail("invalid configuration", error.problems)
        except PreflightError as error:
            return session.fail("the plan cannot run", error.report.problems)
        except BackendUnavailableError as error:
            return session.fail("a selected backend is unavailable", error.problems)
        except (PipelineError, CompositionError, _Failure) as error:
            return session.fail(str(error))


# --- parser --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextmap",
        description="Resolve, plan, run and inspect ContextMap2 pipelines.",
    )
    parser.add_argument("--version", action="version", version=f"contextmap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    config = _config_options()
    run = commands.add_parser(
        "run",
        parents=[config],
        help="run the pipeline, or the stages selected with --stage",
        description="Resolve the configuration and run the pipeline (or a subgraph).",
    )
    run.add_argument(
        "--stage",
        action="append",
        dest="targets",
        metavar="STAGE",
        help="stage to produce; repeat for several (default: the complete pipeline)",
    )
    _execution_flags(run)
    run.set_defaults(handler=_run_command)

    stage = commands.add_parser(
        "stage",
        parents=[config],
        help="run exactly one stage, starting from explicitly selected upstream runs",
        description="Run one stage; select its upstream runs with --select and --catalog.",
    )
    stage.add_argument("stage", metavar="STAGE", help="the stage to produce")
    _execution_flags(stage)
    stage.set_defaults(handler=_stage_command)

    ingest = commands.add_parser(
        "ingest",
        parents=[_ingest_options()],
        help="ingest a recorded source into an immutable sequence artifact",
        description=(
            "Run canonical ingestion through the public ingestion service: preflight, then "
            "read, validate, synchronize and publish a SequenceArtifact."
        ),
    )
    ingest.set_defaults(handler=_ingest)

    inspect = commands.add_parser("inspect", help="show the configuration, the plan or an artifact")
    subjects = inspect.add_subparsers(dest="subject", required=True, metavar="SUBJECT")
    subjects.add_parser(
        "config", parents=[config], help="show the effective configuration and its digest"
    ).set_defaults(handler=_inspect_config)
    subjects.add_parser(
        "plan", parents=[config], help="show the resolved topology without executing anything"
    ).set_defaults(handler=_inspect_plan)
    run_record = subjects.add_parser(
        "run", help="show a run's lifecycle: status, failure record, events and environment"
    )
    _path_argument(run_record)
    run_record.add_argument(
        "--events", action="store_true", help="list every event, not only the summary"
    )
    run_record.set_defaults(handler=_inspect_run)
    artifact = subjects.add_parser(
        "artifact", help="summarise an artifact directory or a runtime document"
    )
    _path_argument(artifact)
    artifact.set_defaults(handler=_inspect_artifact)

    validate = commands.add_parser(
        "validate", help="check the integrity of an artifact directory or a runtime document"
    )
    _path_argument(validate)
    validate.set_defaults(handler=_validate)
    return parser


def _config_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("configuration (each flag is a configuration override)")
    group.add_argument(
        "-c",
        "--config",
        action="append",
        metavar="FILE",
        help="configuration file (.json or .toml); repeat for several, later ones win",
    )
    group.add_argument("--profile", default=CANONICAL_PROFILE_ID, metavar="ID", help="base profile")
    group.add_argument(
        "--set",
        action="append",
        metavar="PATH=VALUE",
        help="override one setting, for example policies.debug_level=full",
    )
    group.add_argument(
        "--sequence", metavar="ID", help="expected identity of the physical sequence"
    )
    group.add_argument(
        "--select",
        action="append",
        type=_parse_select,
        metavar="STAGE=REF[,REF...]",
        help="upstream run(s) of a stage: artifact ids, latest, or named:<name>",
    )
    group.add_argument(
        "--catalog", metavar="FILE", help="catalog file listing the runs to select from"
    )
    group.add_argument("--workspace", metavar="DIR", help="directory that receives run records")
    group.add_argument("--device", metavar="DEVICE", help="device handed to backends that have one")
    group.add_argument("--debug-level", choices=DEBUG_LEVELS, help="amount of debug evidence")
    _json_flag(group)
    return common


def _ingest_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # `ingest` não tem workspace nem run: o destino é o diretório final do artifact (--output-dir).
    common.set_defaults(workspace=None)
    group = common.add_argument_group("configuration")
    group.add_argument("-c", "--config", action="append", metavar="FILE", help="configuration file")
    group.add_argument("--profile", default=CANONICAL_PROFILE_ID, metavar="ID", help="base profile")
    group.add_argument("--set", action="append", metavar="PATH=VALUE", help="override one setting")
    group.add_argument(
        "--output-dir",
        metavar="DIR",
        help="final directory of the sequence artifact (must not exist)",
    )
    source = common.add_argument_group("source and request")
    source.add_argument("--source", required=True, metavar="PATH", help="the recorded source")
    source.add_argument(
        "--sequence-name", required=True, metavar="NAME", help="name of the sequence"
    )
    source.add_argument(
        "--topic",
        action="append",
        type=_parse_topic,
        metavar="KEY=TOPIC",
        help="a topic to read, for example rgb=/camera; repeat for several",
    )
    source.add_argument(
        "--required", action="append", metavar="KEY", help="a topic that must exist in the source"
    )
    source.add_argument("--clock-id", metavar="ID", help="identity of the shared header clock")
    source.add_argument(
        "--sync-reference", required=True, metavar="MODALITY", help="synchronization anchor"
    )
    source.add_argument(
        "--sync-tolerance-ns",
        required=True,
        type=int,
        metavar="N",
        help="synchronization tolerance, in nanoseconds",
    )
    source.add_argument(
        "--on-problems",
        choices=("fail", "warn"),
        default="fail",
        help="what to do with structural problems (default: fail)",
    )
    source.add_argument(
        "--no-duplicate-timestamps",
        action="store_true",
        help="report two observations of one clock at the same timestamp",
    )
    source.add_argument(
        "--no-source-hash", action="store_true", help="skip hashing the source bytes (recorded)"
    )
    source.add_argument(
        "--preflight", action="store_true", help="only check that the request can run"
    )
    _json_flag(common)
    return common


def _parse_topic(text: str) -> tuple[str, str]:
    key, separator, topic = text.partition("=")
    if not separator or not key or not topic:
        raise argparse.ArgumentTypeError(f"{text!r}: expected KEY=TOPIC")
    return key, topic


def _execution_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the resolved plan, inputs, outputs and preflight; load nothing, run nothing",
    )
    reuse = parser.add_argument_group("reuse and resume (need --reuse-index)")
    reuse.add_argument("--reuse-index", metavar="DIR", help="index of completed artifacts to reuse")
    reuse.add_argument(
        "--code-identity", metavar="ID", help="identity of the code producing the results"
    )
    reuse.add_argument(
        "--force",
        action="append",
        metavar="STAGE",
        help="always recompute this stage; repeat for several",
    )
    reuse.add_argument(
        "--resume",
        metavar="RUN",
        help="resume a failed, cancelled or interrupted run (a directory or a run id)",
    )


def _path_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="artifact directory or runtime document")
    _json_flag(parser)


def _json_flag(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def _parse_select(text: str) -> tuple[str, list[str]]:
    stage, separator, refs = text.partition("=")
    references = [ref for ref in refs.split(",") if ref]
    if not separator or not stage or not references:
        raise argparse.ArgumentTypeError(f"{text!r}: expected STAGE=REF[,REF...]")
    return stage, references


# --- configuration -------------------------------------------------------------------


def _overrides(args: argparse.Namespace) -> list[str]:
    """Translate the flags into configuration overrides; a flag beats an earlier --set."""
    overrides = list(args.set or [])
    # Nem todo comando tem todas as flags de configuração (`ingest` tem um subconjunto).
    if getattr(args, "sequence", None) is not None:
        overrides.append(f"inputs.sequence={json.dumps(args.sequence)}")
    for stage, references in getattr(args, "select", None) or []:
        value = references if len(references) > 1 else references[0]
        overrides.append(f"inputs.selections.{stage}={json.dumps(value)}")
    if args.workspace is not None:
        overrides.append(f"resources.workspace={json.dumps(args.workspace)}")
    if getattr(args, "device", None) is not None:
        overrides.append(f"resources.device={json.dumps(args.device)}")
    if getattr(args, "debug_level", None) is not None:
        overrides.append(f"policies.debug_level={json.dumps(args.debug_level)}")
    return overrides


def _effective(args: argparse.Namespace) -> EffectiveConfig:
    return resolve_effective_config(
        profile=args.profile, files=args.config or (), overrides=_overrides(args)
    )


def _scope(
    session: _Session, effective: EffectiveConfig, plan: PipelinePlan, targets: Sequence[str] | None
) -> tuple[ExecutionPlan, ResolvedSelections | None]:
    """Scope the plan, resolving the configured selections against the catalog."""
    inputs = effective.config.inputs
    if not inputs.selections:
        return plan.scope(targets=targets), None
    if session.args.catalog is None:
        raise _UsageError(
            "selections need --catalog FILE, a catalog listing the runs available to select from"
        )
    try:
        catalog = load_catalog(session.args.catalog)
    except ValueError as error:
        raise _Failure(str(error)) from error
    resolved = resolve_selections(plan, inputs, catalog)
    return plan.scope(targets=targets, selections=resolved), resolved


# --- commands ------------------------------------------------------------------------


def _inspect_config(session: _Session) -> int:
    effective = _effective(session.args)
    session.emit(effective.to_document(), _config_lines(effective))
    return EXIT_OK


def _inspect_plan(session: _Session) -> int:
    effective = _effective(session.args)
    plan = resolve_plan(effective)
    document = {
        "plan": plan.to_document(),
        "plan_digest": plan.digest,
        "problems": _problem_documents(plan.problems),
    }
    session.emit(document, [*_plan_lines(plan, None), *_problem_lines("plan", plan.problems)])
    return EXIT_FAILED if plan.problems else EXIT_OK


def _stage_command(session: _Session) -> int:
    return _run(session, [session.args.stage])


def _run_command(session: _Session) -> int:
    return _run(session, session.args.targets)


def _run(session: _Session, targets: Sequence[str] | None) -> int:
    args = session.args
    effective = _effective(args)
    workspace = effective.config.resources.workspace
    if not args.dry_run and workspace is None:
        raise _UsageError(
            "a real run persists its records: pass --workspace DIR (or resources.workspace)"
        )
    plan = resolve_plan(effective)
    execution, resolved = _scope(session, effective, plan, targets)
    provider_overrides: list[str] = []
    executors = _executors_for(session, effective, provider_overrides)
    reuse = _reuse_policy(session)
    if reuse is not None:
        # O dry-run não passa executores ao preflight; a fonte de cada estágio-fonte já vai na
        # política para que a previsão e o preflight vejam a mesma chave que o run usaria.
        reuse = with_source_identities(reuse, executors)
    if args.dry_run:
        return _dry_run(session, effective, plan, execution, resolved, reuse, executors)
    assert workspace is not None  # exigido acima
    _dataset_directory(effective, workspace)  # recusa cedo um run sem dataset
    return _execute(
        session, effective, execution, Path(workspace), reuse, executors, provider_overrides
    )


def _executors_for(
    session: _Session,
    effective: EffectiveConfig,
    provider_overrides: list[str] | None = None,
) -> Mapping[str, StageExecutor]:
    """Merge the executors composed from ``effective`` with the ones the caller injected.

    ``compose_executors`` builds every stage it genuinely can (today: ``state_estimation``,
    ``geometric_mapping``, ``sensor_association``, ``semantic_fusion`` and, once a runtime
    provider is available for every backend that needs one -- explicitly through
    ``session.providers``, or declared as a ``resources.providers`` target in the resolved
    configuration itself -- ``visual_perception``) from the resolved configuration alone, so
    the installed CLI runs them with no Python wrapper; a stage it cannot build for any
    reason is simply absent, never raised (see its own docstring). Whatever the caller of
    :func:`main` passed in ``session.executors`` (tests, a stage composition cannot build
    such as ``ingestion``, or an explicit override) is layered on top and always wins.

    Args:
        session: The current CLI session.
        effective: The resolved configuration.
        provider_overrides: When given, receives (by mutation) the component identities
            where ``session.providers`` won over a ``resources.providers`` target the
            configuration also declared -- so ``_run`` can pass it into ``run_plan`` for a
            real run to record it, exactly as it happened, in the run's own trail.
    """
    composed = compose_executors(
        effective,
        providers=session.providers,
        environ=session.environ,
        module_available=session.module_available,
        on_provider_override=None if provider_overrides is None else provider_overrides.append,
    )
    return {**composed, **session.executors}


def _dry_run(
    session: _Session,
    effective: EffectiveConfig,
    plan: PipelinePlan,
    execution: ExecutionPlan,
    resolved: ResolvedSelections | None,
    reuse: ReusePolicy | None,
    executors: Mapping[str, StageExecutor],
) -> int:
    """Show what the configuration resolves to, without loading, running or writing anything."""
    # Um dry-run não confere executores no preflight: mostra o que a configuração resolve.
    report = preflight(
        execution, environ=session.environ, module_available=session.module_available, reuse=reuse
    )
    missing = [s.stage_id for s in execution.stages if s.stage_id not in executors]
    predicted = {} if reuse is None else predict_reuse(execution, reuse)
    document = {
        "effective_config": effective.to_document(),
        "plan": plan.to_document(),
        "plan_digest": plan.digest,
        "scope": {
            "run": [s.stage_id for s in execution.stages],
            "provided": {
                stage: [ref.artifact_id for ref in refs] for stage, refs in execution.reused.items()
            },
        },
        "selections": None if resolved is None else resolved.to_document(),
        "reuse": {stage: decision.to_document() for stage, decision in predicted.items()},
        "preflight": {"ok": report.ok, "problems": _problem_documents(report.problems)},
        "executors": {"registered": sorted(executors), "missing": missing},
    }
    lines = [
        *_config_lines(effective),
        "",
        *_plan_lines(plan, execution),
        "",
        *(
            [
                "reuse (predicted):",
                *(f"  {s}: {d.kind} - {d.reason}" for s, d in predicted.items()),
                "",
            ]
            if predicted
            else []
        ),
        *_preflight_lines(report),
        *(
            [f"executors: none registered for {', '.join(missing)} (a real run is blocked)"]
            if missing
            else ["executors: registered for every stage to run"]
        ),
    ]
    session.emit(document, lines)
    return EXIT_OK if report.ok else EXIT_FAILED


def _execute(
    session: _Session,
    effective: EffectiveConfig,
    execution: ExecutionPlan,
    workspace: Path,
    reuse: ReusePolicy | None,
    executors: Mapping[str, StageExecutor],
    provider_overrides: Sequence[str] = (),
) -> int:
    """Run for real, journaling every step of the lifecycle into a fresh run directory."""
    args = session.args
    previous: Path | None = None
    if args.resume is not None:
        if reuse is None:
            raise _UsageError(
                "--resume reuses the completed stages: pass --reuse-index and --code-identity"
            )
        previous = _previous_run(_dataset_directory(effective, str(workspace)), args.resume)
        check_resumable(read_run(previous), execution)  # recusa antes de criar um run novo
    secrets = resolve_secrets(effective.config, environ=session.environ)
    journal = RunJournal.create(workspace, effective, execution, code_identity=args.code_identity)
    try:
        if previous is not None and reuse is not None:
            record = resume_plan(
                previous,
                execution,
                executors,
                reuse=reuse,
                environ=session.environ,
                module_available=session.module_available,
                provider_overrides=provider_overrides,
                journal=journal,
                redact=secrets.redact,
            )
        else:
            record = run_plan(
                execution,
                executors,
                environ=session.environ,
                module_available=session.module_available,
                provider_overrides=provider_overrides,
                reuse=reuse,
                journal=journal,
                redact=secrets.redact,
            )
    except PreflightError as error:
        return session.fail(
            "the plan cannot run", error.report.problems, run_directory=journal.directory
        )
    except PipelineError as error:
        return session.fail(str(error), run_directory=journal.directory)
    except KeyboardInterrupt:
        print(f"cancelled; run record: {journal.directory}", file=session.err)
        return EXIT_INTERRUPTED
    result = {
        "ok": True,
        "run_directory": str(journal.directory),
        "execution": record.to_document(),
    }
    lines = [
        f"run completed: {journal.directory}",
        f"  order: {' -> '.join(record.order)}",
        *(f"  {stage.stage_id}: {stage.output.artifact_id}" for stage in record.stages),
    ]
    if record.resume is not None:
        lines.append(
            f"  resumed {record.resume['from']}: reused {record.resume['reused']}, "
            f"recomputed {record.resume['recomputed']}"
        )
    session.emit(result, lines)
    return EXIT_OK


def _reuse_policy(session: _Session) -> ReusePolicy | None:
    """Build the reuse policy from the flags; an artifact verifier has to come from outside."""
    args = session.args
    if args.reuse_index is None:
        if args.force:
            raise _UsageError("--force applies to reuse: pass --reuse-index DIR")
        return None
    if session.verifier is None:
        raise _UsageError(
            "--reuse-index needs an artifact verifier: the CLI cannot tell whether an indexed "
            "artifact still exists, so the owner of the stage executors supplies it to main()"
        )
    if not args.code_identity:
        raise _UsageError(
            "reuse across runs needs --code-identity ID: it must be a decision, not an accident"
        )
    return ReusePolicy(
        store=FileArtifactStore(args.reuse_index, verify=session.verifier),
        code_identity=args.code_identity,
        force_recompute=frozenset(args.force or ()),
    )


def _dataset_directory(effective: EffectiveConfig, workspace: str) -> Path:
    """Resolve ``<workspace>/<dataset>``: a real run needs ``inputs.sequence`` to be placed."""
    try:
        return dataset_directory(workspace, effective.config.inputs.sequence)
    except ValueError as error:
        raise _UsageError(str(error)) from error


def _previous_run(datasets: Path, reference: str) -> Path:
    """Resolve ``--resume`` as a run directory, or as a run id under the dataset directory."""
    candidate = Path(reference)
    if candidate.is_dir():
        return candidate
    under = datasets / reference
    if under.is_dir():
        return under
    raise _Failure(f"no run record at {candidate} or {under}")


def _inspect_run(session: _Session) -> int:
    summary = read_run(session.args.path)
    session.emit(summary.to_document(), _run_lines(summary, events=session.args.events))
    return EXIT_OK


def _run_lines(summary: RunSummary, *, events: bool) -> list[str]:
    state = summary.status.value + (
        " (interrupted: no live process owns it)" if summary.interrupted else ""
    )
    lines = [
        f"run {summary.run_id}: {state}",
        f"  created: {summary.created_at}    updated: {summary.updated_at}",
        f"  plan: {summary.plan_digest}",
        f"  config: {summary.config_digest}",
        f"  stages to run: {', '.join(summary.targets) or '-'}",
        f"  completed: {', '.join(summary.completed_stages) or '-'}",
    ]
    if summary.resumed_from is not None:
        lines.append(f"  resumed from: {summary.resumed_from}")
    if summary.code_identity is not None:
        lines.append(f"  code identity: {summary.code_identity}")
    if summary.failure is not None:
        failure = summary.failure
        lines.append(
            f"  failure: {failure.category} in {failure.stage_id or 'the runner'}: "
            f"{failure.message} ({failure.exception_type})"
        )
    for problem in summary.blocked_problems:
        lines.append(f"  blocked: {problem['path']}: {problem['message']}")
    lines.extend(f"  note: {note}" for note in summary.notes)
    if summary.truncated:
        lines.append("  note: the last event was cut off (the process died while writing it)")
    lines.append(f"  environment: {json.dumps(dict(summary.environment), sort_keys=True)}")
    if events:
        lines.append("  events:")
        lines.extend(
            f"    {e.sequence:>3} {e.time} {e.kind}{' ' + e.stage_id if e.stage_id else ''}"
            for e in summary.events
        )
    return lines


def _ingest(session: _Session) -> int:
    """Ingest a recorded source through the public ingestion service."""
    args = session.args
    effective = _effective(args)
    output_dir = args.output_dir
    if output_dir is None:
        raise _UsageError("ingest publishes a sequence artifact: pass --output-dir DIR")
    adapter = effective.config.components.get("ingestion.source_adapter")
    if adapter is None or adapter.backend is None:
        raise _UsageError(
            "select the source adapter: set components.ingestion.source_adapter.backend "
            "in a configuration file or with --set"
        )
    document = {
        "source_path": args.source,
        "sequence_name": args.sequence_name,
        "topics": dict(args.topic or []),
        "required_topics": list(args.required or []),
        "timestamp_clock_id": args.clock_id,
        "synchronization": {
            "reference_modality": args.sync_reference,
            "tolerance_nanoseconds": args.sync_tolerance_ns,
        },
        "validation": {
            "allow_duplicate_timestamps": not args.no_duplicate_timestamps,
            "on_problems": args.on_problems,
        },
        "hash_source": not args.no_source_hash,
        "config_identity": effective.digest,
    }
    try:
        request = IngestionRequest.from_document(
            document, output_dir=output_dir, source_type=adapter.backend
        )
    except ValueError as error:
        raise _Failure(str(error)) from error
    factory = session.adapter_factory or _composed_adapter_factory(session, effective)
    service = IngestionService(factory)

    if args.preflight:
        report = service.preflight(request)
        lines = [
            f"ingestion preflight {report.identity}: {'ok' if report.ok else 'BLOCKED'}",
            *(f"  - {problem}" for problem in report.problems),
            *(f"  warning: {warning}" for warning in report.warnings),
        ]
        session.emit(report.to_document(), lines)
        return EXIT_OK if report.ok else EXIT_FAILED

    events: list[ExecutionEvent] = []

    class _Progress:
        def emit(self, event: ExecutionEvent) -> None:
            events.append(event)
            if not args.json:
                print(f"{event.kind}", file=session.err)

    secrets = resolve_secrets(effective.config, environ=session.environ)
    try:
        result = service.run(request, event_sink=_Progress(), redact=secrets.redact)
    except KeyboardInterrupt:
        print("cancelled; nothing was published", file=session.err)
        return EXIT_INTERRUPTED
    document_out = result.to_document()
    document_out["events"] = [event.to_document() for event in events]
    session.emit(document_out, _ingestion_lines(result))
    if result.status == "completed":
        return EXIT_OK
    return EXIT_INTERRUPTED if result.status == "cancelled" else EXIT_FAILED


def _composed_adapter_factory(
    session: _Session, effective: EffectiveConfig
) -> SourceAdapterFactory:
    """Compose the configured source adapter; a missing module or selection fails explicitly."""
    composed = compose(
        effective,
        stages=["ingestion"],
        environ=session.environ,
        module_available=session.module_available,
    )
    assert composed.source_adapter is not None  # o estágio `ingestion` foi composto
    return composed.source_adapter


def _ingestion_lines(result: IngestionResult) -> list[str]:
    lines = [f"ingestion {result.status}: {result.sequence_name}"]
    if result.failure is not None:
        failure = result.failure
        lines.append(f"  failure: {failure.category} during {failure.phase}: {failure.message}")
    if result.artifact_path is not None:
        lines.append(f"  artifact: {result.artifact_path}")
    counts = ", ".join(f"{name}={count}" for name, count in result.observation_counts.items())
    lines.append(f"  observations: {counts}")
    lines.extend(f"  warning: {warning}" for warning in result.warnings)
    metrics = result.metrics
    lines.append(
        f"  metrics: {metrics.elapsed_s}s, {metrics.files_written} files, "
        f"{metrics.bytes_written} bytes, {metrics.dropped_events} dropped events"
    )
    return lines


# --- artifacts -----------------------------------------------------------------------


def _inspect_artifact(session: _Session) -> int:
    examined = _examine(session.args.path)
    lines = [f"{examined['kind']}: {examined['path']}"]
    lines.extend(f"  {key}: {value}" for key, value in examined["summary"].items())
    lines.extend(_integrity_lines(examined["integrity"]))
    session.emit(examined, lines)
    return EXIT_OK if examined["integrity"]["ok"] else EXIT_FAILED


def _validate(session: _Session) -> int:
    examined = _examine(session.args.path)
    integrity = examined["integrity"]
    lines = [f"{examined['path']}: {'ok' if integrity['ok'] else 'FAILED'}"]
    lines.extend(_integrity_lines(integrity)[1:])
    session.emit(examined, lines)
    return EXIT_OK if integrity["ok"] else EXIT_FAILED


def _examine(path: Path) -> dict[str, Any]:
    """Inspect an artifact directory or a runtime document with standard-library tools only."""
    if not path.exists():
        raise _Failure(f"{path} does not exist")
    if path.is_dir() and (path / "status.json").is_file():
        return _examine_run(path)
    if path.is_dir():
        return _examine_manifest(path)
    if path.name == "effective_config.json":
        return _examine_document(path, "effective-config", _read_config_summary)
    if path.name in {"plan.json", "execution.json"}:
        kind = "plan" if path.name == "plan.json" else "execution-record"
        return _examine_document(path, kind, _read_plan_summary)
    raise _Failure(
        f"{path} is not recognised: pass an artifact directory containing {_MANIFEST}, or an "
        "effective_config.json, plan.json or execution.json runtime document"
    )


def _examine_run(directory: Path) -> dict[str, Any]:
    """Check a run record: its status and event log, and the digest of each document."""
    problems: list[str] = []
    summary: dict[str, Any] = {}
    try:
        run = read_run(directory)
    except RunRecordError as error:
        problems.append(str(error))
    else:
        summary = {
            "run_id": run.run_id,
            "status": run.status.value,
            "interrupted": run.interrupted,
            "events": len(run.events),
            "completed_stages": len(run.completed_stages),
        }
        problems.extend(run.notes)
    documents: tuple[tuple[str, Callable[[Path], object]], ...] = (
        ("effective_config.json", read_effective_config),
        ("plan.json", read_plan_document),
        ("execution.json", read_plan_document),
    )
    for name, read in documents:
        if (directory / name).is_file():
            try:
                read(directory / name)
            except (ConfigurationError, PipelineError) as error:
                problems.append(f"{name}: {error}")
    return {
        "kind": "run",
        "path": str(directory),
        "summary": summary,
        "integrity": {"ok": not problems, "problems": problems},
    }


def _examine_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / _MANIFEST
    problems: list[str] = []
    summary: dict[str, Any] = {}
    if not manifest_path.is_file():
        problems.append(f"missing {_MANIFEST} in {root}: not a finalized artifact")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            manifest = None
            problems.append(f"cannot read {_MANIFEST}: {error}")
        if isinstance(manifest, dict):
            summary = {key: manifest[key] for key in _SUMMARY_KEYS if key in manifest}
            inventory = _inventory(manifest.get("file_inventory"), problems)
            summary["files"] = len(inventory)
            summary["total_bytes"] = sum(entry.size_bytes for entry in inventory)
            problems.extend(check_file_inventory(root, inventory))
        elif manifest is not None:
            problems.append(f"{_MANIFEST} is not a JSON object")
    return {
        "kind": "artifact",
        "path": str(root),
        "summary": summary,
        "integrity": {"ok": not problems, "problems": problems},
    }


def _inventory(raw: object, problems: list[str]) -> list[FileEntry]:
    """Read the manifest's file inventory, refusing paths that leave the artifact."""
    if not isinstance(raw, list):
        problems.append(f"{_MANIFEST} has no file_inventory list")
        return []
    entries = []
    for item in raw:
        try:
            entry = FileEntry(
                path=item["path"], size_bytes=item["size_bytes"], content_hash=item["content_hash"]
            )
        except (KeyError, TypeError):
            problems.append(f"malformed file_inventory entry: {item!r}")
            continue
        relative = PurePosixPath(entry.path)
        if relative.is_absolute() or ".." in relative.parts:
            problems.append(f"file_inventory path leaves the artifact: {entry.path}")
            continue
        entries.append(entry)
    return entries


def _examine_document(
    path: Path, kind: str, read: Callable[[Path], dict[str, Any]]
) -> dict[str, Any]:
    problems: list[str] = []
    summary: dict[str, Any] = {}
    try:
        summary = read(path)
    except (ConfigurationError, PipelineError) as error:
        details = [problem.message for problem in getattr(error, "problems", ())]
        problems.extend(details or [str(error)])
    return {
        "kind": kind,
        "path": str(path),
        "summary": summary,
        "integrity": {"ok": not problems, "problems": problems},
    }


def _read_config_summary(path: Path) -> dict[str, Any]:
    effective = read_effective_config(path)
    return {
        "digest": effective.digest,
        "schema_version": effective.config.schema_version,
        "preset": effective.config.pipeline.preset,
        "layers": len(effective.sources),
    }


def _read_plan_summary(path: Path) -> dict[str, Any]:
    document = read_plan_document(path)
    summary = {"schema_version": document["schema_version"]}
    if "order" in document:
        summary["order"] = " -> ".join(document["order"])
    if "stages" in document:
        summary["stages"] = len(document["stages"])
    for key in ("preset", "config_digest", "plan_digest"):
        if key in document:
            summary[key] = document[key]
    return summary


# --- rendering -----------------------------------------------------------------------


def _problem_documents(problems: Sequence[ConfigProblem]) -> list[dict[str, str]]:
    return [{"path": problem.path, "message": problem.message} for problem in problems]


def _problem_lines(label: str, problems: Sequence[ConfigProblem]) -> list[str]:
    if not problems:
        return []
    return [f"{label} problems ({len(problems)}):", *(f"  - {problem}" for problem in problems)]


def _integrity_lines(integrity: Mapping[str, Any]) -> list[str]:
    lines = [f"  integrity: {'ok' if integrity['ok'] else 'FAILED'}"]
    lines.extend(f"    - {problem}" for problem in integrity["problems"])
    return lines


def _config_lines(effective: EffectiveConfig) -> list[str]:
    config = effective.config
    lines = [
        f"effective configuration {effective.digest}",
        f"  preset: {config.pipeline.preset}",
        "  layers: "
        + ", ".join(
            f"{source.kind} {source.identity}" if source.kind != "override" else "override"
            for source in effective.sources
        ),
        "  stages: "
        + ", ".join(
            f"{stage}{'' if enabled else ' (off)'}"
            for stage, enabled in config.pipeline.stages.items()
        ),
        "  backends:",
    ]
    lines.extend(
        f"    {component_id}: {component.backend or '(not selected)'}"
        for component_id, component in config.components.items()
    )
    selections = {stage: list(refs) for stage, refs in config.inputs.selections.items()}
    lines.append(
        f"  inputs: sequence={config.inputs.sequence or '-'}, selections={selections or '-'}"
    )
    lines.append(
        f"  resources: device={config.resources.device or '-'}, "
        f"workspace={config.resources.workspace or '-'}"
    )
    lines.append(
        f"  policies: debug_level={config.policies.debug_level}, "
        f"trajectory_mode={config.policies.trajectory_mode}"
    )
    return lines


def _plan_lines(plan: PipelinePlan, execution: ExecutionPlan | None) -> list[str]:
    lines = [f"topology {plan.preset_id} {plan.digest}"]
    running = {s.stage_id for s in execution.stages} if execution is not None else set()
    provided = execution.reused if execution is not None else {}
    for number, stage in enumerate(plan.stages, start=1):
        if not stage.available:
            state = "unavailable"
        elif stage.stage_id in running:
            state = "run"
        elif stage.stage_id in provided:
            state = "provided " + ", ".join(ref.artifact_id for ref in provided[stage.stage_id])
        else:
            state = "-" if execution is not None else ""
        header = f"  {number:>2}. {stage.stage_id}"
        lines.append(
            f"{header} [{state}] -> {stage.output}" if state else f"{header} -> {stage.output}"
        )
        for item in stage.inputs:
            many = " (several runs)" if item.multiple else ""
            lines.append(f"        <- {item.name}: {item.source} ({item.contract}){many}")
        for component_id, backend in stage.components.items():
            lines.append(f"        backend {component_id} = {backend or '(not selected)'}")
        if not stage.available:
            lines.append(f"        {stage.unavailable_reason}")
    return lines


def _preflight_lines(report: PreflightReport) -> list[str]:
    if report.ok:
        return ["preflight: ok"]
    return [f"preflight: BLOCKED ({len(report.problems)})", *(f"  - {p}" for p in report.problems)]
