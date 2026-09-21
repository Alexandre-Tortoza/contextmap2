"""Command-line interface: a thin adapter over the runtime services.

The CLI parses arguments, calls the runtime and prints what came back. It holds no
backend-specific branch and no scientific rule: configuration is resolved by
:func:`~contextmap.runtime.config.resolve_effective_config`, the topology by
:func:`~contextmap.runtime.pipeline.resolve_plan`, selections by
:func:`~contextmap.runtime.selection.resolve_selections` and execution by
:func:`~contextmap.runtime.pipeline.run_plan`. Every flag is translated into a
configuration override, so the CLI never becomes a second place that decides what runs.

Stage executors are not bundled: the executors of the real capabilities are supplied by
the caller of :func:`main` (tests, or a future entry point). Without one, a real run is
blocked by preflight with an explicit message and nothing runs; a dry run needs none.

Exit codes: ``0`` success, ``1`` the request was understood but cannot be satisfied
(invalid configuration, blocked preflight, failed stage, failed integrity check), ``2``
misuse of the command line.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from contextmap import __version__
from contextmap.runtime.catalog import CANONICAL_PROFILE_ID
from contextmap.runtime.config import (
    DEBUG_LEVELS,
    ConfigProblem,
    ConfigurationError,
    EffectiveConfig,
    read_effective_config,
    resolve_effective_config,
    write_effective_config,
)
from contextmap.runtime.errors import PipelineError, PreflightError
from contextmap.runtime.pipeline import (
    ExecutionPlan,
    ExecutionRecord,
    PipelinePlan,
    PreflightReport,
    StageExecutor,
    preflight,
    read_plan_document,
    resolve_plan,
    run_plan,
    write_execution_record,
    write_plan,
)
from contextmap.runtime.selection import ResolvedSelections, load_catalog, resolve_selections
from contextmap.shared import FileEntry, check_file_inventory

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

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
    environ: Mapping[str, str] | None
    module_available: Callable[[str], bool] | None
    out: TextIO
    err: TextIO

    def emit(self, document: Mapping[str, Any], lines: Sequence[str]) -> None:
        """Print a document as JSON, or its human-readable lines."""
        if self.args.json:
            print(json.dumps(document, indent=2, sort_keys=True), file=self.out)
        else:
            print("\n".join(lines), file=self.out)

    def fail(self, message: str, problems: Sequence[ConfigProblem] = ()) -> int:
        """Report a failure, as JSON on stdout or as text on stderr."""
        if self.args.json:
            document = {
                "ok": False,
                "error": message,
                "problems": [{"path": p.path, "message": p.message} for p in problems],
            }
            print(json.dumps(document, indent=2, sort_keys=True), file=self.out)
        else:
            print(f"error: {message}", file=self.err)
            for problem in problems:
                print(f"  - {problem}", file=self.err)
        return EXIT_FAILED


def main(
    argv: Sequence[str] | None = None,
    *,
    executors: Mapping[str, StageExecutor] | None = None,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one CLI command.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.
        executors: One executor per stage this process can execute. There is no default:
            a real run without them is blocked by preflight.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
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
            environ=environ,
            module_available=module_available,
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
        except (PipelineError, _Failure) as error:
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

    inspect = commands.add_parser("inspect", help="show the configuration, the plan or an artifact")
    subjects = inspect.add_subparsers(dest="subject", required=True, metavar="SUBJECT")
    subjects.add_parser(
        "config", parents=[config], help="show the effective configuration and its digest"
    ).set_defaults(handler=_inspect_config)
    subjects.add_parser(
        "plan", parents=[config], help="show the resolved topology without executing anything"
    ).set_defaults(handler=_inspect_plan)
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


def _execution_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the resolved plan, inputs, outputs and preflight; load nothing, run nothing",
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
    if args.sequence is not None:
        overrides.append(f"inputs.sequence={json.dumps(args.sequence)}")
    for stage, references in args.select or []:
        value = references if len(references) > 1 else references[0]
        overrides.append(f"inputs.selections.{stage}={json.dumps(value)}")
    if args.workspace is not None:
        overrides.append(f"resources.workspace={json.dumps(args.workspace)}")
    if args.device is not None:
        overrides.append(f"resources.device={json.dumps(args.device)}")
    if args.debug_level is not None:
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

    if args.dry_run:
        # Um dry-run não confere executores: mostra o que a configuração resolve.
        report = preflight(
            execution, environ=session.environ, module_available=session.module_available
        )
        missing = [s.stage_id for s in execution.stages if s.stage_id not in session.executors]
        document = {
            "effective_config": effective.to_document(),
            "plan": plan.to_document(),
            "plan_digest": plan.digest,
            "scope": {
                "run": [s.stage_id for s in execution.stages],
                "provided": {
                    stage: [ref.artifact_id for ref in refs]
                    for stage, refs in execution.reused.items()
                },
            },
            "selections": None if resolved is None else resolved.to_document(),
            "preflight": {"ok": report.ok, "problems": _problem_documents(report.problems)},
            "executors": {"registered": sorted(session.executors), "missing": missing},
        }
        lines = [
            *_config_lines(effective),
            "",
            *_plan_lines(plan, execution),
            "",
            *_preflight_lines(report),
            *(
                [f"executors: none registered for {', '.join(missing)} (a real run is blocked)"]
                if missing
                else ["executors: registered for every stage to run"]
            ),
        ]
        session.emit(document, lines)
        return EXIT_OK if report.ok else EXIT_FAILED

    record = run_plan(
        execution,
        session.executors,
        environ=session.environ,
        module_available=session.module_available,
    )
    assert workspace is not None  # exigido acima
    run_directory = _publish_run(Path(workspace), effective, plan, record)
    result = {
        "ok": True,
        "run_directory": str(run_directory),
        "execution": record.to_document(),
    }
    lines = [
        f"run completed: {run_directory}",
        f"  order: {' -> '.join(record.order)}",
        *(f"  {stage.stage_id}: {stage.output.artifact_id}" for stage in record.stages),
    ]
    session.emit(result, lines)
    return EXIT_OK


def _publish_run(
    workspace: Path, effective: EffectiveConfig, plan: PipelinePlan, record: ExecutionRecord
) -> Path:
    """Write the run records into a fresh ``runtime/run-NNNN`` directory, atomically.

    The records are written into a temporary directory that is renamed into place only when
    complete, so an interrupted write never leaves something that looks like a run.
    """
    base = workspace / "runtime"
    base.mkdir(parents=True, exist_ok=True)
    taken = [
        int(path.name.removeprefix("run-"))
        for path in base.iterdir()
        if path.name.startswith("run-") and path.name.removeprefix("run-").isdigit()
    ]
    index = max(taken, default=0) + 1
    temporary = base / f".tmp-run-{os.getpid()}-{index:04d}"
    write_effective_config(effective, temporary)
    write_plan(plan, temporary)
    write_execution_record(record, temporary)
    while True:
        final = base / f"run-{index:04d}"
        try:
            os.rename(temporary, final)
        except OSError:
            if not final.exists():
                raise
            index += 1  # outro processo publicou este índice primeiro
            continue
        return final


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
    lines.append(f"  policies: debug_level={config.policies.debug_level}")
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
