"""Shared skeleton of experiment drivers: run layout, retained outputs and child measurement.

An experiment driver is a research script under ``experiments/<experiment>/`` that runs stages,
or arms of one stage, by hand. Every round of an experiment used to copy its drivers and edit
the same three things in them: the run number baked into the output directory, the artifact
locations and the configuration label; the workspace path; and the runner that measures the
wall time and peak memory of a child process. This module is the single, tested version of those
three things, so a new round changes one run number instead of a copy of every script.

Only what the copies actually duplicated lives here. The executor a stage runs, its policies and
its input ``ArtifactRef`` values stay in the driver: they are what each experiment is about. The
historical copies under ``experiments/`` stay as they were executed. See
``src/contextmap/evaluation/docs/experiments.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXPERIMENTS_DIRECTORY = "experiments"
_OUTPUTS_DIRECTORY = "outputs"
_CHECKOUT_MARKER = "pyproject.toml"

# O processo de medição é um interpretador novo que só importa este módulo e executa um único
# filho; a leitura do RUSAGE_CHILDREN acontece lá, nunca no processo de quem chama.
_MEASURING_ENTRY = (
    "import json, sys\n"
    "from contextmap.evaluation.experiment_driver import _measure_as_only_child\n"
    "print(json.dumps(_measure_as_only_child(sys.argv[1:])))\n"
)


class ExperimentDriverError(Exception):
    """Raised when a driver run cannot be laid out or a child process cannot be measured."""


@dataclass(frozen=True, kw_only=True)
class DriverRun:
    """One numbered run of an experiment driver and where its artifacts are retained.

    The layout is the runtime's own, ``<outputs_root>/<experiment>/run-NNNN/<name>``, with the
    outputs root as the workspace: ``StageRequest.run_number()`` reads the run number back from
    :meth:`output_dir`, and ``StageRequest.directory_of()`` opens a :meth:`location`.

    Attributes:
        experiment: Directory of the experiment under the outputs root, the ``<dataset>`` level
            of a runtime location (for example ``"e2e-real"``).
        run_number: Ordinal of the run, from 1. It is the only value a new round changes: every
            directory, location and configuration identity of the run is derived from it.
        outputs_root: Absolute root the run writes under, the workspace of its stage requests.

    Raises:
        ExperimentDriverError: If the experiment is not one directory name, the run number is not
            a positive integer, or the root is relative.
    """

    experiment: str
    run_number: int
    outputs_root: Path

    def __post_init__(self) -> None:
        """Validate the run before any path is derived from it."""
        _require_directory_name(self.experiment, what="experiment")
        # bool é subclasse de int, mas `True` não é um número de run.
        if (
            isinstance(self.run_number, bool)
            or not isinstance(self.run_number, int)
            or self.run_number < 1
        ):
            raise ExperimentDriverError(
                f"the run number must be a positive integer, got {self.run_number!r}"
            )
        if not self.outputs_root.is_absolute():
            raise ExperimentDriverError(
                f"the outputs root must be absolute, got {str(self.outputs_root)!r}: a relative "
                "root would depend on the directory the driver is started from"
            )

    @classmethod
    def for_driver(
        cls,
        driver: Path,
        *,
        experiment: str,
        run_number: int,
        outputs_root: Path | None = None,
    ) -> DriverRun:
        """Return the run of a driver script, retained under its checkout's ``outputs/``.

        Args:
            driver: The driver script, normally ``Path(__file__)``.
            experiment: Directory of the experiment under the outputs root.
            run_number: Ordinal of the run, from 1.
            outputs_root: An explicit root, for example a temporary directory for a smoke run.
                ``None`` retains the run under the ``outputs/`` directory of the checkout whose
                ``experiments/`` holds the driver: the persistent, untracked root every real run
                writes to, so its artifacts can be inspected again without re-executing it.

        Returns:
            The run.

        Raises:
            ExperimentDriverError: If no root is given and the driver is not under the
                ``experiments/`` directory of a checkout, or the run is invalid.
        """
        root = _checkout_outputs_root(driver) if outputs_root is None else outputs_root
        return cls(experiment=experiment, run_number=run_number, outputs_root=root)

    @property
    def run_id(self) -> str:
        """Return the run directory name, ``run-NNNN``, as the runtime names its runs."""
        return f"run-{self.run_number:04d}"

    def location(self, name: str) -> str:
        """Return where one artifact of this run lives, relative to the outputs root.

        It is the ``ArtifactRef.location`` of the artifact and the ``run_dir`` a report records:
        relative, so it stays valid wherever the checkout is.

        Args:
            name: Directory of the artifact: a stage id, or an arm id when several arms of one
                stage run side by side.

        Returns:
            ``<experiment>/run-NNNN/<name>``, written with ``/``.

        Raises:
            ExperimentDriverError: If the name is not one directory name.
        """
        _require_directory_name(name, what="artifact name")
        return f"{self.experiment}/{self.run_id}/{name}"

    def output_dir(self, name: str) -> Path:
        """Return the directory one artifact of this run is published into.

        Args:
            name: Directory of the artifact, as in :meth:`location`.

        Returns:
            ``<outputs_root>/<experiment>/run-NNNN/<name>``. It does not exist yet: the
            capability's writer creates and finalizes it.

        Raises:
            ExperimentDriverError: If the name is not one directory name.
        """
        return self.outputs_root.joinpath(*self.location(name).split("/"))

    def config_digest(self, name: str) -> str:
        """Return the configuration identity a stage request for one artifact carries.

        A hand-built driver has no effective-configuration document to hash, so its
        configuration identity is a label. Deriving it from the run number gives a re-execution
        another identity, so an artifact is never taken for an earlier one. Deriving it from the
        artifact name keeps arms of one stage apart: the runtime's execution identity is the
        stage, this value and the input hashes, and two arms over the same inputs that differ
        only in configuration would otherwise share it.

        Args:
            name: Directory of the artifact, as in :meth:`location`.

        Returns:
            The artifact's :meth:`location`, unique per experiment, run and artifact.

        Raises:
            ExperimentDriverError: If the name is not one directory name.
        """
        return self.location(name)


@dataclass(frozen=True, kw_only=True)
class ChildProcessMeasurement:
    """Wall time and peak memory of one measured child process.

    Attributes:
        command: The command that ran.
        returncode: Its exit status. A child that fails is measured, never raised.
        wall_time_seconds: Elapsed wall time from start to exit on a monotonic clock, the
            child's interpreter start-up included.
        peak_rss_kib: Peak resident set size in KiB (``ru_maxrss`` on Linux) of the largest
            single process among the child and the descendants it waited for: the peak of one
            process, never a sum over a process tree.
        stdout: What the child wrote to its standard output.
        stderr: What the child wrote to its standard error.
    """

    command: tuple[str, ...]
    returncode: int
    wall_time_seconds: float
    peak_rss_kib: int
    stdout: str
    stderr: str

    def to_record(self) -> dict[str, Any]:
        """Return the fields both earlier per-child runners reported, in their exact form.

        ``peak_rss_mb`` is ``peak_rss_kib / 1024`` (MiB) rounded to 0.1 and the wall time is
        rounded to the millisecond, as in the resource profile of issue #181 and the Sensor
        Association scaling report of issue #564, so new numbers stay comparable with them.

        Returns:
            ``returncode``, ``wall_time_seconds`` and ``peak_rss_mb``.
        """
        return {
            "returncode": self.returncode,
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "peak_rss_mb": round(self.peak_rss_kib / 1024, 1),
        }


def measure_child_process(command: Sequence[str]) -> ChildProcessMeasurement:
    """Run a command as the only child of a fresh process and measure it.

    The peak comes from ``resource.getrusage(RUSAGE_CHILDREN).ru_maxrss``, the method of the
    issue #181 resource profile. That value is a running maximum over every child a process has
    ever waited for: read from one long-lived parent, it would charge each measured stage or arm
    the peak of every earlier one, and of anything else that parent ran. So it is never read
    here. A fresh measuring process (this interpreter, ``sys.executable``) runs ``command`` as
    its only child and reports the reading, and it refuses to measure if it has already waited
    for any other child. Calling this function repeatedly from one parent is therefore safe.

    The command inherits this process's environment and working directory. To run it under
    another revision's sources, put the variable in the command (``["env", "PYTHONPATH=...",
    ...]``): ``env`` replaces itself with the target, so the target is still the measured child.

    Args:
        command: The program and its arguments.

    Returns:
        The measurement, whatever the child's exit status.

    Raises:
        ExperimentDriverError: If this is not Linux (``ru_maxrss`` is KiB only there), the
            command is empty, or the measuring process fails, for example because the command
            cannot be started.
    """
    # ru_maxrss é KiB no Linux e bytes no macOS: ler em outra plataforma inflaria o pico 1024
    # vezes em silêncio.
    if not sys.platform.startswith("linux"):
        raise ExperimentDriverError(
            f"peak RSS is read from ru_maxrss, whose unit is KiB only on Linux, not on "
            f"{sys.platform!r}"
        )
    if not command:
        raise ExperimentDriverError("an empty command cannot be measured")
    measuring = subprocess.run(
        [sys.executable, "-c", _MEASURING_ENTRY, *command],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if measuring.returncode != 0:
        raise ExperimentDriverError(
            f"the measuring process failed (exit status {measuring.returncode}) for "
            f"{list(command)!r}: {measuring.stderr.strip()}"
        )
    record = json.loads(measuring.stdout)
    return ChildProcessMeasurement(
        command=tuple(command),
        returncode=int(record["returncode"]),
        wall_time_seconds=float(record["wall_time_seconds"]),
        peak_rss_kib=int(record["peak_rss_kib"]),
        stdout=str(record["stdout"]),
        stderr=str(record["stderr"]),
    )


def _measure_as_only_child(command: Sequence[str]) -> dict[str, Any]:
    """Run a command from this process and read its peak from ``RUSAGE_CHILDREN``.

    This is the body of the measuring process :func:`measure_child_process` starts. The reading
    is the command's own peak only while this process has never waited for another child, so
    that is checked, never assumed.

    Args:
        command: The program and its arguments.

    Returns:
        The JSON-compatible measurement.

    Raises:
        ExperimentDriverError: If this process has already waited for a child.
    """
    # Import local: `resource` só existe em POSIX, e o pacote de avaliação continua importável
    # onde ele não existe.
    import resource

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if before != 0:
        raise ExperimentDriverError(
            f"this process already waited for a child (peak {before} KiB): RUSAGE_CHILDREN is "
            "a running maximum, so the next child's own peak could not be told apart"
        )
    start = time.perf_counter()
    completed = subprocess.run(
        list(command), capture_output=True, encoding="utf-8", errors="replace", check=False
    )
    wall_time_seconds = time.perf_counter() - start
    return {
        "returncode": completed.returncode,
        "wall_time_seconds": wall_time_seconds,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _checkout_outputs_root(driver: Path) -> Path:
    """Return the ``outputs/`` directory of the checkout whose ``experiments/`` holds a driver.

    The checkout is recognized by its layout, never by an absolute path: the nearest ancestor of
    the driver named ``experiments`` whose parent has a ``pyproject.toml``. Nothing is guessed
    otherwise, so a temporary directory is never chosen implicitly.

    Args:
        driver: The driver script.

    Returns:
        ``<checkout>/outputs``.

    Raises:
        ExperimentDriverError: If the driver is not under the ``experiments/`` directory of a
            checkout.
    """
    for ancestor in driver.resolve().parents:
        if (
            ancestor.name == _EXPERIMENTS_DIRECTORY
            and (ancestor.parent / _CHECKOUT_MARKER).is_file()
        ):
            return ancestor.parent / _OUTPUTS_DIRECTORY
    raise ExperimentDriverError(
        f"{driver} is not under the {_EXPERIMENTS_DIRECTORY}/ directory of a checkout: pass an "
        "explicit outputs root"
    )


def _require_directory_name(value: str, *, what: str) -> None:
    """Refuse a value that is not exactly one directory name.

    Raises:
        ExperimentDriverError: If the value is empty, ``.``/``..`` or has a path separator.
    """
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ExperimentDriverError(f"the {what} must be one directory name, got {value!r}")
