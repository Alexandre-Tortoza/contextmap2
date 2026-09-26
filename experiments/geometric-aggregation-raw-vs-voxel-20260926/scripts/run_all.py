"""Run every arm of the raw vs voxel-aggregation experiment and assemble the report (#624).

Each step runs as its own fresh OS process through ``_run_one.py``, so its wall time and
peak RSS are its own. The arms differ only in resolution:

    A   raw GeometricMapArtifact, as it is on disk        (baseline, never rewritten)
    B   voxel centroid, r1 = --cells[0]
    C   voxel centroid, r2 = --cells[1]
    D   voxel centroid, r3 = --cells[2]                  (optional)

With ``--sequence``, ``--trajectory`` and ``--perception`` the paired 2D→3D association runs
too: Sensor Association over A and over every derived arm, then ``compare.py`` per arm.

Usage:
    python run_all.py <output-root> --raw-map DIR [--cells 0.05 0.10 [0.20]]
        [--sequence DIR --trajectory DIR --perception DIR [--window frozen|all]]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextmap.geometric_mapping import GeometricMapArtifactReader, MapArtifactError

HERE = Path(__file__).resolve().parent
RUN_ONE = HERE / "_run_one.py"
ARM_NAMES = ("B", "C", "D")


def _environment() -> dict[str, Any]:
    try:
        total_kb = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal")
            )
        )
    except (OSError, StopIteration, ValueError):
        total_kb = 0
    return {
        "cpu_count": len(os.sched_getaffinity(0)),
        "ram_gib": round(total_kb / 1024 / 1024, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _commit() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=HERE, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


class StepFailedError(Exception):
    """Raised when a step failed or produced an invalid result; the run stops there."""

    def __init__(self, failures: list[str]) -> None:
        """Keep what failed, for the report."""
        super().__init__("; ".join(failures))
        self.failures = failures


def step_failures(step: str, record: Mapping[str, Any]) -> list[str]:
    """Say why a step's record is not valid evidence; empty when it is.

    A step fails when it exited non-zero, left no report, or reported integrity or
    chunking problems: a result that does not verify is never data for the report.
    """
    failures = []
    if record.get("returncode") != 0:
        failures.append(f"{step}: exited with status {record.get('returncode')}")
    report = record.get("arm_report")
    if not isinstance(report, Mapping):
        failures.append(f"{step}: produced no report")
        return failures
    if report.get("integrity_problems"):
        failures.append(f"{step}: integrity problems {report['integrity_problems']}")
    chunking = report.get("chunking_check") or {}
    if chunking.get("problems"):
        failures.append(f"{step}: chunking changed the aggregation {chunking['problems']}")
    return failures


def _checked_step(
    target: dict[str, Any], key: str, step: str, script: str, *arguments: str
) -> None:
    """Run a step, keep its record in ``target[key]`` and stop the run if it is invalid.

    Raises:
        StepFailedError: If :func:`step_failures` finds anything.
    """
    record = _step(script, *arguments)
    target[key] = record
    failures = step_failures(step, record)
    if failures:
        raise StepFailedError(failures)


def _step(script: str, *arguments: str) -> dict[str, Any]:
    command = [sys.executable, str(RUN_ONE), sys.executable, str(HERE / script), *arguments]
    print(f"--- {' '.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr, flush=True)
    try:
        record: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError:
        record = {"returncode": result.returncode, "stdout_tail": result.stdout[-2000:]}
    return record


def _source_snapshot(raw_map: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Record the raw artifact's identity once, before any step, or say why it cannot be read.

    It is never re-read after a step failed: an unreadable raw artifact becomes a failure in
    the report instead of an exception that would leave no report at all.
    """
    try:
        return _source(raw_map), []
    except (MapArtifactError, OSError, ValueError, KeyError) as error:
        return None, [f"source: {type(error).__name__}: {error}"]


def _source(raw_map: Path) -> dict[str, Any]:
    with GeometricMapArtifactReader(raw_map) as reader:
        manifest = reader.manifest
        return {
            "path": str(raw_map),
            "run_id": str(manifest.run_id),
            "map_id": str(manifest.map_id),
            "sequence_artifact_id": str(manifest.sequence_artifact_id),
            "selection_id": manifest.selection_id,
            "trajectory_id": str(manifest.trajectory_id),
            "state_estimation_run_id": manifest.state_estimation_run_id,
            "calibration_identity": manifest.calibration_identity,
            "configuration_fingerprint": manifest.configuration_fingerprint,
            "aggregation_rule": manifest.aggregation_rule,
            "point_count": manifest.point_count,
            "scan_count": manifest.scan_count,
            "motion_correction_policy": reader.read_record("config.json")[
                "motion_correction_policy"
            ],
        }


def main() -> int:
    """Run every arm and write ``report.json``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--raw-map", type=Path, required=True)
    parser.add_argument("--cells", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--perception", type=Path)
    parser.add_argument("--window", default="frozen", choices=("frozen", "all"))
    options = parser.parse_args()
    if not 2 <= len(options.cells) <= len(ARM_NAMES):
        raise SystemExit("between two and three resolutions are compared (arms B, C and D)")
    associating = None not in (options.sequence, options.trajectory, options.perception)

    root = options.output_root
    # Um diretório de saída existente é recusado antes de tudo: o relatório de outro run nunca
    # é sobrescrito. É o único caso em que não sai `report.json`.
    root.mkdir(parents=True, exist_ok=False)
    code = _commit()
    origin = [str(value) for value in options.origin]
    arms: dict[str, dict[str, Any]] = {}
    source, failures = _source_snapshot(options.raw_map)
    try:
        if failures:
            raise StepFailedError(failures)
        arms["A"] = {"cell_m": None}
        _checked_step(
            arms["A"],
            "geometry",
            "arm.py A",
            "arm.py",
            "A",
            str(root / "A"),
            "--raw-map",
            str(options.raw_map),
        )
        for name, cell in zip(ARM_NAMES, options.cells, strict=False):
            arms[name] = {"cell_m": cell}
            _checked_step(
                arms[name],
                "geometry",
                f"arm.py {name}",
                "arm.py",
                name,
                str(root / name),
                "--raw-map",
                str(options.raw_map),
                "--cell-m",
                repr(cell),
                "--origin",
                *origin,
                "--code-version",
                code["commit"],
            )
        if associating:
            _associate_arms(root, arms, options)
    except StepFailedError as error:
        # Fail-fast: nada depois de um passo inválido roda, e o relatório diz que está incompleto.
        failures = error.failures
        print(f"FAILED: {failures}", file=sys.stderr, flush=True)

    report = {
        "experiment": "geometric-aggregation-raw-vs-voxel-20260926",
        "issue": 624,
        "code": code,
        "environment": _environment(),
        "source": source,
        "grid_origin_m": list(options.origin),
        "association": None
        if not associating
        else {
            "sequence": str(options.sequence),
            "trajectory": str(options.trajectory),
            "perception": str(options.perception),
            "window": options.window,
        },
        "status": "failed" if failures else "complete",
        "failures": failures,
        "arms": arms,
    }
    (root / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"report: {root / 'report.json'} ({report['status']})")
    return 1 if failures else 0


def _associate_arms(root: Path, arms: dict[str, dict[str, Any]], options: Any) -> None:
    """Associate over every arm's geometry, then compare each derived arm with arm A.

    Raises:
        StepFailedError: At the first step that fails or does not verify.
    """
    shared = [
        "--raw-map",
        str(options.raw_map),
        "--sequence",
        str(options.sequence),
        "--trajectory",
        str(options.trajectory),
        "--perception",
        str(options.perception),
        "--window",
        options.window,
    ]
    for name in arms:
        extra = (
            [] if name == "A" else ["--voxel-aggregation", str(root / name / "voxel_aggregation")]
        )
        _checked_step(
            arms[name],
            "association",
            f"associate.py {name}",
            "associate.py",
            name,
            str(root / name / "sensor_association"),
            *shared,
            *extra,
        )
    for name in arms:
        if name == "A":
            continue
        _checked_step(
            arms[name],
            "association_stability",
            f"compare.py {name}",
            "compare.py",
            name,
            str(root / name / "association-stability.json"),
            "--raw-map",
            str(options.raw_map),
            "--voxel-aggregation",
            str(root / name / "voxel_aggregation"),
            "--raw-association",
            str(root / "A" / "sensor_association"),
            "--arm-association",
            str(root / name / "sensor_association"),
        )


if __name__ == "__main__":
    sys.exit(main())
