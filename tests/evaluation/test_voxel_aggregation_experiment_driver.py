"""The #624 experiment driver never reports a failed or invalid arm as a finished experiment.

The drivers live under ``experiments/`` (research scripts, outside the package), but whether
their exit status and report can be trusted is behavior: a partial run that looks complete
would be read as evidence.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from geometric_mapping_builders import make_plan, write_run

from contextmap.geometric_mapping import ScanVoxelPolicy

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "geometric-aggregation-raw-vs-voxel-20260926"
    / "scripts"
)


def _run_all(output: Path, raw: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_all.py"),
            str(output),
            "--raw-map",
            str(raw),
            "--cells",
            "0.5",
            "1.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _report(output: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((output / "report.json").read_text(encoding="utf-8"))
    return record


def _driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("voxel_run_all", SCRIPTS / "run_all.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_complete_run_exits_zero_and_says_so(tmp_path: Path) -> None:
    raw = write_run(tmp_path / "raw", make_plan())
    raw.close()

    result = _run_all(tmp_path / "out", tmp_path / "raw" / "run-0001")

    assert result.returncode == 0, result.stderr
    report = _report(tmp_path / "out")
    assert report["status"] == "complete"
    assert report["failures"] == []
    assert set(report["arms"]) == {"A", "B", "C"}


def test_a_failed_arm_stops_the_run_and_exits_non_zero(tmp_path: Path) -> None:
    # Um mapa já agregado dentro dos scans não é bruto: a derivação do braço B é recusada.
    raw = write_run(tmp_path / "raw", make_plan(), aggregation=ScanVoxelPolicy(cell_m=0.5))
    raw.close()

    result = _run_all(tmp_path / "out", tmp_path / "raw" / "run-0001")

    assert result.returncode != 0
    report = _report(tmp_path / "out")
    assert report["status"] == "failed"
    assert any(failure.startswith("arm.py B") for failure in report["failures"])
    assert "C" not in report["arms"]


@pytest.mark.parametrize(
    "record",
    [
        {"returncode": 1, "arm_report": {}},
        {"returncode": 0},
        {"returncode": 0, "arm_report": {"integrity_problems": ["content hash mismatch"]}},
        {"returncode": 0, "arm_report": {"chunking_check": {"problems": ["sums differ"]}}},
    ],
    ids=["exit-status", "no-report", "integrity", "chunking"],
)
def test_an_invalid_step_is_a_failure_even_when_it_exits_zero(record: dict[str, Any]) -> None:
    assert _driver().step_failures("arm.py B", record)


def test_a_valid_step_is_not_a_failure() -> None:
    record = {
        "returncode": 0,
        "arm_report": {"integrity_problems": [], "chunking_check": {"problems": []}},
    }

    assert _driver().step_failures("arm.py B", record) == []
