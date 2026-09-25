"""Run every arm of the Sensor Association scaling experiment and assemble the raw report.

Each arm runs as its own fresh OS process through ``_run_one.py``, so its peak RSS is its own.
The baseline arm runs from a worktree checked out at the revision *before* the optimizations,
because batch retention no longer exists in the code: that is what makes the two optimizations
independently attributable.

    A   full-map candidates + batch retention      (baseline revision)
    D   full-map candidates + streaming            (this revision)      -> isolates #563
    C   culled candidates  + streaming             (this revision)      -> isolates #562

Usage:
    python run_all.py <output-root> --baseline-src <path-to-baseline-worktree>
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ARM = HERE / "arm.py"
RUN_ONE = HERE / "_run_one.py"


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
    gpu = shutil.which("nvidia-smi")
    name = ""
    if gpu:
        probe = subprocess.run(
            [gpu, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        name = probe.stdout.strip().splitlines()[0] if probe.returncode == 0 else ""
    return {
        "cpu_count": len(__import__("os").sched_getaffinity(0)),
        "ram_gib": round(total_kb / 1024 / 1024, 1),
        "gpu": name or "n/a",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _arm(
    *,
    arm: str,
    output_dir: Path,
    interpreter: str,
    script: Path,
    env_src: Path | None,
    extra: list[str],
) -> dict[str, Any]:
    command = [sys.executable, str(RUN_ONE), interpreter, str(script), arm, str(output_dir), *extra]
    environ = dict(__import__("os").environ)
    if env_src is not None:
        environ["PYTHONPATH"] = str(env_src)
    print(f"--- arm {arm}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=environ)
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr, flush=True)
    try:
        return dict(json.loads(result.stdout[result.stdout.index("{") :]))
    except (ValueError, json.JSONDecodeError):
        return {"arm": arm, "returncode": result.returncode, "stderr_tail": result.stderr[-4000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--baseline-src", type=Path, default=None)
    parser.add_argument("--baseline-script", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--full-window", action="store_true", help="also run the full sequence")
    parser.add_argument("--partial-frames", type=int, default=30)
    options = parser.parse_args()

    options.output_root.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, Any]] = {}

    if options.baseline_src is not None:
        arms["A"] = _arm(
            arm="A",
            output_dir=options.output_root / "arm-A-fullmap-batch",
            interpreter=sys.executable,
            script=options.baseline_script or (options.baseline_src.parent / "arm_baseline.py"),
            env_src=options.baseline_src,
            extra=[],
        )
    arms["D"] = _arm(
        arm="D",
        output_dir=options.output_root / "arm-D-fullmap-streaming",
        interpreter=sys.executable,
        script=ARM,
        env_src=None,
        extra=[],
    )
    arms["C"] = _arm(
        arm="C",
        output_dir=options.output_root / "arm-C-culled-streaming",
        interpreter=sys.executable,
        script=ARM,
        env_src=None,
        extra=["--max-range-m", "20"],
    )
    if options.full_window:
        arms["C-full"] = _arm(
            arm="C-full",
            output_dir=options.output_root / "arm-C-culled-streaming-full",
            interpreter=sys.executable,
            script=ARM,
            env_src=None,
            extra=["--max-range-m", "20", "--window", "all"],
        )
        arms["D-partial"] = _arm(
            arm="D-partial",
            output_dir=options.output_root / "arm-D-fullmap-streaming-partial",
            interpreter=sys.executable,
            script=ARM,
            env_src=None,
            extra=["--window", "all", "--frames", str(options.partial_frames)],
        )

    report = {"issue": 564, "environment": _environment(), "arms": arms}
    text = json.dumps(report, indent=1, sort_keys=True)
    if options.report is not None:
        options.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
