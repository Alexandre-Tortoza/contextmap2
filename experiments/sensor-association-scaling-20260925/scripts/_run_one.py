"""Run one arm as a fresh child process and report its own wall time and peak RSS (#564).

A fresh process per arm is required: ``resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`` is a
running maximum across every child a process has ever spawned, so measuring several arms from
one long-lived parent would leak each arm's peak into the next. Same method as the #181
resource profile, so the numbers are comparable with it.

Usage: python _run_one.py <python-executable> <arm.py> <arm-id> <output-dir> [extra args...]
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

interpreter, script, *arguments = sys.argv[1:]

before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
start = time.time()
result = subprocess.run(
    [interpreter, script, *arguments], capture_output=True, text=True, check=False
)
elapsed = time.time() - start
after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

report: dict[str, object] = {
    "arm": arguments[0] if arguments else None,
    "returncode": result.returncode,
    "wall_time_seconds": round(elapsed, 3),
    # ru_maxrss é KB no Linux. `before` é registrado para deixar explícito que a medição é do
    # máximo deste filho: um pai novo por arm garante que `before` seja 0.
    "peak_rss_mb": round(after / 1024, 1),
    "peak_rss_mb_before": round(before / 1024, 1),
}
for line in result.stdout.splitlines():
    stripped = line.strip()
    if stripped.startswith("{"):
        break
else:
    report["stdout_tail"] = result.stdout.splitlines()[-20:]
try:
    report["arm_report"] = json.loads(result.stdout[result.stdout.index("{") :])
except (ValueError, json.JSONDecodeError):
    report["stdout_tail"] = result.stdout.splitlines()[-20:]
if result.returncode != 0:
    report["stderr_tail"] = result.stderr.splitlines()[-25:]

print(json.dumps(report, indent=1))
if result.returncode != 0:
    sys.exit(result.returncode)
