"""Run one profiled stage script as a fresh child process and report its own wall time and
peak RSS (issue #181). A fresh process per stage is required: resource.getrusage(RUSAGE_CHILDREN)
is a running maximum across every child a process has spawned, so measuring N stages from one
long-lived parent would leak each stage's peak into the next.

Usage: python _run_one.py <script.py> <stage_name>
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

script = Path(sys.argv[1])
stage = sys.argv[2]

before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
start = time.time()
result = subprocess.run(
    [sys.executable, str(script)], capture_output=True, text=True, check=False
)
elapsed = time.time() - start
after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

report = {
    "stage": stage,
    "returncode": result.returncode,
    "wall_time_seconds": round(elapsed, 3),
    "peak_rss_mb": round(after / 1024, 1),  # ru_maxrss is KB on Linux
    "stdout_tail": result.stdout.splitlines()[-15:],
    "stderr_tail": result.stderr.splitlines()[-15:] if result.returncode != 0 else [],
}
print(json.dumps(report, indent=1))
if result.returncode != 0:
    sys.exit(result.returncode)
