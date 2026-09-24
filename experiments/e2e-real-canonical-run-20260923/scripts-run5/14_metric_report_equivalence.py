"""Real metric-report equivalence between two independent reruns (PR #438 review, fifth round):
scenario 1.0.4's reproducibility.rerun_equivalence gate text asks for "equivalent artifacts,
metric reports and final map" -- prior rounds proved the final map's content and the artifacts'
lineage, but never compared each stage's own metric report (diagnostic/quality counts) between
the two runs.

Policy, stated explicitly rather than left implicit: every stage's manifest is deterministic
metrics (pose/point/observation/entity/relation counts, diagnostic finding counts by code,
warnings) plus run-specific identity (run_id, trajectory_id, lineage references, timestamps,
digests of upstream artifacts). The metrics are compared for EXACT equality -- zero tolerance --
because these stages are deterministic CPU pipelines over the same real inputs and the same code;
identity fields are excluded because two independent, correct executions are never expected to
produce the same identity strings. This is a real, load-bearing distinction, not a shortcut: an
exact-match policy on the metrics is what actually catches a stage silently processing a
different number of observations, dropping a diagnostic, or losing warnings between two runs.
"""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")

# Fields that are legitimately run-specific identity/timestamps/digests/lineage references, never
# expected to match between two independent, correct executions of the same real inputs.
_EXCLUDED_KEYS = frozenset(
    {
        "run_id", "run_index", "created_at", "trajectory_id", "code_version", "schema_version",
        "file_inventory", "map_id", "state_estimation_run_id", "geometric_map_id",
        "entity_resolution_run_id", "lineage", "semantic_map_id", "context_map_id",
        "content_identity", "configuration_fingerprint", "evidence_identities", "dependencies",
        "written_at", "format_version",
    }
)


def _diff(label: str, manifest_a: Path, manifest_b: Path) -> list[tuple[str, object, object]]:
    a = json.loads(manifest_a.read_text())
    b = json.loads(manifest_b.read_text())
    keys = (set(a) | set(b)) - _EXCLUDED_KEYS
    mismatches = [(key, a.get(key), b.get(key)) for key in sorted(keys) if a.get(key) != b.get(key)]
    status = "ALL METRICS MATCH" if not mismatches else f"{len(mismatches)} MISMATCH(ES)"
    print(f"  [{label}] {status}")
    for key, value_a, value_b in mismatches:
        print(f"      {key}: A={value_a!r} B={value_b!r}")
    return mismatches


def main() -> None:
    pairs = [
        ("state_estimation", "e2e-real/run-0003/state_estimation",
         "corridor-02-repro-check/run-0007/state_estimation"),
        ("geometric_mapping", "e2e-real/run-0003/geometric_mapping",
         "corridor-02-repro-check/run-0007/geometric_mapping"),
        ("sensor_association", "e2e-real/run-0003/sensor_association",
         "corridor-02-repro-check/run-0007/sensor_association"),
        ("semantic_fusion", "e2e-real/run-0003/semantic_fusion",
         "corridor-02-repro-check/run-0008/semantic_fusion"),
        ("semantic_mapping", "e2e-real/run-0003/semantic_mapping",
         "corridor-02-repro-check/run-0008/semantic_mapping"),
        ("entity_resolution", "e2e-real/run-0003/entity_resolution",
         "corridor-02-repro-check/run-0008/entity_resolution"),
        ("spatial_relations", "e2e-real/run-0003/spatial_relations",
         "corridor-02-repro-check/run-0008/spatial_relations"),
        ("context_map", "e2e-real/run-0004/context_map",
         "corridor-02-repro-check/run-0008/context_map"),
    ]
    total_mismatches = 0
    for label, a, b in pairs:
        mismatches = _diff(
            label, WORKSPACE / a / "manifest.json", WORKSPACE / b / "manifest.json"
        )
        total_mismatches += len(mismatches)

    print()
    if total_mismatches:
        raise SystemExit(f"FAILED: {total_mismatches} real metric mismatch(es) across stages")
    print(
        "PASSED: every downstream stage's metric report (counts, diagnostics, warnings) is "
        "exactly equal between the two independent real reruns, excluding legitimately "
        "run-specific identity fields."
    )


if __name__ == "__main__":
    main()
