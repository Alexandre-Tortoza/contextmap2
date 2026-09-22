"""Compact, human-readable digest of the real fusion validation (for the docs and the PR text)."""

from __future__ import annotations

import argparse
import json

from common import VAL

parser = argparse.ArgumentParser()
parser.add_argument("--stage", default="semantic_fusion")
args = parser.parse_args()
root = VAL / args.stage
report = json.loads((root / "report.json").read_text())
print("== upstream")
print(json.dumps(report["upstream"]["association_run_ids"]), report["upstream"]["geometric_map_id"])
print("== grouping", report["grouping"])
print("== supports (primary)")
primary = report["supports"]["primary"]
for key, value in primary.items():
    print(f"  {key}: {value}")
print("== supports sensitivity (structure only)")
for name, item in report["supports"]["sensitivity_structure_only"].items():
    print(
        f"  {name}: supports={item['supports']} excluded={item['excluded_observations']} "
        f"multi_frame={item['supports_with_two_or_more_physical_observations']} "
        f"largest={item['largest_support_observations']} ({item['largest_support_share_of_observations']:.3f})"
    )
print("== per arm")
for name, arm in report["arms"].items():
    ev = arm["evaluation"]
    print(f"--- {name} ({ev['arm_role']})")
    print("  supports", ev["support_count"], "physical", ev["physical_observation_count"], "inference", ev["inference_result_count"])
    print("  correlation", ev["correlation"])
    print("  uncertainty", {k: v for k, v in ev["uncertainty"].items() if k != "supports_by_kind"}, ev["uncertainty"]["supports_by_kind"])
    print("  channels", [(c["channel"], c["active"], c["data_items"]) for c in ev["channels"]])
    if ev["weighting"]:
        print("  weighting", ev["weighting"])
    print("  cost", ev["cost"]["total_payload_bytes"], ev["cost"]["runtime"], arm["runtime"])
print("== comparison")
for entry in report["comparison"]["entries"]:
    print(
        f"  {entry['arm_id']}: channels={entry['active_channels']} hypotheses_match={entry['hypotheses_match_control']} "
        f"stances_match={entry['stances_match_control']} changed_leader={entry['supports_with_changed_leader']}"
    )
print("== determinism", report["determinism"])
base = report["arms"]["baseline_uniform"]["evaluation"]
print("== strata (baseline arm)")
for stratification in base["strata"]:
    print(f"  {stratification['dimension']} ({stratification['unit']}) edges={stratification['edges']}")
    for stratum in [*stratification["strata"], stratification["unavailable"]]:
        print(f"     {stratum['band']}: supports={stratum['supports']} with_uncertainty={stratum['with_uncertainty']}")
quality_arm = report["arms"]["quality_aware_depth_border"]["evaluation"]
print("== strata (quality-aware depth+border arm) weighting")
print(quality_arm["weighting"])
print("== peak", report["peak_rss_mb"], "MB total", report["total_seconds"], "s")
try:
    print("== visual consistency")
    print(json.dumps(json.loads((root / "visual_consistency.json").read_text()), indent=1))
except FileNotFoundError:
    print("(no visual_consistency.json yet)")
