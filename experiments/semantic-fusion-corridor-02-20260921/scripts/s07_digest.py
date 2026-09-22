"""Small facts for the write-up: association totals, excluded breakdown, timings, artifact sizes."""

from __future__ import annotations

import json
from collections import Counter

from common import VAL

fusion = VAL / "semantic_fusion"
report = json.loads((fusion / "report.json").read_text())
print("support build seconds:", report["supports"]["build_seconds"], "| PR run:", report["upstream"]["point_representation_run"])
print("total seconds:", report["total_seconds"], "peak rss MB:", report["peak_rss_mb"])
for directory in sorted((VAL / "sensor_association/runs/sensor-association/corridor-02").glob("run-*")):
    summary = json.loads((directory / "metrics/summary.json").read_text())
    runtime = json.loads((directory / "metrics/runtime.json").read_text())
    print(directory.name, "frames", summary["frame_count"], "obs", summary["observation_count"],
          "without_support", summary["observations_without_support"],
          "rejected", [r["source_observation_id"] + ":" + r["rejection"] for r in summary["rejected_frames"]],
          "states", summary["state_counts"], "runtime", runtime)
excluded = Counter()
run0 = next((fusion / "runs/semantic-fusion/corridor-02").glob("run-0001__*"))
for line in (run0 / "outputs/excluded-observations.jsonl").read_text().splitlines():
    record = json.loads(line)
    excluded[record.get("geometry_count", record.get("count"))] += 1
print("excluded by geometry count:", dict(sorted(excluded.items())))
print("sample excluded record:", (run0 / "outputs/excluded-observations.jsonl").read_text().splitlines()[0][:300])
counts = json.loads((run0 / "metrics/counts.json").read_text())
print("counts:", {k: counts[k] for k in ("supports", "contributions", "hypotheses", "physical_observations", "inference_results", "excluded_observations") if k in counts})
for name in ("baseline_uniform", "quality_aware_depth_border", "ch_all"):
    arm = report["arms"][name]
    print(name, "payload bytes", arm["evaluation"]["cost"]["total_payload_bytes"])
q = report["arms"]["quality_aware_depth_border"]["evaluation"]
print("qa depth+border strata (weighted vs unweighted leader) -> supports_where_leading_changes:", q["weighting"]["supports_where_leading_changes"])
# Fator por estrato de faixa de alcance: quantas contribuições de cada arm tem fator 0
qa_dir = next((fusion / "runs/semantic-fusion/corridor-02").glob("run-*quality-aware-depth-border"))
factors = []
for line in (qa_dir / "outputs/fused-evidence.jsonl").read_text().splitlines():
    record = json.loads(line)
    for item in record["weighting"]["contributions"]:
        factors.append(item["factor"])
factors.sort()
print("depth+border factor deciles:", [round(factors[int(p * (len(factors) - 1))], 3) for p in (0.1, 0.25, 0.5, 0.75, 0.9)])

# --- weighting by condition: share of zero-factor contributions per depth band, for both quality-aware arms ---------
from bisect import bisect_right
import statistics

from contextmap.sensor_association import SensorAssociationRunReader

depth_of = {}
border_of = {}
for directory in sorted((VAL / "sensor_association/runs/sensor-association/corridor-02").glob("run-*")):
    reader = SensorAssociationRunReader(directory)
    for observation in reader.observations():
        quality = reader.quality(observation.spatial_observation_id)
        depth_of[str(observation.spatial_observation_id)] = None if quality.support_depth_m is None else quality.support_depth_m.median
        border_of[str(observation.spatial_observation_id)] = None if quality.border_distance_px is None else quality.border_distance_px.median
contribution_to_observation = {}
for line in (run0 / "outputs/contribution-index.jsonl").read_text().splitlines():
    record = json.loads(line)
    contribution_to_observation[record["contribution_id"]] = record["spatial_observation_id"]
edges = (3.0, 6.0, 12.0)
labels = ["<3 m", "3-6 m", "6-12 m", ">=12 m", "unavailable"]
for arm in ("quality_aware_depth_border", "quality_aware_depth_only"):
    arm_dir = next((fusion / "runs/semantic-fusion/corridor-02").glob(f"run-*{arm.replace('_', '-')}"))
    bands = {label: [] for label in labels}
    for line in (arm_dir / "outputs/fused-evidence.jsonl").read_text().splitlines():
        record = json.loads(line)
        for item in record["weighting"]["contributions"]:
            depth = depth_of[contribution_to_observation[item["contribution_id"]]]
            label = labels[4] if depth is None else labels[bisect_right(edges, depth)]
            bands[label].append(item["factor"])
    print(arm, "factor by depth band:")
    for label, values in bands.items():
        if values:
            print(f"   {label}: contributions={len(values)} zero_share={sum(1 for v in values if v == 0.0) / len(values):.2f} median_factor={statistics.median(values):.2f}")
