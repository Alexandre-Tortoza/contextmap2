"""Compare two Sensor Association arms by persistent geometry identity (issue #564).

Optimizing candidate selection can silently drop valid geometry and still produce a plausible
map, so the arms are compared on what the artifact actually claims about the world: which
geometry supports each ``SpatialObservation``, which regions each frame evaluated, and whether
the two runs are the same run of the same upstream artifacts. Local candidate-array indices
are never compared -- they are not identities.

The arms may be written by **different schema versions** (the baseline arm predates the
candidate schema, ``0.1.0`` vs ``0.2.0``), so this reads the persisted files directly instead
of through the current reader, which would refuse the older manifest. The formats it reads are
the same in both: ``observation-index.jsonl`` locates each observation in
``spatial-observations.jsonl``, whose ``support`` names a run of little-endian ``uint32``
geometry indices in ``geometry-support.u32``. In ``0.1.0`` those indices were candidate rows
over the whole map, which for a full-map arm are exactly the global indices, so the two arms'
values are directly comparable -- and that coincidence is precisely what #562 removed.

Every difference is classified:

``declared``   expected because the arms declare different candidate policies
``lineage``    the arms are not the same run of the same inputs, which invalidates the comparison
``defect``     a difference the candidate policy does not explain

Usage: python equivalence.py <baseline-run-dir> <optimized-run-dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Any

LINEAGE_FIELDS = (
    "sequence_artifact_id",
    "geometric_map_id",
    "trajectory_id",
    "state_estimation_run_id",
    "calibration_identity",
    "perception_run_ids",
    "membership_policy_id",
)


def _records(run_dir: Path, relative: str) -> list[dict[str, Any]]:
    text = (run_dir / relative).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _manifest(run_dir: Path) -> dict[str, Any]:
    return dict(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")))


def _support(run_dir: Path) -> dict[str, tuple[int, ...]]:
    """Every observation's geometry support, as global geometry indices."""
    payload = array("I")
    payload.frombytes((run_dir / "outputs/geometry-support.u32").read_bytes())
    observations = (run_dir / "outputs/spatial-observations.jsonl").read_bytes()
    support: dict[str, tuple[int, ...]] = {}
    for entry in _records(run_dir, "outputs/observation-index.jsonl"):
        span = entry["observation"]
        record = json.loads(
            observations[span["byte_offset"] : span["byte_offset"] + span["byte_length"]]
        )
        run = record["support"]
        support[entry["spatial_observation_id"]] = tuple(
            payload[run["offset"] : run["offset"] + run["count"]]
        )
    return support


def _by_frame(run_dir: Path, relative: str) -> dict[str, dict[str, Any]]:
    return {record["source_observation_id"]: record for record in _records(run_dir, relative)}


def compare(baseline: Path, optimized: Path) -> dict[str, Any]:
    """Compare two persisted arms and classify every difference."""
    left, right = _manifest(baseline), _manifest(optimized)
    findings: list[dict[str, Any]] = []

    for field in LINEAGE_FIELDS:
        if left.get(field) != right.get(field):
            findings.append(
                {
                    "kind": "lineage",
                    "what": field,
                    "baseline": left.get(field),
                    "optimized": right.get(field),
                }
            )

    left_support, right_support = _support(baseline), _support(optimized)
    missing = sorted(set(left_support) - set(right_support))
    extra = sorted(set(right_support) - set(left_support))
    if missing:
        findings.append({"kind": "defect", "what": "observations missing", "ids": missing[:20]})
    if extra:
        findings.append({"kind": "defect", "what": "observations added", "ids": extra[:20]})

    support_changed: list[dict[str, Any]] = []
    for identity in sorted(set(left_support) & set(right_support)):
        before, after = left_support[identity], right_support[identity]
        if before != after:
            lost = sorted(set(before) - set(after))
            gained = sorted(set(after) - set(before))
            support_changed.append(
                {
                    "spatial_observation_id": identity,
                    "lost_count": len(lost),
                    "gained_count": len(gained),
                    "lost_sample": lost[:5],
                    "gained_sample": gained[:5],
                }
            )

    declared_range = (right.get("candidate_policy") or {}).get("max_range_m")
    if support_changed:
        # Suporte alterado só é "declarado" se o braço ótimo declara um alcance: sem alcance,
        # a população avaliada é a mesma e qualquer diferença é defeito.
        findings.append(
            {
                "kind": "declared" if declared_range is not None else "defect",
                "what": "geometry support differs",
                "observations_affected": len(support_changed),
                "geometry_lost_total": sum(item["lost_count"] for item in support_changed),
                "geometry_gained_total": sum(item["gained_count"] for item in support_changed),
                "detail": support_changed[:20],
            }
        )

    left_visibility = _by_frame(baseline, "outputs/visibility-records.jsonl")
    right_visibility = _by_frame(optimized, "outputs/visibility-records.jsonl")
    shared_frames = sorted(set(left_visibility) & set(right_visibility))
    if set(left_visibility) != set(right_visibility):
        findings.append(
            {
                "kind": "defect",
                "what": "frames differ",
                "baseline_only": sorted(set(left_visibility) - set(right_visibility)),
                "optimized_only": sorted(set(right_visibility) - set(left_visibility)),
            }
        )
    for frame_id in shared_frames:
        before = [r["region_id"] for r in left_visibility[frame_id]["regions"]]
        after = [r["region_id"] for r in right_visibility[frame_id]["regions"]]
        if before != after:
            findings.append(
                {"kind": "defect", "what": "evaluated regions differ", "frame": frame_id}
            )
        left_states = left_visibility[frame_id]["state_counts"]
        right_states = right_visibility[frame_id]["state_counts"]
        if left_visibility[frame_id]["visible"] != right_visibility[frame_id]["visible"]:
            findings.append(
                {
                    "kind": "declared" if declared_range is not None else "defect",
                    "what": "visible count differs",
                    "frame": frame_id,
                    "baseline": left_visibility[frame_id]["visible"],
                    "optimized": right_visibility[frame_id]["visible"],
                }
            )
        if left_states.get("occluded") != right_states.get("occluded"):
            findings.append(
                {
                    "kind": "declared" if declared_range is not None else "defect",
                    "what": "occluded count differs",
                    "frame": frame_id,
                    "baseline": left_states.get("occluded"),
                    "optimized": right_states.get("occluded"),
                }
            )

    left_projection = _by_frame(baseline, "outputs/projection-records.jsonl")
    right_projection = _by_frame(optimized, "outputs/projection-records.jsonl")
    return {
        "baseline": str(baseline),
        "optimized": str(optimized),
        "schema_versions": {
            "baseline": left.get("schema_version"),
            "optimized": right.get("schema_version"),
        },
        "candidate_policies": {
            "baseline": left.get("candidate_policy", "absent (0.1.0 schema: whole map)"),
            "optimized": right.get("candidate_policy"),
        },
        "frames": {
            "baseline": left.get("frame_count"),
            "optimized": right.get("frame_count"),
            "rejected_baseline": left.get("rejected_frame_count"),
            "rejected_optimized": right.get("rejected_frame_count"),
        },
        "observations": {
            "baseline": left.get("observation_count"),
            "optimized": right.get("observation_count"),
        },
        "geometry_support_identical": not support_changed and not missing and not extra,
        "per_frame": {
            frame_id: {
                "baseline_evaluated": left_projection[frame_id]["point_count"],
                "optimized_evaluated": right_projection[frame_id]["point_count"],
                "optimized_candidate_fraction": right_projection[frame_id]["candidates"][
                    "candidate_fraction"
                ],
                "baseline_visible": left_visibility[frame_id]["visible"],
                "optimized_visible": right_visibility[frame_id]["visible"],
                "baseline_associated": left_visibility[frame_id]["membership"][
                    "associated_count"
                ],
                "optimized_associated": right_visibility[frame_id]["membership"][
                    "associated_count"
                ],
            }
            for frame_id in shared_frames
        },
        "findings": findings,
        "verdict": "equivalent"
        if not findings
        else (
            "declared-difference"
            if all(f["kind"] == "declared" for f in findings)
            else "review"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    options = parser.parse_args()

    report = compare(options.baseline, options.optimized)
    text = json.dumps(report, indent=1, sort_keys=True)
    if options.json is not None:
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
