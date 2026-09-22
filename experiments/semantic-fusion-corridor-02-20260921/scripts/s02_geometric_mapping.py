"""Stage 2 (real): geometric map over the 90 s window from the LiDAR scans and the ExternalPose run.

Adapted from outputs/validation/2026-09-21/geometric_mapping/scripts/03_run_geometric_mapping.py:
one run, under the MEI-augmented calibration (W3). Usage: s02_geometric_mapping.py [all-points|voxel5cm]
"""

from __future__ import annotations

import json
import sys
import time

from common import (
    CODE_SHA,
    SELECTION,
    SEQ_DIR,
    SEQUENCE_NAME,
    VAL,
    decode_window_observations,
    dump_json,
    mei_calibration,
    peak_rss_mb,
    stage_root,
)

from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    MapDebugLevel,
    MotionCorrectionPolicy,
    ScanDisposition,
    ScanVoxelPolicy,
    allocate_map_run_index,
    assemble_geometry_inputs,
)
from contextmap.ingestion import (
    SequenceArtifactReader,
    SequenceSelectionResult,
    TimestampRangeSelection,
)
from contextmap.state_estimation import LookupPolicy, StateEstimationRunReader

variant = sys.argv[1] if len(sys.argv) > 1 else "all-points"
aggregation = {"all-points": None, "voxel5cm": ScanVoxelPolicy(cell_m=0.05)}[variant]

started = time.time()
calibration, _ = mei_calibration()
se_dir = next((VAL / "state_estimation/runs/state-estimation/corridor-02").glob("run-0001__*"))
se_reader = StateEstimationRunReader(se_dir)
trajectory = se_reader.trajectory()
scans = decode_window_observations("lidar", set(SELECTION["lidar"]["observation_ids"]))
section = SELECTION["selection"]
selection = TimestampRangeSelection(
    clock_id=section["clock_id"], start_seconds=section["start_seconds"], end_seconds=section["end_seconds"]
)
sequence = SequenceSelectionResult(
    sequence_artifact_id=SequenceArtifactReader(SEQ_DIR).manifest.artifact_id,
    selection=selection,
    selection_id=SELECTION["selection_identity"],
    observations=tuple(scans),
)
print(f"{len(scans)} scans, trajectory {len(trajectory.poses)} poses, HWM {peak_rss_mb()} MB", flush=True)

plan = assemble_geometry_inputs(
    sequence=sequence,
    calibration=calibration,
    trajectory=trajectory,
    pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=400_000_000),
    motion_correction_policy=MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN),
    state_estimation_run_id=se_reader.manifest.run_id,
)
rejections: dict[str, int] = {}
for rejection in plan.rejections:
    rejections[rejection.reason.name] = rejections.get(rejection.reason.name, 0) + 1
print(f"plan: {len(plan.inputs)} inputs, rejections {rejections}", flush=True)

workspace = stage_root("geometric_mapping")
index = allocate_map_run_index(workspace_root=workspace, sequence_name=SEQUENCE_NAME)
t = time.time()
writer = GeometricMapArtifactWriter(
    workspace_root=workspace,
    sequence_name=SEQUENCE_NAME,
    run_id=GeometricMapRunId(f"gm-{variant}-mei"),
    run_index=index,
    selection_label="ts-w336-90s",
    profile_label=f"external-pose-{variant}",
    debug_level=MapDebugLevel.NONE,
)
manifest = writer.finalize(plan=plan, aggregation=aggregation, code_version=CODE_SHA, runtime_s=None, peak_memory_bytes=None)
write_s = time.time() - t
run_dir = next(workspace.rglob(f"run-{index:04d}__*"))
with GeometricMapArtifactReader(run_dir) as reader:
    problems = reader.verify_integrity()
size_mb = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file()) / 1e6
summary = {
    "variant": variant,
    "run_dir": str(run_dir),
    "run_id": str(manifest.run_id),
    "inputs": len(plan.inputs),
    "rejections": rejections,
    "write_seconds": round(write_s, 1),
    "artifact_mb": round(size_mb, 1),
    "integrity_problems": problems,
    "peak_rss_mb": peak_rss_mb(),
    "total_seconds": round(time.time() - started, 1),
    "label": "real (ExternalPose trajectory + real LiDAR; not FAST-LIO)",
}
dump_json("geometric_mapping", f"summary-{variant}.json", summary)
print(json.dumps(summary, indent=1, default=str))
