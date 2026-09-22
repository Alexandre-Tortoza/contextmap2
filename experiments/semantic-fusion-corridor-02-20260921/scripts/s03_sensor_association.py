"""Stage 3 (real): 2D <-> 3D association of the perception runs' regions over the real map.

One SensorAssociationRunArtifact per PerceptionRunArtifact (a run cannot hold the same frame twice),
all over the same map, trajectory, MEI calibration and occlusion policy, declared once (no per-run
retuning). Usage: s03_sensor_association.py <perception-run-dir-name> [<name> ...] [--limit N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from common import (
    CODE_SHA,
    PERCEPTION_BASE,
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

from contextmap.geometric_mapping import GeometricMapArtifactReader
from contextmap.ingestion import SequenceArtifactReader
from contextmap.sensor_association import (
    DiagnosticTolerances,
    OcclusionPolicy,
    AssociationFrameInput,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
    allocate_run_index,
)
from contextmap.state_estimation import LookupPolicy, StateEstimationRunReader, TrajectoryLookup
from contextmap.visual_perception import PerceptionRunReader, PreparedImage

args = [a for a in sys.argv[1:] if not a.startswith("--")]
limit = None
stage = "sensor_association"
for i, a in enumerate(sys.argv):
    if a == "--limit":
        limit = int(sys.argv[i + 1])
        args = [x for x in args if x != sys.argv[i + 1]]
    if a == "--stage":
        stage = sys.argv[i + 1]
        args = [x for x in args if x != sys.argv[i + 1]]
run_names = args or ["run-0001__frames-08026-10078__sam2-dinov2-clip"]

# Declared once for every run and arm (AGENTS.md 5: no hidden threshold retuning).
OCCLUSION = OcclusionPolicy(cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02)
TOLERANCES = DiagnosticTolerances(
    max_pose_time_delta_ns=250_000_000,
    max_map_window_offset_ns=1_000_000_000,
    max_reprojection_p95_px=None,
    max_reprojection_invalid_rate=None,
)
POSE_POLICY = LookupPolicy.interpolated(max_interpolation_gap_ns=450_000_000)

calibration, _ = mei_calibration()
sequence_id = SequenceArtifactReader(SEQ_DIR).manifest.artifact_id
se_dir = next((VAL / "state_estimation/runs/state-estimation/corridor-02").glob("run-0001__*"))
se_reader = StateEstimationRunReader(se_dir)
lookup = TrajectoryLookup(se_reader.trajectory())
gm_dir = next((VAL / "geometric_mapping/runs/geometric-mapping/corridor-02").glob("run-0001__*"))
selected = SELECTION["images"]["selected"]
image_ids = [item["observation_id"] for item in selected]
if limit:
    image_ids = image_ids[:limit]
images = {str(o.observation_id): o for o in decode_window_observations("image", set(image_ids))}
print(f"{len(images)} image observations decoded; HWM {peak_rss_mb()} MB", flush=True)
first = next(iter(images.values()))
print("image", first.width, first.height, first.encoding, first.calibration_id, first.frame_id, flush=True)

report = {"policies": {"occlusion": OCCLUSION.to_record(), "pose": "interpolated max_gap 450 ms"}, "runs": {}}
workspace = stage_root(stage)
with GeometricMapArtifactReader(gm_dir) as gm_reader:
    geometry = gm_reader.geometry()
    for name in run_names:
        perception = PerceptionRunReader(PERCEPTION_BASE / name)
        frames = []
        for obs_id in image_ids:
            image = images[obs_id]
            result = perception.result(image.observation_id)
            frames.append(
                AssociationFrameInput(
                    observation=image,
                    prepared_image=PreparedImage(
                        source_observation_id=image.observation_id,
                        payload_reference=f"prepared/{obs_id}.png",
                        width=image.width,
                        height=image.height,
                    ),
                    perception_result=result,
                )
            )
        request = SensorAssociationRequest(
            sequence_artifact_id=sequence_id,
            selection_id=SELECTION["selection_identity"],
            geometry=geometry,
            trajectory=lookup,
            pose_policy=POSE_POLICY,
            calibration=calibration,
            occlusion_policy=OCCLUSION,
            tolerances=TOLERANCES,
            frames=tuple(frames),
            state_estimation_run_id=se_reader.manifest.run_id,
            code_version=CODE_SHA,
        )
        t = time.time()
        outcome = SensorAssociationService().run(request)
        run_s = time.time() - t
        index = allocate_run_index(workspace_root=workspace, sequence_name=SEQUENCE_NAME)
        label = name.split("__")[-1]
        writer = SensorAssociationRunWriter(
            workspace_root=workspace,
            sequence_name=SEQUENCE_NAME,
            run_id=SensorAssociationRunId(f"sa-{perception.manifest.run_id}"),
            run_index=index,
            selection_label="ts-w336-20frames",
            channel_label=label,
        )
        manifest = writer.finalize(outcome, runtime_s=run_s)
        run_dir = next(workspace.rglob(f"run-{index:04d}__*"))
        reader = SensorAssociationRunReader(run_dir)
        problems = reader.verify_integrity()
        observations = sum(len(frame.observations) for frame in outcome.frames)
        skipped = sum(len(frame.membership.skipped) for frame in outcome.frames)
        entry = {
            "run_dir": str(run_dir),
            "run_id": str(manifest.run_id),
            "perception_run": name,
            "perception_run_id": str(perception.manifest.run_id),
            "frames_associated": len(outcome.frames),
            "frames_rejected": len(outcome.rejected),
            "observations": observations,
            "regions_skipped": skipped,
            "association_seconds": round(run_s, 1),
            "artifact_mb": round(sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file()) / 1e6, 1),
            "integrity_problems": problems,
            "peak_rss_mb": peak_rss_mb(),
        }
        report["runs"][name] = entry
        print(json.dumps(entry, indent=1, default=str), flush=True)
dump_json(stage, "summary.json" if not limit else f"summary-limit{limit}.json", report)
