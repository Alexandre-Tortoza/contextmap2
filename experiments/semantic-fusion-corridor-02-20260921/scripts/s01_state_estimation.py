"""Stage 1 (real): ExternalPose trajectory from corridor-02-gt.txt over the 90 s window.

Adapted from outputs/validation/2026-09-21/state_estimation/scripts/02_run_external_pose.py, with one
run (no determinism pair: that was already validated) and under the MEI-augmented calibration (W3).
"""

from __future__ import annotations

import json
import math
import time
from decimal import Decimal

from common import (
    CLOCK,
    GT,
    NS,
    SELECTION,
    SEQ_DIR,
    SEQUENCE_NAME,
    decode_window_observations,
    dump_json,
    mei_calibration,
    peak_rss_mb,
    stage_root,
)

from contextmap.ingestion import (
    ExternalPoseMeasurement,
    FrameId,
    SensorId,
    SequenceArtifactReader,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import (
    BODY_ENDPOINT,
    GeometryRequirements,
    StateEstimationRunId,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    StaticRelationRequirement,
    TrajectoryId,
    allocate_run_index,
    calibration_identity,
    execute_state_estimation,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
    InvalidSamplePolicy,
)
from contextmap.state_estimation.ports import StateEstimationRequest
from contextmap.state_estimation.run_artifact import StateEstimationDebugLevel

BODY, REF = FrameId("epson"), FrameId("map")
w0, w1 = SELECTION["window"]["start_ns"], SELECTION["window"]["end_ns"]
MARGIN = NS  # 1 s of poses outside the window so the edge scans can be interpolated

started = time.time()
calibration, calibration_record = mei_calibration()
sequence_id = SequenceArtifactReader(SEQ_DIR).manifest.artifact_id

measurements = []
for index, line in enumerate(GT.read_text().splitlines()):
    t, x, y, z, qx, qy, qz, qw = line.split()
    ns = int(Decimal(t) * NS)
    if not (w0 - MARGIN <= ns < w1 + MARGIN):
        continue
    seconds, sub = divmod(ns, NS)
    measurements.append(
        ExternalPoseMeasurement(
            observation_id=SourceObservationId(f"corridor-02-gt-{index:05d}"),
            sensor_id=SensorId("corridor-02-gt"),
            frame_id=BODY,
            timestamp=SourceTimestamp(seconds=seconds, nanoseconds=sub, clock_id=CLOCK),
            provenance=SourceProvenance(
                source_type="dataset",
                source_path="datasets/corridor-02/corridor-02-gt.txt",
                source_topic=None,
                source_message_index=index,
                raw_metadata={"format": "TUM t x y z qx qy qz qw", "line_number": index + 1},
            ),
            calibration_id=None,
            parent_frame=REF,
            translation=(float(x), float(y), float(z)),
            orientation=(float(qx), float(qy), float(qz), float(qw)),
        )
    )
print(f"{len(measurements)} pose measurements", flush=True)

lidar = decode_window_observations("lidar", set(SELECTION["lidar"]["observation_ids"]))
print(f"{len(lidar)} LiDAR scans decoded for the geometry preflight; HWM {peak_rss_mb()} MB", flush=True)
merged = sorted([*lidar, *measurements], key=lambda o: (o.timestamp.seconds, o.timestamp.nanoseconds))

requirement = GeometryRequirements(
    capability="geometric_mapping",
    modalities=frozenset({"lidar"}),
    static_relations=(StaticRelationRequirement(from_endpoint=BODY_ENDPOINT, to_endpoint="lidar"),),
)
estimator = ExternalPoseEstimator(
    ExternalPoseConfig(
        reference_frame=REF,
        body_frame=BODY,
        invalid_sample_policy=InvalidSamplePolicy.FAIL,
        max_gap_ns=350_000_000,
    )
)
request = StateEstimationRequest(
    trajectory_id=TrajectoryId("traj-corridor-02-extpose-w336"),
    sequence_artifact_id=sequence_id,
    selection_id=SELECTION["selection_identity"],
    observations=merged,
    calibration=calibration,
)
workspace = stage_root("state_estimation")
t = time.time()
outcome = execute_state_estimation(estimator, request, downstream=(requirement,))
estimate_s = time.time() - t
index = allocate_run_index(workspace_root=workspace, sequence_name=SEQUENCE_NAME)
writer = StateEstimationRunWriter(
    workspace_root=workspace,
    sequence_name=SEQUENCE_NAME,
    run_id=StateEstimationRunId("se-corridor-02-extpose-w336-mei"),
    run_index=index,
    selection_label="ts-w336-90s",
    backend_label="external-pose",
    debug_level=StateEstimationDebugLevel.NONE,
)
manifest = writer.finalize(outcome, runtime_s=estimate_s)
run_dir = next(workspace.rglob(f"run-{index:04d}__*"))
problems = StateEstimationRunReader(run_dir).verify_integrity()
summary = {
    "run_dir": str(run_dir),
    "run_id": str(manifest.run_id),
    "preflight": outcome.preflight.status.name,
    "poses": len(outcome.result.trajectory.poses),
    "consumed": outcome.result.consumed_observation_count,
    "rejected": outcome.result.rejected_observation_count,
    "estimate_seconds": round(estimate_s, 3),
    "integrity_problems": problems,
    "calibration_identity": calibration_identity(calibration),
    "calibration_record": calibration_record,
    "peak_rss_mb": peak_rss_mb(),
    "total_seconds": round(time.time() - started, 1),
    "label": "real (ExternalPose from dataset ground truth; not a FAST-LIO run)",
}
dump_json("state_estimation", "summary.json", summary)
print(json.dumps({k: v for k, v in summary.items() if k != "calibration_record"}, indent=1, default=str))
assert not problems and not math.isnan(estimate_s)
