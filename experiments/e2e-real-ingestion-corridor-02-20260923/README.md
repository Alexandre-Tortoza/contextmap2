# Real re-ingestion of corridor-02 for issue #176 (2026-09-23)

This document is the audit trail for the first real, windowed re-ingestion of `corridor-02.bag`
performed through the current `IngestionService`/`IngestionRequest`/`Ros1BagSourceAdapter` code
path, using the dataset-scoped timestamp normalization policy introduced by issue #554. It is not
committed application code — it documents a real experiment run against the MAIN checkout's
`datasets/`/`outputs/` (both untracked, non-versioned), matching the convention already
established by `experiments/semantic-fusion-corridor-02-20260921/`.

## Result

```text
artifact_id:  720a486de8d44c16a9d3d2ff9fa7b1a4
artifact_dir: outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4
status:       completed
verify_integrity(): [] (clean)
observation_counts: image=2496  imu=20781  lidar=1031  external_pose=0
bytes_written: 2,709,070,198  files_written: 3535
```

This is a **new, independent, immutable** artifact. It never reads, modifies, or depends on the
older `outputs/ingest-full/sequences/corridor-02/e145f73f8d894f18b96ef1f55ca308c2` artifact, which
was produced by an earlier, different ingestion mechanism (its `provenance.json` shape —
`{"sync": ..., "topics": ...}` — does not match the current `IngestionRequest`/`_provenance()`
schema at all, confirming it predates the present `IngestionService`).

## Window and correction decisions

- **Canonical 90 s header-time window** (unchanged from the frozen selection in
  `experiments/semantic-fusion-corridor-02-20260921/selection.json`, clock `corridor-02-header`):
  `start=1646000062.983617  end=1646000152.983617`. This selection was derived independently in an
  earlier validation pass (sliding-window search over pose/LiDAR gap quality, path length and
  turning) and is reused as-is; re-deriving it was out of scope here.
- **+/-5 s padding** added on each side before ingesting (`1646000057.983617` to
  `1646000157.983617`), so the first/last frame of the eventual 90 s selection still has full
  synchronization-tolerance context on both sides once a `TimestampRangeSelection` is applied to
  this artifact.
- **Correction anchor**: the real RGB message at/after the padded window start —
  `source_time=SourceTimestamp(seconds=1646000057, nanoseconds=997985442,
  clock_id="corridor-02-header")`, `reference_time=SourceTimestamp(seconds=1724621416,
  nanoseconds=410417577, clock_id="corridor-02-recording-time")`. `offset_ns = 78621358412432135`
  (`78621358.412432s`, exact, per `ConstantOffsetCorrection.from_anchors`).
- **Write-side `SourceWindow`** (recording_time, `+/-2 s` extra safety buffer beyond the padded
  header bounds, converted through the anchor-derived offset):
  `[1724621414.396049, 1724621518.396049)`.
- **Calibration**: built fresh from `datasets/corridor-02/corridor-02-Intrinsics.yaml` (RGB camera,
  now a real `MeiCameraModel`, not `camera_model=None`) and
  `datasets/corridor-02/corridor-02-extrinsics.yaml` (static transforms `epson->cmu_rc1_velodyne`,
  `epson->d`), passed as `IngestionRequest.calibration` — an ordinary, ingestion-time configuration
  input, not a post-ingestion patch.
- **Validation policy**: `ValidationPolicy(on_problems="warn")`, not the default `"fail"` — see
  below.

## Finding 1: cross-sensor "non-monotonic" warnings (3405 of 24308 observations, 14.01%)

`validate_timestamp_ordering()` compares observations sharing one `clock_id`, in the order the
adapter yields them (bag arrival / `recording_time` order), and flags any pair where a later
arrival has an earlier corrected timestamp than the previous one. Since RGB, LiDAR and IMU on
corridor-02 all share one explicit `timestamp_clock_id="corridor-02-header"` (one physical robot
clock, per `SourceAdapterConfig.resolved_timestamp_clock_id()`'s intended semantics), this check
runs **across sensors**, not just within one.

Measured breakdown of the 3405 violations:

```text
Pairs (later_sensor, earlier_sensor_in_arrival_order):
  camera_1_image_raw -> imu_data       2423
  velodyne_points     -> imu_data        914
  camera_1_image_raw -> velodyne_points   43
  velodyne_points     -> camera_1_image_raw 25

Magnitude (seconds the later one is behind): min=0.3ms  max=147.9ms  mean=113.7ms  median=113.1ms
Magnitude histogram (% of the whole 24308-observation dataset):
  [  0, 20)ms: 0.25%   [ 20, 40)ms: 0.03%   [ 80,100)ms: 0.86%
  [100,120)ms: 8.18%   [120,140)ms: 4.49%   [140,160)ms: 0.20% (max 147.87ms)
```

Per-sensor residual against the configured (camera-anchored) offset —
`(recording_time - source_time) - offset_configured`, computed from every observation's own
preserved `bag_timestamp_nanoseconds` and `source_time_before_correction_*` provenance fields:

```text
imu_data:            n=20781  mean=-139.15ms  stdev=2.85ms   min=-142.12ms  max=-102.41ms
camera_1_image_raw:  n=2496   mean=-18.18ms   stdev=12.58ms  min=-46.74ms   max=18.74ms
velodyne_points:     n=1031   mean=-28.01ms   stdev=4.66ms   min=-39.03ms   max=1.22ms
```

**Interpretation.** The IMU's `recording_time - source_time` gap is a highly stable ~139 ms
*smaller* than the camera's (stdev only 2.85 ms over 20781 samples): the camera pipeline
(capture -> encode -> transfer -> bag write) has real, consistent extra latency the IMU driver
does not. This is a physical property of the recording hardware/pipeline, present in the raw data
before any correction, and **not introduced or worsened by the #554 timestamp policy**.

Crucially, this residual is measured against `recording_time` (the bag's own arrival clock), which
is **only ever used to decide which raw messages to read** (`SourceAdapterConfig.window`) — it is
never used to synchronize sensors against each other. The constant-offset correction adds exactly
the same value to every observation's `header_stamp` regardless of sensor, so the *relative* timing
between camera, LiDAR and IMU after correction is bit-for-bit identical to the relative timing in
the raw header clock. Synchronization quality (`synchronize()`, 50 ms tolerance) is therefore
unaffected by which single message was chosen as the anchor.

**Conclusion: this is a documented, expected characteristic of the dataset, not a defect of the
correction.** Escalating to a per-sensor/affine correction model would require exactly the kind of
real evidence issue #554 asks for before allowing that complexity — and the evidence here shows the
residual is confined to `recording_time` bookkeeping, with zero effect on the event clock's
cross-sensor consistency. No corrective action is taken; this note is the record required by
issue #554's "actionable diagnostics" acceptance criterion.

Because 14% of the dataset triggers this warning, `ValidationPolicy(on_problems="warn")` was used
for this run instead of the default `"fail"` — the sequence still verifies clean
(`verify_integrity() == []`); every warning is preserved in `SequenceProvenance.warnings` and in
the diagnostics files, never silently dropped.

## Finding 2: `external_pose=0` (open, tracked separately)

corridor-02 has no pose/odometry topic in the bag; the trajectory lives in a separate file,
`datasets/corridor-02/corridor-02-gt.txt` (TUM format). `PoseFileSourceAdapter` exists and could
ingest it, but doing so would produce a **second, separate** `SequenceArtifact` — the current
`StateEstimationRequest` (`state_estimation/ports.py`) is scoped to exactly one
`sequence_artifact_id`, and the runtime composes exactly one `ingestion` stage per run. Combining a
bag-sourced sequence with a pose-file-sourced sequence into one canonical run is a real,
un-implemented architectural question, not something #554/#176 were scoped to solve. This
re-ingestion keeps the same behavior as the older ingestion (pose consumed out-of-band by whatever
reads it next, not persisted into this `SequenceArtifact`). A dedicated investigation is being run
to decide the right architecture before this is implemented; see that investigation's report for
the full analysis and recommendation.
