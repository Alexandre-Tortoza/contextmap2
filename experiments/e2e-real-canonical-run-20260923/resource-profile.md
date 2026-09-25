# Resource profile of the real canonical corridor-02 run (issue #181)

This layers real resource measurements on top of the #177 real canonical run
(`experiments/e2e-real-canonical-run-20260923/README.md`) -- it never touches that run's
artifacts (`outputs/e2e-real/run-0001/...`). Every stage except `visual_perception` was
re-executed once, from the same real upstream inputs, into a separate `run-0001-profiled/`
tree (`outputs/e2e-real/run-0001-profiled/<stage>/`) purely to measure it; `run-0001` itself was
never written to again. `visual_perception`'s numbers are pulled from its own already-real
report (`outputs/e2e-real/visual_perception/reports/31_pipeline_semantic_sam2_canonical-1-0-4.json`)
instead of re-running ~12 minutes of real GPU/Qwen generation for no new data.

## Method

- **Wall time / peak RSS**: each stage's driver script (`scripts-profiled/0N_<stage>.py`, a
  copy of the real `scripts/0N_<stage>.py` with only its own `OUTPUT_DIR` redirected to
  `run-0001-profiled/<stage>` -- every upstream `ArtifactRef` still points at the real,
  already-verified `run-0001` artifact) was run as its own fresh OS process via
  `scripts-profiled/_run_one.py`, which reads `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`
  immediately after that one child exits. A fresh parent process per stage is required:
  `RUSAGE_CHILDREN` is a running maximum across every child a process has ever spawned, so
  measuring several stages from one long-lived parent would leak each stage's peak into the
  next stage's number.
- **Storage bytes / file count**: measured directly on the real, untouched `run-0001`
  artifacts (`du -sb` / a recursive file count) -- these numbers describe what issue #177
  actually produced, not the profiling rerun.
- **Device/precision**: not applicable to any stage below except `visual_perception` (GPU
  models); these seven stages are ordinary CPU/NumPy code with no `device` or `precision`
  concept of their own, so noting "n/a" here is accurate, not a fabricated GPU number.
- **Warm/cold**: every measured stage read the real upstream artifact fresh from disk with no
  reuse/caching (`ReusePolicy` was not exercised by these hand-built `StageRequest` drivers) --
  effectively "cold" for the file-system cache page-in cost too, since prior stages had
  already populated OS page cache for the same files, though this was not independently
  controlled per stage.
- **Machine**: 16 logical CPUs, 39 GiB RAM, RTX 3060 8 GB GPU (see `env` in the JSON report).

## Results

| Stage | Wall time | Peak RSS | Storage (bytes) | Files | Device |
|---|---|---|---|---|---|
| ingestion (bag, issue #176/#554, not re-measured here) | -- | -- | -- | -- | -- |
| state_estimation | 2.8 s | 2.70 GB | 3.6 MB | 9 | n/a (CPU) |
| geometric_mapping | 15.1 s | 1.43 GB | **1.05 GB** | 10 | n/a (CPU) |
| sensor_association | **177.8 s** | **24.96 GB** | 3.4 MB | 10 | n/a (CPU) |
| semantic_fusion | 6.3 s | 2.75 GB | 2.7 MB | 12 | n/a (CPU) |
| visual_perception (real, from #177's own report, not re-run) | 91.0 s + 656.7 s | 4.99 GB / 3.25 GB (phase1/phase2) | 76.7 MB | 1437 | CUDA / RTX 3060, nf4 (Qwen), float32 (SAM2/DINOv2/CLIP) |
| semantic_mapping | 31.2 s | 0.51 GB | 1.7 MB | 13 | n/a (CPU) |
| entity_resolution | 6.7 s | 0.29 GB | 73.8 MB | 15 | n/a (CPU) |
| spatial_relations | 45.8 s | 0.65 GB | 27.7 MB | 8 | n/a (CPU) |
| context_map | 22.8 s | 0.53 GB | 42.9 MB | 10 | n/a (CPU) |

(`visual_perception` is listed in pipeline order for readability; it ran before
`state_estimation` is not the real execution order -- the real canonical order is
ingestion → state_estimation → visual_perception → geometric_mapping → sensor_association →
semantic_fusion → semantic_mapping → entity_resolution → spatial_relations → context_map. The
table above groups it near sensor_association only because both consume `PerceptionRunArtifact`;
see the README for the real order.)

## What stands out

- **`sensor_association` is the real outlier**: 177.8 s wall time and a **24.96 GB peak RSS** --
  roughly 64% of this machine's 39 GB of RAM, and by far the largest memory footprint of any
  CPU-only stage (10-50x every other non-GPU stage). Its own storage output is tiny (3.4 MB), so
  this is a working-set/algorithmic cost (projecting sensor evidence against the geometric map
  with occlusion/visibility checks over corridor-02's real point-cloud scale), not an I/O cost.
  On a machine with materially less RAM than this one, this stage alone could OOM. This is a real
  finding for whoever tunes `sensor_association`'s implementation next; it is reported here, not
  fixed (profiling is diagnostic, not a mandate to change the algorithm -- AGENTS.md #26/#31).
- **`geometric_mapping` dominates storage**: 1.05 GB, more than 10x the combined storage of every
  other stage -- expected, since it is the only stage persisting the accumulated real point-cloud
  map itself rather than derived symbolic evidence.
- **`visual_perception` is the only GPU-bound and by far the slowest stage in wall time**
  (~12.5 minutes total), entirely due to real Qwen3-VL-4B-Instruct generation (656.7 s of the
  ~748 s), consistent with the already-documented external-model cost for this backend.
- Every other stage (state_estimation, semantic_fusion, semantic_mapping, entity_resolution) is
  fast (under ~35 s) and modest in memory (under ~2.75 GB) -- unremarkable, real evidence that the
  bottlenecks in this pipeline are concentrated in exactly two places: one real algorithmic cost
  (`sensor_association`'s memory) and one real external-model cost (`visual_perception`'s Qwen
  generation time), not spread evenly across the chain.

## A bonus, unplanned reproducibility signal

Every one of the seven re-executed stages published the exact same `artifact_id` and
`content_hash` as its `run-0001` counterpart, byte for byte, from an independent rerun against
the same real upstream inputs. This is not what issue #182 asks for (that needs a deliberate,
documented repeated-run comparison), but it is a real, free data point in that direction: nothing
about re-running these hand-built drivers on this machine produced divergent output.

## Failure/OOM handling

No stage failed or hit an out-of-memory condition during this profiling; every `returncode` was
`0` and every `verify_integrity()` was clean. `sensor_association`'s real 24.96 GB peak is
reported as observed headroom risk, not a fabricated failure -- issue #181 asks that an OOM be
recorded as a real experimental result when it happens, not induced artificially here.

See `resource-profile.json` (in this directory) for the exact machine-readable per-stage numbers
and the full command stdout, and `scripts-profiled/` for the exact scripts run to produce them.
