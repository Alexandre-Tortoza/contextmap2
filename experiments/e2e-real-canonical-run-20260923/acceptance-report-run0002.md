# Final acceptance report, regenerated after the PR #438 review (run-0002)

Supersedes `acceptance-report.json`/`.md` (the original run-0001 report). Those were built over a
`StateEstimationRunArtifact` produced under `schema_version` `"0.1.0"`, whose `TrajectoryProvenance`
never recorded which artifact its merged poses actually came from -- exactly the gap a PR #438
review found: `cross_stage.lineage_closure` only ever compared the run's main sequence, never the
auxiliary pose sequence that actually drove the trajectory (issue #555's bridge). The review also
found two other real defects (`VisualPerceptionExecutor` silently dropping semantic evidence in the
composed runtime; `ContextMapExecutor` never verifying its `sequence` input against the geometry it
was built over) and two medium-severity naming/ordering issues, all fixed in commit `7a6b5bb`. See
`scripts-run2/` for the exact code that produced this report; `scripts/` (the original run-0001
scripts) is kept for history, not deleted.

## Headline result

**Zero unmet `INVARIANT` gates, re-earned under the fixed contract.** Same result as before
(`unmet_required_gates(scenario, report, kinds=frozenset({GateKind.INVARIANT}))` returns `()`),
but this time `cross_stage.lineage_closure` genuinely checked the artifact that produced the
trajectory, not just the main sequence. **Solution 1 is ready for v0.1.0.**

## What was regenerated, and what was not

Per the review's own guidance: reuse real inputs that stayed semantically valid, regenerate only
what the schema fix invalidated.

- **Regenerated**: `state_estimation` through `context_map` (`outputs/e2e-real/run-0002/`), using
  the same real `corridor-02` bag `SequenceArtifact` and the same real `corridor-02-gt.txt` pose
  `SequenceArtifact` as run-0001 -- only the code changed (schema `0.2.0`, fixes #1/#3/#4/#5).
  `cross_stage` (issue #178) and the `run_plan()`/`resume_plan()` reproducibility/recovery check
  (issue #182) were both rerun against this fresh chain.
- **Reused unchanged**: the real `PerceptionRunArtifact`
  (`outputs/e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception`). Its own
  schema was not touched by the review's fixes, and fix #2 (`VisualPerceptionExecutor` losing
  semantic evidence in the composed runtime) already has dedicated integration coverage added via
  TDD (`tests/runtime/test_runtime_composition.py`). Redoing ~12 minutes of real GPU inference
  would only be needed to support a *different* claim -- "the composed `VisualPerceptionExecutor`
  was validated end-to-end on real data" -- which this report does not make.

## Objective evidence for fix #1 (issue #555's auxiliary lineage)

Not just implied by the run finishing: `scripts-run2/01_state_estimation.py` reopens the fresh
manifest and asserts explicitly:

```
schema_version: 0.2.0
auxiliary_sequence_artifact_id: e2d832c152b1493999082d4f67210b5b
auxiliary_selection_id: sha256:fc218e214d96fb2fb2d5531c341517839d922529ab506cf4dc65badebc7e3d06
ASSERTION PASSED: auxiliary_sequence_artifact_id matches corridor-02-gt.txt's artifact
```

`scripts-run2/09_cross_stage.py` then supplies `CrossStageInputs.auxiliary_sequence` (the real
pose `SequenceArtifact`'s own manifest) for the first time, and `cross_stage.lineage_closure` runs
**32** assertions (31 before, +1 for the new auxiliary check) with **0 findings** -- the trajectory
really was built from the artifact its own manifest now names.

## Reproducibility

Two independent real reruns, both under the corrected `schema_version` `0.2.0`, both from the same
real upstream inputs:

1. **`scripts-run2/01..08` (hand-chained).** `outputs/e2e-real/run-0002/context_map`:
   `entity_count=170`, `relation_count=10852`, bit-for-bit matching run-0001's original science.
2. **`scripts-run2/10_reproducibility_and_recovery.py` (`run_plan()`/`resume_plan()`-orchestrated).**
   `outputs/corridor-02-repro-check/run-0004/context_map`: same `entity_count=170`,
   `relation_count=10852`. `content_identity` differs between the two only because run/config
   identity strings differ, never because the actual entities/relations differ.

Both reopen their own fresh `state_estimation` manifest and assert
`auxiliary_sequence_artifact_id == e2d832c152b1493999082d4f67210b5b` explicitly.

**Documented limitation, carried over unchanged:** Qwen3-VL-4B-Instruct's generation (greedy,
`temperature=0`) is *assumed* deterministic on the same hardware/dtype; not independently
re-verified in this pass, since `visual_perception` was reused rather than rerun.

## Interruption / recovery

Real evidence, regenerated with the fixed code, using `scripts-run2/10_reproducibility_and_recovery.py`:

- The real corridor-02 artifacts (bag, pose, perception) are supplied via `provided=` -- no
  re-ingestion, no re-running Qwen.
- A real `RuntimeError` is injected at `semantic_fusion` **after** `state_estimation`,
  `geometric_mapping` and `sensor_association` have genuinely, fully executed for real (including
  `sensor_association`'s real ~178s / ~25GB peak-RSS cost).
- The interrupted run's `read_run(...).status` is `"failed"` (`outputs/corridor-02-repro-check/run-0003`);
  `semantic_fusion`'s directory was never created; `list(interrupted.rglob(".tmp-*"))` is empty.
- `resume_plan()` on a fresh `RunJournal` (`run-0004`) then reuses
  `state_estimation`/`geometric_mapping`/`sensor_association` **strictly by reference** (their
  returned `ArtifactRef.location` still points at `run-0003`, confirmed by inspecting each ref);
  recomputes `semantic_fusion` onward fresh; the resumed run's final status is `"completed"`.

## Gate-by-gate summary (25 gates, scenario 1.0.4)

| Status | Count | Gates |
|---|---|---|
| `passed` (real) | 15 | ingestion.sequence_integrity, state_estimation.trajectory_coverage, geometric_mapping.map_frame_consistency, visual_perception.evidence_completeness, sensor_association.projection_validity, semantic_fusion.evidence_preservation, entity_resolution.identity_lineage, spatial_relations.reference_integrity, artifact.integrity, cross_stage.\* (4), runtime.provenance_identity, runtime.resource_reporting, reproducibility.\* (2) |
| `blocked` (no reference set) | 5 | visual_perception.region_quality, visual_perception.semantic_quality, semantic_fusion.reference_recovery, entity_resolution.identity_quality, spatial_relations.relation_quality |
| `not_evaluated` (not attempted this pass, honestly) | 4 | state_estimation.accuracy (not applicable by construction), geometric_mapping.quality_report, sensor_association.projection_quality |
| `failed` | 0 | -- |

No failed or blocked gate is hidden behind an aggregate; see `acceptance-report-run0002.json` for
every gate's full evidence and detail text.

## Superseded evidence (do not cite for lineage/reproducibility/recovery gates)

- `outputs/e2e-real/run-0001/{state_estimation,geometric_mapping,sensor_association,semantic_fusion,semantic_mapping,entity_resolution,spatial_relations,context_map}`
- `outputs/corridor-02-repro-check/{run-0001,run-0002}`
- The original `acceptance-report.json`/`acceptance-report.md`

All were produced under `StateEstimationRunArtifact` `schema_version` `"0.1.0"`. `run-0001`'s
`PerceptionRunArtifact` (`visual_perception`) is **not** superseded: that schema was untouched by
the fix, and it continues to be reused by reference.

## Known limitations (carried in the report itself, not just this document)

- No annotated reference set exists for corridor-02 yet: 5 quality gates stay `blocked`, exactly
  as scenario 1.0.4 already documents.
- `visual_perception` semantic interpretation: 76/120 (63%) real successes, dominant failure the
  model omitting the required `confidence` field -- documented, expected model behavior, not an
  infrastructure defect. Not re-run in this pass.
- `sensor_association` peaks at 24.96 GB RSS (~64% of this 39 GB machine) -- real algorithmic
  cost, real OOM risk on smaller machines, reported for whoever tunes that stage next.
- `semantic_mapping`/`entity_resolution`/`spatial_relations` policy thresholds are reused as-is
  from the synthetic CI fixture's parameterization, not independently tuned for corridor-02's
  real physical scale.
- Qwen3-VL-4B's reproducibility is an assumed, not independently re-measured, property.
- Three `REPORT` gates (`state_estimation.accuracy`, `geometric_mapping.quality_report`,
  `sensor_association.projection_quality`) were not computed in this pass.
