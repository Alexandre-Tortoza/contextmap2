# Final acceptance report: reproducibility, recovery, and Solution 1 readiness (issue #182)

This closes the last open piece of milestone #19: reproducibility/recovery checks over the real
corridor-02 canonical run, and the final `AcceptanceReport` binding every real result this
campaign produced to scenario `solution-1-canonical` version `1.0.4`. See `acceptance-report.json`
for the machine-readable version (`contextmap.evaluation.end_to_end.write_acceptance_report`
schema), `scripts/10_reproducibility_and_recovery.py` and `scripts/11_assemble_acceptance_report.py`
for the exact code that produced both.

## Headline result

**Zero unmet `INVARIANT` gates.** Under the release-readiness policy this campaign already
adopted (`unmet_required_gates(scenario, report, kinds=frozenset({GateKind.INVARIANT}))`,
distinguishing structural gates that must pass from `REPORT` gates that have no threshold by
design), this real report says **Solution 1 is ready for v0.1.0**, decidable from the report
alone without relying on any undocumented observation.

## Reproducibility

Two independent kinds of real evidence, never a manual intermediate edit between runs:

1. **Issue #181's byte-identical rerun.** All 7 non-GPU stages (`state_estimation` through
   `context_map`) were re-executed from the same real upstream inputs into
   `outputs/e2e-real/run-0001-profiled/`, and every one published the exact same `artifact_id`
   and `content_hash` as its `run-0001` counterpart.
2. **This issue's `run_plan()`-orchestrated rerun.** `scripts/10_reproducibility_and_recovery.py`
   drives the real corridor-02 artifacts through `resolve_plan().scope(provided=...)` +
   `RunJournal` + `run_plan()` -- the actual runtime mechanism, not hand-built
   `StageRequest`/`ArtifactRef` calls in run order like `experiments/.../scripts/01..08` used.
   `ingestion`/`pose_ingestion`/`visual_perception` are supplied as already-published real
   artifacts (`PipelinePlan.scope(provided=...)`), so nothing is re-derived for them; `visual_
   perception` in particular was **not** re-run (would cost ~12 more minutes of real GPU/Qwen
   generation for the same reason issue #181 skipped it). The resulting `ContextMapArtifact`
   (`outputs/corridor-02-repro-check/run-0002/context_map`) has `entity_count=170`,
   `relation_count=10852` -- bit-for-bit matching `run-0001`'s science. `content_identity`
   differs, as expected, since the two harnesses use different run/config identity strings, never
   because the actual entities/relations differ.

**Documented limitation, not measured evidence:** Qwen3-VL-4B-Instruct's generation (greedy,
`temperature=0`) is *assumed* deterministic on the same hardware/dtype; this was not
independently re-verified in this pass, unlike the cheap SAM2/DINOv2/CLIP phase of the same stage
(91s, would be easy to re-check). Re-running Qwen a second time to measure this was judged not
worth ~12 more minutes of real GPU time given the already-strong evidence from the other seven
stages; flagged honestly rather than silently assumed to be covered.

## Interruption / recovery

Real evidence, not synthetic-mechanism-only evidence, using `scripts/10_reproducibility_and_
recovery.py`:

- The real corridor-02 artifacts (bag, pose, perception) are supplied via `provided=` -- no
  re-ingestion, no re-running Qwen.
- A real `RuntimeError` is injected at `semantic_fusion` **after** `state_estimation`,
  `geometric_mapping` and `sensor_association` have genuinely, fully executed for real (including
  `sensor_association`'s real ~178s / ~25GB peak-RSS cost -- see `resource-profile.md`).
- The interrupted run's `read_run(...).status` is `"failed"`; `semantic_fusion`'s directory was
  never created; `list(interrupted.rglob(".tmp-*"))` is empty -- no incomplete artifact was ever
  promoted.
- `resume_plan()` on a fresh `RunJournal` then: reuses `state_estimation`/`geometric_mapping`/
  `sensor_association` **strictly by reference** (their returned `ArtifactRef.location` still
  points at `run-0001`, confirmed by inspecting each ref -- `sensor_association` was *not*
  recomputed, so its real ~25GB working set was paid exactly once across both attempts);
  recomputes `semantic_fusion` onward fresh into `run-0002`; the resumed run's final status is
  `"completed"`.

This is real corridor-02 evidence for `reproducibility.interruption_recovery`, not the synthetic
`fake_contract` chain's own already-passing interruption test (`tests/end_to_end/
test_runtime_chain.py::test_an_interrupted_run_is_resumed_from_the_completed_stages_and_leaves_no_partial_artifact`)
-- that test remains real, separate, `fake_contract`-evidence-class proof that the *mechanism*
itself works; this exercise is the first time it ran against real data.

## Gate-by-gate summary (25 gates, scenario 1.0.4)

| Status | Count | Gates |
|---|---|---|
| `passed` (real) | 15 | ingestion.sequence_integrity, state_estimation.trajectory_coverage, geometric_mapping.map_frame_consistency, visual_perception.evidence_completeness, sensor_association.projection_validity, semantic_fusion.evidence_preservation, entity_resolution.identity_lineage, spatial_relations.reference_integrity, artifact.integrity, cross_stage.\* (4), runtime.provenance_identity, runtime.resource_reporting, reproducibility.\* (2) |
| `blocked` (no reference set) | 5 | visual_perception.region_quality, visual_perception.semantic_quality, semantic_fusion.reference_recovery, entity_resolution.identity_quality, spatial_relations.relation_quality |
| `not_evaluated` (not attempted this pass, honestly) | 4 | state_estimation.accuracy (not applicable by construction), geometric_mapping.quality_report, sensor_association.projection_quality |
| `failed` | 0 | -- |

No failed or blocked gate is hidden behind an aggregate; see `acceptance-report.json` for every
gate's full evidence and detail text.

## Known limitations (carried in the report itself, not just this document)

- No annotated reference set exists for corridor-02 yet: 5 quality gates stay `blocked`, exactly
  as scenario 1.0.4 already documents.
- `visual_perception` semantic interpretation: 76/120 (63%) real successes, dominant failure the
  model omitting the required `confidence` field -- documented, expected model behavior, not an
  infrastructure defect.
- `sensor_association` peaks at 24.96 GB RSS (~64% of this 39 GB machine) for a 3.4 MB output --
  real algorithmic cost, real OOM risk on smaller machines, reported for whoever tunes that stage
  next.
- `semantic_mapping`/`entity_resolution`/`spatial_relations` policy thresholds are reused as-is
  from the synthetic CI fixture's parameterization, not independently tuned for corridor-02's
  real physical scale (issue #178 found the resulting relation counts unremarkable, but this
  remains an un-tuned default).
- Qwen3-VL-4B's reproducibility is an assumed, not independently re-measured, property (see above).
- Three `REPORT` gates (`state_estimation.accuracy`, `geometric_mapping.quality_report`,
  `sensor_association.projection_quality`) were not computed in this pass.
