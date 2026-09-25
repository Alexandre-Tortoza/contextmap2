> **Superseded by `scripts-run5/` and `acceptance-report-run0005.{json,md}`.** Four PR #438
> review rounds found real gaps in the *evidence*; the fifth found a real gap in the *pipeline's
> reproducibility itself*:
>
> - Round 1: `run-0001` was produced under `StateEstimationRunArtifact` `schema_version` `"0.1.0"`,
>   whose `TrajectoryProvenance` never recorded which artifact the merged auxiliary poses actually
>   came from. Fixed and regenerated as `run-0002` / `scripts-run2/` / `acceptance-report-run0002`.
> - Round 2: `run-0002`'s own `code_version` was stale (dev-venv issue), and `runtime.
>   provenance_identity`/`reproducibility.rerun_equivalence` were spot-checks. Fixed and
>   regenerated as `run-0003` / `scripts-run3/` / `acceptance-report-run0003`.
> - Round 3: `run-0003`'s audit still skipped `context_map` and both `SequenceArtifact`s' own
>   provenance; `ContextMapExecutor` hardcoded `configuration_fingerprint=None` (a real code fix);
>   the reproducibility comparator ignored ambiguity status and relation state. Fixed and
>   regenerated as `run-0004` / `scripts-run4/` / `acceptance-report-run0004`.
> - Round 4: the provenance audit still missed `semantic_fusion.fusion_configuration_fingerprint`,
>   `PolicyRef.configuration_fingerprint` for entity_resolution/spatial_relations policies, and
>   real per-backend identity for visual_perception. Fixed in place (no rerun needed).
> - Round 5: completing `reproducibility.rerun_equivalence` per the frozen scenario 1.0.4 text
>   required an actual second, independent real Visual Perception execution (SAM2+DINOv2+CLIP+
>   Qwen3-VL-4B) over the same 20 real frames -- never done in any prior round. **That real
>   experiment found region discovery perfectly reproducible (0/20 mismatches) but Qwen3-VL-4B's
>   semantic claims only 32.2% (29/90) exactly reproducible between the two runs.** This is a
>   real, measured negative result, recorded as `reproducibility.rerun_equivalence = FAILED`, not
>   forced to a passing status the data does not support. See `acceptance-report-run0005.md`.
>
> **Solution 1 is not yet ready for v0.1.0 under this campaign's own release-readiness policy**
> (`unmet_required_gates(kinds={INVARIANT})` is no longer empty). `run-0001` through
> `run-0004/context_map` are kept below for history; do not cite them as lineage/reproducibility/
> recovery/provenance evidence. `run-0001`'s `PerceptionRunArtifact` remains valid evidence for
> `visual_perception.evidence_completeness`. See `acceptance-report-run0005.md` for the full
> picture and recommended next steps.

# Real canonical run of corridor-02 through ContextMapArtifact (issue #177)

This is the audit trail for the first real, end-to-end execution of the Solution 1 canonical
pipeline over corridor-02, from the real ingested sequence (issue #554/#176) and the real
auxiliary pose sequence (issue #555) through every downstream stage to a real, hash-verified
`ContextMapArtifact`. Not committed application code -- a real experiment run against the MAIN
checkout's `outputs/` (untracked, gitignored), matching the convention already established by
`experiments/semantic-fusion-corridor-02-20260921/` and the corridor-02 re-ingestion experiment.

Each script is a thin driver: it builds the real executor from `contextmap.runtime.executors`
with a hand-built `StageRequest`/`ArtifactRef` pointing at the previous stage's real, already
-verified output, and calls `.execute()` directly -- the same executors `compose_executors()`/
`run_plan()` would build automatically, just wired by hand since this is a one-off, order-known
run rather than a `contextmap run` invocation over a resolved plan.

## Result

```text
01_state_estimation.py     -> StateEstimationRunArtifact   b13c6e588aeef1e5142f890399e190b9
                               5522 poses, 0 gaps, ExternalPose over the #555 bridge
02_geometric_mapping.py    -> GeometricMapArtifact          d6ed712aa1a77b91dc93fd951273d938
03_sensor_association.py   -> SensorAssociationRunArtifact  ec435f284cef18bc1980cfd0188022a4
04_semantic_fusion.py      -> SemanticFusionRunArtifact     7253ce0b68b142d58f52a528ace9af13
05_semantic_mapping.py     -> SemanticEntityArtifact        c9137c48ca2baa50425101497eee6947
06_entity_resolution.py    -> EntityResolutionRunArtifact   07616fecdb889fb0cf492f230e812706
07_spatial_relations.py    -> SpatialRelationsRunArtifact   ca6f0ab8478d46ceb7691ef35114bd8b
08_context_map.py          -> ContextMapArtifact            57484aa0948006e9deb4db5b2bd1c4da
                               entity_count=170  relation_count=10852
```

Every stage's own `verify_integrity()` (or, for `context_map`, `ContextMapArtifactReader.open(
..., verify_hashes=True)`) returned clean. The final `ContextMapArtifact`'s dependency closure
records all six real upstream artifacts by content identity (`entity_resolution_run`,
`geometric_map`, `semantic_fusion_run`, `semantic_map`, `sequence`, `spatial_relations_run`).

Upstream real artifacts this run consumed (produced earlier, not by these scripts):

```text
SequenceArtifact (bag):   outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4
SequenceArtifact (pose):  outputs/ingest-real/sequences/corridor-02-pose/e2d832c152b1493999082d4f67210b5b
PerceptionRunArtifact:    outputs/e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception
```

All new artifacts from this run live under `outputs/e2e-real/run-0001/<stage>/`.

## What is real vs. reused-without-independent-tuning

- **state_estimation, geometric_mapping, sensor_association, semantic_fusion**: policy values
  (lookup interpolation gaps, occlusion cell size, geometry-overlap thresholds, ...) match the
  real precedent already validated against this exact dataset in
  `outputs/validation/2026-09-21/{state_estimation,geometric_mapping}/scripts/` and
  `experiments/semantic-fusion-corridor-02-20260921/scripts/{s01..s04}.py`.
- **semantic_mapping, entity_resolution, spatial_relations, context_map**: **no real precedent
  exists anywhere for these four stages on this dataset** -- this is the first time they have
  ever run against real corridor-02 data. Their policy values (entity materialization geometry
  summary, candidate retrieval radius, geometry/contact predicate thresholds) are reused as-is
  from `tests/end_to_end/test_runtime_chain.py`'s synthetic-chain construction -- the only
  currently-endorsed real parameterization of these policies in the codebase. They are **not**
  independently tuned for corridor-02's real physical scale (a ~90 s, tens-of-meters corridor
  traversal, vs. the synthetic fixture's much smaller synthetic scene). `map_frame="map"` and
  `up_axis=AxisDirection.POSITIVE_Z`/`forward_axis=POSITIVE_X` do match corridor-02's real
  geometric map frame and the standard ROS body-frame convention.
- The resulting `relation_count=10852` over `entity_count=170` (~64 relations/entity) is high
  enough to be worth a closer look before this run is treated as final acceptance evidence for
  #177/#178 -- plausibly the candidate/proximity thresholds carried over from the synthetic
  fixture are too permissive at corridor-02's real scale. Flagged for follow-up, not fixed here:
  changing a spatial-relations threshold is a scientific decision (AGENTS.md #22), out of scope
  for "run the pipeline."

  **Follow-up investigation (issue #178, this session): the threshold is not the cause.**
  `manifest.relation_count`/`ContextMap.relations` count every relation *record* regardless of
  `RelationState`, by design (`manifest.py`: "Records in the relation table"; AGENTS.md #4 --
  an ambiguous or rejected candidate is preserved as explicit knowledge, never silently dropped).
  The real breakdown of the 10852 records: **914 (8.4%) `SUPPORTED`**, 8668 (79.9%) `UNRESOLVED`
  (`insufficient_evidence`, "no measured channel decided: geometry (ambiguous)"), 1270 (11.7%)
  `REJECTED`. 914 confirmed relations over 170 entities is ~5.4/entity, an unremarkable number for
  a real corridor scene -- not high. Candidate generation itself is conservative, not permissive:
  of the 14365 possible entity pairs, only 207 (1.4%) fall within `CandidatePolicy.
  proximity_radius_m=0.6` and 884 (6.2%) within `directional_radius_m=2.0`, computed directly from
  the real resolved entities' `bounds_center_m` (2713 unique pairs end up with at least one
  relation record, at most 4 per pair -- two predicates, `next_to`/`touching`, in each direction).
  **Recommendation: no threshold change needed.** If a future report wants a "confirmed relations"
  headline metric distinct from the raw record count, filter by `state == RelationState.SUPPORTED`
  rather than reading `manifest.relation_count` directly -- the field itself is accurately named
  and documented, the earlier flag above simply read it without checking the state breakdown.

## Semantic quality of the perception evidence this run consumed

The upstream `PerceptionRunArtifact` had 76/120 (63%) successful semantic interpretations
(Qwen3-VL-4B-Instruct, nf4) -- consistent with the already-documented real behavior of this
exact model/quantization on this exact dataset (58% in
`src/contextmap/evaluation/docs/semantic-interpretation.md`; dominant failure: the model
omitting the required `confidence` field). This is expected, already-characterized model
behavior, not a defect in this run.

## Cross-stage check against the real run (issue #178)

`scripts/09_cross_stage.py` builds a real `CrossStageInputs` from every reader above (ingestion
through `ContextMapArtifact`) and runs `contextmap.evaluation.cross_stage.check_cross_stage()` --
the first time this check has run against real artifacts rather than the synthetic CI fixture.

```text
checks_run: lineage_closure=31  coordinate_consistency=1160714  evidence_traceability=58795
            physical_observation_identity=813
findings: 0
notes: the trajectory records no calibration identity (ExternalPose does not consume
       calibration) -- calibration lineage is verified between the map and the associations only
```

**Zero findings across all four gates**, over 1.2M total assertions: no lineage break, no
coordinate-frame discontinuity, no lost traceability from `ContextMapArtifact` back to source
evidence, and no physical observation duplicated into distinct downstream evidence. The one note
is an expected, already-documented limitation of the `external_pose` backend, not a defect.

## Environment

Non-GPU stages (`01` through `08`) ran with the worktree's own editable-installed venv
(`.venv/bin/python`, no ML dependencies needed). The upstream `PerceptionRunArtifact` this run
consumes was produced separately with the real ML stack in
`/home/alexmrtr/.cache/contextmap2-audit/venv` (torch/transformers/sam2/bitsandbytes), documented
in its own `outputs/e2e-real/visual_perception/reports/*.json`.
