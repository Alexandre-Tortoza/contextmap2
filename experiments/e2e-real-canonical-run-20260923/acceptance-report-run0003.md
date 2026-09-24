# Final acceptance report, second PR #438 review round (run-0003)

Supersedes `acceptance-report.json`/`.md` (run-0001) and `acceptance-report-run0002.json`/`.md`.
Each of those closed a real review round but the next round found a new real gap in the
*evidence*, not in the pipeline's science:

- **run-0001** predates the #555 lineage fix (`StateEstimationRunArtifact` schema `"0.1.0"`).
- **run-0002** fixed that, but a second review round found three more problems in the evidence
  itself: (1) `code_version` in both the report and the regenerated `StateEstimationRunArtifact`
  pointed at a commit 23 commits behind the code that supposedly produced it -- a real dev-venv
  bug (the editable install's `setuptools` had gone missing, silently freezing
  `contextmap.__version__` at an old install); (2) `runtime.provenance_identity` was marked
  `passed` from a two-manifest spot-check, not the systematic check the gate's own definition
  requires; (3) `reproducibility.rerun_equivalence` was argued from `entity_count`/`relation_count`
  equality alone, which proves cardinality, not that the two runs' actual entities/relations agree.

This report fixes all three: the dev venv's editable install was refreshed (`setuptools`
reinstalled, `pip install -e .` rerun), `state_estimation` through `context_map` was regenerated a
third time (`run-0003`) under the now-correct `contextmap.__version__`, and two new scripts turn
the remaining spot-checks into real, systematic, committed evidence.

## Headline result

**Zero unmet `INVARIANT` gates, on evidence that is now as rigorous as the gate definitions
require.** `unmet_required_gates(scenario, report, kinds=frozenset({GateKind.INVARIANT}))` returns
`()`. **Solution 1 is ready for v0.1.0.**

## Fix 1: `code_version` provenance

Root cause: the dev venv (`/.../contextmap2/.venv`) had lost its `setuptools` package at some
point, so `pip install -e '.[dev]'` could no longer rebuild the editable-install metadata
`importlib.metadata.version("contextmap")` reads -- `contextmap.__version__` (and therefore every
`ExternalPoseEstimator`-produced `TrajectoryProvenance.code_version`) silently stayed frozen at
whatever commit was `HEAD` the last time the install actually succeeded (`g7915dea8d`, 23 commits
behind `7a6b5bb`). This affected every real `state_estimation` run this campaign produced with this
venv, not just run-0002's -- geometric_mapping/sensor_association/semantic_fusion never call
`current_code_version()` (their manifests recorded `None`/empty already), and
semantic_mapping/entity_resolution/spatial_relations/context_map used explicit, deliberate
run-label strings passed by the driving scripts, unaffected.

Fix: `pip install setuptools`, then `pip install -e . --no-deps`. `contextmap.__version__` now
reads `0.0.1.dev673+gee3021a50`, matching the real `HEAD` at the time run-0003 was produced.
`scripts-run3/11_assemble_acceptance_report.py` reads this live via `current_code_version()`
instead of a hardcoded string, so this class of staleness cannot silently recur in the report
generator itself; `scripts-run3/01_state_estimation.py` asserts the manifest's own
`code_version` matches at generation time.

## Fix 2: `runtime.provenance_identity`, systematically verified

`scripts-run3/13_provenance_identity_audit.py` reopens all 8 real stage artifacts of run-0003
plus the reused real `PerceptionRunArtifact`, and asserts every identity field the gate's
definition requires is present and non-empty: backend id/version/configuration fingerprint for
`state_estimation`; configuration fingerprint and spatial-index identity for `geometric_mapping`;
configuration fingerprint and visibility-policy fingerprint for `sensor_association`;
configuration digest and per-request configuration fingerprint (checked against all 76 real
semantic executions) for `visual_perception`; support/fusion policy ids and configuration
fingerprints for `semantic_fusion`; materialization policy id, configuration fingerprint and code
digest for `semantic_mapping`; all 4 policy roles' policy ids for `entity_resolution`; all 7 policy
roles' fingerprints and the taxonomy version for `spatial_relations`. No model or pipeline
re-execution needed -- this is a read-only audit of already-materialized manifests, and it exits
non-zero on any gap.

## Fix 3: `reproducibility.rerun_equivalence`, real content comparison

`scripts-run3/12_rerun_content_equivalence.py` opens both independent real reruns (`run-0003`,
hand-chained; `corridor-02-repro-check/run-0006`, `run_plan()`/`resume_plan()`-orchestrated with a
real injected interruption), and compares their actual entities and relations, not their counts:

- Each entity's canonical key is `(sorted real (x, y, z) coordinates its geometry_refs resolve to
  via GeometrySource.get(), sorted label hypothesis texts)` -- deliberately never the run-specific
  `map_id`/`geometry_id`/`entity_id` strings, since two independent, correct executions are not
  expected to produce the same identity strings, only the same real content.
- Each relation's canonical key is `(predicate, canonical key of subject, canonical key of
  object)`, order preserved (several predicates are directional).
- The two runs' canonical entity-key multiset and relation-key multiset are compared directly, and
  each side's canonicalized payload is hashed.

Result: entity canonical digest `sha256:13e3c9da...5dec7` and relation canonical digest
`sha256:e7ced261...39174` match **exactly** between `run-0003` and `run-0006`, **0 mismatches**
either way. Coordinates are rounded to 6 decimal places before comparison -- the geometry pipeline
is deterministic over the same real inputs, so exact equality is expected; rounding only guards
against representation noise across independent processes, not real disagreement. The same two
canonical digests were independently obtained comparing the earlier `run-0002` vs `run-0004` pair
(the code_version-stale pair), confirming **zero scientific drift** from any PR #438 review fix --
only lineage/provenance metadata ever changed across the whole campaign.

**Documented limitation, carried over unchanged:** Qwen3-VL-4B-Instruct's generation (greedy,
`temperature=0`) is *assumed* deterministic on the same hardware/dtype; not independently
re-verified in this pass, since `visual_perception` was reused rather than rerun.

## Interruption / recovery

Regenerated with the corrected code_version, using `scripts-run3/10_reproducibility_and_recovery.py`:

- The real corridor-02 artifacts (bag, pose, perception) are supplied via `provided=` -- no
  re-ingestion, no re-running Qwen.
- A real `RuntimeError` is injected at `semantic_fusion` **after** `state_estimation`,
  `geometric_mapping` and `sensor_association` have genuinely, fully executed for real.
- The interrupted run's status is `"failed"` (`outputs/corridor-02-repro-check/run-0005`);
  `semantic_fusion`'s directory was never created; no `.tmp-*` leftovers.
- `resume_plan()` (`run-0006`) reuses the three completed real stages strictly by reference (their
  `ArtifactRef.location` still points at `run-0005`), recomputes `semantic_fusion` onward fresh,
  and the resumed run's final status is `"completed"`.

## Gate-by-gate summary (25 gates, scenario 1.0.4)

| Status | Count | Gates |
|---|---|---|
| `passed` (real) | 17 | ingestion.sequence_integrity, state_estimation.trajectory_coverage, geometric_mapping.map_frame_consistency, visual_perception.evidence_completeness, sensor_association.projection_validity, semantic_fusion.evidence_preservation, entity_resolution.identity_lineage, spatial_relations.reference_integrity, artifact.integrity, cross_stage.\* (4), runtime.provenance_identity, runtime.resource_reporting, reproducibility.\* (2) |
| `blocked` (no reference set) | 5 | visual_perception.region_quality, visual_perception.semantic_quality, semantic_fusion.reference_recovery, entity_resolution.identity_quality, spatial_relations.relation_quality |
| `not_evaluated` (not attempted this pass, honestly) | 3 | state_estimation.accuracy (not applicable by construction), geometric_mapping.quality_report, sensor_association.projection_quality |
| `failed` | 0 | -- |

(16 of the 17 `passed` gates are `INVARIANT`; `runtime.resource_reporting` is the one `REPORT`
gate marked `passed`.) See `acceptance-report-run0003.json` for every gate's full evidence and
detail text.

## Superseded evidence (do not cite for lineage/reproducibility/recovery/provenance gates)

- `outputs/e2e-real/run-0001/*`, `outputs/e2e-real/run-0002/*`
- `outputs/corridor-02-repro-check/{run-0001,run-0002,run-0003,run-0004}`
- `acceptance-report.json`/`.md`, `acceptance-report-run0002.json`/`.md`

`run-0001`'s `PerceptionRunArtifact` (`visual_perception`) is **not** superseded: that schema was
untouched by every fix in this campaign, and it continues to be reused by reference.

## Known limitations (carried in the report itself, not just this document)

- No annotated reference set exists for corridor-02 yet: 5 quality gates stay `blocked`.
- `visual_perception` semantic interpretation: 76/120 (63%) real successes, dominant failure the
  model omitting the required `confidence` field -- documented, expected model behavior. Not
  re-run in this pass.
- `sensor_association` peaks at 24.96 GB RSS (~64% of this 39 GB machine).
- `semantic_mapping`/`entity_resolution`/`spatial_relations` policy thresholds are reused as-is
  from the synthetic CI fixture's parameterization, not independently tuned for corridor-02.
- Qwen3-VL-4B's reproducibility is an assumed, not independently re-measured, property.
- Three `REPORT` gates (`state_estimation.accuracy`, `geometric_mapping.quality_report`,
  `sensor_association.projection_quality`) were not computed in this pass.
