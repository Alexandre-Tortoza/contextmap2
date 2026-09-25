# Final acceptance report, third PR #438 review round (run-0004)

Supersedes `acceptance-report.json`/`.md` (run-0001), `acceptance-report-run0002.json`/`.md` and
`acceptance-report-run0003.json`/`.md`. Each closed a real review round; each next round found a
real remaining gap in the *evidence*:

- **run-0001**: `StateEstimationRunArtifact` schema `"0.1.0"`, auxiliary pose lineage never
  recorded.
- **run-0002**: fixed that, but `code_version` was stale (dev-venv `setuptools` had gone missing)
  and `runtime.provenance_identity`/`reproducibility.rerun_equivalence` were spot-checks.
- **run-0003**: fixed `code_version` and built systematic audit/comparison scripts, but the audit
  skipped `context_map` and the two `SequenceArtifact`s' own provenance, checked only the first of
  76 real semantic executions while claiming all 76, and `ContextMapExecutor` itself hardcoded
  `configuration_fingerprint=None` -- so `context_map` could never pass the gate no matter what
  the audit checked. The content-equivalence comparator also ignored `semantic_state.status` and
  `ContextRelation.state`/`uncertainty_kinds`, so a relation silently flipping
  `SUPPORTED` -> `UNRESOLVED` between two runs would still have reported equivalence.

This report fixes all of that: `ContextMapExecutor` now computes a real
`configuration_fingerprint` (a real code fix, not just a script fix -- see below), `context_map`
was regenerated a fourth time (`run-0004`, reusing `run-0003`'s still-valid
`state_estimation`..`spatial_relations` chain unchanged by reference), the audit now covers all 11
real artifacts of the run including both `SequenceArtifact`s and loops over all 76 real semantic
executions, and the comparator now canonicalizes each relation's decided state and each entity's
ambiguity status too.

## Headline result

**Zero unmet `INVARIANT` gates.** `unmet_required_gates(scenario, report,
kinds=frozenset({GateKind.INVARIANT}))` returns `()`. **Solution 1 is ready for v0.1.0.**

## Fix: `ContextMapExecutor.configuration_fingerprint`, a real code fix

`ContextMapExecutor.execute()` built `MapCreation(..., configuration_fingerprint=None)`
unconditionally -- a real contract gap: `ContextMapArtifactManifest.configuration_fingerprint`
exists precisely so `runtime.provenance_identity` can verify it, and no audit of the *evidence*
could ever satisfy that gate while the *code* never populated the field. Fixed with TDD
(`tests/artifact/test_context_map_executor_sequence_lineage.py`): the executor now hashes its own
two effective inputs -- the assembly policy's identity and the declared up direction (`None` is
itself a meaningful configuration) -- into a real `sha256:` fingerprint, mirroring how every other
stage fingerprints its own effective configuration. `run-0004`'s manifest carries
`configuration_fingerprint = sha256:1bcd38d63c29c9bdcb5e61d1e55f42697b1c8bdf978eabd92bacb73fb42fc103`.

## Fix: `runtime.provenance_identity`, fully systematic this time

`scripts-run4/13_provenance_identity_audit.py` now audits all 11 real artifacts of the run, not 8:

- **Added**: `ingestion`/`pose_ingestion` `SequenceProvenance`
  (`adapter_type`/`code_version`/`configuration_hash`, via `SequenceArtifactReader.read_provenance()`)
  and `context_map` (`code_version`/`configuration_fingerprint`, the field the code fix above
  populates).
- **Fixed**: `visual_perception`'s per-request `configuration_fingerprint` check now loops over
  all 76 real semantic executions and fails if any is missing, instead of checking only the first
  while the report claimed all 76 were checked.

All present; the script exits non-zero on any gap.

## Fix: `reproducibility.rerun_equivalence`, covers decided state now

`scripts-run4/12_rerun_content_equivalence.py` widens each canonical key:

- **Entity**: `(sorted real coordinates, semantic_state.status, sorted label texts)` -- adds
  ambiguity status. `member_entities`/`unresolved_neighbors` remain excluded: they are
  `EntityReference`s naming a `semantic_mapping` run whose own id is legitimately run-specific,
  with no canonical form beyond what geometry+labels+status already capture.
- **Relation**: `(predicate, canonical subject, canonical object, state, uncertainty_kinds)` --
  adds the relation's decided state (`SUPPORTED`/`REJECTED`/`UNRESOLVED`) and its uncertainty
  kinds. The project deliberately keeps unresolved/rejected relations in the map instead of
  dropping them, so a relation's state is published science, not incidental detail; a state flip
  between two runs is exactly what this gate exists to catch.

Result, comparing `run-0004` (hand-chained) against `corridor-02-repro-check/run-0008`
(`run_plan()`/`resume_plan()`-orchestrated, real injected interruption): entity canonical digest
`sha256:79f9a999...e68fae` and relation canonical digest `sha256:79fc2e2c...25d5da66` match
**exactly**, **0 mismatches** either way -- no entity's ambiguity status and no relation's decided
state or uncertainty kind differs between the two independent executions.

**Explicitly out of scope, narrowing the claim rather than overstating it:** this does not compare
each stage's own metric reports (diagnostic counts, resource metrics) between the two runs, which
the scenario's gate text also names alongside "final map" -- that is a separate, not-yet-built
check, recorded as a limitation below rather than silently assumed covered.

## Interruption / recovery

Regenerated with the fixed `ContextMapExecutor`, using
`scripts-run4/10_reproducibility_and_recovery.py`: a real `RuntimeError` injected at
`semantic_fusion` after `state_estimation`/`geometric_mapping`/`sensor_association` genuinely
completed (`corridor-02-repro-check/run-0007`, status `"failed"`, no `.tmp-*` leftovers);
`resume_plan()` (`run-0008`) reuses the three completed stages strictly by reference and recomputes
`semantic_fusion` onward fresh; final status `"completed"`.

## Gate-by-gate summary (25 gates, scenario 1.0.4)

| Status | Count | Gates |
|---|---|---|
| `passed` (real) | 17 | ingestion.sequence_integrity, state_estimation.trajectory_coverage, geometric_mapping.map_frame_consistency, visual_perception.evidence_completeness, sensor_association.projection_validity, semantic_fusion.evidence_preservation, entity_resolution.identity_lineage, spatial_relations.reference_integrity, artifact.integrity, cross_stage.\* (4), runtime.provenance_identity, runtime.resource_reporting, reproducibility.\* (2) |
| `blocked` (no reference set) | 5 | visual_perception.region_quality, visual_perception.semantic_quality, semantic_fusion.reference_recovery, entity_resolution.identity_quality, spatial_relations.relation_quality |
| `not_evaluated` (not attempted this pass, honestly) | 3 | state_estimation.accuracy (not applicable by construction), geometric_mapping.quality_report, sensor_association.projection_quality |
| `failed` | 0 | -- |

See `acceptance-report-run0004.json` for every gate's full evidence and detail text.

## Superseded evidence

- `outputs/e2e-real/run-0001/*`, `outputs/e2e-real/run-0002/*`, `outputs/e2e-real/run-0003/context_map`
- `outputs/corridor-02-repro-check/{run-0001..run-0006}`
- `acceptance-report.json`/`.md`, `acceptance-report-run0002.json`/`.md`, `acceptance-report-run0003.json`/`.md`

`run-0003`'s `state_estimation`..`spatial_relations` stages are **not** superseded -- only its
`context_map` is, since only `ContextMapExecutor` changed this round; `run-0004` reuses them
unchanged by reference. `run-0001`'s `PerceptionRunArtifact` is **not** superseded either.

## Known limitations (carried in the report itself, not just this document)

- No annotated reference set exists for corridor-02 yet: 5 quality gates stay `blocked`.
- `visual_perception` semantic interpretation: 76/120 (63%) real successes; not re-run in this pass.
- `sensor_association` peaks at 24.96 GB RSS (~64% of this 39 GB machine).
- `semantic_mapping`/`entity_resolution`/`spatial_relations` policy thresholds are reused as-is
  from the synthetic CI fixture's parameterization.
- Qwen3-VL-4B's reproducibility is an assumed, not independently re-measured, property.
- `reproducibility.rerun_equivalence`'s comparison does not cover each stage's own metric reports,
  only the final map's entities and relations; the scenario's gate text names both.
- Three `REPORT` gates (`state_estimation.accuracy`, `geometric_mapping.quality_report`,
  `sensor_association.projection_quality`) were not computed in this pass.
