# Final acceptance report, fifth PR #438 review round (run-0005)

Supersedes every prior `acceptance-report*.json`/`.md`. This round is different from the first
four: completing `reproducibility.rerun_equivalence` per the frozen scenario 1.0.4 text
("repeated canonical runs ... yield equivalent artifacts, metric reports and final map") required
an actual **second, independent real execution of Visual Perception** (SAM2 + DINOv2 + CLIP +
Qwen3-VL-4B) over the same 20 real corridor-02 frames -- something no prior round had done, since
every prior "rerun" reused the one real `PerceptionRunArtifact` and only re-executed the
deterministic downstream stages.

## Headline result

**16 of 17 real `passed` gates hold; `reproducibility.rerun_equivalence` FAILS on real, measured
evidence.** This is not a defect in the evidence -- it is a real, measured property of the current
canonical configuration, found by actually running the experiment the requirement describes,
rather than assumed. `unmet_required_gates(scenario, report, kinds=frozenset({GateKind.INVARIANT}))`
returns exactly one entry: `reproducibility.rerun_equivalence`, capability `visual_perception`.

**Solution 1 is not yet ready for v0.1.0 under this campaign's own release-readiness policy.**

## What the real second Visual Perception execution found

Two independent, real executions of the exact same canonical Visual Perception configuration
(SAM2.1-hiera-tiny + DINOv2-base + CLIP-vit-large-patch14 + Qwen3-VL-4B-Instruct nf4, greedy,
`temperature=0.0`) over the same 20 real corridor-02 frames:

| | run-0001 (original) | run-0002 (independent rerun) |
|---|---|---|
| Region discovery (SAM2) | baseline | **0/20 frame mismatches** -- bounding boxes, `region_kind`, `is_accepted` all identical |
| Successful semantic executions | 76/120 | 78/120 |
| Claims | 59 | 60 |
| Scene contexts | 17 | 18 |
| Exact canonical claim agreement | -- | **29/90 (32.2%)** |

Region discovery is perfectly reproducible. Semantic interpretation's aggregate success *rate* is
similar (63% vs 65%), but the actual claim *content* -- the hypothesis text, role, category and
confidence a reader would actually consume -- agrees only 32.2% of the time between the two
independent runs, despite identical greedy/deterministic configuration.

Downstream of Visual Perception, given the *same* perception input, every one of the 8 stages from
`state_estimation` through `context_map` is exactly reproducible: `scripts-run5/
14_metric_report_equivalence.py` found zero metric-report mismatches (counts, diagnostics,
warnings, excluding legitimately run-specific identity) across all 8 stages between two
independent real reruns, and `scripts-run4/12_rerun_content_equivalence.py` already proved exact
entity/relation content equivalence. That evidence is real and stands, but it does not, by itself,
close whole-pipeline rerun equivalence: scenario 1.0.4's requirement is conjunctive over the whole
canonical run, and Visual Perception is a required intermediate artifact of that same run.

**The mechanism is not yet isolated.** GPU floating-point non-determinism cascading through
autoregressive generation is technically plausible (CUDA kernels are not required to be
deterministic unless explicitly configured to be, which this pipeline does not do), but this
experiment does not attribute causality -- that would need a controlled follow-up (fixed seeds,
`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, pinned driver/hardware). This is
recorded as an open question, not asserted as fact.

## Gate-by-gate summary (25 gates, scenario 1.0.4)

| Status | Count | Gates |
|---|---|---|
| `passed` (real) | 16 | ingestion.sequence_integrity, state_estimation.trajectory_coverage, geometric_mapping.map_frame_consistency, visual_perception.evidence_completeness, sensor_association.projection_validity, semantic_fusion.evidence_preservation, entity_resolution.identity_lineage, spatial_relations.reference_integrity, artifact.integrity, cross_stage.\* (4), runtime.provenance_identity, runtime.resource_reporting, reproducibility.interruption_recovery |
| `failed` (real) | 1 | reproducibility.rerun_equivalence |
| `blocked` (no reference set) | 5 | visual_perception.region_quality, visual_perception.semantic_quality, semantic_fusion.reference_recovery, entity_resolution.identity_quality, spatial_relations.relation_quality |
| `not_evaluated` | 3 | state_estimation.accuracy (not applicable by construction), geometric_mapping.quality_report, sensor_association.projection_quality |

See `acceptance-report-run0005.json` for every gate's full evidence and detail text.

## Recommended next steps

1. Open a follow-up issue to isolate the mechanism: does the divergence come from real GPU
   execution non-determinism, from the Qwen output parser/contract being sensitive to tiny
   wording changes, or from the canonical claim-equivalence definition being stricter than the
   scientifically meaningful one? A controlled experiment (deterministic-algorithm flags, fixed
   seeds, pinned hardware) can distinguish these.
2. Depending on that answer, either make the canonical Visual Perception execution deterministic,
   or define and version an explicit semantic-equivalence tolerance for scenario 1.0.4 (or a new
   version) rather than requiring byte-exact claim matches.
3. Do not merge PR #438 or claim v0.1.0 readiness citing this scenario until
   `reproducibility.rerun_equivalence` is resolved one way or the other -- either fixed, or the
   scenario's own tolerance is formally redefined and re-evidenced.

## Superseded evidence

Every prior `acceptance-report*.json`/`.md` (run-0001 through run-0004) is superseded. `run-0001`'s
`PerceptionRunArtifact` remains valid, real evidence for `visual_perception.evidence_completeness`;
it is simply no longer the only real Visual Perception execution this report cites.
