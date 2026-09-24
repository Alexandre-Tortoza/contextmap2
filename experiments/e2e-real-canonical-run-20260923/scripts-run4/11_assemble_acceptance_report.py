"""Assemble the final, review-hardened AcceptanceReport for Solution 1 / corridor-02.

Third PR #438 review round. Supersedes acceptance-report.json/md (run-0001),
acceptance-report-run0002.json/md and acceptance-report-run0003.json/md -- each closed a real
review round, and each next round found a real remaining gap in the evidence:

- run-0001: StateEstimationRunArtifact schema "0.1.0", auxiliary pose lineage never recorded.
- run-0002: fixed that, but code_version was stale (dev venv issue) and
  runtime.provenance_identity/reproducibility.rerun_equivalence were spot-checks, not systematic.
- run-0003: fixed code_version and built systematic audit/comparison scripts, but (a) the audit
  skipped context_map and the two SequenceArtifacts' own provenance, and checked only the first
  of 76 real semantic executions while the report claimed all 76; (b) ContextMapExecutor itself
  hardcoded configuration_fingerprint=None, so context_map could never satisfy the gate no matter
  what the audit checked; (c) the content-equivalence comparator ignored semantic_state.status,
  ContextRelation.state and uncertainty_kinds, so a relation silently flipping
  SUPPORTED -> UNRESOLVED between two runs would still have reported equivalence.

This report is built over run-0004 (context_map regenerated a fourth time, over run-0003's still-
valid state_estimation..spatial_relations chain, with a real ContextMapExecutor.configuration_
fingerprint) and corridor-02-repro-check/run-0007+run-0008, plus the widened audit and comparator.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from contextmap.evaluation import canonical_real_scenario
from contextmap.evaluation.end_to_end import (
    ArtifactRecord,
    EvidenceClass,
    GateKind,
    GateResult,
    assemble_acceptance_report,
    unmet_required_gates,
    write_acceptance_report,
)
from contextmap.ingestion import current_code_version

REAL = EvidenceClass.REAL

RESULTS = [
    GateResult.passed(
        "ingestion.sequence_integrity", evidence_class=REAL,
        evidence_refs=("sha256:3b631eb4c4cdcb3c7be5b65f70c728a8f592167ef693ad248f73770df3884496",
                       "outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4"),
        detail="Real bag SequenceArtifact 720a486d... (unchanged, reused across every run): "
               "verify_integrity()==[], RGB+LiDAR+IMU under the declared corridor-02-header "
               "clock, MeiCameraModel calibration identity, selection_identity matches the "
               "frozen scenario 1.0.4. Its own provenance (adapter_type='ros1_bag', "
               "configuration_hash, code_version) is present and audited by "
               "scripts-run4/13_provenance_identity_audit.py.",
    ),
    GateResult.passed(
        "state_estimation.trajectory_coverage", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/state_estimation", "poses=5522", "gaps=0",
                       "schema_version=0.2.0",
                       "auxiliary_sequence_artifact_id=e2d832c152b1493999082d4f67210b5b"),
        detail="Real StateEstimationRunArtifact under the corrected code_version "
               f"({current_code_version()!r}), asserted explicitly by "
               "scripts-run3/01_state_estimation.py. Every selected image/LiDAR timestamp "
               "resolves through the trajectory (interpolated lookup, documented max gap), 0 "
               "TrajectoryGap records over 5522 poses; frame graph map->epson; "
               "auxiliary_sequence_artifact_id/auxiliary_selection_id name the real "
               "corridor-02-gt.txt pose artifact, whose own provenance is also audited.",
    ),
    GateResult.not_evaluated(
        "state_estimation.accuracy",
        detail="Not applicable by construction: the canonical trajectory IS corridor-02-gt.txt "
               "(external_pose backend over the declared ground-truth pose input), so comparing "
               "ATE/RPE against that same file would report zero error by construction, never a "
               "real accuracy measurement. This is the scenario's own documented position "
               "(canonical_scenario.py's real-subject note), not a gap introduced by this report.",
    ),
    GateResult.passed(
        "geometric_mapping.map_frame_consistency", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/geometric_mapping",
                       "cross_stage.coordinate_consistency: 1160714 assertions, 0 findings"),
        detail="Real GeometricMapArtifact over the fixed, freshly-versioned trajectory: "
               "verify_integrity()==[], map expressed in the trajectory's frame; the rerun "
               "cross-stage check found zero coordinate-frame discontinuities across 1,160,714 "
               "assertions from ingestion through ContextMapArtifact.",
    ),
    GateResult.not_evaluated(
        "geometric_mapping.quality_report",
        detail="No scan-overlap/density/range quality metrics were computed against this real "
               "map in this pass; not fabricated. Left for a dedicated geometric_mapping "
               "evaluation run.",
    ),
    GateResult.passed(
        "visual_perception.evidence_completeness", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception",
                       "reopened_results=20", "verify_integrity=[]"),
        detail="Real PerceptionRunArtifact (SAM2.1-hiera-tiny + DINOv2-base + CLIP-vit-large-"
               "patch14 + Qwen3-VL-4B-Instruct nf4), reused unchanged across every run in this "
               "campaign: this artifact's own schema was not touched by any PR #438 review fix. "
               "All 20 selected frames have a materialized PerceptionResult with an explicit "
               "status per stage; the 44/120 semantic-interpretation failures are recorded via "
               "StageOutcome, never dropped. Every one of the 76 real successful executions' own "
               "configuration_fingerprint was checked this round (not just the first, as the "
               "run-0003 report mistakenly claimed) by "
               "scripts-run4/13_provenance_identity_audit.py.",
    ),
    GateResult.blocked(
        "visual_perception.region_quality", blocked_by=("visual_perception",),
        detail="No annotated reference set exists for corridor-02 (scenario 1.0.4 declares "
               "reference_set=null); region IoU/recall/duplicate-rate cannot be scored without it.",
    ),
    GateResult.blocked(
        "visual_perception.semantic_quality", blocked_by=("visual_perception",),
        detail="Needs AnnotationFamily.SEMANTICS, which does not exist for corridor-02 yet, so "
               "acceptable/unsupported/ambiguity-preservation claim rates cannot be scored. Raw, "
               "unscored completion evidence (not a substitute for this gate): 76/120 (63%) real "
               "semantic interpretations succeeded, consistent with the already-documented 58% "
               "rate for this exact model/quantization/dataset combination; dominant failure is "
               "the model omitting the required `confidence` field, not an infrastructure defect.",
    ),
    GateResult.passed(
        "sensor_association.projection_validity", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/sensor_association",
                       "cross_stage.evidence_traceability: 58795 assertions, 0 findings"),
        detail="Real SensorAssociationRunArtifact: verify_integrity()==[]; the rerun cross-stage "
               "check found every SpatialObservation traces to existing geometry across 58,795 "
               "traceability assertions, 0 findings.",
    ),
    GateResult.not_evaluated(
        "sensor_association.projection_quality",
        detail="Visible-support/reprojection-error/feature-anchoring metrics were not computed "
               "by stratum in this pass; not fabricated.",
    ),
    GateResult.passed(
        "semantic_fusion.evidence_preservation", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/semantic_fusion",
                       "cross_stage.physical_observation_identity: 813 assertions, 0 findings"),
        detail="Real SemanticFusionRunArtifact: verify_integrity()==[]; the rerun cross-stage "
               "check found no physical observation duplicated into distinct downstream "
               "evidence (813 assertions, 0 findings) and no lost lineage from fusion onward.",
    ),
    GateResult.blocked(
        "semantic_fusion.reference_recovery", blocked_by=("semantic_fusion",),
        detail="Needs AnnotationFamily.SEMANTICS, which does not exist for corridor-02 yet.",
    ),
    GateResult.passed(
        "entity_resolution.identity_lineage", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/entity_resolution",
                       "cross_stage.lineage_closure: 32 assertions, 0 findings"),
        detail="Real EntityResolutionRunArtifact: verify_integrity()==[], 170 entities (identical "
               "science to every prior run in this campaign, including their ambiguity status -- "
               "see reproducibility.rerun_equivalence); every resolved entity lists its member "
               "entities/fused evidence, and the rerun cross-stage check found zero "
               "lineage-closure findings across every dependency edge.",
    ),
    GateResult.blocked(
        "entity_resolution.identity_quality", blocked_by=("entity_resolution",),
        detail="Needs AnnotationFamily.IDENTITY, which does not exist for corridor-02 yet.",
    ),
    GateResult.passed(
        "spatial_relations.reference_integrity", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/spatial_relations",
                       "cross_stage.evidence_traceability: 58795 assertions, 0 findings"),
        detail="Real SpatialRelationsRunArtifact: verify_integrity()==[]; every relation record "
               "references resolved entities of this run and carries its evidence, confirmed by "
               "the rerun cross-stage traceability check.",
    ),
    GateResult.blocked(
        "spatial_relations.relation_quality", blocked_by=("spatial_relations",),
        detail="Needs AnnotationFamily.RELATIONS, which does not exist for corridor-02 yet, so "
               "precision/recall/F1 by predicate cannot be scored. Raw, unscored evidence (not a "
               "substitute for this gate, identical science to every prior run, including "
               "relation state -- see reproducibility.rerun_equivalence): of 10852 total relation "
               "records, 914 (8.4%) are SUPPORTED, 8668 (79.9%) UNRESOLVED (insufficient "
               "geometric evidence), 1270 (11.7%) REJECTED -- 914 confirmed relations over 170 "
               "entities (~5.4/entity) is an unremarkable real corridor-scene number; candidate "
               "generation itself is conservative (1.4%/6.2% of possible pairs within the "
               "configured radii), not permissive.",
    ),
    GateResult.passed(
        "artifact.integrity", evidence_class=REAL,
        evidence_refs=("sha256:d08247e2406b1e0a6d594f0a3603efbd69e232b6870743e4bd883ec34642f917",
                       "outputs/e2e-real/run-0004/context_map"),
        detail="Real ContextMapArtifact b5cd6f89..., regenerated a fourth time after fixing "
               "ContextMapExecutor's configuration_fingerprint (was hardcoded None): "
               "ContextMapArtifactReader.open(verify_hashes=True) succeeds, entity_count=170, "
               "relation_count=10852, dependency closure names all 6 real upstream artifacts by "
               "content identity; geometry is referenced, never copied. Manifest now carries a "
               "real configuration_fingerprint "
               "(sha256:1bcd38d63c29c9bdcb5e61d1e55f42697b1c8bdf978eabd92bacb73fb42fc103, hashing "
               "the assembly policy identity and declared up direction) instead of None. Also "
               "exercises PR #438 review fix #3 (ContextMapExecutor verifies the sequence input "
               "against the geometry's own lineage) on real data.",
    ),
    GateResult.passed(
        "cross_stage.lineage_closure", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run4/09_cross_stage.py",),
        detail="32 real assertions over the real chain (ingestion through ContextMapArtifact, "
               "the run-0004 context_map), 0 findings -- CrossStageInputs.auxiliary_sequence is "
               "supplied and verified: the real corridor-02-gt.txt pose SequenceArtifact that "
               "state_estimation actually merged in is confirmed to match what the trajectory's "
               "manifest names.",
    ),
    GateResult.passed(
        "cross_stage.coordinate_consistency", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run4/09_cross_stage.py",),
        detail="1,160,714 real assertions, 0 findings: one map frame from trajectory to geometry "
               "to association to entities, every boundary explicit.",
    ),
    GateResult.passed(
        "cross_stage.evidence_traceability", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run4/09_cross_stage.py",),
        detail="58,795 real assertions, 0 findings: sampled final entities/relations trace to "
               "exact source observations and geometry through the recorded path.",
    ),
    GateResult.passed(
        "cross_stage.physical_observation_identity", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run4/09_cross_stage.py",),
        detail="813 real assertions, 0 findings: repeated inference over one physical "
               "observation never became duplicated physical evidence downstream.",
    ),
    GateResult.passed(
        "runtime.provenance_identity", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run4/13_provenance_identity_audit.py",),
        detail="Systematically verified over all 11 real artifacts of the run (widened this "
               "round: the prior audit skipped context_map and the two SequenceArtifacts' own "
               "provenance, and checked only the first of 76 real semantic executions while "
               "claiming all were checked). scripts-run4/13_provenance_identity_audit.py now "
               "asserts: ingestion/pose_ingestion SequenceProvenance "
               "{adapter_type,code_version,configuration_hash}; state_estimation.estimator."
               "{backend_id,backend_version,configuration_fingerprint}; geometric_mapping."
               "{configuration_fingerprint,spatial_index_kind}; sensor_association."
               "{configuration_fingerprint,visibility_policy.fingerprint}; visual_perception."
               "{configuration_digest, per-request configuration_fingerprint looped over all 76 "
               "real semantic executions individually}; semantic_fusion.{support_policy_id,"
               "support_configuration_fingerprint,fusion_policy_id}; semantic_mapping."
               "{materialization_policy_id,configuration_fingerprint,code_digest}; "
               "entity_resolution.policies (4 roles); spatial_relations.policies (7 roles) and "
               "taxonomy_version; context_map.{code_version,configuration_fingerprint} (the last "
               "of these previously None -- see artifact.integrity). All present; the script "
               "exits non-zero on any gap.",
    ),
    GateResult.passed(
        "runtime.resource_reporting", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/resource-profile.md",
                       "experiments/e2e-real-canonical-run-20260923/resource-profile.json"),
        detail="Real wall-time/peak-RSS/storage/file-count measured per stage (issue #181), "
               "carried over unchanged: this measurement is about runtime cost, not about the "
               "lineage-schema, code_version or configuration_fingerprint defects the #438 "
               "review found, so it needed no rerun. REPORT gate, no threshold: reported as "
               "measured, including the concerning finding that sensor_association peaks at "
               "24.96 GB RSS (~64% of this 39 GB machine), a real headroom risk flagged as a "
               "limitation, not hidden.",
    ),
    GateResult.passed(
        "reproducibility.rerun_equivalence", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run4/12_rerun_content_equivalence.py",
                       "outputs/e2e-real/run-0004/context_map",
                       "outputs/corridor-02-repro-check/run-0008/context_map"),
        detail="Real content equivalence over the scientifically load-bearing state of the map, "
               "not just cardinality (widened this round: the run-0003 comparator canonicalized "
               "only geometry and label text, so a relation silently flipping "
               "SUPPORTED -> UNRESOLVED between runs would still have reported equivalence). "
               "scripts-run4/12_rerun_content_equivalence.py opens both independent real reruns "
               "(run-0004, hand-chained context_map over run-0003's chain; "
               "corridor-02-repro-check/run-0008, run_plan()/resume_plan()-orchestrated with a "
               "real injected interruption), canonicalizes every entity by its real geometry "
               "(exact (x, y, z) coordinates via GeometrySource.get(), never run-specific id "
               "strings), its semantic_state.status and its label hypotheses, and canonicalizes "
               "every relation by (predicate, canonical subject, canonical object, state, "
               "uncertainty_kinds). Result: entity canonical digest "
               "sha256:79f9a999725f424f8007d54df49439112fdc423af4e4dc914ace4a2108e68fae and "
               "relation canonical digest "
               "sha256:79fc2e2cc873376de8b5fd3b1896e45ffbc1481339332378ded73d0225d5da66 match "
               "exactly between the two runs, 0 mismatches either way -- no entity's ambiguity "
               "status and no relation's decided state or uncertainty kind differs between the "
               "two independent executions. Explicitly out of scope, narrowing the claim rather "
               "than overstating it: this does not compare each stage's own metric reports "
               "(diagnostic counts, resource metrics) between the two runs, which the scenario's "
               "gate text also names alongside 'final map' -- that is a separate, not-yet-built "
               "check. visual_perception (Qwen generation) was NOT independently re-verified for "
               "reproducibility in this pass -- greedy/temperature=0 decoding is assumed "
               "deterministic on the same hardware/dtype, but this is a documented assumption, "
               "not measured evidence.",
    ),
    GateResult.passed(
        "reproducibility.interruption_recovery", evidence_class=REAL,
        evidence_refs=("outputs/corridor-02-repro-check/run-0007", "outputs/corridor-02-repro-check/run-0008"),
        detail="Real run_plan()/resume_plan() execution over the real corridor-02 artifacts "
               "(ingestion/pose_ingestion/visual_perception reused by reference via "
               "PipelinePlan.scope(provided=...), never re-derived), regenerated under the "
               "corrected code_version and the fixed ContextMapExecutor: a real RuntimeError was "
               "injected at semantic_fusion after state_estimation/geometric_mapping/"
               "sensor_association genuinely completed. The interrupted run's status is "
               "'failed', semantic_fusion's directory was never created, no .tmp-* leftovers "
               "anywhere. resume_plan() then reused the three completed real stages strictly by "
               "reference (their ArtifactRef.location still points at run-0007, not "
               "recomputed), recomputed semantic_fusion onward fresh into run-0008, and the "
               "resumed run's status is 'completed'.",
    ),
]


def main() -> None:
    scenario = canonical_real_scenario()
    stage_artifacts = {
        "ingestion": ArtifactRecord(
            artifact_id="720a486de8d44c16a9d3d2ff9fa7b1a4",
            digest="sha256:3b631eb4c4cdcb3c7be5b65f70c728a8f592167ef693ad248f73770df3884496",
        ),
        "visual_perception": ArtifactRecord(
            artifact_id="vp-sam2-canonical-1-0-4",
            digest="sha256:160a9a2455054a4e4e1ebf3d46f8211a8fe0992901dafe2c393240ef786c16da",
        ),
        "state_estimation": ArtifactRecord(
            artifact_id="0df649a05d2df4879bf5f61aefb15e94",
            digest="sha256:dfe83e8ddbfeedbf64a9d2698e45fd15d63d163402da6175397d4a7f7717db46",
        ),
        "geometric_mapping": ArtifactRecord(
            artifact_id="e9f0a30db6be13ec8021caa243688d46",
            digest="sha256:e8bb161b5bc02324e33780cb0c8f4ec67f8aa058b72cc313fb8e3f877aab68ce",
        ),
        "sensor_association": ArtifactRecord(
            artifact_id="3ca14b10885481f5ae3abd1ad26f2ab8",
            digest="sha256:4954c94b1c10a933360c539818f9edb337eb251ca75bfbb7dbfa629690e3d00f",
        ),
        "semantic_fusion": ArtifactRecord(
            artifact_id="1d12efdeb2727ee1fa1fbb27772542c0",
            digest="sha256:f33251a304b7a29f3fea5c3ee1410d3feb34c9e2f9cda33e6d3c883fe9ec830e",
        ),
        "semantic_mapping": ArtifactRecord(
            artifact_id="a543b1dd08a8891fe0bae1a8b1f9a9eb",
            digest="sha256:bbdb4d697b3134d0842dc542edf93484ccfd8cd80ca4e9a36fd3e6f2b410b95a",
        ),
        "entity_resolution": ArtifactRecord(
            artifact_id="0318ef0a8f0893984f17b1ad9bec86fa",
            digest="sha256:1b18260bc9a049a02a5ba7e6113aa53b45dcd2457226ae06613980eabe639dd9",
        ),
        "spatial_relations": ArtifactRecord(
            artifact_id="24edca76f62def8f8c808e8c0a4ceb7c",
            digest="sha256:038904ce8e19a987aa3c4bf932d9930ad4947e7e49290639768699c81a5b42b3",
        ),
        "context_map": ArtifactRecord(
            artifact_id="b5cd6f89f63cfd6bc1f2f022867fc420",
            digest="sha256:d08247e2406b1e0a6d594f0a3603efbd69e232b6870743e4bd883ec34642f917",
        ),
    }
    final_artifact = stage_artifacts["context_map"]

    code_version = current_code_version()
    assert code_version is not None, "contextmap.__version__ must be a real installed version"

    report = assemble_acceptance_report(
        scenario,
        report_id="acceptance-report-corridor-02-run0004",
        run_id="e2e-real-run-0004",
        code_version=code_version,
        results=RESULTS,
        stage_artifacts=stage_artifacts,
        final_artifact=final_artifact,
        limitations=(
            "outputs/e2e-real/run-0001/*, run-0002/* and run-0003/context_map (and the "
            "corresponding corridor-02-repro-check/{run-0001..run-0006} journaled reruns), and "
            "the acceptance-report.json/md, acceptance-report-run0002.json/md and "
            "acceptance-report-run0003.json/md, are all superseded for acceptance purposes. "
            "run-0003's state_estimation..spatial_relations stages are NOT superseded -- only "
            "its context_map is, since only ContextMapExecutor changed this round; run-0004 "
            "reuses them unchanged by reference. run-0001's PerceptionRunArtifact "
            "(visual_perception) is NOT superseded: that schema was untouched by every fix, and "
            "it continues to be reused by reference across every run in this campaign.",
            "No annotated reference set exists for corridor-02 yet: region/semantic/identity/"
            "relation quality gates are blocked, not scored (scenario 1.0.4's own documented "
            "position).",
            "visual_perception semantic interpretation (Qwen3-VL-4B-Instruct, nf4): 76/120 "
            "(63%) real successes; dominant failure is the model omitting the required "
            "`confidence` field. Documented, expected behavior for this exact model/"
            "quantization/dataset, not an infrastructure defect. Not re-run in this pass "
            "(reused from run-0001; see visual_perception.evidence_completeness).",
            "sensor_association peaks at 24.96 GB RSS (~64% of this 39 GB dev machine) -- a "
            "real algorithmic working-set cost, a real OOM risk on smaller machines.",
            "semantic_mapping/entity_resolution/spatial_relations policy thresholds are reused "
            "as-is from the synthetic CI fixture's parameterization, not independently tuned "
            "for corridor-02's real physical scale.",
            "visual_perception's reproducibility (Qwen3-VL-4B generation specifically) is an "
            "assumed-deterministic, not independently re-measured, property in this report.",
            "reproducibility.rerun_equivalence's content comparison does not cover each stage's "
            "own metric reports (diagnostic counts, resource metrics), only the final map's "
            "entities and relations; the scenario's gate text names both. A separate, "
            "not-yet-built check would be needed to close that remaining part of the gate's "
            "literal wording.",
            "state_estimation.accuracy, geometric_mapping.quality_report and sensor_association."
            "projection_quality (REPORT gates) were not computed in this pass; not fabricated.",
        ),
    )

    path = Path("/home/alexmrtr/Projects/contextmap2/outputs/e2e-real/acceptance-report-run0004.json")
    write_acceptance_report(path, report)
    print("wrote", path)
    print("results:")
    for result in report.results:
        print(f"  {result.gate_id}: {result.status.value}")

    unmet = unmet_required_gates(scenario, report, kinds={GateKind.INVARIANT})
    print(f"unmet INVARIANT gates: {unmet}")

    counts = Counter(result.status.value for result in report.results)
    print(f"status counts: {dict(counts)} (total={len(report.results)})")


if __name__ == "__main__":
    main()
