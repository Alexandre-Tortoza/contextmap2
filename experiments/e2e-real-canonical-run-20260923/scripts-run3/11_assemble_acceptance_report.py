"""Assemble the final, review-hardened AcceptanceReport for Solution 1 / corridor-02.

Supersedes acceptance-report.json/md (run-0001) and acceptance-report-run0002.json/md: those two
were each closing PR #438 review rounds, but each left a new, real gap the next round found:

- run-0001 was built over a StateEstimationRunArtifact under schema_version "0.1.0", whose
  TrajectoryProvenance never recorded which artifact its merged poses actually came from.
- run-0002 fixed that, but (a) embedded a stale code_version in its own StateEstimationRun
  Artifact and in this report (the dev venv's editable install metadata was 23 commits behind
  the code that actually produced the run -- a real environment bug, not just a report bug,
  fixed here by refreshing the install and reading contextmap.__version__ live, never a literal
  string); (b) marked runtime.provenance_identity `passed` from a two-manifest spot-check, not a
  systematic check of every stage; (c) claimed reproducibility.rerun_equivalence from
  entity_count/relation_count equality alone, which proves cardinality, not content equivalence.

This report is built entirely over run-0003 (state_estimation..context_map, regenerated a third
time after refreshing the dev venv's editable install so contextmap.__version__ -- and therefore
every ExternalPoseEstimator-produced TrajectoryProvenance.code_version -- reflects the actual
commit) and the corridor-02-repro-check/run-0005+run-0006 journaled rerun, plus two new scripts
that turn the two remaining spot-checks into real, systematic, committed evidence:
scripts-run3/12_rerun_content_equivalence.py (canonicalizes and compares real entity/relation
content, not just counts) and scripts-run3/13_provenance_identity_audit.py (reopens every real
stage artifact and asserts the required backend/policy identity and configuration digest fields).
"""

from __future__ import annotations

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
               "frozen scenario 1.0.4.",
    ),
    GateResult.passed(
        "state_estimation.trajectory_coverage", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/state_estimation", "poses=5522", "gaps=0",
                       "schema_version=0.2.0",
                       "auxiliary_sequence_artifact_id=e2d832c152b1493999082d4f67210b5b"),
        detail="Real StateEstimationRunArtifact, regenerated a third time after refreshing the "
               "dev venv's editable install (contextmap.__version__ was stale -- 23 commits "
               "behind -- because the venv's setuptools package had gone missing, silently "
               "breaking `pip install -e` refreshes; fixed by reinstalling setuptools and "
               "reinstalling the package). manifest.code_version is now "
               f"{current_code_version()!r}, the real current commit, asserted explicitly by "
               "scripts-run3/01_state_estimation.py, not read from a hardcoded string. Every "
               "selected image/LiDAR timestamp resolves through the trajectory (interpolated "
               "lookup, documented max gap), 0 TrajectoryGap records over 5522 poses; frame "
               "graph map->epson; auxiliary_sequence_artifact_id/auxiliary_selection_id name "
               "the real corridor-02-gt.txt pose artifact.",
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
        detail="Regenerated real GeometricMapArtifact over the fixed, freshly-versioned "
               "trajectory: verify_integrity()==[], map expressed in the trajectory's frame; "
               "the rerun cross-stage check found zero coordinate-frame discontinuities across "
               "1,160,714 assertions from ingestion through ContextMapArtifact.",
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
               "campaign: this artifact's own schema was not touched by any PR #438 review "
               "fix, and the composed VisualPerceptionExecutor's semantic-evidence-loss finding "
               "(#2) already has dedicated integration coverage (test_runtime_composition.py). "
               "All 20 selected frames have a materialized PerceptionResult with an explicit "
               "status per stage; the 44/120 semantic-interpretation failures are recorded via "
               "StageOutcome, never dropped.",
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
        detail="Regenerated real SensorAssociationRunArtifact: verify_integrity()==[]; the rerun "
               "cross-stage check found every SpatialObservation traces to existing geometry "
               "across 58,795 traceability assertions, 0 findings.",
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
        detail="Regenerated real SemanticFusionRunArtifact: verify_integrity()==[]; the rerun "
               "cross-stage check found no physical observation duplicated into distinct "
               "downstream evidence (813 assertions, 0 findings) and no lost lineage from fusion "
               "onward.",
    ),
    GateResult.blocked(
        "semantic_fusion.reference_recovery", blocked_by=("semantic_fusion",),
        detail="Needs AnnotationFamily.SEMANTICS, which does not exist for corridor-02 yet.",
    ),
    GateResult.passed(
        "entity_resolution.identity_lineage", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/entity_resolution",
                       "cross_stage.lineage_closure: 32 assertions, 0 findings"),
        detail="Regenerated real EntityResolutionRunArtifact: verify_integrity()==[], 170 "
               "entities (identical science to every prior run in this campaign -- see "
               "reproducibility.rerun_equivalence); every resolved entity lists its member "
               "entities/fused evidence, and the rerun cross-stage check found zero "
               "lineage-closure findings across every dependency edge, including the auxiliary "
               "pose sequence's own edge.",
    ),
    GateResult.blocked(
        "entity_resolution.identity_quality", blocked_by=("entity_resolution",),
        detail="Needs AnnotationFamily.IDENTITY, which does not exist for corridor-02 yet.",
    ),
    GateResult.passed(
        "spatial_relations.reference_integrity", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0003/spatial_relations",
                       "cross_stage.evidence_traceability: 58795 assertions, 0 findings"),
        detail="Regenerated real SpatialRelationsRunArtifact: verify_integrity()==[]; every "
               "relation record references resolved entities of this run and carries its "
               "evidence, confirmed by the rerun cross-stage traceability check.",
    ),
    GateResult.blocked(
        "spatial_relations.relation_quality", blocked_by=("spatial_relations",),
        detail="Needs AnnotationFamily.RELATIONS, which does not exist for corridor-02 yet, so "
               "precision/recall/F1 by predicate cannot be scored. Raw, unscored evidence (not a "
               "substitute for this gate, identical science to every prior run): of 10852 total "
               "relation records, 914 (8.4%) are SUPPORTED, 8668 (79.9%) UNRESOLVED (insufficient "
               "geometric evidence), 1270 (11.7%) REJECTED -- 914 confirmed relations over 170 "
               "entities (~5.4/entity) is an unremarkable real corridor-scene number; candidate "
               "generation itself is conservative (1.4%/6.2% of possible pairs within the "
               "configured radii), not permissive.",
    ),
    GateResult.passed(
        "artifact.integrity", evidence_class=REAL,
        evidence_refs=("sha256:a36bc01419c14832859012a409a6ab79657ed2fa9c88d8fe34d2b549a70a158a",
                       "outputs/e2e-real/run-0003/context_map"),
        detail="Regenerated real ContextMapArtifact f777d44d...: ContextMapArtifactReader.open("
               "verify_hashes=True) succeeds, entity_count=170, relation_count=10852, "
               "dependency closure names all 6 real upstream artifacts by content identity; "
               "geometry is referenced, never copied. Also exercises PR #438 review fix #3 "
               "(ContextMapExecutor now opens and verifies the sequence input against the "
               "geometry's own lineage) on real data.",
    ),
    GateResult.passed(
        "cross_stage.lineage_closure", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run3/09_cross_stage.py",),
        detail="32 real assertions over the real chain (ingestion through ContextMapArtifact), "
               "0 findings -- one more than the original run-0001 pass because "
               "CrossStageInputs.auxiliary_sequence is supplied and verified: the real "
               "corridor-02-gt.txt pose SequenceArtifact that state_estimation actually merged "
               "in is confirmed to match what the trajectory's manifest names.",
    ),
    GateResult.passed(
        "cross_stage.coordinate_consistency", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run3/09_cross_stage.py",),
        detail="1,160,714 real assertions, 0 findings: one map frame from trajectory to geometry "
               "to association to entities, every boundary explicit.",
    ),
    GateResult.passed(
        "cross_stage.evidence_traceability", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run3/09_cross_stage.py",),
        detail="58,795 real assertions, 0 findings: sampled final entities/relations trace to "
               "exact source observations and geometry through the recorded path.",
    ),
    GateResult.passed(
        "cross_stage.physical_observation_identity", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run3/09_cross_stage.py",),
        detail="813 real assertions, 0 findings: repeated inference over one physical "
               "observation never became duplicated physical evidence downstream.",
    ),
    GateResult.passed(
        "runtime.provenance_identity", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run3/13_provenance_identity_audit.py",),
        detail="Systematically verified, not spot-checked (PR #438 review, run-0002 report "
               "finding 2): scripts-run3/13_provenance_identity_audit.py reopens all 8 real "
               "stage artifacts of this run plus the reused real PerceptionRunArtifact and "
               "asserts each required identity field is present -- state_estimation.estimator."
               "{backend_id,backend_version,configuration_fingerprint}; geometric_mapping."
               "{configuration_fingerprint,spatial_index_kind}; sensor_association."
               "{configuration_fingerprint,visibility_policy.fingerprint}; visual_perception."
               "{configuration_digest, per-request configuration_fingerprint over 76 real "
               "semantic executions}; semantic_fusion.{support_policy_id,"
               "support_configuration_fingerprint,fusion_policy_id}; semantic_mapping."
               "{materialization_policy_id,configuration_fingerprint,code_digest}; "
               "entity_resolution.policies (4 roles, each with a policy_id); spatial_relations."
               "policies (7 roles, each with a fingerprint) and taxonomy_version. All present; "
               "the script exits non-zero on any gap.",
    ),
    GateResult.passed(
        "runtime.resource_reporting", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/resource-profile.md",
                       "experiments/e2e-real-canonical-run-20260923/resource-profile.json"),
        detail="Real wall-time/peak-RSS/storage/file-count measured per stage (issue #181), "
               "carried over unchanged: this measurement is about runtime cost, not about the "
               "lineage-schema or code_version defects the #438 review found, so it needed no "
               "rerun. REPORT gate, no threshold: reported as measured, including the concerning "
               "finding that sensor_association peaks at 24.96 GB RSS (~64% of this 39 GB "
               "machine), a real headroom risk flagged as a limitation, not hidden.",
    ),
    GateResult.passed(
        "reproducibility.rerun_equivalence", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts-run3/12_rerun_content_equivalence.py",
                       "outputs/e2e-real/run-0003/context_map",
                       "outputs/corridor-02-repro-check/run-0006/context_map"),
        detail="Real content equivalence, not just cardinality equality (PR #438 review, "
               "run-0002 report finding 3): scripts-run3/12_rerun_content_equivalence.py opens "
               "both independent real reruns (run-0003, hand-chained; run-0006, "
               "run_plan()/resume_plan()-orchestrated with a real injected interruption), "
               "canonicalizes every entity by its real geometry (exact (x, y, z) coordinates "
               "in the shared map frame via GeometrySource.get(), never the run-specific "
               "map_id/geometry_id strings) and its label hypotheses, canonicalizes every "
               "relation by (predicate, canonical subject, canonical object), and compares the "
               "two runs' canonical multisets. Result: entity canonical digest "
               "sha256:13e3c9dadfe490559e9c64ca9ffe04c88795a084532115e1539d0fbc5ea6dec7 and "
               "relation canonical digest "
               "sha256:e7ced2618111982cc751239355edf7a30488a020820e2aa2c52f70f736039174 match "
               "exactly between the two runs, 0 mismatches either way (coordinates rounded to "
               "6 decimal places -- the geometry pipeline is deterministic over the same real "
               "inputs, so exact equality is expected; rounding only guards against "
               "representation noise across independent processes, not real disagreement). The "
               "same two canonical digests were also independently obtained comparing run-0002 "
               "vs run-0004 (the prior, code_version-stale pair), confirming zero scientific "
               "drift from any PR #438 review fix -- only lineage/provenance metadata changed. "
               "visual_perception (Qwen generation) was NOT independently re-verified for "
               "reproducibility in this pass -- greedy/temperature=0 decoding is assumed "
               "deterministic on the same hardware/dtype, but this is a documented assumption, "
               "not measured evidence.",
    ),
    GateResult.passed(
        "reproducibility.interruption_recovery", evidence_class=REAL,
        evidence_refs=("outputs/corridor-02-repro-check/run-0005", "outputs/corridor-02-repro-check/run-0006"),
        detail="Real run_plan()/resume_plan() execution over the real corridor-02 artifacts "
               "(ingestion/pose_ingestion/visual_perception reused by reference via "
               "PipelinePlan.scope(provided=...), never re-derived), regenerated under the "
               "corrected code_version: a real RuntimeError was injected at semantic_fusion "
               "after state_estimation/geometric_mapping/sensor_association genuinely "
               "completed. The interrupted run's status is 'failed', semantic_fusion's "
               "directory was never created, no .tmp-* leftovers anywhere. resume_plan() then "
               "reused the three completed real stages strictly by reference (their "
               "ArtifactRef.location still points at run-0005, not recomputed), recomputed "
               "semantic_fusion onward fresh into run-0006, and the resumed run's status is "
               "'completed'.",
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
            artifact_id="f777d44deea093fb759e054e85c0ec04",
            digest="sha256:a36bc01419c14832859012a409a6ab79657ed2fa9c88d8fe34d2b549a70a158a",
        ),
    }
    final_artifact = stage_artifacts["context_map"]

    code_version = current_code_version()
    assert code_version is not None, "contextmap.__version__ must be a real installed version"

    report = assemble_acceptance_report(
        scenario,
        report_id="acceptance-report-corridor-02-run0003",
        run_id="e2e-real-run-0003",
        code_version=code_version,
        results=RESULTS,
        stage_artifacts=stage_artifacts,
        final_artifact=final_artifact,
        limitations=(
            "outputs/e2e-real/run-0001/* and run-0002/* (and the corresponding "
            "corridor-02-repro-check/{run-0001,run-0002,run-0003,run-0004} journaled reruns), "
            "and the original acceptance-report.json/md and acceptance-report-run0002.json/md, "
            "are all superseded for acceptance purposes. run-0001 predates the #555 lineage fix "
            "(schema 0.1.0). run-0002 fixed that but was produced under a stale dev-environment "
            "code_version and its report understated runtime.provenance_identity/reproducibility."
            "rerun_equivalence to spot-checks. Only run-0003 and corridor-02-repro-check/"
            "{run-0005,run-0006} carry the corrected code_version and the systematic evidence "
            "this report cites. run-0001's PerceptionRunArtifact (visual_perception) is NOT "
            "superseded: that schema was untouched by every fix, and it continues to be reused "
            "by reference across every run in this campaign.",
            "No annotated reference set exists for corridor-02 yet: region/semantic/identity/"
            "relation quality gates are blocked, not scored (scenario 1.0.4's own documented "
            "position).",
            "visual_perception semantic interpretation (Qwen3-VL-4B-Instruct, nf4): 76/120 "
            "(63%) real successes; dominant failure is the model omitting the required "
            "`confidence` field. Documented, expected behavior for this exact model/"
            "quantization/dataset, not an infrastructure defect. Not re-run in this pass "
            "(reused from run-0001; see visual_perception.evidence_completeness).",
            "sensor_association peaks at 24.96 GB RSS (~64% of this 39 GB dev machine) -- a "
            "real algorithmic working-set cost, a real OOM risk on smaller machines, reported "
            "as diagnostic evidence for whoever tunes that stage next.",
            "semantic_mapping/entity_resolution/spatial_relations policy thresholds are reused "
            "as-is from the synthetic CI fixture's parameterization, not independently tuned "
            "for corridor-02's real physical scale.",
            "visual_perception's reproducibility (Qwen3-VL-4B generation specifically) is an "
            "assumed-deterministic, not independently re-measured, property in this report.",
            "state_estimation.accuracy, geometric_mapping.quality_report and sensor_association."
            "projection_quality (REPORT gates) were not computed in this pass; not fabricated.",
        ),
    )

    path = Path(__file__).resolve().parents[3] / (

        "outputs/e2e-real/acceptance-report-run0003.json"

    )
    write_acceptance_report(path, report)
    print("wrote", path)
    print("results:")
    for result in report.results:
        print(f"  {result.gate_id}: {result.status.value}")

    unmet = unmet_required_gates(scenario, report, kinds={GateKind.INVARIANT})
    print(f"unmet INVARIANT gates: {unmet}")

    from collections import Counter

    counts = Counter(result.status.value for result in report.results)
    print(f"status counts: {dict(counts)} (total={len(report.results)})")


if __name__ == "__main__":
    main()
