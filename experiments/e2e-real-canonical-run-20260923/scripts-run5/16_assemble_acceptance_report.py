"""Assemble the final, review-hardened AcceptanceReport for Solution 1 / corridor-02.

Fifth PR #438 review round. Supersedes every prior acceptance-report*.json/md -- each closed a
real review round, and most next rounds found a real remaining gap in the evidence. This round is
different: completing reproducibility.rerun_equivalence per the frozen scenario 1.0.4 text
("repeated canonical runs ... yield equivalent artifacts, metric reports and final map") required
an actual second, independent real execution of Visual Perception (SAM2 + DINOv2 + CLIP +
Qwen3-VL-4B) over the same 20 real corridor-02 frames -- something no prior round had done, since
every prior "rerun" reused the one real PerceptionRunArtifact and only re-executed the
deterministic downstream stages.

That real rerun found a genuine, previously undocumented result: region discovery (SAM2) is
perfectly reproducible (0/20 frame mismatches), and every downstream stage's metric report is
exactly equivalent given the same perception input (already known), but Qwen3-VL-4B's semantic
interpretation is NOT reproducible run to run under the current setup -- only 29/90 (32.2%) of
canonical claims matched exactly between the two independent executions, despite identical
greedy/temperature=0 configuration. This is a real, measured negative result, not a defect in the
evidence: reproducibility.rerun_equivalence is recorded as FAILED, with real evidence, rather than
forced to a passing status the data does not support. The mechanism (GPU floating-point
non-determinism cascading through autoregressive generation is technically plausible, but not
isolated by this experiment) is explicitly left as an open question, not asserted as fact.
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
        detail="Systematically verified over all 11 real artifacts of the run, widened across "
               "two rounds: adds context_map and both SequenceArtifacts' own provenance; loops "
               "over all 76 real semantic executions individually (not just the first) for both "
               "configuration_fingerprint and effective_configuration.{model,revision}; checks "
               "every region's and feature's persisted BackendProvenance.{model,version} (SAM2, "
               "DINOv2, CLIP, real checkpoints/revisions found); checks semantic_fusion."
               "fusion_configuration_fingerprint (not just its support-side sibling); checks "
               "PolicyRef.configuration_fingerprint for every entity_resolution/spatial_relations "
               "policy role (not just policy_id), and asserts the exact expected role set for "
               "both (not just a count -- a real, benign shape difference was found and handled: "
               "spatial_relations' 'observation' role names its rule via 'rule_id', not "
               "'policy_id'). All present; the script exits non-zero on any gap.",
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
    GateResult.failed(
        "reproducibility.rerun_equivalence", evidence_class=REAL,
        evidence_refs=(
            "experiments/e2e-real-canonical-run-20260923/scripts-run5/14_metric_report_equivalence.py",
            "experiments/e2e-real-canonical-run-20260923/scripts-run5/15_visual_perception_content_equivalence.py",
            "outputs/e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception",
            "outputs/e2e-real/visual_perception/workspace/corridor-02/run-0002/visual_perception",
        ),
        failing_capabilities=("visual_perception",),
        detail=(
            "Two independent canonical Visual Perception executions over the same 20 real "
            "corridor-02 frames produced identical region outputs (0/20 frame mismatches: same "
            "bounding boxes, region_kind, is_accepted), but semantic outputs were not "
            "equivalent: successful semantic executions changed 76->78, claims 59->60, scene "
            "contexts 17->18, and exact canonical claim agreement (hypothesis, role, category, "
            "confidence, per source_observation_id/region_id) was 29/90 (32.2%). The previously "
            "demonstrated downstream equivalence (all 8 stages from state_estimation through "
            "context_map, exact metric-report and content equivalence -- see "
            "scripts-run5/14_metric_report_equivalence.py and scripts-run4/"
            "12_rerun_content_equivalence.py) used a reused PerceptionRunArtifact and therefore "
            "does not close whole-pipeline rerun equivalence; scenario 1.0.4 requires repeated "
            "canonical runs to yield equivalent artifacts, metric reports AND final map "
            "(conjunctive), and Visual Perception is a required intermediate artifact of that "
            "same canonical run. The mechanism causing semantic nondeterminism has not yet been "
            "isolated: GPU floating-point non-determinism cascading through autoregressive "
            "generation is technically plausible, but attributing causality would need a "
            "controlled experiment (fixed seeds, torch.use_deterministic_algorithms, "
            "CUBLAS_WORKSPACE_CONFIG, pinned driver/hardware) this pass did not run. Recorded as "
            "a real, measured negative result rather than forced to a passing status the data "
            "does not support."
        ),
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
        report_id="acceptance-report-corridor-02-run0005",
        run_id="e2e-real-run-0005",
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
            "(visual_perception) is NOT superseded: it remains valid, real evidence for "
            "visual_perception.evidence_completeness; it is simply no longer the only real "
            "Visual Perception execution this report cites -- run-0002 (a second, independent "
            "execution) is the other, used specifically for reproducibility.rerun_equivalence.",
            "No annotated reference set exists for corridor-02 yet: region/semantic/identity/"
            "relation quality gates are blocked, not scored (scenario 1.0.4's own documented "
            "position).",
            "visual_perception semantic interpretation (Qwen3-VL-4B-Instruct, nf4): 76/120 "
            "(63%) real successes in run-0001, 78/120 (65%) in the independent run-0002 rerun; "
            "dominant failure in both is the model omitting the required `confidence` field. "
            "Documented, expected behavior for this exact model/quantization/dataset, not an "
            "infrastructure defect -- but see reproducibility.rerun_equivalence: the success "
            "*rate* is similar, the actual claim *content* is not reproducible.",
            "sensor_association peaks at 24.96 GB RSS (~64% of this 39 GB dev machine) -- a "
            "real algorithmic working-set cost, a real OOM risk on smaller machines.",
            "semantic_mapping/entity_resolution/spatial_relations policy thresholds are reused "
            "as-is from the synthetic CI fixture's parameterization, not independently tuned "
            "for corridor-02's real physical scale.",
            "reproducibility.rerun_equivalence is FAILED (real, measured, not assumed): a real "
            "second independent Visual Perception execution found only 32.2% exact claim "
            "agreement with the first. Region discovery (SAM2) and every downstream stage "
            "(given the same perception input) remain exactly reproducible; only Qwen3-VL-4B's "
            "semantic interpretation output is not. The mechanism is not yet isolated -- see the "
            "gate's own detail text and scripts-run5/15_visual_perception_content_equivalence.py.",
            "state_estimation.accuracy, geometric_mapping.quality_report and sensor_association."
            "projection_quality (REPORT gates) were not computed in this pass; not fabricated.",
        ),
    )

    path = Path(__file__).resolve().parents[3] / (

        "outputs/e2e-real/acceptance-report-run0005.json"

    )
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
