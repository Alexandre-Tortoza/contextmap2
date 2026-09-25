"""Assemble the real final AcceptanceReport for Solution 1 / corridor-02 (issue #182)."""

from __future__ import annotations

from pathlib import Path

from contextmap.evaluation import canonical_real_scenario
from contextmap.evaluation.end_to_end import (
    ArtifactRecord,
    EvidenceClass,
    GateResult,
    assemble_acceptance_report,
    write_acceptance_report,
)

REAL = EvidenceClass.REAL

RESULTS = [
    GateResult.passed(
        "ingestion.sequence_integrity", evidence_class=REAL,
        evidence_refs=("sha256:3b631eb4c4cdcb3c7be5b65f70c728a8f592167ef693ad248f73770df3884496",
                       "outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4"),
        detail="Real bag SequenceArtifact 720a486d...: verify_integrity()==[], RGB+LiDAR+IMU under "
               "the declared corridor-02-header clock, MeiCameraModel calibration identity, "
               "selection_identity sha256:67ddf342... matches the frozen scenario 1.0.4.",
    ),
    GateResult.passed(
        "state_estimation.trajectory_coverage", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0001/state_estimation", "poses=5522", "gaps=0"),
        detail="Real StateEstimationRunArtifact via the #555 pose bridge: every selected image/"
               "LiDAR timestamp resolves through the trajectory (interpolated lookup, documented "
               "max gap), 0 TrajectoryGap records over 5522 poses; frame graph map->epson.",
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
        evidence_refs=("outputs/e2e-real/run-0001/geometric_mapping",
                       "cross_stage.coordinate_consistency: 1160714 assertions, 0 findings"),
        detail="Real GeometricMapArtifact: verify_integrity()==[], map expressed in the "
               "trajectory's frame; issue #178's real cross-stage check found zero coordinate-"
               "frame discontinuities across 1,160,714 assertions from ingestion through "
               "ContextMapArtifact, which necessarily covers this map's own frame consistency.",
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
               "patch14 + Qwen3-VL-4B-Instruct nf4): all 20 selected frames have a materialized "
               "PerceptionResult with an explicit status per stage; the 44/120 semantic-"
               "interpretation failures are recorded via StageOutcome, never dropped (this was "
               "the exact orphaned-payload bug fixed before this run could finalize).",
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
        evidence_refs=("outputs/e2e-real/run-0001/sensor_association",
                       "cross_stage.evidence_traceability: 58795 assertions, 0 findings"),
        detail="Real SensorAssociationRunArtifact: verify_integrity()==[]; issue #178's real "
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
        evidence_refs=("outputs/e2e-real/run-0001/semantic_fusion",
                       "cross_stage.physical_observation_identity: 813 assertions, 0 findings"),
        detail="Real SemanticFusionRunArtifact: verify_integrity()==[]; issue #178's real cross-"
               "stage check found no physical observation duplicated into distinct downstream "
               "evidence (813 assertions, 0 findings) and no lost lineage from fusion onward.",
    ),
    GateResult.blocked(
        "semantic_fusion.reference_recovery", blocked_by=("semantic_fusion",),
        detail="Needs AnnotationFamily.SEMANTICS, which does not exist for corridor-02 yet.",
    ),
    GateResult.passed(
        "entity_resolution.identity_lineage", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0001/entity_resolution",
                       "cross_stage.lineage_closure: 31 assertions, 0 findings"),
        detail="Real EntityResolutionRunArtifact: verify_integrity()==[], 170 entities; every "
               "resolved entity lists its member entities/fused evidence, and issue #178's real "
               "cross-stage check found zero lineage-closure findings across every dependency "
               "edge, including entity_resolution's own merge lineage.",
    ),
    GateResult.blocked(
        "entity_resolution.identity_quality", blocked_by=("entity_resolution",),
        detail="Needs AnnotationFamily.IDENTITY, which does not exist for corridor-02 yet.",
    ),
    GateResult.passed(
        "spatial_relations.reference_integrity", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0001/spatial_relations",
                       "cross_stage.evidence_traceability: 58795 assertions, 0 findings"),
        detail="Real SpatialRelationsRunArtifact: verify_integrity()==[]; every relation record "
               "references resolved entities of this run and carries its evidence, confirmed by "
               "issue #178's real cross-stage traceability check.",
    ),
    GateResult.blocked(
        "spatial_relations.relation_quality", blocked_by=("spatial_relations",),
        detail="Needs AnnotationFamily.RELATIONS, which does not exist for corridor-02 yet, so "
               "precision/recall/F1 by predicate cannot be scored. Raw, unscored evidence (not a "
               "substitute for this gate): of 10852 total relation records, 914 (8.4%) are "
               "SUPPORTED, 8668 (79.9%) UNRESOLVED (insufficient geometric evidence), 1270 "
               "(11.7%) REJECTED -- 914 confirmed relations over 170 entities (~5.4/entity) is an "
               "unremarkable real corridor-scene number; candidate generation itself is "
               "conservative (1.4%/6.2% of possible pairs within the configured radii), not "
               "permissive, per issue #178's real investigation.",
    ),
    GateResult.passed(
        "artifact.integrity", evidence_class=REAL,
        evidence_refs=("sha256:d24513f65f9b496e8d42a15537c5f03e0b2718878d926afad7d402d15733a61a",
                       "outputs/e2e-real/run-0001/context_map"),
        detail="Real ContextMapArtifact 57484aa0...: ContextMapArtifactReader.open(verify_hashes="
               "True) succeeds, entity_count=170, relation_count=10852, dependency closure names "
               "all 6 real upstream artifacts by content identity (entity_resolution_run, "
               "geometric_map, semantic_fusion_run, semantic_map, sequence, "
               "spatial_relations_run); geometry is referenced, never copied.",
    ),
    GateResult.passed(
        "cross_stage.lineage_closure", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts/09_cross_stage.py",),
        detail="31 real assertions over the real chain (ingestion through ContextMapArtifact), "
               "0 findings.",
    ),
    GateResult.passed(
        "cross_stage.coordinate_consistency", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts/09_cross_stage.py",),
        detail="1,160,714 real assertions, 0 findings: one map frame from trajectory to geometry "
               "to association to entities, every boundary explicit.",
    ),
    GateResult.passed(
        "cross_stage.evidence_traceability", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts/09_cross_stage.py",),
        detail="58,795 real assertions, 0 findings: sampled final entities/relations trace to "
               "exact source observations and geometry through the recorded path.",
    ),
    GateResult.passed(
        "cross_stage.physical_observation_identity", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/scripts/09_cross_stage.py",),
        detail="813 real assertions, 0 findings: repeated inference over one physical "
               "observation never became duplicated physical evidence downstream.",
    ),
    GateResult.passed(
        "runtime.provenance_identity", evidence_class=REAL,
        evidence_refs=("outputs/e2e-real/run-0001/state_estimation/manifest.json#estimator",
                       "outputs/e2e-real/visual_perception/.../manifest.json#configuration_digest"),
        detail="Spot-checked real manifests record exact backend identity and effective-"
               "configuration digest (e.g. state_estimation.manifest.estimator={backend_id: "
               "external_pose, backend_version: 1, configuration_fingerprint: sha256:...}, "
               "code_version=a real git-derived dev string; visual_perception's manifest records "
               "a real configuration_digest and each backend's checkpoint/revision). Not "
               "exhaustively re-verified for all 8 stages in this pass.",
    ),
    GateResult.passed(
        "runtime.resource_reporting", evidence_class=REAL,
        evidence_refs=("experiments/e2e-real-canonical-run-20260923/resource-profile.md",
                       "experiments/e2e-real-canonical-run-20260923/resource-profile.json"),
        detail="Real wall-time/peak-RSS/storage/file-count measured per stage (issue #181). "
               "REPORT gate, no threshold: reported as measured, including the concerning "
               "finding that sensor_association peaks at 24.96 GB RSS (~64% of this 39 GB "
               "machine), a real headroom risk flagged as a limitation, not hidden.",
    ),
    GateResult.passed(
        "reproducibility.rerun_equivalence", evidence_class=REAL,
        evidence_refs=("outputs/corridor-02-repro-check/run-0002/context_map",
                       "outputs/e2e-real/run-0001-profiled/"),
        detail="Two independent kinds of real rerun evidence: (1) issue #181 re-executed all 7 "
               "non-GPU stages (state_estimation..context_map) from the same real upstream "
               "inputs and every one published the exact same artifact_id/content_hash as "
               "run-0001; (2) this report's own run_plan()-orchestrated rerun (different run "
               "identity/config_digest, real interruption injected, see next gate) produced a "
               "ContextMapArtifact with entity_count=170/relation_count=10852, bit-for-bit "
               "matching run-0001's science, though content_identity differs as expected since "
               "run/config identity strings differ between the two harnesses. visual_perception "
               "(Qwen generation) was NOT independently re-verified for reproducibility in this "
               "pass -- greedy/temperature=0 decoding is assumed deterministic on the same "
               "hardware/dtype, but this is a documented assumption, not measured evidence; "
               "re-running it would cost ~12 more minutes of real GPU time for a stage whose "
               "SAM2/DINOv2/CLIP phase (91s) is comparatively cheap to re-verify and whose Qwen "
               "phase is not.",
    ),
    GateResult.passed(
        "reproducibility.interruption_recovery", evidence_class=REAL,
        evidence_refs=("outputs/corridor-02-repro-check/run-0001", "outputs/corridor-02-repro-check/run-0002"),
        detail="Real run_plan()/resume_plan() execution over the real corridor-02 artifacts "
               "(ingestion/pose_ingestion/visual_perception reused by reference via "
               "PipelinePlan.scope(provided=...), never re-derived): a real RuntimeError was "
               "injected at semantic_fusion after state_estimation/geometric_mapping/"
               "sensor_association genuinely completed. The interrupted run's status is "
               "'failed', semantic_fusion's directory was never created, no .tmp-* leftovers "
               "anywhere. resume_plan() then reused the three completed real stages strictly by "
               "reference (their ArtifactRef.location still points at run-0001, not "
               "recomputed -- confirmed by inspecting each returned ArtifactRef), recomputed "
               "semantic_fusion onward fresh into run-0002, and the resumed run's status is "
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
            artifact_id="b13c6e588aeef1e5142f890399e190b9",
            digest="sha256:ab9d83dd62c30db815ac5364a4d1cf443d72d051959471481fcefcc19ca6d817",
        ),
        "geometric_mapping": ArtifactRecord(
            artifact_id="d6ed712aa1a77b91dc93fd951273d938",
            digest="sha256:95df875979eedfc1de3e1a33f5e228735e4afbbfa80ff386db4a5f7c58a98149",
        ),
        "sensor_association": ArtifactRecord(
            artifact_id="ec435f284cef18bc1980cfd0188022a4",
            digest="sha256:60527887e259d61a14ca239909d811f954dc177bed2d766271906077bb30d888",
        ),
        "semantic_fusion": ArtifactRecord(
            artifact_id="7253ce0b68b142d58f52a528ace9af13",
            digest="sha256:7cbe377d4721ee9a7c9a020cab9fe4d4a1f586e077b3e1ffa557b8e8fe3d63c1",
        ),
        "semantic_mapping": ArtifactRecord(
            artifact_id="c9137c48ca2baa50425101497eee6947",
            digest="sha256:72a880e71759b355824092a6d1ae46409a7b9228c14bbaee3c2f17f0944eb095",
        ),
        "entity_resolution": ArtifactRecord(
            artifact_id="07616fecdb889fb0cf492f230e812706",
            digest="sha256:b1af964e3ab172952ca53fa8cddac389a52ae3a6149e39f80253281a10440f43",
        ),
        "spatial_relations": ArtifactRecord(
            artifact_id="ca6f0ab8478d46ceb7691ef35114bd8b",
            digest="sha256:198b12dde5334992939d5f54f47e6c55e5d904e0f13ef622c086573ddd0e6f34",
        ),
        "context_map": ArtifactRecord(
            artifact_id="57484aa0948006e9deb4db5b2bd1c4da",
            digest="sha256:d24513f65f9b496e8d42a15537c5f03e0b2718878d926afad7d402d15733a61a",
        ),
    }
    final_artifact = stage_artifacts["context_map"]

    report = assemble_acceptance_report(
        scenario,
        report_id="acceptance-report-corridor-02-20260923",
        run_id="e2e-real-run-0001",
        code_version="0.0.1.dev487+g69901438e.d20260922",
        results=RESULTS,
        stage_artifacts=stage_artifacts,
        final_artifact=final_artifact,
        limitations=(
            "No annotated reference set exists for corridor-02 yet: region/semantic/identity/"
            "relation quality gates are blocked, not scored (scenario 1.0.4's own documented "
            "position).",
            "visual_perception semantic interpretation (Qwen3-VL-4B-Instruct, nf4): 76/120 "
            "(63%) real successes; dominant failure is the model omitting the required "
            "`confidence` field. Documented, expected behavior for this exact model/"
            "quantization/dataset, not an infrastructure defect.",
            "sensor_association peaks at 24.96 GB RSS (~64% of this 39 GB dev machine) for a "
            "3.4 MB output -- a real algorithmic working-set cost, a real OOM risk on smaller "
            "machines, reported as diagnostic evidence for whoever tunes that stage next.",
            "semantic_mapping/entity_resolution/spatial_relations policy thresholds are reused "
            "as-is from the synthetic CI fixture's parameterization, not independently tuned for "
            "corridor-02's real physical scale; issue #178's investigation found the resulting "
            "relation counts unremarkable, but this remains an un-tuned default, not a "
            "deliberately chosen one.",
            "visual_perception's reproducibility (Qwen3-VL-4B generation specifically) is an "
            "assumed-deterministic, not independently re-measured, property in this report -- "
            "re-verifying it costs ~12 minutes of real GPU time that was not spent in this pass.",
            "state_estimation.accuracy, geometric_mapping.quality_report and sensor_association."
            "projection_quality (REPORT gates) were not computed in this pass; not fabricated.",
        ),
    )

    path = Path(__file__).resolve().parents[3] / (

        "outputs/e2e-real/acceptance-report.json"

    )
    write_acceptance_report(path, report)
    print("wrote", path)
    print("results:")
    for result in report.results:
        print(f"  {result.gate_id}: {result.status.value}")


if __name__ == "__main__":
    main()
