"""Frozen canonical Solution 1 scenario: the real sample, the CI subset and the profile.

Everything here is data that ``contextmap.evaluation.end_to_end`` defines the types
for. A change to any frozen decision (a backend, a gate, a pinned input) changes the
digest and requires a new ``SCENARIO_VERSION`` and a new committed snapshot under
``src/contextmap/evaluation/docs/scenarios/``, never an edit of an existing one. See
``src/contextmap/evaluation/docs/end-to-end.md``.
"""

from __future__ import annotations

from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.end_to_end import (
    CROSS_STAGE,
    SCENARIO_ID,
    SCENARIO_VERSION,
    AblationOnlyOption,
    AcceptanceGate,
    ComponentSelection,
    E2EScenario,
    EvidenceClass,
    ExternalInput,
    GateKind,
    ScenarioStage,
    ScenarioSubject,
    SelectedImage,
    SourceSelection,
)
from contextmap.evaluation.reference_set import ReferenceSetIdentity

_RUNTIME = "runtime"

_GT_POSE_DIGEST = "sha256:cddb6739230ded86c57412e768e071e5cd6d62a5de5ff310618032ca2e38be0a"
_CORRIDOR_MANIFEST_DIGEST = (
    "sha256:74d39984385d63de00480249d1fe9d774f639e6172c89f5ceb0eafa078cc468d"
)
_CORRIDOR_SELECTED_IMAGES: tuple[tuple[str, int, str], ...] = (
    ("camera_1_image_raw-008026", 1646000064964188570, "velodyne_points-003358"),
    ("camera_1_image_raw-008134", 1646000069486096762, "velodyne_points-003403"),
    ("camera_1_image_raw-008242", 1646000073967267474, "velodyne_points-003447"),
    ("camera_1_image_raw-008350", 1646000078489180274, "velodyne_points-003492"),
    ("camera_1_image_raw-008458", 1646000082970349730, "velodyne_points-003537"),
    ("camera_1_image_raw-008566", 1646000087492300082, "velodyne_points-003582"),
    ("camera_1_image_raw-008674", 1646000091973457570, "velodyne_points-003626"),
    ("camera_1_image_raw-008783", 1646000096495447267, "velodyne_points-003671"),
    ("camera_1_image_raw-008890", 1646000100976737607, "velodyne_points-003715"),
    ("camera_1_image_raw-008999", 1646000105498635220, "velodyne_points-003760"),
    ("camera_1_image_raw-009106", 1646000109979812590, "velodyne_points-003804"),
    ("camera_1_image_raw-009215", 1646000114501684605, "velodyne_points-003849"),
    ("camera_1_image_raw-009322", 1646000118982863445, "velodyne_points-003894"),
    ("camera_1_image_raw-009430", 1646000123464043605, "velodyne_points-003938"),
    ("camera_1_image_raw-009538", 1646000127985987661, "velodyne_points-003983"),
    ("camera_1_image_raw-009646", 1646000132467167757, "velodyne_points-004027"),
    ("camera_1_image_raw-009754", 1646000136989077333, "velodyne_points-004072"),
    ("camera_1_image_raw-009862", 1646000141470196866, "velodyne_points-004117"),
    ("camera_1_image_raw-009970", 1646000145991985178, "velodyne_points-004162"),
    ("camera_1_image_raw-010078", 1646000150473153922, "velodyne_points-004206"),
)
# Digest do manifesto do reference set sintético commitado em tests/fixtures/ci_subset/1.0.1.
_CI_REFERENCE_SET = ReferenceSetIdentity(
    reference_set_id="contextmap-ci-subset",
    version="1.0.1",
    digest="sha256:752df30cbd3a5f85b6ab0ea1606c9071e602a83769bb8ebcecc93c0ceff62742",
)


def _real_subject() -> ScenarioSubject:
    return ScenarioSubject(
        subject_id="corridor-02-sample",
        evidence_class=EvidenceClass.REAL,
        dataset_id="corridor-02",
        sequence_artifact_id="e145f73f8d894f18b96ef1f55ca308c2",
        sequence_manifest_digest=_CORRIDOR_MANIFEST_DIGEST,
        selection=SourceSelection(
            clock_id="corridor-02-header",
            selection_identity=(
                "sha256:dc641b345ffc142cbc50452bbaacef2433990478295f4720feb0f165ee4ed1c4"
            ),
            start_ns=1646000062984117000,
            end_ns=1646000152984117000,
            image_count=2160,
            lidar_scan_count=892,
            imu_count=17983,
            selected_images=tuple(
                SelectedImage(observation_id=name, timestamp_ns=stamp, nearest_lidar_id=lidar)
                for name, stamp, lidar in _CORRIDOR_SELECTED_IMAGES
            ),
        ),
        external_inputs=(
            ExternalInput(
                name="pose_source",
                role="pose_input",
                location="corridor-02-gt.txt",
                digest=_GT_POSE_DIGEST,
            ),
        ),
        reference_set=None,
        note=(
            "Real recorded 90 s window of corridor-02 (a straight run plus turns, pose gaps "
            "<= 605 ms). The sequence artifact holds no pose observations, so the trajectory is "
            "read from the dataset trajectory file as a declared pose input, which makes "
            "trajectory accuracy against that same file not applicable. No annotated reference "
            "set exists for it yet: gates that need annotations are blocked, not scored."
        ),
    )


def _ci_subject() -> ScenarioSubject:
    return ScenarioSubject(
        subject_id="ci-synthetic-subset",
        evidence_class=EvidenceClass.FAKE_CONTRACT,
        dataset_id="contextmap-ci-subset",
        sequence_artifact_id=None,
        sequence_manifest_digest=None,
        selection=SourceSelection(
            clock_id="fixture:header",
            selection_identity=None,
            start_ns=0,
            end_ns=2_000_000_000,
            image_count=3,
            lidar_scan_count=3,
            imu_count=0,
            selected_images=tuple(
                SelectedImage(
                    observation_id=f"frame-{index:04d}",
                    timestamp_ns=index * 1_000_000_000,
                    nearest_lidar_id=f"scan-{index:04d}",
                )
                for index in range(3)
            ),
        ),
        external_inputs=(
            ExternalInput(
                name="pose_source",
                role="pose_input",
                location="ExternalPoseMeasurement observations of the synthetic sequence",
                digest=None,
            ),
        ),
        reference_set=_CI_REFERENCE_SET,
        note=(
            "Synthetic formulas with no GPU, network or model. It rehearses the contracts "
            "between stages and protects regressions; it is never evidence that Solution 1 "
            "is validated."
        ),
    )


def _stages() -> tuple[ScenarioStage, ...]:
    return (
        ScenarioStage(
            stage_id="ingestion",
            capability="ingestion",
            components=(
                ComponentSelection(component_id="ingestion.source_adapter", backend="ros1_bag"),
            ),
            rationale="corridor-02 is a ROS 1 bag; the adapter only yields canonical observations.",
        ),
        ScenarioStage(
            stage_id="visual_perception",
            capability="visual_perception",
            components=(
                ComponentSelection(
                    component_id="visual_perception.region_discovery",
                    backend="sam2",
                    model="facebook/sam2.1-hiera-tiny",
                ),
                ComponentSelection(
                    component_id="visual_perception.dense_features",
                    backend="dinov2",
                    model="facebook/dinov2-base",
                ),
                ComponentSelection(
                    component_id="visual_perception.region_features",
                    backend="clip",
                    model="openai/clip-vit-large-patch14",
                ),
                ComponentSelection(
                    component_id="visual_perception.semantic_interpretation",
                    backend="qwen",
                    model="Qwen/Qwen3-VL-4B-Instruct",
                ),
            ),
            rationale=(
                "Local, open-weight backends that already ran on the sample and fit an 8 GB GPU. "
                "SAM2 segments without labels, so segmentation is not read as classification."
            ),
        ),
        ScenarioStage(
            stage_id="state_estimation",
            capability="state_estimation",
            components=(
                ComponentSelection(
                    component_id="state_estimation.estimator", backend="external_pose"
                ),
            ),
            rationale=(
                "The dataset trajectory is a declared pose input; the canonical run does not "
                "need a container. An estimator run is an ablation, not the baseline."
            ),
        ),
        ScenarioStage(
            stage_id="geometric_mapping",
            capability="geometric_mapping",
            rationale="One persistent map in the trajectory frame; geometry has no backend choice.",
        ),
        ScenarioStage(
            stage_id="sensor_association",
            capability="sensor_association",
            rationale="Calibrated projection with explicit visibility and occlusion policies.",
        ),
        ScenarioStage(
            stage_id="semantic_fusion",
            capability="semantic_fusion",
            components=(
                ComponentSelection(
                    component_id="semantic_fusion.support",
                    backend="geometry-jaccard-support-v1",
                ),
                ComponentSelection(
                    component_id="semantic_fusion.accumulation",
                    backend="baseline-evidence-accumulation-v1",
                ),
            ),
            rationale=(
                "The baseline policy is the reference every option is compared with; no decision "
                "to adopt the quality-aware policy has been taken."
            ),
        ),
        ScenarioStage(
            stage_id="semantic_mapping",
            capability="semantic_mapping",
            rationale="Materializes entities from fused evidence without cross-support merging.",
        ),
        ScenarioStage(
            stage_id="entity_resolution",
            capability="entity_resolution",
            rationale="Decides identity across supports; prefers unresolved to a silent merge.",
        ),
        ScenarioStage(
            stage_id="spatial_relations",
            capability="spatial_relations",
            rationale="Derives relations from resolved entities and never replaces their evidence.",
        ),
        ScenarioStage(
            stage_id="context_map",
            capability="artifact",
            rationale="The portable ContextMapArtifact is the product of the repository.",
        ),
    )


def _ablation_only() -> tuple[AblationOnlyOption, ...]:
    return (
        AblationOnlyOption(
            component_id="point_representation",
            option="off-vs-deterministic-vs-ptv3",
            reason=(
                "Optional evidence with a cost that no controlled ablation has justified yet; "
                "off in the canonical run."
            ),
        ),
        AblationOnlyOption(
            component_id="semantic_fusion.accumulation",
            option="quality-aware-evidence-accumulation-v1",
            reason="Not adopted until a controlled end-to-end comparison shows a benefit.",
        ),
        AblationOnlyOption(
            component_id="visual_perception.semantic_interpretation",
            option="gemini",
            reason=(
                "Sends frames to an external paid service; needs exact model identity and "
                "recorded availability and cost, and is compared, not adopted."
            ),
        ),
        AblationOnlyOption(
            component_id="visual_perception.semantic_interpretation",
            option="florence2",
            reason="Speaks in task tokens; its translation to claims is a separate experiment.",
        ),
        AblationOnlyOption(
            component_id="visual_perception.region_discovery",
            option="sam3",
            reason="Text-prompted concept segmentation with different semantics from SAM2.",
        ),
        AblationOnlyOption(
            component_id="visual_perception.dense_features",
            option="dinov3",
            reason="No real execution is recorded; compared against DINOv2, not adopted.",
        ),
        AblationOnlyOption(
            component_id="visual_perception.dense_features",
            option="feature-resolution-enhancement",
            reason="Optional DAG stage; compared on the same DINO artifact, not adopted.",
        ),
        AblationOnlyOption(
            component_id="visual_perception.region_features",
            option="alphaclip",
            reason="No verified checkpoint source; compared against CLIP if one exists.",
        ),
        AblationOnlyOption(
            component_id="state_estimation.estimator",
            option="fast_lio",
            reason="Compared against the pose input as an estimator, not the baseline.",
        ),
        AblationOnlyOption(
            component_id="entity_resolution",
            option="geometry-plus-semantic-and-appearance-evidence",
            reason="The baseline resolves on geometry; extra evidence is an ablation.",
        ),
    )


def _gate(
    gate_id: str,
    capability: str,
    kind: GateKind,
    requirement: str,
    evidence: str,
    *,
    metrics: tuple[str, ...] = (),
    needs: tuple[AnnotationFamily, ...] = (),
) -> AcceptanceGate:
    return AcceptanceGate(
        gate_id=gate_id,
        capability=capability,
        kind=kind,
        requirement=requirement,
        evidence=evidence,
        metrics=metrics,
        needs_reference_annotations=needs,
    )


def _gates() -> tuple[AcceptanceGate, ...]:
    invariant, report = GateKind.INVARIANT, GateKind.REPORT
    return (
        _gate(
            "ingestion.sequence_integrity",
            "ingestion",
            invariant,
            "The selected window of the SequenceArtifact validates with zero integrity "
            "violations, holds RGB and LiDAR under the one declared clock, carries a calibration "
            "identity, and its selection identity equals the scenario's.",
            "Sequence artifact reader verification and the ingestion integrity report.",
            metrics=("ingestion.integrity.violations", "ingestion.modality.coverage"),
        ),
        _gate(
            "state_estimation.trajectory_coverage",
            "state_estimation",
            invariant,
            "Every selected image and LiDAR timestamp resolves to a pose, exact or interpolated "
            "within the declared lookup policy, or is rejected explicitly; the frame graph "
            "connects every sensor frame to the map frame.",
            "State estimation run artifact, lookup outcomes and the pose-gap report.",
            metrics=("state.gap.ratio",),
        ),
        _gate(
            "state_estimation.accuracy",
            "state_estimation",
            report,
            "ATE and RPE are reported against an independent trusted reference; when the "
            "trajectory is the declared pose input itself they are reported as not applicable, "
            "never as zero.",
            "State estimation evaluation report with the comparison protocol.",
            metrics=("state.ate.rmse", "state.rpe.translation.rmse"),
        ),
        _gate(
            "geometric_mapping.map_frame_consistency",
            "geometric_mapping",
            invariant,
            "The map is expressed in the trajectory's frame, every source point of the selected "
            "scans has an auditable transform trace, geometry references are stable, and the "
            "map survives a round trip through its artifact unchanged.",
            "Geometric map artifact, transform traces and the round-trip report.",
        ),
        _gate(
            "geometric_mapping.quality_report",
            "geometric_mapping",
            report,
            "Density, range, bounds and scan-to-scan overlap of the map are reported with their "
            "sampling protocol.",
            "Geometric mapping evaluation report.",
            metrics=("geometry.scan_overlap.plane_distance.median",),
        ),
        _gate(
            "visual_perception.evidence_completeness",
            "visual_perception",
            invariant,
            "Every selected frame has a perception result with an explicit status per stage; "
            "failures are recorded rather than dropped, and each claim carries its backend, "
            "model, prompt policy and source region.",
            "Perception run artifact and its multi-run reader.",
        ),
        _gate(
            "visual_perception.region_quality",
            "visual_perception",
            report,
            "Region overlap, recall and duplicate rate are reported over the annotated frames.",
            "Region discovery evaluation against the reference set.",
            metrics=("region.iou.mean", "region.recall.mean", "region.duplicate_rate.mean"),
            needs=(AnnotationFamily.REGIONS,),
        ),
        _gate(
            "visual_perception.semantic_quality",
            "visual_perception",
            report,
            "Acceptable, unsupported and ambiguity-preserving claim rates are reported; an "
            "unannotated open-vocabulary concept is never counted as wrong.",
            "Semantic interpretation evaluation against the reference set.",
            metrics=(
                "semantic.acceptable_claim_rate",
                "semantic.unsupported_claim_rate",
                "semantic.ambiguity_preservation_rate",
            ),
            needs=(AnnotationFamily.SEMANTICS,),
        ),
        _gate(
            "sensor_association.projection_validity",
            "sensor_association",
            invariant,
            "Every SpatialObservation references geometry that exists in the run's map, the "
            "visible, occluded, outside and invalid counts add up, and a frame without a valid "
            "pose is rejected explicitly.",
            "Sensor association run artifact and its calibration and timing diagnostics.",
        ),
        _gate(
            "sensor_association.projection_quality",
            "sensor_association",
            report,
            "Visible support, reprojection error and feature anchoring are reported by "
            "stratum with an explicit unavailable stratum.",
            "Sensor association evaluation report.",
            metrics=(
                "association.visible_support.ratio",
                "association.reprojection_error.median",
                "association.feature_anchoring.rate",
            ),
        ),
        _gate(
            "semantic_fusion.evidence_preservation",
            "semantic_fusion",
            invariant,
            "Fusion consumes only the run's spatial observations, groups repeated inference "
            "over one physical observation instead of duplicating it, keeps alternatives and "
            "conflicts as uncertainty, and creates no object identity.",
            "Semantic fusion run artifact and its lineage.",
        ),
        _gate(
            "semantic_fusion.reference_recovery",
            "semantic_fusion",
            report,
            "Reference recovery, ambiguity retention and multi-view consistency are reported "
            "over the annotated supports.",
            "Semantic fusion evaluation against the reference set.",
            metrics=(
                "fusion.reference_recovery.rate",
                "fusion.ambiguity_retention.rate",
                "fusion.view_consistency.rate",
            ),
            needs=(AnnotationFamily.SEMANTICS,),
        ),
        _gate(
            "entity_resolution.identity_lineage",
            "entity_resolution",
            invariant,
            "Every resolved entity lists its member entities and their fused evidence, every "
            "merge, split or keep decision records the evidence it used, and an undecided case "
            "stays unresolved.",
            "Entity resolution run artifact and the entity materialization artifact.",
        ),
        _gate(
            "entity_resolution.identity_quality",
            "entity_resolution",
            report,
            "False merges, duplicates and semantic accuracy are reported separately, so a "
            "resolution error is never attributed to fusion or the reverse.",
            "Entity resolution evaluation against the identity annotations.",
            metrics=(
                "entity.false_merge.rate",
                "entity.duplicate.rate",
                "entity.semantic_accuracy.rate",
            ),
            needs=(AnnotationFamily.IDENTITY,),
        ),
        _gate(
            "spatial_relations.reference_integrity",
            "spatial_relations",
            invariant,
            "Every relation references resolved entities of the run and carries its evidence; "
            "a relation never replaces the evidence that supports it.",
            "Spatial relations run artifact.",
        ),
        _gate(
            "spatial_relations.relation_quality",
            "spatial_relations",
            report,
            "Relation precision and recall are reported by predicate, with explicit negatives "
            "kept apart from missing annotations.",
            "Spatial relations evaluation against the relation annotations.",
            metrics=("relations.f1", "relations.negative_violation.rate"),
            needs=(AnnotationFamily.RELATIONS,),
        ),
        _gate(
            "artifact.integrity",
            "artifact",
            invariant,
            "The ContextMapArtifact reads back on the base install with verified hashes and "
            "without mutation, its dependency closure is complete, and it references geometry "
            "rather than copying it.",
            "Artifact reader verification and the round-trip report.",
            metrics=("artifact.integrity.violations", "artifact.round_trip.mismatches"),
        ),
        _gate(
            "cross_stage.lineage_closure",
            CROSS_STAGE,
            invariant,
            "Every stage artifact names the exact upstream artifacts it consumed and each "
            "equals the artifact actually read; the dependency closure of the final artifact "
            "has no unnamed input.",
            "Cross-stage lineage check over the persisted artifacts.",
        ),
        _gate(
            "cross_stage.coordinate_consistency",
            CROSS_STAGE,
            invariant,
            "One map frame runs from trajectory to geometry to association to entities; every "
            "boundary states its frame and units and no stage converts implicitly.",
            "Cross-stage coordinate check over the persisted artifacts.",
        ),
        _gate(
            "cross_stage.evidence_traceability",
            CROSS_STAGE,
            invariant,
            "Sampled final entities and relations trace to the exact source observations and "
            "geometry through the recorded path, and no stage repaired an invalid upstream "
            "artifact.",
            "Cross-stage trace of sampled final facts back to source observations.",
        ),
        _gate(
            "cross_stage.physical_observation_identity",
            CROSS_STAGE,
            invariant,
            "Repeated inference over one physical observation never becomes duplicated "
            "physical evidence downstream.",
            "Cross-stage check of physical observation identity across fusion and entities.",
        ),
        _gate(
            "runtime.provenance_identity",
            _RUNTIME,
            invariant,
            "Every stage artifact records the exact backend, model revision or checkpoint hash "
            "and the effective configuration digest; an external service records its "
            "availability and cost metadata.",
            "Run manifest and stage provenance.",
        ),
        _gate(
            "runtime.resource_reporting",
            _RUNTIME,
            report,
            "Wall time, throughput, peak CPU memory, peak GPU memory with device and precision, "
            "storage and counts are recorded per stage with the warm or cold, reuse and "
            "selection-size conditions; failures and out-of-memory stay explicit.",
            "Resource profile of the canonical run.",
            metrics=(
                "runtime.wall_time",
                "runtime.throughput",
                "runtime.peak_memory",
                "runtime.storage_size",
                "runtime.failure_rate",
            ),
        ),
        _gate(
            "reproducibility.rerun_equivalence",
            _RUNTIME,
            invariant,
            "Repeated canonical runs select the same stages and configuration and yield "
            "equivalent artifacts, metric reports and final map within the documented "
            "tolerance of each stage.",
            "Comparison of repeated canonical runs.",
        ),
        _gate(
            "reproducibility.interruption_recovery",
            _RUNTIME,
            invariant,
            "A run interrupted at a representative stage resumes by reusing immutable "
            "artifacts, never promotes an incomplete artifact and preserves lineage.",
            "Fault-injection run and its resume record.",
        ),
    )


def canonical_real_scenario() -> E2EScenario:
    """Return the frozen scenario over the real corridor-02 sample."""
    return E2EScenario(
        scenario_id=SCENARIO_ID,
        version=SCENARIO_VERSION,
        subject=_real_subject(),
        stages=_stages(),
        ablation_only=_ablation_only(),
        gates=_gates(),
    )


def canonical_ci_scenario() -> E2EScenario:
    """Return the same profile and acceptance matrix over the synthetic CI subset."""
    return E2EScenario(
        scenario_id=SCENARIO_ID,
        version=SCENARIO_VERSION,
        subject=_ci_subject(),
        stages=_stages(),
        ablation_only=_ablation_only(),
        gates=_gates(),
    )
