"""Real, journaled run_plan()/resume_plan() execution over the real corridor-02 artifacts
(issue #182): reuse-by-reference + a real injected interruption + real resume verification.

Unlike experiments/e2e-real-canonical-run-20260923/scripts/, which call each real executor by
hand with hand-built ArtifactRef/StageRequest in run order, this script drives the SAME real
executors through the runtime's own resolve_plan()/RunJournal/run_plan()/resume_plan() mechanism
-- the mechanism issue #182 actually asks to validate. It never touches the existing real
outputs/e2e-real/run-0001/ or run-0001-profiled/ trees; it reads the real upstream artifacts
(ingestion, pose_ingestion, visual_perception) by reference and writes its own new run under
outputs/e2e-real/repro-check/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
)
from contextmap.evaluation import canonical_real_scenario, scenario_runtime_document
from contextmap.geometric_mapping import MotionCorrectionPolicy, ScanDisposition
from contextmap.ingestion import FrameId, SequenceArtifactReader
from contextmap.runtime import (
    ArtifactRef,
    FileArtifactStore,
    ReusePolicy,
    RunJournal,
    StageExecutionError,
    StageRequest,
    read_run,
    resolve_effective_config,
    resolve_plan,
    resume_plan,
    run_plan,
)
from contextmap.runtime.executors import (
    ContextMapExecutor,
    EntityResolutionExecutor,
    GeometricMappingExecutor,
    SemanticFusionExecutor,
    SemanticMappingExecutor,
    SensorAssociationExecutor,
    SpatialRelationsExecutor,
    StateEstimationExecutor,
    inventory_digest,
)
from contextmap.semantic_fusion import BaselineAccumulationPolicy, GeometryOverlapSupportPolicy
from contextmap.semantic_mapping import EntityMaterializationPolicy, GeometrySummaryPolicy, SemanticMapId
from contextmap.sensor_association import DiagnosticTolerances, OcclusionPolicy
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunPolicies,
)
from contextmap.state_estimation import LookupPolicy
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
    InvalidSamplePolicy,
)
from contextmap.visual_perception import PerceptionRunReader

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
BAG_LOCATION = "ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4"
POSE_LOCATION = "ingest-real/sequences/corridor-02-pose/e2d832c152b1493999082d4f67210b5b"
PERCEPTION_LOCATION = "e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception"
DATASET = "corridor-02-repro-check"
UP_AXIS = AxisDirection.POSITIVE_Z


class _Failing:
    def __init__(self, inner: Any, failures: list[str]) -> None:
        self._inner, self._failures = inner, failures

    def execute(self, request: StageRequest) -> ArtifactRef:
        if self._failures:
            raise RuntimeError(self._failures.pop())
        return self._inner.execute(request)  # type: ignore[no-any-return]


def _real_hashes() -> dict[str, str]:
    bag = SequenceArtifactReader(WORKSPACE / BAG_LOCATION).manifest
    pose = SequenceArtifactReader(WORKSPACE / POSE_LOCATION).manifest
    perception = PerceptionRunReader(WORKSPACE / PERCEPTION_LOCATION).manifest
    return {
        "bag": inventory_digest(bag.file_inventory),
        "pose": inventory_digest(pose.file_inventory),
        "perception": inventory_digest(perception.file_inventory),
    }


def _executors() -> dict[str, Any]:
    estimator = ExternalPoseEstimator(
        ExternalPoseConfig(
            reference_frame=FrameId("map"), body_frame=FrameId("epson"),
            invalid_sample_policy=InvalidSamplePolicy.SKIP,
        )
    )
    return {
        "state_estimation": StateEstimationExecutor(estimator, allow_ground_truth_trajectory=True),
        "geometric_mapping": GeometricMappingExecutor(
            pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=400_000_000),
            motion_correction=MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN),
        ),
        "sensor_association": SensorAssociationExecutor(
            occlusion=OcclusionPolicy(cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02),
            tolerances=DiagnosticTolerances(
                max_pose_time_delta_ns=250_000_000, max_map_window_offset_ns=1_000_000_000,
                max_reprojection_p95_px=None, max_reprojection_invalid_rate=None,
            ),
            pose_policy=LookupPolicy.interpolated(max_interpolation_gap_ns=450_000_000),
        ),
        "semantic_fusion": SemanticFusionExecutor(
            support_policy=GeometryOverlapSupportPolicy(min_geometry_count=5, min_overlap=0.3),
            accumulation_policy=BaselineAccumulationPolicy(),
        ),
        "semantic_mapping": SemanticMappingExecutor(
            policy=EntityMaterializationPolicy(geometry=GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)),
            semantic_map_id=SemanticMapId("semantic-map-corridor-02-repro"),
            code_digest="sha256:" + "e2" * 32,
            code_version="e2e-repro-check-20260923",
        ),
        "entity_resolution": EntityResolutionExecutor(
            retrieval=CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1),
            builder=MatchEvidenceBuilder(ComparisonChannels(geometry=GeometryComparisonPolicy(
                min_shared_support_jaccard=0.5, min_bounds_iou=0.5, min_bounds_containment=0.9,
                min_conflict_gap_m=0.5, min_extent_ratio=0.3,
            ))),
            resolution=ConservativeResolutionPolicy(use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1),
            code_version="e2e-repro-check-20260923",
        ),
        "spatial_relations": SpatialRelationsExecutor(
            code_version="e2e-repro-check-20260923",
            policies=RelationsRunPolicies(
                frame_conventions=FrameConventions(map_frame="map", up_axis=UP_AXIS, forward_axis=AxisDirection.POSITIVE_X),
                candidate=CandidatePolicy(predicates=(RelationPredicate.NEXT_TO, RelationPredicate.TOUCHING), proximity_radius_m=0.6, directional_radius_m=2.0),
                geometric=GeometricPredicatePolicy(boundary_tolerance_m=0.02, next_to_max_gap_m=0.5, adjacent_penetration_m=0.05, containment_slack_m=0.05, directional_overlap_fraction=0.5),
                contact=ContactPredicatePolicy(contact_distance_m=0.05, contact_tolerance_m=0.02, min_contact_points=3, support_height_tolerance_m=0.05, support_footprint_fraction=0.5, leaning_min_tilt_deg=10.0, leaning_max_tilt_deg=80.0, tilt_tolerance_deg=2.0, leaning_min_vertical_overlap_m=0.3),
                geometry_summary=GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5),
            ),
        ),
        "context_map": ContextMapExecutor(up_axis=UP_AXIS, code_version="e2e-repro-check-20260923"),
    }


def _scope(hashes: dict[str, str]) -> tuple[Any, Any]:
    document = scenario_runtime_document(canonical_real_scenario())
    document["inputs"] = {"sequence": DATASET}
    document["pipeline"]["stages"]["pose_ingestion"] = True
    document.setdefault("policies", {})["trajectory_mode"] = "allow_ground_truth"
    path = Path("/tmp/repro-check-config.json")
    path.write_text(json.dumps(document), encoding="utf-8")
    effective = resolve_effective_config(files=[path])
    provided = {
        "ingestion": ArtifactRef(
            stage_id="ingestion", contract="SequenceArtifact",
            artifact_id="720a486de8d44c16a9d3d2ff9fa7b1a4", content_hash=hashes["bag"],
            location=BAG_LOCATION,
        ),
        "pose_ingestion": ArtifactRef(
            stage_id="pose_ingestion", contract="SequenceArtifact",
            artifact_id="e2d832c152b1493999082d4f67210b5b", content_hash=hashes["pose"],
            location=POSE_LOCATION,
        ),
        "visual_perception": ArtifactRef(
            stage_id="visual_perception", contract="PerceptionRunArtifact",
            artifact_id="vp-sam2-canonical-1-0-4", content_hash=hashes["perception"],
            location=PERCEPTION_LOCATION,
        ),
    }
    return effective, resolve_plan(effective).scope(targets=["context_map"], provided=provided)


def _outputs(record: Any) -> dict[str, ArtifactRef]:
    return {stage.stage_id: stage.output for stage in record.stages}


def main() -> None:
    hashes = _real_hashes()
    print("real content hashes:", hashes)
    effective, execution = _scope(hashes)
    print("plan order:", [stage.stage_id for stage in execution.stages])
    print("plan problems:", execution.problems)
    print("provided (reused-by-reference, no executor needed):", dict(execution.reused))

    store = FileArtifactStore(
        WORKSPACE / "e2e-real/repro-check-index",
        verify=lambda ref: ref.location is not None and (WORKSPACE / ref.location).is_dir(),
    )
    policy = ReusePolicy(store=store, code_identity="corridor-02-repro-check-1")

    failing = _executors()
    failing["semantic_fusion"] = _Failing(failing["semantic_fusion"], ["injected interruption (issue #182)"])

    journal1 = RunJournal.create(WORKSPACE, effective, execution)
    print("first run journal:", journal1.directory)
    try:
        run_plan(
            execution, failing, environ={}, module_available=lambda _n: True,
            journal=journal1, reuse=policy,
        )
        raise SystemExit("expected the injected failure to raise, it did not")
    except StageExecutionError as error:
        print("real injected interruption raised as expected:", error)

    interrupted = journal1.directory
    record1 = read_run(interrupted)
    print("interrupted run status:", record1.status.value)
    completed_dirs = sorted(p.name for p in interrupted.iterdir() if p.is_dir())
    print("stage directories present after interruption:", completed_dirs)
    tmp_leftovers = list(interrupted.rglob(".tmp-*"))
    print("tmp leftovers (must be empty):", tmp_leftovers)

    journal2 = RunJournal.create(WORKSPACE, effective, execution)
    print("resume run journal:", journal2.directory)
    record2 = resume_plan(
        interrupted, execution, _executors(), reuse=policy,
        environ={}, module_available=lambda _n: True, journal=journal2,
    )
    refs = _outputs(record2)
    for stage_id, ref in refs.items():
        print(f"  {stage_id}: location={ref.location} artifact_id={ref.artifact_id}")

    final_status = read_run(journal2.directory).status.value
    print("resumed run status:", final_status)


if __name__ == "__main__":
    main()
