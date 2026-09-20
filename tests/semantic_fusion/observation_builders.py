"""Builders for the upstream Sensor Association and Visual Perception inputs of fusion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from fusion_builders import MAP_ID, frame_timestamp, geometry_refs, make_interpreter, result_id

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.sensor_association import (
    AssociationProvenance,
    CalibrationRef,
    DepthMetric,
    PoseRef,
    ProjectionSummary,
    SemanticClaimRef,
    SpatialObservation,
    VisibilityDiagnostics,
    VisibilityState,
    spatial_observation_id_for,
)
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import LookupOutcome, PoseEstimateId, TrajectoryId
from contextmap.visual_perception import (
    BackendProvenance,
    ClaimId,
    PerceptionResultId,
    PerceptionRun,
    PerceptionRunId,
    RegionId,
)

SEQUENCE_ID = "sequence-0001"


def make_perception_run(
    run: str = "run-a",
    *,
    sequence: str = SEQUENCE_ID,
    interpreter: BackendProvenance | None = None,
) -> PerceptionRun:
    return PerceptionRun(
        run_id=PerceptionRunId(run),
        run_index=0,
        sequence_artifact_id=sequence,
        selection_id="selection-0001",
        enabled_capabilities=frozenset({"semantic_interpreter"}),
        backend_provenance={"semantic_interpreter": interpreter or make_interpreter()},
    )


def make_spatial_observation(
    run: str = "run-a",
    frame: str = "frame-0120",
    region: str = "region-0001",
    *,
    sequence: str = SEQUENCE_ID,
    map_id: MapId = MAP_ID,
    perception_result_id: PerceptionResultId | None = None,
    source_observation_id: str | None = None,
    support: Sequence[int] = (0, 1, 2),
) -> SpatialObservation:
    """The observation of one region of one frame, as produced under one perception run."""
    result = perception_result_id or result_id(run, frame)
    return SpatialObservation(
        spatial_observation_id=spatial_observation_id_for(
            perception_result_id=result, region_id=RegionId(region)
        ),
        source_observation_id=SourceObservationId(source_observation_id or frame),
        perception_result_id=result,
        region_id=RegionId(region),
        geometry_support=geometry_refs(support, map_id=map_id),
        projection_summary=ProjectionSummary(
            camera_model_kind="mei",
            depth_metric=DepthMetric.RAY_RANGE,
            image_transform_id="sha256:raw-to-prepared",
            prepared_image_size=(640, 480),
            considered_count=50,
            visible_count=20,
        ),
        visual_feature_refs=(),
        semantic_claim_refs=(SemanticClaimRef(claim_id=ClaimId("claim-0001")),),
        calibration_ref=CalibrationRef(
            calibration_identity="sha256:calibration",
            camera_calibration_id=CalibrationReferenceId("camera0"),
            camera_model_kind="mei",
            camera_frame=FrameId("rgb_camera"),
        ),
        pose_ref=PoseRef(
            trajectory_id=TrajectoryId("traj"),
            state_estimation_run_id=None,
            source_estimate_ids=(PoseEstimateId("traj--pose-000007"),),
            lookup_outcome=LookupOutcome.EXACT,
            time_delta_ns=0,
            interpolation_fraction=None,
        ),
        visibility=VisibilityDiagnostics(counts={VisibilityState.ASSOCIATED: len(support)}),
        provenance=AssociationProvenance(
            geometric_map_id=map_id,
            perception_run_id=PerceptionRunId(run),
            sequence_artifact_id=SequenceArtifactId(sequence),
            visibility_policy_id="conservative-depth-support-v1",
            membership_policy_id="mask-membership-v1",
        ),
    )


def timestamps(frames: Iterable[str]) -> dict[SourceObservationId, SourceTimestamp]:
    return {SourceObservationId(frame): frame_timestamp(frame) for frame in frames}
