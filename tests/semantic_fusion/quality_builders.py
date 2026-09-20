"""Builders for the measurable observation quality that Sensor Association produces."""

from __future__ import annotations

from fusion_builders import MAP_ID

from contextmap.ingestion import CalibrationReferenceId, FrameId, SourceObservationId
from contextmap.sensor_association import (
    CalibrationRef,
    DepthMetric,
    ObservationQuality,
    PoseRef,
    QualityComponent,
    ReprojectionStatistics,
    SpatialObservationId,
    ValueSummary,
)
from contextmap.state_estimation import LookupOutcome, PoseEstimateId, TrajectoryId

DEFINITIONS_VERSION = "observation-quality-v1"


def _summary(value: float | None) -> ValueSummary | None:
    return (
        None if value is None else ValueSummary(count=5, minimum=value, median=value, maximum=value)
    )


def make_quality(
    observation_id: SpatialObservationId,
    *,
    frame: str = "frame-0120",
    depth_median_m: float | None = 3.0,
    off_axis_median_rad: float | None = 0.2,
    border_median_px: float | None = 60.0,
    visible_share: float | None = 0.9,
    occluded_fraction: float | None = 0.05,
    outside_valid_fraction: float | None = 0.0,
    reprojection_median_px: float | None = None,
    associated_count: int = 200,
    temporal_offset_ns: int = 0,
    definitions_version: str = DEFINITIONS_VERSION,
) -> ObservationQuality:
    reprojection = (
        None
        if reprojection_median_px is None
        else ReprojectionStatistics(
            reference_id="trusted-correspondences-0001",
            correspondence_count=10,
            invalid_count=0,
            mean_px=reprojection_median_px,
            median_px=reprojection_median_px,
            p95_px=reprojection_median_px,
            max_px=reprojection_median_px,
        )
    )
    values = {
        QualityComponent.SUPPORT_DEPTH: depth_median_m,
        QualityComponent.SUPPORT_OFF_AXIS_ANGLE: off_axis_median_rad,
        QualityComponent.BORDER_DISTANCE: border_median_px,
        QualityComponent.VISIBLE_SHARE: visible_share,
        QualityComponent.OCCLUDED_FRACTION: occluded_fraction,
        QualityComponent.OUTSIDE_VALID_SUPPORT_FRACTION: outside_valid_fraction,
        QualityComponent.REPROJECTION: reprojection,
    }
    return ObservationQuality(
        definitions_version=definitions_version,
        spatial_observation_id=observation_id,
        source_observation_id=SourceObservationId(frame),
        geometric_map_id=MAP_ID,
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
            time_delta_ns=temporal_offset_ns,
            interpolation_fraction=None,
        ),
        image_transform_id="sha256:raw-to-prepared",
        visibility_policy_id="conservative-depth-support-v1",
        depth_metric=DepthMetric.RAY_RANGE,
        associated_count=associated_count,
        footprint_count=associated_count + 20,
        mask_area_px=4_000,
        support_density_per_mask_pixel=associated_count / 4_000,
        support_pixel_coverage=0.5,
        support_depth_m=_summary(depth_median_m),
        support_off_axis_angle_rad=_summary(off_axis_median_rad),
        border_distance_px=_summary(border_median_px),
        visible_share=visible_share,
        occluded_fraction=occluded_fraction,
        outside_valid_support_fraction=outside_valid_fraction,
        reprojection=reprojection,
        unavailable={
            component: "fixture: not measurable"
            for component, value in values.items()
            if value is None
        },
    )
