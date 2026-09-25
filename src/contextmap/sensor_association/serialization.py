"""JSON-friendly encoding of the Sensor Association contracts.

Records contain only JSON primitives, so a persisted observation is readable
without ROS, NumPy or a model SDK. Decoding revalidates the contract: a record
that violates an invariant is rejected instead of being trusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.sensor_association.models import (
    AssociationProvenance,
    CalibrationRef,
    DepthMetric,
    PixelCoordinate,
    PointCorrespondence,
    PoseRef,
    ProjectionSummary,
    SemanticClaimRef,
    SpatialObservation,
    SpatialObservationId,
    VisibilityDiagnostics,
    VisibilityState,
    VisualFeatureRef,
)
from contextmap.sensor_association.quality import (
    ObservationQuality,
    QualityComponent,
    ReprojectionStatistics,
    ValueSummary,
)
from contextmap.state_estimation import (
    LookupOutcome,
    PoseEstimateId,
    StateEstimationRunId,
    TrajectoryId,
)
from contextmap.visual_perception import (
    ClaimId,
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)

__all__ = [
    "decode_observation_quality",
    "decode_point_correspondence",
    "decode_spatial_observation",
    "encode_calibration_ref",
    "encode_observation_quality",
    "encode_point_correspondence",
    "encode_pose_ref",
    "encode_spatial_observation",
]


def _encode_reference(reference: GeometryReference) -> dict[str, str]:
    return {"map_id": str(reference.map_id), "geometry_id": str(reference.geometry_id)}


def _decode_reference(record: Mapping[str, Any]) -> GeometryReference:
    return GeometryReference(
        map_id=MapId(record["map_id"]), geometry_id=GeometryId(record["geometry_id"])
    )


def _encode_pixel(pixel: PixelCoordinate | None) -> list[float] | None:
    return None if pixel is None else [pixel[0], pixel[1]]


def _decode_pixel(record: list[float] | None) -> PixelCoordinate | None:
    return None if record is None else (record[0], record[1])


def encode_calibration_ref(calibration: CalibrationRef) -> dict[str, Any]:
    """Encode which calibration and camera turned geometry into pixels."""
    return {
        "calibration_identity": calibration.calibration_identity,
        "camera_calibration_id": str(calibration.camera_calibration_id),
        "camera_model_kind": calibration.camera_model_kind,
        "camera_frame": str(calibration.camera_frame),
    }


def encode_pose_ref(pose: PoseRef) -> dict[str, Any]:
    """Encode which pose placed the camera, without embedding the pose."""
    return {
        "trajectory_id": str(pose.trajectory_id),
        "state_estimation_run_id": None
        if pose.state_estimation_run_id is None
        else str(pose.state_estimation_run_id),
        "source_estimate_ids": [str(item) for item in pose.source_estimate_ids],
        "lookup_outcome": pose.lookup_outcome.value,
        "time_delta_ns": pose.time_delta_ns,
        "interpolation_fraction": pose.interpolation_fraction,
    }


def _encode_summary(summary: ValueSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "median": summary.median,
        "maximum": summary.maximum,
    }


def _decode_summary(record: Mapping[str, Any] | None) -> ValueSummary | None:
    if record is None:
        return None
    return ValueSummary(
        count=record["count"],
        minimum=record["minimum"],
        median=record["median"],
        maximum=record["maximum"],
    )


def encode_observation_quality(quality: ObservationQuality) -> dict[str, Any]:
    """Encode an observation quality; an unavailable component is an explicit ``null``."""
    reprojection = quality.reprojection
    return {
        "definitions_version": quality.definitions_version,
        "spatial_observation_id": str(quality.spatial_observation_id),
        "source_observation_id": str(quality.source_observation_id),
        "geometric_map_id": str(quality.geometric_map_id),
        "calibration_ref": encode_calibration_ref(quality.calibration_ref),
        "pose_ref": encode_pose_ref(quality.pose_ref),
        "image_transform_id": quality.image_transform_id,
        "visibility_policy_id": quality.visibility_policy_id,
        "depth_metric": quality.depth_metric.value,
        "associated_count": quality.associated_count,
        "footprint_count": quality.footprint_count,
        "mask_area_px": quality.mask_area_px,
        "support_density_per_mask_pixel": quality.support_density_per_mask_pixel,
        "support_pixel_coverage": quality.support_pixel_coverage,
        "support_depth_m": _encode_summary(quality.support_depth_m),
        "support_off_axis_angle_rad": _encode_summary(quality.support_off_axis_angle_rad),
        "border_distance_px": _encode_summary(quality.border_distance_px),
        "visible_share": quality.visible_share,
        "occluded_fraction": quality.occluded_fraction,
        "outside_valid_support_fraction": quality.outside_valid_support_fraction,
        "reprojection": None
        if reprojection is None
        else {
            "reference_id": reprojection.reference_id,
            "correspondence_count": reprojection.correspondence_count,
            "invalid_count": reprojection.invalid_count,
            "unevaluated_count": reprojection.unevaluated_count,
            "mean_px": reprojection.mean_px,
            "median_px": reprojection.median_px,
            "p95_px": reprojection.p95_px,
            "max_px": reprojection.max_px,
        },
        "unavailable": {
            component.value: reason for component, reason in quality.unavailable.items()
        },
    }


def decode_observation_quality(record: Mapping[str, Any]) -> ObservationQuality:
    """Decode an observation quality and revalidate its contract.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    calibration = record["calibration_ref"]
    pose = record["pose_ref"]
    reprojection = record["reprojection"]
    run_id = pose["state_estimation_run_id"]
    return ObservationQuality(
        definitions_version=record["definitions_version"],
        spatial_observation_id=SpatialObservationId(record["spatial_observation_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        geometric_map_id=MapId(record["geometric_map_id"]),
        calibration_ref=CalibrationRef(
            calibration_identity=calibration["calibration_identity"],
            camera_calibration_id=CalibrationReferenceId(calibration["camera_calibration_id"]),
            camera_model_kind=calibration["camera_model_kind"],
            camera_frame=FrameId(calibration["camera_frame"]),
        ),
        pose_ref=PoseRef(
            trajectory_id=TrajectoryId(pose["trajectory_id"]),
            state_estimation_run_id=None if run_id is None else StateEstimationRunId(run_id),
            source_estimate_ids=tuple(PoseEstimateId(item) for item in pose["source_estimate_ids"]),
            lookup_outcome=LookupOutcome(pose["lookup_outcome"]),
            time_delta_ns=pose["time_delta_ns"],
            interpolation_fraction=pose["interpolation_fraction"],
        ),
        image_transform_id=record["image_transform_id"],
        visibility_policy_id=record["visibility_policy_id"],
        depth_metric=DepthMetric(record["depth_metric"]),
        associated_count=record["associated_count"],
        footprint_count=record["footprint_count"],
        mask_area_px=record["mask_area_px"],
        support_density_per_mask_pixel=record["support_density_per_mask_pixel"],
        support_pixel_coverage=record["support_pixel_coverage"],
        support_depth_m=_decode_summary(record["support_depth_m"]),
        support_off_axis_angle_rad=_decode_summary(record["support_off_axis_angle_rad"]),
        border_distance_px=_decode_summary(record["border_distance_px"]),
        visible_share=record["visible_share"],
        occluded_fraction=record["occluded_fraction"],
        outside_valid_support_fraction=record["outside_valid_support_fraction"],
        reprojection=None
        if reprojection is None
        else ReprojectionStatistics(
            reference_id=reprojection["reference_id"],
            correspondence_count=reprojection["correspondence_count"],
            invalid_count=reprojection["invalid_count"],
            unevaluated_count=reprojection["unevaluated_count"],
            mean_px=reprojection["mean_px"],
            median_px=reprojection["median_px"],
            p95_px=reprojection["p95_px"],
            max_px=reprojection["max_px"],
        ),
        unavailable={
            QualityComponent(name): reason for name, reason in record["unavailable"].items()
        },
    )


def encode_point_correspondence(record: PointCorrespondence) -> dict[str, Any]:
    """Encode the 2D outcome of one map element; absent pixels stay explicit ``null``."""
    return {
        "geometry": _encode_reference(record.geometry),
        "visibility": record.visibility.value,
        "camera_depth_m": record.camera_depth_m,
        "raw_pixel": _encode_pixel(record.raw_pixel),
        "prepared_pixel": _encode_pixel(record.prepared_pixel),
        "support_depth_m": record.support_depth_m,
    }


def decode_point_correspondence(record: Mapping[str, Any]) -> PointCorrespondence:
    """Decode a point correspondence and revalidate it.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    return PointCorrespondence(
        geometry=_decode_reference(record["geometry"]),
        visibility=VisibilityState(record["visibility"]),
        camera_depth_m=record["camera_depth_m"],
        raw_pixel=_decode_pixel(record["raw_pixel"]),
        prepared_pixel=_decode_pixel(record["prepared_pixel"]),
        support_depth_m=record["support_depth_m"],
    )


def encode_spatial_observation(observation: SpatialObservation) -> dict[str, Any]:
    """Encode a spatial observation; geometry, features and claims stay references."""
    summary = observation.projection_summary
    calibration = observation.calibration_ref
    pose = observation.pose_ref
    provenance = observation.provenance
    return {
        "spatial_observation_id": str(observation.spatial_observation_id),
        "source_observation_id": str(observation.source_observation_id),
        "perception_result_id": str(observation.perception_result_id),
        "region_id": str(observation.region_id),
        "geometry_support": [
            _encode_reference(reference) for reference in observation.geometry_support
        ],
        "projection_summary": {
            "camera_model_kind": summary.camera_model_kind,
            "depth_metric": summary.depth_metric.value,
            "image_transform_id": summary.image_transform_id,
            "prepared_image_size": list(summary.prepared_image_size),
            "considered_count": summary.considered_count,
            "visible_count": summary.visible_count,
        },
        "visual_feature_refs": [
            {
                "feature_id": str(feature.feature_id),
                "embedding_space_id": feature.embedding_space_id,
                "scope": feature.scope.value,
                "region_id": None if feature.region_id is None else str(feature.region_id),
            }
            for feature in observation.visual_feature_refs
        ],
        "semantic_claim_refs": [
            {"claim_id": str(claim.claim_id)} for claim in observation.semantic_claim_refs
        ],
        "calibration_ref": encode_calibration_ref(calibration),
        "pose_ref": encode_pose_ref(pose),
        "visibility": {
            state.value: count for state, count in observation.visibility.counts.items()
        },
        "provenance": {
            "geometric_map_id": str(provenance.geometric_map_id),
            "perception_run_id": str(provenance.perception_run_id),
            "sequence_artifact_id": str(provenance.sequence_artifact_id),
            "visibility_policy_id": provenance.visibility_policy_id,
            "membership_policy_id": provenance.membership_policy_id,
            "configuration_fingerprint": provenance.configuration_fingerprint,
            "code_version": provenance.code_version,
        },
    }


def decode_spatial_observation(record: Mapping[str, Any]) -> SpatialObservation:
    """Decode a spatial observation and revalidate its contract.

    Raises:
        ValueError: If the record is malformed or violates the observation contract.
    """
    summary = record["projection_summary"]
    calibration = record["calibration_ref"]
    pose = record["pose_ref"]
    provenance = record["provenance"]
    run_id = pose["state_estimation_run_id"]
    width, height = summary["prepared_image_size"]
    return SpatialObservation(
        spatial_observation_id=SpatialObservationId(record["spatial_observation_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        perception_result_id=PerceptionResultId(record["perception_result_id"]),
        region_id=RegionId(record["region_id"]),
        geometry_support=tuple(_decode_reference(item) for item in record["geometry_support"]),
        projection_summary=ProjectionSummary(
            camera_model_kind=summary["camera_model_kind"],
            depth_metric=DepthMetric(summary["depth_metric"]),
            image_transform_id=summary["image_transform_id"],
            prepared_image_size=(width, height),
            considered_count=summary["considered_count"],
            visible_count=summary["visible_count"],
        ),
        visual_feature_refs=tuple(
            VisualFeatureRef(
                feature_id=FeatureId(item["feature_id"]),
                embedding_space_id=item["embedding_space_id"],
                scope=FeatureScope(item["scope"]),
                region_id=None if item["region_id"] is None else RegionId(item["region_id"]),
            )
            for item in record["visual_feature_refs"]
        ),
        semantic_claim_refs=tuple(
            SemanticClaimRef(claim_id=ClaimId(item["claim_id"]))
            for item in record["semantic_claim_refs"]
        ),
        calibration_ref=CalibrationRef(
            calibration_identity=calibration["calibration_identity"],
            camera_calibration_id=CalibrationReferenceId(calibration["camera_calibration_id"]),
            camera_model_kind=calibration["camera_model_kind"],
            camera_frame=FrameId(calibration["camera_frame"]),
        ),
        pose_ref=PoseRef(
            trajectory_id=TrajectoryId(pose["trajectory_id"]),
            state_estimation_run_id=None if run_id is None else StateEstimationRunId(run_id),
            source_estimate_ids=tuple(PoseEstimateId(item) for item in pose["source_estimate_ids"]),
            lookup_outcome=LookupOutcome(pose["lookup_outcome"]),
            time_delta_ns=pose["time_delta_ns"],
            interpolation_fraction=pose["interpolation_fraction"],
        ),
        visibility=VisibilityDiagnostics(
            counts={VisibilityState(state): count for state, count in record["visibility"].items()}
        ),
        provenance=AssociationProvenance(
            geometric_map_id=MapId(provenance["geometric_map_id"]),
            perception_run_id=PerceptionRunId(provenance["perception_run_id"]),
            sequence_artifact_id=SequenceArtifactId(provenance["sequence_artifact_id"]),
            visibility_policy_id=provenance["visibility_policy_id"],
            membership_policy_id=provenance["membership_policy_id"],
            configuration_fingerprint=provenance["configuration_fingerprint"],
            code_version=provenance["code_version"],
        ),
    )
