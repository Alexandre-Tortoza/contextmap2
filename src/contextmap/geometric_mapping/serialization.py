"""JSON-friendly encoding of the Geometric Mapping contracts.

Records contain only JSON primitives, so a persisted map is readable without
ROS, a point-cloud library or NumPy. A map's metadata never embeds its points:
bulk geometry lives in the artifact's storage and is reached through references.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.geometric_mapping.models import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryId,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    MapId,
    PointOrigin,
    SpatialIndexMetadata,
    TransformKind,
    TransformLineage,
    TransformStep,
)
from contextmap.geometric_mapping.motion_correction import (
    MotionCorrectionEvidence,
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
    MotionCorrectionState,
    ScanDisposition,
)
from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import (
    LookupMode,
    LookupPolicy,
    PoseEstimateId,
    StateEstimationRunId,
    TimeBounds,
    TrajectoryId,
)


def encode_bounds(bounds: Bounds3D) -> dict[str, Any]:
    """Encode bounds, keeping the frame they are expressed in."""
    return {
        "frame_id": str(bounds.frame_id),
        "minimum_m": list(bounds.minimum_m),
        "maximum_m": list(bounds.maximum_m),
    }


def decode_bounds(record: Mapping[str, Any]) -> Bounds3D:
    """Decode bounds and revalidate them.

    Raises:
        ValueError: If the record is malformed or the bounds are invalid.
    """
    minimum = record["minimum_m"]
    maximum = record["maximum_m"]
    return Bounds3D(
        frame_id=FrameId(record["frame_id"]),
        minimum_m=(minimum[0], minimum[1], minimum[2]),
        maximum_m=(maximum[0], maximum[1], maximum[2]),
    )


def encode_geometry_reference(reference: GeometryReference) -> dict[str, Any]:
    """Encode a reference to one geometry element."""
    return {"map_id": str(reference.map_id), "geometry_id": str(reference.geometry_id)}


def decode_geometry_reference(record: Mapping[str, Any]) -> GeometryReference:
    """Decode a geometry reference."""
    return GeometryReference(
        map_id=MapId(record["map_id"]), geometry_id=GeometryId(record["geometry_id"])
    )


def encode_transform_lineage(lineage: TransformLineage) -> dict[str, Any]:
    """Encode the transform chain, map side first."""
    return {
        "steps": [
            {
                "kind": step.kind.value,
                "parent_frame": str(step.parent_frame),
                "child_frame": str(step.child_frame),
                "reference": step.reference,
                "source_estimate_ids": [str(item) for item in step.source_estimate_ids],
            }
            for step in lineage.steps
        ]
    }


def decode_transform_lineage(record: Mapping[str, Any]) -> TransformLineage:
    """Decode a transform chain and revalidate that it is contiguous."""
    return TransformLineage(
        steps=tuple(
            TransformStep(
                kind=TransformKind(step["kind"]),
                parent_frame=FrameId(step["parent_frame"]),
                child_frame=FrameId(step["child_frame"]),
                reference=step["reference"],
                source_estimate_ids=tuple(
                    PoseEstimateId(item) for item in step["source_estimate_ids"]
                ),
            )
            for step in record["steps"]
        )
    )


def encode_geometry_point(point: GeometryPoint) -> dict[str, Any]:
    """Encode a point with both coordinate systems and its lineage.

    Returns:
        A record where ``coordinates_m`` is the authoritative map-frame position
        and ``source_coordinates_m`` the original sensor-local one.
    """
    return {
        "geometry_id": str(point.geometry_id),
        "map_id": str(point.map_id),
        "map_frame": str(point.map_frame),
        "coordinates_m": list(point.coordinates_m),
        "source_frame": str(point.source_frame),
        "source_coordinates_m": list(point.source_coordinates_m),
        "source_observation_id": str(point.source_observation_id),
        "source_point_index": point.source_point_index,
        "acquisition_timestamp": point.acquisition_timestamp.to_record(),
        "transform_lineage": encode_transform_lineage(point.transform_lineage),
        "provenance": {
            "origin": point.provenance.origin.value,
            "aggregation_rule": point.provenance.aggregation_rule,
            "contributing_point_count": point.provenance.contributing_point_count,
            "motion_correction": point.provenance.motion_correction.value,
        },
    }


def decode_geometry_point(record: Mapping[str, Any]) -> GeometryPoint:
    """Decode a point and revalidate its contract.

    Raises:
        ValueError: If the record is malformed or violates the point contract.
    """
    x, y, z = record["coordinates_m"]
    sx, sy, sz = record["source_coordinates_m"]
    provenance = record["provenance"]
    return GeometryPoint(
        geometry_id=GeometryId(record["geometry_id"]),
        map_id=MapId(record["map_id"]),
        map_frame=FrameId(record["map_frame"]),
        coordinates_m=(x, y, z),
        source_frame=FrameId(record["source_frame"]),
        source_coordinates_m=(sx, sy, sz),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        source_point_index=record["source_point_index"],
        acquisition_timestamp=SourceTimestamp.from_record(record["acquisition_timestamp"]),
        transform_lineage=decode_transform_lineage(record["transform_lineage"]),
        provenance=GeometryPointProvenance(
            origin=PointOrigin(provenance["origin"]),
            aggregation_rule=provenance["aggregation_rule"],
            contributing_point_count=provenance["contributing_point_count"],
            motion_correction=MotionCorrectionState(provenance["motion_correction"]),
        ),
    )


def encode_geometric_map(geometric_map: GeometricMap) -> dict[str, Any]:
    """Encode a map's metadata; the points are never embedded."""
    provenance = geometric_map.provenance
    index = geometric_map.spatial_index
    return {
        "map_id": str(geometric_map.map_id),
        "frame_id": str(geometric_map.frame_id),
        "point_count": geometric_map.point_count,
        "bounds": encode_bounds(geometric_map.bounds),
        "source_observation_ids": [str(item) for item in geometric_map.source_observation_ids],
        "time_bounds": {
            "start": geometric_map.time_bounds.start.to_record(),
            "end": geometric_map.time_bounds.end.to_record(),
        },
        "spatial_index": None
        if index is None
        else {
            "kind": index.kind,
            "parameters": dict(index.parameters),
            "is_derived": index.is_derived,
        },
        "provenance": {
            "sequence_artifact_id": str(provenance.sequence_artifact_id),
            "selection_id": provenance.selection_id,
            "trajectory_id": str(provenance.trajectory_id),
            "state_estimation_run_id": None
            if provenance.state_estimation_run_id is None
            else str(provenance.state_estimation_run_id),
            "calibration_identity": provenance.calibration_identity,
            "pose_lookup": {
                "mode": provenance.pose_lookup.mode.value,
                "max_time_delta_ns": provenance.pose_lookup.max_time_delta_ns,
                "max_interpolation_gap_ns": provenance.pose_lookup.max_interpolation_gap_ns,
            },
            "configuration_fingerprint": provenance.configuration_fingerprint,
            "code_version": provenance.code_version,
        },
    }


def decode_geometric_map(record: Mapping[str, Any]) -> GeometricMap:
    """Decode a map's metadata and revalidate its contract.

    Raises:
        ValueError: If the record is malformed or violates the map contract.
    """
    provenance = record["provenance"]
    lookup = provenance["pose_lookup"]
    index = record["spatial_index"]
    run_id = provenance["state_estimation_run_id"]
    return GeometricMap(
        map_id=MapId(record["map_id"]),
        frame_id=FrameId(record["frame_id"]),
        point_count=record["point_count"],
        bounds=decode_bounds(record["bounds"]),
        source_observation_ids=tuple(
            SourceObservationId(item) for item in record["source_observation_ids"]
        ),
        time_bounds=TimeBounds(
            start=SourceTimestamp.from_record(record["time_bounds"]["start"]),
            end=SourceTimestamp.from_record(record["time_bounds"]["end"]),
        ),
        spatial_index=None
        if index is None
        else SpatialIndexMetadata(
            kind=index["kind"],
            parameters=dict(index["parameters"]),
            is_derived=index["is_derived"],
        ),
        provenance=GeometricMapProvenance(
            sequence_artifact_id=SequenceArtifactId(provenance["sequence_artifact_id"]),
            selection_id=provenance["selection_id"],
            trajectory_id=TrajectoryId(provenance["trajectory_id"]),
            state_estimation_run_id=None if run_id is None else StateEstimationRunId(run_id),
            calibration_identity=provenance["calibration_identity"],
            pose_lookup=LookupPolicy(
                mode=LookupMode(lookup["mode"]),
                max_time_delta_ns=lookup["max_time_delta_ns"],
                max_interpolation_gap_ns=lookup["max_interpolation_gap_ns"],
            ),
            configuration_fingerprint=provenance["configuration_fingerprint"],
            code_version=provenance["code_version"],
        ),
    )


def encode_motion_correction_record(record: MotionCorrectionRecord) -> dict[str, Any]:
    """Encode a scan's declared correction state and its evidence."""
    evidence = record.evidence
    return {
        "observation_id": str(record.observation_id),
        "state": record.state.value,
        "acquisition_start": None
        if record.acquisition_start is None
        else record.acquisition_start.to_record(),
        "acquisition_end": None
        if record.acquisition_end is None
        else record.acquisition_end.to_record(),
        "per_point_timing_available": record.per_point_timing_available,
        "evidence": None
        if evidence is None
        else {
            "producer": evidence.producer,
            "trajectory_id": str(evidence.trajectory_id),
            "payload_hash": evidence.payload_hash,
            "configuration_fingerprint": evidence.configuration_fingerprint,
        },
    }


def decode_motion_correction_record(record: Mapping[str, Any]) -> MotionCorrectionRecord:
    """Decode a correction record and revalidate it.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    evidence = record["evidence"]
    return MotionCorrectionRecord(
        observation_id=SourceObservationId(record["observation_id"]),
        state=MotionCorrectionState(record["state"]),
        acquisition_start=None
        if record["acquisition_start"] is None
        else SourceTimestamp.from_record(record["acquisition_start"]),
        acquisition_end=None
        if record["acquisition_end"] is None
        else SourceTimestamp.from_record(record["acquisition_end"]),
        per_point_timing_available=record["per_point_timing_available"],
        evidence=None
        if evidence is None
        else MotionCorrectionEvidence(
            producer=evidence["producer"],
            trajectory_id=TrajectoryId(evidence["trajectory_id"]),
            payload_hash=evidence["payload_hash"],
            configuration_fingerprint=evidence["configuration_fingerprint"],
        ),
    )


def encode_motion_correction_policy(policy: MotionCorrectionPolicy) -> dict[str, Any]:
    """Encode the policy so the run manifest records how uncorrected scans were treated."""
    return {"raw": policy.raw.value, "unknown": policy.unknown.value}


def decode_motion_correction_policy(record: Mapping[str, Any]) -> MotionCorrectionPolicy:
    """Decode a motion-correction policy."""
    return MotionCorrectionPolicy(
        raw=ScanDisposition(record["raw"]), unknown=ScanDisposition(record["unknown"])
    )
