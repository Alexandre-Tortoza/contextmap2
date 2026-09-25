"""The Sensor Association application service: from frames to spatial observations.

:class:`SensorAssociationService` composes the capability's steps for a whole run. For each
camera frame it projects the map through the pose chain and the prepared-image chain, resolves
visibility, finds which frozen region masks contain the visible geometry, builds the
spatial observations and their quality, samples the declared dense feature channels and
diagnoses the frame. A frame whose pose the lookup policy rejects is reported, not hidden.

Dense feature maps are declared as **channels**. A native map and an enhanced map are two
distinct evidence channels of a run and are never merged; a channel is identified by the
caller, and every frame must provide the dense map of every declared channel.

The service owns no scientific rule of its own: projection, visibility, membership, quality,
sampling and diagnostics live in their modules, and the runtime supplies the concrete inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from contextmap.geometric_mapping import GeometricMap, GeometryBlockSource
from contextmap.ingestion import (
    CalibrationSet,
    ImageObservation,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.sensor_association.candidate_geometry import CandidateGeometryPolicy
from contextmap.sensor_association.dense_sampling import (
    SAMPLING_POLICY_ID,
    DenseFeatureSamples,
    InterpolationPolicy,
    sample_dense_features,
)
from contextmap.sensor_association.diagnostics import (
    DIAGNOSTICS_DEFINITIONS_VERSION,
    DiagnosticTolerances,
    FrameDiagnostics,
    TrustedCorrespondences,
    diagnose_frame,
    reprojection_statistics,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.frame_projection import FrameProjector, RejectedProjection
from contextmap.sensor_association.membership import (
    COVERAGE_DEFINITIONS_VERSION,
    MEMBERSHIP_POLICY_ID,
    FrameMembership,
    RegionMaskLoader,
    associate_regions,
    build_spatial_observations,
)
from contextmap.sensor_association.models import SpatialObservation
from contextmap.sensor_association.quality import QUALITY_DEFINITIONS_VERSION, ObservationQuality
from contextmap.sensor_association.quality_derivation import derive_observation_quality
from contextmap.sensor_association.visibility import (
    OcclusionPolicy,
    VisibilityResolution,
    resolve_visibility,
)
from contextmap.state_estimation import (
    LookupPolicy,
    StateEstimationRunId,
    TrajectoryId,
    TrajectoryLookup,
    calibration_identity,
)
from contextmap.visual_perception import (
    DenseFeatureMap,
    PerceptionResult,
    PerceptionRunId,
    PreparedImage,
)


@dataclass(frozen=True, kw_only=True)
class DenseChannel:
    """One declared dense-feature evidence channel of a run.

    Attributes:
        channel_id: Identity of the channel within the run, e.g. ``"dinov2-native"`` or
            ``"dinov2-enhanced"``. Channels are never combined.
        interpolation: How the channel's feature map is read at a pixel.
    """

    channel_id: str
    interpolation: InterpolationPolicy

    def __post_init__(self) -> None:
        """Require a channel identity.

        Raises:
            ValueError: If ``channel_id`` is empty.
        """
        if not self.channel_id:
            raise ValueError("channel_id must not be empty")


@dataclass(frozen=True, kw_only=True)
class AssociationFrameInput:
    """Everything one camera frame contributes to a run.

    Attributes:
        observation: The RGB observation.
        prepared_image: The image Visual Perception prepared from it.
        perception_result: The frozen perception result of that observation.
        dense_maps: The dense feature map of each declared channel, by ``channel_id``.
        correspondences: Trusted reference correspondences for this frame, when they exist.
    """

    observation: ImageObservation
    prepared_image: PreparedImage
    perception_result: PerceptionResult
    dense_maps: Mapping[str, DenseFeatureMap] = field(default_factory=dict)
    correspondences: TrustedCorrespondences | None = None


@dataclass(frozen=True, kw_only=True)
class SensorAssociationRequest:
    """The inputs of one association run.

    Attributes:
        sequence_artifact_id: The canonical sequence the run belongs to.
        selection_id: Deterministic identity of the sequence selection.
        geometry: The block read boundary of the persistent map.
        trajectory: The pose lookup of the selected state-estimation trajectory.
        pose_policy: Which pose lookups are acceptable for an image timestamp.
        calibration: The canonical calibration, the one the map and the trajectory used.
        candidate_policy: Which map geometry each frame evaluates, before projection.
        occlusion_policy: The visibility parameters.
        tolerances: The diagnostic tolerances.
        frames: The camera frames to associate.
        dense_channels: The declared dense-feature evidence channels.
        state_estimation_run_id: The persisted state-estimation run the trajectory came from.
        code_version: Code revision that produces the run, when known.
        mask_loader: Resolves a region's mask when a frame's perception result was
            reopened from a persisted run and no longer carries it inline (#378), for
            example that run's ``PerceptionRunReader.mask_store()``. ``None`` keeps a
            region without an inline mask and no resolvable reference as skipped.
    """

    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    geometry: GeometryBlockSource
    trajectory: TrajectoryLookup
    pose_policy: LookupPolicy
    calibration: CalibrationSet
    candidate_policy: CandidateGeometryPolicy
    occlusion_policy: OcclusionPolicy
    tolerances: DiagnosticTolerances
    frames: tuple[AssociationFrameInput, ...]
    dense_channels: tuple[DenseChannel, ...] = ()
    state_estimation_run_id: StateEstimationRunId | None = None
    code_version: str | None = None
    mask_loader: RegionMaskLoader | None = None


@dataclass(frozen=True, kw_only=True, eq=False)
class FrameAssociation:
    """The association of one camera frame.

    Attributes:
        source_observation_id: The camera frame.
        resolution: The visibility of every map point.
        membership: Which region masks contain the visible points.
        observations: One spatial observation per evaluated region.
        qualities: The quality of each observation, in the same order.
        diagnostics: The frame's diagnostic report.
        dense_samples: The dense feature sampling of each declared channel, by ``channel_id``.
    """

    source_observation_id: SourceObservationId
    resolution: VisibilityResolution
    membership: FrameMembership
    observations: tuple[SpatialObservation, ...]
    qualities: tuple[ObservationQuality, ...]
    diagnostics: FrameDiagnostics
    dense_samples: Mapping[str, DenseFeatureSamples]


@dataclass(frozen=True, kw_only=True, eq=False)
class SensorAssociationOutcome:
    """The result of an association run, ready to be persisted.

    Attributes:
        geometric_map: The map's identity, frame and provenance.
        sequence_artifact_id: The canonical sequence.
        selection_id: The sequence selection.
        trajectory_id: The trajectory that placed the camera.
        state_estimation_run_id: The state-estimation run it came from, when there is one.
        calibration_identity: Hash of the calibration used.
        perception_run_ids: The perception runs the frames' evidence came from, sorted.
        candidate_policy: The candidate rule applied to every frame.
        occlusion_policy: The visibility parameters applied.
        pose_policy: The pose lookup policy applied.
        tolerances: The diagnostic tolerances applied.
        dense_channels: The declared dense-feature channels.
        configuration_fingerprint: Hash of the effective configuration of the run.
        code_version: Code revision that produced the run, when known.
        frames: The associated frames, in the order given.
        rejected: The frames whose pose the lookup policy rejected.
    """

    geometric_map: GeometricMap
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    calibration_identity: str
    perception_run_ids: tuple[PerceptionRunId, ...]
    candidate_policy: CandidateGeometryPolicy
    occlusion_policy: OcclusionPolicy
    pose_policy: LookupPolicy
    tolerances: DiagnosticTolerances
    dense_channels: tuple[DenseChannel, ...]
    configuration_fingerprint: str
    code_version: str | None
    frames: tuple[FrameAssociation, ...]
    rejected: tuple[RejectedProjection, ...]


class SensorAssociationService:
    """Runs Sensor Association over the frames of a sequence selection."""

    def run(self, request: SensorAssociationRequest) -> SensorAssociationOutcome:
        """Associate every frame of a request.

        Args:
            request: The inputs of the run.

        Returns:
            The per-frame associations and the frames whose pose was rejected.

        Raises:
            AssociationInputError: If the map cannot be read in blocks, channel or frame
                identities repeat, a frame lacks the dense map of a declared channel, or any
                step finds the inputs incompatible.
        """
        _validate_request(request)
        identity = calibration_identity(request.calibration)
        if identity is None:
            raise AssociationInputError("the calibration set has no identity")
        fingerprint = _configuration_fingerprint(request)
        projector = FrameProjector(
            geometry=request.geometry,
            candidate_policy=request.candidate_policy,
            trajectory=request.trajectory,
            pose_policy=request.pose_policy,
            calibration=request.calibration,
            state_estimation_run_id=request.state_estimation_run_id,
        )
        frames: list[FrameAssociation] = []
        rejected: list[RejectedProjection] = []
        for frame_input in request.frames:
            projection = projector.project(frame_input.observation, frame_input.prepared_image)
            if isinstance(projection, RejectedProjection):
                rejected.append(projection)
                continue
            resolution = resolve_visibility(projection, request.occlusion_policy)
            membership = associate_regions(
                resolution, frame_input.perception_result, mask_loader=request.mask_loader
            )
            observations = build_spatial_observations(
                membership,
                configuration_fingerprint=fingerprint,
                code_version=request.code_version,
            )
            statistics = (
                None
                if frame_input.correspondences is None
                else reprojection_statistics(projection, frame_input.correspondences)
            )
            dense_samples = {
                channel.channel_id: sample_dense_features(
                    resolution,
                    frame_input.perception_result,
                    frame_input.dense_maps[channel.channel_id],
                    interpolation=channel.interpolation,
                )
                for channel in request.dense_channels
            }
            frames.append(
                FrameAssociation(
                    source_observation_id=projection.source_observation_id,
                    resolution=resolution,
                    membership=membership,
                    observations=observations,
                    qualities=derive_observation_quality(
                        membership, observations, reprojection=statistics
                    ),
                    diagnostics=diagnose_frame(
                        resolution,
                        tolerances=request.tolerances,
                        membership=membership,
                        correspondences=frame_input.correspondences,
                        dense_samples=tuple(dense_samples.values()),
                    ),
                    dense_samples=dense_samples,
                )
            )
        geometric_map = request.geometry.geometric_map
        return SensorAssociationOutcome(
            geometric_map=geometric_map,
            sequence_artifact_id=request.sequence_artifact_id,
            selection_id=request.selection_id,
            trajectory_id=request.trajectory.trajectory.trajectory_id,
            state_estimation_run_id=request.state_estimation_run_id,
            calibration_identity=identity,
            perception_run_ids=tuple(sorted({f.perception_result.run_id for f in request.frames})),
            candidate_policy=request.candidate_policy,
            occlusion_policy=request.occlusion_policy,
            pose_policy=request.pose_policy,
            tolerances=request.tolerances,
            dense_channels=request.dense_channels,
            configuration_fingerprint=fingerprint,
            code_version=request.code_version,
            frames=tuple(frames),
            rejected=tuple(rejected),
        )


def _validate_request(request: SensorAssociationRequest) -> None:
    if not isinstance(request.geometry, GeometryBlockSource):
        raise AssociationInputError(
            "sensor association reads the map in vectorized blocks: "
            f"{type(request.geometry).__name__} does not implement GeometryBlockSource"
        )
    channel_ids = [channel.channel_id for channel in request.dense_channels]
    if len(set(channel_ids)) != len(channel_ids):
        raise AssociationInputError(f"dense channel identities must be unique, got {channel_ids}")
    observation_ids = [frame.observation.observation_id for frame in request.frames]
    if len(set(observation_ids)) != len(observation_ids):
        raise AssociationInputError("a frame cannot appear twice in a run")
    for frame in request.frames:
        missing = [c for c in channel_ids if c not in frame.dense_maps]
        if missing:
            raise AssociationInputError(
                f"frame {frame.observation.observation_id!r} has no dense map for the declared "
                f"channel(s) {missing}"
            )


def _configuration_fingerprint(request: SensorAssociationRequest) -> str:
    pose = request.pose_policy
    tolerances = request.tolerances
    payload: dict[str, Any] = {
        "candidates": request.candidate_policy.to_record(),
        "occlusion": request.occlusion_policy.to_record(),
        "pose_policy": {
            "mode": pose.mode.value,
            "max_time_delta_ns": pose.max_time_delta_ns,
            "max_interpolation_gap_ns": pose.max_interpolation_gap_ns,
        },
        "tolerances": {
            "max_pose_time_delta_ns": tolerances.max_pose_time_delta_ns,
            "max_map_window_offset_ns": tolerances.max_map_window_offset_ns,
            "max_reprojection_p95_px": tolerances.max_reprojection_p95_px,
            "max_reprojection_invalid_rate": tolerances.max_reprojection_invalid_rate,
        },
        "dense_channels": [
            {"channel_id": c.channel_id, "interpolation": c.interpolation.value}
            for c in request.dense_channels
        ],
        "policies": {
            "membership": MEMBERSHIP_POLICY_ID,
            "coverage_definitions": COVERAGE_DEFINITIONS_VERSION,
            "quality_definitions": QUALITY_DEFINITIONS_VERSION,
            "diagnostics_definitions": DIAGNOSTICS_DEFINITIONS_VERSION,
            "dense_sampling": SAMPLING_POLICY_ID,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
