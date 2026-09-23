"""Minimal, real entity fixture for the runtime tests that prove a composed executor is real.

Builds one genuine :class:`~contextmap.semantic_mapping.Entity`, with a real ``EntityGeometry``
derived through :func:`~contextmap.semantic_mapping.summarize_geometry` over an in-memory
:class:`~contextmap.geometric_mapping.GeometrySource`, so ``EntityResolutionExecutor`` reads a
genuinely valid ``SemanticMappingRunArtifact`` rather than a hand-typed evidence stand-in. It
mirrors ``tests/semantic_mapping/mapping_builders.py`` and ``mapping_geometry_fake.py``, trimmed
to what a single-entity resolution run needs; it does not attempt to reproduce every field a
semantic_mapping test exercises.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    MapId,
    TransformKind,
    TransformLineage,
    TransformStep,
    geometry_id_for,
)
from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.semantic_fusion import (
    EvidenceContributionId,
    EvidenceStance,
    FusedEvidenceId,
    FusedHypothesisId,
    FusionSupportId,
    HypothesisEvidence,
    SemanticFusionRunId,
    SupportSignal,
    SupportSignalKind,
)
from contextmap.semantic_mapping import (
    Entity,
    EntityEvidenceLinks,
    EntityHypothesis,
    EntityId,
    EntityLifecycle,
    EntityProvenance,
    EntitySemanticState,
    EntityTemporalState,
    FusedEvidenceRef,
    GeometrySummaryPolicy,
    ObservationRef,
    SemanticMapId,
    SemanticStateProvenance,
    TemporalProvenance,
    derive_ambiguity_state,
    summarize_geometry,
)
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import SourceTimestamp, Vector3
from contextmap.state_estimation import (
    LookupPolicy,
    PoseEstimateId,
    StateEstimationRunId,
    TimeBounds,
    TrajectoryId,
)
from contextmap.visual_perception import BackendProvenance, ClaimId, HypothesisRole

_CLOCK = "fixture:header"
_STAMP = SourceTimestamp(seconds=1, nanoseconds=0, clock_id=_CLOCK)
SUMMARY_POLICY = GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)
DEFAULT_SEMANTIC_MAP_ID = SemanticMapId("semantic-map-0001")
DEFAULT_MAP_ID = MapId("map-0001")


def _lineage(map_frame: str) -> TransformLineage:
    return TransformLineage(
        steps=(
            TransformStep(
                kind=TransformKind.DYNAMIC_POSE,
                parent_frame=FrameId(map_frame),
                child_frame=FrameId("body"),
                reference="traj--pose-000007",
                source_estimate_ids=(PoseEstimateId("traj--pose-000007"),),
            ),
            TransformStep(
                kind=TransformKind.STATIC_CALIBRATION,
                parent_frame=FrameId("body"),
                child_frame=FrameId("lidar"),
                reference="sha256:calibration",
            ),
        )
    )


class InMemoryGeometrySource:
    """Serves the points of one map, keyed by their index in the map.

    A minimal, in-memory :class:`~contextmap.geometric_mapping.GeometrySource`: enough to build
    a real ``EntityGeometry`` through ``summarize_geometry``, never a persisted artifact.
    """

    def __init__(self, map_id: MapId, coordinates_by_index: Mapping[int, Vector3]) -> None:
        frame = "map"
        self._points = {
            geometry_id_for(map_id=map_id, index=index): GeometryPoint(
                geometry_id=geometry_id_for(map_id=map_id, index=index),
                map_id=map_id,
                map_frame=FrameId(frame),
                coordinates_m=coordinates,
                source_frame=FrameId("lidar"),
                source_coordinates_m=(0.0, 0.0, 0.0),
                source_observation_id=SourceObservationId("lidar-frame-01824"),
                source_point_index=index,
                acquisition_timestamp=_STAMP,
                transform_lineage=_lineage(frame),
                provenance=GeometryPointProvenance(),
            )
            for index, coordinates in sorted(coordinates_by_index.items())
        }
        self._map = GeometricMap(
            map_id=map_id,
            frame_id=FrameId(frame),
            point_count=len(self._points),
            bounds=Bounds3D.enclosing(coordinates_by_index.values(), frame_id=FrameId(frame)),
            source_observation_ids=(SourceObservationId("lidar-frame-01824"),),
            time_bounds=TimeBounds(start=_STAMP, end=_STAMP),
            spatial_index=None,
            provenance=GeometricMapProvenance(
                sequence_artifact_id=SequenceArtifactId("sequence-0001"),
                selection_id="full-sequence",
                trajectory_id=TrajectoryId("traj"),
                state_estimation_run_id=StateEstimationRunId("run-0001"),
                calibration_identity="sha256:calibration",
                pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=250_000_000),
                configuration_fingerprint="sha256:mapping-config",
                code_version="test",
            ),
        )

    @property
    def geometric_map(self) -> GeometricMap:
        return self._map

    def get(self, reference: GeometryReference) -> GeometryPoint:
        if reference.map_id != self._map.map_id:
            raise KeyError(reference)
        return self._points[reference.geometry_id]

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        return iter(self._points.values())

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        return (
            point
            for point in self._points.values()
            if bounds.contains(point.coordinates_m, frame_id=bounds.frame_id)
        )


def make_entity(
    entity_id: str = "entity--support-000001",
    *,
    semantic_map_id: SemanticMapId = DEFAULT_SEMANTIC_MAP_ID,
    map_id: MapId = DEFAULT_MAP_ID,
    indexes: tuple[int, ...] = (0, 1, 2, 3),
) -> Entity:
    """Build one real, self-consistent ``Entity`` over a small in-memory geometric support."""
    source = InMemoryGeometrySource(
        map_id, {index: (0.1 * index, 0.2 * (index % 3), 0.05 * (index % 5)) for index in indexes}
    )
    geometry_refs = tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))
        for index in indexes
    )
    geometry = summarize_geometry(geometry_refs, source=source, policy=SUMMARY_POLICY)
    interpreter = BackendProvenance(
        backend_id="qwen_vl",
        capability="semantic_interpreter",
        provider="alibaba",
        model="qwen3-vl",
        version="1",
    )
    evidence_item = HypothesisEvidence(
        contribution_id=EvidenceContributionId("contribution--support-000001--spatial-a"),
        claim_id=ClaimId("claim-0001"),
        stance=EvidenceStance.SUPPORTING,
        role=HypothesisRole.PRIMARY,
        signals=(
            SupportSignal(kind=SupportSignalKind.CLAIM_CONFIDENCE, producer=interpreter, value=0.8),
        ),
    )
    hypothesis = EntityHypothesis(
        fused_evidence_id=FusedEvidenceId("fused--support-000001"),
        hypothesis_id=FusedHypothesisId("hypothesis-0001"),
        label="pallet",
        evidence=(evidence_item,),
    )
    semantic_state = EntitySemanticState(
        hypotheses=(hypothesis,),
        ambiguity_state=derive_ambiguity_state((hypothesis,), ()),
        provenance=SemanticStateProvenance(
            mapping_rule_id="fused-evidence-semantic-state-v1",
            primary_policy_id="unambiguous-single-hypothesis-v1",
        ),
        primary_hypothesis=None,
        attributes=(),
        uncertainty=(),
    )
    evidence = EntityEvidenceLinks(
        fused_evidence=(
            FusedEvidenceRef(
                fusion_run_id=SemanticFusionRunId("fusion-run-0001"),
                fusion_schema_version="0.1.0",
                fusion_artifact_digest="sha256:artifact",
                sequence_artifact_id="sequence-0001",
                fused_evidence_id=FusedEvidenceId("fused--support-000001"),
                fusion_support_id=FusionSupportId("support-000001"),
            ),
        ),
        spatial_observation_ids=(SpatialObservationId("spatial--run-a--frame-0120--region-0001"),),
        physical_observation_ids=(SourceObservationId("frame-0120"),),
        visual_feature_refs=(),
        point_representation_refs=(),
    )
    temporal_state = EntityTemporalState(
        first_seen=_STAMP,
        last_seen=_STAMP,
        physical_observation_count=1,
        inference_result_count=1,
        observation_refs=(
            ObservationRef(
                physical_observation_id=SourceObservationId("frame-0120"),
                acquisition_timestamp=_STAMP,
                inference_result_count=1,
            ),
        ),
        provenance=TemporalProvenance(
            rule_id="physical-observation-temporal-summary-v1", input_order_chronological=True
        ),
        lifecycle=EntityLifecycle.OBSERVED,
    )
    provenance = EntityProvenance(
        materialization_policy_id="one-support-one-entity-v1",
        identity_policy_id="support-derived-entity-id-v1",
        configuration_fingerprint="sha256:cfg",
        code_version="test",
    )
    return Entity(
        entity_id=EntityId(entity_id),
        semantic_map_id=semantic_map_id,
        geometry=geometry,
        semantic_state=semantic_state,
        evidence=evidence,
        temporal_state=temporal_state,
        provenance=provenance,
    )
