"""Deterministic builders for Semantic Mapping tests."""

from __future__ import annotations

from collections.abc import Sequence

from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    EvidenceContributionId,
    EvidenceStance,
    FusedEvidenceId,
    FusedHypothesisId,
    FusionSupportId,
    HypothesisEvidence,
    PointRepresentationRef,
    SemanticFusionRunId,
    SupportSignal,
    SupportSignalKind,
)
from contextmap.semantic_mapping import (
    Entity,
    EntityAttribute,
    EntityEvidenceLinks,
    EntityFeatureRef,
    EntityGeometry,
    EntityHypothesis,
    EntityHypothesisRef,
    EntityId,
    EntityLifecycle,
    EntityProvenance,
    EntitySemanticState,
    EntityTemporalState,
    EntityUncertainty,
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
from contextmap.visual_perception import BackendProvenance, ClaimId, HypothesisRole

MAP_ID = MapId("map-0001")
SEMANTIC_MAP_ID = SemanticMapId("semantic-map-0001")
FUSION_RUN_ID = SemanticFusionRunId("fusion-run-0001")
FUSED_EVIDENCE_ID = FusedEvidenceId("fused--support-000001")
FUSION_SUPPORT_ID = FusionSupportId("support-000001")
CLOCK_ID = "fixture:header"
CODE_DIGEST = "sha256:" + "ab" * 32
"""A caller-supplied code digest; the capability records it as given and never computes one."""


def timestamp(seconds: int, nanoseconds: int = 0, *, clock_id: str = CLOCK_ID) -> SourceTimestamp:
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)


def geometry_refs(
    indexes: Sequence[int], *, map_id: MapId = MAP_ID
) -> tuple[GeometryReference, ...]:
    return tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))
        for index in indexes
    )


def make_interpreter() -> BackendProvenance:
    return BackendProvenance(
        backend_id="qwen_vl",
        capability="semantic_interpreter",
        provider="alibaba",
        model="qwen3-vl",
        version="1",
    )


def make_scorer() -> BackendProvenance:
    return BackendProvenance(
        backend_id="clip_scorer",
        capability="semantic_scorer",
        provider="openai",
        model="clip-vit-l14",
        version="1",
    )


def claim_signal(value: float | None) -> SupportSignal:
    return SupportSignal(
        kind=SupportSignalKind.CLAIM_CONFIDENCE, producer=make_interpreter(), value=value
    )


def scorer_signal(value: float | None) -> SupportSignal:
    return SupportSignal(kind=SupportSignalKind.SCORER_SUPPORT, producer=make_scorer(), value=value)


def make_evidence_item(
    *,
    contribution: str = "contribution--support-000001--spatial-a",
    claim: str = "claim-0001",
    stance: EvidenceStance = EvidenceStance.SUPPORTING,
    role: HypothesisRole = HypothesisRole.PRIMARY,
    signals: tuple[SupportSignal, ...] | None = None,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        contribution_id=EvidenceContributionId(contribution),
        claim_id=ClaimId(claim),
        stance=stance,
        role=role,
        signals=(claim_signal(0.8),) if signals is None else signals,
    )


def make_hypothesis(
    hypothesis_id: str = "hypothesis-0001",
    label: str = "pallet",
    *,
    fused_evidence_id: FusedEvidenceId = FUSED_EVIDENCE_ID,
    evidence: tuple[HypothesisEvidence, ...] | None = None,
) -> EntityHypothesis:
    return EntityHypothesis(
        fused_evidence_id=fused_evidence_id,
        hypothesis_id=FusedHypothesisId(hypothesis_id),
        label=label,
        evidence=(make_evidence_item(),) if evidence is None else evidence,
    )


SUMMARY_POLICY = GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)


def default_coordinates(index: int) -> Vector3:
    """Distinct, non-flat coordinates for a geometry index."""
    return (0.1 * index, 0.2 * (index % 3), 0.05 * (index % 5))


def make_geometry(
    indexes: Sequence[int] = (0, 1, 2, 3),
    *,
    map_id: MapId = MAP_ID,
    policy: GeometrySummaryPolicy = SUMMARY_POLICY,
) -> EntityGeometry:
    source = InMemoryGeometrySource(
        map_id, {index: default_coordinates(index) for index in indexes}
    )
    return summarize_geometry(geometry_refs(indexes, map_id=map_id), source=source, policy=policy)


def make_semantic_state(
    hypotheses: tuple[EntityHypothesis, ...] | None = None,
    *,
    uncertainty: tuple[EntityUncertainty, ...] = (),
    primary: EntityHypothesisRef | None = None,
    attributes: tuple[EntityAttribute, ...] = (),
) -> EntitySemanticState:
    used = (make_hypothesis(),) if hypotheses is None else hypotheses
    return EntitySemanticState(
        hypotheses=used,
        ambiguity_state=derive_ambiguity_state(used, uncertainty),
        provenance=SemanticStateProvenance(
            mapping_rule_id="fused-evidence-semantic-state-v1",
            primary_policy_id="unambiguous-single-hypothesis-v1",
        ),
        primary_hypothesis=primary,
        attributes=attributes,
        uncertainty=uncertainty,
    )


def make_fused_evidence_ref(
    *,
    fused_evidence_id: FusedEvidenceId = FUSED_EVIDENCE_ID,
    fusion_support_id: FusionSupportId = FUSION_SUPPORT_ID,
    fusion_run_id: SemanticFusionRunId = FUSION_RUN_ID,
) -> FusedEvidenceRef:
    return FusedEvidenceRef(
        fusion_run_id=fusion_run_id,
        fusion_schema_version="0.1.0",
        fusion_artifact_digest="sha256:artifact",
        sequence_artifact_id="sequence-0001",
        fused_evidence_id=fused_evidence_id,
        fusion_support_id=fusion_support_id,
    )


def make_evidence_links(
    *,
    fused_evidence_id: FusedEvidenceId = FUSED_EVIDENCE_ID,
    fusion_support_id: FusionSupportId = FUSION_SUPPORT_ID,
    spatial: tuple[str, ...] = ("spatial--run-a--frame-0120--region-0001",),
    physical: tuple[str, ...] = ("frame-0120", "frame-0121"),
    features: tuple[EntityFeatureRef, ...] = (),
    representations: tuple[PointRepresentationRef, ...] = (),
) -> EntityEvidenceLinks:
    return EntityEvidenceLinks(
        fused_evidence=(
            make_fused_evidence_ref(
                fused_evidence_id=fused_evidence_id, fusion_support_id=fusion_support_id
            ),
        ),
        spatial_observation_ids=tuple(SpatialObservationId(item) for item in spatial),
        physical_observation_ids=tuple(SourceObservationId(item) for item in physical),
        visual_feature_refs=features,
        point_representation_refs=representations,
    )


FRAMES = (("frame-0120", 10, 2), ("frame-0121", 12, 1))
"""Two physical frames, the first interpreted by two inference runs: ``(id, seconds, results)``."""


def make_temporal_state(
    frames: Sequence[tuple[str, int, int]] = FRAMES,
    *,
    lifecycle: EntityLifecycle | None = EntityLifecycle.OBSERVED,
) -> EntityTemporalState:
    refs = tuple(
        ObservationRef(
            physical_observation_id=SourceObservationId(frame),
            acquisition_timestamp=timestamp(seconds),
            inference_result_count=results,
        )
        for frame, seconds, results in frames
    )
    return EntityTemporalState(
        first_seen=refs[0].acquisition_timestamp,
        last_seen=refs[-1].acquisition_timestamp,
        physical_observation_count=len(refs),
        inference_result_count=sum(item.inference_result_count for item in refs),
        observation_refs=refs,
        provenance=TemporalProvenance(
            rule_id="physical-observation-temporal-summary-v1", input_order_chronological=True
        ),
        lifecycle=lifecycle,
    )


def make_provenance() -> EntityProvenance:
    return EntityProvenance(
        materialization_policy_id="one-support-one-entity-v1",
        identity_policy_id="support-derived-entity-id-v1",
        configuration_fingerprint="sha256:cfg",
        code_version="test",
    )


def make_entity(
    entity_id: str = "entity--support-000001",
    *,
    semantic_map_id: SemanticMapId = SEMANTIC_MAP_ID,
    geometry: EntityGeometry | None = None,
    semantic_state: EntitySemanticState | None = None,
    evidence: EntityEvidenceLinks | None = None,
    temporal_state: EntityTemporalState | None = None,
) -> Entity:
    return Entity(
        entity_id=EntityId(entity_id),
        semantic_map_id=semantic_map_id,
        geometry=geometry or make_geometry(),
        semantic_state=semantic_state or make_semantic_state(),
        evidence=evidence or make_evidence_links(),
        temporal_state=temporal_state or make_temporal_state(),
        provenance=make_provenance(),
    )
