"""Deterministic builders for Semantic Fusion tests."""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.point_representation import PointRepresentationId, PointRepresentationRunId
from contextmap.semantic_fusion import (
    EvidenceContribution,
    EvidenceContributionId,
    EvidenceReference,
    EvidenceStance,
    FusedEvidence,
    FusedEvidenceId,
    FusedEvidenceProvenance,
    FusedHypothesis,
    FusedHypothesisId,
    FusionSupport,
    FusionSupportId,
    FusionSupportProvenance,
    HypothesisEvidence,
    ObservationQualityRef,
    PhysicalObservationGroup,
    PointRepresentationRef,
    ScoreReference,
    SupportSignal,
    SupportSignalKind,
    UncertaintyKind,
    UncertaintyRecord,
    evidence_contribution_id_for,
)
from contextmap.sensor_association import (
    SemanticClaimRef,
    SpatialObservationId,
    VisualFeatureRef,
    spatial_observation_id_for,
)
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import TimeBounds
from contextmap.visual_perception import (
    BackendProvenance,
    ClaimId,
    FeatureId,
    FeatureScope,
    HypothesisRole,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)

MAP_ID = MapId("map-0001")
SUPPORT_ID = FusionSupportId("support-0001")
CLOCK_ID = "fixture:header"
QUALITY_DEFINITIONS_VERSION = "observation-quality-v1"


def timestamp(total_nanoseconds: int) -> SourceTimestamp:
    seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=CLOCK_ID)


def frame_timestamp(frame: str) -> SourceTimestamp:
    """One second per frame number, so ``frame-0120`` precedes ``frame-0121``."""
    return timestamp(int(frame.rsplit("-", 1)[1]) * 1_000_000_000)


def geometry_refs(
    indexes: Sequence[int], *, map_id: MapId = MAP_ID
) -> tuple[GeometryReference, ...]:
    return tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=i))
        for i in indexes
    )


def make_interpreter(backend_id: str = "qwen_vl") -> BackendProvenance:
    return BackendProvenance(
        backend_id=backend_id,
        capability="semantic_interpreter",
        provider="alibaba",
        model="qwen3-vl",
        version="1",
    )


def make_scorer(backend_id: str = "clip_scorer") -> BackendProvenance:
    return BackendProvenance(
        backend_id=backend_id,
        capability="semantic_scorer",
        provider="openai",
        model="clip-vit-l14",
        version="1",
    )


def make_bounds() -> Bounds3D:
    return Bounds3D(frame_id=FrameId("map"), minimum_m=(0.0, 0.0, 0.0), maximum_m=(2.0, 4.0, 1.0))


def make_time_bounds(first_frame: str = "frame-0120", last_frame: str = "frame-0122") -> TimeBounds:
    return TimeBounds(start=frame_timestamp(first_frame), end=frame_timestamp(last_frame))


def make_support(
    *,
    fusion_support_id: FusionSupportId = SUPPORT_ID,
    geometry_support: tuple[GeometryReference, ...] | None = None,
    spatial_observation_ids: tuple[SpatialObservationId, ...] | None = None,
    bounds: Bounds3D | None = None,
    centroid_m: tuple[float, float, float] = (1.0, 2.0, 0.5),
    time_bounds: TimeBounds | None = None,
    provenance: FusionSupportProvenance | None = None,
) -> FusionSupport:
    return FusionSupport(
        fusion_support_id=fusion_support_id,
        geometric_map_id=MAP_ID,
        geometry_support=geometry_refs(range(4)) if geometry_support is None else geometry_support,
        spatial_observation_ids=(
            (spatial_id("run-a", "frame-0120"), spatial_id("run-a", "frame-0121"))
            if spatial_observation_ids is None
            else spatial_observation_ids
        ),
        bounds=make_bounds() if bounds is None else bounds,
        centroid_m=centroid_m,
        time_bounds=make_time_bounds() if time_bounds is None else time_bounds,
        provenance=provenance
        or FusionSupportProvenance(
            support_policy_id="overlap-support-v1", configuration_fingerprint="sha256:cfg"
        ),
    )


def result_id(run: str, frame: str) -> PerceptionResultId:
    return PerceptionResultId(f"{run}--{frame}")


def spatial_id(run: str, frame: str, region: str = "region-0001") -> SpatialObservationId:
    return spatial_observation_id_for(
        perception_result_id=result_id(run, frame), region_id=RegionId(region)
    )


def contribution_id_for(
    run: str, frame: str, region: str = "region-0001"
) -> EvidenceContributionId:
    return evidence_contribution_id_for(
        fusion_support_id=SUPPORT_ID, spatial_observation_id=spatial_id(run, frame, region)
    )


def make_contribution(
    *,
    run: str = "run-a",
    frame: str = "frame-0120",
    region: str = "region-0001",
    geometry_indexes: Sequence[int] = (0, 1, 2),
    claim_ids: Sequence[str] = ("claim-0001",),
    score_refs: tuple[ScoreReference, ...] = (),
    feature_refs: tuple[VisualFeatureRef, ...] = (),
    with_quality: bool = True,
    quality: ObservationQualityRef | None = None,
) -> EvidenceContribution:
    observation_id = spatial_id(run, frame, region)
    return EvidenceContribution(
        contribution_id=contribution_id_for(run, frame, region),
        physical_observation_id=SourceObservationId(frame),
        perception_result_id=result_id(run, frame),
        perception_run_id=PerceptionRunId(run),
        spatial_observation_id=observation_id,
        region_id=RegionId(region),
        geometry_support=geometry_refs(geometry_indexes),
        claim_refs=tuple(SemanticClaimRef(claim_id=ClaimId(c)) for c in claim_ids),
        score_refs=score_refs,
        visual_feature_refs=feature_refs,
        observation_quality=(
            quality
            if quality is not None
            else ObservationQualityRef(
                spatial_observation_id=observation_id,
                definitions_version=QUALITY_DEFINITIONS_VERSION,
            )
            if with_quality
            else None
        ),
    )


def make_region_feature(
    region: str = "region-0001", feature: str = "feature-0001"
) -> VisualFeatureRef:
    return VisualFeatureRef(
        feature_id=FeatureId(feature),
        embedding_space_id="dinov3-vit-b16",
        scope=FeatureScope.REGION,
        region_id=RegionId(region),
    )


def make_group(
    *,
    frame: str = "frame-0120",
    runs: Sequence[str] = ("run-a",),
    region: str = "region-0001",
) -> PhysicalObservationGroup:
    return PhysicalObservationGroup(
        physical_observation_id=SourceObservationId(frame),
        acquisition_timestamp=frame_timestamp(frame),
        spatial_observation_ids=tuple(sorted(spatial_id(run, frame, region) for run in runs)),
        perception_result_ids=tuple(sorted(result_id(run, frame) for run in runs)),
        perception_run_ids=tuple(sorted(PerceptionRunId(run) for run in runs)),
    )


def claim_signal(value: float | None, *, backend_id: str = "qwen_vl") -> SupportSignal:
    return SupportSignal(
        kind=SupportSignalKind.CLAIM_CONFIDENCE,
        producer=make_interpreter(backend_id),
        value=value,
    )


def scorer_signal(value: float | None) -> SupportSignal:
    return SupportSignal(kind=SupportSignalKind.SCORER_SUPPORT, producer=make_scorer(), value=value)


def evidence(
    *,
    run: str = "run-a",
    frame: str = "frame-0120",
    claim_id: str = "claim-0001",
    stance: EvidenceStance = EvidenceStance.SUPPORTING,
    role: HypothesisRole = HypothesisRole.PRIMARY,
    signals: tuple[SupportSignal, ...] | None = None,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        contribution_id=contribution_id_for(run, frame),
        claim_id=ClaimId(claim_id),
        stance=stance,
        role=role,
        signals=(claim_signal(0.8),) if signals is None else signals,
    )


def make_hypothesis(
    hypothesis_id: str = "hypothesis-0001",
    label: str = "pallet",
    *,
    items: tuple[HypothesisEvidence, ...] | None = None,
) -> FusedHypothesis:
    return FusedHypothesis(
        hypothesis_id=FusedHypothesisId(hypothesis_id),
        label=label,
        evidence=(evidence(),) if items is None else items,
    )


def make_point_representation_ref(
    *, representation_id: str = "repr-0001", geometry_index: int = 0
) -> PointRepresentationRef:
    return PointRepresentationRef(
        representation_id=PointRepresentationId(representation_id),
        run_id=PointRepresentationRunId("representation-run-0001"),
        representation_space_id="sha256:space-geometric-descriptor",
        geometry_reference=geometry_refs([geometry_index])[0],
    )


def make_fused_evidence(
    *,
    contributions: tuple[EvidenceContribution, ...] | None = None,
    groups: tuple[PhysicalObservationGroup, ...] | None = None,
    hypotheses: tuple[FusedHypothesis, ...] | None = None,
    point_representation_refs: tuple[PointRepresentationRef, ...] = (),
    uncertainty: tuple[UncertaintyRecord, ...] = (),
    temporal_summary: TimeBounds | None = None,
) -> FusedEvidence:
    """One frame observed once, with one supported hypothesis, unless overridden."""
    return FusedEvidence(
        fused_evidence_id=FusedEvidenceId("fused-support-0001"),
        fusion_support_id=SUPPORT_ID,
        physical_observation_groups=(make_group(),) if groups is None else groups,
        contributions=(make_contribution(),) if contributions is None else contributions,
        hypotheses=(make_hypothesis(),) if hypotheses is None else hypotheses,
        point_representation_refs=point_representation_refs,
        uncertainty=uncertainty,
        temporal_summary=temporal_summary or make_time_bounds("frame-0120", "frame-0120"),
        provenance=FusedEvidenceProvenance(
            grouping_policy_id="physical-observation-grouping-v1",
            fusion_policy_id="baseline-evidence-accumulation-v1",
            configuration_fingerprint="sha256:cfg",
        ),
    )


def make_uncertainty(
    kind: UncertaintyKind,
    *,
    hypothesis_ids: tuple[str, ...] = (),
    references: tuple[EvidenceReference, ...] = (),
    rule_id: str = "rule-v1",
) -> UncertaintyRecord:
    return UncertaintyRecord(
        kind=kind,
        hypothesis_ids=tuple(FusedHypothesisId(h) for h in hypothesis_ids),
        evidence=references,
        rule_id=rule_id,
    )
