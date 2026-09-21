"""Deterministic builders of Entity Resolution contracts for the tests."""

from __future__ import annotations

from typing import Any

from contextmap.entity_resolution import (
    AppearanceEvidence,
    AppearanceMeasurement,
    DecisionProvenance,
    EntityMatchEvidence,
    EvidenceStatus,
    FeatureContribution,
    Finding,
    GateResult,
    GeometryEvidence,
    GeometryMeasurement,
    LabelComparison,
    LabelRelation,
    MatchChannel,
    MatchEvidenceProvenance,
    PointRepresentationEvidence,
    PolicyRef,
    PolicyStage,
    RepresentationMeasurement,
    RepresentationRef,
    ResolutionDecision,
    ResolutionOutcome,
    SemanticEvidence,
    SemanticMeasurement,
    TemporalEvidence,
    TemporalMeasurement,
    TriggeredRule,
    Unavailability,
    UnavailableReason,
    comparison_id_for,
    decision_id_for,
)
from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import PointRepresentationId, PointRepresentationRunId
from contextmap.semantic_mapping import (
    AmbiguityState,
    EntityFeatureRef,
    EntityId,
    EntityReference,
    SemanticMapId,
)
from contextmap.visual_perception import (
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)

SEMANTIC_MAP = SemanticMapId("semantic-map-0001")
MAP_ID = MapId("map-0001")
EMBEDDING_SPACE = "sha256:embedding-space-a"
REPRESENTATION_SPACE = "sha256:representation-space-a"


def ref(entity_id: str, semantic_map_id: SemanticMapId = SEMANTIC_MAP) -> EntityReference:
    return EntityReference(semantic_map_id=semantic_map_id, entity_id=EntityId(entity_id))


REF_A = ref("entity--support-000001")
REF_B = ref("entity--support-000002")


def channel_policy(policy_id: str = "test-channel-policy-v1") -> PolicyRef:
    return PolicyRef(policy_id=policy_id, configuration_fingerprint="sha256:cfg")


def unavailable(
    reason: UnavailableReason = UnavailableReason.MISSING_EVIDENCE,
    detail: str = "nothing to compare",
) -> Unavailability:
    return Unavailability(reason=reason, detail=detail)


def finding(
    status: EvidenceStatus = EvidenceStatus.SUPPORTING,
    rule_id: str = "rule-1",
    *,
    metric: str | None = None,
    observed: float | None = None,
    threshold: float | None = None,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        status=status,
        detail=f"{rule_id} is {status.value}",
        metric=metric,
        observed=observed,
        threshold=threshold,
    )


def geometry_measurement(**overrides: Any) -> GeometryMeasurement:
    values: dict[str, Any] = {
        "geometric_map_id": MAP_ID,
        "map_frame": "map",
        "centroid_distance_m": 0.4,
        "bounds_gap_m": 0.0,
        "bounds_iou": 0.5,
        "bounds_overlap_fraction_a": 0.75,
        "bounds_overlap_fraction_b": 0.6,
        "support_count_a": 4,
        "support_count_b": 4,
        "shared_support_count": 2,
        "support_jaccard": 2 / 6,
        "extent_ratio": 0.8,
        "support_distance": None,
        "orientation_angle_rad": None,
        "caveats": (),
    }
    values.update(overrides)
    return GeometryMeasurement(**values)


def geometry_evidence(
    status: EvidenceStatus = EvidenceStatus.SUPPORTING,
) -> GeometryEvidence:
    return GeometryEvidence(
        policy=channel_policy("entity-geometry-comparison-v1"),
        measurement=geometry_measurement(),
        findings=(finding(status, "geometry-overlap", metric="bounds_iou", observed=0.5),),
    )


def semantic_evidence(status: EvidenceStatus = EvidenceStatus.SUPPORTING) -> SemanticEvidence:
    return SemanticEvidence(
        policy=channel_policy("entity-semantic-compatibility-v1"),
        measurement=SemanticMeasurement(
            ambiguity_a=AmbiguityState.UNAMBIGUOUS,
            ambiguity_b=AmbiguityState.UNAMBIGUOUS,
            hypothesis_count_a=1,
            hypothesis_count_b=1,
            label_comparisons=(
                LabelComparison(
                    label_a="pallet",
                    label_b="wooden pallet",
                    primary_a=True,
                    primary_b=True,
                    relation=LabelRelation.REFINEMENT,
                    rule_id="label-refinement-v1",
                ),
            ),
            attribute_comparisons=(),
        ),
        findings=(finding(status, "semantic-refinement"),),
    )


def feature_ref(index: int, *, space: str = EMBEDDING_SPACE) -> EntityFeatureRef:
    return EntityFeatureRef(
        perception_run_id=PerceptionRunId("perception-run-0001"),
        perception_result_id=PerceptionResultId(f"result-{index:04d}"),
        feature_id=FeatureId(f"feature-{index:04d}"),
        embedding_space_id=space,
        scope=FeatureScope.REGION,
        region_id=RegionId("region-0001"),
    )


def appearance_measurement(**overrides: Any) -> AppearanceMeasurement:
    values: dict[str, Any] = {
        "embedding_space_id": EMBEDDING_SPACE,
        "metric": "cosine-similarity",
        "aggregation_id": "physical-observation-prototype-v1",
        "similarity": 0.91,
        "pair_similarity_min": 0.88,
        "pair_similarity_max": 0.93,
        "contributions_a": (
            FeatureContribution(
                physical_observation_id="frame-0120", feature_refs=(feature_ref(1),)
            ),
        ),
        "contributions_b": (
            FeatureContribution(
                physical_observation_id="frame-0130", feature_refs=(feature_ref(2),)
            ),
        ),
    }
    values.update(overrides)
    return AppearanceMeasurement(**values)


def appearance_evidence(status: EvidenceStatus = EvidenceStatus.SUPPORTING) -> AppearanceEvidence:
    return AppearanceEvidence(
        policy=channel_policy("entity-appearance-comparison-v1"),
        measurement=appearance_measurement(),
        findings=(finding(status, "appearance-similarity", metric="similarity", observed=0.91),),
    )


def temporal_measurement(**overrides: Any) -> TemporalMeasurement:
    values: dict[str, Any] = {
        "clock_id": "fixture:header",
        "interval_overlap_ns": 2_000_000_000,
        "interval_gap_ns": 0,
        "physical_observation_count_a": 2,
        "physical_observation_count_b": 3,
        "inference_result_count_a": 3,
        "inference_result_count_b": 3,
        "shared_physical_observation_count": 1,
        "union_physical_observation_count": 4,
    }
    values.update(overrides)
    return TemporalMeasurement(**values)


def temporal_evidence(status: EvidenceStatus = EvidenceStatus.NEUTRAL) -> TemporalEvidence:
    return TemporalEvidence(
        policy=channel_policy("entity-temporal-compatibility-v1"),
        measurement=temporal_measurement(),
        findings=(finding(status, "temporal-overlap"),),
    )


def representation_ref(index: int) -> RepresentationRef:
    map_id = MAP_ID
    return RepresentationRef(
        run_id=PointRepresentationRunId("representation-run-0001"),
        representation_id=PointRepresentationId(f"representation-{index:04d}"),
        representation_space_id=REPRESENTATION_SPACE,
        geometry_reference=GeometryReference(
            map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
        ),
    )


def representation_evidence(
    status: EvidenceStatus = EvidenceStatus.SUPPORTING,
) -> PointRepresentationEvidence:
    return PointRepresentationEvidence(
        policy=channel_policy("entity-representation-comparison-v1"),
        measurement=RepresentationMeasurement(
            representation_space_id=REPRESENTATION_SPACE,
            metric="cosine-similarity",
            aggregation_id="support-prototype-v1",
            similarity=0.84,
            pair_similarity_min=0.8,
            pair_similarity_max=0.9,
            representations_a=(representation_ref(0),),
            representations_b=(representation_ref(1),),
            dimension=4,
            compared_components=3,
        ),
        findings=(
            finding(status, "representation-similarity", metric="similarity", observed=0.84),
        ),
    )


def passed_gates() -> tuple[GateResult, ...]:
    return (
        GateResult(gate_id="distinct-entities", passed=True, detail="two different entities"),
        GateResult(gate_id="same-geometric-map", passed=True, detail="both over map-0001"),
        GateResult(gate_id="same-map-frame", passed=True, detail="both in frame map"),
    )


def match_evidence(**channels: Any) -> EntityMatchEvidence:
    values: dict[str, Any] = {
        "comparison_id": comparison_id_for(REF_A, REF_B),
        "entity_a_ref": REF_A,
        "entity_b_ref": REF_B,
        "gates": passed_gates(),
        "geometry": geometry_evidence(),
        "provenance": MatchEvidenceProvenance(gate_policy_id="entity-comparison-gates-v1"),
    }
    values.update(channels)
    return EntityMatchEvidence(**values)


def policy_ref() -> PolicyRef:
    return PolicyRef(
        policy_id="conservative-staged-resolution-v1", configuration_fingerprint="sha256:policy"
    )


def triggered_rule(
    outcome: ResolutionOutcome | None,
    *,
    stage: PolicyStage = PolicyStage.DECISION,
    rule_id: str = "geometry-only-match",
) -> TriggeredRule:
    return TriggeredRule(
        stage=stage,
        rule_id=rule_id,
        outcome=outcome,
        detail=f"{rule_id} fired",
        channels=(),
    )


def decision(
    outcome: ResolutionOutcome = ResolutionOutcome.MATCH, **overrides: Any
) -> ResolutionDecision:
    comparison = comparison_id_for(REF_A, REF_B)
    policy = policy_ref()
    values: dict[str, Any] = {
        "decision_id": decision_id_for(comparison, policy),
        "entity_a_ref": REF_A,
        "entity_b_ref": REF_B,
        "decision": outcome,
        "evidence_ref": comparison,
        "policy": policy,
        "triggered_rules": (triggered_rule(outcome),),
        "channels_used": (MatchChannel.GEOMETRY,),
        "channels_ignored": (),
        "unresolved_reason": None,
        "provenance": DecisionProvenance(code_version="test"),
    }
    values.update(overrides)
    return ResolutionDecision(**values)
