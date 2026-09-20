"""JSON-friendly encoding of the Semantic Fusion contracts.

Records contain only JSON primitives, so a persisted fusion run is readable without ROS,
NumPy or any model runtime. Decoding rebuilds the contracts through their constructors, so
every invariant is revalidated and a tampered record is refused rather than trusted.

Geometry is not repeated as identifiers: a geometry identity is positional (``map--geom-N``),
so a sorted set is stored as the first index followed by the differences, which keeps a
region of thousands of points compact. Nothing here copies XYZ, embeddings, claims or
quality components: upstream evidence is referenced by identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryReference,
    MapId,
    geometry_id_for,
    geometry_index_of,
)
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.point_representation import PointRepresentationId, PointRepresentationRunId
from contextmap.semantic_fusion.models import (
    ChannelProvenance,
    ComponentFactor,
    ComponentTreatment,
    ContributionWeight,
    EvidenceChannel,
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
    HypothesisSupport,
    ObservationFactor,
    ObservationQualityRef,
    PhysicalObservationGroup,
    PointRepresentationRef,
    QualityWeighting,
    ScoreReference,
    SupportSignal,
    SupportSignalKind,
    UncertaintyKind,
    UncertaintyRecord,
)
from contextmap.sensor_association import SemanticClaimRef, SpatialObservationId, VisualFeatureRef
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


def encode_fusion_support(support: FusionSupport) -> dict[str, Any]:
    """Encode a support: geometry, observations, extent, time and policy."""
    return {
        "fusion_support_id": str(support.fusion_support_id),
        "geometric_map_id": str(support.geometric_map_id),
        "geometry": _encode_geometry(support.geometry_support),
        "spatial_observation_ids": [str(item) for item in support.spatial_observation_ids],
        "bounds": {
            "frame_id": str(support.bounds.frame_id),
            "minimum_m": list(support.bounds.minimum_m),
            "maximum_m": list(support.bounds.maximum_m),
        },
        "centroid_m": list(support.centroid_m),
        "time_bounds": _encode_time_bounds(support.time_bounds),
        "provenance": {
            "support_policy_id": support.provenance.support_policy_id,
            "configuration_fingerprint": support.provenance.configuration_fingerprint,
            "code_version": support.provenance.code_version,
        },
    }


def decode_fusion_support(record: Mapping[str, Any]) -> FusionSupport:
    """Decode a support and revalidate its contract.

    Raises:
        ValueError: If the record is malformed or violates the support contract.
    """
    bounds = record["bounds"]
    provenance = record["provenance"]
    x, y, z = record["centroid_m"]
    return FusionSupport(
        fusion_support_id=FusionSupportId(record["fusion_support_id"]),
        geometric_map_id=MapId(record["geometric_map_id"]),
        geometry_support=_decode_geometry(record["geometry"]),
        spatial_observation_ids=tuple(
            SpatialObservationId(item) for item in record["spatial_observation_ids"]
        ),
        bounds=Bounds3D(
            frame_id=FrameId(bounds["frame_id"]),
            minimum_m=_vector(bounds["minimum_m"]),
            maximum_m=_vector(bounds["maximum_m"]),
        ),
        centroid_m=(x, y, z),
        time_bounds=_decode_time_bounds(record["time_bounds"]),
        provenance=FusionSupportProvenance(
            support_policy_id=provenance["support_policy_id"],
            configuration_fingerprint=provenance["configuration_fingerprint"],
            code_version=provenance["code_version"],
        ),
    )


def encode_fused_evidence(fused: FusedEvidence) -> dict[str, Any]:
    """Encode the evidence fused over one support, with nothing collapsed."""
    return {
        "fused_evidence_id": str(fused.fused_evidence_id),
        "fusion_support_id": str(fused.fusion_support_id),
        "physical_observation_groups": [
            _encode_group(group) for group in fused.physical_observation_groups
        ],
        "contributions": [_encode_contribution(item) for item in fused.contributions],
        "hypotheses": [_encode_hypothesis(item) for item in fused.hypotheses],
        "temporal_summary": _encode_time_bounds(fused.temporal_summary),
        "provenance": {
            "grouping_policy_id": fused.provenance.grouping_policy_id,
            "fusion_policy_id": fused.provenance.fusion_policy_id,
            "configuration_fingerprint": fused.provenance.configuration_fingerprint,
            "code_version": fused.provenance.code_version,
        },
        "channels": [
            {"channel": item.channel.value, "identities": list(item.identities)}
            for item in fused.channels
        ],
        "point_representation_refs": [
            {
                "representation_id": str(ref.representation_id),
                "run_id": str(ref.run_id),
                "representation_space_id": ref.representation_space_id,
                "geometry": _encode_geometry((ref.geometry_reference,)),
            }
            for ref in fused.point_representation_refs
        ],
        "uncertainty": [
            {
                "kind": record.kind.value,
                "hypothesis_ids": [str(item) for item in record.hypothesis_ids],
                "evidence": [_encode_reference(ref) for ref in record.evidence],
                "rule_id": record.rule_id,
            }
            for record in fused.uncertainty
        ],
        "weighting": None if fused.weighting is None else _encode_weighting(fused.weighting),
    }


def decode_fused_evidence(record: Mapping[str, Any]) -> FusedEvidence:
    """Decode fused evidence and revalidate every invariant of its contract.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    provenance = record["provenance"]
    return FusedEvidence(
        fused_evidence_id=FusedEvidenceId(record["fused_evidence_id"]),
        fusion_support_id=FusionSupportId(record["fusion_support_id"]),
        physical_observation_groups=tuple(
            _decode_group(item) for item in record["physical_observation_groups"]
        ),
        contributions=tuple(_decode_contribution(item) for item in record["contributions"]),
        hypotheses=tuple(_decode_hypothesis(item) for item in record["hypotheses"]),
        temporal_summary=_decode_time_bounds(record["temporal_summary"]),
        provenance=FusedEvidenceProvenance(
            grouping_policy_id=provenance["grouping_policy_id"],
            fusion_policy_id=provenance["fusion_policy_id"],
            configuration_fingerprint=provenance["configuration_fingerprint"],
            code_version=provenance["code_version"],
        ),
        channels=tuple(
            ChannelProvenance(
                channel=EvidenceChannel(item["channel"]), identities=tuple(item["identities"])
            )
            for item in record["channels"]
        ),
        point_representation_refs=tuple(
            PointRepresentationRef(
                representation_id=PointRepresentationId(item["representation_id"]),
                run_id=PointRepresentationRunId(item["run_id"]),
                representation_space_id=item["representation_space_id"],
                geometry_reference=_decode_geometry(item["geometry"])[0],
            )
            for item in record["point_representation_refs"]
        ),
        uncertainty=tuple(
            UncertaintyRecord(
                kind=UncertaintyKind(item["kind"]),
                hypothesis_ids=tuple(FusedHypothesisId(h) for h in item["hypothesis_ids"]),
                evidence=tuple(_decode_reference(ref) for ref in item["evidence"]),
                rule_id=item["rule_id"],
            )
            for item in record["uncertainty"]
        ),
        weighting=None if record["weighting"] is None else _decode_weighting(record["weighting"]),
    )


def _encode_geometry(references: Sequence[GeometryReference]) -> dict[str, Any]:
    map_id = references[0].map_id
    indexes = [
        geometry_index_of(map_id=map_id, geometry_id=reference.geometry_id)
        for reference in references
    ]
    deltas = [indexes[0], *(later - earlier for earlier, later in pairwise(indexes))]
    return {"map_id": str(map_id), "deltas": deltas}


def _decode_geometry(record: Mapping[str, Any]) -> tuple[GeometryReference, ...]:
    map_id = MapId(record["map_id"])
    references: list[GeometryReference] = []
    index = 0
    for delta in record["deltas"]:
        index += delta
        references.append(
            GeometryReference(
                map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
            )
        )
    return tuple(references)


def _vector(values: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = values
    return (x, y, z)


def _encode_time_bounds(bounds: TimeBounds) -> dict[str, Any]:
    return {"start": bounds.start.to_record(), "end": bounds.end.to_record()}


def _decode_time_bounds(record: Mapping[str, Any]) -> TimeBounds:
    return TimeBounds(
        start=SourceTimestamp.from_record(record["start"]),
        end=SourceTimestamp.from_record(record["end"]),
    )


def _encode_producer(producer: BackendProvenance) -> dict[str, Any]:
    return {
        "backend_id": producer.backend_id,
        "capability": producer.capability,
        "provider": producer.provider,
        "model": producer.model,
        "version": producer.version,
        "configuration_fingerprint": producer.configuration_fingerprint,
    }


def _decode_producer(record: Mapping[str, Any]) -> BackendProvenance:
    return BackendProvenance(
        backend_id=record["backend_id"],
        capability=record["capability"],
        provider=record["provider"],
        model=record["model"],
        version=record["version"],
        configuration_fingerprint=record["configuration_fingerprint"],
    )


def _encode_group(group: PhysicalObservationGroup) -> dict[str, Any]:
    return {
        "physical_observation_id": str(group.physical_observation_id),
        "acquisition_timestamp": group.acquisition_timestamp.to_record(),
        "spatial_observation_ids": [str(item) for item in group.spatial_observation_ids],
        "perception_result_ids": [str(item) for item in group.perception_result_ids],
        "perception_run_ids": [str(item) for item in group.perception_run_ids],
    }


def _decode_group(record: Mapping[str, Any]) -> PhysicalObservationGroup:
    return PhysicalObservationGroup(
        physical_observation_id=SourceObservationId(record["physical_observation_id"]),
        acquisition_timestamp=SourceTimestamp.from_record(record["acquisition_timestamp"]),
        spatial_observation_ids=tuple(
            SpatialObservationId(item) for item in record["spatial_observation_ids"]
        ),
        perception_result_ids=tuple(
            PerceptionResultId(item) for item in record["perception_result_ids"]
        ),
        perception_run_ids=tuple(PerceptionRunId(item) for item in record["perception_run_ids"]),
    )


def _encode_contribution(item: EvidenceContribution) -> dict[str, Any]:
    quality = item.observation_quality
    return {
        "contribution_id": str(item.contribution_id),
        "physical_observation_id": str(item.physical_observation_id),
        "perception_result_id": str(item.perception_result_id),
        "perception_run_id": str(item.perception_run_id),
        "spatial_observation_id": str(item.spatial_observation_id),
        "region_id": str(item.region_id),
        "geometry": _encode_geometry(item.geometry_support),
        "claim_ids": [str(ref.claim_id) for ref in item.claim_refs],
        "score_refs": [
            {"claim_id": str(ref.claim_id), "scorer": _encode_producer(ref.scorer)}
            for ref in item.score_refs
        ],
        "visual_feature_refs": [
            {
                "feature_id": str(ref.feature_id),
                "embedding_space_id": ref.embedding_space_id,
                "scope": ref.scope.value,
                "region_id": None if ref.region_id is None else str(ref.region_id),
            }
            for ref in item.visual_feature_refs
        ],
        "observation_quality": None
        if quality is None
        else {
            "spatial_observation_id": str(quality.spatial_observation_id),
            "definitions_version": quality.definitions_version,
        },
    }


def _decode_contribution(record: Mapping[str, Any]) -> EvidenceContribution:
    quality = record["observation_quality"]
    return EvidenceContribution(
        contribution_id=EvidenceContributionId(record["contribution_id"]),
        physical_observation_id=SourceObservationId(record["physical_observation_id"]),
        perception_result_id=PerceptionResultId(record["perception_result_id"]),
        perception_run_id=PerceptionRunId(record["perception_run_id"]),
        spatial_observation_id=SpatialObservationId(record["spatial_observation_id"]),
        region_id=RegionId(record["region_id"]),
        geometry_support=_decode_geometry(record["geometry"]),
        claim_refs=tuple(SemanticClaimRef(claim_id=ClaimId(item)) for item in record["claim_ids"]),
        score_refs=tuple(
            ScoreReference(
                claim_id=ClaimId(item["claim_id"]), scorer=_decode_producer(item["scorer"])
            )
            for item in record["score_refs"]
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
        observation_quality=None
        if quality is None
        else ObservationQualityRef(
            spatial_observation_id=SpatialObservationId(quality["spatial_observation_id"]),
            definitions_version=quality["definitions_version"],
        ),
    )


def _encode_hypothesis(hypothesis: FusedHypothesis) -> dict[str, Any]:
    return {
        "hypothesis_id": str(hypothesis.hypothesis_id),
        "label": hypothesis.label,
        "evidence": [
            {
                "contribution_id": str(item.contribution_id),
                "claim_id": str(item.claim_id),
                "stance": item.stance.value,
                "role": item.role.value,
                "signals": [
                    {
                        "kind": signal.kind.value,
                        "producer": _encode_producer(signal.producer),
                        "value": signal.value,
                    }
                    for signal in item.signals
                ],
            }
            for item in hypothesis.evidence
        ],
    }


def _decode_hypothesis(record: Mapping[str, Any]) -> FusedHypothesis:
    return FusedHypothesis(
        hypothesis_id=FusedHypothesisId(record["hypothesis_id"]),
        label=record["label"],
        evidence=tuple(
            HypothesisEvidence(
                contribution_id=EvidenceContributionId(item["contribution_id"]),
                claim_id=ClaimId(item["claim_id"]),
                stance=EvidenceStance(item["stance"]),
                role=HypothesisRole(item["role"]),
                signals=tuple(
                    SupportSignal(
                        kind=SupportSignalKind(signal["kind"]),
                        producer=_decode_producer(signal["producer"]),
                        value=signal["value"],
                    )
                    for signal in item["signals"]
                ),
            )
            for item in record["evidence"]
        ),
    )


def _encode_reference(reference: EvidenceReference) -> dict[str, Any]:
    return {
        "contribution_id": str(reference.contribution_id),
        "claim_id": None if reference.claim_id is None else str(reference.claim_id),
    }


def _decode_reference(record: Mapping[str, Any]) -> EvidenceReference:
    claim_id = record["claim_id"]
    return EvidenceReference(
        contribution_id=EvidenceContributionId(record["contribution_id"]),
        claim_id=None if claim_id is None else ClaimId(claim_id),
    )


def _encode_weighting(weighting: QualityWeighting) -> dict[str, Any]:
    return {
        "policy_id": weighting.policy_id,
        "combination_rule": weighting.combination_rule,
        "observation_rule": weighting.observation_rule,
        "definitions_version": weighting.definitions_version,
        "contributions": [
            {
                "contribution_id": str(item.contribution_id),
                "factor": item.factor,
                "components": [
                    {
                        "component": component.component,
                        "treatment": component.treatment.value,
                        "measured_value": component.measured_value,
                        "factor": component.factor,
                        "unavailable_reason": component.unavailable_reason,
                    }
                    for component in item.components
                ],
            }
            for item in weighting.contributions
        ],
        "hypotheses": [
            {
                "hypothesis_id": str(item.hypothesis_id),
                "supporting_physical_observations": item.supporting_physical_observations,
                "weighted_support": item.weighted_support,
                "observation_factors": [
                    {
                        "physical_observation_id": str(factor.physical_observation_id),
                        "factor": factor.factor,
                    }
                    for factor in item.observation_factors
                ],
            }
            for item in weighting.hypotheses
        ],
    }


def _decode_weighting(record: Mapping[str, Any]) -> QualityWeighting:
    return QualityWeighting(
        policy_id=record["policy_id"],
        combination_rule=record["combination_rule"],
        observation_rule=record["observation_rule"],
        definitions_version=record["definitions_version"],
        contributions=tuple(
            ContributionWeight(
                contribution_id=EvidenceContributionId(item["contribution_id"]),
                factor=item["factor"],
                components=tuple(
                    ComponentFactor(
                        component=component["component"],
                        treatment=ComponentTreatment(component["treatment"]),
                        measured_value=component["measured_value"],
                        factor=component["factor"],
                        unavailable_reason=component["unavailable_reason"],
                    )
                    for component in item["components"]
                ),
            )
            for item in record["contributions"]
        ),
        hypotheses=tuple(
            HypothesisSupport(
                hypothesis_id=FusedHypothesisId(item["hypothesis_id"]),
                supporting_physical_observations=item["supporting_physical_observations"],
                weighted_support=item["weighted_support"],
                observation_factors=tuple(
                    ObservationFactor(
                        physical_observation_id=SourceObservationId(
                            factor["physical_observation_id"]
                        ),
                        factor=factor["factor"],
                    )
                    for factor in item["observation_factors"]
                ),
            )
            for item in record["hypotheses"]
        ),
    )
