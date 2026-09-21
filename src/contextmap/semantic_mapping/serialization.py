"""JSON-friendly encoding of the Semantic Mapping contracts.

Records contain only JSON primitives, so a persisted semantic map is readable without ROS, NumPy
or any model runtime. Decoding rebuilds the contracts through their constructors, so every
invariant is revalidated and a tampered record is refused rather than trusted.

Geometry is not repeated as identifiers: a geometry identity is positional (``map--geom-N``), so
a sorted set is stored as the first index followed by the differences, which keeps an entity of
thousands of points compact. Nothing here copies XYZ, embeddings, claims or quality components:
upstream evidence is referenced by identity.
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
from contextmap.semantic_fusion import (
    EvidenceContributionId,
    EvidenceReference,
    EvidenceStance,
    FusedEvidenceId,
    FusedHypothesisId,
    FusionSupportId,
    HypothesisEvidence,
    PointRepresentationRef,
    SemanticFusionRunId,
    SupportSignal,
    SupportSignalKind,
    UncertaintyKind,
    UncertaintyRecord,
)
from contextmap.semantic_mapping.evidence import (
    EntityEvidenceLinks,
    EntityFeatureRef,
    FusedEvidenceRef,
)
from contextmap.semantic_mapping.geometry import (
    EntityGeometry,
    EntityOrientation,
    GeometryDiagnostic,
    GeometryDiagnosticKind,
    SpatialSummaryProvenance,
    SupportStatistics,
)
from contextmap.semantic_mapping.models import (
    Entity,
    EntityId,
    EntityProvenance,
    EntityReference,
    SemanticMapId,
)
from contextmap.semantic_mapping.semantic_state import (
    AmbiguityState,
    AttributeOrigin,
    EntityAttribute,
    EntityHypothesis,
    EntityHypothesisRef,
    EntitySemanticState,
    EntityUncertainty,
    SemanticStateProvenance,
)
from contextmap.semantic_mapping.temporal import (
    EntityLifecycle,
    EntityTemporalState,
    ObservationRef,
    TemporalProvenance,
)
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import SourceTimestamp, Vector3
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


def encode_entity_reference(reference: EntityReference) -> dict[str, Any]:
    """Encode the stable handle of an entity."""
    return {
        "semantic_map_id": str(reference.semantic_map_id),
        "entity_id": str(reference.entity_id),
    }


def decode_entity_reference(record: Mapping[str, Any]) -> EntityReference:
    """Decode an entity reference and revalidate it.

    Raises:
        ValueError: If the record is malformed or an identity is empty.
    """
    return EntityReference(
        semantic_map_id=SemanticMapId(record["semantic_map_id"]),
        entity_id=EntityId(record["entity_id"]),
    )


def encode_entity(entity: Entity) -> dict[str, Any]:
    """Encode an entity with its geometry, semantic state, evidence and temporal state."""
    return {
        "entity_id": str(entity.entity_id),
        "semantic_map_id": str(entity.semantic_map_id),
        "geometry": _encode_geometry(entity.geometry),
        "semantic_state": _encode_semantic_state(entity.semantic_state),
        "evidence": _encode_evidence(entity.evidence),
        "temporal_state": _encode_temporal_state(entity.temporal_state),
        "provenance": {
            "materialization_policy_id": entity.provenance.materialization_policy_id,
            "identity_policy_id": entity.provenance.identity_policy_id,
            "configuration_fingerprint": entity.provenance.configuration_fingerprint,
            "code_version": entity.provenance.code_version,
        },
    }


def decode_entity(record: Mapping[str, Any]) -> Entity:
    """Decode an entity and revalidate every invariant of its contract.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    provenance = record["provenance"]
    return Entity(
        entity_id=EntityId(record["entity_id"]),
        semantic_map_id=SemanticMapId(record["semantic_map_id"]),
        geometry=_decode_geometry(record["geometry"]),
        semantic_state=_decode_semantic_state(record["semantic_state"]),
        evidence=_decode_evidence(record["evidence"]),
        temporal_state=_decode_temporal_state(record["temporal_state"]),
        provenance=EntityProvenance(
            materialization_policy_id=provenance["materialization_policy_id"],
            identity_policy_id=provenance["identity_policy_id"],
            configuration_fingerprint=provenance["configuration_fingerprint"],
            code_version=provenance["code_version"],
        ),
    )


def _encode_geometry(geometry: EntityGeometry) -> dict[str, Any]:
    orientation = geometry.orientation
    statistics = geometry.statistics
    summary = geometry.summary
    return {
        "map_frame": str(geometry.map_frame),
        **_encode_geometry_refs(geometry.geometry_refs),
        "centroid_m": list(geometry.centroid_m),
        "bounds": {
            "frame_id": str(geometry.bounds.frame_id),
            "minimum_m": list(geometry.bounds.minimum_m),
            "maximum_m": list(geometry.bounds.maximum_m),
        },
        "extent_m": list(geometry.extent_m),
        "statistics": {
            "point_count": statistics.point_count,
            "volume_m3": statistics.volume_m3,
            "density_per_m3": statistics.density_per_m3,
            "component_count": statistics.component_count,
            "largest_component_fraction": statistics.largest_component_fraction,
        },
        "summary": {
            "algorithm_id": summary.algorithm_id,
            "map_frame": str(summary.map_frame),
            "input_geometry_count": summary.input_geometry_count,
            "input_geometry_digest": summary.input_geometry_digest,
            "numerical_conventions": summary.numerical_conventions,
            "filtering": summary.filtering,
            "configuration_fingerprint": summary.configuration_fingerprint,
        },
        "orientation": None
        if orientation is None
        else {
            "axes": [list(axis) for axis in orientation.axes],
            "variances_m2": list(orientation.variances_m2),
            "method": orientation.method,
        },
        "diagnostics": [
            {"kind": item.kind.value, "detail": item.detail} for item in geometry.diagnostics
        ],
    }


def _decode_geometry(record: Mapping[str, Any]) -> EntityGeometry:
    bounds = record["bounds"]
    statistics = record["statistics"]
    summary = record["summary"]
    orientation = record["orientation"]
    return EntityGeometry(
        geometry_refs=_decode_geometry_refs(record),
        map_frame=FrameId(record["map_frame"]),
        centroid_m=_vector(record["centroid_m"]),
        bounds=Bounds3D(
            frame_id=FrameId(bounds["frame_id"]),
            minimum_m=_vector(bounds["minimum_m"]),
            maximum_m=_vector(bounds["maximum_m"]),
        ),
        extent_m=_vector(record["extent_m"]),
        statistics=SupportStatistics(
            point_count=statistics["point_count"],
            volume_m3=statistics["volume_m3"],
            density_per_m3=statistics["density_per_m3"],
            component_count=statistics["component_count"],
            largest_component_fraction=statistics["largest_component_fraction"],
        ),
        summary=SpatialSummaryProvenance(
            algorithm_id=summary["algorithm_id"],
            map_frame=FrameId(summary["map_frame"]),
            input_geometry_count=summary["input_geometry_count"],
            input_geometry_digest=summary["input_geometry_digest"],
            numerical_conventions=summary["numerical_conventions"],
            filtering=summary["filtering"],
            configuration_fingerprint=summary["configuration_fingerprint"],
        ),
        orientation=None
        if orientation is None
        else EntityOrientation(
            axes=(
                _vector(orientation["axes"][0]),
                _vector(orientation["axes"][1]),
                _vector(orientation["axes"][2]),
            ),
            variances_m2=_vector(orientation["variances_m2"]),
            method=orientation["method"],
        ),
        diagnostics=tuple(
            GeometryDiagnostic(kind=GeometryDiagnosticKind(item["kind"]), detail=item["detail"])
            for item in record["diagnostics"]
        ),
    )


def _vector(values: Sequence[float]) -> Vector3:
    x, y, z = values
    return (x, y, z)


def _encode_geometry_refs(references: Sequence[GeometryReference]) -> dict[str, Any]:
    map_id = references[0].map_id
    indexes = [
        geometry_index_of(map_id=map_id, geometry_id=reference.geometry_id)
        for reference in references
    ]
    deltas = [indexes[0], *(later - earlier for earlier, later in pairwise(indexes))]
    return {"map_id": str(map_id), "deltas": deltas}


def _decode_geometry_refs(record: Mapping[str, Any]) -> tuple[GeometryReference, ...]:
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


def _encode_semantic_state(state: EntitySemanticState) -> dict[str, Any]:
    primary = state.primary_hypothesis
    return {
        "hypotheses": [_encode_hypothesis(item) for item in state.hypotheses],
        "ambiguity_state": state.ambiguity_state.value,
        "primary_hypothesis": None if primary is None else _encode_hypothesis_ref(primary),
        "attributes": [_encode_attribute(item) for item in state.attributes],
        "uncertainty": [_encode_uncertainty(item) for item in state.uncertainty],
        "provenance": {
            "mapping_rule_id": state.provenance.mapping_rule_id,
            "primary_policy_id": state.provenance.primary_policy_id,
        },
    }


def _decode_semantic_state(record: Mapping[str, Any]) -> EntitySemanticState:
    primary = record["primary_hypothesis"]
    provenance = record["provenance"]
    return EntitySemanticState(
        hypotheses=tuple(_decode_hypothesis(item) for item in record["hypotheses"]),
        ambiguity_state=AmbiguityState(record["ambiguity_state"]),
        provenance=SemanticStateProvenance(
            mapping_rule_id=provenance["mapping_rule_id"],
            primary_policy_id=provenance["primary_policy_id"],
        ),
        primary_hypothesis=None if primary is None else _decode_hypothesis_ref(primary),
        attributes=tuple(_decode_attribute(item) for item in record["attributes"]),
        uncertainty=tuple(_decode_uncertainty(item) for item in record["uncertainty"]),
    )


def _encode_hypothesis_ref(ref: EntityHypothesisRef) -> dict[str, Any]:
    return {
        "fused_evidence_id": str(ref.fused_evidence_id),
        "hypothesis_id": str(ref.hypothesis_id),
    }


def _decode_hypothesis_ref(record: Mapping[str, Any]) -> EntityHypothesisRef:
    return EntityHypothesisRef(
        fused_evidence_id=FusedEvidenceId(record["fused_evidence_id"]),
        hypothesis_id=FusedHypothesisId(record["hypothesis_id"]),
    )


def _encode_hypothesis(hypothesis: EntityHypothesis) -> dict[str, Any]:
    return {
        "fused_evidence_id": str(hypothesis.fused_evidence_id),
        "hypothesis_id": str(hypothesis.hypothesis_id),
        "label": hypothesis.label,
        "evidence": [_encode_hypothesis_evidence(item) for item in hypothesis.evidence],
    }


def _decode_hypothesis(record: Mapping[str, Any]) -> EntityHypothesis:
    return EntityHypothesis(
        fused_evidence_id=FusedEvidenceId(record["fused_evidence_id"]),
        hypothesis_id=FusedHypothesisId(record["hypothesis_id"]),
        label=record["label"],
        evidence=tuple(_decode_hypothesis_evidence(item) for item in record["evidence"]),
    )


def _encode_hypothesis_evidence(item: HypothesisEvidence) -> dict[str, Any]:
    return {
        "contribution_id": str(item.contribution_id),
        "claim_id": str(item.claim_id),
        "stance": item.stance.value,
        "role": item.role.value,
        "signals": _encode_signals(item.signals),
    }


def _decode_hypothesis_evidence(record: Mapping[str, Any]) -> HypothesisEvidence:
    return HypothesisEvidence(
        contribution_id=EvidenceContributionId(record["contribution_id"]),
        claim_id=ClaimId(record["claim_id"]),
        stance=EvidenceStance(record["stance"]),
        role=HypothesisRole(record["role"]),
        signals=_decode_signals(record["signals"]),
    )


def _encode_attribute(attribute: EntityAttribute) -> dict[str, Any]:
    return {
        "name": attribute.name,
        "value": attribute.value,
        "origin": attribute.origin.value,
        "derivation_id": attribute.derivation_id,
        "evidence": [_encode_reference(ref) for ref in attribute.evidence],
        "support": _encode_signals(attribute.support),
    }


def _decode_attribute(record: Mapping[str, Any]) -> EntityAttribute:
    return EntityAttribute(
        name=record["name"],
        value=record["value"],
        origin=AttributeOrigin(record["origin"]),
        derivation_id=record["derivation_id"],
        evidence=tuple(_decode_reference(ref) for ref in record["evidence"]),
        support=_decode_signals(record["support"]),
    )


def _encode_uncertainty(item: EntityUncertainty) -> dict[str, Any]:
    record = item.record
    return {
        "fused_evidence_id": str(item.fused_evidence_id),
        "kind": record.kind.value,
        "hypothesis_ids": [str(hypothesis_id) for hypothesis_id in record.hypothesis_ids],
        "evidence": [_encode_reference(ref) for ref in record.evidence],
        "rule_id": record.rule_id,
    }


def _decode_uncertainty(record: Mapping[str, Any]) -> EntityUncertainty:
    return EntityUncertainty(
        fused_evidence_id=FusedEvidenceId(record["fused_evidence_id"]),
        record=UncertaintyRecord(
            kind=UncertaintyKind(record["kind"]),
            hypothesis_ids=tuple(FusedHypothesisId(item) for item in record["hypothesis_ids"]),
            evidence=tuple(_decode_reference(ref) for ref in record["evidence"]),
            rule_id=record["rule_id"],
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


def _encode_signals(signals: Sequence[SupportSignal]) -> list[dict[str, Any]]:
    return [
        {
            "kind": signal.kind.value,
            "producer": _encode_producer(signal.producer),
            "value": signal.value,
        }
        for signal in signals
    ]


def _decode_signals(records: Sequence[Mapping[str, Any]]) -> tuple[SupportSignal, ...]:
    return tuple(
        SupportSignal(
            kind=SupportSignalKind(record["kind"]),
            producer=_decode_producer(record["producer"]),
            value=record["value"],
        )
        for record in records
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


def _encode_evidence(evidence: EntityEvidenceLinks) -> dict[str, Any]:
    return {
        "fused_evidence": [
            {
                "fusion_run_id": str(ref.fusion_run_id),
                "fusion_schema_version": ref.fusion_schema_version,
                "fusion_artifact_digest": ref.fusion_artifact_digest,
                "sequence_artifact_id": ref.sequence_artifact_id,
                "fused_evidence_id": str(ref.fused_evidence_id),
                "fusion_support_id": str(ref.fusion_support_id),
            }
            for ref in evidence.fused_evidence
        ],
        "spatial_observation_ids": [str(item) for item in evidence.spatial_observation_ids],
        "physical_observation_ids": [str(item) for item in evidence.physical_observation_ids],
        "visual_feature_refs": [
            {
                "perception_run_id": str(ref.perception_run_id),
                "perception_result_id": str(ref.perception_result_id),
                "feature_id": str(ref.feature_id),
                "embedding_space_id": ref.embedding_space_id,
                "scope": ref.scope.value,
                "region_id": None if ref.region_id is None else str(ref.region_id),
            }
            for ref in evidence.visual_feature_refs
        ],
        "point_representation_refs": [
            {
                "representation_id": str(ref.representation_id),
                "run_id": str(ref.run_id),
                "representation_space_id": ref.representation_space_id,
                "geometry": _encode_geometry_refs((ref.geometry_reference,)),
            }
            for ref in evidence.point_representation_refs
        ],
    }


def _decode_evidence(record: Mapping[str, Any]) -> EntityEvidenceLinks:
    return EntityEvidenceLinks(
        fused_evidence=tuple(
            FusedEvidenceRef(
                fusion_run_id=SemanticFusionRunId(item["fusion_run_id"]),
                fusion_schema_version=item["fusion_schema_version"],
                fusion_artifact_digest=item["fusion_artifact_digest"],
                sequence_artifact_id=item["sequence_artifact_id"],
                fused_evidence_id=FusedEvidenceId(item["fused_evidence_id"]),
                fusion_support_id=FusionSupportId(item["fusion_support_id"]),
            )
            for item in record["fused_evidence"]
        ),
        spatial_observation_ids=tuple(
            SpatialObservationId(item) for item in record["spatial_observation_ids"]
        ),
        physical_observation_ids=tuple(
            SourceObservationId(item) for item in record["physical_observation_ids"]
        ),
        visual_feature_refs=tuple(
            EntityFeatureRef(
                perception_run_id=PerceptionRunId(item["perception_run_id"]),
                perception_result_id=PerceptionResultId(item["perception_result_id"]),
                feature_id=FeatureId(item["feature_id"]),
                embedding_space_id=item["embedding_space_id"],
                scope=FeatureScope(item["scope"]),
                region_id=None if item["region_id"] is None else RegionId(item["region_id"]),
            )
            for item in record["visual_feature_refs"]
        ),
        point_representation_refs=tuple(
            PointRepresentationRef(
                representation_id=PointRepresentationId(item["representation_id"]),
                run_id=PointRepresentationRunId(item["run_id"]),
                representation_space_id=item["representation_space_id"],
                geometry_reference=_decode_geometry_refs(item["geometry"])[0],
            )
            for item in record["point_representation_refs"]
        ),
    )


def _encode_temporal_state(state: EntityTemporalState) -> dict[str, Any]:
    return {
        "first_seen": state.first_seen.to_record(),
        "last_seen": state.last_seen.to_record(),
        "physical_observation_count": state.physical_observation_count,
        "inference_result_count": state.inference_result_count,
        "observation_refs": [
            {
                "physical_observation_id": str(item.physical_observation_id),
                "acquisition_timestamp": item.acquisition_timestamp.to_record(),
                "inference_result_count": item.inference_result_count,
            }
            for item in state.observation_refs
        ],
        "provenance": {
            "rule_id": state.provenance.rule_id,
            "input_order_chronological": state.provenance.input_order_chronological,
        },
        "lifecycle": None if state.lifecycle is None else state.lifecycle.value,
    }


def _decode_temporal_state(record: Mapping[str, Any]) -> EntityTemporalState:
    provenance = record["provenance"]
    lifecycle = record["lifecycle"]
    return EntityTemporalState(
        first_seen=SourceTimestamp.from_record(record["first_seen"]),
        last_seen=SourceTimestamp.from_record(record["last_seen"]),
        physical_observation_count=record["physical_observation_count"],
        inference_result_count=record["inference_result_count"],
        observation_refs=tuple(
            ObservationRef(
                physical_observation_id=SourceObservationId(item["physical_observation_id"]),
                acquisition_timestamp=SourceTimestamp.from_record(item["acquisition_timestamp"]),
                inference_result_count=item["inference_result_count"],
            )
            for item in record["observation_refs"]
        ),
        provenance=TemporalProvenance(
            rule_id=provenance["rule_id"],
            input_order_chronological=provenance["input_order_chronological"],
        ),
        lifecycle=None if lifecycle is None else EntityLifecycle(lifecycle),
    )
