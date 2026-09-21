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
    GeometryReference,
    MapId,
    geometry_id_for,
    geometry_index_of,
)
from contextmap.ingestion import FrameId
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
from contextmap.semantic_mapping.evidence import EntityEvidenceLinks, FusedEvidenceRef
from contextmap.semantic_mapping.geometry import EntityGeometry
from contextmap.semantic_mapping.models import (
    Entity,
    EntityId,
    EntityProvenance,
    EntityReference,
    SemanticMapId,
)
from contextmap.semantic_mapping.semantic_state import EntityHypothesis, EntitySemanticState
from contextmap.semantic_mapping.temporal import EntityTemporalState
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import BackendProvenance, ClaimId, HypothesisRole


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
    return {"map_frame": str(geometry.map_frame), **_encode_geometry_refs(geometry.geometry_refs)}


def _decode_geometry(record: Mapping[str, Any]) -> EntityGeometry:
    return EntityGeometry(
        geometry_refs=_decode_geometry_refs(record), map_frame=FrameId(record["map_frame"])
    )


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
    return {"hypotheses": [_encode_hypothesis(item) for item in state.hypotheses]}


def _decode_semantic_state(record: Mapping[str, Any]) -> EntitySemanticState:
    return EntitySemanticState(
        hypotheses=tuple(_decode_hypothesis(item) for item in record["hypotheses"])
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
        "signals": [
            {
                "kind": signal.kind.value,
                "producer": _encode_producer(signal.producer),
                "value": signal.value,
            }
            for signal in item.signals
        ],
    }


def _decode_hypothesis_evidence(record: Mapping[str, Any]) -> HypothesisEvidence:
    return HypothesisEvidence(
        contribution_id=EvidenceContributionId(record["contribution_id"]),
        claim_id=ClaimId(record["claim_id"]),
        stance=EvidenceStance(record["stance"]),
        role=HypothesisRole(record["role"]),
        signals=tuple(
            SupportSignal(
                kind=SupportSignalKind(signal["kind"]),
                producer=_decode_producer(signal["producer"]),
                value=signal["value"],
            )
            for signal in record["signals"]
        ),
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
                "fused_evidence_id": str(ref.fused_evidence_id),
                "fusion_support_id": str(ref.fusion_support_id),
            }
            for ref in evidence.fused_evidence
        ]
    }


def _decode_evidence(record: Mapping[str, Any]) -> EntityEvidenceLinks:
    return EntityEvidenceLinks(
        fused_evidence=tuple(
            FusedEvidenceRef(
                fusion_run_id=SemanticFusionRunId(item["fusion_run_id"]),
                fused_evidence_id=FusedEvidenceId(item["fused_evidence_id"]),
                fusion_support_id=FusionSupportId(item["fusion_support_id"]),
            )
            for item in record["fused_evidence"]
        )
    )


def _encode_temporal_state(state: EntityTemporalState) -> dict[str, Any]:
    return {
        "first_seen": state.first_seen.to_record(),
        "last_seen": state.last_seen.to_record(),
        "physical_observation_count": state.physical_observation_count,
        "inference_result_count": state.inference_result_count,
    }


def _decode_temporal_state(record: Mapping[str, Any]) -> EntityTemporalState:
    return EntityTemporalState(
        first_seen=SourceTimestamp.from_record(record["first_seen"]),
        last_seen=SourceTimestamp.from_record(record["last_seen"]),
        physical_observation_count=record["physical_observation_count"],
        inference_result_count=record["inference_result_count"],
    )
