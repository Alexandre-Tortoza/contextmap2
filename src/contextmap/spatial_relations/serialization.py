"""JSON-friendly encoding of the Spatial Relations contracts.

Records contain only JSON primitives, so a persisted relation is readable without ROS, NumPy or any
model runtime. Decoding rebuilds the contracts through their constructors, so every invariant is
revalidated and a tampered record is refused rather than trusted. Nothing here copies an entity or
its geometry: entities are named by reference and geometry by identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.entity_resolution import (
    decode_resolved_entity_reference,
    encode_resolved_entity_reference,
)
from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId
from contextmap.spatial_relations.evidence import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    MeasuredGeometry,
    Quantity,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceId,
    RelationEvidenceProvenance,
    RelationEvidenceStatus,
)
from contextmap.spatial_relations.models import (
    Relation,
    RelationId,
    RelationProvenance,
    RelationState,
    RelationUncertainty,
    RelationUncertaintyKind,
)
from contextmap.spatial_relations.taxonomy import RelationPredicate


def _field(record: Mapping[str, Any], name: str) -> Any:
    """Read a required field, naming it when the record does not have it."""
    try:
        return record[name]
    except KeyError:
        raise ValueError(f"record is missing the field {name!r}") from None


def encode_relation(relation: Relation) -> dict[str, Any]:
    """Encode a relation with its state, evidence trace and uncertainty."""
    return {
        "relation_id": str(relation.relation_id),
        "subject_entity_ref": encode_resolved_entity_reference(relation.subject_entity_ref),
        "predicate": relation.predicate.value,
        "object_entity_ref": encode_resolved_entity_reference(relation.object_entity_ref),
        "state": relation.state.value,
        "relation_evidence_refs": [str(item) for item in relation.relation_evidence_refs],
        "uncertainty": [
            {
                "kind": item.kind.value,
                "detail": item.detail,
                "evidence_refs": [str(reference) for reference in item.evidence_refs],
            }
            for item in relation.uncertainty
        ],
        "derived_from": None if relation.derived_from is None else str(relation.derived_from),
        "provenance": {
            "taxonomy_version": relation.provenance.taxonomy_version,
            "decision_policy_id": relation.provenance.decision_policy_id,
            "configuration_fingerprint": relation.provenance.configuration_fingerprint,
            "code_version": relation.provenance.code_version,
        },
    }


def decode_relation(record: Mapping[str, Any]) -> Relation:
    """Decode a relation and revalidate every invariant of its contract.

    Args:
        record: The output of :func:`encode_relation`.

    Returns:
        The relation.

    Raises:
        ValueError: If a field is missing or the record violates the contract.
    """
    provenance = _field(record, "provenance")
    derived_from = _field(record, "derived_from")
    return Relation(
        relation_id=RelationId(_field(record, "relation_id")),
        subject_entity_ref=decode_resolved_entity_reference(_field(record, "subject_entity_ref")),
        predicate=RelationPredicate(_field(record, "predicate")),
        object_entity_ref=decode_resolved_entity_reference(_field(record, "object_entity_ref")),
        state=RelationState(_field(record, "state")),
        relation_evidence_refs=tuple(
            RelationEvidenceId(item) for item in _field(record, "relation_evidence_refs")
        ),
        uncertainty=tuple(
            RelationUncertainty(
                kind=RelationUncertaintyKind(_field(item, "kind")),
                detail=_field(item, "detail"),
                evidence_refs=tuple(
                    RelationEvidenceId(reference) for reference in _field(item, "evidence_refs")
                ),
            )
            for item in _field(record, "uncertainty")
        ),
        derived_from=None if derived_from is None else RelationId(derived_from),
        provenance=RelationProvenance(
            taxonomy_version=_field(provenance, "taxonomy_version"),
            decision_policy_id=_field(provenance, "decision_policy_id"),
            configuration_fingerprint=_field(provenance, "configuration_fingerprint"),
            code_version=_field(provenance, "code_version"),
        ),
    )


def encode_relation_evidence(evidence: RelationEvidence) -> dict[str, Any]:
    """Encode a piece of evidence with its measurements, geometry and provenance."""
    provenance = evidence.provenance
    return {
        "evidence_id": str(evidence.evidence_id),
        "channel": evidence.channel.value,
        "subject_entity_ref": encode_resolved_entity_reference(evidence.subject_entity_ref),
        "predicate": evidence.predicate.value,
        "object_entity_ref": encode_resolved_entity_reference(evidence.object_entity_ref),
        "status": evidence.status.value,
        "measurements": [_encode_quantity(item) for item in evidence.measurements],
        "thresholds": [_encode_quantity(item) for item in evidence.thresholds],
        "geometry": [
            {
                "role": item.role,
                "point_count": item.point_count,
                "geometry_digest": item.geometry_digest,
                "geometry_refs": [
                    {"map_id": str(ref.map_id), "geometry_id": str(ref.geometry_id)}
                    for ref in item.geometry_refs
                ],
            }
            for item in evidence.geometry
        ],
        "caveats": [{"kind": item.kind.value, "detail": item.detail} for item in evidence.caveats],
        "provenance": {
            "rule_id": provenance.rule_id,
            "configuration_fingerprint": provenance.configuration_fingerprint,
            "taxonomy_version": provenance.taxonomy_version,
            "map_frame": provenance.map_frame,
            "geometric_map_id": (
                None if provenance.geometric_map_id is None else str(provenance.geometric_map_id)
            ),
            "frame_conventions_fingerprint": provenance.frame_conventions_fingerprint,
            "code_version": provenance.code_version,
        },
    }


def decode_relation_evidence(record: Mapping[str, Any]) -> RelationEvidence:
    """Decode a piece of evidence and revalidate every invariant of its contract.

    Args:
        record: The output of :func:`encode_relation_evidence`.

    Returns:
        The evidence.

    Raises:
        ValueError: If a field is missing or the record violates the contract.
    """
    provenance = _field(record, "provenance")
    map_id = _field(provenance, "geometric_map_id")
    return RelationEvidence(
        evidence_id=RelationEvidenceId(_field(record, "evidence_id")),
        channel=RelationEvidenceChannel(_field(record, "channel")),
        subject_entity_ref=decode_resolved_entity_reference(_field(record, "subject_entity_ref")),
        predicate=RelationPredicate(_field(record, "predicate")),
        object_entity_ref=decode_resolved_entity_reference(_field(record, "object_entity_ref")),
        status=RelationEvidenceStatus(_field(record, "status")),
        measurements=tuple(_decode_quantity(item) for item in _field(record, "measurements")),
        thresholds=tuple(_decode_quantity(item) for item in _field(record, "thresholds")),
        geometry=tuple(
            MeasuredGeometry(
                role=_field(item, "role"),
                point_count=_field(item, "point_count"),
                geometry_digest=_field(item, "geometry_digest"),
                geometry_refs=tuple(
                    GeometryReference(
                        map_id=MapId(_field(ref, "map_id")),
                        geometry_id=GeometryId(_field(ref, "geometry_id")),
                    )
                    for ref in _field(item, "geometry_refs")
                ),
            )
            for item in _field(record, "geometry")
        ),
        caveats=tuple(
            EvidenceCaveat(
                kind=EvidenceCaveatKind(_field(item, "kind")), detail=_field(item, "detail")
            )
            for item in _field(record, "caveats")
        ),
        provenance=RelationEvidenceProvenance(
            rule_id=_field(provenance, "rule_id"),
            configuration_fingerprint=_field(provenance, "configuration_fingerprint"),
            taxonomy_version=_field(provenance, "taxonomy_version"),
            map_frame=_field(provenance, "map_frame"),
            geometric_map_id=None if map_id is None else MapId(map_id),
            frame_conventions_fingerprint=_field(provenance, "frame_conventions_fingerprint"),
            code_version=_field(provenance, "code_version"),
        ),
    )


def _encode_quantity(quantity: Quantity) -> dict[str, Any]:
    return {"name": quantity.name, "value": quantity.value, "unit": quantity.unit}


def _decode_quantity(record: Mapping[str, Any]) -> Quantity:
    return Quantity(
        name=_field(record, "name"),
        value=float(_field(record, "value")),
        unit=_field(record, "unit"),
    )
