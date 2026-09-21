"""JSON-friendly encoding of the Spatial Relations contracts.

Records contain only JSON primitives, so a persisted relation is readable without ROS, NumPy or any
model runtime. Decoding rebuilds the contracts through their constructors, so every invariant is
revalidated and a tampered record is refused rather than trusted. Nothing here copies an entity or
its geometry: entities are named by reference and geometry by identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from contextmap.entity_resolution import (
    ResolvedEntityReference,
    decode_resolved_entity_reference,
    encode_resolved_entity_reference,
)
from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId
from contextmap.spatial_relations.candidates import (
    CandidateExclusion,
    CandidateExclusionReason,
    CandidateProvenance,
    CandidateReason,
    RelationCandidate,
    RelationCandidateSet,
    SkippedPredicate,
)
from contextmap.spatial_relations.decision import DecisionRule, EvidenceUse, RelationDecision
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
from contextmap.spatial_relations.statements import (
    EndpointLink,
    ObservationRelationStatement,
    StatementPolarity,
    UpstreamStatementRef,
)
from contextmap.spatial_relations.taxonomy import FrameRequirement, RelationPredicate


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
        "statements": [_encode_statement(item) for item in evidence.statements],
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
        statements=tuple(_decode_statement(item) for item in _field(record, "statements")),
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


def encode_relation_decision(decision: RelationDecision) -> dict[str, Any]:
    """Encode the decision behind a relation: the rule, the evidence that decided and the rest."""
    return {
        "relation_id": str(decision.relation_id),
        "rule": decision.rule.value,
        "deciding_evidence_refs": [str(item) for item in decision.deciding_evidence_refs],
        "ignored": [
            {
                "evidence_id": str(item.evidence_id),
                "channel": item.channel.value,
                "status": item.status.value,
                "contradicts_relation": item.contradicts_relation,
            }
            for item in decision.ignored
        ],
        "detail": decision.detail,
    }


def decode_relation_decision(record: Mapping[str, Any]) -> RelationDecision:
    """Decode a decision and revalidate every invariant of its contract.

    Args:
        record: The output of :func:`encode_relation_decision`.

    Returns:
        The decision.

    Raises:
        ValueError: If a field is missing or the record violates the contract.
    """
    return RelationDecision(
        relation_id=RelationId(_field(record, "relation_id")),
        rule=DecisionRule(_field(record, "rule")),
        deciding_evidence_refs=tuple(
            RelationEvidenceId(item) for item in _field(record, "deciding_evidence_refs")
        ),
        ignored=tuple(
            EvidenceUse(
                evidence_id=RelationEvidenceId(_field(item, "evidence_id")),
                channel=RelationEvidenceChannel(_field(item, "channel")),
                status=RelationEvidenceStatus(_field(item, "status")),
                contradicts_relation=_field(item, "contradicts_relation"),
            )
            for item in _field(record, "ignored")
        ),
        detail=_field(record, "detail"),
    )


def _encode_link(link: EndpointLink) -> dict[str, Any]:
    return {
        "upstream_ref": link.upstream_ref,
        "entity_ref": encode_resolved_entity_reference(link.entity_ref),
        "linked_through": link.linked_through,
    }


def _decode_link(record: Mapping[str, Any]) -> EndpointLink:
    return EndpointLink(
        upstream_ref=_field(record, "upstream_ref"),
        entity_ref=decode_resolved_entity_reference(_field(record, "entity_ref")),
        linked_through=_field(record, "linked_through"),
    )


def _encode_statement(statement: ObservationRelationStatement) -> dict[str, Any]:
    return {
        "source_run_id": statement.source.source_run_id,
        "statement_id": statement.source.statement_id,
        "physical_observation_id": statement.source.physical_observation_id,
        "producer": statement.source.producer,
        "subject": _encode_link(statement.subject),
        "predicate_text": statement.predicate_text,
        "object": _encode_link(statement.object),
        "polarity": statement.polarity.value,
    }


def _decode_statement(record: Mapping[str, Any]) -> ObservationRelationStatement:
    return ObservationRelationStatement(
        source=UpstreamStatementRef(
            source_run_id=_field(record, "source_run_id"),
            statement_id=_field(record, "statement_id"),
            physical_observation_id=_field(record, "physical_observation_id"),
            producer=_field(record, "producer"),
        ),
        subject=_decode_link(_field(record, "subject")),
        predicate_text=_field(record, "predicate_text"),
        object=_decode_link(_field(record, "object")),
        polarity=StatementPolarity(_field(record, "polarity")),
    )


def encode_candidate_set(candidates: RelationCandidateSet) -> list[dict[str, Any]]:
    """Encode a candidate set as records: one ``set`` record, then its members.

    The first record carries the counts and the provenance, so the set can be rebuilt from the
    records alone; the others are one per candidate, exclusion and skipped predicate.
    """
    provenance = candidates.provenance
    rows: list[dict[str, Any]] = [
        {
            "record": "set",
            "entity_count": candidates.entity_count,
            "pairs_not_enumerated": candidates.pairs_not_enumerated,
            "provenance": {
                "policy_id": provenance.policy_id,
                "configuration_fingerprint": provenance.configuration_fingerprint,
                "taxonomy_version": provenance.taxonomy_version,
                "map_frame": provenance.map_frame,
                "geometric_map_id": (
                    None
                    if provenance.geometric_map_id is None
                    else str(provenance.geometric_map_id)
                ),
                "frame_conventions_fingerprint": provenance.frame_conventions_fingerprint,
            },
        }
    ]
    for item in candidates.candidates:
        rows.append(
            {
                "record": "candidate",
                **_encode_pair(item.subject_entity_ref, item.predicate, item.object_entity_ref),
                "reasons": [reason.value for reason in item.reasons],
                "bounds_gap_m": item.bounds_gap_m,
            }
        )
    for exclusion in candidates.exclusions:
        rows.append(
            {
                "record": "exclusion",
                **_encode_pair(
                    exclusion.subject_entity_ref, exclusion.predicate, exclusion.object_entity_ref
                ),
                "reason": exclusion.reason.value,
                "bounds_gap_m": exclusion.bounds_gap_m,
            }
        )
    for skipped in candidates.skipped_predicates:
        rows.append(
            {
                "record": "skipped_predicate",
                "predicate": skipped.predicate.value,
                "requirement": skipped.requirement.value,
                "detail": skipped.detail,
            }
        )
    return rows


def decode_candidate_set(rows: Sequence[Mapping[str, Any]]) -> RelationCandidateSet:
    """Decode the records of :func:`encode_candidate_set` and revalidate the set.

    Raises:
        ValueError: If the records do not hold exactly one ``set`` record, a record is of an
            unknown kind or a field is missing, or the set violates its contract.
    """
    header = [row for row in rows if row.get("record") == "set"]
    if len(header) != 1:
        raise ValueError("candidate records need exactly one 'set' record")
    provenance = _field(header[0], "provenance")
    map_id = _field(provenance, "geometric_map_id")
    candidates: list[RelationCandidate] = []
    exclusions: list[CandidateExclusion] = []
    skipped: list[SkippedPredicate] = []
    for row in rows:
        kind = _field(row, "record")
        if kind == "set":
            continue
        if kind == "candidate":
            subject, predicate, obj = _decode_pair(row)
            candidates.append(
                RelationCandidate(
                    subject_entity_ref=subject,
                    predicate=predicate,
                    object_entity_ref=obj,
                    reasons=tuple(CandidateReason(item) for item in _field(row, "reasons")),
                    bounds_gap_m=float(_field(row, "bounds_gap_m")),
                )
            )
        elif kind == "exclusion":
            subject, predicate, obj = _decode_pair(row)
            exclusions.append(
                CandidateExclusion(
                    subject_entity_ref=subject,
                    predicate=predicate,
                    object_entity_ref=obj,
                    reason=CandidateExclusionReason(_field(row, "reason")),
                    bounds_gap_m=float(_field(row, "bounds_gap_m")),
                )
            )
        elif kind == "skipped_predicate":
            skipped.append(
                SkippedPredicate(
                    predicate=RelationPredicate(_field(row, "predicate")),
                    requirement=FrameRequirement(_field(row, "requirement")),
                    detail=_field(row, "detail"),
                )
            )
        else:
            raise ValueError(f"unknown candidate record kind {kind!r}")
    return RelationCandidateSet(
        candidates=tuple(candidates),
        exclusions=tuple(exclusions),
        skipped_predicates=tuple(skipped),
        entity_count=_field(header[0], "entity_count"),
        pairs_not_enumerated=_field(header[0], "pairs_not_enumerated"),
        provenance=CandidateProvenance(
            policy_id=_field(provenance, "policy_id"),
            configuration_fingerprint=_field(provenance, "configuration_fingerprint"),
            taxonomy_version=_field(provenance, "taxonomy_version"),
            map_frame=_field(provenance, "map_frame"),
            geometric_map_id=None if map_id is None else MapId(map_id),
            frame_conventions_fingerprint=_field(provenance, "frame_conventions_fingerprint"),
        ),
    )


def _encode_pair(
    subject: ResolvedEntityReference,
    predicate: RelationPredicate,
    obj: ResolvedEntityReference,
) -> dict[str, Any]:
    return {
        "subject_entity_ref": encode_resolved_entity_reference(subject),
        "predicate": predicate.value,
        "object_entity_ref": encode_resolved_entity_reference(obj),
    }


def _decode_pair(
    record: Mapping[str, Any],
) -> tuple[ResolvedEntityReference, RelationPredicate, ResolvedEntityReference]:
    return (
        decode_resolved_entity_reference(_field(record, "subject_entity_ref")),
        RelationPredicate(_field(record, "predicate")),
        decode_resolved_entity_reference(_field(record, "object_entity_ref")),
    )
