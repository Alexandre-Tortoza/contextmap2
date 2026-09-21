"""Deterministic builders of relation contracts for Spatial Relations tests."""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.entity_resolution import (
    EntityResolutionRunId,
    ResolvedEntityId,
    ResolvedEntityReference,
)
from contextmap.geometric_mapping import MapId
from contextmap.spatial_relations import (
    TAXONOMY_VERSION,
    EvidenceCaveat,
    MeasuredGeometry,
    Quantity,
    Relation,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceId,
    RelationEvidenceProvenance,
    RelationEvidenceStatus,
    RelationId,
    RelationPredicate,
    RelationProvenance,
    RelationState,
    RelationUncertainty,
    RelationUncertaintyKind,
    evidence_id_for,
    relation_id_for,
)

RESOLUTION_RUN_ID = EntityResolutionRunId("resolution-run-0001")
MAP_ID = MapId("map-0001")


def entity_ref(number: int, *, run: str = "resolution-run-0001") -> ResolvedEntityReference:
    return ResolvedEntityReference(
        resolution_run_id=EntityResolutionRunId(run),
        resolved_entity_id=ResolvedEntityId(f"resolved-{number:04d}"),
    )


def measured_geometry(role: str = "subject", *, digest: str = "sha256:aa") -> MeasuredGeometry:
    return MeasuredGeometry(role=role, point_count=27, geometry_digest=digest)


def evidence_provenance(
    *, rule_id: str = "bounds-next-to-v1", with_scope: bool = True
) -> RelationEvidenceProvenance:
    return RelationEvidenceProvenance(
        rule_id=rule_id,
        configuration_fingerprint="sha256:policy",
        taxonomy_version=TAXONOMY_VERSION,
        map_frame="map" if with_scope else None,
        geometric_map_id=MAP_ID if with_scope else None,
        frame_conventions_fingerprint=None,
    )


def make_evidence(
    *,
    subject: int = 1,
    obj: int = 2,
    predicate: RelationPredicate = RelationPredicate.NEXT_TO,
    channel: RelationEvidenceChannel = RelationEvidenceChannel.GEOMETRY,
    status: RelationEvidenceStatus = RelationEvidenceStatus.SUPPORTS,
    measurements: Sequence[Quantity] | None = None,
    thresholds: Sequence[Quantity] | None = None,
    geometry: Sequence[MeasuredGeometry] | None = None,
    caveats: Sequence[EvidenceCaveat] = (),
    provenance: RelationEvidenceProvenance | None = None,
) -> RelationEvidence:
    subject_ref = entity_ref(subject)
    object_ref = entity_ref(obj)
    return RelationEvidence(
        evidence_id=evidence_id_for(
            channel=channel,
            subject_entity_ref=subject_ref,
            predicate=predicate,
            object_entity_ref=object_ref,
        ),
        channel=channel,
        subject_entity_ref=subject_ref,
        predicate=predicate,
        object_entity_ref=object_ref,
        status=status,
        measurements=(
            (Quantity(name="bounds_gap", value=0.1, unit="m"),)
            if measurements is None
            else tuple(measurements)
        ),
        thresholds=(
            (Quantity(name="next_to_max_gap", value=0.5, unit="m"),)
            if thresholds is None
            else tuple(thresholds)
        ),
        geometry=(
            (measured_geometry("object", digest="sha256:bb"), measured_geometry("subject"))
            if geometry is None
            else tuple(geometry)
        ),
        caveats=tuple(caveats),
        provenance=provenance or evidence_provenance(),
    )


def relation_provenance() -> RelationProvenance:
    return RelationProvenance(
        taxonomy_version=TAXONOMY_VERSION,
        decision_policy_id="baseline-relation-decision-v1",
        configuration_fingerprint="sha256:decision-config",
    )


def make_relation(
    *,
    subject: int = 1,
    obj: int = 2,
    predicate: RelationPredicate = RelationPredicate.NEXT_TO,
    state: RelationState = RelationState.SUPPORTED,
    evidence_refs: Sequence[str] | None = None,
    uncertainty: Sequence[RelationUncertainty] = (),
    derived_from: RelationId | None = None,
) -> Relation:
    subject_ref = entity_ref(subject)
    object_ref = entity_ref(obj)
    refs = (
        (
            evidence_id_for(
                channel=RelationEvidenceChannel.GEOMETRY,
                subject_entity_ref=subject_ref,
                predicate=predicate,
                object_entity_ref=object_ref,
            ),
        )
        if evidence_refs is None
        else tuple(RelationEvidenceId(item) for item in evidence_refs)
    )
    return Relation(
        relation_id=relation_id_for(
            subject_entity_ref=subject_ref, predicate=predicate, object_entity_ref=object_ref
        ),
        subject_entity_ref=subject_ref,
        predicate=predicate,
        object_entity_ref=object_ref,
        state=state,
        relation_evidence_refs=refs,
        uncertainty=tuple(uncertainty),
        derived_from=derived_from,
        provenance=relation_provenance(),
    )


def make_uncertainty(
    kind: RelationUncertaintyKind = RelationUncertaintyKind.INSUFFICIENT_EVIDENCE,
    *,
    detail: str = "no channel decided",
    evidence_refs: Sequence[str] = (),
) -> RelationUncertainty:
    return RelationUncertainty(
        kind=kind,
        detail=detail,
        evidence_refs=tuple(RelationEvidenceId(item) for item in evidence_refs),
    )
