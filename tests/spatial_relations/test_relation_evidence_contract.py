"""RelationEvidence: measurements about one candidate, never a relation by themselves."""

from __future__ import annotations

import dataclasses

import pytest
from relation_builders import (
    MAP_ID,
    entity_ref,
    evidence_provenance,
    make_evidence,
    measured_geometry,
)

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.spatial_relations import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    MeasuredGeometry,
    Quantity,
    RelationEvidenceChannel,
    RelationEvidenceStatus,
    RelationPredicate,
    evidence_id_for,
)


def _caveat(kind: EvidenceCaveatKind = EvidenceCaveatKind.WITHIN_TOLERANCE) -> EvidenceCaveat:
    return EvidenceCaveat(kind=kind, detail="the gap is within the tolerance of the threshold")


def test_a_supporting_geometric_record_keeps_measurements_thresholds_and_geometry() -> None:
    evidence = make_evidence()
    assert evidence.status is RelationEvidenceStatus.SUPPORTS
    assert evidence.channel is RelationEvidenceChannel.GEOMETRY
    assert [item.name for item in evidence.measurements] == ["bounds_gap"]
    assert [item.name for item in evidence.thresholds] == ["next_to_max_gap"]
    assert {item.role for item in evidence.geometry} == {"subject", "object"}


def test_the_candidate_is_part_of_the_record() -> None:
    evidence = make_evidence(subject=3, obj=9, predicate=RelationPredicate.ABOVE)
    assert evidence.subject_entity_ref == entity_ref(3)
    assert evidence.object_entity_ref == entity_ref(9)
    assert evidence.predicate is RelationPredicate.ABOVE


def test_records_are_immutable() -> None:
    evidence = make_evidence()
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.status = RelationEvidenceStatus.CONFLICTS  # type: ignore[misc]


def test_ambiguous_and_unavailable_records_must_say_why() -> None:
    with pytest.raises(ValueError, match="caveat"):
        make_evidence(status=RelationEvidenceStatus.AMBIGUOUS)
    with pytest.raises(ValueError, match="caveat"):
        make_evidence(status=RelationEvidenceStatus.UNAVAILABLE, measurements=())
    ambiguous = make_evidence(status=RelationEvidenceStatus.AMBIGUOUS, caveats=(_caveat(),))
    assert ambiguous.caveats[0].kind is EvidenceCaveatKind.WITHIN_TOLERANCE


def test_unavailable_evidence_may_have_no_measurements() -> None:
    evidence = make_evidence(
        status=RelationEvidenceStatus.UNAVAILABLE,
        measurements=(),
        caveats=(_caveat(EvidenceCaveatKind.MISSING_INPUT),),
    )
    assert evidence.measurements == ()


def test_a_decisive_record_needs_measurements() -> None:
    for status in (RelationEvidenceStatus.SUPPORTS, RelationEvidenceStatus.CONFLICTS):
        with pytest.raises(ValueError, match="measurements"):
            make_evidence(status=status, measurements=())


def test_a_geometric_record_needs_the_geometry_it_measured() -> None:
    with pytest.raises(ValueError, match="geometry"):
        make_evidence(geometry=())


def test_a_geometric_record_needs_the_map_scope_of_its_coordinates() -> None:
    with pytest.raises(ValueError, match="map"):
        make_evidence(provenance=evidence_provenance(with_scope=False))


def test_an_entity_is_never_related_to_itself() -> None:
    with pytest.raises(ValueError, match="itself"):
        make_evidence(subject=4, obj=4)


def test_both_entities_must_come_from_one_resolution_artifact() -> None:
    subject = entity_ref(1)
    foreign = entity_ref(2, run="resolution-run-0002")
    with pytest.raises(ValueError, match="resolution"):
        dataclasses.replace(make_evidence(), subject_entity_ref=subject, object_entity_ref=foreign)


def test_evidence_is_recorded_for_the_evaluated_direction_only() -> None:
    with pytest.raises(ValueError, match="derived"):
        make_evidence(predicate=RelationPredicate.BELOW)


def test_measurements_and_thresholds_are_canonical() -> None:
    gap = Quantity(name="bounds_gap", value=0.1, unit="m")
    other = Quantity(name="overlap", value=0.2, unit="ratio")
    with pytest.raises(ValueError, match="sorted"):
        make_evidence(measurements=(other, gap))
    with pytest.raises(ValueError, match="sorted"):
        make_evidence(measurements=(gap, gap))
    with pytest.raises(ValueError, match="sorted"):
        make_evidence(thresholds=(other, gap))


def test_caveats_are_canonical() -> None:
    first = _caveat(EvidenceCaveatKind.WITHIN_TOLERANCE)
    second = _caveat(EvidenceCaveatKind.UNRELIABLE_GEOMETRY)
    ordered = sorted((first, second), key=lambda item: (item.kind.value, item.detail))
    make_evidence(status=RelationEvidenceStatus.AMBIGUOUS, caveats=tuple(ordered))
    with pytest.raises(ValueError, match="sorted"):
        make_evidence(status=RelationEvidenceStatus.AMBIGUOUS, caveats=tuple(reversed(ordered)))


def test_a_quantity_is_a_finite_number_with_a_unit() -> None:
    Quantity(name="bounds_gap", value=0.0, unit="m")
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            Quantity(name="bounds_gap", value=value, unit="m")
    with pytest.raises(ValueError, match="unit"):
        Quantity(name="bounds_gap", value=1.0, unit=" ")
    with pytest.raises(ValueError, match="name"):
        Quantity(name="", value=1.0, unit="m")


def test_a_caveat_needs_an_explanation() -> None:
    with pytest.raises(ValueError, match="detail"):
        EvidenceCaveat(kind=EvidenceCaveatKind.MISSING_INPUT, detail="")


def test_measured_geometry_says_which_geometry_was_used() -> None:
    whole = measured_geometry("subject")
    assert whole.geometry_refs == ()
    reference = GeometryReference(
        map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=3)
    )
    subset = MeasuredGeometry(
        role="subject_contact_band",
        point_count=1,
        geometry_digest="sha256:cc",
        geometry_refs=(reference,),
    )
    assert subset.geometry_refs == (reference,)


def test_measured_geometry_invariants() -> None:
    reference = GeometryReference(
        map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=3)
    )
    other = GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=4))
    with pytest.raises(ValueError, match="role"):
        MeasuredGeometry(role="", point_count=1, geometry_digest="sha256:cc")
    with pytest.raises(ValueError, match="digest"):
        MeasuredGeometry(role="subject", point_count=1, geometry_digest="")
    with pytest.raises(ValueError, match="point_count"):
        MeasuredGeometry(role="subject", point_count=0, geometry_digest="sha256:cc")
    with pytest.raises(ValueError, match="point_count"):
        MeasuredGeometry(
            role="subject",
            point_count=3,
            geometry_digest="sha256:cc",
            geometry_refs=(reference,),
        )
    with pytest.raises(ValueError, match="sorted"):
        MeasuredGeometry(
            role="subject",
            point_count=2,
            geometry_digest="sha256:cc",
            geometry_refs=(other, reference),
        )


def test_geometry_roles_are_unique_within_a_record() -> None:
    with pytest.raises(ValueError, match="role"):
        make_evidence(geometry=(measured_geometry("subject"), measured_geometry("subject")))


def test_evidence_identity_is_deterministic_and_directional() -> None:
    def ident(
        subject: int,
        obj: int,
        predicate: RelationPredicate = RelationPredicate.NEXT_TO,
        channel: RelationEvidenceChannel = RelationEvidenceChannel.GEOMETRY,
    ) -> str:
        return str(
            evidence_id_for(
                channel=channel,
                subject_entity_ref=entity_ref(subject),
                predicate=predicate,
                object_entity_ref=entity_ref(obj),
            )
        )

    assert ident(1, 2) == ident(1, 2)
    assert len({ident(1, 2), ident(2, 1)}) == 2
    assert ident(1, 2) != ident(1, 2, RelationPredicate.ABOVE)
    assert ident(1, 2) != ident(1, 2, channel=RelationEvidenceChannel.CONTACT)
    assert "geometry" in ident(1, 2)
    assert "next_to" in ident(1, 2)


def test_a_tampered_status_cannot_hide_an_unexplained_ambiguity() -> None:
    supported = make_evidence()
    with pytest.raises(ValueError, match="caveat"):
        dataclasses.replace(supported, status=RelationEvidenceStatus.AMBIGUOUS)
