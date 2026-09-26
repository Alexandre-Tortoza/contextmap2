"""Relations and their evidence serialize as plain JSON and are revalidated on the way back."""

from __future__ import annotations

import json

import pytest
from relation_builders import (
    MAP_ID,
    entity_ref,
    make_evidence,
    make_relation,
    make_uncertainty,
    measured_geometry,
)
from relation_run_fixture import CANDIDATES, CONVENTIONS
from relation_storeroom_fixture import exclusion_ceiling, storeroom_scene

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.spatial_relations import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    MeasuredGeometry,
    RelationCandidateSet,
    RelationEvidenceChannel,
    RelationEvidenceStatus,
    RelationPredicate,
    RelationState,
    RelationUncertaintyKind,
    decode_candidate_set,
    decode_relation,
    decode_relation_evidence,
    encode_candidate_set,
    encode_relation,
    encode_relation_evidence,
    generate_relation_candidates,
)


def _round_trip_json(record: dict[str, object]) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(json.dumps(record, sort_keys=True))
    return loaded


def test_a_supported_relation_round_trips_through_json() -> None:
    relation = make_relation(subject=1, obj=2, predicate=RelationPredicate.ABOVE)
    record = _round_trip_json(encode_relation(relation))
    assert decode_relation(record) == relation


def test_an_unresolved_relation_keeps_its_uncertainty_and_derivation() -> None:
    source = make_relation(subject=1, obj=2, predicate=RelationPredicate.INSIDE)
    relation = make_relation(
        subject=2,
        obj=1,
        predicate=RelationPredicate.CONTAINS,
        state=RelationState.UNRESOLVED,
        evidence_refs=("evidence--a", "evidence--b"),
        uncertainty=(
            make_uncertainty(
                RelationUncertaintyKind.CONFLICTING_EVIDENCE,
                detail="geometry supports, contact conflicts",
                evidence_refs=("evidence--a", "evidence--b"),
            ),
        ),
        derived_from=source.relation_id,
    )
    decoded = decode_relation(_round_trip_json(encode_relation(relation)))
    assert decoded == relation
    assert decoded.uncertainty[0].evidence_refs == relation.relation_evidence_refs


def test_the_relation_record_states_the_entities_by_reference_only() -> None:
    record = encode_relation(make_relation())
    assert record["subject_entity_ref"] == {
        "resolution_run_id": "resolution-run-0001",
        "resolved_entity_id": "resolved-0001",
    }
    assert record["object_entity_ref"] == {
        "resolution_run_id": "resolution-run-0001",
        "resolved_entity_id": "resolved-0002",
    }
    assert record["predicate"] == "next_to"
    assert record["state"] == "supported"


def test_a_geometric_record_round_trips_with_every_measurement_and_geometry_input() -> None:
    reference = GeometryReference(
        map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=7)
    )
    evidence = make_evidence(
        predicate=RelationPredicate.ABOVE,
        geometry=(
            measured_geometry("object", digest="sha256:bb"),
            MeasuredGeometry(
                role="subject_contact_band",
                point_count=1,
                geometry_digest="sha256:cc",
                geometry_refs=(reference,),
            ),
        ),
    )
    decoded = decode_relation_evidence(_round_trip_json(encode_relation_evidence(evidence)))
    assert decoded == evidence
    assert decoded.geometry[1].geometry_refs == (reference,)
    assert decoded.measurements[0].value == 0.1


def test_an_ambiguous_record_keeps_its_caveats() -> None:
    evidence = make_evidence(
        status=RelationEvidenceStatus.AMBIGUOUS,
        channel=RelationEvidenceChannel.CONTACT,
        predicate=RelationPredicate.TOUCHING,
        caveats=(
            EvidenceCaveat(
                kind=EvidenceCaveatKind.UNRELIABLE_GEOMETRY,
                detail="the subject support is sparse",
            ),
        ),
    )
    decoded = decode_relation_evidence(_round_trip_json(encode_relation_evidence(evidence)))
    assert decoded == evidence
    assert decoded.caveats[0].kind is EvidenceCaveatKind.UNRELIABLE_GEOMETRY


def test_the_evidence_record_carries_the_provenance_of_the_measurement() -> None:
    record = encode_relation_evidence(make_evidence())
    provenance = record["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["rule_id"] == "bounds-next-to-v1"
    assert provenance["map_frame"] == "map"
    assert provenance["geometric_map_id"] == "map-0001"
    assert provenance["taxonomy_version"] == "spatial-relation-taxonomy-v1"
    assert record["subject_entity_ref"] == {
        "resolution_run_id": entity_ref(1).resolution_run_id,
        "resolved_entity_id": entity_ref(1).resolved_entity_id,
    }


def test_the_encoding_is_deterministic() -> None:
    first = json.dumps(encode_relation_evidence(make_evidence()), sort_keys=True)
    second = json.dumps(encode_relation_evidence(make_evidence()), sort_keys=True)
    assert first == second


def test_a_tampered_relation_is_refused_rather_than_trusted() -> None:
    record = _round_trip_json(encode_relation(make_relation()))
    record["relation_evidence_refs"] = []
    with pytest.raises(ValueError, match="evidence"):
        decode_relation(record)


def test_an_unknown_predicate_is_refused() -> None:
    record = _round_trip_json(encode_relation(make_relation()))
    record["predicate"] = "near"
    with pytest.raises(ValueError):
        decode_relation(record)


def test_a_record_missing_a_field_names_it() -> None:
    record = _round_trip_json(encode_relation(make_relation()))
    del record["state"]
    with pytest.raises(ValueError, match="state"):
        decode_relation(record)


def test_a_tampered_evidence_record_is_refused() -> None:
    record = _round_trip_json(encode_relation_evidence(make_evidence()))
    record["status"] = "ambiguous"
    with pytest.raises(ValueError, match="caveat"):
        decode_relation_evidence(record)


# --- candidate sets past the exclusion ceiling (#601) ---


def _capped_storeroom(monkeypatch: pytest.MonkeyPatch) -> RelationCandidateSet:
    _, _, entities = storeroom_scene()
    exclusion_ceiling(monkeypatch, 4)
    return generate_relation_candidates(entities, policy=CANDIDATES, conventions=CONVENTIONS)


def test_a_candidate_set_with_summarized_exclusions_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SR-01: o registro de exclusões não tinha como carregar os grupos resumidos.
    capped = _capped_storeroom(monkeypatch)
    records = [_round_trip_json(row) for row in encode_candidate_set(capped)]

    summaries = [row for row in records if row["record"] == "exclusion_summary"]
    assert len(summaries) == len(capped.exclusion_summaries) > 0
    assert set(summaries[0]) == {
        "record",
        "predicate",
        "reason",
        "count",
        "min_gap_m",
        "max_gap_m",
        "digest",
        "nearest",
    }
    assert decode_candidate_set(records) == capped


def test_a_tampered_exclusion_summary_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _round_trip_json(row) for row in encode_candidate_set(_capped_storeroom(monkeypatch))
    ]
    summary = next(row for row in records if row["record"] == "exclusion_summary")
    summary["count"] = len(summary["nearest"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="listed"):
        decode_candidate_set(records)
