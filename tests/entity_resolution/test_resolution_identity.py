"""Identity contracts of Entity Resolution: the artifact-scoped handle of a resolved entity."""

from __future__ import annotations

import dataclasses

import pytest

from contextmap.entity_resolution import (
    EntityResolutionRunId,
    ResolvedEntityId,
    ResolvedEntityReference,
    decode_resolved_entity_reference,
    encode_resolved_entity_reference,
)
from contextmap.semantic_mapping import EntityId, EntityReference, SemanticMapId


def make_reference(
    run: str = "resolution-run-0001", resolved: str = "resolved--0001"
) -> ResolvedEntityReference:
    return ResolvedEntityReference(
        resolution_run_id=EntityResolutionRunId(run), resolved_entity_id=ResolvedEntityId(resolved)
    )


def test_a_reference_names_the_resolution_artifact_and_the_resolved_entity() -> None:
    reference = make_reference()

    assert reference.resolution_run_id == "resolution-run-0001"
    assert reference.resolved_entity_id == "resolved--0001"


@pytest.mark.parametrize("field", ["resolution_run_id", "resolved_entity_id"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_a_reference_requires_both_identities(field: str, blank: str) -> None:
    values = {"resolution_run_id": "resolution-run-0001", "resolved_entity_id": "resolved--0001"}
    values[field] = blank

    with pytest.raises(ValueError, match=field):
        ResolvedEntityReference(
            resolution_run_id=EntityResolutionRunId(values["resolution_run_id"]),
            resolved_entity_id=ResolvedEntityId(values["resolved_entity_id"]),
        )


def test_the_same_resolved_id_in_two_artifacts_is_two_different_references() -> None:
    # Um id só vale dentro do artifact que o possui: reruns não herdam identidade.
    first = make_reference(run="resolution-run-0001")
    second = make_reference(run="resolution-run-0002")

    assert first != second
    assert len({first, second}) == 2


def test_a_reference_is_immutable_and_hashable() -> None:
    reference = make_reference()

    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.resolved_entity_id = ResolvedEntityId("other")  # type: ignore[misc]
    assert {reference: 1}[make_reference()] == 1


def test_a_resolved_reference_is_never_a_source_entity_reference() -> None:
    # A entidade de origem continua endereçável pela própria referência e nunca é sobrescrita.
    source = EntityReference(
        semantic_map_id=SemanticMapId("semantic-map-0001"),
        entity_id=EntityId("entity--support-000001"),
    )

    assert make_reference() != source
    assert not hasattr(make_reference(), "semantic_map_id")


def test_a_reference_survives_a_round_trip() -> None:
    reference = make_reference()

    record = encode_resolved_entity_reference(reference)

    assert record == {
        "resolution_run_id": "resolution-run-0001",
        "resolved_entity_id": "resolved--0001",
    }
    assert decode_resolved_entity_reference(record) == reference


def test_decoding_revalidates_the_reference() -> None:
    with pytest.raises(ValueError, match="resolved_entity_id"):
        decode_resolved_entity_reference(
            {"resolution_run_id": "resolution-run-0001", "resolved_entity_id": ""}
        )


def test_decoding_names_the_missing_field() -> None:
    with pytest.raises(ValueError, match="resolution_run_id"):
        decode_resolved_entity_reference({"resolved_entity_id": "resolved--0001"})
