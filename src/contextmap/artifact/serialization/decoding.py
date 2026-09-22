"""Encoding and decoding of the parts of a ContextMap that the artifact stores one by one.

The schema writes and reads a whole map as one record (``context_map_to_record`` and
``context_map_from_record``). The artifact stores the metadata, the geometry reference, the
lineage, every entity and every relation as separate documents and lines so that one of them can
be read without the rest. This module wraps each stored record in the envelope the tables index
by, and decodes one part at a time through the schema's own strict decoder, so every invariant
of the schema is revalidated and nothing is defaulted or repaired.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.artifact.composition import ContextEntity, ContextRelation
from contextmap.artifact.metadata import ContextMapMetadata
from contextmap.artifact.models import ContextMap, GeometricMapLink
from contextmap.artifact.provenance import UpstreamArtifact
from contextmap.artifact.records import ContextMapRecordError, _decode, context_map_to_record

ENTITY_LINE_FIELDS = frozenset({"key", "record"})
RELATION_LINE_FIELDS = frozenset({"key", "subject", "object", "record"})


def entity_lines(context_map: ContextMap) -> list[dict[str, Any]]:
    """Wrap every entity record of a map in the envelope the entity table stores.

    Args:
        context_map: The map.

    Returns:
        One ``{"key", "record"}`` line per entity; the key is the entity id and the record is the
        schema's canonical record of the entity.
    """
    records = context_map_to_record(context_map)["entities"]
    return [
        {"key": str(entity.entity_id), "record": record}
        for entity, record in zip(context_map.entities, records, strict=True)
    ]


def relation_lines(context_map: ContextMap) -> list[dict[str, Any]]:
    """Wrap every relation record of a map in the envelope the relation table stores.

    Args:
        context_map: The map.

    Returns:
        One ``{"key", "subject", "object", "record"}`` line per relation; the subject and the
        object are the ids of the entities the relation connects.
    """
    records = context_map_to_record(context_map)["relations"]
    return [
        {
            "key": str(relation.relation_id),
            "subject": str(relation.subject.entity_id),
            "object": str(relation.object.entity_id),
            "record": record,
        }
        for relation, record in zip(context_map.relations, records, strict=True)
    ]


def decode_entity(line: Mapping[str, Any]) -> ContextEntity:
    """Decode the entity of a stored line.

    Raises:
        ContextMapRecordError: If the line has the wrong fields or the record is not a valid
            entity of the schema.
        ValueError: If the record violates an invariant of the schema.
    """
    if line.keys() != ENTITY_LINE_FIELDS:
        raise ContextMapRecordError(
            f"an entity line must have exactly the fields {sorted(ENTITY_LINE_FIELDS)}"
        )
    entity = _decode(ContextEntity, line["record"], f"entity {line['key']!r}")
    assert isinstance(entity, ContextEntity)
    return entity


def decode_relation(line: Mapping[str, Any]) -> ContextRelation:
    """Decode the relation of a stored line.

    Raises:
        ContextMapRecordError: If the line has the wrong fields or the record is not a valid
            relation of the schema.
        ValueError: If the record violates an invariant of the schema.
    """
    if line.keys() != RELATION_LINE_FIELDS:
        raise ContextMapRecordError(
            f"a relation line must have exactly the fields {sorted(RELATION_LINE_FIELDS)}"
        )
    relation = _decode(ContextRelation, line["record"], f"relation {line['key']!r}")
    assert isinstance(relation, ContextRelation)
    return relation


def decode_metadata(record: Mapping[str, Any]) -> ContextMapMetadata:
    """Decode the stored metadata document."""
    metadata = _decode(ContextMapMetadata, record, "metadata")
    assert isinstance(metadata, ContextMapMetadata)
    return metadata


def decode_geometry_link(record: Mapping[str, Any]) -> GeometricMapLink:
    """Decode the stored geometry reference document."""
    link = _decode(GeometricMapLink, record, "geometry_ref")
    assert isinstance(link, GeometricMapLink)
    return link


def decode_upstream_artifacts(records: Any) -> tuple[UpstreamArtifact, ...]:
    """Decode the lineage entries of the artifact.

    Raises:
        ContextMapRecordError: If ``records`` is not a list of valid lineage entries.
    """
    if not isinstance(records, list):
        raise ContextMapRecordError("the lineage must be a list of upstream artifacts")
    return tuple(
        _decode(UpstreamArtifact, item, f"lineage[{number}]") for number, item in enumerate(records)
    )
