"""Cross-record invariants of a ContextMap, checked at construction.

Each function validates one relationship that no single record can check on its own: whether a
reference resolves inside the scope it names, whether an upstream identity is mapped only once,
and whether the declared capabilities agree with the content that is really there. A failure
is a :class:`ReferenceIntegrityError` or a ``ValueError`` that names the offender; nothing is
skipped, defaulted or repaired.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from contextmap.artifact._checks import require_canonical
from contextmap.artifact.composition import ContextEntity, ContextRelation
from contextmap.artifact.metadata import DeclaredCapabilities, MapCapability
from contextmap.artifact.references import (
    ContextEntityId,
    ContextMapId,
    ReferenceIntegrityError,
)
from contextmap.geometric_mapping import MapId, geometry_index_of
from contextmap.spatial_relations import RelationPredicate


def check_entities(
    entities: Sequence[ContextEntity], *, geometry_map_id: MapId, point_count: int
) -> dict[ContextEntityId, ContextEntity]:
    """Validate the entities of a map and index them by identity.

    Args:
        entities: The entities, expected sorted by id and unique.
        geometry_map_id: The geometric-map artifact the map references.
        point_count: Number of geometry elements in that artifact.

    Returns:
        The entities by id.

    Raises:
        ValueError: If the entities are not sorted and unique.
        ReferenceIntegrityError: If geometry support belongs to another map, uses a
            non-canonical identity or lies beyond the referenced map, or two entities map to
            the same upstream record.
    """
    require_canonical("entities", entities, lambda item: (item.entity_id,), detail="by id ")
    for entity in entities:
        for reference in entity.geometry_refs:
            if reference.map_id != geometry_map_id:
                raise ReferenceIntegrityError(
                    f"entity {entity.entity_id!r} is supported by geometry of map "
                    f"{reference.map_id!r}, not of the referenced map {geometry_map_id!r}"
                )
            try:
                index = geometry_index_of(map_id=geometry_map_id, geometry_id=reference.geometry_id)
            except ValueError:
                raise ReferenceIntegrityError(
                    f"entity {entity.entity_id!r} names geometry {reference.geometry_id!r}, "
                    f"which is not a geometry identity of map {geometry_map_id!r}"
                ) from None
            if index >= point_count:
                raise ReferenceIntegrityError(
                    f"entity {entity.entity_id!r} names geometry element {index}, which exceeds "
                    f"the {point_count} elements of map {geometry_map_id!r}"
                )
    _require_distinct_sources(
        [(entity.entity_id, entity.source) for entity in entities], kind="entity"
    )
    return {entity.entity_id: entity for entity in entities}


def check_relations(
    relations: Sequence[ContextRelation],
    *,
    context_map_id: ContextMapId,
    entities: dict[ContextEntityId, ContextEntity],
) -> None:
    """Validate the relations of a map against its entities.

    Args:
        relations: The relations, expected sorted by id and unique.
        context_map_id: The map that owns the relations and the entities.
        entities: The entities of the map by id.

    Raises:
        ValueError: If the relations are not sorted and unique.
        ReferenceIntegrityError: If an endpoint names another map or an entity the map does
            not have, or two relations map to the same upstream record.
    """
    require_canonical("relations", relations, lambda item: (item.relation_id,), detail="by id ")
    for relation in relations:
        for role, endpoint in (("subject", relation.subject), ("object", relation.object)):
            if endpoint.context_map_id != context_map_id:
                raise ReferenceIntegrityError(
                    f"relation {relation.relation_id!r} has its {role} in map "
                    f"{endpoint.context_map_id!r}, not in {context_map_id!r}"
                )
            if endpoint.entity_id not in entities:
                raise ReferenceIntegrityError(
                    f"relation {relation.relation_id!r} has its {role} "
                    f"{endpoint.entity_id!r}, which is not an entity of the map"
                )
    _require_distinct_sources(
        [
            (relation.relation_id, (relation.source_run_id, relation.source_relation_id))
            for relation in relations
        ],
        kind="relation",
    )


def check_declared_capabilities(
    capabilities: DeclaredCapabilities,
    *,
    entities: Sequence[ContextEntity],
    relations: Sequence[ContextRelation],
) -> None:
    """Validate that the declared capabilities agree with the content of the map.

    Content can never be hidden from the declaration. A capability may be declared while its
    content is empty: the stage ran and found nothing, which differs from a stage that never
    ran.

    Args:
        capabilities: The capabilities the metadata declares.
        entities: The entities of the map.
        relations: The relations of the map.

    Raises:
        ValueError: If entities or relations exist without being declared, or the declared
            relation types differ from the predicates present.
    """
    if entities and MapCapability.ENTITIES not in capabilities.content:
        raise ValueError(f"the map has entities but does not declare {MapCapability.ENTITIES.name}")
    if relations and MapCapability.RELATIONS not in capabilities.content:
        raise ValueError(
            f"the map has relations but does not declare {MapCapability.RELATIONS.name}"
        )
    if MapCapability.RELATIONS in capabilities.content:
        present = tuple(
            sorted({relation.predicate for relation in relations}, key=_predicate_order)
        )
        if present != capabilities.relation_predicates:
            raise ValueError(
                f"relation_predicates {[item.value for item in capabilities.relation_predicates]} "
                f"differ from the predicates present {[item.value for item in present]}"
            )


def _predicate_order(predicate: RelationPredicate) -> str:
    return predicate.value


def _require_distinct_sources(records: Sequence[tuple[str, Hashable]], *, kind: str) -> None:
    """Require that no upstream record is mapped by two records of the map.

    Args:
        records: ``(identity in the map, upstream record)`` pairs.
        kind: What the records are, for the error message.

    Raises:
        ReferenceIntegrityError: If two records share an upstream record, which would make the
            identity mapping ambiguous.
    """
    seen: dict[Hashable, str] = {}
    for identity, source in records:
        if source in seen:
            raise ReferenceIntegrityError(
                f"{kind} {identity!r} and {seen[source]!r} map to the same upstream record "
                f"{source!r}"
            )
        seen[source] = identity
