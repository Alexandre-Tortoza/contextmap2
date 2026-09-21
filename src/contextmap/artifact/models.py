"""Canonical contract of the ContextMap.

A :class:`ContextMap` is the public data product of Solution 1: one immutable, versioned
description of a mapped place that connects persistent geometry with what was recognized in it.
It is a schema, nothing more. It does not search, interpret language, plan, navigate, draw or
talk to ROS, and it does not say where or how it is stored: identity is never a path.

Geometry is *referenced*, never embedded: the authoritative points stay in the immutable
geometric-map artifact the map names, so millions of coordinates are not copied into a second
structure. Entities and relations are composed by reference to that geometry and to each other,
and every reference is validated when the map is built. See
``src/contextmap/artifact/docs/contracts.md`` and ``composition.md`` for the field reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from contextmap.artifact._checks import require_artifact_identity
from contextmap.artifact._invariants import (
    check_declared_capabilities,
    check_entities,
    check_relations,
)
from contextmap.artifact.composition import ContextEntity, ContextRelation
from contextmap.artifact.metadata import ContextMapMetadata
from contextmap.artifact.references import (
    ContextEntityId,
    ContextEntityReference,
    ContextMapId,
    ForeignContextEntityReferenceError,
    UnknownContextEntityError,
)
from contextmap.artifact.versioning import require_supported_schema_version
from contextmap.geometric_mapping import MapId


@dataclass(frozen=True, kw_only=True)
class GeometricMapLink:
    """The authoritative geometry of a map, referenced by identity.

    Attributes:
        map_id: The immutable geometric-map artifact that owns every geometry element the map
            refers to.
        point_count: Number of geometry elements in that artifact, so a reference can be checked
            against the range that exists without opening the geometry.
    """

    map_id: MapId
    point_count: int

    def __post_init__(self) -> None:
        """Validate the identity and the size.

        Raises:
            ValueError: If the identity is blank or a path, or ``point_count`` is not positive.
        """
        require_artifact_identity(self, "map_id")
        if self.point_count < 1:
            raise ValueError(f"point_count must be at least 1, got {self.point_count}")


@dataclass(frozen=True, kw_only=True)
class ContextMap:
    """The final contextual map, as a schema.

    Attributes:
        context_map_id: Identity of the map; unique per immutable artifact, never a path.
        schema_version: Version of the data semantics this map was written under.
        metadata: What the map is, where it came from and how it was created.
        geometry_ref: The geometric-map artifact that owns the geometry.
        entities: The resolved entities, sorted by id and unique; may be empty when the entities
            capability is not declared, or declared and empty.
        relations: The relations between entities, sorted by id and unique.
    """

    context_map_id: ContextMapId
    schema_version: str
    metadata: ContextMapMetadata
    geometry_ref: GeometricMapLink
    entities: tuple[ContextEntity, ...]
    relations: tuple[ContextRelation, ...]

    def __post_init__(self) -> None:
        """Validate the identity, the version and every reference inside the map.

        Raises:
            ValueError: If the identity is blank or a path, a collection is not canonical, or
                the declared capabilities disagree with the content.
            UnsupportedSchemaVersionError: If the schema version is malformed or unreadable.
            ReferenceIntegrityError: If a geometry, entity or upstream reference does not
                resolve inside the scope it names.
        """
        require_artifact_identity(self, "context_map_id")
        require_supported_schema_version(self.schema_version)
        entities = check_entities(
            self.entities,
            geometry_map_id=self.geometry_ref.map_id,
            point_count=self.geometry_ref.point_count,
        )
        check_relations(self.relations, context_map_id=self.context_map_id, entities=entities)
        check_declared_capabilities(
            self.metadata.capabilities, entities=self.entities, relations=self.relations
        )

    def entity(self, reference: ContextEntityReference) -> ContextEntity:
        """Resolve a reference to its entity.

        Args:
            reference: A reference to an entity of this map.

        Returns:
            The entity the reference names.

        Raises:
            ForeignContextEntityReferenceError: If the reference names another map: an id is
                only meaningful inside the map that owns it.
            UnknownContextEntityError: If this map has no such entity.
        """
        if reference.context_map_id != self.context_map_id:
            raise ForeignContextEntityReferenceError(
                f"reference names map {reference.context_map_id!r}, but this is "
                f"{self.context_map_id!r}"
            )
        try:
            return self._entities_by_id[reference.entity_id]
        except KeyError:
            raise UnknownContextEntityError(reference.entity_id) from None

    def relations_for(self, reference: ContextEntityReference) -> tuple[ContextRelation, ...]:
        """List the relations in which an entity is the subject or the object.

        Args:
            reference: A reference to an entity of this map.

        Returns:
            The relations of the entity in relation-id order; empty when it has none.

        Raises:
            ForeignContextEntityReferenceError: If the reference names another map.
            UnknownContextEntityError: If this map has no such entity.
        """
        entity = self.entity(reference)
        return self._relations_by_entity.get(entity.entity_id, ())

    @cached_property
    def _entities_by_id(self) -> dict[ContextEntityId, ContextEntity]:
        """Index of the entities; derived, never authoritative."""
        return {entity.entity_id: entity for entity in self.entities}

    @cached_property
    def _relations_by_entity(self) -> dict[ContextEntityId, tuple[ContextRelation, ...]]:
        """Index of the relations by endpoint; derived, never authoritative."""
        grouped: dict[ContextEntityId, list[ContextRelation]] = {}
        for relation in self.relations:
            for endpoint in (relation.subject, relation.object):
                grouped.setdefault(endpoint.entity_id, []).append(relation)
        return {entity_id: tuple(items) for entity_id, items in grouped.items()}
