"""Canonical contracts for Semantic Mapping.

Semantic Mapping turns fused evidence into persistent semantic entities inside one semantic-map
artifact. An :class:`Entity` is the persistent record of *something in the map*: the exact 3D
support that justifies it, the semantic state it may have, the evidence chain that created it and
when it was observed. It is not a raw region, a single spatial observation, a fusion support, a
single winning label, a spatial relation or a guarantee about identity across maps.

Identity has an explicit scope. An :class:`EntityId` is unique inside one semantic map and says
nothing about any other map: the same string in two independently built maps names two different
records, and only an explicit alignment stage could ever say they are the same physical object.
That is why a stable handle is an :class:`EntityReference`, ``(semantic_map_id, entity_id)``, and
never a bare id.

Nothing here decides whether two entities are the same object: that belongs to Entity Resolution.
Everything upstream is referenced by identity and nothing is copied. See
``src/contextmap/semantic_mapping/docs/contracts.md`` for the field reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import NewType

from contextmap.semantic_mapping._checks import require_canonical, require_present
from contextmap.semantic_mapping.evidence import EntityEvidenceLinks
from contextmap.semantic_mapping.geometry import EntityGeometry
from contextmap.semantic_mapping.semantic_state import EntitySemanticState
from contextmap.semantic_mapping.temporal import EntityTemporalState

SemanticMapId = NewType("SemanticMapId", str)
"""Identity of one immutable semantic-map artifact: the scope inside which entity ids are unique."""

EntityId = NewType("EntityId", str)
"""Identity of one entity, local to its semantic map."""


class UnknownEntityError(KeyError):
    """Raised when a semantic map has no entity with the referenced id."""


class ForeignEntityReferenceError(ValueError):
    """Raised when a reference names another semantic map than the one asked to resolve it."""


@dataclass(frozen=True, kw_only=True)
class EntityReference:
    """A stable handle to one entity of one semantic map.

    Attributes:
        semantic_map_id: The semantic map that owns the entity.
        entity_id: The entity, local to that map.
    """

    semantic_map_id: SemanticMapId
    entity_id: EntityId

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "semantic_map_id", "entity_id")


@dataclass(frozen=True, kw_only=True)
class EntityProvenance:
    """How an entity was materialized.

    Attributes:
        materialization_policy_id: Versioned rule that turned fused evidence into the entity.
        identity_policy_id: Versioned rule that allocated the entity id.
        configuration_fingerprint: Hash of the materialization configuration, when configurable.
        code_version: Code revision that produced the entity, when known.
    """

    materialization_policy_id: str
    identity_policy_id: str
    configuration_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require both policy identities.

        Raises:
            ValueError: If a policy identity is empty.
        """
        require_present(self, "materialization_policy_id", "identity_policy_id")


@dataclass(frozen=True, kw_only=True)
class Entity:
    """A persistent semantic entity of one semantic map.

    Attributes:
        entity_id: Identity of the entity, local to ``semantic_map_id``.
        semantic_map_id: The semantic map that owns the entity; makes the identity scope
            explicit on the record itself.
        geometry: The persistent 3D support that justifies the entity.
        semantic_state: What the entity may be, with every hypothesis kept.
        evidence: The fused evidence the entity was materialized from.
        temporal_state: When and how often the entity was observed.
        provenance: How the entity was materialized.
    """

    entity_id: EntityId
    semantic_map_id: SemanticMapId
    geometry: EntityGeometry
    semantic_state: EntitySemanticState
    evidence: EntityEvidenceLinks
    temporal_state: EntityTemporalState
    provenance: EntityProvenance

    def __post_init__(self) -> None:
        """Validate the identity and that the semantic state can be traced to the evidence.

        Raises:
            ValueError: If an identity is empty, a hypothesis or an uncertainty record comes
                from a fused evidence the entity does not link to, the observation history and
                the evidence links list different physical observations, or a 3D representation
                is anchored outside the entity's geometry support.
        """
        require_present(self, "entity_id", "semantic_map_id")
        linked = {ref.fused_evidence_id for ref in self.evidence.fused_evidence}
        for hypothesis in self.semantic_state.hypotheses:
            if hypothesis.fused_evidence_id not in linked:
                raise ValueError(
                    f"hypothesis {hypothesis.label!r} comes from fused evidence "
                    f"{hypothesis.fused_evidence_id!r}, which the entity does not link to"
                )
        for item in self.semantic_state.uncertainty:
            if item.fused_evidence_id not in linked:
                raise ValueError(
                    f"uncertainty comes from fused evidence {item.fused_evidence_id!r}, which "
                    f"the entity does not link to"
                )
        contributed = {
            item.physical_observation_id for item in self.temporal_state.observation_refs
        }
        if contributed != set(self.evidence.physical_observation_ids):
            raise ValueError(
                "the observation history and the evidence links must list the same physical "
                "observations"
            )
        support = set(self.geometry.geometry_refs)
        for representation in self.evidence.point_representation_refs:
            if representation.geometry_reference not in support:
                raise ValueError(
                    f"point representation {representation.representation_id!r} is anchored "
                    f"outside the geometry support of the entity"
                )

    @property
    def reference(self) -> EntityReference:
        """The stable handle of this entity."""
        return EntityReference(semantic_map_id=self.semantic_map_id, entity_id=self.entity_id)


@dataclass(frozen=True, kw_only=True)
class EntitySet:
    """The entities of one semantic map, addressable by reference.

    Attributes:
        semantic_map_id: The semantic map the entities belong to.
        entities: Every entity of the map, sorted by ``entity_id`` and unique, all owned by
            ``semantic_map_id``.
    """

    semantic_map_id: SemanticMapId
    entities: tuple[Entity, ...]

    def __post_init__(self) -> None:
        """Validate that ids are unique and every entity belongs to this map.

        Raises:
            ValueError: If the map identity is empty, the entities are not sorted by id and
                unique, or one is owned by another semantic map.
        """
        require_present(self, "semantic_map_id")
        require_canonical("entities", self.entities, lambda item: (item.entity_id,))
        for entity in self.entities:
            if entity.semantic_map_id != self.semantic_map_id:
                raise ValueError(
                    f"entity {entity.entity_id!r} belongs to semantic map "
                    f"{entity.semantic_map_id!r}, not {self.semantic_map_id!r}"
                )

    @classmethod
    def of(cls, semantic_map_id: SemanticMapId, entities: Sequence[Entity]) -> EntitySet:
        """Build a set from entities in any order.

        Args:
            semantic_map_id: The semantic map the entities belong to.
            entities: The entities, in any order.

        Returns:
            The set with the entities in canonical order.

        Raises:
            ValueError: If two entities share an id or one belongs to another map.
        """
        return cls(
            semantic_map_id=semantic_map_id,
            entities=tuple(sorted(entities, key=lambda item: item.entity_id)),
        )

    def resolve(self, reference: EntityReference) -> Entity:
        """Resolve a reference to its entity.

        Args:
            reference: A reference to an entity of this map.

        Returns:
            The entity the reference names.

        Raises:
            ForeignEntityReferenceError: If the reference names another semantic map: an id is
                only meaningful inside the map that owns it.
            UnknownEntityError: If this map has no such entity.
        """
        if reference.semantic_map_id != self.semantic_map_id:
            raise ForeignEntityReferenceError(
                f"reference names semantic map {reference.semantic_map_id!r}, but this is "
                f"{self.semantic_map_id!r}"
            )
        try:
            return self._by_id[reference.entity_id]
        except KeyError:
            raise UnknownEntityError(reference.entity_id) from None

    @cached_property
    def _by_id(self) -> dict[EntityId, Entity]:
        return {entity.entity_id: entity for entity in self.entities}
