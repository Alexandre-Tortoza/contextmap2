"""Identities and references of the ContextMap, and the errors raised when one does not resolve.

Every identity here has an explicit scope. A :class:`ContextEntityId` is unique inside one
ContextMap and says nothing about any other map: the same text in two maps names two different
records, and nothing is inferred from it. That is why a stable handle to an entity is a
:class:`ContextEntityReference`, ``(context_map_id, entity_id)``, and never a bare id.

Records the map takes from other artifacts keep their own identity. An
:class:`UpstreamRecordRef` names the exact record in the exact upstream artifact, so the
mapping between an identity in the map and an identity upstream is always explicit and never a
silent rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from contextmap.artifact._checks import require_artifact_identity, require_present

ContextMapId = NewType("ContextMapId", str)
"""Identity of one immutable ContextMap artifact; never a filesystem path."""

ContextEntityId = NewType("ContextEntityId", str)
"""Identity of one entity, unique inside its ContextMap."""

ContextRelationId = NewType("ContextRelationId", str)
"""Identity of one relation, unique inside its ContextMap."""


class ReferenceIntegrityError(ValueError):
    """Raised when a reference cannot be resolved inside the scope it names."""


class ForeignContextEntityReferenceError(ValueError):
    """Raised when a reference names another ContextMap than the one asked to resolve it."""


class UnknownContextEntityError(KeyError):
    """Raised when a ContextMap has no entity with the referenced id."""


@dataclass(frozen=True, kw_only=True, order=True)
class UpstreamRecordRef:
    """The exact record, in the exact upstream artifact, that something in the map comes from.

    Attributes:
        artifact_id: Identity of the immutable upstream artifact; never a path.
        record_id: Identity of the record, local to that artifact.
    """

    artifact_id: str
    record_id: str

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is blank or the artifact identity is a filesystem path.
        """
        require_artifact_identity(self, "artifact_id")
        require_present(self, "record_id")


@dataclass(frozen=True, kw_only=True, order=True)
class ContextEntityReference:
    """A stable handle to one entity of one ContextMap.

    Attributes:
        context_map_id: The map that owns the entity.
        entity_id: The entity, local to that map.
    """

    context_map_id: ContextMapId
    entity_id: ContextEntityId

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is blank.
        """
        require_present(self, "context_map_id", "entity_id")
