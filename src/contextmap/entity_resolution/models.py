"""Identity contracts of Entity Resolution.

Entity Resolution decides whether entity records of a semantic map describe one physical object,
distinct objects, or cannot be told apart, and materializes the entities those decisions justify.
Its public product is an immutable resolution artifact; this module owns the identities that
artifact hands out.

Identity has an explicit scope, exactly like in Semantic Mapping. A :class:`ResolvedEntityId` is
unique inside one resolution artifact, named by an :class:`EntityResolutionRunId`, and says nothing
about any other artifact: rerunning the resolution creates a new artifact and its ids are new
identities, so permanence across independently rebuilt maps is never implied. The stable handle
is therefore a :class:`ResolvedEntityReference`, ``(resolution_run_id, resolved_entity_id)``, and
never a bare id.

A resolved entity never replaces its members: the source entities keep their own
:class:`~contextmap.semantic_mapping.EntityReference` and stay individually addressable.
See ``src/contextmap/entity_resolution/docs/contracts.md`` for the field reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from contextmap.entity_resolution._checks import require_present

EntityResolutionRunId = NewType("EntityResolutionRunId", str)
"""Identity of one immutable resolution artifact: the scope inside which resolved ids are unique."""

ResolvedEntityId = NewType("ResolvedEntityId", str)
"""Identity of one resolved entity, local to its resolution artifact."""


@dataclass(frozen=True, kw_only=True)
class ResolvedEntityReference:
    """A stable handle to one resolved entity of one resolution artifact.

    It mirrors :class:`~contextmap.semantic_mapping.EntityReference`: the id alone is only
    meaningful inside the artifact that allocated it.

    Attributes:
        resolution_run_id: The resolution artifact that owns the resolved entity.
        resolved_entity_id: The resolved entity, local to that artifact.
    """

    resolution_run_id: EntityResolutionRunId
    resolved_entity_id: ResolvedEntityId

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "resolution_run_id", "resolved_entity_id")
