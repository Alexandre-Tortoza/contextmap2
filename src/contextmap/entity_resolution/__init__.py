"""Public contract for the entity resolution capability.

Entity Resolution decides, from typed and separately preserved evidence, whether semantic entities
describe the same physical object, distinct objects, or cannot be told apart, and materializes the
resolved entities that explicit decisions justify inside one immutable resolution artifact. It
never mutates a source entity and never turns a single score into an identity. See
``src/contextmap/entity_resolution/docs/README.md`` for the full capability documentation.
"""

from contextmap.entity_resolution.models import (
    EntityResolutionRunId,
    ResolvedEntityId,
    ResolvedEntityReference,
)
from contextmap.entity_resolution.serialization import (
    decode_resolved_entity_reference,
    encode_resolved_entity_reference,
)

__all__ = [
    "EntityResolutionRunId",
    "ResolvedEntityId",
    "ResolvedEntityReference",
    "decode_resolved_entity_reference",
    "encode_resolved_entity_reference",
]
