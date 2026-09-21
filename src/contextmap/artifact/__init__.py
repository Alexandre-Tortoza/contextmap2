"""Public contract for the artifact capability.

The artifact capability owns the final public product of Solution 1: the :class:`ContextMap`
schema, its metadata and the rules that keep it interpretable. It composes what other
capabilities produced (geometry, entities, relations) by reference and adds nothing scientific
of its own. It is a schema only: it is independent of the filesystem layout, of any serializer,
of ROS and of model stacks. See ``src/contextmap/artifact/docs/README.md`` for the full
capability documentation.
"""

from contextmap.artifact.composition import (
    AmbiguityStatus,
    ContextEntity,
    ContextRelation,
    ContextSemanticState,
    LabelHypothesis,
    RelationState,
)
from contextmap.artifact.frame import AnchorKind, Handedness, LengthUnit, MapAnchor, MapFrame
from contextmap.artifact.metadata import (
    ContextMapMetadata,
    DeclaredCapabilities,
    MapCapability,
    MapCreation,
    ObservationWindow,
    PolicyRef,
    SourceSequence,
)
from contextmap.artifact.models import ContextMap, GeometricMapLink
from contextmap.artifact.records import (
    ContextMapRecordError,
    context_map_from_record,
    context_map_to_record,
)
from contextmap.artifact.references import (
    ContextEntityId,
    ContextEntityReference,
    ContextMapId,
    ContextRelationId,
    ForeignContextEntityReferenceError,
    ReferenceIntegrityError,
    UnknownContextEntityError,
    UpstreamRecordRef,
)
from contextmap.artifact.versioning import (
    CONTEXT_MAP_SCHEMA_VERSION,
    SchemaVersion,
    UnsupportedSchemaVersionError,
    require_supported_schema_version,
)

__all__ = [
    "CONTEXT_MAP_SCHEMA_VERSION",
    "AmbiguityStatus",
    "AnchorKind",
    "ContextEntity",
    "ContextEntityId",
    "ContextEntityReference",
    "ContextMap",
    "ContextMapId",
    "ContextMapMetadata",
    "ContextMapRecordError",
    "ContextRelation",
    "ContextRelationId",
    "ContextSemanticState",
    "DeclaredCapabilities",
    "ForeignContextEntityReferenceError",
    "GeometricMapLink",
    "Handedness",
    "LabelHypothesis",
    "LengthUnit",
    "MapAnchor",
    "MapCapability",
    "MapCreation",
    "MapFrame",
    "ObservationWindow",
    "PolicyRef",
    "ReferenceIntegrityError",
    "RelationState",
    "SchemaVersion",
    "SourceSequence",
    "UnknownContextEntityError",
    "UnsupportedSchemaVersionError",
    "UpstreamRecordRef",
    "context_map_from_record",
    "context_map_to_record",
    "require_supported_schema_version",
]
