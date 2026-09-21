"""Public contract for the artifact capability.

The artifact capability owns the final public product of Solution 1: the :class:`ContextMap`
schema, its metadata and the rules that keep it interpretable. It composes what other
capabilities produced (geometry, entities, relations) by reference and adds nothing scientific
of its own. It is a schema only: it is independent of the filesystem layout, of any serializer,
of ROS and of model stacks. See ``src/contextmap/artifact/docs/README.md`` for the full
capability documentation.
"""

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
from contextmap.artifact.models import ContextMap, ContextMapId, GeometricMapLink
from contextmap.artifact.records import (
    ContextMapRecordError,
    context_map_from_record,
    context_map_to_record,
)
from contextmap.artifact.versioning import (
    CONTEXT_MAP_SCHEMA_VERSION,
    SchemaVersion,
    UnsupportedSchemaVersionError,
    require_supported_schema_version,
)

__all__ = [
    "CONTEXT_MAP_SCHEMA_VERSION",
    "AnchorKind",
    "ContextMap",
    "ContextMapId",
    "ContextMapMetadata",
    "ContextMapRecordError",
    "DeclaredCapabilities",
    "GeometricMapLink",
    "Handedness",
    "LengthUnit",
    "MapAnchor",
    "MapCapability",
    "MapCreation",
    "MapFrame",
    "ObservationWindow",
    "PolicyRef",
    "SchemaVersion",
    "SourceSequence",
    "UnsupportedSchemaVersionError",
    "context_map_from_record",
    "context_map_to_record",
    "require_supported_schema_version",
]
