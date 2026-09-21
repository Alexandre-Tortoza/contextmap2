"""Public contract for the artifact capability.

The artifact capability owns the final public product of Solution 1: the :class:`ContextMap`
schema, its metadata and the rules that keep it interpretable, and the portable on-disk
ContextMapArtifact that persists it. It composes what other capabilities produced (geometry,
entities, relations) by reference and adds nothing scientific of its own. The schema is
independent of the filesystem layout, of any serializer, of ROS and of model stacks; the layout
and the storage formats are decided apart from it. See ``src/contextmap/artifact/docs/README.md``
for the full capability documentation and ``src/contextmap/artifact/docs/storage-layout.md`` for
the layout.
"""

from contextmap.artifact.dependencies import UpstreamArtifact
from contextmap.artifact.errors import (
    ArtifactExistsError,
    ContextMapArtifactError,
    InvalidContentError,
    ManifestError,
    RecordTableError,
    UnsupportedFormatVersionError,
    UpstreamArtifactError,
)
from contextmap.artifact.frame import AnchorKind, Handedness, LengthUnit, MapAnchor, MapFrame
from contextmap.artifact.layout import ARTIFACT_TYPE, FORMAT_VERSION
from contextmap.artifact.manifest import (
    ContextMapArtifactManifest,
    DependencyRecord,
    Requirement,
    inventory_digest,
)
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
from contextmap.artifact.writer import ContextMapArtifactWriter, EntityEntry, RelationEntry

__all__ = [
    "ARTIFACT_TYPE",
    "CONTEXT_MAP_SCHEMA_VERSION",
    "FORMAT_VERSION",
    "AnchorKind",
    "ArtifactExistsError",
    "ContextMap",
    "ContextMapArtifactError",
    "ContextMapArtifactManifest",
    "ContextMapArtifactWriter",
    "ContextMapId",
    "ContextMapMetadata",
    "ContextMapRecordError",
    "DeclaredCapabilities",
    "DependencyRecord",
    "EntityEntry",
    "GeometricMapLink",
    "Handedness",
    "InvalidContentError",
    "LengthUnit",
    "ManifestError",
    "MapAnchor",
    "MapCapability",
    "MapCreation",
    "MapFrame",
    "ObservationWindow",
    "PolicyRef",
    "RecordTableError",
    "RelationEntry",
    "Requirement",
    "SchemaVersion",
    "SourceSequence",
    "UnsupportedFormatVersionError",
    "UnsupportedSchemaVersionError",
    "UpstreamArtifact",
    "UpstreamArtifactError",
    "context_map_from_record",
    "context_map_to_record",
    "inventory_digest",
    "require_supported_schema_version",
]
