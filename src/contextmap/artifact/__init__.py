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

from contextmap.artifact.bundle import (
    BundleManifest,
    ClosurePolicy,
    EmbeddedDependency,
    OmittedDependency,
    export_bundle,
    verify_bundle,
)
from contextmap.artifact.dependencies import UpstreamArtifact
from contextmap.artifact.entries import EntityEntry, RelationEntry
from contextmap.artifact.errors import (
    ArtifactExistsError,
    ArtifactIntegrityError,
    BundleError,
    ContextMapArtifactError,
    DependencyMismatchError,
    IncompleteContextMapArtifactError,
    InvalidContentError,
    ManifestError,
    MissingDependencyError,
    RecordTableError,
    UnresolvedReferenceError,
    UnsupportedArtifactSchemaError,
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
from contextmap.artifact.reader import ContextMapArtifactReader
from contextmap.artifact.records import (
    ContextMapRecordError,
    context_map_from_record,
    context_map_to_record,
)
from contextmap.artifact.validation import (
    VALIDATOR_VERSION,
    CheckOutcome,
    FileStatus,
    Severity,
    ValidationLevel,
    ValidationReport,
    ValidationStatus,
    validate_context_map_artifact,
)
from contextmap.artifact.versioning import (
    CONTEXT_MAP_SCHEMA_VERSION,
    SchemaVersion,
    UnsupportedSchemaVersionError,
    require_supported_schema_version,
)
from contextmap.artifact.writer import ContextMapArtifactWriter

__all__ = [
    "ARTIFACT_TYPE",
    "CONTEXT_MAP_SCHEMA_VERSION",
    "FORMAT_VERSION",
    "VALIDATOR_VERSION",
    "AnchorKind",
    "ArtifactExistsError",
    "ArtifactIntegrityError",
    "BundleError",
    "BundleManifest",
    "CheckOutcome",
    "ClosurePolicy",
    "ContextMap",
    "ContextMapArtifactError",
    "ContextMapArtifactManifest",
    "ContextMapArtifactReader",
    "ContextMapArtifactWriter",
    "ContextMapId",
    "ContextMapMetadata",
    "ContextMapRecordError",
    "DeclaredCapabilities",
    "DependencyMismatchError",
    "DependencyRecord",
    "EmbeddedDependency",
    "EntityEntry",
    "FileStatus",
    "GeometricMapLink",
    "Handedness",
    "IncompleteContextMapArtifactError",
    "InvalidContentError",
    "LengthUnit",
    "ManifestError",
    "MapAnchor",
    "MapCapability",
    "MapCreation",
    "MapFrame",
    "MissingDependencyError",
    "ObservationWindow",
    "OmittedDependency",
    "PolicyRef",
    "RecordTableError",
    "RelationEntry",
    "Requirement",
    "SchemaVersion",
    "Severity",
    "SourceSequence",
    "UnresolvedReferenceError",
    "UnsupportedArtifactSchemaError",
    "UnsupportedFormatVersionError",
    "UnsupportedSchemaVersionError",
    "UpstreamArtifact",
    "UpstreamArtifactError",
    "ValidationLevel",
    "ValidationReport",
    "ValidationStatus",
    "context_map_from_record",
    "context_map_to_record",
    "export_bundle",
    "inventory_digest",
    "require_supported_schema_version",
    "validate_context_map_artifact",
    "verify_bundle",
]
