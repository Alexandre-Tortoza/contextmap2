"""Public contract for the artifact capability.

The artifact capability owns the final public product of Solution 1: the :class:`ContextMap`
schema, its metadata and the rules that keep it interpretable, and the portable on-disk
ContextMapArtifact that persists it. It composes what other capabilities produced (geometry,
entities, relations) by reference and adds nothing scientific of its own. The schema is
independent of the filesystem layout, of any serializer, of ROS and of model stacks; the layout,
the storage formats, the writer, the reader, the validator and the bundle export are decided
apart from it. See ``src/contextmap/artifact/docs/README.md`` for the full capability
documentation and ``src/contextmap/artifact/docs/storage-layout.md`` for the layout.
"""

from contextmap.artifact.composition import (
    AmbiguityStatus,
    ContextEntity,
    ContextRelation,
    ContextSemanticState,
    LabelHypothesis,
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
from contextmap.artifact.provenance import (
    ArtifactKind,
    DerivationKind,
    EvidenceOrigin,
    ProvenanceError,
    UpstreamArtifact,
)
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
from contextmap.artifact.schema_identity import describe_schema, schema_fingerprint
from contextmap.artifact.serialization.bundle import (
    BundleManifest,
    ClosurePolicy,
    EmbeddedDependency,
    OmittedDependency,
    export_bundle,
    verify_bundle,
)
from contextmap.artifact.serialization.dependencies import artifact_digest
from contextmap.artifact.serialization.errors import (
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
from contextmap.artifact.serialization.layout import ARTIFACT_TYPE, FORMAT_VERSION
from contextmap.artifact.serialization.manifest import (
    ContextMapArtifactManifest,
    DependencyRecord,
    Requirement,
)
from contextmap.artifact.serialization.reader import ContextMapArtifactReader
from contextmap.artifact.serialization.validation import (
    VALIDATOR_VERSION,
    CheckOutcome,
    FileStatus,
    Severity,
    ValidationLevel,
    ValidationReport,
    ValidationStatus,
    validate_context_map_artifact,
)
from contextmap.artifact.serialization.writer import ContextMapArtifactWriter
from contextmap.artifact.versioning import (
    CONTEXT_MAP_SCHEMA_VERSION,
    SchemaVersion,
    UnsupportedSchemaVersionError,
    require_supported_schema_version,
)

__all__ = [
    "ARTIFACT_TYPE",
    "CONTEXT_MAP_SCHEMA_VERSION",
    "FORMAT_VERSION",
    "VALIDATOR_VERSION",
    "AmbiguityStatus",
    "AnchorKind",
    "ArtifactExistsError",
    "ArtifactIntegrityError",
    "ArtifactKind",
    "BundleError",
    "BundleManifest",
    "CheckOutcome",
    "ClosurePolicy",
    "ContextEntity",
    "ContextEntityId",
    "ContextEntityReference",
    "ContextMap",
    "ContextMapArtifactError",
    "ContextMapArtifactManifest",
    "ContextMapArtifactReader",
    "ContextMapArtifactWriter",
    "ContextMapId",
    "ContextMapMetadata",
    "ContextMapRecordError",
    "ContextRelation",
    "ContextRelationId",
    "ContextSemanticState",
    "DeclaredCapabilities",
    "DependencyMismatchError",
    "DependencyRecord",
    "DerivationKind",
    "EmbeddedDependency",
    "EvidenceOrigin",
    "FileStatus",
    "ForeignContextEntityReferenceError",
    "GeometricMapLink",
    "Handedness",
    "IncompleteContextMapArtifactError",
    "InvalidContentError",
    "LabelHypothesis",
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
    "ProvenanceError",
    "RecordTableError",
    "ReferenceIntegrityError",
    "Requirement",
    "SchemaVersion",
    "Severity",
    "SourceSequence",
    "UnknownContextEntityError",
    "UnresolvedReferenceError",
    "UnsupportedArtifactSchemaError",
    "UnsupportedFormatVersionError",
    "UnsupportedSchemaVersionError",
    "UpstreamArtifact",
    "UpstreamArtifactError",
    "UpstreamRecordRef",
    "ValidationLevel",
    "ValidationReport",
    "ValidationStatus",
    "artifact_digest",
    "context_map_from_record",
    "context_map_to_record",
    "describe_schema",
    "export_bundle",
    "require_supported_schema_version",
    "schema_fingerprint",
    "validate_context_map_artifact",
    "verify_bundle",
]
