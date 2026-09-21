"""Public contract for the artifact capability: the on-disk ContextMapArtifact.

The artifact capability owns the final product of Solution 1, the ContextMap and the
ContextMapArtifact that persists it as a portable, versioned and auditable directory. This
package surface exposes the manifest of that directory and its errors; see
``src/contextmap/artifact/docs/storage-layout.md`` for the layout and the storage-format
decisions.
"""

from contextmap.artifact.errors import (
    ContextMapArtifactError,
    ManifestError,
    UnsupportedFormatVersionError,
)
from contextmap.artifact.layout import ARTIFACT_TYPE, FORMAT_VERSION
from contextmap.artifact.manifest import (
    ColumnPayload,
    ContextMapArtifactManifest,
    DependencyRecord,
    Payload,
    PayloadRole,
    RecordPayload,
    Requirement,
    inventory_digest,
)

__all__ = [
    "ARTIFACT_TYPE",
    "FORMAT_VERSION",
    "ColumnPayload",
    "ContextMapArtifactError",
    "ContextMapArtifactManifest",
    "DependencyRecord",
    "ManifestError",
    "Payload",
    "PayloadRole",
    "RecordPayload",
    "Requirement",
    "UnsupportedFormatVersionError",
    "inventory_digest",
]
