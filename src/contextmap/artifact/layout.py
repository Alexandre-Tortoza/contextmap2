"""Where every part of a ContextMapArtifact directory lives.

A ContextMapArtifact is a directory: the directory is the canonical baseline and any archive is
only a transport wrapper of it. The names below are the v0 layout; the map's schema semantics
(``ContextMap``, ``ContextMapMetadata``, entities, relations) do not depend on them, and an
artifact declares what it contains in its manifest and metadata, never through the presence of a
file. See ``src/contextmap/artifact/docs/storage-layout.md`` for the reasons behind the layout.

There is no ``debug/`` directory: debug evidence is human-only, is not part of the map contract
and is never a place a reader may look for authoritative data.
"""

from __future__ import annotations

from contextmap.artifact.errors import ManifestError

ARTIFACT_TYPE = "context_map"
"""``artifact_type`` recorded in the manifest of every ContextMapArtifact."""

FORMAT_VERSION = "0.1.0"
"""Version of the on-disk layout and encodings written by this code.

It versions how the map is stored, not what it means: the semantics carry their own
``schema_version``. The two move independently.
"""

SUPPORTED_FORMAT_VERSIONS = frozenset({FORMAT_VERSION})
"""Format versions this code opens; anything else is rejected, never partially read."""

MANIFEST = "manifest.json"
README = "README.md"
MAP_METADATA = "map-metadata.json"
GEOMETRY_REFERENCE = "geometry/geometry-reference.json"
ENTITIES = "entities/entities.jsonl"
RELATIONS = "relations/relations.jsonl"
LINEAGE = "lineage/lineage.json"
ENTITY_INDEX = "indexes/entity-index.jsonl"
RELATION_INDEX = "indexes/relation-index.jsonl"
ENTITY_RELATION_INDEX = "indexes/entity-relation-index.jsonl"

CONTRACTUAL_FILES = (
    MAP_METADATA,
    GEOMETRY_REFERENCE,
    ENTITIES,
    RELATIONS,
    LINEAGE,
    ENTITY_INDEX,
    RELATION_INDEX,
    ENTITY_RELATION_INDEX,
)
"""Files every artifact has, empty tables included: a missing file is damage, never "no content"."""

DEBUG_DIRECTORY = "debug"


def require_contractual_path(relative_path: str) -> None:
    """Check that a path may name a contractual file of an artifact.

    Args:
        relative_path: Path of the file relative to the artifact directory.

    Raises:
        ManifestError: If the path is empty, absolute, uses a backslash, contains an empty,
            ``.`` or ``..`` segment, or lies under ``debug/``.
    """
    segments = relative_path.split("/")
    if (
        not relative_path
        or "\\" in relative_path
        or "\x00" in relative_path
        or any(segment in ("", ".", "..") for segment in segments)
    ):
        raise ManifestError(
            f"expected a plain relative POSIX path inside the artifact, got {relative_path!r}"
        )
    if segments[0] == DEBUG_DIRECTORY:
        raise ManifestError(
            f"{relative_path!r} is under debug/, which is never part of the artifact contract"
        )
