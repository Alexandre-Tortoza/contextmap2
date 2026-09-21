"""Errors of the ContextMapArtifact on-disk format.

Every failure to write, open or verify an artifact is an explicit, typed error: an artifact that
cannot be trusted is never partially interpreted.
"""

from __future__ import annotations

from contextmap.artifact.versioning import UnsupportedSchemaVersionError


class ContextMapArtifactError(Exception):
    """Base class for ContextMapArtifact read, write and validation failures."""


class ManifestError(ContextMapArtifactError):
    """Raised when a manifest is malformed, inconsistent or not a ContextMapArtifact manifest."""


class UnsupportedFormatVersionError(ManifestError):
    """Raised when the format version of a manifest is not one this code understands."""


class MissingPayloadError(ContextMapArtifactError):
    """Raised when a file the artifact promises is not on disk."""


class RecordTableError(ContextMapArtifactError):
    """Raised when a record table cannot be encoded or its lines cannot be trusted."""


class BrokenIndexError(RecordTableError):
    """Raised when an index disagrees with the payload it claims to describe."""


class RecordNotFoundError(RecordTableError):
    """Raised when a table has no record with the requested key."""


class IncompleteContextMapArtifactError(ContextMapArtifactError):
    """Raised when a directory is not a complete artifact: not a directory, or no manifest.

    The manifest is written last, at publication, so an interrupted write never has one.
    """


class ArtifactIntegrityError(ContextMapArtifactError):
    """Raised when the bytes of an artifact do not match what its manifest promises."""


class UnsupportedArtifactSchemaError(ContextMapArtifactError, UnsupportedSchemaVersionError):
    """Raised when an artifact was written under a schema version this code cannot read.

    It is both an artifact error and the schema's own unsupported-version error, so a caller may
    catch either.
    """


class MissingDependencyError(ContextMapArtifactError):
    """Raised when an upstream artifact the map needs cannot be found."""


class DependencyMismatchError(ContextMapArtifactError):
    """Raised when what was found is not the upstream artifact the manifest recorded."""


class UnresolvedReferenceError(ContextMapArtifactError):
    """Raised when a reference does not resolve inside the artifact."""


class BundleError(ContextMapArtifactError):
    """Raised when a bundle cannot be exported, or a directory is not a readable bundle."""


class ArtifactExistsError(ContextMapArtifactError):
    """Raised when a write would land on an artifact that already exists.

    A finished artifact is immutable; writing again creates a new artifact elsewhere.
    """


class InvalidContentError(ContextMapArtifactError):
    """Raised when the content handed to the writer is inconsistent, so nothing is published."""


class UpstreamArtifactError(ContextMapArtifactError):
    """Raised when an upstream artifact the map refers to is missing, damaged or inconsistent."""
