"""Errors of the ContextMapArtifact on-disk format.

Every failure to write, open or verify an artifact is an explicit, typed error: an artifact that
cannot be trusted is never partially interpreted.
"""

from __future__ import annotations


class ContextMapArtifactError(Exception):
    """Base class for ContextMapArtifact read, write and validation failures."""


class ManifestError(ContextMapArtifactError):
    """Raised when a manifest is malformed, inconsistent or not a ContextMapArtifact manifest."""


class UnsupportedFormatVersionError(ManifestError):
    """Raised when the format version of a manifest is not one this code understands."""
