"""Version of the ContextMap schema.

The schema version describes the *semantics of the data*: which fields exist, what they mean,
which invariants hold. It is not the version of the Python package that produced the map and it
is not the version of any serializer, so all three can change independently. The compatibility
rules are documented in ``src/contextmap/artifact/docs/versioning.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTEXT_MAP_SCHEMA_VERSION = "0.1.0"
"""Schema version this code reads and writes, as ``MAJOR.MINOR.PATCH``."""

_VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a schema version is malformed or this code cannot read it."""


@dataclass(frozen=True, order=True)
class SchemaVersion:
    """A ``MAJOR.MINOR.PATCH`` schema version.

    Attributes:
        major: Incremented by any breaking change to the data semantics.
        minor: Incremented by a backward-compatible addition.
        patch: Incremented by a clarification that changes neither data nor validation.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> SchemaVersion:
        """Parse a canonical ``MAJOR.MINOR.PATCH`` string.

        Args:
            text: The version text; no prefix, pre-release tag, padding or spaces are accepted.

        Returns:
            The parsed version.

        Raises:
            UnsupportedSchemaVersionError: If ``text`` is not a canonical version.
        """
        matched = _VERSION_PATTERN.fullmatch(text)
        if matched is None:
            raise UnsupportedSchemaVersionError(
                f"malformed schema version {text!r}: expected MAJOR.MINOR.PATCH"
            )
        major, minor, patch = (int(part) for part in matched.groups())
        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        """Return the canonical ``MAJOR.MINOR.PATCH`` text."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_readable_by(self, reader: SchemaVersion) -> bool:
        """Check whether a reader at version ``reader`` may interpret data at this version.

        The rule is deliberately narrow and never guesses across a breaking change. While the
        schema is in its validation phase (``major == 0``, see ``docs/versioning.md``) a MINOR
        bump may be breaking, so only the same ``MAJOR.MINOR`` is readable. From ``1.0.0`` on,
        any version of the same major is readable: additions of a newer minor are optional by
        rule and an older reader ignores what it does not know.

        Args:
            reader: The schema version the reading code implements.

        Returns:
            ``True`` when the data can be interpreted without reinterpreting any field.
        """
        if self.major != reader.major:
            return False
        if self.major == 0:
            return self.minor == reader.minor
        return True


def require_supported_schema_version(text: str) -> SchemaVersion:
    """Parse ``text`` and require that this code can read it.

    Args:
        text: The schema version recorded in a map.

    Returns:
        The parsed version.

    Raises:
        UnsupportedSchemaVersionError: If ``text`` is malformed or names a version this code
            cannot interpret; nothing is read partially.
    """
    version = SchemaVersion.parse(text)
    reader = SchemaVersion.parse(CONTEXT_MAP_SCHEMA_VERSION)
    if not version.is_readable_by(reader):
        raise UnsupportedSchemaVersionError(
            f"unsupported schema version {text}: this reader implements {reader}"
        )
    return version
