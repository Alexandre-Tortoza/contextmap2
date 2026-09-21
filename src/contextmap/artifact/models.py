"""Canonical contract of the ContextMap.

A :class:`ContextMap` is the public data product of Solution 1: one immutable, versioned
description of a mapped place that connects persistent geometry with what was recognized in it.
It is a schema, nothing more. It does not search, interpret language, plan, navigate, draw or
talk to ROS, and it does not say where or how it is stored: identity is never a path.

Geometry is *referenced*, never embedded: the authoritative points stay in the immutable
geometric-map artifact the map names, so millions of coordinates are not copied into a second
structure. See ``src/contextmap/artifact/docs/contracts.md`` for the field reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from contextmap.artifact._checks import require_artifact_identity
from contextmap.artifact.metadata import ContextMapMetadata
from contextmap.artifact.versioning import require_supported_schema_version
from contextmap.geometric_mapping import MapId

ContextMapId = NewType("ContextMapId", str)
"""Identity of one immutable ContextMap artifact; never a filesystem path."""


@dataclass(frozen=True, kw_only=True)
class GeometricMapLink:
    """The authoritative geometry of a map, referenced by identity.

    Attributes:
        map_id: The immutable geometric-map artifact that owns every geometry element the map
            refers to.
        point_count: Number of geometry elements in that artifact, so a reference can be checked
            against the range that exists without opening the geometry.
    """

    map_id: MapId
    point_count: int

    def __post_init__(self) -> None:
        """Validate the identity and the size.

        Raises:
            ValueError: If the identity is blank or a path, or ``point_count`` is not positive.
        """
        require_artifact_identity(self, "map_id")
        if self.point_count < 1:
            raise ValueError(f"point_count must be at least 1, got {self.point_count}")


@dataclass(frozen=True, kw_only=True)
class ContextMap:
    """The final contextual map, as a schema.

    Attributes:
        context_map_id: Identity of the map; unique per immutable artifact, never a path.
        schema_version: Version of the data semantics this map was written under.
        metadata: What the map is, where it came from and how it was created.
        geometry_ref: The geometric-map artifact that owns the geometry.
    """

    context_map_id: ContextMapId
    schema_version: str
    metadata: ContextMapMetadata
    geometry_ref: GeometricMapLink

    def __post_init__(self) -> None:
        """Validate the identity and that this code can read the schema version.

        Raises:
            ValueError: If the identity is blank or a path.
            UnsupportedSchemaVersionError: If the schema version is malformed or unreadable.
        """
        require_artifact_identity(self, "context_map_id")
        require_supported_schema_version(self.schema_version)
