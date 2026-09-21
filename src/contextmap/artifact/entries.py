"""The entity and relation records a ContextMapArtifact stores.

An entry is a plain, JSON-compatible record with the identity the map uses for it. The artifact
keeps the record verbatim and does not interpret it; the key and the relation endpoints are the
only parts of an entry the artifact itself relies on, to index, traverse and check references.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class EntityEntry:
    """One entity of the map, as the artifact stores it.

    Attributes:
        key: Identity of the entity inside this map, unique and non-empty. It is what relations
            and readers refer to; the writer never rewrites it.
        record: The entity's canonical, JSON-compatible record. The artifact keeps it verbatim
            and does not interpret it.
    """

    key: str
    record: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class RelationEntry:
    """One relation of the map, as the artifact stores it.

    Attributes:
        key: Identity of the relation inside this map, unique and non-empty.
        subject_key: Key of the entity that is the subject; it must exist in the map.
        object_key: Key of the entity that is the object; it must exist in the map.
        record: The relation's canonical, JSON-compatible record, kept verbatim.
    """

    key: str
    subject_key: str
    object_key: str
    record: Mapping[str, Any]
