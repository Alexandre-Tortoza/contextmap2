"""Deterministic, atomic writer of a ContextMapArtifact directory.

The writer turns a :class:`~contextmap.artifact.ContextMap` and its entity and relation records
into the directory described in ``src/contextmap/artifact/docs/storage-layout.md``. It validates
everything it can before touching the disk, writes into a hidden temporary sibling and publishes
with one rename (:class:`~contextmap.shared.AtomicRunDirectory`), so an interrupted write never
looks like a finished artifact and a finished artifact is never overwritten.

The output is a function of the content: records are ordered by key, every line is canonical and
the manifest carries a content identity that ignores the write time. Nothing invalid is dropped
silently: an inconsistent map is refused whole with the reason. No model, ROS or runtime object
ever reaches the files; the writer accepts only plain records.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextmap.artifact.dependencies import (
    GEOMETRIC_MAP_ARTIFACT_TYPE,
    UpstreamArtifact,
    read_inventory,
    relative_locator,
    verify_inventory,
)
from contextmap.artifact.errors import (
    ArtifactExistsError,
    ContextMapArtifactError,
    InvalidContentError,
    UpstreamArtifactError,
)
from contextmap.artifact.layout import (
    ENTITIES,
    ENTITY_INDEX,
    ENTITY_RELATION_INDEX,
    GEOMETRY_REFERENCE,
    LINEAGE,
    MANIFEST,
    MAP_METADATA,
    RELATION_INDEX,
    RELATIONS,
)
from contextmap.artifact.manifest import (
    ContextMapArtifactManifest,
    DependencyRecord,
    Payload,
    PayloadRole,
    RecordPayload,
    Requirement,
    create_manifest,
    decode_manifest,
    encode_manifest,
    inventory_digest,
)
from contextmap.artifact.metadata import MapCapability
from contextmap.artifact.models import ContextMap
from contextmap.artifact.records import context_map_to_record
from contextmap.artifact.tables import (
    document_json,
    encode_entity_relation_index,
    encode_record_table,
)
from contextmap.geometric_mapping import GeometricMapArtifactReader, MapArtifactError
from contextmap.shared import AtomicRunDirectory, RunDirectoryError, file_entry


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


class ContextMapArtifactWriter:
    """Writes one immutable ContextMapArtifact directory.

    The caller names the final directory; the writer never derives a location, allocates a run
    index or keeps a registry, so the runtime is free to place the artifact where its workspace
    layout says.
    """

    def __init__(self, *, output_dir: Path, written_at: datetime | None = None) -> None:
        """Prepare a write.

        Args:
            output_dir: The final directory of the artifact. It must not exist yet.
            written_at: The time recorded as provenance in the manifest, timezone-aware; the
                current UTC time when omitted. It is not part of the content identity.
        """
        self._output_dir = output_dir
        self._written_at = written_at if written_at is not None else datetime.now(UTC)

    def write(
        self,
        context_map: ContextMap,
        *,
        geometry_dir: Path,
        entities: Iterable[EntityEntry] = (),
        relations: Iterable[RelationEntry] = (),
        evidence: Iterable[UpstreamArtifact] = (),
    ) -> ContextMapArtifactManifest:
        """Validate the map and publish it atomically.

        Args:
            context_map: The map to persist.
            geometry_dir: The GeometricMapArtifact that owns the geometry ``context_map``
                refers to. It is referenced and verified, never copied.
            entities: The entity records, in any order.
            relations: The relation records, in any order.
            evidence: Further upstream artifacts the map refers to, required or optional. A
                geometric map is not evidence: it is ``geometry_dir``.

        Returns:
            The manifest of the published artifact.

        Raises:
            ArtifactExistsError: If ``output_dir`` already exists; a finished artifact is never
                overwritten.
            InvalidContentError: If the content is inconsistent: a relation names an unknown
                entity, content is present that the map does not declare, the geometry is not
                the one the map names, or an upstream artifact is repeated.
            RecordTableError: If a key is duplicated or a record is not plain JSON.
            UpstreamArtifactError: If an upstream artifact is missing or does not match its own
                inventory.
            ContextMapArtifactError: If the directory cannot be published.
        """
        if self._output_dir.exists():
            raise ArtifactExistsError(f"the artifact already exists: {self._output_dir}")
        entity_entries, relation_entries = tuple(entities), tuple(relations)
        _check_declared_content(context_map, entity_entries, relation_entries)
        _check_endpoints(entity_entries, relation_entries)
        dependencies = self._dependencies(context_map, geometry_dir, tuple(evidence))

        record = context_map_to_record(context_map)
        entity_table = encode_record_table(
            {"key": entry.key, "record": entry.record} for entry in entity_entries
        )
        relation_table = encode_record_table(
            {
                "key": entry.key,
                "subject": entry.subject_key,
                "object": entry.object_key,
                "record": entry.record,
            }
            for entry in relation_entries
        )
        traversal_index = encode_entity_relation_index(
            (entry.key for entry in entity_entries),
            ((rel.key, rel.subject_key, rel.object_key) for rel in relation_entries),
        )
        files: dict[str, bytes] = {
            MAP_METADATA: document_json(record["metadata"]),
            GEOMETRY_REFERENCE: document_json(record["geometry_ref"]),
            LINEAGE: document_json(
                {
                    "upstream_artifacts": [
                        {
                            "artifact_type": dependency.artifact_type,
                            "artifact_id": dependency.artifact_id,
                            "content_identity": dependency.content_identity,
                            "requirement": dependency.requirement.value,
                        }
                        for dependency in dependencies
                    ]
                }
            ),
            ENTITIES: entity_table.payload,
            ENTITY_INDEX: entity_table.index,
            RELATIONS: relation_table.payload,
            RELATION_INDEX: relation_table.index,
            ENTITY_RELATION_INDEX: traversal_index,
        }
        creation = context_map.metadata.creation
        manifest = create_manifest(
            context_map_id=str(context_map.context_map_id),
            schema_version=context_map.schema_version,
            written_at=self._written_at.isoformat(),
            code_version=creation.code_version,
            configuration_fingerprint=creation.configuration_fingerprint,
            entity_count=entity_table.record_count,
            relation_count=relation_table.record_count,
            payloads=_payloads(entity_table.record_count, relation_table.record_count),
            dependencies=dependencies,
            file_inventory=[file_entry(path, data) for path, data in files.items()],
        )

        try:
            with AtomicRunDirectory(self._output_dir) as run:
                for path, data in files.items():
                    run.write_bytes(path, data)
                run.publish(manifest=encode_manifest(manifest), readme=_render_readme(manifest))
        except RunDirectoryError as error:
            raise ContextMapArtifactError(str(error)) from error

        published = decode_manifest(
            json.loads((self._output_dir / MANIFEST).read_text(encoding="utf-8"))
        )
        if published != manifest:
            raise ContextMapArtifactError(
                "the published manifest differs from the one that was computed"
            )
        return published

    def _dependencies(
        self, context_map: ContextMap, geometry_dir: Path, evidence: tuple[UpstreamArtifact, ...]
    ) -> tuple[DependencyRecord, ...]:
        """Verify every upstream artifact and pin it by the digest of its inventory."""
        seen: set[tuple[str, str]] = set()
        for upstream in evidence:
            if upstream.artifact_type == GEOMETRIC_MAP_ARTIFACT_TYPE:
                raise InvalidContentError(
                    f"a {GEOMETRIC_MAP_ARTIFACT_TYPE} is not evidence: pass it as geometry_dir"
                )
            key = (upstream.artifact_type, upstream.artifact_id)
            if key in seen:
                raise InvalidContentError(f"duplicate upstream artifact {key!r}")
            seen.add(key)

        geometry = self._geometry_dependency(context_map, geometry_dir)
        records = [geometry]
        for upstream in evidence:
            inventory = read_inventory(upstream.location)
            verify_inventory(upstream.location, inventory)
            records.append(
                DependencyRecord(
                    artifact_type=upstream.artifact_type,
                    artifact_id=upstream.artifact_id,
                    content_identity=inventory_digest(inventory),
                    requirement=upstream.requirement,
                    locator=relative_locator(self._output_dir, upstream.location),
                )
            )
        return tuple(records)

    def _geometry_dependency(self, context_map: ContextMap, geometry_dir: Path) -> DependencyRecord:
        link = context_map.geometry_ref
        try:
            with GeometricMapArtifactReader(geometry_dir) as reader:
                manifest = reader.manifest
                problems = reader.verify_integrity(check_index=False)
        except MapArtifactError as error:
            raise UpstreamArtifactError(
                f"the geometric map at {geometry_dir.name!r} cannot be opened: {error}"
            ) from error
        if problems:
            raise UpstreamArtifactError(
                f"the geometric map {manifest.map_id!r} does not match its inventory: "
                + "; ".join(problems)
            )
        if manifest.map_id != link.map_id:
            raise InvalidContentError(
                f"the map refers to the geometric map {link.map_id!r} but geometry_dir holds "
                f"{manifest.map_id!r}"
            )
        if manifest.point_count != link.point_count:
            raise InvalidContentError(
                f"the map declares {link.point_count} geometry elements but the geometric map "
                f"{manifest.map_id!r} has {manifest.point_count}"
            )
        map_frame = context_map.metadata.frame.frame_id
        if manifest.map_frame != map_frame:
            raise InvalidContentError(
                f"the map is expressed in frame {map_frame!r} but the geometric map "
                f"{manifest.map_id!r} is in frame {manifest.map_frame!r}"
            )
        return DependencyRecord(
            artifact_type=GEOMETRIC_MAP_ARTIFACT_TYPE,
            artifact_id=str(manifest.map_id),
            content_identity=inventory_digest(manifest.file_inventory),
            requirement=Requirement.REQUIRED,
            locator=relative_locator(self._output_dir, geometry_dir),
        )


def _check_declared_content(
    context_map: ContextMap,
    entities: tuple[EntityEntry, ...],
    relations: tuple[RelationEntry, ...],
) -> None:
    """Refuse content that the map does not declare: a capability is never implied by content."""
    declared = context_map.metadata.capabilities.content
    if entities and MapCapability.ENTITIES not in declared:
        raise InvalidContentError(
            f"{len(entities)} entities were given but the map does not declare the entities "
            "capability"
        )
    if relations and MapCapability.RELATIONS not in declared:
        raise InvalidContentError(
            f"{len(relations)} relations were given but the map does not declare the relations "
            "capability"
        )


def _check_endpoints(
    entities: tuple[EntityEntry, ...], relations: tuple[RelationEntry, ...]
) -> None:
    """Refuse a relation whose subject or object is not an entity of the map."""
    keys = {entry.key for entry in entities}
    for relation in relations:
        for role, key in (("subject", relation.subject_key), ("object", relation.object_key)):
            if key not in keys:
                raise InvalidContentError(
                    f"relation {relation.key!r} has the {role} {key!r}, which is not an entity "
                    "of the map"
                )


def _payloads(entity_count: int, relation_count: int) -> tuple[Payload, ...]:
    return (
        RecordPayload(
            path=ENTITIES,
            role=PayloadRole.AUTHORITATIVE,
            semantics=(
                "One entity per line as {key, record}, ordered by key; record is the canonical "
                "entity record."
            ),
            record_count=entity_count,
        ),
        RecordPayload(
            path=RELATIONS,
            role=PayloadRole.AUTHORITATIVE,
            semantics=(
                "One relation per line as {key, subject, object, record}, ordered by key; "
                "subject and object are entity keys."
            ),
            record_count=relation_count,
        ),
        RecordPayload(
            path=ENTITY_INDEX,
            role=PayloadRole.DERIVED_INDEX,
            semantics="Entity key to byte offset and length of its line in entities.jsonl.",
            record_count=entity_count,
            derived_from=(ENTITIES,),
        ),
        RecordPayload(
            path=RELATION_INDEX,
            role=PayloadRole.DERIVED_INDEX,
            semantics="Relation key to byte offset and length of its line in relations.jsonl.",
            record_count=relation_count,
            derived_from=(RELATIONS,),
        ),
        RecordPayload(
            path=ENTITY_RELATION_INDEX,
            role=PayloadRole.DERIVED_INDEX,
            semantics=(
                "Per entity, the keys of the relations where it is the subject and where it is "
                "the object."
            ),
            record_count=entity_count,
            derived_from=(ENTITIES, RELATIONS),
        ),
    )


def _render_readme(manifest: ContextMapArtifactManifest) -> str:
    """Summarize the artifact for people; no time and no path, so it is deterministic."""
    lines = [
        f"# Context map `{manifest.context_map_id}`",
        "",
        f"- Content identity: `{manifest.content_identity}`",
        f"- Format version: `{manifest.format_version}`",
        f"- Schema version: `{manifest.schema_version}`",
        f"- Entities: {manifest.entity_count}; relations: {manifest.relation_count}",
        "",
        "Upstream artifacts referred to, never copied:",
        "",
    ]
    lines += [
        f"- `{item.artifact_type}` `{item.artifact_id}` ({item.requirement.value})"
        for item in manifest.dependencies
    ]
    lines += [
        "",
        "`manifest.json` inventories every contractual file with size and SHA-256. There is no "
        "debug data in this artifact.",
        "",
    ]
    return "\n".join(lines)
