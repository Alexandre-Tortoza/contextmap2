"""Deterministic, atomic writer of a ContextMapArtifact directory.

The writer turns a :class:`~contextmap.artifact.ContextMap` into the directory described in
``src/contextmap/artifact/docs/storage-layout.md``. It validates everything it can before
touching the disk, writes into a hidden temporary sibling and publishes with one rename
(:class:`~contextmap.shared.AtomicRunDirectory`), so an interrupted write never looks like a
finished artifact and a finished artifact is never overwritten.

The map is encoded once, and the entity and relation tables, which grow with the map, are written
line by line and hashed as they are written; only the small documents are held as bytes. The
output is a function of the map: records are ordered by key, every line is canonical and the
manifest carries a content identity that ignores the write time. Nothing invalid is dropped
silently: an inconsistent map is refused whole with the reason. The schema already refuses an
invalid map when it is built (references, provenance, declared capabilities), so the writer adds
only what the schema cannot know: that the upstream artifacts the map cites are on disk, are
intact and are exactly the ones its lineage names. Pinning a structural dependency by the digest
of its manifest is not enough on its own to prove that: ``artifact_id`` and ``content_identity``
are independent fields, so the writer also opens the Entity Resolution and Spatial Relations runs
with their own readers (:mod:`contextmap.artifact.serialization.structural_dependencies`) to
check their own identity, that Spatial Relations was built over the very Entity Resolution run
the map also cites, and that every resolved entity and relation the map's records point at
actually exists upstream. No model, ROS or runtime object ever reaches the files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextmap.artifact.models import ContextMap
from contextmap.artifact.provenance import ArtifactKind
from contextmap.artifact.records import context_map_to_record
from contextmap.artifact.serialization.decoding import entity_lines, relation_lines
from contextmap.artifact.serialization.dependencies import (
    artifact_digest,
    read_inventory,
    relative_locator,
    verify_inventory,
)
from contextmap.artifact.serialization.errors import (
    ArtifactExistsError,
    ContextMapArtifactError,
    InvalidContentError,
    UpstreamArtifactError,
)
from contextmap.artifact.serialization.layout import (
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
from contextmap.artifact.serialization.manifest import (
    ContextMapArtifactManifest,
    DependencyRecord,
    Payload,
    PayloadRole,
    RecordPayload,
    Requirement,
    create_manifest,
    decode_manifest,
    encode_manifest,
)
from contextmap.artifact.serialization.structural_dependencies import (
    check_structural_dependencies,
)
from contextmap.artifact.serialization.tables import (
    document_json,
    encode_entity_relation_index,
    write_record_table,
)
from contextmap.geometric_mapping import GeometricMapArtifactReader, MapArtifactError
from contextmap.shared import AtomicRunDirectory, RunDirectoryError


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
        self, context_map: ContextMap, *, upstream_locations: Mapping[str, Path]
    ) -> ContextMapArtifactManifest:
        """Validate the map against the upstream artifacts it cites and publish it atomically.

        Args:
            context_map: The map to persist.
            upstream_locations: Where the upstream artifacts the map cites are now, keyed by
                artifact id. The location of every *structural* dependency (the geometric map,
                the entity-resolution and spatial-relations runs) is required; the location of
                optional evidence is not, but when given it is verified. A location is used to
                read and verify the artifact and to compute a relative hint; it is never
                recorded as an identity.

        Returns:
            The manifest of the published artifact.

        Raises:
            ArtifactExistsError: If ``output_dir`` already exists; a finished artifact is never
                overwritten.
            InvalidContentError: If a location names an artifact the map does not cite, a
                structural dependency has no location, an upstream artifact is not the one the
                lineage names, the geometric map is not the one the map declares (identity, point
                count or frame), an Entity Resolution or Spatial Relations run's own identity or
                geometric-map lineage disagrees with what the map declares for it, a Spatial
                Relations run was built over a different Entity Resolution run than the one the
                map also cites, or a ``ContextEntity.source`` or ``ContextRelation`` the map
                lists does not resolve in the corresponding run.
            RecordTableError: If a record is not plain JSON. A record of a table is only found
                while the tables are written: the temporary directory is then discarded and
                nothing is published.
            UpstreamArtifactError: If an upstream artifact is missing or does not match its own
                inventory.
            ContextMapArtifactError: If the directory cannot be published.
        """
        if self._output_dir.exists():
            raise ArtifactExistsError(f"the artifact already exists: {self._output_dir}")
        dependencies = self._dependencies(context_map, upstream_locations)

        # O mapa é codificado uma vez só; documentos e tabelas são partes do mesmo registro.
        record = context_map_to_record(context_map)
        documents = {
            MAP_METADATA: document_json(record["metadata"]),
            GEOMETRY_REFERENCE: document_json(record["geometry_ref"]),
            LINEAGE: document_json({"upstream_artifacts": record["lineage"]}),
            ENTITY_RELATION_INDEX: encode_entity_relation_index(
                (str(entity.entity_id) for entity in context_map.entities),
                (
                    (str(item.relation_id), str(item.subject.entity_id), str(item.object.entity_id))
                    for item in context_map.relations
                ),
            ),
        }
        creation = context_map.metadata.creation

        try:
            with AtomicRunDirectory(self._output_dir) as run:
                for path, data in documents.items():
                    run.write_bytes(path, data)
                entity_count = _write_table(
                    run, ENTITIES, ENTITY_INDEX, entity_lines(context_map, record)
                )
                relation_count = _write_table(
                    run, RELATIONS, RELATION_INDEX, relation_lines(context_map, record)
                )
                # A identidade de conteúdo cobre o inventário: tamanho e hash vêm da própria
                # escrita, sem reler nenhum arquivo.
                manifest = create_manifest(
                    context_map_id=str(context_map.context_map_id),
                    schema_version=context_map.schema_version,
                    written_at=self._written_at.isoformat(),
                    code_version=creation.code_version,
                    configuration_fingerprint=creation.configuration_fingerprint,
                    entity_count=entity_count,
                    relation_count=relation_count,
                    payloads=_payloads(entity_count, relation_count),
                    dependencies=dependencies,
                    file_inventory=run.inventory(),
                )
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
        self, context_map: ContextMap, locations: Mapping[str, Path]
    ) -> tuple[DependencyRecord, ...]:
        """Verify every upstream artifact that has a location and pin each one by its digest."""
        cited = {item.artifact_id for item in context_map.lineage}
        for artifact_id in sorted(set(locations) - cited):
            raise InvalidContentError(
                f"a location was given for {artifact_id!r}, which the map does not cite"
            )
        records: list[DependencyRecord] = []
        entity_resolution_locations: dict[str, Path] = {}
        spatial_relations_locations: dict[str, Path] = {}
        for upstream in context_map.lineage:
            requirement = (
                Requirement.REQUIRED if upstream.kind.is_structural else Requirement.OPTIONAL
            )
            location = locations.get(upstream.artifact_id)
            if location is None:
                if requirement is Requirement.REQUIRED:
                    raise InvalidContentError(
                        f"the structural dependency {upstream.kind.value} "
                        f"{upstream.artifact_id!r} has no location: pass it in upstream_locations"
                    )
            else:
                self._verify_upstream(context_map, upstream.artifact_id, upstream.kind, location)
                found = artifact_digest(location)
                if found != upstream.content_identity:
                    raise InvalidContentError(
                        f"the lineage names {upstream.kind.value} {upstream.artifact_id!r} with "
                        f"content identity {upstream.content_identity} but the artifact at "
                        f"{location.name!r} has {found}"
                    )
                if upstream.kind is ArtifactKind.ENTITY_RESOLUTION_RUN:
                    entity_resolution_locations[upstream.artifact_id] = location
                elif upstream.kind is ArtifactKind.SPATIAL_RELATIONS_RUN:
                    spatial_relations_locations[upstream.artifact_id] = location
            records.append(
                DependencyRecord(
                    artifact_type=upstream.kind.value,
                    artifact_id=upstream.artifact_id,
                    content_identity=upstream.content_identity,
                    requirement=requirement,
                    locator=None
                    if location is None
                    else relative_locator(self._output_dir, location),
                )
            )
        problems = check_structural_dependencies(
            context_map,
            entity_resolution_locations=entity_resolution_locations,
            spatial_relations_locations=spatial_relations_locations,
        )
        if problems:
            raise InvalidContentError(
                "the structural dependencies do not match what the map's own entities and "
                "relations need: " + "; ".join(problems)
            )
        return tuple(records)

    def _verify_upstream(
        self, context_map: ContextMap, artifact_id: str, kind: ArtifactKind, location: Path
    ) -> None:
        """Check the files of one upstream artifact and, for the geometry, what the map declares."""
        verify_inventory(location, read_inventory(location))
        if kind is not ArtifactKind.GEOMETRIC_MAP:
            return
        link = context_map.geometry_ref
        try:
            with GeometricMapArtifactReader(location) as reader:
                upstream = reader.manifest
        except MapArtifactError as error:
            raise UpstreamArtifactError(
                f"the geometric map at {location.name!r} cannot be opened: {error}"
            ) from error
        if upstream.map_id != link.map_id or artifact_id != str(link.map_id):
            raise InvalidContentError(
                f"the map refers to the geometric map {link.map_id!r} but {location.name!r} "
                f"holds {upstream.map_id!r}"
            )
        if upstream.point_count != link.point_count:
            raise InvalidContentError(
                f"the map declares {link.point_count} geometry elements but the geometric map "
                f"{upstream.map_id!r} has {upstream.point_count}"
            )
        frame = context_map.metadata.frame.frame_id
        if upstream.map_frame != frame:
            raise InvalidContentError(
                f"the map is expressed in frame {frame!r} but the geometric map "
                f"{upstream.map_id!r} is in frame {upstream.map_frame!r}"
            )


def _write_table(
    run: AtomicRunDirectory,
    payload_path: str,
    index_path: str,
    lines: Iterable[Mapping[str, Any]],
) -> int:
    """Stream one record table and its offset index into the run; return its record count."""
    with run.open_binary(payload_path) as payload, run.open_binary(index_path) as index:
        return write_record_table(lines, payload=payload, index=index)


def _payloads(entity_count: int, relation_count: int) -> tuple[Payload, ...]:
    return (
        RecordPayload(
            path=ENTITIES,
            role=PayloadRole.AUTHORITATIVE,
            semantics=(
                "One entity per line as {key, record}, ordered by key; record is the schema's "
                "canonical entity record."
            ),
            record_count=entity_count,
        ),
        RecordPayload(
            path=RELATIONS,
            role=PayloadRole.AUTHORITATIVE,
            semantics=(
                "One relation per line as {key, subject, object, record}, ordered by key; "
                "subject and object are entity ids and record is the schema's canonical "
                "relation record."
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
        "Upstream artifacts the map cites, referred to and never copied:",
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
