"""Builders for ContextMapArtifact serialization tests.

A map is assembled from the schema test builders, a real GeometricMapArtifact and small upstream
artifacts published the way sibling capabilities publish theirs, so no perception, robotics or
model runtime is involved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from context_map_builders import capabilities, geometry_link, metadata
from context_map_builders import context_map as schema_context_map
from context_map_serialization_geometry import MAP_ID, POINT_COUNT, build_geometry_artifact

from contextmap.artifact import (
    ContextMap,
    ContextMapArtifactManifest,
    ContextMapArtifactWriter,
    EntityEntry,
    MapCapability,
    RelationEntry,
    Requirement,
    UpstreamArtifact,
)
from contextmap.geometric_mapping import MapId
from contextmap.shared import AtomicRunDirectory

WRITTEN_AT = "2026-09-21T12:00:00+00:00"


def make_context_map(*, entities: bool = True, relations: bool = True) -> ContextMap:
    """A valid map that declares the content the given flags say it has."""
    content = [MapCapability.GEOMETRY]
    if entities:
        content.append(MapCapability.ENTITIES)
    if relations:
        content.append(MapCapability.RELATIONS)
    declared = capabilities(
        content=tuple(sorted(content, key=lambda item: item.value)),
        relation_predicates=("next_to",) if relations else (),
    )
    return schema_context_map(
        metadata=metadata(capabilities=declared),
        geometry_ref=geometry_link(map_id=MapId(MAP_ID), point_count=POINT_COUNT),
    )


def make_geometry(root: Path) -> Path:
    """Write the real geometric map the maps above refer to and return its directory."""
    run_dir, _ = build_geometry_artifact(root / "geometry-workspace")
    return run_dir


def make_evidence(
    root: Path,
    *,
    name: str = "semantic-fusion",
    artifact_id: str = "run-0003",
    requirement: Requirement = Requirement.OPTIONAL,
    artifact_type: str = "semantic_fusion_run",
) -> UpstreamArtifact:
    """Publish a small evidence artifact: a manifest with an inventory, a payload and debug."""
    final_dir = root / name
    with AtomicRunDirectory(final_dir) as run:
        run.write_bytes("outputs/evidence.jsonl", b'{"claim": "door"}\n' * 4)
        run.write_text("outputs/summary.json", json.dumps({"run": artifact_id}))
        run.write_text("debug/notes.txt", "human only", contractual=False)
        run.publish(manifest={"run_id": artifact_id, "schema_version": "0.1.0"}, readme="# run\n")
    return UpstreamArtifact(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        location=final_dir,
        requirement=requirement,
    )


def entity(key: str, **fields: Any) -> EntityEntry:
    """An entity entry whose record carries its own key and a label."""
    return EntityEntry(key=key, record={"entity_id": key, "label": f"label-of-{key}", **fields})


def relation(key: str, subject: str, obj: str) -> RelationEntry:
    """A relation entry between two entity keys."""
    return RelationEntry(
        key=key,
        subject_key=subject,
        object_key=obj,
        record={"relation_id": key, "predicate": "next_to", "subject": subject, "object": obj},
    )


def default_entities() -> tuple[EntityEntry, ...]:
    """Three entities, keyed ``entity-a`` to ``entity-c``."""
    return (entity("entity-a"), entity("entity-b"), entity("entity-c"))


def default_relations() -> tuple[RelationEntry, ...]:
    """Two relations: a to b, and c to a."""
    return (
        relation("relation-1", "entity-a", "entity-b"),
        relation("relation-2", "entity-c", "entity-a"),
    )


def write_artifact(
    tmp_path: Path,
    geometry_dir: Path,
    *,
    name: str = "context_map",
    context_map: ContextMap | None = None,
    entities: tuple[EntityEntry, ...] | None = None,
    relations: tuple[RelationEntry, ...] | None = None,
    evidence: tuple[UpstreamArtifact, ...] = (),
    written_at: datetime | None = None,
) -> tuple[Path, ContextMapArtifactManifest]:
    """Write a small artifact under ``tmp_path/out/<name>`` and return it with its manifest."""
    output_dir = tmp_path / "out" / name
    manifest = ContextMapArtifactWriter(
        output_dir=output_dir, written_at=written_at or datetime.fromisoformat(WRITTEN_AT)
    ).write(
        context_map if context_map is not None else make_context_map(),
        geometry_dir=geometry_dir,
        entities=default_entities() if entities is None else entities,
        relations=default_relations() if relations is None else relations,
        evidence=evidence,
    )
    return output_dir, manifest


def tree_files(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` by relative path, with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Every entry under ``root`` with a content hash and its modification time.

    Used to prove that a read did not change anything: same files, same bytes, same times.
    """
    snapshot: dict[str, tuple[str, int]] = {}
    for path in sorted(root.rglob("*")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
        snapshot[path.relative_to(root).as_posix()] = (digest, path.stat().st_mtime_ns)
    return snapshot


def declared_without(context_map: ContextMap, capability: MapCapability) -> ContextMap:
    """The same map without one declared capability (and its predicates when relations)."""
    content = tuple(
        item for item in context_map.metadata.capabilities.content if item is not capability
    )
    predicates = (
        ()
        if capability is MapCapability.RELATIONS
        else context_map.metadata.capabilities.relation_predicates
    )
    declared = replace(
        context_map.metadata.capabilities, content=content, relation_predicates=predicates
    )
    return replace(context_map, metadata=replace(context_map.metadata, capabilities=declared))
