"""Builders for ContextMapArtifact serialization tests.

A map is assembled from the schema test builders, a real GeometricMapArtifact and small upstream
artifacts published the way sibling capabilities publish theirs, so no perception, robotics or
model runtime is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from context_map_builders import capabilities, geometry_link, metadata
from context_map_builders import context_map as schema_context_map
from context_map_serialization_geometry import MAP_ID, POINT_COUNT, build_geometry_artifact

from contextmap.artifact import ContextMap, MapCapability, Requirement
from contextmap.artifact.dependencies import UpstreamArtifact
from contextmap.artifact.writer import EntityEntry, RelationEntry
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
