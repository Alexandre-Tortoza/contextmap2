"""Builders for ContextMapArtifact serialization tests.

A map is assembled from the schema test builders and a small *world* of upstream artifacts on
disk. Three of them are written by the real writers of their capabilities: a GeometricMapArtifact
(Geometric Mapping), an EntityResolutionRunArtifact and a SpatialRelationsRunArtifact (on a tiny
scene, see ``context_map_serialization_upstream``); the fourth, one optional evidence run, is a
stand-in with the same manifest and inventory conventions. The populated map is assembled from the
two real runs: its entities are the resolved entities of the resolution run and its relations
are the relations of the spatial-relations run, with their real identities and geometry
references; only the semantic state and the origin of each record, which the assembly stage that
does not exist yet would derive, come from the schema builders. The lineage of the map pins every
located artifact by the digest of its manifest, so no perception, robotics or model runtime is
involved. Results are contract/synthetic tests over real artifact formats.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from context_map_builders import (
    ENTITY_RESOLUTION_ARTIFACT_ID,
    FUSION_ARTIFACT_ID,
    GEOMETRIC_MAP_ID,
    SEMANTIC_MAP_ARTIFACT_ID,
    SPATIAL_RELATIONS_ARTIFACT_ID,
    context_map,
    entity_capabilities,
    metadata,
    semantic_state,
)
from context_map_builders import entity as schema_entity
from context_map_builders import relation as schema_relation
from context_map_serialization_geometry import POINT_COUNT, build_geometry_artifact
from context_map_serialization_upstream import write_relations_run, write_resolution_run

from contextmap.artifact import (
    AmbiguityStatus,
    ContextMap,
    ContextMapArtifactManifest,
    ContextMapArtifactWriter,
)
from contextmap.artifact.serialization.dependencies import artifact_digest
from contextmap.entity_resolution import EntityResolutionRunReader, ResolvedEntityReference
from contextmap.shared import AtomicRunDirectory
from contextmap.spatial_relations import RelationId, SpatialRelationsRunReader

WRITTEN_AT = "2026-09-21T12:00:00+00:00"
MAP_ID = str(GEOMETRIC_MAP_ID)
__all__ = ["MAP_ID", "POINT_COUNT", "WRITTEN_AT"]


@dataclass(frozen=True)
class World:
    """The upstream artifacts a test map is written against, all real and on disk."""

    root: Path
    geometry_dir: Path
    resolution_dir: Path
    relations_dir: Path
    fusion_dir: Path

    @property
    def locations(self) -> dict[str, Path]:
        """Where each located upstream artifact is, keyed by artifact id."""
        return {
            MAP_ID: self.geometry_dir,
            ENTITY_RESOLUTION_ARTIFACT_ID: self.resolution_dir,
            SPATIAL_RELATIONS_ARTIFACT_ID: self.relations_dir,
            FUSION_ARTIFACT_ID: self.fusion_dir,
        }

    @property
    def structural_locations(self) -> dict[str, Path]:
        """Only the locations the writer requires."""
        return {
            MAP_ID: self.geometry_dir,
            ENTITY_RESOLUTION_ARTIFACT_ID: self.resolution_dir,
            SPATIAL_RELATIONS_ARTIFACT_ID: self.relations_dir,
        }


def make_upstream(root: Path, name: str, artifact_id: str) -> Path:
    """Publish a small upstream run artifact the way sibling capabilities do.

    It has a manifest with a file inventory, two contractual files and a ``debug/`` file that is
    not inventoried.
    """
    final_dir = root / name
    with AtomicRunDirectory(final_dir) as run:
        run.write_bytes("outputs/records.jsonl", b'{"record": "one"}\n' * 4)
        run.write_text("outputs/summary.json", json.dumps({"run": artifact_id}))
        run.write_text("debug/notes.txt", "human only", contractual=False)
        run.publish(manifest={"run_id": artifact_id, "schema_version": "0.1.0"}, readme="# run\n")
    return final_dir


def make_world(root: Path, *, geometry_scans: int = 10, points_per_scan: int = 100) -> World:
    """Write the upstream artifacts a populated test map cites.

    The geometric map, the entity-resolution run and the spatial-relations run are written by
    their capabilities' real writers; the evidence run is a stand-in.
    """
    geometry_dir, _ = build_geometry_artifact(
        root / "geometry-workspace", scans=geometry_scans, points_per_scan=points_per_scan
    )
    resolution_dir, relations_dir = root / "entity-resolution", root / "spatial-relations"
    write_resolution_run(
        resolution_dir,
        run_id=ENTITY_RESOLUTION_ARTIFACT_ID,
        geometric_map_id=MAP_ID,
        semantic_map_id=SEMANTIC_MAP_ARTIFACT_ID,
    )
    write_relations_run(
        relations_dir, run_id=SPATIAL_RELATIONS_ARTIFACT_ID, resolution_dir=resolution_dir
    )
    return World(
        root=root,
        geometry_dir=geometry_dir,
        resolution_dir=resolution_dir,
        relations_dir=relations_dir,
        fusion_dir=make_upstream(root, "semantic-fusion", FUSION_ARTIFACT_ID),
    )


def pinned(context_map: ContextMap, world: World) -> ContextMap:
    """The same map with the lineage identity of every located artifact set to its real digest."""
    digests = {
        artifact_id: artifact_digest(location) for artifact_id, location in world.locations.items()
    }
    lineage = tuple(
        replace(item, content_identity=digests.get(item.artifact_id, item.content_identity))
        for item in context_map.lineage
    )
    return replace(context_map, lineage=lineage)


_SEMANTIC_STATES = (
    lambda: semantic_state(),
    lambda: semantic_state(AmbiguityStatus.AMBIGUOUS, ("box", "table")),
    lambda: semantic_state(AmbiguityStatus.INSUFFICIENT_EVIDENCE, ()),
)


def assemble_from_runs(world: World) -> ContextMap:
    """Assemble a map from the real resolution and spatial-relations runs of the world.

    Entities are ``entity-0001`` to ``entity-000N`` in the order of the resolved entities, each
    mapping to its resolved entity with the real members, decisions, unresolved neighbours and
    geometry references; relations are ``relation-0001`` to ``relation-000M`` in the order of the
    relations of the run, with their real ends, predicate and state. The semantic state of each
    entity is set by the test (the first is plain, the second ambiguous, the third has no
    evidence), which is what the missing assembly stage would derive.
    """
    resolution = EntityResolutionRunReader(world.resolution_dir)
    relations_run = SpatialRelationsRunReader(world.relations_dir)
    identities: dict[ResolvedEntityReference, str] = {}
    entities = []
    for number, resolved in enumerate(resolution.resolved_entities().entities, 1):
        source = ResolvedEntityReference(
            resolution_run_id=resolved.resolution_run_id,
            resolved_entity_id=resolved.resolved_entity_id,
        )
        identities[source] = f"entity-{number:04d}"
        entities.append(
            replace(
                schema_entity(identities[source]),
                source=source,
                member_entities=tuple(member.entity_ref for member in resolved.members),
                resolution_decisions=resolved.resolution_decision_refs,
                unresolved_neighbors=resolved.unresolved_neighbor_refs,
                geometry_refs=resolved.geometry.geometry_refs,
                semantic_state=_SEMANTIC_STATES[(number - 1) % len(_SEMANTIC_STATES)](),
            )
        )
    relations = []
    ordered = sorted(relations_run.iter_relations(), key=lambda item: str(item.relation_id))
    for number, relation in enumerate(ordered, 1):
        relations.append(
            schema_relation(
                f"relation-{number:04d}",
                identities[relation.subject_entity_ref],
                relation.predicate,
                identities[relation.object_entity_ref],
                source_relation_id=RelationId(str(relation.relation_id)),
                state=relation.state,
                uncertainty_kinds=tuple(
                    sorted(
                        {item.kind for item in relation.uncertainty}, key=lambda kind: kind.value
                    )
                ),
            )
        )
    capabilities = entity_capabilities(*{item.predicate for item in relations})
    return context_map(
        metadata=metadata(capabilities=capabilities),
        entities=tuple(entities),
        relations=tuple(relations),
    )


def make_context_map(world: World, *, kind: str = "populated") -> ContextMap:
    """A valid map pinned to the world: populated (entities and relations) or geometry only."""
    if kind == "populated":
        return pinned(assemble_from_runs(world), world)
    if kind == "geometry-only":
        return pinned(context_map(), world)
    raise ValueError(kind)


def write_artifact(
    world: World,
    *,
    name: str = "context_map",
    context_map: ContextMap | None = None,
    locations: dict[str, Path] | None = None,
    written_at: datetime | None = None,
) -> tuple[Path, ContextMapArtifactManifest]:
    """Write an artifact under ``world.root/out/<name>`` and return it with its manifest.

    By default every located upstream artifact the map cites is passed; a test passes
    ``locations`` to give fewer (or others).
    """
    output_dir = world.root / "out" / name
    written = context_map if context_map is not None else make_context_map(world)
    cited = {item.artifact_id for item in written.lineage}
    manifest = ContextMapArtifactWriter(
        output_dir=output_dir,
        written_at=written_at or datetime.fromisoformat(WRITTEN_AT),
    ).write(
        written,
        upstream_locations=(
            {key: value for key, value in world.locations.items() if key in cited}
            if locations is None
            else locations
        ),
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
