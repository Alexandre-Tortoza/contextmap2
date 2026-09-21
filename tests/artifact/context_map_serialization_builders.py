"""Builders for ContextMapArtifact serialization tests.

A map is assembled from the schema test builders and a small *world* of real upstream artifacts:
a GeometricMapArtifact written by the public writer of Geometric Mapping and generic run
artifacts (a manifest with an inventory) for the entity-resolution and spatial-relations runs and
for one optional evidence run. The lineage of the map is pinned to those artifacts by the digest
of their inventories, so no perception, robotics or model runtime is involved.
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
    SPATIAL_RELATIONS_ARTIFACT_ID,
    context_map,
    populated_map,
)
from context_map_serialization_geometry import POINT_COUNT, build_geometry_artifact

from contextmap.artifact import (
    ContextMap,
    ContextMapArtifactManifest,
    ContextMapArtifactWriter,
    inventory_digest,
)
from contextmap.artifact.serialization.dependencies import read_inventory
from contextmap.shared import AtomicRunDirectory

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


def make_world(root: Path) -> World:
    """Write the geometry and the upstream runs a populated test map cites."""
    geometry_dir, _ = build_geometry_artifact(root / "geometry-workspace")
    return World(
        root=root,
        geometry_dir=geometry_dir,
        resolution_dir=make_upstream(root, "entity-resolution", ENTITY_RESOLUTION_ARTIFACT_ID),
        relations_dir=make_upstream(root, "spatial-relations", SPATIAL_RELATIONS_ARTIFACT_ID),
        fusion_dir=make_upstream(root, "semantic-fusion", FUSION_ARTIFACT_ID),
    )


def pinned(context_map: ContextMap, world: World) -> ContextMap:
    """The same map with the lineage identity of every located artifact set to its real digest."""
    digests = {
        artifact_id: inventory_digest(read_inventory(location))
        for artifact_id, location in world.locations.items()
    }
    lineage = tuple(
        replace(item, content_identity=digests.get(item.artifact_id, item.content_identity))
        for item in context_map.lineage
    )
    return replace(context_map, lineage=lineage)


def make_context_map(world: World, *, kind: str = "populated") -> ContextMap:
    """A valid map pinned to the world: populated (entities and relations) or geometry only."""
    if kind == "populated":
        return pinned(populated_map(), world)
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
