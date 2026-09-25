"""Real content comparison of two independent ContextMapArtifact reruns (PR #438 review, finding
3 on the run-0002 acceptance report): cardinality equality (entity_count/relation_count) proves
nothing about the actual entities/relations. This canonicalizes and compares them.

Canonicalization strategy: entity/relation identity strings (context_map_id, entity_id,
resolution_run_id, ...) are deliberately run-specific -- two independent, correct executions are
not expected to produce the same strings. What must match is the real content: which physical
points each entity's geometry_refs actually resolve to (real (x, y, z) coordinates in the shared
map frame, via GeometrySource.get(), never the reference's own map_id/geometry_id string) and
which label hypotheses it carries. An entity's canonical key is
(sorted rounded coordinates, sorted label texts); a relation's canonical key is
(predicate, canonical key of subject, canonical key of object) -- order preserved, since several
predicates are directional.

Coordinates are rounded to COORDINATE_DECIMALS places before comparison: the geometry pipeline is
deterministic over the same real inputs, so exact equality is expected, but comparing floats
verbatim would be fragile to accumulated representation noise across independent processes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from contextmap.artifact import ContextMap, ContextMapArtifactReader
from contextmap.artifact.composition import ContextEntity, ContextRelation
from contextmap.artifact.references import ContextEntityId
from contextmap.geometric_mapping import GeometricMapArtifactReader, GeometryReference, GeometrySource

COORDINATE_DECIMALS = 6

EntityKey = tuple[tuple[tuple[float, float, float], ...], tuple[str, ...]]


def _entity_canonical_key(entity: ContextEntity, geometry: GeometrySource) -> EntityKey:
    points = tuple(
        sorted(
            tuple(round(value, COORDINATE_DECIMALS) for value in geometry.get(ref).coordinates_m)
            for ref in entity.geometry_refs
        )
    )
    labels = tuple(sorted(hypothesis.label for hypothesis in entity.semantic_state.hypotheses))
    return (points, labels)


def _entity_keys(context_map: ContextMap, geometry: GeometrySource) -> dict[ContextEntityId, EntityKey]:
    return {entity.entity_id: _entity_canonical_key(entity, geometry) for entity in context_map.entities}


def _relation_canonical_key(
    relation: ContextRelation, entity_keys: dict[ContextEntityId, EntityKey]
) -> tuple[str, EntityKey, EntityKey]:
    return (
        relation.predicate.value,
        entity_keys[relation.subject.entity_id],
        entity_keys[relation.object.entity_id],
    )


def _digest(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _load(context_map_dir: Path, geometry_dir: Path) -> tuple[ContextMap, GeometricMapArtifactReader]:
    context_map = ContextMapArtifactReader.open(context_map_dir, verify_hashes=True).context_map()
    geometry_ref = next(item for item in context_map.lineage if item.kind.value == "geometric_map")
    geometry_reader = GeometricMapArtifactReader(geometry_dir)
    if str(geometry_reader.manifest.map_id) != geometry_ref.artifact_id:
        raise SystemExit(
            f"the geometric map at {geometry_dir} is {geometry_reader.manifest.map_id!r}, "
            f"but {context_map_dir} cites {geometry_ref.artifact_id!r}"
        )
    return context_map, geometry_reader


def compare(
    label: str, dir_a: Path, geometry_dir_a: Path, dir_b: Path, geometry_dir_b: Path
) -> bool:
    context_map_a, geometry_reader_a = _load(dir_a, geometry_dir_a)
    context_map_b, geometry_reader_b = _load(dir_b, geometry_dir_b)
    with geometry_reader_a as geometry_a, geometry_reader_b as geometry_b:
        keys_a = _entity_keys(context_map_a, geometry_a.geometry())
        keys_b = _entity_keys(context_map_b, geometry_b.geometry())

        entity_multiset_a = Counter(keys_a.values())
        entity_multiset_b = Counter(keys_b.values())
        entity_digest_a = _digest(sorted(json.dumps(k, default=str) for k in entity_multiset_a.elements()))
        entity_digest_b = _digest(sorted(json.dumps(k, default=str) for k in entity_multiset_b.elements()))

        relation_multiset_a = Counter(
            _relation_canonical_key(relation, keys_a) for relation in context_map_a.relations
        )
        relation_multiset_b = Counter(
            _relation_canonical_key(relation, keys_b) for relation in context_map_b.relations
        )
        relation_digest_a = _digest(
            sorted(json.dumps(k, default=str) for k in relation_multiset_a.elements())
        )
        relation_digest_b = _digest(
            sorted(json.dumps(k, default=str) for k in relation_multiset_b.elements())
        )

    entity_mismatches = entity_multiset_a - entity_multiset_b
    entity_mismatches += entity_multiset_b - entity_multiset_a
    relation_mismatches = relation_multiset_a - relation_multiset_b
    relation_mismatches += relation_multiset_b - relation_multiset_a

    print(f"=== {label} ===")
    print(f"  A: {dir_a}")
    print(f"  B: {dir_b}")
    print(f"  entity_count: A={len(context_map_a.entities)} B={len(context_map_b.entities)}")
    print(f"  relation_count: A={len(context_map_a.relations)} B={len(context_map_b.relations)}")
    print(f"  entity canonical digest: A={entity_digest_a}")
    print(f"                           B={entity_digest_b}")
    print(f"  relation canonical digest: A={relation_digest_a}")
    print(f"                             B={relation_digest_b}")
    print(f"  entity canonical mismatches: {sum(entity_mismatches.values())}")
    print(f"  relation canonical mismatches: {sum(relation_mismatches.values())}")

    equivalent = (
        entity_digest_a == entity_digest_b
        and relation_digest_a == relation_digest_b
        and not entity_mismatches
        and not relation_mismatches
    )
    print(f"  CONTENT EQUIVALENT: {equivalent}")
    return equivalent


def main() -> None:
    workspace = Path(__file__).resolve().parents[3] / "outputs"
    ok = compare(
        "run-0003 (hand-chained) vs run-0006 (run_plan()/resume_plan()-orchestrated), "
        "both under the corrected code_version (contextmap.__version__ reflecting the real "
        "HEAD, not the stale editable-install metadata run-0002 was produced under)",
        workspace / "e2e-real/run-0003/context_map",
        workspace / "e2e-real/run-0003/geometric_mapping",
        workspace / "corridor-02-repro-check/run-0006/context_map",
        workspace / "corridor-02-repro-check/run-0005/geometric_mapping",
    )
    if not ok:
        raise SystemExit("content mismatch: see mismatches above")


if __name__ == "__main__":
    main()
