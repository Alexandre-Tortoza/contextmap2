"""Real Entity Resolution and Spatial Relations run artifacts for the serialization tests.

The runs are written by the real ``EntityResolutionRunWriter`` and ``SpatialRelationsRunWriter``
on a tiny scene built the way those capabilities' own tests build theirs: four semantic entities
that are boxes one metre apart, two of which the resolution merges, one it leaves unresolved
against its neighbour and one it declares distinct; the relations are then decided from the
resolved geometry (neighbours are next to each other; the geometry rejects containment).

What is real: the two run artifacts (their manifests, inventories, indexes and identities), the
identities they hand out and the digests that pin them. What is synthetic: the scene (boxes in an
in-memory geometry source instead of the points of the real geometric map) and the entity records
of the map itself, which the still non-existent assembly stage would derive from the runs. The
helper modules of the other capabilities' tests are imported by base name, which needs their
directories on ``sys.path`` (the Spatial Relations wiring tests do the same).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

for _directory in ("semantic_fusion", "semantic_mapping", "entity_resolution"):
    _path = str(Path(__file__).resolve().parents[1] / _directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from resolution_entity_builders import box_corners, entity_at, scene_source  # noqa: E402
from resolution_run_fixtures import DISTINCT, MATCH, UNRESOLVED, resolution_of  # noqa: E402
from resolution_run_fixtures import LINEAGE as ER_LINEAGE  # noqa: E402

from contextmap.entity_resolution import (  # noqa: E402
    CandidateRetrievalPolicy,
    EntityResolutionRunId,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    ResolvedEntityReference,
    materialize_resolved_entities,
    retrieve_candidate_sets,
)
from contextmap.geometric_mapping import MapId  # noqa: E402
from contextmap.semantic_mapping import GeometrySummaryPolicy, SemanticMapId  # noqa: E402
from contextmap.shared import Vector3  # noqa: E402
from contextmap.spatial_relations import (  # noqa: E402
    AxisDirection,
    CandidatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunWriter,
    decide_relations,
    evaluate_geometric_candidates,
    generate_relation_candidates,
    lineage_from_resolution_manifest,
    resolved_entity_geometries,
)

NAMES = "abcd"
_STEP_INDEXES = 100
_SUMMARY_POLICY = GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=1000.0)
_CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
_GEOMETRIC = GeometricPredicatePolicy(
    boundary_tolerance_m=0.02,
    next_to_max_gap_m=0.6,
    adjacent_penetration_m=0.05,
    containment_slack_m=0.05,
    directional_overlap_fraction=0.5,
)
_CANDIDATES = CandidatePolicy(
    predicates=(RelationPredicate.NEXT_TO, RelationPredicate.ABOVE, RelationPredicate.INSIDE),
    proximity_radius_m=0.6,
    directional_radius_m=2.0,
)
_POLICIES = RelationsRunPolicies(
    frame_conventions=_CONVENTIONS, candidate=_CANDIDATES, geometric=_GEOMETRIC
)


def write_resolution_run(
    output_dir: Path, *, run_id: str, geometric_map_id: str, semantic_map_id: str
) -> None:
    """Write a real Entity Resolution run over the four boxes.

    Args:
        output_dir: The final directory of the run.
        run_id: Identity of the run, which the map's lineage will cite.
        geometric_map_id: The geometric map the entities' geometry references point into.
        semantic_map_id: The semantic map the source entities belong to.
    """
    resolution_run = EntityResolutionRunId(run_id)
    entities = {
        name: entity_at(
            name,
            (index * 1.0, 0.0, 0.0),
            support_number=index + 1,
            spatial=(f"spatial--{name}",),
            map_id=MapId(geometric_map_id),
            semantic_map_id=SemanticMapId(semantic_map_id),
            first_index=index * _STEP_INDEXES,
        )
        for index, name in enumerate(NAMES)
    }
    candidate_sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
    )
    resolutions = tuple(
        resolution_of(entities[first], entities[second], outcome)
        for first, second, outcome in (
            ("a", "b", MATCH),
            ("b", "c", UNRESOLVED),
            ("c", "d", DISTINCT),
        )
    )
    materialization = materialize_resolved_entities(
        entities.values(), [item.decision for item in resolutions], resolution_run_id=resolution_run
    )
    lineage = dataclasses.replace(
        ER_LINEAGE,
        geometric_map_id=MapId(geometric_map_id),
        semantic_map_ids=(SemanticMapId(semantic_map_id),),
    )
    EntityResolutionRunWriter(
        output_dir=output_dir, run_id=resolution_run, lineage=lineage, code_version="test"
    ).write(candidate_sets=candidate_sets, resolutions=resolutions, materialization=materialization)


def _scene_points() -> dict[int, Vector3]:
    points: dict[int, Vector3] = {}
    for index in range(len(NAMES)):
        corners = box_corners((index * 1.0, 0.0, 0.0), 0.5)
        first = index * _STEP_INDEXES
        points.update(zip(range(first, first + len(corners)), corners, strict=True))
    return points


def write_relations_run(output_dir: Path, *, run_id: str, resolution_dir: Path) -> None:
    """Write a real Spatial Relations run over the resolved entities of a resolution run.

    The geometry decides that neighbouring resolved entities are next to each other (supported)
    and that none contains another (rejected), so the run holds relations in two states.

    Args:
        output_dir: The final directory of the run.
        run_id: Identity of the run, which the map's lineage will cite.
        resolution_dir: The real resolution run whose resolved entities are related.
    """
    resolution = EntityResolutionRunReader(resolution_dir)
    geometries = resolved_entity_geometries(
        resolution.resolved_entities(),
        source=scene_source(_scene_points(), map_id=resolution.manifest.lineage.geometric_map_id),
        policy=_SUMMARY_POLICY,
    )
    candidates = generate_relation_candidates(
        geometries, policy=_CANDIDATES, conventions=_CONVENTIONS
    )
    evidence = list(
        evaluate_geometric_candidates(
            candidates, entities=geometries, policy=_GEOMETRIC, conventions=_CONVENTIONS
        )
    )
    decisions = decide_relations(candidates, evidence)
    SpatialRelationsRunWriter(
        output_dir=output_dir,
        run_id=SpatialRelationsRunId(run_id),
        lineage=lineage_from_resolution_manifest(resolution.manifest),
        policies=_POLICIES,
        code_version="test",
    ).write(candidates=candidates, evidence=evidence, decisions=decisions)


__all__ = [
    "NAMES",
    "ResolvedEntityReference",
    "write_relations_run",
    "write_resolution_run",
]
