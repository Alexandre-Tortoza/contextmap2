"""The geometry of resolved entities, in the form the relation evaluators read.

Entity Resolution persists the geometry of a resolved entity as the union of its members' supports:
the exact geometry references and their tight bounds. The evaluators of this capability read an
:class:`~contextmap.semantic_mapping.EntityGeometry`, which also carries the statistics, diagnostics
and optional orientation of the support. This module produces that summary for the *union* support,
with the same algorithm Semantic Mapping uses for a single entity, and it checks the result against
what the resolution persisted: the references are the ones Entity Resolution named, and the bounds
recomputed from the geometric map are exactly the bounds it recorded. A relation is then measured
on the geometry the resolved entity has, never on an approximation of it.
"""

from __future__ import annotations

from contextmap.entity_resolution import ResolvedEntityReference, ResolvedEntitySet
from contextmap.geometric_mapping import GeometrySource
from contextmap.semantic_mapping import EntityGeometry, GeometrySummaryPolicy, summarize_geometry


def resolved_entity_geometries(
    resolved: ResolvedEntitySet, *, source: GeometrySource, policy: GeometrySummaryPolicy
) -> dict[ResolvedEntityReference, EntityGeometry]:
    """Summarize the union support of every resolved entity from the geometric map.

    Args:
        resolved: The resolved entities of one resolution run.
        source: The read boundary of the geometric map the entities' geometry belongs to.
        policy: The thresholds of the spatial summary, the same kind Semantic Mapping uses.

    Returns:
        The geometry of each resolved entity, by reference, ready for candidate generation and the
        relation evaluators.

    Raises:
        GeometryResolutionError: If a geometry reference cannot be resolved against the map.
        ValueError: If the frame or the bounds recomputed from the map differ from what the
            resolution persisted, which means the map is not the one the entities were built on.
    """
    geometries: dict[ResolvedEntityReference, EntityGeometry] = {}
    for entity in resolved.entities:
        persisted = entity.geometry
        geometry = summarize_geometry(persisted.geometry_refs, source=source, policy=policy)
        if geometry.map_frame != persisted.map_frame or geometry.bounds != persisted.bounds:
            raise ValueError(
                f"the geometry of resolved entity {entity.resolved_entity_id!r} read from the "
                f"geometric map has bounds or frame other than the ones the resolution "
                f"persisted, so the map is not the one the entity was resolved on"
            )
        reference = ResolvedEntityReference(
            resolution_run_id=resolved.resolution_run_id,
            resolved_entity_id=entity.resolved_entity_id,
        )
        geometries[reference] = geometry
    return geometries
