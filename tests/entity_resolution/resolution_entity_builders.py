"""Deterministic builders of positioned semantic entities for the Entity Resolution tests.

Entities are built with the same builders and the same ``summarize_geometry`` Semantic Mapping
uses, so their geometry, semantic state, evidence links and temporal state are the real contracts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from mapping_builders import (
    CLOCK_ID,
    FUSED_EVIDENCE_ID,
    MAP_ID,
    SEMANTIC_MAP_ID,
    SUMMARY_POLICY,
    make_evidence_links,
    make_hypothesis,
    make_provenance,
    make_semantic_state,
    timestamp,
)
from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import (
    Entity,
    EntityGeometry,
    EntityHypothesis,
    EntityId,
    EntityLifecycle,
    EntityTemporalState,
    GeometrySummaryPolicy,
    ObservationRef,
    SemanticMapId,
    TemporalProvenance,
    summarize_geometry,
)
from contextmap.shared import Vector3

Center = tuple[float, float, float]


def stable_index_base(entity_id: str) -> int:
    """A deterministic first geometry index, so distinct entities never share support."""
    return int(hashlib.sha256(entity_id.encode()).hexdigest()[:8], 16) * 16


def box_corners(center: Center, size: float | Vector3) -> list[Vector3]:
    """The eight corners of an axis-aligned box, so the bounds of the support are exactly it."""
    sizes = (size, size, size) if isinstance(size, int | float) else size
    (cx, cy, cz), (sx, sy, sz) = center, sizes
    return [
        (cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2)
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (-1, 1)
    ]


def geometry_at(
    center: Center,
    *,
    size: float | Vector3 = 0.5,
    map_id: MapId = MAP_ID,
    frame: str = "map",
    first_index: int,
    extra_points: Sequence[Vector3] = (),
) -> EntityGeometry:
    """Summarize a box-shaped support, optionally with extra points, from the real algorithm."""
    points = [*box_corners(center, size), *extra_points]
    indexes = list(range(first_index, first_index + len(points)))
    source = InMemoryGeometrySource(map_id, dict(zip(indexes, points, strict=True)), frame=frame)
    references = tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))
        for index in indexes
    )
    return summarize_geometry(references, source=source, policy=SUMMARY_POLICY)


def temporal_state_at(seconds: Sequence[int], *, clock_id: str = CLOCK_ID) -> EntityTemporalState:
    """A history with one physical observation, interpreted once, per second given."""
    refs = tuple(
        ObservationRef(
            physical_observation_id=SourceObservationId(f"frame-{second:04d}"),
            acquisition_timestamp=timestamp(second, clock_id=clock_id),
            inference_result_count=1,
        )
        for second in sorted(seconds)
    )
    return EntityTemporalState(
        first_seen=refs[0].acquisition_timestamp,
        last_seen=refs[-1].acquisition_timestamp,
        physical_observation_count=len(refs),
        inference_result_count=len(refs),
        observation_refs=refs,
        provenance=TemporalProvenance(
            rule_id="physical-observation-temporal-summary-v1", input_order_chronological=True
        ),
        lifecycle=EntityLifecycle.OBSERVED,
    )


def entity_at(
    entity_id: str,
    center: Center = (0.0, 0.0, 0.0),
    *,
    size: float | Vector3 = 0.5,
    seconds: Sequence[int] = (10, 12),
    map_id: MapId = MAP_ID,
    semantic_map_id: SemanticMapId = SEMANTIC_MAP_ID,
    frame: str = "map",
    clock_id: str = CLOCK_ID,
    first_index: int | None = None,
    extra_points: Sequence[Vector3] = (),
    hypotheses: tuple[EntityHypothesis, ...] | None = None,
) -> Entity:
    """A real entity whose support is a box at ``center`` and which was seen at ``seconds``."""
    base = stable_index_base(entity_id) if first_index is None else first_index
    return Entity(
        entity_id=EntityId(entity_id),
        semantic_map_id=semantic_map_id,
        geometry=geometry_at(
            center,
            size=size,
            map_id=map_id,
            frame=frame,
            first_index=base,
            extra_points=extra_points,
        ),
        semantic_state=make_semantic_state(
            (make_hypothesis(fused_evidence_id=FUSED_EVIDENCE_ID),)
            if hypotheses is None
            else hypotheses
        ),
        evidence=make_evidence_links(
            physical=tuple(f"frame-{second:04d}" for second in sorted(seconds))
        ),
        temporal_state=temporal_state_at(seconds, clock_id=clock_id),
        provenance=make_provenance(),
    )


def lattice(center: Center, size: float | Vector3, *, steps: int = 3) -> list[Vector3]:
    """``steps`` points per axis across a box, so subsets of the points are meaningful supports."""
    sizes = (size, size, size) if isinstance(size, int | float) else size
    axes = [
        [c + (index / (steps - 1) - 0.5) * extent for index in range(steps)]
        for c, extent in zip(center, sizes, strict=True)
    ]
    return [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]


def scene_source(
    points: dict[int, Vector3], *, map_id: MapId = MAP_ID, frame: str = "map"
) -> InMemoryGeometrySource:
    """One geometric map holding every point of a scene, keyed by geometry index."""
    return InMemoryGeometrySource(map_id, points, frame=frame)


def entity_over(
    entity_id: str,
    source: InMemoryGeometrySource,
    indexes: Sequence[int],
    *,
    seconds: Sequence[int] = (10, 12),
    semantic_map_id: SemanticMapId = SEMANTIC_MAP_ID,
    policy: GeometrySummaryPolicy = SUMMARY_POLICY,
) -> Entity:
    """A real entity supported by the given points of a shared scene, so supports can overlap."""
    map_id = source.geometric_map.map_id
    references = tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))
        for index in indexes
    )
    return Entity(
        entity_id=EntityId(entity_id),
        semantic_map_id=semantic_map_id,
        geometry=summarize_geometry(references, source=source, policy=policy),
        semantic_state=make_semantic_state((make_hypothesis(),)),
        evidence=make_evidence_links(
            physical=tuple(f"frame-{second:04d}" for second in sorted(seconds))
        ),
        temporal_state=temporal_state_at(seconds),
        provenance=make_provenance(),
    )
