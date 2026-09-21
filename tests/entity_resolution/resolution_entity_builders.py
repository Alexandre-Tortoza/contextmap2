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
from contextmap.semantic_fusion import EvidenceContributionId, EvidenceReference
from contextmap.semantic_mapping import (
    AttributeOrigin,
    Entity,
    EntityAttribute,
    EntityGeometry,
    EntityHypothesis,
    EntityId,
    EntityLifecycle,
    EntityTemporalState,
    ObservationRef,
    SemanticMapId,
    TemporalProvenance,
    summarize_geometry,
)
from contextmap.shared import Vector3
from contextmap.visual_perception import ClaimId

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


def temporal_state_at(
    seconds: Sequence[int], *, clock_id: str = CLOCK_ID, inference_results: int = 1
) -> EntityTemporalState:
    """A history with one physical observation per second given, each interpreted N times."""
    refs = tuple(
        ObservationRef(
            physical_observation_id=SourceObservationId(f"frame-{second:04d}"),
            acquisition_timestamp=timestamp(second, clock_id=clock_id),
            inference_result_count=inference_results,
        )
        for second in sorted(seconds)
    )
    return EntityTemporalState(
        first_seen=refs[0].acquisition_timestamp,
        last_seen=refs[-1].acquisition_timestamp,
        physical_observation_count=len(refs),
        inference_result_count=len(refs) * inference_results,
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
    attributes: tuple[EntityAttribute, ...] = (),
    inference_results: int = 1,
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
            else hypotheses,
            attributes=attributes,
        ),
        evidence=make_evidence_links(
            physical=tuple(f"frame-{second:04d}" for second in sorted(seconds))
        ),
        temporal_state=temporal_state_at(
            seconds, clock_id=clock_id, inference_results=inference_results
        ),
        provenance=make_provenance(),
    )


def hypotheses_of(*labels: str) -> tuple[EntityHypothesis, ...]:
    """Competing hypotheses of one fused evidence, one per label, so more than one is ambiguous."""
    return tuple(
        make_hypothesis(f"hypothesis-{index:04d}", label) for index, label in enumerate(labels)
    )


def attribute(
    name: str,
    value: str,
    *,
    origin: AttributeOrigin = AttributeOrigin.OBSERVED,
    derivation_id: str = "observed-attribute-v1",
) -> EntityAttribute:
    """An attribute citing one claim, unless it is external knowledge, which cites nothing."""
    evidence = (
        ()
        if origin is AttributeOrigin.EXTERNAL_KNOWLEDGE
        else (
            EvidenceReference(
                contribution_id=EvidenceContributionId("contribution--support-000001--spatial-a"),
                claim_id=ClaimId("claim-0001"),
            ),
        )
    )
    return EntityAttribute(
        name=name, value=value, origin=origin, derivation_id=derivation_id, evidence=evidence
    )
