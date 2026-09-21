"""Deterministic builders for ContextMap schema tests.

Every builder returns a valid object built only from public contracts, so a test states
what it changes instead of repeating the whole map.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from contextmap.artifact import (
    AmbiguityStatus,
    AnchorKind,
    ContextEntity,
    ContextEntityId,
    ContextEntityReference,
    ContextMap,
    ContextMapId,
    ContextMapMetadata,
    ContextRelation,
    ContextRelationId,
    ContextSemanticState,
    DeclaredCapabilities,
    GeometricMapLink,
    Handedness,
    LabelHypothesis,
    LengthUnit,
    MapAnchor,
    MapCapability,
    MapCreation,
    MapFrame,
    ObservationWindow,
    PolicyRef,
    RelationState,
    SourceSequence,
    UpstreamRecordRef,
)
from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import FrameId
from contextmap.shared import SourceTimestamp

CONTEXT_MAP_ID = ContextMapId("context-map--corridor-02--0001")
GEOMETRIC_MAP_ID = MapId("corridor-02--map-run-0001")
SEQUENCE_ARTIFACT_ID = "sequence--corridor-02--e145f73f"
SELECTION_ID = "selection--90s-20frames"
POINT_COUNT = 1_000
MAP_FRAME_ID = "map"
CLOCK_ID = "fixture:header"
ENTITY_RESOLUTION_ARTIFACT_ID = "entity-resolution--run-0001"
SPATIAL_RELATIONS_ARTIFACT_ID = "spatial-relations--run-0001"


def policy(policy_id: str = "context-map-assembly", version: str = "1") -> PolicyRef:
    return PolicyRef(policy_id=policy_id, version=version)


def creation(**overrides: Any) -> MapCreation:
    return replace(
        MapCreation(
            assembly_policy=policy(),
            code_version="a1b2c3d",
            configuration_fingerprint="sha256:cfg-0001",
        ),
        **overrides,
    )


def source_sequence(
    sequence_artifact_id: str = SEQUENCE_ARTIFACT_ID, selection_id: str = SELECTION_ID
) -> SourceSequence:
    return SourceSequence(sequence_artifact_id=sequence_artifact_id, selection_id=selection_id)


def timestamp(seconds: int, nanoseconds: int = 0, *, clock_id: str = CLOCK_ID) -> SourceTimestamp:
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)


def window(**overrides: Any) -> ObservationWindow:
    return replace(ObservationWindow(start=timestamp(100), end=timestamp(190)), **overrides)


def bounds(frame_id: str = MAP_FRAME_ID, **overrides: Any) -> Bounds3D:
    return replace(
        Bounds3D(
            frame_id=FrameId(frame_id), minimum_m=(-10.0, -5.0, 0.0), maximum_m=(10.0, 5.0, 3.0)
        ),
        **overrides,
    )


def estimator_local_anchor(**overrides: Any) -> MapAnchor:
    return replace(
        MapAnchor(
            kind=AnchorKind.ESTIMATOR_LOCAL,
            origin_definition="pose of the first accepted scan of the estimator run",
            reference_frame_id=None,
        ),
        **overrides,
    )


def external_anchor(reference_frame_id: str | None = "site-a/enu", **overrides: Any) -> MapAnchor:
    return replace(
        MapAnchor(
            kind=AnchorKind.EXTERNALLY_ANCHORED,
            origin_definition="surveyed control point of the site",
            reference_frame_id=reference_frame_id,
        ),
        **overrides,
    )


def map_frame(**overrides: Any) -> MapFrame:
    return replace(
        MapFrame(
            frame_id=MAP_FRAME_ID,
            unit=LengthUnit.METER,
            handedness=Handedness.RIGHT_HANDED,
            up_direction=(0.0, 0.0, 1.0),
            anchor=estimator_local_anchor(),
        ),
        **overrides,
    )


def capabilities(**overrides: Any) -> DeclaredCapabilities:
    return replace(
        DeclaredCapabilities(content=(MapCapability.GEOMETRY,), relation_predicates=()),
        **overrides,
    )


def metadata(**overrides: Any) -> ContextMapMetadata:
    return replace(
        ContextMapMetadata(
            creation=creation(),
            source_sequences=(source_sequence(),),
            frame=map_frame(),
            bounds=bounds(),
            time_bounds=window(),
            capabilities=capabilities(),
        ),
        **overrides,
    )


def geometry_link(**overrides: Any) -> GeometricMapLink:
    return replace(GeometricMapLink(map_id=GEOMETRIC_MAP_ID, point_count=POINT_COUNT), **overrides)


def geometry_ref(index: int, *, map_id: MapId = GEOMETRIC_MAP_ID) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


def upstream_record(artifact_id: str, record_id: str) -> UpstreamRecordRef:
    return UpstreamRecordRef(artifact_id=artifact_id, record_id=record_id)


def entity_reference(
    entity_id: str, *, context_map_id: str = CONTEXT_MAP_ID
) -> ContextEntityReference:
    return ContextEntityReference(
        context_map_id=ContextMapId(context_map_id), entity_id=ContextEntityId(entity_id)
    )


def semantic_state(
    status: AmbiguityStatus = AmbiguityStatus.UNAMBIGUOUS, labels: tuple[str, ...] = ("chair",)
) -> ContextSemanticState:
    return ContextSemanticState(
        status=status, hypotheses=tuple(LabelHypothesis(label=label) for label in labels)
    )


def entity(
    entity_id: str = "entity-0001", *, geometry: tuple[int, ...] = (0, 1, 2), **overrides: Any
) -> ContextEntity:
    """Build an entity mapped to the resolved entity of the same local identity."""
    return replace(
        ContextEntity(
            entity_id=ContextEntityId(entity_id),
            source=upstream_record(ENTITY_RESOLUTION_ARTIFACT_ID, f"resolved-{entity_id}"),
            geometry_refs=tuple(geometry_ref(index) for index in geometry),
            semantic_state=semantic_state(),
        ),
        **overrides,
    )


def relation(
    relation_id: str = "relation-0001",
    subject: str = "entity-0001",
    predicate: str = "on",
    object_: str = "entity-0002",
    **overrides: Any,
) -> ContextRelation:
    return replace(
        ContextRelation(
            relation_id=ContextRelationId(relation_id),
            source=upstream_record(SPATIAL_RELATIONS_ARTIFACT_ID, f"source-{relation_id}"),
            subject=entity_reference(subject),
            predicate=predicate,
            object=entity_reference(object_),
            state=RelationState.SUPPORTED,
        ),
        **overrides,
    )


def entity_capabilities(*predicates: str) -> DeclaredCapabilities:
    """Declare geometry and entities, and relations when predicates are given."""
    if not predicates:
        return DeclaredCapabilities(
            content=(MapCapability.ENTITIES, MapCapability.GEOMETRY), relation_predicates=()
        )
    return DeclaredCapabilities(
        content=(MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS),
        relation_predicates=tuple(sorted(set(predicates))),
    )


def context_map(**overrides: Any) -> ContextMap:
    """Build a valid geometry-only map; override the parts a test changes."""
    return replace(
        ContextMap(
            context_map_id=CONTEXT_MAP_ID,
            schema_version="0.1.0",
            metadata=metadata(),
            geometry_ref=geometry_link(),
            entities=(),
            relations=(),
        ),
        **overrides,
    )


def populated_map(**overrides: Any) -> ContextMap:
    """Build a valid map with three entities and two relations, one of them unresolved."""
    entities = (
        entity("entity-0001", geometry=(0, 1, 2)),
        entity(
            "entity-0002",
            geometry=(10, 11),
            semantic_state=semantic_state(AmbiguityStatus.AMBIGUOUS, ("box", "table")),
        ),
        entity(
            "entity-0003",
            geometry=(20,),
            semantic_state=semantic_state(AmbiguityStatus.INSUFFICIENT_EVIDENCE, ()),
        ),
    )
    relations = (
        relation("relation-0001", "entity-0001", "on", "entity-0002"),
        relation(
            "relation-0002",
            "entity-0002",
            "next_to",
            "entity-0003",
            state=RelationState.UNRESOLVED,
        ),
    )
    defaults: dict[str, Any] = {
        "metadata": metadata(capabilities=entity_capabilities("on", "next_to")),
        "entities": entities,
        "relations": relations,
    }
    return context_map(**{**defaults, **overrides})
