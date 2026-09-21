"""Deterministic builders for ContextMap schema tests.

Every builder returns a valid object built only from public contracts, so a test states
what it changes instead of repeating the whole map.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from contextmap.artifact import (
    AmbiguityStatus,
    AnchorKind,
    ArtifactKind,
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
    DerivationKind,
    EvidenceOrigin,
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
    UpstreamArtifact,
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
PERCEPTION_ARTIFACT_ID = "perception-run--0001"
FUSION_ARTIFACT_ID = "semantic-fusion--run-0001"
SEMANTIC_MAP_ARTIFACT_ID = "semantic-map--0001"
POINT_REPRESENTATION_ARTIFACT_ID = "point-representation--run-0001"
ANNOTATION_ARTIFACT_ID = "human-annotations--0001"
MODEL_IDENTITIES = ("qwen2.5-vl-7b@rev-2", "sam2-hiera-large@rev-1")


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


def digest(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def upstream_artifact(artifact_id: str, kind: ArtifactKind, **overrides: Any) -> UpstreamArtifact:
    return replace(
        UpstreamArtifact(
            artifact_id=artifact_id,
            kind=kind,
            content_identity=digest(artifact_id),
            configuration_fingerprint=digest(f"configuration-of-{artifact_id}"),
            code_version="a1b2c3d",
            model_identities=(),
        ),
        **overrides,
    )


def default_lineage(capabilities: DeclaredCapabilities) -> tuple[UpstreamArtifact, ...]:
    """List the upstream artifacts a map with these capabilities cites, sorted by id."""
    items = [
        upstream_artifact(SEQUENCE_ARTIFACT_ID, ArtifactKind.SEQUENCE),
        upstream_artifact(str(GEOMETRIC_MAP_ID), ArtifactKind.GEOMETRIC_MAP),
    ]
    if MapCapability.ENTITIES in capabilities.content:
        items += [
            upstream_artifact(
                PERCEPTION_ARTIFACT_ID,
                ArtifactKind.PERCEPTION_RUN,
                model_identities=MODEL_IDENTITIES,
            ),
            upstream_artifact(FUSION_ARTIFACT_ID, ArtifactKind.SEMANTIC_FUSION_RUN),
            upstream_artifact(SEMANTIC_MAP_ARTIFACT_ID, ArtifactKind.SEMANTIC_MAP),
            upstream_artifact(ENTITY_RESOLUTION_ARTIFACT_ID, ArtifactKind.ENTITY_RESOLUTION_RUN),
        ]
    if MapCapability.RELATIONS in capabilities.content:
        items.append(
            upstream_artifact(SPATIAL_RELATIONS_ARTIFACT_ID, ArtifactKind.SPATIAL_RELATIONS_RUN)
        )
    if MapCapability.POINT_REPRESENTATION_EVIDENCE in capabilities.content:
        items.append(
            upstream_artifact(
                POINT_REPRESENTATION_ARTIFACT_ID, ArtifactKind.POINT_REPRESENTATION_RUN
            )
        )
    return tuple(sorted(items, key=lambda item: item.artifact_id))


def origin(
    kind: DerivationKind, *records: UpstreamRecordRef, policy_ref: PolicyRef | None = None
) -> EvidenceOrigin:
    return EvidenceOrigin(kind=kind, derived_from=tuple(sorted(records)), policy=policy_ref)


def fused_origin(seed: str) -> EvidenceOrigin:
    return origin(
        DerivationKind.MULTIVIEW_FUSED,
        upstream_record(FUSION_ARTIFACT_ID, f"fused-{seed}"),
        policy_ref=policy("baseline-evidence-accumulation"),
    )


def inferred_origin(seed: str) -> EvidenceOrigin:
    return origin(
        DerivationKind.MODEL_INFERRED, upstream_record(PERCEPTION_ARTIFACT_ID, f"claim-{seed}")
    )


def geometric_origin(seed: str) -> EvidenceOrigin:
    return origin(
        DerivationKind.GEOMETRY_DERIVED,
        upstream_record(SPATIAL_RELATIONS_ARTIFACT_ID, f"evidence-{seed}"),
        upstream_record(str(GEOMETRIC_MAP_ID), f"geometry-{seed}"),
        policy_ref=policy("geometric-relations"),
    )


def hypothesis(label: str, origin_: EvidenceOrigin | None = None) -> LabelHypothesis:
    return LabelHypothesis(label=label, origin=origin_ or inferred_origin(label))


def semantic_state(
    status: AmbiguityStatus = AmbiguityStatus.UNAMBIGUOUS, labels: tuple[str, ...] = ("chair",)
) -> ContextSemanticState:
    return ContextSemanticState(
        status=status, hypotheses=tuple(hypothesis(label) for label in labels)
    )


def entity(
    entity_id: str = "entity-0001", *, geometry: tuple[int, ...] = (0, 1, 2), **overrides: Any
) -> ContextEntity:
    """Build an entity mapped to the resolved entity of the same local identity."""
    return replace(
        ContextEntity(
            entity_id=ContextEntityId(entity_id),
            source=upstream_record(ENTITY_RESOLUTION_ARTIFACT_ID, f"resolved-{entity_id}"),
            member_entities=(upstream_record(SEMANTIC_MAP_ARTIFACT_ID, f"semantic-{entity_id}"),),
            resolution_decisions=(),
            geometry_refs=tuple(geometry_ref(index) for index in geometry),
            semantic_state=semantic_state(),
            origin=fused_origin(entity_id),
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
            origin=geometric_origin(relation_id),
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
    """Build a valid geometry-only map; override the parts a test changes.

    The lineage follows the declared capabilities unless a test overrides it.
    """
    fields: dict[str, Any] = {
        "context_map_id": CONTEXT_MAP_ID,
        "schema_version": "0.1.0",
        "metadata": metadata(),
        "geometry_ref": geometry_link(),
        "entities": (),
        "relations": (),
    }
    fields.update(overrides)
    fields.setdefault("lineage", default_lineage(fields["metadata"].capabilities))
    return ContextMap(**fields)


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
