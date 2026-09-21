"""Builds the human-readable representative ContextMap fixture from public contracts only.

The fixture is a small map of a corridor: four entities (one merged from two source entities,
one ambiguous with an unresolved neighbor, one conflicting, one with insufficient evidence),
three relations (supported, unresolved and rejected) and the full provenance behind them. Every
identity and digest is synthetic and deterministic. Run this module to regenerate the committed
JSON file::

    python tests/artifact/context_map_fixture_builder.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    SourceSequence,
    UpstreamArtifact,
    UpstreamRecordRef,
    context_map_to_record,
)
from contextmap.entity_resolution import (
    EntityResolutionRunId,
    ResolutionDecisionId,
    ResolvedEntityId,
    ResolvedEntityReference,
)
from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import FrameId
from contextmap.semantic_mapping import EntityId, EntityReference, SemanticMapId
from contextmap.shared import SourceTimestamp
from contextmap.spatial_relations import (
    RelationId,
    RelationPredicate,
    RelationState,
    RelationUncertaintyKind,
    SpatialRelationsRunId,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "context_map" / "corridor.json"

CONTEXT_MAP = "context-map--corridor-02--fixture"
GEOMETRY = "corridor-02--map--fixture"
SEQUENCE = "sequence--corridor-02--fixture"
PERCEPTION = "perception-run--corridor-02--fixture"
FUSION = "semantic-fusion-run--corridor-02--fixture"
SEMANTIC_MAP = "semantic-map--corridor-02--fixture"
POINT_REPRESENTATION = "point-representation-run--corridor-02--fixture"
ENTITY_RESOLUTION = "entity-resolution-run--corridor-02--fixture"
SPATIAL_RELATIONS = "spatial-relations-run--corridor-02--fixture"
CLOCK = "corridor-02:header"


def _digest(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def _record(artifact_id: str, record_id: str) -> UpstreamRecordRef:
    return UpstreamRecordRef(artifact_id=artifact_id, record_id=record_id)


def _origin(
    kind: DerivationKind, *records: UpstreamRecordRef, policy: PolicyRef | None = None
) -> EvidenceOrigin:
    return EvidenceOrigin(kind=kind, derived_from=tuple(sorted(records)), policy=policy)


def _source_entity(name: str) -> EntityReference:
    return EntityReference(semantic_map_id=SemanticMapId(SEMANTIC_MAP), entity_id=EntityId(name))


def _geometry(*indices: int) -> tuple[GeometryReference, ...]:
    map_id = MapId(GEOMETRY)
    return tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))
        for index in indices
    )


def _hypothesis(label: str, *claims: str) -> LabelHypothesis:
    origin = _origin(
        DerivationKind.MODEL_INFERRED, *(_record(PERCEPTION, claim) for claim in claims)
    )
    return LabelHypothesis(label=label, origin=origin)


def _reference(entity_id: str) -> ContextEntityReference:
    return ContextEntityReference(
        context_map_id=ContextMapId(CONTEXT_MAP), entity_id=ContextEntityId(entity_id)
    )


def _fusion_policy() -> PolicyRef:
    return PolicyRef(policy_id="baseline-evidence-accumulation", version="1")


def _relation_policy() -> PolicyRef:
    return PolicyRef(policy_id="geometric-relations", version="1")


def _entity_order(reference: EntityReference) -> tuple[str, str]:
    return (str(reference.semantic_map_id), str(reference.entity_id))


def _entities() -> tuple[ContextEntity, ...]:
    def entity(
        number: int,
        *,
        geometry: tuple[int, ...],
        state: ContextSemanticState,
        members: tuple[str, ...],
        decisions: tuple[str, ...] = (),
        unresolved: tuple[str, ...] = (),
        evidence: tuple[UpstreamRecordRef, ...],
    ) -> ContextEntity:
        identity = f"entity-{number:04d}"
        return ContextEntity(
            entity_id=ContextEntityId(identity),
            source=ResolvedEntityReference(
                resolution_run_id=EntityResolutionRunId(ENTITY_RESOLUTION),
                resolved_entity_id=ResolvedEntityId(f"resolved-{number:04d}"),
            ),
            member_entities=tuple(sorted(map(_source_entity, members), key=_entity_order)),
            resolution_decisions=tuple(sorted(ResolutionDecisionId(item) for item in decisions)),
            unresolved_neighbors=tuple(sorted(map(_source_entity, unresolved), key=_entity_order)),
            geometry_refs=_geometry(*geometry),
            semantic_state=state,
            origin=_origin(DerivationKind.MULTIVIEW_FUSED, *evidence, policy=_fusion_policy()),
        )

    return (
        entity(
            1,
            geometry=(10, 11, 12),
            state=ContextSemanticState(
                status=AmbiguityStatus.UNAMBIGUOUS,
                hypotheses=(_hypothesis("chair", "claim-000120-a", "claim-000135-a"),),
            ),
            # Resolvida a partir de duas entidades de origem: a fusão de identidade é preservada.
            members=("entity-a", "entity-b"),
            decisions=("decision-a-b",),
            evidence=(
                _record(FUSION, "fused-support-0001"),
                _record(FUSION, "fused-support-0002"),
                _record(SEQUENCE, "observation-000120"),
                _record(SEQUENCE, "observation-000135"),
                _record(POINT_REPRESENTATION, "representation-0001"),
            ),
        ),
        entity(
            2,
            geometry=(20, 21, 22),
            state=ContextSemanticState(
                status=AmbiguityStatus.AMBIGUOUS,
                hypotheses=(
                    _hypothesis("desk", "claim-000120-b"),
                    _hypothesis("table", "claim-000135-b"),
                ),
            ),
            members=("entity-c",),
            # A resolução deixou UNRESOLVED contra outra entidade de origem:
            # nem fundida, nem declarada distinta.
            unresolved=("entity-f",),
            evidence=(_record(FUSION, "fused-support-0003"),),
        ),
        entity(
            3,
            geometry=(40, 41),
            state=ContextSemanticState(
                status=AmbiguityStatus.CONFLICTING,
                hypotheses=(
                    _hypothesis("bin", "claim-000120-c"),
                    _hypothesis("box", "claim-000150-c"),
                ),
            ),
            members=("entity-d",),
            evidence=(_record(FUSION, "fused-support-0004"),),
        ),
        entity(
            4,
            geometry=(60,),
            # Abstenção: nenhuma hipótese é afirmada, e isso não é evidência negativa.
            state=ContextSemanticState(status=AmbiguityStatus.INSUFFICIENT_EVIDENCE, hypotheses=()),
            members=("entity-e",),
            evidence=(_record(FUSION, "fused-support-0005"),),
        ),
    )


def _relations() -> tuple[ContextRelation, ...]:
    def relation(
        number: int,
        subject: str,
        predicate: RelationPredicate,
        target: str,
        state: RelationState,
        uncertainty: tuple[RelationUncertaintyKind, ...] = (),
    ) -> ContextRelation:
        return ContextRelation(
            relation_id=ContextRelationId(f"relation-{number:04d}"),
            source_run_id=SpatialRelationsRunId(SPATIAL_RELATIONS),
            source_relation_id=RelationId(f"relation-{number:04d}"),
            subject=_reference(subject),
            predicate=predicate,
            object=_reference(target),
            state=state,
            uncertainty_kinds=uncertainty,
            origin=_origin(
                DerivationKind.GEOMETRY_DERIVED,
                _record(SPATIAL_RELATIONS, f"relation-evidence-{number:04d}"),
                _record(GEOMETRY, geometry_id_for(map_id=MapId(GEOMETRY), index=20)),
                policy=_relation_policy(),
            ),
        )

    return (
        relation(
            1, "entity-0001", RelationPredicate.NEXT_TO, "entity-0002", RelationState.SUPPORTED
        ),
        # Evidência que apoia e contradiz: não resolvida, e o motivo fica registrado.
        relation(
            2,
            "entity-0003",
            RelationPredicate.ON_TOP_OF,
            "entity-0002",
            RelationState.UNRESOLVED,
            (RelationUncertaintyKind.CONFLICTING_EVIDENCE,),
        ),
        # Evidência que contradiz o predicado: rejeitada, mantida como conhecimento negativo.
        relation(3, "entity-0004", RelationPredicate.INSIDE, "entity-0001", RelationState.REJECTED),
    )


def _lineage() -> tuple[UpstreamArtifact, ...]:
    def upstream(
        artifact_id: str, kind: ArtifactKind, models: tuple[str, ...] = ()
    ) -> UpstreamArtifact:
        return UpstreamArtifact(
            artifact_id=artifact_id,
            kind=kind,
            content_identity=_digest(artifact_id),
            configuration_fingerprint=_digest(f"configuration-of-{artifact_id}"),
            code_version="fixture",
            model_identities=models,
        )

    return tuple(
        sorted(
            (
                upstream(SEQUENCE, ArtifactKind.SEQUENCE),
                upstream(
                    PERCEPTION,
                    ArtifactKind.PERCEPTION_RUN,
                    ("qwen2.5-vl-7b@fixture", "sam2-hiera-large@fixture"),
                ),
                upstream(FUSION, ArtifactKind.SEMANTIC_FUSION_RUN),
                upstream(SEMANTIC_MAP, ArtifactKind.SEMANTIC_MAP),
                upstream(GEOMETRY, ArtifactKind.GEOMETRIC_MAP),
                upstream(POINT_REPRESENTATION, ArtifactKind.POINT_REPRESENTATION_RUN),
                upstream(ENTITY_RESOLUTION, ArtifactKind.ENTITY_RESOLUTION_RUN),
                upstream(SPATIAL_RELATIONS, ArtifactKind.SPATIAL_RELATIONS_RUN),
            ),
            key=lambda item: item.artifact_id,
        )
    )


def build_fixture_map() -> ContextMap:
    """Build the representative fixture map from public contracts only."""
    frame_id = "map"
    metadata = ContextMapMetadata(
        creation=MapCreation(
            assembly_policy=PolicyRef(policy_id="context-map-assembly", version="1"),
            code_version="fixture",
            configuration_fingerprint=None,
        ),
        source_sequences=(
            SourceSequence(sequence_artifact_id=SEQUENCE, selection_id="selection--fixture-window"),
        ),
        frame=MapFrame(
            frame_id=frame_id,
            unit=LengthUnit.METER,
            handedness=Handedness.RIGHT_HANDED,
            # Um frame local de estimador não garante que z aponte para cima: desconhecido.
            up_direction=None,
            anchor=MapAnchor(
                kind=AnchorKind.ESTIMATOR_LOCAL,
                origin_definition="first accepted lidar scan of the state-estimation run",
                reference_frame_id=None,
            ),
        ),
        bounds=Bounds3D(
            frame_id=FrameId(frame_id), minimum_m=(-4.0, -2.0, 0.0), maximum_m=(8.0, 2.0, 2.5)
        ),
        time_bounds=ObservationWindow(
            start=SourceTimestamp(seconds=1_700_000_000, nanoseconds=0, clock_id=CLOCK),
            end=SourceTimestamp(seconds=1_700_000_090, nanoseconds=0, clock_id=CLOCK),
        ),
        capabilities=DeclaredCapabilities(
            content=(
                MapCapability.ENTITIES,
                MapCapability.GEOMETRY,
                MapCapability.POINT_REPRESENTATION_EVIDENCE,
                MapCapability.RELATIONS,
            ),
            relation_predicates=(
                RelationPredicate.INSIDE,
                RelationPredicate.NEXT_TO,
                RelationPredicate.ON_TOP_OF,
            ),
        ),
    )
    return ContextMap(
        context_map_id=ContextMapId(CONTEXT_MAP),
        schema_version="0.1.0",
        metadata=metadata,
        geometry_ref=GeometricMapLink(map_id=MapId(GEOMETRY), point_count=120),
        entities=_entities(),
        relations=_relations(),
        lineage=_lineage(),
    )


def render_fixture() -> str:
    """Render the fixture as the exact text committed to the repository."""
    return (
        json.dumps(context_map_to_record(build_fixture_map()), indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(render_fixture(), encoding="utf-8")
    print(f"wrote {FIXTURE_PATH}")
