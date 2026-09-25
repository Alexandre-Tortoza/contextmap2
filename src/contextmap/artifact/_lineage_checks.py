"""Provenance-closure invariants of a ContextMap, checked at construction.

The map lists every upstream artifact it cites in its lineage. These checks require that every
reference in the map resolves to that table with the right kind, that each origin is honest
about what it was derived from, and that the declared capabilities are backed by the artifacts
that produce them. Nothing is skipped, defaulted or repaired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from contextmap.artifact._checks import require_canonical
from contextmap.artifact.composition import ContextEntity, ContextRelation
from contextmap.artifact.metadata import DeclaredCapabilities, MapCapability, SourceSequence
from contextmap.artifact.provenance import (
    ArtifactKind,
    DerivationKind,
    EvidenceOrigin,
    ProvenanceError,
    UpstreamArtifact,
)
from contextmap.artifact.references import ReferenceIntegrityError
from contextmap.geometric_mapping import MapId

_CAPABILITY_ARTIFACT = {
    MapCapability.ENTITIES: ArtifactKind.ENTITY_RESOLUTION_RUN,
    MapCapability.RELATIONS: ArtifactKind.SPATIAL_RELATIONS_RUN,
    MapCapability.POINT_REPRESENTATION_EVIDENCE: ArtifactKind.POINT_REPRESENTATION_RUN,
}

_SENSOR_GROUNDING = frozenset({ArtifactKind.SEQUENCE, ArtifactKind.GEOMETRIC_MAP})

# Pelo menos um artifact citado precisa ser de um destes tipos para a categoria ser honesta.
_GROUNDING: Mapping[DerivationKind, frozenset[ArtifactKind]] = {
    DerivationKind.SENSOR_OBSERVED: _SENSOR_GROUNDING,
    DerivationKind.MODEL_INFERRED: frozenset({ArtifactKind.PERCEPTION_RUN}),
    DerivationKind.GEOMETRY_DERIVED: frozenset(
        {ArtifactKind.GEOMETRIC_MAP, ArtifactKind.SPATIAL_RELATIONS_RUN}
    ),
    # SEMANTIC_FUSION_RUN é a fusão multi-vista de uma entidade de origem; ENTITY_RESOLUTION_RUN
    # é acrescentado porque a materialização de Entity Resolution também acumula, por uma regra
    # versionada (a política de materialização), várias contribuições de evidência — as dos
    # membros que fundiu — e é o único artifact que a montagem do ContextMap (que nunca abre um
    # run de Semantic Fusion) tem de fato aberto para citar essa derivação com honestidade.
    DerivationKind.MULTIVIEW_FUSED: frozenset(
        {ArtifactKind.SEMANTIC_FUSION_RUN, ArtifactKind.ENTITY_RESOLUTION_RUN}
    ),
    DerivationKind.HUMAN_ANNOTATED: frozenset({ArtifactKind.HUMAN_ANNOTATION_SET}),
}


def check_lineage(
    lineage: Sequence[UpstreamArtifact],
    *,
    geometry_map_id: MapId,
    source_sequences: Sequence[SourceSequence],
    capabilities: DeclaredCapabilities,
    entities: Sequence[ContextEntity],
    relations: Sequence[ContextRelation],
) -> dict[str, UpstreamArtifact]:
    """Validate the provenance closure of a map.

    Args:
        lineage: The upstream artifacts the map cites, expected sorted by id and unique.
        geometry_map_id: The geometric-map artifact the map references.
        source_sequences: The sequence selections the map was built from.
        capabilities: The capabilities the metadata declares.
        entities: The entities of the map.
        relations: The relations of the map.

    Returns:
        The lineage by artifact id.

    Raises:
        ValueError: If the lineage is not sorted and unique, or a declared capability is not
            backed by its artifact, or the reverse.
        ReferenceIntegrityError: If a cited artifact is not listed or is listed with the wrong
            kind.
        ProvenanceError: If an origin claims a derivation its evidence does not support.
    """
    require_canonical("lineage", lineage, lambda item: (item.artifact_id,), detail="by id ")
    by_id = {item.artifact_id: item for item in lineage}

    _require_kind(by_id, geometry_map_id, ArtifactKind.GEOMETRIC_MAP, "geometry_ref")
    for sequence in source_sequences:
        _require_kind(
            by_id, sequence.sequence_artifact_id, ArtifactKind.SEQUENCE, "source_sequences"
        )
    for entity in entities:
        _check_entity(entity, by_id)
    for relation in relations:
        _check_relation(relation, by_id)
    _check_capabilities_backed_by_lineage(capabilities, lineage)
    return by_id


def _check_entity(entity: ContextEntity, by_id: Mapping[str, UpstreamArtifact]) -> None:
    owner = f"entity {entity.entity_id!r}"
    _require_kind(
        by_id,
        entity.source.resolution_run_id,
        ArtifactKind.ENTITY_RESOLUTION_RUN,
        f"{owner} source",
    )
    for name, references in (
        ("member_entities", entity.member_entities),
        ("unresolved_neighbors", entity.unresolved_neighbors),
    ):
        for reference in references:
            _require_kind(
                by_id, reference.semantic_map_id, ArtifactKind.SEMANTIC_MAP, f"{owner} {name}"
            )
    _check_origin(entity.origin, by_id, owner=owner)
    for hypothesis in entity.semantic_state.hypotheses:
        _check_origin(hypothesis.origin, by_id, owner=f"{owner} hypothesis {hypothesis.label!r}")


def _check_relation(relation: ContextRelation, by_id: Mapping[str, UpstreamArtifact]) -> None:
    owner = f"relation {relation.relation_id!r}"
    _require_kind(
        by_id, relation.source_run_id, ArtifactKind.SPATIAL_RELATIONS_RUN, f"{owner} source"
    )
    if relation.origin.kind is DerivationKind.SENSOR_OBSERVED:
        raise ProvenanceError(
            f"{owner} cannot be SENSOR_OBSERVED: a relation is derived, never a direct sensor "
            f"observation"
        )
    _check_origin(relation.origin, by_id, owner=owner)


def _check_capabilities_backed_by_lineage(
    capabilities: DeclaredCapabilities, lineage: Sequence[UpstreamArtifact]
) -> None:
    """Require that a capability is declared exactly when its producing artifact is cited.

    Args:
        capabilities: The capabilities the metadata declares.
        lineage: The upstream artifacts the map cites.

    Raises:
        ValueError: If a capability is declared without its artifact, or the artifact is cited
            without the capability being declared.
    """
    for capability, kind in _CAPABILITY_ARTIFACT.items():
        declared = capability in capabilities.content
        listed = any(item.kind is kind for item in lineage)
        if declared != listed:
            raise ValueError(
                f"{capability.name} is {'declared' if declared else 'not declared'}, but the "
                f"lineage {'lacks' if declared else 'lists'} a {kind.name} artifact"
            )


def _require_kind(
    by_id: Mapping[str, UpstreamArtifact], artifact_id: str, kind: ArtifactKind, what: str
) -> None:
    """Require that an artifact is listed in the lineage with the expected kind.

    Args:
        by_id: The lineage by artifact id.
        artifact_id: The artifact a record names.
        kind: The kind the record needs it to be.
        what: What names the artifact, for the error message.

    Raises:
        ReferenceIntegrityError: If the artifact is not listed or has another kind.
    """
    listed = by_id.get(artifact_id)
    if listed is None:
        raise ReferenceIntegrityError(
            f"{what} names artifact {artifact_id!r}, which is not in the lineage "
            f"(expected a {kind.name} artifact)"
        )
    if listed.kind is not kind:
        raise ReferenceIntegrityError(
            f"{what} names artifact {artifact_id!r} as a {kind.name} artifact, but the lineage "
            f"lists it as {listed.kind.name}"
        )


def _check_origin(
    origin: EvidenceOrigin, by_id: Mapping[str, UpstreamArtifact], *, owner: str
) -> None:
    """Validate that an origin is honest about what it was derived from.

    The category never overrides the evidence: a model output cannot be called observed because
    it consumed an image, and an annotation cannot enter as anything but an annotation.

    Args:
        origin: The origin to validate.
        by_id: The lineage by artifact id.
        owner: What carries the origin, for the error message.

    Raises:
        ReferenceIntegrityError: If a cited artifact is not in the lineage.
        ProvenanceError: If the category is not supported by the artifacts cited.
    """
    cited: list[UpstreamArtifact] = []
    for reference in origin.derived_from:
        listed = by_id.get(reference.artifact_id)
        if listed is None:
            raise ReferenceIntegrityError(
                f"{owner} cites artifact {reference.artifact_id!r}, which is not in the lineage"
            )
        cited.append(listed)
    kinds = {item.kind for item in cited}
    name = origin.kind.name

    if (
        ArtifactKind.HUMAN_ANNOTATION_SET in kinds
        and origin.kind is not DerivationKind.HUMAN_ANNOTATED
    ):
        raise ProvenanceError(
            f"{owner} is {name} but cites a human annotation set: annotations enter only as a "
            f"HUMAN_ANNOTATED origin"
        )
    grounding = _GROUNDING.get(origin.kind)
    if grounding is not None and not kinds & grounding:
        expected = ", ".join(sorted(kind.name for kind in grounding))
        raise ProvenanceError(f"{owner} is {name} but cites no {expected} artifact")
    if origin.kind is DerivationKind.SENSOR_OBSERVED and not kinds <= _SENSOR_GROUNDING:
        stray = ", ".join(sorted(kind.name for kind in kinds - _SENSOR_GROUNDING))
        raise ProvenanceError(
            f"{owner} is SENSOR_OBSERVED but also cites {stray}: a model output is not a direct "
            f"sensor observation"
        )
    if origin.kind is DerivationKind.MODEL_INFERRED and not any(
        item.kind is ArtifactKind.PERCEPTION_RUN and item.model_identities for item in cited
    ):
        raise ProvenanceError(
            f"{owner} is MODEL_INFERRED but the inference artifact it cites names no model"
        )
