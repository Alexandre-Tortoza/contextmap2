"""Deterministic assembly of a :class:`ContextMap` from already-decided upstream results.

Entity Resolution and Spatial Relations already decided everything a map composes: which source
entities are one physical object, what that object may be, and how resolved entities relate in
space. This module never resolves an entity and never decides a relation; it *translates* those
finished decisions into the map's own schema (:mod:`contextmap.artifact.composition`), the way
``structural_dependencies.py`` already reads the same two runs to *prove* a map's claims instead
of computing them. Geometry stays referenced: the geometric-map artifact is opened only for its
identity and size, never for its points.

Assembly is intentionally asymmetric between what it copies and what it defers:

* identity, resolution lineage (``member_entities``, ``resolution_decisions``,
  ``unresolved_neighbors``) and geometry references are copied verbatim from
  :class:`~contextmap.entity_resolution.ResolvedEntity`, because :mod:`composition` already
  requires the map's own copy of these fields to equal the upstream record exactly
  (``structural_dependencies.py``'s "prove content, not just existence");
* the semantic state is *translated*, not copied verbatim, because
  :class:`~contextmap.entity_resolution.ResolvedSemanticState` and
  :class:`~contextmap.artifact.ContextSemanticState` are different granularities of the same
  belief (see :func:`_translate_semantic_state`);
* every relation is translated the same way, endpoint by endpoint, through the entities this
  module already composed.

Every entity and relation cites, in its own :class:`~contextmap.artifact.EvidenceOrigin`, the one
upstream record this module actually has open (the resolved entity, or the relation) — never a
perception or semantic-fusion record this module never opens. See
``src/contextmap/artifact/docs/lineage.md`` for the corresponding, deliberate widening of what a
``MULTIVIEW_FUSED`` origin may cite.

This module lives under ``serialization/``, not beside :mod:`contextmap.artifact.composition`,
because it genuinely performs file I/O (opening the Entity Resolution, Spatial Relations and
Geometric Mapping run directories it is given, and reading their manifests to pin each one by
its content digest): exactly the same category of work ``structural_dependencies.py`` already
does from the same place, and exactly what the schema modules (``composition.py``, ``models.py``
and their siblings, checked by ``tests/artifact/test_context_map_invariants.py::
test_the_schema_package_performs_no_file_io_and_owns_no_format``) are not allowed to do. It is
still re-exported at the capability's public root (``contextmap.artifact.assemble_context_map``),
the same way ``ContextMapArtifactWriter`` and ``artifact_digest`` are.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from contextmap.artifact.composition import (
    AmbiguityStatus,
    ContextEntity,
    ContextRelation,
    ContextSemanticState,
    LabelHypothesis,
)
from contextmap.artifact.metadata import ContextMapMetadata, PolicyRef
from contextmap.artifact.models import ContextMap, GeometricMapLink
from contextmap.artifact.provenance import (
    ArtifactKind,
    DerivationKind,
    EvidenceOrigin,
    UpstreamArtifact,
)
from contextmap.artifact.references import (
    ContextEntityId,
    ContextEntityReference,
    ContextMapId,
    ContextRelationId,
    UpstreamRecordRef,
)
from contextmap.artifact.serialization.dependencies import artifact_digest
from contextmap.artifact.serialization.errors import UnresolvedReferenceError, UpstreamArtifactError
from contextmap.artifact.versioning import CONTEXT_MAP_SCHEMA_VERSION
from contextmap.entity_resolution import (
    EntityResolutionRunReader,
    ResolutionRunLineage,
    ResolvedEntityReference,
    ResolvedEntitySet,
    ResolvedSemanticState,
)
from contextmap.entity_resolution import PolicyRef as ResolutionPolicyRef
from contextmap.geometric_mapping import GeometricMapArtifactManifest, GeometricMapArtifactReader
from contextmap.spatial_relations import SpatialRelationsRunReader

__all__ = ["assemble_context_map"]


def assemble_context_map(
    *,
    context_map_id: ContextMapId,
    metadata: ContextMapMetadata,
    geometric_map_location: Path,
    entity_resolution_location: Path | None = None,
    spatial_relations_location: Path | None = None,
    additional_lineage: tuple[UpstreamArtifact, ...] = (),
) -> ContextMap:
    """Compose a validated :class:`ContextMap` from real upstream runs, by reference.

    This is pure composition: it opens each run's own public reader, translates what it already
    decided into the map's schema and pins every located artifact by the digest of its manifest
    (:func:`~contextmap.artifact.artifact_digest`). It never performs entity resolution or
    relation inference, and it never copies geometry: ``geometric_map_location`` is opened only
    for its identity and point count.

    Args:
        context_map_id: Identity of the map being assembled; unique per immutable artifact.
        metadata: What the map is, where it came from and how it was created — including the
            explicit, versioned assembly policy (``metadata.creation.assembly_policy``). Nothing
            here is derived from Entity Resolution or Spatial Relations, so this function never
            invents a frame, a time window or a declared capability on the caller's behalf; a
            declaration that disagrees with what is composed is rejected by
            :class:`~contextmap.artifact.ContextMap` itself.
        geometric_map_location: Directory of the immutable geometric-map artifact the map
            references. Required: every map has geometry.
        entity_resolution_location: Directory of the Entity Resolution run whose resolved
            entities become this map's entities; ``None`` for a geometry-only map.
        spatial_relations_location: Directory of the Spatial Relations run whose relations
            become this map's relations; ``None`` when the map declares no relations. Requires
            ``entity_resolution_location``, since a relation names entities.
        additional_lineage: Upstream artifacts this function itself never opens but that
            ``metadata`` still requires in the lineage. Every ``SEQUENCE`` artifact named by
            ``metadata.source_sequences`` is the one this always needs: pinning it needs the
            ingestion artifact's own digest, and ``contextmap.artifact`` has no dependency on
            ``contextmap.ingestion`` to compute one (see ``tests/architecture/test_boundaries.
            py``). An optional evidence artifact (a perception run, a point-representation run, a
            human-annotation set) that some other ``origin`` this function did not build might
            cite is another. The one structural dependency this function *can* derive for free
            from the Entity Resolution manifest it already opened — the ``SEMANTIC_MAP`` of every
            ``member_entities``/``unresolved_neighbors`` reference — is computed by this function
            itself and must not be repeated here. Merged with the geometry, entity-resolution,
            semantic-map and spatial-relations entries this function does compute; a duplicate
            ``artifact_id`` is rejected by :class:`~contextmap.artifact.ContextMap` itself, never
            silently overridden.

    Returns:
        A :class:`ContextMap` whose own construction has already validated every reference,
        the provenance closure and the declared capabilities. Not yet written to disk: persisting
        it is :class:`~contextmap.artifact.ContextMapArtifactWriter`'s job.

    Raises:
        ValueError: If ``spatial_relations_location`` is given without
            ``entity_resolution_location``.
        UpstreamArtifactError: If the Entity Resolution or Spatial Relations run was built over a
            different geometric map than ``geometric_map_location``, or the Spatial Relations run
            was built over a different Entity Resolution run than ``entity_resolution_location``
            (a dangling or swapped structural dependency).
        UnresolvedReferenceError: If a relation names a resolved entity that is not among the
            entities this run actually has (defensive: :meth:`SpatialRelationsRunReader.
            validate_resolution` already reports this as an upstream-mismatch problem first).
    """
    if spatial_relations_location is not None and entity_resolution_location is None:
        raise ValueError(
            "spatial_relations_location requires entity_resolution_location: a relation names "
            "entities, and a map cannot declare relations without entities"
        )

    with GeometricMapArtifactReader(geometric_map_location) as geometry_reader:
        geometry_manifest = geometry_reader.manifest
    geometry_ref = GeometricMapLink(
        map_id=geometry_manifest.map_id, point_count=geometry_manifest.point_count
    )
    lineage = [
        _geometry_upstream_artifact(geometry_manifest, geometric_map_location),
        *additional_lineage,
    ]

    entities: tuple[ContextEntity, ...] = ()
    relations: tuple[ContextRelation, ...] = ()
    if entity_resolution_location is not None:
        resolution_reader = EntityResolutionRunReader(entity_resolution_location)
        _require_same_geometric_map(
            declared=resolution_reader.manifest.lineage.geometric_map_id,
            referenced=geometry_manifest.map_id,
            owner=f"entity resolution run {resolution_reader.run_id!r}",
        )
        materialization = resolution_reader.materialization()
        entities, identities = _translate_entities(
            context_map_id, resolution_reader.resolved_entities(), materialization.policy
        )
        lineage.append(_resolution_upstream_artifact(resolution_reader, entity_resolution_location))
        lineage.extend(_semantic_map_upstream_artifacts(resolution_reader.manifest.lineage))

        if spatial_relations_location is not None:
            relations_reader = SpatialRelationsRunReader(spatial_relations_location)
            _require_same_geometric_map(
                declared=relations_reader.manifest.lineage.geometric_map_id,
                referenced=geometry_manifest.map_id,
                owner=f"spatial relations run {relations_reader.manifest.run_id!r}",
            )
            problems = relations_reader.validate_resolution(resolution_reader)
            if problems:
                raise UpstreamArtifactError(
                    f"spatial relations run {relations_reader.manifest.run_id!r} does not match "
                    f"entity resolution run {resolution_reader.run_id!r} as its own lineage "
                    "records: " + "; ".join(problems)
                )
            relations = _translate_relations(relations_reader, identities)
            lineage.append(
                _relations_upstream_artifact(relations_reader, spatial_relations_location)
            )

    return ContextMap(
        context_map_id=context_map_id,
        schema_version=CONTEXT_MAP_SCHEMA_VERSION,
        metadata=metadata,
        geometry_ref=geometry_ref,
        entities=entities,
        relations=relations,
        lineage=tuple(sorted(lineage, key=lambda item: item.artifact_id)),
    )


def _require_same_geometric_map(*, declared: object, referenced: object, owner: str) -> None:
    """Reject a run built over a geometric map other than the one this map references.

    Um `ResolvedEntity`/`Relation` que aponta para outro mapa geométrico nunca é composto
    silenciosamente: a montagem falha aqui, antes de qualquer referência de geometria ser
    copiada para uma entidade.
    """
    if declared != referenced:
        raise UpstreamArtifactError(
            f"{owner} was built over geometric map {declared!r}, not the referenced {referenced!r}"
        )


def _geometry_upstream_artifact(
    manifest: GeometricMapArtifactManifest, location: Path
) -> UpstreamArtifact:
    return UpstreamArtifact(
        artifact_id=str(manifest.map_id),
        kind=ArtifactKind.GEOMETRIC_MAP,
        content_identity=artifact_digest(location),
        configuration_fingerprint=manifest.configuration_fingerprint,
        code_version=manifest.code_version,
        model_identities=(),
    )


def _resolution_upstream_artifact(
    reader: EntityResolutionRunReader, location: Path
) -> UpstreamArtifact:
    return UpstreamArtifact(
        artifact_id=str(reader.run_id),
        kind=ArtifactKind.ENTITY_RESOLUTION_RUN,
        content_identity=artifact_digest(location),
        # A leitura pública do run não expõe uma impressão digital única de configuração
        # efetiva (só políticas por papel); nada aqui é inventado para preencher o campo.
        configuration_fingerprint=None,
        code_version=reader.manifest.code_version,
        model_identities=(),
    )


def _semantic_map_upstream_artifacts(lineage: ResolutionRunLineage) -> tuple[UpstreamArtifact, ...]:
    """Derive the ``SEMANTIC_MAP`` lineage entry of every semantic map an entity may reference.

    Unlike ``SEQUENCE``, this one is free: the Entity Resolution manifest this module already
    opened names the exact semantic-mapping run it was built from and that run's own pinned
    digest (:func:`~contextmap.entity_resolution.mapping_artifact_digest`, computed once by
    Semantic Mapping's writer and carried unchanged since). No extra reader is opened.

    A resolution run built the ordinary way (:func:`~contextmap.entity_resolution.
    lineage_from_mapping_manifest`) names exactly one semantic-mapping run, so
    ``semantic_map_ids`` has exactly one entry in practice and every entry gets the same digest
    honestly; ``ResolutionRunLineage`` has no way to record a different digest per semantic map,
    so this function cannot invent one either if a lineage were ever built with more than one.
    """
    return tuple(
        UpstreamArtifact(
            artifact_id=str(semantic_map_id),
            kind=ArtifactKind.SEMANTIC_MAP,
            content_identity=lineage.semantic_mapping_artifact_digest,
            configuration_fingerprint=None,
            code_version=None,
            model_identities=(),
        )
        for semantic_map_id in lineage.semantic_map_ids
    )


def _relations_upstream_artifact(
    reader: SpatialRelationsRunReader, location: Path
) -> UpstreamArtifact:
    return UpstreamArtifact(
        artifact_id=str(reader.manifest.run_id),
        kind=ArtifactKind.SPATIAL_RELATIONS_RUN,
        content_identity=artifact_digest(location),
        configuration_fingerprint=None,
        code_version=reader.manifest.code_version,
        model_identities=(),
    )


def _translate_resolution_policy(policy: ResolutionPolicyRef) -> PolicyRef:
    """Translate Entity Resolution's ``(policy_id, configuration_fingerprint)`` into the map's.

    Entity Resolution's own :class:`~contextmap.entity_resolution.PolicyRef` has no separate
    ``version``: the rule's version is a suffix of ``policy_id`` by that capability's own
    convention (for example ``connected-components-materialization-v1``), and
    ``configuration_fingerprint`` is the per-execution effective configuration. The map's
    :class:`~contextmap.artifact.PolicyRef` requires an explicit ``version`` ("a different
    version is a different rule"), so ``configuration_fingerprint`` is carried over as it: it is
    the only signal Entity Resolution's own contract records that actually distinguishes one
    execution of the rule from another, and nothing is fabricated to fill the field.
    """
    return PolicyRef(policy_id=policy.policy_id, version=policy.configuration_fingerprint)


def _translate_entities(
    context_map_id: ContextMapId,
    resolved: ResolvedEntitySet,
    materialization_policy: ResolutionPolicyRef,
) -> tuple[tuple[ContextEntity, ...], dict[ResolvedEntityReference, ContextEntityReference]]:
    """Translate every resolved entity of a run into a ``ContextEntity``, in canonical order.

    ``resolved.entities`` is already sorted and unique by ``resolved_entity_id``
    (:class:`~contextmap.entity_resolution.ResolvedEntitySet` requires it at construction), a
    hash of the members and therefore independent of how the run's own table happens to be
    stored or iterated. Local identities are assigned in that order, so two assemblies of the
    same run always number entities the same way.
    """
    policy = _translate_resolution_policy(materialization_policy)
    width = max(4, len(str(len(resolved.entities))))
    entities: list[ContextEntity] = []
    identities: dict[ResolvedEntityReference, ContextEntityReference] = {}
    for index, item in enumerate(resolved.entities, start=1):
        local_id = ContextEntityId(f"entity-{index:0{width}d}")
        record = UpstreamRecordRef(
            artifact_id=str(item.resolution_run_id), record_id=str(item.resolved_entity_id)
        )
        origin = EvidenceOrigin(
            kind=DerivationKind.MULTIVIEW_FUSED, derived_from=(record,), policy=policy
        )
        entities.append(
            ContextEntity(
                entity_id=local_id,
                source=item.reference,
                member_entities=item.member_entity_refs,
                resolution_decisions=item.resolution_decision_refs,
                unresolved_neighbors=item.unresolved_neighbor_refs,
                geometry_refs=item.geometry.geometry_refs,
                semantic_state=_translate_semantic_state(item.semantic_state, origin=origin),
                origin=origin,
            )
        )
        identities[item.reference] = ContextEntityReference(
            context_map_id=context_map_id, entity_id=local_id
        )
    return tuple(entities), identities


def _translate_semantic_state(
    state: ResolvedSemanticState, *, origin: EvidenceOrigin
) -> ContextSemanticState:
    """Translate a resolved entity's belief into the map's semantic-state schema.

    This is the one genuinely lossy step of assembly, and it is lossy by the target schema's own
    design, not by an assembly shortcut: ``ResolvedSemanticState.hypotheses`` is keyed by
    ``(fused_evidence_id, hypothesis_id)`` and may hold two entries with the *same* label text
    (independent evidence that happens to agree), while :class:`LabelHypothesis` is keyed by
    label text alone (``ContextSemanticState`` refuses two hypotheses with the same label). Two
    items that agree on a label are reinforcing evidence for one candidate, not two competing
    ones, so collapsing them by label is not a flattening of *ambiguity* — ambiguity is exactly
    the presence of more than one *distinct* label, which this function still preserves in full:
    every distinct label proposed by any member survives as its own hypothesis, and
    ``ambiguity_state`` is carried over unchanged (``AmbiguityState`` and ``AmbiguityStatus`` are
    the same four values). Neither a score nor a hypothesis count is invented for the merge: the
    schema has no field for either, and the evidence behind each label stays reachable through
    ``origin`` and the entity-resolution run it cites, never re-derived here.

    ``attributes`` and the detailed ``uncertainty`` records of ``ResolvedSemanticState`` have no
    corresponding field in ``ContextSemanticState`` (composition.py's own module docstring:
    "Support values are deliberately not copied; the evidence behind a result stays reachable
    through the upstream artifact"), so they are not copied here either; a consumer that needs
    them opens the entity-resolution run this entity's ``origin`` and ``source`` both cite.
    """
    labels = sorted({hypothesis.label for hypothesis in state.hypotheses})
    hypotheses = tuple(LabelHypothesis(label=label, origin=origin) for label in labels)
    status = AmbiguityStatus(state.ambiguity_state.value)
    return ContextSemanticState(status=status, hypotheses=hypotheses)


def _translate_relations(
    reader: SpatialRelationsRunReader,
    identities: Mapping[ResolvedEntityReference, ContextEntityReference],
) -> tuple[ContextRelation, ...]:
    """Translate every relation of a run into a ``ContextRelation``, in canonical order.

    ``iter_relations()`` yields relations in the run's own canonical order (subject, predicate,
    object), which has no relationship to ``relation_id`` (a digest of the same triple): sorting
    by ``relation_id`` here is what actually makes the map's own numbering independent of that
    reader order, not a no-op.
    """
    ordered = sorted(reader.iter_relations(), key=lambda item: str(item.relation_id))
    width = max(4, len(str(len(ordered))))
    run_id = reader.manifest.run_id
    relations: list[ContextRelation] = []
    for index, relation in enumerate(ordered, start=1):
        try:
            subject = identities[relation.subject_entity_ref]
            object_ = identities[relation.object_entity_ref]
        except KeyError as error:
            raise UnresolvedReferenceError(
                f"relation {relation.relation_id!r} of run {run_id!r} references resolved "
                f"entity {error.args[0]!r}, which entity resolution does not have"
            ) from None
        policy = PolicyRef(
            policy_id=relation.provenance.decision_policy_id,
            version=relation.provenance.taxonomy_version,
        )
        origin = EvidenceOrigin(
            kind=DerivationKind.GEOMETRY_DERIVED,
            derived_from=(
                UpstreamRecordRef(artifact_id=str(run_id), record_id=str(relation.relation_id)),
            ),
            policy=policy,
        )
        relations.append(
            ContextRelation(
                relation_id=ContextRelationId(f"relation-{index:0{width}d}"),
                source_run_id=run_id,
                source_relation_id=relation.relation_id,
                subject=subject,
                predicate=relation.predicate,
                object=object_,
                state=relation.state,
                uncertainty_kinds=tuple(
                    sorted(
                        {item.kind for item in relation.uncertainty}, key=lambda kind: kind.value
                    )
                ),
                origin=origin,
            )
        )
    return tuple(relations)
