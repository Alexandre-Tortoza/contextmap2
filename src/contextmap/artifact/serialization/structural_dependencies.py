"""Typed validation of the structural upstream dependencies of a ContextMap.

Pinning an upstream artifact by the digest of its manifest
(:func:`~contextmap.artifact.serialization.dependencies.artifact_digest`) proves that the bytes
found are exactly the artifact the digest was computed from. It does not prove that the artifact
is the *one the map's own records need*: ``artifact_id`` and ``content_identity`` are independent
fields of an :class:`~contextmap.artifact.UpstreamArtifact`, so a lineage entry can declare an
``artifact_id`` that disagrees with the digest's real artifact, and nothing about the digest check
alone notices that a Spatial Relations run was produced over a different Entity Resolution run
than the one the map's entities actually come from, or that a particular resolved entity or
relation the map cites does not exist upstream at all.

This module opens the two structural runs with their own public readers and checks what only
they can tell: their own identity (``run_id``) against what the map's lineage declares for them,
the geometric map they were themselves built over against the map's own ``geometry_ref``, that the
Spatial Relations run was built over the very Entity Resolution run the map also cites (via its
own :meth:`~contextmap.spatial_relations.SpatialRelationsRunReader.validate_resolution`), and that
every ``ContextEntity.source`` and every ``ContextRelation.source_relation_id`` the map lists
actually resolves in the corresponding run. It is used identically by the writer, before
publishing, and by the FULL validator, before a report may say ``VERIFIED``, so a dangling or
swapped structural dependency is refused the same way in both places. It has no public consumer
outside ``contextmap.artifact.serialization`` and is not exported at the package root.

Both readers load their tables lazily, so a corrupted upstream file is not necessarily caught by
opening the reader: it only surfaces once the table it damages is actually read. Every call that
reaches into a run's own tables (``resolved_entity``, ``relation``, ``validate_resolution``) is
therefore checked the same way construction is, and a run that cannot even be read to answer one
of these questions is reported as a problem exactly like a dangling or swapped reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from contextmap.artifact.models import ContextMap
from contextmap.entity_resolution import (
    EntityResolutionRunReader,
    ForeignResolvedEntityReferenceError,
    IncompleteRunArtifactError,
    RunArtifactError,
    UnknownResolvedEntityError,
)
from contextmap.spatial_relations import (
    IncompleteRelationsRunArtifactError,
    RelationsRunArtifactError,
    SpatialRelationsRunReader,
)


def check_structural_dependencies(
    context_map: ContextMap,
    *,
    entity_resolution_locations: Mapping[str, Path],
    spatial_relations_locations: Mapping[str, Path],
) -> tuple[str, ...]:
    """Check the Entity Resolution and Spatial Relations runs the map's own records need.

    Args:
        context_map: The map whose entities and relations are checked.
        entity_resolution_locations: Where each cited Entity Resolution run is, keyed by the
            ``artifact_id`` the map's lineage declares for it.
        spatial_relations_locations: Where each cited Spatial Relations run is, keyed the same
            way.

    Returns:
        Human-readable problems, empty when the map's structural claims are fully upheld. Never
        raises: an artifact that cannot even be opened is itself reported as a problem, exactly
        like a mismatched or dangling reference.
    """
    problems: list[str] = []
    resolution_readers = _open_resolution_runs(context_map, entity_resolution_locations, problems)
    _check_entities(context_map, resolution_readers, problems)
    relations_readers = _open_relations_runs(
        context_map, spatial_relations_locations, resolution_readers, problems
    )
    _check_relations(context_map, relations_readers, problems)
    return tuple(problems)


def _open_resolution_runs(
    context_map: ContextMap,
    locations: Mapping[str, Path],
    problems: list[str],
) -> dict[str, EntityResolutionRunReader]:
    readers: dict[str, EntityResolutionRunReader] = {}
    for artifact_id, location in locations.items():
        try:
            reader = EntityResolutionRunReader(location)
        except (IncompleteRunArtifactError, RunArtifactError) as error:
            problems.append(
                f"the entity resolution run cited as {artifact_id!r} cannot be opened at "
                f"{location.name!r}: {error}"
            )
            continue
        if str(reader.run_id) != artifact_id:
            problems.append(
                f"the map cites entity resolution run {artifact_id!r}, but the artifact found "
                f"at {location.name!r} is the run {reader.run_id!r}"
            )
            continue
        map_frame = str(context_map.geometry_ref.map_id)
        if str(reader.manifest.lineage.geometric_map_id) != map_frame:
            problems.append(
                f"entity resolution run {artifact_id!r} was built over geometric map "
                f"{reader.manifest.lineage.geometric_map_id!r}, not the map's own {map_frame!r}"
            )
        readers[artifact_id] = reader
    return readers


def _check_entities(
    context_map: ContextMap,
    resolution_readers: Mapping[str, EntityResolutionRunReader],
    problems: list[str],
) -> None:
    for entity in context_map.entities:
        reader = resolution_readers.get(str(entity.source.resolution_run_id))
        if reader is None:
            continue
        try:
            reader.resolved_entity(entity.source)
        except (UnknownResolvedEntityError, ForeignResolvedEntityReferenceError) as error:
            problems.append(
                f"entity {entity.entity_id!r} cites resolved entity "
                f"{entity.source.resolved_entity_id!r} of run {entity.source.resolution_run_id!r}, "
                f"which the run does not have: {error}"
            )
        except (IncompleteRunArtifactError, RunArtifactError) as error:
            # A leitura é preguiçosa: abrir o reader não garante que a tabela que resolve esta
            # entidade específica ainda esteja íntegra, então a corrupção só aparece aqui.
            problems.append(
                f"entity {entity.entity_id!r} cites resolved entity "
                f"{entity.source.resolved_entity_id!r} of run {entity.source.resolution_run_id!r}, "
                f"which could not be read to check: {error}"
            )


def _open_relations_runs(
    context_map: ContextMap,
    locations: Mapping[str, Path],
    resolution_readers: Mapping[str, EntityResolutionRunReader],
    problems: list[str],
) -> dict[str, SpatialRelationsRunReader]:
    readers: dict[str, SpatialRelationsRunReader] = {}
    for artifact_id, location in locations.items():
        try:
            reader = SpatialRelationsRunReader(location)
        except (IncompleteRelationsRunArtifactError, RelationsRunArtifactError) as error:
            problems.append(
                f"the spatial relations run cited as {artifact_id!r} cannot be opened at "
                f"{location.name!r}: {error}"
            )
            continue
        if str(reader.manifest.run_id) != artifact_id:
            problems.append(
                f"the map cites spatial relations run {artifact_id!r}, but the artifact found "
                f"at {location.name!r} is the run {reader.manifest.run_id!r}"
            )
            continue
        map_frame = str(context_map.geometry_ref.map_id)
        if str(reader.manifest.lineage.geometric_map_id) != map_frame:
            problems.append(
                f"spatial relations run {artifact_id!r} was built over geometric map "
                f"{reader.manifest.lineage.geometric_map_id!r}, not the map's own {map_frame!r}"
            )
        matching = resolution_readers.get(str(reader.manifest.lineage.entity_resolution_run_id))
        if matching is None:
            problems.append(
                f"spatial relations run {artifact_id!r} was built over entity resolution run "
                f"{reader.manifest.lineage.entity_resolution_run_id!r}, which the map does not "
                "cite as its entity resolution dependency"
            )
        else:
            try:
                resolution_problems = reader.validate_resolution(matching)
            except (
                IncompleteRelationsRunArtifactError,
                RelationsRunArtifactError,
                IncompleteRunArtifactError,
                RunArtifactError,
            ) as error:
                # Preguiçoso dos dois lados: validate_resolution só lê a própria tabela de
                # índice (e, através dela, a resolução) quando é chamado, então uma tabela
                # corrompida só se revela aqui, não na abertura do reader.
                problems.append(
                    f"spatial relations run {artifact_id!r} could not be checked against "
                    f"entity resolution run {matching.run_id!r}: {error}"
                )
            else:
                problems.extend(
                    f"spatial relations run {artifact_id!r}: {problem}"
                    for problem in resolution_problems
                )
        readers[artifact_id] = reader
    return readers


def _check_relations(
    context_map: ContextMap,
    relations_readers: Mapping[str, SpatialRelationsRunReader],
    problems: list[str],
) -> None:
    for relation in context_map.relations:
        reader = relations_readers.get(str(relation.source_run_id))
        if reader is None:
            continue
        try:
            reader.relation(relation.source_relation_id)
        except KeyError:
            problems.append(
                f"relation {relation.relation_id!r} cites relation "
                f"{relation.source_relation_id!r} of run {relation.source_run_id!r}, which the "
                "run does not have"
            )
        except (IncompleteRelationsRunArtifactError, RelationsRunArtifactError) as error:
            problems.append(
                f"relation {relation.relation_id!r} cites relation "
                f"{relation.source_relation_id!r} of run {relation.source_run_id!r}, which could "
                f"not be read to check: {error}"
            )


__all__ = ["check_structural_dependencies"]
