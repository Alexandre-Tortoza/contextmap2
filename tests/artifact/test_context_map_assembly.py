"""Tests for :func:`contextmap.artifact.assemble_context_map`.

Every real-run fixture is written by the actual writers of its capability (``build_geometry_
artifact``, ``write_resolution_run``, ``write_relations_run``, all reused from the serialization
test helpers), never hand-built JSON: assembly is exercised against the same artifact shapes a
real pipeline run produces. ``_ambiguous_world`` is the one bespoke fixture, needed because the
real Entity Resolution fixture the serialization tests share is unambiguous by construction (a
single hypothesis per entity); it uses Entity Resolution's own public materialization API, not a
hand-written record.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from context_map_builders import SEQUENCE_ARTIFACT_ID, digest, entity_capabilities, metadata
from context_map_serialization_geometry import build_geometry_artifact
from context_map_serialization_upstream import write_relations_run, write_resolution_run
from mapping_builders import make_hypothesis
from resolution_entity_builders import entity_at, fused_id
from resolution_run_fixtures import LINEAGE as BASE_RESOLUTION_LINEAGE

from contextmap.artifact import (
    AmbiguityStatus,
    ArtifactKind,
    ContextMap,
    ContextMapArtifactReader,
    ContextMapArtifactWriter,
    ContextMapId,
    UpstreamArtifact,
    UpstreamArtifactError,
    artifact_digest,
    assemble_context_map,
    context_map_to_record,
)
from contextmap.artifact.references import ContextEntityReference
from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    EntityResolutionRunId,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    materialize_resolved_entities,
    retrieve_candidate_sets,
)
from contextmap.geometric_mapping import GeometricMapArtifactManifest
from contextmap.semantic_mapping import SemanticMapId
from contextmap.spatial_relations import SpatialRelationsRunReader

CONTEXT_MAP_ID = ContextMapId("context-map--assembly-test--0001")


def _sequence_lineage() -> tuple[UpstreamArtifact, ...]:
    """The one lineage entry ``assemble_context_map`` never derives itself.

    Every ``ContextMapMetadata`` requires at least one ``source_sequences`` entry, and the map's
    own construction requires a matching ``SEQUENCE`` artifact in the lineage — a dependency this
    module (scoped to geometry, entity resolution and spatial relations) never opens. Every test
    here uses ``context_map_builders.metadata()``'s default source sequence, so one fixed,
    synthetic entry covers them all.
    """
    return (
        UpstreamArtifact(
            artifact_id=SEQUENCE_ARTIFACT_ID,
            kind=ArtifactKind.SEQUENCE,
            content_identity=digest(SEQUENCE_ARTIFACT_ID),
            configuration_fingerprint=None,
            code_version=None,
            model_identities=(),
        ),
    )


def _assemble(**kwargs: Any) -> ContextMap:
    """``assemble_context_map`` with the mandatory sequence lineage filled in by default."""
    kwargs.setdefault("additional_lineage", _sequence_lineage())
    return assemble_context_map(**kwargs)


@dataclass(frozen=True)
class World:
    """The upstream artifacts one populated-map test assembles from, all real and on disk."""

    geometry_dir: Path
    geometry_manifest: GeometricMapArtifactManifest
    resolution_dir: Path
    relations_dir: Path


def _build_world(tmp_path: Path) -> World:
    geometry_dir, geometry_manifest = build_geometry_artifact(tmp_path / "geometry-workspace")
    resolution_dir = tmp_path / "entity-resolution"
    write_resolution_run(
        resolution_dir,
        run_id="entity-resolution--run-0001",
        geometric_map_id=str(geometry_manifest.map_id),
        semantic_map_id="semantic-map--0001",
    )
    relations_dir = tmp_path / "spatial-relations"
    write_relations_run(
        relations_dir, run_id="spatial-relations--run-0001", resolution_dir=resolution_dir
    )
    return World(
        geometry_dir=geometry_dir,
        geometry_manifest=geometry_manifest,
        resolution_dir=resolution_dir,
        relations_dir=relations_dir,
    )


def _build_ambiguous_resolution(
    tmp_path: Path, geometry_manifest: GeometricMapArtifactManifest
) -> Path:
    """A resolution run with one unambiguous, one ambiguous and one insufficient-evidence entity.

    Built from Entity Resolution's own public materialization API on three singleton entities
    that are never compared (they are far apart), so no decision is needed to keep each one its
    own resolved entity: this isolates the semantic-state translation from any merge logic, which
    is already covered by ``_build_world``'s real Spatial-Relations-ready fixture.
    """
    map_id = geometry_manifest.map_id
    semantic_map_id = SemanticMapId("semantic-map--ambiguous-0001")
    run_id = EntityResolutionRunId("entity-resolution--ambiguous-run-0001")
    entities = {
        "one": entity_at(
            "one",
            (0.0, 0.0, 0.0),
            support_number=1,
            spatial=("spatial--one",),
            map_id=map_id,
            semantic_map_id=semantic_map_id,
            first_index=0,
        ),
        "empty": entity_at(
            "empty",
            (20.0, 0.0, 0.0),
            support_number=2,
            hypotheses=(),
            spatial=("spatial--empty",),
            map_id=map_id,
            semantic_map_id=semantic_map_id,
            first_index=100,
        ),
        "ambiguous": entity_at(
            "ambiguous",
            (40.0, 0.0, 0.0),
            support_number=3,
            # Duas hipóteses da mesma evidência fundida: nenhuma domina, então o estado é
            # ambíguo, não uma escolha entre rótulos por confiança.
            hypotheses=(
                make_hypothesis("hypothesis-0001", "desk", fused_evidence_id=fused_id(3)),
                make_hypothesis("hypothesis-0002", "table", fused_evidence_id=fused_id(3)),
            ),
            spatial=("spatial--ambiguous",),
            map_id=map_id,
            semantic_map_id=semantic_map_id,
            first_index=200,
        ),
    }
    candidate_sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.1)
    )
    materialization = materialize_resolved_entities(entities.values(), [], resolution_run_id=run_id)
    resolution_dir = tmp_path / "entity-resolution-ambiguous"
    lineage = dataclasses.replace(
        BASE_RESOLUTION_LINEAGE,
        geometric_map_id=map_id,
        semantic_map_ids=(semantic_map_id,),
        # BASE_RESOLUTION_LINEAGE traz um placeholder que não é um sha256 real; a montagem do
        # ContextMap deriva dele uma UpstreamArtifact real e exige o formato "sha256:<64 hex>".
        semantic_mapping_artifact_digest=digest(str(semantic_map_id)),
    )
    EntityResolutionRunWriter(
        output_dir=resolution_dir, run_id=run_id, lineage=lineage, code_version="test"
    ).write(candidate_sets=candidate_sets, resolutions=(), materialization=materialization)
    return resolution_dir


# --- geometry-only map --------------------------------------------------------------------------


def test_geometry_only_map_declares_only_geometry(tmp_path: Path) -> None:
    geometry_dir, geometry_manifest = build_geometry_artifact(tmp_path / "geometry-workspace")

    context_map = _assemble(
        context_map_id=CONTEXT_MAP_ID, metadata=metadata(), geometric_map_location=geometry_dir
    )

    assert context_map.entities == ()
    assert context_map.relations == ()
    assert context_map.geometry_ref.map_id == geometry_manifest.map_id
    assert context_map.geometry_ref.point_count == geometry_manifest.point_count
    assert {item.kind for item in context_map.lineage} == {
        ArtifactKind.GEOMETRIC_MAP,
        ArtifactKind.SEQUENCE,
    }
    geometry_entry = next(
        item for item in context_map.lineage if item.kind is ArtifactKind.GEOMETRIC_MAP
    )
    assert geometry_entry.content_identity == artifact_digest(geometry_dir)


# --- populated map from real runs ---------------------------------------------------------------


def test_populated_map_from_real_entity_resolution_and_spatial_relations_runs(
    tmp_path: Path,
) -> None:
    world = _build_world(tmp_path)
    resolution = EntityResolutionRunReader(world.resolution_dir)
    relations_run = SpatialRelationsRunReader(world.relations_dir)
    resolved_entities = resolution.resolved_entities().entities
    upstream_relations = tuple(relations_run.iter_relations())
    predicates = {relation.predicate for relation in upstream_relations}

    context_map = _assemble(
        context_map_id=CONTEXT_MAP_ID,
        metadata=metadata(capabilities=entity_capabilities(*predicates)),
        geometric_map_location=world.geometry_dir,
        entity_resolution_location=world.resolution_dir,
        spatial_relations_location=world.relations_dir,
    )

    assert len(context_map.entities) == len(resolved_entities)
    assert len(context_map.relations) == len(upstream_relations)

    by_source = {entity.source: entity for entity in context_map.entities}
    assert set(by_source) == {resolved.reference for resolved in resolved_entities}
    for resolved in resolved_entities:
        entity = by_source[resolved.reference]
        assert entity.member_entities == resolved.member_entity_refs
        assert entity.resolution_decisions == resolved.resolution_decision_refs
        assert entity.unresolved_neighbors == resolved.unresolved_neighbor_refs
        assert entity.geometry_refs == resolved.geometry.geometry_refs

    lineage_by_kind = {item.kind: item for item in context_map.lineage}
    assert lineage_by_kind[ArtifactKind.GEOMETRIC_MAP].content_identity == artifact_digest(
        world.geometry_dir
    )
    assert lineage_by_kind[ArtifactKind.ENTITY_RESOLUTION_RUN].content_identity == artifact_digest(
        world.resolution_dir
    )
    assert lineage_by_kind[ArtifactKind.SPATIAL_RELATIONS_RUN].content_identity == artifact_digest(
        world.relations_dir
    )


def test_relation_endpoints_predicate_state_and_uncertainty_are_preserved(tmp_path: Path) -> None:
    world = _build_world(tmp_path)
    relations_run = SpatialRelationsRunReader(world.relations_dir)
    upstream_relations = tuple(relations_run.iter_relations())
    predicates = {relation.predicate for relation in upstream_relations}

    context_map = _assemble(
        context_map_id=CONTEXT_MAP_ID,
        metadata=metadata(capabilities=entity_capabilities(*predicates)),
        geometric_map_location=world.geometry_dir,
        entity_resolution_location=world.resolution_dir,
        spatial_relations_location=world.relations_dir,
    )

    entity_ref_of = {
        entity.source: ContextEntityReference(
            context_map_id=context_map.context_map_id, entity_id=entity.entity_id
        )
        for entity in context_map.entities
    }
    by_source_relation = {
        relation.source_relation_id: relation for relation in context_map.relations
    }
    assert len(by_source_relation) == len(upstream_relations)
    for upstream in upstream_relations:
        translated = by_source_relation[upstream.relation_id]
        assert translated.subject == entity_ref_of[upstream.subject_entity_ref]
        assert translated.object == entity_ref_of[upstream.object_entity_ref]
        assert translated.predicate == upstream.predicate
        assert translated.state == upstream.state
        expected_uncertainty = tuple(
            sorted({item.kind for item in upstream.uncertainty}, key=lambda kind: kind.value)
        )
        assert translated.uncertainty_kinds == expected_uncertainty


# --- semantic-state translation ------------------------------------------------------------------


def test_ambiguous_and_insufficient_evidence_semantic_state_is_preserved(tmp_path: Path) -> None:
    geometry_dir, geometry_manifest = build_geometry_artifact(tmp_path / "geometry-workspace")
    resolution_dir = _build_ambiguous_resolution(tmp_path, geometry_manifest)
    context_map = _assemble(
        context_map_id=CONTEXT_MAP_ID,
        metadata=metadata(capabilities=entity_capabilities()),
        geometric_map_location=geometry_dir,
        entity_resolution_location=resolution_dir,
    )

    # Cada entidade resolvida deste cenário tem exatamente um membro de origem, então o nome do
    # membro identifica a entidade composta sem depender de qualquer ordem específica.
    assert len(context_map.entities) == 3
    states = {
        str(entity.member_entities[0].entity_id): entity.semantic_state
        for entity in context_map.entities
    }

    assert states["one"].status is AmbiguityStatus.UNAMBIGUOUS
    assert [h.label for h in states["one"].hypotheses] == ["pallet"]

    assert states["empty"].status is AmbiguityStatus.INSUFFICIENT_EVIDENCE
    assert states["empty"].hypotheses == ()

    assert states["ambiguous"].status is AmbiguityStatus.AMBIGUOUS
    assert [h.label for h in states["ambiguous"].hypotheses] == ["desk", "table"]
    # A ambiguidade nunca é achatada em um rótulo: as duas hipóteses concorrentes sobrevivem.
    assert len(states["ambiguous"].hypotheses) == 2


# --- deterministic ordering -----------------------------------------------------------------------


def test_assembly_is_deterministic_regardless_of_relation_reader_natural_order(
    tmp_path: Path,
) -> None:
    world = _build_world(tmp_path)
    relations_run = SpatialRelationsRunReader(world.relations_dir)
    natural_order = tuple(relations_run.iter_relations())
    expected_order = tuple(sorted(natural_order, key=lambda item: str(item.relation_id)))
    # A ordem natural do reader (sujeito, predicado, objeto) não é a ordem por relation_id: a
    # montagem tem de ordenar explicitamente, não apenas repassar a ordem do reader.
    assert [item.relation_id for item in natural_order] != [
        item.relation_id for item in expected_order
    ]

    context_map = _assemble(
        context_map_id=CONTEXT_MAP_ID,
        metadata=metadata(
            capabilities=entity_capabilities(*{item.predicate for item in natural_order})
        ),
        geometric_map_location=world.geometry_dir,
        entity_resolution_location=world.resolution_dir,
        spatial_relations_location=world.relations_dir,
    )

    assert [relation.source_relation_id for relation in context_map.relations] == [
        item.relation_id for item in expected_order
    ]


def test_assembling_the_same_runs_twice_gives_the_same_context_map(tmp_path: Path) -> None:
    world = _build_world(tmp_path)
    relations_run = SpatialRelationsRunReader(world.relations_dir)
    predicates = {item.predicate for item in relations_run.iter_relations()}
    kwargs = dict(
        context_map_id=CONTEXT_MAP_ID,
        metadata=metadata(capabilities=entity_capabilities(*predicates)),
        geometric_map_location=world.geometry_dir,
        entity_resolution_location=world.resolution_dir,
        spatial_relations_location=world.relations_dir,
    )

    first = _assemble(**kwargs)
    second = _assemble(**kwargs)

    assert first == second


# --- explicit rejection ---------------------------------------------------------------------------


def test_spatial_relations_without_entity_resolution_is_rejected(tmp_path: Path) -> None:
    world = _build_world(tmp_path)

    with pytest.raises(ValueError, match="entity_resolution_location"):
        assemble_context_map(
            context_map_id=CONTEXT_MAP_ID,
            metadata=metadata(),
            geometric_map_location=world.geometry_dir,
            spatial_relations_location=world.relations_dir,
        )


def test_geometry_reference_mismatch_is_rejected(tmp_path: Path) -> None:
    geometry_dir, _ = build_geometry_artifact(tmp_path / "geometry-workspace")
    resolution_dir = tmp_path / "entity-resolution"
    write_resolution_run(
        resolution_dir,
        run_id="entity-resolution--run-0001",
        geometric_map_id="a-different-geometric-map",
        semantic_map_id="semantic-map--0001",
    )

    with pytest.raises(UpstreamArtifactError, match="geometric map"):
        assemble_context_map(
            context_map_id=CONTEXT_MAP_ID,
            metadata=metadata(capabilities=entity_capabilities()),
            geometric_map_location=geometry_dir,
            entity_resolution_location=resolution_dir,
        )


def test_dangling_spatial_relations_dependency_is_rejected(tmp_path: Path) -> None:
    world = _build_world(tmp_path)
    # Um segundo run de resolução real, sobre a mesma geometria mas com identidade diferente: a
    # relação foi decidida sobre o primeiro run, não sobre este.
    other_resolution_dir = tmp_path / "entity-resolution-other"
    write_resolution_run(
        other_resolution_dir,
        run_id="entity-resolution--run-0002",
        geometric_map_id=str(world.geometry_manifest.map_id),
        semantic_map_id="semantic-map--0001",
    )

    with pytest.raises(UpstreamArtifactError, match="does not match"):
        assemble_context_map(
            context_map_id=CONTEXT_MAP_ID,
            metadata=metadata(capabilities=entity_capabilities()),
            geometric_map_location=world.geometry_dir,
            entity_resolution_location=other_resolution_dir,
            spatial_relations_location=world.relations_dir,
        )


# --- provenance closure -----------------------------------------------------------------------


def test_provenance_closure_is_complete(tmp_path: Path) -> None:
    world = _build_world(tmp_path)
    relations_run = SpatialRelationsRunReader(world.relations_dir)
    predicates = {item.predicate for item in relations_run.iter_relations()}

    context_map = _assemble(
        context_map_id=CONTEXT_MAP_ID,
        metadata=metadata(capabilities=entity_capabilities(*predicates)),
        geometric_map_location=world.geometry_dir,
        entity_resolution_location=world.resolution_dir,
        spatial_relations_location=world.relations_dir,
    )

    # ContextMap.__post_init__ já rejeitaria uma origem que cita um artifact fora da linhagem;
    # aqui confirmamos explicitamente que toda citação resolve, como consumidor faria.
    for entity in context_map.entities:
        for record in entity.origin.derived_from:
            context_map.upstream_artifact(record.artifact_id)
        for hypothesis in entity.semantic_state.hypotheses:
            for record in hypothesis.origin.derived_from:
                context_map.upstream_artifact(record.artifact_id)
    for relation in context_map.relations:
        for record in relation.origin.derived_from:
            context_map.upstream_artifact(record.artifact_id)


# --- round trip ------------------------------------------------------------------------------


def test_round_trip_assemble_write_reopen_same_record(tmp_path: Path) -> None:
    world = _build_world(tmp_path)
    relations_run = SpatialRelationsRunReader(world.relations_dir)
    predicates = {item.predicate for item in relations_run.iter_relations()}

    context_map = _assemble(
        context_map_id=CONTEXT_MAP_ID,
        metadata=metadata(capabilities=entity_capabilities(*predicates)),
        geometric_map_location=world.geometry_dir,
        entity_resolution_location=world.resolution_dir,
        spatial_relations_location=world.relations_dir,
    )

    output_dir = tmp_path / "out" / "context_map"
    ContextMapArtifactWriter(output_dir=output_dir).write(
        context_map,
        upstream_locations={
            str(world.geometry_manifest.map_id): world.geometry_dir,
            "entity-resolution--run-0001": world.resolution_dir,
            "spatial-relations--run-0001": world.relations_dir,
        },
    )

    with ContextMapArtifactReader.open(output_dir) as reader:
        reopened = reader.context_map()

    assert context_map_to_record(reopened) == context_map_to_record(context_map)
