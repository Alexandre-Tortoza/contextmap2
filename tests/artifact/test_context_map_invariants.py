"""Schema invariants, reference integrity and the representative fixture of the ContextMap.

Every test here runs on public contracts and on the canonical record view of the schema. None
depends on a serializer, a file layout, ROS or a model stack, so all of them stay valid whichever
serializer the artifact uses.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from context_map_builders import (
    context_map,
    entity_capabilities,
    metadata,
    populated_map,
    relation,
)
from context_map_fixture_builder import (
    FIXTURE_PATH,
    FUSION,
    GEOMETRY,
    POINT_REPRESENTATION,
    build_fixture_map,
    render_fixture,
)

from contextmap.artifact import (
    AmbiguityStatus,
    ArtifactKind,
    ContextEntityReference,
    ContextMapRecordError,
    DerivationKind,
    MapCapability,
    ProvenanceError,
    ReferenceIntegrityError,
    UnsupportedSchemaVersionError,
    context_map_from_record,
    context_map_to_record,
)
from contextmap.geometric_mapping import MapId, geometry_id_for
from contextmap.spatial_relations import RelationPredicate, RelationState, RelationUncertaintyKind

Record = dict[str, Any]

# --- the representative fixture ----------------------------------------------------------------


def _fixture_record() -> Record:
    loaded: Record = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return loaded


def test_the_fixture_is_the_committed_rendering_of_the_public_contracts() -> None:
    assert FIXTURE_PATH.read_text(encoding="utf-8") == render_fixture(), (
        "the fixture drifted from the public contracts: regenerate it with "
        "'python tests/artifact/context_map_fixture_builder.py'"
    )


def test_the_fixture_is_valid_and_rebuilds_the_same_map() -> None:
    loaded = context_map_from_record(_fixture_record())

    assert loaded == build_fixture_map()


def test_the_fixture_is_human_inspectable() -> None:
    text = FIXTURE_PATH.read_text(encoding="utf-8")

    assert len(text.splitlines()) > 100
    for word in ("chair", "desk", "table", "unresolved", "conflicting", "multiview_fused"):
        assert word in text
    assert "NaN" not in text and "Infinity" not in text


def test_the_fixture_reads_the_same_whatever_the_json_layout() -> None:
    record = _fixture_record()
    layouts = [
        json.dumps(record),
        json.dumps(record, separators=(",", ":")),
        json.dumps(record, indent=4, sort_keys=True),
    ]

    maps = [context_map_from_record(json.loads(layout)) for layout in layouts]

    assert maps[0] == maps[1] == maps[2] == build_fixture_map()


def test_the_fixture_shows_geometry_entities_relations_and_provenance_together() -> None:
    fixture = build_fixture_map()

    assert fixture.geometry_ref.map_id == GEOMETRY
    assert len(fixture.entities) >= 3 and len(fixture.relations) >= 3
    assert all(entity.geometry_refs for entity in fixture.entities)
    assert {kind for kind in (item.origin.kind for item in fixture.entities)} == {
        DerivationKind.MULTIVIEW_FUSED
    }
    hypothesis_kinds = {
        hypothesis.origin.kind
        for entity in fixture.entities
        for hypothesis in entity.semantic_state.hypotheses
    }
    assert hypothesis_kinds == {DerivationKind.MODEL_INFERRED}
    assert {item.origin.kind for item in fixture.relations} == {DerivationKind.GEOMETRY_DERIVED}
    assert {item.kind for item in fixture.lineage} >= {
        ArtifactKind.SEQUENCE,
        ArtifactKind.GEOMETRIC_MAP,
        ArtifactKind.PERCEPTION_RUN,
        ArtifactKind.SEMANTIC_FUSION_RUN,
        ArtifactKind.ENTITY_RESOLUTION_RUN,
        ArtifactKind.SPATIAL_RELATIONS_RUN,
    }


def test_the_fixture_preserves_unresolved_and_conflicting_state() -> None:
    fixture = build_fixture_map()

    statuses = {item.entity_id: item.semantic_state.status for item in fixture.entities}
    assert statuses == {
        "entity-0001": AmbiguityStatus.UNAMBIGUOUS,
        "entity-0002": AmbiguityStatus.AMBIGUOUS,
        "entity-0003": AmbiguityStatus.CONFLICTING,
        "entity-0004": AmbiguityStatus.INSUFFICIENT_EVIDENCE,
    }
    conflicting = fixture.entities[2].semantic_state.hypotheses
    assert [item.label for item in conflicting] == ["bin", "box"]
    assert fixture.entities[3].semantic_state.hypotheses == ()
    assert [item.state for item in fixture.relations] == [
        RelationState.SUPPORTED,
        RelationState.UNRESOLVED,
        RelationState.REJECTED,
    ]
    assert fixture.relations[1].uncertainty_kinds == (RelationUncertaintyKind.CONFLICTING_EVIDENCE,)


def test_the_fixture_keeps_the_merge_lineage_of_a_resolved_entity() -> None:
    merged = build_fixture_map().entities[0]

    assert len(merged.member_entities) == 2
    assert list(merged.resolution_decisions) == ["decision-a-b"]


def test_the_fixture_keeps_an_entity_whose_identity_resolution_left_a_neighbor_unresolved() -> None:
    ambiguous = build_fixture_map().entities[1]

    assert [item.entity_id for item in ambiguous.unresolved_neighbors] == ["entity-f"]
    assert not set(ambiguous.unresolved_neighbors) & set(ambiguous.member_entities)


def test_a_consumer_walks_from_a_relation_to_geometry_and_evidence_using_public_types() -> None:
    fixture = build_fixture_map()
    relation_ = fixture.relations[0]

    subject = fixture.entity(relation_.subject)
    target = fixture.entity(relation_.object)
    first_geometry = subject.geometry_refs[0]
    fusion = fixture.upstream_artifact(FUSION)
    physical = [
        ref
        for ref in subject.origin.derived_from
        if fixture.upstream_artifact(ref.artifact_id).kind.is_physical_observation
    ]

    assert first_geometry.map_id == fixture.geometry_ref.map_id
    assert first_geometry.geometry_id == geometry_id_for(map_id=MapId(GEOMETRY), index=10)
    assert target.entity_id == "entity-0002"
    assert fusion.kind is ArtifactKind.SEMANTIC_FUSION_RUN
    assert len(physical) == 2
    assert fixture.upstream_artifact(POINT_REPRESENTATION).kind is (
        ArtifactKind.POINT_REPRESENTATION_RUN
    )
    assert len(fixture.relations_for(relation_.subject)) == 2


def test_identities_are_unique_and_scoped_inside_the_fixture() -> None:
    fixture = build_fixture_map()

    entity_ids = [item.entity_id for item in fixture.entities]
    relation_ids = [item.relation_id for item in fixture.relations]
    assert len(set(entity_ids)) == len(entity_ids)
    assert len(set(relation_ids)) == len(relation_ids)
    assert {item.context_map_id for item in _endpoints(fixture)} == {fixture.context_map_id}


def _endpoints(fixture: Any) -> list[ContextEntityReference]:
    return [endpoint for item in fixture.relations for endpoint in (item.subject, item.object)]


# --- invalid inputs fail predictably, never repaired -------------------------------------------


def _entity(record: Record, entity_id: str) -> Record:
    return next(item for item in record["entities"] if item["entity_id"] == entity_id)


def _relation(record: Record, relation_id: str) -> Record:
    return next(item for item in record["relations"] if item["relation_id"] == relation_id)


def _artifact(record: Record, artifact_id: str) -> Record:
    return next(item for item in record["lineage"] if item["artifact_id"] == artifact_id)


def _drop_artifact(record: Record, artifact_id: str) -> None:
    record["lineage"] = [item for item in record["lineage"] if item["artifact_id"] != artifact_id]


def _drop_capability(record: Record, name: str) -> None:
    content = record["metadata"]["capabilities"]["content"]
    record["metadata"]["capabilities"]["content"] = [item for item in content if item != name]


def _hide_capabilities(*names: str) -> Callable[[Record], None]:
    """Build a mutation that stops declaring content the map still carries."""

    def mutate(record: Record) -> None:
        for name in names:
            _drop_capability(record, name)
        if "relations" in names:
            record["metadata"]["capabilities"]["relation_predicates"] = []

    return mutate


def _swap_first_two_entities(record: Record) -> None:
    record["entities"][0], record["entities"][1] = record["entities"][1], record["entities"][0]


def _set(path: list[str | int], value: Any) -> Callable[[Record], None]:
    """Build a mutation that sets the value at a path of the record."""

    def mutate(record: Record) -> None:
        target: Any = record
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value

    return mutate


def _mutate_entity(entity_id: str, key: str, value: Any) -> Callable[[Record], None]:
    def mutate(record: Record) -> None:
        _entity(record, entity_id)[key] = value

    return mutate


def _mutate_relation(relation_id: str, path: list[str], value: Any) -> Callable[[Record], None]:
    def mutate(record: Record) -> None:
        target: Any = _relation(record, relation_id)
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value

    return mutate


def _mutate_member_map(entity_id: str, semantic_map_id: str) -> Callable[[Record], None]:
    def mutate(record: Record) -> None:
        _entity(record, entity_id)["member_entities"][0]["semantic_map_id"] = semantic_map_id

    return mutate


def _set_last_geometry_map(record: Record) -> None:
    _entity(record, "entity-0001")["geometry_refs"][-1]["map_id"] = "zz-another-map"


def _set_last_geometry_beyond(record: Record) -> None:
    beyond = geometry_id_for(map_id=MapId(GEOMETRY), index=500)
    _entity(record, "entity-0001")["geometry_refs"][-1]["geometry_id"] = beyond


def _duplicate_entity_id(record: Record) -> None:
    record["entities"][1]["entity_id"] = record["entities"][0]["entity_id"]


def _share_entity_source(record: Record) -> None:
    record["entities"][1]["source"] = deepcopy(record["entities"][0]["source"])


def _relate_entity_to_itself(record: Record) -> None:
    _relation(record, "relation-0001")["object"] = deepcopy(
        _relation(record, "relation-0001")["subject"]
    )


def _collapse_conflict(record: Record) -> None:
    _entity(record, "entity-0003")["semantic_state"]["hypotheses"].pop()


def _hypothesise_from_nothing(record: Record) -> None:
    hypotheses = _entity(record, "entity-0001")["semantic_state"]["hypotheses"]
    _entity(record, "entity-0004")["semantic_state"]["hypotheses"] = deepcopy(hypotheses)


def _mark_vlm_output_observed(record: Record) -> None:
    hypothesis = _entity(record, "entity-0001")["semantic_state"]["hypotheses"][0]
    hypothesis["origin"]["kind"] = "sensor_observed"


INVALID_CASES: list[tuple[str, Callable[[Record], None], type[Exception], str]] = [
    # --- schema version
    (
        "unsupported-major",
        _set(["schema_version"], "1.0.0"),
        UnsupportedSchemaVersionError,
        "1.0.0",
    ),
    (
        "unsupported-minor",
        _set(["schema_version"], "0.2.0"),
        UnsupportedSchemaVersionError,
        "0.2.0",
    ),
    (
        "malformed-version",
        _set(["schema_version"], "latest"),
        UnsupportedSchemaVersionError,
        "malformed",
    ),
    # --- map frame, units, anchor and extent
    ("unknown-unit", _set(["metadata", "frame", "unit"], "foot"), ContextMapRecordError, "foot"),
    ("blank-frame", _set(["metadata", "frame", "frame_id"], " "), ValueError, "frame_id"),
    (
        "up-direction-not-a-unit-vector",
        _set(["metadata", "frame", "up_direction"], [0.0, 0.0, 2.0]),
        ValueError,
        "up_direction",
    ),
    (
        "local-origin-claims-a-reference",
        _set(["metadata", "frame", "anchor", "reference_frame_id"], "site/enu"),
        ValueError,
        "reference_frame_id",
    ),
    (
        "external-origin-without-a-reference",
        _set(["metadata", "frame", "anchor", "kind"], "externally_anchored"),
        ValueError,
        "reference_frame_id",
    ),
    (
        "bounds-in-another-frame",
        _set(["metadata", "bounds", "frame_id"], "odom"),
        ValueError,
        "frame",
    ),
    (
        "time-window-mixes-clocks",
        _set(["metadata", "time_bounds", "end", "clock_id"], "other:clock"),
        ValueError,
        "clock",
    ),
    (
        "no-source-sequence",
        _set(["metadata", "source_sequences"], []),
        ValueError,
        "source_sequences",
    ),
    # --- geometry references
    ("empty-geometry-map", _set(["geometry_ref", "point_count"], 0), ValueError, "point_count"),
    ("geometry-of-another-map", _set_last_geometry_map, ReferenceIntegrityError, "zz-another-map"),
    ("geometry-beyond-the-map", _set_last_geometry_beyond, ReferenceIntegrityError, "exceeds"),
    (
        "geometry-map-missing-from-the-lineage",
        lambda record: _drop_artifact(record, GEOMETRY),
        ReferenceIntegrityError,
        "GEOMETRIC_MAP",
    ),
    # --- entities: identity, scope and support
    ("duplicate-entity-id", _duplicate_entity_id, ValueError, "unique"),
    ("entities-out-of-order", _swap_first_two_entities, ValueError, "sorted"),
    ("two-entities-one-source", _share_entity_source, ReferenceIntegrityError, "same upstream"),
    (
        "entity-without-geometry",
        _mutate_entity("entity-0001", "geometry_refs", []),
        ValueError,
        "geometry_refs",
    ),
    (
        "entity-resolved-from-nothing",
        _mutate_entity("entity-0001", "member_entities", []),
        ValueError,
        "member_entities",
    ),
    (
        "unresolved-neighbor-also-a-member",
        lambda record: _entity(record, "entity-0002").update(
            unresolved_neighbors=deepcopy(_entity(record, "entity-0002")["member_entities"])
        ),
        ValueError,
        "member",
    ),
    (
        "member-entity-from-the-wrong-artifact",
        _mutate_member_map("entity-0001", FUSION),
        ReferenceIntegrityError,
        "SEMANTIC_MAP",
    ),
    (
        "entity-from-the-wrong-artifact",
        _set(["entities", 0, "source", "resolution_run_id"], FUSION),
        ReferenceIntegrityError,
        "ENTITY_RESOLUTION_RUN",
    ),
    # --- relations: subject and object validity
    (
        "relation-to-an-unknown-entity",
        _mutate_relation("relation-0001", ["object", "entity_id"], "entity-9999"),
        ReferenceIntegrityError,
        "entity-9999",
    ),
    (
        "relation-to-an-entity-of-another-map",
        _mutate_relation("relation-0001", ["subject", "context_map_id"], "other-map"),
        ReferenceIntegrityError,
        "other-map",
    ),
    ("relation-to-itself", _relate_entity_to_itself, ValueError, "itself"),
    (
        "relation-with-a-predicate-outside-the-taxonomy",
        _mutate_relation("relation-0001", ["predicate"], "adjacent"),
        ContextMapRecordError,
        "RelationPredicate",
    ),
    (
        "unresolved-relation-without-a-reason",
        _mutate_relation("relation-0002", ["uncertainty_kinds"], []),
        ValueError,
        "says why",
    ),
    (
        "decided-relation-with-uncertainty",
        _mutate_relation("relation-0001", ["uncertainty_kinds"], ["conflicting_evidence"]),
        ValueError,
        "no uncertainty",
    ),
    (
        "rejected-state-unknown-to-the-taxonomy",
        _mutate_relation("relation-0001", ["state"], "conflicting"),
        ContextMapRecordError,
        "RelationState",
    ),
    (
        "relation-from-the-wrong-artifact",
        _mutate_relation("relation-0001", ["source_run_id"], FUSION),
        ReferenceIntegrityError,
        "SPATIAL_RELATIONS_RUN",
    ),
    # --- semantic state: uncertainty is not collapsed
    (
        "unambiguous-with-two-hypotheses",
        _set(["entities", 1, "semantic_state", "status"], "unambiguous"),
        ValueError,
        "exactly one",
    ),
    ("conflict-collapsed-to-one-label", _collapse_conflict, ValueError, "at least two"),
    ("abstention-with-a-hypothesis", _hypothesise_from_nothing, ValueError, "no hypothesis"),
    # --- capability declarations versus the content and the lineage
    (
        "entities-hidden-from-the-declaration",
        _hide_capabilities("entities", "relations"),
        ValueError,
        "ENTITIES",
    ),
    (
        "relations-hidden-from-the-declaration",
        _hide_capabilities("relations"),
        ValueError,
        "RELATIONS",
    ),
    (
        "relation-types-drift-from-the-content",
        _set(["metadata", "capabilities", "relation_predicates"], ["inside", "on_top_of"]),
        ValueError,
        "relation_predicates",
    ),
    (
        "evidence-capability-without-declaration",
        lambda record: _drop_capability(record, "point_representation_evidence"),
        ValueError,
        "POINT_REPRESENTATION_EVIDENCE",
    ),
    # --- lineage and provenance closure
    (
        "cited-artifact-missing-from-the-lineage",
        lambda record: _drop_artifact(record, POINT_REPRESENTATION),
        ReferenceIntegrityError,
        "not in the lineage",
    ),
    (
        "artifact-listed-with-the-wrong-kind",
        lambda record: _artifact(record, GEOMETRY).update(kind="semantic_map"),
        ReferenceIntegrityError,
        "GEOMETRIC_MAP",
    ),
    ("lineage-out-of-order", lambda record: record["lineage"].reverse(), ValueError, "sorted"),
    (
        "content-identity-not-a-digest",
        lambda record: record["lineage"][0].update(content_identity="abc"),
        ValueError,
        "content_identity",
    ),
    (
        "fused-result-without-its-policy",
        _set(["entities", 0, "origin", "policy"], None),
        ProvenanceError,
        "policy",
    ),
    (
        "origin-without-evidence",
        _set(["entities", 0, "origin", "derived_from"], []),
        ValueError,
        "derived_from",
    ),
    (
        "vlm-output-marked-as-observed",
        _mark_vlm_output_observed,
        ProvenanceError,
        "SENSOR_OBSERVED",
    ),
    (
        "relation-marked-as-a-direct-observation",
        _mutate_relation("relation-0001", ["origin", "kind"], "sensor_observed"),
        ProvenanceError,
        "relation",
    ),
    (
        "unknown-derivation-kind",
        _set(["entities", 0, "origin", "kind"], "gut_feeling"),
        ContextMapRecordError,
        "gut_feeling",
    ),
    # --- shape of the record itself
    ("unknown-field", _set(["confidence"], 0.9), ContextMapRecordError, "confidence"),
    (
        "missing-field",
        lambda record: record["metadata"].pop("frame"),
        ContextMapRecordError,
        "metadata.frame",
    ),
    (
        "wrong-type",
        _set(["geometry_ref", "point_count"], "120"),
        ContextMapRecordError,
        "geometry_ref.point_count",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [pytest.param(*case[1:], id=case[0]) for case in INVALID_CASES],
)
def test_an_invalid_map_fails_predictably_and_is_never_repaired(
    mutate: Callable[[Record], None], error: type[Exception], message: str
) -> None:
    record = _fixture_record()
    mutate(record)

    with pytest.raises(error, match=message.replace(".", r"\.")):
        context_map_from_record(record)


def test_the_unmutated_fixture_record_is_accepted() -> None:
    context_map_from_record(_fixture_record())


# --- optional content absence is valid and explicit --------------------------------------------


def test_a_map_with_geometry_only_is_valid() -> None:
    geometry_only = context_map()

    assert geometry_only.metadata.capabilities.content == (MapCapability.GEOMETRY,)
    assert geometry_only.entities == () and geometry_only.relations == ()


def test_entities_without_relations_do_not_declare_relations() -> None:
    entities = populated_map(metadata=metadata(capabilities=entity_capabilities()), relations=())

    assert MapCapability.RELATIONS not in entities.metadata.capabilities.content
    assert entities.metadata.capabilities.relation_predicates == ()


def test_declared_but_empty_relations_differ_from_absent_relations() -> None:
    absent = populated_map(metadata=metadata(capabilities=entity_capabilities()), relations=())
    declared_empty = context_map(
        metadata=metadata(
            capabilities=type(absent.metadata.capabilities)(
                content=(MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS),
                relation_predicates=(),
            )
        ),
    )

    assert MapCapability.RELATIONS not in absent.metadata.capabilities.content
    assert MapCapability.RELATIONS in declared_empty.metadata.capabilities.content
    assert declared_empty.relations == ()


def test_evidence_capability_is_optional() -> None:
    fixture = build_fixture_map()
    record = context_map_to_record(fixture)
    _drop_capability(record, "point_representation_evidence")
    _drop_artifact(record, POINT_REPRESENTATION)
    for entity in record["entities"]:
        entity["origin"]["derived_from"] = [
            ref
            for ref in entity["origin"]["derived_from"]
            if ref["artifact_id"] != POINT_REPRESENTATION
        ]

    without = context_map_from_record(record)

    assert MapCapability.POINT_REPRESENTATION_EVIDENCE not in without.metadata.capabilities.content


def test_a_relation_between_valid_entities_is_accepted_once_declared() -> None:
    added = populated_map(
        relations=(
            *populated_map().relations,
            relation("relation-0003", "entity-0001", RelationPredicate.NEXT_TO, "entity-0003"),
        )
    )

    assert [item.relation_id for item in added.relations][-1] == "relation-0003"


# --- the schema is independent of the filesystem and of any serializer -------------------------

_FILE_IO_MODULES = {
    "os",
    "pathlib",
    "shutil",
    "io",
    "json",
    "pickle",
    "tempfile",
    "glob",
    "sqlite3",
}


def test_the_schema_package_performs_no_file_io_and_owns_no_format() -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "contextmap" / "artifact"
    offenders: list[str] = []
    for source in sorted(package.glob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [
                f"{source.name}: {name}" for name in names if name.split(".")[0] in _FILE_IO_MODULES
            ]

    assert not offenders
