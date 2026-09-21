"""Lineage, evidence references and provenance closure of the ContextMap."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from typing import Any

import pytest
from context_map_builders import (
    ANNOTATION_ARTIFACT_ID,
    ENTITY_RESOLUTION_ARTIFACT_ID,
    FUSION_ARTIFACT_ID,
    GEOMETRIC_MAP_ID,
    MODEL_IDENTITIES,
    PERCEPTION_ARTIFACT_ID,
    SEMANTIC_MAP_ARTIFACT_ID,
    SEQUENCE_ARTIFACT_ID,
    SPATIAL_RELATIONS_ARTIFACT_ID,
    context_map,
    default_lineage,
    digest,
    entity,
    entity_capabilities,
    hypothesis,
    metadata,
    origin,
    policy,
    populated_map,
    relation,
    upstream_artifact,
    upstream_record,
)

from contextmap import artifact
from contextmap.artifact import (
    AmbiguityStatus,
    ArtifactKind,
    ContextSemanticState,
    DeclaredCapabilities,
    DerivationKind,
    EvidenceOrigin,
    MapCapability,
    ProvenanceError,
    ReferenceIntegrityError,
    context_map_from_record,
    context_map_to_record,
)

FUSED = "fused-entity-0001"


def _with_entity(**entity_overrides: Any) -> Any:
    """Build a map whose only entity carries the given overrides, with a fitting lineage."""
    return context_map(
        metadata=metadata(capabilities=entity_capabilities()),
        entities=(entity("entity-0001", **entity_overrides),),
    )


# --- the six derivation categories -------------------------------------------------------------


def test_the_six_derivation_categories_exist_with_stable_values() -> None:
    assert {kind.name: kind.value for kind in DerivationKind} == {
        "SENSOR_OBSERVED": "sensor_observed",
        "MODEL_INFERRED": "model_inferred",
        "GEOMETRY_DERIVED": "geometry_derived",
        "MULTIVIEW_FUSED": "multiview_fused",
        "HUMAN_ANNOTATED": "human_annotated",
        "PRIOR_KNOWLEDGE": "prior_knowledge",
    }


def test_provenance_is_never_a_confidence() -> None:
    forbidden = ("confidence", "score", "probability", "weight", "likelihood")
    for name in ("EvidenceOrigin", "UpstreamArtifact", "UpstreamRecordRef", "PolicyRef"):
        cls = getattr(artifact, name)
        assert is_dataclass(cls)
        for field in fields(cls):
            assert not any(word in field.name for word in forbidden), (name, field.name)
            assert "float" not in str(field.type), (name, field.name)


def test_an_origin_never_explains_a_result_by_its_category_alone() -> None:
    with pytest.raises(ValueError, match="derived_from"):
        EvidenceOrigin(kind=DerivationKind.MODEL_INFERRED, derived_from=(), policy=None)


def test_an_origin_cites_sorted_and_unique_evidence() -> None:
    first = upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a")
    second = upstream_record(PERCEPTION_ARTIFACT_ID, "claim-b")

    assert origin(DerivationKind.MODEL_INFERRED, first, second).derived_from == (first, second)
    with pytest.raises(ValueError, match="sorted"):
        EvidenceOrigin(
            kind=DerivationKind.MODEL_INFERRED, derived_from=(second, first), policy=None
        )
    with pytest.raises(ValueError, match="unique"):
        EvidenceOrigin(kind=DerivationKind.MODEL_INFERRED, derived_from=(first, first), policy=None)


@pytest.mark.parametrize(
    "kind",
    [
        DerivationKind.GEOMETRY_DERIVED,
        DerivationKind.MULTIVIEW_FUSED,
        DerivationKind.PRIOR_KNOWLEDGE,
    ],
)
def test_derivation_by_rule_records_the_policy_and_its_version(kind: DerivationKind) -> None:
    record = upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a")

    with pytest.raises(ValueError, match="policy"):
        EvidenceOrigin(kind=kind, derived_from=(record,), policy=None)
    assert EvidenceOrigin(kind=kind, derived_from=(record,), policy=policy("rule", "3")).policy


# --- what each category may cite ---------------------------------------------------------------


def test_sensor_observed_may_cite_only_sensor_grounded_artifacts() -> None:
    grounded = origin(
        DerivationKind.SENSOR_OBSERVED,
        upstream_record(str(GEOMETRIC_MAP_ID), "geometry-x"),
        upstream_record(SEQUENCE_ARTIFACT_ID, "observation-1"),
    )

    assert _with_entity(origin=grounded).entities[0].origin.kind is DerivationKind.SENSOR_OBSERVED


def test_a_vlm_output_is_not_sensor_observed_because_it_consumed_an_image() -> None:
    dishonest = origin(
        DerivationKind.SENSOR_OBSERVED,
        upstream_record(SEQUENCE_ARTIFACT_ID, "observation-1"),
        upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a"),
    )

    with pytest.raises(ProvenanceError, match="SENSOR_OBSERVED"):
        _with_entity(origin=dishonest)


def test_a_model_inference_needs_an_inference_artifact_that_names_its_model() -> None:
    no_inference = origin(
        DerivationKind.MODEL_INFERRED, upstream_record(SEQUENCE_ARTIFACT_ID, "observation-1")
    )
    with pytest.raises(ProvenanceError, match="MODEL_INFERRED"):
        _with_entity(
            semantic_state=ContextSemanticState(
                status=AmbiguityStatus.UNAMBIGUOUS, hypotheses=(hypothesis("chair", no_inference),)
            )
        )

    anonymous_lineage = tuple(
        replace(item, model_identities=()) if item.kind is ArtifactKind.PERCEPTION_RUN else item
        for item in default_lineage(entity_capabilities())
    )
    with pytest.raises(ProvenanceError, match="model"):
        context_map(
            metadata=metadata(capabilities=entity_capabilities()),
            entities=(entity(),),
            lineage=anonymous_lineage,
        )


def test_a_geometrically_derived_result_cites_geometry_or_relation_evidence() -> None:
    wrong = origin(
        DerivationKind.GEOMETRY_DERIVED,
        upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a"),
        policy_ref=policy("geometric-relations"),
    )

    with pytest.raises(ProvenanceError, match="GEOMETRY_DERIVED"):
        populated_map(
            relations=(
                relation("relation-0001", "entity-0001", "on", "entity-0002", origin=wrong),
                populated_map().relations[1],
            )
        )


def test_a_fused_result_cites_a_fusion_artifact() -> None:
    wrong = origin(
        DerivationKind.MULTIVIEW_FUSED,
        upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a"),
        policy_ref=policy("baseline-evidence-accumulation"),
    )

    with pytest.raises(ProvenanceError, match="MULTIVIEW_FUSED"):
        _with_entity(origin=wrong)


def test_a_geometrically_derived_relation_is_never_a_direct_sensor_observation() -> None:
    dishonest = origin(
        DerivationKind.SENSOR_OBSERVED, upstream_record(str(GEOMETRIC_MAP_ID), "geometry-x")
    )

    with pytest.raises(ProvenanceError, match="relation"):
        populated_map(
            relations=(
                relation("relation-0001", "entity-0001", "on", "entity-0002", origin=dishonest),
                populated_map().relations[1],
            )
        )


def test_human_annotations_enter_only_as_a_declared_human_annotated_origin() -> None:
    annotated = origin(
        DerivationKind.HUMAN_ANNOTATED, upstream_record(ANNOTATION_ARTIFACT_ID, "annotation-1")
    )
    lineage = (
        *default_lineage(entity_capabilities()),
        upstream_artifact(ANNOTATION_ARTIFACT_ID, ArtifactKind.HUMAN_ANNOTATION_SET),
    )
    lineage = tuple(sorted(lineage, key=lambda item: item.artifact_id))
    accepted = context_map(
        metadata=metadata(capabilities=entity_capabilities()),
        entities=(
            entity(
                semantic_state=ContextSemanticState(
                    status=AmbiguityStatus.UNAMBIGUOUS,
                    hypotheses=(hypothesis("chair", annotated),),
                )
            ),
        ),
        lineage=lineage,
    )
    assert accepted.entities[0].semantic_state.hypotheses[0].origin.kind is (
        DerivationKind.HUMAN_ANNOTATED
    )

    smuggled = origin(
        DerivationKind.MODEL_INFERRED,
        upstream_record(ANNOTATION_ARTIFACT_ID, "annotation-1"),
        upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a"),
    )
    with pytest.raises(ProvenanceError, match="annotation"):
        context_map(
            metadata=metadata(capabilities=entity_capabilities()),
            entities=(
                entity(
                    semantic_state=ContextSemanticState(
                        status=AmbiguityStatus.UNAMBIGUOUS,
                        hypotheses=(hypothesis("chair", smuggled),),
                    )
                ),
            ),
            lineage=lineage,
        )


def test_a_human_annotated_origin_needs_an_annotation_artifact() -> None:
    unfounded = origin(
        DerivationKind.HUMAN_ANNOTATED, upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a")
    )

    with pytest.raises(ProvenanceError, match="HUMAN_ANNOTATED"):
        _with_entity(
            semantic_state=ContextSemanticState(
                status=AmbiguityStatus.UNAMBIGUOUS, hypotheses=(hypothesis("chair", unfounded),)
            )
        )


def test_an_evaluation_only_reference_set_has_no_place_in_the_lineage() -> None:
    assert {kind.value for kind in ArtifactKind}.isdisjoint(
        {"evaluation_reference_set", "reference_set", "ground_truth", "debug", "debug_output"}
    )
    with pytest.raises(ValueError, match="evaluation_reference_set"):
        ArtifactKind("evaluation_reference_set")


def test_prior_knowledge_is_representable_without_changing_other_origins() -> None:
    reasoned = origin(
        DerivationKind.PRIOR_KNOWLEDGE,
        upstream_record(FUSION_ARTIFACT_ID, FUSED),
        policy_ref=policy("commonsense-context", "0"),
    )
    state = ContextSemanticState(
        status=AmbiguityStatus.AMBIGUOUS,
        hypotheses=(hypothesis("chair"), hypothesis("stool", reasoned)),
    )

    result = _with_entity(semantic_state=state)

    kinds = [item.origin.kind for item in result.entities[0].semantic_state.hypotheses]
    assert kinds == [DerivationKind.MODEL_INFERRED, DerivationKind.PRIOR_KNOWLEDGE]


# --- upstream artifacts: exact identities ------------------------------------------------------


def test_an_upstream_artifact_preserves_exact_content_config_code_and_model_identities() -> None:
    item = upstream_artifact(
        PERCEPTION_ARTIFACT_ID,
        ArtifactKind.PERCEPTION_RUN,
        model_identities=MODEL_IDENTITIES,
        configuration_fingerprint="sha256:cfg",
        code_version="deadbee",
    )

    assert item.content_identity == digest(PERCEPTION_ARTIFACT_ID)
    assert item.configuration_fingerprint == "sha256:cfg"
    assert item.code_version == "deadbee"
    assert item.model_identities == MODEL_IDENTITIES


@pytest.mark.parametrize("bad", ["", "abc", "md5:" + "0" * 32, "sha256:" + "G" * 64, "sha256:12"])
def test_the_content_identity_is_a_sha256_digest(bad: str) -> None:
    with pytest.raises(ValueError, match="content_identity"):
        upstream_artifact("a", ArtifactKind.SEQUENCE, content_identity=bad)


def test_optional_identities_are_explicitly_none_never_blank() -> None:
    item = upstream_artifact(
        "a", ArtifactKind.SEQUENCE, configuration_fingerprint=None, code_version=None
    )

    assert item.configuration_fingerprint is None and item.code_version is None
    with pytest.raises(ValueError, match="code_version"):
        upstream_artifact("a", ArtifactKind.SEQUENCE, code_version=" ")


def test_model_identities_are_sorted_unique_and_not_blank() -> None:
    with pytest.raises(ValueError, match="sorted"):
        upstream_artifact("a", ArtifactKind.PERCEPTION_RUN, model_identities=("b", "a"))
    with pytest.raises(ValueError, match="unique"):
        upstream_artifact("a", ArtifactKind.PERCEPTION_RUN, model_identities=("a", "a"))
    with pytest.raises(ValueError, match="model_identities"):
        upstream_artifact("a", ArtifactKind.PERCEPTION_RUN, model_identities=(" ",))


def test_an_upstream_artifact_identity_is_not_a_path() -> None:
    with pytest.raises(ValueError, match="path"):
        upstream_artifact("outputs/run-0001", ArtifactKind.SEQUENCE)


def test_the_lineage_is_sorted_and_unique_by_artifact() -> None:
    lineage = default_lineage(entity_capabilities())

    with pytest.raises(ValueError, match="sorted"):
        context_map(
            metadata=metadata(capabilities=entity_capabilities()),
            lineage=tuple(reversed(lineage)),
        )
    with pytest.raises(ValueError, match="unique"):
        context_map(lineage=(lineage[0], lineage[0], *lineage[1:]))


# --- physical observation versus inference; structural versus evidence -------------------------


def test_only_a_sequence_is_a_physical_observation() -> None:
    physical = {kind for kind in ArtifactKind if kind.is_physical_observation}

    assert physical == {ArtifactKind.SEQUENCE}


def test_structural_dependencies_are_those_needed_to_resolve_the_map() -> None:
    structural = {kind for kind in ArtifactKind if kind.is_structural}

    assert structural == {
        ArtifactKind.GEOMETRIC_MAP,
        ArtifactKind.ENTITY_RESOLUTION_RUN,
        ArtifactKind.SPATIAL_RELATIONS_RUN,
    }


def test_a_consumer_separates_physical_from_inference_provenance_without_debug_data() -> None:
    result = populated_map()
    chair = result.entities[0]
    state_origin = chair.semantic_state.hypotheses[0].origin
    mixed = origin(
        DerivationKind.MODEL_INFERRED,
        upstream_record(PERCEPTION_ARTIFACT_ID, "claim-a"),
        upstream_record(SEQUENCE_ARTIFACT_ID, "observation-1"),
    )

    kinds = [result.upstream_artifact(ref.artifact_id).kind for ref in mixed.derived_from]

    assert kinds == [ArtifactKind.PERCEPTION_RUN, ArtifactKind.SEQUENCE]
    assert [kind.is_physical_observation for kind in kinds] == [False, True]
    assert state_origin.kind is DerivationKind.MODEL_INFERRED


# --- closure: every cited artifact is listed with the right kind ------------------------------


def test_every_cited_artifact_must_be_in_the_lineage() -> None:
    stray = origin(
        DerivationKind.MODEL_INFERRED, upstream_record("perception-run--9999", "claim-a")
    )

    with pytest.raises(ReferenceIntegrityError, match="perception-run--9999"):
        _with_entity(
            semantic_state=ContextSemanticState(
                status=AmbiguityStatus.UNAMBIGUOUS, hypotheses=(hypothesis("chair", stray),)
            )
        )


def test_the_geometry_must_be_a_geometric_map_in_the_lineage() -> None:
    without = tuple(
        item
        for item in default_lineage(metadata().capabilities)
        if item.kind is not (ArtifactKind.GEOMETRIC_MAP)
    )
    with pytest.raises(ReferenceIntegrityError, match="GEOMETRIC_MAP"):
        context_map(lineage=without)

    mislabelled = tuple(
        replace(item, kind=ArtifactKind.SEMANTIC_MAP)
        if item.artifact_id == GEOMETRIC_MAP_ID
        else item
        for item in default_lineage(metadata().capabilities)
    )
    with pytest.raises(ReferenceIntegrityError, match="GEOMETRIC_MAP"):
        context_map(lineage=mislabelled)


def test_every_source_sequence_must_be_a_sequence_in_the_lineage() -> None:
    without = tuple(
        item
        for item in default_lineage(metadata().capabilities)
        if item.kind is not ArtifactKind.SEQUENCE
    )

    with pytest.raises(ReferenceIntegrityError, match="SEQUENCE"):
        context_map(lineage=without)


def test_an_entity_maps_to_an_entity_resolution_artifact() -> None:
    wrong = upstream_record(FUSION_ARTIFACT_ID, "resolved-entity-0001")

    with pytest.raises(ReferenceIntegrityError, match="ENTITY_RESOLUTION_RUN"):
        _with_entity(source=wrong)


def test_a_relation_maps_to_a_spatial_relations_artifact() -> None:
    wrong = upstream_record(ENTITY_RESOLUTION_ARTIFACT_ID, "source-relation-0001")

    with pytest.raises(ReferenceIntegrityError, match="SPATIAL_RELATIONS_RUN"):
        populated_map(
            relations=(
                relation("relation-0001", "entity-0001", "on", "entity-0002", source=wrong),
                populated_map().relations[1],
            )
        )


# --- entity merge and resolution lineage -------------------------------------------------------


def test_a_resolved_entity_keeps_the_source_entities_it_was_resolved_from() -> None:
    members = (
        upstream_record(SEMANTIC_MAP_ARTIFACT_ID, "semantic-a"),
        upstream_record(SEMANTIC_MAP_ARTIFACT_ID, "semantic-b"),
    )
    decisions = (upstream_record(ENTITY_RESOLUTION_ARTIFACT_ID, "decision-a-b"),)

    merged = _with_entity(member_entities=members, resolution_decisions=decisions).entities[0]

    assert merged.member_entities == members
    assert merged.resolution_decisions == decisions


def test_an_entity_is_resolved_from_at_least_one_source_entity() -> None:
    with pytest.raises(ValueError, match="member_entities"):
        entity(member_entities=())


def test_merge_lineage_is_sorted_and_unique() -> None:
    a = upstream_record(SEMANTIC_MAP_ARTIFACT_ID, "semantic-a")
    b = upstream_record(SEMANTIC_MAP_ARTIFACT_ID, "semantic-b")

    with pytest.raises(ValueError, match="sorted"):
        entity(member_entities=(b, a))
    with pytest.raises(ValueError, match="unique"):
        entity(member_entities=(a, a))


def test_member_entities_come_from_a_semantic_map_and_decisions_from_resolution() -> None:
    with pytest.raises(ReferenceIntegrityError, match="SEMANTIC_MAP"):
        _with_entity(member_entities=(upstream_record(FUSION_ARTIFACT_ID, "x"),))
    with pytest.raises(ReferenceIntegrityError, match="ENTITY_RESOLUTION_RUN"):
        _with_entity(resolution_decisions=(upstream_record(FUSION_ARTIFACT_ID, "x"),))


# --- capabilities are backed by the lineage ---------------------------------------------------


def test_the_entities_capability_needs_an_entity_resolution_artifact() -> None:
    lineage = tuple(
        item
        for item in default_lineage(entity_capabilities())
        if item.kind is not ArtifactKind.ENTITY_RESOLUTION_RUN
    )

    with pytest.raises(ValueError, match="ENTITIES"):
        context_map(metadata=metadata(capabilities=entity_capabilities()), lineage=lineage)


def test_an_entity_resolution_artifact_requires_the_entities_capability() -> None:
    with pytest.raises(ValueError, match="ENTITIES"):
        context_map(lineage=default_lineage(entity_capabilities()))


def test_the_relations_capability_needs_a_spatial_relations_artifact() -> None:
    both = DeclaredCapabilities(
        content=(MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS),
        relation_predicates=(),
    )
    lineage = tuple(
        item
        for item in default_lineage(both)
        if item.kind is not ArtifactKind.SPATIAL_RELATIONS_RUN
    )

    with pytest.raises(ValueError, match="RELATIONS"):
        context_map(metadata=metadata(capabilities=both), lineage=lineage)


def test_point_representation_evidence_is_declared_exactly_when_its_artifact_is_cited() -> None:
    declared = DeclaredCapabilities(
        content=(MapCapability.GEOMETRY, MapCapability.POINT_REPRESENTATION_EVIDENCE),
        relation_predicates=(),
    )
    ok = context_map(metadata=metadata(capabilities=declared))
    assert MapCapability.POINT_REPRESENTATION_EVIDENCE in ok.metadata.capabilities.content

    with pytest.raises(ValueError, match="POINT_REPRESENTATION_EVIDENCE"):
        context_map(
            metadata=metadata(capabilities=declared),
            lineage=default_lineage(metadata().capabilities),
        )
    with pytest.raises(ValueError, match="POINT_REPRESENTATION_EVIDENCE"):
        context_map(lineage=default_lineage(declared))


# --- relation derivation evidence --------------------------------------------------------------


def test_a_relation_cites_the_evidence_it_was_derived_from() -> None:
    found = populated_map().relations[0]

    assert found.origin.kind is DerivationKind.GEOMETRY_DERIVED
    assert {ref.artifact_id for ref in found.origin.derived_from} == {
        SPATIAL_RELATIONS_ARTIFACT_ID,
        str(GEOMETRIC_MAP_ID),
    }
    assert found.origin.policy == policy("geometric-relations")


# --- serialization round trip ------------------------------------------------------------------


def _round_trip(value: Any) -> Any:
    return context_map_from_record(json.loads(json.dumps(context_map_to_record(value))))


def test_provenance_survives_a_serialization_round_trip() -> None:
    original = populated_map()

    restored = _round_trip(original)

    assert restored == original
    assert restored.lineage == original.lineage
    for before, after in zip(original.entities, restored.entities, strict=True):
        assert after.origin == before.origin
        assert after.member_entities == before.member_entities
        assert [h.origin for h in after.semantic_state.hypotheses] == [
            h.origin for h in before.semantic_state.hypotheses
        ]
    assert [r.origin for r in restored.relations] == [r.origin for r in original.relations]


def test_the_record_writes_categories_and_identities_verbatim() -> None:
    record = context_map_to_record(populated_map())

    first = record["entities"][0]
    assert first["origin"]["kind"] == "multiview_fused"
    assert first["origin"]["policy"] == {
        "policy_id": "baseline-evidence-accumulation",
        "version": "1",
    }
    assert first["semantic_state"]["hypotheses"][0]["origin"]["kind"] == "model_inferred"
    perception = next(a for a in record["lineage"] if a["kind"] == "perception_run")
    assert perception["model_identities"] == list(MODEL_IDENTITIES)
    assert perception["content_identity"] == digest(PERCEPTION_ARTIFACT_ID)


def test_a_tampered_origin_is_not_repaired_on_decode() -> None:
    record = context_map_to_record(populated_map())
    record["entities"][0]["origin"]["kind"] = "sensor_observed"

    with pytest.raises(ProvenanceError, match="SENSOR_OBSERVED"):
        context_map_from_record(record)


def test_an_unknown_derivation_kind_in_a_record_is_rejected() -> None:
    record = context_map_to_record(populated_map())
    record["entities"][0]["origin"]["kind"] = "gut_feeling"

    with pytest.raises(ValueError, match="gut_feeling"):
        context_map_from_record(record)


def test_the_lineage_types_are_part_of_the_public_surface() -> None:
    for name in ("EvidenceOrigin", "UpstreamArtifact", "ArtifactKind", "DerivationKind"):
        assert getattr(artifact, name).__module__.startswith("contextmap.artifact")
