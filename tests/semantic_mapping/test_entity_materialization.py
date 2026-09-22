import json
from pathlib import Path

import pytest
from mapping_builders import (
    MAP_ID,
    SEMANTIC_MAP_ID,
    SUMMARY_POLICY,
)
from mapping_fusion import ClaimSpec, FusionRun, View, outcomes_for, write_fusion_run, write_run
from mapping_geometry_fake import InMemoryGeometrySource

import contextmap.semantic_mapping as semantic_mapping
from contextmap.geometric_mapping import MapId
from contextmap.semantic_fusion import FusionSupportId, SemanticFusionRunId
from contextmap.semantic_mapping import (
    ENTITY_ID_POLICY_ID,
    ENTITY_MATERIALIZATION_POLICY_ID,
    AmbiguityState,
    EmptyGeometrySupportError,
    EntityId,
    EntityMaterialization,
    EntityMaterializationPolicy,
    MaterializationInputError,
    RejectionReason,
    TemporalEvidenceError,
    entity_id_for,
    materialize_entities,
    validate_entity_evidence,
)
from contextmap.semantic_mapping.serialization import encode_entity

POLICY = EntityMaterializationPolicy(geometry=SUMMARY_POLICY)
RUN_ID = SemanticFusionRunId("fusion-run-0001")


@pytest.fixture
def run(tmp_path: Path) -> FusionRun:
    return write_fusion_run(tmp_path)


def _materialize(
    run: FusionRun,
    *,
    outcomes: object = None,
    geometry: InMemoryGeometrySource | None = None,
    policy: EntityMaterializationPolicy = POLICY,
) -> EntityMaterialization:
    return materialize_entities(
        run.outcomes if outcomes is None else outcomes,  # type: ignore[arg-type]
        fusion_manifest=run.manifest,
        geometry=run.geometry if geometry is None else geometry,
        semantic_map_id=SEMANTIC_MAP_ID,
        policy=policy,
        code_version="test",
    )


class TestMaterializationFromFusedEvidence:
    def test_a_valid_fused_support_becomes_a_complete_canonical_entity(
        self, run: FusionRun
    ) -> None:
        result = _materialize(run)

        assert result.rejections == ()
        assert [item.entity_id for item in result.entities] == [
            "entity--support-000001",
            "entity--support-000002",
            "entity--support-000003",
        ]
        for entity in result.entities:
            assert entity.semantic_map_id == SEMANTIC_MAP_ID
            assert entity.geometry.geometry_refs
            assert entity.evidence.fused_evidence
            assert entity.temporal_state.observation_refs

    def test_every_entity_validates_against_the_run_it_came_from(self, run: FusionRun) -> None:
        for entity in _materialize(run).entities:
            assert (
                validate_entity_evidence(
                    entity, fusion_runs={RUN_ID: run.reader}, geometry=run.geometry
                )
                == ()
            )

    def test_geometry_semantic_uncertainty_temporal_state_and_provenance_are_preserved(
        self, run: FusionRun
    ) -> None:
        contradiction, agreement, abstention = _materialize(run).entities

        assert contradiction.geometry.geometry_refs == run.outcomes[0].support.geometry_support
        assert contradiction.semantic_state.ambiguity_state is AmbiguityState.CONFLICTING
        assert len(contradiction.semantic_state.conflicts) == 1
        assert agreement.semantic_state.ambiguity_state is AmbiguityState.UNAMBIGUOUS
        assert agreement.semantic_state.primary is not None
        assert agreement.semantic_state.primary.label == "pallet"
        assert abstention.semantic_state.ambiguity_state is AmbiguityState.INSUFFICIENT_EVIDENCE
        assert abstention.semantic_state.hypotheses == ()
        assert contradiction.temporal_state.physical_observation_count == 2
        assert contradiction.temporal_state.inference_result_count == 3
        assert contradiction.evidence.point_representation_refs
        assert (
            contradiction.provenance.materialization_policy_id == ENTITY_MATERIALIZATION_POLICY_ID
        )
        assert contradiction.provenance.identity_policy_id == ENTITY_ID_POLICY_ID
        assert contradiction.provenance.configuration_fingerprint == POLICY.fingerprint()
        assert contradiction.provenance.code_version == "test"

    def test_a_support_without_any_hypothesis_is_still_a_valid_unknown_entity(
        self, run: FusionRun
    ) -> None:
        *_, abstention = _materialize(run).entities

        assert abstention.semantic_state.primary_hypothesis is None
        assert abstention.semantic_state.uncertainty

    def test_the_fused_evidence_is_never_rewritten(self, run: FusionRun) -> None:
        _materialize(run)

        for outcome in run.outcomes:
            reread = run.reader.fused_evidence(outcome.support.fusion_support_id)
            assert reread == outcome.evidence

    def test_only_the_selected_supports_are_materialized(self, run: FusionRun) -> None:
        result = _materialize(run, outcomes=run.outcomes[1:2])

        assert [item.entity_id for item in result.entities] == ["entity--support-000002"]


class TestMaterializationBoundary:
    def test_two_independent_supports_stay_two_entities_however_similar_they_are(
        self, tmp_path: Path
    ) -> None:
        # Mesmo label e geometria adjacente: só Entity Resolution poderia decidir que é um objeto.
        outcomes = outcomes_for(
            [
                View(
                    "run-a",
                    "frame-0120",
                    (ClaimSpec("pallet", confidence=0.8),),
                    geometry=range(0, 20),
                ),
                View(
                    "run-a",
                    "frame-0130",
                    (ClaimSpec("pallet", confidence=0.8),),
                    geometry=range(20, 40),
                ),
            ]
        )
        run = write_run(tmp_path, outcomes)

        result = _materialize(run)

        assert len(outcomes) == 2 and len(result.entities) == 2
        first, second = result.entities
        assert (
            first.semantic_state.primary is not None and second.semantic_state.primary is not None
        )
        assert first.semantic_state.primary.label == second.semantic_state.primary.label == "pallet"
        assert first.entity_id != second.entity_id
        assert set(first.geometry.geometry_refs).isdisjoint(second.geometry.geometry_refs)

    def test_a_support_seen_from_two_runs_is_still_one_entity_not_two(self, tmp_path: Path) -> None:
        outcomes = outcomes_for(
            [
                View("run-a", "frame-0120", (ClaimSpec("pallet"),), geometry=range(0, 20)),
                View("run-b", "frame-0120", (ClaimSpec("pallet"),), geometry=range(0, 20)),
            ]
        )

        result = _materialize(write_run(tmp_path, outcomes))

        assert len(result.entities) == 1
        assert result.entities[0].temporal_state.physical_observation_count == 1
        assert result.entities[0].temporal_state.inference_result_count == 2

    def test_the_public_api_offers_no_merge_split_match_or_reidentification(self) -> None:
        forbidden = ("merge", "split", "match", "reidentif", "resolve_entities", "same_object")

        public = [name.lower() for name in semantic_mapping.__all__]

        assert not [name for name in public if any(word in name for word in forbidden)]

    def test_the_selection_is_an_explicit_input_and_nothing_is_loaded_implicitly(
        self, run: FusionRun
    ) -> None:
        assert _materialize(run, outcomes=[]).entities == ()


class TestDeterministicIdentity:
    def test_the_identity_is_a_pure_function_of_the_support(self) -> None:
        assert entity_id_for(fusion_support_id=FusionSupportId("support-000007")) == EntityId(
            "entity--support-000007"
        )

    def test_repeated_execution_reproduces_the_same_entities_and_identities(
        self, run: FusionRun
    ) -> None:
        first, second = _materialize(run), _materialize(run)

        assert first == second
        assert [json.dumps(encode_entity(e), sort_keys=True) for e in first.entities] == [
            json.dumps(encode_entity(e), sort_keys=True) for e in second.entities
        ]

    def test_the_order_of_the_selection_does_not_matter(self, run: FusionRun) -> None:
        assert _materialize(run, outcomes=tuple(reversed(run.outcomes))) == _materialize(run)

    def test_a_rejected_sibling_does_not_shift_the_identity_of_the_others(
        self, run: FusionRun
    ) -> None:
        partial = InMemoryGeometrySource(MAP_ID, {i: (i * 0.1, 0.0, 0.0) for i in range(50)})

        result = _materialize(run, geometry=partial)

        assert [item.entity_id for item in result.entities] == ["entity--support-000001"]
        assert result.entities[0] == _materialize(run).entities[0]

    def test_a_different_configuration_changes_the_recorded_fingerprint_not_the_identity(
        self, run: FusionRun
    ) -> None:
        other = EntityMaterializationPolicy(
            geometry=type(SUMMARY_POLICY)(sparse_point_threshold=10, connectivity_radius_m=2.0)
        )

        default, changed = _materialize(run), _materialize(run, policy=other)

        assert [e.entity_id for e in default.entities] == [e.entity_id for e in changed.entities]
        assert default.entities[0].provenance.configuration_fingerprint != (
            changed.entities[0].provenance.configuration_fingerprint
        )


class TestCandidateRejection:
    def test_candidates_that_cannot_be_materialized_are_reported_never_dropped(
        self, run: FusionRun
    ) -> None:
        partial = InMemoryGeometrySource(MAP_ID, {i: (i * 0.1, 0.0, 0.0) for i in range(50)})

        result = _materialize(run, geometry=partial)

        assert len(result.entities) + len(result.rejections) == len(run.outcomes)
        assert [item.fusion_support_id for item in result.rejections] == [
            "support-000002",
            "support-000003",
        ]
        assert {item.reason for item in result.rejections} == {
            RejectionReason.UNRESOLVABLE_GEOMETRY
        }
        assert all(item.detail for item in result.rejections)

    def test_an_empty_support_is_a_rejection(
        self, run: FusionRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def empty(*_: object, **__: object) -> None:
            raise EmptyGeometrySupportError("an entity needs at least one geometry reference")

        monkeypatch.setattr("contextmap.semantic_mapping.materialization.summarize_geometry", empty)

        result = _materialize(run)

        assert result.entities == ()
        assert {item.reason for item in result.rejections} == {
            RejectionReason.EMPTY_GEOMETRY_SUPPORT
        }

    def test_invalid_temporal_evidence_is_a_rejection(
        self, run: FusionRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*_: object, **__: object) -> None:
            raise TemporalEvidenceError("no physical observation contributed")

        monkeypatch.setattr(
            "contextmap.semantic_mapping.materialization.summarize_temporal_state", broken
        )

        result = _materialize(run)

        assert {item.reason for item in result.rejections} == {
            RejectionReason.INVALID_TEMPORAL_EVIDENCE
        }

    def test_parts_that_break_the_entity_contract_are_a_rejection(
        self, run: FusionRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*_: object, **__: object) -> None:
            raise ValueError("the parts disagree")

        monkeypatch.setattr(
            "contextmap.semantic_mapping.materialization.semantic_state_from_fused_evidence", broken
        )

        result = _materialize(run)

        assert {item.reason for item in result.rejections} == {RejectionReason.INVALID_ENTITY}
        assert result.rejections[0].detail == "the parts disagree"

    def test_a_clean_run_rejects_nothing(self, run: FusionRun) -> None:
        assert _materialize(run).rejections == ()


class TestSelectionErrors:
    def test_a_support_selected_twice_is_refused(self, run: FusionRun) -> None:
        with pytest.raises(MaterializationInputError, match="more than once"):
            _materialize(run, outcomes=(run.outcomes[0], run.outcomes[0]))

    def test_a_geometry_source_of_another_map_is_refused(self, run: FusionRun) -> None:
        other = InMemoryGeometrySource(MapId("map-0002"), {0: (0.0, 0.0, 0.0)})

        with pytest.raises(MaterializationInputError, match="geometry source serves"):
            _materialize(run, geometry=other)


class TestResultContract:
    def test_a_support_cannot_be_both_an_entity_and_a_rejection(self, run: FusionRun) -> None:
        result = _materialize(run)
        (support,) = result.entities[0].evidence.fused_evidence
        partial = InMemoryGeometrySource(MAP_ID, {i: (i * 0.1, 0.0, 0.0) for i in range(50)})
        rejection = _materialize(run, geometry=partial).rejections[0]
        clash = type(rejection)(
            fusion_support_id=support.fusion_support_id,
            fused_evidence_id=rejection.fused_evidence_id,
            reason=rejection.reason,
            detail=rejection.detail,
        )

        with pytest.raises(ValueError, match="both materialized and rejected"):
            EntityMaterialization(
                semantic_map_id=SEMANTIC_MAP_ID, entities=result.entities[:1], rejections=(clash,)
            )

    def test_entities_must_belong_to_the_map_of_the_materialization(self, run: FusionRun) -> None:
        result = _materialize(run)

        with pytest.raises(ValueError, match="belongs to semantic map"):
            EntityMaterialization(
                semantic_map_id=type(SEMANTIC_MAP_ID)("semantic-map-other"),
                entities=result.entities,
            )
