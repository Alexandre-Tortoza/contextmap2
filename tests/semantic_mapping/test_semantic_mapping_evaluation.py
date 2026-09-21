import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from mapping_builders import (
    SEMANTIC_MAP_ID,
    SUMMARY_POLICY,
    make_semantic_state,
)
from mapping_fusion import FusionRun, write_fusion_run, write_mapping_run, write_run
from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.evaluation import (
    SemanticMappingEvaluationError,
    SemanticMappingEvaluationReport,
    SemanticMappingValidationLayer,
    encode_semantic_mapping_report,
    evaluate_semantic_mapping,
)
from contextmap.evaluation.semantic_mapping import EVALUATOR_VERSION
from contextmap.geometric_mapping import MapId
from contextmap.semantic_fusion import FusionSupportId, SemanticFusionRunReader
from contextmap.semantic_mapping import (
    Entity,
    EntityId,
    EntityMaterializationPolicy,
    EntityTemporalState,
    GeometrySummaryPolicy,
    SemanticMappingRunReader,
    materialize_entities,
    summarize_temporal_state,
)

POLICY = EntityMaterializationPolicy(geometry=SUMMARY_POLICY)
LAYERS = list(SemanticMappingValidationLayer)


@pytest.fixture
def fusion(tmp_path: Path) -> FusionRun:
    return write_fusion_run(tmp_path / "fusion")


def _entities(fusion: FusionRun) -> tuple[Entity, ...]:
    return materialize_entities(
        fusion.outcomes,
        fusion_manifest=fusion.manifest,
        geometry=fusion.geometry,
        semantic_map_id=SEMANTIC_MAP_ID,
        policy=POLICY,
        code_version="test",
    ).entities


def _evaluate(
    tmp_path: Path,
    fusion: FusionRun,
    *,
    tamper: Callable[[tuple[Entity, ...]], Sequence[Entity]] | None = None,
    policy: EntityMaterializationPolicy = POLICY,
) -> SemanticMappingEvaluationReport:
    entities = _entities(fusion)
    run_dir = write_mapping_run(
        tmp_path / "mapping", fusion, entities if tamper is None else tamper(entities)
    )
    return evaluate_semantic_mapping(
        SemanticMappingRunReader(run_dir),
        fusion=SemanticFusionRunReader(fusion.run_dir),
        geometry=fusion.geometry,
        policy=policy,
    )


def _failed(report: SemanticMappingEvaluationReport) -> set[str]:
    return {check.check_id for check in report.failed_checks}


class TestACleanRun:
    def test_every_check_of_every_layer_passes(self, tmp_path: Path, fusion: FusionRun) -> None:
        report = _evaluate(tmp_path, fusion)

        assert report.passed, [(c.check_id, c.failures) for c in report.failed_checks]
        for layer in LAYERS:
            assert report.layer(layer), f"no check for {layer.value}"
        assert report.entity_count == 3 and report.rejected_count == 0

    def test_each_check_says_what_it_examined(self, tmp_path: Path, fusion: FusionRun) -> None:
        report = _evaluate(tmp_path, fusion)

        assert all(check.examined >= 1 for check in report.checks)
        assert {c.check_id for c in report.checks} >= {
            "entity_ids_unique_and_counted",
            "entity_references_resolve",
            "hypotheses_preserved",
            "uncertainty_preserved",
            "abstention_preserved",
            "unscored_signals_preserved",
            "no_forced_single_label",
            "references_resolve",
            "entity_to_source_observation_traversal",
            "first_and_last_seen_reproducible",
            "physical_observations_not_inflated",
            "one_support_one_entity",
            "rematerialization_reproduces_the_entities",
            "run_integrity",
            "entities_reopen_unchanged",
            "derived_indexes_agree_with_the_entities",
        }

    def test_the_report_records_every_identity_needed_to_reproduce_it(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        lineage = _evaluate(tmp_path, fusion).lineage

        assert lineage.fusion_run_id == fusion.manifest.run_id
        assert lineage.fusion_artifact_digest.startswith("sha256:")
        assert lineage.geometric_map_id == fusion.manifest.lineage.geometric_map_id
        assert lineage.materialization_policy_id == "one-support-one-entity-v1"
        assert lineage.identity_policy_id == "support-derived-entity-id-v1"
        assert lineage.entity_schema_version == "0.1.0"
        assert lineage.configuration_fingerprint == POLICY.fingerprint()
        assert lineage.code_version == "test"
        assert lineage.evaluator_version == EVALUATOR_VERSION

    def test_the_report_is_deterministic_and_encodes_without_a_composite_score(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        first = _evaluate(tmp_path / "a", fusion)
        second = _evaluate(tmp_path / "b", fusion)

        assert first.checks == second.checks
        encoded = encode_semantic_mapping_report(first)
        assert json.loads(json.dumps(encoded)) == encoded
        assert set(encoded) == {
            "evaluator_version",
            "lineage",
            "entity_count",
            "rejected_count",
            "passed",
            "checks",
        }
        assert encoded["passed"] is True


class TestDetectsRealDefects:
    def test_a_semantic_state_that_dropped_alternatives_and_forced_a_label(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        def force_a_label(entities: tuple[Entity, ...]) -> list[Entity]:
            contradiction = entities[0]
            door = next(h for h in contradiction.semantic_state.hypotheses if h.label == "door")
            forced = make_semantic_state((door,), primary=door.ref)
            return [dataclasses.replace(contradiction, semantic_state=forced), *entities[1:]]

        report = _evaluate(tmp_path, fusion, tamper=force_a_label)

        assert {
            "hypotheses_preserved",
            "uncertainty_preserved",
            "no_forced_single_label",
            "semantic_state_is_the_mapping_of_the_evidence",
        } <= _failed(report)
        assert not report.passed

    def test_lost_unscored_evidence_is_detected(self, tmp_path: Path, fusion: FusionRun) -> None:
        def score_everything(entities: tuple[Entity, ...]) -> list[Entity]:
            contradiction = entities[0]
            (door,) = [h for h in contradiction.semantic_state.hypotheses if h.label == "door"]
            zeroed = dataclasses.replace(
                door,
                evidence=tuple(
                    dataclasses.replace(
                        item,
                        signals=tuple(
                            dataclasses.replace(
                                signal, value=0.0 if signal.value is None else signal.value
                            )
                            for signal in item.signals
                        ),
                    )
                    for item in door.evidence
                ),
            )
            others = [h for h in contradiction.semantic_state.hypotheses if h.label != "door"]
            state = make_semantic_state(
                tuple(sorted([zeroed, *others], key=lambda h: h.hypothesis_id)),
                uncertainty=contradiction.semantic_state.uncertainty,
            )
            return [dataclasses.replace(contradiction, semantic_state=state), *entities[1:]]

        report = _evaluate(tmp_path, fusion, tamper=score_everything)

        assert "unscored_signals_preserved" in _failed(report)

    def test_inflated_or_incomplete_temporal_state_is_detected(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        def only_first_frame(entities: tuple[Entity, ...]) -> list[Entity]:
            contradiction = entities[0]
            groups = fusion.outcomes[0].evidence.physical_observation_groups[:1]
            state: EntityTemporalState = summarize_temporal_state(groups)
            links = dataclasses.replace(
                contradiction.evidence,
                physical_observation_ids=(groups[0].physical_observation_id,),
            )
            return [
                dataclasses.replace(contradiction, temporal_state=state, evidence=links),
                *entities[1:],
            ]

        report = _evaluate(tmp_path, fusion, tamper=only_first_frame)

        assert {
            "first_and_last_seen_reproducible",
            "physical_observations_not_inflated",
        } <= _failed(report)

    def test_evidence_links_that_list_a_stranger_are_detected(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        def add_a_stranger(entities: tuple[Entity, ...]) -> list[Entity]:
            agreement = entities[1]
            links = dataclasses.replace(
                agreement.evidence,
                spatial_observation_ids=(
                    *agreement.evidence.spatial_observation_ids,
                    "spatial--zzz--stranger",  # type: ignore[arg-type]
                ),
            )
            return [entities[0], dataclasses.replace(agreement, evidence=links), entities[2]]

        report = _evaluate(tmp_path, fusion, tamper=add_a_stranger)

        assert {"references_resolve", "entity_to_source_observation_traversal"} & _failed(report)

    def test_a_support_that_became_two_entities_or_a_renamed_identity_breaks_the_boundary(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        def rename(entities: tuple[Entity, ...]) -> list[Entity]:
            first = dataclasses.replace(entities[0], entity_id=EntityId("entity--support-000001-b"))
            return [entities[0], first, *entities[1:]]

        report = _evaluate(tmp_path, fusion, tamper=rename)

        assert {
            "one_support_one_entity",
            "identity_is_a_function_of_the_support",
            "rematerialization_reproduces_the_entities",
        } <= _failed(report)

    def test_a_tampered_geometry_summary_is_detected(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        def move_centroid(entities: tuple[Entity, ...]) -> list[Entity]:
            first = entities[0]
            low, high = first.geometry.bounds.minimum_m, first.geometry.bounds.maximum_m
            centroid = ((low[0] + high[0]) / 4, low[1], low[2])
            moved = dataclasses.replace(first.geometry, centroid_m=centroid)
            return [dataclasses.replace(first, geometry=moved), *entities[1:]]

        report = _evaluate(tmp_path, fusion, tamper=move_centroid)

        assert "geometry_valid_and_authoritative" in _failed(report)

    def test_an_entity_that_names_another_sequence_than_the_lineage_breaks_provenance(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        run_dir = write_mapping_run(tmp_path / "mapping", fusion, _entities(fusion))
        table = run_dir / "outputs" / "entities.jsonl"
        # Mesmo comprimento, para que os deslocamentos do índice continuem válidos.
        table.write_text(
            table.read_text(encoding="utf-8").replace(
                '"sequence_artifact_id":"sequence-0001"', '"sequence_artifact_id":"sequence-0002"'
            ),
            encoding="utf-8",
        )

        report = evaluate_semantic_mapping(
            SemanticMappingRunReader(run_dir),
            fusion=SemanticFusionRunReader(fusion.run_dir),
            geometry=fusion.geometry,
            policy=POLICY,
        )

        assert "provenance_complete" in _failed(report)

    def test_evaluating_with_another_policy_than_the_one_used_is_detected(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        other = EntityMaterializationPolicy(
            geometry=GeometrySummaryPolicy(sparse_point_threshold=10, connectivity_radius_m=2.0)
        )

        report = _evaluate(tmp_path, fusion, policy=other)

        assert {"provenance_complete", "rematerialization_reproduces_the_entities"} <= _failed(
            report
        )

    def test_fused_evidence_that_cannot_be_read_is_reported_per_entity(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        def dangling(entities: tuple[Entity, ...]) -> list[Entity]:
            first = entities[0]
            (ref,) = first.evidence.fused_evidence
            gone = dataclasses.replace(ref, fusion_support_id=FusionSupportId("support-9999"))
            broken = dataclasses.replace(
                first, evidence=dataclasses.replace(first.evidence, fused_evidence=(gone,))
            )
            return [broken, *entities[1:]]

        report = _evaluate(tmp_path, fusion, tamper=dangling)

        preservation = report.check("hypotheses_preserved")
        assert any("cannot read its fused evidence" in failure for failure in preservation.failures)

    def test_corruption_of_the_persisted_run_is_detected(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        run_dir = write_mapping_run(tmp_path / "mapping", fusion, _entities(fusion))
        index = run_dir / "outputs" / "entity-semantic-state.jsonl"
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        rows[0]["hypothesis_labels"] = ["something else"]
        index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

        report = evaluate_semantic_mapping(
            SemanticMappingRunReader(run_dir),
            fusion=SemanticFusionRunReader(fusion.run_dir),
            geometry=fusion.geometry,
            policy=POLICY,
        )

        assert {"run_integrity", "derived_indexes_agree_with_the_entities"} <= _failed(report)
        assert (
            report.check("run_integrity").layer
            is SemanticMappingValidationLayer.ARTIFACT_ROUND_TRIP
        )


class TestWrongUpstream:
    def test_another_fusion_run_than_the_one_in_the_lineage_is_refused(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        run_dir = write_mapping_run(tmp_path / "mapping", fusion, _entities(fusion))
        other = write_run(tmp_path / "other-fusion", fusion.outcomes[:1])

        with pytest.raises(SemanticMappingEvaluationError, match="not the one"):
            evaluate_semantic_mapping(
                SemanticMappingRunReader(run_dir),
                fusion=SemanticFusionRunReader(other.run_dir),
                geometry=fusion.geometry,
                policy=POLICY,
            )

    def test_a_corrupt_fusion_payload_is_reported_not_raised(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        run_dir = write_mapping_run(tmp_path / "mapping", fusion, _entities(fusion))
        target = fusion.run_dir / "outputs" / "fused-evidence.jsonl"
        target.write_bytes(target.read_bytes() + b"\n")

        report = evaluate_semantic_mapping(
            SemanticMappingRunReader(run_dir),
            fusion=SemanticFusionRunReader(fusion.run_dir),
            geometry=fusion.geometry,
            policy=POLICY,
        )

        assert "references_resolve" in _failed(report)

    def test_a_geometry_source_of_another_map_is_refused(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        run_dir = write_mapping_run(tmp_path / "mapping", fusion, _entities(fusion))
        other_map = InMemoryGeometrySource(MapId("map-0002"), {0: (0.0, 0.0, 0.0)})

        with pytest.raises(SemanticMappingEvaluationError, match="names"):
            evaluate_semantic_mapping(
                SemanticMappingRunReader(run_dir),
                fusion=SemanticFusionRunReader(fusion.run_dir),
                geometry=other_map,
                policy=POLICY,
            )


class TestScope:
    def test_the_harness_never_evaluates_same_object_accuracy_or_relations(self) -> None:
        from contextmap.evaluation import semantic_mapping as harness

        names = " ".join(dir(harness)).lower()

        assert "relation" not in names and "resolution_accuracy" not in names
        assert "repair" not in names

    def test_an_empty_run_is_validated_without_error(
        self, tmp_path: Path, fusion: FusionRun
    ) -> None:
        run_dir = write_mapping_run(tmp_path / "mapping", fusion, ())

        report = evaluate_semantic_mapping(
            SemanticMappingRunReader(run_dir),
            fusion=SemanticFusionRunReader(fusion.run_dir),
            geometry=fusion.geometry,
            policy=POLICY,
        )

        assert report.entity_count == 0
        assert report.passed, [(c.check_id, c.failures) for c in report.failed_checks]
