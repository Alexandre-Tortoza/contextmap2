import dataclasses
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from mapping_builders import MAP_ID, SEMANTIC_MAP_ID, SUMMARY_POLICY
from mapping_fusion import FusionRun, write_fusion_run
from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import SemanticFusionRunId
from contextmap.semantic_mapping import (
    AmbiguityState,
    CandidateRejection,
    Entity,
    EntityId,
    EntityMaterialization,
    EntityMaterializationPolicy,
    EntityReference,
    EvidenceIntegrityKind,
    ForeignEntityReferenceError,
    IncompleteMappingRunArtifactError,
    MappingDebugLevel,
    MappingRunArtifactError,
    MappingRunLineage,
    RejectionReason,
    SemanticMapId,
    SemanticMappingRunId,
    SemanticMappingRunReader,
    SemanticMappingRunWriter,
    UnknownEntityError,
    lineage_from_fusion_manifest,
    materialize_entities,
)

FUSION_RUN_ID = SemanticFusionRunId("fusion-run-0001")


@dataclass
class Mapped:
    fusion: FusionRun
    materialization: EntityMaterialization
    workspace: Path
    run_dir: Path
    reader: SemanticMappingRunReader


def _materialize(fusion: FusionRun, geometry: InMemoryGeometrySource | None = None) -> Any:
    return materialize_entities(
        fusion.outcomes,
        fusion_manifest=fusion.manifest,
        geometry=fusion.geometry if geometry is None else geometry,
        semantic_map_id=SEMANTIC_MAP_ID,
        policy=EntityMaterializationPolicy(geometry=SUMMARY_POLICY),
        code_version="test",
    )


def _writer(
    workspace: Path,
    fusion: FusionRun,
    *,
    run_index: int = 1,
    debug: MappingDebugLevel = MappingDebugLevel.NONE,
    semantic_map_id: SemanticMapId = SEMANTIC_MAP_ID,
    lineage: MappingRunLineage | None = None,
) -> SemanticMappingRunWriter:
    return SemanticMappingRunWriter(
        output_dir=_run_dir(workspace, run_index),
        sequence_name="sequence-0001",
        run_id=SemanticMappingRunId(f"mapping-run-{run_index:04d}"),
        run_index=run_index,
        semantic_map_id=semantic_map_id,
        lineage=lineage or lineage_from_fusion_manifest(fusion.manifest),
        code_version="test",
        debug_level=debug,
    )


def _run_dir(workspace: Path, run_index: int = 1) -> Path:
    """Onde o writer grava: o chamador decide o diretório final, o writer não calcula caminho."""
    return workspace / f"run-{run_index:04d}"


def _write(
    tmp_path: Path,
    *,
    debug: MappingDebugLevel = MappingDebugLevel.NONE,
    geometry: InMemoryGeometrySource | None = None,
    runtime: dict[str, float | int | None] | None = None,
) -> Mapped:
    fusion = write_fusion_run(tmp_path / "fusion")
    result = _materialize(fusion, geometry)
    workspace = tmp_path / "mapping"
    _writer(workspace, fusion, debug=debug).write(
        result.entities,
        rejections=result.rejections,
        warnings=("one warning",),
        runtime=runtime,
    )
    run_dir = _run_dir(workspace)
    return Mapped(
        fusion=fusion,
        materialization=result,
        workspace=workspace,
        run_dir=run_dir,
        reader=SemanticMappingRunReader(run_dir),
    )


@pytest.fixture
def mapped(tmp_path: Path) -> Mapped:
    return _write(tmp_path)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _partial() -> InMemoryGeometrySource:
    return InMemoryGeometrySource(MAP_ID, {i: (i * 0.1, 0.0, 0.0) for i in range(50)})


class TestRoundTrip:
    def test_a_reference_resolves_to_the_same_canonical_entity_after_reopen(
        self, mapped: Mapped
    ) -> None:
        reopened = SemanticMappingRunReader(mapped.run_dir)

        for entity in mapped.materialization.entities:
            assert reopened.entity(entity.reference) == entity

    def test_the_whole_run_reloads_as_an_entity_set(self, mapped: Mapped) -> None:
        loaded = mapped.reader.entities()

        assert loaded.semantic_map_id == SEMANTIC_MAP_ID
        assert loaded.entities == mapped.materialization.entities
        first = mapped.materialization.entities[0]
        assert loaded.resolve(first.reference) == first

    def test_geometry_alternatives_conflicts_evidence_and_time_survive_intact(
        self, mapped: Mapped
    ) -> None:
        contradiction, agreement, abstention = mapped.materialization.entities

        back = mapped.reader.entity(contradiction.reference)

        assert back.geometry == contradiction.geometry
        assert back.semantic_state.ambiguity_state is AmbiguityState.CONFLICTING
        assert back.semantic_state.hypotheses == contradiction.semantic_state.hypotheses
        assert back.semantic_state.uncertainty == contradiction.semantic_state.uncertainty
        assert back.evidence == contradiction.evidence
        assert back.temporal_state == contradiction.temporal_state
        unscored = [
            signal.value
            for hypothesis in back.semantic_state.hypotheses
            for item in hypothesis.evidence
            for signal in item.signals
            if signal.value is None
        ]
        assert unscored
        assert mapped.reader.entity(abstention.reference).semantic_state.hypotheses == ()
        assert mapped.reader.entity(agreement.reference).semantic_state.primary is not None

    def test_one_entity_is_read_without_loading_the_others(self, mapped: Mapped) -> None:
        first, second, _ = mapped.materialization.entities
        target = mapped.run_dir / "outputs" / "entities.jsonl"
        lines = target.read_bytes().splitlines(keepends=True)
        # Adultera a segunda linha sem mudar offsets: só ler a entidade certa continua funcionando.
        lines[1] = b"x" * (len(lines[1]) - 1) + b"\n"
        target.write_bytes(b"".join(lines))
        reader = SemanticMappingRunReader(mapped.run_dir)

        assert reader.entity(first.reference) == first
        with pytest.raises(MappingRunArtifactError, match="malformed entity"):
            reader.entity(second.reference)

    def test_it_reopens_without_perception_fusion_or_model_runtimes(self, tmp_path: Path) -> None:
        mapped = _write(tmp_path)
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "from contextmap.semantic_mapping import SemanticMappingRunReader\n"
            f"reader = SemanticMappingRunReader(Path({str(mapped.run_dir)!r}))\n"
            "assert len(list(reader.iter_entities())) == 3\n"
            "heavy = {'torch', 'transformers', 'rclpy', 'rosbags', 'segment_anything'}\n"
            "assert not heavy & set(sys.modules), sorted(heavy & set(sys.modules))\n"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

        completed = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )

        assert completed.returncode == 0, completed.stderr


class TestLayoutAndManifest:
    def test_contractual_outputs_exist_and_each_is_inventoried(self, mapped: Mapped) -> None:
        manifest = mapped.reader.manifest
        expected = {
            "outputs/entities.jsonl",
            "outputs/entity-index.jsonl",
            "outputs/entity-geometry-index.jsonl",
            "outputs/entity-evidence-index.jsonl",
            "outputs/entity-observation-index.jsonl",
            "outputs/entity-semantic-state.jsonl",
            "outputs/entity-temporal-state.jsonl",
            "outputs/rejected-candidates.jsonl",
            "metrics/counts.json",
            "metrics/distributions.json",
            "metrics/payload.json",
        }

        assert {entry.path for entry in manifest.file_inventory} == expected
        assert (mapped.run_dir / "README.md").is_file()
        assert mapped.reader.verify_integrity() == []

    def test_the_manifest_records_identity_lineage_policies_and_counts(
        self, mapped: Mapped
    ) -> None:
        manifest = mapped.reader.manifest
        fusion_manifest = mapped.fusion.manifest

        assert manifest.run_id == "mapping-run-0001" and manifest.run_index == 1
        assert manifest.semantic_map_id == SEMANTIC_MAP_ID
        assert manifest.lineage.fusion_run_id == fusion_manifest.run_id
        assert manifest.lineage.fusion_schema_version == fusion_manifest.schema_version
        assert manifest.lineage.geometric_map_id == fusion_manifest.lineage.geometric_map_id
        assert manifest.lineage.association_run_ids == fusion_manifest.lineage.association_run_ids
        assert manifest.lineage.perception_run_ids == fusion_manifest.lineage.perception_run_ids
        assert manifest.lineage.point_representation_run_ids == (
            fusion_manifest.lineage.point_representation_run_ids
        )
        assert manifest.materialization_policy_id == "one-support-one-entity-v1"
        assert manifest.identity_policy_id == "support-derived-entity-id-v1"
        assert manifest.configuration_fingerprint
        assert (manifest.entity_count, manifest.rejected_count) == (3, 0)
        assert manifest.warnings == ("one warning",)
        assert manifest.code_version == "test"

    def test_an_empty_run_is_valid_and_explicit(self, tmp_path: Path) -> None:
        fusion = write_fusion_run(tmp_path / "fusion")
        _writer(tmp_path / "mapping", fusion).write([])

        reader = SemanticMappingRunReader(_run_dir(tmp_path / "mapping"))

        assert reader.manifest.entity_count == 0
        assert reader.manifest.materialization_policy_id is None
        assert reader.entity_ids() == [] and reader.entities().entities == ()

    def test_the_indexes_summarize_each_entity(self, mapped: Mapped) -> None:
        contradiction, *_ = mapped.materialization.entities
        outputs = mapped.run_dir / "outputs"

        geometry = _rows(outputs / "entity-geometry-index.jsonl")[0]
        semantic = _rows(outputs / "entity-semantic-state.jsonl")[0]
        temporal = _rows(outputs / "entity-temporal-state.jsonl")[0]
        evidence = _rows(outputs / "entity-evidence-index.jsonl")[0]

        assert geometry["entity_id"] == contradiction.entity_id
        assert geometry["point_count"] == len(contradiction.geometry.geometry_refs)
        assert semantic["ambiguity_state"] == "conflicting"
        assert semantic["primary_label"] is None and semantic["conflict_count"] == 1
        assert temporal["physical_observation_count"] == 2
        assert evidence["fused_evidence"][0]["fusion_run_id"] == FUSION_RUN_ID
        assert evidence["visual_feature_count"] == 3

    def test_a_physical_observation_resolves_to_the_entities_it_contributed_to(
        self, mapped: Mapped
    ) -> None:
        reader = mapped.reader

        assert reader.entities_of_observation(SourceObservationId("frame-0120")) == [
            "entity--support-000001"
        ]
        assert reader.entities_of_observation(SourceObservationId("frame-9999")) == []

    def test_a_downstream_consumer_reads_contractual_tables_and_never_debug(
        self, tmp_path: Path
    ) -> None:
        mapped = _write(tmp_path, debug=MappingDebugLevel.FULL)

        assert len(mapped.reader.read_table("outputs/entity-index.jsonl")) == 3
        with pytest.raises(MappingRunArtifactError, match="not a contractual table"):
            mapped.reader.read_table("debug/entities/entity--support-000001/summary.json")
        with pytest.raises(MappingRunArtifactError, match="not a contractual JSON record"):
            mapped.reader.read_record("debug/entities/entity--support-000001/summary.json")

    def test_there_is_no_resolution_state_in_the_artifact(self, mapped: Mapped) -> None:
        for path in (mapped.run_dir / "outputs").iterdir():
            text = path.read_text()
            assert "merge" not in text and "resolved" not in text.lower().replace("unresolved", "")


class TestReferences:
    def test_a_reference_of_another_semantic_map_is_refused(self, mapped: Mapped) -> None:
        foreign = EntityReference(
            semantic_map_id=SemanticMapId("semantic-map-other"),
            entity_id=EntityId("entity--support-000001"),
        )

        with pytest.raises(ForeignEntityReferenceError):
            mapped.reader.entity(foreign)

    def test_an_unknown_entity_is_an_explicit_error(self, mapped: Mapped) -> None:
        unknown = EntityReference(
            semantic_map_id=SEMANTIC_MAP_ID, entity_id=EntityId("entity--support-999999")
        )

        with pytest.raises(UnknownEntityError):
            mapped.reader.entity(unknown)

    def test_the_upstream_references_still_validate_after_the_reopen(self, mapped: Mapped) -> None:
        issues = mapped.reader.validate_references(
            fusion_runs={FUSION_RUN_ID: mapped.fusion.reader}, geometry=mapped.fusion.geometry
        )

        assert issues == ()

    def test_a_missing_upstream_run_is_reported_explicitly(self, mapped: Mapped) -> None:
        issues = mapped.reader.validate_references(fusion_runs={})

        assert {item.kind for item in issues} == {EvidenceIntegrityKind.MISSING_ARTIFACT}

    def test_a_corrupt_upstream_payload_is_reported_explicitly(self, mapped: Mapped) -> None:
        target = mapped.fusion.run_dir / "outputs" / "fused-evidence.jsonl"
        target.write_bytes(target.read_bytes() + b"\n")

        issues = mapped.reader.validate_references(
            fusion_runs={FUSION_RUN_ID: mapped.fusion.reader}
        )

        assert EvidenceIntegrityKind.CORRUPT_ARTIFACT in {item.kind for item in issues}


class TestIntegrityAndCorruption:
    def test_changed_content_is_detected(self, mapped: Mapped) -> None:
        target = mapped.run_dir / "outputs" / "entities.jsonl"
        target.write_bytes(target.read_bytes() + b"\n")

        problems = SemanticMappingRunReader(mapped.run_dir).verify_integrity()

        assert any("entities.jsonl" in problem for problem in problems)

    def test_a_missing_contractual_file_is_detected(self, mapped: Mapped) -> None:
        (mapped.run_dir / "outputs" / "entity-index.jsonl").unlink()

        problems = SemanticMappingRunReader(mapped.run_dir).verify_integrity()

        assert any("missing file" in problem for problem in problems)

    def test_a_truncated_payload_is_an_explicit_error(self, mapped: Mapped) -> None:
        target = mapped.run_dir / "outputs" / "entities.jsonl"
        data = target.read_bytes()
        target.write_bytes(data[: len(data) // 2])
        reader = SemanticMappingRunReader(mapped.run_dir)

        with pytest.raises(MappingRunArtifactError, match="truncated"):
            reader.entity(mapped.materialization.entities[-1].reference)

    def test_a_malformed_table_is_an_explicit_error(self, mapped: Mapped) -> None:
        (mapped.run_dir / "outputs" / "entity-index.jsonl").write_text("{broken\n")

        with pytest.raises(MappingRunArtifactError, match="malformed table"):
            SemanticMappingRunReader(mapped.run_dir).entity_ids()

    def test_a_directory_without_manifest_is_not_a_run(self, tmp_path: Path) -> None:
        with pytest.raises(IncompleteMappingRunArtifactError):
            SemanticMappingRunReader(tmp_path)

    def test_an_unsupported_schema_version_is_refused(self, mapped: Mapped) -> None:
        manifest = mapped.run_dir / "manifest.json"
        record = json.loads(manifest.read_text())
        record["schema_version"] = "9.9.9"
        manifest.write_text(json.dumps(record))

        with pytest.raises(MappingRunArtifactError, match="unsupported"):
            SemanticMappingRunReader(mapped.run_dir)

    def test_a_tampered_entity_record_is_refused_rather_than_trusted(self, mapped: Mapped) -> None:
        first = mapped.materialization.entities[0]
        row = _rows(mapped.run_dir / "outputs" / "entity-index.jsonl")[0]
        target = mapped.run_dir / "outputs" / "entities.jsonl"
        data = target.read_bytes()
        start, end = row["offset"], row["offset"] + row["length"]
        record = json.loads(data[start:end])
        # O histórico tem 2 frames: uma contagem de 3 viola o contrato, com o mesmo tamanho.
        record["temporal_state"]["physical_observation_count"] = 3
        patched = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        assert len(patched) == row["length"]
        target.write_bytes(data[:start] + patched + data[end:])

        with pytest.raises(MappingRunArtifactError, match="malformed entity"):
            SemanticMappingRunReader(mapped.run_dir).entity(first.reference)


class TestDebugEvidence:
    def test_none_writes_no_debug(self, mapped: Mapped) -> None:
        assert not (mapped.run_dir / "debug").exists()

    @pytest.mark.parametrize("level", [MappingDebugLevel.STANDARD, MappingDebugLevel.FULL])
    def test_debug_summarizes_each_entity_and_is_never_inventoried(
        self, tmp_path: Path, level: MappingDebugLevel
    ) -> None:
        mapped = _write(tmp_path, debug=level)
        base = mapped.run_dir / "debug" / "entities" / "entity--support-000001"

        assert {path.name for path in base.iterdir()} == {
            "summary.json",
            "geometry-summary.json",
            "semantic-state.json",
            "evidence-trace.json",
            "temporal-history.json",
        }
        assert json.loads((base / "summary.json").read_text())["ambiguity_state"] == "conflicting"
        assert not any(
            entry.path.startswith("debug/") for entry in mapped.reader.manifest.file_inventory
        )

    def test_removing_debug_cannot_invalidate_the_run(self, tmp_path: Path) -> None:
        mapped = _write(tmp_path, debug=MappingDebugLevel.FULL)
        shutil.rmtree(mapped.run_dir / "debug")
        reader = SemanticMappingRunReader(mapped.run_dir)

        assert reader.verify_integrity() == []
        assert reader.entities().entities == mapped.materialization.entities


class TestMetrics:
    def test_counts_are_derived_from_the_entities(self, mapped: Mapped) -> None:
        counts = mapped.reader.read_record("metrics/counts.json")
        entities = mapped.materialization.entities

        assert counts["entities"] == 3
        assert counts["rejected_candidates"]["total"] == 0
        assert counts["semantic_state"] == {
            "unambiguous": 1,
            "ambiguous": 0,
            "conflicting": 1,
            "insufficient_evidence": 1,
        }
        assert counts["entities_with_primary_hypothesis"] == 1
        assert counts["entities_with_conflicts"] == 1
        assert counts["entities_without_visual_features"] == 2
        assert counts["entities_without_point_representations"] == 2
        assert counts["physical_observations"] == len(
            {f for e in entities for f in e.evidence.physical_observation_ids}
        )
        assert counts["inference_results"] == sum(
            e.temporal_state.inference_result_count for e in entities
        )
        assert counts["warnings"] == 1

    def test_distributions_report_size_and_observation_counts_per_entity(
        self, mapped: Mapped
    ) -> None:
        distributions = mapped.reader.read_record("metrics/distributions.json")

        for name in (
            "geometry_points_per_entity",
            "physical_observations_per_entity",
            "inference_results_per_entity",
            "hypotheses_per_entity",
        ):
            assert distributions[name]["count"] == 3
        assert distributions["physical_observations_per_entity"]["max"] == 2

    def test_payload_sizes_match_the_files(self, mapped: Mapped) -> None:
        payload = mapped.reader.read_record("metrics/payload.json")

        for relative, size in payload["files"].items():
            assert (mapped.run_dir / relative).stat().st_size == size
        assert payload["total_bytes"] == sum(payload["files"].values())

    def test_runtime_is_recorded_only_when_measured(self, tmp_path: Path) -> None:
        without = _write(tmp_path / "a")
        measured = _write(tmp_path / "b", runtime={"seconds": 1.5, "peak_memory_mb": None})

        assert not (without.run_dir / "metrics" / "runtime.json").exists()
        assert measured.reader.read_record("metrics/runtime.json") == {
            "seconds": 1.5,
            "peak_memory_mb": None,
        }

    def test_rejected_candidates_are_persisted_with_their_reason(self, tmp_path: Path) -> None:
        mapped = _write(tmp_path, geometry=_partial())

        rejected = mapped.reader.rejected_candidates()
        counts = mapped.reader.read_record("metrics/counts.json")

        assert [item.fusion_support_id for item in rejected] == ["support-000002", "support-000003"]
        assert {item.reason for item in rejected} == {RejectionReason.UNRESOLVABLE_GEOMETRY}
        assert counts["rejected_candidates"]["by_reason"]["unresolvable_geometry"] == 2
        assert mapped.reader.manifest.rejected_count == 2
        assert mapped.reader.manifest.entity_count == 1


class TestWriterRefusals:
    def _fusion(self, tmp_path: Path) -> tuple[FusionRun, EntityMaterialization]:
        fusion = write_fusion_run(tmp_path / "fusion")
        return fusion, _materialize(fusion)

    def test_entities_must_be_strictly_sorted(self, tmp_path: Path) -> None:
        fusion, result = self._fusion(tmp_path)

        with pytest.raises(MappingRunArtifactError, match="strictly sorted"):
            _writer(tmp_path / "m", fusion).write(tuple(reversed(result.entities)))
        with pytest.raises(MappingRunArtifactError, match="strictly sorted"):
            _writer(tmp_path / "m2", fusion).write((result.entities[0], result.entities[0]))

    def test_an_entity_of_another_semantic_map_is_refused(self, tmp_path: Path) -> None:
        fusion, result = self._fusion(tmp_path)

        with pytest.raises(MappingRunArtifactError, match="belongs to semantic map"):
            _writer(
                tmp_path / "m", fusion, semantic_map_id=SemanticMapId("semantic-map-other")
            ).write(result.entities)

    def test_a_lineage_over_another_map_is_refused(self, tmp_path: Path) -> None:
        fusion, result = self._fusion(tmp_path)
        lineage = dataclasses.replace(
            lineage_from_fusion_manifest(fusion.manifest), geometric_map_id=MapId("map-0002")
        )

        with pytest.raises(MappingRunArtifactError, match="lineage names"):
            _writer(tmp_path / "m", fusion, lineage=lineage).write(result.entities)

    def test_an_entity_from_a_fusion_run_the_lineage_does_not_name_is_refused(
        self, tmp_path: Path
    ) -> None:
        fusion, result = self._fusion(tmp_path)
        lineage = dataclasses.replace(
            lineage_from_fusion_manifest(fusion.manifest),
            fusion_run_id=SemanticFusionRunId("another-fusion-run"),
        )

        with pytest.raises(MappingRunArtifactError, match="not the run the lineage names"):
            _writer(tmp_path / "m", fusion, lineage=lineage).write(result.entities)

    def test_one_run_keeps_one_policy(self, tmp_path: Path) -> None:
        fusion, result = self._fusion(tmp_path)
        first, second, third = result.entities
        other = dataclasses.replace(
            second,
            provenance=dataclasses.replace(second.provenance, configuration_fingerprint="sha256:x"),
        )

        with pytest.raises(MappingRunArtifactError, match="different policy"):
            _writer(tmp_path / "m", fusion).write((first, other, third))

    def test_a_candidate_cannot_be_both_an_entity_and_a_rejection(self, tmp_path: Path) -> None:
        fusion, result = self._fusion(tmp_path)
        entity = result.entities[0]
        (ref,) = entity.evidence.fused_evidence
        clash = CandidateRejection(
            fusion_support_id=ref.fusion_support_id,
            fused_evidence_id=ref.fused_evidence_id,
            reason=RejectionReason.INVALID_ENTITY,
            detail="clash",
        )

        with pytest.raises(MappingRunArtifactError, match="both an entity and a rejection"):
            _writer(tmp_path / "m", fusion).write(result.entities, rejections=(clash,))

    def test_rejections_must_be_sorted(self, tmp_path: Path) -> None:
        fusion = write_fusion_run(tmp_path / "fusion")
        result = _materialize(fusion, _partial())
        first, second = result.rejections

        with pytest.raises(MappingRunArtifactError, match="rejected candidates must be sorted"):
            _writer(tmp_path / "m", fusion).write(result.entities, rejections=(second, first))

    def test_a_run_is_never_overwritten(self, tmp_path: Path) -> None:
        mapped = _write(tmp_path)

        with pytest.raises(MappingRunArtifactError, match="already exists"):
            _writer(mapped.workspace, mapped.fusion).write(mapped.materialization.entities)

    def test_an_interrupted_write_leaves_no_visible_run(self, tmp_path: Path) -> None:
        fusion, result = self._fusion(tmp_path)

        def interrupted() -> Iterator[Entity]:
            yield result.entities[0]
            raise RuntimeError("interrupted")

        with pytest.raises(RuntimeError, match="interrupted"):
            _writer(tmp_path / "m", fusion).write(interrupted())

        parent = _run_dir(tmp_path / "m").parent
        assert not _run_dir(tmp_path / "m").exists()
        assert [item for item in parent.iterdir() if not item.name.startswith(".tmp-")] == []


class TestOutputDirectory:
    def test_the_run_is_written_exactly_where_the_caller_says_and_nothing_else_is_created(
        self, tmp_path: Path
    ) -> None:
        fusion = write_fusion_run(tmp_path / "fusion")
        result = _materialize(fusion, None)
        target = tmp_path / "ws" / "corridor-02" / "run-0001" / "semantic_mapping"
        writer = SemanticMappingRunWriter(
            output_dir=target,
            sequence_name="sequence-0001",
            run_id=SemanticMappingRunId("mapping-run"),
            run_index=1,
            semantic_map_id=SEMANTIC_MAP_ID,
            lineage=lineage_from_fusion_manifest(fusion.manifest),
            code_version="test",
        )

        writer.write(result.entities, rejections=result.rejections)

        assert SemanticMappingRunReader(target).manifest.run_id == SemanticMappingRunId(
            "mapping-run"
        )
        # Sem registro `runs.json` e sem `runs/<capability>/...`: só o diretório do artifact.
        assert sorted(path.name for path in target.parent.iterdir()) == ["semantic_mapping"]

    def test_the_run_id_and_index_are_recorded_as_supplied_and_never_allocated(
        self, tmp_path: Path
    ) -> None:
        fusion = write_fusion_run(tmp_path / "fusion")
        result = _materialize(fusion, None)

        _writer(tmp_path / "m", fusion, run_index=7).write(result.entities)

        manifest = SemanticMappingRunReader(_run_dir(tmp_path / "m", 7)).manifest
        assert (manifest.run_id, manifest.run_index) == (
            SemanticMappingRunId("mapping-run-0007"),
            7,
        )
        assert not _run_dir(tmp_path / "m", 1).exists()
