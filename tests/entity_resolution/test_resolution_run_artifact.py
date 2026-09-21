"""The persisted run artifact: immutable, atomic, auditable, readable without any runtime."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from resolution_builders import channel_policy, geometry_evidence, match_evidence
from resolution_run_fixtures import (
    LINEAGE,
    RUN,
    RunInputs,
    build_inputs,
    resolution_of,
    write_run,
)

from contextmap.entity_resolution import (
    ComparisonId,
    EntityResolutionRunManifest,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    ForeignResolvedEntityReferenceError,
    IncompleteRunArtifactError,
    PairResolution,
    PolicyRef,
    ResolutionDebugLevel,
    ResolutionOutcome,
    ResolutionRunLineage,
    ResolvedEntityReference,
    RunArtifactError,
    UnknownResolvedEntityError,
    comparison_id_for,
    lineage_from_mapping_manifest,
    mapping_artifact_digest,
    materialize_resolved_entities,
)
from contextmap.entity_resolution.channels import GeometryEvidence
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import (
    MappingRunLineage,
    SemanticMapId,
    SemanticMappingRunManifest,
)
from contextmap.shared import FileEntry

MATCH, DISTINCT, UNRESOLVED = (
    ResolutionOutcome.MATCH,
    ResolutionOutcome.DISTINCT,
    ResolutionOutcome.UNRESOLVED,
)


def opened(tmp_path: Path, **kwargs: Any) -> tuple[RunInputs, Path, EntityResolutionRunReader]:
    directory = tmp_path / "artifact"
    inputs, _ = write_run(directory, **kwargs)
    return inputs, directory, EntityResolutionRunReader(directory)


# --- round trip ---------------------------------------------------------------------------------


def test_everything_survives_a_round_trip_intact(tmp_path: Path) -> None:
    inputs, _, reader = opened(tmp_path)

    assert reader.candidate_sets() == inputs.candidate_sets
    assert {item.comparison_id: item for item in reader.match_evidence()} == {
        item.evidence.comparison_id: item.evidence for item in inputs.resolutions
    }
    assert tuple(sorted(reader.decisions(), key=lambda d: d.decision_id)) == tuple(
        sorted((item.decision for item in inputs.resolutions), key=lambda d: d.decision_id)
    )
    assert reader.resolved_entities() == inputs.materialization.resolved
    assert reader.contradictions() == inputs.materialization.contradictions
    assert reader.materialization() == inputs.materialization
    assert reader.split_candidates() == ()


def test_match_distinct_and_unresolved_survive_intact(tmp_path: Path) -> None:
    inputs, _, reader = opened(tmp_path)

    written = {item.decision.decision_id: item.decision.decision for item in inputs.resolutions}
    read = {item.decision_id: item.decision for item in reader.decisions()}

    assert read == written
    assert set(read.values()) == {MATCH, DISTINCT, UNRESOLVED}
    assert reader.manifest.count("decision_match") == 3
    assert reader.manifest.count("decision_distinct") == 2
    assert reader.manifest.count("decision_unresolved") == 1


def test_contradictions_and_unresolved_evidence_are_preserved_not_discarded(tmp_path: Path) -> None:
    inputs, _, reader = opened(tmp_path)

    (contradiction,) = reader.contradictions()
    assert contradiction == inputs.materialization.contradictions[0]
    unresolved = {str(item.entity_ref.entity_id): item for item in reader.unresolved_entities()}
    assert set(unresolved) == {"a", "b", "c", "d"}
    assert unresolved["a"].contradiction_ids == (contradiction.contradiction_id,)
    assert [str(ref.entity_id) for ref in unresolved["d"].unresolved_neighbor_refs] == ["a"]


def test_every_resolved_entity_traces_to_its_exact_source_entities_and_decisions(
    tmp_path: Path,
) -> None:
    inputs, _, reader = opened(tmp_path)
    decisions = {item.decision_id for item in reader.decisions()}
    entities = inputs.entities

    merged = reader.resolved_entity(reader.resolved_of(entities["e"].reference))

    assert [str(ref.entity_id) for ref in merged.member_entity_refs] == ["e", "f"]
    assert set(merged.resolution_decision_refs) <= decisions
    assert reader.resolved_of(entities["f"].reference) == merged.reference
    for name, entity in entities.items():
        resolved = reader.resolved_entity(reader.resolved_of(entity.reference))
        assert entity.reference in resolved.member_entity_refs, name


# --- layout, identity and immutability ----------------------------------------------------------


def test_the_run_lives_exactly_at_the_output_directory_without_a_registry(tmp_path: Path) -> None:
    directory = tmp_path / "workspace" / "corridor-02" / "run-x" / "entity_resolution"
    directory.parent.mkdir(parents=True)

    write_run(directory)

    assert (directory / "manifest.json").is_file()
    assert {path.name for path in directory.parent.iterdir()} == {"entity_resolution"}
    assert not list(tmp_path.rglob("runs.json"))
    assert not list(tmp_path.rglob(".tmp-*"))
    assert {"manifest.json", "README.md", "outputs", "metrics"} <= {
        p.name for p in directory.iterdir()
    }


def test_a_finished_run_is_never_overwritten(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    write_run(directory)
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}

    with pytest.raises(RunArtifactError, match="already exists"):
        write_run(directory)

    assert {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


def test_an_interrupted_write_leaves_no_run_behind(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    inputs = build_inputs()
    writer = EntityResolutionRunWriter(
        output_dir=directory, run_id=RUN, lineage=LINEAGE, code_version="test-version"
    )

    with pytest.raises(TypeError):
        writer.write(
            candidate_sets=inputs.candidate_sets,
            resolutions=inputs.resolutions,
            materialization=inputs.materialization,
            runtime={"seconds": object()},  # type: ignore[dict-item]
        )

    assert not directory.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_the_run_identity_is_the_scope_of_every_resolved_id(tmp_path: Path) -> None:
    inputs, _, reader = opened(tmp_path)
    resolved = inputs.materialization.resolved.entities[0]

    assert reader.run_id == RUN
    assert reader.manifest.run_id == RUN
    assert resolved.reference.resolution_run_id == RUN
    with pytest.raises(ForeignResolvedEntityReferenceError):
        reader.resolved_entity(
            ResolvedEntityReference(
                resolution_run_id=type(RUN)("another-run"),
                resolved_entity_id=resolved.resolved_entity_id,
            )
        )
    with pytest.raises(UnknownResolvedEntityError):
        reader.resolved_entity(
            ResolvedEntityReference(
                resolution_run_id=RUN, resolved_entity_id=type(resolved.resolved_entity_id)("nope")
            )
        )
    with pytest.raises(KeyError):
        reader.resolved_of(
            inputs.entities["a"].reference.__class__(
                semantic_map_id=SemanticMapId("x"),
                entity_id="y",  # type: ignore[arg-type]
            )
        )


def test_one_resolved_entity_is_read_without_loading_the_others(tmp_path: Path) -> None:
    inputs, directory, reader = opened(tmp_path)
    target = inputs.materialization.resolved.entities[-1]
    offsets = {
        row["resolved_entity_id"]: (row["offset"], row["length"])
        for row in reader.read_table("outputs/resolved-entity-index.jsonl")
    }

    assert reader.resolved_entity(target.reference) == target
    offset, length = offsets[str(target.resolved_entity_id)]
    data = (directory / "outputs/resolved-entities.jsonl").read_bytes()
    assert json.loads(data[offset : offset + length])["resolved_entity_id"] == str(
        target.resolved_entity_id
    )


# --- manifest, lineage and metrics --------------------------------------------------------------


def test_the_manifest_records_lineage_policies_counts_and_an_inventory(tmp_path: Path) -> None:
    _, directory, reader = opened(tmp_path)
    manifest = reader.manifest

    assert isinstance(manifest, EntityResolutionRunManifest)
    assert manifest.lineage == LINEAGE
    assert manifest.schema_version == "0.1.0"
    assert manifest.code_version == "test-version"
    roles = {item.role for item in manifest.policies}
    assert {"candidate_retrieval", "resolution", "materialization", "channel_geometry"} <= roles
    resolution = manifest.policy("resolution")
    assert resolution is not None and resolution.policy_id == "conservative-staged-resolution-v1"
    assert manifest.policy("split_detection") is None
    assert manifest.count("source_entities") == 6
    assert manifest.count("resolved_entities") == 5
    assert manifest.count("merged_entities") == 1
    assert manifest.count("transitivity_contradictions") == 1
    assert manifest.count("compared_pairs") == 6
    assert manifest.count("candidate_pairs") == 15
    paths = {entry.path for entry in manifest.file_inventory}
    assert "outputs/resolved-entities.jsonl" in paths and "metrics/counts.json" in paths
    assert not any(
        path.startswith("debug/") or path in {"manifest.json", "README.md"} for path in paths
    )
    assert all(isinstance(entry, FileEntry) for entry in manifest.file_inventory)
    assert reader.verify_integrity() == []
    assert (directory / "README.md").read_text().startswith("# Entity resolution run")


def test_per_channel_availability_and_group_sizes_are_recorded_apart(tmp_path: Path) -> None:
    _, _, reader = opened(tmp_path)

    counts = reader.read_record("metrics/counts.json")["counts"]
    distributions = reader.read_record("metrics/distributions.json")

    assert counts["channel_geometry_measured"] == 6
    assert counts["channel_appearance_not_evaluated"] == 6
    assert counts["channel_semantic_unavailable"] == 1
    assert counts["channel_semantic_measured"] == 0
    assert counts["channel_semantic_not_evaluated"] == 5
    sizes = {row["size"]: row["count"] for row in distributions["merge_group_size"]["histogram"]}
    assert sizes == {1: 4, 2: 1}
    payload = reader.read_record("metrics/payload.json")
    assert payload["total_bytes"] == sum(payload["files"].values())


def test_runtime_is_recorded_only_when_measured(tmp_path: Path) -> None:
    inputs = build_inputs()
    directory = tmp_path / "with-runtime"
    EntityResolutionRunWriter(
        output_dir=directory, run_id=RUN, lineage=LINEAGE, code_version="v"
    ).write(
        candidate_sets=inputs.candidate_sets,
        resolutions=inputs.resolutions,
        materialization=inputs.materialization,
        runtime={"seconds": 1.5, "peak_memory_mb": None},
    )
    _, _, without = opened(tmp_path)

    assert EntityResolutionRunReader(directory).read_record("metrics/runtime.json") == {
        "seconds": 1.5,
        "peak_memory_mb": None,
    }
    assert "metrics/runtime.json" not in {e.path for e in without.manifest.file_inventory}


def test_the_lineage_is_derived_from_the_semantic_mapping_manifest() -> None:
    manifest = SemanticMappingRunManifest(
        run_id="mapping-run-0007",  # type: ignore[arg-type]
        run_index=7,
        sequence_name="corridor-02",
        semantic_map_id=SemanticMapId("semantic-map-0001"),
        lineage=MappingRunLineage(
            sequence_artifact_id="sequence-0001",
            geometric_map_id=MapId("map-0001"),
            fusion_run_id="fusion-run-0001",  # type: ignore[arg-type]
            fusion_schema_version="0.1.0",
            fusion_artifact_digest="sha256:fusion",
            association_run_ids=("association-run-0001",),
            perception_run_ids=("perception-run-0001",),  # type: ignore[arg-type]
            point_representation_run_ids=("representation-run-0001",),  # type: ignore[arg-type]
        ),
        materialization_policy_id="one-support-one-entity-v1",
        identity_policy_id="support-derived-entity-id-v1",
        configuration_fingerprint="sha256:cfg",
        code_version="test",
        entity_count=6,
        rejected_count=0,
        warnings=(),
        debug_level="none",
        schema_version="0.1.0",
        created_at="2026-09-21T00:00:00+00:00",
        file_inventory=(
            FileEntry(path="outputs/entities.jsonl", size_bytes=10, content_hash="sha256:a"),
        ),
    )

    lineage = lineage_from_mapping_manifest(manifest)
    changed = dataclasses.replace(
        manifest,
        file_inventory=(
            FileEntry(path="outputs/entities.jsonl", size_bytes=10, content_hash="sha256:b"),
        ),
    )

    assert lineage.semantic_mapping_run_id == "mapping-run-0007"
    assert lineage.geometric_map_id == "map-0001"
    assert lineage.point_representation_run_ids == ("representation-run-0001",)
    assert lineage.perception_run_ids == ("perception-run-0001",)
    assert lineage.semantic_mapping_artifact_digest == mapping_artifact_digest(manifest)
    assert mapping_artifact_digest(manifest) != mapping_artifact_digest(changed)


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence_artifact_id": ""},
        {"semantic_map_ids": ()},
        {"semantic_map_ids": (SemanticMapId("b"), SemanticMapId("a"))},
        {"perception_run_ids": ("b", "a")},
    ],
)
def test_the_lineage_refuses_incomplete_or_unordered_selections(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(LINEAGE, **changes)


# --- consistency checks before anything is written ---------------------------------------------


def test_only_candidate_pairs_may_be_compared(tmp_path: Path) -> None:
    inputs = build_inputs()
    from resolution_entity_builders import entity_at

    stranger = entity_at("z", (9.0, 0.0, 0.0), support_number=9)
    extra = resolution_of(inputs.entities["a"], stranger, MATCH)

    with pytest.raises(RunArtifactError, match="not a candidate pair"):
        write_run(
            tmp_path / "artifact",
            dataclasses.replace(inputs, resolutions=(*inputs.resolutions, extra)),
        )
    assert not (tmp_path / "artifact").exists()


def test_a_pair_cannot_be_resolved_twice(tmp_path: Path) -> None:
    inputs = build_inputs()

    with pytest.raises(RunArtifactError, match="more than once"):
        write_run(
            tmp_path / "artifact",
            dataclasses.replace(inputs, resolutions=(*inputs.resolutions, inputs.resolutions[0])),
        )


def test_a_role_cannot_have_two_policies(tmp_path: Path) -> None:
    inputs = build_inputs()
    first = inputs.resolutions[0]
    other_policy = PolicyRef(
        policy_id="entity-geometry-comparison-v1", configuration_fingerprint="sha256:other"
    )
    rewritten = PairResolution(
        evidence=match_evidence(
            comparison_id=first.evidence.comparison_id,
            entity_a_ref=first.evidence.entity_a_ref,
            entity_b_ref=first.evidence.entity_b_ref,
            geometry=GeometryEvidence(
                policy=other_policy, measurement=geometry_evidence().measurement, findings=()
            ),
        ),
        decision=first.decision,
    )

    with pytest.raises(RunArtifactError, match="not unique"):
        write_run(
            tmp_path / "artifact",
            dataclasses.replace(inputs, resolutions=(rewritten, *inputs.resolutions[1:])),
        )


def test_resolved_entities_of_another_run_or_map_are_refused(tmp_path: Path) -> None:
    inputs = build_inputs()
    other_run = materialize_resolved_entities(
        inputs.entities.values(),
        [item.decision for item in inputs.resolutions],
        resolution_run_id=type(RUN)("another-run"),
    )

    with pytest.raises(RunArtifactError, match="belong to run"):
        write_run(tmp_path / "one", dataclasses.replace(inputs, materialization=other_run))
    foreign_lineage = dataclasses.replace(LINEAGE, geometric_map_id=MapId("map-0002"))
    with pytest.raises(RunArtifactError, match="lineage names"):
        EntityResolutionRunWriter(
            output_dir=tmp_path / "two", run_id=RUN, lineage=foreign_lineage, code_version="v"
        ).write(
            candidate_sets=inputs.candidate_sets,
            resolutions=inputs.resolutions,
            materialization=inputs.materialization,
        )
    unnamed_map = dataclasses.replace(LINEAGE, semantic_map_ids=(SemanticMapId("other"),))
    with pytest.raises(RunArtifactError, match="does not name"):
        EntityResolutionRunWriter(
            output_dir=tmp_path / "three", run_id=RUN, lineage=unnamed_map, code_version="v"
        ).write(
            candidate_sets=inputs.candidate_sets,
            resolutions=inputs.resolutions,
            materialization=inputs.materialization,
        )


def test_a_resolved_entity_cannot_cite_a_decision_that_was_not_written(tmp_path: Path) -> None:
    inputs = build_inputs()

    with pytest.raises(RunArtifactError, match="were not written"):
        write_run(
            tmp_path / "artifact",
            dataclasses.replace(inputs, resolutions=inputs.resolutions[:1]),
        )


# --- corruption is detected, nothing is trusted -------------------------------------------------


def test_a_tampered_file_and_a_missing_file_are_reported(tmp_path: Path) -> None:
    _, directory, reader = opened(tmp_path)

    with (directory / "outputs/resolution-decisions.jsonl").open("a") as handle:
        handle.write("{}\n")
    (directory / "outputs/candidate-sets.jsonl").unlink()

    problems = reader.verify_integrity()

    assert any("resolution-decisions.jsonl" in item for item in problems)
    assert any("missing file" in item and "candidate-sets.jsonl" in item for item in problems)


def test_an_incomplete_directory_and_an_unknown_schema_are_refused(tmp_path: Path) -> None:
    _, directory, _ = opened(tmp_path)

    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["schema_version"] = "9.9.9"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RunArtifactError, match="unsupported schema_version"):
        EntityResolutionRunReader(directory)
    (directory / "manifest.json").unlink()
    with pytest.raises(IncompleteRunArtifactError):
        EntityResolutionRunReader(directory)
    with pytest.raises(IncompleteRunArtifactError):
        EntityResolutionRunReader(tmp_path / "missing")


def test_a_malformed_manifest_or_row_is_refused_not_trusted(tmp_path: Path) -> None:
    _, directory, reader = opened(tmp_path)

    (directory / "outputs/resolution-decisions.jsonl").write_text("not json\n")
    with pytest.raises(RunArtifactError, match="malformed table"):
        reader.decisions()
    (directory / "outputs/resolution-decisions.jsonl").write_text('{"decision": "probably"}\n')
    with pytest.raises(RunArtifactError, match="malformed record"):
        reader.decisions()
    manifest = json.loads((directory / "manifest.json").read_text())
    del manifest["lineage"]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IncompleteRunArtifactError, match="malformed"):
        EntityResolutionRunReader(directory)


def test_a_truncated_resolved_entity_is_detected(tmp_path: Path) -> None:
    inputs, directory, reader = opened(tmp_path)
    target = inputs.materialization.resolved.entities[-1]
    path = directory / "outputs/resolved-entities.jsonl"
    path.write_bytes(path.read_bytes()[:50])

    with pytest.raises(RunArtifactError, match="truncated"):
        reader.resolved_entity(target.reference)


# --- debug is never contractual ------------------------------------------------------------------


def test_no_debug_by_default_and_debug_never_enters_the_inventory(tmp_path: Path) -> None:
    _, plain, _ = opened(tmp_path)
    assert not (plain / "debug").exists()

    inputs = build_inputs()
    full_dir = tmp_path / "full"
    write_run(full_dir, inputs, debug_level=ResolutionDebugLevel.FULL)
    reader = EntityResolutionRunReader(full_dir)
    first = inputs.resolutions[0].decision.decision_id

    trace = full_dir / "debug" / "decisions" / str(first)
    assert {"decision.json", "entity-a.json", "entity-b.json", "candidate-retrieval.json"} <= {
        p.name for p in trace.iterdir()
    }
    assert (trace / "geometry-evidence.json").is_file()
    assert not (trace / "appearance-evidence.json").exists()  # só os canais aplicáveis
    assert len(list((full_dir / "debug" / "decisions").iterdir())) == len(inputs.resolutions)
    assert not any(e.path.startswith("debug/") for e in reader.manifest.file_inventory)
    assert reader.manifest.debug_level == "full"


def test_the_standard_level_traces_a_sample_and_removing_debug_breaks_nothing(
    tmp_path: Path,
) -> None:
    inputs = build_inputs()
    directory = tmp_path / "standard"
    write_run(directory, inputs, debug_level=ResolutionDebugLevel.STANDARD)
    reader = EntityResolutionRunReader(directory)

    shutil.rmtree(directory / "debug")

    assert reader.verify_integrity() == []
    assert reader.resolved_entities() == inputs.materialization.resolved


def test_downstream_consumers_can_only_read_contractual_files(tmp_path: Path) -> None:
    inputs = build_inputs()
    directory = tmp_path / "artifact"
    write_run(directory, inputs, debug_level=ResolutionDebugLevel.FULL)
    reader = EntityResolutionRunReader(directory)

    with pytest.raises(RunArtifactError, match="not a contractual table"):
        reader.read_table("debug/decisions/x/decision.json")
    with pytest.raises(RunArtifactError, match="not a contractual JSON record"):
        reader.read_record("debug/decisions/x/decision.json")
    with pytest.raises(RunArtifactError, match="not a contractual"):
        reader.read_table("manifest.json")
    shutil.rmtree(directory / "debug")
    assert reader.resolved_entities() == inputs.materialization.resolved


# --- readable without any runtime ---------------------------------------------------------------


def test_a_run_reopens_without_numpy_or_any_model_runtime(tmp_path: Path) -> None:
    inputs, directory, _ = opened(tmp_path)
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from contextmap.entity_resolution import EntityResolutionRunReader\n"
        f"reader = EntityResolutionRunReader(Path({str(directory)!r}))\n"
        "resolved = reader.resolved_entities()\n"
        "assert reader.verify_integrity() == []\n"
        "names = ('numpy', 'torch', 'transformers', 'rosbags', 'PIL')\n"
        "heavy = [name for name in names if name in sys.modules]\n"
        "print(len(resolved.entities), heavy)\n"
    )

    output = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert output == f"{len(inputs.materialization.resolved.entities)} []"


def test_comparison_ids_of_the_written_evidence_match_its_decisions(tmp_path: Path) -> None:
    _, _, reader = opened(tmp_path)

    evidence = {item.comparison_id: item for item in reader.match_evidence()}
    for decision in reader.decisions():
        assert decision.evidence_ref in evidence
        assert evidence[decision.evidence_ref].comparison_id == comparison_id_for(
            decision.entity_a_ref, decision.entity_b_ref
        )
    assert isinstance(next(iter(evidence)), str) and ComparisonId is not None
    assert channel_policy() is not None and ResolutionRunLineage is not None


def test_an_unknown_count_and_a_missing_materialization_policy_are_explicit(tmp_path: Path) -> None:
    _, directory, reader = opened(tmp_path)

    with pytest.raises(KeyError):
        reader.manifest.count("no-such-count")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["policies"] = [
        item for item in manifest["policies"] if item["role"] != "materialization"
    ]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IncompleteRunArtifactError, match="materialization policy"):
        EntityResolutionRunReader(directory).materialization()


def test_a_resolved_entity_record_that_is_not_json_is_refused(tmp_path: Path) -> None:
    inputs, directory, reader = opened(tmp_path)
    target = inputs.materialization.resolved.entities[0]
    offset, length = next(
        (row["offset"], row["length"])
        for row in reader.read_table("outputs/resolved-entity-index.jsonl")
        if row["resolved_entity_id"] == str(target.resolved_entity_id)
    )
    path = directory / "outputs/resolved-entities.jsonl"
    data = bytearray(path.read_bytes())
    data[offset : offset + length] = b"x" * length
    path.write_bytes(bytes(data))

    with pytest.raises(RunArtifactError, match="malformed record"):
        reader.resolved_entity(target.reference)


def test_an_empty_run_is_valid_and_reads_back_empty(tmp_path: Path) -> None:
    empty = materialize_resolved_entities([], [], resolution_run_id=RUN)
    directory = tmp_path / "empty"
    EntityResolutionRunWriter(
        output_dir=directory, run_id=RUN, lineage=LINEAGE, code_version="v"
    ).write(candidate_sets=(), resolutions=(), materialization=empty)

    reader = EntityResolutionRunReader(directory)

    assert reader.resolved_entities().entities == ()
    assert reader.manifest.count("resolved_entities") == 0
    assert reader.read_record("metrics/distributions.json")["candidates_per_entity"]["count"] == 0
    assert reader.verify_integrity() == []
