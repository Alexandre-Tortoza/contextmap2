"""The SpatialRelationsRunArtifact: immutable, self-describing and reusable as it is."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from relation_builders import entity_ref
from relation_run_fixture import (
    CANDIDATES,
    CONTACT,
    CONVENTIONS,
    GEOMETRIC,
    LINEAGE,
    POLICIES,
    RUN,
    Run,
    build_run,
    write_run,
)

from contextmap.entity_resolution import EntityResolutionRunId
from contextmap.geometric_mapping import MapId
from contextmap.spatial_relations import (  # noqa: F401
    CONSERVATIVE_DECISION_POLICY_ID,
    IncompleteRelationsRunArtifactError,
    RelationEvidenceStatus,
    RelationsRunArtifactError,
    RelationsRunDebugLevel,
    RelationsRunLineage,
    RelationsRunPolicies,
    RelationState,
    SpatialRelationsRunManifest,
    SpatialRelationsRunReader,
    decision_policy_fingerprint,
)


@pytest.fixture(scope="module")
def run() -> Run:
    return build_run()


def _write(run: Run, directory: Path, **options: object) -> SpatialRelationsRunManifest:
    return write_run(run, directory, **options)  # type: ignore[arg-type]


# --- write and reopen ---


def test_a_run_reopens_with_everything_that_was_written(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    assert tuple(reader.iter_relations()) == run.decisions.relations
    assert tuple(reader.iter_decisions()) == run.decisions.decisions
    assert tuple(reader.iter_evidence()) == tuple(
        sorted(run.evidence, key=lambda item: item.evidence_id)
    )
    assert reader.candidate_set() == run.candidates


def test_the_run_is_written_where_the_caller_says_and_nowhere_else(
    run: Run, tmp_path: Path
) -> None:
    _write(run, tmp_path / "chosen" / "spatial_relations")
    assert sorted(path.name for path in (tmp_path / "chosen").iterdir()) == ["spatial_relations"]
    assert not list(tmp_path.rglob("runs.json"))
    assert not list(tmp_path.rglob("run-0*"))


def test_a_finished_run_is_never_overwritten(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    with pytest.raises(RelationsRunArtifactError, match="exists"):
        _write(run, tmp_path / "relations")


def test_the_manifest_records_lineage_policies_and_counts(run: Run, tmp_path: Path) -> None:
    manifest = _write(run, tmp_path / "relations", warnings=("one warning",))
    assert manifest.run_id == "relations-run-0001"
    assert manifest.lineage == LINEAGE
    assert manifest.taxonomy_version == run.decisions.relations[0].provenance.taxonomy_version
    assert manifest.warnings == ("one warning",)
    assert manifest.code_version == "test"
    assert manifest.schema_version
    raw = json.loads((tmp_path / "relations" / "manifest.json").read_text())
    policies = raw["policies"]
    assert policies["decision"] == {
        "policy_id": CONSERVATIVE_DECISION_POLICY_ID,
        "fingerprint": decision_policy_fingerprint(),
    }
    assert policies["candidate"]["fingerprint"] == CANDIDATES.fingerprint()
    assert policies["geometric"]["fingerprint"] == GEOMETRIC.fingerprint()
    assert policies["contact"]["fingerprint"] == CONTACT.fingerprint()
    assert policies["frame_conventions"]["up_axis"] == "+z"
    assert policies["frame_conventions"]["fingerprint"] == CONVENTIONS.fingerprint()
    assert policies["geometric"]["parameters"]["next_to_max_gap_m"] == 0.5
    assert manifest.counts["relations"] == len(run.decisions.relations)


def test_the_manifest_holds_no_absolute_path_and_no_secret(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    text = (tmp_path / "relations" / "manifest.json").read_text()
    assert str(tmp_path) not in text
    for marker in ("hf_", "AIza", "Bearer", "token", "api_key"):
        assert marker not in text


def test_the_inventory_lists_the_contractual_files_and_leaves_debug_out(
    run: Run, tmp_path: Path
) -> None:
    manifest = _write(run, tmp_path / "relations", debug=RelationsRunDebugLevel.FULL)
    paths = {entry.path for entry in manifest.file_inventory}
    assert {
        "outputs/relations.jsonl",
        "outputs/relation-evidence.jsonl",
        "outputs/relation-candidates.jsonl",
        "outputs/relation-decisions.jsonl",
        "outputs/entity-relation-index.jsonl",
        "metrics/counts.json",
    } <= paths
    assert not any(path.startswith("debug/") for path in paths)
    assert "manifest.json" not in paths


# --- states, evidence and the entity index ---


def test_supported_rejected_and_unresolved_relations_stay_distinguishable(
    run: Run, tmp_path: Path
) -> None:
    manifest = _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    states = {item.state for item in reader.iter_relations()}
    assert states == {
        RelationState.SUPPORTED,
        RelationState.REJECTED,
        RelationState.UNRESOLVED,
    }
    by_state = manifest.counts["by_state"]
    assert sum(by_state.values()) == manifest.counts["relations"]
    assert by_state["supported"] == sum(
        1 for item in run.decisions.relations if item.state is RelationState.SUPPORTED
    )


def test_every_supported_relation_traces_to_exact_entities_and_evidence(
    run: Run, tmp_path: Path
) -> None:
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    supported = [item for item in reader.iter_relations() if item.state is RelationState.SUPPORTED]
    assert supported
    for relation in supported:
        assert relation.subject_entity_ref.resolution_run_id == RUN
        evidence = reader.evidence_of(relation)
        assert {item.evidence_id for item in evidence} == set(relation.relation_evidence_refs)
        decision = reader.decision(relation.relation_id)
        assert set(decision.deciding_evidence_refs) <= set(relation.relation_evidence_refs)
        assert any(item.status is RelationEvidenceStatus.SUPPORTS for item in evidence)


def test_a_relation_and_its_evidence_are_read_by_identity(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    relation = run.decisions.relations[0]
    assert reader.relation(relation.relation_id) == relation
    with pytest.raises(KeyError):
        reader.relation("relation--nope")  # type: ignore[arg-type]
    record = run.evidence[0]
    assert reader.evidence(record.evidence_id) == record


def test_the_entity_index_lists_the_relations_of_each_resolved_entity(
    run: Run, tmp_path: Path
) -> None:
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    expected = {
        item.relation_id
        for item in run.decisions.relations
        if entity_ref(2) in (item.subject_entity_ref, item.object_entity_ref)
    }
    assert expected
    found = reader.relations_of(entity_ref(2))
    assert {item.relation_id for item in found} == expected
    subject_only = reader.relations_of(entity_ref(2), as_subject=True, as_object=False)
    assert all(item.subject_entity_ref == entity_ref(2) for item in subject_only)
    assert reader.relations_of(entity_ref(9)) == ()


def test_the_artifact_does_not_copy_entities_or_geometry(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    for path in (tmp_path / "relations" / "outputs").iterdir():
        text = path.read_text()
        assert "coordinates_m" not in text
        assert "centroid" not in text


# --- integrity, immutability and portability ---


def test_an_intact_run_verifies_and_corruption_is_detected(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    assert reader.verify_integrity() == []
    target = tmp_path / "relations" / "outputs" / "relations.jsonl"
    target.write_text(target.read_text().replace("supported", "rejected", 1))
    assert any("relations.jsonl" in problem for problem in reader.verify_integrity())
    (tmp_path / "relations" / "outputs" / "relation-decisions.jsonl").unlink()
    assert any("relation-decisions.jsonl" in item for item in reader.verify_integrity())


def test_a_relocated_run_still_opens_and_verifies(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "here")
    shutil.move(tmp_path / "here", tmp_path / "there")
    reader = SpatialRelationsRunReader(tmp_path / "there")
    assert reader.verify_integrity() == []
    assert tuple(reader.iter_relations()) == run.decisions.relations


def test_two_writes_of_the_same_result_have_identical_contractual_files(
    run: Run, tmp_path: Path
) -> None:
    first = _write(run, tmp_path / "a")
    second = _write(run, tmp_path / "b")
    assert first.file_inventory == second.file_inventory


def test_removing_debug_never_invalidates_the_run(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations", debug=RelationsRunDebugLevel.FULL)
    shutil.rmtree(tmp_path / "relations" / "debug")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    assert reader.verify_integrity() == []
    assert tuple(reader.iter_relations()) == run.decisions.relations


# --- debug ---


def test_debug_holds_the_measurements_and_traces_only_on_request(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "none")
    assert not (tmp_path / "none" / "debug").exists()
    _write(run, tmp_path / "full", debug=RelationsRunDebugLevel.FULL)
    files = sorted((tmp_path / "full" / "debug" / "relations").iterdir())
    assert len(files) == len(run.decisions.relations)
    record = json.loads(files[0].read_text())
    assert {"relation", "decision", "evidence"} <= set(record)


def test_consumers_can_only_read_contractual_files(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations", debug=RelationsRunDebugLevel.FULL)
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    with pytest.raises(RelationsRunArtifactError, match="contractual"):
        reader.read_table("debug/relations/x.json")
    with pytest.raises(RelationsRunArtifactError, match="contractual"):
        reader.read_record("outputs/relations.jsonl")
    assert reader.read_record("metrics/counts.json")["relations"] == len(run.decisions.relations)


# --- validation at the writer ---


def test_relations_of_another_resolution_run_are_refused(run: Run, tmp_path: Path) -> None:
    other = dataclasses.replace(
        LINEAGE, entity_resolution_run_id=EntityResolutionRunId("resolution-run-0002")
    )
    with pytest.raises(RelationsRunArtifactError, match="resolution"):
        _write(run, tmp_path / "relations", lineage=other)
    assert not (tmp_path / "relations").exists()


def test_a_run_over_another_geometric_map_is_refused(run: Run, tmp_path: Path) -> None:
    other = dataclasses.replace(LINEAGE, geometric_map_id=MapId("map-0002"))
    with pytest.raises(RelationsRunArtifactError, match="geometric map"):
        _write(run, tmp_path / "relations", lineage=other)


def test_evidence_without_its_declared_policy_is_refused(run: Run, tmp_path: Path) -> None:
    without_contact = dataclasses.replace(POLICIES, contact=None)
    with pytest.raises(RelationsRunArtifactError, match="contact"):
        _write(run, tmp_path / "relations", policies=without_contact)
    changed = dataclasses.replace(
        POLICIES, geometric=dataclasses.replace(GEOMETRIC, next_to_max_gap_m=0.9)
    )
    with pytest.raises(RelationsRunArtifactError, match="fingerprint"):
        _write(run, tmp_path / "relations", policies=changed)


def test_a_relation_that_cites_missing_evidence_is_refused(run: Run, tmp_path: Path) -> None:
    incomplete = dataclasses.replace(run, evidence=run.evidence[1:])
    with pytest.raises(RelationsRunArtifactError, match="evidence"):
        _write(incomplete, tmp_path / "relations")


def test_evidence_that_is_not_about_a_candidate_is_refused(run: Run, tmp_path: Path) -> None:
    fewer = dataclasses.replace(
        run.candidates, candidates=run.candidates.candidates[1:], exclusions=()
    )
    with pytest.raises(RelationsRunArtifactError, match="candidate"):
        _write(dataclasses.replace(run, candidates=fewer), tmp_path / "relations")


# --- reading errors ---


def test_a_directory_that_is_not_a_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(IncompleteRelationsRunArtifactError, match="manifest"):
        SpatialRelationsRunReader(tmp_path)


def test_an_unknown_schema_version_is_refused(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    manifest = tmp_path / "relations" / "manifest.json"
    raw = json.loads(manifest.read_text())
    raw["schema_version"] = "99.0.0"
    manifest.write_text(json.dumps(raw))
    with pytest.raises(RelationsRunArtifactError, match="schema_version"):
        SpatialRelationsRunReader(tmp_path / "relations")


def test_references_are_checked_against_the_resolved_entities_of_the_run(
    run: Run, tmp_path: Path
) -> None:
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    assert reader.validate_references(set(run.entities)) == ()
    assert reader.validate_references(set(run.entities) - {entity_ref(3)}) == ()
    assert reader.validate_references(set(run.entities) - {entity_ref(2)}) == (entity_ref(2),)
