"""The SpatialRelationsRunArtifact: immutable, self-describing and reusable as it is."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from relation_storeroom_fixture import exclusion_ceiling

from contextmap.entity_resolution import EntityResolutionRunId
from contextmap.geometric_mapping import MapId
from contextmap.spatial_relations import (  # noqa: F401
    CONSERVATIVE_DECISION_POLICY_ID,
    IncompleteRelationsRunArtifactError,
    Relation,
    RelationEvidenceStatus,
    RelationsRunArtifactError,
    RelationsRunDebugLevel,
    RelationsRunLineage,
    RelationsRunPolicies,
    RelationState,
    SpatialRelationsRunManifest,
    SpatialRelationsRunReader,
    decision_policy_fingerprint,
    encode_candidate_set,
    encode_relation,
    encode_relation_decision,
    encode_relation_evidence,
    generate_relation_candidates,
    run_artifact,
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
    assert policies["geometry_summary"]["fingerprint"] == POLICIES.geometry_summary.fingerprint()
    assert manifest.counts["relations"] == len(run.decisions.relations)


def test_a_reopened_run_exposes_the_geometry_summary_policy_that_produced_it(
    run: Run, tmp_path: Path
) -> None:
    """P2 #4 of the PR #540 review: ``geometry_summary`` is part of the run's own provenance.

    ``resolved_entity_geometries`` (inside ``SpatialRelationsExecutor``) uses this policy
    before candidate generation, and it measurably affects the geometric/contact predicate
    outputs -- so the artifact must be self-describing about it, exactly like
    ``frame_conventions``, ``candidate``, ``geometric`` and ``contact`` already are.
    """
    _write(run, tmp_path / "relations")

    reopened = SpatialRelationsRunReader(tmp_path / "relations").manifest.policies

    geometry_summary = reopened["geometry_summary"]
    assert geometry_summary["policy_id"] == "entity-geometry-summary-v1"
    assert geometry_summary["fingerprint"] == POLICIES.geometry_summary.fingerprint()
    assert geometry_summary["parameters"] == dataclasses.asdict(POLICIES.geometry_summary)


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


def contractual_digest(directory: Path) -> str:
    """Digest of every contractual file and of the manifest apart from its creation time."""
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest.pop("created_at")
    files = {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.relative_to(directory).parts[0] in ("outputs", "metrics")
    }
    record = json.dumps({"manifest": manifest, "files": files}, sort_keys=True)
    return hashlib.sha256(record.encode()).hexdigest()


def answers_digest(reader: SpatialRelationsRunReader, run: Run) -> str:
    """Digest of every answer the reader gives: lookups, the entity index and the iterations."""
    relation_ids = sorted(item.relation_id for item in run.decisions.relations)
    evidence_ids = sorted(item.evidence_id for item in run.evidence)
    entities = [*sorted(run.entities, key=lambda item: item.resolved_entity_id), entity_ref(9)]
    record = {
        "relation": [encode_relation(reader.relation(item)) for item in relation_ids],
        "decision": [encode_relation_decision(reader.decision(item)) for item in relation_ids],
        "evidence": [encode_relation_evidence(reader.evidence(item)) for item in evidence_ids],
        "evidence_of": [
            [encode_relation_evidence(item) for item in reader.evidence_of(reader.relation(key))]
            for key in relation_ids
        ],
        "relations_of": [
            [
                encode_relation(item)
                for item in reader.relations_of(entity, as_subject=subject, as_object=obj)
            ]
            for entity in entities
            for subject, obj in ((True, True), (True, False), (False, True), (False, False))
        ],
        "iter_relations": [encode_relation(item) for item in reader.iter_relations()],
        "iter_evidence": [encode_relation_evidence(item) for item in reader.iter_evidence()],
        "iter_decisions": [encode_relation_decision(item) for item in reader.iter_decisions()],
        "candidate_set": encode_candidate_set(reader.candidate_set()),
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


# #601 (SR-06): gravados antes de o leitor passar a buscar por deslocamento. O schema 0.2.0
# (SR-01) só mudou, neste run, a schema_version do manifest e a chave unlisted_exclusions (zero) de
# metrics/counts.json, com o inventário que as descreve; as tabelas estão fixadas abaixo.
RECORDED_RUN_BYTES = "d12a86a75c50385f529467ca9b91aa7bd6b9006c090fd429d510ee956c68aca3"
RECORDED_READER_ANSWERS = "a2cf3c4b17dcf49e5110ec5c59bf66bae87f18124c37904aeca23da807e8971b"


def test_the_persisted_run_matches_the_recorded_bytes(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    assert contractual_digest(tmp_path / "relations") == RECORDED_RUN_BYTES


# #601 (SR-01): gravados antes do teto de exclusões. Nenhum grupo do fixture passa do teto, então
# nenhuma tabela de outputs/ pode mudar; só o manifest e as métricas mudam com o schema 0.2.0.
RECORDED_OUTPUT_TABLES = {
    "outputs/entity-relation-index.jsonl": (
        "58dfef4e2e6b528933a0e9d0a848075837b79b2ff727479d4055b44a56d857dd"
    ),
    "outputs/relation-candidates.jsonl": (
        "37ff6b291c4fee6af3cde90e991d789b5ac0895e88c55155e176d93921e1c1ab"
    ),
    "outputs/relation-decisions.jsonl": (
        "f400d13d4582c4bdaebe9df987413507d254fde4db5780df832f92a98fdb4ed5"
    ),
    "outputs/relation-evidence.jsonl": (
        "ce913c16ccbee34e16ae3fd354c35df5a6b340e4302bd0f666bddc7066edefb8"
    ),
    "outputs/relations.jsonl": "72e203fa784b0d2adfd29004dfc2608d7f0634f2dbaab5704214a75d9c79ef34",
}


def test_the_tables_below_the_exclusion_ceiling_match_the_recorded_bytes(
    run: Run, tmp_path: Path
) -> None:
    _write(run, tmp_path / "relations")
    tables = {
        path: hashlib.sha256((tmp_path / "relations" / path).read_bytes()).hexdigest()
        for path in RECORDED_OUTPUT_TABLES
    }
    assert tables == RECORDED_OUTPUT_TABLES


def test_the_reader_gives_the_recorded_answers(run: Run, tmp_path: Path) -> None:
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    assert answers_digest(reader, run) == RECORDED_READER_ANSWERS


def _count_relation_decodes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the identity of every relation record the reader decodes."""
    decoded: list[str] = []
    decode = run_artifact.decode_relation

    def counting(record: Mapping[str, Any]) -> Relation:
        decoded.append(record["relation_id"])
        return decode(record)

    monkeypatch.setattr(run_artifact, "decode_relation", counting)
    return decoded


def test_a_relation_is_read_without_decoding_the_others(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SR-06: buscar uma relação decodificava a tabela inteira.
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    first, last = run.decisions.relations[0], run.decisions.relations[-1]
    decoded = _count_relation_decodes(monkeypatch)

    assert reader.relation(last.relation_id) == last
    assert reader.relation(first.relation_id) == first

    assert decoded == [last.relation_id, first.relation_id]


def test_the_relations_of_an_entity_are_read_without_decoding_the_others(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SR-06: relations_of decodificava todas as relações para filtrar as da entidade.
    _write(run, tmp_path / "relations")
    reader = SpatialRelationsRunReader(tmp_path / "relations")
    expected = tuple(
        item
        for item in run.decisions.relations
        if entity_ref(4) in (item.subject_entity_ref, item.object_entity_ref)
    )
    decoded = _count_relation_decodes(monkeypatch)

    assert reader.relations_of(entity_ref(4)) == expected
    assert reader.relations_of(entity_ref(9)) == ()

    assert 0 < len(decoded) == len(expected) < len(run.decisions.relations)


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
        POLICIES, geometric=dataclasses.replace(GEOMETRIC, next_to_max_gap_m=0.55)
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


def test_a_run_under_incoherent_policies_is_refused_before_anything_is_published(
    run: Run, tmp_path: Path
) -> None:
    wider = dataclasses.replace(GEOMETRIC, next_to_max_gap_m=0.9)
    with pytest.raises(ValueError, match="next_to_max_gap_m"):
        _write(
            run,
            tmp_path / "relations",
            policies=dataclasses.replace(POLICIES, geometric=wider),
        )
    assert not (tmp_path / "relations").exists()


# --- coherence of the candidate reach with the evaluators' tolerances ---


def test_a_next_to_gap_beyond_the_proximity_reach_is_refused() -> None:
    wider = dataclasses.replace(GEOMETRIC, next_to_max_gap_m=0.9)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, geometric=wider)
    message = str(error.value)
    assert "next_to_max_gap_m=0.9 exceeds proximity_radius_m=0.6" in message
    assert "NEXT_TO relations would be excluded" in message
    assert "before evaluation" in message


def test_a_contact_reach_beyond_the_proximity_reach_is_refused() -> None:
    # Cada parâmetro cabe sozinho no alcance; é a soma, o raio de busca real do canal de contato,
    # que passa dele.
    farther = dataclasses.replace(CONTACT, contact_distance_m=0.5, contact_tolerance_m=0.2)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, contact=farther)
    message = str(error.value)
    assert "contact_distance_m + contact_tolerance_m = 0.7 (0.5 + 0.2)" in message
    assert "exceeds proximity_radius_m=0.6" in message
    assert "TOUCHING, ON_TOP_OF and LEANING_AGAINST relations would be excluded" in message


def test_a_containment_slack_on_both_faces_beyond_the_proximity_reach_is_refused() -> None:
    # containment_slack_m <= proximity_radius_m < 2 * containment_slack_m: o sujeito pode
    # ultrapassar as duas faces de um eixo, então o avaliador aceita um excesso de extensão
    # que a pré-condição de INSIDE já teria descartado.
    looser = dataclasses.replace(GEOMETRIC, containment_slack_m=0.4)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, geometric=looser)
    message = str(error.value)
    assert "2 * containment_slack_m = 0.8" in message
    assert "containment_slack_m=0.4" in message
    assert "exceeds proximity_radius_m=0.6" in message
    assert "INSIDE relations would be excluded" in message


def test_every_incoherence_is_reported_at_once() -> None:
    short = dataclasses.replace(CANDIDATES, proximity_radius_m=0.04)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, candidate=short)
    message = str(error.value)
    for name in ("next_to_max_gap_m=", "2 * containment_slack_m", "contact_distance_m +"):
        assert name in message


def test_policies_whose_reach_covers_every_tolerance_are_accepted_silently() -> None:
    # Os limites são inclusivos: um alcance exatamente igual à tolerância não perde nada.
    geometric = dataclasses.replace(GEOMETRIC, next_to_max_gap_m=0.6, containment_slack_m=0.3)
    contact = dataclasses.replace(CONTACT, contact_distance_m=0.5, contact_tolerance_m=0.1)
    short = dataclasses.replace(CANDIDATES, proximity_radius_m=0.04)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        at_the_limit = dataclasses.replace(POLICIES, geometric=geometric, contact=contact)
        # Um canal ausente não tem tolerância a cobrir, por menor que seja o alcance.
        without_channels = dataclasses.replace(
            POLICIES, candidate=short, geometric=None, contact=None
        )
    assert at_the_limit.geometric == geometric
    assert at_the_limit.contact == contact
    assert without_channels.candidate == short


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


# --- schema 0.2.0: the exclusion ceiling (#601) ---


def test_a_run_is_written_under_schema_0_2_0(run: Run, tmp_path: Path) -> None:
    # SR-01: o registro de exclusões passou a ter teto, e o schema do artifact muda com ele.
    assert _write(run, tmp_path / "relations").schema_version == "0.2.0"


def test_a_run_of_schema_0_1_0_is_still_read(run: Run, tmp_path: Path) -> None:
    # Os runs 0.1.0 (entre eles a demo congelada de examples/v0.1.0) listam toda exclusão, o que
    # o leitor atual lê sem distinção: um conjunto sem grupo resumido.
    _write(run, tmp_path / "relations")
    manifest = tmp_path / "relations" / "manifest.json"
    raw = json.loads(manifest.read_text())
    raw["schema_version"] = "0.1.0"
    manifest.write_text(json.dumps(raw))

    reader = SpatialRelationsRunReader(tmp_path / "relations")

    assert reader.manifest.schema_version == "0.1.0"
    assert reader.candidate_set() == run.candidates
    assert tuple(reader.iter_relations()) == run.decisions.relations


def test_the_counts_record_every_exclusion_and_how_many_are_not_listed(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exclusion_ceiling(monkeypatch, 1)
    capped = generate_relation_candidates(run.entities, policy=CANDIDATES, conventions=CONVENTIONS)
    listed = len(capped.exclusions) + sum(len(item.nearest) for item in capped.exclusion_summaries)
    assert capped.exclusion_summaries

    _write(dataclasses.replace(run, candidates=capped), tmp_path / "relations")

    reader = SpatialRelationsRunReader(tmp_path / "relations")
    counts = reader.read_record("metrics/counts.json")
    assert counts["exclusions"] == len(run.candidates.exclusions)
    assert counts["unlisted_exclusions"] == len(run.candidates.exclusions) - listed > 0
    assert reader.candidate_set() == capped
