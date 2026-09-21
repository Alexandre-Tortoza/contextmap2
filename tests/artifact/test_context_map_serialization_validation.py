"""The ContextMapArtifact integrity validator (issue #158).

The validator never raises for a damaged artifact and never repairs it: it returns a
deterministic, machine-readable report with errors, warnings, the inventory it checked, the
identities it found and the version of the validator. Structural validation is fast and says so;
only full verification (hashes, records, references, rebuilt indexes, upstream files) may call an
artifact verified.
"""

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from context_map_builders import (
    ENTITY_RESOLUTION_ARTIFACT_ID,
    FUSION_ARTIFACT_ID,
    SPATIAL_RELATIONS_ARTIFACT_ID,
    entity_capabilities,
    metadata,
)
from context_map_builders import context_map as schema_context_map
from context_map_serialization_builders import (
    MAP_ID,
    World,
    make_context_map,
    make_world,
    pinned,
    tree_snapshot,
    write_artifact,
)

from contextmap.artifact import (
    VALIDATOR_VERSION,
    CheckOutcome,
    FileStatus,
    Severity,
    ValidationLevel,
    ValidationReport,
    ValidationStatus,
    validate_context_map_artifact,
)
from contextmap.artifact.serialization.manifest import (
    create_manifest,
    decode_manifest,
    encode_manifest,
)
from contextmap.shared import file_entry

FULL = ValidationLevel.FULL
STRUCTURAL = ValidationLevel.STRUCTURAL


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


@pytest.fixture
def artifact(world: World) -> Path:
    return write_artifact(world)[0]


def _codes(report: ValidationReport, severity: Severity | None = None) -> set[str]:
    return {
        finding.code
        for finding in report.findings
        if severity is None or finding.severity is severity
    }


def _errors(report: ValidationReport) -> set[str]:
    return _codes(report, Severity.ERROR)


def _outcomes(report: ValidationReport) -> dict[str, CheckOutcome]:
    return {check.name: check.outcome for check in report.checks}


def _reseal(artifact: Path) -> None:
    """Recompute inventory and identity from the disk so only the intended damage remains."""
    manifest = decode_manifest(json.loads((artifact / "manifest.json").read_text()))
    resealed = create_manifest(
        context_map_id=manifest.context_map_id,
        schema_version=manifest.schema_version,
        written_at=manifest.written_at,
        code_version=manifest.code_version,
        configuration_fingerprint=manifest.configuration_fingerprint,
        entity_count=manifest.entity_count,
        relation_count=manifest.relation_count,
        payloads=manifest.payloads,
        dependencies=manifest.dependencies,
        file_inventory=[
            file_entry(entry.path, (artifact / entry.path).read_bytes())
            for entry in manifest.file_inventory
        ],
    )
    (artifact / "manifest.json").write_text(json.dumps(encode_manifest(resealed)))


def _replace_in(artifact: Path, relative_path: str, old: bytes, new: bytes) -> None:
    path = artifact / relative_path
    data = path.read_bytes()
    assert old in data and len(old) == len(new)
    path.write_bytes(data.replace(old, new))


def _edit_json(artifact: Path, relative_path: str, edit: Callable[[Any], None]) -> None:
    path = artifact / relative_path
    record = json.loads(path.read_text(encoding="utf-8"))
    edit(record)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def test_an_intact_artifact_is_fully_verified_with_nothing_to_report(
    world: World, artifact: Path
) -> None:
    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.VERIFIED
    assert report.level is FULL
    assert _errors(report) == set()
    assert report.validator_version == VALIDATOR_VERSION
    assert report.artifact_type == "context_map"
    assert report.format_version == "0.1.0"
    assert report.schema_version == "0.1.0"
    assert report.context_map_id == str(make_context_map(world).context_map_id)
    assert report.content_identity is not None and report.content_identity.startswith("sha256:")
    assert {item.status for item in report.files} == {FileStatus.VERIFIED}
    assert [item.path for item in report.files] == sorted(item.path for item in report.files)
    assert all(item.outcome is not CheckOutcome.FAILED for item in report.checks)


def test_the_report_lists_every_check_and_a_full_run_skips_none(artifact: Path) -> None:
    outcomes = _outcomes(validate_context_map_artifact(artifact))

    assert list(outcomes)[:2] == ["manifest", "inventory"]
    assert set(outcomes.values()) == {CheckOutcome.PASSED}
    for name in (
        "file_hashes",
        "reference_integrity",
        "index_rebuild",
        "schema_invariants",
        "dependency_integrity",
    ):
        assert outcomes[name] is CheckOutcome.PASSED


def test_structural_validation_is_fast_and_never_claims_the_artifact_is_verified(
    artifact: Path,
) -> None:
    report = validate_context_map_artifact(artifact, level=STRUCTURAL)

    assert report.status is ValidationStatus.STRUCTURALLY_VALID
    assert report.status is not ValidationStatus.VERIFIED
    assert _errors(report) == set()
    assert {item.status for item in report.files} == {FileStatus.SIZE_OK}
    outcomes = _outcomes(report)
    for name in (
        "file_hashes",
        "reference_integrity",
        "index_rebuild",
        "schema_invariants",
        "dependency_integrity",
    ):
        assert outcomes[name] is CheckOutcome.SKIPPED
    skipped = {check.name: check.detail for check in report.checks}
    assert "structural" in (skipped["file_hashes"] or "")


def test_a_change_that_keeps_the_size_is_only_seen_by_full_verification(artifact: Path) -> None:
    _replace_in(artifact, "map-metadata.json", b"corridor", b"corridoq")

    structural = validate_context_map_artifact(artifact, level=STRUCTURAL)
    full = validate_context_map_artifact(artifact, level=FULL)

    assert structural.status is ValidationStatus.STRUCTURALLY_VALID
    assert full.status is ValidationStatus.INVALID
    assert "file.hash_mismatch" in _errors(full)
    finding = next(item for item in full.findings if item.code == "file.hash_mismatch")
    assert finding.subject == "map-metadata.json"
    assert {item.path: item.status for item in full.files}["map-metadata.json"] is (
        FileStatus.HASH_MISMATCH
    )


def test_a_missing_payload_is_an_error_at_both_levels(artifact: Path) -> None:
    (artifact / "relations/relations.jsonl").unlink()

    for level in (STRUCTURAL, FULL):
        report = validate_context_map_artifact(artifact, level=level)
        assert report.status is ValidationStatus.INVALID
        assert "file.missing" in _errors(report)
        assert {item.path: item.status for item in report.files}["relations/relations.jsonl"] is (
            FileStatus.MISSING
        )


def test_a_truncated_payload_is_an_error_at_both_levels(artifact: Path) -> None:
    path = artifact / "entities/entities.jsonl"
    path.write_bytes(path.read_bytes()[:-15])

    for level in (STRUCTURAL, FULL):
        report = validate_context_map_artifact(artifact, level=level)
        assert report.status is ValidationStatus.INVALID
        assert "file.size_mismatch" in _errors(report)


def test_a_broken_index_is_detected_structurally(artifact: Path) -> None:
    index = artifact / "indexes/entity-index.jsonl"
    lines = index.read_bytes().split(b"\n")[:-1]
    index.write_bytes(b"\n".join(reversed(lines)) + b"\n")
    _reseal(artifact)

    for level in (STRUCTURAL, FULL):
        report = validate_context_map_artifact(artifact, level=level)
        assert report.status is ValidationStatus.INVALID
        assert "index.broken" in _errors(report)


def test_an_index_that_points_at_another_record_is_detected_by_full_verification(
    artifact: Path,
) -> None:
    _replace_in(artifact, "entities/entities.jsonl", b'"key":"entity-0002"', b'"key":"entity-0009"')
    _reseal(artifact)

    structural = validate_context_map_artifact(artifact, level=STRUCTURAL)
    full = validate_context_map_artifact(artifact)

    assert structural.status is ValidationStatus.STRUCTURALLY_VALID
    assert full.status is ValidationStatus.INVALID
    assert "index.broken" in _errors(full)


def test_a_relation_that_names_an_unknown_entity_is_a_broken_reference(artifact: Path) -> None:
    _replace_in(
        artifact,
        "relations/relations.jsonl",
        b'"subject":"entity-0002"',
        b'"subject":"entity-0009"',
    )
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "reference.relation_endpoint_missing" in _errors(report)
    finding = next(
        item for item in report.findings if item.code == "reference.relation_endpoint_missing"
    )
    assert finding.subject == "relation-0002"
    assert "entity-0009" in finding.message


def test_a_stale_traversal_index_is_detected_by_rebuilding_it(artifact: Path) -> None:
    _replace_in(
        artifact,
        "indexes/entity-relation-index.jsonl",
        b'"as_object":["relation-0001"]',
        b'"as_object":["relation-0009"]',
    )
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "index.mismatch" in _errors(report)
    finding = next(item for item in report.findings if item.code == "index.mismatch")
    assert finding.subject == "indexes/entity-relation-index.jsonl"


def test_a_record_that_the_schema_refuses_is_found_by_full_verification_only(
    artifact: Path,
) -> None:
    # Um suporte geométrico fora do mapa: só a verificação completa decodifica as entidades.
    _replace_in(
        artifact,
        "entities/entities.jsonl",
        b"corridor-02--map-run-0001--geom-000000020",
        b"corridor-02--map-run-0001--geom-000009999",
    )
    _reseal(artifact)

    structural = validate_context_map_artifact(artifact, level=STRUCTURAL)
    full = validate_context_map_artifact(artifact)

    assert structural.status is ValidationStatus.STRUCTURALLY_VALID
    assert full.status is ValidationStatus.INVALID
    assert "map.invalid" in _errors(full)
    assert _outcomes(full)["schema_invariants"] is CheckOutcome.FAILED


def test_a_lineage_that_disagrees_with_the_manifest_is_an_incompatible_lineage(
    artifact: Path,
) -> None:
    def flip(record: Any) -> None:
        item = next(entry for entry in record["upstream_artifacts"] if entry["kind"] == "sequence")
        item["content_identity"] = "sha256:" + "f" * 64

    _edit_json(artifact, "lineage/lineage.json", flip)
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "lineage.mismatch" in _errors(report)


def test_the_geometry_the_map_names_must_be_the_one_the_dependency_records(
    artifact: Path,
) -> None:
    _replace_in(
        artifact, "geometry/geometry-reference.json", b'"point_count": 1000', b'"point_count": 1001'
    )
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "geometry.point_count_mismatch" in _errors(report)


def test_content_the_map_does_not_declare_is_an_error(artifact: Path) -> None:
    def undeclare(record: Any) -> None:
        record["capabilities"]["content"] = ["geometry"]
        record["capabilities"]["relation_predicates"] = []

    _edit_json(artifact, "map-metadata.json", undeclare)
    _reseal(artifact)

    for level in (STRUCTURAL, FULL):
        report = validate_context_map_artifact(artifact, level=level)
        assert report.status is ValidationStatus.INVALID
        assert "capabilities.undeclared_entities" in _errors(report)


def test_a_declared_capability_with_no_records_is_valid(world: World) -> None:
    declared_only = pinned(
        schema_context_map(metadata=metadata(capabilities=entity_capabilities())), world
    )
    artifact, _ = write_artifact(world, context_map=declared_only)

    assert validate_context_map_artifact(artifact).status is ValidationStatus.VERIFIED


def test_a_missing_required_dependency_is_an_error_and_an_optional_one_a_warning(
    world: World, artifact: Path
) -> None:
    shutil.rmtree(world.fusion_dir)

    only_optional = validate_context_map_artifact(artifact)
    assert only_optional.status is ValidationStatus.VERIFIED
    assert _codes(only_optional, Severity.WARNING) == {"dependency.optional_missing"}
    assert _errors(only_optional) == set()

    shutil.rmtree(world.geometry_dir.parents[3])
    both = validate_context_map_artifact(artifact)
    assert both.status is ValidationStatus.INVALID
    assert _errors(both) == {"dependency.required_missing"}
    statuses = {(item.artifact_id, item.requirement): item.status for item in both.dependencies}
    assert statuses[(MAP_ID, "required")] == "missing"
    assert statuses[(FUSION_ARTIFACT_ID, "optional")] == "missing"


def test_every_structural_kind_is_required_and_every_other_kind_optional(
    artifact: Path,
) -> None:
    report = validate_context_map_artifact(artifact)

    requirement = {item.artifact_id: item.requirement for item in report.dependencies}
    assert requirement[MAP_ID] == "required"
    assert requirement[ENTITY_RESOLUTION_ARTIFACT_ID] == "required"
    assert requirement[SPATIAL_RELATIONS_ARTIFACT_ID] == "required"
    assert requirement[FUSION_ARTIFACT_ID] == "optional"


def test_a_dependency_that_is_not_the_recorded_artifact_is_an_error_even_when_optional(
    world: World, artifact: Path
) -> None:
    manifest_path = world.fusion_dir / "manifest.json"
    record = json.loads(manifest_path.read_text())
    record["file_inventory"][0]["content_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(record))

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert _errors(report) == {"dependency.mismatch"}


def test_a_moved_artifact_needs_its_dependencies_told_where_they_are(
    tmp_path: Path, world: World, artifact: Path
) -> None:
    elsewhere = tmp_path / "another-machine" / "copy"
    shutil.copytree(artifact, elsewhere)
    moved: dict[str, Path] = {}
    for artifact_id, directory in world.structural_locations.items():
        moved[artifact_id] = tmp_path / "another-machine" / artifact_id
        shutil.copytree(directory, moved[artifact_id])
    shutil.rmtree(world.geometry_dir.parents[3])
    shutil.rmtree(world.resolution_dir)
    shutil.rmtree(world.relations_dir)

    alone = validate_context_map_artifact(elsewhere)
    told = validate_context_map_artifact(elsewhere, dependency_paths=moved)

    assert _errors(alone) == {"dependency.required_missing"}
    assert told.status is ValidationStatus.VERIFIED


def test_damaged_upstream_files_are_found_by_full_verification_only(
    world: World, artifact: Path
) -> None:
    payload = world.geometry_dir / "outputs/geometry.bin"
    data = bytearray(payload.read_bytes())
    data[10] ^= 0xFF
    payload.write_bytes(bytes(data))

    structural = validate_context_map_artifact(artifact, level=STRUCTURAL)
    full = validate_context_map_artifact(artifact, level=FULL)

    assert structural.status is ValidationStatus.STRUCTURALLY_VALID
    assert full.status is ValidationStatus.INVALID
    assert "dependency.upstream_damaged" in _errors(full)


def test_an_unsupported_format_version_stops_the_validation_with_a_clear_reason(
    artifact: Path,
) -> None:
    _edit_json(artifact, "manifest.json", lambda record: record.update(format_version="9.0.0"))

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert _errors(report) == {"manifest.unsupported_format_version"}
    assert _outcomes(report)["manifest"] is CheckOutcome.FAILED
    assert _outcomes(report)["file_hashes"] is CheckOutcome.SKIPPED
    assert report.format_version is None


def test_an_unsupported_schema_version_is_reported_as_such(artifact: Path) -> None:
    _edit_json(artifact, "manifest.json", lambda record: record.update(schema_version="9.0.0"))

    report = validate_context_map_artifact(artifact)

    assert _errors(report) == {"manifest.unsupported_schema_version"}


def test_a_tampered_manifest_is_reported_and_incomplete_directories_are_named(
    tmp_path: Path, artifact: Path
) -> None:
    _edit_json(artifact, "manifest.json", lambda record: record.update(entity_count=99))
    assert _errors(validate_context_map_artifact(artifact)) == {"manifest.identity_mismatch"}

    (tmp_path / ".tmp-context_map-1234abcd").mkdir()
    incomplete = validate_context_map_artifact(tmp_path / ".tmp-context_map-1234abcd")
    assert incomplete.status is ValidationStatus.INVALID
    assert _errors(incomplete) == {"artifact.incomplete"}
    assert _errors(validate_context_map_artifact(tmp_path / "nothing")) == {"artifact.incomplete"}


def test_files_outside_the_contract_are_warnings_and_are_never_read(artifact: Path) -> None:
    (artifact / "debug").mkdir()
    (artifact / "debug" / "entities.jsonl").write_text("not the data")
    (artifact / "notes.txt").write_text("stray")

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.VERIFIED
    assert {"debug.present", "file.unlisted"} <= _codes(report, Severity.WARNING)
    unlisted = next(item for item in report.findings if item.code == "file.unlisted")
    assert unlisted.subject == "notes.txt"


def test_the_report_is_deterministic_and_carries_no_machine_specific_data(
    tmp_path: Path, world: World, artifact: Path
) -> None:
    first = validate_context_map_artifact(artifact).to_json()
    second = validate_context_map_artifact(artifact).to_json()
    copied = tmp_path / "another-place" / "copy"
    shutil.copytree(artifact, copied)
    third = validate_context_map_artifact(
        copied, dependency_paths=world.structural_locations
    ).to_json()

    assert first == second == third
    assert str(tmp_path) not in first


def test_the_report_is_inspectable_json_with_the_documented_fields(artifact: Path) -> None:
    report = validate_context_map_artifact(artifact)

    record = json.loads(report.to_json())

    assert set(record) == {
        "artifact_type",
        "checks",
        "content_identity",
        "context_map_id",
        "dependencies",
        "files",
        "findings",
        "format_version",
        "level",
        "schema_version",
        "status",
        "validator_version",
    }
    assert record["status"] == "verified"
    assert record["level"] == "full"
    assert record["files"][0].keys() == {"content_hash", "path", "size_bytes", "status"}
    assert record["dependencies"][0].keys() == {
        "artifact_id",
        "artifact_type",
        "detail",
        "requirement",
        "status",
    }
    assert report.to_record() == record


def test_findings_are_ordered_errors_first_then_by_code_and_subject(artifact: Path) -> None:
    (artifact / "notes.txt").write_text("stray")
    (artifact / "relations/relations.jsonl").unlink()

    report = validate_context_map_artifact(artifact)

    severities = [item.severity for item in report.findings]
    assert severities == sorted(severities, key=lambda severity: severity is Severity.WARNING)
    errors = [
        (item.code, item.subject or "")
        for item in report.findings
        if item.severity is Severity.ERROR
    ]
    assert errors == sorted(errors)


def test_validation_never_changes_or_repairs_the_artifact(artifact: Path) -> None:
    (artifact / "entities/entities.jsonl").write_bytes(b"damaged\n")
    before = tree_snapshot(artifact)

    for level in (STRUCTURAL, FULL):
        assert validate_context_map_artifact(artifact, level=level).status is (
            ValidationStatus.INVALID
        )

    assert tree_snapshot(artifact) == before


def test_the_validator_needs_no_model_or_robotics_runtime() -> None:
    program = (
        "import sys\n"
        "import contextmap.artifact.serialization.validation\n"
        "blocked = ('torch', 'transformers', 'rclpy', 'rosbags', 'cv2', 'PIL')\n"
        "found = [name for name in blocked if name in sys.modules]\n"
        "assert not found, found\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
