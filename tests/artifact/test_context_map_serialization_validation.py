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
from pathlib import Path

import pytest
from context_map_serialization_builders import (
    make_context_map,
    make_evidence,
    make_geometry,
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
from contextmap.artifact.manifest import create_manifest, decode_manifest, encode_manifest
from contextmap.artifact.tables import canonical_json_line, encode_record_table
from contextmap.shared import file_entry

FULL = ValidationLevel.FULL
STRUCTURAL = ValidationLevel.STRUCTURAL


@pytest.fixture
def geometry_dir(tmp_path: Path) -> Path:
    return make_geometry(tmp_path)


@pytest.fixture
def artifact(tmp_path: Path, geometry_dir: Path) -> Path:
    return write_artifact(tmp_path, geometry_dir)[0]


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


def _reseal(artifact: Path, *, entity_count: int | None = None) -> None:
    """Recompute inventory and identity from the disk so only the intended damage remains."""
    manifest = decode_manifest(json.loads((artifact / "manifest.json").read_text()))
    payloads = manifest.payloads
    resealed = create_manifest(
        context_map_id=manifest.context_map_id,
        schema_version=manifest.schema_version,
        written_at=manifest.written_at,
        code_version=manifest.code_version,
        configuration_fingerprint=manifest.configuration_fingerprint,
        entity_count=manifest.entity_count if entity_count is None else entity_count,
        relation_count=manifest.relation_count,
        payloads=payloads,
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


def test_an_intact_artifact_is_fully_verified_with_nothing_to_report(artifact: Path) -> None:
    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.VERIFIED
    assert report.level is FULL
    assert report.findings == ()
    assert report.validator_version == VALIDATOR_VERSION
    assert report.artifact_type == "context_map"
    assert report.format_version == "0.1.0"
    assert report.schema_version == "0.1.0"
    assert report.context_map_id == str(make_context_map().context_map_id)
    assert report.content_identity is not None and report.content_identity.startswith("sha256:")
    assert {item.status for item in report.files} == {FileStatus.VERIFIED}
    assert [item.path for item in report.files] == sorted(item.path for item in report.files)
    assert all(item.outcome is not CheckOutcome.FAILED for item in report.checks)


def test_the_report_lists_every_check_including_the_ones_that_could_not_run(
    artifact: Path,
) -> None:
    outcomes = _outcomes(validate_context_map_artifact(artifact))

    assert outcomes["manifest"] is CheckOutcome.PASSED
    assert outcomes["file_hashes"] is CheckOutcome.PASSED
    assert outcomes["reference_integrity"] is CheckOutcome.PASSED
    assert outcomes["index_rebuild"] is CheckOutcome.PASSED
    assert outcomes["dependencies"] is CheckOutcome.PASSED
    assert outcomes["entity_geometry_support"] is CheckOutcome.SKIPPED


def test_structural_validation_is_fast_and_never_claims_the_artifact_is_verified(
    artifact: Path,
) -> None:
    report = validate_context_map_artifact(artifact, level=STRUCTURAL)

    assert report.status is ValidationStatus.STRUCTURALLY_VALID
    assert report.status is not ValidationStatus.VERIFIED
    assert report.findings == ()
    assert {item.status for item in report.files} == {FileStatus.SIZE_OK}
    outcomes = _outcomes(report)
    for name in ("file_hashes", "reference_integrity", "index_rebuild", "dependency_integrity"):
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
    _replace_in(artifact, "entities/entities.jsonl", b'"key":"entity-b"', b'"key":"entity-x"')
    _reseal(artifact)

    structural = validate_context_map_artifact(artifact, level=STRUCTURAL)
    full = validate_context_map_artifact(artifact)

    assert structural.status is ValidationStatus.STRUCTURALLY_VALID
    assert full.status is ValidationStatus.INVALID
    assert "index.broken" in _errors(full)


def test_a_relation_that_names_an_unknown_entity_is_a_broken_reference(artifact: Path) -> None:
    _replace_in(
        artifact, "relations/relations.jsonl", b'"subject":"entity-c"', b'"subject":"entity-z"'
    )
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "reference.relation_endpoint_missing" in _errors(report)
    finding = next(
        item for item in report.findings if item.code == "reference.relation_endpoint_missing"
    )
    assert finding.subject == "relation-2"
    assert "entity-z" in finding.message


def test_a_stale_traversal_index_is_detected_by_rebuilding_it(artifact: Path) -> None:
    _replace_in(
        artifact,
        "indexes/entity-relation-index.jsonl",
        b'"as_object":["relation-1"]',
        b'"as_object":["relation-9"]',
    )
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "index.mismatch" in _errors(report)
    finding = next(item for item in report.findings if item.code == "index.mismatch")
    assert finding.subject == "indexes/entity-relation-index.jsonl"


def test_a_lineage_that_disagrees_with_the_manifest_is_an_incompatible_lineage(
    artifact: Path,
) -> None:
    _replace_in(artifact, "lineage/lineage.json", b'"required"', b'"optional"')
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "lineage.mismatch" in _errors(report)


def test_the_geometry_the_map_names_must_be_the_one_the_dependency_records(
    artifact: Path,
) -> None:
    _replace_in(
        artifact, "geometry/geometry-reference.json", b'"point_count": 24', b'"point_count": 25'
    )
    _reseal(artifact)

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "geometry.point_count_mismatch" in _errors(report)


def test_content_the_map_does_not_declare_is_an_error(tmp_path: Path, geometry_dir: Path) -> None:
    artifact, manifest = write_artifact(
        tmp_path,
        geometry_dir,
        context_map=make_context_map(entities=False, relations=False),
        entities=(),
        relations=(),
    )
    table = encode_record_table([{"key": "entity-a", "record": {"label": "ghost"}}])
    (artifact / "entities/entities.jsonl").write_bytes(table.payload)
    (artifact / "indexes/entity-index.jsonl").write_bytes(table.index)
    (artifact / "indexes/entity-relation-index.jsonl").write_bytes(
        canonical_json_line({"key": "entity-a", "as_subject": [], "as_object": []}) + b"\n"
    )
    payloads = tuple(
        payload.__class__(**{**payload.__dict__, "record_count": 1})
        if payload.path.endswith(
            ("entities.jsonl", "entity-index.jsonl", "entity-relation-index.jsonl")
        )
        else payload
        for payload in manifest.payloads
    )
    resealed = create_manifest(
        context_map_id=manifest.context_map_id,
        schema_version=manifest.schema_version,
        written_at=manifest.written_at,
        code_version=manifest.code_version,
        configuration_fingerprint=manifest.configuration_fingerprint,
        entity_count=1,
        relation_count=0,
        payloads=payloads,
        dependencies=manifest.dependencies,
        file_inventory=[
            file_entry(entry.path, (artifact / entry.path).read_bytes())
            for entry in manifest.file_inventory
        ],
    )
    (artifact / "manifest.json").write_text(json.dumps(encode_manifest(resealed)))

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert "capabilities.undeclared_entities" in _errors(report)


def test_a_declared_capability_with_no_records_is_valid(tmp_path: Path, geometry_dir: Path) -> None:
    artifact, _ = write_artifact(tmp_path, geometry_dir, entities=(), relations=())

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.VERIFIED


def test_a_missing_required_dependency_is_an_error_and_an_optional_one_a_warning(
    tmp_path: Path, geometry_dir: Path
) -> None:
    evidence = make_evidence(tmp_path)
    artifact, _ = write_artifact(tmp_path, geometry_dir, evidence=(evidence,))
    shutil.rmtree(evidence.location)

    only_optional = validate_context_map_artifact(artifact)
    assert only_optional.status is ValidationStatus.VERIFIED
    assert _codes(only_optional, Severity.WARNING) == {"dependency.optional_missing"}
    assert _errors(only_optional) == set()

    shutil.rmtree(tmp_path / "geometry-workspace")
    both = validate_context_map_artifact(artifact)
    assert both.status is ValidationStatus.INVALID
    assert _errors(both) == {"dependency.required_missing"}
    statuses = {(item.artifact_type, item.requirement): item.status for item in both.dependencies}
    assert statuses[("geometric_map", "required")] == "missing"
    assert statuses[("semantic_fusion_run", "optional")] == "missing"


def test_a_dependency_that_is_not_the_recorded_artifact_is_an_error_even_when_optional(
    tmp_path: Path, geometry_dir: Path
) -> None:
    evidence = make_evidence(tmp_path)
    artifact, _ = write_artifact(tmp_path, geometry_dir, evidence=(evidence,))
    manifest_path = evidence.location / "manifest.json"
    record = json.loads(manifest_path.read_text())
    record["file_inventory"][0]["content_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(record))

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert _errors(report) == {"dependency.mismatch"}


def test_a_moved_artifact_needs_its_dependencies_told_where_they_are(
    tmp_path: Path, artifact: Path, geometry_dir: Path
) -> None:
    elsewhere = tmp_path / "another-machine" / "copy"
    shutil.copytree(artifact, elsewhere)
    moved_geometry = tmp_path / "another-machine" / "geometry"
    shutil.copytree(geometry_dir, moved_geometry)
    shutil.rmtree(tmp_path / "geometry-workspace")

    alone = validate_context_map_artifact(elsewhere)
    told = validate_context_map_artifact(
        elsewhere, dependency_paths={"corridor-02--run-0001": moved_geometry}
    )

    assert _errors(alone) == {"dependency.required_missing"}
    assert told.status is ValidationStatus.VERIFIED


def test_damaged_upstream_files_are_found_by_full_verification_only(
    artifact: Path, geometry_dir: Path
) -> None:
    payload = geometry_dir / "outputs/geometry.bin"
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
    path = artifact / "manifest.json"
    record = json.loads(path.read_text())
    record["format_version"] = "9.0.0"
    path.write_text(json.dumps(record))

    report = validate_context_map_artifact(artifact)

    assert report.status is ValidationStatus.INVALID
    assert _errors(report) == {"manifest.unsupported_format_version"}
    assert _outcomes(report)["manifest"] is CheckOutcome.FAILED
    assert _outcomes(report)["file_hashes"] is CheckOutcome.SKIPPED
    assert report.format_version is None


def test_an_unsupported_schema_version_is_reported_as_such(artifact: Path) -> None:
    path = artifact / "manifest.json"
    record = json.loads(path.read_text())
    record["schema_version"] = "9.0.0"
    path.write_text(json.dumps(record))

    report = validate_context_map_artifact(artifact)

    assert _errors(report) == {"manifest.unsupported_schema_version"}


def test_a_tampered_manifest_is_reported_and_incomplete_directories_are_named(
    tmp_path: Path, artifact: Path
) -> None:
    path = artifact / "manifest.json"
    record = json.loads(path.read_text())
    record["entity_count"] = 99
    path.write_text(json.dumps(record))
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
    assert _codes(report, Severity.WARNING) == {"debug.present", "file.unlisted"}
    unlisted = next(item for item in report.findings if item.code == "file.unlisted")
    assert unlisted.subject == "notes.txt"


def test_the_report_is_deterministic_and_carries_no_machine_specific_data(
    tmp_path: Path, artifact: Path
) -> None:
    first = validate_context_map_artifact(artifact).to_json()
    second = validate_context_map_artifact(artifact).to_json()
    copied = tmp_path / "another-place" / "copy"
    shutil.copytree(artifact, copied)
    third = validate_context_map_artifact(
        copied,
        dependency_paths={
            "corridor-02--run-0001": tmp_path
            / "geometry-workspace"
            / "runs"
            / "geometric-mapping"
            / "corridor-02"
            / "run-0001__full__baseline"
        },
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
        "import contextmap.artifact.validation\n"
        "blocked = ('torch', 'transformers', 'rclpy', 'rosbags', 'cv2', 'PIL')\n"
        "found = [name for name in blocked if name in sys.modules]\n"
        "assert not found, found\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
