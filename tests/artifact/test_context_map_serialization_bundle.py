"""The portable bundle export of a ContextMapArtifact (issue #159).

A bundle is a directory that carries the artifact and, by an explicit closure policy, the upstream
artifacts it depends on. It preserves identities and provenance, rewrites only transport-level
hints, verifies every byte it copies, never carries debug data, and stays distinguishable from
the immutable source artifact.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from context_map_serialization_builders import (
    make_context_map,
    make_evidence,
    make_geometry,
    tree_files,
    tree_snapshot,
    write_artifact,
)

from contextmap.artifact import (
    ArtifactExistsError,
    BundleError,
    ClosurePolicy,
    ContextMapArtifactReader,
    ManifestError,
    Requirement,
    ValidationStatus,
    export_bundle,
    validate_context_map_artifact,
    verify_bundle,
)
from contextmap.artifact.serialization import bundle as bundle_module
from contextmap.artifact.serialization.bundle import read_bundle_manifest
from contextmap.artifact.serialization.dependencies import read_inventory
from contextmap.artifact.serialization.layout import BUNDLE_ARTIFACT_TYPE, MANIFEST
from contextmap.artifact.serialization.manifest import decode_manifest, inventory_digest
from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for

MAP = "corridor-02--run-0001"
NOW = datetime(2026, 9, 21, 15, 0, tzinfo=UTC)


@pytest.fixture
def geometry_dir(tmp_path: Path) -> Path:
    return make_geometry(tmp_path)


@pytest.fixture
def source(tmp_path: Path, geometry_dir: Path) -> Path:
    evidence = make_evidence(tmp_path)
    return write_artifact(tmp_path, geometry_dir, evidence=(evidence,))[0]


def _export(
    tmp_path: Path,
    source: Path,
    policy: ClosurePolicy,
    *,
    name: str = "bundle",
    evidence: tuple[tuple[str, str], ...] = (),
    dependency_paths: dict[str, Path] | None = None,
) -> Path:
    output = tmp_path / "exports" / name
    export_bundle(
        source,
        output,
        policy=policy,
        evidence=evidence,
        dependency_paths=dependency_paths,
        exported_at=NOW,
    )
    return output


def _reference(index: int) -> GeometryReference:
    return GeometryReference(
        map_id=MapId(MAP), geometry_id=geometry_id_for(map_id=MapId(MAP), index=index)
    )


def test_a_core_only_bundle_carries_the_artifact_and_no_dependency(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.CORE_ONLY)

    manifest = read_bundle_manifest(bundle)
    assert manifest.closure_policy is ClosurePolicy.CORE_ONLY
    assert manifest.embedded == ()
    assert {(item.artifact_type, item.requirement) for item in manifest.omitted} == {
        ("geometric_map", "required"),
        ("semantic_fusion_run", "optional"),
    }
    assert all("core-only" in item.reason for item in manifest.omitted)
    assert not (bundle / "dependencies").exists()
    assert (bundle / "artifact" / "manifest.json").is_file()


def test_a_bundle_is_distinguishable_from_the_artifact_it_carries(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.REQUIRED)

    record = json.loads((bundle / MANIFEST).read_text())
    assert record["artifact_type"] == BUNDLE_ARTIFACT_TYPE == "context_map_bundle"
    assert record["closure_policy"] == "core+required"
    with pytest.raises(ManifestError, match="bundle"):
        ContextMapArtifactReader.open(bundle)
    assert validate_context_map_artifact(bundle).status is ValidationStatus.INVALID
    assert record.keys().isdisjoint({"run_id", "run_index", "lineage", "configuration_fingerprint"})


def test_a_required_closure_carries_the_geometry_and_opens_after_relocation(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.REQUIRED)
    relocated = tmp_path / "another-filesystem" / "moved-bundle"
    shutil.copytree(bundle, relocated)
    # Nada do workspace de origem sobrevive: a origem e a geometria original somem.
    shutil.rmtree(tmp_path / "out")
    shutil.rmtree(tmp_path / "geometry-workspace")
    shutil.rmtree(tmp_path / "semantic-fusion")

    with ContextMapArtifactReader.open(relocated / "artifact", verify_hashes=True) as reader:
        assert reader.context_map() == make_context_map()
        assert reader.entity("entity-a").key == "entity-a"
        assert reader.geometry(_reference(7)).reference == _reference(7)
    report = validate_context_map_artifact(relocated / "artifact")
    assert report.status is ValidationStatus.VERIFIED
    assert verify_bundle(relocated) == ()


def test_the_required_closure_leaves_optional_evidence_out_explicitly(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.REQUIRED)

    manifest = read_bundle_manifest(bundle)
    assert [item.artifact_type for item in manifest.embedded] == ["geometric_map"]
    assert [(item.artifact_type, item.reason) for item in manifest.omitted] == [
        ("semantic_fusion_run", "optional evidence was not selected")
    ]
    assert not (bundle / "dependencies" / "semantic_fusion_run").exists()


def test_selected_evidence_is_carried_with_the_required_closure(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(
        tmp_path,
        source,
        ClosurePolicy.SELECTED_EVIDENCE,
        evidence=(("semantic_fusion_run", "run-0003"),),
    )

    manifest = read_bundle_manifest(bundle)
    assert {item.artifact_type for item in manifest.embedded} == {
        "geometric_map",
        "semantic_fusion_run",
    }
    assert manifest.omitted == ()
    assert (
        bundle / "dependencies" / "semantic_fusion_run" / "run-0003" / "manifest.json"
    ).is_file()
    assert validate_context_map_artifact(bundle / "artifact").status is ValidationStatus.VERIFIED


def test_selecting_evidence_the_artifact_does_not_record_is_refused(
    tmp_path: Path, source: Path
) -> None:
    with pytest.raises(BundleError, match="run-9999"):
        _export(
            tmp_path,
            source,
            ClosurePolicy.SELECTED_EVIDENCE,
            evidence=(("semantic_fusion_run", "run-9999"),),
        )
    assert not (tmp_path / "exports").exists()


def test_selecting_evidence_with_another_policy_is_refused(tmp_path: Path, source: Path) -> None:
    with pytest.raises(BundleError, match="selected evidence"):
        _export(
            tmp_path,
            source,
            ClosurePolicy.REQUIRED,
            evidence=(("semantic_fusion_run", "run-0003"),),
        )


def test_a_selected_evidence_that_cannot_be_found_blocks_the_export(
    tmp_path: Path, source: Path
) -> None:
    shutil.rmtree(tmp_path / "semantic-fusion")

    with pytest.raises(BundleError, match="run-0003"):
        _export(
            tmp_path,
            source,
            ClosurePolicy.SELECTED_EVIDENCE,
            evidence=(("semantic_fusion_run", "run-0003"),),
        )
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("policy", list(ClosurePolicy))
def test_a_missing_required_dependency_blocks_every_export(
    tmp_path: Path, source: Path, policy: ClosurePolicy
) -> None:
    shutil.rmtree(tmp_path / "geometry-workspace")

    with pytest.raises(BundleError, match=r"dependency\.required_missing"):
        _export(tmp_path, source, policy)
    assert not (tmp_path / "exports").exists()


def test_a_damaged_source_blocks_the_export_and_is_never_carried(
    tmp_path: Path, source: Path
) -> None:
    path = source / "entities/entities.jsonl"
    data = bytearray(path.read_bytes())
    data[5] ^= 0x01
    path.write_bytes(bytes(data))

    with pytest.raises(BundleError, match=r"file\.hash_mismatch"):
        _export(tmp_path, source, ClosurePolicy.REQUIRED)
    assert not (tmp_path / "exports").exists()


def test_the_source_and_its_dependencies_are_left_untouched(
    tmp_path: Path, source: Path, geometry_dir: Path
) -> None:
    before = (tree_snapshot(source), tree_snapshot(geometry_dir))

    _export(
        tmp_path,
        source,
        ClosurePolicy.SELECTED_EVIDENCE,
        evidence=(("semantic_fusion_run", "run-0003"),),
    )

    assert (tree_snapshot(source), tree_snapshot(geometry_dir)) == before


def test_identities_and_content_are_preserved_and_only_locators_change(
    tmp_path: Path, source: Path, geometry_dir: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.REQUIRED)

    original = decode_manifest(json.loads((source / "manifest.json").read_text()))
    carried = decode_manifest(json.loads((bundle / "artifact" / "manifest.json").read_text()))
    assert carried.content_identity == original.content_identity
    assert carried.file_inventory == original.file_inventory
    assert carried.written_at == original.written_at
    assert carried.context_map_id == original.context_map_id
    for path in (entry.path for entry in original.file_inventory):
        assert (bundle / "artifact" / path).read_bytes() == (source / path).read_bytes()
    by_type = {item.artifact_type: item for item in carried.dependencies}
    assert by_type["geometric_map"].locator == f"../dependencies/geometric_map/{MAP}"
    assert by_type["semantic_fusion_run"].locator is None
    assert {item.artifact_type: item.content_identity for item in carried.dependencies} == {
        item.artifact_type: item.content_identity for item in original.dependencies
    }
    manifest = read_bundle_manifest(bundle)
    assert manifest.source_content_identity == original.content_identity
    assert manifest.source_context_map_id == original.context_map_id
    embedded = manifest.embedded[0]
    assert embedded.content_identity == inventory_digest(read_inventory(geometry_dir))


def test_only_contractual_files_are_carried_never_debug_or_strays(
    tmp_path: Path, source: Path, geometry_dir: Path
) -> None:
    (geometry_dir / "debug").mkdir()
    (geometry_dir / "debug" / "trace.json").write_text("human only")
    (geometry_dir / "checkpoint.pt").write_bytes(b"weights")
    assert (tmp_path / "semantic-fusion" / "debug" / "notes.txt").exists()

    bundle = _export(
        tmp_path,
        source,
        ClosurePolicy.SELECTED_EVIDENCE,
        evidence=(("semantic_fusion_run", "run-0003"),),
    )

    carried = set(tree_files(bundle))
    assert not any(path.startswith("debug/") or "/debug/" in path for path in carried)
    assert not any(path.endswith(".pt") for path in carried)
    assert not any("notes.txt" in path for path in carried)


def test_every_copied_byte_is_verified_against_the_upstream_inventory(
    tmp_path: Path, source: Path, geometry_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = bundle_module.validate_context_map_artifact

    def racing(*args: Any, **kwargs: Any) -> Any:
        report = real(*args, **kwargs)
        payload = geometry_dir / "outputs/geometry.bin"
        data = bytearray(payload.read_bytes())
        data[0] ^= 0xFF
        payload.write_bytes(bytes(data))
        return report

    monkeypatch.setattr(
        "contextmap.artifact.serialization.bundle.validate_context_map_artifact", racing
    )

    with pytest.raises(BundleError, match=r"geometry\.bin"):
        _export(tmp_path, source, ClosurePolicy.REQUIRED)

    assert not (tmp_path / "exports" / "bundle").exists()
    assert list((tmp_path / "exports").iterdir()) == []


def test_an_existing_bundle_is_never_overwritten(tmp_path: Path, source: Path) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.CORE_ONLY)
    before = tree_files(bundle)

    with pytest.raises(ArtifactExistsError):
        _export(tmp_path, source, ClosurePolicy.CORE_ONLY)

    assert tree_files(bundle) == before


def test_the_same_inputs_give_the_same_bundle_except_the_export_time(
    tmp_path: Path, source: Path
) -> None:
    first = _export(tmp_path, source, ClosurePolicy.REQUIRED, name="first")
    second = tmp_path / "exports" / "second"
    export_bundle(
        source,
        second,
        policy=ClosurePolicy.REQUIRED,
        exported_at=datetime(2031, 1, 1, tzinfo=UTC),
    )

    first_files, second_files = tree_files(first), tree_files(second)
    assert first_files.keys() == second_files.keys()
    assert {path for path in first_files if first_files[path] != second_files[path]} == {MANIFEST}
    assert (
        read_bundle_manifest(first).bundle_identity == read_bundle_manifest(second).bundle_identity
    )


def test_the_policies_produce_distinguishable_bundles(tmp_path: Path, source: Path) -> None:
    identities = {
        policy: read_bundle_manifest(_export(tmp_path, source, policy, name=policy.value))
        for policy in (ClosurePolicy.CORE_ONLY, ClosurePolicy.REQUIRED)
    }

    assert (
        identities[ClosurePolicy.CORE_ONLY].bundle_identity
        != identities[ClosurePolicy.REQUIRED].bundle_identity
    )
    assert (
        identities[ClosurePolicy.CORE_ONLY].source_content_identity
        == identities[ClosurePolicy.REQUIRED].source_content_identity
    )


def test_a_moved_source_is_exported_when_its_dependencies_are_told_where_they_are(
    tmp_path: Path, source: Path, geometry_dir: Path
) -> None:
    elsewhere = tmp_path / "moved" / "source"
    shutil.copytree(source, elsewhere)
    moved_geometry = tmp_path / "moved" / "geometry"
    shutil.copytree(geometry_dir, moved_geometry)
    shutil.rmtree(tmp_path / "geometry-workspace")

    with pytest.raises(BundleError, match=r"dependency\.required_missing"):
        _export(tmp_path, elsewhere, ClosurePolicy.REQUIRED)
    bundle = _export(
        tmp_path, elsewhere, ClosurePolicy.REQUIRED, dependency_paths={MAP: moved_geometry}
    )

    assert verify_bundle(bundle) == ()


def test_a_core_only_bundle_verifies_as_intact_with_its_omissions_stated(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.CORE_ONLY)

    assert verify_bundle(bundle) == ()
    assert validate_context_map_artifact(bundle / "artifact").status is ValidationStatus.INVALID, (
        "as a standalone artifact the core-only bundle lacks its required geometry"
    )


def test_a_modified_embedded_file_is_detected_when_the_bundle_is_verified(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.REQUIRED)
    payload = next(bundle.glob("dependencies/geometric_map/*/outputs/geometry.bin"))
    data = bytearray(payload.read_bytes())
    data[3] ^= 0xFF
    payload.write_bytes(bytes(data))

    problems = verify_bundle(bundle)

    assert problems and any("geometry.bin" in problem for problem in problems)


def test_a_modified_artifact_file_or_bundle_manifest_is_detected(
    tmp_path: Path, source: Path
) -> None:
    bundle = _export(tmp_path, source, ClosurePolicy.REQUIRED)
    path = bundle / "artifact" / "lineage" / "lineage.json"
    path.write_bytes(path.read_bytes().replace(b"required", b"optional"))
    assert any("lineage.json" in problem for problem in verify_bundle(bundle))

    other = _export(tmp_path, source, ClosurePolicy.REQUIRED, name="other")
    manifest_path = other / MANIFEST
    record = json.loads(manifest_path.read_text())
    record["closure_policy"] = "core-only"
    manifest_path.write_text(json.dumps(record))
    assert any("identity" in problem for problem in verify_bundle(other))


def test_a_directory_that_is_not_a_bundle_is_refused(tmp_path: Path, source: Path) -> None:
    with pytest.raises(BundleError, match="not a bundle"):
        read_bundle_manifest(source)


def test_evidence_can_be_required_and_is_then_carried_with_the_required_closure(
    tmp_path: Path, geometry_dir: Path
) -> None:
    required = make_evidence(tmp_path, requirement=Requirement.REQUIRED)
    source, _ = write_artifact(tmp_path, geometry_dir, evidence=(required,))

    bundle = _export(tmp_path, source, ClosurePolicy.REQUIRED)

    assert {item.artifact_type for item in read_bundle_manifest(bundle).embedded} == {
        "geometric_map",
        "semantic_fusion_run",
    }
