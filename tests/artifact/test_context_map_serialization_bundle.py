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
from context_map_builders import (
    ENTITY_RESOLUTION_ARTIFACT_ID,
    FUSION_ARTIFACT_ID,
    SEQUENCE_ARTIFACT_ID,
    SPATIAL_RELATIONS_ARTIFACT_ID,
)
from context_map_serialization_builders import (
    MAP_ID,
    World,
    make_context_map,
    make_world,
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
    ValidationStatus,
    export_bundle,
    validate_context_map_artifact,
    verify_bundle,
)
from contextmap.artifact.serialization import bundle as bundle_module
from contextmap.artifact.serialization.bundle import read_bundle_manifest
from contextmap.artifact.serialization.dependencies import artifact_digest
from contextmap.artifact.serialization.layout import BUNDLE_ARTIFACT_TYPE, MANIFEST
from contextmap.artifact.serialization.manifest import decode_manifest
from contextmap.entity_resolution import EntityResolutionRunReader
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometryReference,
    MapId,
    geometry_id_for,
)
from contextmap.spatial_relations import SpatialRelationsRunReader

NOW = datetime(2026, 9, 21, 15, 0, tzinfo=UTC)
FUSION = ("semantic_fusion_run", FUSION_ARTIFACT_ID)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


@pytest.fixture
def source(world: World) -> Path:
    return write_artifact(world)[0]


def _export(
    world: World,
    source: Path,
    policy: ClosurePolicy,
    *,
    name: str = "bundle",
    evidence: tuple[tuple[str, str], ...] = (),
    dependency_paths: dict[str, Path] | None = None,
) -> Path:
    output = world.root / "exports" / name
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
        map_id=MapId(MAP_ID), geometry_id=geometry_id_for(map_id=MapId(MAP_ID), index=index)
    )


def _remove_upstream(world: World) -> None:
    for directory in (
        world.geometry_dir,
        world.resolution_dir,
        world.relations_dir,
        world.fusion_dir,
    ):
        shutil.rmtree(directory)


def test_a_core_only_bundle_carries_the_artifact_and_no_dependency(
    world: World, source: Path
) -> None:
    bundle = _export(world, source, ClosurePolicy.CORE_ONLY)

    manifest = read_bundle_manifest(bundle)
    lineage = {item.artifact_id for item in make_context_map(world).lineage}
    assert manifest.closure_policy is ClosurePolicy.CORE_ONLY
    assert manifest.embedded == ()
    assert {item.artifact_id for item in manifest.omitted} == lineage
    assert all("core-only" in item.reason for item in manifest.omitted)
    assert not (bundle / "dependencies").exists()
    assert (bundle / "artifact" / "manifest.json").is_file()


def test_a_bundle_is_distinguishable_from_the_artifact_it_carries(
    world: World, source: Path
) -> None:
    bundle = _export(world, source, ClosurePolicy.REQUIRED)

    record = json.loads((bundle / MANIFEST).read_text())
    assert record["artifact_type"] == BUNDLE_ARTIFACT_TYPE == "context_map_bundle"
    assert record["closure_policy"] == "core+required"
    with pytest.raises(ManifestError, match="bundle"):
        ContextMapArtifactReader.open(bundle)
    assert validate_context_map_artifact(bundle).status is ValidationStatus.INVALID
    assert record.keys().isdisjoint({"run_id", "run_index", "lineage", "configuration_fingerprint"})


def test_a_required_closure_carries_the_structural_artifacts_and_opens_after_relocation(
    world: World, source: Path
) -> None:
    written = make_context_map(world)
    bundle = _export(world, source, ClosurePolicy.REQUIRED)
    relocated = world.root / "another-filesystem" / "moved-bundle"
    carried = relocated / "dependencies"
    shutil.copytree(bundle, relocated)
    # Nada do workspace de origem sobrevive: a origem e todos os upstream originais somem.
    shutil.rmtree(world.root / "out")
    _remove_upstream(world)

    with ContextMapArtifactReader.open(relocated / "artifact", verify_hashes=True) as reader:
        assert reader.context_map() == written
        assert reader.geometry(_reference(7)).reference == _reference(7)
    report = validate_context_map_artifact(relocated / "artifact")
    assert report.status is ValidationStatus.VERIFIED
    assert verify_bundle(relocated) == ()
    # Os runs reais copiados continuam íntegros para os leitores dos próprios donos.
    resolution = EntityResolutionRunReader(
        carried / "entity_resolution_run" / ENTITY_RESOLUTION_ARTIFACT_ID
    )
    relations = SpatialRelationsRunReader(
        carried / "spatial_relations_run" / SPATIAL_RELATIONS_ARTIFACT_ID
    )
    assert resolution.verify_integrity() == []
    assert relations.verify_integrity() == []
    assert relations.validate_resolution(resolution) == ()
    with GeometricMapArtifactReader(carried / "geometric_map" / MAP_ID) as geometry:
        assert geometry.verify_integrity() == []


def test_the_required_closure_leaves_optional_evidence_out_explicitly(
    world: World, source: Path
) -> None:
    bundle = _export(world, source, ClosurePolicy.REQUIRED)

    manifest = read_bundle_manifest(bundle)
    assert [item.artifact_type for item in manifest.embedded] == [
        "entity_resolution_run",
        "geometric_map",
        "spatial_relations_run",
    ]
    assert {item.reason for item in manifest.omitted} == {"optional evidence was not selected"}
    assert FUSION_ARTIFACT_ID in {item.artifact_id for item in manifest.omitted}
    assert not (bundle / "dependencies" / "semantic_fusion_run").exists()


def test_selected_evidence_is_carried_with_the_required_closure(world: World, source: Path) -> None:
    bundle = _export(world, source, ClosurePolicy.SELECTED_EVIDENCE, evidence=(FUSION,))

    manifest = read_bundle_manifest(bundle)
    assert "semantic_fusion_run" in {item.artifact_type for item in manifest.embedded}
    assert FUSION_ARTIFACT_ID not in {item.artifact_id for item in manifest.omitted}
    assert (
        bundle / "dependencies" / "semantic_fusion_run" / FUSION_ARTIFACT_ID / "manifest.json"
    ).is_file()
    assert validate_context_map_artifact(bundle / "artifact").status is ValidationStatus.VERIFIED


def test_selecting_evidence_the_artifact_does_not_record_is_refused(
    world: World, source: Path
) -> None:
    with pytest.raises(BundleError, match="run-9999"):
        _export(
            world,
            source,
            ClosurePolicy.SELECTED_EVIDENCE,
            evidence=(("semantic_fusion_run", "run-9999"),),
        )
    assert not (world.root / "exports").exists()


def test_selecting_evidence_with_another_policy_is_refused(world: World, source: Path) -> None:
    with pytest.raises(BundleError, match="selected evidence"):
        _export(world, source, ClosurePolicy.REQUIRED, evidence=(FUSION,))


def test_a_selected_evidence_that_cannot_be_found_blocks_the_export(
    world: World, source: Path
) -> None:
    shutil.rmtree(world.fusion_dir)

    with pytest.raises(BundleError, match=FUSION_ARTIFACT_ID):
        _export(world, source, ClosurePolicy.SELECTED_EVIDENCE, evidence=(FUSION,))
    with pytest.raises(BundleError, match=SEQUENCE_ARTIFACT_ID):
        _export(
            world,
            source,
            ClosurePolicy.SELECTED_EVIDENCE,
            evidence=(("sequence", SEQUENCE_ARTIFACT_ID),),
        )
    assert not (world.root / "exports").exists()


@pytest.mark.parametrize("policy", list(ClosurePolicy))
def test_a_missing_required_dependency_blocks_every_export(
    world: World, source: Path, policy: ClosurePolicy
) -> None:
    shutil.rmtree(world.geometry_dir)

    with pytest.raises(BundleError, match=r"dependency\.required_missing"):
        _export(world, source, policy)
    assert not (world.root / "exports").exists()


def test_a_damaged_source_blocks_the_export_and_is_never_carried(
    world: World, source: Path
) -> None:
    path = source / "entities/entities.jsonl"
    data = bytearray(path.read_bytes())
    data[5] ^= 0x01
    path.write_bytes(bytes(data))

    with pytest.raises(BundleError, match=r"file\.hash_mismatch"):
        _export(world, source, ClosurePolicy.REQUIRED)
    assert not (world.root / "exports").exists()


def test_the_source_and_its_dependencies_are_left_untouched(world: World, source: Path) -> None:
    before = (tree_snapshot(source), tree_snapshot(world.geometry_dir))

    _export(world, source, ClosurePolicy.SELECTED_EVIDENCE, evidence=(FUSION,))

    assert (tree_snapshot(source), tree_snapshot(world.geometry_dir)) == before


def test_identities_and_content_are_preserved_and_only_locators_change(
    world: World, source: Path
) -> None:
    bundle = _export(world, source, ClosurePolicy.REQUIRED)

    original = decode_manifest(json.loads((source / "manifest.json").read_text()))
    carried = decode_manifest(json.loads((bundle / "artifact" / "manifest.json").read_text()))
    assert carried.content_identity == original.content_identity
    assert carried.file_inventory == original.file_inventory
    assert carried.written_at == original.written_at
    assert carried.context_map_id == original.context_map_id
    for path in (entry.path for entry in original.file_inventory):
        assert (bundle / "artifact" / path).read_bytes() == (source / path).read_bytes()
    by_id = {item.artifact_id: item for item in carried.dependencies}
    assert by_id[MAP_ID].locator == f"../dependencies/geometric_map/{MAP_ID}"
    assert by_id[FUSION_ARTIFACT_ID].locator is None
    assert {item.artifact_id: item.content_identity for item in carried.dependencies} == {
        item.artifact_id: item.content_identity for item in original.dependencies
    }
    manifest = read_bundle_manifest(bundle)
    assert manifest.source_content_identity == original.content_identity
    assert manifest.source_context_map_id == original.context_map_id
    embedded = next(item for item in manifest.embedded if item.artifact_id == MAP_ID)
    assert embedded.content_identity == artifact_digest(world.geometry_dir)


def test_only_contractual_files_are_carried_never_debug_or_strays(
    world: World, source: Path
) -> None:
    (world.geometry_dir / "debug").mkdir()
    (world.geometry_dir / "debug" / "trace.json").write_text("human only")
    (world.geometry_dir / "checkpoint.pt").write_bytes(b"weights")
    assert (world.fusion_dir / "debug" / "notes.txt").exists()

    bundle = _export(world, source, ClosurePolicy.SELECTED_EVIDENCE, evidence=(FUSION,))

    carried = set(tree_files(bundle))
    assert not any(path.startswith("debug/") or "/debug/" in path for path in carried)
    assert not any(path.endswith(".pt") for path in carried)
    assert not any("notes.txt" in path for path in carried)


def test_every_copied_byte_is_verified_against_the_upstream_inventory(
    world: World, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = bundle_module.validate_context_map_artifact

    def racing(*args: Any, **kwargs: Any) -> Any:
        report = real(*args, **kwargs)
        payload = world.geometry_dir / "outputs/geometry.bin"
        data = bytearray(payload.read_bytes())
        data[0] ^= 0xFF
        payload.write_bytes(bytes(data))
        return report

    monkeypatch.setattr(
        "contextmap.artifact.serialization.bundle.validate_context_map_artifact", racing
    )

    with pytest.raises(BundleError, match=r"geometry\.bin"):
        _export(world, source, ClosurePolicy.REQUIRED)

    assert not (world.root / "exports" / "bundle").exists()
    assert list((world.root / "exports").iterdir()) == []


def test_an_existing_bundle_is_never_overwritten(world: World, source: Path) -> None:
    bundle = _export(world, source, ClosurePolicy.CORE_ONLY)
    before = tree_files(bundle)

    with pytest.raises(ArtifactExistsError):
        _export(world, source, ClosurePolicy.CORE_ONLY)

    assert tree_files(bundle) == before


def test_the_same_inputs_give_the_same_bundle_except_the_export_time(
    world: World, source: Path
) -> None:
    first = _export(world, source, ClosurePolicy.REQUIRED, name="first")
    second = world.root / "exports" / "second"
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


def test_the_policies_produce_distinguishable_bundles(world: World, source: Path) -> None:
    identities = {
        policy: read_bundle_manifest(_export(world, source, policy, name=policy.value))
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
    world: World, source: Path
) -> None:
    elsewhere = world.root / "moved" / "source"
    shutil.copytree(source, elsewhere)
    moved: dict[str, Path] = {}
    for artifact_id, directory in world.structural_locations.items():
        moved[artifact_id] = world.root / "moved" / artifact_id
        shutil.copytree(directory, moved[artifact_id])
    shutil.rmtree(world.geometry_dir)
    shutil.rmtree(world.resolution_dir)
    shutil.rmtree(world.relations_dir)

    with pytest.raises(BundleError, match=r"dependency\.required_missing"):
        _export(world, elsewhere, ClosurePolicy.REQUIRED)
    bundle = _export(world, elsewhere, ClosurePolicy.REQUIRED, dependency_paths=moved)

    assert verify_bundle(bundle) == ()


def test_a_core_only_bundle_verifies_as_intact_with_its_omissions_stated(
    world: World, source: Path
) -> None:
    bundle = _export(world, source, ClosurePolicy.CORE_ONLY)

    assert verify_bundle(bundle) == ()
    assert validate_context_map_artifact(bundle / "artifact").status is ValidationStatus.INVALID, (
        "as a standalone artifact the core-only bundle lacks its required dependencies"
    )


def test_a_modified_embedded_file_is_detected_when_the_bundle_is_verified(
    world: World, source: Path
) -> None:
    bundle = _export(world, source, ClosurePolicy.REQUIRED)
    payload = next(bundle.glob("dependencies/geometric_map/*/outputs/geometry.bin"))
    data = bytearray(payload.read_bytes())
    data[3] ^= 0xFF
    payload.write_bytes(bytes(data))

    problems = verify_bundle(bundle)

    assert problems and any("geometry.bin" in problem for problem in problems)


def test_a_modified_artifact_file_or_bundle_manifest_is_detected(
    world: World, source: Path
) -> None:
    bundle = _export(world, source, ClosurePolicy.REQUIRED)
    path = bundle / "artifact" / "lineage" / "lineage.json"
    path.write_bytes(path.read_bytes() + b" ")
    assert any("lineage.json" in problem for problem in verify_bundle(bundle))

    other = _export(world, source, ClosurePolicy.REQUIRED, name="other")
    manifest_path = other / MANIFEST
    record = json.loads(manifest_path.read_text())
    record["closure_policy"] = "core-only"
    manifest_path.write_text(json.dumps(record))
    assert any("identity" in problem for problem in verify_bundle(other))


def test_a_directory_that_is_not_a_bundle_is_refused(source: Path) -> None:
    with pytest.raises(BundleError, match="not a bundle"):
        read_bundle_manifest(source)
