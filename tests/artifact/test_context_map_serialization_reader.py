"""The lightweight ContextMapArtifact reader (issue #157).

It opens an artifact from its own directory, reads entities, relations and metadata lazily as the
schema's own types, resolves geometry through the real GeometricMapArtifactReader and never
mutates anything. Every damaged or unsupported input is an explicit, typed error.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from context_map_builders import (
    CONTEXT_MAP_ID,
    ENTITY_RESOLUTION_ARTIFACT_ID,
    FUSION_ARTIFACT_ID,
    entity_reference,
)
from context_map_serialization_builders import (
    MAP_ID,
    World,
    make_context_map,
    make_world,
    tree_snapshot,
    write_artifact,
)

from contextmap.artifact import (
    ArtifactIntegrityError,
    ContextEntityReference,
    ContextMapArtifactError,
    ContextMapArtifactReader,
    DependencyMismatchError,
    ForeignContextEntityReferenceError,
    ManifestError,
    MissingDependencyError,
    UnknownContextEntityError,
    UnresolvedReferenceError,
    UnsupportedArtifactSchemaError,
    UnsupportedFormatVersionError,
    UnsupportedSchemaVersionError,
)
from contextmap.artifact.serialization.errors import (
    BrokenIndexError,
    IncompleteContextMapArtifactError,
    MissingPayloadError,
    RecordNotFoundError,
)
from contextmap.artifact.serialization.manifest import (
    create_manifest,
    decode_manifest,
    encode_manifest,
)
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometryReference,
    MapId,
    geometry_id_for,
)
from contextmap.shared import file_entry

MAP = MapId(MAP_ID)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


@pytest.fixture
def artifact(world: World) -> Path:
    return write_artifact(world)[0]


@pytest.fixture
def reader(artifact: Path) -> Iterator[ContextMapArtifactReader]:
    with ContextMapArtifactReader.open(artifact) as opened:
        yield opened


def _reference(index: int, map_id: MapId = MAP) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


def _rewrite_manifest(artifact: Path, **changes: Any) -> None:
    path = artifact / "manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(changes)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def _reseal(artifact: Path, *, drop: tuple[str, ...] = ()) -> None:
    """Recompute the inventory and identity from the disk, so only what a test changed is wrong."""
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
            if entry.path not in drop
        ],
    )
    (artifact / "manifest.json").write_text(json.dumps(encode_manifest(resealed)))


def test_the_reader_returns_the_map_that_was_written(
    world: World, reader: ContextMapArtifactReader
) -> None:
    written = make_context_map(world)

    assert reader.context_map() == written
    assert reader.metadata() == written.metadata
    assert reader.geometry_link() == written.geometry_ref
    assert reader.lineage() == written.lineage
    assert reader.manifest.entity_count == 3


def test_map_bounds_are_the_declared_extent_in_the_map_frame(
    world: World, reader: ContextMapArtifactReader
) -> None:
    written = make_context_map(world)

    assert reader.map_bounds() == written.metadata.bounds
    assert reader.map_bounds().frame_id == written.metadata.frame.frame_id


def test_entities_round_trip_as_schema_types_and_stream_in_id_order(
    world: World, reader: ContextMapArtifactReader
) -> None:
    written = make_context_map(world)

    assert reader.entity_ids() == ("entity-0001", "entity-0002", "entity-0003")
    assert tuple(reader.entities()) == written.entities
    reference = entity_reference("entity-0002")
    assert reader.entity(reference) == written.entity(reference)


def test_relations_round_trip_as_schema_types(
    world: World, reader: ContextMapArtifactReader
) -> None:
    written = make_context_map(world)

    assert reader.relation_ids() == tuple(str(item.relation_id) for item in written.relations)
    assert tuple(reader.relations()) == written.relations
    assert reader.relation(str(written.relations[1].relation_id)) == written.relations[1]


def test_relations_for_an_entity_lists_those_it_takes_part_in(
    world: World, reader: ContextMapArtifactReader
) -> None:
    written = make_context_map(world)

    for entity_id in ("entity-0001", "entity-0002", "entity-0003"):
        reference = entity_reference(entity_id)
        assert {item.relation_id for item in reader.relations_for(reference)} == {
            item.relation_id for item in written.relations_for(reference)
        }
    assert len(reader.relations_for(entity_reference("entity-0003"))) > 2


def test_a_map_without_entities_reads_as_empty(tmp_path: Path) -> None:
    workspace = make_world(tmp_path / "geometry-only")
    written = make_context_map(workspace, kind="geometry-only")
    artifact, _ = write_artifact(workspace, context_map=written)

    with ContextMapArtifactReader.open(artifact) as opened:
        assert opened.entity_ids() == ()
        assert opened.relation_ids() == ()
        assert list(opened.entities()) == []
        assert opened.context_map() == written


def test_unknown_and_foreign_references_are_explicit_errors(
    reader: ContextMapArtifactReader,
) -> None:
    with pytest.raises(UnknownContextEntityError, match="entity-9999"):
        reader.entity(entity_reference("entity-9999"))
    with pytest.raises(UnknownContextEntityError, match="entity-9999"):
        reader.relations_for(entity_reference("entity-9999"))
    with pytest.raises(RecordNotFoundError, match="relation-9999"):
        reader.relation("relation-9999")
    foreign = entity_reference("entity-0001", context_map_id="another-map")
    with pytest.raises(ForeignContextEntityReferenceError, match="another-map"):
        reader.entity(foreign)
    with pytest.raises(ForeignContextEntityReferenceError):
        reader.relations_for(foreign)


def test_geometry_resolves_through_the_geometric_map_reader(
    world: World, reader: ContextMapArtifactReader
) -> None:
    with GeometricMapArtifactReader(world.geometry_dir) as direct:
        expected = direct.geometry().get(_reference(5))

    point = reader.geometry(_reference(5))

    assert point == expected
    assert point.reference == _reference(5)
    assert reader.geometry_source().geometric_map.map_id == MAP
    assert sum(1 for _ in reader.geometry_source().iter_geometry()) == 1000


def test_the_geometry_is_opened_only_when_it_is_asked_for(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    original = GeometricMapArtifactReader

    class Counting(original):  # type: ignore[valid-type, misc]
        def __init__(self, run_dir: Path) -> None:
            opened.append(run_dir)
            super().__init__(run_dir)

    monkeypatch.setattr(
        "contextmap.artifact.serialization.reader.GeometricMapArtifactReader", Counting
    )

    with ContextMapArtifactReader.open(artifact) as reader:
        reader.metadata()
        reader.entity(entity_reference("entity-0001"))
        list(reader.relations())
        assert opened == []
        reader.geometry(_reference(0))
        reader.geometry(_reference(1))
        assert len(opened) == 1


def test_references_are_validated_against_the_map_without_loading_geometry(
    reader: ContextMapArtifactReader,
) -> None:
    reader.validate_reference(_reference(0))
    reader.validate_reference(_reference(999))

    with pytest.raises(UnresolvedReferenceError, match="another-map"):
        reader.validate_reference(_reference(0, MapId("another-map")))
    with pytest.raises(UnresolvedReferenceError, match="1000"):
        reader.validate_reference(_reference(1000))
    with pytest.raises(UnresolvedReferenceError, match="canonical"):
        reader.validate_reference(
            GeometryReference(map_id=MAP, geometry_id=geometry_id_for(map_id=MAP, index=1) + "x")  # type: ignore[arg-type]
        )


def test_a_moved_artifact_opens_and_reads_everything_that_needs_no_geometry(
    tmp_path: Path, world: World, artifact: Path
) -> None:
    written = make_context_map(world)
    elsewhere = tmp_path / "another-machine" / "copy"
    shutil.copytree(artifact, elsewhere)
    for directory in (
        world.geometry_dir,
        world.resolution_dir,
        world.relations_dir,
        world.fusion_dir,
    ):
        shutil.rmtree(directory)

    with ContextMapArtifactReader.open(elsewhere) as reader:
        assert reader.context_map() == written
        assert reader.entity(entity_reference("entity-0001")).entity_id == "entity-0001"
        assert reader.relations_for(entity_reference("entity-0001")) == written.relations_for(
            entity_reference("entity-0001")
        )
        with pytest.raises(MissingDependencyError, match=MAP_ID):
            reader.geometry(_reference(0))


def test_a_moved_artifact_finds_its_geometry_when_told_where_it_is(
    tmp_path: Path, world: World, artifact: Path
) -> None:
    elsewhere = tmp_path / "another-machine" / "copy"
    shutil.copytree(artifact, elsewhere)
    moved_geometry = tmp_path / "another-machine" / "maps" / "geometry"
    shutil.copytree(world.geometry_dir, moved_geometry)
    shutil.rmtree(world.geometry_dir)

    with ContextMapArtifactReader.open(elsewhere, dependency_paths={MAP_ID: moved_geometry}) as r:
        assert r.geometry(_reference(3)).reference == _reference(3)


def test_the_relative_hint_finds_the_geometry_when_the_workspace_moves_as_a_whole(
    tmp_path: Path,
) -> None:
    workspace = make_world(tmp_path / "workspace")
    write_artifact(workspace)
    shutil.move(tmp_path / "workspace", tmp_path / "relocated")

    with ContextMapArtifactReader.open(tmp_path / "relocated" / "out" / "context_map") as reader:
        assert reader.geometry(_reference(2)).reference == _reference(2)
        assert reader.dependency_location(ENTITY_RESOLUTION_ARTIFACT_ID).name == "entity-resolution"


def test_a_dependency_that_is_not_the_recorded_artifact_is_a_mismatch(
    world: World, artifact: Path
) -> None:
    manifest_path = world.geometry_dir / "manifest.json"
    record = json.loads(manifest_path.read_text())
    for entry in record["file_inventory"]:
        if entry["path"] == "outputs/geometry.bin":
            entry["content_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(record))

    with (
        ContextMapArtifactReader.open(artifact) as reader,
        pytest.raises(DependencyMismatchError, match=MAP_ID),
    ):
        reader.geometry(_reference(0))


def test_an_explicit_dependency_path_never_falls_back_to_the_hint(
    tmp_path: Path, artifact: Path
) -> None:
    nowhere = tmp_path / "does-not-exist"

    with (
        ContextMapArtifactReader.open(artifact, dependency_paths={MAP_ID: nowhere}) as reader,
        pytest.raises(MissingDependencyError),
    ):
        reader.geometry(_reference(0))


def test_the_optional_evidence_is_located_and_verified_on_request(
    world: World, artifact: Path
) -> None:
    with ContextMapArtifactReader.open(artifact) as reader:
        assert reader.dependency_location(FUSION_ARTIFACT_ID) == world.fusion_dir.resolve()
        shutil.rmtree(world.fusion_dir)
        with pytest.raises(MissingDependencyError, match=FUSION_ARTIFACT_ID):
            reader.dependency_location(FUSION_ARTIFACT_ID)
        with pytest.raises(RecordNotFoundError, match="nothing-like-this"):
            reader.dependency_location("nothing-like-this")
        assert reader.entity(entity_reference("entity-0001")).entity_id == "entity-0001"


def test_a_directory_without_a_manifest_is_incomplete(tmp_path: Path) -> None:
    (tmp_path / ".tmp-context_map-1234abcd").mkdir()

    with pytest.raises(IncompleteContextMapArtifactError, match=r"manifest\.json"):
        ContextMapArtifactReader.open(tmp_path / ".tmp-context_map-1234abcd")
    with pytest.raises(IncompleteContextMapArtifactError, match="not a directory"):
        ContextMapArtifactReader.open(tmp_path / "missing")


def test_an_unsupported_format_version_is_rejected_with_the_supported_ones(
    artifact: Path,
) -> None:
    _rewrite_manifest(artifact, format_version="9.0.0")

    with pytest.raises(UnsupportedFormatVersionError, match=r"'9\.0\.0'.*0\.1\.0"):
        ContextMapArtifactReader.open(artifact)


def test_an_unsupported_schema_version_is_rejected_before_any_field_is_read(
    artifact: Path,
) -> None:
    _rewrite_manifest(artifact, schema_version="9.0.0")

    with pytest.raises(UnsupportedArtifactSchemaError, match=r"9\.0\.0") as raised:
        ContextMapArtifactReader.open(artifact)
    assert isinstance(raised.value, UnsupportedSchemaVersionError)
    assert isinstance(raised.value, ContextMapArtifactError)


def test_a_manifest_that_is_not_json_is_an_explicit_error(artifact: Path) -> None:
    (artifact / "manifest.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(ContextMapArtifactError, match=r"manifest\.json"):
        ContextMapArtifactReader.open(artifact)


def test_a_tampered_manifest_no_longer_matches_its_identity(artifact: Path) -> None:
    _rewrite_manifest(artifact, entity_count=99)

    with pytest.raises(ArtifactIntegrityError, match="content identity"):
        ContextMapArtifactReader.open(artifact)


def test_a_missing_payload_is_detected_when_the_artifact_opens(artifact: Path) -> None:
    (artifact / "entities/entities.jsonl").unlink()

    with pytest.raises(MissingPayloadError, match=r"entities/entities\.jsonl"):
        ContextMapArtifactReader.open(artifact)


def test_a_truncated_payload_is_detected_when_the_artifact_opens(artifact: Path) -> None:
    path = artifact / "relations/relations.jsonl"
    path.write_bytes(path.read_bytes()[:-20])

    with pytest.raises(ArtifactIntegrityError, match=r"relations/relations\.jsonl.*size"):
        ContextMapArtifactReader.open(artifact)


def test_a_change_of_the_same_size_needs_the_hash_check_to_be_seen(artifact: Path) -> None:
    path = artifact / "map-metadata.json"
    data = bytearray(path.read_bytes())
    data[data.index(b"corridor") + 1] ^= 0x01
    path.write_bytes(bytes(data))

    with ContextMapArtifactReader.open(artifact):
        pass
    with pytest.raises(ArtifactIntegrityError, match=r"map-metadata\.json.*hash"):
        ContextMapArtifactReader.open(artifact, verify_hashes=True)


def test_a_required_file_the_manifest_does_not_inventory_is_never_trusted(artifact: Path) -> None:
    _reseal(artifact, drop=("lineage/lineage.json",))

    with pytest.raises(IncompleteContextMapArtifactError, match=r"lineage/lineage\.json"):
        ContextMapArtifactReader.open(artifact)


def test_a_broken_index_fails_explicitly_when_the_table_is_used(artifact: Path) -> None:
    index = artifact / "indexes/entity-index.jsonl"
    lines = index.read_bytes().split(b"\n")
    index.write_bytes(b"\n".join(reversed(lines[:-1])) + b"\n")
    _reseal(artifact)

    with ContextMapArtifactReader.open(artifact) as reader:
        assert reader.geometry_link().point_count == 1000
        with pytest.raises(BrokenIndexError, match="sorted"):
            reader.entity(entity_reference("entity-0001"))


def test_a_stored_entity_that_is_not_a_valid_entity_is_an_explicit_error(artifact: Path) -> None:
    path = artifact / "entities/entities.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"entity_id"', b'"entity_ix"', 1))
    _reseal(artifact)

    with (
        ContextMapArtifactReader.open(artifact) as reader,
        pytest.raises(ContextMapArtifactError, match="not a valid entity"),
    ):
        reader.entity(entity_reference("entity-0001"))


def test_the_debug_directory_is_never_a_fallback(artifact: Path) -> None:
    (artifact / "debug").mkdir()
    (artifact / "debug" / "entities.jsonl").write_bytes(
        (artifact / "entities/entities.jsonl").read_bytes()
    )
    (artifact / "entities/entities.jsonl").unlink()

    with pytest.raises(MissingPayloadError):
        ContextMapArtifactReader.open(artifact)


def test_reading_never_changes_the_artifact(world: World, artifact: Path) -> None:
    before = tree_snapshot(artifact)
    geometry_before = tree_snapshot(world.geometry_dir)

    with ContextMapArtifactReader.open(artifact, verify_hashes=True) as reader:
        reader.context_map()
        list(reader.entities())
        list(reader.relations())
        reader.relations_for(entity_reference("entity-0001"))
        reader.map_bounds()
        reader.lineage()
        reader.geometry(_reference(4))
        list(reader.geometry_source().iter_geometry())

    assert tree_snapshot(artifact) == before
    assert tree_snapshot(world.geometry_dir) == geometry_before


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_a_read_only_artifact_can_be_read(world: World, artifact: Path) -> None:
    for path in [*artifact.rglob("*"), artifact]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        with ContextMapArtifactReader.open(artifact, verify_hashes=True) as reader:
            assert reader.entity(entity_reference("entity-0001")).entity_id == "entity-0001"
            assert reader.context_map() == make_context_map(world)
    finally:
        for path in [artifact, *artifact.rglob("*")]:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def test_the_reader_has_no_search_query_or_planning_behavior() -> None:
    forbidden = ("search", "query", "find", "plan", "navigate", "infer", "resolve_entity")
    public = [name for name in dir(ContextMapArtifactReader) if not name.startswith("_")]

    assert not [name for name in public if any(word in name for word in forbidden)]


def test_the_reader_needs_no_model_or_robotics_runtime() -> None:
    program = (
        "import sys\n"
        "import contextmap.artifact.serialization.reader\n"
        "blocked = ('torch', 'transformers', 'rclpy', 'rosbags', 'cv2', 'PIL')\n"
        "found = [name for name in blocked if name in sys.modules]\n"
        "assert not found, found\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_a_closed_reader_refuses_to_read(artifact: Path) -> None:
    reader = ContextMapArtifactReader.open(artifact)
    reader.geometry(_reference(0))
    reader.close()
    reader.close()

    with pytest.raises(ContextMapArtifactError, match="closed"):
        reader.geometry(_reference(0))
    with pytest.raises(ContextMapArtifactError, match="closed"):
        reader.entity(entity_reference("entity-0001"))


def test_a_manifest_that_is_not_a_context_map_manifest_is_rejected(artifact: Path) -> None:
    _rewrite_manifest(artifact, artifact_type="geometric_map")

    with pytest.raises(ManifestError, match="artifact_type"):
        ContextMapArtifactReader.open(artifact)


def test_entity_references_are_the_schemas_own_references() -> None:
    reference = entity_reference("entity-0001")

    assert isinstance(reference, ContextEntityReference)
    assert str(reference.context_map_id) == str(CONTEXT_MAP_ID)
