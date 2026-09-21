"""The lightweight ContextMapArtifact reader (issue #157).

It opens an artifact from its own directory, reads entities, relations and metadata lazily,
resolves geometry through the real GeometricMapArtifactReader and never mutates anything. Every
damaged or unsupported input is an explicit, typed error.
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
from context_map_serialization_builders import (
    default_entities,
    default_relations,
    entity,
    make_context_map,
    make_evidence,
    make_geometry,
    relation,
    tree_snapshot,
    write_artifact,
)

from contextmap.artifact import (
    ArtifactIntegrityError,
    ContextMapArtifactError,
    ContextMapArtifactReader,
    DependencyMismatchError,
    EntityEntry,
    IncompleteContextMapArtifactError,
    MissingDependencyError,
    RelationEntry,
    UnresolvedReferenceError,
    UnsupportedArtifactSchemaError,
    UnsupportedFormatVersionError,
    UnsupportedSchemaVersionError,
)
from contextmap.artifact.serialization.errors import (
    BrokenIndexError,
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

MAP = MapId("corridor-02--run-0001")


@pytest.fixture
def geometry_dir(tmp_path: Path) -> Path:
    return make_geometry(tmp_path)


@pytest.fixture
def artifact(tmp_path: Path, geometry_dir: Path) -> Path:
    return write_artifact(tmp_path, geometry_dir)[0]


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


def test_the_reader_returns_the_map_that_was_written(
    tmp_path: Path, geometry_dir: Path, reader: ContextMapArtifactReader
) -> None:
    assert reader.context_map() == make_context_map()
    assert reader.metadata() == make_context_map().metadata
    assert reader.manifest.entity_count == 3


def test_map_bounds_are_the_declared_extent_in_the_map_frame(
    reader: ContextMapArtifactReader,
) -> None:
    bounds = reader.map_bounds()

    assert bounds == make_context_map().metadata.bounds
    assert bounds.frame_id == make_context_map().metadata.frame.frame_id


def test_entities_round_trip_and_stream_in_key_order(reader: ContextMapArtifactReader) -> None:
    assert reader.entity_keys() == ("entity-a", "entity-b", "entity-c")
    assert list(reader.entities()) == sorted(default_entities(), key=lambda entry: entry.key)
    assert reader.entity("entity-b") == EntityEntry(
        key="entity-b", record={"entity_id": "entity-b", "label": "label-of-entity-b"}
    )


def test_relations_round_trip_with_their_endpoints(reader: ContextMapArtifactReader) -> None:
    assert reader.relation_keys() == ("relation-1", "relation-2")
    assert list(reader.relations()) == sorted(default_relations(), key=lambda entry: entry.key)
    assert reader.relation("relation-2") == RelationEntry(
        key="relation-2",
        subject_key="entity-c",
        object_key="entity-a",
        record={
            "relation_id": "relation-2",
            "predicate": "next_to",
            "subject": "entity-c",
            "object": "entity-a",
        },
    )


def test_relations_for_an_entity_lists_those_it_takes_part_in(
    reader: ContextMapArtifactReader,
) -> None:
    assert [item.key for item in reader.relations_for("entity-a")] == ["relation-1", "relation-2"]
    assert [item.key for item in reader.relations_for("entity-b")] == ["relation-1"]
    assert [item.key for item in reader.relations_for("entity-c")] == ["relation-2"]


def test_an_entity_without_relations_has_an_empty_traversal(
    tmp_path: Path, geometry_dir: Path
) -> None:
    artifact, _ = write_artifact(
        tmp_path,
        geometry_dir,
        entities=(*default_entities(), entity("entity-lonely")),
        relations=(relation("relation-1", "entity-a", "entity-b"),),
    )
    with ContextMapArtifactReader.open(artifact) as opened:
        assert opened.relations_for("entity-lonely") == ()


def test_unknown_entities_and_relations_are_explicit_errors(
    reader: ContextMapArtifactReader,
) -> None:
    with pytest.raises(RecordNotFoundError, match="entity-zzz"):
        reader.entity("entity-zzz")
    with pytest.raises(RecordNotFoundError, match="relation-9"):
        reader.relation("relation-9")
    with pytest.raises(RecordNotFoundError, match="entity-zzz"):
        reader.relations_for("entity-zzz")


def test_geometry_resolves_through_the_geometric_map_reader(
    geometry_dir: Path, reader: ContextMapArtifactReader
) -> None:
    with GeometricMapArtifactReader(geometry_dir) as direct:
        expected = direct.geometry().get(_reference(5))

    point = reader.geometry(_reference(5))

    assert point == expected
    assert point.reference == _reference(5)
    assert reader.geometry_source().geometric_map.map_id == MAP
    assert sum(1 for _ in reader.geometry_source().iter_geometry()) == 24


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
        reader.entity("entity-a")
        list(reader.relations())
        assert opened == []
        reader.geometry(_reference(0))
        reader.geometry(_reference(1))
        assert len(opened) == 1


def test_references_are_validated_against_the_map_without_loading_geometry(
    artifact: Path, reader: ContextMapArtifactReader
) -> None:
    reader.validate_reference(_reference(0))
    reader.validate_reference(_reference(23))

    with pytest.raises(UnresolvedReferenceError, match="another-map"):
        reader.validate_reference(_reference(0, MapId("another-map")))
    with pytest.raises(UnresolvedReferenceError, match="24"):
        reader.validate_reference(_reference(24))
    with pytest.raises(UnresolvedReferenceError, match="canonical"):
        reader.validate_reference(
            GeometryReference(map_id=MAP, geometry_id=geometry_id_for(map_id=MAP, index=1) + "x")  # type: ignore[arg-type]
        )


def test_a_reference_outside_the_geometry_is_not_resolved(
    reader: ContextMapArtifactReader,
) -> None:
    with pytest.raises(UnresolvedReferenceError):
        reader.geometry(_reference(99))


def test_a_moved_artifact_opens_and_reads_everything_that_needs_no_geometry(
    tmp_path: Path, artifact: Path
) -> None:
    elsewhere = tmp_path / "another-machine" / "copy"
    shutil.copytree(artifact, elsewhere)
    shutil.rmtree(tmp_path / "geometry-workspace")

    with ContextMapArtifactReader.open(elsewhere) as reader:
        assert reader.context_map() == make_context_map()
        assert reader.entity("entity-a").key == "entity-a"
        assert [item.key for item in reader.relations_for("entity-a")] == [
            "relation-1",
            "relation-2",
        ]
        with pytest.raises(MissingDependencyError, match="corridor-02--run-0001"):
            reader.geometry(_reference(0))


def test_a_moved_artifact_finds_its_geometry_when_told_where_it_is(
    tmp_path: Path, artifact: Path, geometry_dir: Path
) -> None:
    elsewhere = tmp_path / "another-machine" / "copy"
    shutil.copytree(artifact, elsewhere)
    moved_geometry = tmp_path / "another-machine" / "maps" / "geometry"
    shutil.copytree(geometry_dir, moved_geometry)
    shutil.rmtree(tmp_path / "geometry-workspace")

    with ContextMapArtifactReader.open(
        elsewhere, dependency_paths={"corridor-02--run-0001": moved_geometry}
    ) as reader:
        assert reader.geometry(_reference(3)).reference == _reference(3)


def test_the_relative_hint_finds_the_geometry_when_the_workspace_moves_as_a_whole(
    tmp_path: Path, geometry_dir: Path
) -> None:
    workspace = tmp_path / "workspace"
    shutil.move(tmp_path / "geometry-workspace", workspace / "geometry-workspace")
    geometry_moved = (
        workspace / "geometry-workspace" / geometry_dir.relative_to(tmp_path / "geometry-workspace")
    )
    artifact, _ = write_artifact(workspace, geometry_moved)
    relocated = tmp_path / "relocated"
    shutil.move(workspace, relocated)

    with ContextMapArtifactReader.open(relocated / "out" / "context_map") as reader:
        assert reader.geometry(_reference(2)).reference == _reference(2)
    assert artifact.name == "context_map"


def test_a_dependency_that_is_not_the_recorded_artifact_is_a_mismatch(
    tmp_path: Path, artifact: Path, geometry_dir: Path
) -> None:
    manifest_path = geometry_dir / "manifest.json"
    record = json.loads(manifest_path.read_text())
    for entry in record["file_inventory"]:
        if entry["path"] == "outputs/geometry.bin":
            entry["content_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(record))

    with (
        ContextMapArtifactReader.open(artifact) as reader,
        pytest.raises(DependencyMismatchError, match="corridor-02--run-0001"),
    ):
        reader.geometry(_reference(0))


def test_an_explicit_dependency_path_never_falls_back_to_the_hint(
    tmp_path: Path, artifact: Path
) -> None:
    nowhere = tmp_path / "does-not-exist"

    with (
        ContextMapArtifactReader.open(
            artifact, dependency_paths={"corridor-02--run-0001": nowhere}
        ) as reader,
        pytest.raises(MissingDependencyError),
    ):
        reader.geometry(_reference(0))


def test_the_optional_evidence_is_located_and_verified_on_request(
    tmp_path: Path, geometry_dir: Path
) -> None:
    evidence = make_evidence(tmp_path)
    artifact, _ = write_artifact(tmp_path, geometry_dir, evidence=(evidence,))

    with ContextMapArtifactReader.open(artifact) as reader:
        assert (
            reader.dependency_location("semantic_fusion_run", "run-0003")
            == evidence.location.resolve()
        )
        shutil.rmtree(evidence.location)
        with pytest.raises(MissingDependencyError, match="run-0003"):
            reader.dependency_location("semantic_fusion_run", "run-0003")
        assert reader.entity("entity-a").key == "entity-a"


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
        assert reader.metadata() == make_context_map().metadata
        with pytest.raises(BrokenIndexError, match="sorted"):
            reader.entity("entity-a")


def test_the_debug_directory_is_never_a_fallback(artifact: Path) -> None:
    (artifact / "debug").mkdir()
    (artifact / "debug" / "entities.jsonl").write_bytes(
        (artifact / "entities/entities.jsonl").read_bytes()
    )
    (artifact / "entities/entities.jsonl").unlink()

    with pytest.raises(MissingPayloadError):
        ContextMapArtifactReader.open(artifact)


def test_reading_never_changes_the_artifact(
    tmp_path: Path, artifact: Path, geometry_dir: Path
) -> None:
    before = tree_snapshot(artifact)
    geometry_before = tree_snapshot(geometry_dir)

    with ContextMapArtifactReader.open(artifact, verify_hashes=True) as reader:
        reader.context_map()
        list(reader.entities())
        list(reader.relations())
        reader.relations_for("entity-a")
        reader.map_bounds()
        reader.geometry(_reference(4))
        list(reader.geometry_source().iter_geometry())

    assert tree_snapshot(artifact) == before
    assert tree_snapshot(geometry_dir) == geometry_before


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_a_read_only_artifact_can_be_read(tmp_path: Path, artifact: Path) -> None:
    for path in [*artifact.rglob("*"), artifact]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        with ContextMapArtifactReader.open(artifact, verify_hashes=True) as reader:
            assert reader.entity("entity-a").key == "entity-a"
            assert reader.context_map() == make_context_map()
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
        reader.entity("entity-a")
