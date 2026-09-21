"""The deterministic, atomic ContextMapArtifact writer (issue #156).

The tests use the real schema, a real GeometricMapArtifact and small real upstream artifacts;
they need no perception, robotics or model runtime, which is part of the acceptance criteria.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from context_map_builders import (
    ENTITY_RESOLUTION_ARTIFACT_ID,
    FUSION_ARTIFACT_ID,
    SEQUENCE_ARTIFACT_ID,
    entity_capabilities,
    metadata,
)
from context_map_builders import context_map as schema_context_map
from context_map_serialization_builders import (
    MAP_ID,
    WRITTEN_AT,
    World,
    make_context_map,
    make_world,
    pinned,
    tree_files,
    write_artifact,
)

from contextmap.artifact import (
    ContextMapArtifactError,
    Requirement,
    context_map_from_record,
    context_map_to_record,
)
from contextmap.artifact.serialization.dependencies import artifact_digest
from contextmap.artifact.serialization.errors import (
    ArtifactExistsError,
    InvalidContentError,
    UpstreamArtifactError,
)
from contextmap.artifact.serialization.layout import (
    CONTRACTUAL_FILES,
    FORMAT_VERSION,
    MANIFEST,
    README,
)
from contextmap.artifact.serialization.manifest import (
    ContextMapArtifactManifest,
    PayloadRole,
    RecordPayload,
    decode_manifest,
    manifest_content_identity,
)
from contextmap.artifact.serialization.tables import RecordTable
from contextmap.ingestion import FrameId
from contextmap.shared import check_file_inventory


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


def _read_manifest(output_dir: Path) -> ContextMapArtifactManifest:
    return decode_manifest(json.loads((output_dir / MANIFEST).read_text(encoding="utf-8")))


def test_the_writer_publishes_exactly_the_documented_layout(world: World) -> None:
    output_dir, _ = write_artifact(world)

    assert set(tree_files(output_dir)) == {MANIFEST, README, *CONTRACTUAL_FILES}
    assert not (output_dir / "debug").exists()


def test_the_manifest_inventories_every_contractual_file_with_matching_hashes(
    world: World,
) -> None:
    output_dir, manifest = write_artifact(world)

    assert {entry.path for entry in manifest.file_inventory} == set(CONTRACTUAL_FILES)
    assert check_file_inventory(output_dir, manifest.file_inventory) == []
    assert _read_manifest(output_dir) == manifest
    for entry in manifest.file_inventory:
        data = (output_dir / entry.path).read_bytes()
        assert entry.size_bytes == len(data)
        assert entry.content_hash == f"sha256:{hashlib.sha256(data).hexdigest()}"


def test_the_manifest_records_schema_format_code_and_configuration_identities(
    world: World,
) -> None:
    context_map = make_context_map(world)
    _, manifest = write_artifact(world, context_map=context_map)

    creation = context_map.metadata.creation
    assert manifest.context_map_id == str(context_map.context_map_id)
    assert manifest.schema_version == context_map.schema_version
    assert manifest.format_version == FORMAT_VERSION
    assert manifest.code_version == creation.code_version
    assert manifest.configuration_fingerprint == creation.configuration_fingerprint
    assert manifest.written_at == WRITTEN_AT
    assert (manifest.entity_count, manifest.relation_count) == (
        len(context_map.entities),
        len(context_map.relations),
    )
    assert (manifest.entity_count, manifest.relation_count) == (3, 10)
    assert manifest.content_identity == manifest_content_identity(manifest)


def test_the_schema_record_is_recoverable_from_the_artifact_files(world: World) -> None:
    context_map = make_context_map(world)
    output_dir, manifest = write_artifact(world, context_map=context_map)
    table = RecordTable(
        output_dir / "entities/entities.jsonl",
        output_dir / "indexes/entity-index.jsonl",
        record_count=manifest.entity_count,
    )
    relations = RecordTable(
        output_dir / "relations/relations.jsonl",
        output_dir / "indexes/relation-index.jsonl",
        record_count=manifest.relation_count,
    )

    rebuilt = context_map_from_record(
        {
            "context_map_id": manifest.context_map_id,
            "schema_version": manifest.schema_version,
            "metadata": json.loads((output_dir / "map-metadata.json").read_text()),
            "geometry_ref": json.loads(
                (output_dir / "geometry/geometry-reference.json").read_text()
            ),
            "entities": [line["record"] for line in table.iter_lines()],
            "relations": [line["record"] for line in relations.iter_lines()],
            "lineage": json.loads((output_dir / "lineage/lineage.json").read_text())[
                "upstream_artifacts"
            ],
        }
    )

    assert rebuilt == context_map


def test_the_tables_are_written_ordered_with_indexes_that_describe_them(world: World) -> None:
    output_dir, manifest = write_artifact(world)

    entities = RecordTable(
        output_dir / "entities/entities.jsonl",
        output_dir / "indexes/entity-index.jsonl",
        record_count=manifest.entity_count,
    )
    relations = RecordTable(
        output_dir / "relations/relations.jsonl",
        output_dir / "indexes/relation-index.jsonl",
        record_count=manifest.relation_count,
    )
    written = make_context_map(world)
    assert entities.keys == tuple(str(item.entity_id) for item in written.entities)
    assert relations.keys == tuple(str(item.relation_id) for item in written.relations)
    line = relations.read(str(written.relations[1].relation_id))
    assert (line["subject"], line["object"]) == (
        str(written.relations[1].subject.entity_id),
        str(written.relations[1].object.entity_id),
    )
    assert line["record"]["predicate"] == written.relations[1].predicate.value
    assert line["record"]["source_relation_id"] == str(written.relations[1].source_relation_id)


def test_the_traversal_index_lists_the_relations_of_every_entity(world: World) -> None:
    output_dir, _ = write_artifact(world)

    lines = [
        json.loads(line)
        for line in (output_dir / "indexes/entity-relation-index.jsonl").read_bytes().splitlines()
    ]
    written = make_context_map(world)
    assert lines == [
        {
            "key": str(item.entity_id),
            "as_subject": sorted(
                str(rel.relation_id)
                for rel in written.relations
                if rel.subject.entity_id == item.entity_id
            ),
            "as_object": sorted(
                str(rel.relation_id)
                for rel in written.relations
                if rel.object.entity_id == item.entity_id
            ),
        }
        for item in written.entities
    ]
    assert any(line["as_subject"] and line["as_object"] for line in lines)


def test_payloads_are_described_with_their_role_counts_and_sources(world: World) -> None:
    _, manifest = write_artifact(world)

    payloads = {payload.path: payload for payload in manifest.payloads}
    assert set(payloads) == {
        "entities/entities.jsonl",
        "relations/relations.jsonl",
        "indexes/entity-index.jsonl",
        "indexes/relation-index.jsonl",
        "indexes/entity-relation-index.jsonl",
    }
    entities = payloads["entities/entities.jsonl"]
    assert isinstance(entities, RecordPayload)
    assert entities.role is PayloadRole.AUTHORITATIVE and entities.record_count == 3
    index = payloads["indexes/entity-index.jsonl"]
    assert index.role is PayloadRole.DERIVED_INDEX
    assert index.derived_from == ("entities/entities.jsonl",)
    assert payloads["indexes/entity-relation-index.jsonl"].derived_from == (
        "entities/entities.jsonl",
        "relations/relations.jsonl",
    )


def test_the_geometry_is_referenced_by_identity_and_never_copied(world: World) -> None:
    output_dir, _ = write_artifact(world)

    reference = json.loads((output_dir / "geometry/geometry-reference.json").read_text())
    assert reference == {"map_id": MAP_ID, "point_count": 1000}
    assert not any(path.endswith(".bin") for path in tree_files(output_dir))
    assert sum(len(data) for data in tree_files(output_dir).values()) < 150_000


def test_dependencies_follow_the_lineage_and_its_structural_kinds(world: World) -> None:
    output_dir, manifest = write_artifact(world)

    by_id = {item.artifact_id: item for item in manifest.dependencies}
    lineage = {item.artifact_id: item for item in make_context_map(world).lineage}
    assert set(by_id) == set(lineage)
    for artifact_id in (MAP_ID, ENTITY_RESOLUTION_ARTIFACT_ID):
        assert by_id[artifact_id].requirement is Requirement.REQUIRED
    assert by_id[FUSION_ARTIFACT_ID].requirement is Requirement.OPTIONAL
    assert by_id[SEQUENCE_ARTIFACT_ID].requirement is Requirement.OPTIONAL
    for artifact_id, record in by_id.items():
        assert record.artifact_type == lineage[artifact_id].kind.value
        assert record.content_identity == lineage[artifact_id].content_identity
    for artifact_id, directory in world.locations.items():
        locator = by_id[artifact_id].locator
        assert locator is not None and not locator.startswith("/")
        assert (output_dir / locator).resolve() == directory.resolve()
        assert by_id[artifact_id].content_identity == artifact_digest(directory)
    assert by_id[SEQUENCE_ARTIFACT_ID].locator is None


def test_the_lineage_document_is_the_schema_lineage(world: World) -> None:
    context_map = make_context_map(world)
    output_dir, _ = write_artifact(world, context_map=context_map)

    lineage = json.loads((output_dir / "lineage/lineage.json").read_text())
    assert lineage == {"upstream_artifacts": context_map_to_record(context_map)["lineage"]}


def test_repeated_writes_are_semantically_equal_with_a_stable_identity(world: World) -> None:
    first_dir, first = write_artifact(world, name="first")
    second_dir, second = write_artifact(
        world, name="second", written_at=datetime(2030, 1, 1, tzinfo=UTC)
    )

    assert first.content_identity == second.content_identity
    assert first.written_at != second.written_at
    first_files, second_files = tree_files(first_dir), tree_files(second_dir)
    assert first_files.keys() == second_files.keys()
    differing = {path for path in first_files if first_files[path] != second_files[path]}
    assert differing <= {MANIFEST}


def test_a_different_map_has_a_different_identity(world: World) -> None:
    _, populated = write_artifact(world, name="populated")
    _, geometry_only = write_artifact(
        world,
        name="geometry-only",
        context_map=make_context_map(world, kind="geometry-only"),
        locations={MAP_ID: world.geometry_dir},
    )

    assert populated.content_identity != geometry_only.content_identity


def test_a_geometry_only_map_is_valid_and_explicit(world: World) -> None:
    output_dir, manifest = write_artifact(
        world,
        context_map=make_context_map(world, kind="geometry-only"),
        locations={MAP_ID: world.geometry_dir},
    )

    assert (manifest.entity_count, manifest.relation_count) == (0, 0)
    for path in CONTRACTUAL_FILES:
        assert (output_dir / path).is_file()
    assert (output_dir / "entities/entities.jsonl").read_bytes() == b""


def test_a_declared_capability_may_be_empty(world: World) -> None:
    declared_only = pinned(
        schema_context_map(metadata=metadata(capabilities=entity_capabilities())), world
    )

    _, manifest = write_artifact(world, context_map=declared_only)

    assert (manifest.entity_count, manifest.relation_count) == (0, 0)


def test_an_existing_artifact_is_never_overwritten(world: World) -> None:
    output_dir, _ = write_artifact(world)
    before = tree_files(output_dir)

    with pytest.raises(ArtifactExistsError, match="already exists"):
        write_artifact(world)

    assert tree_files(output_dir) == before


def test_a_failure_never_leaves_a_partial_artifact_or_temporary_files(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(self: Any, **_: Any) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr("contextmap.shared.run_directory.AtomicRunDirectory.publish", interrupted)
    with pytest.raises(OSError, match="disk went away"):
        write_artifact(world)

    assert not (world.root / "out" / "context_map").exists()
    assert list((world.root / "out").iterdir()) == []


def test_a_crash_leaves_a_directory_that_cannot_be_mistaken_for_an_artifact(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(self: Any, **_: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("contextmap.shared.run_directory.AtomicRunDirectory.publish", crash)
    # Simula um processo morto: a limpeza do diretório temporário não roda.
    monkeypatch.setattr("contextmap.shared.run_directory.shutil.rmtree", lambda *a, **k: None)
    with pytest.raises(KeyboardInterrupt):
        write_artifact(world)

    assert not (world.root / "out" / "context_map").exists()
    leftovers = list((world.root / "out").iterdir())
    assert len(leftovers) == 1 and leftovers[0].name.startswith(".tmp-")
    assert not (leftovers[0] / MANIFEST).exists()


def test_a_structural_dependency_without_a_location_is_refused(world: World) -> None:
    without_resolution = {
        key: value for key, value in world.locations.items() if key != ENTITY_RESOLUTION_ARTIFACT_ID
    }

    with pytest.raises(InvalidContentError, match=ENTITY_RESOLUTION_ARTIFACT_ID):
        write_artifact(world, locations=without_resolution)
    assert not (world.root / "out").exists()


def test_a_location_for_an_artifact_the_map_does_not_cite_is_refused(world: World) -> None:
    with pytest.raises(InvalidContentError, match="not-cited"):
        write_artifact(world, locations={**world.locations, "not-cited": world.fusion_dir})
    assert not (world.root / "out").exists()


def test_the_lineage_must_name_the_artifact_that_is_actually_there(world: World) -> None:
    unpinned = make_context_map(world)
    stale = replace(
        unpinned,
        lineage=tuple(
            replace(item, content_identity="sha256:" + "0" * 64)
            if item.artifact_id == FUSION_ARTIFACT_ID
            else item
            for item in unpinned.lineage
        ),
    )

    with pytest.raises(InvalidContentError, match=r"content identity sha256:0{64}"):
        write_artifact(world, context_map=stale)
    assert not (world.root / "out").exists()


def test_the_geometry_must_be_the_one_the_map_declares(world: World) -> None:
    populated = make_context_map(world)
    wrong_size = replace(populated, geometry_ref=replace(populated.geometry_ref, point_count=1001))
    frame = populated.metadata.frame
    other_frame = replace(
        populated,
        metadata=replace(
            populated.metadata,
            frame=replace(frame, frame_id="odom"),
            bounds=replace(populated.metadata.bounds, frame_id=FrameId("odom")),
        ),
    )

    with pytest.raises(InvalidContentError, match="1001"):
        write_artifact(world, name="a", context_map=wrong_size)
    with pytest.raises(InvalidContentError, match="frame"):
        write_artifact(world, name="b", context_map=other_frame)
    assert not (world.root / "out").exists()


def test_a_damaged_or_missing_geometry_is_refused_before_anything_is_written(
    world: World,
) -> None:
    payload = world.geometry_dir / "outputs/geometry.bin"
    original = payload.read_bytes()
    payload.write_bytes(b"\xff" + original[1:])
    with pytest.raises(UpstreamArtifactError, match=r"geometry\.bin"):
        write_artifact(world, name="a")
    payload.write_bytes(original)
    (world.geometry_dir / MANIFEST).unlink()
    with pytest.raises(UpstreamArtifactError, match="manifest"):
        write_artifact(world, name="b")
    assert not (world.root / "out").exists()


def test_evidence_that_is_given_must_match_its_own_inventory(world: World) -> None:
    (world.fusion_dir / "outputs/summary.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(UpstreamArtifactError, match=r"summary\.json"):
        write_artifact(world)
    assert not (world.root / "out").exists()


def test_nothing_machine_specific_is_written_into_the_artifact(world: World) -> None:
    output_dir, _ = write_artifact(world)

    for path, data in tree_files(output_dir).items():
        text = data.decode("utf-8")
        assert str(world.root) not in text, path
        assert os.path.expanduser("~") not in text, path


def test_the_readme_is_deterministic_and_carries_no_time_or_path(world: World) -> None:
    first_dir, _ = write_artifact(world, name="first")
    second_dir, _ = write_artifact(
        world, name="second", written_at=datetime(2031, 5, 5, tzinfo=UTC)
    )

    readme = (first_dir / README).read_text(encoding="utf-8")
    assert readme == (second_dir / README).read_text(encoding="utf-8")
    assert "context-map--corridor-02--0001" in readme
    assert "2026" not in readme


def test_the_writer_needs_no_model_or_robotics_runtime() -> None:
    program = (
        "import sys\n"
        "import contextmap.artifact.serialization.writer\n"
        "blocked = ('torch', 'transformers', 'rclpy', 'rosbags', 'cv2', 'PIL')\n"
        "found = [name for name in blocked if name in sys.modules]\n"
        "assert not found, found\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_every_error_of_the_writer_belongs_to_the_artifact_error_family() -> None:
    for error in (ArtifactExistsError, InvalidContentError, UpstreamArtifactError):
        assert issubclass(error, ContextMapArtifactError)
