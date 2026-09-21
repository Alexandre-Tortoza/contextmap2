"""The deterministic, atomic ContextMapArtifact writer (issue #156).

The tests use the real schema, a real GeometricMapArtifact and small upstream artifacts; they
need no perception, robotics or model runtime, which is part of the acceptance criteria.
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
from context_map_serialization_builders import (
    WRITTEN_AT,
    declared_without,
    default_entities,
    default_relations,
    entity,
    make_context_map,
    make_evidence,
    make_geometry,
    relation,
    tree_files,
    write_artifact,
)

from contextmap.artifact import (
    MapCapability,
    Requirement,
    context_map_from_record,
    inventory_digest,
)
from contextmap.artifact.serialization.dependencies import EvidenceArtifact, read_inventory
from contextmap.artifact.serialization.errors import (
    ArtifactExistsError,
    ContextMapArtifactError,
    InvalidContentError,
    RecordTableError,
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
from contextmap.geometric_mapping import GeometricMapArtifactReader, MapId
from contextmap.ingestion import FrameId
from contextmap.shared import check_file_inventory


@pytest.fixture
def geometry_dir(tmp_path: Path) -> Path:
    return make_geometry(tmp_path)


@pytest.fixture
def evidence(tmp_path: Path) -> EvidenceArtifact:
    return make_evidence(tmp_path)


def _read_manifest(output_dir: Path) -> ContextMapArtifactManifest:
    return decode_manifest(json.loads((output_dir / MANIFEST).read_text(encoding="utf-8")))


def test_the_writer_publishes_exactly_the_documented_layout(
    tmp_path: Path, geometry_dir: Path
) -> None:
    output_dir, _ = write_artifact(tmp_path, geometry_dir)

    assert set(tree_files(output_dir)) == {MANIFEST, README, *CONTRACTUAL_FILES}
    assert not (output_dir / "debug").exists()


def test_the_manifest_inventories_every_contractual_file_with_matching_hashes(
    tmp_path: Path, geometry_dir: Path
) -> None:
    output_dir, manifest = write_artifact(tmp_path, geometry_dir)

    assert {entry.path for entry in manifest.file_inventory} == set(CONTRACTUAL_FILES)
    assert check_file_inventory(output_dir, manifest.file_inventory) == []
    assert _read_manifest(output_dir) == manifest
    for entry in manifest.file_inventory:
        data = (output_dir / entry.path).read_bytes()
        assert entry.size_bytes == len(data)
        assert entry.content_hash == f"sha256:{hashlib.sha256(data).hexdigest()}"


def test_the_manifest_records_schema_format_code_and_configuration_identities(
    tmp_path: Path, geometry_dir: Path
) -> None:
    context_map = make_context_map()
    _, manifest = write_artifact(tmp_path, geometry_dir, context_map=context_map)

    creation = context_map.metadata.creation
    assert manifest.context_map_id == str(context_map.context_map_id)
    assert manifest.schema_version == context_map.schema_version
    assert manifest.format_version == FORMAT_VERSION
    assert manifest.code_version == creation.code_version
    assert manifest.configuration_fingerprint == creation.configuration_fingerprint
    assert manifest.written_at == WRITTEN_AT
    assert (manifest.entity_count, manifest.relation_count) == (3, 2)
    assert manifest.content_identity == manifest_content_identity(manifest)


def test_the_schema_record_is_recoverable_from_the_artifacttree_files(
    tmp_path: Path, geometry_dir: Path
) -> None:
    context_map = make_context_map()
    output_dir, manifest = write_artifact(tmp_path, geometry_dir, context_map=context_map)

    rebuilt = context_map_from_record(
        {
            "context_map_id": manifest.context_map_id,
            "schema_version": manifest.schema_version,
            "metadata": json.loads((output_dir / "map-metadata.json").read_text()),
            "geometry_ref": json.loads(
                (output_dir / "geometry/geometry-reference.json").read_text()
            ),
        }
    )

    assert rebuilt == context_map


def test_the_tables_are_written_ordered_with_indexes_that_describe_them(
    tmp_path: Path, geometry_dir: Path
) -> None:
    output_dir, manifest = write_artifact(tmp_path, geometry_dir)

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
    assert entities.keys == ("entity-a", "entity-b", "entity-c")
    assert relations.keys == ("relation-1", "relation-2")
    assert entities.read("entity-b")["record"]["label"] == "label-of-entity-b"
    assert relations.read("relation-2")["subject"] == "entity-c"
    assert relations.read("relation-2")["object"] == "entity-a"


def test_the_traversal_index_lists_the_relations_of_every_entity(
    tmp_path: Path, geometry_dir: Path
) -> None:
    output_dir, _ = write_artifact(tmp_path, geometry_dir)

    lines = [
        json.loads(line)
        for line in (output_dir / "indexes/entity-relation-index.jsonl").read_bytes().splitlines()
    ]
    assert lines == [
        {"key": "entity-a", "as_subject": ["relation-1"], "as_object": ["relation-2"]},
        {"key": "entity-b", "as_subject": [], "as_object": ["relation-1"]},
        {"key": "entity-c", "as_subject": ["relation-2"], "as_object": []},
    ]


def test_payloads_are_described_with_their_role_counts_and_sources(
    tmp_path: Path, geometry_dir: Path
) -> None:
    _, manifest = write_artifact(tmp_path, geometry_dir)

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


def test_the_geometry_is_referenced_by_identity_and_never_copied(
    tmp_path: Path, geometry_dir: Path
) -> None:
    output_dir, _ = write_artifact(tmp_path, geometry_dir)

    reference = json.loads((output_dir / "geometry/geometry-reference.json").read_text())
    assert reference == {"map_id": "corridor-02--run-0001", "point_count": 24}
    assert not any(path.endswith(".bin") for path in tree_files(output_dir))
    assert sum(len(data) for data in tree_files(output_dir).values()) < 20_000


def test_dependencies_are_pinned_by_content_and_located_by_a_relative_hint(
    tmp_path: Path, geometry_dir: Path, evidence: EvidenceArtifact
) -> None:
    output_dir, manifest = write_artifact(tmp_path, geometry_dir, evidence=(evidence,))

    by_type = {dependency.artifact_type: dependency for dependency in manifest.dependencies}
    geometry, fusion = by_type["geometric_map"], by_type["semantic_fusion_run"]
    assert geometry.requirement is Requirement.REQUIRED
    assert fusion.requirement is Requirement.OPTIONAL
    assert geometry.artifact_id == "corridor-02--run-0001"
    assert fusion.artifact_id == "run-0003"
    with GeometricMapArtifactReader(geometry_dir) as reader:
        assert geometry.content_identity == inventory_digest(reader.manifest.file_inventory)
    assert fusion.content_identity == inventory_digest(read_inventory(evidence.location))
    for record, directory in ((geometry, geometry_dir), (fusion, evidence.location)):
        assert record.locator is not None and not record.locator.startswith("/")
        assert (output_dir / record.locator).resolve() == directory.resolve()


def test_the_lineage_lists_the_exact_upstream_identities_without_paths(
    tmp_path: Path, geometry_dir: Path, evidence: EvidenceArtifact
) -> None:
    output_dir, manifest = write_artifact(tmp_path, geometry_dir, evidence=(evidence,))

    lineage = json.loads((output_dir / "lineage/lineage.json").read_text())
    assert lineage == {
        "upstream_artifacts": [
            {
                "artifact_type": item.artifact_type,
                "artifact_id": item.artifact_id,
                "content_identity": item.content_identity,
                "requirement": item.requirement.value,
            }
            for item in manifest.dependencies
        ]
    }


def test_repeated_writes_are_semantically_equal_with_a_stable_identity(
    tmp_path: Path, geometry_dir: Path
) -> None:
    first_dir, first = write_artifact(tmp_path, geometry_dir, name="first")
    second_dir, second = write_artifact(
        tmp_path, geometry_dir, name="second", written_at=datetime(2030, 1, 1, tzinfo=UTC)
    )

    assert first.content_identity == second.content_identity
    assert first.written_at != second.written_at
    first_files, second_files = tree_files(first_dir), tree_files(second_dir)
    assert first_files.keys() == second_files.keys()
    differing = {path for path in first_files if first_files[path] != second_files[path]}
    assert differing <= {MANIFEST}


def test_the_artifact_does_not_depend_on_the_order_the_parts_arrive_in(
    tmp_path: Path, geometry_dir: Path
) -> None:
    first_dir, first = write_artifact(tmp_path, geometry_dir, name="ordered")
    second_dir, second = write_artifact(
        tmp_path,
        geometry_dir,
        name="shuffled",
        entities=tuple(reversed(default_entities())),
        relations=tuple(reversed(default_relations())),
    )

    assert first.content_identity == second.content_identity
    for path in CONTRACTUAL_FILES:
        assert (first_dir / path).read_bytes() == (second_dir / path).read_bytes()


def test_a_different_map_has_a_different_identity(tmp_path: Path, geometry_dir: Path) -> None:
    _, first = write_artifact(tmp_path, geometry_dir, name="first")
    _, second = write_artifact(
        tmp_path,
        geometry_dir,
        name="second",
        entities=(*default_entities(), entity("entity-d")),
    )

    assert first.content_identity != second.content_identity


def test_a_map_without_entities_or_relations_is_valid_and_explicit(
    tmp_path: Path, geometry_dir: Path
) -> None:
    output_dir, manifest = write_artifact(
        tmp_path,
        geometry_dir,
        context_map=make_context_map(entities=False, relations=False),
        entities=(),
        relations=(),
    )

    assert (manifest.entity_count, manifest.relation_count) == (0, 0)
    for path in CONTRACTUAL_FILES:
        assert (output_dir / path).is_file()
    assert (output_dir / "entities/entities.jsonl").read_bytes() == b""


def test_a_declared_capability_may_be_empty(tmp_path: Path, geometry_dir: Path) -> None:
    _, manifest = write_artifact(tmp_path, geometry_dir, entities=(), relations=())

    assert (manifest.entity_count, manifest.relation_count) == (0, 0)


def test_an_existing_artifact_is_never_overwritten(tmp_path: Path, geometry_dir: Path) -> None:
    output_dir, _ = write_artifact(tmp_path, geometry_dir)
    before = tree_files(output_dir)

    with pytest.raises(ArtifactExistsError, match="already exists"):
        write_artifact(tmp_path, geometry_dir)

    assert tree_files(output_dir) == before


def test_a_failure_never_leaves_a_partial_artifact_or_temporarytree_files(
    tmp_path: Path, geometry_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(self: Any, **_: Any) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr("contextmap.shared.run_directory.AtomicRunDirectory.publish", interrupted)
    with pytest.raises(OSError, match="disk went away"):
        write_artifact(tmp_path, geometry_dir)

    assert not (tmp_path / "out" / "context_map").exists()
    assert list((tmp_path / "out").iterdir()) == []


def test_a_crash_leaves_a_directory_that_cannot_be_mistaken_for_an_artifact(
    tmp_path: Path, geometry_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(self: Any, **_: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("contextmap.shared.run_directory.AtomicRunDirectory.publish", crash)
    # Simula um processo morto: a limpeza do diretório temporário não roda.
    monkeypatch.setattr("contextmap.shared.run_directory.shutil.rmtree", lambda *a, **k: None)
    with pytest.raises(KeyboardInterrupt):
        write_artifact(tmp_path, geometry_dir)

    assert not (tmp_path / "out" / "context_map").exists()
    leftovers = list((tmp_path / "out").iterdir())
    assert len(leftovers) == 1 and leftovers[0].name.startswith(".tmp-")
    assert not (leftovers[0] / MANIFEST).exists()


def test_a_relation_with_an_unknown_endpoint_is_refused_not_dropped(
    tmp_path: Path, geometry_dir: Path
) -> None:
    with pytest.raises(InvalidContentError, match=r"relation-1.*'entity-zzz'"):
        write_artifact(
            tmp_path,
            geometry_dir,
            relations=(relation("relation-1", "entity-a", "entity-zzz"),),
        )

    assert not (tmp_path / "out").exists()


def test_duplicate_entity_or_relation_keys_are_refused(tmp_path: Path, geometry_dir: Path) -> None:
    with pytest.raises(RecordTableError, match="duplicate key 'entity-a'"):
        write_artifact(
            tmp_path,
            geometry_dir,
            name="a",
            entities=(entity("entity-a"), entity("entity-a")),
            relations=(),
        )
    with pytest.raises(RecordTableError, match="duplicate key 'relation-1'"):
        write_artifact(
            tmp_path,
            geometry_dir,
            name="b",
            relations=(
                relation("relation-1", "entity-a", "entity-b"),
                relation("relation-1", "entity-b", "entity-a"),
            ),
        )
    assert not (tmp_path / "out").exists()


def test_a_record_that_json_cannot_represent_is_refused_naming_it(
    tmp_path: Path, geometry_dir: Path
) -> None:
    with pytest.raises(RecordTableError, match="entity-a"):
        write_artifact(
            tmp_path,
            geometry_dir,
            entities=(entity("entity-a", score=float("nan")), entity("entity-b")),
            relations=(),
        )
    assert not (tmp_path / "out").exists()


def test_content_the_map_does_not_declare_is_refused(tmp_path: Path, geometry_dir: Path) -> None:
    no_relations = declared_without(make_context_map(), MapCapability.RELATIONS)
    no_content = make_context_map(entities=False, relations=False)

    with pytest.raises(InvalidContentError, match="relations capability"):
        write_artifact(tmp_path, geometry_dir, name="a", context_map=no_relations)
    with pytest.raises(InvalidContentError, match="entities capability"):
        write_artifact(tmp_path, geometry_dir, name="b", context_map=no_content, relations=())
    assert not (tmp_path / "out").exists()


def test_the_geometry_must_be_the_one_the_map_names(tmp_path: Path, geometry_dir: Path) -> None:
    other_map = replace(
        make_context_map(),
        geometry_ref=replace(make_context_map().geometry_ref, map_id=MapId("another-map")),
    )
    wrong_size = replace(
        make_context_map(),
        geometry_ref=replace(make_context_map().geometry_ref, point_count=25),
    )
    metadata = make_context_map().metadata
    other_frame = replace(
        make_context_map(),
        metadata=replace(
            metadata,
            frame=replace(metadata.frame, frame_id="odom"),
            bounds=replace(metadata.bounds, frame_id=FrameId("odom")),
        ),
    )

    with pytest.raises(InvalidContentError, match="another-map"):
        write_artifact(tmp_path, geometry_dir, name="a", context_map=other_map)
    with pytest.raises(InvalidContentError, match="25"):
        write_artifact(tmp_path, geometry_dir, name="b", context_map=wrong_size)
    with pytest.raises(InvalidContentError, match="frame"):
        write_artifact(tmp_path, geometry_dir, name="c", context_map=other_frame)
    assert not (tmp_path / "out").exists()


def test_a_damaged_or_missing_geometry_is_refused_before_anything_is_written(
    tmp_path: Path, geometry_dir: Path
) -> None:
    payload = geometry_dir / "outputs/geometry.bin"
    original = payload.read_bytes()
    payload.write_bytes(b"\xff" + original[1:])
    with pytest.raises(UpstreamArtifactError, match=r"geometry\.bin"):
        write_artifact(tmp_path, geometry_dir, name="a")
    payload.write_bytes(original)
    (geometry_dir / MANIFEST).unlink()
    with pytest.raises(UpstreamArtifactError, match="cannot be opened"):
        write_artifact(tmp_path, geometry_dir, name="b")
    assert not (tmp_path / "out").exists()


def test_evidence_that_is_damaged_repeated_or_a_geometric_map_is_refused(
    tmp_path: Path, geometry_dir: Path, evidence: EvidenceArtifact
) -> None:
    with pytest.raises(InvalidContentError, match="duplicate"):
        write_artifact(tmp_path, geometry_dir, name="a", evidence=(evidence, evidence))
    with pytest.raises(InvalidContentError, match="geometry_dir"):
        write_artifact(
            tmp_path,
            geometry_dir,
            name="b",
            evidence=(replace(evidence, artifact_type="geometric_map"),),
        )
    payload = evidence.location / "outputs/summary.json"
    payload.write_text("tampered", encoding="utf-8")
    with pytest.raises(UpstreamArtifactError, match=r"summary\.json"):
        write_artifact(tmp_path, geometry_dir, name="c", evidence=(evidence,))
    assert not (tmp_path / "out").exists()


def test_a_required_evidence_dependency_is_recorded_as_required(
    tmp_path: Path, geometry_dir: Path
) -> None:
    required = make_evidence(tmp_path, requirement=Requirement.REQUIRED)

    _, manifest = write_artifact(tmp_path, geometry_dir, evidence=(required,))

    recorded = {item.artifact_type: item.requirement for item in manifest.dependencies}
    assert recorded["semantic_fusion_run"] is Requirement.REQUIRED


def test_nothing_machine_specific_is_written_into_the_artifact(
    tmp_path: Path, geometry_dir: Path
) -> None:
    output_dir, _ = write_artifact(tmp_path, geometry_dir)

    for path, data in tree_files(output_dir).items():
        text = data.decode("utf-8")
        assert str(tmp_path) not in text, path
        assert os.path.expanduser("~") not in text, path


def test_the_readme_is_deterministic_and_carries_no_time_or_path(
    tmp_path: Path, geometry_dir: Path
) -> None:
    first_dir, _ = write_artifact(tmp_path, geometry_dir, name="first")
    second_dir, _ = write_artifact(
        tmp_path, geometry_dir, name="second", written_at=datetime(2031, 5, 5, tzinfo=UTC)
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
