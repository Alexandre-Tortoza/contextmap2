"""Round trip, corruption, portability and version handling of the ContextMapArtifact (issue #160).

The artifact is the product boundary of Solution 1: serialization must preserve the semantic state
of the map exactly and fail safely under corruption, an interrupted write or an unsupported
version. Everything here runs on the base install: no perception, robotics or model stack.
"""

import json
import pathlib
import shutil
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from context_map_builders import (
    GEOMETRIC_MAP_ID,
    entity,
    entity_capabilities,
    entity_reference,
    metadata,
    populated_map,
    relation,
)
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
    CONTEXT_MAP_SCHEMA_VERSION,
    ClosurePolicy,
    ContextMapArtifactError,
    ContextMapArtifactReader,
    GeometricMapLink,
    Severity,
    UnsupportedArtifactSchemaError,
    UnsupportedFormatVersionError,
    ValidationLevel,
    ValidationStatus,
    context_map_to_record,
    export_bundle,
    validate_context_map_artifact,
    verify_bundle,
)
from contextmap.artifact.serialization.layout import (
    CONTRACTUAL_FILES,
    FORMAT_VERSION,
    MANIFEST,
    SUPPORTED_FORMAT_VERSIONS,
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
from contextmap.spatial_relations import RelationPredicate

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "context_map_artifact" / "v0.1.0"
FIXTURE_CONTENT_IDENTITY = "sha256:a2be8fdd3668bb05f31639154db31e636783bca282a638ee795da41c5cc43f40"


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


def _reseal(artifact: Path, **changes: Any) -> None:
    manifest = decode_manifest(json.loads((artifact / "manifest.json").read_text()))
    fields: dict[str, Any] = {
        "context_map_id": manifest.context_map_id,
        "schema_version": manifest.schema_version,
        "written_at": manifest.written_at,
        "code_version": manifest.code_version,
        "configuration_fingerprint": manifest.configuration_fingerprint,
        "entity_count": manifest.entity_count,
        "relation_count": manifest.relation_count,
        "payloads": manifest.payloads,
        "dependencies": manifest.dependencies,
        **changes,
    }
    resealed = create_manifest(
        **fields,
        file_inventory=[
            file_entry(entry.path, (artifact / entry.path).read_bytes())
            for entry in manifest.file_inventory
        ],
    )
    (artifact / "manifest.json").write_text(json.dumps(encode_manifest(resealed)))


def _errors(artifact: Path, *, level: ValidationLevel = ValidationLevel.FULL) -> set[str]:
    report = validate_context_map_artifact(artifact, level=level)
    return {item.code for item in report.findings if item.severity is Severity.ERROR}


# --- round trip ---------------------------------------------------------------------------------


def test_write_validate_close_reopen_and_compare_the_canonical_state(world: World) -> None:
    written = make_context_map(world)
    artifact, manifest = write_artifact(world, context_map=written)
    assert validate_context_map_artifact(artifact).status is ValidationStatus.VERIFIED

    with ContextMapArtifactReader.open(artifact, verify_hashes=True) as first:
        first.context_map()
        for item in first.entities():
            for reference in item.geometry_refs:
                first.geometry(reference)
    with ContextMapArtifactReader.open(artifact) as reopened:
        assert reopened.manifest == manifest
        assert reopened.context_map() == written
        assert context_map_to_record(reopened.context_map()) == context_map_to_record(written)
        assert tuple(reopened.entities()) == written.entities
        assert tuple(reopened.relations()) == written.relations
        assert reopened.lineage() == written.lineage
        with GeometricMapArtifactReader(world.geometry_dir) as direct:
            for item in written.entities:
                for reference in item.geometry_refs:
                    assert reopened.geometry(reference) == direct.geometry().get(reference)


def test_uncertainty_and_provenance_survive_the_round_trip(world: World) -> None:
    written = make_context_map(world)
    artifact, _ = write_artifact(world, context_map=written)

    with ContextMapArtifactReader.open(artifact) as reader:
        ambiguous = reader.entity(entity_reference("entity-0002"))
        insufficient = reader.entity(entity_reference("entity-0003"))
        unresolved = reader.relation("relation-0002")

    original = {str(item.entity_id): item for item in written.entities}
    assert ambiguous.semantic_state == original["entity-0002"].semantic_state
    assert len(ambiguous.semantic_state.hypotheses) == 2
    assert insufficient.semantic_state.hypotheses == ()
    assert insufficient.origin == original["entity-0003"].origin
    assert unresolved.state is written.relations[1].state
    assert unresolved.origin == written.relations[1].origin


def test_the_round_trip_is_repeatable_across_repeated_writes(world: World) -> None:
    written = make_context_map(world)
    first, _ = write_artifact(world, name="first", context_map=written)
    second, _ = write_artifact(
        world,
        name="second",
        context_map=written,
        written_at=datetime(2032, 3, 4, tzinfo=UTC),
    )

    with (
        ContextMapArtifactReader.open(first) as one,
        ContextMapArtifactReader.open(second) as two,
    ):
        assert one.context_map() == two.context_map() == written
        assert one.manifest.content_identity == two.manifest.content_identity


# --- lazy access to large payloads ---------------------------------------------------------------


class _CountingFile:
    """Wraps a file object and records how many bytes were read from it."""

    def __init__(self, handle: Any, counter: list[int]) -> None:
        self._handle = handle
        self._counter = counter

    def __enter__(self) -> "_CountingFile":
        self._handle.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._handle.__exit__(*args)

    def read(self, size: int = -1) -> bytes:
        data: bytes = self._handle.read(size)
        self._counter.append(len(data))
        return data

    def seek(self, *args: Any) -> int:
        return int(self._handle.seek(*args))


def test_one_entity_of_a_large_map_is_read_without_reading_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = make_world(tmp_path)
    entities = tuple(
        entity(f"entity-{n:05d}", geometry=(n % 990, n % 990 + 1, n % 990 + 2)) for n in range(2000)
    )
    chain = tuple(
        relation(
            f"relation-{n:05d}",
            f"entity-{n:05d}",
            RelationPredicate.ON_TOP_OF,
            f"entity-{n + 1:05d}",
        )
        for n in range(0, 1998, 2)
    )
    large = pinned(
        populated_map(
            entities=entities,
            relations=chain,
            metadata=metadata(capabilities=entity_capabilities(RelationPredicate.ON_TOP_OF)),
        ),
        world,
    )
    artifact, manifest = write_artifact(world, context_map=large)
    payload = artifact / "entities/entities.jsonl"
    assert payload.stat().st_size > 500_000
    assert manifest.entity_count == 2000

    counter: list[int] = []
    real_open = pathlib.Path.open

    def counting(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        return _CountingFile(handle, counter) if self.name == "entities.jsonl" else handle

    with ContextMapArtifactReader.open(artifact) as reader:
        monkeypatch.setattr(pathlib.Path, "open", counting)
        found = reader.entity(entity_reference("entity-01234"))
        monkeypatch.undo()

    assert found.entity_id == "entity-01234"
    assert sum(counter) < 5_000, "reading one entity must not read the entity table"


def test_a_large_geometry_is_resolved_lazily_without_loading_the_payload(
    tmp_path: Path,
) -> None:
    world = make_world(tmp_path, geometry_scans=8, points_per_scan=5000)
    payload_size = (world.geometry_dir / "outputs/geometry.bin").stat().st_size
    large = pinned(
        populated_map(geometry_ref=GeometricMapLink(map_id=GEOMETRIC_MAP_ID, point_count=40_000)),
        world,
    )
    artifact, _ = write_artifact(world, context_map=large)
    reference = GeometryReference(
        map_id=MapId(MAP_ID), geometry_id=geometry_id_for(map_id=MapId(MAP_ID), index=31_337)
    )

    with ContextMapArtifactReader.open(artifact) as reader:
        tracemalloc.start()
        point = reader.geometry(reference)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    assert payload_size > 2_000_000
    assert point.reference == reference
    assert peak < payload_size // 4, "the geometry payload must be memory-mapped, not read"


# --- corruption is always detected ---------------------------------------------------------------


@pytest.mark.parametrize("path", CONTRACTUAL_FILES)
def test_a_deleted_truncated_or_altered_contractual_file_is_never_valid(
    world: World, path: str
) -> None:
    for damage in ("delete", "truncate", "alter"):
        artifact, _ = write_artifact(world, name=f"{path.replace('/', '-')}-{damage}")
        target = artifact / path
        if damage == "delete":
            target.unlink()
        elif damage == "truncate":
            target.write_bytes(target.read_bytes()[:-3])
        else:
            data = bytearray(target.read_bytes())
            if data:
                data[len(data) // 2] ^= 0x01
            else:
                data = bytearray(b"x")
            target.write_bytes(bytes(data))

        report = validate_context_map_artifact(artifact)
        assert report.status is ValidationStatus.INVALID, (path, damage)
        assert _errors(artifact), (path, damage)
        with pytest.raises(ContextMapArtifactError):
            ContextMapArtifactReader.open(artifact, verify_hashes=True)


def test_an_altered_upstream_file_is_reported_by_the_validator_and_refused_by_the_writer(
    world: World,
) -> None:
    artifact, _ = write_artifact(world)
    payload = world.relations_dir / "outputs/records.jsonl"
    payload.write_bytes(payload.read_bytes() + b"x")

    assert "dependency.upstream_damaged" in _errors(artifact)
    with pytest.raises(ContextMapArtifactError):
        write_artifact(world, name="another")


def test_a_directory_left_by_an_interrupted_write_is_never_an_artifact(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(self: Any, **_: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("contextmap.shared.run_directory.AtomicRunDirectory.publish", crash)
    monkeypatch.setattr("contextmap.shared.run_directory.shutil.rmtree", lambda *a, **k: None)
    with pytest.raises(KeyboardInterrupt):
        write_artifact(world)
    monkeypatch.undo()

    leftover = next((world.root / "out").iterdir())
    assert _errors(leftover) == {"artifact.incomplete"}
    with pytest.raises(ContextMapArtifactError):
        ContextMapArtifactReader.open(leftover)


def test_corruption_is_never_repaired_from_debug_output(world: World) -> None:
    artifact, _ = write_artifact(world)
    (artifact / "debug").mkdir()
    for path in ("entities/entities.jsonl", "indexes/entity-index.jsonl"):
        (artifact / "debug" / Path(path).name).write_bytes((artifact / path).read_bytes())
    (artifact / "entities/entities.jsonl").write_bytes(b"corrupted\n")
    before = tree_snapshot(artifact)

    assert validate_context_map_artifact(artifact).status is ValidationStatus.INVALID
    with pytest.raises(ContextMapArtifactError):
        ContextMapArtifactReader.open(artifact)

    assert tree_snapshot(artifact) == before


# --- portability --------------------------------------------------------------------------------


def test_the_whole_workspace_can_move_and_the_artifact_still_opens_by_its_hints(
    tmp_path: Path,
) -> None:
    workspace = make_world(tmp_path / "workspace")
    written = make_context_map(workspace)
    write_artifact(workspace, context_map=written)
    shutil.move(tmp_path / "workspace", tmp_path / "relocated")
    artifact = tmp_path / "relocated" / "out" / "context_map"

    assert validate_context_map_artifact(artifact).status is ValidationStatus.VERIFIED
    with ContextMapArtifactReader.open(artifact) as reader:
        assert reader.context_map() == written
        assert (
            reader.geometry(written.entities[0].geometry_refs[0]).reference
            == (written.entities[0].geometry_refs[0])
        )


def test_a_copied_artifact_works_with_explicit_locations_and_a_missing_optional_evidence(
    tmp_path: Path, world: World
) -> None:
    written = make_context_map(world)
    artifact, _ = write_artifact(world, context_map=written)
    elsewhere = tmp_path / "far-away" / "copy"
    shutil.copytree(artifact, elsewhere)
    moved = {}
    for artifact_id, directory in world.structural_locations.items():
        moved[artifact_id] = tmp_path / "far-away" / artifact_id
        shutil.copytree(directory, moved[artifact_id])

    report = validate_context_map_artifact(elsewhere, dependency_paths=moved)

    assert report.status is ValidationStatus.VERIFIED
    assert {item.code for item in report.findings} == {"dependency.optional_missing"}
    with ContextMapArtifactReader.open(elsewhere, dependency_paths=moved) as reader:
        assert reader.context_map() == written


def test_a_portable_bundle_opens_and_round_trips_after_relocation(world: World) -> None:
    written = make_context_map(world)
    artifact, _ = write_artifact(world, context_map=written)
    bundle = world.root / "exports" / "bundle"
    export_bundle(artifact, bundle, policy=ClosurePolicy.REQUIRED)
    relocated = world.root / "another-filesystem" / "bundle"
    shutil.copytree(bundle, relocated)
    shutil.rmtree(world.root / "out")
    for directory in (
        world.geometry_dir.parents[3],
        world.resolution_dir,
        world.relations_dir,
        world.fusion_dir,
    ):
        shutil.rmtree(directory)

    assert verify_bundle(relocated) == ()
    with ContextMapArtifactReader.open(relocated / "artifact", verify_hashes=True) as reader:
        assert context_map_to_record(reader.context_map()) == context_map_to_record(written)
        assert reader.geometry(written.entities[0].geometry_refs[0]) is not None


# --- version handling and compatibility fixtures --------------------------------------------------


def test_the_versions_promised_readable_by_v0_1_0_are_the_ones_the_fixture_uses() -> None:
    manifest = decode_manifest(json.loads((FIXTURE / MANIFEST).read_text()))

    assert manifest.format_version == FORMAT_VERSION == "0.1.0"
    assert manifest.format_version in SUPPORTED_FORMAT_VERSIONS
    assert manifest.schema_version == CONTEXT_MAP_SCHEMA_VERSION == "0.1.0"
    assert manifest.content_identity == FIXTURE_CONTENT_IDENTITY, (
        "the v0.1.0 fixture must never be regenerated: it proves old artifacts stay readable"
    )


def test_the_v0_1_0_fixture_is_still_readable_and_says_what_it_depends_on() -> None:
    expected = populated_map()

    with ContextMapArtifactReader.open(FIXTURE, verify_hashes=True) as reader:
        assert reader.metadata() == expected.metadata
        assert reader.geometry_link() == expected.geometry_ref
        assert tuple(reader.entities()) == expected.entities
        assert tuple(reader.relations()) == expected.relations
        assert [item.artifact_id for item in reader.lineage()] == [
            item.artifact_id for item in expected.lineage
        ]
        assert {
            item.relation_id for item in reader.relations_for(entity_reference("entity-0002"))
        } == {
            "relation-0001",
            "relation-0002",
        }


def test_the_v0_1_0_fixture_validates_completely_except_for_the_upstream_it_does_not_carry() -> (
    None
):
    report = validate_context_map_artifact(FIXTURE)

    assert {item.code for item in report.findings if item.severity is Severity.ERROR} == {
        "dependency.required_missing"
    }
    outcomes = {check.name: check.outcome.value for check in report.checks}
    for name in ("file_hashes", "reference_integrity", "index_rebuild", "schema_invariants"):
        assert outcomes[name] == "passed", name


def test_a_schema_patch_release_of_the_same_minor_is_readable(world: World) -> None:
    artifact, _ = write_artifact(world)
    manifest_path = artifact / "manifest.json"
    record = json.loads(manifest_path.read_text())
    record["schema_version"] = "0.1.7"
    manifest_path.write_text(json.dumps(record))
    _reseal(artifact, schema_version="0.1.7")

    with ContextMapArtifactReader.open(artifact) as reader:
        assert reader.manifest.schema_version == "0.1.7"


@pytest.mark.parametrize("version", ["0.2.0", "1.0.0", "9.9.9", "0.1", "v0.1.0"])
def test_an_unsupported_schema_version_fails_before_any_record_is_interpreted(
    world: World, version: str
) -> None:
    artifact, _ = write_artifact(world)
    # Se algum registro fosse lido, o erro seria outro: a versão é rejeitada antes de tudo.
    (artifact / "entities/entities.jsonl").write_bytes(b"not even json\n")
    record = json.loads((artifact / "manifest.json").read_text())
    record["schema_version"] = version
    (artifact / "manifest.json").write_text(json.dumps(record))

    with pytest.raises(UnsupportedArtifactSchemaError, match=r"schema version|MAJOR"):
        ContextMapArtifactReader.open(artifact)
    assert _errors(artifact) == {"manifest.unsupported_schema_version"}


@pytest.mark.parametrize("version", ["0.0.9", "0.2.0", "1.0.0", "", "0.1.0-beta"])
def test_an_unsupported_format_version_lists_the_supported_ones(world: World, version: str) -> None:
    artifact, _ = write_artifact(world)
    record = json.loads((artifact / "manifest.json").read_text())
    record["format_version"] = version
    (artifact / "manifest.json").write_text(json.dumps(record))

    with pytest.raises(UnsupportedFormatVersionError, match=r"0\.1\.0"):
        ContextMapArtifactReader.open(artifact)
    assert _errors(artifact) == {"manifest.unsupported_format_version"}
