"""Contract of the on-disk manifest of a ContextMapArtifact (issue #155).

The manifest is the one file a human or a tool opens first: it names the format and schema
versions, inventories every contractual file with size and hash, describes each binary or
tabular payload (dtype, shape, unit, frame, semantics) and lists the artifacts the map depends
on. These tests pin what is inspectable and what is identity.
"""

import json
from dataclasses import replace
from typing import Any

import pytest

from contextmap.artifact.serialization.errors import (
    ContextMapArtifactError,
    ManifestError,
    UnsupportedFormatVersionError,
)
from contextmap.artifact.serialization.layout import ARTIFACT_TYPE, FORMAT_VERSION
from contextmap.artifact.serialization.manifest import (
    COLUMN_DTYPES,
    ColumnPayload,
    ContextMapArtifactManifest,
    DependencyRecord,
    PayloadRole,
    RecordPayload,
    Requirement,
    create_manifest,
    decode_manifest,
    encode_manifest,
    inventory_digest,
    manifest_content_identity,
)
from contextmap.shared import FileEntry, file_entry

GEOMETRY_DIGEST = "sha256:" + "a" * 64
EVIDENCE_DIGEST = "sha256:" + "b" * 64


def _entities_payload() -> RecordPayload:
    return RecordPayload(
        path="entities/entities.jsonl",
        role=PayloadRole.AUTHORITATIVE,
        semantics="One resolved entity per line, ordered by entity id.",
        record_count=2,
    )


def _entity_index_payload() -> RecordPayload:
    return RecordPayload(
        path="indexes/entity-index.jsonl",
        role=PayloadRole.DERIVED_INDEX,
        semantics="Entity id to byte offset and length in entities.jsonl.",
        record_count=2,
        derived_from=("entities/entities.jsonl",),
    )


def _support_column() -> ColumnPayload:
    return ColumnPayload(
        path="indexes/entity-geometry-support.u32",
        role=PayloadRole.DERIVED_INDEX,
        semantics="Positional geometry index of the support of each entity, entity after entity.",
        dtype="uint32",
        shape=(6,),
        derived_from=("entities/entities.jsonl",),
        unit=None,
        frame_id="map",
    )


def _dependencies() -> tuple[DependencyRecord, ...]:
    return (
        DependencyRecord(
            artifact_type="geometric_map",
            artifact_id="corridor-02--run-0001",
            content_identity=GEOMETRY_DIGEST,
            requirement=Requirement.REQUIRED,
            locator="../geometric_mapping",
        ),
        DependencyRecord(
            artifact_type="semantic_fusion_run",
            artifact_id="run-0003",
            content_identity=EVIDENCE_DIGEST,
            requirement=Requirement.OPTIONAL,
        ),
    )


def _inventory() -> tuple[FileEntry, ...]:
    return (
        file_entry("entities/entities.jsonl", b'{"id": "e1"}\n{"id": "e2"}\n'),
        file_entry("indexes/entity-index.jsonl", b"index"),
        file_entry("indexes/entity-geometry-support.u32", b"\x00" * 24),
        file_entry("map-metadata.json", b"{}"),
    )


def _manifest(**overrides: Any) -> ContextMapArtifactManifest:
    fields: dict[str, Any] = {
        "context_map_id": "context-map-0001",
        "schema_version": "0.1.0",
        "written_at": "2026-09-21T12:00:00+00:00",
        "code_version": "abc1234",
        "configuration_fingerprint": "sha256:" + "c" * 64,
        "entity_count": 2,
        "relation_count": 0,
        "payloads": (_entities_payload(), _entity_index_payload(), _support_column()),
        "dependencies": _dependencies(),
        "file_inventory": _inventory(),
    }
    fields.update(overrides)
    return create_manifest(**fields)


def test_manifest_round_trips_through_its_json_record() -> None:
    manifest = _manifest()

    record = json.loads(json.dumps(encode_manifest(manifest)))

    assert decode_manifest(record) == manifest


def test_manifest_record_is_inspectable_with_ordinary_tools() -> None:
    record = encode_manifest(_manifest())

    assert record["artifact_type"] == ARTIFACT_TYPE == "context_map"
    assert record["format_version"] == FORMAT_VERSION
    assert record["schema_version"] == "0.1.0"
    assert record["context_map_id"] == "context-map-0001"
    assert record["entity_count"] == 2
    column = next(item for item in record["payloads"] if item["encoding"] == "raw-le")
    assert column == {
        "encoding": "raw-le",
        "path": "indexes/entity-geometry-support.u32",
        "role": "derived_index",
        "semantics": (
            "Positional geometry index of the support of each entity, entity after entity."
        ),
        "dtype": "uint32",
        "byte_order": "little",
        "shape": [6],
        "unit": None,
        "frame_id": "map",
        "derived_from": ["entities/entities.jsonl"],
    }
    rows = next(item for item in record["payloads"] if item["path"] == "entities/entities.jsonl")
    assert rows["encoding"] == "jsonl"
    assert rows["record_count"] == 2
    assert record["dependencies"][0] == {
        "artifact_type": "geometric_map",
        "artifact_id": "corridor-02--run-0001",
        "content_identity": GEOMETRY_DIGEST,
        "requirement": "required",
        "locator": "../geometric_mapping",
    }
    assert [item["path"] for item in record["file_inventory"]] == sorted(
        item["path"] for item in record["file_inventory"]
    )
    assert record["file_inventory"][0]["content_hash"].startswith("sha256:")


def test_content_identity_is_stable_for_equivalent_content() -> None:
    first = _manifest()
    second = _manifest(
        written_at="2027-01-01T00:00:00+00:00",
        dependencies=tuple(replace(item, locator="elsewhere") for item in _dependencies()),
    )

    assert first.content_identity == second.content_identity
    assert first.content_identity == manifest_content_identity(first)
    assert first.content_identity.startswith("sha256:")


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "0.2.0"},
        {"context_map_id": "context-map-0002"},
        {"code_version": "def5678"},
        {"configuration_fingerprint": None},
        {"entity_count": 3},
        {"file_inventory": (*_inventory()[:-1], file_entry("map-metadata.json", b"{ }"))},
        {"payloads": (_entities_payload(), _entity_index_payload())},
        {
            "dependencies": (
                replace(_dependencies()[0], requirement=Requirement.OPTIONAL),
                _dependencies()[1],
            )
        },
        {"dependencies": (replace(_dependencies()[0], content_identity=EVIDENCE_DIGEST),)},
    ],
)
def test_content_identity_changes_with_any_semantic_content(overrides: dict[str, Any]) -> None:
    assert _manifest(**overrides).content_identity != _manifest().content_identity


def test_content_identity_does_not_depend_on_the_order_of_inventory_or_dependencies() -> None:
    reordered = _manifest(
        file_inventory=tuple(reversed(_inventory())),
        dependencies=tuple(reversed(_dependencies())),
    )

    assert reordered == _manifest()


def test_a_tampered_manifest_no_longer_matches_its_recorded_identity() -> None:
    record = encode_manifest(_manifest())
    record["entity_count"] = 99

    tampered = decode_manifest(record)

    assert manifest_content_identity(tampered) != tampered.content_identity


def test_inventory_digest_depends_on_content_not_on_order() -> None:
    inventory = _inventory()

    assert inventory_digest(inventory) == inventory_digest(tuple(reversed(inventory)))
    changed = (*inventory[:-1], file_entry("map-metadata.json", b"other"))
    assert inventory_digest(changed) != inventory_digest(inventory)


@pytest.mark.parametrize("version", ["0.0.1", "1.0.0", "", "not-a-version"])
def test_an_unsupported_format_version_is_rejected_with_the_supported_ones(version: str) -> None:
    record = encode_manifest(_manifest())
    record["format_version"] = version

    with pytest.raises(UnsupportedFormatVersionError) as raised:
        decode_manifest(record)

    assert repr(version) in str(raised.value)
    assert FORMAT_VERSION in str(raised.value)
    assert isinstance(raised.value, ContextMapArtifactError)


def test_a_manifest_of_another_artifact_type_is_rejected() -> None:
    record = encode_manifest(_manifest())
    record["artifact_type"] = "geometric_map"

    with pytest.raises(ManifestError, match="artifact_type"):
        decode_manifest(record)


def test_a_missing_or_unknown_field_is_rejected_instead_of_ignored() -> None:
    missing = encode_manifest(_manifest())
    del missing["dependencies"]
    unknown = encode_manifest(_manifest())
    unknown["future_field"] = 1

    with pytest.raises(ManifestError, match="dependencies"):
        decode_manifest(missing)
    with pytest.raises(ManifestError, match="future_field"):
        decode_manifest(unknown)


def test_a_manifest_that_is_not_a_json_object_is_rejected() -> None:
    with pytest.raises(ManifestError, match="object"):
        decode_manifest([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field, value",
    [
        ("entity_count", -1),
        ("entity_count", True),
        ("relation_count", "2"),
        ("context_map_id", ""),
        ("schema_version", 1),
        ("written_at", "yesterday"),
        ("content_identity", "sha256:short"),
    ],
)
def test_decoding_validates_every_scalar_field(field: str, value: object) -> None:
    record = encode_manifest(_manifest())
    record[field] = value

    with pytest.raises(ManifestError, match=field):
        decode_manifest(record)


def test_payload_paths_and_inventory_paths_must_be_unique() -> None:
    with pytest.raises(ManifestError, match="duplicate payload"):
        _manifest(payloads=(_entities_payload(), _entities_payload()))
    with pytest.raises(ManifestError, match="duplicate inventory"):
        _manifest(file_inventory=(*_inventory(), _inventory()[0]))
    with pytest.raises(ManifestError, match="duplicate dependency"):
        _manifest(dependencies=(_dependencies()[0], _dependencies()[0]))


def test_every_payload_must_be_inventoried_and_every_source_must_be_a_payload() -> None:
    with pytest.raises(ManifestError, match="not inventoried"):
        _manifest(file_inventory=_inventory()[1:])
    orphan = replace(_entity_index_payload(), derived_from=("entities/unknown.jsonl",))
    with pytest.raises(ManifestError, match="derived_from"):
        _manifest(payloads=(orphan, _entities_payload(), _support_column()))


def test_the_inventory_never_lists_debug_or_escaping_paths() -> None:
    with pytest.raises(ManifestError, match="debug"):
        _manifest(file_inventory=(*_inventory(), file_entry("debug/trace.json", b"{}")))
    with pytest.raises(ManifestError, match="relative"):
        _manifest(file_inventory=(*_inventory(), file_entry("../escape.json", b"{}")))


def test_column_payloads_describe_their_size_from_dtype_and_shape() -> None:
    column = ColumnPayload(
        path="indexes/bounds.f64",
        role=PayloadRole.DERIVED_INDEX,
        semantics="Axis-aligned bounds, [entity, (min_xyz, max_xyz)].",
        dtype="float64",
        shape=(3, 6),
        derived_from=("entities/entities.jsonl",),
        unit="m",
        frame_id="map",
    )

    assert column.expected_size_bytes == 3 * 6 * 8
    assert set(COLUMN_DTYPES) == {
        "uint8",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    }
    empty = replace(column, shape=(0, 6))
    assert empty.expected_size_bytes == 0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dtype": "float16"}, "dtype"),
        ({"dtype": "object"}, "dtype"),
        ({"shape": ()}, "shape"),
        ({"shape": (-1,)}, "shape"),
        ({"shape": (2.5,)}, "shape"),
        ({"unit": ""}, "unit"),
        ({"frame_id": ""}, "frame_id"),
        ({"semantics": ""}, "semantics"),
        ({"role": PayloadRole.AUTHORITATIVE}, "derived_from"),
        ({"derived_from": ()}, "derived_from"),
    ],
)
def test_invalid_column_payloads_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        replace(_support_column(), **overrides)


def test_invalid_record_payloads_are_rejected() -> None:
    with pytest.raises(ManifestError, match="record_count"):
        replace(_entities_payload(), record_count=-1)
    with pytest.raises(ManifestError, match="derived_from"):
        replace(_entities_payload(), derived_from=("entities/other.jsonl",))
    with pytest.raises(ManifestError, match="derived_from"):
        replace(_entity_index_payload(), derived_from=())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"artifact_type": ""}, "artifact_type"),
        ({"artifact_id": ""}, "artifact_id"),
        ({"content_identity": "md5:abc"}, "content_identity"),
        ({"locator": "/absolute/path"}, "locator"),
        ({"locator": "C:\\workspace"}, "locator"),
        ({"locator": ""}, "locator"),
    ],
)
def test_invalid_dependency_records_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        replace(_dependencies()[0], **overrides)


def test_a_dependency_is_required_or_optional_never_implicit() -> None:
    record = encode_manifest(_manifest())
    record["dependencies"][0]["requirement"] = "maybe"

    with pytest.raises(ManifestError, match="requirement"):
        decode_manifest(record)
