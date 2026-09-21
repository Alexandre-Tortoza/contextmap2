"""Contract of the top-level ContextMap: identity, version, metadata and geometry reference."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest
from context_map_builders import (
    CONTEXT_MAP_ID,
    GEOMETRIC_MAP_ID,
    context_map,
    creation,
    geometry_link,
    metadata,
    policy,
    source_sequence,
)

from contextmap.artifact import (
    CONTEXT_MAP_SCHEMA_VERSION,
    ContextMap,
    ContextMapMetadata,
    GeometricMapLink,
    MapCreation,
    PolicyRef,
    SourceSequence,
    UnsupportedSchemaVersionError,
)
from contextmap.geometric_mapping import MapId


def test_a_map_exposes_the_parts_a_consumer_needs() -> None:
    result = context_map()

    assert result.context_map_id == CONTEXT_MAP_ID
    assert result.schema_version == CONTEXT_MAP_SCHEMA_VERSION
    assert result.metadata.creation.assembly_policy == policy()
    assert result.geometry_ref.map_id == GEOMETRIC_MAP_ID


def test_a_map_is_immutable() -> None:
    result = context_map()

    with pytest.raises(FrozenInstanceError):
        result.context_map_id = CONTEXT_MAP_ID  # type: ignore[misc]


def test_equivalent_maps_are_equal_and_hashable() -> None:
    assert context_map() == context_map()
    assert hash(context_map()) == hash(context_map())


@pytest.mark.parametrize("blank", ["", "   "])
def test_the_map_identity_must_be_present(blank: str) -> None:
    with pytest.raises(ValueError, match="context_map_id"):
        context_map(context_map_id=blank)


def test_the_identity_is_not_a_path() -> None:
    # A identidade semântica nunca é um caminho: o layout em disco é decisão do serializador.
    with pytest.raises(ValueError, match="path"):
        context_map(context_map_id="workspace/corridor-02/context_map")


@pytest.mark.parametrize("blank", ["", "  "])
def test_the_assembly_policy_must_be_identified(blank: str) -> None:
    with pytest.raises(ValueError, match="policy_id"):
        PolicyRef(policy_id=blank, version="1")
    with pytest.raises(ValueError, match="version"):
        PolicyRef(policy_id="context-map-assembly", version=blank)


def test_creation_records_optional_identities_explicitly() -> None:
    result = MapCreation(
        assembly_policy=policy(), code_version=None, configuration_fingerprint=None
    )

    assert result.code_version is None
    assert result.configuration_fingerprint is None


@pytest.mark.parametrize("field", ["code_version", "configuration_fingerprint"])
def test_a_present_creation_identity_must_not_be_blank(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        creation(**{field: " "})


def test_at_least_one_source_sequence_is_required() -> None:
    with pytest.raises(ValueError, match="source_sequences"):
        metadata(source_sequences=())


def test_source_sequences_are_sorted_and_unique() -> None:
    first = source_sequence("sequence--a")
    second = source_sequence("sequence--b")

    assert metadata(source_sequences=(first, second)).source_sequences == (first, second)
    with pytest.raises(ValueError, match="sorted"):
        metadata(source_sequences=(second, first))
    with pytest.raises(ValueError, match="unique"):
        metadata(source_sequences=(first, first))


def test_one_sequence_may_appear_with_distinct_selections() -> None:
    early = source_sequence(selection_id="selection--early")
    late = source_sequence(selection_id="selection--late")

    assert metadata(source_sequences=(early, late)).source_sequences == (early, late)


@pytest.mark.parametrize("field", ["sequence_artifact_id", "selection_id"])
def test_a_source_sequence_needs_both_identities(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SourceSequence(**{"sequence_artifact_id": "s", "selection_id": "x", field: ""})


def test_geometry_is_referenced_by_identity_and_size() -> None:
    link = geometry_link()

    assert link.map_id == GEOMETRIC_MAP_ID
    assert link.point_count == 1_000


def test_geometry_reference_needs_an_identity_and_a_positive_size() -> None:
    with pytest.raises(ValueError, match="map_id"):
        GeometricMapLink(map_id=MapId(""), point_count=1)
    with pytest.raises(ValueError, match="point_count"):
        geometry_link(point_count=0)


def test_geometry_is_never_embedded() -> None:
    # O contrato referencia o mapa geométrico: nenhum campo carrega coordenadas.
    fields = {name for name in GeometricMapLink.__dataclass_fields__}

    assert fields == {"map_id", "point_count"}


@pytest.mark.parametrize("version", ["1.0.0", "0.2.0", "0.0.9"])
def test_an_unsupported_schema_version_is_rejected_explicitly(version: str) -> None:
    with pytest.raises(UnsupportedSchemaVersionError, match=version):
        context_map(schema_version=version)


@pytest.mark.parametrize("version", ["", "0.1", "v0.1.0", "0.1.0-rc1", "0.01.0", "a.b.c", " 0.1.0"])
def test_a_malformed_schema_version_is_rejected_explicitly(version: str) -> None:
    with pytest.raises(UnsupportedSchemaVersionError, match="schema version"):
        context_map(schema_version=version)


def test_a_patch_revision_of_the_supported_version_is_readable() -> None:
    assert context_map(schema_version="0.1.7").schema_version == "0.1.7"


def test_the_contract_types_are_the_public_surface() -> None:
    assert ContextMap.__module__.startswith("contextmap.artifact")
    assert ContextMapMetadata.__module__.startswith("contextmap.artifact")


@pytest.mark.parametrize("module", ["torch", "transformers", "rclpy", "rosbags", "cv2", "PIL"])
def test_the_schema_imports_without_robotics_or_model_stacks(module: str) -> None:
    program = (
        "import sys\n"
        "import contextmap.artifact\n"
        f"assert {module!r} not in sys.modules, {module!r}\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_metadata_is_independent_of_filesystem_layout() -> None:
    forbidden = {"path", "directory", "filename", "uri", "url"}
    fields: set[str] = set()
    for cls in (ContextMap, ContextMapMetadata, MapCreation, SourceSequence, GeometricMapLink):
        fields |= set(cls.__dataclass_fields__)

    assert not {name for name in fields if any(word in name for word in forbidden)}
