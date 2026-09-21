"""Schema versioning: negotiation, evolution rules, classification examples and identity."""

from __future__ import annotations

from dataclasses import make_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from context_map_builders import context_map

from contextmap.artifact import (
    CONTEXT_MAP_SCHEMA_VERSION,
    SchemaVersion,
    UnsupportedSchemaVersionError,
    context_map_from_record,
    context_map_to_record,
    describe_schema,
    require_supported_schema_version,
    schema_fingerprint,
)

VERSIONING_DOC = (
    Path(__file__).resolve().parents[2] / "src/contextmap/artifact/docs/versioning.md"
).read_text(encoding="utf-8")

# Impressão digital estrutural de cada versão do schema prometida como legível. Quando a
# estrutura muda, este teste falha de propósito: classifique a mudança em docs/versioning.md,
# suba CONTEXT_MAP_SCHEMA_VERSION quando exigido e registre a nova impressão aqui.
PINNED_FINGERPRINTS = {
    "0.1.0": "sha256:d372e4ff26cb378912393fe1a014ed8d1a583a737006aa53c04c1a10a8155047",
}


# --- parsing and ordering ----------------------------------------------------------------------


def test_a_version_prints_canonically_and_round_trips() -> None:
    version = SchemaVersion.parse("1.12.3")

    assert (version.major, version.minor, version.patch) == (1, 12, 3)
    assert str(version) == "1.12.3"
    assert SchemaVersion.parse(str(version)) == version


def test_versions_order_numerically_not_textually() -> None:
    assert SchemaVersion.parse("0.10.0") > SchemaVersion.parse("0.9.0")
    assert SchemaVersion.parse("1.0.0") > SchemaVersion.parse("0.99.99")


@pytest.mark.parametrize("text", ["1", "1.0", "1.0.0.0", "01.0.0", "1.0.0-rc.1", "v1.0.0", "1.0.x"])
def test_only_canonical_versions_are_accepted(text: str) -> None:
    with pytest.raises(UnsupportedSchemaVersionError, match="malformed"):
        SchemaVersion.parse(text)


# --- version negotiation and the reader window -------------------------------------------------


@pytest.mark.parametrize(
    ("document", "reader", "readable"),
    [
        # Fase de validação (major 0): só a mesma MAJOR.MINOR; o PATCH nunca importa.
        ("0.1.0", "0.1.0", True),
        ("0.1.9", "0.1.0", True),
        ("0.1.0", "0.1.9", True),
        ("0.2.0", "0.1.0", False),
        ("0.1.0", "0.2.0", False),
        ("1.0.0", "0.1.0", False),
        # A partir da 1.0.0: qualquer versão do mesmo major é legível.
        ("1.0.0", "1.0.0", True),
        ("1.0.5", "1.2.0", True),
        ("1.3.0", "1.1.0", True),
        ("2.0.0", "1.9.9", False),
        ("1.9.9", "2.0.0", False),
    ],
)
def test_the_reader_window(document: str, reader: str, readable: bool) -> None:
    result = SchemaVersion.parse(document).is_readable_by(SchemaVersion.parse(reader))

    assert result is readable


def test_this_reader_supports_its_own_version() -> None:
    assert require_supported_schema_version(CONTEXT_MAP_SCHEMA_VERSION) == SchemaVersion.parse(
        CONTEXT_MAP_SCHEMA_VERSION
    )


def test_an_unsupported_major_is_rejected_explicitly_naming_both_versions() -> None:
    reader = SchemaVersion.parse(CONTEXT_MAP_SCHEMA_VERSION)
    unsupported = f"{reader.major + 1}.0.0"

    with pytest.raises(UnsupportedSchemaVersionError) as failure:
        require_supported_schema_version(unsupported)

    assert unsupported in str(failure.value)
    assert str(reader) in str(failure.value)


def test_nothing_is_read_from_a_record_of_an_unsupported_version() -> None:
    record = context_map_to_record(context_map())
    record["schema_version"] = "9.0.0"
    record["metadata"] = "would fail elsewhere if it were read"

    with pytest.raises(UnsupportedSchemaVersionError, match=r"9\.0\.0"):
        context_map_from_record(record)


def test_a_map_records_the_schema_version_it_was_written_under() -> None:
    assert context_map().schema_version == CONTEXT_MAP_SCHEMA_VERSION
    assert context_map_to_record(context_map())["schema_version"] == CONTEXT_MAP_SCHEMA_VERSION


# --- classification examples -------------------------------------------------------------------

# Cada exemplo aparece, com o mesmo texto, na tabela de docs/versioning.md.
EXAMPLES = [
    ("novo campo opcional com padrão explícito", "minor"),
    ("nova capacidade opcional em MapCapability", "minor"),
    ("novo predicado de relação", "minor"),
    ("novo ArtifactKind apenas de evidência", "minor"),
    ("novo DerivationKind", "minor"),
    ("esclarecimento de documentação", "patch"),
    ("correção de mensagem de erro", "patch"),
    ("remover ou renomear um campo", "major"),
    ("novo campo obrigatório", "major"),
    ("mudar o significado, a unidade ou o frame de um campo", "major"),
    (
        "novo membro de LengthUnit, Handedness, AnchorKind, AmbiguityStatus ou RelationState",
        "major",
    ),
    ("novo ArtifactKind estrutural", "major"),
    ("mudar o escopo de uma identidade", "major"),
    ("invalidar um mapa que era válido", "major"),
]


def _bumped(version: str, kind: str) -> str:
    current = SchemaVersion.parse(version)
    if kind == "major":
        return f"{current.major + 1}.0.0"
    if kind == "minor":
        return f"{current.major}.{current.minor + 1}.0"
    return f"{current.major}.{current.minor}.{current.patch + 1}"


@pytest.mark.parametrize(("description", "kind"), EXAMPLES)
def test_every_example_is_documented_with_its_classification(description: str, kind: str) -> None:
    row = next((line for line in VERSIONING_DOC.splitlines() if description in line), None)

    assert row is not None, f"docs/versioning.md does not list {description!r}"
    assert kind.upper() in row


@pytest.mark.parametrize(("description", "kind"), EXAMPLES)
def test_a_classified_change_has_the_promised_reader_consequences(
    description: str, kind: str
) -> None:
    # A partir da 1.0.0, PATCH e MINOR continuam legíveis por leitores antigos; MAJOR não.
    old, new = SchemaVersion.parse("1.2.0"), SchemaVersion.parse(_bumped("1.2.0", kind))
    older_reads_newer = new.is_readable_by(old)
    newer_reads_older = old.is_readable_by(new)

    if kind == "major":
        assert not older_reads_newer and not newer_reads_older, description
    else:
        assert older_reads_newer and newer_reads_older, description


@pytest.mark.parametrize(("description", "kind"), EXAMPLES)
def test_during_validation_only_a_patch_keeps_readers_working(description: str, kind: str) -> None:
    old, new = SchemaVersion.parse("0.1.0"), SchemaVersion.parse(_bumped("0.1.0", kind))

    assert new.is_readable_by(old) is (kind == "patch"), description


# --- identity: a fingerprint of the structure --------------------------------------------------


def test_the_fingerprint_is_a_deterministic_sha256() -> None:
    first, second = schema_fingerprint(), schema_fingerprint()

    assert first == second
    assert first.startswith("sha256:") and len(first) == len("sha256:") + 64


def test_the_structure_of_each_promised_version_is_pinned() -> None:
    assert CONTEXT_MAP_SCHEMA_VERSION in PINNED_FINGERPRINTS, (
        f"register the fingerprint of schema {CONTEXT_MAP_SCHEMA_VERSION}: {schema_fingerprint()}"
    )
    assert schema_fingerprint() == PINNED_FINGERPRINTS[CONTEXT_MAP_SCHEMA_VERSION], (
        "the structure of the ContextMap schema changed: classify the change in "
        "docs/versioning.md, bump CONTEXT_MAP_SCHEMA_VERSION when the rules require it and pin "
        f"{schema_fingerprint()}"
    )


def test_the_fingerprint_covers_the_types_the_schema_reuses() -> None:
    description = describe_schema()

    for name in (
        "GeometryReference",
        "Bounds3D",
        "SourceTimestamp",
        "MapId",
        "ContextEntity",
        "ResolvedEntityReference",
        "EntityReference",
        "RelationPredicate",
        "RelationState",
    ):
        assert name in description


def _unit(*members: str) -> Any:
    """Build a synthetic enum named ``_Unit`` with the given members."""
    return Enum("_Unit", {member.upper(): member for member in members})


_UNIT = _unit("meter")
_BASE = make_dataclass("_Base", [("label", str)], frozen=True)


def _root(fields: list[tuple[str, object]]) -> type:
    """Build a synthetic schema root named ``_Root`` with the given fields."""
    return make_dataclass("_Root", fields, frozen=True)


def _standard_fields() -> list[tuple[str, object]]:
    return [("base", _BASE), ("unit", _UNIT), ("tags", tuple[str, ...])]


def test_the_fingerprint_changes_when_the_structure_changes() -> None:
    baseline = schema_fingerprint(_root(_standard_fields()))
    extra_field = _root([*_standard_fields(), ("note", str | None)])
    retyped = _root([("base", _BASE), ("unit", _UNIT), ("tags", tuple[int, ...])])
    unit_with_foot = _unit("meter", "foot")
    new_member = _root([("base", _BASE), ("unit", unit_with_foot), ("tags", tuple[str, ...])])

    assert schema_fingerprint(extra_field) != baseline
    assert schema_fingerprint(retyped) != baseline
    assert schema_fingerprint(new_member) != baseline


def test_the_fingerprint_ignores_declaration_order() -> None:
    reordered = _root(list(reversed(_standard_fields())))

    assert schema_fingerprint(reordered) == schema_fingerprint(_root(_standard_fields()))


def test_the_fingerprint_follows_types_that_a_field_only_references() -> None:
    other_base = make_dataclass("_Base", [("label", str), ("extra", int)], frozen=True)
    deeper = _root([("base", other_base), ("unit", _UNIT), ("tags", tuple[str, ...])])

    assert schema_fingerprint(deeper) != schema_fingerprint(_root(_standard_fields()))
