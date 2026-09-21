"""Canonical record view of the schema: JSON-compatible, strict and format independent."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from context_map_builders import context_map

from contextmap.artifact import (
    ContextMapRecordError,
    UnsupportedSchemaVersionError,
    context_map_from_record,
    context_map_to_record,
)


def _json_round_trip(record: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    return decoded


def test_a_map_round_trips_through_plain_json() -> None:
    original = context_map()

    record = _json_round_trip(context_map_to_record(original))

    assert context_map_from_record(record) == original


def test_the_record_names_the_schema_version_and_the_fields_verbatim() -> None:
    record = context_map_to_record(context_map())

    assert record["schema_version"] == "0.1.0"
    assert record["geometry_ref"] == {"map_id": "corridor-02--map-run-0001", "point_count": 1000}
    assert record["metadata"]["creation"]["assembly_policy"] == {
        "policy_id": "context-map-assembly",
        "version": "1",
    }


def test_optional_values_are_written_explicitly_as_null() -> None:
    record = context_map_to_record(context_map())

    assert "code_version" in record["metadata"]["creation"]
    absent = deepcopy(record)
    absent["metadata"]["creation"]["code_version"] = None

    assert context_map_from_record(absent).metadata.creation.code_version is None


def test_encoding_is_deterministic() -> None:
    first = json.dumps(context_map_to_record(context_map()), sort_keys=True)
    second = json.dumps(context_map_to_record(context_map()), sort_keys=True)

    assert first == second


def test_an_unknown_field_is_rejected_not_ignored() -> None:
    record = context_map_to_record(context_map())
    record["confidence"] = 0.9

    with pytest.raises(ContextMapRecordError, match="confidence"):
        context_map_from_record(record)


def test_a_missing_field_is_rejected_not_defaulted() -> None:
    record = context_map_to_record(context_map())
    del record["metadata"]["source_sequences"]

    with pytest.raises(ContextMapRecordError, match=r"metadata\.source_sequences"):
        context_map_from_record(record)


def test_a_wrong_type_is_rejected_with_its_location() -> None:
    record = context_map_to_record(context_map())
    record["geometry_ref"]["point_count"] = "1000"

    with pytest.raises(ContextMapRecordError, match=r"geometry_ref\.point_count"):
        context_map_from_record(record)


def test_a_boolean_is_not_accepted_as_a_number() -> None:
    record = context_map_to_record(context_map())
    record["geometry_ref"]["point_count"] = True

    with pytest.raises(ContextMapRecordError, match=r"geometry_ref\.point_count"):
        context_map_from_record(record)


def test_the_version_is_checked_before_the_rest_of_the_record() -> None:
    record = {"schema_version": "9.0.0", "anything": "else"}

    with pytest.raises(UnsupportedSchemaVersionError, match=r"9\.0\.0"):
        context_map_from_record(record)


def test_a_record_without_a_version_is_refused() -> None:
    with pytest.raises(UnsupportedSchemaVersionError, match="schema version"):
        context_map_from_record({})


def test_an_invalid_value_is_not_silently_repaired() -> None:
    record = context_map_to_record(context_map())
    record["geometry_ref"]["point_count"] = 0

    with pytest.raises(ValueError, match="point_count"):
        context_map_from_record(record)


def test_a_record_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(ContextMapRecordError, match="mapping"):
        context_map_from_record([])  # type: ignore[arg-type]
