"""JSON Lines record tables with a byte-offset index (issues #156 and #157).

Entities and relations are stored one canonical record per line. The index gives random access:
one record is read by key without parsing the others, and a damaged index is an explicit error,
never a silently wrong record.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from contextmap.artifact.errors import (
    BrokenIndexError,
    ContextMapArtifactError,
    MissingPayloadError,
    RecordNotFoundError,
    RecordTableError,
)
from contextmap.artifact.tables import (
    RecordTable,
    canonical_json_line,
    document_json,
    encode_record_table,
)


def _lines() -> list[dict[str, Any]]:
    return [
        {"key": "entity-b", "record": {"label": "door", "score": 0.5}},
        {"key": "entity-a", "record": {"label": "chair", "score": None}},
        {"key": "entity-c", "record": {"label": "cadeira á", "score": 1.0}},
    ]


def _write(tmp_path: Path, lines: list[dict[str, Any]]) -> tuple[Path, Path]:
    table = encode_record_table(lines)
    payload, index = tmp_path / "table.jsonl", tmp_path / "table-index.jsonl"
    payload.write_bytes(table.payload)
    index.write_bytes(table.index)
    return payload, index


def _open(tmp_path: Path, lines: list[dict[str, Any]] | None = None) -> RecordTable:
    payload, index = _write(tmp_path, _lines() if lines is None else lines)
    return RecordTable(payload, index, record_count=len(lines if lines is not None else _lines()))


def test_a_table_holds_one_canonical_line_per_record_ordered_by_key() -> None:
    table = encode_record_table(_lines())

    lines = table.payload.split(b"\n")
    assert lines[-1] == b""
    keys = [json.loads(line)["key"] for line in lines[:-1]]
    assert keys == ["entity-a", "entity-b", "entity-c"]
    assert table.record_count == 3
    assert lines[0] == canonical_json_line(_lines()[1])


def test_the_encoding_is_deterministic_and_independent_of_the_input_order() -> None:
    assert encode_record_table(_lines()) == encode_record_table(list(reversed(_lines())))
    assert encode_record_table(_lines()) == encode_record_table(_lines())


def test_a_line_is_ascii_json_with_sorted_keys_and_no_raw_newline() -> None:
    line = canonical_json_line({"b": 1, "a": f"linha\nnova {chr(0x2028)} é"})

    assert line == b'{"a":"linha\\nnova \\u2028 \\u00e9","b":1}'
    assert b"\n" not in line


def test_documents_are_indented_sorted_json_with_a_trailing_newline() -> None:
    data = document_json({"b": [1, 2], "a": {"z": None}})

    assert data.endswith(b"\n")
    assert data.decode("utf-8").splitlines()[1].startswith('  "a"')
    assert json.loads(data) == {"b": [1, 2], "a": {"z": None}}


def test_non_finite_numbers_are_refused_because_they_are_not_json() -> None:
    with pytest.raises(RecordTableError, match="entity-a"):
        encode_record_table([{"key": "entity-a", "record": {"score": float("nan")}}])
    with pytest.raises(RecordTableError, match="finite"):
        document_json({"value": float("inf")})


@pytest.mark.parametrize("key", ["", None, 3])
def test_every_line_needs_a_non_empty_string_key(key: object) -> None:
    with pytest.raises(RecordTableError, match="key"):
        encode_record_table([{"key": key, "record": {}}])


def test_a_line_without_a_key_is_refused() -> None:
    with pytest.raises(RecordTableError, match="key"):
        encode_record_table([{"record": {}}])


def test_duplicate_keys_are_refused_instead_of_dropped() -> None:
    with pytest.raises(RecordTableError, match="duplicate key 'entity-a'"):
        encode_record_table([{"key": "entity-a", "record": {}}, {"key": "entity-a", "record": {}}])


def test_the_index_records_where_every_line_starts_and_how_long_it_is() -> None:
    table = encode_record_table(_lines())

    entries = [json.loads(line) for line in table.index.splitlines()]
    assert [entry["key"] for entry in entries] == ["entity-a", "entity-b", "entity-c"]
    for entry in entries:
        line = table.payload[entry["offset"] : entry["offset"] + entry["length"]]
        assert json.loads(line)["key"] == entry["key"]
        assert table.payload[entry["offset"] + entry["length"]] == ord("\n")


def test_a_table_reads_one_record_by_key(tmp_path: Path) -> None:
    table = _open(tmp_path)

    assert table.keys == ("entity-a", "entity-b", "entity-c")
    assert len(table) == 3
    assert "entity-b" in table and "entity-z" not in table
    assert table.read("entity-b") == {"key": "entity-b", "record": {"label": "door", "score": 0.5}}
    assert table.read("entity-c")["record"]["label"] == "cadeira á"


def test_a_table_iterates_every_line_in_order(tmp_path: Path) -> None:
    table = _open(tmp_path)

    assert [line["key"] for line in table.iter_lines()] == ["entity-a", "entity-b", "entity-c"]


def test_reading_one_record_does_not_parse_the_others(tmp_path: Path) -> None:
    payload, index = _write(tmp_path, _lines())
    data = bytearray(payload.read_bytes())
    damaged_at = data.index(b"entity-b") + 2
    data[damaged_at : damaged_at + 1] = b"\x00"
    payload.write_bytes(bytes(data))
    table = RecordTable(payload, index, record_count=3)

    assert table.read("entity-a")["key"] == "entity-a"
    with pytest.raises(BrokenIndexError, match="entity-b"):
        table.read("entity-b")


def test_an_unknown_key_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(RecordNotFoundError, match="entity-z"):
        _open(tmp_path).read("entity-z")


def test_an_empty_table_is_valid(tmp_path: Path) -> None:
    table = _open(tmp_path, [])

    assert table.keys == ()
    assert list(table.iter_lines()) == []
    assert encode_record_table([]).payload == b""


def test_a_missing_payload_or_index_is_an_explicit_error(tmp_path: Path) -> None:
    payload, index = _write(tmp_path, _lines())
    payload.unlink()
    with pytest.raises(MissingPayloadError, match=r"table\.jsonl"):
        RecordTable(payload, index, record_count=3)
    payload.write_bytes(encode_record_table(_lines()).payload)
    index.unlink()
    with pytest.raises(MissingPayloadError, match=r"table-index\.jsonl"):
        RecordTable(payload, index, record_count=3)


def test_a_truncated_payload_is_detected_when_the_table_opens(tmp_path: Path) -> None:
    payload, index = _write(tmp_path, _lines())
    payload.write_bytes(payload.read_bytes()[:-10])

    with pytest.raises(BrokenIndexError, match="bytes"):
        RecordTable(payload, index, record_count=3)


def test_a_record_count_that_disagrees_with_the_index_is_detected(tmp_path: Path) -> None:
    payload, index = _write(tmp_path, _lines())

    with pytest.raises(BrokenIndexError, match="record_count"):
        RecordTable(payload, index, record_count=4)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda entries: entries.reverse(), "sorted"),
        (lambda entries: entries[1].update(offset=entries[1]["offset"] + 1), "offset"),
        (lambda entries: entries[0].update(length=entries[0]["length"] + 1), "offset"),
        (lambda entries: entries[2].update(key="entity-b"), "sorted"),
        (lambda entries: entries[0].update(length=-1), "length"),
        (lambda entries: entries[0].update(key=""), "key"),
        (lambda entries: entries[0].update(extra=1), "fields"),
        (lambda entries: entries[0].pop("offset"), "fields"),
    ],
)
def test_a_damaged_index_is_an_explicit_error(tmp_path: Path, mutate: Any, message: str) -> None:
    payload, index = _write(tmp_path, _lines())
    entries = [json.loads(line) for line in index.read_bytes().splitlines()]
    mutate(entries)
    index.write_bytes(b"".join(canonical_json_line(entry) + b"\n" for entry in entries))

    with pytest.raises(BrokenIndexError, match=message):
        RecordTable(payload, index, record_count=3)


def test_an_index_line_that_is_not_json_is_an_explicit_error(tmp_path: Path) -> None:
    payload, index = _write(tmp_path, _lines())
    index.write_bytes(b"not json\n")

    with pytest.raises(BrokenIndexError, match="index"):
        RecordTable(payload, index, record_count=3)


def test_an_index_that_points_at_another_record_is_detected_on_read(tmp_path: Path) -> None:
    lines = [{"key": "entity-a", "record": {"n": 1}}, {"key": "entity-b", "record": {"n": 1}}]
    payload, index = _write(tmp_path, lines)
    first, second, _ = payload.read_bytes().split(b"\n")
    assert len(first) == len(second)
    payload.write_bytes(second + b"\n" + first + b"\n")
    table = RecordTable(payload, index, record_count=2)

    with pytest.raises(BrokenIndexError, match="entity-b"):
        table.read("entity-a")


def test_a_line_that_is_not_terminated_by_a_newline_is_detected(tmp_path: Path) -> None:
    payload, index = _write(tmp_path, _lines())
    data = bytearray(payload.read_bytes())
    first_end = data.index(b"\n")
    data[first_end : first_end + 1] = b" "
    payload.write_bytes(bytes(data))
    table = RecordTable(payload, index, record_count=3)

    with pytest.raises(BrokenIndexError, match="newline"):
        table.read("entity-a")


def test_the_table_errors_belong_to_the_artifact_error_family() -> None:
    for error in (BrokenIndexError, MissingPayloadError, RecordNotFoundError, RecordTableError):
        assert issubclass(error, ContextMapArtifactError)
