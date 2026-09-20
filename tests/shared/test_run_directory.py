import json
from pathlib import Path

import pytest

from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
    file_entry,
    next_run_index,
    write_run_registry,
)


def test_file_entry_records_size_and_sha256() -> None:
    entry = file_entry("outputs/data.bin", b"abc")

    assert entry == FileEntry(
        path="outputs/data.bin",
        size_bytes=3,
        content_hash="sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )


def test_a_matching_inventory_reports_no_problem(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"abc")

    assert check_file_inventory(tmp_path, [file_entry("a.txt", b"abc")]) == []


def test_the_inventory_check_detects_missing_resized_and_altered_files(tmp_path: Path) -> None:
    (tmp_path / "resized.txt").write_bytes(b"abcd")
    (tmp_path / "altered.txt").write_bytes(b"xyz")
    inventory = [
        file_entry("missing.txt", b"abc"),
        file_entry("resized.txt", b"abc"),
        file_entry("altered.txt", b"abc"),
    ]

    problems = check_file_inventory(tmp_path, inventory)

    assert any("missing.txt" in problem and "missing" in problem for problem in problems)
    assert any("resized.txt" in problem and "size" in problem for problem in problems)
    assert any("altered.txt" in problem and "hash" in problem for problem in problems)


def test_publishing_makes_the_run_visible_only_when_complete(tmp_path: Path) -> None:
    final_dir = tmp_path / "runs" / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        run.write_text("outputs/result.json", '{"ok": true}\n')
        run.write_bytes("outputs/blob.bin", b"\x00\x01")
        assert not final_dir.exists()
        run.publish(manifest={"run_id": "r1"}, readme="# Run\n")

    assert (final_dir / "outputs" / "result.json").read_text() == '{"ok": true}\n'
    manifest = json.loads((final_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "r1"
    assert [entry["path"] for entry in manifest["file_inventory"]] == [
        "outputs/blob.bin",
        "outputs/result.json",
    ]
    assert (final_dir / "README.md").read_text() == "# Run\n"
    assert not list(final_dir.parent.glob(".tmp-*"))


def test_manifest_and_readme_are_not_part_of_their_own_inventory(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        run.write_text("outputs/a.txt", "a")
        run.publish(manifest={}, readme="# x\n")

    paths = [
        entry["path"]
        for entry in json.loads((final_dir / "manifest.json").read_text())["file_inventory"]
    ]
    assert "manifest.json" not in paths
    assert "README.md" not in paths


def test_non_contractual_files_are_written_but_not_inventoried(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        run.write_text("outputs/a.txt", "a")
        run.write_text("debug/notes.txt", "human only", contractual=False)
        run.publish(manifest={}, readme="# x\n")

    assert (final_dir / "debug" / "notes.txt").read_text() == "human only"
    paths = [
        entry["path"]
        for entry in json.loads((final_dir / "manifest.json").read_text())["file_inventory"]
    ]
    assert paths == ["outputs/a.txt"]
    (final_dir / "debug" / "notes.txt").unlink()
    manifest = json.loads((final_dir / "manifest.json").read_text())
    inventory = [FileEntry(**entry) for entry in manifest["file_inventory"]]
    assert check_file_inventory(final_dir, inventory) == []


def test_an_interrupted_write_never_looks_like_a_finished_run(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"

    with pytest.raises(RuntimeError, match="boom"), AtomicRunDirectory(final_dir) as run:
        run.write_text("outputs/a.txt", "a")
        raise RuntimeError("boom")

    assert not final_dir.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_leaving_without_publishing_discards_the_temporary_directory(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        run.write_text("outputs/a.txt", "a")

    assert not final_dir.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_an_existing_run_is_never_overwritten(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"
    final_dir.mkdir()
    (final_dir / "marker").write_text("immutable")

    with pytest.raises(RunDirectoryError, match="already exists"):
        AtomicRunDirectory(final_dir)

    assert (final_dir / "marker").read_text() == "immutable"


@pytest.mark.parametrize("bad_path", ["../escape.txt", "/absolute.txt", "a/../../b.txt", ""])
def test_writes_outside_the_run_directory_are_rejected(tmp_path: Path, bad_path: str) -> None:
    with (
        AtomicRunDirectory(tmp_path / "run-0001") as run,
        pytest.raises(RunDirectoryError, match="relative path"),
    ):
        run.write_text(bad_path, "x")


def test_the_same_path_cannot_be_written_twice(tmp_path: Path) -> None:
    with AtomicRunDirectory(tmp_path / "run-0001") as run:
        run.write_text("outputs/a.txt", "a")
        with pytest.raises(RunDirectoryError, match="already written"):
            run.write_text("outputs/a.txt", "b")


def _index_of(run_dir: Path) -> int | None:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    index: int = json.loads(manifest.read_text())["run_index"]
    return index


def _make_run(sequence_dir: Path, name: str, index: int, *, complete: bool = True) -> None:
    run_dir = sequence_dir / name
    run_dir.mkdir(parents=True)
    if complete:
        (run_dir / "manifest.json").write_text(json.dumps({"run_index": index}))


def test_run_indexes_start_at_one_and_increase_monotonically(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "corridor"

    assert next_run_index(sequence_dir, index_of=_index_of) == 1

    _make_run(sequence_dir, "run-0001__a__b", 1)
    _make_run(sequence_dir, "run-0004__a__b", 4)
    assert next_run_index(sequence_dir, index_of=_index_of) == 5


def test_incomplete_and_temporary_runs_are_never_counted(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "corridor"
    _make_run(sequence_dir, "run-0001__a__b", 1)
    _make_run(sequence_dir, "run-0009__a__b", 9, complete=False)
    _make_run(sequence_dir, ".tmp-run-0010__a__b-abc", 10)

    assert next_run_index(sequence_dir, index_of=_index_of) == 2


def test_the_registry_is_rebuilt_from_valid_runs_only(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "corridor"
    _make_run(sequence_dir, "run-0001__a__b", 1)
    _make_run(sequence_dir, "run-0002__a__b", 2, complete=False)

    write_run_registry(
        sequence_dir,
        describe=lambda run_dir: (
            {"run_index": _index_of(run_dir), "directory": run_dir.name}
            if _index_of(run_dir) is not None
            else None
        ),
    )

    registry = json.loads((sequence_dir / "runs.json").read_text())
    assert registry == {"runs": [{"directory": "run-0001__a__b", "run_index": 1}]}
