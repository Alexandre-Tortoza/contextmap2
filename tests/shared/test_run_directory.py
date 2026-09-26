import json
from pathlib import Path

import pytest

from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
    file_entry,
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


def test_a_streamed_file_is_inventoried_like_one_written_at_once(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        with run.open_binary("outputs/big.bin") as handle:
            handle.write(b"abc")
            handle.write(b"def")
        run.publish(manifest={}, readme="# x\n")

    manifest = json.loads((final_dir / "manifest.json").read_text())
    assert manifest["file_inventory"] == [
        {
            "path": "outputs/big.bin",
            "size_bytes": 6,
            "content_hash": file_entry("outputs/big.bin", b"abcdef").content_hash,
        }
    ]
    assert (final_dir / "outputs" / "big.bin").read_bytes() == b"abcdef"


def test_a_streamed_file_larger_than_the_hashing_chunk_is_hashed_whole(tmp_path: Path) -> None:
    payload = bytes(range(256)) * (3 * 4096)  # 3 MiB
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        with run.open_binary("outputs/big.bin") as handle:
            for start in range(0, len(payload), 100_000):
                handle.write(payload[start : start + 100_000])
        run.publish(manifest={}, readme="# x\n")

    entry = FileEntry(**json.loads((final_dir / "manifest.json").read_text())["file_inventory"][0])
    assert entry == file_entry("outputs/big.bin", payload)
    assert check_file_inventory(final_dir, [entry]) == []


def test_the_inventory_check_detects_a_change_in_the_last_byte_of_a_large_file(
    tmp_path: Path,
) -> None:
    payload = bytes(range(256)) * (3 * 4096)
    # O último byte original é 0xff; troca-se por 0x00.
    (tmp_path / "big.bin").write_bytes(payload[:-1] + b"\x00")

    problems = check_file_inventory(tmp_path, [file_entry("big.bin", payload)])

    assert any("hash" in problem for problem in problems)


def test_a_non_contractual_stream_stays_out_of_the_inventory(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        with run.open_binary("debug/samples.bin", contractual=False) as handle:
            handle.write(b"human only")
        run.publish(manifest={}, readme="# x\n")

    assert json.loads((final_dir / "manifest.json").read_text())["file_inventory"] == []
    assert (final_dir / "debug" / "samples.bin").read_bytes() == b"human only"


def test_a_written_file_can_be_read_back_before_publishing(tmp_path: Path) -> None:
    with AtomicRunDirectory(tmp_path / "run-0001") as run:
        with run.open_binary("outputs/big.bin") as handle:
            handle.write(b"abc")

        assert run.written_path("outputs/big.bin").read_bytes() == b"abc"
        with pytest.raises(RunDirectoryError, match="not been written"):
            run.written_path("outputs/other.bin")


def test_the_inventory_is_known_before_publishing_and_is_the_one_published(
    tmp_path: Path,
) -> None:
    # Um manifest cuja identidade cobre o inventário precisa dele antes de publicar (#600).
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        run.write_bytes("outputs/b.txt", b"abc")
        with run.open_binary("outputs/a.bin") as handle:
            handle.write(b"def")
            assert run.inventory() == (file_entry("outputs/b.txt", b"abc"),)
        run.write_text("debug/notes.txt", "human only", contractual=False)
        inventory = run.inventory()
        run.publish(manifest={}, readme="# x\n")

    assert inventory == (file_entry("outputs/a.bin", b"def"), file_entry("outputs/b.txt", b"abc"))
    published = json.loads((final_dir / "manifest.json").read_text())["file_inventory"]
    assert tuple(FileEntry(**item) for item in published) == inventory


def test_publishing_while_a_stream_is_open_is_refused(tmp_path: Path) -> None:
    with (
        AtomicRunDirectory(tmp_path / "run-0001") as run,
        run.open_binary("outputs/big.bin") as out,
    ):
        out.write(b"abc")
        with pytest.raises(RunDirectoryError, match="open"):
            run.publish(manifest={}, readme="# x\n")


def test_a_stream_that_failed_can_never_be_published(tmp_path: Path) -> None:
    final_dir = tmp_path / "run-0001"

    with AtomicRunDirectory(final_dir) as run:
        with (
            pytest.raises(RuntimeError, match="boom"),
            run.open_binary("outputs/big.bin") as handle,
        ):
            handle.write(b"partial")
            raise RuntimeError("boom")

        with pytest.raises(RunDirectoryError, match="failed"):
            run.publish(manifest={}, readme="# x\n")

    assert not final_dir.exists()


def test_a_streamed_path_follows_the_same_rules_as_a_written_one(tmp_path: Path) -> None:
    with AtomicRunDirectory(tmp_path / "run-0001") as run:
        run.write_text("outputs/a.txt", "a")
        with (
            pytest.raises(RunDirectoryError, match="already written"),
            run.open_binary("outputs/a.txt"),
        ):
            pass
        with pytest.raises(RunDirectoryError, match="relative path"), run.open_binary("../x"):
            pass


@pytest.mark.parametrize("path", ["../outside.bin", "/etc/hostname", ""])
def test_the_inventory_check_never_reads_outside_the_run(tmp_path: Path, path: str) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "outside.bin").write_bytes(b"x")

    problems = check_file_inventory(run, [file_entry(path, b"x")])

    assert problems == [f"invalid path in manifest: {path!r}"]
