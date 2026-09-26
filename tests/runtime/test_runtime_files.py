"""Tests for the durable, atomic publication of the runtime's small JSON documents."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from contextmap.runtime._files import publish_text, replace_text

_Identity = tuple[int, int]


def _identity(path: Path) -> _Identity | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


class _SyncRecorder:
    """Records every ``os.fsync`` together with what the published path was at that moment.

    Identities are ``(device, inode)``: a link or a replace keeps them, so the temporary file
    that was synced is recognized under its published name.
    """

    def __init__(self, published: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._published = published
        self._calls: list[tuple[_Identity, _Identity | None]] = []
        real_fsync = os.fsync

        def recording_fsync(descriptor: int) -> None:
            status = os.fstat(descriptor)
            self._calls.append(((status.st_dev, status.st_ino), _identity(published)))
            real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", recording_fsync)

    def synced_before_publication(self, path: Path) -> bool:
        target, published = _identity(path), _identity(self._published)
        return any(synced == target and seen != published for synced, seen in self._calls)

    def synced_after_publication(self, path: Path) -> bool:
        target, published = _identity(path), _identity(self._published)
        return any(synced == target and seen == published for synced, seen in self._calls)


def test_a_published_document_reaches_the_disk_before_its_name_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sem fsync, uma queda de energia pode persistir o link antes dos dados (#616).
    recorder = _SyncRecorder(tmp_path / "plan.json", monkeypatch)

    published = publish_text(tmp_path, "plan.json", '{"ok": true}\n')

    assert published.read_text(encoding="utf-8") == '{"ok": true}\n'
    assert recorder.synced_before_publication(published)
    assert recorder.synced_after_publication(tmp_path)


def test_a_replaced_document_reaches_the_disk_before_it_replaces_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "status.json").write_text("{}\n", encoding="utf-8")
    recorder = _SyncRecorder(tmp_path / "status.json", monkeypatch)

    replaced = replace_text(tmp_path, "status.json", '{"status": "running"}\n')

    assert replaced.read_text(encoding="utf-8") == '{"status": "running"}\n'
    assert recorder.synced_before_publication(replaced)
    assert recorder.synced_after_publication(tmp_path)


def test_a_document_that_cannot_reach_the_disk_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_fsync(descriptor: int) -> None:
        raise OSError(errno.EIO, "simulated I/O error")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="simulated"):
        publish_text(tmp_path, "plan.json", "{}\n")

    assert list(tmp_path.iterdir()) == []
