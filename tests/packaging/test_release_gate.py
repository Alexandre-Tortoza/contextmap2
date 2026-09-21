"""Behavior of the release gate script against throwaway git repositories."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "verify_release_tag.sh"

# Isola o teste da configuração global de git da máquina (assinatura, hooks, branch inicial).
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "Release Gate Test",
    "GIT_AUTHOR_EMAIL": "gate@example.invalid",
    "GIT_COMMITTER_NAME": "Release Gate Test",
    "GIT_COMMITTER_EMAIL": "gate@example.invalid",
}


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        env=_GIT_ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _commit(repository: Path, message: str) -> str:
    _git(repository, "commit", "--allow-empty", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "main")
    _commit(tmp_path, "root")
    return tmp_path


def _gate(repository: Path, tag: str, commit: str, main_ref: str = "main") -> tuple[int, str]:
    result = subprocess.run(
        ["bash", str(SCRIPT), tag, commit, main_ref],
        cwd=repository,
        env=_GIT_ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def test_accepts_a_tag_on_a_commit_that_is_on_main(repository: Path) -> None:
    released = _commit(repository, "release candidate")
    _git(repository, "switch", "-c", "dev")
    _commit(repository, "work after the release")

    code, output = _gate(repository, "v0.1.0", released)

    assert code == 0, output


def test_rejects_a_commit_that_is_not_on_main(repository: Path) -> None:
    _git(repository, "switch", "-c", "dev")
    only_on_dev = _commit(repository, "not promoted yet")

    code, output = _gate(repository, "v0.1.0", only_on_dev)

    assert code == 1
    assert "not on main" in output


def test_rejects_when_main_does_not_exist(repository: Path) -> None:
    head = _git(repository, "rev-parse", "HEAD")

    code, output = _gate(repository, "v0.1.0", head, main_ref="refs/remotes/origin/main")

    assert code == 1
    assert "main" in output


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1", "v0.1.0-rc1", "v1.2.3.4", "release-1"])
def test_rejects_tags_that_are_not_strict_semantic_versions(repository: Path, tag: str) -> None:
    head = _git(repository, "rev-parse", "HEAD")

    code, output = _gate(repository, tag, head)

    assert code == 1
    assert "vMAJOR.MINOR.PATCH" in output
