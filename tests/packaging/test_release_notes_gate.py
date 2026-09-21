"""Behavior of the release-notes gate: no release without a dated changelog and final notes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "verify_release_notes.sh"

DATED_CHANGELOG = "# Changelog\n\n## [Não lançado]\n\n## [0.1.0] - 2026-10-01\n\n- notas\n"
FINAL_NOTES = "# ContextMap2 v0.1.0\n\nNotas finais.\n"
DRAFT_NOTES = "> **Status: rascunho.** Não publicar.\n\n# ContextMap2 v0.1.0\n"


def _repository(root: Path, *, changelog: str | None, notes: str | None) -> Path:
    if changelog is not None:
        (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    if notes is not None:
        (root / "docs" / "releases").mkdir(parents=True)
        (root / "docs" / "releases" / "v0.1.0.md").write_text(notes, encoding="utf-8")
    return root


def _gate(root: Path, tag: str = "v0.1.0") -> tuple[int, str]:
    result = subprocess.run(
        ["bash", str(SCRIPT), tag, str(root)], capture_output=True, text=True, check=False
    )
    return result.returncode, result.stdout + result.stderr


def test_accepts_a_dated_changelog_entry_and_final_notes(tmp_path: Path) -> None:
    code, output = _gate(_repository(tmp_path, changelog=DATED_CHANGELOG, notes=FINAL_NOTES))

    assert code == 0, output


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n\n## [Não lançado]\n",
        "# Changelog\n\n## [0.1.0] - não lançado\n",
        "# Changelog\n\n## [0.10.0] - 2026-10-01\n",
    ],
    ids=["no entry", "undated entry", "another version"],
)
def test_rejects_a_changelog_without_a_dated_entry_for_the_tag(
    tmp_path: Path, changelog: str
) -> None:
    code, output = _gate(_repository(tmp_path, changelog=changelog, notes=FINAL_NOTES))

    assert code == 1
    assert "CHANGELOG.md" in output


def test_rejects_a_missing_changelog(tmp_path: Path) -> None:
    code, output = _gate(_repository(tmp_path, changelog=None, notes=FINAL_NOTES))

    assert code == 1
    assert "CHANGELOG.md" in output


def test_rejects_missing_release_notes(tmp_path: Path) -> None:
    code, output = _gate(_repository(tmp_path, changelog=DATED_CHANGELOG, notes=None))

    assert code == 1
    assert "docs/releases/v0.1.0.md" in output


def test_rejects_release_notes_that_are_still_a_draft(tmp_path: Path) -> None:
    code, output = _gate(_repository(tmp_path, changelog=DATED_CHANGELOG, notes=DRAFT_NOTES))

    assert code == 1
    assert "draft" in output
