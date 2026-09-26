"""Release hygiene: nothing generated, private, oversized or machine-specific is tracked."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

MAX_TRACKED_FILE_BYTES = 2 * 1024 * 1024
# Pesos de modelo, bags e nuvens de pontos: o conteúdo não é redistribuível por este
# repositório (docs/third-party-licenses.md) e o setuptools-scm o empacotaria no sdist.
FORBIDDEN_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".gguf", ".bag", ".db3", ".mcap"}
    | {".npz", ".pcd", ".ply"}
)
# Padrões de credenciais e de caminhos de máquina que nunca devem aparecer em um release.
LEAK_PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "local home directory": re.compile(r"(?:/home|/Users)/[A-Za-z_][A-Za-z0-9_-]*/"),
}

# Caminhos que precisam estar ignorados: dados de pesquisa, workspaces locais, segredos.
MUST_BE_IGNORED = (
    "workspace/corridor-02/run-0001/manifest.json",
    "datasets/corridor-02/calibration.yaml",
    ".claude/worktrees/agent-1/notes.md",
    "outputs/validation/report.json",
    "runs/run-0001/config.json",
    "artifacts/context-map/manifest.json",
    "checkpoints/model.ckpt",
    "weights/model.safetensors",
    "weights/model.gguf",
    "recording/data.mcap",
    ".env",
    ".env.local",
    "secrets/token.txt",
    "certs/private.pem",
    "certs/private.key",
)


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPOSITORY_ROOT, capture_output=True, check=False
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return [REPOSITORY_ROOT / name for name in result.stdout.decode().split("\0") if name]


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_research_data_workspaces_and_secrets_are_ignored(path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", path], cwd=REPOSITORY_ROOT, check=False
    )

    assert result.returncode == 0, f"{path} is not covered by .gitignore"


def test_no_example_file_is_ignored() -> None:
    """Every file of a committed example must actually be committed.

    The release examples are fixtures, and an artifact's contractual layout has an ``outputs/``
    directory -- the same name ``.gitignore`` uses to keep research outputs out of the tree. The
    first version of the demo bundle lost 20 of its 52 files to that rule and was published
    invalid, while the test that validates it passed locally because the files were still on
    disk. Only a clean checkout showed it, so this asserts the property directly.
    """
    examples = REPOSITORY_ROOT / "examples"
    if not examples.is_dir():
        pytest.skip("there is no examples directory")
    present = sorted(path for path in examples.rglob("*") if path.is_file())
    tracked = {
        REPOSITORY_ROOT / name
        for name in subprocess.run(
            ["git", "ls-files", "-z", "examples"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
        )
        .stdout.decode()
        .split("\0")
        if name
    }

    assert present, "the examples directory has no file"
    assert [str(path.relative_to(REPOSITORY_ROOT)) for path in present if path not in tracked] == []


def test_no_tracked_file_is_a_model_weight_a_recording_or_oversized() -> None:
    offenders = []
    for path in _tracked_files():
        forbidden = path.suffix.lower() in FORBIDDEN_SUFFIXES
        oversized = path.stat().st_size > MAX_TRACKED_FILE_BYTES
        if forbidden or oversized:
            offenders.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert not offenders, f"tracked files that must not be released: {offenders}"


def test_no_tracked_file_contains_a_credential_or_a_local_path() -> None:
    leaks = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in LEAK_PATTERNS.items():
            if pattern.search(text):
                leaks.append(f"{path.relative_to(REPOSITORY_ROOT)}: {label}")

    assert not leaks, f"possible leaks in tracked files: {leaks}"
