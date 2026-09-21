"""Package metadata that a release publishes: license, version source, entry points."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

import contextmap

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT: dict[str, Any] = tomllib.loads(
    (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)
PROJECT: dict[str, Any] = PYPROJECT["project"]

REPOSITORY_URL = "https://github.com/Alexandre-Tortoza/contextmap2"

# A CLI de runtime só existe em ``milestone/runtime-configuration``. Enquanto o
# módulo não faz parte desta árvore, o ponto de entrada não pode ser declarado
# (apontaria para um módulo inexistente); depois do merge, os testes abaixo
# passam a valer e exigem a declaração.
_RUNTIME_CLI_MISSING = importlib.util.find_spec("contextmap.runtime") is None
_RUNTIME_CLI_REASON = "contextmap.runtime is not on this branch yet; enabled by the runtime merge"


def test_version_comes_from_the_git_tag_only() -> None:
    assert PROJECT["dynamic"] == ["version"]
    assert "version" not in PROJECT
    assert "setuptools_scm" in PYPROJECT["tool"]


def test_runtime_version_is_the_installed_distribution_version() -> None:
    assert contextmap.__version__ == importlib.metadata.version("contextmap")


def test_license_is_an_spdx_expression_backed_by_the_license_file() -> None:
    assert PROJECT["license"] == "AGPL-3.0-only"
    assert PROJECT["license-files"] == ["LICENSE"]
    assert not [c for c in PROJECT["classifiers"] if c.startswith("License ::")], (
        "license classifiers are superseded by the SPDX expression"
    )
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3" in license_text


def test_project_urls_point_to_the_repository() -> None:
    urls: dict[str, str] = PROJECT["urls"]

    assert urls["Repository"] == REPOSITORY_URL
    assert urls["Issues"] == f"{REPOSITORY_URL}/issues"
    assert all(url.startswith(REPOSITORY_URL) for url in urls.values())


def test_python_classifiers_start_at_the_supported_floor() -> None:
    assert PROJECT["requires-python"] == ">=3.11"
    versions = [
        c.rsplit("::", 1)[1].strip()
        for c in PROJECT["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    ]

    assert versions, "declare every tested Python minor version"
    assert versions[0] == "3.11"
    assert versions == sorted(versions, key=lambda v: int(v.split(".")[1]))


@pytest.mark.skipif(_RUNTIME_CLI_MISSING, reason=_RUNTIME_CLI_REASON)
def test_console_script_points_to_the_runtime_cli() -> None:
    assert PROJECT["scripts"] == {"contextmap": "contextmap.runtime.cli:main"}

    cli = importlib.import_module("contextmap.runtime.cli")

    assert callable(cli.main)


@pytest.mark.skipif(_RUNTIME_CLI_MISSING, reason=_RUNTIME_CLI_REASON)
def test_module_entry_point_runs_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "contextmap", "--help"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "contextmap" in result.stdout.lower()
