"""The incremental-context documentation names only APIs and commands that exist (issue #503)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import contextmap.runtime as runtime
from contextmap.runtime.cli import main

DECISION = Path(__file__).resolve().parents[2] / "docs" / "runtime-composition.md"
CLI_DOC = Path(__file__).resolve().parents[2] / "src/contextmap/runtime/docs/cli.md"

DOCUMENTED_API = (
    "resolve_spatial_foundation",
    "foundation_of_run",
    "create_branch",
    "open_branch",
    "append_to_branch",
    "context_scope",
    "publish_context_run",
    "read_context_run",
    "plan_context_build",
    "publish_context_build",
    "read_context_build",
)
DOCUMENTED_COMMANDS = (
    ("context", "branch", "create"),
    ("context", "run"),
    ("context", "build"),
    ("context", "inspect", "branch"),
    ("context", "inspect", "run"),
)


@pytest.mark.parametrize("name", DOCUMENTED_API)
def test_every_documented_function_is_public_and_documented(name: str) -> None:
    assert name in runtime.__all__
    assert f"`{name}" in DECISION.read_text(encoding="utf-8")


@pytest.mark.parametrize("command", DOCUMENTED_COMMANDS)
def test_every_documented_command_exists(command: tuple[str, ...]) -> None:
    out, err = io.StringIO(), io.StringIO()

    code = main([*command, "--help"], stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert f"contextmap {' '.join(command)}" in CLI_DOC.read_text(encoding="utf-8")
