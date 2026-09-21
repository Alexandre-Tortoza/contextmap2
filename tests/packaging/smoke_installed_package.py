"""Smoke test for an *installed* ContextMap2 distribution (wheel or sdist).

Run it with the interpreter of a fresh virtual environment that holds only the
built distribution (plus the extras under test), never with the development
environment: it checks what a consumer gets, not what the source tree contains.

    python tests/packaging/smoke_installed_package.py --expect-version 0.1.0
    python tests/packaging/smoke_installed_package.py --extra ros1

It uses the standard library only, so it also runs in the lightest environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import pkgutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# Adapters que importam o SDK opcional já no import (ver ingestion/adapters/__init__.py).
_ROSBAGS_ADAPTERS = (
    "contextmap.ingestion.adapters.ros1_bag",
    "contextmap.ingestion.adapters.ros2_bag",
)
_EXTRA_MODULES = {"ros1": ("rosbags",), "ros2": ("rosbags",)}
# Módulos que a instalação base nunca deve arrastar.
_HEAVY_MODULES = ("torch", "transformers", "PIL", "rosbags", "rclpy")
_FALLBACK_VERSION = "0.0.0"


def _check_installed_distribution(expected_version: str | None) -> list[str]:
    problems: list[str] = []
    import contextmap

    location = Path(contextmap.__file__).resolve()
    if "site-packages" not in location.parts and "dist-packages" not in location.parts:
        problems.append(f"contextmap was imported from a source tree, not an install: {location}")

    installed = importlib.metadata.version("contextmap")
    if contextmap.__version__ != installed:
        problems.append(
            f"contextmap.__version__ ({contextmap.__version__}) differs from "
            f"the distribution metadata ({installed})"
        )
    if installed == _FALLBACK_VERSION:
        problems.append("version is the setuptools-scm fallback: the build had no git metadata")
    if expected_version is not None and installed != expected_version:
        problems.append(f"installed version {installed} != expected {expected_version}")
    return problems


def _check_base_requirements() -> list[str]:
    requirements = importlib.metadata.requires("contextmap") or []
    base = [entry for entry in requirements if "extra ==" not in entry]
    if len(base) != 1 or not base[0].lower().startswith("numpy"):
        return [f"the base install must depend on NumPy only, found: {base}"]
    return []


def _check_optional_modules(extras: Sequence[str]) -> list[str]:
    expected_present = {module for extra in extras for module in _EXTRA_MODULES[extra]}
    problems = []
    for module in _HEAVY_MODULES:
        present = importlib.util.find_spec(module) is not None
        if module in expected_present and not present:
            problems.append(f"extra {list(extras)} did not install {module}")
        if module not in expected_present and present:
            problems.append(f"{module} is installed but no selected extra provides it")
    return problems


def _check_imports(extras: Sequence[str]) -> list[str]:
    import contextmap

    has_rosbags = any("rosbags" in _EXTRA_MODULES[extra] for extra in extras)
    skipped = set() if has_rosbags else set(_ROSBAGS_ADAPTERS)
    problems = []
    imported = 0
    for info in pkgutil.walk_packages(contextmap.__path__, prefix="contextmap."):
        if info.name in skipped:
            continue
        try:
            importlib.import_module(info.name)
        except Exception as error:
            problems.append(f"cannot import {info.name}: {type(error).__name__}: {error}")
        else:
            imported += 1
    print(f"imported {imported} modules ({len(skipped)} adapters need an extra)")
    return problems


def _check_console_scripts() -> list[str]:
    problems = []
    distribution = importlib.metadata.distribution("contextmap")
    scripts = [entry for entry in distribution.entry_points if entry.group == "console_scripts"]
    for entry in scripts:
        try:
            if not callable(entry.load()):
                problems.append(f"console script {entry.name} does not point to a callable")
        except Exception as error:
            problems.append(f"console script {entry.name} cannot be loaded: {error}")
            continue
        executable = Path(sys.executable).parent / entry.name
        result = subprocess.run(
            [str(executable), "--help"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            problems.append(f"`{entry.name} --help` exited {result.returncode}: {result.stderr}")
        installed = importlib.metadata.version("contextmap")
        reported = subprocess.run(
            [str(executable), "--version"], capture_output=True, text=True, check=False
        )
        if reported.returncode != 0 or installed not in reported.stdout:
            problems.append(
                f"`{entry.name} --version` must report {installed}, got {reported.stdout!r}"
            )
    print(f"checked {len(scripts)} console script(s)")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Run every check and return a process exit code (0 when all pass)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--expect-version", help="version the distribution must report")
    parser.add_argument(
        "--extra",
        action="append",
        choices=sorted(_EXTRA_MODULES),
        default=[],
        help="extra installed next to the base package (repeatable)",
    )
    args = parser.parse_args(argv)

    problems = [
        *_check_installed_distribution(args.expect_version),
        *_check_base_requirements(),
        *_check_optional_modules(args.extra),
        *_check_imports(args.extra),
        *_check_console_scripts(),
    ]
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if not problems:
        print(f"OK: contextmap {importlib.metadata.version('contextmap')} smoke test passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
