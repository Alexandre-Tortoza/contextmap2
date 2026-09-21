"""Installation contract: what the package declares is what the code imports.

The base install must stay light (NumPy only) and every optional third-party
module the source can import must be tied to a declared extra, or be listed
here as an external runtime that PyPI cannot provide.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "contextmap"

PROJECT: dict[str, Any] = tomllib.loads(
    (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]

# Nome de distribuição no PyPI -> nome do módulo importado, quando diferem.
_IMPORT_ROOT_OF_DISTRIBUTION = {"pillow": "PIL"}

# Runtimes que o código importa, mas que o PyPI não distribui (verificado em
# 2026-09-21): a instalação é feita pelo usuário a partir do repositório do
# fornecedor, e o motivo de não haver extra fica em docs/installation.md.
EXTERNAL_RUNTIME_MODULES = {
    "alpha_clip": "AlphaCLIP is installed from its GitHub repository, not from PyPI",
    "sam2": "SAM 2 is installed from its GitHub repository, not from PyPI",
}

# Módulos que importam um SDK opcional já no import: só são alcançados pelo
# caminho pontilhado completo, nunca reexportados (ver ingestion/adapters/__init__).
IMPORT_TIME_OPTIONAL_MODULES = frozenset(
    {
        "contextmap.ingestion.adapters.ros1_bag",
        "contextmap.ingestion.adapters.ros2_bag",
    }
)


def _distribution_names(requirements: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(entry).name) for entry in requirements}


def _import_roots(requirements: list[str]) -> set[str]:
    roots = set()
    for name in _distribution_names(requirements):
        roots.add(_IMPORT_ROOT_OF_DISTRIBUTION.get(name, name.replace("-", "_")))
    return roots


def _feature_extras() -> dict[str, list[str]]:
    extras: dict[str, list[str]] = PROJECT["optional-dependencies"]
    return {name: entries for name, entries in extras.items() if name != "dev"}


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level modules a file imports, including lazy ``import_module``."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and _is_import_module_call(node):
            argument = node.args[0] if node.args else None
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                roots.add(argument.value.split(".")[0])
    return roots


def _is_import_module_call(node: ast.Call) -> bool:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id == "import_module"
    return isinstance(function, ast.Attribute) and function.attr == "import_module"


def test_base_install_depends_only_on_numpy() -> None:
    assert _distribution_names(PROJECT["dependencies"]) == {"numpy"}


def test_declared_extras_cover_the_advertised_capabilities() -> None:
    extras = _feature_extras()

    assert {"ros1", "ros2", "vision"} <= set(extras)
    assert _distribution_names(extras["ros1"]) == {"rosbags"}
    assert extras["ros2"] == extras["ros1"], "both ROS adapters read bags through rosbags"
    assert {"torch", "torchvision", "transformers", "pillow"} <= _distribution_names(
        extras["vision"]
    )


def test_every_third_party_import_is_declared_or_documented_as_external() -> None:
    declared = _import_roots(PROJECT["dependencies"])
    for entries in _feature_extras().values():
        declared |= _import_roots(entries)
    allowed = declared | set(EXTERNAL_RUNTIME_MODULES) | {"contextmap"}
    stdlib = set(sys.stdlib_module_names)

    undeclared: dict[str, set[str]] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        unknown = _imported_roots(path) - stdlib - allowed
        if unknown:
            undeclared[str(path.relative_to(REPOSITORY_ROOT))] = unknown

    assert not undeclared, f"imports without a declared extra: {undeclared}"


def test_base_install_can_import_every_module_without_optional_dependencies() -> None:
    # Bloqueia qualquer módulo de terceiros que não seja o NumPy, para que uma
    # dependência opcional importada no topo de um módulo falhe aqui e não no
    # ambiente de um consumidor leve do artefato.
    program = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import pkgutil
        import sys

        ALLOWED = set(sys.stdlib_module_names) | {"contextmap", "numpy"}
        EXEMPT = set(sys.argv[1:])


        class BlockThirdParty(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] not in ALLOWED:
                    raise ModuleNotFoundError(f"blocked: {fullname}", name=fullname)
                return None


        sys.meta_path.insert(0, BlockThirdParty())
        import contextmap

        failures = []
        for info in pkgutil.walk_packages(contextmap.__path__, prefix="contextmap."):
            if info.name in EXEMPT:
                continue
            try:
                importlib.import_module(info.name)
            except Exception as error:
                failures.append(f"{info.name}: {type(error).__name__}: {error}")
        print("\\n".join(failures))
        sys.exit(1 if failures else 0)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program, *sorted(IMPORT_TIME_OPTIONAL_MODULES)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
