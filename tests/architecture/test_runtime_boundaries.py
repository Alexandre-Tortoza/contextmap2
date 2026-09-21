"""The runtime composes; only its composition root knows other capabilities and backends.

``test_boundaries.py`` lets the runtime import anything, because a composition root has to
name concrete backends. These tests narrow that permission: the composition root may import
capabilities, lazily; the ingestion application service may import the ingestion capability's
public root (and nothing below it, so never an adapter); every other runtime module, meaning
configuration, DAG, reuse, selection, lifecycle and CLI code, can never grow a dependency on a
capability or on a concrete backend.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
RUNTIME = SRC / "contextmap" / "runtime"
COMPOSITION = RUNTIME / "composition.py"
# Módulos de aplicação que podem importar a raiz pública de uma capability, e só ela.
PUBLIC_ROOT_IMPORTERS = {"ingestion_service.py": {"contextmap.ingestion"}}
CAPABILITIES = frozenset(
    {
        "ingestion",
        "visual_perception",
        "state_estimation",
        "geometric_mapping",
        "sensor_association",
        "point_representation",
        "semantic_fusion",
        "semantic_mapping",
        "entity_resolution",
        "spatial_relations",
        "artifact",
        "evaluation",
    }
)
BACKEND_PARTS = frozenset({"backends", "adapters", "infrastructure"})


def _module_level_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Imports that run when the module is imported, skipping ``if TYPE_CHECKING`` blocks."""
    found: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def _all_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def _capability_of(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "contextmap" and parts[1] in CAPABILITIES:
        return parts[1]
    return None


def _runtime_modules() -> list[Path]:
    return sorted(path for path in RUNTIME.glob("*.py"))


def test_only_the_composition_root_imports_a_capability() -> None:
    violations = []
    for path in _runtime_modules():
        if path == COMPOSITION:
            continue
        allowed = PUBLIC_ROOT_IMPORTERS.get(path.name, set())
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, module in _all_imports(tree):
            capability = _capability_of(module)
            if capability is not None and module not in allowed:
                violations.append(f"{path.name}:{line} imports {module} ({capability})")

    assert not violations, "\n".join(violations)


def test_the_composition_root_imports_capabilities_lazily() -> None:
    tree = ast.parse(COMPOSITION.read_text(encoding="utf-8"))

    eager = [
        f"composition.py:{line} imports {module} at module level"
        for line, module in _module_level_imports(tree)
        if _capability_of(module) is not None
    ]

    assert not eager, "\n".join(eager)


def test_concrete_backends_are_imported_only_inside_the_factory_that_uses_them() -> None:
    tree = ast.parse(COMPOSITION.read_text(encoding="utf-8"))
    inside_functions: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            inside_functions.update(
                child.lineno for child in ast.walk(node) if isinstance(child, ast.ImportFrom)
            )

    misplaced = []
    for line, module in _all_imports(tree):
        parts = set(module.split("."))
        is_backend = _capability_of(module) is not None and bool(parts & BACKEND_PARTS)
        if is_backend and line not in inside_functions:
            misplaced.append(f"composition.py:{line} imports backend {module} outside a factory")

    assert not misplaced, "\n".join(misplaced)


def test_importing_the_runtime_loads_no_backend_no_sdk_and_only_the_ingestion_root() -> None:
    code = (
        "import sys\n"
        "import contextmap.runtime, contextmap.runtime.cli\n"
        f"capabilities = {sorted(CAPABILITIES)!r}\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('contextmap.') "
        "and m.split('.')[1] in capabilities)\n"
        "sdks = [m for m in ('torch', 'transformers', 'rosbags', 'numpy') if m in sys.modules]\n"
        "print('\\n'.join(loaded + sdks))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(SRC), "PATH": ""},
    )

    loaded = result.stdout.split()
    assert not [m for m in loaded if not m.startswith("contextmap.ingestion")], loaded
    assert not [m for m in loaded if ".adapters" in m or ".backends" in m], loaded


def test_the_public_api_module_depends_only_on_the_standard_library_and_the_runtime() -> None:
    tree = ast.parse((RUNTIME / "api.py").read_text(encoding="utf-8"))

    foreign = sorted(
        {
            module
            for _, module in _all_imports(tree)
            if module.split(".")[0] not in sys.stdlib_module_names
            and module != "contextmap"
            and not module.startswith("contextmap.runtime")
        }
    )

    # Nenhuma capability, backend, adaptador ou biblioteca de terceiros entra na API pública.
    assert not foreign, foreign


def test_no_runtime_module_depends_on_a_ui_framework() -> None:
    frameworks = {
        "textual",
        "typer",
        "click",
        "rich",
        "prompt_toolkit",
        "urwid",
        "curses",
        "tkinter",
    }
    violations = [
        f"{path.name}:{line} imports {module}"
        for path in _runtime_modules()
        for line, module in _all_imports(ast.parse(path.read_text(encoding="utf-8")))
        if module.split(".")[0] in frameworks or module.split(".")[0].startswith(("PyQt", "PySide"))
    ]

    assert not violations, "\n".join(violations)


def test_the_public_api_exposes_no_ros_native_type_and_no_backend_class() -> None:
    import contextmap.runtime as runtime
    from contextmap.runtime import api

    contracts = [getattr(api, name) for name in api.__all__ if isinstance(getattr(api, name), type)]
    foreign = {
        f"{contract.__name__}.{annotation}"
        for contract in contracts
        for annotation in getattr(contract, "__annotations__", {}).values()
        if any(token in str(annotation) for token in ("rosbags", "rclpy", "rospy", "torch", "PIL"))
    }

    assert foreign == set()
    assert set(api.__all__) <= set(runtime.__all__)


def test_the_runtime_public_api_hides_the_concrete_backend_classes() -> None:
    import contextmap.runtime as runtime

    exported = set(runtime.__all__)
    concrete = {
        name
        for name in exported
        if name.startswith(("Sam", "Dino", "Clip", "Qwen", "Gemini", "Florence", "FastLio", "PTv3"))
    }

    assert concrete == set()
