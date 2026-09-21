from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

CAPABILITIES = frozenset(
    {
        "shared",
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
        "runtime",
        "evaluation",
    }
)

# Dependências abaixo representam imports permitidos pelo consumer para a API
# pública do producer. O sentido é, portanto, o inverso do fluxo de dados
# desenhado como producer -> consumer na documentação arquitetural.
ALLOWED_DEPENDENCIES: dict[str, frozenset[str]] = {
    "shared": frozenset(),
    "ingestion": frozenset({"shared"}),
    "visual_perception": frozenset({"shared", "ingestion"}),
    "state_estimation": frozenset({"shared", "ingestion"}),
    "geometric_mapping": frozenset({"shared", "ingestion", "state_estimation"}),
    "sensor_association": frozenset(
        {
            "shared",
            "ingestion",
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
        }
    ),
    "point_representation": frozenset({"shared", "geometric_mapping", "sensor_association"}),
    "semantic_fusion": frozenset(
        {
            "shared",
            "ingestion",
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "point_representation",
        }
    ),
    # Além da API pública de semantic_fusion, Semantic Mapping usa identidades de ingestion
    # (FrameId, SourceObservationId), visual_perception (BackendProvenance, ClaimId, FeatureId...),
    # sensor_association (SpatialObservationId), point_representation (PointRepresentationId) e
    # state_estimation (TimeBounds) apenas como tipos: as evidências das hipóteses e as
    # referências à evidência fundida chegam com esses tipos e são preservadas, sem usar a
    # lógica dessas capabilities.
    "semantic_mapping": frozenset(
        {
            "shared",
            "ingestion",
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "point_representation",
            "semantic_fusion",
        }
    ),
    "entity_resolution": frozenset(
        {
            "shared",
            "geometric_mapping",
            "visual_perception",
            "point_representation",
            "semantic_mapping",
        }
    ),
    "spatial_relations": frozenset(
        {"shared", "geometric_mapping", "semantic_mapping", "entity_resolution"}
    ),
    "artifact": frozenset(
        {
            "shared",
            "geometric_mapping",
            "semantic_mapping",
            "entity_resolution",
            "spatial_relations",
        }
    ),
    "runtime": CAPABILITIES - {"runtime", "evaluation"},
    "evaluation": CAPABILITIES - {"evaluation"},
}

PRIVATE_CROSS_MODULE_PARTS = frozenset({"backends", "infrastructure", "_internal"})
BACKEND_IMPLEMENTATION_PARTS = frozenset({"backends", "infrastructure", "adapters"})
HEAVY_SDK_ROOTS = frozenset(
    {
        "torch",
        "transformers",
        "rclpy",
        "rosbag",
        "rosbag2_py",
        "rosbags",
        "segment_anything",
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    rule: str
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


def _module_name(path: Path) -> str | None:
    try:
        root_index = path.parts.index("contextmap")
    except ValueError:
        return None

    relative_parts = list(path.parts[root_index:])
    filename = relative_parts[-1]
    if not filename.endswith(".py"):
        return None

    stem = filename.removesuffix(".py")
    if stem == "__init__":
        relative_parts = relative_parts[:-1]
    else:
        relative_parts[-1] = stem
    return ".".join(relative_parts)


def _source_capability(path: Path) -> str | None:
    module = _module_name(path)
    if module is None:
        return None

    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "contextmap":
        return None
    return parts[1] if parts[1] in CAPABILITIES else None


def _resolve_import_from(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    source_module = _module_name(path)
    if source_module is None:
        return node.module or ""

    package = source_module if path.name == "__init__.py" else source_module.rpartition(".")[0]
    package_parts = package.split(".")
    parent_hops = node.level - 1
    if parent_hops > len(package_parts):
        return node.module or ""

    base = package_parts[: len(package_parts) - parent_hops]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _imported_modules(path: Path, tree: ast.AST) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.lineno, _resolve_import_from(path, node)))
    return imports


def _is_backend_implementation(path: Path, capability: str) -> bool:
    try:
        capability_index = path.parts.index(capability)
    except ValueError:
        return False
    internal_parts = set(path.parts[capability_index + 1 : -1])
    return bool(internal_parts & BACKEND_IMPLEMENTATION_PARTS)


def _contextmap_target(module: str) -> tuple[str, tuple[str, ...]] | None:
    parts = tuple(part for part in module.split(".") if part)
    if len(parts) < 2 or parts[0] != "contextmap" or parts[1] not in CAPABILITIES:
        return None
    return parts[1], parts[2:]


def _check_import(
    *,
    path: Path,
    line: int,
    source: str,
    imported_module: str,
) -> list[Violation]:
    violations: list[Violation] = []
    imported_root = imported_module.split(".", maxsplit=1)[0]

    if imported_root in HEAVY_SDK_ROOTS and not _is_backend_implementation(path, source):
        violations.append(
            Violation(
                path,
                line,
                "heavy-sdk-isolation",
                f"{source} imports {imported_root!r} outside backends/infrastructure/adapters",
            )
        )

    target = _contextmap_target(imported_module)
    if target is None:
        return violations

    target_capability, target_parts = target
    if target_capability == source:
        return violations

    if target_capability == "runtime" and source != "evaluation":
        violations.append(
            Violation(
                path,
                line,
                "runtime-isolation",
                f"{source} must not depend on runtime",
            )
        )
        return violations

    if source == "runtime":
        # A composition root pode conhecer implementações concretas para construí-las.
        return violations

    if target_capability not in ALLOWED_DEPENDENCIES[source]:
        violations.append(
            Violation(
                path,
                line,
                "dependency-direction",
                f"{source} must not depend on {target_capability}",
            )
        )
        return violations

    private_parts = PRIVATE_CROSS_MODULE_PARTS.intersection(target_parts)
    private_symbol = any(part.startswith("_") for part in target_parts)
    if private_parts or private_symbol:
        violations.append(
            Violation(
                path,
                line,
                "private-cross-module-import",
                f"{source} imports internal path {imported_module!r}",
            )
        )
        return violations

    if target_parts:
        violations.append(
            Violation(
                path,
                line,
                "public-api-boundary",
                (
                    f"{source} must import {target_capability} through "
                    f"'contextmap.{target_capability}', not {imported_module!r}"
                ),
            )
        )

    return violations


def find_violations(path: Path, source_text: str) -> list[Violation]:
    source_capability = _source_capability(path)
    if source_capability is None:
        return []

    tree = ast.parse(source_text, filename=str(path))
    violations: list[Violation] = []
    for line, imported_module in _imported_modules(path, tree):
        violations.extend(
            _check_import(
                path=path,
                line=line,
                source=source_capability,
                imported_module=imported_module,
            )
        )
    return violations


def _fixture_violations(path: str, source: str) -> list[Violation]:
    return find_violations(Path(path), source)


def test_repository_respects_architecture_boundaries() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source_root = repository_root / "src" / "contextmap"
    violations: list[Violation] = []

    for path in sorted(source_root.rglob("*.py")):
        violations.extend(find_violations(path, path.read_text(encoding="utf-8")))

    rendered = "\n".join(violation.render() for violation in violations)
    assert not violations, f"Architecture boundary violations:\n{rendered}"


def test_valid_public_cross_module_import_is_allowed() -> None:
    violations = _fixture_violations(
        "src/contextmap/semantic_fusion/service.py",
        "from contextmap.sensor_association import SpatialObservation\n",
    )
    assert violations == []


def test_cross_module_submodule_import_is_rejected() -> None:
    violations = _fixture_violations(
        "src/contextmap/semantic_fusion/service.py",
        "from contextmap.sensor_association.models import SpatialObservation\n",
    )
    assert {violation.rule for violation in violations} == {"public-api-boundary"}


def test_cross_module_backend_import_is_rejected() -> None:
    violations = _fixture_violations(
        "src/contextmap/semantic_fusion/service.py",
        "from contextmap.visual_perception.backends.sam3 import SamNativeMask\n",
    )
    assert {violation.rule for violation in violations} == {"private-cross-module-import"}


def test_capability_importing_runtime_is_rejected() -> None:
    violations = _fixture_violations(
        "src/contextmap/visual_perception/service.py",
        "from contextmap.runtime import pipeline\n",
    )
    assert {violation.rule for violation in violations} == {"runtime-isolation"}


def test_forbidden_dependency_direction_is_rejected() -> None:
    violations = _fixture_violations(
        "src/contextmap/ingestion/models.py",
        "from contextmap.semantic_fusion import FusedEvidence\n",
    )
    assert {violation.rule for violation in violations} == {"dependency-direction"}


def test_relative_cross_capability_import_is_checked() -> None:
    violations = _fixture_violations(
        "src/contextmap/semantic_fusion/service.py",
        "from ..sensor_association.models import SpatialObservation\n",
    )
    assert {violation.rule for violation in violations} == {"public-api-boundary"}


def test_heavy_sdk_import_outside_backend_is_rejected() -> None:
    violations = _fixture_violations(
        "src/contextmap/visual_perception/models.py",
        "import torch\n",
    )
    assert {violation.rule for violation in violations} == {"heavy-sdk-isolation"}


def test_heavy_sdk_import_inside_backend_is_allowed() -> None:
    violations = _fixture_violations(
        "src/contextmap/visual_perception/backends/dino.py",
        "import torch\n",
    )
    assert violations == []


def test_runtime_can_import_concrete_backend_for_composition() -> None:
    violations = _fixture_violations(
        "src/contextmap/runtime/composition.py",
        "from contextmap.visual_perception.backends.sam3 import Sam3RegionDiscovery\n",
    )
    assert violations == []
