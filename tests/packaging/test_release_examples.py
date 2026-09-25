"""The committed v0.1.0 examples work on a base install, with no extra and no GPU.

These run in the ``lightweight-install`` job against the wheel installed with NumPy only, which
is the point: issue #188 promises that a clean base installation opens and validates the demo
artifact and that the documented commands are covered by release smoke tests. A README command
that stops working fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextmap.artifact import (
    ContextMapArtifactReader,
    Severity,
    ValidationLevel,
    ValidationStatus,
    validate_context_map_artifact,
)
from contextmap.runtime.cli import main as cli

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "v0.1.0"
DEMO = EXAMPLES / "demo" / "context-map"
CONFIG = EXAMPLES / "canonical-config.toml"


def test_a_base_install_validates_the_demo_artifact_with_no_dependency_paths() -> None:
    """Nothing is passed: the relative hints the manifest recorded must resolve on their own."""
    report = validate_context_map_artifact(DEMO, level=ValidationLevel.FULL)

    assert report.status is ValidationStatus.VERIFIED
    errors = [finding for finding in report.findings if finding.severity is Severity.ERROR]
    assert errors == []
    # Os avisos esperados são só as dependências opcionais ausentes, que não vão no pacote.
    assert {finding.code for finding in report.findings} <= {"dependency.optional_missing"}


def test_the_demo_artifact_exercises_entities_geometry_alternatives_and_relations() -> None:
    """#188 forbids an empty smoke artifact; this pins what it actually carries."""
    reader = ContextMapArtifactReader.open(DEMO, verify_hashes=True)
    entities = list(reader.entities())
    relations = list(reader.relations())

    assert len(entities) == 3
    assert len(relations) == 10
    assert all(entity.geometry_refs for entity in entities)
    assert any(len(entity.semantic_state.hypotheses) > 1 for entity in entities)
    assert {entity.semantic_state.status.value for entity in entities} == {
        "unambiguous",
        "ambiguous",
        "insufficient_evidence",
    }


def test_the_documented_validate_command_reports_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(["validate", str(DEMO)]) == 0

    assert "ok" in capsys.readouterr().out


def test_the_documented_inspect_artifact_command_summarises_the_demo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli(["inspect", "artifact", str(DEMO)]) == 0

    output = capsys.readouterr().out
    assert "schema_version: 0.1.0" in output
    assert "integrity: ok" in output


def test_the_canonical_configuration_resolves_with_no_undocumented_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The example must parse and resolve cleanly, which is an acceptance criterion of #188."""
    assert cli(["inspect", "config", "-c", str(CONFIG), "--json"]) == 0

    document = json.loads(capsys.readouterr().out)
    assert document.get("problems", []) == []
    assert document["config"]["pipeline"]["preset"] == "canonical/1"
    assert (
        document["config"]["components"]["visual_perception"]["semantic_interpretation"]["backend"]
        == "qwen"
    )


def test_the_committed_effective_configuration_is_what_the_example_resolves_to(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A committed example that drifts from what the code produces is worse than none."""
    assert cli(["inspect", "config", "-c", str(CONFIG), "--json"]) == 0
    resolved = json.loads(capsys.readouterr().out)

    committed = json.loads((EXAMPLES / "effective-config.json").read_text(encoding="utf-8"))

    assert committed["config"] == resolved["config"]
    assert committed["schema_version"] == resolved["schema_version"]
    # O digest cobre o documento efetivo, não onde o arquivo estava: é ele que prova que o
    # exemplo commitado é o que este código resolve.
    assert committed["digest"] == resolved["digest"]
    # A identidade de uma fonte é um caminho, e um caminho depende de onde a suíte roda; o que
    # é comparável é o hash do conteúdo do arquivo lido.
    assert [source["kind"] for source in committed["sources"]] == [
        source["kind"] for source in resolved["sources"]
    ]
    assert [source["content_hash"] for source in committed["sources"]] == [
        source["content_hash"] for source in resolved["sources"]
    ]


def test_the_committed_resolved_plan_is_what_the_example_resolves_to(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli(["inspect", "plan", "-c", str(CONFIG)]) == 0

    assert capsys.readouterr().out == (EXAMPLES / "resolved-plan.txt").read_text(encoding="utf-8")


def test_a_context_map_distributed_without_its_required_lineage_is_invalid(tmp_path: Path) -> None:
    """The README says so, and it is the contract working: the map references, never copies.

    Copying the artifact alone, away from the upstream runs its manifest points at, must be
    reported as invalid rather than silently accepted.
    """
    import shutil

    alone = tmp_path / "context-map"
    shutil.copytree(DEMO, alone)

    report = validate_context_map_artifact(alone, level=ValidationLevel.FULL)

    assert report.status is ValidationStatus.INVALID
    assert {finding.code for finding in report.findings if finding.severity is Severity.ERROR} == {
        "dependency.required_missing"
    }
