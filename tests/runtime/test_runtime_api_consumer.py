"""A frontend written against the public runtime API only, the way the CLI and the TUI will be.

Everything it needs comes from ``contextmap.runtime``: no internal runtime module, no backend, no
capability. It discovers what the runtime can do, lets a "user" pick backends, resolves the
topology and derives its executors from it (it never hard-codes the DAG), preflights, runs with a
terminal-style event renderer, inspects the run, runs again with reuse and cancels a third run.
A second test reads this very file and proves the imports above are all it uses.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from runtime_documents import selected_document
from runtime_fixtures import unavailable_context_map  # noqa: F401

import contextmap.runtime as public
from contextmap.runtime import (
    ArtifactRef,
    CancellationToken,
    Runtime,
    RuntimeExecutionEvent,
    RuntimeExecutionResult,
    StageRequest,
)


class PureStage:
    """A stage that is a pure function of what it consumed and how it is configured."""

    def __init__(self, stage_id: str, contract: str, produced: set[str]) -> None:
        self._stage_id = stage_id
        self._contract = contract
        self._produced = produced

    def execute(self, request: StageRequest) -> ArtifactRef:
        parts = [request.config_digest]
        parts += [
            f"{name}={ref.content_hash}"
            for name, refs in sorted(request.inputs.items())
            for ref in refs
        ]
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        artifact_id = f"{self._stage_id}-{digest[:8]}"
        self._produced.add(artifact_id)
        return ArtifactRef(
            stage_id=self._stage_id,
            contract=self._contract,
            artifact_id=artifact_id,
            content_hash=f"sha256:{digest}",
        )


class TerminalRenderer:
    """What a terminal frontend does with the events: one line each, in order."""

    def __init__(self, token: CancellationToken | None = None, stop_after: str | None = None):
        self.lines: list[str] = []
        self._token = token
        self._stop_after = stop_after

    def emit(self, event: RuntimeExecutionEvent) -> None:
        self.lines.append(f"{event.sequence:03d} {event.kind} {event.stage_id or '-'}")
        finished = event.kind in {"stage_completed", "stage_reused"}
        if self._token is not None and finished and event.stage_id == self._stop_after:
            self._token.cancel("stop requested by the user")


def _summary(result: RuntimeExecutionResult) -> dict[str, str]:
    return {stage.stage_id: stage.outcome for stage in result.record.stages}


@pytest.mark.usefixtures("unavailable_context_map")
def test_a_frontend_drives_discovery_configuration_preflight_run_and_inspection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    produced: set[str] = set()
    config_file = tmp_path / "experiment.json"
    config_file.write_text(json.dumps(selected_document()), encoding="utf-8")

    # 1. Descoberta: o que esta instalação sabe fazer, e por que algo não está disponível.
    probe = Runtime(workspace=workspace, module_available=lambda _name: True, environ={})
    capabilities = probe.capabilities()
    not_implemented = {c.stage_id: c.reason for c in capabilities if not c.implemented}
    assert probe.status().profiles == ("canonical/1",)
    assert set(not_implemented) == {"context_map"}

    # 2. Configuração e topologia: o frontend nunca infere o DAG, ele o lê.
    config = probe.resolve_config(files=[config_file], overrides=["policies.debug_level=standard"])
    plan = probe.resolve_plan(config, targets=["semantic_fusion"])
    assert plan.order is not None
    executors = {
        stage.stage_id: PureStage(stage.stage_id, stage.output or "", produced)
        for stage in plan.stages
        if stage.available
    }
    runtime = Runtime(
        workspace=workspace,
        executors=executors,
        verifier=lambda ref: ref.artifact_id in produced,
        module_available=lambda _name: True,
        environ={},
    )

    # 3. Preflight antes de qualquer execução.
    report = runtime.preflight(config, targets=list(plan.run_stages))
    assert report.ok, report.problems
    assert produced == set()

    # 4. Execução com eventos renderizados em ordem.
    renderer = TerminalRenderer()
    result = runtime.run(config, targets=list(plan.run_stages), events=renderer)
    assert result.ok
    assert renderer.lines[0].endswith("run_planned -")
    assert renderer.lines[-1].endswith("run_completed -")
    assert [line[:3] for line in renderer.lines] == [
        f"{n:03d}" for n in range(1, len(renderer.lines) + 1)
    ]

    # 5. Inspeção: a lista e o detalhe vêm do registro persistido, com a linhagem exata.
    (summary,) = runtime.list_runs()
    record = runtime.inspect_run(summary.run_id)
    assert record == result.record
    assert record.config_digest == config.digest
    assert [stage.stage_id for stage in record.stages] == list(plan.run_stages)
    perception = next(stage for stage in record.stages if stage.stage_id == "visual_perception")
    ingestion = next(stage for stage in record.stages if stage.stage_id == "ingestion")
    assert ingestion.output is not None
    assert perception.inputs == {"sequence": (ingestion.output["artifact_id"],)}

    # 6. Reuso explícito: a segunda execução aponta o artifact exato da primeira.
    policy = runtime.reuse_policy(tmp_path / "index", code_identity="frontend-1")
    warm = runtime.run(config, targets=list(plan.run_stages), reuse=policy)
    again = runtime.run(config, targets=list(plan.run_stages), reuse=policy)
    assert set(_summary(again).values()) == {"reused"}
    assert warm.run_id != again.run_id

    # 7. Cancelamento cooperativo: o run termina cancelado, num limite de estágio.
    token = CancellationToken()
    stopped = runtime.run(
        config,
        targets=list(plan.run_stages),
        cancellation=token,
        events=TerminalRenderer(token, stop_after="visual_perception"),
    )
    assert stopped.status == "cancelled"
    assert _summary(stopped)["state_estimation"] == "pending"

    # 8. Tudo o que o frontend viu é serializável para um log ou uma tela.
    json.dumps([run.to_document() for run in runtime.list_runs()], allow_nan=False)
    json.dumps(stopped.to_document(), allow_nan=False)


def test_this_consumer_uses_only_documented_public_runtime_imports() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "contextmap":
                modules.add(node.module)
                names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(
                alias.name for alias in node.names if alias.name.split(".")[0] == "contextmap"
            )

    assert modules == {"contextmap.runtime"}
    assert names <= set(public.__all__), sorted(names - set(public.__all__))
