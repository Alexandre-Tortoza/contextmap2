"""The canonical runtime path, from a configuration file to run records, with fake stages.

The stages are pure functions of their inputs and configuration (see ``runtime_worlds``), so
everything the runtime owns is exercised for real: configuration resolution, the DAG, reuse,
selection, the lifecycle, resume and the CLI. Only the science is replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runtime_documents import selected_document
from runtime_worlds import World, run_cli, world_executors

CANONICAL = [
    "ingestion",
    "visual_perception",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "point_representation",
    "semantic_fusion",
]


def _ready(_name: str) -> bool:
    return True


def _config(tmp_path: Path, *, point_representation: bool = True) -> Path:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = point_representation
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _document(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


class Campaign:
    """One workspace, one reuse index and one fake world, driven through the CLI."""

    def __init__(self, tmp_path: Path, *, point_representation: bool = True) -> None:
        self.tmp_path = tmp_path
        self.world = World()
        self.config = _config(tmp_path, point_representation=point_representation)
        self.workspace = tmp_path / "ws"

    def cli(self, *arguments: str, executors: bool = True) -> tuple[int, str, str]:
        return run_cli(
            *arguments,
            executors=world_executors(self.world) if executors else {},
            verifier=lambda ref: ref.artifact_id in self.world.existing,
            module_available=_ready,
        )

    def run(self, *extra: str, target: str = "semantic_fusion") -> tuple[int, str, str]:
        return self.cli(
            "run",
            "-c",
            str(self.config),
            "--stage",
            target,
            "--workspace",
            str(self.workspace),
            "--reuse-index",
            str(self.tmp_path / "index"),
            "--code-identity",
            "code-1",
            *extra,
        )

    def record(self, number: int) -> Path:
        return self.workspace / "S1" / f"run-{number:04d}"

    def status(self, number: int) -> dict[str, Any]:
        return _document(self.record(number) / "status.json")

    def execution(self, number: int) -> dict[str, Any]:
        document: dict[str, Any] = _document(self.record(number) / "execution.json")["document"]
        return document

    def plan(self, number: int) -> dict[str, Any]:
        document: dict[str, Any] = _document(self.record(number) / "plan.json")["document"]
        return document

    def decisions(self, number: int) -> dict[str, str]:
        return {
            stage["stage_id"]: stage["decision"]["kind"]
            for stage in self.execution(number)["stages"]
        }


def assert_lineage_is_consistent(execution: dict[str, Any], plan: dict[str, Any]) -> None:
    """Every input of every stage is exactly the artifact its producer stage output."""
    outputs = {s["stage_id"]: s["output"]["artifact_id"] for s in execution["stages"]}
    wiring = {
        stage["stage_id"]: {item["name"]: item["source"] for item in stage["inputs"]}
        for stage in plan["stages"]
    }
    for stage in execution["stages"]:
        for name, refs in stage["inputs"].items():
            source = wiring[stage["stage_id"]][name]
            assert [ref["artifact_id"] for ref in refs] == [outputs[source]], (
                f"{stage['stage_id']}.{name} does not read {source}'s output"
            )


def test_the_canonical_path_from_a_configuration_file_to_records(tmp_path: Path) -> None:
    campaign = Campaign(tmp_path)

    # 1. O dry-run resolve o DAG completo, com o ramo opcional, sem executar nada.
    code, out, err = campaign.run("--dry-run", "--json")
    dry = json.loads(out)
    assert code == 0, out + err
    assert dry["scope"]["run"] == CANONICAL
    assert dry["preflight"]["ok"] is True
    assert {decision["kind"] for decision in dry["reuse"].values()} == {"recomputed"}
    assert campaign.world.runs == [] and not campaign.workspace.exists()

    # 2. A primeira execução real percorre o canônico em ordem de dependência.
    code, out, err = campaign.run()
    assert code == 0, out + err
    assert campaign.world.runs == CANONICAL
    assert campaign.status(1)["status"] == "completed"
    assert campaign.execution(1)["order"] == CANONICAL
    assert_lineage_is_consistent(campaign.execution(1), campaign.plan(1))
    assert campaign.plan(1)["config_digest"] == dry["effective_config"]["digest"]

    # 3. O mesmo pedido de novo reutiliza tudo: nenhum estágio roda.
    campaign.world.runs.clear()
    assert campaign.run()[0] == 0
    assert campaign.world.runs == []
    assert set(campaign.decisions(2).values()) == {"reused"}

    # 4. Mudar a política de fusão recomputa só a fusão.
    change = [
        "--set",
        "components.semantic_fusion.support.geometry-jaccard-support-v1.min_overlap=0.7",
    ]
    assert campaign.run(*change)[0] == 0
    assert campaign.world.runs == ["semantic_fusion"]
    assert campaign.decisions(3)["semantic_fusion"] == "recomputed"
    assert_lineage_is_consistent(campaign.execution(3), campaign.plan(3))

    # 5. Uma falha na fusão deixa um registro; os estágios anteriores foram reutilizados.
    campaign.world.runs.clear()
    campaign.world.fail_at = "semantic_fusion"
    failing = [
        "--set",
        "components.semantic_fusion.support.geometry-jaccard-support-v1.min_overlap=0.8",
    ]
    code, out, err = campaign.run(*failing)
    assert code == 1
    assert "run record" in err
    failed = campaign.status(4)
    assert failed["status"] == "failed"
    assert failed["failure"]["stage_id"] == "semantic_fusion"
    assert failed["completed_stages"] == CANONICAL[:-1]
    assert not (campaign.record(4) / "execution.json").exists()

    # 6. A retomada continua do estágio que falhou, como um run novo.
    campaign.world.fail_at = None
    campaign.world.runs.clear()
    code, out, err = campaign.run(*failing, "--resume", "run-0004")
    assert code == 0, out + err
    assert campaign.world.runs == ["semantic_fusion"]
    assert campaign.status(5)["resumed_from"] == "run-0004"
    assert campaign.execution(5)["resume"]["reused"] == CANONICAL[:-1]
    assert campaign.execution(5)["resume"]["recomputed"] == []
    assert campaign.status(4)["status"] == "failed"  # o histórico não foi tocado
    assert_lineage_is_consistent(campaign.execution(5), campaign.plan(5))

    # 7. Todo registro é íntegro e inspecionável.
    for number in range(1, 6):
        assert campaign.cli("validate", str(campaign.record(number)))[0] == 0
    code, out, _ = campaign.cli("inspect", "run", str(campaign.record(4)), "--json")
    assert code == 0 and json.loads(out)["failure"]["category"] == "execution"


def test_enabling_the_optional_stage_shares_the_upstream_artifacts_of_the_native_arm(
    tmp_path: Path,
) -> None:
    campaign = Campaign(tmp_path, point_representation=False)
    assert campaign.run()[0] == 0
    native_stages = campaign.execution(1)["order"]
    assert "point_representation" not in native_stages
    campaign.world.runs.clear()

    campaign.config.write_text(
        json.dumps(_with_optional_stage(_document(campaign.config))), encoding="utf-8"
    )
    assert campaign.run()[0] == 0

    enhanced = campaign.execution(2)["order"]
    assert enhanced == CANONICAL
    # Só o ramo novo e o que depende dele rodam; o restante vem do braço nativo.
    assert campaign.world.runs == ["point_representation", "semantic_fusion"]
    assert campaign.decisions(2)["ingestion"] == "reused"
    assert campaign.decisions(2)["sensor_association"] == "reused"
    native = {s["stage_id"]: s["output"]["artifact_id"] for s in campaign.execution(1)["stages"]}
    both = {s["stage_id"]: s["output"]["artifact_id"] for s in campaign.execution(2)["stages"]}
    assert all(both[stage] == native[stage] for stage in native if stage != "semantic_fusion")
    assert campaign.plan(1)["config_digest"] != campaign.plan(2)["config_digest"]
    assert_lineage_is_consistent(campaign.execution(2), campaign.plan(2))


def _with_optional_stage(document: dict[str, Any]) -> dict[str, Any]:
    document["pipeline"]["stages"]["point_representation"] = True
    return document


def test_a_run_that_cannot_run_is_recorded_as_blocked_and_nothing_executes(
    tmp_path: Path,
) -> None:
    campaign = Campaign(tmp_path)

    code, out, err = campaign.cli(
        "run",
        "-c",
        str(campaign.config),
        "--stage",
        "context_map",
        "--workspace",
        str(campaign.workspace),
    )

    assert code == 1
    assert "milestone" in out + err
    assert campaign.status(1)["status"] == "blocked"
    assert campaign.world.runs == []
    code, out, _ = campaign.cli("inspect", "run", str(campaign.record(1)))
    assert code == 0 and "blocked" in out
