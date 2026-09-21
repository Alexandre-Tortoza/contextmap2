"""Regressions for the configuration, selection, cache and secret mistakes that would silently
invalidate a scientific comparison even when every capability is correct.

Each test names one mistake and pins the behavior that prevents it. Stages are fakes: nothing
here needs a model, a GPU or the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_worlds import World, run_cli, world_executors

from contextmap.runtime import (
    ArtifactRef,
    CatalogEntry,
    ConfigurationError,
    Lineage,
    ReusePolicy,
    StaticCatalog,
    resolve_effective_config,
    resolve_plan,
    resolve_selections,
    run_plan,
)

PROVIDED = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)
SRC = Path(__file__).resolve().parents[2] / "src"


def _ready(_name: str) -> bool:
    return True


def _document(**changes: Any) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    document.update(changes)
    return document


def _config_file(tmp_path: Path, document: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document or _document()), encoding="utf-8")
    return path


class TestConfigurationMistakes:
    def test_the_environment_never_changes_the_effective_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline = effective_from(tmp_path, _document()).digest
        for name in ("GEMINI_API_KEY", "CONTEXTMAP_DEVICE", "RESOURCES_DEVICE", "DEVICE", "HOME"):
            monkeypatch.setenv(name, "cuda:9")

        assert effective_from(tmp_path, _document()).digest == baseline

    def test_digests_are_stable_across_processes_and_hash_seeds(self, tmp_path: Path) -> None:
        config = _config_file(tmp_path)
        code = (
            "import json, sys\n"
            "from contextmap.runtime import resolve_effective_config, resolve_plan\n"
            f"effective = resolve_effective_config(files=[{str(config)!r}])\n"
            "plan = resolve_plan(effective)\n"
            "print(json.dumps([effective.digest, plan.digest, plan.order,"
            " [s.config_digest for s in plan.stages]]))\n"
        )

        outputs = []
        for seed in ("1", "2", "123"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONPATH": str(SRC), "PYTHONHASHSEED": seed, "PATH": ""},
            )
            outputs.append(result.stdout)

        assert len(set(outputs)) == 1

    def test_a_backend_of_another_variation_point_is_rejected_listing_the_supported_ones(
        self, tmp_path: Path
    ) -> None:
        document = _document()
        document["components"]["visual_perception"]["region_features"]["backend"] = "dinov3"

        with pytest.raises(ConfigurationError, match="alphaclip") as error:
            effective_from(tmp_path, document)

        assert "clip" in str(error.value)

    def test_a_secret_passed_as_an_override_is_refused_and_its_value_is_never_echoed(
        self, tmp_path: Path
    ) -> None:
        value = "hunter2-do-not-echo"

        code, out, err = run_cli(
            "inspect",
            "config",
            "--set",
            f"components.visual_perception.region_discovery.sam3.api_key={value}",
        )

        assert code == 1
        assert "secret" in out + err
        assert value not in out + err

    def test_switching_the_selected_backend_never_leaves_the_other_ones_parameters_behind(
        self, tmp_path: Path
    ) -> None:
        document = _document()
        region = document["components"]["visual_perception"]["region_discovery"]
        region["florence2"] = {"checkpoint": "stale", "task": "<OD>"}

        sam3 = effective_from(tmp_path, document)
        florence = resolve_effective_config(
            files=[_config_file(tmp_path, document)],
            overrides=["components.visual_perception.region_discovery.backend=florence2"],
        )

        parameters = sam3.config.components["visual_perception.region_discovery"].parameters
        assert "task" not in parameters
        assert sam3.digest != florence.digest


class TestSelectionMistakes:
    def _catalog(self) -> StaticCatalog:
        def entry(stage: str, contract: str, artifact_id: str) -> CatalogEntry:
            return CatalogEntry(
                ref=ArtifactRef(
                    stage_id=stage,
                    contract=contract,
                    artifact_id=artifact_id,
                    content_hash=f"sha256:{artifact_id}",
                ),
                lineage=Lineage(sequence="S1"),
                run_index=1,
            )

        return StaticCatalog(
            [
                entry("ingestion", "SequenceArtifact", "seq-1"),
                entry("state_estimation", "PerceptionRunArtifact", "traj-wrong-kind"),
                entry("point_representation", "PointRepresentationRunArtifact", "pr-1"),
            ]
        )

    def _resolve(self, tmp_path: Path, selections: dict[str, str]) -> Any:
        document = _document(inputs={"selections": selections})
        effective = effective_from(tmp_path, document)
        return resolve_selections(resolve_plan(effective), effective.config.inputs, self._catalog())

    def test_a_run_of_the_right_stage_but_the_wrong_artifact_kind_is_refused(
        self, tmp_path: Path
    ) -> None:
        resolved = self._resolve(tmp_path, {"state_estimation": "traj-wrong-kind"})

        assert not resolved.ok
        assert any("PerceptionRunArtifact" in problem.message for problem in resolved.problems)

    def test_selecting_a_stage_that_does_not_take_part_in_the_plan_is_a_problem(
        self, tmp_path: Path
    ) -> None:
        resolved = self._resolve(tmp_path, {"point_representation": "pr-1"})

        assert not resolved.ok
        assert any(
            "does not take part" in p.message or "not a stage" in p.message
            for p in resolved.problems
        )

    def test_an_unlisted_stage_is_never_selected_even_when_runs_exist(self, tmp_path: Path) -> None:
        resolved = self._resolve(tmp_path, {"ingestion": "seq-1"})

        assert set(resolved.provided) == {"ingestion"}


class TestCacheMistakes:
    def _run(
        self,
        tmp_path: Path,
        world: World,
        provided: dict[str, ArtifactRef] | None,
        *,
        document: dict[str, Any] | None = None,
    ) -> Any:
        effective = effective_from(tmp_path, document or _document())
        execution = resolve_plan(effective).scope(targets=["semantic_fusion"], provided=provided)
        return run_plan(
            execution,
            world.executors(execution.plan),
            environ={},
            module_available=_ready,
            provided_runtimes=PROVIDED,
            reuse=ReusePolicy(store=world.store(tmp_path / "index"), code_identity="code-1"),
        )

    @staticmethod
    def _sequence(artifact_id: str, content: str) -> dict[str, ArtifactRef]:
        return {
            "ingestion": ArtifactRef(
                stage_id="ingestion",
                contract="SequenceArtifact",
                artifact_id=artifact_id,
                content_hash=content,
            )
        }

    def test_reuse_follows_the_content_of_an_input_and_not_its_name(self, tmp_path: Path) -> None:
        world = World()
        self._run(tmp_path, world, self._sequence("seq-a", "sha256:H1"))
        world.runs.clear()

        self._run(tmp_path, world, self._sequence("renamed-seq", "sha256:H1"))

        assert world.runs == []  # mesmo conteúdo, outro nome: reutiliza tudo

    def test_the_same_artifact_name_with_other_content_is_never_reused(
        self, tmp_path: Path
    ) -> None:
        world = World()
        self._run(tmp_path, world, self._sequence("seq-a", "sha256:H1"))
        world.runs.clear()

        self._run(tmp_path, world, self._sequence("seq-a", "sha256:H2"))

        assert "visual_perception" in world.runs and "state_estimation" in world.runs

    def test_the_order_of_keys_in_a_configuration_file_never_defeats_reuse(
        self, tmp_path: Path
    ) -> None:
        world = World()
        self._run(tmp_path, world, None)
        world.runs.clear()
        document = _document()
        reordered = {key: document[key] for key in reversed(list(document))}

        self._run(tmp_path, world, None, document=reordered)

        assert world.runs == []

    def test_a_failed_stage_indexes_nothing_beyond_what_completed(self, tmp_path: Path) -> None:
        world = World()
        world.fail_at = "geometric_mapping"
        with pytest.raises(Exception, match="geometric_mapping"):
            self._run(tmp_path, world, None)

        entries = sorted((tmp_path / "index").glob("sha256-*.json"))

        assert len(entries) == 3  # ingestion, visual_perception e state_estimation
        indexed = {json.loads(p.read_text("utf-8"))["output"]["stage_id"] for p in entries}
        assert "geometric_mapping" not in indexed


class TestSecretsLeaveNoTrace:
    SECRET = "s3cr3t-token-value"

    def _gemini(self) -> dict[str, Any]:
        document = _document()
        document["components"]["visual_perception"]["semantic_interpretation"] = {
            "backend": "gemini",
            "gemini": {"model": "g", "timeout_s": 1, "max_retries": 1, "temperature": 0.0},
        }
        return document

    def test_a_successful_run_persists_and_prints_no_secret(self, tmp_path: Path) -> None:
        world = World()

        code, out, err = run_cli(
            "run",
            "-c",
            str(_config_file(tmp_path, self._gemini())),
            "--stage",
            "visual_perception",
            "--workspace",
            str(tmp_path / "ws"),
            executors=world_executors(world),
            environ={"GEMINI_API_KEY": self.SECRET, "OTHER_SECRET": "other-value-xyz"},
        )

        assert code == 0, out + err
        assert self.SECRET not in out + err
        files = [p for p in (tmp_path / "ws").rglob("*") if p.is_file()]
        assert files
        for path in files:
            text = path.read_text("utf-8")
            assert self.SECRET not in text and "other-value-xyz" not in text, path.name

    def test_the_recorded_environment_holds_no_environment_variable(self, tmp_path: Path) -> None:
        world = World()
        run_cli(
            "run",
            "-c",
            str(_config_file(tmp_path)),
            "--stage",
            "ingestion",
            "--workspace",
            str(tmp_path / "ws"),
            executors=world_executors(world),
            environ={"PRIVATE_MARKER": "marker-value-123"},
        )

        status = (tmp_path / "ws" / "S1" / "run-0001" / "status.json").read_text("utf-8")

        assert "marker-value-123" not in status and "PRIVATE_MARKER" not in status


class TestDryRunMatchesTheRealRun:
    def test_the_dry_run_plan_is_the_plan_the_real_run_executes(self, tmp_path: Path) -> None:
        world = World()
        # O workspace faz parte da configuração efetiva: o mesmo pedido, com e sem --dry-run.
        base = [
            "run",
            "-c",
            str(_config_file(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--workspace",
            str(tmp_path / "ws"),
        ]

        _, out, _ = run_cli(*base, "--dry-run", "--json")
        code, _, _ = run_cli(*base, executors=world_executors(world))

        run_dir = tmp_path / "ws" / "S1" / "run-0001"
        status = json.loads((run_dir / "status.json").read_text("utf-8"))
        plan = json.loads((run_dir / "plan.json").read_text("utf-8"))
        dry = json.loads(out)
        assert code == 0
        assert status["plan_digest"] == dry["plan_digest"]
        assert plan["document"] == dry["plan"]
        assert status["config_digest"] == dry["effective_config"]["digest"]

    def test_a_dry_run_loads_no_model_sdk(self, tmp_path: Path) -> None:
        loaded_before = {name for name in ("torch", "transformers") if name in sys.modules}

        code, _, _ = run_cli(
            "run", "-c", str(_config_file(tmp_path)), "--stage", "visual_perception", "--dry-run"
        )

        assert code == 0
        assert {name for name in ("torch", "transformers") if name in sys.modules} == loaded_before


class TestEquivalentRerunsAreEquivalent:
    @staticmethod
    def _trail(run_dir: Path) -> list[dict[str, Any]]:
        events = []
        for line in (run_dir / "events.jsonl").read_text("utf-8").splitlines():
            event = json.loads(line)
            event.pop("time")
            event["data"].pop("elapsed_s", None)
            events.append(event)
        return events

    def test_identical_requests_leave_identical_records_and_event_trails(
        self, tmp_path: Path
    ) -> None:
        config = _config_file(tmp_path)
        # O mesmo pedido (mesmo arquivo, mesmo workspace) duas vezes: dois runs distintos.
        for _ in range(2):
            world = World()
            code, out, err = run_cli(
                "run",
                "-c",
                str(config),
                "--stage",
                "semantic_fusion",
                "--workspace",
                str(tmp_path / "ws"),
                "--code-identity",
                "code-1",
                executors=world_executors(world),
            )
            assert code == 0, out + err

        first = tmp_path / "ws" / "S1" / "run-0001"
        second = tmp_path / "ws" / "S1" / "run-0002"
        for name in ("effective_config.json", "plan.json", "execution.json"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name
        assert self._trail(first) == self._trail(second)
