"""Tests for the command-line adapter: it parses, calls runtime services and reports."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import selected_document
from runtime_fixtures import unavailable_context_map  # noqa: F401
from runtime_ingestion import factory as fake_factory
from runtime_worlds import World

from contextmap.runtime import (
    ArtifactRef,
    CatalogEntry,
    Lineage,
    StageRequest,
    load_catalog,
)
from contextmap.runtime.catalog import COMPONENTS
from contextmap.runtime.cli import main


def _ready(_name: str) -> bool:
    return True


def _document(**overrides: Any) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    document.update(overrides)
    return document


def _config(tmp_path: Path, document: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document or _document()), encoding="utf-8")
    return path


class Stage:
    """A lightweight stage executor: an artifact per call, content from its request."""

    def __init__(self, stage: str, contract: str, log: list[str]) -> None:
        self.stage, self.contract, self.log = stage, contract, log

    def execute(self, request: StageRequest) -> ArtifactRef:
        self.log.append(self.stage)
        return ArtifactRef(
            stage_id=self.stage,
            contract=self.contract,
            artifact_id=f"{self.stage}#1",
            content_hash=f"sha256:{self.stage}",
        )


def _executors(log: list[str]) -> dict[str, Stage]:
    from contextmap.runtime.catalog import CANONICAL_PRESET

    return {s.stage_id: Stage(s.stage_id, s.output or "", log) for s in CANONICAL_PRESET.stages}


def cli(*argv: str, **options: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    options.setdefault("module_available", _ready)
    options.setdefault("environ", {})
    code = main(list(argv), stdout=out, stderr=err, **options)
    return code, out.getvalue(), err.getvalue()


def _status(run_dir: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((run_dir / "status.json").read_text("utf-8"))
    return document


def _json(text: str) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(text)
    return result


class TestDryRun:
    def test_shows_the_resolved_dag_inputs_outputs_and_effective_config(
        self, tmp_path: Path
    ) -> None:
        code, out, _ = cli(
            "run", "-c", str(_config(tmp_path)), "--stage", "geometric_mapping", "--dry-run"
        )

        assert code == 0
        assert "sha256:" in out
        for stage in ("ingestion", "state_estimation", "geometric_mapping"):
            assert stage in out
        assert "SequenceArtifact" in out and "GeometricMapArtifact" in out
        assert "ros1_bag" in out

    def test_json_output_is_machine_readable_and_complete(self, tmp_path: Path) -> None:
        code, out, _ = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--dry-run",
            "--json",
        )

        document = _json(out)
        assert code == 0
        assert document["preflight"]["ok"] is True
        assert document["scope"]["run"] == ["ingestion", "state_estimation", "geometric_mapping"]
        assert document["effective_config"]["digest"].startswith("sha256:")
        assert [s["stage_id"] for s in document["plan"]["stages"]][:2] == [
            "ingestion",
            "visual_perception",
        ]

    def test_loads_no_model_and_needs_no_executor(self, tmp_path: Path) -> None:
        code, out, _ = cli(
            "run", "-c", str(_config(tmp_path)), "--stage", "semantic_fusion", "--dry-run", "--json"
        )

        assert code == 0
        assert _json(out)["executors"]["missing"]

    @pytest.mark.usefixtures("unavailable_context_map")
    def test_the_complete_pipeline_is_blocked_with_the_reason_per_stage(
        self, tmp_path: Path
    ) -> None:
        code, out, err = cli("run", "-c", str(_config(tmp_path)), "--dry-run")

        assert code == 1
        text = out + err
        assert "semantic_mapping" in text and "milestone" in text

    def test_a_missing_optional_module_is_explained_with_an_install_hint(
        self, tmp_path: Path
    ) -> None:
        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "visual_perception",
            "--dry-run",
            module_available=lambda name: name != "torch",
        )

        assert code == 1
        assert "torch" in out + err

    def test_a_missing_secret_is_reported_by_name(self, tmp_path: Path) -> None:
        document = _document()
        document["components"]["visual_perception"]["semantic_interpretation"] = {
            "backend": "gemini",
            "gemini": {"model": "g", "timeout_s": 1, "max_retries": 1, "temperature": 0.0},
        }

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path, document)),
            "--stage",
            "visual_perception",
            "--dry-run",
        )

        assert code == 1
        assert "GEMINI_API_KEY" in out + err

    def test_writes_nothing(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"

        cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "state_estimation",
            "--dry-run",
            "--workspace",
            str(workspace),
        )

        assert not workspace.exists()


class TestConfigurationFlags:
    def test_set_and_flags_are_translated_into_overrides(self, tmp_path: Path) -> None:
        code, out, _ = cli(
            "inspect",
            "config",
            "-c",
            str(_config(tmp_path)),
            "--set",
            "policies.debug_level=standard",
            "--device",
            "cuda",
            "--workspace",
            "ws",
            "--sequence",
            "S1",
            "--debug-level",
            "full",
            "--json",
        )

        config = _json(out)["config"]
        assert code == 0
        assert config["policies"]["debug_level"] == "full"  # a flag vence o --set
        assert config["resources"] == {"device": "cuda", "workspace": "ws"}
        assert config["inputs"]["sequence"] == "S1"

    def test_a_numeric_looking_sequence_stays_a_string(self, tmp_path: Path) -> None:
        code, out, _ = cli(
            "inspect", "config", "-c", str(_config(tmp_path)), "--sequence", "123", "--json"
        )

        assert code == 0
        assert _json(out)["config"]["inputs"]["sequence"] == "123"

    def test_an_unknown_profile_lists_the_known_ones(self, tmp_path: Path) -> None:
        code, out, err = cli("inspect", "config", "--profile", "canonical/99")

        assert code == 1
        assert "canonical/1" in out + err

    def test_an_invalid_configuration_reports_every_problem_with_its_path(
        self, tmp_path: Path
    ) -> None:
        document = _document(resources={"device": 1}, policies={"debug_level": "loud"})

        code, out, err = cli("inspect", "config", "-c", str(_config(tmp_path, document)))

        assert code == 1
        assert "resources.device" in out + err and "policies.debug_level" in out + err

    def test_a_failure_is_machine_readable_with_json(self, tmp_path: Path) -> None:
        code, out, _ = cli("inspect", "config", "--profile", "nope", "--json")

        document = _json(out)
        assert code == 1
        assert document["ok"] is False
        assert document["problems"]

    def test_a_malformed_override_is_a_usage_error(self, tmp_path: Path) -> None:
        code, out, err = cli("inspect", "config", "--set", "no-equals")

        assert code == 1
        assert "override" in out + err


class TestInspectPlan:
    def test_prints_the_topology_without_executing_anything(self, tmp_path: Path) -> None:
        code, out, _ = cli("inspect", "plan", "-c", str(_config(tmp_path)), "--json")

        document = _json(out)
        assert code == 0
        assert [s["stage_id"] for s in document["plan"]["stages"]] == [
            "ingestion",
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "semantic_fusion",
            "semantic_mapping",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        ]
        assert document["plan_digest"].startswith("sha256:")

    def test_the_optional_stage_enters_the_plan_by_configuration_only(self, tmp_path: Path) -> None:
        code, out, _ = cli(
            "inspect",
            "plan",
            "-c",
            str(_config(tmp_path)),
            "--set",
            "pipeline.stages.point_representation=true",
            "--json",
        )

        stages = [s["stage_id"] for s in _json(out)["plan"]["stages"]]
        assert code == 0 and "point_representation" in stages


class TestRun:
    def test_executes_a_subgraph_and_persists_the_run_records(self, tmp_path: Path) -> None:
        log: list[str] = []
        workspace = tmp_path / "ws"

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--workspace",
            str(workspace),
            executors=_executors(log),
        )

        assert code == 0, out + err
        assert log == ["ingestion", "state_estimation", "geometric_mapping"]
        run_dir = workspace / "S1" / "run-0001"
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "effective_config.json",
            "events.jsonl",
            "execution.json",
            "plan.json",
            "status.json",
        ]
        execution = json.loads((run_dir / "execution.json").read_text("utf-8"))["document"]
        assert execution["order"] == ["ingestion", "state_estimation", "geometric_mapping"]
        assert "run-0001" in out

    def test_every_run_gets_its_own_directory_and_never_overwrites_a_previous_one(
        self, tmp_path: Path
    ) -> None:
        options = ["--stage", "ingestion", "--workspace", str(tmp_path / "ws")]

        for _ in range(2):
            code, _, _ = cli(
                "run", "-c", str(_config(tmp_path)), *options, executors=_executors([])
            )
            assert code == 0

        assert sorted(p.name for p in (tmp_path / "ws" / "S1").iterdir()) == [
            "run-0001",
            "run-0002",
        ]

    def test_json_output_carries_the_execution_and_the_run_directory(self, tmp_path: Path) -> None:
        code, out, _ = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "ingestion",
            "--workspace",
            str(tmp_path / "ws"),
            "--json",
            executors=_executors([]),
        )

        document = _json(out)
        assert code == 0
        assert document["execution"]["order"] == ["ingestion"]
        assert document["run_directory"].endswith("run-0001")

    def test_a_real_run_needs_a_workspace(self, tmp_path: Path) -> None:
        log: list[str] = []

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "ingestion",
            executors=_executors(log),
        )

        assert code == 2
        assert "workspace" in out + err
        assert log == []

    def test_a_stage_without_an_executor_blocks_the_run_and_records_it(
        self, tmp_path: Path
    ) -> None:
        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "ingestion",
            "--workspace",
            str(tmp_path / "ws"),
        )

        assert code == 1
        assert "executor" in out + err and "ingestion" in out + err
        assert "run record" in err
        status = _status(tmp_path / "ws" / "S1" / "run-0001")
        assert status["status"] == "blocked"
        assert not (tmp_path / "ws" / "S1" / "run-0001" / "execution.json").exists()

    def test_a_failing_stage_reports_what_completed_and_leaves_a_failure_record(
        self, tmp_path: Path
    ) -> None:
        executors: dict[str, Any] = _executors([])

        class Boom:
            def execute(self, request: StageRequest) -> ArtifactRef:
                raise RuntimeError("out of memory")

        executors["state_estimation"] = Boom()

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--workspace",
            str(tmp_path / "ws"),
            executors=executors,
        )

        assert code == 1
        text = out + err
        assert "state_estimation" in text and "out of memory" in text and "ingestion" in text
        run_dir = tmp_path / "ws" / "S1" / "run-0001"
        assert str(run_dir) in err
        status = _status(run_dir)
        assert status["status"] == "failed"
        assert status["failure"]["stage_id"] == "state_estimation"
        assert not (run_dir / "execution.json").exists()

    def test_the_stage_command_targets_exactly_one_stage(self, tmp_path: Path) -> None:
        log: list[str] = []

        code, out, err = cli(
            "stage",
            "state_estimation",
            "-c",
            str(_config(tmp_path)),
            "--workspace",
            str(tmp_path / "ws"),
            executors=_executors(log),
        )

        assert code == 0, out + err
        assert log == ["ingestion", "state_estimation"]

    def test_an_unknown_stage_is_reported(self, tmp_path: Path) -> None:
        code, out, err = cli("stage", "nope", "-c", str(_config(tmp_path)), "--dry-run")

        assert code == 1
        assert "nope" in out + err


class TestSelections:
    def _catalog(self, tmp_path: Path) -> Path:
        def entry(stage: str, contract: str, artifact_id: str, index: int, sequence: str) -> Any:
            return CatalogEntry(
                ref=ArtifactRef(
                    stage_id=stage,
                    contract=contract,
                    artifact_id=artifact_id,
                    content_hash=f"sha256:{artifact_id}",
                ),
                lineage=Lineage(sequence=sequence),
                run_index=index,
            ).to_document()

        path = tmp_path / "catalog.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "0.1.0",
                    "entries": [
                        entry("ingestion", "SequenceArtifact", "seq-1", 1, "S1"),
                        entry("ingestion", "SequenceArtifact", "seq-2", 2, "S2"),
                        entry("state_estimation", "StateEstimationRunArtifact", "traj-1", 1, "S1"),
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_selected_runs_feed_the_execution_and_are_recorded(self, tmp_path: Path) -> None:
        log: list[str] = []

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--select",
            "ingestion=seq-1",
            "--select",
            "state_estimation=traj-1",
            "--catalog",
            str(self._catalog(tmp_path)),
            "--workspace",
            str(tmp_path / "ws"),
            executors=_executors(log),
        )

        assert code == 0, out + err
        assert log == ["geometric_mapping"]
        record = json.loads(
            (tmp_path / "ws" / "S1" / "run-0001" / "execution.json").read_text("utf-8")
        )["document"]
        assert record["selections"]["stages"]["ingestion"][0]["artifact_id"] == "seq-1"

    def test_an_incompatible_selection_fails_before_any_stage_runs(self, tmp_path: Path) -> None:
        log: list[str] = []

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--select",
            "ingestion=seq-2",
            "--select",
            "state_estimation=traj-1",
            "--catalog",
            str(self._catalog(tmp_path)),
            "--workspace",
            str(tmp_path / "ws"),
            executors=_executors(log),
        )

        assert code == 1
        assert "sequence" in out + err and "seq-2" in out + err
        assert log == []

    def test_selections_need_an_explicit_catalog(self, tmp_path: Path) -> None:
        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--select",
            "ingestion=seq-1",
            "--dry-run",
        )

        assert code == 2
        assert "--catalog" in out + err

    def test_a_list_and_latest_are_expressed_on_the_command_line(self, tmp_path: Path) -> None:
        code, out, _ = cli(
            "inspect",
            "config",
            "-c",
            str(_config(tmp_path)),
            "--select",
            "visual_perception=perc-1,perc-2",
            "--select",
            "state_estimation=latest",
            "--json",
        )

        selections = _json(out)["config"]["inputs"]["selections"]
        assert code == 0
        assert selections == {
            "visual_perception": ["perc-1", "perc-2"],
            "state_estimation": ["latest"],
        }

    def test_a_malformed_select_is_a_usage_error(self, tmp_path: Path) -> None:
        code, out, err = cli("inspect", "config", "--select", "ingestion")

        assert code == 2
        assert "STAGE=REF" in out + err

    def test_load_catalog_rejects_a_bad_document(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema_version": "9", "entries": []}), encoding="utf-8")

        with pytest.raises(ValueError, match="schema_version"):
            load_catalog(bad)
        bad.write_text(json.dumps({"schema_version": "0.1.0", "entries": [{}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="entry"):
            load_catalog(bad)
        with pytest.raises(ValueError, match="cannot read"):
            load_catalog(tmp_path / "missing.json")


class TestArtifactInspectionAndValidation:
    def _artifact(self, tmp_path: Path) -> Path:
        import hashlib

        root = tmp_path / "artifact"
        (root / "outputs").mkdir(parents=True)
        payload = b"hello"
        (root / "outputs" / "data.bin").write_bytes(payload)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_id": "seq-1",
                    "schema_version": "0.1.0",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "file_inventory": [
                        {
                            "path": "outputs/data.bin",
                            "size_bytes": len(payload),
                            "content_hash": "sha256:" + hashlib.sha256(payload).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_inspect_summarises_a_manifest_and_its_integrity(self, tmp_path: Path) -> None:
        code, out, _ = cli("inspect", "artifact", str(self._artifact(tmp_path)), "--json")

        document = _json(out)
        assert code == 0
        assert document["kind"] == "artifact"
        assert document["summary"]["artifact_id"] == "seq-1"
        assert document["summary"]["files"] == 1
        assert document["integrity"] == {"ok": True, "problems": []}

    def test_validate_passes_an_intact_artifact_and_names_a_corrupted_file(
        self, tmp_path: Path
    ) -> None:
        root = self._artifact(tmp_path)
        assert cli("validate", str(root))[0] == 0

        (root / "outputs" / "data.bin").write_bytes(b"HELLO")
        code, out, err = cli("validate", str(root))

        assert code == 1
        assert "data.bin" in out + err

    def test_validate_reports_a_missing_manifest(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        code, out, err = cli("validate", str(empty))

        assert code == 1
        assert "manifest.json" in out + err

    def test_runtime_documents_are_verified_by_digest(self, tmp_path: Path) -> None:
        run = tmp_path / "ws" / "S1" / "run-0001"
        cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "ingestion",
            "--workspace",
            str(tmp_path / "ws"),
            executors=_executors([]),
        )
        for name in ("effective_config.json", "plan.json", "execution.json"):
            assert cli("validate", str(run / name))[0] == 0, name

        path = run / "plan.json"
        document = json.loads(path.read_text("utf-8"))
        document["document"]["stages"][0]["capability"] = "other"
        path.write_text(json.dumps(document), encoding="utf-8")

        code, out, err = cli("validate", str(path))
        assert code == 1
        assert "digest" in out + err

    def test_an_unrecognised_file_is_explained(self, tmp_path: Path) -> None:
        stray = tmp_path / "stray.json"
        stray.write_text("{}", encoding="utf-8")

        code, out, err = cli("validate", str(stray))

        assert code == 1
        assert "manifest.json" in out + err

    def test_a_missing_path_is_reported(self, tmp_path: Path) -> None:
        code, out, err = cli("inspect", "artifact", str(tmp_path / "nowhere"))

        assert code == 1
        assert "nowhere" in out + err


class TestThinness:
    def test_the_cli_names_no_backend(self) -> None:
        source = Path(sys.modules["contextmap.runtime.cli"].__file__ or "").read_text("utf-8")
        backend_ids = {b for spec in COMPONENTS.values() for b in spec.backends}

        named = [backend for backend in backend_ids if f'"{backend}"' in source]
        assert named == []

    def test_importing_the_cli_loads_no_heavy_sdk(self) -> None:
        code = (
            "import sys; import contextmap.runtime.cli; "
            "heavy = [m for m in ('torch','transformers','rosbags','numpy') if m in sys.modules]; "
            "print(','.join(heavy))"
        )
        src = Path(__file__).resolve().parents[2] / "src"

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONPATH": str(src), "PATH": ""},
        )

        assert result.stdout.strip() == ""

    def test_the_module_entry_point_runs(self) -> None:
        src = Path(__file__).resolve().parents[2] / "src"

        result = subprocess.run(
            [sys.executable, "-m", "contextmap", "inspect", "plan", "--json"],
            capture_output=True,
            text=True,
            check=False,
            env={"PYTHONPATH": str(src), "PATH": ""},
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["plan"]["preset"] == "canonical/1"

    def test_help_documents_every_command(self) -> None:
        code, text, _ = cli("--help")

        assert code == 0
        for command in ("run", "stage", "inspect", "validate"):
            assert command in text


class TestLifecycleCommands:
    def _world_executors(self, world: World) -> dict[str, Any]:
        from contextmap.runtime.catalog import CANONICAL_PRESET

        return {
            s.stage_id: world.executor(s.stage_id, s.output or "") for s in CANONICAL_PRESET.stages
        }

    def _resumable(
        self, tmp_path: Path, *, fail_at: str = "geometric_mapping"
    ) -> tuple[World, list[str]]:
        world = World()
        world.fail_at = fail_at
        args = [
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "semantic_fusion",
            "--workspace",
            str(tmp_path / "ws"),
            "--reuse-index",
            str(tmp_path / "index"),
            "--code-identity",
            "code-1",
        ]
        code, _, _ = cli(
            *args,
            executors=self._world_executors(world),
            verifier=lambda ref: ref.artifact_id in world.existing,
            module_available=_ready,
        )
        assert code == 1
        world.fail_at = None
        world.runs.clear()
        return world, args

    def test_inspect_run_shows_the_failure_record_and_the_trail(self, tmp_path: Path) -> None:
        self._resumable(tmp_path)
        run_dir = str(tmp_path / "ws" / "S1" / "run-0001")

        code, out, _ = cli("inspect", "run", run_dir, "--events")

        assert code == 0
        assert "failed" in out and "geometric_mapping" in out and "execution" in out
        assert "stage_failed" in out and "run_failed" in out and "code-1" in out

    def test_inspect_run_json_carries_every_event_and_the_environment(self, tmp_path: Path) -> None:
        self._resumable(tmp_path)

        code, out, _ = cli("inspect", "run", str(tmp_path / "ws" / "S1" / "run-0001"), "--json")

        document = _json(out)
        assert code == 0
        assert document["status"] == "failed"
        assert document["failure"]["category"] == "execution"
        assert [e["kind"] for e in document["events"]][:2] == ["run_planned", "run_started"]
        assert "python" in document["environment"]

    def test_inspect_run_reports_a_blocked_run_with_its_problems(self, tmp_path: Path) -> None:
        cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "ingestion",
            "--workspace",
            str(tmp_path / "ws"),
        )

        code, out, _ = cli("inspect", "run", str(tmp_path / "ws" / "S1" / "run-0001"))

        assert code == 0
        assert "blocked" in out and "executor" in out

    def test_validate_checks_a_run_record_and_catches_a_corrupt_event_log(
        self, tmp_path: Path
    ) -> None:
        self._resumable(tmp_path)
        run_dir = tmp_path / "ws" / "S1" / "run-0001"
        assert cli("validate", str(run_dir))[0] == 0

        events = run_dir / "events.jsonl"
        lines = events.read_text("utf-8").splitlines()
        lines[1] = "{broken"
        events.write_text("\n".join(lines) + "\n", encoding="utf-8")
        code, out, err = cli("validate", str(run_dir))

        assert code == 1
        assert "line 2" in out + err

    def test_resume_continues_from_the_failed_stage_as_a_new_run(self, tmp_path: Path) -> None:
        world, args = self._resumable(tmp_path)

        code, out, err = cli(
            *args,
            "--resume",
            "run-0001",
            executors=self._world_executors(world),
            verifier=lambda ref: ref.artifact_id in world.existing,
            module_available=_ready,
        )

        assert code == 0, out + err
        assert world.runs == ["geometric_mapping", "sensor_association", "semantic_fusion"]
        assert "resumed run-0001" in out
        new = tmp_path / "ws" / "S1" / "run-0002"
        assert _status(new)["resumed_from"] == "run-0001"
        assert _status(tmp_path / "ws" / "S1" / "run-0001")["status"] == "failed"

    def test_resuming_a_completed_run_is_refused_and_creates_no_run(self, tmp_path: Path) -> None:
        world, args = self._resumable(tmp_path)
        options: dict[str, Any] = {
            "executors": self._world_executors(world),
            "verifier": lambda ref: ref.artifact_id in world.existing,
            "module_available": _ready,
        }
        assert cli(*args, "--resume", "run-0001", **options)[0] == 0

        code, out, err = cli(*args, "--resume", "run-0002", **options)

        assert code == 1
        assert "nothing to resume" in out + err
        assert not (tmp_path / "ws" / "S1" / "run-0003").exists()

    def test_resume_and_reuse_flags_have_explicit_requirements(self, tmp_path: Path) -> None:
        base = [
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "ingestion",
            "--workspace",
            str(tmp_path / "ws"),
        ]

        no_index = cli(*base, "--resume", "run-0001", executors=_executors([]))
        no_verifier = cli(*base, "--reuse-index", str(tmp_path / "i"), "--code-identity", "c")
        no_identity = cli(*base, "--reuse-index", str(tmp_path / "i"), verifier=lambda ref: True)
        force_alone = cli(*base, "--force", "ingestion")

        for code, out, err in (no_index, no_verifier, no_identity, force_alone):
            assert code == 2, out + err
        assert "--reuse-index" in no_index[1] + no_index[2]
        assert "verifier" in no_verifier[1] + no_verifier[2]
        assert "--code-identity" in no_identity[1] + no_identity[2]
        assert "--reuse-index" in force_alone[1] + force_alone[2]

    def test_an_unknown_run_to_resume_is_reported(self, tmp_path: Path) -> None:
        world = World()

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "ingestion",
            "--workspace",
            str(tmp_path / "ws"),
            "--reuse-index",
            str(tmp_path / "i"),
            "--code-identity",
            "c",
            "--resume",
            "run-0009",
            verifier=lambda ref: ref.artifact_id in world.existing,
            executors=self._world_executors(world),
        )

        assert code == 1
        assert "run-0009" in out + err

    def test_an_interrupt_is_a_cancelled_run_with_the_conventional_exit_code(
        self, tmp_path: Path
    ) -> None:
        world = World()
        world.fail_at = "state_estimation"
        world.fail_with = KeyboardInterrupt()

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "geometric_mapping",
            "--workspace",
            str(tmp_path / "ws"),
            executors=self._world_executors(world),
        )

        assert code == 130
        assert "cancelled" in out + err
        assert _status(tmp_path / "ws" / "S1" / "run-0001")["status"] == "cancelled"

    def test_a_secret_never_reaches_the_output_or_the_run_record(self, tmp_path: Path) -> None:
        secret = "s3cr3t-token-value"
        document = _document()
        document["components"]["visual_perception"]["semantic_interpretation"] = {
            "backend": "gemini",
            "gemini": {"model": "g", "timeout_s": 1, "max_retries": 1, "temperature": 0.0},
        }
        world = World()
        world.fail_at = "visual_perception"
        world.fail_with = RuntimeError(f"401 for key {secret}")

        code, out, err = cli(
            "run",
            "-c",
            str(_config(tmp_path, document)),
            "--stage",
            "visual_perception",
            "--workspace",
            str(tmp_path / "ws"),
            executors=self._world_executors(world),
            environ={"GEMINI_API_KEY": secret},
        )

        assert code == 1
        assert secret not in out + err
        for path in (tmp_path / "ws" / "S1" / "run-0001").iterdir():
            assert secret not in path.read_text("utf-8"), path.name

    def test_the_dry_run_predicts_what_reuse_would_do(self, tmp_path: Path) -> None:
        world, _ = self._resumable(tmp_path)

        code, out, _ = cli(
            "run",
            "-c",
            str(_config(tmp_path)),
            "--stage",
            "semantic_fusion",
            "--dry-run",
            "--reuse-index",
            str(tmp_path / "index"),
            "--code-identity",
            "code-1",
            "--json",
            verifier=lambda ref: ref.artifact_id in world.existing,
            module_available=_ready,
        )

        reuse = _json(out)["reuse"]
        assert code == 0
        assert reuse["ingestion"]["kind"] == "reused"
        assert reuse["geometric_mapping"]["kind"] == "recomputed"


class TestIngestCommand:
    def _args(self, tmp_path: Path, *extra: str) -> list[str]:
        source = tmp_path / "recording.bag"
        source.write_bytes(b"raw-source-bytes")
        return [
            "ingest",
            "-c",
            str(_config(tmp_path)),
            "--source",
            str(source),
            "--sequence-name",
            "corridor-02",
            "--topic",
            "rgb=/camera",
            "--topic",
            "imu=/imu",
            "--clock-id",
            "fake:header",
            "--sync-reference",
            "image",
            "--sync-tolerance-ns",
            "100000000",
            "--workspace",
            str(tmp_path / "ws"),
            *extra,
        ]

    def _published(self, tmp_path: Path) -> list[Path]:
        root = tmp_path / "ws" / "sequences" / "corridor-02"
        return sorted(root.iterdir()) if root.exists() else []

    def test_the_preflight_reports_readiness_without_reading_or_writing(
        self, tmp_path: Path
    ) -> None:
        code, out, _ = cli(
            *self._args(tmp_path, "--preflight", "--json"), adapter_factory=fake_factory()
        )

        document = _json(out)
        assert code == 0 and document["ok"] is True
        assert document["capabilities"]["rgb"] is True
        assert not (tmp_path / "ws").exists()

    def test_a_blocked_preflight_lists_the_problems_and_exits_non_zero(
        self, tmp_path: Path
    ) -> None:
        args = self._args(tmp_path, "--preflight", "--required", "lidar")

        code, out, err = cli(*args, adapter_factory=fake_factory())

        assert code == 1
        assert "lidar" in out + err and "BLOCKED" in out

    def test_it_publishes_a_sequence_artifact_and_reports_progress(self, tmp_path: Path) -> None:
        code, out, err = cli(*self._args(tmp_path), adapter_factory=fake_factory())

        assert code == 0, out + err
        assert "ingestion completed: corridor-02" in out
        assert "observations:" in out and "artifact:" in out
        assert "ingestion.reading-source" in err and "ingestion.completed" in err
        published = self._published(tmp_path)
        assert len(published) == 1 and (published[0] / "manifest.json").is_file()

    def test_the_json_result_carries_the_events_and_the_identity(self, tmp_path: Path) -> None:
        code, out, err = cli(*self._args(tmp_path, "--json"), adapter_factory=fake_factory())

        document = _json(out)
        assert code == 0, out + err
        assert document["status"] == "completed"
        assert document["request_identity"].startswith("sha256:")
        assert document["events"][0]["kind"] == "ingestion.planned"
        assert document["metrics"]["observations_read"] == 6

    def test_a_failure_names_the_category_and_the_phase_and_publishes_nothing(
        self, tmp_path: Path
    ) -> None:
        code, out, _ = cli(*self._args(tmp_path), adapter_factory=fake_factory(fail_after=2))

        assert code == 1
        assert "source during reading-source" in out and "bag corrupt" in out
        assert self._published(tmp_path) == []

    def test_an_interrupt_exits_with_the_conventional_code_and_publishes_nothing(
        self, tmp_path: Path
    ) -> None:
        def interrupt(index: int) -> None:
            if index == 1:
                raise KeyboardInterrupt

        code, out, err = cli(
            *self._args(tmp_path), adapter_factory=fake_factory(on_observation=interrupt)
        )

        assert code == 130 and "cancelled" in out + err
        assert self._published(tmp_path) == []

    def test_a_workspace_is_required(self, tmp_path: Path) -> None:
        args = self._args(tmp_path)
        position = args.index("--workspace")
        without = args[:position] + args[position + 2 :]

        code, out, err = cli(*without, adapter_factory=fake_factory())

        assert code == 2 and "--workspace" in out + err

    def test_a_selected_adapter_is_required(self, tmp_path: Path) -> None:
        document = _document()
        del document["components"]["ingestion"]
        bare = tmp_path / "bare.json"
        bare.write_text(json.dumps(document), encoding="utf-8")
        args = self._args(tmp_path)
        args[args.index("-c") + 1] = str(bare)

        code, out, err = cli(*args, adapter_factory=fake_factory())

        assert code == 2
        assert "components.ingestion.source_adapter.backend" in out + err

    def test_an_unknown_topic_key_lists_the_known_ones(self, tmp_path: Path) -> None:
        code, out, err = cli(
            *self._args(tmp_path, "--topic", "thermal=/cam"), adapter_factory=fake_factory()
        )

        assert code == 1
        assert "thermal" in out + err and "lidar" in out + err

    def test_a_missing_optional_module_is_explained_with_its_install_hint(
        self, tmp_path: Path
    ) -> None:
        code, out, err = cli(*self._args(tmp_path), module_available=lambda name: name != "rosbags")

        assert code == 1
        assert "rosbags" in out + err and "contextmap[ros1]" in out + err
        assert self._published(tmp_path) == []

    def test_the_adapter_family_is_not_a_command_line_choice(self) -> None:
        # Sem `adapter_factory`, a CLI compõe o backend selecionado na configuração.
        source = Path(sys.modules["contextmap.runtime.cli"].__file__ or "").read_text("utf-8")

        assert "--source-type" not in source and "--adapter" not in source

    def test_a_malformed_topic_flag_is_a_usage_error(self, tmp_path: Path) -> None:
        code, out, err = cli(
            *self._args(tmp_path, "--topic", "rgb"), adapter_factory=fake_factory()
        )

        assert code == 2 and "KEY=TOPIC" in out + err
