"""The public runtime application API: discovery, configuration, preflight, execution and runs.

A frontend uses ``Runtime`` and nothing behind it. These tests drive it with the fake world of
``runtime_worlds`` (pure stages over content identity), so the facade is checked against the real
DAG, reuse, selection and lifecycle services with no GPU, model SDK or network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import selected_document
from runtime_fixtures import unavailable_future_stage  # noqa: F401
from runtime_ingestion import factory, request
from runtime_worlds import World, source_identities, world_executors

from contextmap.runtime import (
    ArtifactRef,
    BackendUnavailableError,
    CancellationToken,
    CatalogEntry,
    ConfigurationError,
    ExecutionEvent,
    IngestionService,
    Lineage,
    ResumeError,
    RunRecordError,
    Runtime,
    RuntimeRunRecord,
    StaticCatalog,
    check_selection,
    resolve_effective_config,
)

SRC = Path(__file__).resolve().parents[2] / "src"
SECRET = "s3cr3t-value-123"
TARGET = ["semantic_fusion"]
PLAN_ORDER = (
    "ingestion",
    "pose_ingestion",
    "visual_perception",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "point_representation",
    "semantic_fusion",
    "semantic_mapping",
    "entity_resolution",
    "spatial_relations",
    "context_map",
)
# ``pose_ingestion`` (issue #555) stays off by default in this suite's config, unlike
# ``point_representation``, which ``selected_document()`` enables explicitly -- so a resolved
# plan's actual stages/order exclude it, while a full catalog listing (``status.executors``,
# ``capabilities()``) still names it.
ACTIVE_PLAN_ORDER = tuple(stage for stage in PLAN_ORDER if stage != "pose_ingestion")
# The order a run scoped to ``TARGET`` (``semantic_fusion``) actually executes: the active
# stages' prefix up to and including it, since nothing after it is a dependency of that target.
RUN_ORDER = ACTIVE_PLAN_ORDER[: ACTIVE_PLAN_ORDER.index("semantic_fusion") + 1]
# `canonical/1`'s own topology, extended by the ``unavailable_future_stage`` fixture with a
# fictional ``scene_graph`` stage marked unavailable -- it never joins ``run_stages`` because it
# is not a dependency of any real target.
PLAN_ORDER_WITH_UNAVAILABLE_FUTURE_STAGE = (*PLAN_ORDER, "scene_graph")
ACTIVE_PLAN_ORDER_WITH_UNAVAILABLE_FUTURE_STAGE = (*ACTIVE_PLAN_ORDER, "scene_graph")


def _ready(_name: str) -> bool:
    return True


class Sink:
    """A frontend's event sink: it only records."""

    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []

    def emit(self, event: ExecutionEvent) -> None:
        self.events.append(event)


def _runtime(
    tmp_path: Path, world: World | None = None, *, executors: bool = True, **options: Any
) -> tuple[Runtime, World]:
    world = world or World()
    options.setdefault("module_available", _ready)
    options.setdefault("environ", {})
    runtime = Runtime(
        workspace=tmp_path / "ws",
        executors=world_executors(world) if executors else {},
        verifier=lambda ref: ref.artifact_id in world.existing,
        **options,
    )
    return runtime, world


# O dataset é a sequência física do catálogo de teste (`Lineage(sequence=...)`).
DATASET = "corridor-02"


def _write(tmp_path: Path, document: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "experiment.json"
    document = document or selected_document()
    document["inputs"] = {**document.get("inputs", {}), "sequence": DATASET}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _config(runtime: Runtime, tmp_path: Path, *overrides: str) -> Any:
    return runtime.resolve_config(files=[_write(tmp_path)], overrides=list(overrides))


def _outcomes(record: RuntimeRunRecord) -> dict[str, str]:
    return {stage.stage_id: stage.outcome for stage in record.stages}


def _catalog() -> StaticCatalog:
    def entry(artifact_id: str, run_index: int) -> CatalogEntry:
        return CatalogEntry(
            ref=ArtifactRef(
                stage_id="ingestion",
                contract="SequenceArtifact",
                artifact_id=artifact_id,
                content_hash=f"sha256:{artifact_id}",
            ),
            lineage=Lineage(sequence="corridor-02"),
            run_index=run_index,
        )

    return StaticCatalog([entry("seq-1", 1), entry("seq-2", 2)])


# --- discovery -------------------------------------------------------------------------


def test_status_describes_the_runtime_and_what_it_is_wired_to(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)

    status = runtime.status()
    bare = Runtime().status()

    assert status.workspace == str(tmp_path / "ws")
    assert status.profiles == ("canonical/1",)
    assert set(status.schemas) == {"configuration", "plan", "run", "reuse", "catalog"}
    assert status.verifier_configured is True
    assert status.executors == tuple(sorted(PLAN_ORDER))
    assert (bare.workspace, bare.executors, bare.verifier_configured) == (None, (), False)


@pytest.mark.usefixtures("unavailable_future_stage")
def test_capabilities_list_every_stage_in_order_with_its_variation_points() -> None:
    capabilities = Runtime(module_available=_ready, environ={}).capabilities()
    by_stage = {capability.stage_id: capability for capability in capabilities}

    assert tuple(by_stage) == PLAN_ORDER_WITH_UNAVAILABLE_FUTURE_STAGE
    ingestion = by_stage["ingestion"]
    assert ingestion.implemented
    assert [component.component_id for component in ingestion.components] == [
        "ingestion.source_adapter"
    ]
    assert [b.backend_id for b in ingestion.components[0].backends] == ["ros1_bag", "ros2_bag"]
    unimplemented = by_stage["scene_graph"]
    assert not unimplemented.implemented
    assert "not implemented yet" in unimplemented.reason
    assert unimplemented.components == ()
    assert by_stage["point_representation"].optional
    assert not by_stage["point_representation"].default_enabled


def test_an_optional_component_round_trips_through_discovery_edit_and_validation(
    tmp_path: Path,
) -> None:
    """P2 #3 of the PR #540 review: ``RuntimeComponent.optional`` and the edit contract agree.

    ``entity_resolution.semantic_compatibility`` may legitimately have no backend selected.
    Discovery must say so (``RuntimeComponent.optional``), and the edit that lets a frontend
    choose it must accept "no backend" as one of ``allowed`` -- not describe a ``current`` of
    ``None`` while claiming only concrete backend ids are accepted.
    """
    runtime = Runtime(module_available=_ready, environ={})
    component_id = "entity_resolution.semantic_compatibility"

    # Discovery: o componente se declara opcional.
    capability = next(c for c in runtime.capabilities() if c.stage_id == "entity_resolution")
    component = next(c for c in capability.components if c.component_id == component_id)
    assert component.optional

    document = selected_document()
    document["inputs"] = {**document.get("inputs", {}), "sequence": DATASET}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    config = runtime.resolve_config(files=[path])

    # Edição: "sem backend" é um valor aceito, não só o que `current` descreve.
    plan = runtime.resolve_plan(config)
    edit = next(e for e in plan.editable if e.path == f"components.{component_id}.backend")
    assert edit.current is None
    assert edit.allowed is not None
    assert None in edit.allowed

    # Validação: aplicar explicitamente "sem backend" continua uma seleção completa.
    validated = runtime.resolve_config(files=[path], overrides=[f"{edit.path}=null"])
    assert validated.config.components[component_id].backend is None
    assert check_selection(validated.config) == ()


def test_a_backend_that_cannot_be_used_says_why() -> None:
    runtime = Runtime(module_available=lambda name: name != "rosbags", environ={})

    ros1 = runtime.capabilities()[0].components[0].backends[0]

    assert ros1.backend_id == "ros1_bag"
    assert not ros1.available
    assert ros1.requires == ("rosbags",)
    assert ros1.install_hint == "pip install 'contextmap[ros1]'"
    assert any("rosbags" in reason and ros1.install_hint in reason for reason in ros1.reasons)


def test_a_missing_secret_is_named_and_a_present_one_is_never_valued() -> None:
    def gemini(environ: dict[str, str]) -> tuple[Any, Runtime]:
        runtime = Runtime(module_available=_ready, environ=environ)
        perception = next(c for c in runtime.capabilities() if c.stage_id == "visual_perception")
        component = next(
            c
            for c in perception.components
            if c.component_id == "visual_perception.semantic_interpretation"
        )
        return next(b for b in component.backends if b.backend_id == "gemini"), runtime

    missing, _ = gemini({})
    present, runtime = gemini({"GEMINI_API_KEY": SECRET})

    assert not missing.available
    assert any("GEMINI_API_KEY" in reason for reason in missing.reasons)
    assert present.available
    assert present.reasons == ()
    assert SECRET not in json.dumps([c.to_document() for c in runtime.capabilities()])


def test_every_backend_is_available_when_nothing_is_missing() -> None:
    runtime = Runtime(module_available=_ready, environ={"GEMINI_API_KEY": SECRET})

    backends = [
        backend
        for capability in runtime.capabilities()
        for component in capability.components
        for backend in component.backends
    ]

    assert backends
    assert all(backend.available for backend in backends)


def test_an_unknown_profile_is_refused_explicitly() -> None:
    runtime = Runtime()

    with pytest.raises(ConfigurationError, match="unknown profile 'nope'"):
        runtime.capabilities(profile="nope")
    with pytest.raises(ConfigurationError, match="unknown profile 'nope'"):
        runtime.resolve_config(profile="nope")


def test_discovery_and_inspection_import_no_optional_sdk(tmp_path: Path) -> None:
    heavy = ("torch", "transformers", "PIL", "rosbags", "alpha_clip")
    sentinels = tmp_path / "sentinels"
    sentinels.mkdir()
    fakes = tmp_path / "fakes"
    for name in heavy:
        (fakes / name).mkdir(parents=True)
        (fakes / name / "__init__.py").write_text(
            f"from pathlib import Path\nPath({str(sentinels / name)!r}).write_text('imported')\n",
            encoding="utf-8",
        )
    script = "\n".join(
        [
            "import json, sys",
            "from contextmap.runtime import Runtime",
            f"runtime = Runtime(workspace={str(tmp_path / 'ws')!r})",
            "runtime.status()",
            "capabilities = runtime.capabilities()",
            "config = runtime.resolve_config()",
            "runtime.resolve_plan(config)",
            "runtime.preflight(config)",
            "runtime.list_runs()",
            f"heavy = {heavy!r}",
            "available = sorted(b.backend_id for c in capabilities for k in c.components "
            "for b in k.backends if b.available)",
            "loaded = sorted(m for m in sys.modules if m.split('.')[0] in heavy)",
            "print(json.dumps({'available': available, 'loaded': loaded}))",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": os.pathsep.join([str(fakes), str(SRC)]), "PATH": ""},
    )

    seen = json.loads(result.stdout)
    assert seen["loaded"] == []
    assert list(sentinels.iterdir()) == []  # nenhum pacote pesado chegou a executar
    # Os pacotes falsos foram *encontrados*: a disponibilidade é uma consulta, não um import.
    assert {"ros1_bag", "dinov2", "alphaclip"} <= set(seen["available"])


def test_discovery_and_preflight_are_fast_enough_for_an_interactive_frontend(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)

    started = time.perf_counter()
    runtime.capabilities()
    discovery = time.perf_counter() - started
    started = time.perf_counter()
    runtime.preflight(config, targets=TARGET)
    preflight = time.perf_counter() - started

    assert discovery < 2.0
    assert preflight < 2.0


# --- configuration and topology --------------------------------------------------------


def test_resolve_config_is_deterministic_and_matches_the_direct_resolution(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    path = _write(tmp_path)

    first = runtime.resolve_config(files=[path])
    second = runtime.resolve_config(files=[path])
    direct = resolve_effective_config(
        files=[path], overrides=[f"resources.workspace={json.dumps(str(tmp_path / 'ws'))}"]
    )

    assert first.digest == second.digest == direct.digest
    assert first.to_document() == second.to_document()
    assert first.config.resources.workspace == str(tmp_path / "ws")
    assert (first.sources[0].kind, first.sources[0].identity) == ("profile", "canonical/1")


def test_an_explicit_workspace_override_wins_over_the_runtimes(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    elsewhere = json.dumps(str(tmp_path / "elsewhere"))

    config = _config(runtime, tmp_path, f"resources.workspace={elsewhere}")

    assert config.config.resources.workspace == str(tmp_path / "elsewhere")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("components.ingestion.source_adapter.backend=nope", "nope"),
        ("pipeline.stages.nonexistent=true", "unknown stage"),
        ("pipeline.stages.ingestion=false", "not optional"),
    ],
)
def test_an_unsupported_stage_or_backend_is_refused_at_resolution(
    tmp_path: Path, override: str, message: str
) -> None:
    runtime, _ = _runtime(tmp_path)

    with pytest.raises(ConfigurationError, match=message):
        _config(runtime, tmp_path, override)


@pytest.mark.usefixtures("unavailable_future_stage")
def test_the_resolved_plan_exposes_topology_wiring_and_selected_backends(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)

    plan = runtime.resolve_plan(config, targets=TARGET)

    by_stage = {stage.stage_id: stage for stage in plan.stages}
    assert plan.preset == "canonical/1"
    assert plan.order == ACTIVE_PLAN_ORDER_WITH_UNAVAILABLE_FUTURE_STAGE
    assert tuple(by_stage) == ACTIVE_PLAN_ORDER_WITH_UNAVAILABLE_FUTURE_STAGE
    assert plan.run_stages == RUN_ORDER
    # pose_ingestion (issue #555) is off by default; nothing else is disabled in this config.
    assert (plan.config_digest, plan.disabled_stages, plan.problems) == (
        config.digest,
        ("pose_ingestion",),
        (),
    )
    association = by_stage["sensor_association"]
    assert [(item.name, item.source) for item in association.inputs] == [
        ("sequence", "ingestion"),
        ("perception", "visual_perception"),
        ("trajectory", "state_estimation"),
        ("geometry", "geometric_mapping"),
    ]
    assert association.inputs[1].multiple
    assert by_stage["ingestion"].backends == {"ingestion.source_adapter": "ros1_bag"}
    assert by_stage["visual_perception"].backends["visual_perception.region_discovery"] == "sam3"
    assert by_stage["point_representation"].optional
    assert by_stage["point_representation"].output == "PointRepresentationRunArtifact"
    assert by_stage["ingestion"].in_scope
    assert not by_stage["scene_graph"].available
    assert not by_stage["scene_graph"].in_scope
    assert by_stage["ingestion"].config_digest


def test_a_disabled_optional_stage_is_listed_and_its_branch_disappears(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)

    plan = runtime.resolve_plan(
        _config(runtime, tmp_path, "pipeline.stages.point_representation=false"), targets=TARGET
    )

    fusion = next(stage for stage in plan.stages if stage.stage_id == "semantic_fusion")
    toggle = next(e for e in plan.editable if e.path == "pipeline.stages.point_representation")
    # pose_ingestion is also off by default (issue #555); only point_representation was toggled.
    assert set(plan.disabled_stages) == {"pose_ingestion", "point_representation"}
    assert plan.order is not None
    assert "point_representation" not in plan.order
    assert "representation" not in [item.name for item in fusion.inputs]
    assert (toggle.kind, toggle.current, toggle.allowed) == ("toggle", False, (True, False))


def test_supplied_upstream_artifacts_take_the_place_of_their_stages(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)
    sequence = ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id="sequence-7",
        content_hash="sha256:seq",
    )
    wrong = ArtifactRef(
        stage_id="ingestion", contract="PerceptionRunArtifact", artifact_id="x", content_hash=None
    )

    plan = runtime.resolve_plan(
        config, targets=["visual_perception"], provided={"ingestion": sequence}
    )
    report = runtime.preflight(
        config, targets=["visual_perception"], provided={"ingestion": sequence}
    )
    refused = runtime.preflight(
        config, targets=["visual_perception"], provided={"ingestion": wrong}
    )

    by_stage = {stage.stage_id: stage for stage in plan.stages}
    assert plan.run_stages == ("visual_perception",)
    assert by_stage["ingestion"].provided == ("sequence-7",)
    assert not by_stage["ingestion"].in_scope
    assert report.ok
    assert report.provided == {"ingestion": ("sequence-7",)}
    assert report.stages == ("visual_perception",)
    assert not refused.ok
    assert [problem.path for problem in refused.problems] == ["provided.ingestion"]


@pytest.mark.usefixtures("unavailable_future_stage")
def test_every_declared_edit_is_a_real_override_path_with_a_truthful_current_value(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime(tmp_path)
    path = _write(tmp_path)
    base = runtime.resolve_config(files=[path])

    plan = runtime.resolve_plan(base, targets=TARGET)

    edits = {edit.path: edit for edit in plan.editable}
    assert "components.ingestion.source_adapter.backend" in edits
    assert edits["components.ingestion.source_adapter.backend"].allowed == ("ros1_bag", "ros2_bag")
    assert edits["policies.debug_level"].allowed == ("none", "standard", "full")
    assert "inputs.selections.scene_graph" not in edits  # capability ainda inexistente
    for edit in plan.editable:
        if edit.current is None:
            continue
        again = runtime.resolve_config(
            files=[path], overrides=[f"{edit.path}={json.dumps(edit.current)}"]
        )
        assert again.digest == base.digest, edit.path
    switched = runtime.resolve_config(
        files=[path], overrides=["components.ingestion.source_adapter.backend=ros2_bag"]
    )
    (ingestion, *_) = runtime.resolve_plan(switched).stages
    assert ingestion.backends == {"ingestion.source_adapter": "ros2_bag"}


# --- preflight --------------------------------------------------------------------------


def test_a_runtime_with_no_injected_executors_still_composes_them_from_configuration(
    tmp_path: Path,
) -> None:
    """Blocker #2 of the PR #387 review: ``Runtime(executors={})`` is no longer a dead end.

    A frontend that never builds an executor by hand can still preflight (and run) the
    stages ``compose_executors`` can build from the configuration alone; only ``ingestion``
    (needs a request no configuration carries) and ``visual_perception`` (no real executor
    yet) remain genuinely blocked.
    """
    runtime = Runtime(
        workspace=tmp_path / "ws", module_available=_ready, environ={}
    )  # executors=None: nada injetado
    config = _config(runtime, tmp_path)

    report = runtime.preflight(config, targets=TARGET)

    assert not report.ok  # capabilities sem executor real continuam honestamente ausentes
    assert set(report.missing_executors) == {
        "ingestion",
        "visual_perception",
        "point_representation",
    }
    for stage in ("state_estimation", "geometric_mapping", "sensor_association", "semantic_fusion"):
        assert stage not in report.missing_executors


def test_supplied_providers_let_visual_perception_compose_through_the_runtime_facade(
    tmp_path: Path,
) -> None:
    """``providers=`` reaches the facade's own composition path exactly like the CLI's.

    ``sam3`` and ``qwen`` (the default fixture's region discovery and semantic
    interpretation backends) have no bundled model loader: without a ``RuntimeProvider`` for
    each, ``visual_perception`` stays a genuine ``missing_executors`` entry (see
    ``test_a_runtime_with_no_injected_executors_still_composes_them_from_configuration``).
    Supplying them through ``Runtime(providers=...)`` -- not a hand-built executor for the
    whole stage -- lets ``compose_executors`` build the real thing.
    """
    providers = {
        "visual_perception.region_discovery": lambda _config, _secrets: object(),
        "visual_perception.semantic_interpretation": lambda _config, _secrets: object(),
    }
    runtime = Runtime(
        workspace=tmp_path / "ws", providers=providers, module_available=_ready, environ={}
    )
    config = _config(runtime, tmp_path)

    report = runtime.preflight(config, targets=TARGET)

    assert "visual_perception" not in report.missing_executors


def test_a_declared_provider_target_lets_visual_perception_compose_with_no_python_providers(
    tmp_path: Path,
) -> None:
    """The facade's counterpart of the CLI's decisive #507 proof.

    Unlike ``test_supplied_providers_let_visual_perception_compose_through_the_runtime_facade``
    above, this ``Runtime`` is constructed with no ``providers=`` at all: the runtime for
    ``sam3``/``qwen`` instead comes from a ``resources.providers`` target declared in the
    resolved configuration, resolved lazily by
    ``contextmap.runtime.composition.resolve_provider`` from a real importable module.
    """
    runtime = Runtime(workspace=tmp_path / "ws", module_available=_ready, environ={})
    targets = json.dumps(
        {
            "visual_perception.region_discovery": "runtime_provider_fixtures:load_region_discovery",
            "visual_perception.semantic_interpretation": (
                "runtime_provider_fixtures:load_semantic_interpretation"
            ),
        }
    )
    config = _config(runtime, tmp_path, f"resources.providers={targets}")

    report = runtime.preflight(config, targets=TARGET)

    assert "visual_perception" not in report.missing_executors


def test_a_provider_given_to_runtime_reaches_an_optional_entity_resolution_channel(
    tmp_path: Path,
) -> None:
    """P2 #2 of the PR #540 review: ``Runtime(providers=...)`` must reach ``compose_executors``.

    ``entity_resolution.appearance``, once selected, needs a ``FeatureVectorSource`` from a
    ``RuntimeProvider`` -- the same mechanism sam3/qwen/gemini already use. Before this fix,
    ``Runtime`` had no way to accept or forward ``providers``, so selecting this optional
    channel silently dropped the whole ``entity_resolution`` executor instead of composing it.
    """
    document = selected_document()
    document["inputs"] = {**document.get("inputs", {}), "sequence": DATASET}
    document["components"]["entity_resolution"]["appearance"] = {
        "backend": "entity-appearance-comparison-v1",
        "entity-appearance-comparison-v1": {
            "embedding_space_id": "clip-vit-b32",
            "min_supporting_similarity": 0.8,
        },
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    def provide(config: Any, secrets: Any) -> object:
        return object()  # um FeatureVectorSource real não importa aqui, só a propagação

    runtime = Runtime(
        workspace=tmp_path / "ws",
        providers={"entity_resolution.appearance": provide},
        module_available=_ready,
        environ={},
    )
    config = runtime.resolve_config(files=[path])

    report = runtime.preflight(config, targets=["entity_resolution"])

    assert "entity_resolution" not in report.missing_executors


def test_preflight_succeeds_and_reports_the_identities_it_would_use(tmp_path: Path) -> None:
    runtime, world = _runtime(tmp_path)
    config = _config(runtime, tmp_path)

    report = runtime.preflight(config, targets=TARGET)

    plan = runtime.resolve_plan(config, targets=TARGET)
    assert report.ok
    assert (report.problems, report.warnings) == ((), ())
    assert report.stages == plan.run_stages
    assert (report.config_digest, report.plan_digest) == (plan.config_digest, plan.plan_digest)
    assert report.outputs["ingestion"] == "SequenceArtifact"
    assert report.outputs["semantic_fusion"] == "SemanticFusionRunArtifact"
    assert (report.missing_executors, report.predicted_reuse) == ((), {})
    assert world.runs == []


@pytest.mark.usefixtures("unavailable_future_stage")
def test_preflight_reports_every_problem_at_once(tmp_path: Path) -> None:
    runtime, _ = _runtime(
        tmp_path, executors=False, module_available=lambda name: name != "rosbags"
    )
    config = _config(runtime, tmp_path)

    report = runtime.preflight(config, targets=[*TARGET, "scene_graph", "nonexistent"])

    paths = [problem.path for problem in report.problems]
    assert not report.ok
    assert "stages.scene_graph" in paths  # capability ainda inexistente
    assert "targets.nonexistent" in paths
    assert "components.ingestion.source_adapter" in paths  # módulo opcional ausente
    assert any(
        problem.path == "stages.ingestion" and "no executor" in problem.message
        for problem in report.problems
    )
    assert "ingestion" in report.missing_executors
    assert "scene_graph" not in report.missing_executors


def test_preflight_names_a_missing_secret_without_any_value(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(
        runtime, tmp_path, 'components.visual_perception.semantic_interpretation.backend="gemini"'
    )

    report = runtime.preflight(config, targets=TARGET)

    assert not report.ok
    assert any("GEMINI_API_KEY" in problem.message for problem in report.problems)


def test_preflight_warns_when_no_workspace_can_hold_the_run(tmp_path: Path) -> None:
    runtime = Runtime(executors=world_executors(World()), module_available=_ready, environ={})
    config = runtime.resolve_config(files=[_write(tmp_path)])

    report = runtime.preflight(config, targets=TARGET)

    assert report.ok
    assert any("workspace" in warning for warning in report.warnings)
    with pytest.raises(ValueError, match="workspace"):
        runtime.run(config, targets=TARGET)
    with pytest.raises(ValueError, match="workspace"):
        runtime.list_runs()


def test_preflight_predicts_reuse_without_running_anything(tmp_path: Path) -> None:
    runtime, world = _runtime(tmp_path)
    config = _config(runtime, tmp_path)
    policy = runtime.reuse_policy(tmp_path / "index", code_identity="code-1")
    runtime.run(config, targets=TARGET, reuse=policy)
    ran = list(world.runs)

    report = runtime.preflight(config, targets=TARGET, reuse=policy)

    assert {stage: item["kind"] for stage, item in report.predicted_reuse.items()} == dict.fromkeys(
        RUN_ORDER, "reused"
    )
    assert world.runs == ran
    # Tudo será reaproveitado: nenhum executor é necessário para o que não vai rodar.
    bare = Runtime(
        workspace=tmp_path / "ws",
        verifier=lambda ref: ref.artifact_id in world.existing,
        module_available=_ready,
        environ={},
    )
    # Sem executor, a fonte do estágio de ingestion precisa ser declarada na política.
    bare_policy = bare.reuse_policy(
        tmp_path / "index", code_identity="code-1", identities=source_identities("ingestion")
    )
    bare_report = bare.preflight(config, targets=TARGET, reuse=bare_policy)
    assert bare_report.ok
    assert bare_report.missing_executors == ()


def test_selecting_upstream_runs_without_a_catalog_is_a_problem_not_a_crash(
    tmp_path: Path,
) -> None:
    runtime, world = _runtime(tmp_path)
    config = _config(runtime, tmp_path, 'inputs.selections.ingestion="seq-1"')

    report = runtime.preflight(config, targets=["visual_perception"])
    plan = runtime.resolve_plan(config, targets=["visual_perception"])
    result = runtime.run(config, targets=["visual_perception"])

    assert not report.ok
    assert "inputs.selections" in [problem.path for problem in report.problems]
    assert "inputs.selections" in [problem.path for problem in plan.problems]
    assert result.status == "blocked"
    assert [p.path for p in result.record.blocked_problems] == ["inputs.selections"]
    assert world.runs == []


def test_a_latest_selection_resolves_visibly_and_feeds_the_run(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path, 'inputs.selections.ingestion="latest"')

    report = runtime.preflight(config, targets=["visual_perception"], catalog=_catalog())
    plan = runtime.resolve_plan(config, targets=["visual_perception"], catalog=_catalog())
    result = runtime.run(config, targets=["visual_perception"], catalog=_catalog())

    assert report.ok
    assert report.provided == {"ingestion": ("seq-2",)}
    assert any("'latest' resolved to 'seq-2'" in warning for warning in report.warnings)
    assert plan.selections is not None
    assert plan.selections["stages"]["ingestion"][0]["origin"] == "latest"
    assert result.ok
    assert result.record.stages[0].inputs == {"sequence": ("seq-2",)}
    assert result.record.provided == {"ingestion": ("seq-2",)}
    assert result.record.selections is not None
    assert result.record.selections["stages"]["ingestion"][0]["artifact_id"] == "seq-2"


def test_supplied_artifacts_and_configured_selections_cannot_be_mixed(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path, 'inputs.selections.ingestion="seq-1"')
    sequence = ArtifactRef(
        stage_id="ingestion", contract="SequenceArtifact", artifact_id="seq-1", content_hash=None
    )

    with pytest.raises(ValueError, match="not both"):
        runtime.preflight(config, targets=["visual_perception"], provided={"ingestion": sequence})


# --- execution --------------------------------------------------------------------------


def test_a_run_executes_the_canonical_path_and_returns_the_persisted_record(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)

    result = runtime.run(config, targets=TARGET)

    record = result.record
    assert result.ok
    assert (result.status, result.run_id) == ("completed", "run-0001")
    assert (Path(record.directory) / "status.json").is_file()
    assert record.config_digest == config.digest
    assert record.targets == RUN_ORDER
    assert tuple(stage.stage_id for stage in record.stages) == RUN_ORDER
    assert set(_outcomes(record).values()) == {"completed"}
    assert record.backends is not None
    assert record.backends["ingestion.source_adapter"] == "ros1_bag"
    assert (record.failure, record.notes) == (None, ())
    assert all(stage.decision is None for stage in record.stages)  # sem política de reuso


def test_the_lineage_of_a_run_is_consistent_with_the_resolved_topology(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)

    record = runtime.run(config, targets=TARGET).record

    plan = runtime.resolve_plan(config, targets=TARGET)
    wiring = {s.stage_id: {i.name: i.source for i in s.inputs} for s in plan.stages}
    outputs = {}
    for stage in record.stages:
        assert stage.output is not None
        outputs[stage.stage_id] = stage.output["artifact_id"]
    for stage in record.stages:
        assert stage.inputs is not None
        for name, ids in stage.inputs.items():
            assert ids == (outputs[wiring[stage.stage_id][name]],), (stage.stage_id, name)


def test_events_reach_the_frontend_in_order_and_match_the_persisted_trail(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    sink = Sink()

    result = runtime.run(_config(runtime, tmp_path), targets=TARGET, events=sink)

    kinds = [event.kind for event in sink.events]
    assert [event.sequence for event in sink.events] == list(range(1, len(sink.events) + 1))
    assert kinds[:2] == ["run_planned", "run_started"]
    assert kinds[2:-1] == ["stage_started", "stage_completed"] * 7
    assert kinds[-1] == "run_completed"
    started = [event.stage_id for event in sink.events if event.kind == "stage_started"]
    assert tuple(started) == RUN_ORDER
    assert [e.to_document() for e in sink.events] == [e.to_document() for e in result.record.events]


def test_a_failing_stage_is_a_result_with_a_categorised_failure(tmp_path: Path) -> None:
    world = World()
    world.fail_at = "state_estimation"
    runtime, _ = _runtime(tmp_path, world)

    result = runtime.run(_config(runtime, tmp_path), targets=TARGET)

    failure = result.record.failure
    assert failure is not None
    assert (result.status, result.ok) == ("failed", False)
    assert failure["stage_id"] == "state_estimation"
    assert failure["category"] == "execution"
    assert failure["completed"] == ["ingestion", "visual_perception"]
    outcomes = _outcomes(result.record)
    assert outcomes["visual_perception"] == "completed"
    assert outcomes["state_estimation"] == "failed"
    assert outcomes["geometric_mapping"] == outcomes["semantic_fusion"] == "pending"


def test_a_blocked_run_is_a_result_and_nothing_executed(tmp_path: Path) -> None:
    runtime, world = _runtime(tmp_path, executors=False)

    result = runtime.run(_config(runtime, tmp_path), targets=TARGET)

    assert result.status == "blocked"
    assert world.runs == []
    assert any("no executor" in problem.message for problem in result.record.blocked_problems)
    assert set(_outcomes(result.record).values()) == {"pending"}


@pytest.mark.usefixtures("unavailable_future_stage")
def test_an_unimplemented_stage_blocks_the_run_explicitly(tmp_path: Path) -> None:
    runtime, world = _runtime(tmp_path)

    result = runtime.run(_config(runtime, tmp_path), targets=["scene_graph"])

    assert result.status == "blocked"
    assert world.runs == []
    assert any(
        problem.path == "stages.scene_graph" and "not implemented yet" in problem.message
        for problem in result.record.blocked_problems
    )


def test_cancellation_is_a_result_recorded_at_a_stage_boundary(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    token = CancellationToken()

    class CancelsAfterPerception(Sink):
        def emit(self, event: ExecutionEvent) -> None:
            super().emit(event)
            if event.kind == "stage_completed" and event.stage_id == "visual_perception":
                token.cancel("operator asked")

    result = runtime.run(
        _config(runtime, tmp_path),
        targets=TARGET,
        cancellation=token,
        events=CancelsAfterPerception(),
    )

    cancelled = [event for event in result.record.events if event.kind == "run_cancelled"]
    outcomes = _outcomes(result.record)
    assert result.status == "cancelled"
    assert outcomes["visual_perception"] == "completed"
    assert outcomes["state_estimation"] == "pending"
    assert [event.data["reason"] for event in cancelled] == ["operator asked"]


def test_a_keyboard_interrupt_is_recorded_as_a_cancellation_and_propagates(
    tmp_path: Path,
) -> None:
    world = World()
    world.fail_at = "visual_perception"
    world.fail_with = KeyboardInterrupt()
    runtime, _ = _runtime(tmp_path, world)

    with pytest.raises(KeyboardInterrupt):
        runtime.run(_config(runtime, tmp_path), targets=TARGET)

    (summary,) = runtime.list_runs()
    assert (summary.run_id, summary.status) == ("run-0001", "cancelled")


def test_an_event_sink_that_raises_never_changes_the_run(tmp_path: Path) -> None:
    class Broken:
        def emit(self, event: ExecutionEvent) -> None:
            raise RuntimeError("the terminal is gone")

    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)

    broken = runtime.run(config, targets=TARGET, events=Broken())
    clean = runtime.run(config, targets=TARGET)

    assert broken.ok
    assert len(broken.event_errors) == len(broken.record.events)
    assert all("the terminal is gone" in error for error in broken.event_errors)
    assert [e.kind for e in broken.record.events] == [e.kind for e in clean.record.events]
    assert clean.event_errors == ()


def test_secrets_never_reach_the_record_the_sink_or_the_disk(tmp_path: Path) -> None:
    world = World()
    world.fail_at = "visual_perception"
    world.fail_with = RuntimeError(f"quota exceeded for {SECRET}")
    runtime, _ = _runtime(tmp_path, world, environ={"GEMINI_API_KEY": SECRET})
    config = _config(
        runtime, tmp_path, 'components.visual_perception.semantic_interpretation.backend="gemini"'
    )
    sink = Sink()

    result = runtime.run(config, targets=TARGET, events=sink)

    failure = result.record.failure
    assert failure is not None
    assert result.status == "failed"
    assert "***" in failure["message"]
    assert SECRET not in json.dumps(result.to_document())
    assert SECRET not in json.dumps([event.to_document() for event in sink.events])
    on_disk = "".join(
        path.read_text(encoding="utf-8")
        for path in Path(result.record.directory).iterdir()
        if path.is_file()
    )
    assert SECRET not in on_disk


def test_a_configuration_pointing_at_another_workspace_is_refused(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    elsewhere = json.dumps(str(tmp_path / "elsewhere"))
    config = _config(runtime, tmp_path, f"resources.workspace={elsewhere}")

    with pytest.raises(ValueError, match="workspace"):
        runtime.run(config, targets=TARGET)

    assert runtime.list_runs() == ()


# --- reuse and resume -------------------------------------------------------------------


def test_reuse_and_recompute_decisions_are_visible_with_the_exact_prior_artifact(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)
    index = tmp_path / "index"
    policy = runtime.reuse_policy(index, code_identity="code-1")

    first = runtime.run(config, targets=TARGET, reuse=policy)
    second = runtime.run(config, targets=TARGET, reuse=policy)
    forced = runtime.run(
        config,
        targets=TARGET,
        reuse=runtime.reuse_policy(index, code_identity="code-1", force=["visual_perception"]),
    )

    first_outputs = {}
    for stage in first.record.stages:
        assert stage.output is not None
        assert stage.decision is not None
        assert stage.decision["kind"] == "recomputed"
        first_outputs[stage.stage_id] = stage.output["artifact_id"]
    assert first.record.code_identity == "code-1"
    for stage in second.record.stages:
        assert stage.outcome == "reused"
        assert stage.decision is not None
        assert stage.output is not None
        assert stage.decision["kind"] == "reused"
        assert stage.decision["reused_from"]["artifact_id"] == first_outputs[stage.stage_id]
        assert stage.output["artifact_id"] == first_outputs[stage.stage_id]
    decision = next(s.decision for s in forced.record.stages if s.stage_id == "visual_perception")
    assert decision is not None
    assert decision["kind"] == "recomputed"
    assert "forced" in decision["reason"]


def test_resume_reuses_what_completed_and_says_so(tmp_path: Path) -> None:
    world = World()
    world.fail_at = "state_estimation"
    runtime, _ = _runtime(tmp_path, world)
    config = _config(runtime, tmp_path)
    policy = runtime.reuse_policy(tmp_path / "index", code_identity="code-1")
    failed = runtime.run(config, targets=TARGET, reuse=policy)
    world.fail_at = None

    resumed = runtime.run(config, targets=TARGET, reuse=policy, resume=failed.run_id)

    outcomes = _outcomes(resumed.record)
    assert (failed.status, resumed.status) == ("failed", "completed")
    assert resumed.record.resumed_from == failed.run_id
    assert resumed.record.resume == {
        "from": "run-0001",
        "previous_status": "failed",
        "reused": ["ingestion", "visual_perception"],
        "recomputed": [],
    }
    assert outcomes["ingestion"] == outcomes["visual_perception"] == "reused"
    assert outcomes["state_estimation"] == "completed"


def test_a_resume_that_cannot_happen_is_refused_and_creates_no_run(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)
    policy = runtime.reuse_policy(tmp_path / "index", code_identity="code-1")
    done = runtime.run(config, targets=TARGET, reuse=policy)

    with pytest.raises(ValueError, match="reuse policy"):
        runtime.run(config, targets=TARGET, resume=done.run_id)
    with pytest.raises(ResumeError, match="already completed"):
        runtime.run(config, targets=TARGET, reuse=policy, resume=done.run_id)
    with pytest.raises(RunRecordError, match="run-0042"):
        runtime.run(config, targets=TARGET, reuse=policy, resume="run-0042")

    assert [summary.run_id for summary in runtime.list_runs()] == ["run-0001"]


def test_reuse_needs_a_verifier_and_a_code_identity(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)

    with pytest.raises(ValueError, match="verifier"):
        Runtime().reuse_policy(tmp_path / "index", code_identity="code-1")
    with pytest.raises(ValueError, match="code_identity"):
        runtime.reuse_policy(tmp_path / "index", code_identity=" ")


# --- runs -------------------------------------------------------------------------------


def test_list_runs_is_deterministic_numeric_and_lists_unreadable_records(tmp_path: Path) -> None:
    world = World()
    runtime, _ = _runtime(tmp_path, world)
    config = _config(runtime, tmp_path)
    runtime.run(config, targets=TARGET)
    world.fail_at = "ingestion"
    runtime.run(config, targets=TARGET)
    base = tmp_path / "ws" / DATASET
    (base / "run-9999").mkdir()
    (base / "run-9999" / "status.json").write_text("{not json", encoding="utf-8")
    (base / "run-10000").mkdir()
    (base / "run-abc").mkdir()
    (base / "notes").mkdir()

    runs = runtime.list_runs()

    assert [run.run_id for run in runs] == ["run-0001", "run-0002", "run-9999", "run-10000"]
    assert [(run.readable, run.status) for run in runs] == [
        (True, "completed"),
        (True, "failed"),
        (False, None),
        (False, None),
    ]
    assert runs[1].failure_category == "execution"
    assert runs[2].error is not None
    assert "status.json" in runs[2].error
    assert runtime.list_runs() == runs
    assert Runtime(workspace=tmp_path / "empty").list_runs() == ()


def test_inspect_run_accepts_a_run_id_or_a_directory_and_refuses_unknown_runs(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime(tmp_path)
    result = runtime.run(_config(runtime, tmp_path), targets=TARGET)

    by_id = runtime.inspect_run("run-0001")

    assert by_id == runtime.inspect_run(result.record.directory) == result.record
    assert Runtime().inspect_run(result.record.directory) == by_id
    with pytest.raises(RunRecordError, match="run-0042"):
        runtime.inspect_run("run-0042")
    with pytest.raises(RunRecordError, match="run-0001"):
        Runtime().inspect_run("run-0001")  # sem workspace, só um diretório é uma referência


def test_inspect_run_never_infers_what_the_record_does_not_hold(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)
    policy = runtime.reuse_policy(tmp_path / "index", code_identity="code-1")
    plain = runtime.run(config, targets=TARGET)
    runtime.run(config, targets=TARGET, reuse=policy)
    reused = runtime.run(config, targets=TARGET, reuse=policy)
    assert all(stage.inputs is not None for stage in reused.record.stages)  # de execution.json

    for run in (plain, reused):
        (Path(run.record.directory) / "execution.json").unlink()
    (Path(reused.record.directory) / "effective_config.json").unlink()
    without_execution = runtime.inspect_run(plain.run_id)
    without_anything = runtime.inspect_run(reused.run_id)

    # Sem execution.json, um estágio executado ainda tem as entradas do evento `stage_started`...
    perception = next(s for s in without_execution.stages if s.stage_id == "visual_perception")
    ingestion = next(s for s in without_execution.stages if s.stage_id == "ingestion")
    assert ingestion.output is not None
    assert perception.inputs == {"sequence": (ingestion.output["artifact_id"],)}
    # ...mas um estágio reaproveitado não tem: as entradas ficam desconhecidas, nunca deduzidas.
    assert all(stage.inputs is None for stage in without_anything.stages)
    assert without_anything.backends is None
    assert without_anything.selections is None
    assert without_anything.resume is None
    assert any("execution.json" in note for note in without_anything.notes)
    assert any("backends are unknown" in note for note in without_anything.notes)


def test_repeated_queries_return_equivalent_documents(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    first, second = _config(runtime, tmp_path), _config(runtime, tmp_path)
    runtime.run(first, targets=TARGET)

    def snapshot(config: Any) -> list[Any]:
        return [
            runtime.status().to_document(),
            [capability.to_document() for capability in runtime.capabilities()],
            runtime.resolve_plan(config, targets=TARGET).to_document(),
            runtime.preflight(config, targets=TARGET).to_document(),
            [summary.to_document() for summary in runtime.list_runs()],
            runtime.inspect_run("run-0001").to_document(),
        ]

    assert first.digest == second.digest
    assert snapshot(first) == snapshot(second)


def test_every_contract_is_json_serialisable(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    config = _config(runtime, tmp_path)
    result = runtime.run(config, targets=TARGET)

    documents = [
        runtime.status().to_document(),
        [capability.to_document() for capability in runtime.capabilities()],
        runtime.resolve_plan(config, targets=TARGET).to_document(),
        runtime.preflight(config, targets=TARGET).to_document(),
        result.to_document(),
        [summary.to_document() for summary in runtime.list_runs()],
        runtime.inspect_run(result.run_id).to_document(),
    ]

    for document in documents:
        assert json.loads(json.dumps(document, allow_nan=False)) == json.loads(json.dumps(document))


def test_two_runtimes_share_no_hidden_state(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    first, world = _runtime(tmp_path / "one")
    second, _ = _runtime(tmp_path / "two", executors=False)

    first.run(_config(first, tmp_path / "one"), targets=TARGET)

    assert len(first.list_runs()) == 1
    assert second.list_runs() == ()
    assert second.status().executors == ()
    assert world.runs


# --- ingestion --------------------------------------------------------------------------


def test_the_ingestion_service_is_composed_from_the_configuration(tmp_path: Path) -> None:
    runtime = Runtime(module_available=_ready, environ={})
    config = runtime.resolve_config(files=[_write(tmp_path)])
    missing = Runtime(module_available=lambda name: name != "rosbags", environ={})

    assert isinstance(runtime.ingestion(config), IngestionService)
    with pytest.raises(BackendUnavailableError, match="rosbags"):
        missing.ingestion(missing.resolve_config(files=[_write(tmp_path)]))
    with pytest.raises(ConfigurationError, match="source_adapter"):
        runtime.ingestion(runtime.resolve_config())  # nenhum adapter selecionado


def test_an_injected_adapter_factory_backs_the_ingestion_service(tmp_path: Path) -> None:
    runtime = Runtime(adapter_factory=factory(), environ={})
    config = runtime.resolve_config(files=[_write(tmp_path)])

    report = runtime.ingestion(config).preflight(request(tmp_path))

    assert report.ok, report.problems


def test_list_runs_spans_every_dataset_of_the_workspace_and_orders_by_dataset_then_number(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime(tmp_path)
    base = tmp_path / "ws"
    for dataset, names in (
        ("corridor-03", ["run-0002"]),
        ("corridor-02", ["run-0010", "run-0002"]),
    ):
        for name in names:
            (base / dataset / name).mkdir(parents=True)

    runs = runtime.list_runs()

    assert [(run.run_id, run.dataset) for run in runs] == [
        ("run-0002", "corridor-02"),
        ("run-0010", "corridor-02"),
        ("run-0002", "corridor-03"),
    ]


def test_a_run_id_present_in_two_datasets_is_ambiguous_and_never_guessed(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    for dataset in ("corridor-02", "corridor-03"):
        (tmp_path / "ws" / dataset / "run-0001").mkdir(parents=True)

    with pytest.raises(RunRecordError, match="several datasets"):
        runtime.inspect_run("run-0001")
