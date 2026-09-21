"""Tests for explicit run selection and lineage binding across multi-run experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.runtime import (
    ArtifactRef,
    CatalogEntry,
    ConfigurationError,
    Lineage,
    PreflightError,
    StageRequest,
    StaticCatalog,
    check_lineage,
    preflight,
    resolve_plan,
    resolve_selections,
    run_plan,
)

_INDEX = {"n": 0}


def entry(
    stage: str,
    contract: str,
    artifact_id: str,
    index: int,
    *,
    sequence: str | None = "S1",
    selection: str | None = "sel-A",
    calibration: str | None = "cal-1",
    schema: str | None = "0.1.0",
    upstream: dict[str, tuple[str, ...]] | None = None,
    content: str | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        ref=ArtifactRef(
            stage_id=stage,
            contract=contract,
            artifact_id=artifact_id,
            content_hash=content or f"sha256:{artifact_id}",
        ),
        lineage=Lineage(
            sequence=sequence,
            selection=selection,
            calibration=calibration,
            schema_version=schema,
            upstream=upstream or {},
        ),
        run_index=index,
    )


def entries() -> list[CatalogEntry]:
    seq, perc, traj, geo = (
        "SequenceArtifact",
        "PerceptionRunArtifact",
        "StateEstimationRunArtifact",
        "GeometricMapArtifact",
    )
    return [
        entry("ingestion", seq, "seq-1", 1),
        entry("ingestion", seq, "seq-2", 2, sequence="S2"),
        entry("visual_perception", perc, "perc-1", 1, upstream={"ingestion": ("seq-1",)}),
        entry("visual_perception", perc, "perc-2", 2, upstream={"ingestion": ("seq-1",)}),
        entry("visual_perception", perc, "perc-3", 3, sequence="S2"),
        entry("state_estimation", traj, "traj-1", 1),
        entry("state_estimation", traj, "traj-1b", 2),
        entry("state_estimation", traj, "traj-2", 3, sequence="S2"),
        entry(
            "geometric_mapping",
            geo,
            "map-1",
            1,
            upstream={"ingestion": ("seq-1",), "state_estimation": ("traj-1",)},
        ),
        entry(
            "sensor_association",
            "SensorAssociationRunArtifact",
            "assoc-1",
            1,
            upstream={"visual_perception": ("perc-1",), "geometric_mapping": ("map-1",)},
        ),
        entry(
            "sensor_association",
            "SensorAssociationRunArtifact",
            "assoc-2",
            2,
            upstream={"visual_perception": ("perc-2",), "geometric_mapping": ("map-1",)},
        ),
    ]


def _plan(tmp_path: Path, inputs: dict[str, Any] | None = None) -> Any:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    if inputs is not None:
        document["inputs"] = inputs
    effective = effective_from(tmp_path, document)
    return resolve_plan(effective), effective.config.inputs


def _resolve(tmp_path: Path, inputs: dict[str, Any], **options: Any) -> Any:
    plan, config_inputs = _plan(tmp_path, inputs)
    return resolve_selections(plan, config_inputs, StaticCatalog(entries()), **options)


def _ids(resolved: Any) -> dict[str, list[str]]:
    return {
        stage: [run.entry.ref.artifact_id for run in runs] for stage, runs in resolved.runs.items()
    }


class TestExplicitSelection:
    def test_exact_ids_select_exactly_those_artifacts(self, tmp_path: Path) -> None:
        resolved = _resolve(
            tmp_path, {"selections": {"ingestion": "seq-1", "state_estimation": "traj-1"}}
        )

        assert resolved.ok
        assert _ids(resolved) == {"ingestion": ["seq-1"], "state_estimation": ["traj-1"]}
        assert resolved.provided["ingestion"] == (
            ArtifactRef(
                stage_id="ingestion",
                contract="SequenceArtifact",
                artifact_id="seq-1",
                content_hash="sha256:seq-1",
            ),
        )

    def test_a_stage_that_is_not_listed_is_never_selected_implicitly(self, tmp_path: Path) -> None:
        resolved = _resolve(tmp_path, {"selections": {"ingestion": "seq-1"}})

        assert set(resolved.provided) == {"ingestion"}

    def test_the_same_selection_and_configuration_resolve_the_same_input_set(
        self, tmp_path: Path
    ) -> None:
        inputs = {"selections": {"visual_perception": ["perc-2", "perc-1"], "ingestion": "seq-1"}}
        plan, config_inputs = _plan(tmp_path, inputs)

        catalog_a = StaticCatalog(entries())
        shuffled = entries()
        random.Random(7).shuffle(shuffled)
        catalog_b = StaticCatalog(shuffled)

        first = resolve_selections(plan, config_inputs, catalog_a).to_document()
        second = resolve_selections(plan, config_inputs, catalog_b).to_document()

        assert first == second
        assert [run["artifact_id"] for run in first["stages"]["visual_perception"]] == [
            "perc-1",
            "perc-2",
        ]

    def test_an_unknown_artifact_or_one_of_another_stage_is_a_problem(self, tmp_path: Path) -> None:
        resolved = _resolve(
            tmp_path, {"selections": {"ingestion": "seq-404", "visual_perception": "seq-1"}}
        )

        text = " ".join(problem.message for problem in resolved.problems)
        assert "seq-404" in text
        assert "belongs to stage 'ingestion'" in text

    def test_named_selections_resolve_to_their_exact_artifacts_and_say_so(
        self, tmp_path: Path
    ) -> None:
        resolved = _resolve(
            tmp_path,
            {
                "named": {"canonical-v1": {"ingestion": "seq-1", "state_estimation": "traj-1"}},
                "selections": {
                    "ingestion": "named:canonical-v1",
                    "state_estimation": "named:canonical-v1",
                },
            },
        )

        assert resolved.ok
        assert _ids(resolved) == {"ingestion": ["seq-1"], "state_estimation": ["traj-1"]}
        document = resolved.to_document()
        assert document["stages"]["ingestion"][0]["origin"] == "named:canonical-v1"

    def test_the_selection_grammar_is_validated_with_the_configuration(
        self, tmp_path: Path
    ) -> None:
        bad: list[dict[str, Any]] = [
            {"selections": {"nope": "seq-1"}},
            {"selections": {"ingestion": "named:missing"}},
            {"selections": {"ingestion": ["latest", "seq-1"]}},
            {"selections": {"ingestion": []}},
            {"selections": {"ingestion": ["seq-1", "seq-1"]}},
            {"named": {"n": {"ingestion": "latest"}}},
            {"named": {"n": {"nope": "x"}}},
        ]
        for inputs in bad:
            with pytest.raises(ConfigurationError, match="inputs"):
                _plan(tmp_path, inputs)


class TestLatestCompatible:
    def test_latest_is_used_only_when_asked_for_and_is_compatible_with_the_rest(
        self, tmp_path: Path
    ) -> None:
        resolved = _resolve(
            tmp_path, {"selections": {"ingestion": "seq-1", "visual_perception": "latest"}}
        )

        # perc-3 é a mais recente, mas é de outra sequência: a última compatível é perc-2.
        assert _ids(resolved)["visual_perception"] == ["perc-2"]
        assert resolved.to_document()["stages"]["visual_perception"][0]["origin"] == "latest"

    def test_latest_follows_the_run_index_not_the_insertion_order(self, tmp_path: Path) -> None:
        plan, config_inputs = _plan(tmp_path, {"selections": {"state_estimation": "latest"}})
        reversed_entries = list(reversed(entries()))

        resolved = resolve_selections(plan, config_inputs, StaticCatalog(reversed_entries))

        assert _ids(resolved)["state_estimation"] == ["traj-2"]

    def test_downstream_latest_respects_upstream_choices(self, tmp_path: Path) -> None:
        resolved = _resolve(
            tmp_path,
            {
                "selections": {
                    "ingestion": "seq-1",
                    "visual_perception": "perc-1",
                    "sensor_association": "latest",
                }
            },
        )

        # assoc-2 é mais recente, mas foi construída sobre perc-2: a compatível é assoc-1.
        assert _ids(resolved)["sensor_association"] == ["assoc-1"]

    def test_no_compatible_run_is_a_problem(self, tmp_path: Path) -> None:
        resolved = _resolve(
            tmp_path,
            {
                "selections": {
                    "ingestion": "seq-2",
                    "sensor_association": "latest",
                }
            },
        )

        assert not resolved.ok
        assert any("compatible" in problem.message for problem in resolved.problems)


class TestMultipleRuns:
    def test_repeated_inference_runs_stay_distinct_and_share_the_physical_identity(
        self, tmp_path: Path
    ) -> None:
        resolved = _resolve(
            tmp_path,
            {"selections": {"ingestion": "seq-1", "visual_perception": ["perc-1", "perc-2"]}},
        )

        assert resolved.ok
        runs = resolved.runs["visual_perception"]
        assert [run.entry.ref.artifact_id for run in runs] == ["perc-1", "perc-2"]
        assert {run.entry.lineage.sequence for run in runs} == {"S1"}
        assert len({run.entry.ref.content_hash for run in runs}) == 2

    def test_a_single_run_input_refuses_several_runs(self, tmp_path: Path) -> None:
        resolved = _resolve(tmp_path, {"selections": {"state_estimation": ["traj-1", "traj-1b"]}})

        assert not resolved.ok
        assert any(
            "geometric_mapping" in problem.path and "one run" in problem.message
            for problem in resolved.problems
        )

    def test_the_consumer_receives_every_selected_run(self, tmp_path: Path) -> None:
        plan, config_inputs = _plan(
            tmp_path,
            {"selections": {"ingestion": "seq-1", "visual_perception": ["perc-1", "perc-2"]}},
        )
        resolved = resolve_selections(plan, config_inputs, StaticCatalog(entries()))
        seen: dict[str, StageRequest] = {}

        class Recorder:
            def __init__(self, stage: str, contract: str) -> None:
                self.stage, self.contract = stage, contract

            def execute(self, request: StageRequest) -> ArtifactRef:
                seen[self.stage] = request
                return ArtifactRef(
                    stage_id=self.stage, contract=self.contract, artifact_id=f"{self.stage}#1"
                )

        executors = {s.stage_id: Recorder(s.stage_id, s.output or "") for s in plan.stages}
        run_plan(
            plan.scope(targets=["sensor_association"], selections=resolved),
            executors,
            environ={},
            module_available=lambda _n: True,
            provided_runtimes=(
                "visual_perception.region_discovery",
                "visual_perception.dense_features",
                "visual_perception.region_features",
                "visual_perception.semantic_interpretation",
            ),
        )

        perception = seen["sensor_association"].inputs["perception"]
        assert [ref.artifact_id for ref in perception] == ["perc-1", "perc-2"]


class TestLineageCompatibility:
    def test_a_different_physical_sequence_fails_before_execution(self, tmp_path: Path) -> None:
        plan, config_inputs = _plan(
            tmp_path, {"selections": {"ingestion": "seq-1", "visual_perception": "perc-3"}}
        )
        resolved = resolve_selections(plan, config_inputs, StaticCatalog(entries()))
        ran: list[str] = []

        class Executor:
            def execute(self, request: StageRequest) -> ArtifactRef:  # pragma: no cover
                ran.append(request.stage_id)
                raise AssertionError("must not run")

        assert not resolved.ok
        text = " ".join(problem.message for problem in resolved.problems)
        assert "sequence" in text and "perc-3" in text and "seq-1" in text
        with pytest.raises(PreflightError):
            run_plan(
                plan.scope(targets=["visual_perception"], selections=resolved),
                {"visual_perception": Executor()},
                environ={},
                module_available=lambda _n: True,
                provided_runtimes=(
                    "visual_perception.region_discovery",
                    "visual_perception.dense_features",
                    "visual_perception.region_features",
                    "visual_perception.semantic_interpretation",
                ),
            )
        assert ran == []

    def test_compatibility_comes_from_the_declared_lineage_not_from_names(
        self, tmp_path: Path
    ) -> None:
        catalog = StaticCatalog(
            [
                entry("ingestion", "SequenceArtifact", "lab-S1-run", 1, sequence="S1"),
                # O nome diz S1, a linhagem declara outra sequência.
                entry(
                    "visual_perception",
                    "PerceptionRunArtifact",
                    "perception-of-lab-S1-run",
                    1,
                    sequence="S9",
                ),
            ]
        )
        plan, config_inputs = _plan(
            tmp_path,
            {
                "selections": {
                    "ingestion": "lab-S1-run",
                    "visual_perception": "perception-of-lab-S1-run",
                }
            },
        )

        resolved = resolve_selections(plan, config_inputs, catalog)

        assert not resolved.ok

    def test_calibration_and_observation_selection_must_agree(self, tmp_path: Path) -> None:
        catalog = StaticCatalog(
            [
                entry("ingestion", "SequenceArtifact", "seq-1", 1),
                entry(
                    "visual_perception",
                    "PerceptionRunArtifact",
                    "perc-x",
                    1,
                    calibration="cal-2",
                    selection="sel-B",
                ),
            ]
        )
        plan, config_inputs = _plan(
            tmp_path, {"selections": {"ingestion": "seq-1", "visual_perception": "perc-x"}}
        )

        resolved = resolve_selections(plan, config_inputs, catalog)

        text = " ".join(problem.message for problem in resolved.problems)
        assert "calibration" in text and "selection" in text

    def test_a_downstream_artifact_must_have_been_built_from_the_selected_upstream(
        self, tmp_path: Path
    ) -> None:
        resolved = _resolve(
            tmp_path,
            {
                "selections": {
                    "ingestion": "seq-1",
                    "visual_perception": "perc-2",
                    "sensor_association": "assoc-1",
                }
            },
        )

        assert not resolved.ok
        text = " ".join(problem.message for problem in resolved.problems)
        assert "assoc-1" in text and "perc-1" in text and "perc-2" in text

    def test_the_geometry_identity_is_part_of_the_dependency_check(self, tmp_path: Path) -> None:
        catalog = StaticCatalog(
            [
                *entries(),
                entry("geometric_mapping", "GeometricMapArtifact", "map-9", 2, upstream={}),
            ]
        )
        plan, config_inputs = _plan(
            tmp_path,
            {"selections": {"geometric_mapping": "map-9", "sensor_association": "assoc-1"}},
        )

        resolved = resolve_selections(plan, config_inputs, catalog)

        assert any("map-1" in problem.message for problem in resolved.problems)

    def test_unsupported_schema_versions_are_refused(self, tmp_path: Path) -> None:
        resolved = _resolve(
            tmp_path,
            {"selections": {"ingestion": "seq-1"}},
            supported_schemas={"SequenceArtifact": {"0.2.0"}},
        )

        assert any("0.1.0" in problem.message for problem in resolved.problems)

    def test_the_expected_sequence_identity_is_an_explicit_assertion(self, tmp_path: Path) -> None:
        resolved = _resolve(tmp_path, {"sequence": "S2", "selections": {"ingestion": "seq-1"}})

        assert any("S2" in problem.message for problem in resolved.problems)

    def test_check_lineage_reports_every_disagreement(self) -> None:
        problems = check_lineage(
            [
                entry("ingestion", "SequenceArtifact", "a", 1, sequence="S1", calibration="c1"),
                entry(
                    "visual_perception",
                    "PerceptionRunArtifact",
                    "b",
                    1,
                    sequence="S2",
                    calibration="c2",
                ),
            ]
        )

        assert len(problems) == 2


class TestRecordedSelections:
    def test_the_execution_record_persists_the_resolved_selection_and_exact_inputs(
        self, tmp_path: Path
    ) -> None:
        plan, config_inputs = _plan(
            tmp_path, {"selections": {"ingestion": "seq-1", "state_estimation": "traj-1"}}
        )
        resolved = resolve_selections(plan, config_inputs, StaticCatalog(entries()))

        class Recorder:
            def __init__(self, stage: str, contract: str) -> None:
                self.stage, self.contract = stage, contract

            def execute(self, request: StageRequest) -> ArtifactRef:
                return ArtifactRef(
                    stage_id=self.stage, contract=self.contract, artifact_id=f"{self.stage}#1"
                )

        record = run_plan(
            plan.scope(targets=["geometric_mapping"], selections=resolved),
            {s.stage_id: Recorder(s.stage_id, s.output or "") for s in plan.stages},
            environ={},
            module_available=lambda _n: True,
        )

        document = record.to_document()
        assert document["selections"]["stages"]["ingestion"][0]["artifact_id"] == "seq-1"
        inputs = document["stages"][0]["inputs"]
        assert inputs["sequence"][0]["artifact_id"] == "seq-1"
        assert inputs["trajectory"][0]["artifact_id"] == "traj-1"

    def test_scoping_takes_selections_or_provided_artifacts_but_not_both(
        self, tmp_path: Path
    ) -> None:
        plan, config_inputs = _plan(tmp_path, {"selections": {"ingestion": "seq-1"}})
        resolved = resolve_selections(plan, config_inputs, StaticCatalog(entries()))

        with pytest.raises(ValueError, match="both"):
            plan.scope(selections=resolved, provided={})

    def test_selection_problems_reach_preflight(self, tmp_path: Path) -> None:
        plan, config_inputs = _plan(
            tmp_path, {"selections": {"ingestion": "seq-1", "visual_perception": "perc-3"}}
        )
        resolved = resolve_selections(plan, config_inputs, StaticCatalog(entries()))

        report = preflight(plan.scope(targets=["visual_perception"], selections=resolved))

        assert any("perc-3" in problem.message for problem in report.problems)
