"""Canonical end-to-end scenario, acceptance matrix and acceptance report contract.

The scenario is the frozen definition of what ``Solution 1 validated`` means. These
tests protect its identity (a committed snapshot regenerated from code), its
structure (required versus ablation-only stages, stage-specific gates that name a
capability) and the report rules that keep a failed gate from hiding behind an
overall label.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.canonical_scenario import (
    canonical_ci_scenario,
    canonical_real_scenario,
)
from contextmap.evaluation.end_to_end import (
    ACCEPTANCE_REPORT_SCHEMA,
    CROSS_STAGE,
    SCENARIO_ID,
    SCENARIO_SCHEMA,
    SCENARIO_VERSION,
    AcceptanceGate,
    AcceptanceReport,
    ArtifactRecord,
    EvidenceClass,
    GateKind,
    GateResult,
    GateStatus,
    ScenarioError,
    assemble_acceptance_report,
    encode_acceptance_report,
    encode_scenario,
    scenario_runtime_document,
    scenario_snapshot_name,
    unmet_required_gates,
    write_acceptance_report,
)
from contextmap.evaluation.metrics import default_metric_registry
from contextmap.semantic_fusion import (
    BASELINE_ACCUMULATION_POLICY_ID,
    GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
    QUALITY_AWARE_ACCUMULATION_POLICY_ID,
)

SNAPSHOT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "src/contextmap/evaluation/docs/scenarios"
)

# Áreas que a issue #176 exige como gates próprios, cada uma atribuível a um estágio.
REQUIRED_AREAS = {
    "ingestion": "ingestion",
    "pose": "state_estimation",
    "geometry": "geometric_mapping",
    "perception": "visual_perception",
    "association": "sensor_association",
    "fusion": "semantic_fusion",
    "identity": "entity_resolution",
    "relations": "spatial_relations",
    "artifact": "artifact",
    "reproducibility": "runtime",
    "resources": "runtime",
}


def _passed(gate_id: str, *, evidence_class: EvidenceClass = EvidenceClass.REAL) -> GateResult:
    return GateResult.passed(
        gate_id, evidence_class=evidence_class, evidence_refs=("artifact:example",), detail="ok"
    )


def _all_passed(
    scenario_gates: tuple[AcceptanceGate, ...], *, evidence_class: EvidenceClass
) -> list[GateResult]:
    return [_passed(gate.gate_id, evidence_class=evidence_class) for gate in scenario_gates]


def test_the_committed_scenario_snapshots_are_exactly_what_the_code_produces() -> None:
    for scenario in (canonical_real_scenario(), canonical_ci_scenario()):
        snapshot = SNAPSHOT_DIRECTORY / scenario_snapshot_name(scenario)

        assert snapshot.read_text(encoding="utf-8") == (
            json.dumps(encode_scenario(scenario), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        ), (
            "the frozen scenario changed: a new SCENARIO_VERSION and a new snapshot are required, "
            f"never an edit of {snapshot.name}"
        )


def test_the_scenario_digest_is_deterministic_and_tracks_every_frozen_decision() -> None:
    scenario = canonical_real_scenario()

    assert scenario.digest == canonical_real_scenario().digest
    assert scenario.digest.startswith("sha256:")
    other_backend = replace(
        scenario,
        stages=tuple(
            replace(
                stage,
                components=tuple(
                    replace(component, backend="sam3") if component.backend == "sam2" else component
                    for component in stage.components
                ),
            )
            for stage in scenario.stages
        ),
    )
    assert other_backend.digest != scenario.digest
    looser = replace(scenario, gates=scenario.gates[:-1])
    assert looser.digest != scenario.digest
    assert looser.matrix_digest != scenario.matrix_digest


def test_real_and_ci_scenarios_share_the_acceptance_matrix_but_not_the_evidence_class() -> None:
    real, ci = canonical_real_scenario(), canonical_ci_scenario()

    assert real.matrix_digest == ci.matrix_digest
    assert real.scenario_id == ci.scenario_id == SCENARIO_ID
    assert real.version == ci.version == SCENARIO_VERSION
    assert real.subject.evidence_class is EvidenceClass.REAL
    assert ci.subject.evidence_class is EvidenceClass.FAKE_CONTRACT
    assert real.digest != ci.digest
    assert encode_scenario(real)["schema"] == SCENARIO_SCHEMA


def test_the_real_scenario_pins_the_corridor_02_sample_it_was_frozen_from() -> None:
    subject = canonical_real_scenario().subject

    assert subject.dataset_id == "corridor-02"
    assert subject.sequence_artifact_id == "e145f73f8d894f18b96ef1f55ca308c2"
    assert subject.selection.clock_id == "corridor-02-header"
    assert subject.selection.selection_identity == (
        "sha256:dc641b345ffc142cbc50452bbaacef2433990478295f4720feb0f165ee4ed1c4"
    )
    assert subject.selection.end_ns - subject.selection.start_ns == 90_000_000_000
    assert len(subject.selection.selected_images) == 20
    assert subject.selection.lidar_scan_count == 892
    # A pose do ExternalPose é declarada como entrada, nunca como referência de avaliação.
    (pose_input,) = subject.external_inputs
    assert pose_input.name == "pose_source"
    assert pose_input.role == "pose_input"
    assert pose_input.digest is not None
    assert pose_input.digest.startswith("sha256:")
    # Ainda não há reference set anotado para o corridor-02: nada é inventado.
    assert subject.reference_set is None


def test_the_ci_scenario_pins_the_committed_synthetic_reference_set() -> None:
    subject = canonical_ci_scenario().subject
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures/ci_subset/1.0.1/manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert subject.reference_set is not None
    assert subject.reference_set.reference_set_id == manifest["reference_set_id"]
    assert subject.reference_set.version == manifest["version"]
    assert subject.reference_set.digest == manifest["digest"]


def test_required_stages_follow_the_canonical_topology_and_optional_ones_are_ablation_only() -> (
    None
):
    scenario = canonical_real_scenario()
    stage_ids = [stage.stage_id for stage in scenario.stages]

    assert stage_ids == [
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
    options = {(option.component_id, option.option) for option in scenario.ablation_only}
    # PTv3 e os demais opcionais só entram por ablação, nunca pelo perfil canônico.
    assert ("point_representation", "off-vs-deterministic-vs-ptv3") in options
    assert ("semantic_fusion.accumulation", QUALITY_AWARE_ACCUMULATION_POLICY_ID) in options
    assert ("visual_perception.semantic_interpretation", "gemini") in options
    assert ("visual_perception.dense_features", "feature-resolution-enhancement") in options
    assert all(option.reason for option in scenario.ablation_only)


def test_the_canonical_profile_names_only_local_backends_and_the_fusion_policies_that_exist() -> (
    None
):
    scenario = canonical_real_scenario()
    selected = {
        component.component_id: component
        for stage in scenario.stages
        for component in stage.components
    }

    assert selected["visual_perception.region_discovery"].backend == "sam2"
    assert selected["visual_perception.dense_features"].backend == "dinov2"
    assert selected["visual_perception.region_features"].backend == "clip"
    assert selected["visual_perception.semantic_interpretation"].backend == "qwen"
    assert selected["state_estimation.estimator"].backend == "external_pose"
    assert selected["semantic_fusion.support"].backend == GEOMETRY_OVERLAP_SUPPORT_POLICY_ID
    assert selected["semantic_fusion.accumulation"].backend == BASELINE_ACCUMULATION_POLICY_ID
    # Nenhum componente canônico depende de serviço externo: custo/disponibilidade não entram.
    assert not any(component.external_service for component in selected.values())
    assert "point_representation" not in {stage.stage_id for stage in scenario.stages}


def test_every_required_area_has_a_gate_owned_by_one_capability() -> None:
    gates = canonical_real_scenario().gates

    for area, capability in REQUIRED_AREAS.items():
        assert any(capability == gate.capability for gate in gates), area
    assert {gate.capability for gate in gates} >= set(REQUIRED_AREAS.values()) | {CROSS_STAGE}
    assert len({gate.gate_id for gate in gates}) == len(gates)


def test_gate_metrics_exist_in_the_registry_and_annotations_are_named_families() -> None:
    registry = default_metric_registry()
    names = {definition.name for definition in registry.definitions}
    gates = canonical_real_scenario().gates

    for gate in gates:
        assert set(gate.metrics) <= names, gate.gate_id
        for family in gate.needs_reference_annotations:
            assert isinstance(family, AnnotationFamily)
    # Sem score global: nenhum gate agrega outro; cada um tem o seu enunciado e a sua evidência.
    assert all(gate.requirement and gate.evidence for gate in gates)


def test_quality_gates_need_reference_annotations_and_invariants_do_not_hide_behind_them() -> None:
    gates = {gate.gate_id: gate for gate in canonical_real_scenario().gates}

    assert gates["entity_resolution.identity_quality"].kind is GateKind.REPORT
    assert (
        AnnotationFamily.IDENTITY
        in gates["entity_resolution.identity_quality"].needs_reference_annotations
    )
    assert gates["spatial_relations.relation_quality"].needs_reference_annotations == (
        AnnotationFamily.RELATIONS,
    )
    for gate in gates.values():
        if gate.kind is GateKind.INVARIANT:
            assert gate.needs_reference_annotations == (), gate.gate_id


def test_a_gate_must_name_the_capability_that_owns_its_failure() -> None:
    gate = canonical_real_scenario().gates[0]

    with pytest.raises(ScenarioError, match="capability"):
        replace(gate, capability="")
    with pytest.raises(ScenarioError, match="requirement"):
        replace(gate, requirement=" ")


def test_a_passed_or_failed_result_needs_evidence_and_a_blocked_one_names_its_blocker() -> None:
    with pytest.raises(ScenarioError, match="evidence"):
        GateResult.passed(
            "x.y", evidence_class=EvidenceClass.REAL, evidence_refs=(), detail="nothing shown"
        )
    with pytest.raises(ScenarioError, match="failing capability"):
        GateResult.failed(
            "x.y",
            evidence_class=EvidenceClass.REAL,
            evidence_refs=("artifact:a",),
            failing_capabilities=(),
            detail="broken",
        )
    with pytest.raises(ScenarioError, match="blocked_by"):
        GateResult.blocked("x.y", blocked_by=(), detail="waiting")

    blocked = GateResult.blocked(
        "entity_resolution.identity_lineage",
        blocked_by=("entity_resolution",),
        detail="capability not implemented on dev",
    )
    assert blocked.status is GateStatus.BLOCKED
    assert blocked.evidence_class is None
    assert blocked.failing_capabilities == ("entity_resolution",)


def test_the_report_needs_exactly_one_result_per_gate_of_the_scenario() -> None:
    scenario = canonical_ci_scenario()
    results = _all_passed(scenario.gates, evidence_class=EvidenceClass.FAKE_CONTRACT)

    with pytest.raises(ScenarioError, match="missing"):
        assemble_acceptance_report(
            scenario, report_id="r", run_id="run", code_version="c", results=results[1:]
        )
    with pytest.raises(ScenarioError, match="unknown"):
        assemble_acceptance_report(
            scenario,
            report_id="r",
            run_id="run",
            code_version="c",
            results=[*results, _passed("not.a.gate")],
        )
    with pytest.raises(ScenarioError, match="twice"):
        assemble_acceptance_report(
            scenario, report_id="r", run_id="run", code_version="c", results=[*results, results[0]]
        )


def test_a_failed_gate_stays_visible_and_names_the_capability_instead_of_an_overall_label() -> None:
    scenario = canonical_real_scenario()
    results = _all_passed(scenario.gates, evidence_class=EvidenceClass.REAL)
    broken_index = next(
        index
        for index, r in enumerate(results)
        if r.gate_id == "geometric_mapping.map_frame_consistency"
    )
    results[broken_index] = GateResult.failed(
        "geometric_mapping.map_frame_consistency",
        evidence_class=EvidenceClass.REAL,
        evidence_refs=("artifact:map-0001",),
        failing_capabilities=("geometric_mapping",),
        detail="map frame differs from the trajectory frame",
    )

    report = assemble_acceptance_report(
        scenario, report_id="r", run_id="run", code_version="c", results=results
    )
    unmet = unmet_required_gates(scenario, report)

    assert [(gate.gate_id, gate.capabilities, gate.status) for gate in unmet] == [
        ("geometric_mapping.map_frame_consistency", ("geometric_mapping",), GateStatus.FAILED)
    ]
    assert not hasattr(report, "passed")
    assert not hasattr(report, "score")


def test_blocked_and_not_evaluated_gates_are_unmet_and_explained() -> None:
    scenario = canonical_real_scenario()
    results = _all_passed(scenario.gates, evidence_class=EvidenceClass.REAL)
    results[0] = GateResult.blocked(
        results[0].gate_id, blocked_by=("ingestion",), detail="sequence not available"
    )
    results[1] = GateResult.not_evaluated(results[1].gate_id, detail="not run in this pass")

    unmet = unmet_required_gates(
        scenario,
        assemble_acceptance_report(
            scenario, report_id="r", run_id="run", code_version="c", results=results
        ),
    )

    assert [gate.status for gate in unmet] == [GateStatus.BLOCKED, GateStatus.NOT_EVALUATED]
    assert all(gate.reason for gate in unmet)


def test_release_blocking_filter_excludes_report_gates_without_hiding_invariants() -> None:
    scenario = canonical_real_scenario()
    results = _all_passed(scenario.gates, evidence_class=EvidenceClass.REAL)
    quality_index = next(
        index
        for index, r in enumerate(results)
        if r.gate_id == "entity_resolution.identity_quality"
    )
    frame_index = next(
        index
        for index, r in enumerate(results)
        if r.gate_id == "geometric_mapping.map_frame_consistency"
    )
    assert scenario.gates[quality_index].kind is GateKind.REPORT
    assert scenario.gates[frame_index].kind is GateKind.INVARIANT
    results[quality_index] = GateResult.blocked(
        "entity_resolution.identity_quality",
        blocked_by=("entity_resolution",),
        detail="identity annotations not available",
    )
    results[frame_index] = GateResult.failed(
        "geometric_mapping.map_frame_consistency",
        evidence_class=EvidenceClass.REAL,
        evidence_refs=("artifact:map-0001",),
        failing_capabilities=("geometric_mapping",),
        detail="map frame differs from the trajectory frame",
    )
    report = assemble_acceptance_report(
        scenario, report_id="r", run_id="run", code_version="c", results=results
    )

    unfiltered = unmet_required_gates(scenario, report)
    assert {gate.gate_id for gate in unfiltered} == {
        "entity_resolution.identity_quality",
        "geometric_mapping.map_frame_consistency",
    }

    release_blocking = unmet_required_gates(scenario, report, kinds=frozenset({GateKind.INVARIANT}))
    assert [gate.gate_id for gate in release_blocking] == [
        "geometric_mapping.map_frame_consistency"
    ]


def test_contract_evidence_alone_never_satisfies_a_gate() -> None:
    scenario = canonical_ci_scenario()
    report = assemble_acceptance_report(
        scenario,
        report_id="r",
        run_id="run",
        code_version="c",
        results=_all_passed(scenario.gates, evidence_class=EvidenceClass.FAKE_CONTRACT),
    )

    unmet = unmet_required_gates(scenario, report)

    assert {gate.gate_id for gate in unmet} == {gate.gate_id for gate in scenario.gates}
    assert all("contract" in gate.reason for gate in unmet)


def test_a_fully_real_report_leaves_no_unmet_gate() -> None:
    scenario = canonical_real_scenario()
    report = assemble_acceptance_report(
        scenario,
        report_id="r",
        run_id="run",
        code_version="c",
        results=_all_passed(scenario.gates, evidence_class=EvidenceClass.REAL),
        stage_artifacts={
            "ingestion": ArtifactRecord(artifact_id="seq", digest="sha256:" + "0" * 64)
        },
        final_artifact=ArtifactRecord(artifact_id="ctx", digest="sha256:" + "1" * 64),
    )

    assert unmet_required_gates(scenario, report) == ()
    assert report.final_artifact is not None


def test_the_report_records_scenario_identity_limitations_and_exact_artifacts(
    tmp_path: Path,
) -> None:
    scenario = canonical_real_scenario()
    report = assemble_acceptance_report(
        scenario,
        report_id="report-0001",
        run_id="run-0001",
        code_version="ac1eb59",
        results=_all_passed(scenario.gates, evidence_class=EvidenceClass.REAL),
        stage_artifacts={
            "ingestion": ArtifactRecord(artifact_id="seq", digest="sha256:" + "0" * 64)
        },
        limitations=("no annotated reference set for corridor-02",),
    )

    document = encode_acceptance_report(report)

    assert document["schema"] == ACCEPTANCE_REPORT_SCHEMA
    assert document["scenario"] == {
        "scenario_id": SCENARIO_ID,
        "version": SCENARIO_VERSION,
        "digest": scenario.digest,
        "matrix_digest": scenario.matrix_digest,
        "evidence_class": "real",
    }
    assert document["limitations"] == ["no annotated reference set for corridor-02"]
    assert document["final_artifact"] is None
    assert [result["gate_id"] for result in document["gates"]] == [
        gate.gate_id for gate in scenario.gates
    ]
    assert "passed" not in document and "score" not in document
    assert isinstance(report, AcceptanceReport)

    path = tmp_path / "acceptance-report.json"
    write_acceptance_report(path, report)
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(document))
    with pytest.raises(FileExistsError):
        write_acceptance_report(path, report)


def test_the_scenario_selects_its_backends_in_a_runtime_configuration_document() -> None:
    scenario = canonical_real_scenario()

    document = scenario_runtime_document(scenario)

    assert document["pipeline"] == {
        "preset": "canonical/1",
        # Point Representation é só por ablação: o perfil canônico a mantém desligada.
        "stages": {"point_representation": False},
    }
    components = document["components"]
    assert components["visual_perception"]["region_discovery"] == {"backend": "sam2"}
    assert components["visual_perception"]["semantic_interpretation"] == {"backend": "qwen"}
    assert components["state_estimation"]["estimator"] == {"backend": "external_pose"}
    assert components["semantic_fusion"]["accumulation"] == {
        "backend": BASELINE_ACCUMULATION_POLICY_ID
    }
    # Só a seleção de backend: checkpoint, revisão e limiares pertencem a quem os possui.
    assert all(
        set(slot) == {"backend"}
        for capability in components.values()
        for slot in capability.values()
    )
    selected = {
        component.component_id for stage in scenario.stages for component in stage.components
    }
    assert {
        f"{capability}.{name}" for capability, slots in components.items() for name in slots
    } == selected
