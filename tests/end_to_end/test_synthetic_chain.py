"""The synthetic Solution 1 chain: stage artifacts written and read back through public readers.

Contract evidence only (fake/contract): the geometry, poses and projections come from formulas and
the perception evidence is canned. The chain fills the gap the CI fixture catalogue declares as
"multi-view fusion not available": every stage runs its real code and persists its real artifact.
"""

from __future__ import annotations

from pathlib import Path

from acceptance import contract_report, rerun_is_equivalent
from chain import synthetic_chain

from contextmap.evaluation.canonical_scenario import canonical_ci_scenario
from contextmap.evaluation.end_to_end import (
    EvidenceClass,
    GateStatus,
    encode_acceptance_report,
    unmet_required_gates,
)


def test_every_stage_of_the_chain_persists_an_artifact_that_verifies(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path) as chain:
        assert chain.sequence.verify_integrity() == []
        assert chain.trajectory.verify_integrity() == []
        assert chain.geometry.verify_integrity() == []
        for _, association in chain.associations:
            assert association.verify_integrity() == []
        assert chain.fusion.verify_integrity() == []
        assert chain.geometry.manifest.point_count == 12
        # Duas runs de percepção: a run-b repete o frame-0000 (3 frames físicos, 4 vistas).
        assert len(chain.spatial_observations) == 8
        assert len(chain.fusion_outcomes) == 2
        assert chain.fusion_excluded == ()


def test_the_fused_evidence_keeps_the_disagreement_between_repeated_inferences(
    tmp_path: Path,
) -> None:
    with synthetic_chain(tmp_path) as chain:
        pallet, shelf = (outcome.evidence for outcome in chain.fusion_outcomes)

        labels = {hypothesis.label for hypothesis in pallet.hypotheses}
        assert labels == {"pallet", "crate"}
        assert [item.kind.value for item in pallet.uncertainty] == ["contradiction"]
        assert {hypothesis.label for hypothesis in shelf.hypotheses} == {"shelf post"}
        assert len(pallet.contributions) == 4
        assert len(pallet.physical_observation_groups) == 3


def test_rerunning_the_chain_reproduces_every_contractual_file_hash(tmp_path: Path) -> None:
    with (
        synthetic_chain(tmp_path / "first") as first,
        synthetic_chain(tmp_path / "second") as second,
    ):
        assert rerun_is_equivalent(first, second)


def test_the_contract_report_decides_only_what_the_chain_checked(tmp_path: Path) -> None:
    with (
        synthetic_chain(tmp_path / "first") as chain,
        synthetic_chain(tmp_path / "second") as rerun,
    ):
        report = contract_report(chain, rerun=rerun)

    by_gate = {result.gate_id: result for result in report.results}
    passed = {gate for gate, result in by_gate.items() if result.status is GateStatus.PASSED}
    blocked = {
        gate: r.failing_capabilities for gate, r in by_gate.items() if r.status.value == "blocked"
    }

    assert passed == {
        "ingestion.sequence_integrity",
        "state_estimation.trajectory_coverage",
        "sensor_association.projection_validity",
        "semantic_fusion.evidence_preservation",
        "reproducibility.rerun_equivalence",
    }
    assert all(by_gate[gate].evidence_class is EvidenceClass.FAKE_CONTRACT for gate in passed)
    # Os gates que dependem de ER, SR e do artifact final ficam bloqueados, nomeando a capability.
    assert blocked["entity_resolution.identity_lineage"] == ("entity_resolution",)
    assert blocked["spatial_relations.relation_quality"] == ("spatial_relations",)
    assert blocked["artifact.integrity"] == ("artifact",)
    for gate in (
        "cross_stage.lineage_closure",
        "cross_stage.coordinate_consistency",
        "cross_stage.evidence_traceability",
        "cross_stage.physical_observation_identity",
    ):
        assert blocked[gate] == ("entity_resolution", "spatial_relations", "artifact")
        assert "0 findings" in by_gate[gate].detail
        # O ExternalPose não registra calibração: a lacuna é declarada, não escondida.
        assert "records no calibration identity" in by_gate[gate].detail


def test_contract_evidence_never_validates_the_scenario(tmp_path: Path) -> None:
    scenario = canonical_ci_scenario()
    with synthetic_chain(tmp_path) as chain:
        report = contract_report(chain)

    unmet = unmet_required_gates(scenario, report)
    document = encode_acceptance_report(report)

    assert {gate.gate_id for gate in unmet} == {gate.gate_id for gate in scenario.gates}
    assert document["final_artifact"] is None
    assert set(document["stage_artifacts"]) == {
        "ingestion",
        "state_estimation",
        "geometric_mapping",
        "semantic_fusion",
    }
    assert document["scenario"]["evidence_class"] == "fake_contract"
