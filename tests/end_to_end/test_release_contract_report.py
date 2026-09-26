"""The committed v0.1.0 release acceptance report stays consistent with scenario 1.0.5.

The report decides the release contract's matrix from the evidence milestone #19's campaign
recorded against 1.0.4. CI cannot reproduce that campaign -- its artifacts were pruned -- but it
can refuse a report that drifts from the scenario it claims to be about, that hides a failed
gate, or that quietly drops the caveat saying no hash in it was re-verified.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from contextmap.evaluation import ACCEPTANCE_REPORT_SCHEMA, GateKind, canonical_real_scenario

REPORT = (
    Path(__file__).resolve().parents[2]
    / "src/contextmap/evaluation/docs/validation"
    / "v0.1.0-release-contract-20260925.acceptance-report.json"
)


def _report() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(REPORT.read_text(encoding="utf-8"))
    return document


def test_the_release_report_is_about_the_frozen_release_contract() -> None:
    """Unlike the superseded 1.0.3 record, this one must track live code.

    It is the evidence the release ships with, so it is about the contract that ships. If a
    later scenario version is ever frozen, this test fails and forces the decision instead of
    letting the release cite a matrix nobody validated.
    """
    scenario = canonical_real_scenario()
    document = _report()

    assert document["schema"] == ACCEPTANCE_REPORT_SCHEMA
    assert document["scenario"] == {
        "scenario_id": scenario.scenario_id,
        "version": "1.0.5",
        "digest": scenario.digest,
        "matrix_digest": scenario.matrix_digest,
        "evidence_class": "real",
    }
    assert [gate["gate_id"] for gate in document["gates"]] == [g.gate_id for g in scenario.gates]


def test_every_invariant_gate_of_the_release_contract_is_met_with_real_evidence() -> None:
    """Re-derives the release-readiness rule instead of trusting a recorded count."""
    scenario = canonical_real_scenario()
    kind = {gate.gate_id: gate.kind for gate in scenario.gates}

    unmet = [
        gate["gate_id"]
        for gate in _report()["gates"]
        if kind[gate["gate_id"]] is GateKind.INVARIANT
        and not (gate["status"] == "passed" and gate["evidence_class"] == "real")
    ]

    assert unmet == []


def test_the_release_report_hides_no_failed_gate_and_blocks_only_report_gates() -> None:
    scenario = canonical_real_scenario()
    kind = {gate.gate_id: gate.kind for gate in scenario.gates}
    document = _report()
    gates = document["gates"]

    assert Counter(gate["status"] for gate in gates) == {
        "passed": 18,
        "blocked": 5,
        "not_evaluated": 3,
    }
    assert "passed" not in document and "score" not in document
    # Nada que não seja gate de relatório pode ficar sem decisão: um invariante bloqueado ou não
    # avaliado seria uma lacuna do contrato, não uma medida ausente.
    for gate in gates:
        if gate["status"] in ("blocked", "not_evaluated"):
            assert kind[gate["gate_id"]] is GateKind.REPORT, gate["gate_id"]
        if gate["status"] == "blocked":
            assert gate["failing_capabilities"]


def test_the_report_keeps_saying_that_nothing_in_it_was_re_measured() -> None:
    """The caveat is the honesty of the document; dropping it would overstate the evidence."""
    limitations = " ".join(_report()["limitations"])

    assert "not a rerun" in limitations
    assert "pruned" in limitations
    assert "re-verified" in limitations
    # E que a equivalência vale sob a condição estreitada, não de sensores a artifact.
    assert "same PerceptionRunArtifact" in limitations
    assert "#556" in limitations
