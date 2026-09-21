"""The committed report of the partial real run stays consistent with the frozen scenario.

The report records real evidence, produced outside CI on the corridor-02 sample. CI cannot rerun it,
but it can refuse a report that no longer matches the scenario it claims to be about or that hides a
failed gate behind an aggregate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from contextmap.evaluation import ACCEPTANCE_REPORT_SCHEMA, canonical_real_scenario

REPORT = (
    Path(__file__).resolve().parents[2]
    / "src/contextmap/evaluation/docs/validation/e2e-real-sample-20260921.acceptance-report.json"
)


def _report() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(REPORT.read_text(encoding="utf-8"))
    return document


def test_the_recorded_report_is_about_the_frozen_real_scenario() -> None:
    scenario = canonical_real_scenario()
    document = _report()

    assert document["schema"] == ACCEPTANCE_REPORT_SCHEMA
    assert document["scenario"] == {
        "scenario_id": scenario.scenario_id,
        "version": scenario.version,
        "digest": scenario.digest,
        "matrix_digest": scenario.matrix_digest,
        "evidence_class": "real",
    }
    assert [gate["gate_id"] for gate in document["gates"]] == [g.gate_id for g in scenario.gates]
    assert document["final_artifact"] is None


def test_the_recorded_report_hides_no_failed_or_blocked_gate() -> None:
    document = _report()
    gates = document["gates"]

    assert Counter(gate["status"] for gate in gates) == {
        "passed": 3,
        "failed": 1,
        "blocked": 12,
        "not_evaluated": 9,
    }
    assert "passed" not in document and "score" not in document
    for gate in gates:
        if gate["status"] in ("passed", "failed"):
            assert gate["evidence_class"] == "real"
            assert gate["evidence_refs"]
        if gate["status"] in ("failed", "blocked"):
            assert gate["failing_capabilities"]
    failed = next(gate for gate in gates if gate["status"] == "failed")
    assert failed["gate_id"] == "cross_stage.lineage_closure"
    assert failed["failing_capabilities"] == ["geometric_mapping", "sensor_association"]


def test_the_recorded_report_states_what_the_run_did_not_cover() -> None:
    limitations = " ".join(_report()["limitations"])

    assert "camera model" in limitations
    assert "no semantic interpretation" in limitations
    assert "no annotated reference set" in limitations
