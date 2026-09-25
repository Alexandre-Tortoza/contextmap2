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
    # This report is real evidence recorded on 2026-09-21 against scenario 1.0.3's real subject,
    # whose SequenceArtifact had camera_model=None (see the "camera model" limitation below).
    # Issue #554 re-froze the real subject on a new artifact with a real MeiCameraModel in
    # 1.0.4, so this report's pinned scenario identity is no longer reproducible from the live
    # canonical_real_scenario() -- correctly so: it is evidence about the old artifact, and
    # relabeling it as evidence about the new one would misstate what was actually run (AGENTS.md
    # #8, immutability). Scenario 1.0.5 then changed the matrix itself -- it narrowed
    # reproducibility.rerun_equivalence and added reproducibility.semantic_rerun_agreement -- so
    # matrix_digest is pinned here as a literal too. Nothing about this report is reproducible
    # from live code any more, and that is the point: it is evidence about the definitions that
    # were in force when it ran.
    scenario = canonical_real_scenario()
    document = _report()

    assert document["schema"] == ACCEPTANCE_REPORT_SCHEMA
    assert document["scenario"] == {
        "scenario_id": "solution-1-canonical",
        "version": "1.0.3",
        "digest": "sha256:c96634be0bda5ca29b820851630188406da580614781fa1c498bc9dd25f3ddd9",
        "matrix_digest": "sha256:ac91ea8697efb77692584b839a81c00cdf179fa1016a4e850d9c6bdab1ed9e6a",
        "evidence_class": "real",
    }
    # O que continua exigido da matriz viva é que ela só tenha crescido: um gate removido
    # deixaria esta evidência apontando para um critério que o projeto não define mais.
    live = [gate.gate_id for gate in scenario.gates]
    recorded = [gate["gate_id"] for gate in document["gates"]]
    assert set(recorded) <= set(live)
    assert recorded == [gate_id for gate_id in live if gate_id in set(recorded)]
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
