"""Assemble the v0.1.0 release acceptance report against scenario 1.0.5 (issues #183, #190).

This is **not** a rerun and does not claim to be one. The real artifacts of run-0005 were
pruned from ``outputs/`` after the campaign, so nothing here re-measures or re-verifies a
hash. What it does is decide scenario 1.0.5's matrix using the evidence run-0005 actually
recorded, which is a different question from the one 1.0.4 asked:

* 24 gates are carried forward verbatim, with their recorded status, detail and evidence
  references. Their definitions did not change between 1.0.4 and 1.0.5.
* ``reproducibility.rerun_equivalence`` is re-decided, because 1.0.5 narrowed it to repeated
  runs **from the same PerceptionRunArtifact**. Run-0005 measured exactly that property and
  found it holds; it failed 1.0.4's wider, conjunctive requirement. The evidence is the same;
  the requirement it is judged against is not.
* ``reproducibility.semantic_rerun_agreement`` is new in 1.0.5 and is a report gate: run-0005's
  measured agreement is reported with its denominator and no pass threshold.
* ``visual_perception.semantic_quality`` gained ``semantic.parse_failure_rate`` in 1.0.5. It was
  blocked for want of an annotated reference set and stays blocked: a new required metric does
  not unblock a gate that has no reference data.

Scenario 1.0.4 is untouched, still fails, and remains the authoritative record of its campaign.

Usage:
    python 17_release_contract_report.py <output-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from contextmap.evaluation import (
    ArtifactRecord,
    EvidenceClass,
    GateKind,
    GateResult,
    assemble_acceptance_report,
    canonical_real_scenario,
    unmet_required_gates,
    write_acceptance_report,
)

CAMPAIGN = Path(__file__).resolve().parents[1]
SOURCE = CAMPAIGN / "acceptance-report-run0005.json"

REPORT_ID = "acceptance-report-v0.1.0-release-contract"
# Nome do documento commitado, na convenção de src/contextmap/evaluation/docs/validation/.
DOCUMENT_STEM = "v0.1.0-release-contract-20260925"

# Os dois gates que a 1.0.5 redefiniu; todo o resto é carregado literalmente.
_REDECIDED = "reproducibility.rerun_equivalence"
_NEW = "reproducibility.semantic_rerun_agreement"

_PRUNED = (
    "This report re-decides scenario 1.0.5's matrix from the evidence recorded by run-0005 "
    "against 1.0.4; it is not a rerun. The real artifacts under outputs/e2e-real/ were pruned "
    "after that campaign, so no hash in it was re-verified here and no measurement was repeated. "
    "Every evidence reference names what run-0005 recorded, and the committed "
    "acceptance-report-run0005.json remains the primary record."
)
_CONDITIONED = (
    "reproducibility.rerun_equivalence is satisfied under 1.0.5's narrowed requirement only: "
    "repeated runs from the same PerceptionRunArtifact. Recorded sensors to ContextMapArtifact is "
    "NOT claimed to be reproducible end to end while the Qwen3-VL backend participates in the "
    "chain -- that is what 1.0.4 required and what run-0005 measured as failing, and it is why "
    "the backend is declared experimental in 1.0.5 (see issue #556)."
)


def _carried(record: dict[str, object]) -> GateResult:
    """Rebuild one recorded gate result, preserving its status, evidence and wording."""
    status = str(record["status"])
    gate_id = str(record["gate_id"])
    detail = str(record["detail"])
    refs = [str(ref) for ref in record["evidence_refs"]]  # type: ignore[union-attr]
    blame = [str(name) for name in record["failing_capabilities"]]  # type: ignore[union-attr]
    if status == "passed":
        return GateResult.passed(
            gate_id,
            evidence_class=EvidenceClass(str(record["evidence_class"])),
            evidence_refs=refs,
            detail=detail,
        )
    if status == "failed":
        return GateResult.failed(
            gate_id,
            evidence_class=EvidenceClass(str(record["evidence_class"])),
            evidence_refs=refs,
            failing_capabilities=blame,
            detail=detail,
        )
    if status == "blocked":
        return GateResult.blocked(gate_id, blocked_by=blame, detail=detail)
    return GateResult.not_evaluated(gate_id, detail=detail)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    output_dir = Path(sys.argv[1])
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    recorded = {str(gate["gate_id"]): gate for gate in source["gates"]}
    if source["scenario"]["version"] != "1.0.4":
        raise SystemExit(f"expected 1.0.4 evidence, got {source['scenario']['version']}")

    scenario = canonical_real_scenario()
    if scenario.version != "1.0.5":
        raise SystemExit(f"this script targets 1.0.5, live scenario is {scenario.version}")

    results = []
    for gate in scenario.gates:
        if gate.gate_id == _NEW:
            results.append(
                GateResult.passed(
                    gate.gate_id,
                    evidence_class=EvidenceClass.REAL,
                    evidence_refs=recorded[_REDECIDED]["evidence_refs"],  # type: ignore[arg-type]
                    detail=(
                        "Reported with its denominator, no pass threshold: two independent real "
                        "executions of the identical canonical configuration (SAM2.1-hiera-tiny + "
                        "DINOv2-base + CLIP-ViT-L/14 + Qwen3-VL-4B-Instruct nf4, greedy, "
                        "temperature=0.0) over the same 20 real corridor-02 frames agreed on "
                        "29/90 compared canonical claims (32.2%), comparing hypothesis, role, "
                        "category and confidence per source_observation_id/region_id. Region "
                        "discovery matched on 0/20 frames. semantic.rerun_agreement.rate = "
                        "29/90 = 0.322. The mechanism is not isolated and is not asserted here; "
                        "issue #556 investigates it."
                    ),
                )
            )
        elif gate.gate_id == _REDECIDED:
            results.append(
                GateResult.passed(
                    gate.gate_id,
                    evidence_class=EvidenceClass.REAL,
                    evidence_refs=recorded[_REDECIDED]["evidence_refs"],  # type: ignore[arg-type]
                    detail=(
                        "Judged against 1.0.5's narrowed requirement: repeated runs from the same "
                        "PerceptionRunArtifact. Run-0005 measured exactly that and found it holds "
                        "-- scripts-run5/14_metric_report_equivalence.py found zero metric-report "
                        "mismatches (counts, diagnostics, warnings, excluding legitimately "
                        "run-specific identity) across all eight stages from state_estimation "
                        "through context_map between two independent real reruns, and "
                        "scripts-run4/12_rerun_content_equivalence.py proved exact entity and "
                        "relation content equivalence including ambiguity status and relation "
                        "state. The same evidence fails 1.0.4's wider conjunctive requirement, "
                        "which also covered the generative backend; 1.0.4 keeps that verdict."
                    ),
                )
            )
        else:
            results.append(_carried(recorded[gate.gate_id]))

    report = assemble_acceptance_report(
        scenario,
        report_id=REPORT_ID,
        run_id=str(source["run_id"]),
        code_version=str(source["code_version"]),
        results=results,
        stage_artifacts={
            stage: ArtifactRecord(artifact_id=str(value["artifact_id"]), digest=str(value["digest"]))
            for stage, value in (source["stage_artifacts"] or {}).items()
        },
        final_artifact=(
            ArtifactRecord(
                artifact_id=str(source["final_artifact"]["artifact_id"]),
                digest=str(source["final_artifact"]["digest"]),
            )
            if source["final_artifact"]
            else None
        ),
        limitations=(_PRUNED, _CONDITIONED, *[str(item) for item in source["limitations"]]),
    )

    unmet = unmet_required_gates(scenario, report, kinds=frozenset({GateKind.INVARIANT}))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_acceptance_report(output_dir / f"{DOCUMENT_STEM}.acceptance-report.json", report)

    statuses: dict[str, int] = {}
    for result in report.results:
        statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
    print(json.dumps({"statuses": statuses, "unmet_invariant_gates": [g.gate_id for g in unmet]}))
    return 0 if not unmet else 1


if __name__ == "__main__":
    sys.exit(main())
