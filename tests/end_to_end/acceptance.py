"""Gate results of the synthetic chain, as an acceptance report over the CI scenario.

Only `fake_contract` evidence: the report shows which gates the chain can decide today and names
the capability that blocks each of the others. It never claims a gate it did not check.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from chain import SyntheticChain, cross_stage_inputs

from contextmap.artifact import (
    Severity,
    ValidationLevel,
    validate_context_map_artifact,
)
from contextmap.evaluation.canonical_scenario import canonical_ci_scenario
from contextmap.evaluation.cross_stage import (
    CrossStageReport,
    check_cross_stage,
    cross_stage_gate_results,
)
from contextmap.evaluation.end_to_end import (
    AcceptanceReport,
    ArtifactRecord,
    EvidenceClass,
    GateResult,
    assemble_acceptance_report,
)

_CONTRACT = EvidenceClass.FAKE_CONTRACT
# check_cross_stage() now covers every boundary through the ContextMapArtifact (issue #178):
# nothing is left unverified for the synthetic chain.
_MISSING: tuple[str, ...] = ()


def _inventory(manifest: object) -> dict[str, str]:
    return {entry.path: entry.content_hash for entry in manifest.file_inventory}  # type: ignore[attr-defined]


def inventory_digest(manifest: object) -> str:
    """Digest of the contractual inventory of one artifact: path and content hash of every file."""
    text = json.dumps(_inventory(manifest), sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def stage_artifacts(chain: SyntheticChain) -> dict[str, ArtifactRecord]:
    """The exact artifact of each stage the chain produced."""
    return {
        "ingestion": ArtifactRecord(
            artifact_id=str(chain.sequence.manifest.artifact_id),
            digest=inventory_digest(chain.sequence.manifest),
        ),
        "state_estimation": ArtifactRecord(
            artifact_id=str(chain.trajectory.manifest.run_id),
            digest=inventory_digest(chain.trajectory.manifest),
        ),
        "geometric_mapping": ArtifactRecord(
            artifact_id=str(chain.geometry.manifest.run_id),
            digest=inventory_digest(chain.geometry.manifest),
        ),
        "semantic_fusion": ArtifactRecord(
            artifact_id=str(chain.fusion.manifest.run_id),
            digest=inventory_digest(chain.fusion.manifest),
        ),
        "semantic_mapping": ArtifactRecord(
            artifact_id=str(chain.mapping.manifest.run_id),
            digest=inventory_digest(chain.mapping.manifest),
        ),
        "entity_resolution": ArtifactRecord(
            artifact_id=str(chain.resolution.manifest.run_id),
            digest=inventory_digest(chain.resolution.manifest),
        ),
        "spatial_relations": ArtifactRecord(
            artifact_id=str(chain.relations.manifest.run_id),
            digest=inventory_digest(chain.relations.manifest),
        ),
    }


def rerun_is_equivalent(first: SyntheticChain, second: SyntheticChain) -> bool:
    """Whether two runs produced the same contractual file hashes at every stage."""
    pairs = [
        (first.sequence.manifest, second.sequence.manifest),
        (first.trajectory.manifest, second.trajectory.manifest),
        (first.geometry.manifest, second.geometry.manifest),
        (first.fusion.manifest, second.fusion.manifest),
        (first.mapping.manifest, second.mapping.manifest),
        (first.resolution.manifest, second.resolution.manifest),
        (first.relations.manifest, second.relations.manifest),
        *(
            (a[1].manifest, b[1].manifest)
            for a, b in zip(first.associations, second.associations, strict=True)
        ),
    ]
    identities = (
        validate_context_map_artifact(
            chain.context_map_dir, level=ValidationLevel.STRUCTURAL
        ).content_identity
        for chain in (first, second)
    )
    return (
        all(_inventory(left) == _inventory(right) for left, right in pairs)
        and len(set(identities)) == 1
    )


def _findings_for(report: CrossStageReport, capability: str) -> list[str]:
    return [f.message for f in report.findings if f.failing_capability == capability]


def _decide(
    gate_id: str,
    capability: str,
    problems: Iterable[str],
    refs: list[str],
    detail: str,
) -> GateResult:
    found = list(problems)
    if found:
        return GateResult.failed(
            gate_id,
            evidence_class=_CONTRACT,
            evidence_refs=refs,
            failing_capabilities=(capability,),
            detail="; ".join(found[:3]),
        )
    return GateResult.passed(gate_id, evidence_class=_CONTRACT, evidence_refs=refs, detail=detail)


def contract_results(
    chain: SyntheticChain, *, rerun: SyntheticChain | None = None
) -> list[GateResult]:
    """One result per gate of the CI scenario, decided only from what the chain checked."""
    cross = check_cross_stage(cross_stage_inputs(chain))
    artifacts = stage_artifacts(chain)
    rejected = [frame for outcome, _ in chain.associations for frame in outcome.rejected]
    results: dict[str, GateResult] = {}

    def ref(stage: str) -> list[str]:
        return [f"{stage}:{artifacts[stage].artifact_id}@{artifacts[stage].digest}"]

    results["ingestion.sequence_integrity"] = _decide(
        "ingestion.sequence_integrity",
        "ingestion",
        chain.sequence.verify_integrity(),
        ref("ingestion"),
        "sequence artifact verifies; 3 images, 3 scans, 3 poses under one clock",
    )
    results["state_estimation.trajectory_coverage"] = _decide(
        "state_estimation.trajectory_coverage",
        "state_estimation",
        [*chain.trajectory.verify_integrity(), *(f"rejected frame {f}" for f in rejected)],
        ref("state_estimation"),
        "trajectory verifies and every associated image resolved a pose",
    )
    results["state_estimation.accuracy"] = GateResult.not_evaluated(
        "state_estimation.accuracy",
        detail="not applicable: the trajectory is the declared pose input, not an estimate",
    )
    results["geometric_mapping.map_frame_consistency"] = GateResult.not_evaluated(
        "geometric_mapping.map_frame_consistency",
        detail=(
            "partially checked: the map verifies and shares the trajectory frame and clock, and "
            "every reference resolves; transform traces and the artifact round trip were not run"
        ),
    )
    results["sensor_association.projection_validity"] = _decide(
        "sensor_association.projection_validity",
        "sensor_association",
        [
            *(p for _, reader in chain.associations for p in reader.verify_integrity()),
            *_findings_for(cross, "sensor_association"),
        ],
        [f"sensor_association:{r.manifest.run_id}" for _, r in chain.associations],
        "association runs verify; every reference resolves in the run's map",
    )
    results["semantic_fusion.evidence_preservation"] = _decide(
        "semantic_fusion.evidence_preservation",
        "semantic_fusion",
        [*chain.fusion.verify_integrity(), *_findings_for(cross, "semantic_fusion")],
        ref("semantic_fusion"),
        "fusion verifies; repeated inference grouped by physical frame, no claim dropped",
    )
    results["entity_resolution.identity_lineage"] = _decide(
        "entity_resolution.identity_lineage",
        "entity_resolution",
        chain.resolution.verify_integrity(),
        ref("entity_resolution"),
        "resolution run verifies; every merge, split or keep decision records its evidence",
    )
    results["spatial_relations.reference_integrity"] = _decide(
        "spatial_relations.reference_integrity",
        "spatial_relations",
        chain.relations.verify_integrity(),
        ref("spatial_relations"),
        "relations run verifies; every relation references a resolved entity of the run",
    )
    context_map_validation = validate_context_map_artifact(
        chain.context_map_dir, level=ValidationLevel.FULL
    )
    results["artifact.integrity"] = _decide(
        "artifact.integrity",
        "artifact",
        [f.message for f in context_map_validation.findings if f.severity is Severity.ERROR],
        [f"artifact:{context_map_validation.context_map_id}"],
        "context map validates: full closure, no dangling reference, no unlisted file",
    )
    for gate_id in (
        "geometric_mapping.quality_report",
        "visual_perception.region_quality",
        "visual_perception.semantic_quality",
        "sensor_association.projection_quality",
        "semantic_fusion.reference_recovery",
        # As duas linhas abaixo são gates de qualidade (GateKind.REPORT) que precisam de anotações
        # de referência (IDENTITY/RELATIONS): esta chain nunca as fabrica, mesmo tendo evidência
        # estrutural real para os gates de integridade acima.
        "entity_resolution.identity_quality",
        "spatial_relations.relation_quality",
    ):
        results[gate_id] = GateResult.not_evaluated(
            gate_id, detail="the chain does not run this evaluator; its own tests cover it"
        )
    results["visual_perception.evidence_completeness"] = GateResult.not_evaluated(
        "visual_perception.evidence_completeness",
        detail="the perception evidence is canned and in memory; no perception run artifact",
    )
    for result in cross_stage_gate_results(
        cross,
        evidence_class=_CONTRACT,
        evidence_refs=[f"chain:{artifacts['ingestion'].artifact_id}"],
        unverified_boundaries=_MISSING,
    ):
        results[result.gate_id] = result
    for gate_id in ("runtime.provenance_identity", "runtime.resource_reporting"):
        results[gate_id] = GateResult.not_evaluated(
            gate_id, detail="stage executors are wired by the runtime integration, not this chain"
        )
    if rerun is None:
        results["reproducibility.rerun_equivalence"] = GateResult.not_evaluated(
            "reproducibility.rerun_equivalence", detail="the chain ran once"
        )
    else:
        results["reproducibility.rerun_equivalence"] = _decide(
            "reproducibility.rerun_equivalence",
            "runtime",
            [] if rerun_is_equivalent(chain, rerun) else ["contractual file hashes differ"],
            ref("ingestion"),
            "two runs produced identical contractual file hashes at every stage",
        )
    results["reproducibility.interruption_recovery"] = GateResult.not_evaluated(
        "reproducibility.interruption_recovery",
        detail="atomic finalization is covered by each writer's tests; no chain-level injection",
    )
    return list(results.values())


def final_artifact(chain: SyntheticChain) -> ArtifactRecord:
    """The exact identity of the chain's ContextMapArtifact."""
    validation = validate_context_map_artifact(chain.context_map_dir, level=ValidationLevel.FULL)
    assert validation.context_map_id is not None and validation.content_identity is not None
    return ArtifactRecord(artifact_id=validation.context_map_id, digest=validation.content_identity)


def contract_report(
    chain: SyntheticChain, *, rerun: SyntheticChain | None = None
) -> AcceptanceReport:
    """The acceptance report of the synthetic chain against the CI scenario."""
    scenario = canonical_ci_scenario()
    return assemble_acceptance_report(
        scenario,
        report_id="ci-chain-report",
        run_id="ci-chain-run",
        code_version="test",
        results=contract_results(chain, rerun=rerun),
        stage_artifacts=stage_artifacts(chain),
        final_artifact=final_artifact(chain),
        limitations=(
            "synthetic formulas and canned model output: contract evidence, never real evidence",
            "entity resolution and spatial relations run with no geometric/contact evaluator "
            "selected, so every relation candidate stays honestly UNRESOLVED",
        ),
    )
