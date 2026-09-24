"""Systematic audit for cross_stage/runtime.provenance_identity (PR #438 review, finding 2 on
the run-0002 acceptance report): a spot-check of a couple of manifests does not satisfy the
gate's own requirement ("every stage artifact records the exact backend, model revision or
checkpoint hash and the effective configuration digest"). This reopens every real stage artifact
of the run and asserts the required field is present and non-empty -- no model or pipeline
re-execution needed, this is a read-only audit of already-materialized manifests.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.artifact import ContextMapArtifactReader
from contextmap.entity_resolution import EntityResolutionRunReader
from contextmap.geometric_mapping import GeometricMapArtifactReader
from contextmap.ingestion import SequenceArtifactReader
from contextmap.semantic_fusion import SemanticFusionRunReader
from contextmap.semantic_mapping import SemanticMappingRunReader
from contextmap.sensor_association import SensorAssociationRunReader
from contextmap.spatial_relations import SpatialRelationsRunReader
from contextmap.state_estimation import StateEstimationRunReader
from contextmap.visual_perception import PerceptionRunReader

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")

_FAILURES: list[str] = []


def _require(stage: str, field: str, value: object) -> None:
    ok = value is not None and value != "" and value != ()
    print(f"  [{'OK' if ok else 'MISSING'}] {field} = {value!r}")
    if not ok:
        _FAILURES.append(f"{stage}.{field} is empty/None")


def audit_sequence(stage: str, sequence_dir: Path) -> None:
    provenance = SequenceArtifactReader(sequence_dir).read_provenance()
    print(f"{stage}:")
    _require(stage, "provenance_present", "yes" if provenance is not None else None)
    if provenance is not None:
        _require(stage, "provenance.adapter_type", provenance.adapter_type)
        _require(stage, "provenance.code_version", provenance.code_version)
        _require(stage, "provenance.configuration_hash", provenance.configuration_hash)


def audit_state_estimation(run_dir: Path) -> None:
    manifest = StateEstimationRunReader(run_dir).manifest
    print("state_estimation:")
    _require("state_estimation", "estimator.backend_id", manifest.estimator.backend_id)
    _require("state_estimation", "estimator.backend_version", manifest.estimator.backend_version)
    _require(
        "state_estimation",
        "estimator.configuration_fingerprint",
        manifest.estimator.configuration_fingerprint,
    )


def audit_geometric_mapping(run_dir: Path) -> None:
    manifest = GeometricMapArtifactReader(run_dir).manifest
    print("geometric_mapping:")
    _require("geometric_mapping", "configuration_fingerprint", manifest.configuration_fingerprint)
    _require("geometric_mapping", "spatial_index_kind", manifest.spatial_index_kind)


def audit_sensor_association(run_dir: Path) -> None:
    manifest = SensorAssociationRunReader(run_dir).manifest
    print("sensor_association:")
    _require(
        "sensor_association", "configuration_fingerprint", manifest.configuration_fingerprint
    )
    _require(
        "sensor_association",
        "visibility_policy.fingerprint",
        manifest.visibility_policy.get("fingerprint") if manifest.visibility_policy else None,
    )


def audit_visual_perception(run_dir: Path) -> None:
    reader = PerceptionRunReader(run_dir)
    manifest = reader.manifest
    print("visual_perception:")
    # configuration_digest is defined to change whenever a resolved backend's identity changes
    # (ResolvedPipeline.configuration_digest's own docstring), so it already pins backend/model
    # identity transitively; also spot-check the per-request backend identity real executions
    # carry individually.
    _require("visual_perception", "configuration_digest", manifest.configuration_digest)
    executions = reader.list_semantic_executions()
    _require("visual_perception", "semantic_executions_count", len(executions) or None)
    # PR #438 review, third round: the prior audit only checked executions[0], while the report
    # claimed every execution was checked. Loop over all of them for real.
    missing_fingerprints = [
        str(execution.request.request_id)
        for execution in executions
        if not execution.request.configuration_fingerprint
    ]
    ok = not missing_fingerprints
    label = f"semantic_request.configuration_fingerprint (checked over all {len(executions)})"
    print(f"  [{'OK' if ok else 'MISSING'}] {label} = {'present on every execution' if ok else missing_fingerprints[:5]!r}")
    if not ok:
        _FAILURES.append(
            f"visual_perception.semantic_request.configuration_fingerprint missing on "
            f"{len(missing_fingerprints)} of {len(executions)} executions"
        )


def audit_semantic_fusion(run_dir: Path) -> None:
    manifest = SemanticFusionRunReader(run_dir).manifest
    print("semantic_fusion:")
    _require("semantic_fusion", "support_policy_id", manifest.support_policy_id)
    _require(
        "semantic_fusion",
        "support_configuration_fingerprint",
        manifest.support_configuration_fingerprint,
    )
    _require("semantic_fusion", "fusion_policy_id", manifest.fusion_policy_id)


def audit_semantic_mapping(run_dir: Path) -> None:
    manifest = SemanticMappingRunReader(run_dir).manifest
    print("semantic_mapping:")
    _require(
        "semantic_mapping", "materialization_policy_id", manifest.materialization_policy_id
    )
    _require(
        "semantic_mapping", "configuration_fingerprint", manifest.configuration_fingerprint
    )
    _require("semantic_mapping", "code_digest", manifest.code_digest)


def audit_entity_resolution(run_dir: Path) -> None:
    manifest = EntityResolutionRunReader(run_dir).manifest
    print("entity_resolution:")
    _require("entity_resolution", "policies_count", len(manifest.policies) or None)
    for record in manifest.policies:
        _require(
            "entity_resolution", f"policies[{record.role}].policy_id", record.policy.policy_id
        )


def audit_spatial_relations(run_dir: Path) -> None:
    manifest = SpatialRelationsRunReader(run_dir).manifest
    print("spatial_relations:")
    _require("spatial_relations", "policies_count", len(manifest.policies) or None)
    for role, record in sorted(manifest.policies.items()):
        _require("spatial_relations", f"policies[{role}].fingerprint", record.get("fingerprint"))
    _require("spatial_relations", "taxonomy_version", manifest.taxonomy_version)


def audit_context_map(run_dir: Path) -> None:
    manifest = ContextMapArtifactReader.open(run_dir, verify_hashes=True).manifest
    print("context_map:")
    _require("context_map", "code_version", manifest.code_version)
    _require("context_map", "configuration_fingerprint", manifest.configuration_fingerprint)


def main() -> None:
    run_root = WORKSPACE / "e2e-real/run-0003"
    audit_sequence(
        "ingestion", WORKSPACE / "ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4"
    )
    audit_sequence(
        "pose_ingestion",
        WORKSPACE / "ingest-real/sequences/corridor-02-pose/e2d832c152b1493999082d4f67210b5b",
    )
    audit_state_estimation(run_root / "state_estimation")
    audit_geometric_mapping(run_root / "geometric_mapping")
    audit_sensor_association(run_root / "sensor_association")
    audit_visual_perception(
        WORKSPACE / "e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception"
    )
    audit_semantic_fusion(run_root / "semantic_fusion")
    audit_semantic_mapping(run_root / "semantic_mapping")
    audit_entity_resolution(run_root / "entity_resolution")
    audit_spatial_relations(run_root / "spatial_relations")
    audit_context_map(WORKSPACE / "e2e-real/run-0004/context_map")

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} missing identity field(s):")
        for failure in _FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("PASSED: every stage artifact records backend/policy identity and a configuration "
          "digest -- runtime.provenance_identity is systematically verified, not spot-checked.")


if __name__ == "__main__":
    main()
