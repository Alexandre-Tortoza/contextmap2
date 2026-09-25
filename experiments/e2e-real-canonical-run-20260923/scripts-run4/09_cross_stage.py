"""Real cross-stage check (issue #178) over the regenerated run-0003 chain (PR #438 review).

Unlike the original run against run-0001, this supplies CrossStageInputs.auxiliary_sequence: the
real pose SequenceArtifact (corridor-02-gt.txt) that state_estimation actually merged in, so
cross_stage.lineage_closure genuinely verifies the artifact that produced the trajectory --
before this fix, that check only ever compared the main bag sequence, which is not what the
poses actually came from.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.artifact import ContextMapArtifactReader
from contextmap.entity_resolution import EntityResolutionRunReader, ResolvedEntityReference
from contextmap.evaluation.cross_stage import CrossStageInputs, check_cross_stage
from contextmap.geometric_mapping import GeometricMapArtifactReader
from contextmap.ingestion import SequenceArtifactReader
from contextmap.semantic_fusion import FusionOutcome, SemanticFusionRunReader
from contextmap.semantic_mapping import EntityReference, SemanticMappingRunReader
from contextmap.sensor_association import SensorAssociationRunReader
from contextmap.spatial_relations import SpatialRelationsRunReader
from contextmap.state_estimation import StateEstimationRunReader
from contextmap.state_estimation.preflight import calibration_identity
from contextmap.visual_perception import PerceptionRun, PerceptionRunReader

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
SEQUENCE_DIR = WORKSPACE / "ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4"
POSE_SEQUENCE_DIR = WORKSPACE / "ingest-real/sequences/corridor-02-pose/e2d832c152b1493999082d4f67210b5b"
TRAJECTORY_DIR = WORKSPACE / "e2e-real/run-0003/state_estimation"
GEOMETRY_DIR = WORKSPACE / "e2e-real/run-0003/geometric_mapping"
PERCEPTION_DIR = WORKSPACE / "e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception"
ASSOCIATION_DIR = WORKSPACE / "e2e-real/run-0003/sensor_association"
FUSION_DIR = WORKSPACE / "e2e-real/run-0003/semantic_fusion"
MAPPING_DIR = WORKSPACE / "e2e-real/run-0003/semantic_mapping"
RESOLUTION_DIR = WORKSPACE / "e2e-real/run-0003/entity_resolution"
RELATIONS_DIR = WORKSPACE / "e2e-real/run-0003/spatial_relations"
CONTEXT_MAP_DIR = WORKSPACE / "e2e-real/run-0004/context_map"


def main() -> None:
    sequence_reader = SequenceArtifactReader(SEQUENCE_DIR)
    sequence = sequence_reader.manifest
    sequence_calibration = calibration_identity(sequence_reader.read_calibration())

    pose_sequence = SequenceArtifactReader(POSE_SEQUENCE_DIR).manifest

    trajectory_reader = StateEstimationRunReader(TRAJECTORY_DIR)
    trajectory = trajectory_reader.manifest
    print("trajectory.auxiliary_sequence_artifact_id:", trajectory.auxiliary_sequence_artifact_id)

    geometry_reader = GeometricMapArtifactReader(GEOMETRY_DIR)
    geometry = geometry_reader.manifest
    geometry_source = geometry_reader.geometry()

    perception_reader = PerceptionRunReader(PERCEPTION_DIR)
    perception_manifest = perception_reader.manifest
    perception_results = {result.result_id: result for result in perception_reader.list_results()}
    perception_run = PerceptionRun(
        run_id=perception_manifest.run_id,
        run_index=perception_manifest.run_index,
        sequence_artifact_id=perception_manifest.sequence_artifact_id,
        selection_id=perception_manifest.selection_id,
        enabled_capabilities=frozenset(perception_manifest.enabled_capabilities),
        backend_provenance={},
        code_version=None,
    )

    association_reader = SensorAssociationRunReader(ASSOCIATION_DIR)
    association_manifest = association_reader.manifest
    spatial_observations = {
        observation.spatial_observation_id: observation
        for observation in association_reader.observations()
    }

    fusion_reader = SemanticFusionRunReader(FUSION_DIR)
    fusion = fusion_reader.manifest
    fusion_outcomes = tuple(
        FusionOutcome(
            support=fusion_reader.support(support_id),
            evidence=fusion_reader.fused_evidence(support_id),
        )
        for support_id in fusion_reader.support_ids()
    )

    mapping_reader = SemanticMappingRunReader(MAPPING_DIR)
    mapping = mapping_reader.manifest
    entities = {
        EntityReference(semantic_map_id=entity.semantic_map_id, entity_id=entity.entity_id): entity
        for entity in mapping_reader.iter_entities()
    }

    resolution_reader = EntityResolutionRunReader(RESOLUTION_DIR)
    resolution = resolution_reader.manifest
    resolved_entities = {
        ResolvedEntityReference(
            resolution_run_id=resolved.resolution_run_id,
            resolved_entity_id=resolved.resolved_entity_id,
        ): resolved
        for resolved in resolution_reader.resolved_entities().entities
    }

    relations_reader = SpatialRelationsRunReader(RELATIONS_DIR)
    relations = relations_reader.manifest
    relation_records = tuple(relations_reader.iter_relations())

    context_map = ContextMapArtifactReader.open(CONTEXT_MAP_DIR, verify_hashes=True).context_map()

    inputs = CrossStageInputs(
        sequence=sequence,
        sequence_calibration_identity=sequence_calibration,
        trajectory=trajectory,
        geometry=geometry,
        geometry_source=geometry_source,
        perception_runs=(perception_run,),
        perception_results=perception_results,
        associations=(association_manifest,),
        spatial_observations=spatial_observations,
        fusion=fusion,
        fusion_outcomes=fusion_outcomes,
        mapping=mapping,
        entities=entities,
        resolution=resolution,
        resolved_entities=resolved_entities,
        relations=relations,
        relation_records=relation_records,
        context_map=context_map,
        auxiliary_sequence=pose_sequence,
    )

    report = check_cross_stage(inputs)
    print(f"checks_run: {dict(report.checks_run)}")
    print(f"notes: {report.notes}")
    print(f"findings ({len(report.findings)}):")
    for finding in report.findings:
        print(f"  [{finding.gate_id}] {finding.failing_capability}: {finding.message}")
    if not report.findings:
        print("NO FINDINGS -- every checked boundary held, including the auxiliary pose sequence.")


if __name__ == "__main__":
    main()
