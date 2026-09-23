"""A contract-level Solution 1 chain over the synthetic CI sequence, through the real artifacts.

Every stage runs the real capability code and persists its real artifact, then is read back through
the capability's public reader. Only the model outputs are fake: the perception evidence (masks and
semantic claims) is hand-written, because a model is exactly what CI must not depend on. Nothing
here is evidence about real data.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from contextmap.artifact import (
    CONTEXT_MAP_ASSEMBLY_POLICY_ID,
    ArtifactKind,
    ContextMapId,
    ContextMapMetadata,
    DeclaredCapabilities,
    MapCapability,
    MapCreation,
    ObservationWindow,
    PolicyRef,
    SourceSequence,
    UpstreamArtifact,
    artifact_digest,
    assemble_context_map_with_metrics,
    estimator_local_map_frame,
    write_context_map_with_metrics,
)
from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    EntityResolutionRunId,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
    lineage_from_mapping_manifest,
    materialize_resolved_entities,
    resolve_candidate_pairs,
    retrieve_candidate_sets,
)
from contextmap.evaluation.ci_fixtures import (
    CI_FIXTURE_ID,
    LANDMARKS,
    build_synthetic_sequence,
    project_landmark,
)
from contextmap.evaluation.cross_stage import CrossStageInputs
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    MapDebugLevel,
    MotionCorrectionPolicy,
    ScanDisposition,
    assemble_geometry_inputs_from_artifacts,
)
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    FullSequenceSelection,
    ImageObservation,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SourceObservationId,
)
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    ExcludedObservation,
    FusionOutcome,
    FusionRunLineage,
    GeometryOverlapSupportPolicy,
    SemanticFusionRunId,
    SemanticFusionRunReader,
    SemanticFusionRunWriter,
    accumulate_baseline_evidence,
    build_fusion_supports,
    group_by_physical_observation,
)
from contextmap.semantic_mapping import (
    EntityMaterializationPolicy,
    GeometrySummaryPolicy,
    SemanticMapId,
    SemanticMappingRunId,
    SemanticMappingRunReader,
    SemanticMappingRunWriter,
    lineage_from_fusion_manifest,
    materialize_entities,
)
from contextmap.sensor_association import (
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationOutcome,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
    SpatialObservation,
    SpatialObservationId,
)
from contextmap.sensor_association.service import AssociationFrameInput
from contextmap.shared import SourceTimestamp
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    FrameConventions,
    RelationPredicate,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunReader,
    SpatialRelationsRunWriter,
    decide_relations,
    generate_relation_candidates,
    lineage_from_resolution_manifest,
    resolved_entity_geometries,
)
from contextmap.state_estimation import (
    BODY_ENDPOINT,
    GeometryRequirements,
    LookupPolicy,
    StateEstimationRequest,
    StateEstimationRunId,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    StaticRelationRequirement,
    TrajectoryId,
    TrajectoryLookup,
    calibration_identity,
    execute_state_estimation,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox2D,
    ClaimId,
    HypothesisRole,
    InlineMask,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRun,
    PerceptionRunId,
    Region2D,
    RegionId,
    SemanticClaim,
    SemanticInferenceProvenance,
    SourceImage,
    prepare_image,
)

SEQUENCE_NAME = CI_FIXTURE_ID
SEQUENCE_ARTIFACT_ID = SequenceArtifactId("ci-fixture")
REFERENCE_FRAME = FrameId("odom")
BODY_FRAME = FrameId("base_link")
_CAMERA_CALIBRATION = CalibrationReferenceId("front_camera-calib")
PALLET_REGION = RegionId("region-pallet")
SHELF_REGION = RegionId("region-shelf")
RUN_A = PerceptionRunId("run-a")
RUN_B = PerceptionRunId("run-b")

# Duas inferências sobre a mesma observação física (frame-0000): a run-b repete o frame com outro
# modelo e discorda sobre a palete. As demais observações vêm só da run-a.
_CLAIMS: dict[tuple[str, int, RegionId], tuple[tuple[str, HypothesisRole, float], ...]] = {
    ("run-a", 0, PALLET_REGION): (
        ("pallet", HypothesisRole.PRIMARY, 0.8),
        ("crate", HypothesisRole.ALTERNATIVE, 0.15),
    ),
    ("run-a", 0, SHELF_REGION): (("shelf post", HypothesisRole.PRIMARY, 0.7),),
    ("run-a", 1, PALLET_REGION): (("pallet", HypothesisRole.PRIMARY, 0.7),),
    ("run-a", 1, SHELF_REGION): (("shelf post", HypothesisRole.PRIMARY, 0.6),),
    ("run-a", 2, PALLET_REGION): (("pallet", HypothesisRole.PRIMARY, 0.9),),
    ("run-a", 2, SHELF_REGION): (("unknown", HypothesisRole.PRIMARY, 0.4),),
    ("run-b", 0, PALLET_REGION): (("crate", HypothesisRole.PRIMARY, 0.6),),
    ("run-b", 0, SHELF_REGION): (("shelf post", HypothesisRole.PRIMARY, 0.9),),
}
_LANDMARK_BY_REGION = {PALLET_REGION: "pallet-corner", SHELF_REGION: "shelf-post"}
_INTERPRETER = BackendProvenance(
    backend_id="canned_interpreter",
    capability="semantic_interpreter",
    provider="fixture",
    model="canned",
    version="1",
)
_REGION_BACKEND = BackendProvenance(
    backend_id="canned_region_discovery",
    capability="region_discovery",
    provider="fixture",
    model="canned",
    version="1",
)


@dataclass(frozen=True)
class SyntheticChain:
    """Every stage artifact of the synthetic chain, opened through its public reader."""

    sequence: SequenceArtifactReader
    trajectory: StateEstimationRunReader
    geometry: GeometricMapArtifactReader
    perception_runs: tuple[PerceptionRun, ...]
    perception_results: dict[PerceptionResultId, PerceptionResult]
    associations: tuple[tuple[SensorAssociationOutcome, SensorAssociationRunReader], ...]
    fusion_outcomes: tuple[FusionOutcome, ...]
    fusion_excluded: tuple[ExcludedObservation, ...]
    fusion: SemanticFusionRunReader
    spatial_observations: dict[SpatialObservationId, SpatialObservation]
    mapping: SemanticMappingRunReader
    resolution: EntityResolutionRunReader
    relations: SpatialRelationsRunReader
    context_map_dir: Path


def _ingest(workspace: Path) -> SequenceArtifactReader:
    sequence = build_synthetic_sequence()
    directory = workspace / "ingestion"
    with SequenceArtifactWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        artifact_id=SEQUENCE_ARTIFACT_ID,
    ) as writer:
        writer.set_calibration(sequence.calibration)
        for observation in sequence.observations:
            writer.add_observation(observation)
        writer.finalize()
    return SequenceArtifactReader(directory)


def _estimate(workspace: Path, sequence: SequenceArtifactReader) -> StateEstimationRunReader:
    request = StateEstimationRequest(
        trajectory_id=TrajectoryId("ci-fixture--trajectory"),
        sequence_artifact_id=sequence.manifest.artifact_id,
        selection_id="full-sequence",
        observations=tuple(sequence.list_observations()),
        calibration=sequence.read_calibration(),
    )
    outcome = execute_state_estimation(
        ExternalPoseEstimator(
            ExternalPoseConfig(reference_frame=REFERENCE_FRAME, body_frame=BODY_FRAME)
        ),
        request,
        downstream=(
            GeometryRequirements(
                capability="geometric_mapping",
                modalities=frozenset({"lidar"}),
                static_relations=(
                    StaticRelationRequirement(from_endpoint=BODY_ENDPOINT, to_endpoint="lidar"),
                ),
            ),
        ),
    )
    directory = workspace / "state_estimation"
    StateEstimationRunWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=StateEstimationRunId("state-run-0001"),
        run_index=1,
    ).finalize(outcome)
    return StateEstimationRunReader(directory)


def _map(
    workspace: Path, sequence: SequenceArtifactReader, trajectory: StateEstimationRunReader
) -> GeometricMapArtifactReader:
    plan = assemble_geometry_inputs_from_artifacts(
        sequence=sequence,
        selection=FullSequenceSelection(),
        run=trajectory,
        pose_lookup=LookupPolicy.exact(),
        motion_correction_policy=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN
        ),
    )
    directory = workspace / "geometric_mapping"
    GeometricMapArtifactWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=GeometricMapRunId("map-run-0001"),
        run_index=1,
        debug_level=MapDebugLevel.NONE,
    ).finalize(plan=plan, aggregation=None, code_version="test")
    return GeometricMapArtifactReader(directory)


def _landmark_world(name: str) -> tuple[float, float, float]:
    return next(point for label, point in LANDMARKS if label == name)


def _mask_around_landmark(region: RegionId, frame_index: int) -> InlineMask:
    """A one-pixel mask on the analytic projection of the region's landmark.

    A single pixel keeps the two regions disjoint at 16 x 12: each contains exactly the three
    copies (one per scan) of its own landmark.
    """
    projection = project_landmark(_landmark_world(_LANDMARK_BY_REGION[region]), frame_index)
    assert projection["pixel"] is not None, "a region is only drawn for a visible landmark"
    # A associação usa o centro do pixel em coordenadas inteiras, então o pixel é o arredondado.
    u, v = (round(value) for value in projection["pixel"])
    return InlineMask(
        width=16,
        height=12,
        data=tuple(x == u and y == v for y in range(12) for x in range(16)),
    )


def _perception_result(
    run: PerceptionRunId,
    frame_index: int,
    *,
    claims_of: str | None = None,
    sequence_artifact_id: str | None = None,
) -> PerceptionResult:
    observation_id = SourceObservationId(f"frame-{frame_index:04d}")
    result_id = PerceptionResultId(f"{run}--{observation_id}")
    regions = []
    claims = []
    for region_id in (PALLET_REGION, SHELF_REGION):
        regions.append(
            Region2D(
                region_id=region_id,
                bounding_box=BoundingBox2D(x=0.0, y=0.0, width=16.0, height=12.0),
                provenance=_REGION_BACKEND,
                mask=_mask_around_landmark(region_id, frame_index),
                image_width=16,
                image_height=12,
                is_accepted=True,
                rejection_reason=None,
                source_observation_id=observation_id,
            )
        )
        for claim_index, (label, role, confidence) in enumerate(
            _CLAIMS.get((claims_of or str(run), frame_index, region_id), ())
        ):
            claims.append(
                SemanticClaim(
                    claim_id=ClaimId(f"{result_id}--{region_id}--claim-{claim_index:04d}"),
                    source_observation_id=observation_id,
                    perception_result_id=result_id,
                    hypothesis=label,
                    role=role,
                    provenance=SemanticInferenceProvenance(
                        backend=_INTERPRETER,
                        task_identity="canned-task",
                        prompt_template_id="canned-prompt-v1",
                        output_schema_version="schema-v1",
                    ),
                    confidence=confidence,
                    region_id=region_id,
                )
            )
    return PerceptionResult(
        result_id=result_id,
        source_observation_id=observation_id,
        run_id=run,
        sequence_artifact_id=sequence_artifact_id or str(SEQUENCE_ARTIFACT_ID),
        created_at="2026-01-01T00:00:00Z",
        regions=tuple(regions),
        features=(),
        claims=tuple(claims),
    )


def _frame_input(image: ImageObservation, result: PerceptionResult) -> AssociationFrameInput:
    prepared = prepare_image(
        SourceImage(
            source_observation_id=str(image.observation_id),
            image=ArtifactReference(
                uri=f"sequence://{image.observation_id}", sha256="0" * 64, media_type="image/rgb8"
            ),
            width=image.width,
            height=image.height,
        ),
        operations=(),
    )
    return AssociationFrameInput(
        observation=image, prepared_image=prepared, perception_result=result, dense_maps={}
    )


def _associate(
    workspace: Path,
    sequence: SequenceArtifactReader,
    trajectory: StateEstimationRunReader,
    geometry: GeometricMapArtifactReader,
    results: dict[PerceptionResultId, PerceptionResult],
    run: PerceptionRunId,
    run_index: int,
) -> tuple[SensorAssociationOutcome, SensorAssociationRunReader]:
    """Associate the regions of one perception run; a frame appears once per association run."""
    # O subconjunto de CI grava as imagens sem `calibration_id`, e a associação recusa uma imagem
    # que não nomeia a sua calibração: o teste a nomeia aqui, sem alterar o fixture versionado.
    images = {
        str(item.observation_id): replace(item, calibration_id=_CAMERA_CALIBRATION)
        for item in sequence.list_observations()
        if isinstance(item, ImageObservation)
    }
    frames = tuple(
        _frame_input(images[str(result.source_observation_id)], result)
        for result in results.values()
        if result.run_id == run
    )
    calibration = sequence.read_calibration()
    assert calibration is not None, "the synthetic sequence carries its calibration"
    request = SensorAssociationRequest(
        sequence_artifact_id=sequence.manifest.artifact_id,
        selection_id="full-sequence",
        geometry=geometry.geometry(),
        trajectory=TrajectoryLookup(trajectory.trajectory()),
        pose_policy=LookupPolicy.exact(),
        calibration=calibration,
        occlusion_policy=OcclusionPolicy(
            cell_size_px=4, neighborhood_radius_cells=0, depth_margin_m=0.1, depth_margin_ratio=0.02
        ),
        tolerances=DiagnosticTolerances(
            max_pose_time_delta_ns=1_000_000,
            max_map_window_offset_ns=None,
            max_reprojection_p95_px=None,
            max_reprojection_invalid_rate=None,
        ),
        frames=frames,
        state_estimation_run_id=trajectory.manifest.run_id,
        code_version="test",
    )
    outcome = SensorAssociationService().run(request)
    # A cadeia sintética tem uma associação por run de percepção, então cada uma ganha o seu
    # diretório; no runtime há um único `<run>/sensor_association/` por execução.
    directory = workspace / f"sensor_association-{run}"
    SensorAssociationRunWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=SensorAssociationRunId(f"association-{run}"),
        run_index=run_index,
    ).finalize(outcome)
    return outcome, SensorAssociationRunReader(directory)


def _fuse(
    workspace: Path,
    sequence: SequenceArtifactReader,
    geometry: GeometricMapArtifactReader,
    associations: tuple[tuple[SensorAssociationOutcome, SensorAssociationRunReader], ...],
    results: dict[PerceptionResultId, PerceptionResult],
    runs: tuple[PerceptionRun, ...],
) -> tuple[tuple[FusionOutcome, ...], tuple[ExcludedObservation, ...], SemanticFusionRunReader]:
    observations = [
        item
        for outcome, _ in associations
        for frame in outcome.frames
        for item in frame.observations
    ]
    timestamps = {
        item.observation_id: item.timestamp
        for item in sequence.list_observations()
        if isinstance(item, ImageObservation)
    }
    grouping = group_by_physical_observation(
        observations, selected_runs=runs, acquisition_timestamps=timestamps
    )
    build = build_fusion_supports(
        observations,
        geometry=geometry.geometry(),
        acquisition_timestamps=timestamps,
        policy=GeometryOverlapSupportPolicy(min_geometry_count=1, min_overlap=0.5),
        code_version="test",
    )
    by_id = {item.spatial_observation_id: item for item in observations}
    policy = BaselineAccumulationPolicy(abstention_labels=frozenset({"unknown"}))
    outcomes = tuple(
        FusionOutcome(
            support=support,
            evidence=accumulate_baseline_evidence(
                support,
                observations=by_id,
                grouping=grouping,
                perception_results=results,
                policy=policy,
                code_version="test",
            ),
        )
        for support in build.supports
    )
    directory = workspace / "semantic_fusion"
    SemanticFusionRunWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=SemanticFusionRunId("fusion-run-0001"),
        run_index=1,
        lineage=FusionRunLineage(
            sequence_artifact_id=str(sequence.manifest.artifact_id),
            geometric_map_id=geometry.manifest.map_id,
            association_run_ids=tuple(str(reader.manifest.run_id) for _, reader in associations),
            perception_run_ids=tuple(run.run_id for run in runs),
            point_representation_run_ids=(),
        ),
        code_version="test",
    ).write(outcomes, excluded=build.excluded)
    return outcomes, build.excluded, SemanticFusionRunReader(directory)


_MAPPING_GEOMETRY_SUMMARY = GeometrySummaryPolicy(
    sparse_point_threshold=3, connectivity_radius_m=0.5
)
_UP_AXIS = AxisDirection.POSITIVE_Z
_SEMANTIC_MAP_ID = SemanticMapId("semantic-map-ci")


def _materialize(
    workspace: Path, fusion: SemanticFusionRunReader, geometry: GeometricMapArtifactReader
) -> SemanticMappingRunReader:
    materialization = materialize_entities(
        fusion.iter_outcomes(),
        fusion_manifest=fusion.manifest,
        geometry=geometry.geometry(),
        semantic_map_id=_SEMANTIC_MAP_ID,
        policy=EntityMaterializationPolicy(geometry=_MAPPING_GEOMETRY_SUMMARY),
        code_version="test",
    )
    directory = workspace / "semantic_mapping"
    SemanticMappingRunWriter(
        output_dir=directory,
        sequence_name=fusion.manifest.sequence_name,
        run_id=SemanticMappingRunId("mapping-run-0001"),
        run_index=1,
        semantic_map_id=_SEMANTIC_MAP_ID,
        lineage=lineage_from_fusion_manifest(fusion.manifest),
        code_version="test",
        code_digest="sha256:" + "cd" * 32,
    ).write(materialization.entities, rejections=materialization.rejections)
    return SemanticMappingRunReader(directory)


def _resolve(workspace: Path, mapping: SemanticMappingRunReader) -> EntityResolutionRunReader:
    entities = list(mapping.iter_entities())
    candidate_sets = retrieve_candidate_sets(
        entities, CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
    )
    builder = MatchEvidenceBuilder(
        ComparisonChannels(
            geometry=GeometryComparisonPolicy(
                min_shared_support_jaccard=0.5,
                min_bounds_iou=0.5,
                min_bounds_containment=0.9,
                min_conflict_gap_m=0.5,
                min_extent_ratio=0.3,
            )
        )
    )
    resolutions = resolve_candidate_pairs(
        entities,
        candidate_sets,
        builder,
        ConservativeResolutionPolicy(
            use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
        ),
        code_version="test",
    )
    run_id = EntityResolutionRunId("resolution-run-0001")
    materialization = materialize_resolved_entities(
        entities,
        [item.decision for item in resolutions],
        resolution_run_id=run_id,
        code_version="test",
    )
    directory = workspace / "entity_resolution"
    EntityResolutionRunWriter(
        output_dir=directory,
        run_id=run_id,
        lineage=lineage_from_mapping_manifest(mapping.manifest),
        code_version="test",
    ).write(candidate_sets=candidate_sets, resolutions=resolutions, materialization=materialization)
    return EntityResolutionRunReader(directory)


def _relate(
    workspace: Path, resolution: EntityResolutionRunReader, geometry: GeometricMapArtifactReader
) -> SpatialRelationsRunReader:
    conventions = FrameConventions(
        map_frame=str(REFERENCE_FRAME), up_axis=_UP_AXIS, forward_axis=AxisDirection.POSITIVE_X
    )
    candidate_policy = CandidatePolicy(
        predicates=(RelationPredicate.NEXT_TO,), proximity_radius_m=2.0, directional_radius_m=2.0
    )
    source = geometry.geometry()
    entities = resolved_entity_geometries(
        resolution.resolved_entities(), source=source, policy=_MAPPING_GEOMETRY_SUMMARY
    )
    # O raio generoso garante ao menos um candidato geometricamente real entre a palete e o poste
    # (~1.36 m de separação): nenhum avaliador geométrico/de contato foi selecionado, então o
    # candidato fica honestamente UNRESOLVED, nunca decidido sem evidência.
    candidates = generate_relation_candidates(
        entities, policy=candidate_policy, conventions=conventions
    )
    decisions = decide_relations(candidates, [])
    directory = workspace / "spatial_relations"
    SpatialRelationsRunWriter(
        output_dir=directory,
        run_id=SpatialRelationsRunId("relations-run-0001"),
        lineage=lineage_from_resolution_manifest(resolution.manifest),
        policies=RelationsRunPolicies(
            frame_conventions=conventions,
            candidate=candidate_policy,
            geometry_summary=_MAPPING_GEOMETRY_SUMMARY,
        ),
        code_version="test",
    ).write(candidates=candidates, evidence=[], decisions=decisions)
    return SpatialRelationsRunReader(directory)


def _observation_window(clock_id: str, *, start_ns: int, end_ns: int) -> ObservationWindow:
    def _timestamp(total_nanoseconds: int) -> SourceTimestamp:
        seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
        return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)

    return ObservationWindow(start=_timestamp(start_ns), end=_timestamp(end_ns))


def _assemble(
    workspace: Path,
    geometry: GeometricMapArtifactReader,
    resolution: EntityResolutionRunReader,
    relations: SpatialRelationsRunReader,
) -> Path:
    geometry_manifest = geometry.manifest
    bounds = geometry.geometry().geometric_map.bounds
    predicates = sorted(
        {relation.predicate for relation in relations.iter_relations()},
        key=lambda predicate: predicate.value,
    )
    metadata = ContextMapMetadata(
        creation=MapCreation(
            assembly_policy=PolicyRef(policy_id=CONTEXT_MAP_ASSEMBLY_POLICY_ID, version="1"),
            code_version="test",
            configuration_fingerprint=None,
        ),
        source_sequences=(
            SourceSequence(
                sequence_artifact_id=str(geometry_manifest.sequence_artifact_id),
                selection_id=geometry_manifest.selection_id,
            ),
        ),
        frame=estimator_local_map_frame(
            frame_id=str(geometry_manifest.map_frame), up_direction=_UP_AXIS.vector
        ),
        bounds=bounds,
        time_bounds=_observation_window(
            geometry_manifest.clock_id,
            start_ns=geometry_manifest.start_time_ns,
            end_ns=geometry_manifest.end_time_ns,
        ),
        capabilities=DeclaredCapabilities(
            content=(MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS),
            relation_predicates=tuple(predicates),
        ),
    )
    sequence_dir = workspace / "ingestion"
    geometry_dir = workspace / "geometric_mapping"
    resolution_dir = workspace / "entity_resolution"
    relations_dir = workspace / "spatial_relations"
    sequence_lineage = (
        UpstreamArtifact(
            artifact_id=str(geometry_manifest.sequence_artifact_id),
            kind=ArtifactKind.SEQUENCE,
            content_identity=artifact_digest(sequence_dir),
            configuration_fingerprint=None,
            code_version=None,
            model_identities=(),
        ),
    )
    result = assemble_context_map_with_metrics(
        context_map_id=ContextMapId("context-map-ci-0001"),
        metadata=metadata,
        geometric_map_location=geometry_dir,
        entity_resolution_location=resolution_dir,
        spatial_relations_location=relations_dir,
        additional_lineage=sequence_lineage,
    )
    location_by_kind = {
        ArtifactKind.GEOMETRIC_MAP: geometry_dir,
        ArtifactKind.ENTITY_RESOLUTION_RUN: resolution_dir,
        ArtifactKind.SPATIAL_RELATIONS_RUN: relations_dir,
    }
    upstream_locations = {
        item.artifact_id: location_by_kind[item.kind]
        for item in result.context_map.lineage
        if item.kind.is_structural
    }
    directory = workspace / "context_map"
    write_context_map_with_metrics(
        result.context_map, output_dir=directory, upstream_locations=upstream_locations
    )
    return directory


@contextmanager
def synthetic_chain(workspace: Path) -> Iterator[SyntheticChain]:
    """Run every stage over the synthetic sequence and keep the geometry reader open."""
    sequence = _ingest(workspace)
    trajectory = _estimate(workspace, sequence)
    with _map(workspace, sequence, trajectory) as geometry:
        results = {
            PerceptionResultId(f"{run}--frame-{index:04d}"): _perception_result(run, index)
            for run, frames in ((RUN_A, (0, 1, 2)), (RUN_B, (0,)))
            for index in frames
        }
        runs = tuple(
            PerceptionRun(
                run_id=run,
                run_index=position,
                sequence_artifact_id=str(SEQUENCE_ARTIFACT_ID),
                selection_id="full-sequence",
                enabled_capabilities=frozenset({"semantic_interpreter"}),
                backend_provenance={"semantic_interpreter": _INTERPRETER},
            )
            for position, run in enumerate((RUN_A, RUN_B))
        )
        associations = tuple(
            _associate(workspace, sequence, trajectory, geometry, results, run, run_index)
            for run_index, run in enumerate((RUN_A, RUN_B), start=1)
        )
        fusion_outcomes, excluded, fusion = _fuse(
            workspace, sequence, geometry, associations, results, runs
        )
        mapping = _materialize(workspace, fusion, geometry)
        resolution = _resolve(workspace, mapping)
        relations = _relate(workspace, resolution, geometry)
        context_map_dir = _assemble(workspace, geometry, resolution, relations)
        yield SyntheticChain(
            sequence=sequence,
            trajectory=trajectory,
            geometry=geometry,
            perception_runs=runs,
            perception_results=results,
            associations=associations,
            fusion_outcomes=fusion_outcomes,
            fusion_excluded=excluded,
            fusion=fusion,
            spatial_observations={
                item.spatial_observation_id: item
                for outcome, _ in associations
                for frame in outcome.frames
                for item in frame.observations
            },
            mapping=mapping,
            resolution=resolution,
            relations=relations,
            context_map_dir=context_map_dir,
        )


def cross_stage_inputs(chain: SyntheticChain) -> CrossStageInputs:
    """The manifests and objects the cross-stage evaluator reads, taken from the readers."""
    return CrossStageInputs(
        sequence=chain.sequence.manifest,
        sequence_calibration_identity=calibration_identity(chain.sequence.read_calibration()),
        trajectory=chain.trajectory.manifest,
        geometry=chain.geometry.manifest,
        geometry_source=chain.geometry.geometry(),
        perception_runs=chain.perception_runs,
        perception_results=chain.perception_results,
        associations=tuple(reader.manifest for _, reader in chain.associations),
        spatial_observations=chain.spatial_observations,
        fusion=chain.fusion.manifest,
        fusion_outcomes=chain.fusion_outcomes,
    )


def canned_perception_results(
    run: PerceptionRunId, sequence_artifact_id: str
) -> list[PerceptionResult]:
    """The canned perception evidence of the first run (three frames) under another run identity."""
    return [
        _perception_result(run, index, claims_of="run-a", sequence_artifact_id=sequence_artifact_id)
        for index in range(3)
    ]
