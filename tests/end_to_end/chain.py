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
    allocate_map_run_index,
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
    allocate_fusion_run_index,
    build_fusion_supports,
    group_by_physical_observation,
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
from contextmap.sensor_association import allocate_run_index as allocate_association_index
from contextmap.sensor_association.service import AssociationFrameInput
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
    index = allocate_map_run_index(workspace_root=workspace, sequence_name=SEQUENCE_NAME)
    GeometricMapArtifactWriter(
        workspace_root=workspace,
        sequence_name=SEQUENCE_NAME,
        run_id=GeometricMapRunId("map-run-0001"),
        run_index=index,
        selection_label="full-sequence",
        profile_label="all-points",
        debug_level=MapDebugLevel.NONE,
    ).finalize(plan=plan, aggregation=None, code_version="test")
    directory = next((workspace / "runs" / "geometric-mapping" / SEQUENCE_NAME).glob("run-0001__*"))
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


def _perception_result(run: PerceptionRunId, frame_index: int) -> PerceptionResult:
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
            _CLAIMS.get((str(run), frame_index, region_id), ())
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
        sequence_artifact_id=str(SEQUENCE_ARTIFACT_ID),
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
    index = allocate_association_index(workspace_root=workspace, sequence_name=SEQUENCE_NAME)
    SensorAssociationRunWriter(
        workspace_root=workspace,
        sequence_name=SEQUENCE_NAME,
        run_id=SensorAssociationRunId(f"association-{run}"),
        run_index=index,
        selection_label=str(run),
        channel_label="geometry-only",
    ).finalize(outcome)
    directory = next(
        (workspace / "runs" / "sensor-association" / SEQUENCE_NAME).glob(f"run-{index:04d}__*")
    )
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
    index = allocate_fusion_run_index(workspace_root=workspace, sequence_name=SEQUENCE_NAME)
    SemanticFusionRunWriter(
        workspace_root=workspace,
        sequence_name=SEQUENCE_NAME,
        run_id=SemanticFusionRunId("fusion-run-0001"),
        run_index=index,
        selection_label="full-sequence",
        policy_label="baseline",
        lineage=FusionRunLineage(
            sequence_artifact_id=str(sequence.manifest.artifact_id),
            geometric_map_id=geometry.manifest.map_id,
            association_run_ids=tuple(str(reader.manifest.run_id) for _, reader in associations),
            perception_run_ids=tuple(run.run_id for run in runs),
            point_representation_run_ids=(),
        ),
        code_version="test",
    ).write(outcomes, excluded=build.excluded)
    directory = next((workspace / "runs" / "semantic-fusion" / SEQUENCE_NAME).glob("run-0001__*"))
    return outcomes, build.excluded, SemanticFusionRunReader(directory)


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
            _associate(workspace, sequence, trajectory, geometry, results, run)
            for run in (RUN_A, RUN_B)
        )
        fusion_outcomes, excluded, fusion = _fuse(
            workspace, sequence, geometry, associations, results, runs
        )
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
