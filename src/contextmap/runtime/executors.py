"""Stage executors: each runs one capability and writes only where the request says.

An executor is thin. It opens its inputs through :meth:`StageRequest.directory_of`, calls the
capability's public service with the scientific policies it was built with, hands the capability's
writer the directory it was given (``StageRequest.output_dir``) and reports the artifact as an
:class:`ArtifactRef` with its location. It computes no path, allocates no identity and decides no
science: every threshold and policy is supplied by whoever builds the executor, and the run id it
records is derived from the execution (:meth:`StageRequest.identity`), so an identical execution
produces the identical artifact.

This module is imported explicitly (``contextmap.runtime.executors``), never by importing
``contextmap.runtime``: it depends on the public root of every capability it executes, and the
runtime package itself must load none of them. It imports public roots only, never a backend.

A stage produces exactly one artifact, and the executors consume exactly one run per input: a
request with several runs for one input is refused, never resolved by picking one. Comparing runs
means running the runtime more than once and reading the artifacts through their references.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ConservativeResolutionPolicy,
    EntityResolutionRunId,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    MatchEvidenceBuilder,
    lineage_from_mapping_manifest,
    materialize_resolved_entities,
    resolve_candidate_pairs,
    retrieve_candidate_sets,
)
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    MotionCorrectionPolicy,
    ScanVoxelPolicy,
    assemble_geometry_inputs_from_artifacts,
)
from contextmap.ingestion import (
    FullSequenceSelection,
    ImageObservation,
    SequenceArtifactReader,
    selection_identity,
)
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.catalog import (
    ASSOCIATION,
    ENTITIES,
    FUSION,
    GEOMETRY,
    RELATIONS,
    RESOLUTION,
    TRAJECTORY,
)
from contextmap.runtime.pipeline import StageRequest
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
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
    AssociationFrameInput,
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
)
from contextmap.spatial_relations import (
    RelationEvidence,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunWriter,
    decide_relations,
    evaluate_contact_candidates,
    evaluate_geometric_candidates,
    generate_relation_candidates,
    lineage_from_resolution_manifest,
    resolved_entity_geometries,
)
from contextmap.state_estimation import (
    GeometryRequirements,
    LookupPolicy,
    StateEstimationRequest,
    StateEstimationRunId,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    StateEstimator,
    TrajectoryId,
    TrajectoryLookup,
    execute_state_estimation,
)
from contextmap.visual_perception import (
    ArtifactReference,
    PerceptionRun,
    PerceptionRunId,
    PerceptionRunReader,
    SourceImage,
    prepare_image,
)

__all__ = [
    "EntityResolutionExecutor",
    "ExecutorError",
    "GeometricMappingExecutor",
    "SemanticFusionExecutor",
    "SemanticMappingExecutor",
    "SensorAssociationExecutor",
    "SpatialRelationsExecutor",
    "StateEstimationExecutor",
    "inventory_digest",
]


class ExecutorError(ValueError):
    """Raised when a request cannot be served: no directory, or not exactly one run per input."""


def inventory_digest(inventory: Sequence[object]) -> str:
    """Return the content hash of an artifact: the digest of its contractual file inventory.

    Args:
        inventory: The ``file_inventory`` of a manifest; each entry has ``path`` and
            ``content_hash``.

    Returns:
        ``sha256:`` of the canonical JSON of ``{path: content_hash}``. Two artifacts with the same
        contractual files have the same digest, whatever their names, ids or timestamps.
    """
    files = {entry.path: entry.content_hash for entry in inventory}  # type: ignore[attr-defined]
    text = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _output(request: StageRequest) -> Path:
    if request.output_dir is None or request.workspace is None:
        raise ExecutorError(
            f"stage {request.stage_id!r} needs the run's output directory: run it with a journal"
        )
    return request.output_dir


def _one(request: StageRequest, name: str) -> Path:
    refs = request.inputs.get(name, ())
    if len(refs) != 1:
        raise ExecutorError(
            f"stage {request.stage_id!r} consumes exactly one run of input {name!r}, got "
            f"{len(refs)}: run the runtime once per run instead of choosing one"
        )
    return request.directory_of(refs[0])


def _reference(
    request: StageRequest, contract: str, artifact_id: str, inventory: Sequence[object]
) -> ArtifactRef:
    output = _output(request)
    assert request.workspace is not None  # garantido por `_output`
    return ArtifactRef(
        stage_id=request.stage_id,
        contract=contract,
        artifact_id=artifact_id,
        content_hash=inventory_digest(inventory),
        location=output.relative_to(request.workspace).as_posix(),
    )


class StateEstimationExecutor:
    """Estimates the trajectory of the ingested sequence with the selected estimator."""

    def __init__(
        self, estimator: StateEstimator, *, downstream: Sequence[GeometryRequirements] = ()
    ) -> None:
        """Bind the executor to a composed estimator and the readiness it reports downstream."""
        self._estimator = estimator
        self._downstream = tuple(downstream)

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Estimate, persist and reference the trajectory of the ``sequence`` input."""
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        identity = request.identity()
        outcome = execute_state_estimation(
            self._estimator,
            StateEstimationRequest(
                trajectory_id=TrajectoryId(f"{identity}--trajectory"),
                sequence_artifact_id=sequence.manifest.artifact_id,
                selection_id=selection_identity(
                    sequence.manifest.artifact_id, FullSequenceSelection()
                ),
                observations=tuple(sequence.list_observations()),
                calibration=sequence.read_calibration(),
            ),
            downstream=self._downstream,
        )
        manifest = StateEstimationRunWriter(
            output_dir=output,
            sequence_name=sequence.manifest.sequence_name,
            run_id=StateEstimationRunId(identity),
            run_index=request.run_number(),
        ).finalize(outcome)
        return _reference(request, TRAJECTORY, str(manifest.run_id), manifest.file_inventory)


class GeometricMappingExecutor:
    """Accumulates the LiDAR scans into a persistent map in the trajectory's frame."""

    def __init__(
        self,
        *,
        pose_lookup: LookupPolicy,
        motion_correction: MotionCorrectionPolicy,
        aggregation: ScanVoxelPolicy | None = None,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the policies that decide which scans become geometry."""
        self._pose_lookup = pose_lookup
        self._motion_correction = motion_correction
        self._aggregation = aggregation
        self._code_version = code_version

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Map the ``sequence`` input through the ``trajectory`` input and reference the map."""
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        plan = assemble_geometry_inputs_from_artifacts(
            sequence=sequence,
            selection=FullSequenceSelection(),
            run=StateEstimationRunReader(_one(request, "trajectory")),
            pose_lookup=self._pose_lookup,
            motion_correction_policy=self._motion_correction,
        )
        manifest = GeometricMapArtifactWriter(
            output_dir=output,
            sequence_name=sequence.manifest.sequence_name,
            run_id=GeometricMapRunId(request.identity()),
            run_index=request.run_number(),
        ).finalize(plan=plan, aggregation=self._aggregation, code_version=self._code_version)
        return _reference(request, GEOMETRY, str(manifest.run_id), manifest.file_inventory)


class SensorAssociationExecutor:
    """Associates the regions of one perception run with the persistent map."""

    def __init__(
        self,
        *,
        occlusion: OcclusionPolicy,
        tolerances: DiagnosticTolerances,
        pose_policy: LookupPolicy,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the visibility policy, the diagnostic tolerances and the pose rule.

        The dense-feature channels are not used: this executor produces the geometry-only
        association, the one evidence channel a run can derive without a feature payload.
        """
        self._occlusion = occlusion
        self._tolerances = tolerances
        self._pose_policy = pose_policy
        self._code_version = code_version

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Associate every perceived frame of the ``perception`` input and reference the run."""
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        trajectory = StateEstimationRunReader(_one(request, "trajectory"))
        calibration = sequence.read_calibration()
        if calibration is None:
            raise ExecutorError("the sequence artifact carries no calibration")
        images = {
            str(item.observation_id): item
            for item in sequence.list_observations()
            if isinstance(item, ImageObservation)
        }
        frames = tuple(
            self._frame(images[str(result.source_observation_id)], result)
            for result in PerceptionRunReader(_one(request, "perception")).list_results()
            if str(result.source_observation_id) in images
        )
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            outcome = SensorAssociationService().run(
                SensorAssociationRequest(
                    sequence_artifact_id=sequence.manifest.artifact_id,
                    selection_id=selection_identity(
                        sequence.manifest.artifact_id, FullSequenceSelection()
                    ),
                    geometry=geometry.geometry(),
                    trajectory=TrajectoryLookup(trajectory.trajectory()),
                    pose_policy=self._pose_policy,
                    calibration=calibration,
                    occlusion_policy=self._occlusion,
                    tolerances=self._tolerances,
                    frames=frames,
                    state_estimation_run_id=trajectory.manifest.run_id,
                    code_version=self._code_version,
                )
            )
        manifest = SensorAssociationRunWriter(
            output_dir=output,
            sequence_name=sequence.manifest.sequence_name,
            run_id=SensorAssociationRunId(request.identity()),
            run_index=request.run_number(),
        ).finalize(outcome)
        return _reference(request, ASSOCIATION, str(manifest.run_id), manifest.file_inventory)

    @staticmethod
    def _frame(image: ImageObservation, result: object) -> AssociationFrameInput:
        digest = hashlib.sha256(image.data).hexdigest()
        prepared = prepare_image(
            SourceImage(
                source_observation_id=str(image.observation_id),
                image=ArtifactReference(
                    uri=f"sequence://{image.observation_id}",
                    sha256=digest,
                    media_type=f"image/{image.encoding.value}",
                ),
                width=image.width,
                height=image.height,
            ),
            operations=(),
        )
        return AssociationFrameInput(
            observation=image,
            prepared_image=prepared,
            perception_result=result,  # type: ignore[arg-type]
            dense_maps={},
        )


class SemanticFusionExecutor:
    """Fuses the association evidence over the persistent geometry, without creating identity."""

    def __init__(
        self,
        *,
        support_policy: GeometryOverlapSupportPolicy,
        accumulation_policy: BaselineAccumulationPolicy,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the support policy and the baseline accumulation policy."""
        self._support_policy = support_policy
        self._accumulation_policy = accumulation_policy
        self._code_version = code_version

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Fuse the ``association`` run over the ``geometry`` map and reference the fusion run."""
        output = _output(request)
        if request.inputs.get("representation"):
            raise ExecutorError(
                "the point representation channel is not supported by this executor: "
                "run the stage without a representation input"
            )
        association = SensorAssociationRunReader(_one(request, "association"))
        perception = PerceptionRunReader(_one(request, "perception"))
        results = {result.result_id: result for result in perception.list_results()}
        observations = list(association.observations())
        described = perception.manifest
        run = PerceptionRun(
            run_id=PerceptionRunId(str(described.run_id)),
            run_index=described.run_index,
            sequence_artifact_id=str(described.sequence_artifact_id),
            selection_id=described.selection_id,
            enabled_capabilities=frozenset(described.enabled_capabilities),
            backend_provenance={},
        )
        timestamps = {
            item.observation_id: item.timestamp
            for item in SequenceArtifactReader(_one(request, "sequence")).list_observations()
            if isinstance(item, ImageObservation)
        }
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            source = geometry.geometry()
            grouping = group_by_physical_observation(
                observations, selected_runs=(run,), acquisition_timestamps=timestamps
            )
            build = build_fusion_supports(
                observations,
                geometry=source,
                acquisition_timestamps=timestamps,
                policy=self._support_policy,
                code_version=self._code_version,
            )
            by_id = {item.spatial_observation_id: item for item in observations}
            outcomes = tuple(
                FusionOutcome(
                    support=support,
                    evidence=accumulate_baseline_evidence(
                        support,
                        observations=by_id,
                        grouping=grouping,
                        perception_results=results,
                        policy=self._accumulation_policy,
                        code_version=self._code_version,
                    ),
                )
                for support in build.supports
            )
            manifest = SemanticFusionRunWriter(
                output_dir=output,
                sequence_name=association.manifest.sequence_name,
                run_id=SemanticFusionRunId(request.identity()),
                run_index=request.run_number(),
                lineage=FusionRunLineage(
                    sequence_artifact_id=str(association.manifest.sequence_artifact_id),
                    geometric_map_id=geometry.manifest.map_id,
                    association_run_ids=(str(association.manifest.run_id),),
                    perception_run_ids=(run.run_id,),
                    point_representation_run_ids=(),
                ),
                code_version=self._code_version or "",
            ).write(outcomes, excluded=build.excluded)
        return _reference(request, FUSION, str(manifest.run_id), manifest.file_inventory)


class SemanticMappingExecutor:
    """Materializes the fused evidence as persistent entities, without merging supports."""

    def __init__(
        self,
        *,
        policy: EntityMaterializationPolicy,
        semantic_map_id: SemanticMapId,
        code_digest: str,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the materialization policy and the semantic map it fills.

        Args:
            policy: The materialization policy.
            semantic_map_id: The semantic map the run fills.
            code_digest: Digest of the code producing the run; ``SemanticMappingRunWriter``
                refuses an empty one, so unlike ``code_version`` it has no silent default.
            code_version: Code revision that produced the run.
        """
        self._policy = policy
        self._semantic_map_id = semantic_map_id
        self._code_digest = code_digest
        self._code_version = code_version

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Materialize the ``fusion`` run over its geometry and reference the entity run."""
        output = _output(request)
        fusion = SemanticFusionRunReader(_one(request, "fusion"))
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            materialization = materialize_entities(
                fusion.iter_outcomes(),
                fusion_manifest=fusion.manifest,
                geometry=geometry.geometry(),
                semantic_map_id=self._semantic_map_id,
                policy=self._policy,
                code_version=self._code_version,
            )
        manifest = SemanticMappingRunWriter(
            output_dir=output,
            sequence_name=fusion.manifest.sequence_name,
            run_id=SemanticMappingRunId(request.identity()),
            run_index=request.run_number(),
            semantic_map_id=self._semantic_map_id,
            lineage=lineage_from_fusion_manifest(fusion.manifest),
            code_version=self._code_version or "",
            code_digest=self._code_digest,
        ).write(materialization.entities, rejections=materialization.rejections)
        return _reference(request, ENTITIES, str(manifest.run_id), manifest.file_inventory)


class EntityResolutionExecutor:
    """Decides identity across the materialized entities; without proof it stays unresolved."""

    def __init__(
        self,
        *,
        retrieval: CandidateRetrievalPolicy,
        builder: MatchEvidenceBuilder,
        resolution: ConservativeResolutionPolicy,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the retrieval, the evidence channels and the decision policy."""
        self._retrieval = retrieval
        self._builder = builder
        self._resolution = resolution
        self._code_version = code_version

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Resolve the ``entities`` run without mutating it and reference the resolution run."""
        output = _output(request)
        mapping = SemanticMappingRunReader(_one(request, "entities"))
        entities = list(mapping.iter_entities())
        candidate_sets = retrieve_candidate_sets(entities, self._retrieval)
        resolutions = resolve_candidate_pairs(
            entities,
            candidate_sets,
            self._builder,
            self._resolution,
            code_version=self._code_version,
        )
        run_id = EntityResolutionRunId(request.identity())
        materialization = materialize_resolved_entities(
            entities,
            [item.decision for item in resolutions],
            resolution_run_id=run_id,
            code_version=self._code_version,
        )
        manifest = EntityResolutionRunWriter(
            output_dir=output,
            run_id=run_id,
            lineage=lineage_from_mapping_manifest(mapping.manifest),
            code_version=self._code_version or "",
        ).write(
            candidate_sets=candidate_sets, resolutions=resolutions, materialization=materialization
        )
        return _reference(request, RESOLUTION, str(manifest.run_id), manifest.file_inventory)


class SpatialRelationsExecutor:
    """Derives the relations between resolved entities and keeps the evidence behind each one."""

    def __init__(
        self,
        *,
        policies: RelationsRunPolicies,
        geometry_summary: GeometrySummaryPolicy,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the frame conventions, the predicate policies and the summary."""
        self._policies = policies
        self._geometry_summary = geometry_summary
        self._code_version = code_version

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Relate the ``entities`` resolution over the ``geometry`` map and reference the run."""
        output = _output(request)
        resolution = EntityResolutionRunReader(_one(request, "entities"))
        conventions = self._policies.frame_conventions
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            source = geometry.geometry()
            entities = resolved_entity_geometries(
                resolution.resolved_entities(), source=source, policy=self._geometry_summary
            )
            candidates = generate_relation_candidates(
                entities, policy=self._policies.candidate, conventions=conventions
            )
            evidence: list[RelationEvidence] = []
            if self._policies.geometric is not None:
                evidence.extend(
                    evaluate_geometric_candidates(
                        candidates,
                        entities=entities,
                        policy=self._policies.geometric,
                        conventions=conventions,
                    )
                )
            if self._policies.contact is not None:
                evidence.extend(
                    evaluate_contact_candidates(
                        candidates,
                        entities=entities,
                        geometry_source=source,
                        policy=self._policies.contact,
                        conventions=conventions,
                    )
                )
            decisions = decide_relations(candidates, evidence)
        manifest = SpatialRelationsRunWriter(
            output_dir=output,
            run_id=SpatialRelationsRunId(request.identity()),
            lineage=lineage_from_resolution_manifest(resolution.manifest),
            policies=self._policies,
            code_version=self._code_version or "",
        ).write(candidates=candidates, evidence=evidence, decisions=decisions)
        return _reference(request, RELATIONS, str(manifest.run_id), manifest.file_inventory)
