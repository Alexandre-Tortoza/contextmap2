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
import math
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    ClockPlausibilityError,
    FullSequenceSelection,
    ImageEncoding,
    ImageObservation,
    SequenceArtifactReader,
    SourceObservation,
    SourceObservationId,
    selection_identity,
    validate_cross_source_clock_plausibility,
)
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.catalog import (
    ASSOCIATION,
    CONTEXT_MAP,
    ENTITIES,
    FUSION,
    GEOMETRY,
    PERCEPTION,
    RELATIONS,
    RESOLUTION,
    TRAJECTORY,
)
from contextmap.runtime.composition import FeatureBuildScope, FeatureFactory
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
    SemanticMapId,
    SemanticMappingRunId,
    SemanticMappingRunReader,
    SemanticMappingRunWriter,
    lineage_from_fusion_manifest,
    materialize_entities,
)
from contextmap.sensor_association import (
    AssociationFrameInput,
    CandidateGeometryPolicy,
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
)
from contextmap.shared import SourceTimestamp
from contextmap.spatial_relations import (
    AxisDirection,
    RelationEvidence,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunReader,
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
    CANONICAL_PRESET_V1,
    ArtifactReference,
    AuditedRegionDiscovery,
    BackendProvenance,
    PerceptionRun,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    Region2D,
    RegionId,
    SceneContext,
    SemanticClaim,
    SemanticInterpretationExecution,
    SemanticInterpretationFailedError,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    SourceImage,
    StageOutcome,
    StageStatus,
    VisualViewKind,
    assemble_perception_result,
    box_mask_shape,
    execute_stage_graph,
    perception_result_id_for,
    prepare_image,
    resolve_pipeline,
)

__all__ = [
    "EntityResolutionExecutor",
    "ExecutorError",
    "GeometricMappingExecutor",
    "SemanticFusionExecutor",
    "SensorAssociationExecutor",
    "SpatialRelationsExecutor",
    "StateEstimationExecutor",
    "VisualPerceptionExecutor",
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


def _optional_one(request: StageRequest, name: str) -> Path | None:
    refs = request.inputs.get(name, ())
    if not refs:
        return None
    if len(refs) != 1:
        raise ExecutorError(
            f"stage {request.stage_id!r} consumes at most one run of input {name!r}, got "
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


_GROUND_TRUTH_POSE_ROLE = "ground_truth"
"""issue #555: an auxiliary pose sequence tagged with this role never drives the trajectory
without :attr:`StateEstimationExecutor._allow_ground_truth_trajectory`'s explicit opt-in."""


class StateEstimationExecutor:
    """Estimates the trajectory of the ingested sequence with the selected estimator."""

    def __init__(
        self,
        estimator: StateEstimator,
        *,
        downstream: Sequence[GeometryRequirements] = (),
        allow_ground_truth_trajectory: bool = False,
    ) -> None:
        """Bind the executor to a composed estimator and the readiness it reports downstream.

        Args:
            estimator: The selected state estimation backend.
            downstream: Requirements later capabilities report as readiness, never blocking.
            allow_ground_truth_trajectory: Explicit opt-in (issue #555) letting an optional
                auxiliary pose sequence tagged ``pose_role="ground_truth"`` become the
                operational trajectory. ``False`` (default) drops a ground-truth-tagged
                auxiliary pose from the merge entirely: the run behaves exactly as if no
                auxiliary pose sequence were configured.
        """
        self._estimator = estimator
        self._downstream = tuple(downstream)
        self._allow_ground_truth_trajectory = allow_ground_truth_trajectory

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Estimate, persist and reference the trajectory of the ``sequence`` input.

        When the optional ``pose_sequence`` input is present (issue #555's auxiliary pose
        bridge), its observations are merged in after its clock relationship with ``sequence``
        passes an explicit plausibility check (never accepted on a matching
        ``timestamp_clock_id`` string alone) and its declared ``pose_role`` is checked. When the
        merge actually contributes observations, ``StateEstimationRequest`` also names the
        auxiliary artifact and selection explicitly (``auxiliary_sequence_artifact_id``/
        ``auxiliary_selection_id``), so the trajectory's own persisted provenance stays
        self-portable even when a reader only has the ``StateEstimationRunArtifact``.
        """
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        observations: tuple[SourceObservation, ...] = tuple(sequence.list_observations())

        auxiliary_sequence_artifact_id = None
        auxiliary_selection_id = None
        pose_path = _optional_one(request, "pose_sequence")
        if pose_path is not None:
            pose_sequence = SequenceArtifactReader(pose_path)
            auxiliary = tuple(pose_sequence.list_observations())
            merged = self._select_auxiliary_pose(observations, auxiliary)
            observations = observations + merged
            if merged:
                auxiliary_sequence_artifact_id = pose_sequence.manifest.artifact_id
                auxiliary_selection_id = selection_identity(
                    pose_sequence.manifest.artifact_id, FullSequenceSelection()
                )

        identity = request.identity()
        outcome = execute_state_estimation(
            self._estimator,
            StateEstimationRequest(
                trajectory_id=TrajectoryId(f"{identity}--trajectory"),
                sequence_artifact_id=sequence.manifest.artifact_id,
                selection_id=selection_identity(
                    sequence.manifest.artifact_id, FullSequenceSelection()
                ),
                observations=observations,
                calibration=sequence.read_calibration(),
                auxiliary_sequence_artifact_id=auxiliary_sequence_artifact_id,
                auxiliary_selection_id=auxiliary_selection_id,
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

    def _select_auxiliary_pose(
        self, primary: Sequence[SourceObservation], auxiliary: Sequence[SourceObservation]
    ) -> tuple[SourceObservation, ...]:
        """Validate an auxiliary pose sequence and decide whether it joins the merge.

        Args:
            primary: The main sequence's observations, before any merge.
            auxiliary: The auxiliary pose sequence's observations.

        Returns:
            ``auxiliary`` unchanged, or ``()`` when its declared ``pose_role`` is
            ``"ground_truth"`` and :attr:`_allow_ground_truth_trajectory` is not set.

        Raises:
            ExecutorError: If ``auxiliary`` does not declare one single, recognized
                ``pose_role``, or if the two sequences' clock relationship fails the
                plausibility check (issue #555 requires this before ever combining two
                artifacts' observations). The ground-truth safeguard is checked *before* the
                clock check: a ground-truth-tagged auxiliary sequence that will be dropped
                anyway must not block the run just because its clock is implausible --
                ``operational_only`` must behave exactly as if the auxiliary sequence were
                never configured, including when it is malformed in a way that is irrelevant to
                a dropped sequence.
        """
        pose_role = self._auxiliary_pose_role(auxiliary)
        if pose_role == _GROUND_TRUTH_POSE_ROLE and not self._allow_ground_truth_trajectory:
            return ()
        try:
            validate_cross_source_clock_plausibility(primary, auxiliary)
        except ClockPlausibilityError as error:
            raise ExecutorError(f"cannot merge auxiliary pose sequence: {error}") from error
        return tuple(auxiliary)

    @staticmethod
    def _auxiliary_pose_role(auxiliary: Sequence[SourceObservation]) -> str:
        """Return the auxiliary sequence's single, consistent ``pose_role``.

        Raises:
            ExecutorError: If ``auxiliary`` is empty, declares no ``pose_role`` on some
                observation, or declares more than one distinct value -- inconsistent or
                missing pose role is never silently resolved (issue #555).
        """
        roles = {observation.provenance.raw_metadata.get("pose_role") for observation in auxiliary}
        if len(roles) != 1 or None in roles:
            raise ExecutorError(
                "auxiliary pose sequence observations must declare a single, consistent "
                f"pose_role in their provenance; got {sorted(str(role) for role in roles)!r}"
            )
        (role,) = roles
        assert isinstance(role, str)
        return role


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
        candidates: CandidateGeometryPolicy,
        occlusion: OcclusionPolicy,
        tolerances: DiagnosticTolerances,
        pose_policy: LookupPolicy,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the candidate, visibility, tolerance and pose rules.

        The dense-feature channels are not used: this executor produces the geometry-only
        association, the one evidence channel a run can derive without a feature payload.
        """
        self._candidates = candidates
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
        # Só as imagens são decodificadas: list_observations() decodificaria também os 364 MB de
        # pointcloud e os 20781 registros de IMU do corridor-02, que esta associação nunca usa
        # (#511). O índice guarda apenas o offset de cada imagem: o payload é lido quando o
        # frame daquela observação é construído e solto logo depois, porque mantê-los todos
        # vivos custava 2,19 GB no corridor-02 e faria a entrada escalar com o número de
        # frames, exatamente o que o #563 removeu do lado da saída.
        offsets = {
            str(entry.observation_id): entry.offset
            for entry in sequence.iter_index()
            if entry.modality == "image"
        }
        perception = PerceptionRunReader(_one(request, "perception"))

        def frames() -> Iterator[AssociationFrameInput]:
            for result in perception.iter_results():
                offset = offsets.get(str(result.source_observation_id))
                if offset is None:
                    continue
                image = sequence.observation_at(offset)
                if isinstance(image, ImageObservation):
                    yield self._frame(image, result)

        writer = SensorAssociationRunWriter(
            output_dir=output,
            sequence_name=sequence.manifest.sequence_name,
            run_id=SensorAssociationRunId(request.identity()),
            run_index=request.run_number(),
        )
        # Cada frame é construído sob demanda, persistido e liberado dentro da transação: nem
        # a entrada nem o resultado do run ficam inteiros em memória (#563).
        with (
            GeometricMapArtifactReader(_one(request, "geometry")) as geometry,
            writer.transaction() as run,
        ):
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
                    candidate_policy=self._candidates,
                    occlusion_policy=self._occlusion,
                    tolerances=self._tolerances,
                    frames=frames(),
                    state_estimation_run_id=trajectory.manifest.run_id,
                    code_version=self._code_version,
                    # A reopened perception run never inlines mask pixels (#378); this
                    # is how association resolves a region's mask_reference on demand.
                    mask_loader=perception.mask_store(),
                ),
                sink=run,
            )
            manifest = run.finalize(outcome)
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
        results = {result.result_id: result for result in perception.iter_results()}
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
        # Pelo índice, não por list_observations(): esta fusão só precisa do timestamp de cada
        # imagem, e decodificar a sequência inteira para isso custava os 2,6 GB de payload do
        # corridor-02 (#514). O índice já carrega identidade, timestamp e modalidade.
        timestamps = {
            entry.observation_id: entry.timestamp
            for entry in SequenceArtifactReader(_one(request, "sequence")).iter_index()
            if entry.modality == "image"
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


class _DeferredFeaturePayloadSink:
    """Forwards feature payloads to a writer that does not exist yet.

    ``PerceptionRunWriter`` needs ``configuration_digest``
    (:meth:`~contextmap.visual_perception.pipeline.ResolvedPipeline.configuration_digest`)
    at construction time, but that digest can only be computed *after*
    :func:`~contextmap.visual_perception.resolve_pipeline` has built every
    backend — including the feature extractors, which need a payload sink
    already at their own construction time. This breaks that ordering
    cycle: backends are given this forwarding sink immediately, and it is
    bound to the real writer once one exists, before any observation is
    processed and therefore before any payload is actually queued.
    """

    def __init__(self) -> None:
        """Create an unbound sink."""
        self._writer: PerceptionRunWriter | None = None

    def bind(self, writer: PerceptionRunWriter) -> None:
        """Bind every subsequent call to the real writer."""
        self._writer = writer

    def add_feature_payload(
        self, feature: object, source_observation_id: object, array: object
    ) -> None:
        """Forward one payload to the bound writer.

        Raises:
            ExecutorError: If called before :meth:`bind`.
        """
        if self._writer is None:
            raise ExecutorError("feature payload queued before the run writer was bound")
        self._writer.add_feature_payload(feature, source_observation_id, array)  # type: ignore[arg-type]


class _AuditPublishingRegionDiscovery:
    """Serves the plain ``RegionDiscovery`` port to the stage graph and publishes each audit.

    ``CANONICAL_PRESET_V1``'s ``region_discovery`` stage calls ``discover()`` and hands its regions
    to the stages that depend on them. This asks the composed backend for
    ``discover_audited()`` instead, gives the frame's
    :class:`~contextmap.visual_perception.RegionDiscoveryAudit` to the writer and returns exactly
    the regions ``discover()`` would have. What the audit contains is decided in
    :mod:`contextmap.visual_perception`; this only routes it, so rejected candidates and merge
    decisions reach the artifact instead of dying with the stage output.
    """

    def __init__(self, discovery: AuditedRegionDiscovery) -> None:
        """Wrap the composed backend; nothing is published until :meth:`bind`."""
        self._discovery = discovery
        self._writer: PerceptionRunWriter | None = None

    def bind(self, writer: PerceptionRunWriter) -> None:
        """Bind every subsequent audit to the real writer.

        Same ordering cycle as :class:`_DeferredFeaturePayloadSink`: this has to exist before
        ``resolve_pipeline()`` builds the stage graph, but the writer needs that graph's
        ``configuration_digest``. Binding happens before any image is processed.
        """
        self._writer = writer

    def backend_provenance(self) -> BackendProvenance:
        """Pass the wrapped backend's provenance through: the configuration digest is unchanged."""
        return self._discovery.backend_provenance()

    def discover(self, image: PreparedImage) -> tuple[Region2D, ...]:
        """Discover one frame's regions and hand its audit to the writer before returning them.

        Raises:
            ExecutorError: If called before :meth:`bind`.
        """
        if self._writer is None:
            raise ExecutorError("region discovery used before bind(): no writer for its audit")
        audited = self._discovery.discover_audited(image)
        self._writer.add_region_discovery_audit(audited.audit)
        return audited.regions


class _LegacySemanticInterpreterBridge:
    """Adapts a real ``SemanticInterpreter`` (``interpret()``) to the pipeline's legacy shape.

    ``CANONICAL_PRESET_V1``'s ``scene_interpretation``/``region_interpretation`` stages still
    dispatch through the pre-request-contract shape (``interpret_scene``/``interpret_regions``),
    but no real backend (Qwen, Gemini, Florence-2) implements it any more -- all three finished
    migrating to :class:`~contextmap.visual_perception.SemanticInterpreter`'s ``interpret()``
    port. This bridges the one remaining caller of the legacy shape to the real port: one
    single-view request per call (the whole frame for a scene, one tight crop per region), the
    minimum evidence the request contract requires. It does not implement the multi-view/prompt
    policy work multi-view semantic requests still need (#524, #529, #547, #549) -- that is
    real, separately-tracked capability work, not a runtime concern.

    Each ``interpret()`` call answers with a real :class:`~contextmap.visual_perception.
    SemanticInterpretationExecution` -- the rendered prompt, the raw response, diagnostics and
    the effective configuration, not just the parsed claims/scene context the legacy shape
    returns. A request that fails schema conformance raises before returning one, so every
    execution this bridge collects (:meth:`evidence`) already succeeded; ``interpret_scene``/
    ``interpret_regions`` still return the reduced value the stage graph needs, but the caller
    (:class:`VisualPerceptionExecutor`) also registers every collected execution and its view
    payload with the writer, so this evidence is never silently dropped.
    """

    def __init__(
        self, *, interpreter: SemanticInterpreter, run_id: PerceptionRunId, view_root: Path
    ) -> None:
        """Bind the bridge to the real interpreter, this run's identity, and its view directory."""
        self._interpreter = interpreter
        self._run_id = run_id
        self._view_root = view_root
        self._views_dir = view_root / "outputs" / "semantic-views"
        self._views_dir.mkdir(parents=True, exist_ok=True)
        provenance = interpreter.backend_provenance()
        self._configuration_fingerprint = provenance.configuration_fingerprint or provenance.model
        self._writer: PerceptionRunWriter | None = None

    def bind(self, writer: PerceptionRunWriter) -> None:
        """Bind every subsequent execution to the real writer.

        Same ordering cycle as :class:`_DeferredFeaturePayloadSink`: the bridge has to exist
        before ``resolve_pipeline()`` can build the stage graph, but the writer needs that
        graph's ``configuration_digest``. Binding happens before any image is processed, so
        every execution is forwarded, never buffered.
        """
        self._writer = writer

    def backend_provenance(self) -> BackendProvenance:
        """Pass through the wrapped interpreter's provenance unchanged."""
        return self._interpreter.backend_provenance()

    def _publish(
        self, execution: SemanticInterpretationExecution, view: SemanticVisualView
    ) -> None:
        """Hand one finished execution and its view payload to the writer, retaining neither.

        The payload is read back from the view file only here and dropped as soon as the
        writer has it. Holding ``(execution, view, payload)`` until the image loop ended cost
        O(semantic requests x image size) resident and undid the writer's own streaming.
        """
        if self._writer is None:
            raise ExecutorError("semantic bridge used before bind(): no writer to publish to")
        stage_id = (
            "scene_interpretation"
            if execution.request.mode is SemanticInterpretationMode.SCENE
            else "region_interpretation"
        )
        self._writer.add_stage_outcomes(
            (StageOutcome(stage_id=stage_id, status=StageStatus.SUCCEEDED, output=execution),)
        )
        payload = (self._view_root / view.payload_reference).read_bytes()
        self._writer.add_semantic_view_payload(view, payload)

    def _write_view(self, name: str, pil_image: object) -> tuple[str, str]:
        target = self._views_dir / name
        pil_image.save(target)  # type: ignore[attr-defined]
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return f"outputs/semantic-views/{name}", digest

    def _interpret_preserving_failures(
        self, request: SemanticInterpretationRequest
    ) -> SemanticInterpretationExecution:
        """Run one interpretation, persisting the response even when the parser rejects it.

        The stage still fails — ``execute_stage_graph()`` records the error exactly as before —
        but the model's real answer is no longer reduced to that error string: it reaches the
        artifact's own failures stream first, so ``attempted`` can be reconciled against
        ``parsed`` and ``parse_failed`` instead of being inferred.
        """
        try:
            return self._interpreter.interpret(request)
        except SemanticInterpretationFailedError as failed:
            if self._writer is None:
                raise ExecutorError(
                    "semantic bridge used before bind(): no writer to record the failure in"
                ) from failed
            # As views entram antes da failure: elas sao a evidencia visual exata que produziu
            # a resposta rejeitada, e sem isto ficariam so no scratch, que o executor apaga --
            # o artifact citaria um payload_reference irrecuperavel.
            for view in failed.failure.request.visual_views:
                self._writer.add_semantic_view_payload(
                    view, (self._view_root / view.payload_reference).read_bytes()
                )
            self._writer.add_failed_semantic_interpretation(failed.failure)
            raise

    def interpret_scene(self, image: PreparedImage) -> SceneContext | None:
        """Build one single-view SCENE request from the whole frame and delegate to interpret()."""
        import importlib

        image_module = importlib.import_module("PIL.Image")
        pil_image = image_module.open(self._view_root / image.payload_reference).convert("RGB")
        reference, digest = self._write_view(f"{image.source_observation_id}__scene.png", pil_image)
        view = SemanticVisualView(
            view_id=f"v-{image.source_observation_id}-scene",
            kind=VisualViewKind.FULL_FRAME,
            payload_reference=reference,
            source_observation_id=image.source_observation_id,
            sha256=digest,
        )
        request = SemanticInterpretationRequest(
            request_id=SemanticRequestId(f"scene-{image.source_observation_id}"),
            source_observation_id=image.source_observation_id,
            perception_result_id=perception_result_id_for(
                run_id=self._run_id, source_observation_id=image.source_observation_id
            ),
            mode=SemanticInterpretationMode.SCENE,
            visual_views=(view,),
            prompt_template_id="scene/v1",
            requested_output_schema="semantic-response/1",
            configuration_fingerprint=self._configuration_fingerprint,
        )
        execution = self._interpret_preserving_failures(request)
        self._publish(execution, view)
        return execution.parsed.scene_context

    def interpret_regions(
        self, image: PreparedImage, regions: Sequence[Region2D]
    ) -> Sequence[SemanticClaim]:
        """Build one single-view REGION request per region (tight crop) and delegate."""
        import importlib

        image_module = importlib.import_module("PIL.Image")
        frame = image_module.open(self._view_root / image.payload_reference).convert("RGB")
        claims: list[SemanticClaim] = []
        for region in regions:
            box = region.bounding_box
            crop = frame.crop(
                (
                    int(box.x),
                    int(box.y),
                    int(box.x + box.width),
                    int(box.y + box.height),
                )
            )
            reference, digest = self._write_view(
                f"{image.source_observation_id}__{region.region_id}__tight_crop.png", crop
            )
            view = SemanticVisualView(
                view_id=f"v-{image.source_observation_id}-{region.region_id}",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference=reference,
                source_observation_id=image.source_observation_id,
                sha256=digest,
                region_id=RegionId(str(region.region_id)),
            )
            request = SemanticInterpretationRequest(
                request_id=SemanticRequestId(
                    f"region-{image.source_observation_id}-{region.region_id}"
                ),
                source_observation_id=image.source_observation_id,
                perception_result_id=perception_result_id_for(
                    run_id=self._run_id, source_observation_id=image.source_observation_id
                ),
                mode=SemanticInterpretationMode.REGION,
                visual_views=(view,),
                region_id=RegionId(str(region.region_id)),
                prompt_template_id="region/v1",
                requested_output_schema="semantic-response/1",
                configuration_fingerprint=self._configuration_fingerprint,
            )
            execution = self._interpret_preserving_failures(request)
            self._publish(execution, view)
            claims.extend(execution.parsed.claims)
        return tuple(claims)


def _materialize_prepared_image(image: ImageObservation, root: Path) -> PreparedImage:
    """Decode one raw observation into an RGB/L PNG file a real backend can open.

    Ingestion's :class:`~contextmap.ingestion.ImageObservation` carries a raw,
    uncompressed pixel buffer; every real Visual Perception backend (SAM2,
    SAM3, DINOv2/v3, CLIP, AlphaCLIP) decodes ``PreparedImage.payload_reference``
    as an actual image file through Pillow. This is the one format
    conversion this executor performs on its own: no resize, crop or
    rectification decision is made — exactly the identity transform
    :func:`~contextmap.visual_perception.prepare_image` already models with
    ``operations=()``, the same shape ``SensorAssociationExecutor._frame``
    already uses for its own (non-decodable) placeholder reference.

    Args:
        image: The raw observation to materialize.
        root: Scratch directory prepared images are written under; never
            part of the published ``PerceptionRunArtifact``.

    Returns:
        A ``PreparedImage`` whose reference resolves, under ``root``, to a
        real, hash-verifiable PNG file.

    Raises:
        ExecutorError: If the observation's encoding is not supported.
    """
    import importlib

    import numpy as np

    image_module = importlib.import_module("PIL.Image")

    if image.encoding is ImageEncoding.RGB8:
        rgb = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        pil_image = image_module.fromarray(rgb, mode="RGB")
    elif image.encoding is ImageEncoding.BGR8:
        bgr = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        pil_image = image_module.fromarray(bgr[:, :, ::-1], mode="RGB")
    elif image.encoding is ImageEncoding.MONO8:
        mono8 = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width)
        pil_image = image_module.fromarray(mono8, mode="L")
    elif image.encoding is ImageEncoding.MONO16:
        mono16 = np.frombuffer(image.data, dtype=np.uint16).reshape(image.height, image.width)
        pil_image = image_module.fromarray(mono16, mode="I;16")
    else:
        raise ExecutorError(
            f"unsupported image encoding for prepared-image materialization: {image.encoding!r}"
        )

    filename = f"{image.observation_id}.png"
    output_path = root / filename
    pil_image.save(output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return prepare_image(
        SourceImage(
            source_observation_id=str(image.observation_id),
            image=ArtifactReference(uri=filename, sha256=digest, media_type="image/png"),
            width=image.width,
            height=image.height,
        ),
        operations=(),
    )


class _RegionInlineMaskSource:
    """Crops each region's own already-decoded inline mask to its box-local raster envelope.

    A mask-conditioned region-features backend (AlphaCLIP) is conditioned on the region's
    mask, not only its crop, and asks for it box-local (``box_mask_shape``), not full-image
    (``contextmap.visual_perception.backends.alphaclip.RegionMaskSource``). A mask-based
    Region Discovery backend (SAM2, SAM3) already attaches that mask to each ``Region2D`` at
    full-image resolution (``region.mask``) before ``AlphaClipRegionFeatureBackend.extract``
    ever runs, in the same per-image stage graph -- this only crops it. No new evidence is
    produced, no threshold applied, and nothing is loaded from disk: it is the exact inverse
    of how the backend itself re-expands a box-local mask back into full-image coordinates.

    Raises:
        ValueError: If ``region`` was produced by a discovery backend that never attaches an
            inline mask (a mask-conditioned backend cannot run over a box-only region).
    """

    def load_box_local_mask(self, region: Region2D) -> Any:
        """Return ``region``'s own mask, cropped to ``region.bounding_box``'s raster envelope."""
        import numpy as np

        if region.mask is None:
            raise ValueError(
                f"region {region.region_id!r} has no inline mask: a mask-conditioned region "
                "features backend needs a mask-based region discovery backend"
            )
        full = region.mask.as_array()
        height, width = box_mask_shape(region.bounding_box)
        top = math.floor(region.bounding_box.y)
        left = math.floor(region.bounding_box.x)
        cropped = np.zeros((height, width), dtype=np.bool_)
        source_top, source_left = max(0, top), max(0, left)
        source_bottom = min(full.shape[0], top + height)
        source_right = min(full.shape[1], left + width)
        copy_height, copy_width = source_bottom - source_top, source_right - source_left
        if copy_height > 0 and copy_width > 0:
            cropped[
                source_top - top : source_top - top + copy_height,
                source_left - left : source_left - left + copy_width,
            ] = full[source_top:source_bottom, source_left:source_right]
        return cropped


class VisualPerceptionExecutor:
    """Runs the canonical Visual Perception pipeline over one sequence's images.

    This is orchestration only: every scientific decision (segmentation,
    embedding, prompting) belongs to the composed backends and to
    :mod:`contextmap.visual_perception`'s own ``resolve_pipeline()``/
    ``execute_stage_graph()``/``assemble_perception_result()``, the exact
    public path already exercised manually/in validation scripts (#507).
    The executor only materializes decodable images, wires the run-scoped
    feature extractors (including a ``mask_source`` for a mask-conditioned region-features
    backend such as AlphaCLIP, see ``_RegionInlineMaskSource``), loops over observations and
    hands results to ``PerceptionRunWriter``, together with each frame's region discovery
    audit (see ``_AuditPublishingRegionDiscovery``).

    A backend that does not (yet) satisfy the canonical preset's stage
    shape — for example a ``SemanticInterpreter`` that only implements the
    newer ``interpret(request)`` port while ``CANONICAL_PRESET_V1``'s
    ``scene_interpretation``/``region_interpretation`` stages still dispatch
    through the legacy ``interpret_scene``/``interpret_regions`` shape — is
    not special-cased here: ``execute_stage_graph()``'s own failure
    isolation records that stage ``FAILED`` for the affected observations
    and every independent stage (region discovery, dense/region features)
    still produces its real evidence. This executor never works around a
    backend/preset mismatch; that is a capability-level gap, not a runtime
    one.
    """

    def __init__(
        self,
        *,
        region_discovery: AuditedRegionDiscovery,
        dense_features: FeatureFactory,
        region_features: FeatureFactory,
        semantic_interpreter: SemanticInterpreter,
    ) -> None:
        """Bind the executor to the composed backends of the canonical preset.

        Raises:
            TypeError: If ``region_discovery`` cannot report the audit of its regions
                (``discover_audited``): the run would silently lose every rejection and merge.
        """
        if not isinstance(region_discovery, AuditedRegionDiscovery):
            raise TypeError(
                "the canonical perception path needs a region discovery backend that implements "
                f"AuditedRegionDiscovery.discover_audited(); {type(region_discovery).__name__} "
                "only returns regions, so its rejections and merges could not be persisted"
            )
        self._region_discovery = region_discovery
        self._dense_features = dense_features
        self._region_features = region_features
        self._semantic_interpreter = semantic_interpreter

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Process every image observation of the ``sequence`` input and reference the run."""
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))

        def images() -> Iterator[ImageObservation]:
            """Decodifica uma imagem por vez, guiado pelo índice.

            O laço abaixo consome um frame de cada vez, então materializar a sequência inteira
            custaria os 2,2 GB de RGB do corridor-02 (mais 364 MB de pointcloud e 20781
            registros de IMU que este estágio nem usa) sem nenhum ganho (#511).
            """
            for entry in sequence.iter_index():
                if entry.modality != "image":
                    continue
                observation = sequence.observation_at(entry.offset)
                if isinstance(observation, ImageObservation):
                    yield observation

        identity = request.identity()
        run_id = PerceptionRunId(identity)
        payload_sink = _DeferredFeaturePayloadSink()

        # Diretório de rascunho, irmão do output_dir final: nunca faz parte do
        # PerceptionRunArtifact publicado (que só existe depois de finalize()).
        scratch = output.parent / f".tmp-{output.name}-prepared-{uuid4().hex[:8]}"
        scratch.mkdir(parents=True, exist_ok=False)
        try:
            dense_scope = FeatureBuildScope(
                run_id=run_id,
                feature_stage_id="dense_feature_extraction",
                source_artifact_id=identity,
                payload_sink=payload_sink,
                prepared_image_root=scratch,
            )
            # mask_source é sempre fornecido: backends que não o exigem (CLIP, DINOv2/v3)
            # simplesmente o ignoram, e um backend condicionado a máscara (AlphaCLIP) só o usa
            # quando a região tem, de fato, uma máscara inline (#535).
            region_scope = replace(
                dense_scope,
                feature_stage_id="region_feature_extraction",
                mask_source=_RegionInlineMaskSource(),
            )
            semantic_bridge = _LegacySemanticInterpreterBridge(
                interpreter=self._semantic_interpreter, run_id=run_id, view_root=scratch
            )
            region_discovery = _AuditPublishingRegionDiscovery(self._region_discovery)
            resolved = resolve_pipeline(
                CANONICAL_PRESET_V1,
                backend_factories={
                    "region_discovery": lambda _parameters: region_discovery,
                    "dense_feature_extraction": (
                        lambda _parameters: self._dense_features(dense_scope)
                    ),
                    "region_feature_extraction": (
                        lambda _parameters: self._region_features(region_scope)
                    ),
                    "scene_interpretation": lambda _parameters: semantic_bridge,
                    "region_interpretation": lambda _parameters: semantic_bridge,
                },
            )
            writer = PerceptionRunWriter(
                output_dir=output,
                sequence_name=sequence.manifest.sequence_name,
                run_id=run_id,
                run_index=request.run_number(),
                sequence_artifact_id=sequence.manifest.artifact_id,
                selection_id=selection_identity(
                    sequence.manifest.artifact_id, FullSequenceSelection()
                ),
                enabled_capabilities=frozenset(
                    stage.capability
                    for stage in CANONICAL_PRESET_V1.stages
                    if stage.backend_id is not None
                ),
                pipeline_preset=CANONICAL_PRESET_V1,
                configuration_digest=resolved.configuration_digest(),
            )
            payload_sink.bind(writer)
            semantic_bridge.bind(writer)
            region_discovery.bind(writer)

            for image in images():
                prepared = _materialize_prepared_image(image, scratch)
                result_id = perception_result_id_for(
                    run_id=run_id, source_observation_id=image.observation_id
                )
                outcomes = execute_stage_graph(
                    resolved.build_stage_graph({"image_preparation": prepared})
                )
                result = assemble_perception_result(
                    result_id=result_id,
                    source_observation_id=SourceObservationId(str(image.observation_id)),
                    run_id=run_id,
                    sequence_artifact_id=sequence.manifest.artifact_id,
                    created_at=datetime.now(UTC).isoformat(),
                    outcomes=outcomes,
                    region_stage_id="region_discovery",
                    feature_stage_ids=("dense_feature_extraction", "region_feature_extraction"),
                    claim_stage_ids=("region_interpretation",),
                    scene_context_stage_id="scene_interpretation",
                )
                writer.add_result(result)
                writer.add_stage_outcomes(outcomes)

            manifest = writer.finalize()
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return _reference(request, PERCEPTION, str(manifest.run_id), manifest.file_inventory)


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
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the frame conventions and the predicate policies.

        ``policies.geometry_summary`` is the only source of the summary policy: it is also
        what gets persisted in the run's own provenance (see
        ``contextmap.spatial_relations.run_artifact``), so there is exactly one place that can
        say which policy actually produced the evidence, never a second parameter that could
        name a different one (review of PR #540, second round).
        """
        self._policies = policies
        self._code_version = code_version

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Relate the ``entities`` resolution over the ``geometry`` map and reference the run."""
        output = _output(request)
        resolution = EntityResolutionRunReader(_one(request, "entities"))
        conventions = self._policies.frame_conventions
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            source = geometry.geometry()
            entities = resolved_entity_geometries(
                resolution.resolved_entities(),
                source=source,
                policy=self._policies.geometry_summary,
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


def _observation_window(clock_id: str, *, start_ns: int, end_ns: int) -> ObservationWindow:
    def _timestamp(total_nanoseconds: int) -> SourceTimestamp:
        seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
        return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)

    return ObservationWindow(start=_timestamp(start_ns), end=_timestamp(end_ns))


def _context_map_configuration_fingerprint(
    *, assembly_policy: PolicyRef, up_direction: tuple[float, float, float] | None
) -> str:
    """Deterministic hash of the configuration ``ContextMapExecutor`` actually assembled with.

    ``ContextMapExecutor`` has no backend to select, but it does have two effective inputs of
    its own that change what it assembles: the assembly policy's identity and the map's
    declared up direction (``None`` is itself a distinct, meaningful configuration, not an
    absence of one). ``runtime.provenance_identity`` requires every stage artifact to record an
    effective configuration digest; this was previously hardcoded to ``None`` (PR #438 review).
    """
    payload = json.dumps(
        {
            "assembly_policy_id": assembly_policy.policy_id,
            "assembly_policy_version": assembly_policy.version,
            "up_direction": up_direction,
        },
        sort_keys=True,
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class ContextMapExecutor:
    """Assembles the repository's public product: the ``ContextMap``, by reference.

    Every entity, relation and piece of evidence was already decided by Entity Resolution and
    Spatial Relations; this only translates their real, finished runs -- and the geometric map
    and sequence they were built over -- into what :func:`~contextmap.artifact.
    assemble_context_map_with_metrics` needs, and never infers anything itself (that function
    already owns the translation rule, see :data:`~contextmap.artifact.
    CONTEXT_MAP_ASSEMBLY_POLICY_ID`).
    """

    def __init__(
        self,
        *,
        up_axis: AxisDirection | None = None,
        assembly_policy: PolicyRef | None = None,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the map frame's declared up axis and the assembly policy.

        Args:
            up_axis: Spatial Relations' own composed ``FrameConventions.up_axis``, when
                declared -- the same value that already shaped this run's relations, never
                re-derived independently here. ``None`` when no run declared one.
            assembly_policy: Identity of the assembly rule; defaults to
                :data:`~contextmap.artifact.CONTEXT_MAP_ASSEMBLY_POLICY_ID` version ``"1"``, the
                one :func:`~contextmap.artifact.assemble_context_map_with_metrics` actually
                implements.
            code_version: Code revision that produced the run.
        """
        self._up_direction = up_axis.vector if up_axis is not None else None
        self._assembly_policy = assembly_policy or PolicyRef(
            policy_id=CONTEXT_MAP_ASSEMBLY_POLICY_ID, version="1"
        )
        self._code_version = code_version
        self._configuration_fingerprint = _context_map_configuration_fingerprint(
            assembly_policy=self._assembly_policy, up_direction=self._up_direction
        )

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Assemble and write the final ``ContextMap`` from this run's real upstream artifacts.

        Raises:
            ExecutorError: If the ``sequence`` input is not the artifact the ``geometry`` input
                was actually built over. ``ArtifactKind.SEQUENCE`` is not a structural
                dependency (:attr:`~contextmap.artifact.ArtifactKind.is_structural`), so
                :class:`~contextmap.artifact.ContextMapArtifactWriter` never opens or verifies
                it; without this check, a caller wiring the wrong ``sequence`` run could publish
                a lineage entry whose ``artifact_id`` (taken from the geometry manifest) and
                ``content_identity`` (the digest of whatever directory was actually passed)
                silently name two different artifacts.
        """
        output = _output(request)
        sequence_dir = _one(request, "sequence")
        geometry_dir = _one(request, "geometry")
        resolution_dir = _one(request, "entities")
        relations_dir = _one(request, "relations")

        sequence = SequenceArtifactReader(sequence_dir)
        with GeometricMapArtifactReader(geometry_dir) as geometry_reader:
            geometry_manifest = geometry_reader.manifest
            bounds = geometry_reader.geometry().geometric_map.bounds
        if sequence.manifest.artifact_id != geometry_manifest.sequence_artifact_id:
            raise ExecutorError(
                f"stage {request.stage_id!r} was given sequence "
                f"{sequence.manifest.artifact_id!r}, but the geometric map it was also given "
                f"was built over sequence {geometry_manifest.sequence_artifact_id!r}"
            )
        expected_selection_id = selection_identity(
            sequence.manifest.artifact_id, FullSequenceSelection()
        )
        if geometry_manifest.selection_id != expected_selection_id:
            raise ExecutorError(
                f"stage {request.stage_id!r} was given a sequence selection "
                f"{expected_selection_id!r} that does not match the geometric map's own "
                f"selection {geometry_manifest.selection_id!r}"
            )
        relations_reader = SpatialRelationsRunReader(relations_dir)
        predicates = sorted(
            {relation.predicate for relation in relations_reader.iter_relations()},
            key=lambda predicate: predicate.value,
        )

        metadata = ContextMapMetadata(
            creation=MapCreation(
                assembly_policy=self._assembly_policy,
                code_version=self._code_version,
                configuration_fingerprint=self._configuration_fingerprint,
            ),
            source_sequences=(
                SourceSequence(
                    sequence_artifact_id=str(geometry_manifest.sequence_artifact_id),
                    selection_id=geometry_manifest.selection_id,
                ),
            ),
            frame=estimator_local_map_frame(
                frame_id=str(geometry_manifest.map_frame), up_direction=self._up_direction
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
            context_map_id=ContextMapId(request.identity()),
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
        manifest, _write_metrics = write_context_map_with_metrics(
            result.context_map, output_dir=output, upstream_locations=upstream_locations
        )
        return _reference(
            request, CONTEXT_MAP, str(manifest.context_map_id), manifest.file_inventory
        )
