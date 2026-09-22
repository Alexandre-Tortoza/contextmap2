"""The synthetic Solution 1 chain driven by the runtime, through the real stage executors.

Contract evidence only (fake/contract): the sequence and the perception evidence come from test
doubles (a model is exactly what CI must not depend on), but every other stage runs its real
executor through ``run_plan`` and a ``RunJournal``, and writes only into the directory the runtime
hands it. The tests check the layout ``<workspace>/<dataset>/<run>/<stage>/``, the derived identity
that makes reruns identical, reuse by reference, and recovery of an interrupted run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from chain import canned_perception_results

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    EntityResolutionRunReader,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
)
from contextmap.evaluation import canonical_real_scenario, scenario_runtime_document
from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID, build_synthetic_sequence
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    MotionCorrectionPolicy,
    ScanDisposition,
)
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    ImageObservation,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
)
from contextmap.runtime import (
    ArtifactRef,
    ExecutionPlan,
    FileArtifactStore,
    ReusePolicy,
    RunJournal,
    StageExecutionError,
    StageRequest,
    read_run,
    resolve_effective_config,
    resolve_plan,
    resume_plan,
    run_plan,
)
from contextmap.runtime.executors import (
    EntityResolutionExecutor,
    GeometricMappingExecutor,
    SemanticFusionExecutor,
    SemanticMappingExecutor,
    SensorAssociationExecutor,
    SpatialRelationsExecutor,
    StateEstimationExecutor,
    inventory_digest,
)
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    GeometryOverlapSupportPolicy,
    SemanticFusionRunReader,
)
from contextmap.semantic_mapping import (
    EntityMaterializationPolicy,
    GeometrySummaryPolicy,
    SemanticMapId,
    SemanticMappingRunReader,
)
from contextmap.sensor_association import (
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRunReader,
)
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunPolicies,
    SpatialRelationsRunReader,
)
from contextmap.state_estimation import (
    BODY_ENDPOINT,
    GeometryRequirements,
    LookupPolicy,
    StateEstimationRunReader,
    StaticRelationRequirement,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
)

STAGES = [
    "ingestion",
    "visual_perception",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "semantic_fusion",
    "semantic_mapping",
    "entity_resolution",
    "spatial_relations",
]
PROVIDED = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)
_CAMERA_CALIBRATION = CalibrationReferenceId("front_camera-calib")


class _Ingestion:
    """Test double: publishes the synthetic sequence into the stage directory it is given."""

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        sequence = build_synthetic_sequence()
        artifact_id = SequenceArtifactId(request.identity())
        with SequenceArtifactWriter(
            output_dir=request.output_dir, sequence_name=CI_FIXTURE_ID, artifact_id=artifact_id
        ) as writer:
            writer.set_calibration(sequence.calibration)
            for observation in sequence.observations:
                if isinstance(observation, ImageObservation):
                    # O subconjunto de CI grava imagens sem `calibration_id`; a associação o exige.
                    observation = dataclasses.replace(
                        observation, calibration_id=_CAMERA_CALIBRATION
                    )
                writer.add_observation(observation)
            writer.finalize()
        manifest = SequenceArtifactReader(request.output_dir).manifest
        return ArtifactRef(
            stage_id="ingestion",
            contract="SequenceArtifact",
            artifact_id=str(artifact_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class _Perception:
    """Test double: publishes canned perception evidence as a real PerceptionRunArtifact."""

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        (sequence,) = request.inputs["sequence"]
        run_id = PerceptionRunId(request.identity())
        writer = PerceptionRunWriter(
            output_dir=request.output_dir,
            sequence_name=CI_FIXTURE_ID,
            run_id=run_id,
            run_index=request.run_number(),
            sequence_artifact_id=sequence.artifact_id,
            selection_id="full-sequence",
            enabled_capabilities=frozenset({"semantic_interpreter"}),
            pipeline_preset=CANONICAL_PRESET_V1,
            configuration_digest=request.config_digest,
        )
        for result in canned_perception_results(run_id, sequence.artifact_id):
            writer.add_result(result)
        manifest = writer.finalize()
        return ArtifactRef(
            stage_id="visual_perception",
            contract="PerceptionRunArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class _Failing:
    """Wraps an executor and fails once, the way a process that dies mid-stage would."""

    def __init__(self, inner: Any, stage_id: str, failures: list[str]) -> None:
        self._inner, self._stage_id, self._failures = inner, stage_id, failures

    def execute(self, request: StageRequest) -> ArtifactRef:
        if self._failures:
            raise RuntimeError(self._failures.pop())
        ref: ArtifactRef = self._inner.execute(request)
        return ref


def _executors() -> dict[str, Any]:
    estimator = ExternalPoseEstimator(
        ExternalPoseConfig(reference_frame=FrameId("odom"), body_frame=FrameId("base_link"))
    )
    return {
        "ingestion": _Ingestion(),
        "visual_perception": _Perception(),
        "state_estimation": StateEstimationExecutor(
            estimator,
            downstream=(
                GeometryRequirements(
                    capability="geometric_mapping",
                    modalities=frozenset({"lidar"}),
                    static_relations=(
                        StaticRelationRequirement(from_endpoint=BODY_ENDPOINT, to_endpoint="lidar"),
                    ),
                ),
            ),
        ),
        "geometric_mapping": GeometricMappingExecutor(
            pose_lookup=LookupPolicy.exact(),
            motion_correction=MotionCorrectionPolicy(
                raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN
            ),
        ),
        "sensor_association": SensorAssociationExecutor(
            occlusion=OcclusionPolicy(
                cell_size_px=4,
                neighborhood_radius_cells=0,
                depth_margin_m=0.1,
                depth_margin_ratio=0.02,
            ),
            tolerances=DiagnosticTolerances(
                max_pose_time_delta_ns=1_000_000,
                max_map_window_offset_ns=None,
                max_reprojection_p95_px=None,
                max_reprojection_invalid_rate=None,
            ),
            pose_policy=LookupPolicy.exact(),
        ),
        "semantic_fusion": SemanticFusionExecutor(
            support_policy=GeometryOverlapSupportPolicy(min_geometry_count=1, min_overlap=0.5),
            accumulation_policy=BaselineAccumulationPolicy(
                abstention_labels=frozenset({"unknown"})
            ),
        ),
        "semantic_mapping": SemanticMappingExecutor(
            policy=EntityMaterializationPolicy(
                geometry=GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)
            ),
            semantic_map_id=SemanticMapId("semantic-map-ci"),
            code_digest="sha256:" + "cd" * 32,
        ),
        "entity_resolution": EntityResolutionExecutor(
            retrieval=CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1),
            builder=MatchEvidenceBuilder(
                ComparisonChannels(
                    geometry=GeometryComparisonPolicy(
                        min_shared_support_jaccard=0.5,
                        min_bounds_iou=0.5,
                        min_bounds_containment=0.9,
                        min_conflict_gap_m=0.5,
                        min_extent_ratio=0.3,
                    )
                )
            ),
            resolution=ConservativeResolutionPolicy(
                use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
            ),
        ),
        "spatial_relations": SpatialRelationsExecutor(
            policies=RelationsRunPolicies(
                frame_conventions=FrameConventions(
                    map_frame="odom",  # o mapa sintético está expresso no referencial odom
                    up_axis=AxisDirection.POSITIVE_Z,
                    forward_axis=AxisDirection.POSITIVE_X,
                ),
                candidate=CandidatePolicy(
                    predicates=(RelationPredicate.NEXT_TO, RelationPredicate.TOUCHING),
                    proximity_radius_m=0.6,
                    directional_radius_m=2.0,
                ),
                geometric=GeometricPredicatePolicy(
                    boundary_tolerance_m=0.02,
                    next_to_max_gap_m=0.5,
                    adjacent_penetration_m=0.05,
                    containment_slack_m=0.05,
                    directional_overlap_fraction=0.5,
                ),
                contact=ContactPredicatePolicy(
                    contact_distance_m=0.05,
                    contact_tolerance_m=0.02,
                    min_contact_points=3,
                    support_height_tolerance_m=0.05,
                    support_footprint_fraction=0.5,
                    leaning_min_tilt_deg=10.0,
                    leaning_max_tilt_deg=80.0,
                    tilt_tolerance_deg=2.0,
                    leaning_min_vertical_overlap_m=0.3,
                ),
            ),
            geometry_summary=GeometrySummaryPolicy(
                sparse_point_threshold=3, connectivity_radius_m=0.5
            ),
        ),
    }


def _scope(tmp_path: Path) -> tuple[Any, ExecutionPlan]:
    # O perfil canônico do cenário, expresso como configuração do runtime, com o dataset do run.
    document = scenario_runtime_document(canonical_real_scenario())
    document["inputs"] = {"sequence": CI_FIXTURE_ID}
    # O cenário congelado ainda não escolhe backend para estes componentes: eles só passaram a
    # existir no catálogo depois que o cenário foi congelado. A escolha é só deste teste, nunca
    # do cenário: mudar o cenário mudaria seu digest e invalidaria o relatório real registrado.
    components = document["components"]
    components.setdefault("geometric_mapping", {})["pose_lookup"] = {"backend": "lookup-policy-v1"}
    components.setdefault("geometric_mapping", {})["motion_correction"] = {
        "backend": "motion-correction-v1"
    }
    components.setdefault("sensor_association", {})["occlusion"] = {
        "backend": "conservative-depth-support-v1"
    }
    components.setdefault("sensor_association", {})["tolerances"] = {
        "backend": "diagnostic-tolerances-v1"
    }
    components.setdefault("sensor_association", {})["pose_policy"] = {"backend": "lookup-policy-v1"}
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    effective = resolve_effective_config(files=[path])
    return effective, resolve_plan(effective).scope(targets=["spatial_relations"])


def _run(
    effective: Any,
    execution: ExecutionPlan,
    workspace: Path,
    executors: dict[str, Any],
    **options: Any,
) -> tuple[RunJournal, Any]:
    journal = RunJournal.create(workspace, effective, execution)
    record = run_plan(
        execution,
        executors,
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        journal=journal,
        **options,
    )
    return journal, record


def _outputs(record: Any) -> dict[str, ArtifactRef]:
    return {stage.stage_id: stage.output for stage in record.stages}


def test_every_stage_writes_its_artifact_into_its_own_directory_of_the_run(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    journal, record = _run(effective, execution, workspace, _executors())

    run = workspace / CI_FIXTURE_ID / "run-0001"
    assert journal.directory == run
    assert read_run(run).status.value == "completed"
    assert sorted(path.name for path in run.iterdir() if path.is_dir()) == sorted(STAGES)
    refs = _outputs(record)
    assert {stage: ref.location for stage, ref in refs.items()} == {
        stage: f"{CI_FIXTURE_ID}/run-0001/{stage}" for stage in STAGES
    }
    assert SequenceArtifactReader(run / "ingestion").verify_integrity() == []
    assert PerceptionRunReader(run / "visual_perception").verify_integrity() == []
    assert StateEstimationRunReader(run / "state_estimation").verify_integrity() == []
    with GeometricMapArtifactReader(run / "geometric_mapping") as geometry:
        assert geometry.verify_integrity() == []
    assert SensorAssociationRunReader(run / "sensor_association").verify_integrity() == []
    assert SemanticFusionRunReader(run / "semantic_fusion").verify_integrity() == []
    mapping = SemanticMappingRunReader(run / "semantic_mapping")
    assert mapping.verify_integrity() == []
    assert len(mapping.entity_ids()) == 2  # a palete e o poste, sem fusão entre suportes
    resolution = EntityResolutionRunReader(run / "entity_resolution")
    assert resolution.verify_integrity() == []
    relations = SpatialRelationsRunReader(run / "spatial_relations")
    assert relations.verify_integrity() == []


def test_no_stage_writes_outside_the_directory_the_runtime_gave_it(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    _run(effective, execution, workspace, _executors())

    # Só `<dataset>/run-0001/...`: nenhum `sequences/`, `runs/` ou registro ao lado.
    assert [path.name for path in workspace.iterdir()] == [CI_FIXTURE_ID]
    assert [path.name for path in (workspace / CI_FIXTURE_ID).iterdir()] == ["run-0001"]
    assert not any(path.name == "runs.json" for path in workspace.rglob("*"))


def test_a_rerun_of_the_same_execution_produces_the_same_identities_and_content(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    _, first = _run(effective, execution, workspace, _executors())
    _, second = _run(effective, execution, workspace, _executors())

    one, two = _outputs(first), _outputs(second)
    assert set(one) == set(two) == set(STAGES)
    for stage in STAGES:
        assert one[stage].artifact_id == two[stage].artifact_id, stage
        assert one[stage].content_hash == two[stage].content_hash, stage
        assert one[stage].location != two[stage].location  # runs distintos, pastas distintas


def test_a_third_run_reuses_every_stage_by_reference_and_copies_nothing(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"
    store = FileArtifactStore(
        tmp_path / "index",
        verify=lambda ref: ref.location is not None and (workspace / ref.location).is_dir(),
    )
    policy = ReusePolicy(store=store, code_identity="code-1")
    _, first = _run(effective, execution, workspace, _executors(), reuse=policy)

    journal, second = _run(effective, execution, workspace, _executors(), reuse=policy)

    assert {s.decision.kind for s in second.stages if s.decision is not None} == {"reused"}
    assert not any(path.is_dir() for path in journal.directory.iterdir())  # só o diário
    assert _outputs(second) == _outputs(first)  # as mesmas referências, para a pasta do run 1
    assert all(
        ref.location is not None and ref.location.startswith(f"{CI_FIXTURE_ID}/run-0001/")
        for ref in _outputs(second).values()
    )


def test_an_interrupted_run_is_resumed_from_the_completed_stages_and_leaves_no_partial_artifact(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"
    store = FileArtifactStore(
        tmp_path / "index",
        verify=lambda ref: ref.location is not None and (workspace / ref.location).is_dir(),
    )
    policy = ReusePolicy(store=store, code_identity="code-1")
    failing = _executors()
    failing["semantic_fusion"] = _Failing(failing["semantic_fusion"], "semantic_fusion", ["boom"])

    with pytest.raises(StageExecutionError, match="boom"):
        _run(effective, execution, workspace, failing, reuse=policy)

    interrupted = workspace / CI_FIXTURE_ID / "run-0001"
    assert read_run(interrupted).status.value == "failed"
    # Os estágios concluídos existem; o que falhou não deixou pasta, nem temporária.
    assert (interrupted / "geometric_mapping").is_dir()
    assert (interrupted / "sensor_association").is_dir()
    assert not (interrupted / "semantic_fusion").exists()
    assert not list(interrupted.rglob(".tmp-*"))

    journal = RunJournal.create(workspace, effective, execution)
    record = resume_plan(
        interrupted,
        execution,
        _executors(),
        reuse=policy,
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        journal=journal,
    )

    refs = _outputs(record)
    resumed = f"{CI_FIXTURE_ID}/run-0002/"
    assert refs["semantic_fusion"].location == f"{resumed}semantic_fusion"
    assert refs["semantic_mapping"].location == f"{resumed}semantic_mapping"
    assert refs["entity_resolution"].location == f"{resumed}entity_resolution"
    assert refs["spatial_relations"].location == f"{resumed}spatial_relations"
    for stage in STAGES[:5]:  # os concluídos foram reutilizados por referência
        assert refs[stage].location == f"{CI_FIXTURE_ID}/run-0001/{stage}"
    assert read_run(journal.directory).status.value == "completed"
    assert SemanticFusionRunReader(journal.directory / "semantic_fusion").verify_integrity() == []


def test_a_stage_that_gets_several_runs_for_one_input_is_refused_not_resolved(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"
    executors = _executors()
    _, record = _run(effective, execution, workspace, executors)
    refs = _outputs(record)
    request = StageRequest(
        stage_id="sensor_association",
        inputs={
            "sequence": (refs["ingestion"],),
            "perception": (refs["visual_perception"], refs["visual_perception"]),
            "trajectory": (refs["state_estimation"],),
            "geometry": (refs["geometric_mapping"],),
        },
        components={},
        config_digest="d",
        output_dir=workspace / CI_FIXTURE_ID / "run-0009" / "sensor_association",
        workspace=workspace,
    )

    with pytest.raises(ValueError, match="exactly one run of input 'perception'"):
        executors["sensor_association"].execute(request)


def test_resolution_and_relations_are_derived_from_the_entities_without_rewriting_them(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    _, record = _run(effective, execution, workspace, _executors())

    run = workspace / CI_FIXTURE_ID / "run-0001"
    mapping = SemanticMappingRunReader(run / "semantic_mapping")
    resolution = EntityResolutionRunReader(run / "entity_resolution")
    relations = SpatialRelationsRunReader(run / "spatial_relations")
    # A resolução decide sobre as entidades materializadas e guarda a origem de cada uma; um
    # candidato sem prova de identidade fica separado (sem fusão silenciosa).
    resolved = resolution.resolved_entities()
    members = {
        member.entity_ref.entity_id for entity in resolved.entities for member in entity.members
    }
    assert members == set(mapping.entity_ids())
    lineage = resolution.manifest.lineage
    assert str(lineage.semantic_mapping_run_id) == str(mapping.manifest.run_id)
    # As relações cobrem exatamente as entidades resolvidas e nascem do run de resolução.
    assert str(relations.manifest.lineage.entity_resolution_run_id) == str(resolution.run_id)
    refs = _outputs(record)
    assert refs["entity_resolution"].artifact_id == str(resolution.run_id)
    assert refs["spatial_relations"].artifact_id == str(relations.manifest.run_id)
    assert refs["spatial_relations"].content_hash == inventory_digest(
        relations.manifest.file_inventory
    )
