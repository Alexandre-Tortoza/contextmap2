"""Tests for the composition root: configuration in, constructed implementations out."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_documents import SUPPORT_POLICY, effective_from, selected_document
from runtime_fixtures import unavailable_context_map  # noqa: F401

from contextmap.ingestion import SourceAdapterConfig, SourceTopicMapping
from contextmap.point_representation.backends.geometric_descriptor import GeometricDescriptorEncoder
from contextmap.runtime import ConfigurationError
from contextmap.runtime.composition import (
    ComposedRuntime,
    FeatureBuildScope,
    RuntimeProvider,
    composable_backends,
    compose,
    compose_executors,
    composed_stages,
)
from contextmap.runtime.errors import (
    BackendConfigurationError,
    BackendRuntimeMissingError,
    BackendUnavailableError,
    StageUnavailableError,
)
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceChannel,
    GeometryOverlapSupportPolicy,
    QualityAwareAccumulationPolicy,
)
from contextmap.state_estimation.backends.external_pose import ExternalPoseEstimator
from contextmap.state_estimation.backends.fast_lio import FastLioEstimator
from contextmap.visual_perception import PerceptionRunId
from contextmap.visual_perception.backends.clip import ClipVisualFeatureBackend
from contextmap.visual_perception.backends.dinov3 import DinoV3DenseFeatureBackend
from contextmap.visual_perception.backends.florence2 import Florence2RegionDiscovery
from contextmap.visual_perception.backends.gemini import GeminiSemanticInterpreter
from contextmap.visual_perception.backends.qwen import QwenSemanticInterpreter
from contextmap.visual_perception.backends.sam3 import Sam3Config, Sam3RegionDiscovery

REGION = "visual_perception.region_discovery"
INTERPRETER = "visual_perception.semantic_interpretation"


class _Recorder:
    """Stands in for a caller-supplied model runtime and records how it was requested."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[str, ...]]] = []

    def provider(self, name: str = "runtime") -> RuntimeProvider:
        def provide(config: Any, secrets: Any) -> object:
            self.calls.append((config, secrets.names))
            return _FakeRuntime(name)

        return provide


class _FakeRuntime:
    def __init__(self, name: str) -> None:
        self.name = name


def _providers(recorder: _Recorder) -> dict[str, RuntimeProvider]:
    return {
        REGION: recorder.provider("sam3"),
        INTERPRETER: recorder.provider("qwen"),
    }


def _compose(
    tmp_path: Path,
    *,
    document: dict[str, Any] | None = None,
    recorder: _Recorder | None = None,
    **options: Any,
) -> ComposedRuntime:
    recorder = recorder or _Recorder()
    options.setdefault("module_available", lambda _name: True)
    options.setdefault("environ", {})
    return compose(effective_from(tmp_path, document), providers=_providers(recorder), **options)


class TestCanonicalComposition:
    def test_constructs_each_selected_implementation_from_configuration(
        self, tmp_path: Path
    ) -> None:
        composed = _compose(tmp_path)

        assert isinstance(composed.region_discovery, Sam3RegionDiscovery)
        assert isinstance(composed.semantic_interpreter, QwenSemanticInterpreter)
        assert isinstance(composed.state_estimator, ExternalPoseEstimator)
        assert isinstance(composed.point_encoder, GeometricDescriptorEncoder)
        assert composed.support_policy == GeometryOverlapSupportPolicy(
            min_geometry_count=5, min_overlap=0.3
        )
        assert composed.accumulation_policy == BaselineAccumulationPolicy()

    def test_the_capability_config_carries_the_parameters_and_the_resource_device(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["device"] = "cuda"
        composed = _compose(tmp_path, document=document, recorder=recorder)

        config, _secrets = recorder.calls[0]
        assert isinstance(config, Sam3Config)
        assert (config.checkpoint, config.device, config.prompt) == ("sam3-x", "cuda", "chair")
        assert (
            composed.region_discovery.backend_provenance().configuration_fingerprint  # type: ignore[union-attr]
            == config.digest
        )

    def test_swapping_a_backend_is_a_configuration_change_only(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["visual_perception"]["region_discovery"] = {
            "backend": "florence2",
            "florence2": {"checkpoint": "microsoft/Florence-2-x", "task": "<OD>"},
        }

        composed = _compose(tmp_path, document=document)

        assert isinstance(composed.region_discovery, Florence2RegionDiscovery)

    def test_quality_aware_policy_is_built_with_its_nested_values(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["semantic_fusion"]["accumulation"] = {
            "backend": "quality-aware-evidence-accumulation-v1",
            "quality-aware-evidence-accumulation-v1": {
                "definitions_version": "1",
                "neutral_factor": 0.5,
                "ramps": [{"quality_input": "visible_share", "good": 1.0, "bad": 0.2}],
                "baseline": {"channels": ["semantic_claims", "point_representation"]},
            },
        }

        policy = _compose(tmp_path, document=document).accumulation_policy

        assert isinstance(policy, QualityAwareAccumulationPolicy)
        assert policy.baseline.channels == frozenset(
            {EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.POINT_REPRESENTATION}
        )
        assert policy.ramps[0].good == 1.0

    def test_composes_the_external_secret_only_for_backends_that_declare_it(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["components"]["visual_perception"]["semantic_interpretation"] = {
            "backend": "gemini",
            "gemini": {"model": "gemini-x", "timeout_s": 30, "max_retries": 2, "temperature": 0.0},
        }

        composed = _compose(
            tmp_path,
            document=document,
            recorder=recorder,
            environ={"GEMINI_API_KEY": "s3cr3t"},
        )

        assert isinstance(composed.semantic_interpreter, GeminiSemanticInterpreter)
        interpreter_call = recorder.calls[1]
        assert interpreter_call[1] == ("GEMINI_API_KEY",)
        assert recorder.calls[0][1] == ()


class TestFeatureBackendsAreRunScoped:
    def _scope(self, tmp_path: Path) -> FeatureBuildScope:
        return FeatureBuildScope(
            run_id=PerceptionRunId("run-1"),
            feature_stage_id="dense_feature_extraction",
            source_artifact_id="artifact-1",
            payload_sink=object(),
            prepared_image_root=tmp_path,
        )

    def test_builds_the_configured_extractors_once_a_run_scope_exists(self, tmp_path: Path) -> None:
        composed = _compose(tmp_path)
        assert composed.dense_features is not None
        assert composed.region_features is not None

        dense = composed.dense_features(self._scope(tmp_path))
        region = composed.region_features(self._scope(tmp_path))

        assert isinstance(dense, DinoV3DenseFeatureBackend)
        assert isinstance(region, ClipVisualFeatureBackend)

    def test_the_region_features_slot_fixes_the_clip_scope(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["visual_perception"]["region_features"]["clip"]["scope"] = "global"

        with pytest.raises(BackendConfigurationError, match="scope"):
            _compose(tmp_path, document=document)

    def test_configuration_errors_surface_at_composition_not_at_run_time(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["visual_perception"]["dense_features"]["dinov3"]["revision"] = "main"

        with pytest.raises(BackendConfigurationError, match="revision"):
            _compose(tmp_path, document=document)


class TestOtherBackends:
    def test_fast_lio_gets_its_bundled_subprocess_runner_from_configuration(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {
            "backend": "fast_lio",
            "fast_lio": {
                "reference_frame": "map",
                "body_frame": "base",
                "fast_lio_ref": "v1",
                "scan_period_ns": 100_000_000,
                "runner": {
                    "command": ["fast-lio", "{input_bag}", "{job}", "{output_dir}"],
                    "timeout_s": 60,
                },
            },
        }

        composed = _compose(tmp_path, document=document)

        assert isinstance(composed.state_estimator, FastLioEstimator)

    def test_fast_lio_without_a_runner_is_a_configuration_error(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {
            "backend": "fast_lio",
            "fast_lio": {
                "reference_frame": "map",
                "body_frame": "base",
                "fast_lio_ref": "v1",
                "scan_period_ns": 100_000_000,
            },
        }

        with pytest.raises(BackendConfigurationError, match="runner"):
            _compose(tmp_path, document=document)

    def test_fast_lio_rejects_a_command_without_the_placeholders(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {
            "backend": "fast_lio",
            "fast_lio": {
                "reference_frame": "map",
                "body_frame": "base",
                "fast_lio_ref": "v1",
                "scan_period_ns": 100_000_000,
                "runner": {"command": ["fast-lio"], "timeout_s": 60},
            },
        }

        with pytest.raises(BackendConfigurationError, match="runner"):
            _compose(tmp_path, document=document)

    def test_a_learned_encoder_needs_a_support_policy_and_a_caller_runtime(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        ptv3 = {
            "variant": "ptv3-base",
            "checkpoint": "ptv3-x",
            "checkpoint_hash": "sha256:" + "0" * 64,
            "precision": "float32",
            "grid_size_m": 0.02,
            "output_dimension": 64,
            "pooling": "mean",
            "normalization": "l2",
            "min_support_points": 8,
        }
        document["components"]["point_representation"]["encoder"] = {
            "backend": "ptv3",
            "ptv3": ptv3,
        }
        with pytest.raises(BackendConfigurationError, match="support_policy"):
            _compose(tmp_path, document=document)

        ptv3["support_policy"] = SUPPORT_POLICY
        with pytest.raises(BackendRuntimeMissingError, match="PTv3Runtime"):
            _compose(tmp_path, document=document)

        recorder = _Recorder()
        providers = {**_providers(recorder), "point_representation.encoder": recorder.provider()}
        composed = compose(
            effective_from(tmp_path, document),
            providers=providers,
            module_available=lambda _name: True,
            environ={},
        )
        assert type(composed.point_encoder).__name__ == "PTv3PointEncoder"

    def test_region_discovery_forwards_its_pass_and_normalization_policies(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        sam3 = document["components"]["visual_perception"]["region_discovery"]["sam3"]
        sam3["pass_config"] = {"include_full_frame": True, "max_candidates_per_pass": 10}
        sam3["normalization_config"] = {"minimum_area_pixels": 32}

        composed = _compose(tmp_path, document=document)

        backend = composed.region_discovery
        assert backend._pass_config.max_candidates_per_pass == 10  # type: ignore[union-attr]
        assert backend._normalization_config.minimum_area_pixels == 32.0  # type: ignore[union-attr]

    def test_a_mask_conditioned_extractor_needs_the_mask_source_of_its_run(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["visual_perception"]["region_features"] = {
            "backend": "alphaclip",
            "alphaclip": {
                "model_name": "ViT-B/16",
                "base_checkpoint_path": "clip.pt",
                "alpha_checkpoint_path": "alpha.pt",
                "checkpoint_fingerprint": "sha256:" + "1" * 64,
            },
        }
        composed = _compose(tmp_path, document=document)
        assert composed.region_features is not None
        scope = FeatureBuildScope(
            run_id=PerceptionRunId("run-1"),
            feature_stage_id="region_feature_extraction",
            source_artifact_id="artifact-1",
            payload_sink=object(),
            prepared_image_root=tmp_path,
        )

        with pytest.raises(BackendConfigurationError, match="mask_source"):
            composed.region_features(scope)

    def test_a_supplied_runtime_replaces_the_bundled_loader_and_its_module_checks(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        composed = compose(
            effective_from(tmp_path),
            providers={
                **_providers(recorder),
                "visual_perception.dense_features": recorder.provider("dino"),
                "visual_perception.region_features": recorder.provider("clip"),
            },
            module_available=lambda name: name not in {"torch", "transformers", "PIL"},
            environ={},
        )
        assert composed.dense_features is not None
        scope = FeatureBuildScope(
            run_id=PerceptionRunId("run-1"),
            feature_stage_id="dense",
            source_artifact_id="artifact-1",
            payload_sink=object(),
        )

        backend = composed.dense_features(scope)

        assert backend._runtime.name == "dino"  # type: ignore[attr-defined]


class TestSourceAdapter:
    def _request(self, source_type: str) -> SourceAdapterConfig:
        return SourceAdapterConfig(
            source_type=source_type, path="bag.bag", topics=SourceTopicMapping()
        )

    def test_builds_the_configured_adapter_for_one_request(self, tmp_path: Path) -> None:
        pytest.importorskip("rosbags")
        composed = _compose(tmp_path)
        assert composed.source_adapter is not None

        adapter = composed.source_adapter(self._request("ros1_bag"))

        assert type(adapter).__name__ == "Ros1BagSourceAdapter"

    def test_refuses_a_request_for_another_adapter_family(self, tmp_path: Path) -> None:
        composed = _compose(tmp_path)
        assert composed.source_adapter is not None

        with pytest.raises(BackendConfigurationError, match="ros2_bag"):
            composed.source_adapter(self._request("ros2_bag"))


class TestExplicitFailures:
    def test_a_backend_without_a_bundled_runtime_needs_one_from_the_caller(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(BackendRuntimeMissingError) as error:
            compose(
                effective_from(tmp_path),
                providers={INTERPRETER: _Recorder().provider()},
                module_available=lambda _name: True,
                environ={},
            )

        assert REGION in str(error.value)
        assert "sam3" in str(error.value)

    def test_a_missing_optional_module_is_reported_with_its_install_hint(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(BackendUnavailableError, match="torch"):
            _compose(tmp_path, module_available=lambda name: name != "torch")

    def test_a_missing_secret_is_reported_by_name(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["visual_perception"]["semantic_interpretation"] = {
            "backend": "gemini",
            "gemini": {"model": "gemini-x", "timeout_s": 30, "max_retries": 2, "temperature": 0.0},
        }

        with pytest.raises(BackendUnavailableError, match="GEMINI_API_KEY"):
            _compose(tmp_path, document=document, environ={})

    def test_a_capability_rejection_happens_before_any_runtime_is_requested(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["components"]["visual_perception"]["region_discovery"]["sam3"] = {
            "checkpoint": "x",
            "strategy": "text_prompt",
        }

        with pytest.raises(BackendConfigurationError, match="requires a prompt"):
            _compose(tmp_path, document=document, recorder=recorder)

        assert recorder.calls == []

    def test_an_unknown_parameter_names_the_valid_ones(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"]["external_pose"]["nope"] = 1

        with pytest.raises(BackendConfigurationError, match="reference_frame"):
            _compose(tmp_path, document=document)

    def test_an_incomplete_selection_is_a_configuration_error(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {"backend": None}

        with pytest.raises(ConfigurationError, match=r"state_estimation\.estimator"):
            _compose(tmp_path, document=document)


class TestStagesAndExtensionPoints:
    def test_a_stage_that_is_not_selected_is_not_built(self, tmp_path: Path) -> None:
        document = selected_document()
        document["pipeline"]["stages"]["point_representation"] = False

        composed = _compose(tmp_path, document=document)

        assert composed.point_encoder is None
        assert "point_representation" not in composed.stages

    @pytest.mark.usefixtures("unavailable_context_map")
    def test_stages_without_an_implemented_capability_are_listed_not_simulated(
        self, tmp_path: Path
    ) -> None:
        composed = _compose(tmp_path)

        assert set(composed.unavailable_stages) == {"context_map"}
        assert "milestone" in composed.unavailable_stages["context_map"]

    @pytest.mark.usefixtures("unavailable_context_map")
    def test_asking_explicitly_for_an_unavailable_stage_fails_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(StageUnavailableError, match="context_map"):
            _compose(tmp_path, stages=["context_map"])

    def test_a_subset_of_stages_builds_only_those(self, tmp_path: Path) -> None:
        recorder = _Recorder()

        composed = _compose(tmp_path, recorder=recorder, stages=["state_estimation"])

        assert composed.state_estimator is not None
        assert composed.region_discovery is None
        assert recorder.calls == []

    def test_an_unknown_stage_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="nope"):
            _compose(tmp_path, stages=["nope"])


class TestCatalogAgreement:
    def test_every_catalog_backend_has_a_builder_and_no_builder_is_undeclared(self) -> None:
        from contextmap.runtime.catalog import COMPONENTS

        declared = {
            (component_id, backend_id)
            for component_id, spec in COMPONENTS.items()
            for backend_id in spec.backends
        }

        assert composable_backends() == declared

    def test_every_available_stage_of_the_catalog_has_a_composer(self) -> None:
        from contextmap.runtime.catalog import CANONICAL_PRESET

        available = {stage.stage_id for stage in CANONICAL_PRESET.stages if stage.available}

        assert composed_stages() == available


class TestComposeExecutors:
    """Blocker #1/#2 of the PR #387 review: real ``StageExecutor``s from configuration alone.

    ``compose_executors`` is the automatic counterpart of ``compose``: it is what lets the
    installed ``contextmap`` binary run ``state_estimation``, ``geometric_mapping``,
    ``sensor_association`` and ``semantic_fusion`` with no Python caller building an
    executor by hand.
    """

    def test_composes_a_real_executor_for_every_stage_it_can_build_from_configuration(
        self, tmp_path: Path
    ) -> None:
        from contextmap.runtime.executors import (
            EntityResolutionExecutor,
            GeometricMappingExecutor,
            SemanticFusionExecutor,
            SensorAssociationExecutor,
            SpatialRelationsExecutor,
            StateEstimationExecutor,
        )

        executors = compose_executors(
            effective_from(tmp_path), module_available=lambda _name: True, environ={}
        )

        assert set(executors) == {
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "semantic_fusion",
            "entity_resolution",
            "spatial_relations",
        }
        assert isinstance(executors["state_estimation"], StateEstimationExecutor)
        assert isinstance(executors["geometric_mapping"], GeometricMappingExecutor)
        assert isinstance(executors["sensor_association"], SensorAssociationExecutor)
        assert isinstance(executors["semantic_fusion"], SemanticFusionExecutor)
        assert isinstance(executors["entity_resolution"], EntityResolutionExecutor)
        assert isinstance(executors["spatial_relations"], SpatialRelationsExecutor)
        # Nunca fabricado: capabilities sem executor real continuam ausentes, honestamente.
        assert "ingestion" not in executors
        assert "visual_perception" not in executors
        assert "point_representation" not in executors
        assert "semantic_mapping" not in executors

    def test_a_stage_whose_variation_points_are_not_selected_is_left_out_not_fabricated(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        del document["components"]["geometric_mapping"]
        del document["components"]["sensor_association"]

        executors = compose_executors(
            effective_from(tmp_path, document), module_available=lambda _name: True, environ={}
        )

        assert set(executors) == {
            "state_estimation",
            "semantic_fusion",
            "entity_resolution",
            "spatial_relations",
        }

    def test_a_broken_backend_in_one_stage_never_costs_another_stage_its_executor(
        self, tmp_path: Path
    ) -> None:
        """A rejected parameter is isolated to its own stage, one at a time.

        ``sensor_association.tolerances`` rejects ``max_reprojection_invalid_rate`` above 1
        (see ``DiagnosticTolerances.__post_init__``), so composing that stage fails; the
        other stages, whose own configuration is unrelated, still compose normally.
        """
        document = selected_document()
        document["components"]["sensor_association"]["tolerances"]["diagnostic-tolerances-v1"][
            "max_reprojection_invalid_rate"
        ] = 2.0

        executors = compose_executors(
            effective_from(tmp_path, document), module_available=lambda _name: True, environ={}
        )

        assert set(executors) == {
            "state_estimation",
            "geometric_mapping",
            "semantic_fusion",
            "entity_resolution",
            "spatial_relations",
        }

    def test_an_unselected_optional_channel_leaves_it_none_never_a_default_policy(
        self, tmp_path: Path
    ) -> None:
        """Every optional Entity Resolution channel and Spatial Relations predicate is absent.

        ``selected_document()`` never selects ``semantic_compatibility``, ``temporal_
        compatibility``, ``appearance``, ``representation``, ``geometric_predicate`` or
        ``contact_predicate``: the stages still compose, with the geometry-only path.
        """
        composed = _compose(tmp_path)

        channels = composed.entity_comparison_channels
        assert channels is not None
        assert channels.geometry is not None
        assert channels.semantic is None
        assert channels.temporal is None
        assert channels.appearance is None
        assert channels.representation is None

        policies = composed.spatial_relations_policies
        assert policies is not None
        assert policies.frame_conventions is not None
        assert policies.candidate is not None
        assert policies.geometric is None
        assert policies.contact is None

    def test_an_incomplete_required_selection_leaves_the_stage_absent_not_fabricated(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        del document["components"]["entity_resolution"]["geometry_comparison"]

        executors = compose_executors(
            effective_from(tmp_path, document), module_available=lambda _name: True, environ={}
        )

        assert "entity_resolution" not in executors
        # spatial_relations depende de entity_resolution rio abaixo, não da sua composição:
        # a seleção incompleta de um estágio nunca contamina a composição de outro.
        assert "spatial_relations" in executors

    def test_a_composed_executor_is_the_real_thing_and_runs_against_a_real_artifact(
        self, tmp_path: Path
    ) -> None:
        """The composed ``StateEstimationExecutor`` is not a type-compatible stub.

        It reads a real ``SequenceArtifact`` through ``StageRequest.directory_of`` and
        writes a real ``StateEstimationRunArtifact`` where the request tells it to,
        exactly as the runtime's own DAG runner would call it.
        """
        from contextmap.ingestion import (
            ExternalPoseMeasurement,
            FrameId,
            SensorId,
            SequenceArtifactId,
            SequenceArtifactWriter,
            SourceObservationId,
            SourceProvenance,
        )
        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.executors import inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.shared import SourceTimestamp

        workspace = tmp_path / "ws"
        ingestion_dir = workspace / "corridor-02" / "run-0001" / "ingestion"
        with SequenceArtifactWriter(
            output_dir=ingestion_dir,
            sequence_name="corridor-02",
            artifact_id=SequenceArtifactId("sequence-0001"),
        ) as writer:
            for index in range(3):
                writer.add_observation(
                    ExternalPoseMeasurement(
                        observation_id=SourceObservationId(f"pose-{index:04d}"),
                        sensor_id=SensorId("external_pose_source"),
                        frame_id=FrameId("base"),
                        timestamp=SourceTimestamp(
                            seconds=index, nanoseconds=0, clock_id="fixture:header"
                        ),
                        provenance=SourceProvenance(
                            source_type="fixture", source_path="fixtures/poses"
                        ),
                        parent_frame=FrameId("map"),
                        translation=(float(index), 0.0, 0.0),
                        orientation=(0.0, 0.0, 0.0, 1.0),
                    )
                )
            manifest = writer.finalize()
        sequence_ref = ArtifactRef(
            stage_id="ingestion",
            contract="SequenceArtifact",
            artifact_id=str(manifest.artifact_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location="corridor-02/run-0001/ingestion",
        )
        effective = effective_from(tmp_path)
        executors = compose_executors(effective, module_available=lambda _name: True, environ={})
        output_dir = workspace / "corridor-02" / "run-0001" / "state_estimation"
        request = StageRequest(
            stage_id="state_estimation",
            inputs={"sequence": (sequence_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=output_dir,
            workspace=workspace,
        )

        ref = executors["state_estimation"].execute(request)

        assert ref.contract == "StateEstimationRunArtifact"
        assert (output_dir / "manifest.json").is_file()

    def test_the_entity_resolution_executor_is_the_real_thing_and_runs_against_a_real_artifact(
        self, tmp_path: Path
    ) -> None:
        """The composed ``EntityResolutionExecutor`` reads a real ``SemanticMappingRunArtifact``.

        Not a fake type-compatible stand-in: ``make_entity`` builds one genuinely self-consistent
        ``Entity`` (its geometry derived through ``summarize_geometry`` over a real
        ``GeometrySource``), persisted with the capability's own ``SemanticMappingRunWriter``. The
        composed executor reads it back through ``SemanticMappingRunReader`` and writes a real
        ``EntityResolutionRunArtifact``, exactly as the runtime's own DAG runner would call it.
        """
        from runtime_entities import make_entity

        from contextmap.geometric_mapping import MapId
        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.executors import inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.semantic_fusion import SemanticFusionRunId
        from contextmap.semantic_mapping import (
            MappingRunLineage,
            SemanticMapId,
            SemanticMappingRunId,
            SemanticMappingRunWriter,
        )
        from contextmap.visual_perception import PerceptionRunId

        workspace = tmp_path / "ws"
        mapping_dir = workspace / "corridor-02" / "run-0001" / "semantic_mapping"
        semantic_map_id = SemanticMapId("semantic-map-ci")
        entity = make_entity(semantic_map_id=semantic_map_id)
        manifest = SemanticMappingRunWriter(
            output_dir=mapping_dir,
            sequence_name="corridor-02",
            run_id=SemanticMappingRunId("run-0001--semantic-mapping"),
            run_index=1,
            semantic_map_id=semantic_map_id,
            lineage=MappingRunLineage(
                sequence_artifact_id="sequence-0001",
                geometric_map_id=MapId("map-0001"),
                fusion_run_id=SemanticFusionRunId("fusion-run-0001"),
                fusion_schema_version="0.1.0",
                fusion_artifact_digest="sha256:artifact",
                association_run_ids=("association-run-0001",),
                perception_run_ids=(PerceptionRunId("perception-run-0001"),),
                point_representation_run_ids=(),
            ),
            code_version="test",
            code_digest="sha256:" + "cd" * 32,
        ).write([entity])
        entities_ref = ArtifactRef(
            stage_id="semantic_mapping",
            contract="SemanticEntityArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location="corridor-02/run-0001/semantic_mapping",
        )
        effective = effective_from(tmp_path)
        executors = compose_executors(effective, module_available=lambda _name: True, environ={})
        output_dir = workspace / "corridor-02" / "run-0001" / "entity_resolution"
        request = StageRequest(
            stage_id="entity_resolution",
            inputs={"entities": (entities_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=output_dir,
            workspace=workspace,
        )

        ref = executors["entity_resolution"].execute(request)

        assert ref.contract == "EntityResolutionRunArtifact"
        assert (output_dir / "manifest.json").is_file()


def test_composition_does_not_load_a_model_or_import_a_heavy_sdk(tmp_path: Path) -> None:
    import sys

    before = {name for name in ("torch", "transformers", "rosbags") if name in sys.modules}

    _compose(tmp_path)

    after = {name for name in ("torch", "transformers") if name in sys.modules}
    assert after <= before
