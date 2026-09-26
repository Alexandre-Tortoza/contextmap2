"""Tests for the composition root: configuration in, constructed implementations out."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import runtime_provider_fixtures
from runtime_documents import SUPPORT_POLICY, effective_from, selected_document
from runtime_fixtures import unavailable_future_stage  # noqa: F401

from contextmap.geometric_mapping import GeometricMapArtifactManifest
from contextmap.ingestion import SourceAdapterConfig, SourceTopicMapping
from contextmap.point_representation.backends.geometric_descriptor import GeometricDescriptorEncoder
from contextmap.runtime import (
    ConfigProblem,
    ConfigurationError,
    PreflightReport,
    preflight,
    resolve_plan,
)
from contextmap.runtime.composition import (
    ComposedRuntime,
    FeatureBuildScope,
    RuntimeProvider,
    composable_backends,
    compose,
    compose_executors,
    composed_stages,
    resolve_provider,
)
from contextmap.runtime.errors import (
    BackendConfigurationError,
    BackendRuntimeMissingError,
    BackendUnavailableError,
    CompositionError,
    ProviderConfigurationError,
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
DENSE_FEATURES = "visual_perception.dense_features"
REGION_FEATURES = "visual_perception.region_features"


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


# Mesmas coordenadas que ``runtime_entities.InMemoryGeometrySource`` deriva para os índices
# 0..7 (``(0.1 * i, 0.2 * (i % 3), 0.05 * (i % 5))``): oito pontos bastam para dois ``make_entity``
# de quatro índices cada, e são o único jeito de um ``GeometricMapArtifact`` real concordar, ponto
# a ponto, com a geometria que ``make_entity`` já resume sozinho a partir daquele fixture.
_ALIGNED_MAP_POINTS: tuple[tuple[float, float, float], ...] = tuple(
    (0.1 * index, 0.2 * (index % 3), 0.05 * (index % 5)) for index in range(8)
)


def _write_aligned_geometric_map(output_dir: Path) -> GeometricMapArtifactManifest:
    """Persist a real, tiny ``GeometricMapArtifact`` whose points are ``make_entity``'s own support.

    Uma única captura LiDAR, uma única pose e um extrínseco identidade: o ponto do mapa fica
    numericamente igual ao ponto bruto do "scan" (``P_map = T_map_body · T_body_lidar · P_source``
    com translação zero e rotação identidade nos dois fatores), então os oito pontos persistidos
    aqui caem exatamente sobre a geometria que ``runtime_entities.make_entity`` resume por conta
    própria a partir do seu próprio ``InMemoryGeometrySource`` -- mesmos índices (via
    ``geometry_id_for``, uma função pura de ``map_id`` e índice), mesmas coordenadas. É o que
    permite ``resolved_entity_geometries`` recomputar bounds e frame idênticos aos que a resolução
    persistiu, sem precisar hackear identidades de geometria.

    O ``MapId`` final é o que ``GeometricMapArtifactWriter`` deriva (``sequence_name--run_id``);
    o chamador lê ``manifest.map_id`` e passa exatamente esse valor a ``make_entity``.
    """
    import struct

    from contextmap.geometric_mapping import (
        GeometricMapArtifactWriter,
        GeometricMapRunId,
        MotionCorrectionPolicy,
        ScanDisposition,
        assemble_geometry_inputs,
    )
    from contextmap.ingestion import (
        CalibrationSet,
        FrameId,
        FullSequenceSelection,
        LidarObservation,
        PointFieldDataType,
        PointFieldDescriptor,
        RigidTransform,
        SensorId,
        SequenceArtifactId,
        SequenceSelectionResult,
        SourceObservationId,
        SourceProvenance,
        selection_identity,
    )
    from contextmap.shared import SourceTimestamp
    from contextmap.state_estimation import (
        EstimatorProvenance,
        LookupPolicy,
        PoseEstimate,
        PoseProvenance,
        PoseValidity,
        Trajectory,
        TrajectoryId,
        TrajectoryProvenance,
        calibration_identity,
        pose_estimate_id_for,
    )

    stamp = SourceTimestamp(seconds=0, nanoseconds=0, clock_id="fixture:aligned-map")
    sequence_artifact_id = SequenceArtifactId("aligned-sequence")
    trajectory_id = TrajectoryId("aligned--trajectory")
    calibration = CalibrationSet(
        entries={},
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("body"),
                child_frame=FrameId("lidar"),
                translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
        poses=(
            PoseEstimate(
                estimate_id=pose_estimate_id_for(trajectory_id=trajectory_id, index=0),
                timestamp=stamp,
                parent_frame=FrameId("map"),
                child_frame=FrameId("body"),
                translation_m=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                validity=PoseValidity.VALID,
                provenance=PoseProvenance(source_observation_ids=(SourceObservationId("pose-0"),)),
            ),
        ),
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(backend_id="fixture_estimator", backend_version="0"),
            sequence_artifact_id=sequence_artifact_id,
            selection_id="full-sequence",
            calibration_identity=calibration_identity(calibration),
            code_version="test",
        ),
    )
    # Precisão dupla: ``InMemoryGeometrySource`` (usado por ``make_entity``) guarda coordenadas
    # float64 exatas. Empacotar em float32 arredondaria os pontos do "scan" e o mapa recomputado
    # nunca bateria, bit a bit, com os bounds que a resolução persistiu.
    fields = tuple(
        PointFieldDescriptor(
            name=name, offset_bytes=index * 8, data_type=PointFieldDataType.FLOAT64
        )
        for index, name in enumerate(("x", "y", "z"))
    )
    scan = LidarObservation(
        observation_id=SourceObservationId("scan-aligned-0000"),
        sensor_id=SensorId("velodyne"),
        frame_id=FrameId("lidar"),
        timestamp=stamp,
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/lidar"),
        point_count=len(_ALIGNED_MAP_POINTS),
        point_step_bytes=24,
        fields=fields,
        data=b"".join(struct.pack("<3d", *point) for point in _ALIGNED_MAP_POINTS),
        is_dense=True,
    )
    selection = FullSequenceSelection()
    plan = assemble_geometry_inputs(
        sequence=SequenceSelectionResult(
            sequence_artifact_id=sequence_artifact_id,
            selection=selection,
            selection_id=selection_identity(sequence_artifact_id, selection),
            observations=(scan,),
        ),
        calibration=calibration,
        trajectory=trajectory,
        pose_lookup=LookupPolicy.exact(),
        motion_correction_policy=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.ACCEPT
        ),
        motion_correction=None,
        state_estimation_run_id=None,
    )
    return GeometricMapArtifactWriter(
        output_dir=output_dir,
        sequence_name="aligned-map",
        run_id=GeometricMapRunId("run-0001"),
        run_index=1,
    ).finalize(plan=plan, aggregation=None, code_version="test")


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


class TestResolveProvider:
    """Unit tests for resolving a ``"module:attribute"`` string into a callable."""

    def test_resolves_a_real_importable_target(self) -> None:
        provider = resolve_provider(REGION, "runtime_provider_fixtures:load_region_discovery")

        assert provider is runtime_provider_fixtures.load_region_discovery

    @pytest.mark.parametrize(
        "target",
        ["no-colon-at-all", ":load_region_discovery", "runtime_provider_fixtures:", ":"],
    )
    def test_rejects_a_malformed_target(self, target: str) -> None:
        with pytest.raises(ProviderConfigurationError) as error:
            resolve_provider(REGION, target)

        assert error.value.component_id == REGION
        assert error.value.target == target

    def test_rejects_a_module_that_cannot_be_imported(self) -> None:
        with pytest.raises(ProviderConfigurationError, match="could not be imported"):
            resolve_provider(REGION, "no_such_package_exists_for_contextmap:load")

    def test_rejects_an_attribute_the_module_does_not_have(self) -> None:
        with pytest.raises(ProviderConfigurationError, match="no attribute"):
            resolve_provider(REGION, "runtime_provider_fixtures:does_not_exist")

    def test_rejects_an_attribute_that_is_not_callable(self) -> None:
        with pytest.raises(ProviderConfigurationError, match="not callable"):
            resolve_provider(REGION, "runtime_provider_fixtures:not_callable")


class TestDeclaredProviderTargets:
    """``resources.providers``: a declared target resolved lazily by the composition root.

    This is the config-driven counterpart of ``providers=``: a caller of the installed
    ``contextmap`` binary, which never passes ``providers=`` at all, still gets a real
    backend composed when the effective configuration names a ``"module:attribute"``
    target for it (#507's real, previously unmet acceptance criterion).
    """

    def test_a_declared_target_supplies_the_runtime_with_no_explicit_provider(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }

        composed = compose(
            effective_from(tmp_path, document),
            stages=["visual_perception"],
            providers={INTERPRETER: recorder.provider("qwen")},  # isola o slot sob teste
            module_available=lambda _name: True,
            environ={},
        )

        assert isinstance(composed.region_discovery, Sam3RegionDiscovery)

    def test_ensure_available_skips_the_module_check_for_a_declared_target(
        self, tmp_path: Path
    ) -> None:
        """dense_features' own bundled loader needs torch/transformers/PIL -- but a
        declared provider supplies the runtime instead, so composing it must never demand
        those modules be importable (composition.py's own module docstring)."""
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            DENSE_FEATURES: "runtime_provider_fixtures:load_dense_features",
            REGION_FEATURES: "runtime_provider_fixtures:load_region_features",
        }

        composed = compose(
            effective_from(tmp_path, document),
            stages=["visual_perception"],
            providers=_providers(recorder),  # region_discovery e semantic_interpretation
            module_available=lambda _name: False,  # nada está instalado
            environ={},
        )

        assert composed.dense_features is not None
        scope = FeatureBuildScope(
            run_id=PerceptionRunId("run-0001"),
            feature_stage_id="dense_feature_extraction",
            source_artifact_id="run-0001",
            payload_sink=object(),
            prepared_image_root=tmp_path,
        )
        composed.dense_features(scope)
        assert runtime_provider_fixtures.CALLS[-1][0] == "dense_features"

    def test_an_explicit_provider_wins_over_a_declared_target_and_is_reported(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        explicit = recorder.provider("explicit-sam3")
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }
        overrides: list[str] = []

        composed = compose(
            effective_from(tmp_path, document),
            stages=["visual_perception"],
            providers={REGION: explicit, INTERPRETER: recorder.provider("qwen")},
            module_available=lambda _name: True,
            environ={},
            on_provider_override=overrides.append,
        )

        assert composed.region_discovery is not None
        assert overrides == [REGION]
        # o provider explícito de fato construiu o backend, não o alvo declarado.
        assert recorder.calls[0][0] is not None

    def test_no_override_is_reported_when_only_a_declared_target_exists(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }
        overrides: list[str] = []

        compose(
            effective_from(tmp_path, document),
            stages=["visual_perception"],
            providers={INTERPRETER: recorder.provider("qwen")},
            module_available=lambda _name: True,
            environ={},
            on_provider_override=overrides.append,
        )

        assert overrides == []

    def test_an_unresolvable_declared_target_raises_a_provider_configuration_error(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {REGION: "not-a-valid-target"}

        with pytest.raises(ProviderConfigurationError) as error:
            compose(
                effective_from(tmp_path, document),
                stages=["visual_perception"],
                providers={INTERPRETER: recorder.provider("qwen")},
                module_available=lambda _name: True,
                environ={},
            )

        assert error.value.component_id == REGION

    def test_compose_executors_reports_the_override_through_its_own_callback(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }
        overrides: list[str] = []

        executors = compose_executors(
            effective_from(tmp_path, document),
            providers=_providers(recorder),
            module_available=lambda _name: True,
            environ={},
            on_provider_override=overrides.append,
        )

        assert "visual_perception" in executors
        assert overrides == [REGION]


class TestStagesAndExtensionPoints:
    def test_a_stage_that_is_not_selected_is_not_built(self, tmp_path: Path) -> None:
        document = selected_document()
        document["pipeline"]["stages"]["point_representation"] = False

        composed = _compose(tmp_path, document=document)

        assert composed.point_encoder is None
        assert "point_representation" not in composed.stages

    @pytest.mark.usefixtures("unavailable_future_stage")
    def test_stages_without_an_implemented_capability_are_listed_not_simulated(
        self, tmp_path: Path
    ) -> None:
        composed = _compose(tmp_path)

        assert set(composed.unavailable_stages) == {"scene_graph"}
        assert "milestone" in composed.unavailable_stages["scene_graph"]

    @pytest.mark.usefixtures("unavailable_future_stage")
    def test_asking_explicitly_for_an_unavailable_stage_fails_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(StageUnavailableError, match="scene_graph"):
            _compose(tmp_path, stages=["scene_graph"])

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
            ContextMapExecutor,
            EntityResolutionExecutor,
            GeometricMappingExecutor,
            SemanticFusionExecutor,
            SensorAssociationExecutor,
            SpatialRelationsExecutor,
            StateEstimationExecutor,
        )

        executors = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
        )

        assert set(executors) == {
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "semantic_fusion",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        }
        assert isinstance(executors["state_estimation"], StateEstimationExecutor)
        assert isinstance(executors["geometric_mapping"], GeometricMappingExecutor)
        assert isinstance(executors["sensor_association"], SensorAssociationExecutor)
        assert isinstance(executors["semantic_fusion"], SemanticFusionExecutor)
        assert isinstance(executors["entity_resolution"], EntityResolutionExecutor)
        assert isinstance(executors["spatial_relations"], SpatialRelationsExecutor)
        assert isinstance(executors["context_map"], ContextMapExecutor)
        # Nunca fabricado: capabilities sem executor real continuam ausentes, honestamente.
        # visual_perception fica de fora aqui porque a seleção padrão usa sam3, que não tem
        # loader embutido (precisa de um provider) -- não porque falte um VisualPerceptionExecutor
        # (#507); ver TestComposeVisualPerceptionExecutor para composição bem-sucedida.
        assert "ingestion" not in executors
        assert "visual_perception" not in executors
        assert "point_representation" not in executors
        assert "semantic_mapping" not in executors

    def test_the_trajectory_mode_policy_is_threaded_into_state_estimation(
        self, tmp_path: Path
    ) -> None:
        """Issue #555: the ground-truth opt-in is a runtime policy, never a new
        StateEstimationRequest field -- it only decides how the executor is built."""
        from contextmap.runtime.executors import StateEstimationExecutor

        default_executors = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
        )
        document = selected_document()
        document["policies"] = {"trajectory_mode": "allow_ground_truth"}
        opted_in_executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
        )

        default_state_estimation = default_executors["state_estimation"]
        opted_in_state_estimation = opted_in_executors["state_estimation"]
        assert isinstance(default_state_estimation, StateEstimationExecutor)
        assert isinstance(opted_in_state_estimation, StateEstimationExecutor)
        assert default_state_estimation._allow_ground_truth_trajectory is False
        assert opted_in_state_estimation._allow_ground_truth_trajectory is True

    def test_pose_ingestion_is_never_composed_even_when_enabled(self, tmp_path: Path) -> None:
        """Issue #555: like "ingestion" itself, this stage always needs a caller-supplied
        IngestionRequest (here, a pose file path) -- never something this function can
        derive from configuration alone."""
        document = selected_document()
        document["pipeline"]["stages"]["pose_ingestion"] = True

        executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
        )

        assert "pose_ingestion" not in executors

    def test_semantic_mapping_composes_only_when_map_id_and_code_digest_are_both_given(
        self, tmp_path: Path
    ) -> None:
        """Neither identity is configuration: this function never invents one (issue #177)."""
        from contextmap.runtime.executors import SemanticMappingExecutor
        from contextmap.semantic_mapping import SemanticMapId

        without_identity = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
        )
        assert "semantic_mapping" not in without_identity

        with_identity = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
            semantic_map_id=SemanticMapId("semantic-map--test"),
            code_digest="sha256:" + "cd" * 32,
        )
        assert isinstance(with_identity["semantic_mapping"], SemanticMappingExecutor)

    def test_spatial_relations_executor_reads_geometry_summary_only_from_its_policies(
        self, tmp_path: Path
    ) -> None:
        """P2 provenance finding of the PR #540 review, second round.

        ``SpatialRelationsExecutor`` used to also accept a separate ``geometry_summary``
        keyword, independent of ``policies.geometry_summary`` -- the one persisted in the run's
        own provenance. Two independent values could diverge, making the artifact record a
        different policy than the one that actually produced its evidence. There is now only
        one place to supply it.
        """
        from contextmap.runtime.executors import SpatialRelationsExecutor

        executors = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
        )
        spatial_relations = executors["spatial_relations"]
        assert isinstance(spatial_relations, SpatialRelationsExecutor)

        # Argumento removido: a assinatura não aceita mais um segundo valor independente.
        with pytest.raises(TypeError):
            SpatialRelationsExecutor(policies=None, geometry_summary=object())  # type: ignore[call-arg,arg-type]

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
            "context_map",
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
            "context_map",
        }

    def test_an_unselected_optional_channel_leaves_it_none_never_a_default_policy(
        self, tmp_path: Path
    ) -> None:
        """Every optional Entity Resolution channel and Spatial Relations predicate is absent.

        ``selected_document()`` never selects ``semantic_compatibility``, ``temporal_
        compatibility``, ``appearance``, ``representation``, ``geometric_predicate`` or
        ``contact_predicate``: the stages still compose, with the geometry-only path.
        """
        composed = _compose(tmp_path, document=selected_document())

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
        effective = effective_from(tmp_path, selected_document())
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

    def test_the_spatial_relations_executor_is_the_real_thing_and_runs_against_real_artifacts(
        self, tmp_path: Path
    ) -> None:
        """The composed ``SpatialRelationsExecutor`` reads a real resolution and a real map.

        Two real upstream artifacts, not fakes: an ``EntityResolutionRunArtifact`` produced by
        running the composed ``entity_resolution`` executor itself (the same construction as
        ``test_the_entity_resolution_executor_is_the_real_thing_and_runs_against_a_real_artifact``,
        chained), and a ``GeometricMapArtifact`` written by the capability's own
        ``GeometricMapArtifactWriter`` (``_write_aligned_geometric_map``) whose eight points are,
        coordinate for coordinate, the same support ``make_entity`` already summarized for its two
        entities. That alignment is what lets ``resolved_entity_geometries`` (inside the executor)
        recompute bounds identical to what entity resolution persisted, instead of raising -- so
        candidate generation runs on genuine, map-derived positions, not on a coincidence of
        matching test doubles. The two entities are 0.1 m apart along x (well inside the
        configured ``proximity_radius_m=0.6``), so at least one real, geometry-driven candidate
        is produced; the composed executor writes a real ``SpatialRelationsRunArtifact``, exactly
        as the runtime's own DAG runner would call it.
        """
        import dataclasses

        from runtime_entities import make_entity

        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.composition import compose
        from contextmap.runtime.executors import inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.semantic_fusion import SemanticFusionRunId
        from contextmap.semantic_mapping import (
            GEOMETRY_SUMMARY_ALGORITHM_ID,
            MappingRunLineage,
            SemanticMapId,
            SemanticMappingRunId,
            SemanticMappingRunWriter,
        )
        from contextmap.spatial_relations import RelationState, SpatialRelationsRunReader
        from contextmap.visual_perception import PerceptionRunId

        workspace = tmp_path / "ws"

        # As convenções de eixo do documento estendido declaram "odom" como o frame do mapa; o
        # fixture de geometria (tanto o real quanto o InMemoryGeometrySource de ``make_entity``)
        # está em "map". A escolha do nome do frame é um detalhe de configuração deste teste, não
        # uma invariante do domínio -- então o documento local é ajustado para concordar com a
        # geometria real, em vez de forçar a geometria a se chamar "odom".
        document = selected_document()
        document["components"]["spatial_relations"]["frame_conventions"][
            "map-frame-conventions-v1"
        ]["map_frame"] = "map"
        effective = effective_from(tmp_path, document)
        executors = compose_executors(effective, module_available=lambda _name: True, environ={})

        geometry_dir = workspace / "corridor-02" / "run-0001" / "geometric_mapping"
        geometry_manifest = _write_aligned_geometric_map(geometry_dir)
        geometry_ref = ArtifactRef(
            stage_id="geometric_mapping",
            contract="GeometricMapArtifact",
            artifact_id=str(geometry_manifest.run_id),
            content_hash=inventory_digest(geometry_manifest.file_inventory),
            location=geometry_dir.relative_to(workspace).as_posix(),
        )

        # Duas entidades reais sobre o mesmo mapa, com suportes de geometria disjuntos (índices
        # 0-3 e 4-7) e vizinhos: 0.1 m de vão no eixo x, dentro do proximity_radius_m=0.6
        # configurado, então a geração de candidatos tem um par real para medir.
        semantic_map_id = SemanticMapId("semantic-map-ci")
        entities = [
            make_entity(
                "entity--support-000001",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(0, 1, 2, 3),
            ),
            make_entity(
                "entity--support-000002",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(4, 5, 6, 7),
            ),
        ]
        mapping_dir = workspace / "corridor-02" / "run-0001" / "semantic_mapping"
        mapping_manifest = SemanticMappingRunWriter(
            output_dir=mapping_dir,
            sequence_name="corridor-02",
            run_id=SemanticMappingRunId("run-0001--semantic-mapping"),
            run_index=1,
            semantic_map_id=semantic_map_id,
            lineage=MappingRunLineage(
                sequence_artifact_id="sequence-0001",
                geometric_map_id=geometry_manifest.map_id,
                fusion_run_id=SemanticFusionRunId("fusion-run-0001"),
                fusion_schema_version="0.1.0",
                fusion_artifact_digest="sha256:artifact",
                association_run_ids=("association-run-0001",),
                perception_run_ids=(PerceptionRunId("perception-run-0001"),),
                point_representation_run_ids=(),
            ),
            code_version="test",
            code_digest="sha256:" + "cd" * 32,
        ).write(entities)
        entities_ref = ArtifactRef(
            stage_id="semantic_mapping",
            contract="SemanticEntityArtifact",
            artifact_id=str(mapping_manifest.run_id),
            content_hash=inventory_digest(mapping_manifest.file_inventory),
            location=mapping_dir.relative_to(workspace).as_posix(),
        )

        # Passo 1: entity_resolution real, encadeado, exatamente como o teste irmão o constrói --
        # a forma mais simples de obter um EntityResolutionRunArtifact genuíno para este teste.
        resolution_output_dir = workspace / "corridor-02" / "run-0001" / "entity_resolution"
        resolution_request = StageRequest(
            stage_id="entity_resolution",
            inputs={"entities": (entities_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=resolution_output_dir,
            workspace=workspace,
        )
        resolution_ref = executors["entity_resolution"].execute(resolution_request)
        assert resolution_ref.contract == "EntityResolutionRunArtifact"

        # Passo 2: spatial_relations real, sobre a resolução e o mapa geométrico reais.
        relations_output_dir = workspace / "corridor-02" / "run-0001" / "spatial_relations"
        relations_request = StageRequest(
            stage_id="spatial_relations",
            inputs={"entities": (resolution_ref,), "geometry": (geometry_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=relations_output_dir,
            workspace=workspace,
        )

        ref = executors["spatial_relations"].execute(relations_request)

        assert ref.contract == "SpatialRelationsRunArtifact"
        assert (relations_output_dir / "manifest.json").is_file()

        # A geometria real e as duas entidades vizinhas produziram trabalho genuíno: pelo menos
        # um candidato de relação, ainda sem estado decidido (nenhum avaliador geométrico ou de
        # contato foi selecionado no documento estendido) -- honestamente UNRESOLVED, nunca uma
        # relação fabricada.
        reader = SpatialRelationsRunReader(relations_output_dir)
        relations = tuple(reader.iter_relations())
        assert len(relations) >= 1
        assert all(relation.state is RelationState.UNRESOLVED for relation in relations)

        # Reabre o manifesto e confere a proveniência de ``geometry_summary`` de ponta a ponta:
        # a política persistida é a mesma que a composição de fato construiu a partir do
        # documento estendido, não uma segunda instância que só coincide por acaso.
        composed = compose(
            effective,
            stages=["spatial_relations"],
            module_available=lambda _name: True,
            environ={},
        )
        assert composed.spatial_relations_policies is not None
        expected_policy = composed.spatial_relations_policies.geometry_summary
        assert reader.manifest.policies["geometry_summary"] == {
            "policy_id": GEOMETRY_SUMMARY_ALGORITHM_ID,
            "fingerprint": expected_policy.fingerprint(),
            "parameters": dataclasses.asdict(expected_policy),
        }

    def test_the_context_map_executor_is_the_real_thing_and_assembles_a_real_artifact(
        self, tmp_path: Path
    ) -> None:
        """The composed ``ContextMapExecutor`` assembles a real ``ContextMap`` (issue #177).

        Chains the same real entity_resolution -> spatial_relations construction as
        ``test_the_spatial_relations_executor_is_the_real_thing_and_runs_against_real_artifacts``,
        one hop further, feeding the composed executor the real geometric map, the real
        spatial-relations run it just produced, and a real, minimal sequence artifact whose
        identity matches the one the geometric map was actually built over (issue #438 review:
        ``ContextMapExecutor`` now opens and verifies this, so a stand-in with an unrelated
        identity is no longer accepted).
        """
        import json

        from runtime_entities import make_entity

        from contextmap.ingestion import (
            FrameId,
            ImuObservation,
            SensorId,
            SequenceArtifactWriter,
            SourceObservationId,
            SourceProvenance,
        )
        from contextmap.ingestion.sequence_provenance import SequenceProvenance
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
        from contextmap.shared import SourceTimestamp
        from contextmap.visual_perception import PerceptionRunId

        workspace = tmp_path / "ws"
        document = selected_document()
        document["components"]["spatial_relations"]["frame_conventions"][
            "map-frame-conventions-v1"
        ]["map_frame"] = "map"
        effective = effective_from(tmp_path, document)
        executors = compose_executors(
            effective, module_available=lambda _name: True, environ={}, code_version="test"
        )
        assert "context_map" in executors

        geometry_dir = workspace / "corridor-02" / "run-0001" / "geometric_mapping"
        geometry_manifest = _write_aligned_geometric_map(geometry_dir)
        geometry_ref = ArtifactRef(
            stage_id="geometric_mapping",
            contract="GeometricMapArtifact",
            artifact_id=str(geometry_manifest.map_id),
            content_hash=inventory_digest(geometry_manifest.file_inventory),
            location=geometry_dir.relative_to(workspace).as_posix(),
        )

        semantic_map_id = SemanticMapId("semantic-map-ci")
        fusion_artifact_digest = "sha256:" + "ab" * 32
        entities = [
            make_entity(
                "entity--support-000001",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(0, 1, 2, 3),
                fusion_artifact_digest=fusion_artifact_digest,
            ),
            make_entity(
                "entity--support-000002",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(4, 5, 6, 7),
                fusion_artifact_digest=fusion_artifact_digest,
            ),
        ]
        mapping_dir = workspace / "corridor-02" / "run-0001" / "semantic_mapping"
        mapping_manifest = SemanticMappingRunWriter(
            output_dir=mapping_dir,
            sequence_name="corridor-02",
            run_id=SemanticMappingRunId("run-0001--semantic-mapping"),
            run_index=1,
            semantic_map_id=semantic_map_id,
            lineage=MappingRunLineage(
                sequence_artifact_id="sequence-0001",
                geometric_map_id=geometry_manifest.map_id,
                fusion_run_id=SemanticFusionRunId("fusion-run-0001"),
                fusion_schema_version="0.1.0",
                fusion_artifact_digest=fusion_artifact_digest,
                association_run_ids=("association-run-0001",),
                perception_run_ids=(PerceptionRunId("perception-run-0001"),),
                point_representation_run_ids=(),
            ),
            code_version="test",
            code_digest="sha256:" + "cd" * 32,
        ).write(entities)
        entities_ref = ArtifactRef(
            stage_id="semantic_mapping",
            contract="SemanticEntityArtifact",
            artifact_id=str(mapping_manifest.run_id),
            content_hash=inventory_digest(mapping_manifest.file_inventory),
            location=mapping_dir.relative_to(workspace).as_posix(),
        )

        resolution_output_dir = workspace / "corridor-02" / "run-0001" / "entity_resolution"
        resolution_ref = executors["entity_resolution"].execute(
            StageRequest(
                stage_id="entity_resolution",
                inputs={"entities": (entities_ref,)},
                components={},
                config_digest=effective.digest,
                output_dir=resolution_output_dir,
                workspace=workspace,
            )
        )

        relations_output_dir = workspace / "corridor-02" / "run-0001" / "spatial_relations"
        relations_ref = executors["spatial_relations"].execute(
            StageRequest(
                stage_id="spatial_relations",
                inputs={"entities": (resolution_ref,), "geometry": (geometry_ref,)},
                components={},
                config_digest=effective.digest,
                output_dir=relations_output_dir,
                workspace=workspace,
            )
        )
        # Sequence real, mínima: ContextMapExecutor agora abre este artifact e exige que sua
        # identidade seja exatamente a que o mapa geométrico foi construído sobre
        # (``geometry_manifest.sequence_artifact_id``, "aligned-sequence" -- ver
        # ``_write_aligned_geometric_map``), não um stand-in com identidade arbitrária.
        sequence_dir = workspace / "corridor-02" / "sequence"
        with SequenceArtifactWriter(
            output_dir=sequence_dir,
            sequence_name="corridor-02",
            artifact_id=geometry_manifest.sequence_artifact_id,
        ) as sequence_writer:
            sequence_writer.add_observation(
                ImuObservation(
                    observation_id=SourceObservationId("imu-0000"),
                    sensor_id=SensorId("imu"),
                    frame_id=FrameId("imu"),
                    timestamp=SourceTimestamp(
                        seconds=0, nanoseconds=0, clock_id="fixture:aligned-map"
                    ),
                    provenance=SourceProvenance(source_type="fixture", source_path="fixtures/imu"),
                )
            )
            sequence_writer.set_provenance(
                SequenceProvenance(source_type="fixture", source_path="fixtures/imu")
            )
            sequence_manifest = sequence_writer.finalize()
        sequence_ref = ArtifactRef(
            stage_id="ingestion",
            contract="SequenceArtifact",
            artifact_id=str(sequence_manifest.artifact_id),
            content_hash=inventory_digest(sequence_manifest.file_inventory),
            location=sequence_dir.relative_to(workspace).as_posix(),
        )

        context_map_output_dir = workspace / "corridor-02" / "run-0001" / "context_map"
        context_map_ref = executors["context_map"].execute(
            StageRequest(
                stage_id="context_map",
                inputs={
                    "sequence": (sequence_ref,),
                    "geometry": (geometry_ref,),
                    "entities": (resolution_ref,),
                    "relations": (relations_ref,),
                },
                components={},
                config_digest=effective.digest,
                output_dir=context_map_output_dir,
                workspace=workspace,
            )
        )

        assert context_map_ref.contract == "ContextMapArtifact"
        manifest_record = json.loads(
            (context_map_output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_record["entity_count"] == 2


class TestCompositionFailuresReachPreflight:
    """Issue #602 (RT-02): why a stage could not be composed reaches the preflight report.

    ``compose_executors`` still leaves such a stage out and never raises; the cause it used to
    swallow is now handed to its caller, and ``preflight`` reports it at the component it
    failed on instead of only saying that no executor is registered.
    """

    def _preflight(
        self,
        tmp_path: Path,
        document: dict[str, Any],
        targets: list[str],
        providers: dict[str, RuntimeProvider] | None = None,
    ) -> PreflightReport:
        effective = effective_from(tmp_path, document)
        failures: dict[str, CompositionError | ConfigurationError] = {}
        executors = compose_executors(
            effective,
            providers=providers,
            module_available=lambda _name: True,
            environ={},
            on_composition_failure=failures.__setitem__,
        )
        return preflight(
            resolve_plan(effective).scope(targets=targets),
            executors=executors,
            environ={},
            module_available=lambda _name: True,
            composition_failures=failures,
        )

    def test_a_rejected_backend_parameter_is_reported_at_its_component(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["sensor_association"]["tolerances"]["diagnostic-tolerances-v1"][
            "max_reprojection_invalid_rate"
        ] = 2.0

        report = self._preflight(tmp_path, document, ["sensor_association"])

        paths = [problem.path for problem in report.problems]
        assert "components.sensor_association.tolerances" in paths
        cause = report.problems[paths.index("components.sensor_association.tolerances")]
        assert "sensor_association" in cause.message
        assert "max_reprojection_invalid_rate" in cause.message
        assert "stages.sensor_association" not in paths
        # ingestion nunca é composto aqui e não tem causa capturada: o problema genérico fica.
        generic = ConfigProblem(path="stages.ingestion", message="no executor is registered for it")
        assert generic in report.problems

    def test_an_invalid_provider_target_is_reported_with_the_target_and_the_reason(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["resources"]["providers"] = {REGION: "no-attribute-here"}

        report = self._preflight(
            tmp_path,
            document,
            ["visual_perception"],
            providers={INTERPRETER: _Recorder().provider("qwen")},
        )

        causes = [p for p in report.problems if p.path == f"components.{REGION}"]
        assert len(causes) == 1
        assert "'no-attribute-here'" in causes[0].message
        assert "must look like 'module:attribute'" in causes[0].message
        assert "stages.visual_perception" not in [p.path for p in report.problems]

    def test_a_candidate_reach_short_of_an_evaluator_tolerance_is_reported_at_the_candidate(
        self, tmp_path: Path
    ) -> None:
        # Issue #606: cada política é válida sozinha, mas o alcance de proximidade dos
        # candidatos (0.6 m) não cobre o next_to_max_gap_m do avaliador geométrico.
        document = selected_document()
        document["components"]["spatial_relations"]["geometric_predicate"] = {
            "backend": "bounds-geometric-predicates-v1",
            "bounds-geometric-predicates-v1": {
                "boundary_tolerance_m": 0.02,
                "next_to_max_gap_m": 0.9,
                "adjacent_penetration_m": 0.05,
                "containment_slack_m": 0.05,
                "directional_overlap_fraction": 0.5,
            },
        }

        report = self._preflight(tmp_path, document, ["spatial_relations"])

        paths = [problem.path for problem in report.problems]
        assert "components.spatial_relations.candidate" in paths
        cause = report.problems[paths.index("components.spatial_relations.candidate")]
        assert "next_to_max_gap_m=0.9 exceeds proximity_radius_m=0.6" in cause.message
        assert "stages.spatial_relations" not in paths

    def test_a_missing_model_runtime_is_reported_at_its_component(self, tmp_path: Path) -> None:
        report = self._preflight(tmp_path, selected_document(), ["visual_perception"])

        causes = [p for p in report.problems if p.path == f"components.{REGION}"]
        assert len(causes) == 1
        assert "has no bundled model loader" in causes[0].message

    def test_a_cause_preflight_already_reports_is_not_repeated(self, tmp_path: Path) -> None:
        # Seleção incompleta: o próprio preflight já aponta o componente; o estágio segue
        # reportado como sem executor, sem duplicar a causa.
        document = selected_document()
        del document["components"]["entity_resolution"]["geometry_comparison"]

        report = self._preflight(tmp_path, document, ["entity_resolution"])

        paths = [problem.path for problem in report.problems]
        assert paths.count("components.entity_resolution.geometry_comparison") == 1
        assert "stages.entity_resolution" in paths

    def test_the_cause_is_handed_over_without_costing_another_stage_its_executor(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["sensor_association"]["tolerances"]["diagnostic-tolerances-v1"][
            "max_reprojection_invalid_rate"
        ] = 2.0
        failures: dict[str, CompositionError | ConfigurationError] = {}

        executors = compose_executors(
            effective_from(tmp_path, document),
            providers=_providers(_Recorder()),
            module_available=lambda _name: True,
            environ={},
            on_composition_failure=failures.__setitem__,
        )

        assert set(failures) == {"sensor_association"}
        failure = failures["sensor_association"]
        assert isinstance(failure, BackendConfigurationError)
        assert failure.component_id == "sensor_association.tolerances"
        assert set(executors) == {
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
            "semantic_fusion",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        }


def _perception_request(workspace: Path, *, width: int, height: int) -> Any:
    """Publish a one-frame ``bgr8`` sequence and build the ``visual_perception`` request over it."""
    from contextmap.ingestion import (
        FrameId,
        ImageEncoding,
        ImageObservation,
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

    with SequenceArtifactWriter(
        output_dir=workspace / "corridor-02" / "run-0001" / "ingestion",
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-0001"),
    ) as writer:
        writer.add_observation(
            ImageObservation(
                observation_id=SourceObservationId("frame-0000"),
                sensor_id=SensorId("camera_1"),
                frame_id=FrameId("camera_1_optical"),
                timestamp=SourceTimestamp(seconds=0, nanoseconds=0, clock_id="fixture:header"),
                provenance=SourceProvenance(source_type="fixture", source_path="fixtures/images"),
                width=width,
                height=height,
                encoding=ImageEncoding.BGR8,
                data=bytes([10, 20, 30] * (width * height)),
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
    return StageRequest(
        stage_id="visual_perception",
        inputs={"sequence": (sequence_ref,)},
        components={},
        config_digest="sha256:test",
        output_dir=workspace / "corridor-02" / "run-0001" / "visual_perception",
        workspace=workspace,
    )


class _FakePillowImage:
    """The few Pillow image operations the perception executor performs, over a NumPy array."""

    def __init__(self, pixels: Any) -> None:
        self._pixels = pixels

    def save(self, path: Path) -> None:
        import numpy as np

        with open(path, "wb") as handle:
            np.save(handle, self._pixels)

    def convert(self, mode: str) -> _FakePillowImage:
        return self

    def crop(self, box: tuple[int, int, int, int]) -> _FakePillowImage:
        left, top, right, bottom = box
        return _FakePillowImage(self._pixels[top:bottom, left:right])


@pytest.fixture
def fake_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for ``PIL.Image``, which ``.[dev]`` does not install, so the executor still runs.

    The executor only materializes the prepared image and the semantic views through it; neither
    is what the tests using this fixture examine.
    """
    import sys
    import types

    import numpy as np

    module = types.ModuleType("PIL.Image")
    module.__dict__.update(
        fromarray=lambda pixels, mode=None: _FakePillowImage(pixels),
        open=lambda path: _FakePillowImage(np.load(path)),
    )
    monkeypatch.setitem(sys.modules, "PIL.Image", module)


def _audited_without_rejections(image: Any, regions: list[Any], backend: Any) -> Any:
    """Report hand-built regions as a frame whose discovery rejected and merged nothing."""
    from contextmap.visual_perception import (
        AuditedRegions,
        NormalizationConfig,
        RegionDiscoveryAudit,
    )

    return AuditedRegions(
        regions=tuple(regions),
        audit=RegionDiscoveryAudit(
            source_observation_id=image.source_observation_id,
            backend=backend,
            passes=(),
            diagnostics=(),
            pass_rejections=(),
            normalization_config_digest=NormalizationConfig().digest,
            normalization_rejections=(),
            merge_decisions=(),
        ),
    )


class _NoFeatures:
    """A feature extractor of one scope that extracts nothing: features are not under test."""

    def __init__(self, scope: Any) -> None:
        self._scope = scope

    def backend_provenance(self) -> Any:
        from contextmap.visual_perception import BackendProvenance

        return BackendProvenance(
            backend_id=f"fake_{self._scope.value}_feature_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def required_scope(self) -> Any:
        return self._scope

    def extract(self, image: Any, regions: Any = ()) -> list[Any]:
        return []


class _AbstainingSemanticInterpreter:
    """Answers every real ``interpret()`` request with an abstention: semantics are not tested."""

    def backend_provenance(self) -> Any:
        from contextmap.visual_perception import BackendProvenance

        return BackendProvenance(
            backend_id="fake_semantic_interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def interpret(self, request: Any) -> object:
        import json

        from contextmap.visual_perception import (
            SemanticBackendDiagnostics,
            SemanticConfidencePolicy,
            SemanticInferenceProvenance,
            SemanticInterpretationExecution,
            SemanticPromptTemplate,
            parse_semantic_response,
            render_semantic_prompt,
        )

        raw_response = json.dumps({"abstained": True, "claims": [], "scene_context": None})
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"fake-{request.mode.value}",
            prompt_template_id=request.prompt_template_id,
            output_schema_version=request.requested_output_schema,
        )
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=render_semantic_prompt(
                request,
                SemanticPromptTemplate.default_for(request.mode),
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            ),
            raw_response=raw_response,
            parsed=parse_semantic_response(
                raw_response,
                request,
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            ),
            diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
            effective_configuration={"backend": "fake"},
        )


class TestComposeVisualPerceptionExecutor:
    """#507: ``visual_perception`` gets a real, wired ``VisualPerceptionExecutor``.

    ``sam3`` and ``qwen`` (the default fixture's region discovery and semantic
    interpretation backends) have no bundled model loader, so a real provider for both is
    required for the four variation points to compose successfully -- exactly like
    ``TestCanonicalComposition`` already needs for those two slots individually.
    """

    def test_composes_a_real_executor_when_all_four_slots_are_selected(
        self, tmp_path: Path
    ) -> None:
        from contextmap.runtime.executors import VisualPerceptionExecutor
        from contextmap.visual_perception.backends.dinov3 import DinoV3DenseFeatureBackend

        recorder = _Recorder()
        executors = compose_executors(
            effective_from(tmp_path),
            providers=_providers(recorder),
            module_available=lambda _name: True,
            environ={},
        )

        assert "visual_perception" in executors
        executor = executors["visual_perception"]
        assert isinstance(executor, VisualPerceptionExecutor)
        # Backends de verdade, não um stub type-compatible: region_discovery e
        # semantic_interpreter já vêm prontos; dense_features/region_features são
        # fábricas run-scoped que, quando chamadas, devolvem o adapter real.
        assert isinstance(executor._region_discovery, Sam3RegionDiscovery)  # type: ignore[attr-defined]
        assert isinstance(executor._semantic_interpreter, QwenSemanticInterpreter)  # type: ignore[attr-defined]
        scope = FeatureBuildScope(
            run_id=PerceptionRunId("run-0001"),
            feature_stage_id="dense_feature_extraction",
            source_artifact_id="run-0001",
            payload_sink=object(),
            prepared_image_root=tmp_path,
        )
        assert isinstance(executor._dense_features(scope), DinoV3DenseFeatureBackend)  # type: ignore[attr-defined]
        assert isinstance(executor._region_features(scope), ClipVisualFeatureBackend)  # type: ignore[attr-defined]

    def test_an_incomplete_visual_perception_selection_leaves_it_out_not_fabricated(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        del document["components"]["visual_perception"]["semantic_interpretation"]

        recorder = _Recorder()
        executors = compose_executors(
            effective_from(tmp_path, document),
            providers=_providers(recorder),
            module_available=lambda _name: True,
            environ={},
        )

        assert "visual_perception" not in executors
        # Nenhum estágio irmão paga pelo problema de visual_perception (#507).
        assert "state_estimation" in executors

    def test_composed_executor_actually_runs_and_produces_a_real_perception_run_artifact(
        self, tmp_path: Path
    ) -> None:
        """Prove the executor's own orchestration (#507), not the backends' science.

        Region discovery, feature extraction and semantic interpretation are
        deterministic fakes injected directly (constructing the real ``Sam3``/``DINOv3``/
        ``CLIP``/``Qwen`` backends would need real weights and, for two of them, a real
        model runtime) -- but the sequence, the prepared-image materialization, the
        resolved stage graph, the assembled ``PerceptionResult`` and the written/reopened
        ``PerceptionRunArtifact`` are all real. See the milestone PR for the separate,
        real-backend GPU validation this local test cannot perform.
        """
        pytest.importorskip("PIL")  # Pillow decodes/writes the prepared-image PNG; not a base dep.

        from collections.abc import Sequence

        from contextmap.ingestion import (
            FrameId,
            ImageEncoding,
            ImageObservation,
            SensorId,
            SequenceArtifactId,
            SequenceArtifactWriter,
            SourceObservationId,
            SourceProvenance,
        )
        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.executors import VisualPerceptionExecutor, inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.shared import SourceTimestamp
        from contextmap.visual_perception import (
            BackendProvenance,
            BoundingBox2D,
            FeatureId,
            FeatureScope,
            PerceptionRunReader,
            PreparedImage,
            Region2D,
            RegionId,
            VisualFeature,
        )

        class _FakeRegionDiscovery:
            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id="fake_region_discovery",
                    capability="region_discovery",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def discover(self, image: PreparedImage) -> list[Region2D]:
                return [
                    Region2D(
                        region_id=RegionId("region-0000"),
                        bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
                        provenance=self.backend_provenance(),
                    )
                ]

            def discover_audited(self, image: PreparedImage) -> Any:
                return _audited_without_rejections(
                    image, self.discover(image), self.backend_provenance()
                )

        class _FakeFeatureExtractor:
            def __init__(self, scope: FeatureScope) -> None:
                self._scope = scope

            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id=f"fake_{self._scope.value}_feature_extractor",
                    capability="feature_extractor",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def required_scope(self) -> FeatureScope:
                return self._scope

            def extract(
                self, image: PreparedImage, regions: Sequence[Region2D] = ()
            ) -> Sequence[VisualFeature]:
                if self._scope is FeatureScope.DENSE:
                    return [
                        VisualFeature(
                            feature_id=FeatureId("feature-dense-0000"),
                            scope=FeatureScope.DENSE,
                            embedding_space_id="fake-dense-space",
                            shape=(2,),
                            dtype="float32",
                            payload_reference=f"{image.source_observation_id}-dense.npy",
                            provenance=self.backend_provenance(),
                        )
                    ]
                return [
                    VisualFeature(
                        feature_id=FeatureId(f"feature-{region.region_id}"),
                        scope=FeatureScope.REGION,
                        embedding_space_id="fake-region-space",
                        shape=(2,),
                        dtype="float32",
                        payload_reference=f"{image.source_observation_id}-{region.region_id}.npy",
                        provenance=self.backend_provenance(),
                        region_id=region.region_id,
                    )
                    for region in regions
                ]

        class _FakeSemanticInterpreter:
            """Implements the real ``interpret()`` port, like every real backend does today.

            The executor bridges this to its own legacy ``interpret_scene``/``interpret_regions``
            dispatch (see ``_LegacySemanticInterpreterBridge``); a fake standing in for a real
            backend must match what a real backend implements.
            """

            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id="fake_semantic_interpreter",
                    capability="semantic_interpreter",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def interpret(self, request: Any) -> object:
                import json

                from contextmap.visual_perception import (
                    SemanticBackendDiagnostics,
                    SemanticConfidencePolicy,
                    SemanticInferenceProvenance,
                    SemanticInterpretationExecution,
                    SemanticPromptTemplate,
                    parse_semantic_response,
                    render_semantic_prompt,
                )

                template = SemanticPromptTemplate.default_for(request.mode)  # type: ignore[attr-defined]
                rendered = render_semantic_prompt(
                    request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
                )
                raw_response = json.dumps({"abstained": True, "claims": [], "scene_context": None})
                provenance = SemanticInferenceProvenance(
                    backend=self.backend_provenance(),
                    task_identity=f"fake-{request.mode.value}",  # type: ignore[attr-defined]
                    prompt_template_id=request.prompt_template_id,  # type: ignore[attr-defined]
                    output_schema_version=request.requested_output_schema,  # type: ignore[attr-defined]
                )
                return SemanticInterpretationExecution(
                    request=request,  # type: ignore[arg-type]
                    rendered_prompt=rendered,
                    raw_response=raw_response,
                    parsed=parse_semantic_response(
                        raw_response,
                        request,  # type: ignore[arg-type]
                        provenance,
                        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
                    ),
                    diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
                    effective_configuration={"backend": "fake"},
                )

        executor = VisualPerceptionExecutor(
            region_discovery=_FakeRegionDiscovery(),  # type: ignore[arg-type]
            dense_features=lambda _scope: _FakeFeatureExtractor(FeatureScope.DENSE),  # type: ignore[arg-type]
            region_features=lambda _scope: _FakeFeatureExtractor(FeatureScope.REGION),  # type: ignore[arg-type]
            semantic_interpreter=_FakeSemanticInterpreter(),  # type: ignore[arg-type]
        )

        workspace = tmp_path / "ws"
        ingestion_dir = workspace / "corridor-02" / "run-0001" / "ingestion"
        pixels = bytes([10, 20, 30] * (2 * 2))  # 2x2 bgr8, solid color
        with SequenceArtifactWriter(
            output_dir=ingestion_dir,
            sequence_name="corridor-02",
            artifact_id=SequenceArtifactId("sequence-0001"),
        ) as writer:
            for index in range(2):
                writer.add_observation(
                    ImageObservation(
                        observation_id=SourceObservationId(f"frame-{index:04d}"),
                        sensor_id=SensorId("camera_1"),
                        frame_id=FrameId("camera_1_optical"),
                        timestamp=SourceTimestamp(
                            seconds=index, nanoseconds=0, clock_id="fixture:header"
                        ),
                        provenance=SourceProvenance(
                            source_type="fixture", source_path="fixtures/images"
                        ),
                        width=2,
                        height=2,
                        encoding=ImageEncoding.BGR8,
                        data=pixels,
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
        output_dir = workspace / "corridor-02" / "run-0001" / "visual_perception"
        request = StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (sequence_ref,)},
            components={},
            config_digest="sha256:test",
            output_dir=output_dir,
            workspace=workspace,
        )

        ref = executor.execute(request)

        assert ref.contract == "PerceptionRunArtifact"
        assert (output_dir / "manifest.json").is_file()
        # O diretório de rascunho das imagens preparadas nunca sobrevive à execução.
        assert list(workspace.glob(".tmp-*")) == []

        reader = PerceptionRunReader(output_dir)
        assert reader.verify_integrity() == []
        results = reader.list_results()
        assert {str(result.source_observation_id) for result in results} == {
            "frame-0000",
            "frame-0001",
        }
        for result in results:
            assert len(result.regions) == 1
            assert len(result.features) == 2  # one dense + one region feature

        # The legacy scene/region dispatch (`_LegacySemanticInterpreterBridge`) reduces every
        # real `interpret()` call to the `SceneContext`/`SemanticClaim` the stage graph needs,
        # but the real `SemanticInterpretationExecution` evidence (rendered prompt, raw
        # response, diagnostics) and its view payload must still be recoverable from the
        # finalized artifact -- one execution per image (scene) plus one per region.
        from contextmap.visual_perception.backends._semantic_views import read_view_payload

        semantic_executions = reader.list_semantic_executions()
        assert {
            str(execution.request.source_observation_id) for execution in semantic_executions
        } == {"frame-0000", "frame-0001"}
        assert {execution.request.mode.value for execution in semantic_executions} == {
            "scene",
            "region",
        }
        for execution in semantic_executions:
            for view in execution.request.visual_views:
                assert read_view_payload(output_dir, view) is not None

    def test_a_mask_conditioned_region_features_backend_gets_the_region_own_mask(
        self, tmp_path: Path
    ) -> None:
        """Blocker #2 of the PR #535 review: the executor now supplies a real ``mask_source``.

        ``region_features=alphaclip`` used to compose into an executor that failed
        deterministically the moment any image was processed: ``VisualPerceptionExecutor``
        built its region-features scope with ``mask_source=None``, and AlphaCLIP's own
        composition factory (``composition.py``'s ``_alphaclip``, left unmodified here)
        refuses exactly that. This runs the real ``_alphaclip`` factory through the real
        executor, with a mask-based fake region-discovery backend that attaches an inline
        mask to its one region -- exactly what SAM2/SAM3 do -- and a fake AlphaCLIP model
        runtime standing in for the SDK.
        """
        pytest.importorskip("PIL")

        from collections.abc import Sequence

        import numpy as np

        from contextmap.ingestion import (
            FrameId,
            ImageEncoding,
            ImageObservation,
            SensorId,
            SequenceArtifactId,
            SequenceArtifactWriter,
            SourceObservationId,
            SourceProvenance,
        )
        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.executors import VisualPerceptionExecutor, inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.shared import SourceTimestamp
        from contextmap.visual_perception import (
            BackendProvenance,
            BoundingBox2D,
            FeatureId,
            FeatureScope,
            InlineMask,
            PerceptionRunReader,
            PreparedImage,
            Region2D,
            RegionId,
            VisualFeature,
        )
        from contextmap.visual_perception.backends.alphaclip import AlphaClipNativeOutput

        class _FakeMaskedRegionDiscovery:
            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id="fake_region_discovery",
                    capability="region_discovery",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def discover(self, image: PreparedImage) -> list[Region2D]:
                return [
                    Region2D(
                        region_id=RegionId("region-0000"),
                        bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
                        provenance=self.backend_provenance(),
                        mask_reference="masks/region-0000.npy",
                        mask=InlineMask(
                            np.array((True, True, True, True), dtype=bool).reshape(2, 2)
                        ),
                        image_width=2,
                        image_height=2,
                    )
                ]

            def discover_audited(self, image: PreparedImage) -> Any:
                return _audited_without_rejections(
                    image, self.discover(image), self.backend_provenance()
                )

        class _FakeDenseFeatureExtractor:
            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id="fake_dense_feature_extractor",
                    capability="feature_extractor",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def required_scope(self) -> FeatureScope:
                return FeatureScope.DENSE

            def extract(
                self, image: PreparedImage, regions: Sequence[Region2D] = ()
            ) -> Sequence[VisualFeature]:
                return [
                    VisualFeature(
                        feature_id=FeatureId("feature-dense-0000"),
                        scope=FeatureScope.DENSE,
                        embedding_space_id="fake-dense-space",
                        shape=(2,),
                        dtype="float32",
                        payload_reference=f"{image.source_observation_id}-dense.npy",
                        provenance=self.backend_provenance(),
                    )
                ]

        class _FakeSemanticInterpreter:
            """Implements the real ``interpret()`` port, like every real backend does today.

            The executor bridges this to its own legacy ``interpret_scene``/``interpret_regions``
            dispatch (see ``_LegacySemanticInterpreterBridge``); a fake standing in for a real
            backend must match what a real backend implements.
            """

            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id="fake_semantic_interpreter",
                    capability="semantic_interpreter",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def interpret(self, request: Any) -> object:
                import json

                from contextmap.visual_perception import (
                    SemanticBackendDiagnostics,
                    SemanticConfidencePolicy,
                    SemanticInferenceProvenance,
                    SemanticInterpretationExecution,
                    SemanticPromptTemplate,
                    parse_semantic_response,
                    render_semantic_prompt,
                )

                template = SemanticPromptTemplate.default_for(request.mode)  # type: ignore[attr-defined]
                rendered = render_semantic_prompt(
                    request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
                )
                raw_response = json.dumps({"abstained": True, "claims": [], "scene_context": None})
                provenance = SemanticInferenceProvenance(
                    backend=self.backend_provenance(),
                    task_identity=f"fake-{request.mode.value}",  # type: ignore[attr-defined]
                    prompt_template_id=request.prompt_template_id,  # type: ignore[attr-defined]
                    output_schema_version=request.requested_output_schema,  # type: ignore[attr-defined]
                )
                return SemanticInterpretationExecution(
                    request=request,  # type: ignore[arg-type]
                    rendered_prompt=rendered,
                    raw_response=raw_response,
                    parsed=parse_semantic_response(
                        raw_response,
                        request,  # type: ignore[arg-type]
                        provenance,
                        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
                    ),
                    diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
                    effective_configuration={"backend": "fake"},
                )

        class _FakeAlphaClipRuntime:
            def encode(
                self, image: PreparedImage, requests: Sequence[Any]
            ) -> AlphaClipNativeOutput:
                return AlphaClipNativeOutput(
                    array=np.array([[1.0, 2.0]] * len(requests), dtype=np.float32),
                    elapsed_seconds=0.01,
                    peak_memory_bytes=None,
                )

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
        recorder = _Recorder()
        providers = {
            **_providers(recorder),
            "visual_perception.region_features": lambda _config, _secrets: _FakeAlphaClipRuntime(),
        }
        composed = compose(
            effective_from(tmp_path, document),
            providers=providers,
            module_available=lambda _name: True,
            environ={},
        )
        assert composed.region_features is not None

        executor = VisualPerceptionExecutor(
            region_discovery=_FakeMaskedRegionDiscovery(),  # type: ignore[arg-type]
            dense_features=lambda _scope: _FakeDenseFeatureExtractor(),  # type: ignore[arg-type]
            region_features=composed.region_features,
            semantic_interpreter=_FakeSemanticInterpreter(),  # type: ignore[arg-type]
        )

        workspace = tmp_path / "ws"
        ingestion_dir = workspace / "corridor-02" / "run-0001" / "ingestion"
        pixels = bytes([10, 20, 30] * (2 * 2))  # 2x2 bgr8, solid color
        with SequenceArtifactWriter(
            output_dir=ingestion_dir,
            sequence_name="corridor-02",
            artifact_id=SequenceArtifactId("sequence-0001"),
        ) as writer:
            writer.add_observation(
                ImageObservation(
                    observation_id=SourceObservationId("frame-0000"),
                    sensor_id=SensorId("camera_1"),
                    frame_id=FrameId("camera_1_optical"),
                    timestamp=SourceTimestamp(seconds=0, nanoseconds=0, clock_id="fixture:header"),
                    provenance=SourceProvenance(
                        source_type="fixture", source_path="fixtures/images"
                    ),
                    width=2,
                    height=2,
                    encoding=ImageEncoding.BGR8,
                    data=pixels,
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
        output_dir = workspace / "corridor-02" / "run-0001" / "visual_perception"
        request = StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (sequence_ref,)},
            components={},
            config_digest="sha256:test",
            output_dir=output_dir,
            workspace=workspace,
        )

        ref = executor.execute(request)

        assert ref.contract == "PerceptionRunArtifact"
        reader = PerceptionRunReader(output_dir)
        assert reader.verify_integrity() == []
        results = reader.list_results()
        assert len(results) == 1
        assert len(results[0].regions) == 1
        assert len(results[0].features) == 2  # one dense + one AlphaCLIP region feature

    @pytest.mark.usefixtures("fake_pillow")
    def test_the_run_artifact_keeps_every_discovery_rejection_and_merge(
        self, tmp_path: Path
    ) -> None:
        """#611: what the canonical discovery rejected and merged survives the run.

        The real SAM2 adapter runs over a fake SDK runtime that proposes the same mask twice, so
        normalization merges the second proposal into the first. Only the surviving region used
        to reach the artifact: the rejection and the merge decision were lost with the run.
        """
        import json

        import numpy as np

        from contextmap.ingestion import SourceObservationId
        from contextmap.runtime.executors import VisualPerceptionExecutor
        from contextmap.visual_perception import (
            FeatureScope,
            InlineMask,
            MergeKind,
            NormalizationConfig,
            PerceptionRunReader,
            RejectionReason,
        )
        from contextmap.visual_perception.backends.sam2 import (
            Sam2Config,
            Sam2NativeProposal,
            Sam2RegionDiscovery,
        )
        from contextmap.visual_perception.discovery import DiscoveryInput

        class _SameMaskTwice:
            def predict(
                self, discovery_input: DiscoveryInput, config: Sam2Config
            ) -> tuple[Sam2NativeProposal, ...]:
                width = discovery_input.discovery_pass.input_width
                height = discovery_input.discovery_pass.input_height
                return tuple(
                    Sam2NativeProposal(
                        proposal_id=proposal_id,
                        box=(0.0, 0.0, float(width), float(height)),
                        mask=InlineMask(np.ones((height, width), dtype=bool)),
                        predicted_iou=0.9,
                        stability_score=0.9,
                    )
                    for proposal_id in ("sam2-a", "sam2-b")
                )

        discovery = Sam2RegionDiscovery(
            config=Sam2Config(checkpoint="facebook/sam2-hiera-large"), runtime=_SameMaskTwice()
        )
        executor = VisualPerceptionExecutor(
            region_discovery=discovery,
            dense_features=lambda _scope: _NoFeatures(FeatureScope.DENSE),
            region_features=lambda _scope: _NoFeatures(FeatureScope.REGION),
            semantic_interpreter=_AbstainingSemanticInterpreter(),  # type: ignore[arg-type]
        )
        request = _perception_request(tmp_path / "ws", width=4, height=3)

        executor.execute(request)

        run_dir = request.output_dir
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "outputs/region-discovery-audit.jsonl" in {
            entry["path"] for entry in manifest["file_inventory"]
        }, "the discovery audit never reached the run artifact"
        reader = PerceptionRunReader(run_dir)
        assert reader.verify_integrity() == []
        assert reader.records_region_discovery_audit() is True
        audit = reader.region_discovery_audit(SourceObservationId("frame-0000"))
        assert audit.backend == discovery.backend_provenance()
        assert [
            (rejection.candidate_id, rejection.reason, rejection.detail)
            for rejection in audit.normalization_rejections
        ] == [
            (
                "full-frame/sam2-b",
                RejectionReason.MERGED_DUPLICATE,
                "merged into full-frame/sam2-a",
            )
        ]
        assert [
            (decision.representative_candidate_id, decision.merged_candidate_id, decision.kind)
            for decision in audit.merge_decisions
        ] == [("full-frame/sam2-a", "full-frame/sam2-b", MergeKind.IOU_DUPLICATE)]
        assert audit.normalization_config_digest == NormalizationConfig().digest
        assert [discovery_pass.pass_id for discovery_pass in audit.passes] == ["full-frame"]
        assert [diagnostics.proposal_count for diagnostics in audit.diagnostics] == [2]
        assert audit.pass_rejections == ()
        # A região sobrevivente nomeia os dois contribuintes: auditoria e resultado se reconciliam.
        (region,) = reader.result(SourceObservationId("frame-0000")).regions
        assert region.contributor_candidate_ids == ("full-frame/sam2-a", "full-frame/sam2-b")

    def test_a_region_discovery_backend_that_cannot_report_its_audit_is_refused(self) -> None:
        """#611: the canonical path never runs a discovery whose rejections it cannot persist."""
        from contextmap.runtime.executors import VisualPerceptionExecutor
        from contextmap.visual_perception import BackendProvenance, FeatureScope

        class _RegionsOnly:
            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id="fake_region_discovery",
                    capability="region_discovery",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def discover(self, image: Any) -> list[Any]:
                return []

        with pytest.raises(TypeError, match="discover_audited"):
            VisualPerceptionExecutor(
                region_discovery=_RegionsOnly(),  # type: ignore[arg-type]
                dense_features=lambda _scope: _NoFeatures(FeatureScope.DENSE),
                region_features=lambda _scope: _NoFeatures(FeatureScope.REGION),
                semantic_interpreter=_AbstainingSemanticInterpreter(),  # type: ignore[arg-type]
            )


def test_composition_does_not_load_a_model_or_import_a_heavy_sdk(tmp_path: Path) -> None:
    import sys

    before = {name for name in ("torch", "transformers", "rosbags") if name in sys.modules}

    _compose(tmp_path)

    after = {name for name in ("torch", "transformers") if name in sys.modules}
    assert after <= before


class TestSemanticBridgeStreamsItsEvidence:
    """PR #438 review: the bridge must not hold view payloads until the image loop ends."""

    @staticmethod
    def _fake_interpreter() -> Any:
        from contextmap.visual_perception import BackendProvenance

        class _Fake:
            def backend_provenance(self) -> BackendProvenance:
                return BackendProvenance(
                    backend_id="fake_semantic_interpreter",
                    capability="semantic_interpreter",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def interpret(self, request: Any) -> object:
                import json

                from contextmap.visual_perception import (
                    SemanticBackendDiagnostics,
                    SemanticConfidencePolicy,
                    SemanticInferenceProvenance,
                    SemanticInterpretationExecution,
                    SemanticPromptTemplate,
                    parse_semantic_response,
                    render_semantic_prompt,
                )

                template = SemanticPromptTemplate.default_for(request.mode)
                rendered = render_semantic_prompt(
                    request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
                )
                raw = json.dumps({"abstained": True, "claims": [], "scene_context": None})
                provenance = SemanticInferenceProvenance(
                    backend=self.backend_provenance(),
                    task_identity=f"fake-{request.mode.value}",
                    prompt_template_id=request.prompt_template_id,
                    output_schema_version=request.requested_output_schema,
                )
                return SemanticInterpretationExecution(
                    request=request,
                    rendered_prompt=rendered,
                    raw_response=raw,
                    parsed=parse_semantic_response(
                        raw,
                        request,
                        provenance,
                        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
                    ),
                    diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
                    effective_configuration={"backend": "fake"},
                )

        return _Fake()

    @staticmethod
    def _prepared_image(view_root: Path, observation_id: str) -> Any:
        # Pillow is not a dependency of this project (the bridge itself reaches it through
        # importlib for that reason), so the test skips instead of failing where it is absent.
        image_module = pytest.importorskip("PIL.Image")

        from contextmap.ingestion import SourceObservationId
        from contextmap.visual_perception import PreparedImage

        prepared_dir = view_root / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        image_module.new("RGB", (16, 12), (10, 20, 30)).save(prepared_dir / f"{observation_id}.png")
        return PreparedImage(
            source_observation_id=SourceObservationId(observation_id),
            payload_reference=f"prepared/{observation_id}.png",
            width=16,
            height=12,
        )

    def test_each_execution_reaches_the_writer_before_the_next_frame(self, tmp_path: Path) -> None:
        """_write_view() read the bytes back and kept them in self._evidence for the whole run.

        That is O(semantic requests x image size) resident, which undoes the writer's streaming.
        """
        from contextmap.runtime.executors import _LegacySemanticInterpreterBridge

        class _RecordingWriter:
            def __init__(self) -> None:
                self.outcomes: list[object] = []
                self.views: list[tuple[str, int]] = []

            def add_stage_outcomes(self, outcomes: Any) -> None:
                self.outcomes.extend(outcomes)

            def add_semantic_view_payload(self, view: Any, payload: bytes) -> None:
                self.views.append((view.payload_reference, len(payload)))

        writer = _RecordingWriter()
        bridge = _LegacySemanticInterpreterBridge(
            interpreter=self._fake_interpreter(),
            run_id=PerceptionRunId("run-0001"),
            view_root=tmp_path,
        )
        bridge.bind(writer)  # type: ignore[arg-type]

        bridge.interpret_scene(self._prepared_image(tmp_path, "frame-0000"))

        assert writer.views, "the view payload must reach the writer as soon as interpret() returns"
        assert writer.outcomes, "the execution must reach the writer as soon as interpret() returns"
        assert not getattr(bridge, "_evidence", []), "the bridge must retain no payload"

    def test_a_rejected_response_is_recorded_in_the_artifact_not_lost(self, tmp_path: Path) -> None:
        """The stage still fails, but the model's real answer reaches the failures stream."""
        from contextmap.runtime.executors import _LegacySemanticInterpreterBridge
        from contextmap.visual_perception import (
            SemanticBackendDiagnostics,
            SemanticConfidencePolicy,
            SemanticInferenceProvenance,
            SemanticPromptTemplate,
            render_semantic_prompt,
            semantic_failure_from_parse_error,
        )
        from contextmap.visual_perception.semantic_prompt import SemanticResponseParseError

        raw = '{"abstained": false, "claims": [{"hypothesis": "a door"}]}'

        class _RejectingInterpreter:
            def backend_provenance(self) -> Any:
                from contextmap.visual_perception import BackendProvenance

                return BackendProvenance(
                    backend_id="fake_semantic_interpreter",
                    capability="semantic_interpreter",
                    provider="fake",
                    model="fake",
                    version="0.1",
                )

            def interpret(self, request: Any) -> object:
                template = SemanticPromptTemplate.default_for(request.mode)
                rendered = render_semantic_prompt(
                    request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
                )
                raise semantic_failure_from_parse_error(
                    SemanticResponseParseError(
                        "claim[0] is missing required fields: ['confidence']", raw_response=raw
                    ),
                    request=request,
                    rendered_prompt=rendered,
                    raw_response=raw,
                    provenance=SemanticInferenceProvenance(
                        backend=self.backend_provenance(),
                        task_identity=f"fake-{request.mode.value}",
                        prompt_template_id=request.prompt_template_id,
                        output_schema_version=request.requested_output_schema,
                    ),
                    diagnostics=SemanticBackendDiagnostics(latency_ms=3.0),
                    effective_configuration={"backend": "fake"},
                )

        recorded: list[Any] = []
        recorded_views: list[tuple[str, int]] = []

        class _RecordingWriter:
            def add_stage_outcomes(self, outcomes: Any) -> None: ...

            def add_semantic_view_payload(self, view: Any, payload: bytes) -> None:
                recorded_views.append((view.payload_reference, len(payload)))

            def add_failed_semantic_interpretation(self, failed: Any) -> None:
                recorded.append(failed)

        bridge = _LegacySemanticInterpreterBridge(
            interpreter=_RejectingInterpreter(),  # type: ignore[arg-type]
            run_id=PerceptionRunId("run-0001"),
            view_root=tmp_path,
        )
        bridge.bind(_RecordingWriter())  # type: ignore[arg-type]

        with pytest.raises(Exception, match="confidence"):
            bridge.interpret_scene(self._prepared_image(tmp_path, "frame-0000"))

        assert len(recorded) == 1, "the rejected response must be recorded before the stage fails"
        assert recorded[0].raw_response == raw
        assert recorded[0].failure.kind == "SemanticResponseParseError"
        # The view that produced the rejected response must reach the artifact too, otherwise
        # it stays in the scratch the executor deletes and the reference dangles.
        assert [reference for reference, _ in recorded_views] == [
            view.payload_reference for view in recorded[0].request.visual_views
        ]
        assert all(size > 0 for _, size in recorded_views)
