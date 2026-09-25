"""Contract tests for the versioned runtime configuration and effective-config resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextmap.runtime import (
    CANONICAL_PROFILE_ID,
    CONFIG_SCHEMA_VERSION,
    ConfigurationError,
    ConfigurationSource,
    EffectiveConfig,
    check_availability,
    check_selection,
    parse_override,
    read_effective_config,
    resolve_effective_config,
    resolve_secrets,
    write_effective_config,
)

REGION = "visual_perception.region_discovery"
DENSE = "visual_perception.dense_features"
ESTIMATOR = "state_estimation.estimator"
INTERPRETER = "visual_perception.semantic_interpretation"
ENCODER = "point_representation.encoder"


def _write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _sam3_document(**parameters: object) -> dict[str, object]:
    return {
        "components": {
            "visual_perception": {
                "region_discovery": {
                    "backend": "sam3",
                    "sam3": {"checkpoint": "sam3-x", **parameters},
                }
            }
        }
    }


class TestCanonicalProfile:
    def test_resolves_deterministically(self) -> None:
        first = resolve_effective_config()
        second = resolve_effective_config()

        assert first.digest == second.digest
        assert first.config == second.config
        assert first.config.pipeline.preset == CANONICAL_PROFILE_ID
        assert first.config.schema_version == CONFIG_SCHEMA_VERSION

    def test_lists_every_stage_with_its_resolved_state(self) -> None:
        stages = resolve_effective_config().config.pipeline.stages

        assert stages["ingestion"] is True
        assert stages["semantic_fusion"] is True
        assert stages["point_representation"] is False
        assert stages["semantic_mapping"] is True
        assert stages["entity_resolution"] is True
        assert stages["spatial_relations"] is True
        assert stages["context_map"] is True

    def test_makes_no_backend_choice_on_the_users_behalf(self) -> None:
        config = resolve_effective_config().config

        assert config.components[REGION].backend is None
        assert config.components[ESTIMATOR].backend is None

    def test_unknown_profile_is_rejected_with_the_known_ones(self) -> None:
        with pytest.raises(ConfigurationError, match="canonical/1"):
            resolve_effective_config(profile="canonical/99")


class TestPrecedence:
    def test_file_overrides_the_profile_and_a_later_file_overrides_an_earlier_one(
        self, tmp_path: Path
    ) -> None:
        first = _write(tmp_path / "a.json", {"resources": {"device": "cuda"}})
        second = _write(tmp_path / "b.json", {"resources": {"device": "cpu"}})

        assert resolve_effective_config(files=[first]).config.resources.device == "cuda"
        assert resolve_effective_config(files=[first, second]).config.resources.device == "cpu"

    def test_override_wins_over_every_file(self, tmp_path: Path) -> None:
        file = _write(tmp_path / "a.json", {"resources": {"device": "cuda"}})

        effective = resolve_effective_config(files=[file], overrides=["resources.device=cpu"])

        assert effective.config.resources.device == "cpu"

    def test_nested_mappings_merge_and_scalars_replace(self, tmp_path: Path) -> None:
        first = _write(tmp_path / "a.json", _sam3_document(score_threshold=0.1, prompt="a"))
        second = _write(
            tmp_path / "b.json",
            {"components": {"visual_perception": {"region_discovery": {"sam3": {"prompt": "b"}}}}},
        )

        parameters = (
            resolve_effective_config(files=[first, second]).config.components[REGION].parameters
        )

        assert parameters["checkpoint"] == "sam3-x"
        assert parameters["score_threshold"] == 0.1
        assert parameters["prompt"] == "b"

    def test_sources_record_each_layer_in_order_without_override_values(
        self, tmp_path: Path
    ) -> None:
        file = _write(tmp_path / "a.json", {"resources": {"device": "cuda"}})

        effective = resolve_effective_config(files=[file], overrides=["resources.workspace=/w"])

        kinds = [source.kind for source in effective.sources]
        assert kinds == ["profile", "file", "override"]
        assert effective.sources[0].identity == CANONICAL_PROFILE_ID
        assert effective.sources[1].identity == str(file)
        assert effective.sources[1].content_hash is not None
        assert effective.sources[2].identity == "resources.workspace"
        assert "/w" not in json.dumps([source.identity for source in effective.sources])

    @pytest.mark.parametrize(
        ("text", "path", "value"),
        [
            ("resources.device=cpu", ["resources", "device"], "cpu"),
            ("resources.device=null", ["resources", "device"], None),
            ("a.b=12", ["a", "b"], 12),
            ("a.b=true", ["a", "b"], True),
            ("a.b=[1,2]", ["a", "b"], [1, 2]),
            ("a.b=/plain/path", ["a", "b"], "/plain/path"),
        ],
    )
    def test_parse_override_reads_json_literals_and_falls_back_to_text(
        self, text: str, path: list[str], value: object
    ) -> None:
        assert parse_override(text) == (tuple(path), value)

    @pytest.mark.parametrize("text", ["no-equals", "=value", ".a=1", "a..b=1"])
    def test_parse_override_rejects_malformed_text(self, text: str) -> None:
        with pytest.raises(ConfigurationError):
            parse_override(text)


class TestDigest:
    def test_is_independent_of_key_order_and_file_format(self, tmp_path: Path) -> None:
        as_json = _write(tmp_path / "a.json", {"resources": {"device": "cpu", "workspace": "w"}})
        as_toml = tmp_path / "a.toml"
        as_toml.write_text('[resources]\nworkspace = "w"\ndevice = "cpu"\n', encoding="utf-8")

        assert (
            resolve_effective_config(files=[as_json]).digest
            == resolve_effective_config(files=[as_toml]).digest
        )

    def test_changes_when_an_effective_value_changes(self) -> None:
        base = resolve_effective_config().digest

        changed = resolve_effective_config(overrides=["policies.debug_level=full"]).digest

        assert changed != base
        assert changed.startswith("sha256:")

    def test_trajectory_mode_defaults_to_operational_only_and_can_be_overridden(self) -> None:
        """Issue #555: default behavior with a ground-truth auxiliary pose is unchanged
        unless this opt-in is set explicitly."""
        default = resolve_effective_config().config.policies.trajectory_mode
        overridden = resolve_effective_config(
            overrides=["policies.trajectory_mode=allow_ground_truth"]
        ).config.policies.trajectory_mode

        assert default == "operational_only"
        assert overridden == "allow_ground_truth"

    def test_ignores_configuration_that_does_not_reach_the_execution(self, tmp_path: Path) -> None:
        selected = _sam3_document()
        with_unused = _sam3_document()
        region = with_unused["components"]["visual_perception"]["region_discovery"]  # type: ignore[index]
        region["sam2"] = {"checkpoint": "never-used"}
        first = _write(tmp_path / "a.json", selected)
        second = _write(tmp_path / "b.json", with_unused)

        assert (
            resolve_effective_config(files=[first]).digest
            == resolve_effective_config(files=[second]).digest
        )

    def test_a_disabled_stage_contributes_nothing(self, tmp_path: Path) -> None:
        document = {
            "components": {
                "point_representation": {
                    "encoder": {"backend": "geometric_descriptor", "geometric_descriptor": {}}
                }
            }
        }
        file = _write(tmp_path / "a.json", document)

        effective = resolve_effective_config(files=[file])

        assert effective.config.pipeline.stages["point_representation"] is False
        assert ENCODER not in effective.config.components
        assert effective.digest == resolve_effective_config().digest


class TestBackendScoping:
    def test_keeps_only_the_selected_backends_parameters(self, tmp_path: Path) -> None:
        document = _sam3_document()
        region = document["components"]["visual_perception"]["region_discovery"]  # type: ignore[index]
        region["florence2"] = {"checkpoint": "other"}
        file = _write(tmp_path / "a.json", document)

        component = resolve_effective_config(files=[file]).config.components[REGION]

        assert component.backend == "sam3"
        assert dict(component.parameters) == {"checkpoint": "sam3-x"}

    def test_switching_backend_by_override_uses_that_backends_block(self, tmp_path: Path) -> None:
        document = _sam3_document()
        region = document["components"]["visual_perception"]["region_discovery"]  # type: ignore[index]
        region["florence2"] = {"checkpoint": "flo"}
        file = _write(tmp_path / "a.json", document)

        component = resolve_effective_config(
            files=[file],
            overrides=["components.visual_perception.region_discovery.backend=florence2"],
        ).config.components[REGION]

        assert component.backend == "florence2"
        assert dict(component.parameters) == {"checkpoint": "flo"}

    def test_device_setting_reaches_only_backends_that_declare_a_device(
        self, tmp_path: Path
    ) -> None:
        document = _sam3_document()
        document["resources"] = {"device": "cuda"}
        document["components"]["state_estimation"] = {  # type: ignore[index]
            "estimator": {"backend": "external_pose", "external_pose": {}}
        }
        file = _write(tmp_path / "a.json", document)

        components = resolve_effective_config(files=[file]).config.components

        assert components[REGION].parameters["device"] == "cuda"
        assert "device" not in components[ESTIMATOR].parameters

    def test_an_explicit_backend_device_wins_over_the_resource_setting(
        self, tmp_path: Path
    ) -> None:
        document = _sam3_document(device="cpu")
        document["resources"] = {"device": "cuda"}
        file = _write(tmp_path / "a.json", document)

        component = resolve_effective_config(files=[file]).config.components[REGION]

        assert component.parameters["device"] == "cpu"


class TestResourceProviders:
    """``resources.providers``: declarative ``RuntimeProvider`` targets (#507's real gap)."""

    def test_parses_a_declared_target_per_component(self, tmp_path: Path) -> None:
        document = _sam3_document()
        document["resources"] = {
            "providers": {REGION: "pkg.loaders:load_sam3", INTERPRETER: "pkg.loaders:load_qwen"}
        }
        file = _write(tmp_path / "a.json", document)

        resources = resolve_effective_config(files=[file]).config.resources

        assert dict(resources.providers) == {
            REGION: "pkg.loaders:load_sam3",
            INTERPRETER: "pkg.loaders:load_qwen",
        }

    def test_defaults_to_an_empty_mapping(self) -> None:
        assert dict(resolve_effective_config().config.resources.providers) == {}

    def test_rejects_a_non_string_or_empty_target(self, tmp_path: Path) -> None:
        document = {"resources": {"providers": {REGION: "", INTERPRETER: 3}}}
        file = _write(tmp_path / "a.json", document)

        with pytest.raises(ConfigurationError) as excinfo:
            resolve_effective_config(files=[file])

        paths = {problem.path for problem in excinfo.value.problems}
        assert paths == {f"resources.providers.{REGION}", f"resources.providers.{INTERPRETER}"}

    def test_a_later_file_adds_to_the_declared_targets_without_wiping_earlier_ones(
        self, tmp_path: Path
    ) -> None:
        first = _write(tmp_path / "a.json", {"resources": {"providers": {REGION: "pkg:a"}}})
        second = _write(tmp_path / "b.json", {"resources": {"providers": {INTERPRETER: "pkg:b"}}})

        resources = resolve_effective_config(files=[first, second]).config.resources

        assert dict(resources.providers) == {REGION: "pkg:a", INTERPRETER: "pkg:b"}

    def test_an_override_replaces_the_whole_declared_providers_mapping(
        self, tmp_path: Path
    ) -> None:
        # component_id já contém um ponto ("visual_perception.region_discovery"), então um
        # override pontual não consegue nomear uma única chave sem ambiguidade com o próprio
        # separador de caminho: o override substitui o mapa inteiro, como em qualquer outro
        # valor JSON (a fusão chave a chave é exclusiva de camadas de arquivo).
        file = _write(tmp_path / "a.json", {"resources": {"providers": {REGION: "pkg:a"}}})
        replacement = json.dumps({REGION: "pkg:b"})

        resources = resolve_effective_config(
            files=[file], overrides=[f"resources.providers={replacement}"]
        ).config.resources

        assert dict(resources.providers) == {REGION: "pkg:b"}

    def test_round_trips_through_to_document(self) -> None:
        target = {"visual_perception.region_discovery": "pkg:load"}

        config = resolve_effective_config(
            overrides=[f"resources.providers={json.dumps(target)}"]
        ).config

        assert config.to_document()["resources"]["providers"] == target

    def test_changes_the_digest(self) -> None:
        base = resolve_effective_config().digest

        changed = resolve_effective_config(
            overrides=[f"resources.providers={json.dumps({REGION: 'pkg:a'})}"]
        ).digest

        assert changed != base


class TestStructuralValidation:
    @pytest.mark.parametrize(
        ("document", "fragment"),
        [
            ({"unknown_section": {}}, "unknown_section"),
            ({"schema_version": "9.9.9"}, "schema_version"),
            ({"pipeline": {"preset": "nope/1"}}, "nope/1"),
            ({"pipeline": {"stages": {"no_such_stage": True}}}, "no_such_stage"),
            ({"pipeline": {"stages": {"ingestion": False}}}, "ingestion"),
            ({"components": {"no_capability": {"x": {}}}}, "no_capability"),
            ({"components": {"visual_perception": {"no_slot": {}}}}, "no_slot"),
            (
                {"components": {"visual_perception": {"region_discovery": {"backend": "nope"}}}},
                "sam3",
            ),
            (
                {"components": {"visual_perception": {"region_discovery": {"nope": {}}}}},
                "nope",
            ),
            ({"resources": {"device": 3}}, "resources.device"),
            ({"policies": {"debug_level": "loud"}}, "debug_level"),
            ({"policies": {"trajectory_mode": "always_ground_truth"}}, "trajectory_mode"),
            ({"inputs": {"selections": {"ingestion": 3}}}, "inputs.selections"),
        ],
    )
    def test_rejects_invalid_documents_naming_the_offending_path(
        self, tmp_path: Path, document: dict[str, object], fragment: str
    ) -> None:
        file = _write(tmp_path / "a.json", document)

        with pytest.raises(ConfigurationError, match=fragment):
            resolve_effective_config(files=[file])

    def test_reports_every_problem_at_once(self, tmp_path: Path) -> None:
        file = _write(tmp_path / "a.json", {"resources": {"device": 1}, "policies": {"x": 1}})

        with pytest.raises(ConfigurationError) as error:
            resolve_effective_config(files=[file])

        assert len(error.value.problems) >= 2

    def test_rejects_non_finite_and_non_json_parameter_values(self, tmp_path: Path) -> None:
        bad = tmp_path / "a.toml"
        bad.write_text(
            '[components.visual_perception.region_discovery]\nbackend = "sam3"\n'
            "[components.visual_perception.region_discovery.sam3]\nwhen = 1979-05-27\n",
            encoding="utf-8",
        )

        with pytest.raises(ConfigurationError, match="when"):
            resolve_effective_config(files=[bad])

    @pytest.mark.parametrize("name", ["api_key", "GEMINI_API_KEY", "password", "authToken"])
    def test_rejects_secret_looking_parameters(self, tmp_path: Path, name: str) -> None:
        file = _write(tmp_path / "a.json", _sam3_document(**{name: "hunter2"}))

        with pytest.raises(ConfigurationError, match="secret"):
            resolve_effective_config(files=[file])

    @pytest.mark.parametrize("name", ["max_new_tokens", "monkey", "keyframe_stride"])
    def test_does_not_mistake_ordinary_parameters_for_secrets(
        self, tmp_path: Path, name: str
    ) -> None:
        file = _write(tmp_path / "a.json", _sam3_document(**{name: 1}))

        assert (
            resolve_effective_config(files=[file]).config.components[REGION].parameters[name] == 1
        )

    def test_rejects_an_unreadable_or_unsupported_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match=r"missing\.json"):
            resolve_effective_config(files=[tmp_path / "missing.json"])
        yaml = tmp_path / "a.yaml"
        yaml.write_text("a: 1", encoding="utf-8")
        with pytest.raises(ConfigurationError, match=r"a\.yaml"):
            resolve_effective_config(files=[yaml])
        broken = tmp_path / "broken.json"
        broken.write_text("{", encoding="utf-8")
        with pytest.raises(ConfigurationError, match=r"broken\.json"):
            resolve_effective_config(files=[broken])
        root = _write(tmp_path / "list.json", [1])
        with pytest.raises(ConfigurationError, match=r"list\.json"):
            resolve_effective_config(files=[root])


class TestIncompatibleCombinations:
    def test_fusion_channel_over_3d_structure_needs_the_point_representation_stage(
        self, tmp_path: Path
    ) -> None:
        document = {
            "components": {
                "semantic_fusion": {
                    "accumulation": {
                        "backend": "baseline-evidence-accumulation-v1",
                        "baseline-evidence-accumulation-v1": {
                            "channels": ["semantic_claims", "point_representation"]
                        },
                    }
                }
            }
        }
        file = _write(tmp_path / "a.json", document)

        with pytest.raises(ConfigurationError, match="point_representation"):
            resolve_effective_config(files=[file])

        enabled = resolve_effective_config(
            files=[file], overrides=["pipeline.stages.point_representation=true"]
        )
        assert enabled.config.pipeline.stages["point_representation"] is True


class TestQualityAwareChannels:
    def test_the_quality_aware_policy_declares_its_channels_under_baseline(
        self, tmp_path: Path
    ) -> None:
        document = {
            "components": {
                "semantic_fusion": {
                    "accumulation": {
                        "backend": "quality-aware-evidence-accumulation-v1",
                        "quality-aware-evidence-accumulation-v1": {
                            "baseline": {"channels": ["semantic_claims", "point_representation"]}
                        },
                    }
                }
            }
        }
        file = _write(tmp_path / "a.json", document)

        with pytest.raises(ConfigurationError, match="point_representation"):
            resolve_effective_config(files=[file])


class TestSelectionCompleteness:
    def test_canonical_profile_is_incomplete_until_backends_are_selected(self) -> None:
        problems = check_selection(resolve_effective_config().config)

        paths = {problem.path for problem in problems}
        assert f"components.{REGION}" in paths
        assert f"components.{ESTIMATOR}" in paths
        assert ENCODER not in {path.removeprefix("components.") for path in paths}
        assert any("sam3" in problem.message for problem in problems)

    def test_a_fully_selected_configuration_has_no_problems(self, tmp_path: Path) -> None:
        file = _write(tmp_path / "a.json", _all_selected())

        effective = resolve_effective_config(files=[file])

        assert check_selection(effective.config) == ()

    def test_an_unselected_optional_evidence_channel_is_never_reported_as_incomplete(
        self, tmp_path: Path
    ) -> None:
        """The geometry-only path is a complete selection, not an incomplete one.

        ``_all_selected()`` never chooses a backend for the optional Entity Resolution channels
        or Spatial Relations predicates: their absence must not surface as a problem.
        """
        file = _write(tmp_path / "a.json", _all_selected())

        effective = resolve_effective_config(files=[file])

        paths = {
            problem.path.removeprefix("components.")
            for problem in check_selection(effective.config)
        }
        assert not paths & {
            "entity_resolution.semantic_compatibility",
            "entity_resolution.temporal_compatibility",
            "entity_resolution.appearance",
            "entity_resolution.representation",
            "spatial_relations.geometric_predicate",
            "spatial_relations.contact_predicate",
        }


def _all_selected() -> dict[str, object]:
    return {
        "components": {
            "ingestion": {"source_adapter": {"backend": "ros1_bag"}},
            "visual_perception": {
                "region_discovery": {"backend": "sam3"},
                "dense_features": {"backend": "dinov3"},
                "region_features": {"backend": "alphaclip"},
                "semantic_interpretation": {"backend": "qwen"},
            },
            "state_estimation": {"estimator": {"backend": "external_pose"}},
            "geometric_mapping": {
                "pose_lookup": {"backend": "lookup-policy-v1"},
                "motion_correction": {"backend": "motion-correction-v1"},
            },
            "sensor_association": {
                "occlusion": {"backend": "conservative-depth-support-v1"},
                "tolerances": {"backend": "diagnostic-tolerances-v1"},
                "pose_policy": {"backend": "lookup-policy-v1"},
            },
            "semantic_fusion": {
                "support": {"backend": "geometry-jaccard-support-v1"},
                "accumulation": {"backend": "baseline-evidence-accumulation-v1"},
            },
            "semantic_mapping": {"geometry_summary": {"backend": "entity-geometry-summary-v1"}},
            "entity_resolution": {
                "retrieval": {"backend": "entity-candidate-retrieval-v1"},
                "resolution": {"backend": "conservative-staged-resolution-v1"},
                "geometry_comparison": {"backend": "entity-geometry-comparison-v1"},
            },
            "spatial_relations": {
                "frame_conventions": {"backend": "map-frame-conventions-v1"},
                "candidate": {"backend": "bounds-neighborhood-candidates-v1"},
                "geometry_summary": {"backend": "entity-geometry-summary-v1"},
            },
        }
    }


class TestAvailability:
    def _effective(self, tmp_path: Path) -> EffectiveConfig:
        return resolve_effective_config(files=[_write(tmp_path / "a.json", _all_selected())])

    def test_reports_missing_optional_modules_with_an_install_hint(self, tmp_path: Path) -> None:
        effective = self._effective(tmp_path)

        problems = check_availability(
            effective.config, environ={}, module_available=lambda name: name != "torch"
        )

        message = " ".join(problem.message for problem in problems)
        assert "torch" in message
        assert any(problem.path == f"components.{DENSE}" for problem in problems)

    def test_passes_when_every_module_and_secret_is_present(self, tmp_path: Path) -> None:
        effective = self._effective(tmp_path)

        assert (
            check_availability(effective.config, environ={}, module_available=lambda _: True) == ()
        )

    def test_reports_a_missing_secret_by_name_only(self, tmp_path: Path) -> None:
        document = _all_selected()
        document["components"]["visual_perception"]["semantic_interpretation"]["backend"] = "gemini"  # type: ignore[index]
        effective = resolve_effective_config(files=[_write(tmp_path / "a.json", document)])

        problems = check_availability(effective.config, environ={}, module_available=lambda _: True)

        assert [problem.message for problem in problems if "GEMINI_API_KEY" in problem.message]


class TestSecrets:
    def _gemini(self, tmp_path: Path) -> EffectiveConfig:
        document = _all_selected()
        document["components"]["visual_perception"]["semantic_interpretation"]["backend"] = "gemini"  # type: ignore[index]
        return resolve_effective_config(files=[_write(tmp_path / "a.json", document)])

    def test_resolves_only_the_secrets_the_selected_backends_declare(self, tmp_path: Path) -> None:
        effective = self._gemini(tmp_path)

        secrets = resolve_secrets(
            effective.config, environ={"GEMINI_API_KEY": "s3cr3t", "UNRELATED": "x"}
        )

        assert secrets.names == ("GEMINI_API_KEY",)
        assert secrets.get("GEMINI_API_KEY") == "s3cr3t"
        assert secrets.missing == ()

    def test_never_renders_a_secret_value(self, tmp_path: Path) -> None:
        secrets = resolve_secrets(
            self._gemini(tmp_path).config, environ={"GEMINI_API_KEY": "s3cr3t"}
        )

        assert "s3cr3t" not in repr(secrets)
        assert "s3cr3t" not in str(secrets)

    def test_reports_missing_secrets_without_failing_resolution(self, tmp_path: Path) -> None:
        secrets = resolve_secrets(self._gemini(tmp_path).config, environ={})

        assert secrets.missing == ("GEMINI_API_KEY",)
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            secrets.get("GEMINI_API_KEY")

    def test_secrets_never_reach_the_effective_config_or_its_persisted_form(
        self, tmp_path: Path
    ) -> None:
        effective = self._gemini(tmp_path)
        resolve_secrets(effective.config, environ={"GEMINI_API_KEY": "s3cr3t"})

        written = write_effective_config(effective, tmp_path / "run")

        assert "s3cr3t" not in written.read_text(encoding="utf-8")
        assert "s3cr3t" not in json.dumps(effective.config.to_document())


class TestPersistence:
    def test_round_trips_and_verifies_the_digest(self, tmp_path: Path) -> None:
        effective = resolve_effective_config(overrides=["policies.debug_level=full"])

        path = write_effective_config(effective, tmp_path / "run")
        loaded = read_effective_config(path)

        assert path.name == "effective_config.json"
        assert loaded.config == effective.config
        assert loaded.digest == effective.digest
        assert loaded.sources == effective.sources

    def test_records_schema_version_digest_and_sources(self, tmp_path: Path) -> None:
        effective = resolve_effective_config()

        document = json.loads(write_effective_config(effective, tmp_path).read_text("utf-8"))

        assert document["schema_version"] == CONFIG_SCHEMA_VERSION
        assert document["digest"] == effective.digest
        assert document["sources"][0] == {
            "kind": "profile",
            "identity": CANONICAL_PROFILE_ID,
            "content_hash": None,
        }

    def test_detects_a_tampered_document(self, tmp_path: Path) -> None:
        path = write_effective_config(resolve_effective_config(), tmp_path)
        document = json.loads(path.read_text("utf-8"))
        document["config"]["policies"]["debug_level"] = "full"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ConfigurationError, match="digest"):
            read_effective_config(path)

    def test_writing_is_idempotent_but_never_replaces_a_different_config(
        self, tmp_path: Path
    ) -> None:
        effective = resolve_effective_config()
        first = write_effective_config(effective, tmp_path)
        again = write_effective_config(effective, tmp_path)
        assert first == again

        other = resolve_effective_config(overrides=["policies.debug_level=full"])
        with pytest.raises(ConfigurationError, match="already"):
            write_effective_config(other, tmp_path)
        assert read_effective_config(first).digest == effective.digest

    def test_leaves_no_temporary_files_behind(self, tmp_path: Path) -> None:
        write_effective_config(resolve_effective_config(), tmp_path / "run")

        assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
            "effective_config.json"
        ]

    def test_rejects_a_document_from_another_schema_version(self, tmp_path: Path) -> None:
        path = write_effective_config(resolve_effective_config(), tmp_path)
        document = json.loads(path.read_text("utf-8"))
        document["schema_version"] = "0.0.0"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ConfigurationError, match="schema_version"):
            read_effective_config(path)


def test_configuration_source_is_a_plain_value() -> None:
    source = ConfigurationSource(kind="profile", identity="canonical/1", content_hash=None)

    assert source == ConfigurationSource(kind="profile", identity="canonical/1", content_hash=None)
