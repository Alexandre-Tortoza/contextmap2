"""Guard the static runtime catalog against drifting from the capabilities it names."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from contextmap.runtime import (
    CANONICAL_PROFILE_ID,
    check_selection,
    resolve_effective_config,
)
from contextmap.runtime.catalog import CANONICAL_PRESET, COMPONENTS, PRESETS
from contextmap.semantic_fusion import (
    BASELINE_ACCUMULATION_POLICY_ID,
    GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
    QUALITY_AWARE_ACCUMULATION_POLICY_ID,
    EvidenceChannel,
)


def _package_exists(capability: str) -> bool:
    return importlib.util.find_spec(f"contextmap.{capability}") is not None


class TestCatalogAgainstCapabilities:
    def test_fusion_policy_identities_are_the_ones_the_capability_versions(self) -> None:
        assert set(COMPONENTS["semantic_fusion.support"].backends) == {
            GEOMETRY_OVERLAP_SUPPORT_POLICY_ID
        }
        assert set(COMPONENTS["semantic_fusion.accumulation"].backends) == {
            BASELINE_ACCUMULATION_POLICY_ID,
            QUALITY_AWARE_ACCUMULATION_POLICY_ID,
        }

    def test_the_channel_that_needs_the_point_representation_stage_still_exists(self) -> None:
        assert EvidenceChannel.POINT_REPRESENTATION.value == "point_representation"

    def test_every_available_stage_names_an_implemented_capability(self) -> None:
        for stage in CANONICAL_PRESET.stages:
            if stage.available:
                assert _package_exists(stage.capability), stage.stage_id

    def test_every_unavailable_stage_names_a_capability_that_is_still_missing(self) -> None:
        # Quando a capability passar a existir, o catálogo precisa ser atualizado junto.
        for stage in CANONICAL_PRESET.stages:
            if not stage.available:
                assert not _package_exists(stage.capability), (
                    f"{stage.capability} now exists: mark stage {stage.stage_id!r} available"
                )
                assert stage.unavailable_reason

    def test_every_component_belongs_to_exactly_one_stage_of_its_capability(self) -> None:
        owners: dict[str, list[str]] = {}
        for stage in CANONICAL_PRESET.stages:
            for component_id in stage.components:
                assert component_id in COMPONENTS
                assert COMPONENTS[component_id].capability == stage.capability
                owners.setdefault(component_id, []).append(stage.stage_id)
        assert set(owners) == set(COMPONENTS)
        assert all(len(stages) == 1 for stages in owners.values())

    def test_only_the_point_representation_stage_is_optional_and_off_by_default(self) -> None:
        optional = [stage.stage_id for stage in CANONICAL_PRESET.stages if stage.optional]

        assert optional == ["point_representation"]
        assert not CANONICAL_PRESET.stage("point_representation").default_enabled

    def test_every_input_is_wired_to_a_stage_that_produces_its_contract(self) -> None:
        stages = {stage.stage_id: stage for stage in CANONICAL_PRESET.stages}

        for stage in CANONICAL_PRESET.stages:
            for item in stage.inputs:
                assert item.source in stages, f"{stage.stage_id}.{item.name}"
                assert stages[item.source].output == item.contract, f"{stage.stage_id}.{item.name}"

    def test_every_stage_declares_what_it_produces(self) -> None:
        assert all(stage.output for stage in CANONICAL_PRESET.stages)

    def test_the_canonical_topology_resolves_without_structural_problems(self) -> None:
        from contextmap.runtime import resolve_effective_config, resolve_plan

        plan = resolve_plan(resolve_effective_config())

        assert plan.problems == ()
        assert plan.order is not None

    def test_the_canonical_profile_is_a_known_preset(self) -> None:
        assert PRESETS[CANONICAL_PROFILE_ID] is CANONICAL_PRESET

    def test_an_unknown_stage_lookup_fails_loudly(self) -> None:
        with pytest.raises(KeyError):
            CANONICAL_PRESET.stage("nope")


class TestRealisticConfigurationFile:
    def test_a_toml_configuration_selects_backends_and_enables_the_optional_stage(
        self, tmp_path: Path
    ) -> None:
        file = tmp_path / "experiment.toml"
        file.write_text(
            """
[pipeline.stages]
point_representation = true

[resources]
device = "cuda"
workspace = "workspace/run-a"

[inputs]
sequence = "seq-01"
[inputs.selections]
state_estimation = "seq-01--run-0003"

[policies]
debug_level = "standard"

[components.visual_perception.region_discovery]
backend = "sam3"
[components.visual_perception.region_discovery.sam3]
checkpoint = "sam3-x"
score_threshold = 0.25

[components.point_representation.encoder]
backend = "geometric_descriptor"
""",
            encoding="utf-8",
        )

        config = resolve_effective_config(files=[file]).config

        assert config.pipeline.stages["point_representation"] is True
        assert config.components["point_representation.encoder"].backend == "geometric_descriptor"
        assert config.components["visual_perception.region_discovery"].parameters == {
            "checkpoint": "sam3-x",
            "score_threshold": 0.25,
            "device": "cuda",
        }
        assert config.inputs.sequence == "seq-01"
        assert config.inputs.selections == {"state_estimation": "seq-01--run-0003"}
        assert config.resources.workspace == "workspace/run-a"
        assert config.policies.debug_level == "standard"

    def test_enabling_the_optional_stage_makes_its_backend_a_required_choice(self) -> None:
        disabled = check_selection(resolve_effective_config().config)
        enabled = check_selection(
            resolve_effective_config(overrides=["pipeline.stages.point_representation=true"]).config
        )

        assert not any("point_representation" in problem.path for problem in disabled)
        assert any("point_representation.encoder" in problem.path for problem in enabled)
