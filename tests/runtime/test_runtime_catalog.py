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
from contextmap.runtime.catalog import (
    CANONICAL_PRESET,
    COMPONENTS,
    PRESETS,
)
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

    def test_entity_resolution_policy_identities_are_the_ones_the_capability_versions(
        self,
    ) -> None:
        from contextmap.entity_resolution import (
            APPEARANCE_COMPARISON_POLICY_ID,
            CANDIDATE_RETRIEVAL_POLICY_ID,
            CONSERVATIVE_RESOLUTION_POLICY_ID,
            GEOMETRY_COMPARISON_POLICY_ID,
            REPRESENTATION_COMPARISON_POLICY_ID,
            SEMANTIC_COMPATIBILITY_POLICY_ID,
            TEMPORAL_COMPATIBILITY_POLICY_ID,
        )

        assert set(COMPONENTS["entity_resolution.retrieval"].backends) == {
            CANDIDATE_RETRIEVAL_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.resolution"].backends) == {
            CONSERVATIVE_RESOLUTION_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.geometry_comparison"].backends) == {
            GEOMETRY_COMPARISON_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.semantic_compatibility"].backends) == {
            SEMANTIC_COMPATIBILITY_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.temporal_compatibility"].backends) == {
            TEMPORAL_COMPATIBILITY_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.appearance"].backends) == {
            APPEARANCE_COMPARISON_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.representation"].backends) == {
            REPRESENTATION_COMPARISON_POLICY_ID
        }

    def test_spatial_relations_policy_identities_are_the_ones_the_capability_versions(
        self,
    ) -> None:
        from contextmap.semantic_mapping import GEOMETRY_SUMMARY_ALGORITHM_ID
        from contextmap.spatial_relations import (
            CANDIDATE_POLICY_ID,
            CONTACT_POLICY_ID,
            FRAME_CONVENTIONS_POLICY_ID,
            GEOMETRIC_POLICY_ID,
        )

        assert set(COMPONENTS["spatial_relations.frame_conventions"].backends) == {
            FRAME_CONVENTIONS_POLICY_ID
        }
        assert set(COMPONENTS["spatial_relations.candidate"].backends) == {CANDIDATE_POLICY_ID}
        assert set(COMPONENTS["spatial_relations.geometry_summary"].backends) == {
            GEOMETRY_SUMMARY_ALGORITHM_ID
        }
        assert set(COMPONENTS["spatial_relations.geometric_predicate"].backends) == {
            GEOMETRIC_POLICY_ID
        }
        assert set(COMPONENTS["spatial_relations.contact_predicate"].backends) == {
            CONTACT_POLICY_ID
        }

    def test_only_the_genuinely_optional_evidence_channels_are_marked_optional(self) -> None:
        optional = {component_id for component_id, spec in COMPONENTS.items() if spec.optional}

        assert optional == {
            # Grounding por prompt é evidência adicional de Visual Perception (#569): sem ele,
            # o estágio roda exatamente como antes.
            "visual_perception.region_grounding",
            # O refinamento por máscara é um estágio de evidência separado e ablável (#568).
            "visual_perception.region_refinement",
            "entity_resolution.semantic_compatibility",
            "entity_resolution.temporal_compatibility",
            "entity_resolution.appearance",
            "entity_resolution.representation",
            "spatial_relations.geometric_predicate",
            "spatial_relations.contact_predicate",
        }

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

    def test_only_point_representation_and_pose_ingestion_are_optional_and_off_by_default(
        self,
    ) -> None:
        optional = [stage.stage_id for stage in CANONICAL_PRESET.stages if stage.optional]

        assert optional == ["pose_ingestion", "point_representation"]
        assert not CANONICAL_PRESET.stage("point_representation").default_enabled
        assert not CANONICAL_PRESET.stage("pose_ingestion").default_enabled

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
        assert plan.order[-1] == "context_map"

    def test_the_canonical_profile_is_a_known_preset(self) -> None:
        assert PRESETS[CANONICAL_PROFILE_ID] is CANONICAL_PRESET

    def test_an_unknown_stage_lookup_fails_loudly(self) -> None:
        with pytest.raises(KeyError):
            CANONICAL_PRESET.stage("nope")

    def test_the_canonical_topology_is_the_whole_solution_1_pipeline(self) -> None:
        """Pre-v0.1.0 there is one topology, end to end (see ``CANONICAL_PROFILE_ID``'s own
        docstring for why this identity is free to keep evolving until the release)."""
        assert CANONICAL_PROFILE_ID == "canonical/1"
        assert [stage.stage_id for stage in CANONICAL_PRESET.stages] == [
            "ingestion",
            "pose_ingestion",
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "point_representation",
            "semantic_fusion",
            "semantic_mapping",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        ]
        assert CANONICAL_PRESET.stages[-1].output == "ContextMapArtifact"


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
        assert config.inputs.selections == {"state_estimation": ("seq-01--run-0003",)}
        assert config.resources.workspace == "workspace/run-a"
        assert config.policies.debug_level == "standard"

    def test_enabling_the_optional_stage_makes_its_backend_a_required_choice(self) -> None:
        disabled = check_selection(resolve_effective_config().config)
        enabled = check_selection(
            resolve_effective_config(overrides=["pipeline.stages.point_representation=true"]).config
        )

        assert not any("point_representation" in problem.path for problem in disabled)
        assert any("point_representation.encoder" in problem.path for problem in enabled)
